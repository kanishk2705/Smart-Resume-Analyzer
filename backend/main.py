import io
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)

from backend.ratelimit import RateLimiter
from backend.schemas import (
    AnalysisResponse,
    HealthResponse,
    HistoryResponse,
    LearningPathRequest,
    LearningPathResponse,
    RankingResponse,
    RewriteRequest,
    RewriteResponse,
)
from backend.storage import HistoryStore
from core.ats import check_resume
from core.coaching import learning_path, rewrite_bullets
from core.config import Settings, get_settings
from core.llm import LLMError
from core.logging_config import configure_logging
from core.parsing import (
    ParsedDocument,
    ResumeParseError,
    UnsupportedFileTypeError,
    extract_document,
)
from core.report import build_report
from core.scoring import ResumeAnalyzer, get_analyzer

logger = logging.getLogger(__name__)

MAX_JOB_TITLE_CHARS = 80
SESSION_ID = re.compile(r"[A-Za-z0-9_-]{16,64}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.preload_model:
        # Load the model at startup so the first request isn't slow
        get_analyzer()
    yield


app = FastAPI(
    title="Smart Resume Analyzer API",
    description="Scores resumes against job descriptions and explains the result.",
    version="2.0.0",
    lifespan=lifespan,
)


@lru_cache(maxsize=1)
def _history_store(path: str) -> HistoryStore:
    return HistoryStore(path)


def get_history(settings: Annotated[Settings, Depends(get_settings)]) -> HistoryStore:
    return _history_store(settings.history_db)


def get_session(
    x_session_id: Annotated[
        str | None, Header(description="Random id of the browser session")
    ] = None,
) -> str | None:
    """The caller's session id, if it sent a well-formed one."""
    if x_session_id is None:
        return None
    if not SESSION_ID.fullmatch(x_session_id):
        raise HTTPException(400, "X-Session-Id must be 16-64 letters, digits, '-' or '_'.")
    return x_session_id


def get_owner(
    session: Annotated[str | None, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str | None:
    """Whose history a request may see: its own session's, or (on a private install) all of it."""
    if settings.history_scope == "shared":
        return None
    if session is None:
        raise HTTPException(400, "This server keeps history per session; send X-Session-Id.")
    return session


@lru_cache(maxsize=1)
def _coach_limiter(limit: int) -> RateLimiter:
    return RateLimiter(limit)


def limit_coaching(
    request: Request,
    session: Annotated[str | None, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Caps AI coach calls per visitor, so one visitor cannot use up a public demo's quota."""
    client = session or (request.client.host if request.client else "unknown")
    wait = _coach_limiter(settings.coach_requests_per_hour).acquire(client)
    if wait is not None:
        minutes = max(1, round(wait / 60))
        raise HTTPException(
            429,
            f"AI coach limit reached ({settings.coach_requests_per_hour} requests an hour). "
            f"Try again in about {minutes} minute{'s' if minutes != 1 else ''}.",
            headers={"Retry-After": str(int(wait) + 1)},
        )


def _read_resume(upload: UploadFile, settings: Settings) -> ParsedDocument:
    """Parses an upload, raising the HTTP error that describes what is wrong with it."""
    content = upload.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"Resume is larger than the {settings.max_upload_mb} MB limit.")

    try:
        return extract_document(io.BytesIO(content), upload.filename or "")
    except UnsupportedFileTypeError as e:
        raise HTTPException(415, str(e)) from e
    except ResumeParseError as e:
        raise HTTPException(422, str(e)) from e


def _require_job_description(job_description: str) -> None:
    if not job_description.strip():
        raise HTTPException(422, "Job description is empty.")


def _job_title(job_description: str) -> str:
    """A short label for the job: the first line of the description, trimmed."""
    first_line = job_description.strip().splitlines()[0].strip(" -•*#:")
    if len(first_line) > MAX_JOB_TITLE_CHARS:
        return first_line[: MAX_JOB_TITLE_CHARS - 1].rstrip() + "…"
    return first_line or "Untitled job"


@app.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]):
    return HealthResponse(
        status="ok",
        embedding_model=settings.embedding_model,
        llm_configured=bool(settings.groq_api_key),
        history_scope=settings.history_scope,
    )


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze(
    resume: Annotated[UploadFile, File(description="Resume as PDF or DOCX")],
    job_description: Annotated[str, Form(min_length=1)],
    analyzer: Annotated[ResumeAnalyzer, Depends(get_analyzer)],
    history: Annotated[HistoryStore, Depends(get_history)],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[str | None, Depends(get_session)],
):
    """Scores one resume against a job description and saves the result to the history."""
    _require_job_description(job_description)
    document = _read_resume(resume, settings)

    result = asdict(analyzer.analyze(document.text, job_description))
    result["ats"] = asdict(check_resume(document))
    saved = history.save(
        resume.filename or "resume", _job_title(job_description), result, owner=session
    )
    logger.info("Analyzed '%s': match_score=%.2f", resume.filename, saved["match_score"])
    return saved


@app.post("/api/v1/rank", response_model=RankingResponse)
def rank(
    resumes: Annotated[list[UploadFile], File(description="Resumes as PDF or DOCX")],
    job_description: Annotated[str, Form(min_length=1)],
    analyzer: Annotated[ResumeAnalyzer, Depends(get_analyzer)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Ranks several resumes against one job description. Results are not saved."""
    _require_job_description(job_description)
    if len(resumes) > settings.max_batch_files:
        raise HTTPException(
            413, f"At most {settings.max_batch_files} resumes can be ranked at once."
        )

    scored, failed = [], []
    for upload in resumes:
        filename = upload.filename or "resume"
        try:
            document = _read_resume(upload, settings)
        except HTTPException as e:
            # One unreadable file should not sink the whole batch
            failed.append({"filename": filename, "error": e.detail})
            continue

        result = analyzer.analyze(document.text, job_description)
        by_status = {status: [] for status in ("matched", "related", "missing")}
        for skill in result.skills:
            by_status[skill.status].append(skill.name)
        scored.append({
            "filename": filename,
            "match_score": result.match_score,
            "components": [asdict(c) for c in result.components],
            "matched_skills": by_status["matched"],
            "related_skills": by_status["related"],
            "missing_skills": by_status["missing"],
        })

    scored.sort(key=lambda candidate: candidate["match_score"], reverse=True)
    candidates = [{"rank": i, **candidate} for i, candidate in enumerate(scored, start=1)]
    logger.info("Ranked %d resumes (%d unreadable)", len(candidates), len(failed))
    return {"job_title": _job_title(job_description), "candidates": candidates, "failed": failed}


History = Annotated[HistoryStore, Depends(get_history)]
Owner = Annotated[str | None, Depends(get_owner)]


@app.get("/api/v1/history", response_model=HistoryResponse)
def list_history(history: History, owner: Owner, limit: int = 50):
    return {"analyses": history.list(min(max(limit, 1), 200), owner=owner)}


def _found(analysis: dict | None, description: str) -> dict:
    if analysis is None:
        raise HTTPException(404, f"No {description}.")
    return analysis


def _pdf(analysis: dict) -> Response:
    return Response(
        build_report(analysis),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="resume-report-{analysis["id"]}.pdf"'
        },
    )


@app.get("/api/v1/history/{analysis_id}", response_model=AnalysisResponse)
def get_history_item(analysis_id: int, history: History, owner: Owner):
    return _found(history.get(analysis_id, owner), f"analysis with id {analysis_id}")


@app.get("/api/v1/history/{analysis_id}/report", response_class=Response)
def get_report(analysis_id: int, history: History, owner: Owner):
    """The analysis as a downloadable PDF."""
    return _pdf(_found(history.get(analysis_id, owner), f"analysis with id {analysis_id}"))


@app.delete("/api/v1/history/{analysis_id}", status_code=204)
def delete_history_item(analysis_id: int, history: History, owner: Owner):
    if not history.delete(analysis_id, owner):
        raise HTTPException(404, f"No analysis with id {analysis_id}.")


@app.get("/api/v1/shared/{share_token}", response_model=AnalysisResponse)
def get_shared(share_token: str, history: History):
    """An analysis opened from a share link. Anyone holding the link can see it."""
    return _found(history.get_shared(share_token), "shared analysis at this link")


@app.get("/api/v1/shared/{share_token}/report", response_class=Response)
def get_shared_report(share_token: str, history: History):
    return _pdf(_found(history.get_shared(share_token), "shared analysis at this link"))

def _coach(advise, *args):
    """Runs an LLM-backed function, translating its failures into HTTP errors."""
    if not get_settings().groq_api_key:
        raise HTTPException(503, "AI coaching is unavailable: GROQ_API_KEY is not set.")
    try:
        return advise(*args)
    except LLMError as e:
        raise HTTPException(502, f"The AI service did not respond ({e}).") from e


@app.post(
    "/api/v1/coach/learning-path",
    response_model=LearningPathResponse,
    dependencies=[Depends(limit_coaching)],
)
def coach_learning_path(request: LearningPathRequest):
    """A learning resource and a project idea for each of the top missing skills."""
    return {"recommendations": _coach(learning_path, request.missing_skills, request.job_title)}


@app.post(
    "/api/v1/coach/rewrite",
    response_model=RewriteResponse,
    dependencies=[Depends(limit_coaching)],
)
def coach_rewrite(request: RewriteRequest):
    """Stronger wordings of resume statements, without adding facts that were not there."""
    return {"rewrites": _coach(rewrite_bullets, request.statements, request.job_title)}