import io
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile

from backend.schemas import (
    AnalysisResponse,
    HealthResponse,
    HistoryResponse,
    RankingResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from backend.storage import HistoryStore
from core.ats import check_resume
from core.config import Settings, get_settings
from core.logging_config import configure_logging
from core.parsing import (
    ParsedDocument,
    ResumeParseError,
    UnsupportedFileTypeError,
    extract_document,
)
from core.recommendations import RecommendationError, get_ai_recommendations
from core.report import build_report
from core.scoring import ResumeAnalyzer, get_analyzer

logger = logging.getLogger(__name__)

MAX_JOB_TITLE_CHARS = 80


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
    description="Scores a resume against a job description and suggests how to close skill gaps.",
    version="2.0.0",
    lifespan=lifespan,
)


@lru_cache(maxsize=1)
def _history_store(path: str) -> HistoryStore:
    return HistoryStore(path)


def get_history(settings: Annotated[Settings, Depends(get_settings)]) -> HistoryStore:
    return _history_store(settings.history_db)


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
        gemini_configured=bool(settings.gemini_api_key),
    )


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze(
    resume: Annotated[UploadFile, File(description="Resume as PDF or DOCX")],
    job_description: Annotated[str, Form(min_length=1)],
    analyzer: Annotated[ResumeAnalyzer, Depends(get_analyzer)],
    history: Annotated[HistoryStore, Depends(get_history)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Scores one resume against a job description and saves the result to the history."""
    _require_job_description(job_description)
    document = _read_resume(resume, settings)

    result = asdict(analyzer.analyze(document.text, job_description))
    result["ats"] = asdict(check_resume(document))
    saved = history.save(resume.filename or "resume", _job_title(job_description), result)
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


@app.get("/api/v1/history", response_model=HistoryResponse)
def list_history(history: Annotated[HistoryStore, Depends(get_history)], limit: int = 50):
    return {"analyses": history.list(min(max(limit, 1), 200))}


def _stored_analysis(analysis_id: int, history: HistoryStore) -> dict:
    analysis = history.get(analysis_id)
    if analysis is None:
        raise HTTPException(404, f"No analysis with id {analysis_id}.")
    return analysis


@app.get("/api/v1/history/{analysis_id}", response_model=AnalysisResponse)
def get_history_item(analysis_id: int, history: Annotated[HistoryStore, Depends(get_history)]):
    return _stored_analysis(analysis_id, history)


@app.get("/api/v1/history/{analysis_id}/report", response_class=Response)
def get_report(analysis_id: int, history: Annotated[HistoryStore, Depends(get_history)]):
    """The analysis as a downloadable PDF."""
    pdf = build_report(_stored_analysis(analysis_id, history))
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="resume-report-{analysis_id}.pdf"'},
    )


@app.delete("/api/v1/history/{analysis_id}", status_code=204)
def delete_history_item(analysis_id: int, history: Annotated[HistoryStore, Depends(get_history)]):
    if not history.delete(analysis_id):
        raise HTTPException(404, f"No analysis with id {analysis_id}.")


@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
def recommendations(
    request: RecommendationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    x_gemini_api_key: Annotated[str | None, Header()] = None,
):
    # A key sent by the client takes priority over the one configured on the server
    api_key = x_gemini_api_key or settings.gemini_api_key
    if not api_key:
        raise HTTPException(
            503,
            "No Gemini API key. Set GEMINI_API_KEY on the server or send X-Gemini-Api-Key.",
        )

    try:
        recs = get_ai_recommendations(request.missing_skills, api_key)
    except RecommendationError as e:
        raise HTTPException(502, f"Gemini request failed ({e}).") from e
    return {"recommendations": recs}
