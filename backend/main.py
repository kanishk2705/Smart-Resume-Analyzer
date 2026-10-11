import io
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from backend.schemas import (
    AnalysisResponse,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from core.config import Settings, get_settings
from core.logging_config import configure_logging
from core.parsing import ResumeParseError, UnsupportedFileTypeError, extract_text
from core.recommendations import RecommendationError, get_ai_recommendations
from core.scoring import ResumeAnalyzer, get_analyzer

logger = logging.getLogger(__name__)


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
    settings: Annotated[Settings, Depends(get_settings)],
):
    if not job_description.strip():
        raise HTTPException(422, "Job description is empty.")

    content = resume.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            413,
            f"Resume is larger than the {settings.max_upload_mb} MB limit.",
        )

    try:
        resume_text = extract_text(io.BytesIO(content), resume.filename or "")
    except UnsupportedFileTypeError as e:
        raise HTTPException(415, str(e)) from e
    except ResumeParseError as e:
        raise HTTPException(422, str(e)) from e

    result = analyzer.analyze(resume_text, job_description)
    logger.info("Analyzed '%s': match_score=%.2f", resume.filename, result.match_score)
    return asdict(result)


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
