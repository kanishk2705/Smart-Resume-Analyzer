from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    gemini_configured: bool


class ScoredSentence(BaseModel):
    text: str
    score: float = Field(description="Cosine similarity to the closest job requirement, -1 to 1")


class ScoreComponent(BaseModel):
    key: Literal["skills", "coverage", "experience", "education"]
    label: str
    score: float | None = Field(description="0-100, or null when it could not be assessed")
    weight: float = Field(description="Share of the overall score, before renormalising")
    detail: str


class SkillMatch(BaseModel):
    name: str
    category: str
    status: Literal["matched", "related", "missing"]
    related_to: str | None = Field(
        default=None, description="For related skills, the substitute found in the resume"
    )


class RequirementMatch(BaseModel):
    requirement: str = Field(description="A sentence from the job description")
    evidence: str = Field(description="The resume sentence that supports it best")
    similarity: float
    status: Literal["strong", "partial", "weak"]


class AnalysisResponse(BaseModel):
    match_score: float = Field(description="Weighted mean of the assessed components, 0-100")
    components: list[ScoreComponent]
    skills: list[SkillMatch] = Field(description="Every skill the job description asks for")
    additional_skills: list[str] = Field(description="Resume skills the job did not ask for")
    requirements: list[RequirementMatch]
    top_sentences: list[ScoredSentence]
    sections_found: list[str]


class RecommendationRequest(BaseModel):
    missing_skills: list[str] = Field(min_length=1)


class Recommendation(BaseModel):
    skill: str = ""
    resource_name: str = "Recommended Course"
    project: str = "No project generated"
    link: str = "#"


class RecommendationResponse(BaseModel):
    recommendations: list[Recommendation]
