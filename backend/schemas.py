from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    gemini_configured: bool


class ScoredSentence(BaseModel):
    text: str
    score: float = Field(description="Cosine similarity to the job description, -1 to 1")


class AnalysisResponse(BaseModel):
    match_score: float = Field(description="Resume-to-JD semantic similarity as a percentage")
    missing_keywords: list[str]
    top_sentences: list[ScoredSentence]


class RecommendationRequest(BaseModel):
    missing_skills: list[str] = Field(min_length=1)


class Recommendation(BaseModel):
    skill: str = ""
    resource_name: str = "Recommended Course"
    project: str = "No project generated"
    link: str = "#"


class RecommendationResponse(BaseModel):
    recommendations: list[Recommendation]
