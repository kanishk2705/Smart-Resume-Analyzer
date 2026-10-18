from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # Scoring
    embedding_model: str = "all-MiniLM-L6-v2"
    preload_model: bool = True
    top_k_sentences: int = 3
    # Longest sentence sent to the model in one piece; all-MiniLM-L6-v2 reads 256 tokens
    max_unit_words: int = 120
    top_experience_units: int = 5

    # Share of the overall score each component carries
    weight_skills: float = 0.40
    weight_coverage: float = 0.30
    weight_experience: float = 0.20
    weight_education: float = 0.10

    # Cosine similarities at or below the floor score 0, at or above the ceiling score 100
    similarity_floor: float = 0.15
    similarity_ceiling: float = 0.70
    strong_match_threshold: float = 0.50
    partial_match_threshold: float = 0.35

    # Uploads
    max_upload_mb: int = 5
    max_batch_files: int = 25

    # SQLite file holding past analyses
    history_db: str = "data/history.db"
    # "shared": everyone sees every analysis (a private, local install).
    # "session": each browser session sees only its own (a public demo).
    history_scope: Literal["shared", "session"] = "shared"

    # Most AI coach requests one visitor may make per hour; 0 means no limit
    coach_requests_per_hour: int = 0

    # Groq (AI coaching: learning path and bullet rewrites)
    groq_api_key: str = ""
    groq_models: str = "openai/gpt-oss-120b,openai/gpt-oss-20b"
    llm_timeout_seconds: float = 60.0
    # Most skills or statements sent to the LLM in one request
    max_coached_items: int = 5

    @property
    def groq_model_list(self) -> list[str]:
        return [m.strip() for m in self.groq_models.split(",") if m.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
