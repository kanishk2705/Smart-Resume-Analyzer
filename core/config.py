from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # Scoring
    embedding_model: str = "all-MiniLM-L6-v2"
    preload_model: bool = True
    top_k_sentences: int = 3
    max_missing_keywords: int = 10

    # Uploads
    max_upload_mb: int = 5

    # Gemini (recommendations)
    gemini_api_key: str = ""
    gemini_models: str = "gemini-2.5-flash,gemini-2.0-flash-exp,gemini-1.5-flash"
    gemini_timeout_seconds: float = 30.0
    max_recommendation_skills: int = 3

    @property
    def gemini_model_list(self) -> list[str]:
        return [m.strip() for m in self.gemini_models.split(",") if m.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
