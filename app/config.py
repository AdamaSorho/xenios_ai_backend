"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Database (Supabase PostgreSQL)
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Authentication
    supabase_jwt_secret: str
    xenios_backend_api_key: str

    # LLM Providers
    openrouter_api_key: str

    # External Services
    deepgram_api_key: str = ""

    # Transcription settings (Spec 0003)
    transcription_max_file_size_mb: int = 500
    transcription_max_duration_minutes: int = 120
    transcription_min_duration_seconds: int = 10
    transcription_s3_bucket: str = "xenios-transcriptions"

    # S3/R2 Storage (Spec 0002: Document Extraction)
    s3_endpoint_url: str = ""  # For R2 or S3-compatible services
    s3_bucket: str = "xenios-extractions"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "auto"

    # Extraction settings
    extraction_max_file_size_mb: int = 50
    extraction_webhook_url: str = ""  # Optional webhook for completion notifications

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sentry_dsn: str = ""

    # CORS
    cors_origins: str = "http://localhost:3000"

    # Flower
    flower_user: str = "admin"
    flower_password: str = "change-this-password"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
