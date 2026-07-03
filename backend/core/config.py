"""
Centralized application configuration.

All runtime settings are loaded from environment variables with sensible
defaults that match the pre-Milestone-3 hardcoded values.  Import ``settings``
as a module-level singleton rather than instantiating ``Settings`` repeatedly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    """Resolve repository root (parent of ``backend/``)."""
    return Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    app_name: str = "ASL Sign Language Communication Platform"
    app_version: str = "1.1.0"
    api_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # ── Paths ────────────────────────────────────────────────────────────
    # Override in Docker with PROJECT_ROOT=/app when only backend/ is copied.
    project_root: Path = Field(default_factory=_default_project_root)

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def dataset_dir(self) -> Path:
        return self.project_root / "dataset"

    @property
    def static_model_path(self) -> Path:
        return self.models_dir / "asl_xgboost.pkl"

    @property
    def static_encoder_path(self) -> Path:
        return self.models_dir / "label_encoder.pkl"


    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── CORS ─────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # Default is localhost ports to prevent wildcard Starlette credential exceptions.
    # For production, override this by setting the CORS_ORIGINS environment variable, e.g.:
    #   CORS_ORIGINS=https://app.example.com,https://www.example.com
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"
    cors_allow_credentials: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # ── Request limits ───────────────────────────────────────────────────
    max_upload_bytes: int = 1 * 1024 * 1024            # 1 MB
    max_base64_chars: int = 1_500_000                  # ~1 MB decoded
    max_tts_chars: int = 5_000
    max_session_id_length: int = 128


    # ── Session management ───────────────────────────────────────────────
    session_idle_seconds: float = 600.0
    max_sessions: int = 1_000


    # ── Log normalisation ────────────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


settings = get_settings()
