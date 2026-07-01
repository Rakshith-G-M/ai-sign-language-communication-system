"""
FastAPI dependency providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from core.config import Settings, get_settings

if TYPE_CHECKING:
    from services.prediction_service import ASLPredictionService


def get_app_settings() -> Settings:
    """Return cached application settings."""
    return get_settings()


def get_prediction_service(request: Request) -> "ASLPredictionService":
    """Return the lifespan-managed prediction service singleton."""
    return request.app.state.prediction_service
