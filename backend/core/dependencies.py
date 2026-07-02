"""
FastAPI dependency providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from core.config import Settings, get_settings
from services.tts_service import TTSService

if TYPE_CHECKING:
    from services.prediction_service import ASLPredictionService

_TTS_SERVICE_SINGLETON = TTSService()


def get_app_settings() -> Settings:
    """Return cached application settings."""
    return get_settings()


def get_prediction_service(request: Request) -> "ASLPredictionService":
    """Return the lifespan-managed prediction service singleton."""
    return request.app.state.prediction_service


def get_tts_service() -> TTSService:
    """Return the global Text-to-Speech service singleton."""
    return _TTS_SERVICE_SINGLETON

