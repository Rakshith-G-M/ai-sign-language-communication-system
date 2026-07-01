"""Prediction API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from core.config import settings


class PredictionResponse(BaseModel):
    """ASL prediction result returned to the frontend."""

    letter: str | None = Field(None, description="Stabilised static ASL letter, if any.")
    confidence: float = Field(..., description="Prediction confidence score (0.0–1.0).")
    word: str = Field(..., description="Current in-progress word being assembled.")
    sentence: str = Field(..., description="Current sentence text.")
    suggestions: list[str] = Field(default_factory=list, description="Spell-check suggestions.")
    finalized_sentence: str | None = Field(
        None, description="Sentence just completed for optional TTS playback."
    )
    hand_detected: bool = Field(..., description="Whether MediaPipe detected a hand.")
    latency: float = Field(..., description="End-to-end request latency in milliseconds.")


class Base64Request(BaseModel):
    """Base64-encoded image payload for frame prediction."""

    image: str = Field(..., min_length=1, description="Base64-encoded image bytes.")

    @field_validator("image")
    @classmethod
    def validate_image_size(cls, v: str) -> str:
        if len(v) > settings.max_base64_chars:
            raise ValueError(
                f"Base64 payload exceeds maximum allowed size "
                f"({settings.max_base64_chars} characters)."
            )
        return v


class TTSRequest(BaseModel):
    """Text-to-speech generation request."""

    text: str = Field(..., description="Text to synthesise as speech.")

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        if len(v) > settings.max_tts_chars:
            raise ValueError(
                f"Text exceeds maximum allowed length ({settings.max_tts_chars} characters)."
            )
        return v


class ResetResponse(BaseModel):
    status: str = "reset successful"


class StateResponse(BaseModel):
    word: str
    sentence: str
