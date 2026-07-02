"""Health, readiness, and metrics response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    status: str = Field(..., description="Process liveness status.")


class ReadinessCheck(BaseModel):
    static_model: bool
    mediapipe: bool
    prediction_service: bool


class ReadinessResponse(BaseModel):
    status: str = Field(..., description="'ready' or 'not_ready'.")
    checks: ReadinessCheck


class MetricsResponse(BaseModel):
    uptime_seconds: float
    active_sessions: int
    total_predictions: int
    static_predictions: int
    dynamic_predictions: int   # Always 0; retained for frontend schema compatibility
