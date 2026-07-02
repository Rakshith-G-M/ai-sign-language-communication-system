"""
Prediction Router
──────────────────────────────────────────────

Handles API endpoints for ASL prediction and state management.
"""

from __future__ import annotations

import logging
import time
import io

import edge_tts
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile

from core.config import settings
from core.dependencies import get_prediction_service
from schemas.prediction import (
    Base64Request,
    PredictionResponse,
    ResetResponse,
    StateResponse,
    TTSRequest,
)
from services.prediction_service import ASLPredictionService

router = APIRouter(prefix="/api/v1", tags=["prediction"])
log = logging.getLogger(__name__)

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/tts",
    summary="Generate text-to-speech audio",
    response_class=Response,
    responses={
        200: {"content": {"audio/mpeg": {}}, "description": "MP3 audio bytes."},
        400: {"description": "Empty or invalid text."},
        500: {"description": "Speech generation failed."},
    },
)
async def generate_tts(request: TTSRequest) -> Response:
    """Generate speech dynamically via edge-tts."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        communicate = edge_tts.Communicate(request.text.strip(), "en-US-AriaNeural")
        audio_data = bytearray()
        chunk_count = 0
        audio_chunk_count = 0

        async for chunk in communicate.stream():
            chunk_count += 1
            # Only process audio chunks, skip metadata
            if chunk["type"] == "audio":
                audio_chunk_count += 1
                audio_data.extend(chunk["data"])

        # Enhanced validation: check if we actually received audio chunks
        if not audio_data:
            log.warning(
                "TTS generation produced no audio data. Chunks received: %d, Audio chunks: %d",
                chunk_count,
                audio_chunk_count,
            )
            raise HTTPException(status_code=502, detail="No audio generated")

        if audio_chunk_count == 0:
            log.warning(
                "TTS generation received %d total chunks but zero audio chunks",
                chunk_count,
            )
            raise HTTPException(status_code=502, detail="No audio chunks in response")

        audio_bytes = bytes(audio_data)
        
        # Return response with proper headers for browser audio playback
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                # Essential for browser playback
                "Content-Length": str(len(audio_bytes)),
                "Content-Type": "audio/mpeg",
                # Prevent caching to ensure fresh audio on each call
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                # Help browser handle the response correctly
                "Accept-Ranges": "bytes",
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        log.error("TTS engine error: %s - Type: %s", exc, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Speech generation failed") from exc


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict ASL letter from uploaded image",
)
async def predict(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP image frame."),
    session_id: str | None = None,
    service: ASLPredictionService = Depends(get_prediction_service),
) -> dict:
    """Predict ASL letter from an uploaded image frame."""
    start_time = time.time()

    content_type = file.content_type or ""
    if content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()
        if len(contents) > settings.max_upload_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file exceeds size limit")

        return service.predict_from_bytes(contents, start_time, session_id=session_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post(
    "/predict-base64",
    response_model=PredictionResponse,
    summary="Predict ASL letter from base64 image",
)
async def predict_base64(
    request: Base64Request,
    session_id: str | None = None,
    service: ASLPredictionService = Depends(get_prediction_service),
) -> dict:
    """Predict ASL letter from a base64-encoded image."""
    start_time = time.time()

    try:
        return service.predict_from_base64(
            request.image, start_time, session_id=session_id
        )
    except Exception as exc:
        log.error("Base64 prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post(
    "/reset",
    response_model=ResetResponse,
    summary="Reset word and sentence state",
)
async def reset(
    session_id: str | None = None,
    service: ASLPredictionService = Depends(get_prediction_service),
) -> dict:
    """Reset word and sentence state for the given session."""
    service.reset(session_id=session_id)
    return {"status": "reset successful"}


@router.get(
    "/state",
    response_model=StateResponse,
    summary="Get current word and sentence state",
)
async def get_state(
    session_id: str | None = None,
    service: ASLPredictionService = Depends(get_prediction_service),
) -> dict:
    """Return the current word and sentence for the given session."""
    return service.get_state(session_id=session_id)


@router.get(
    "/health",
    summary="Liveness health check (legacy)",
    description="Returns ok when the process is running. Preserved for frontend compatibility.",
)
async def health() -> dict:
    """Health check endpoint — always returns ok when the process is alive."""
    return {"status": "ok"}


@router.get("/info", summary="Service metadata")
async def info() -> dict:
    """Return service metadata and available endpoints."""
    return {
        "service": "ASL Recognition Engine",
        "version": settings.api_version,
        "endpoints": [
            "/predict",
            "/predict-base64",
            "/reset",
            "/state",
            "/health",
            "/info",
            "/tts",
        ],
    }