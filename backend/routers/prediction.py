"""
Prediction Router
──────────────────────────────────────────────

Handles API endpoints for ASL prediction and state management.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
import logging
import time
import edge_tts

# Import service (updated path)
from services.prediction_service import ASLPredictionService

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/api/v1", tags=["prediction"])
log = logging.getLogger(__name__)

# Initialize service (singleton for now)
prediction_service = ASLPredictionService()

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    letter: str | None
    confidence: float
    word: str
    sentence: str
    suggestions: list[str] = []  # ✨ New field
    finalized_sentence: str | None
    hand_detected: bool
    latency: float


class Base64Request(BaseModel):
    image: str

class TTSRequest(BaseModel):
    text: str

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/tts")
async def generate_tts(request: TTSRequest):
    """
    Generate speech dynamically via edge-tts. Runs async to prevent blocking pipeline.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
        
    try:
        communicate = edge_tts.Communicate(request.text, "en-US-AriaNeural")
        audio_data = bytearray()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
                
        return Response(content=bytes(audio_data), media_type="audio/mp3")
    except Exception as exc:
        log.error(f"TTS Engine Error: {exc}")
        raise HTTPException(status_code=500, detail="Speech generation failed")

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict ASL letter from uploaded image frame.
    """
    start_time = time.time()

    # ✅ File type validation
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        contents = await file.read()

        result = prediction_service.predict_from_bytes(contents, start_time)
        return result

    except Exception as exc:
        log.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict-base64", response_model=PredictionResponse)
async def predict_base64(request: Base64Request):
    """
    Predict ASL letter from base64 image.
    """
    start_time = time.time()

    try:
        result = prediction_service.predict_from_base64(
            request.image, start_time
        )
        return result

    except Exception as exc:
        log.error(f"Base64 prediction error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reset")
async def reset():
    """
    Reset word and sentence state.
    """
    prediction_service.reset()
    return {"status": "reset successful"}


@router.get("/state")
async def get_state():
    """
    Get current word and sentence state.
    """
    return prediction_service.get_state()


@router.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


@router.get("/info")
async def info():
    """
    Service metadata.
    """
    return {
        "service": "ASL Recognition Engine",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/predict-base64",
            "/reset",
            "/state",
            "/health",
        ],
    }