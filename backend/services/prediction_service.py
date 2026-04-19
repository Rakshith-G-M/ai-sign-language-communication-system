"""
ASL Prediction Service
──────────────────────────────────────────────

Handles the core inference pipeline:
image → prediction → word/sentence building
"""

import cv2
import numpy as np
import logging
import base64
import time
from collections import deque, Counter

# ✅ Updated imports (no sys.path hacks anymore)
from core.inference.realtime_asl_predictor import predict_frame
from core.inference.text_builder import TextBuilder

log = logging.getLogger(__name__)


class ASLPredictionService:
    """
    Wraps prediction + text building logic.
    Maintains state across frames.
    """

    def __init__(self):
        self.text_builder = TextBuilder()
        log.info("ASL Prediction Service initialized")

    # ─────────────────────────────────────────────────────────────────────────
    # Predict from raw image bytes
    # ─────────────────────────────────────────────────────────────────────────
    def predict_from_bytes(self, image_bytes: bytes, start_time: float) -> dict:
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return self._error_response("Invalid image format", start_time)

            return self._run_pipeline(frame, start_time)

        except Exception as exc:
            log.error(f"Prediction error: {exc}")
            return self._error_response(str(exc), start_time)

    # ─────────────────────────────────────────────────────────────────────────
    # Predict from base64
    # ─────────────────────────────────────────────────────────────────────────
    def predict_from_base64(self, base64_str: str, start_time: float) -> dict:
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_str)
            return self.predict_from_bytes(image_bytes, start_time)

        except Exception as exc:
            log.error(f"Base64 decode error: {exc}")
            return self._error_response("Invalid base64 input", start_time)

    # ─────────────────────────────────────────────────────────────────────────
    # Core pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _run_pipeline(self, frame: np.ndarray, start_time: float) -> dict:
        try:
            # 🔥 Core ML call - Returns fully stabilized letter
            annotated_frame, stable_letter, hand_detected = predict_frame(frame)

            # 🧠 Update word + sentence with stabilized letter
            current_word, sentence, suggestions = self.text_builder.update(
                stable_letter,
                hand_detected,
                time.time(),
            )

            # Check if a sentence was just finalized for TTS
            finalized_sentence = self.text_builder.pop_final_sentence()

            latency = round((time.time() - start_time) * 1000, 2)

            return {
                "letter": stable_letter,
                "confidence": 1.0 if stable_letter else 0.0,
                "word": current_word,
                "sentence": sentence,
                "suggestions": suggestions, # ✨ New field
                "finalized_sentence": finalized_sentence,
                "hand_detected": hand_detected,
                "latency": latency,
            }

        except Exception as exc:
            log.error(f"Pipeline error: {exc}")
            return self._error_response(str(exc), start_time)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _error_response(self, message: str, start_time: float) -> dict:
        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "letter": None,
            "confidence": 0.0,
            "word": self.text_builder.current_word,
            "sentence": self.text_builder.sentence,
            "suggestions": [],
            "finalized_sentence": None,
            "hand_detected": False,
            "latency": latency,
        }

    def reset(self):
        """Reset word/sentence state"""
        self.text_builder.reset()
        log.info("State reset")

    def get_state(self):
        """Return current state"""
        return {
            "word": self.text_builder.current_word,
            "sentence": self.text_builder.sentence,
        }