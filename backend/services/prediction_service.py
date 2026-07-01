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


from core.inference.dynamic_predictor import DynamicPredictor
from core.inference.prediction_session import PredictionSession

log = logging.getLogger(__name__)


class ASLPredictionService:
    """
    Wraps prediction + text building logic.
    Maintains state across frames isolated inside sessions.
    """

    def __init__(self):
        self.sessions: dict[str, PredictionSession] = {}
        self.dynamic_predictor = DynamicPredictor()
        log.info("ASL Prediction Service initialized with session isolation")

    def get_or_create_session(self, session_id: str | None) -> PredictionSession:
        """Retrieve an existing user session or create a new one."""
        if not session_id:
            session_id = "default"

        # Periodically clean up sessions (approx. 5% chance on request)
        import random
        if random.random() < 0.05:
            self._cleanup_inactive_sessions()

        if session_id not in self.sessions:
            self.sessions[session_id] = PredictionSession()
            log.info("Created new prediction session: %s", session_id)

        return self.sessions[session_id]

    def _cleanup_inactive_sessions(self, max_idle_seconds: float = 600.0) -> None:
        """Remove inactive sessions to prevent memory leaks."""
        now = time.time()
        expired = [
            sid for sid, s in self.sessions.items()
            if sid != "default" and (now - s.last_active_time) > max_idle_seconds
        ]
        for sid in expired:
            del self.sessions[sid]
            log.info("Cleaned up expired prediction session: %s", sid)

    # ─────────────────────────────────────────────────────────────────────────
    # Predict from raw image bytes
    # ─────────────────────────────────────────────────────────────────────────
    def predict_from_bytes(self, image_bytes: bytes, start_time: float, session_id: str | None = None) -> dict:
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            session = self.get_or_create_session(session_id)

            if frame is None:
                return self._error_response("Invalid image format", start_time, session)

            return self._run_pipeline(frame, start_time, session)

        except Exception as exc:
            log.error(f"Prediction error: {exc}")
            session = self.get_or_create_session(session_id)
            return self._error_response(str(exc), start_time, session)

    # ─────────────────────────────────────────────────────────────────────────
    # Predict from base64
    # ─────────────────────────────────────────────────────────────────────────
    def predict_from_base64(self, base64_str: str, start_time: float, session_id: str | None = None) -> dict:
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_str)
            return self.predict_from_bytes(image_bytes, start_time, session_id)

        except Exception as exc:
            log.error(f"Base64 decode error: {exc}")
            session = self.get_or_create_session(session_id)
            return self._error_response("Invalid base64 input", start_time, session)

    # ─────────────────────────────────────────────────────────────────────────
    # Core pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _run_pipeline(self, frame: np.ndarray, start_time: float, session: PredictionSession) -> dict:
        try:
            # 🔥 Core static ML call - Returns fully stabilized letter
            annotated_frame, stable_letter, hand_detected = predict_frame(frame, session)

            stable_word = None
            if hand_detected:
                # Check for dynamic gestures if hand is in motion
                from core.inference.motion_classifier import is_hand_moving
                if is_hand_moving(session.landmarks_history):
                    # Suppress static letter updates while hand is moving
                    stable_letter = None
                    
                    # Query dynamic predictor for whole-word recognition
                    predicted_word, confidence = self.dynamic_predictor.predict_sequence(session.landmarks_history)
                    if predicted_word:
                        stable_word = predicted_word

            # 🧠 Update word + sentence with stabilized letter or word
            current_word, sentence, suggestions = session.text_builder.update(
                stable_letter,
                hand_detected,
                time.time(),
                stable_word=stable_word
            )

            # Check if a sentence was just finalized for TTS
            finalized_sentence = session.text_builder.pop_final_sentence()

            latency = round((time.time() - start_time) * 1000, 2)

            return {
                "letter": stable_letter,
                "confidence": 1.0 if (stable_letter or stable_word) else 0.0,
                "word": current_word,
                "sentence": sentence,
                "suggestions": suggestions,
                "finalized_sentence": finalized_sentence,
                "hand_detected": hand_detected,
                "latency": latency,
            }

        except Exception as exc:
            log.error(f"Pipeline error: {exc}")
            return self._error_response(str(exc), start_time, session)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _error_response(self, message: str, start_time: float, session: PredictionSession) -> dict:
        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "letter": None,
            "confidence": 0.0,
            "word": session.text_builder.current_word,
            "sentence": session.text_builder.sentence,
            "suggestions": [],
            "finalized_sentence": None,
            "hand_detected": False,
            "latency": latency,
        }

    def reset(self, session_id: str | None = None):
        """Reset word/sentence state for the specified session"""
        session = self.get_or_create_session(session_id)
        session.reset_all()
        log.info("State reset for session: %s", session_id or "default")

    def get_state(self, session_id: str | None = None):
        """Return current state of the specified session"""
        session = self.get_or_create_session(session_id)
        return {
            "word": session.text_builder.current_word,
            "sentence": session.text_builder.sentence,
        }