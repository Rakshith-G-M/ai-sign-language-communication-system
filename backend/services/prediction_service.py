"""
ASL Prediction Service
──────────────────────────────────────────────

Handles the static alphabet inference pipeline:
    image → MediaPipe → Static XGBoost Model → Prediction Stabilizer
          → Stabilized Letter → TextBuilder (word / sentence) → response
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.config import settings
from core.inference.prediction_session import PredictionSession
from core.inference.realtime_asl_predictor import is_static_predictor_ready, predict_frame
from schemas.common import normalize_session_id

log = logging.getLogger(__name__)


@dataclass
class ServiceMetrics:
    """Lightweight in-process operational counters."""

    total_predictions: int = 0
    static_predictions: int = 0


@dataclass
class ASLPredictionService:
    """
    Wraps static-alphabet prediction + text building logic.
    Maintains per-session state for multi-client / concurrent support.
    """

    started_at: float = field(default_factory=time.time)
    sessions: dict[str, PredictionSession] = field(default_factory=dict)
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    def __post_init__(self) -> None:
        log.info("ASL Prediction Service initialized (static alphabet pipeline)")

    # ─────────────────────────────────────────────────────────────────────
    # Session management
    # ─────────────────────────────────────────────────────────────────────
    def get_or_create_session(self, session_id: str | None) -> PredictionSession:
        """Retrieve an existing user session or create a new one."""
        normalized_id = normalize_session_id(session_id)
        if session_id and normalized_id == "default" and session_id.strip() != "default":
            log.warning(
                "Invalid session_id normalised to default",
                extra={"original_session_id": session_id[:32]},
            )

        self._cleanup_inactive_sessions()

        if normalized_id not in self.sessions:
            if len(self.sessions) >= settings.max_sessions:
                self._evict_oldest_session()
            self.sessions[normalized_id] = PredictionSession()
            log.info("Created new prediction session: %s", normalized_id)

        return self.sessions[normalized_id]

    def _cleanup_inactive_sessions(self) -> None:
        """Remove inactive sessions to prevent memory leaks."""
        now = time.time()
        expired = [
            sid
            for sid, s in self.sessions.items()
            if sid != "default" and (now - s.last_active_time) > settings.session_idle_seconds
        ]
        for sid in expired:
            del self.sessions[sid]
            log.info("Cleaned up expired prediction session: %s", sid)

    def _evict_oldest_session(self) -> None:
        """Evict the oldest non-default session when at capacity."""
        candidates = [
            (sid, s.last_active_time)
            for sid, s in self.sessions.items()
            if sid != "default"
        ]
        if not candidates:
            return
        oldest_sid = min(candidates, key=lambda x: x[1])[0]
        del self.sessions[oldest_sid]
        log.warning("Evicted oldest session due to capacity limit: %s", oldest_sid)

    # ─────────────────────────────────────────────────────────────────────
    # Predict from raw image bytes
    # ─────────────────────────────────────────────────────────────────────
    def predict_from_bytes(
        self, image_bytes: bytes, start_time: float, session_id: str | None = None
    ) -> dict:
        if not image_bytes:
            session = self.get_or_create_session(session_id)
            return self._error_response("Empty image payload", start_time, session)

        if len(image_bytes) > settings.max_upload_bytes:
            session = self.get_or_create_session(session_id)
            return self._error_response("Image payload too large", start_time, session)

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            session = self.get_or_create_session(session_id)

            if frame is None:
                return self._error_response("Invalid image format", start_time, session)

            return self._run_pipeline(frame, start_time, session, session_id)

        except Exception as exc:
            log.error("Prediction error: %s", exc)
            session = self.get_or_create_session(session_id)
            return self._error_response(str(exc), start_time, session)

    # ─────────────────────────────────────────────────────────────────────
    # Predict from base64
    # ─────────────────────────────────────────────────────────────────────
    def predict_from_base64(
        self, base64_str: str, start_time: float, session_id: str | None = None
    ) -> dict:
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            image_bytes = base64.b64decode(base64_str, validate=True)
            return self.predict_from_bytes(image_bytes, start_time, session_id)

        except Exception as exc:
            log.error("Base64 decode error: %s", exc)
            session = self.get_or_create_session(session_id)
            return self._error_response("Invalid base64 input", start_time, session)

    # ─────────────────────────────────────────────────────────────────────
    # Core static alphabet pipeline
    # ─────────────────────────────────────────────────────────────────────
    def _run_pipeline(
        self,
        frame: np.ndarray,
        start_time: float,
        session: PredictionSession,
        session_id: str | None = None,
    ) -> dict:
        prediction_start = time.perf_counter()

        try:
            annotated_frame, stable_letter, hand_detected = predict_frame(frame, session)

            current_word, sentence, suggestions = session.text_builder.update(
                stable_letter,
                hand_detected,
                time.time(),
            )

            finalized_sentence = session.text_builder.pop_final_sentence()
            latency = round((time.time() - start_time) * 1000, 2)
            prediction_ms = round((time.perf_counter() - prediction_start) * 1000, 2)

            self.metrics.total_predictions += 1
            if stable_letter:
                self.metrics.static_predictions += 1

            log.info(
                "prediction completed",
                extra={
                    "prediction_ms": prediction_ms,
                    "latency_ms": latency,
                    "hand_detected": hand_detected,
                    "letter": stable_letter,
                    "word": current_word,
                    "confidence": 1.0 if stable_letter else 0.0,
                    "model": "static",
                    "session_id": normalize_session_id(session_id),
                },
            )

            return {
                "letter": stable_letter,
                "confidence": 1.0 if stable_letter else 0.0,
                "word": current_word,
                "sentence": sentence,
                "suggestions": suggestions,
                "finalized_sentence": finalized_sentence,
                "hand_detected": hand_detected,
                "latency": latency,
            }

        except Exception as exc:
            log.error("Pipeline error: %s", exc)
            return self._error_response(str(exc), start_time, session)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────
    def _error_response(
        self, message: str, start_time: float, session: PredictionSession
    ) -> dict:
        latency = round((time.time() - start_time) * 1000, 2)
        log.warning("prediction degraded: %s", message, extra={"latency_ms": latency})

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

    def reset(self, session_id: str | None = None) -> None:
        """Reset word/sentence state for the specified session."""
        session = self.get_or_create_session(session_id)
        session.reset_all()
        log.info("State reset for session: %s", normalize_session_id(session_id))

    def get_state(self, session_id: str | None = None) -> dict:
        """Return current state of the specified session."""
        session = self.get_or_create_session(session_id)
        return {
            "word": session.text_builder.current_word,
            "sentence": session.text_builder.sentence,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Health & metrics
    # ─────────────────────────────────────────────────────────────────────
    def readiness_checks(self) -> dict[str, bool]:
        """Component readiness for the /ready probe."""
        return {
            "static_model": is_static_predictor_ready(),
            "mediapipe": is_static_predictor_ready(),
            "prediction_service": True,
        }

    def is_ready(self) -> bool:
        checks = self.readiness_checks()
        return all(checks.values())

    def get_metrics(self) -> dict:
        return {
            "uptime_seconds": round(time.time() - self.started_at, 2),
            "active_sessions": len(self.sessions),
            "total_predictions": self.metrics.total_predictions,
            "static_predictions": self.metrics.static_predictions,
            # Retained at zero for frontend schema compatibility
            "dynamic_predictions": 0,
        }

    def shutdown(self) -> None:
        """Release resources during application shutdown."""
        session_count = len(self.sessions)
        self.sessions.clear()
        log.info("Prediction service shutdown — cleared %d sessions", session_count)
