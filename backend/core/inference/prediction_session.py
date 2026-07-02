"""
prediction_session.py
───────────────────────
Encapsulates all frame buffers, predictions, and text building state for a
single stream or user session. Eliminates the need for global state and
enables concurrent multi-client support.
"""

from collections import deque
import time
from core.inference.text_builder import TextBuilder


class PredictionSession:
    """
    State container for a single ASL static alphabet prediction session.

    Holds:
        - Rolling prediction buffer (majority vote — Layer 2)
        - Hysteresis stability counter (Layer 3)
        - TextBuilder for letter → word → sentence accumulation
        - Session activity timestamp for idle cleanup
    """

    def __init__(self, buffer_size: int = 12) -> None:
        # ── Static recognition buffers ────────────────────────────────────
        self.buffer: deque = deque(maxlen=buffer_size)
        self.candidate_letter: str | None = None
        self.stability_counter: int = 0
        self.stable_letter: str | None = None
        self.hand_missing_counter: int = 0

        # ── Text building state ───────────────────────────────────────────
        self.text_builder: TextBuilder = TextBuilder()

        # ── Session activity tracking ─────────────────────────────────────
        self.last_active_time: float = time.time()

    def touch(self) -> None:
        """Update last active timestamp to keep the session alive."""
        self.last_active_time = time.time()

    def reset_prediction_state(self) -> None:
        """Reset only prediction stability layers (e.g. when hand goes missing)."""
        self.buffer.clear()
        self.candidate_letter = None
        self.stability_counter = 0
        self.stable_letter = None

    def reset_all(self) -> None:
        """Hard reset of both prediction buffers and text building state."""
        self.reset_prediction_state()
        self.hand_missing_counter = 0
        self.text_builder.reset()
