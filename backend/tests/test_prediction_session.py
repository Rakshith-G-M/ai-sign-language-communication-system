"""
test_prediction_session.py
───────────────────────────
Unit tests for the static ASL alphabet prediction session and TextBuilder.
"""

import sys
from pathlib import Path
import unittest
import time

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_BACKEND_DIR))

from core.inference.prediction_session import PredictionSession
from core.inference.text_builder import TextBuilder


class TestPredictionSession(unittest.TestCase):

    def test_session_isolation(self) -> None:
        """Verify that two prediction sessions maintain completely separate state."""
        session_a = PredictionSession(buffer_size=5)
        session_b = PredictionSession(buffer_size=5)

        session_a.buffer.append("H")
        session_a.text_builder.current_word = "HE"

        self.assertEqual(len(session_b.buffer), 0)
        self.assertEqual(session_b.text_builder.current_word, "")

        session_b.buffer.append("W")
        session_b.text_builder.current_word = "WO"

        self.assertEqual(list(session_a.buffer), ["H"])
        self.assertEqual(list(session_b.buffer), ["W"])

    def test_reset_all_clears_state(self) -> None:
        """Verify reset_all() wipes buffers and text builder state."""
        session = PredictionSession(buffer_size=5)
        session.buffer.append("A")
        session.candidate_letter = "A"
        session.stability_counter = 3
        session.text_builder.current_word = "AB"
        session.text_builder.sentence = "HELLO "

        session.reset_all()

        self.assertEqual(len(session.buffer), 0)
        self.assertIsNone(session.candidate_letter)
        self.assertEqual(session.stability_counter, 0)
        self.assertEqual(session.text_builder.current_word, "")
        self.assertEqual(session.text_builder.sentence, "")


class TestTextBuilder(unittest.TestCase):

    def test_letter_accumulation_after_hold(self) -> None:
        """Letters held past MIN_LETTER_DURATION are appended to current_word."""
        tb = TextBuilder()
        tb.MIN_LETTER_DURATION = 0.0   # disable hold timer for test speed

        now = time.time()
        tb.update("H", True, now)        # DETECTING
        tb.update("H", True, now + 0.1)  # LOCKED → appends "H"

        self.assertEqual(tb.current_word, "H")

    def test_duplicate_suppression_while_held(self) -> None:
        """Same letter held continuously must not produce duplicates."""
        tb = TextBuilder()
        tb.MIN_LETTER_DURATION = 0.0

        now = time.time()
        tb.update("A", True, now)
        tb.update("A", True, now + 0.1)  # locks in "A"
        tb.update("A", True, now + 0.2)  # still held — no re-emission
        tb.update("A", True, now + 0.3)

        self.assertEqual(tb.current_word, "A")

    def test_word_commit_on_hand_absent(self) -> None:
        """Word is committed to sentence after hand is absent past SPACE_TIMEOUT."""
        tb = TextBuilder()
        tb.MIN_LETTER_DURATION = 0.0
        tb.SPACE_TIMEOUT = 0.1

        now = time.time()
        # Sign H then I so current_word becomes "HI" (exact dict word → kept as-is)
        tb.update("H", True, now)
        tb.update("H", True, now + 0.05)   # lock "H"
        tb.update("I", True, now + 0.06)   # switch to "I" — resets DETECTING
        tb.update("I", True, now + 0.12)   # lock "I"
        tb.update(None, False, now + 0.25) # hand absent > SPACE_TIMEOUT → commit

        self.assertEqual(tb.current_word, "")      # word flushed
        self.assertIn("HI", tb.sentence)           # "HI" is an exact dict match

    def test_update_returns_correct_tuple_type(self) -> None:
        """update() must accept exactly (stable_letter, hand_detected, timestamp)."""
        tb = TextBuilder()
        word, sentence, suggestions = tb.update(None, False, time.time())
        self.assertIsInstance(word, str)
        self.assertIsInstance(sentence, str)
        self.assertIsInstance(suggestions, list)

    def test_reset_clears_everything(self) -> None:
        """reset() wipes all state."""
        tb = TextBuilder()
        tb.MIN_LETTER_DURATION = 0.0
        now = time.time()
        tb.update("X", True, now)
        tb.update("X", True, now + 0.1)
        tb.reset()

        self.assertEqual(tb.current_word, "")
        self.assertEqual(tb.sentence, "")


if __name__ == "__main__":
    unittest.main()
