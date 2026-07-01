"""
test_prediction_session.py
───────────────────────────
Unit tests for ASL Prediction Session, Motion Classification,
Dynamic Prediction, and TextBuilder integration.
"""

import sys
from pathlib import Path
import unittest
from collections import deque
import numpy as np

# Add backend directory to path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_BACKEND_DIR))

from core.inference.prediction_session import PredictionSession
from core.inference.motion_classifier import is_hand_moving
from core.inference.dynamic_predictor import DynamicPredictor
from core.inference.text_builder import TextBuilder


class MockLandmark:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarkList:
    def __init__(self, landmarks: list[MockLandmark]):
        self.landmark = landmarks


class TestPredictionSessionAndPipeline(unittest.TestCase):
    
    def test_session_isolation(self) -> None:
        """Verify that two prediction sessions maintain completely separate state."""
        session_a = PredictionSession(buffer_size=5)
        session_b = PredictionSession(buffer_size=5)
        
        # Modify session A
        session_a.buffer.append("H")
        session_a.text_builder.current_word = "HE"
        
        # Assert Session B remains unmodified
        self.assertEqual(len(session_b.buffer), 0)
        self.assertEqual(session_b.text_builder.current_word, "")
        
        # Modify session B
        session_b.buffer.append("W")
        session_b.text_builder.current_word = "WO"
        
        self.assertEqual(list(session_a.buffer), ["H"])
        self.assertEqual(list(session_b.buffer), ["W"])

    def test_motion_classifier_stationary(self) -> None:
        """Verify that a stationary hand yields is_hand_moving = False."""
        history = deque(maxlen=10)
        
        # Generate stationary landmarks (tremors around a point)
        for i in range(10):
            # Wrist at index 0, others dummy
            wrist = MockLandmark(0.5 + (i % 2) * 0.001, 0.5 - (i % 2) * 0.001, 0.0)
            history.append(MockLandmarkList([wrist] + [MockLandmark(0, 0, 0)] * 20))
            
        self.assertFalse(is_hand_moving(history))

    def test_motion_classifier_moving(self) -> None:
        """Verify that a moving hand yields is_hand_moving = True."""
        history = deque(maxlen=10)
        
        # Generate moving landmarks (sweeping from x=0.2 to x=0.8)
        for i in range(10):
            wrist = MockLandmark(0.2 + i * 0.06, 0.5, 0.0)
            history.append(MockLandmarkList([wrist] + [MockLandmark(0, 0, 0)] * 20))
            
        self.assertTrue(is_hand_moving(history))

    def test_dynamic_predictor_hello_mock(self) -> None:
        """Verify the mock prediction of 'HELLO' from horizontal waving."""
        predictor = DynamicPredictor()
        history = deque(maxlen=20)
        
        # Oscillating hand on X axis
        for i in range(20):
            # Sine wave oscillation
            x_val = 0.5 + 0.04 * np.sin(i * 0.8)
            wrist = MockLandmark(x_val, 0.5, 0.0)
            history.append(MockLandmarkList([wrist] + [MockLandmark(0, 0, 0)] * 20))
            
        word, confidence = predictor.predict_sequence(history)
        self.assertEqual(word, "HELLO")
        self.assertGreater(confidence, 0.70)

    def test_dynamic_predictor_thank_you_mock(self) -> None:
        """Verify the mock prediction of 'THANK-YOU' from vertical swiping."""
        predictor = DynamicPredictor()
        history = deque(maxlen=20)
        
        # Moving downwards on Y axis (increasing Y coordinates)
        for i in range(20):
            wrist = MockLandmark(0.5, 0.3 + i * 0.015, 0.0)
            history.append(MockLandmarkList([wrist] + [MockLandmark(0, 0, 0)] * 20))
            
        word, confidence = predictor.predict_sequence(history)
        self.assertEqual(word, "THANK-YOU")
        self.assertGreater(confidence, 0.70)

    def test_text_builder_direct_word(self) -> None:
        """Verify that TextBuilder seamlessly integrates dynamic whole words."""
        tb = TextBuilder()
        
        # Simulate partial letter spelling of "I" (which is a valid dictionary word)
        tb.update("I", True, 100.0)
        tb.update("I", True, 101.0) # lock it in (needs > MIN_LETTER_DURATION which is 0.6)
        
        self.assertEqual(tb.current_word, "I")
        
        # Now receive a dynamic word (should commit typed letter "I" to sentence, and append "HELLO" directly)
        tb.update(None, True, 102.0, stable_word="HELLO")
        
        self.assertEqual(tb.current_word, "")
        self.assertEqual(tb.sentence, "I HELLO ")


if __name__ == "__main__":
    unittest.main()
