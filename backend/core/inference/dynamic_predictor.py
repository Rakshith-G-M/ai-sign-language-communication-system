"""
dynamic_predictor.py
───────────────────────
Handles dynamic gesture and whole-word prediction from landmark sequences.

Architecture
────────────
    Priority 1 — ONNX Runtime (trained model):
        Loads asl_dynamic.onnx + dynamic_label_encoder.pkl + dynamic_scaler.pkl
        from the models/ directory.  When all three exist, real ML inference runs.

    Priority 2 — Rule-based geometric fallback:
        Used during development before any training data is collected.
        Recognises a small set of gestures from wrist trajectory statistics:
            HELLO     →  horizontal wave (oscillating X with high std)
            THANK-YOU →  vertical downward swipe

ONNX input contract  (must match train_dynamic_gesture.py export)
───────────────────────────────────────────────────────────────────
    input  name : "landmarks_sequence"
    shape       : (1, SEQUENCE_LENGTH, TOTAL_FEATURES_V2)
    dtype       : float32
    values      : StandardScaler-normalised v2 engineered features

    output name : "class_logits"
    shape       : (1, num_classes)
    dtype       : float32
    values      : raw logits — softmax applied here for confidence score
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.ml.constants import SEQUENCE_LENGTH, TOTAL_FEATURES_V2
from core.ml.landmark_utils import extract_v2_features_from_landmarks, zero_feature_frame
from core.ml.training_utils import validate_label_encoder, validate_scaler

log = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import joblib
except ImportError:
    joblib = None

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
_MODELS_DIR = _BASE_DIR / "models"

DYNAMIC_MODEL_PATH = _MODELS_DIR / "asl_dynamic.onnx"
ENCODER_PATH = _MODELS_DIR / "dynamic_label_encoder.pkl"
SCALER_PATH = _MODELS_DIR / "dynamic_scaler.pkl"

INPUT_SIZE = TOTAL_FEATURES_V2
CONFIDENCE_GATE = 0.70
MIN_HISTORY = 10


class DynamicPredictor:
    """
    Sequence-based predictor for ASL words.

    Automatically selects ONNX-backed ML inference when all three artefacts
    (model, encoder, scaler) are present.  Gracefully falls back to
    geometric rule-based detection when they are not.
    """

    def __init__(self) -> None:
        self.ort_session = None
        self.encoder = None
        self.scaler = None
        self.model_loaded = False

        self._try_load_onnx_model()

    def _try_load_onnx_model(self) -> None:
        """Attempt to load all three artefacts required for ONNX inference."""
        if ort is None:
            log.info("DynamicPredictor: onnxruntime not installed. Using geometric fallback.")
            return
        if joblib is None:
            log.info("DynamicPredictor: joblib not installed. Using geometric fallback.")
            return

        missing = [
            p for p in (DYNAMIC_MODEL_PATH, ENCODER_PATH, SCALER_PATH)
            if not p.exists()
        ]
        if missing:
            log.info(
                "DynamicPredictor: missing artefacts %s. Using geometric fallback.\n"
                "  Run preprocess_wlasl_dynamic.py → train_dynamic_gesture.py to create them.",
                [p.name for p in missing],
            )
            return

        try:
            self.ort_session = ort.InferenceSession(str(DYNAMIC_MODEL_PATH))
            self.encoder = joblib.load(ENCODER_PATH)
            self.scaler = joblib.load(SCALER_PATH)

            validate_scaler(self.scaler, expected_features=TOTAL_FEATURES_V2, label="Dynamic scaler")
            validate_label_encoder(self.encoder, label="Dynamic label encoder")

            input_meta = self.ort_session.get_inputs()[0]
            input_shape = tuple(dim if isinstance(dim, int) else -1 for dim in input_meta.shape)
            if len(input_shape) != 3:
                raise ValueError(f"Unexpected ONNX input rank: {input_shape}")

            if input_shape[1] not in (-1, SEQUENCE_LENGTH):
                raise ValueError(
                    f"ONNX sequence length mismatch: expected {SEQUENCE_LENGTH}, got {input_shape[1]}."
                )
            if input_shape[2] not in (-1, TOTAL_FEATURES_V2):
                raise ValueError(
                    f"ONNX feature dim mismatch: expected {TOTAL_FEATURES_V2}, got {input_shape[2]}."
                )

            self.model_loaded = True
            log.info(
                "DynamicPredictor: ONNX model loaded. Classes: %s  input=%s",
                list(self.encoder.classes_),
                input_shape,
            )
        except Exception as exc:
            log.error("DynamicPredictor: failed to load ONNX artefacts: %s. Falling back.", exc)
            self.ort_session = None
            self.encoder = None
            self.scaler = None
            self.model_loaded = False

    def readiness_info(self) -> dict:
        """Return readiness metadata for health probes."""
        return {
            "onnx_loaded": self.model_loaded,
            "mode": "onnx" if self.model_loaded else "geometric_fallback",
        }

    def predict_sequence(self, landmarks_history) -> tuple[str | None, float]:
        """
        Predict a whole word from a deque of MediaPipe NormalizedLandmarkList objects.

        Returns:
            (word, confidence) or (None, 0.0) if no gesture is detected.
        """
        if not landmarks_history or len(landmarks_history) < MIN_HISTORY:
            return None, 0.0

        if self.model_loaded:
            features = self._extract_feature_sequence(landmarks_history)
            if len(features) < MIN_HISTORY:
                return None, 0.0
            return self._run_onnx_inference(features)

        wrist_coords = self._extract_wrist_trajectory(landmarks_history)
        if len(wrist_coords) < MIN_HISTORY:
            return None, 0.0
        return self._run_mock_inference(wrist_coords)

    def _extract_feature_sequence(self, landmarks_history) -> list[list[float]]:
        """Convert landmark history into canonical v2 feature frames."""
        features = []
        zero_frame = zero_feature_frame()

        for lm_list in landmarks_history:
            if not lm_list or not getattr(lm_list, "landmark", None):
                features.append(zero_frame)
                continue

            vector = extract_v2_features_from_landmarks(lm_list, handedness=None)
            if vector is None:
                features.append(zero_frame)
            else:
                features.append(vector.astype(np.float32).tolist())

        return features

    def _extract_wrist_trajectory(self, landmarks_history) -> list[list[float]]:
        """Extract raw wrist coordinates for geometric fallback inference."""
        coords = []
        for lm_list in landmarks_history:
            if lm_list and lm_list.landmark:
                wrist = lm_list.landmark[0]
                coords.append([wrist.x, wrist.y, wrist.z])
        return coords

    def _run_onnx_inference(self, features: list[list[float]]) -> tuple[str | None, float]:
        """Run ONNX inference on a fixed-length, scaled v2 feature sequence."""
        try:
            seq = features[-SEQUENCE_LENGTH:]
            zero_frame = zero_feature_frame()
            while len(seq) < SEQUENCE_LENGTH:
                seq = [zero_frame] + seq

            arr = np.array(seq, dtype=np.float32)

            flat = arr.reshape(-1, INPUT_SIZE)
            scaled = self.scaler.transform(flat).astype(np.float32)
            arr = scaled.reshape(1, SEQUENCE_LENGTH, INPUT_SIZE)

            input_name = self.ort_session.get_inputs()[0].name
            output_name = self.ort_session.get_outputs()[0].name
            logits = self.ort_session.run([output_name], {input_name: arr})[0][0]

            exp_l = np.exp(logits - logits.max())
            probs = exp_l / exp_l.sum()
            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])

            if confidence < CONFIDENCE_GATE:
                return None, 0.0

            word = str(self.encoder.classes_[best_idx])
            log.debug("Dynamic ONNX: %s  conf=%.3f", word, confidence)
            return word, confidence

        except Exception as exc:
            log.error("ONNX inference error: %s", exc)
            return None, 0.0

    def _run_mock_inference(self, wrist_coords: list[list[float]]) -> tuple[str | None, float]:
        """
        Lightweight trajectory analysis using wrist coordinates only.

        Geometric fallback behaviour is unchanged from the legacy implementation.
        """
        wrist = np.array(wrist_coords)
        x = wrist[:, 0]
        y = wrist[:, 1]

        dy = float(y[-1] - y[0])
        std_x = float(np.std(x))
        std_y = float(np.std(y))

        if std_x > 0.015 and std_x > std_y * 1.5:
            sign_changes = int(np.sum(np.diff(np.sign(np.diff(x))) != 0))
            if sign_changes >= 2:
                return "HELLO", 0.85

        if dy > 0.06 and std_y > 0.02 and std_y > std_x * 1.2:
            return "THANK-YOU", 0.90

        return None, 0.0
