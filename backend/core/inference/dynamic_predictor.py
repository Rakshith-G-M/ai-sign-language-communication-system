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
    shape       : (1, SEQUENCE_LENGTH, 63)   — batch=1, time=30, features=63
    dtype       : float32
    values      : StandardScaler-normalised landmark coordinates

    output name : "class_logits"
    shape       : (1, num_classes)
    dtype       : float32
    values      : raw logits — softmax applied here for confidence score
"""

import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports (training-only / production)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import joblib
except ImportError:
    joblib = None

# ─────────────────────────────────────────────────────────────────────────────
# Model artefact paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
_MODELS_DIR  = _BASE_DIR / "models"

DYNAMIC_MODEL_PATH   = _MODELS_DIR / "asl_dynamic.onnx"
ENCODER_PATH         = _MODELS_DIR / "dynamic_label_encoder.pkl"
SCALER_PATH          = _MODELS_DIR / "dynamic_scaler.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Inference constants
# ─────────────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30      # frames per gesture — must match training
INPUT_SIZE      = 63      # 21 landmarks × 3 coords
CONFIDENCE_GATE = 0.70    # minimum softmax probability to emit a prediction
MIN_HISTORY     = 10      # minimum frames needed before prediction is attempted


class DynamicPredictor:
    """
    Sequence-based predictor for ASL words.

    Automatically selects ONNX-backed ML inference when all three artefacts
    (model, encoder, scaler) are present.  Gracefully falls back to
    geometric rule-based detection when they are not.
    """

    def __init__(self) -> None:
        self.ort_session  = None
        self.encoder      = None
        self.scaler       = None
        self.model_loaded = False

        self._try_load_onnx_model()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────
    def _try_load_onnx_model(self) -> None:
        """
        Attempt to load all three artefacts required for ONNX inference.
        Sets self.model_loaded = True only when all three succeed.
        """
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
                "  Run generate_dynamic_dataset.py → train_dynamic_gesture.py to create them.",
                [str(p.name) for p in missing],
            )
            return

        try:
            self.ort_session = ort.InferenceSession(str(DYNAMIC_MODEL_PATH))
            self.encoder     = joblib.load(ENCODER_PATH)
            self.scaler      = joblib.load(SCALER_PATH)
            self.model_loaded = True
            log.info(
                "DynamicPredictor: ONNX model loaded. Classes: %s",
                list(self.encoder.classes_),
            )
        except Exception as exc:
            log.error("DynamicPredictor: failed to load ONNX artefacts: %s. Falling back.", exc)
            self.ort_session  = None
            self.encoder      = None
            self.scaler       = None
            self.model_loaded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def predict_sequence(self, landmarks_history) -> tuple[str | None, float]:
        """
        Predict a whole word from a deque of MediaPipe NormalizedLandmarkList objects.

        Args:
            landmarks_history : deque of NormalizedLandmarkList (or MockLandmarkList
                                in tests).  Length should be >= MIN_HISTORY.

        Returns:
            (word, confidence) or (None, 0.0) if no gesture is detected.
        """
        if not landmarks_history or len(landmarks_history) < MIN_HISTORY:
            return None, 0.0

        coords = self._extract_coords(landmarks_history)
        if len(coords) < MIN_HISTORY:
            return None, 0.0

        if self.model_loaded:
            return self._run_onnx_inference(coords)

        return self._run_mock_inference(coords)

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinate extraction helper
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_coords(self, landmarks_history) -> list[list[float]]:
        """Convert a history deque into a plain list of 63-float vectors."""
        coords = []
        for lm_list in landmarks_history:
            if lm_list and lm_list.landmark:
                frame_coords = []
                for lm in lm_list.landmark:
                    frame_coords.extend([lm.x, lm.y, lm.z])
                coords.append(frame_coords)
        return coords

    # ─────────────────────────────────────────────────────────────────────────
    # ONNX inference
    # ─────────────────────────────────────────────────────────────────────────
    def _run_onnx_inference(self, coords: list[list[float]]) -> tuple[str | None, float]:
        """
        Run the ONNX session on a fixed-length, scaled landmark sequence.

        Steps:
            1. Pad / truncate to SEQUENCE_LENGTH frames.
            2. Apply the saved StandardScaler.
            3. Run the ONNX session.
            4. Apply softmax and confidence gate.
            5. Decode the class index via LabelEncoder.
        """
        try:
            # 1. Fixed-length sequence
            seq = coords[-SEQUENCE_LENGTH:]                          # most recent frames
            while len(seq) < SEQUENCE_LENGTH:
                seq = [[0.0] * INPUT_SIZE] + seq                   # left-pad with zeros

            arr = np.array(seq, dtype=np.float32)                  # (T, 63)

            # 2. Scale
            flat   = arr.reshape(-1, INPUT_SIZE)
            scaled = self.scaler.transform(flat).astype(np.float32)
            arr    = scaled.reshape(1, SEQUENCE_LENGTH, INPUT_SIZE) # (1, T, 63)

            # 3. ONNX forward pass
            input_name  = self.ort_session.get_inputs()[0].name
            output_name = self.ort_session.get_outputs()[0].name
            logits      = self.ort_session.run([output_name], {input_name: arr})[0][0]

            # 4. Softmax + confidence gate
            exp_l      = np.exp(logits - logits.max())
            probs      = exp_l / exp_l.sum()
            best_idx   = int(np.argmax(probs))
            confidence = float(probs[best_idx])

            if confidence < CONFIDENCE_GATE:
                return None, 0.0

            # 5. Decode
            word = str(self.encoder.classes_[best_idx])
            log.debug("Dynamic ONNX: %s  conf=%.3f", word, confidence)
            return word, confidence

        except Exception as exc:
            log.error("ONNX inference error: %s", exc)
            return None, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Geometric rule-based fallback
    # ─────────────────────────────────────────────────────────────────────────
    def _run_mock_inference(self, coords: list[list[float]]) -> tuple[str | None, float]:
        """
        Lightweight trajectory analysis using wrist coordinates only.

        Rules
        ─────
        HELLO     : oscillating X trajectory (≥ 2 direction reversals) with
                    higher X variance than Y variance.
        THANK-YOU : sustained downward Y displacement (dy > 0.06) with higher
                    Y variance than X variance.
        """
        wrist = np.array([frame[:3] for frame in coords])   # (N, 3)
        x     = wrist[:, 0]
        y     = wrist[:, 1]

        dx    = float(x[-1] - x[0])
        dy    = float(y[-1] - y[0])
        std_x = float(np.std(x))
        std_y = float(np.std(y))

        # ── HELLO: horizontal wave ─────────────────────────────────────────
        if std_x > 0.015 and std_x > std_y * 1.5:
            sign_changes = int(np.sum(np.diff(np.sign(np.diff(x))) != 0))
            if sign_changes >= 2:
                return "HELLO", 0.85

        # ── THANK-YOU: downward swipe ──────────────────────────────────────
        if dy > 0.06 and std_y > 0.02 and std_y > std_x * 1.2:
            return "THANK-YOU", 0.90

        return None, 0.0
