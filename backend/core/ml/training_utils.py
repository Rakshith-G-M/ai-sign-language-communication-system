"""
training_utils.py
─────────────────
Shared training utilities: deterministic seeds, artefact validation,
and metadata manifest generation.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.ml.constants import RANDOM_SEED, TOTAL_FEATURES_V2

log = logging.getLogger(__name__)

METADATA_DIRNAME = "metadata"


def set_deterministic_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for numpy, Python random, and PyTorch when available."""
    np.random.seed(seed)
    random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metadata_dir(model_dir: str | Path) -> Path:
    """Return the metadata subdirectory inside a model directory."""
    path = Path(model_dir) / METADATA_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_training_manifest(model_dir: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    """Write a JSON training manifest under models/metadata/."""
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    out_path = metadata_dir(model_dir) / filename
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("Training manifest saved → %s", out_path.resolve())
    return out_path


def validate_saved_pickle(path: str | Path, expected_type: type | tuple[type, ...], label: str) -> Any:
    """Load a pickle artefact and verify its runtime type."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path.resolve()}")

    obj = joblib.load(file_path)
    if not isinstance(obj, expected_type):
        raise TypeError(
            f"{label} at {file_path} has unexpected type {type(obj).__name__}, "
            f"expected {expected_type}."
        )
    log.info("Validated %s → %s", label, file_path.resolve())
    return obj


def validate_scaler(scaler, *, expected_features: int = TOTAL_FEATURES_V2, label: str = "Scaler") -> None:
    """Verify a fitted StandardScaler matches the canonical feature dimension."""
    n_features = getattr(scaler, "n_features_in_", None)
    if n_features != expected_features:
        raise ValueError(
            f"{label}: expected n_features_in_={expected_features}, got {n_features}."
        )
    log.info("%s validated (%d features).", label, expected_features)


def validate_label_encoder(encoder, *, min_classes: int = 1, label: str = "Label encoder") -> None:
    """Verify a fitted LabelEncoder exposes class labels."""
    classes = getattr(encoder, "classes_", None)
    if classes is None or len(classes) < min_classes:
        raise ValueError(f"{label}: invalid or empty classes_.")
    log.info("%s validated (%d classes).", label, len(classes))


def validate_onnx_model(
    onnx_path: str | Path,
    *,
    input_shape: tuple[int, ...],
    output_names: list[str] | None = None,
) -> None:
    """
    Load an ONNX model with onnxruntime and verify input/output contracts.

    Raises:
        ImportError: When onnxruntime is not installed.
        ValueError: When input/output shapes or names do not match expectations.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX validation. Install via: pip install onnxruntime"
        ) from exc

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path.resolve()}")

    session = ort.InferenceSession(str(path))
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if not inputs:
        raise ValueError(f"ONNX model at {path} has no inputs.")

    actual_input_shape = tuple(dim if isinstance(dim, int) else -1 for dim in inputs[0].shape)
    expected = tuple(input_shape)

    if len(actual_input_shape) != len(expected):
        raise ValueError(
            f"ONNX input rank mismatch: expected {len(expected)}, got {len(actual_input_shape)}."
        )

    for index, (actual, expected_dim) in enumerate(zip(actual_input_shape, expected)):
        if expected_dim != -1 and actual not in (-1, expected_dim):
            raise ValueError(
                f"ONNX input dim {index}: expected {expected_dim}, got {actual}."
            )

    if output_names is not None:
        actual_output_names = [output.name for output in outputs]
        for name in output_names:
            if name not in actual_output_names:
                raise ValueError(
                    f"ONNX model missing output '{name}'. Found: {actual_output_names}."
                )

    dummy = np.zeros(input_shape, dtype=np.float32)
    session.run(None, {inputs[0].name: dummy})
    log.info("ONNX model validated → %s  input=%s", path.resolve(), input_shape)
