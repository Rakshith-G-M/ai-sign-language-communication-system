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

from core.ml.constants import RANDOM_SEED

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


def validate_label_encoder(encoder, *, min_classes: int = 1, label: str = "Label encoder") -> None:
    """Verify a fitted LabelEncoder exposes class labels."""
    classes = getattr(encoder, "classes_", None)
    if classes is None or len(classes) < min_classes:
        raise ValueError(f"{label}: invalid or empty classes_.")
    log.info("%s validated (%d classes).", label, len(classes))
