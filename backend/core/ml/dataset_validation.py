"""
dataset_validation.py
─────────────────────
Dataset validation helpers for the static CSV landmark pipeline.
Duplicate detection warns by default; callers may choose to fail on duplicates.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.ml.constants import (
    STATIC_FEATURE_PREFIX,
    STATIC_LABEL_COLUMN,
    TOTAL_FEATURES_V2,
)

log = logging.getLogger(__name__)


def static_feature_columns() -> list[str]:
    """Return canonical static CSV feature column names f1 … fN."""
    return [f"{STATIC_FEATURE_PREFIX}{i + 1}" for i in range(TOTAL_FEATURES_V2)]


def report_class_distribution(labels: Iterable[str], title: str = "Class distribution") -> dict[str, int]:
    """Log and return per-label sample counts."""
    counts = dict(sorted(Counter(labels).items()))
    log.info("%s (%d classes, %d samples):", title, len(counts), sum(counts.values()))
    for label, count in counts.items():
        log.info("  %-20s  %6d", label, count)
    return counts


def find_duplicate_rows(
    X: np.ndarray,
    labels: np.ndarray | None = None,
) -> list[int]:
    """
    Return indices of duplicate feature rows (keeping the first occurrence).

    When labels are provided, duplicates are detected within the same label only.
    """
    seen: set[bytes] = set()
    duplicate_indices: list[int] = []

    for index, row in enumerate(X):
        label_suffix = b""
        if labels is not None:
            label_suffix = str(labels[index]).encode("utf-8")
        key = label_suffix + row.tobytes()

        if key in seen:
            duplicate_indices.append(index)
        else:
            seen.add(key)

    return duplicate_indices


def warn_duplicates(
    duplicate_indices: list[int],
    context: str,
    fail_on_duplicates: bool = False,
) -> None:
    """Warn about duplicate rows; optionally raise when duplicates are invalid."""
    if not duplicate_indices:
        return

    message = (
        f"{context}: found {len(duplicate_indices)} duplicate row(s) "
        f"(indices include {duplicate_indices[:5]}…)"
    )
    if fail_on_duplicates:
        raise ValueError(message)
    log.warning(message)


def validate_finite_array(array: np.ndarray, context: str) -> None:
    """Raise ValueError when array contains NaN or infinite values."""
    if not np.isfinite(array).all():
        raise ValueError(f"{context}: contains NaN or infinite values.")


def validate_static_csv(
    df: pd.DataFrame,
    *,
    fail_on_duplicates: bool = False,
    min_samples_per_class: int = 1,
) -> list[str]:
    """
    Validate a static landmark CSV DataFrame.

    Returns:
        Canonical feature column names.

    Raises:
        ValueError: On schema, numeric, or class-count violations.
    """
    if STATIC_LABEL_COLUMN not in df.columns:
        raise ValueError(f"CSV is missing required column '{STATIC_LABEL_COLUMN}'.")

    feature_cols = [c for c in df.columns if c != STATIC_LABEL_COLUMN]
    expected_cols = static_feature_columns()

    if len(feature_cols) != TOTAL_FEATURES_V2:
        raise ValueError(
            f"Feature count mismatch: expected {TOTAL_FEATURES_V2} features, "
            f"but CSV contains {len(feature_cols)}. "
            "Re-run generate_landmark_dataset.py with the current pipeline."
        )

    if feature_cols != expected_cols:
        raise ValueError(
            "Feature column names must be f1 … "
            f"f{TOTAL_FEATURES_V2} in order. "
            f"Got unexpected columns: {feature_cols[:5]}…"
        )

    labels = df[STATIC_LABEL_COLUMN].astype(str).values
    if (labels == "").any() or pd.isna(df[STATIC_LABEL_COLUMN]).any():
        raise ValueError("CSV contains empty or missing labels.")

    X = df[feature_cols].values.astype(np.float32)
    validate_finite_array(X, "Static dataset features")

    duplicates = find_duplicate_rows(X, labels)
    warn_duplicates(duplicates, "Static dataset", fail_on_duplicates=fail_on_duplicates)

    class_counts = Counter(labels)
    empty_classes = [label for label, count in class_counts.items() if count < min_samples_per_class]
    if empty_classes:
        raise ValueError(
            f"Classes with fewer than {min_samples_per_class} sample(s): "
            f"{sorted(empty_classes)}"
        )

    return feature_cols

