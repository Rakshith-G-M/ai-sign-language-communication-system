"""Unit tests for shared ML pipeline validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.ml.constants import TOTAL_FEATURES_V2
from core.ml.dataset_validation import (
    find_duplicate_rows,
    static_feature_columns,
    validate_static_csv,
)
from core.ml.feature_engineering import validate_feature_vector_v2


def test_static_feature_columns_length():
    cols = static_feature_columns()
    assert len(cols) == TOTAL_FEATURES_V2
    assert cols[0] == "f1"
    assert cols[-1] == f"f{TOTAL_FEATURES_V2}"


def test_validate_static_csv_accepts_valid_frame():
    feature_cols = static_feature_columns()
    row = {col: 0.1 for col in feature_cols}
    row["label"] = "A"
    df = pd.DataFrame([row])
    returned = validate_static_csv(df)
    assert returned == feature_cols


def test_validate_static_csv_rejects_wrong_feature_count():
    df = pd.DataFrame({"label": ["A"], "f1": [0.1]})
    with pytest.raises(ValueError, match="Feature count mismatch"):
        validate_static_csv(df)


def test_find_duplicate_rows_warn_indices():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = np.array(["A", "A", "B"])
    duplicates = find_duplicate_rows(X, labels)
    assert duplicates == [1]


def test_validate_feature_vector_v2():
    vector = np.zeros(TOTAL_FEATURES_V2, dtype=np.float32)
    validate_feature_vector_v2(vector)

    with pytest.raises(ValueError, match="expected shape"):
        validate_feature_vector_v2(np.zeros(10, dtype=np.float32))
