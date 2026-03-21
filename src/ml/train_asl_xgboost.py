"""
train_asl_xgboost.py
─────────────────────
Trains an XGBoost classifier on the 134-feature ASL landmark dataset
produced by generate_landmark_dataset.py + extract_hand_features_v2().

Pipeline:
    1. Load     →  dataset/asl_landmarks_dataset.csv
    2. Validate →  exactly 134 feature columns  (f1 … f134)
    3. Encode   →  LabelEncoder  (A–Z → 0–25)
    4. Split    →  stratified 80 / 20 train-test  (random_state=42)
    5. Train    →  XGBClassifier  (300 trees, lr=0.1, depth=6)
    6. Evaluate →  train accuracy + test accuracy + classification report
    7. Save     →  models/asl_xgboost.pkl
                   models/label_encoder.pkl

Usage:
    python train_asl_xgboost.py
    python train_asl_xgboost.py --dataset dataset/asl_landmarks_dataset.csv \\
                                 --model_dir models --test_size 0.2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.ml.feature_engineering import TOTAL_FEATURES_V2

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET   = "dataset/asl_landmarks_dataset.csv"
DEFAULT_MODEL_DIR = "models"
MODEL_FILENAME    = "asl_xgboost.pkl"
ENCODER_FILENAME  = "label_encoder.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading & validation
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the CSV, validate feature count, and return X, y_raw, feature_names.

    Args:
        csv_path: Path to the CSV file (label, f1 … f134).

    Returns:
        X            : float32 array of shape (n_samples, 134)
        y_raw        : string array of shape (n_samples,)  — raw label strings
        feature_cols : list of feature column names ["f1" … "f134"]

    Raises:
        FileNotFoundError : CSV not found.
        ValueError        : Feature column count ≠ TOTAL_FEATURES_V2 (134).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    log.info("Loading dataset from: %s", path.resolve())
    df = pd.read_csv(path)

    # ── Schema check ─────────────────────────────────────────────────────────
    if "label" not in df.columns:
        raise ValueError("CSV is missing the required 'label' column.")

    feature_cols = [c for c in df.columns if c != "label"]

    if len(feature_cols) != TOTAL_FEATURES_V2:
        raise ValueError(
            f"Feature count mismatch: expected {TOTAL_FEATURES_V2} features "
            f"(TOTAL_FEATURES_V2), but CSV contains {len(feature_cols)}. "
            "Re-run generate_landmark_dataset.py with the current "
            "extract_hand_features_v2() to rebuild the dataset."
        )

    # ── Extract arrays ────────────────────────────────────────────────────────
    X     = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].astype(str).values

    # Hard assertion — second safety net catches shape bugs immediately
    assert X.shape[1] == TOTAL_FEATURES_V2, (
        f"X.shape[1] == {X.shape[1]}, expected {TOTAL_FEATURES_V2}."
    )

    log.info("Dataset shape   : %s", df.shape)
    log.info("  Samples       : %d", X.shape[0])
    log.info("  Features      : %d", X.shape[1])
    log.info("  Classes       : %s", sorted(set(y_raw)))

    return X, y_raw, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. Label encoding
# ─────────────────────────────────────────────────────────────────────────────
def encode_labels(y_raw: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """
    Fit a LabelEncoder and return integer-encoded labels.

    XGBoost requires integer class labels starting from 0.
    The fitted encoder must be saved alongside the model so that
    encoder.inverse_transform() can recover letter strings at inference.

    Args:
        y_raw : 1-D string array  (e.g. ["A", "B", "A", …]).

    Returns:
        y       : 1-D int array of encoded labels.
        encoder : Fitted LabelEncoder.
    """
    encoder = LabelEncoder()
    y       = encoder.fit_transform(y_raw)

    log.info("Label encoding  : %d classes → %s",
             len(encoder.classes_), list(encoder.classes_))
    return y, encoder


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / test split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split preserving class distribution in both partitions.

    Args:
        X            : Feature matrix  (n_samples, 134).
        y            : Encoded label vector.
        test_size    : Fraction held out for testing.
        random_state : RNG seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    log.info("Train/test split: %.0f%% / %.0f%%  (%d / %d samples)",
             (1 - test_size) * 100, test_size * 100,
             len(X_train), len(X_test))

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    n_estimators: int = 300,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Train an XGBClassifier with hyperparameters tuned for landmark data.

    Hyperparameter rationale:
        n_estimators=300    — more trees than RF default; XGBoost is less
                              prone to overfitting per additional tree due
                              to gradient correction.
        max_depth=6         — moderate depth; ASL gesture boundaries are
                              non-linear but not extremely deep.
        learning_rate=0.1   — standard conservative step size.
        subsample=0.8       — row sub-sampling adds stochasticity,
                              improves generalisation.
        colsample_bytree=0.8 — feature sub-sampling per tree; works well
                               with the 134 correlated landmark features.
        eval_metric=mlogloss — multi-class log-loss, appropriate for 26
                               softmax output classes.
        n_jobs=-1           — parallel across all CPU cores.

    Args:
        X_train      : Training features (n_samples, 134).
        y_train      : Encoded integer labels.
        n_classes    : Number of unique classes (26 for full A–Z).
        n_estimators : Number of boosting rounds.
        random_state : RNG seed.

    Returns:
        Fitted XGBClassifier.
    """
    log.info(
        "Training XGBClassifier  "
        "(n_estimators=%d, max_depth=6, lr=0.1, n_classes=%d) …",
        n_estimators, n_classes,
    )

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        num_class=n_classes,
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,            # suppress XGBoost's own progress output
        use_label_encoder=False,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    log.info("Training complete in %.2f seconds.", elapsed)
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    clf: XGBClassifier,
    encoder: LabelEncoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Print training accuracy, test accuracy, and a full per-class report.

    Args:
        clf              : Fitted XGBClassifier.
        encoder          : Fitted LabelEncoder (supplies class name strings).
        X_train, y_train : Training partition.
        X_test,  y_test  : Test partition.
    """
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test))

    print()
    print("═" * 58)
    print("  MODEL EVALUATION")
    print("═" * 58)
    print(f"  Training accuracy  : {train_acc * 100:.2f}%")
    print(f"  Test     accuracy  : {test_acc  * 100:.2f}%")
    print("─" * 58)
    print("  Classification Report (test set):")
    print()
    print(
        classification_report(
            y_test,
            clf.predict(X_test),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )
    print("═" * 58)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save artefacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artefacts(
    clf: XGBClassifier,
    encoder: LabelEncoder,
    model_dir: str,
) -> tuple[Path, Path]:
    """
    Serialise the trained classifier and label encoder to disk with joblib.

    Both files must be loaded together at inference time:
        clf     = joblib.load("models/asl_xgboost.pkl")
        encoder = joblib.load("models/label_encoder.pkl")
        label   = encoder.inverse_transform(clf.predict([features]))[0]

    Args:
        clf       : Fitted XGBClassifier.
        encoder   : Fitted LabelEncoder.
        model_dir : Output directory.

    Returns:
        (model_path, encoder_path)
    """
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path   = out / MODEL_FILENAME
    encoder_path = out / ENCODER_FILENAME

    joblib.dump(clf,     model_path)
    joblib.dump(encoder, encoder_path)

    log.info("Model   saved → %s", model_path.resolve())
    log.info("Encoder saved → %s", encoder_path.resolve())

    return model_path, encoder_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost ASL classifier on 134-feature v2 landmarks."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to the CSV dataset (default: '{DEFAULT_DATASET}')",
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Output directory for saved models (default: '{DEFAULT_MODEL_DIR}')",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test fraction (default: 0.2)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="Number of boosting rounds (default: 300)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    # 1. Load & validate
    X, y_raw, _ = load_dataset(args.dataset)

    # 2. Encode labels
    y, encoder = encode_labels(y_raw)
    n_classes  = len(encoder.classes_)

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)

    # 4. Train
    clf = train_model(X_train, y_train, n_classes=n_classes,
                      n_estimators=args.n_estimators)

    # 5. Evaluate
    evaluate_model(clf, encoder, X_train, y_train, X_test, y_test)

    # 6. Save
    save_artefacts(clf, encoder, args.model_dir)

    log.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        log.error("Training failed: %s", exc)
        sys.exit(1)