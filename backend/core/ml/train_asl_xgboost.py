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
                   models/metadata/static_training_manifest.json

Usage:
    python -m core.ml.train_asl_xgboost
    python -m core.ml.train_asl_xgboost --dataset dataset/asl_landmarks_dataset.csv \\
                                        --model_dir models --test_size 0.2
"""

from __future__ import annotations

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

from core.ml.constants import RANDOM_SEED, STATIC_MODEL_VERSION, TOTAL_FEATURES_V2
from core.ml.dataset_validation import report_class_distribution, validate_static_csv
from core.ml.training_utils import (
    set_deterministic_seeds,
    validate_label_encoder,
    validate_saved_pickle,
    write_training_manifest,
)

DEFAULT_DATASET = "dataset/asl_landmarks_dataset.csv"
DEFAULT_MODEL_DIR = "models"
MODEL_FILENAME = "asl_xgboost.pkl"
ENCODER_FILENAME = "label_encoder.pkl"
MANIFEST_FILENAME = "static_training_manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and validate the static CSV dataset."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    log.info("Loading dataset from: %s", path.resolve())
    df = pd.read_csv(path)
    feature_cols = validate_static_csv(df)

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].astype(str).values

    assert X.shape[1] == TOTAL_FEATURES_V2

    log.info("Dataset shape   : %s", df.shape)
    log.info("  Samples       : %d", X.shape[0])
    log.info("  Features      : %d", X.shape[1])
    report_class_distribution(y_raw, title="Static training dataset")

    return X, y_raw, feature_cols


def encode_labels(y_raw: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """Fit a LabelEncoder and return integer-encoded labels."""
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    log.info("Label encoding  : %d classes → %s", len(encoder.classes_), list(encoder.classes_))
    return y, encoder


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    log.info(
        "Train/test split: %.0f%% / %.0f%%  (%d / %d samples)",
        (1 - test_size) * 100, test_size * 100, len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    n_estimators: int = 300,
    random_state: int = RANDOM_SEED,
) -> XGBClassifier:
    """Train an XGBClassifier with landmark-tuned hyperparameters."""
    log.info(
        "Training XGBClassifier  (n_estimators=%d, max_depth=6, lr=0.1, n_classes=%d) …",
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
        verbosity=0,
        use_label_encoder=False,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.2f seconds.", elapsed)
    return clf


def evaluate_model(
    clf: XGBClassifier,
    encoder: LabelEncoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Print evaluation metrics and return train/test accuracy."""
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    print()
    print("═" * 58)
    print("  MODEL EVALUATION")
    print("═" * 58)
    print(f"  Training accuracy  : {train_acc * 100:.2f}%")
    print(f"  Test     accuracy  : {test_acc * 100:.2f}%")
    print("─" * 58)
    print("  Classification Report (test set):")
    print()
    print(classification_report(
        y_test, clf.predict(X_test), target_names=encoder.classes_, zero_division=0,
    ))
    print("═" * 58)
    print()
    return train_acc, test_acc


def save_artefacts(
    clf: XGBClassifier,
    encoder: LabelEncoder,
    model_dir: str,
    *,
    dataset_path: str,
    train_acc: float,
    test_acc: float,
    n_samples: int,
) -> tuple[Path, Path]:
    """Serialise model artefacts and write a training manifest."""
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / MODEL_FILENAME
    encoder_path = out / ENCODER_FILENAME

    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)

    log.info("Model   saved → %s", model_path.resolve())
    log.info("Encoder saved → %s", encoder_path.resolve())

    validate_saved_pickle(model_path, XGBClassifier, "Static model")
    loaded_encoder = validate_saved_pickle(encoder_path, LabelEncoder, "Static label encoder")
    validate_label_encoder(loaded_encoder, label="Static label encoder")

    sample_prediction = clf.predict(np.zeros((1, TOTAL_FEATURES_V2), dtype=np.float32))
    log.info("Post-save smoke prediction: %s", sample_prediction)

    write_training_manifest(
        out,
        MANIFEST_FILENAME,
        {
            "model_version": STATIC_MODEL_VERSION,
            "model_type": "xgboost",
            "dataset_path": str(Path(dataset_path).resolve()),
            "feature_dim": TOTAL_FEATURES_V2,
            "n_samples": n_samples,
            "n_classes": len(encoder.classes_),
            "classes": list(encoder.classes_),
            "train_accuracy": round(float(train_acc), 6),
            "test_accuracy": round(float(test_acc), 6),
            "artefacts": {
                "model": MODEL_FILENAME,
                "label_encoder": ENCODER_FILENAME,
            },
        },
    )

    return model_path, encoder_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train an XGBoost ASL classifier on {TOTAL_FEATURES_V2}-feature v2 landmarks."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_deterministic_seeds(args.seed)

    X, y_raw, _ = load_dataset(args.dataset)
    y, encoder = encode_labels(y_raw)
    n_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=args.test_size, random_state=args.seed,
    )

    clf = train_model(
        X_train, y_train, n_classes=n_classes,
        n_estimators=args.n_estimators, random_state=args.seed,
    )

    train_acc, test_acc = evaluate_model(clf, encoder, X_train, y_train, X_test, y_test)

    save_artefacts(
        clf, encoder, args.model_dir,
        dataset_path=args.dataset,
        train_acc=train_acc,
        test_acc=test_acc,
        n_samples=len(X),
    )

    log.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, AssertionError, TypeError) as exc:
        log.error("Training failed: %s", exc)
        sys.exit(1)
