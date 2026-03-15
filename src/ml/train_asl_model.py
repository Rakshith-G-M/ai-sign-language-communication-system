"""
train_asl_model.py
------------------
Part of the AI Sign Language Communication System.

Loads the ASL landmark feature CSV, encodes the letter labels, trains a
Random Forest classifier, evaluates it, and saves both the fitted model
and the label encoder to disk for use in real-time inference.

Dataset : dataset/asl_landmarks_dataset.csv
Outputs : models/asl_alphabet_model.pkl
          models/asl_label_encoder.pkl

Location: src/ml/train_asl_model.py
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH   = os.path.join("dataset", "asl_landmarks_dataset.csv")
MODELS_DIR     = "models"
MODEL_PATH     = os.path.join(MODELS_DIR, "asl_alphabet_model.pkl")
ENCODER_PATH   = os.path.join(MODELS_DIR, "asl_label_encoder.pkl")

# Training hyper-parameters
N_ESTIMATORS  = 500
TEST_SIZE     = 0.2
RANDOM_STATE  = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_models_dir() -> None:
    """Create the models/ directory if it does not already exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the landmark CSV and return the feature matrix and raw string labels.

    Args:
        path: Path to asl_landmarks_dataset.csv.

    Returns:
        X: float32 array of shape (n_samples, 63) — landmark coordinates.
        y: object array of shape (n_samples,)      — letter strings e.g. 'A'.
    """
    print(f"[INFO] Loading dataset from : {path}")
    df = pd.read_csv(path)

    print(f"[INFO] Raw dataframe shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # All columns except 'label' are landmark features
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values   # string labels A–Z

    return X, y


def print_class_distribution(y_raw: np.ndarray) -> None:
    """Log the per-class sample counts for a quick data-quality check."""
    labels, counts = np.unique(y_raw, return_counts=True)
    print("\n[INFO] Class distribution:")
    print("-" * 30)
    for label, count in zip(labels, counts):
        bar = "█" * (count // 100)
        print(f"  {label} : {count:>5,}  {bar}")
    print("-" * 30 + "\n")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train() -> None:
    """
    End-to-end training pipeline:
      1. Load dataset
      2. Encode string labels → integers
      3. Train / test split
      4. Fit RandomForestClassifier
      5. Evaluate and print metrics
      6. Persist model and label encoder
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    X, y_raw = load_dataset(DATASET_PATH)
    print_class_distribution(y_raw)

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y_raw))

    print("=" * 55)
    print("  Dataset summary")
    print("=" * 55)
    print(f"  Total samples  : {n_samples:,}")
    print(f"  Features       : {n_features}  (21 landmarks × 3 coords)")
    print(f"  Classes        : {n_classes}  {sorted(np.unique(y_raw).tolist())}")
    print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # 2. Encode string labels (A–Z) → integer class indices
    # ------------------------------------------------------------------
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)   # e.g. 'A'→0, 'B'→1, …, 'Z'→25

    print(f"[INFO] Label encoding  : {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}\n")

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,        # preserve class proportions in both splits
    )

    print(f"[INFO] Training samples : {len(X_train):,}")
    print(f"[INFO] Test samples     : {len(X_test):,}\n")

    # ------------------------------------------------------------------
    # 4. Train model
    # ------------------------------------------------------------------
    print(f"[INFO] Training RandomForestClassifier "
          f"(n_estimators={N_ESTIMATORS}, n_jobs=-1, random_state={RANDOM_STATE}) …\n")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        n_jobs=-1,           # parallelise across all CPU cores
        random_state=RANDOM_STATE,
        verbose=1,           # scikit-learn's own progress output
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    print(f"\n[INFO] Training completed in {elapsed:.2f} seconds.\n")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)

    # Decode integer predictions back to letter strings for the report
    y_test_labels = encoder.inverse_transform(y_test)
    y_pred_labels = encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)

    print("=" * 55)
    print("  Evaluation results")
    print("=" * 55)
    print(f"  Accuracy : {accuracy * 100:.2f}%\n")
    print("  Classification report:")
    print("-" * 55)
    print(classification_report(y_test_labels, y_pred_labels))
    print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # 6. Save model and label encoder
    # ------------------------------------------------------------------
    ensure_models_dir()

    joblib.dump(clf, MODEL_PATH)
    print(f"[INFO] Model saved to        : {MODEL_PATH}  "
          f"({os.path.getsize(MODEL_PATH) / 1024:.1f} KB)")

    joblib.dump(encoder, ENCODER_PATH)
    print(f"[INFO] Label encoder saved to: {ENCODER_PATH}  "
          f"({os.path.getsize(ENCODER_PATH) / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()