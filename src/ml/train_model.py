"""
train_model.py
--------------
Part of the AI Sign Language Communication System.

Loads the pre-extracted landmark feature CSV, trains a Random Forest
classifier to recognise hand gesture digits, evaluates it, and saves
the fitted model to disk for use in inference.

Dataset : dataset/digits_landmarks_dataset.csv
Output  : models/digit_model.pkl

Location: src/ml/train_model.py
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH  = os.path.join("dataset", "digits_landmarks_dataset.csv")
MODELS_DIR    = "models"
MODEL_PATH    = os.path.join(MODELS_DIR, "digit_model.pkl")

# Training hyper-parameters
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
N_ESTIMATORS  = 200


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_models_dir() -> None:
    """Create the models/ directory if it does not already exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the landmark CSV and split it into feature matrix X and label vector y.

    Args:
        path: Path to the CSV file produced by convert_dataset_to_landmarks.py.

    Returns:
        X: NumPy float32 array of shape (n_samples, 63) — landmark coordinates.
        y: NumPy int array of shape (n_samples,)        — digit class labels.
    """
    print(f"[INFO] Loading dataset from : {path}")
    df = pd.read_csv(path)

    print(f"[INFO] Raw dataframe shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[INFO] Class distribution :\n{df['label'].value_counts().sort_index().to_string()}\n")

    # All columns except 'label' are landmark features
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)

    return X, y


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Full training pipeline:
      1. Load dataset
      2. Train / test split
      3. Fit RandomForestClassifier
      4. Evaluate and print metrics
      5. Persist model to disk
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    X, y = load_dataset(DATASET_PATH)

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    print("=" * 55)
    print("  Dataset summary")
    print("=" * 55)
    print(f"  Total samples  : {n_samples:,}")
    print(f"  Features       : {n_features}  (21 landmarks × 3 coords)")
    print(f"  Classes        : {n_classes}  {sorted(np.unique(y).tolist())}")
    print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # preserve class proportions in both splits
    )

    print(f"[INFO] Training samples : {len(X_train):,}")
    print(f"[INFO] Test samples     : {len(X_test):,}\n")

    # ------------------------------------------------------------------
    # 3. Train model
    # ------------------------------------------------------------------
    print(f"[INFO] Training RandomForestClassifier  "
          f"(n_estimators={N_ESTIMATORS}, random_state={RANDOM_STATE}) …")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,           # use all available CPU cores
        verbose=1,           # print scikit-learn's own progress output
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    print(f"\n[INFO] Training completed in {elapsed:.2f} seconds.\n")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 55)
    print("  Evaluation results")
    print("=" * 55)
    print(f"  Accuracy : {accuracy * 100:.2f}%\n")
    print("  Classification report:")
    print("-" * 55)
    print(classification_report(y_test, y_pred))
    print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # 5. Save model
    # ------------------------------------------------------------------
    ensure_models_dir()
    joblib.dump(clf, MODEL_PATH)
    print(f"[INFO] Model saved to : {MODEL_PATH}")
    print(f"[INFO] Model file size: "
          f"{os.path.getsize(MODEL_PATH) / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()