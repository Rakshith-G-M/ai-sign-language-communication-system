"""
generate_landmark_dataset.py
─────────────────────────────
Builds the ASL landmark dataset CSV from a folder of hand-sign images.

For every image it:
    1. Loads the image with OpenCV.
    2. Detects the hand using MediaPipe Hands (static_image_mode=True).
    3. Reads handedness ("Left" / "Right") when available.
    4. Normalises left-hand landmarks: x = 1.0 - x  (horizontal mirror)
       so all samples match the right-hand orientation the model expects.
    5. Passes the (normalised) landmarks to extract_hand_features_v2().
    6. Appends the resulting 134-feature vector + label to the dataset.
    7. Skips images where no hand is detected or features are invalid.

Output
──────
    dataset/asl_landmarks_dataset.csv
    Columns: label, f1, f2, … f134   (135 columns total)

Expected input directory layout
────────────────────────────────
    data/
    ├── A/
    │   ├── img1.jpg
    │   └── ...
    ├── B/
    │   └── ...
    └── ...

Each sub-folder name becomes the label for every image inside it.

Usage
─────
    python -m core.data.generate_landmark_dataset
    python -m core.data.generate_landmark_dataset --data_dir data --out_dir dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from core.ml.constants import RANDOM_SEED, STATIC_LABEL_COLUMN, TOTAL_FEATURES_V2
from core.ml.dataset_validation import (
    find_duplicate_rows,
    report_class_distribution,
    static_feature_columns,
    validate_static_csv,
    warn_duplicates,
)
from core.ml.landmark_utils import extract_v2_features_from_landmarks
from core.ml.training_utils import set_deterministic_seeds

DEFAULT_DATA_DIR = "dataset/asl_alphabet_train"
DEFAULT_OUT_DIR = "dataset"
OUTPUT_FILENAME = "asl_landmarks_dataset.csv"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_hands_detector(
    static_image_mode: bool = True,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.3,
) -> mp.solutions.hands.Hands:
    """Return a configured MediaPipe Hands instance for still images."""
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
    )


def process_image(image_path: Path, hands: mp.solutions.hands.Hands) -> tuple[np.ndarray | None, str]:
    """
    Load one image, detect a hand, and extract the canonical v2 feature vector.

    Returns:
        (features, reason) where reason is one of: ok, load_error, no_hand,
        invalid_features.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        log.warning("Could not load image: %s — skipping.", image_path)
        return None, "load_error"

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None, "no_hand"

    hand_landmarks = results.multi_hand_landmarks[0]
    handedness = None
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label

    features = extract_v2_features_from_landmarks(hand_landmarks, handedness)
    if features is None:
        log.debug("Feature extraction returned None for: %s", image_path.name)
        return None, "invalid_features"

    if features.shape[0] != TOTAL_FEATURES_V2:
        log.warning(
            "Feature length %d ≠ %d for %s — skipping.",
            features.shape[0], TOTAL_FEATURES_V2, image_path.name,
        )
        return None, "invalid_features"

    return features, "ok"


def generate_dataset(
    data_dir: str,
    out_dir: str,
    *,
    min_samples_per_class: int = 1,
) -> Path:
    """
    Walk the data directory, extract features for every image, and save a CSV.

    Raises:
        FileNotFoundError: If data_dir does not exist.
        RuntimeError: If no valid samples were found.
        ValueError: If validation fails before save.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    label_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())
    if not label_dirs:
        raise RuntimeError(f"No sub-folders found inside: {data_path}")

    log.info("Found %d label class(es): %s", len(label_dirs), [d.name for d in label_dirs])

    feature_cols = static_feature_columns()
    columns = [STATIC_LABEL_COLUMN] + feature_cols
    rows: list[list] = []

    total_images = 0
    skipped_load = 0
    skipped_hand = 0
    skipped_invalid = 0

    with _build_hands_detector() as hands:
        for label_dir in label_dirs:
            label = label_dir.name
            image_files = sorted(
                f for f in label_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            if not image_files:
                log.warning("No images found in label folder: %s", label_dir)
                continue

            label_ok = 0
            for img_path in image_files:
                total_images += 1
                features, reason = process_image(img_path, hands)

                if features is None:
                    if reason == "load_error":
                        skipped_load += 1
                    elif reason == "no_hand":
                        skipped_hand += 1
                    else:
                        skipped_invalid += 1
                    continue

                rows.append([label] + features.tolist())
                label_ok += 1

            log.info("  %-6s  processed %4d / %4d images", label, label_ok, len(image_files))

    if not rows:
        raise RuntimeError(
            "No valid samples were generated. "
            "Check that hand images are present and detectable."
        )

    df = pd.DataFrame(rows, columns=columns)
    df[STATIC_LABEL_COLUMN] = df[STATIC_LABEL_COLUMN].astype(str)
    df[feature_cols] = df[feature_cols].astype(np.float32)

    duplicates = find_duplicate_rows(
        df[feature_cols].values,
        df[STATIC_LABEL_COLUMN].values,
    )
    warn_duplicates(duplicates, "Static dataset generation")

    validate_static_csv(df, min_samples_per_class=min_samples_per_class)
    report_class_distribution(df[STATIC_LABEL_COLUMN].values, title="Static dataset")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / OUTPUT_FILENAME
    df.to_csv(csv_path, index=False)

    log.info("─" * 55)
    log.info("Dataset generation complete")
    log.info("  Total images found   : %d", total_images)
    log.info("  Skipped (load error) : %d", skipped_load)
    log.info("  Skipped (no hand)    : %d", skipped_hand)
    log.info("  Skipped (invalid)    : %d", skipped_invalid)
    log.info("  Duplicate rows       : %d (warned)", len(duplicates))
    log.info("  Rows saved           : %d", len(df))
    log.info("  CSV shape            : %s  (%d label + %d feature columns)",
             df.shape, 1, TOTAL_FEATURES_V2)
    log.info("  Saved to             : %s", csv_path.resolve())
    log.info("─" * 55)

    return csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Generate {TOTAL_FEATURES_V2}-feature ASL landmark dataset from images."
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help=f"Root folder with per-label sub-folders (default: '{DEFAULT_DATA_DIR}')",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for the CSV (default: '{DEFAULT_OUT_DIR}')",
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=1,
        help="Minimum samples required per class before saving (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for deterministic processing (default: {RANDOM_SEED}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_deterministic_seeds(args.seed)

    try:
        generate_dataset(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            min_samples_per_class=args.min_samples_per_class,
        )
        sys.exit(0)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log.error("Dataset generation failed: %s", exc)
        sys.exit(1)
