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
    6. Appends the resulting 126-feature vector + label to the dataset.
    7. Skips images where no hand is detected or features are invalid.

Output
──────
    dataset/asl_landmarks_dataset.csv
    Columns: label, f1, f2, … f126   (127 columns total)

Expected input directory layout
────────────────────────────────
    data/
    ├── A/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── B/
    │   └── ...
    └── ...

Each sub-folder name becomes the label for every image inside it.

Usage
─────
    python generate_landmark_dataset.py
    python generate_landmark_dataset.py --data_dir data --out_dir dataset
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ── Shared feature extractor: identical function used for dataset generation
#    AND real-time inference — guarantees consistent feature ordering.
#    v2 produces 126 features with wrist-origin / middle-MCP-scale normalisation.
from core.ml.feature_engineering import extract_hand_features_v2, TOTAL_FEATURES_V2

# Alias used throughout for readability — must equal _V2_STANDALONE_FEATURES
# in feature_engineering.py (126).
_EXPECTED_FEATURES = TOTAL_FEATURES_V2   # 126

# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "dataset/asl_alphabet_train"        # root folder containing per-label sub-folders
DEFAULT_OUT_DIR  = "dataset"     # where the CSV will be saved
OUTPUT_FILENAME  = "asl_landmarks_dataset.csv"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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
# MediaPipe initialisation  (done once for the whole run)
# ─────────────────────────────────────────────────────────────────────────────
def _build_hands_detector(
    static_image_mode: bool = True,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.3,
) -> mp.solutions.hands.Hands:
    """
    Return a configured MediaPipe Hands instance.

    static_image_mode=True  → optimised for still images (no temporal tracking).
    max_num_hands=1         → we only need the primary hand per image.
    """
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orientation normalisation helper
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_orientation(hand_landmarks, handedness: str | None):
    """
    Horizontally flip landmark x-coordinates for left hands so that every
    sample in the dataset has a consistent right-hand orientation.

    The training model expects right-hand geometry.  A left-hand sign is a
    mirror image: x_new = 1.0 - x_old maps it to the equivalent right-hand
    layout without touching y or z.

    Operates ON A COPY — the original MediaPipe object is never mutated so
    it can still be used for drawing or other purposes by the caller.

    Args:
        hand_landmarks : MediaPipe NormalizedLandmarkList (21 landmarks).
        handedness     : "Left", "Right", or None.
                         None is treated as "Right" (safer default — no flip).

    Returns:
        The (possibly flipped) landmark list; either the original object
        unchanged (Right/None) or a deep-copied and flipped version (Left).
    """
    if handedness != "Left":
        return hand_landmarks   # right hand or unknown → no change needed

    # Deep-copy via protobuf so we never corrupt the original object.
    from mediapipe.framework.formats import landmark_pb2
    normalised = landmark_pb2.NormalizedLandmarkList()
    normalised.CopyFrom(hand_landmarks)

    for lm in normalised.landmark:
        lm.x = 1.0 - lm.x   # horizontal mirror in normalised [0, 1] space

    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# Core per-image processing
# ─────────────────────────────────────────────────────────────────────────────
def process_image(image_path: Path, hands: mp.solutions.hands.Hands):
    """
    Load one image, run hand detection, normalise orientation, and extract
    the 126-feature v2 vector.

    Args:
        image_path : Path to the image file.
        hands      : Shared MediaPipe Hands instance (already initialised).

    Returns:
        numpy.ndarray shape (126,) on success.
        None if the image cannot be loaded, no hand is detected, or
        extract_hand_features_v2() returns None (degenerate pose / error).
    """
    # ── Load ─────────────────────────────────────────────────────────────────
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        log.warning("Could not load image: %s — skipping.", image_path)
        return None, "load_error"

    # MediaPipe requires RGB input
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Detect ───────────────────────────────────────────────────────────────
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None, "no_hand"  # no hand in frame — caller increments skipped_hand

    # Use only the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    # ── Read handedness (optional metadata) ──────────────────────────────────
    # multi_handedness may be absent even when landmarks are present (rare but
    # possible at low confidence).  Default to None → treated as right hand.
    handedness = None
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label

    # ── Orientation normalisation ─────────────────────────────────────────────
    # Flip left-hand x-coordinates so all samples share right-hand geometry.
    # Returns a deep copy for left hands; the original object for right hands.
    hand_landmarks = _normalise_orientation(hand_landmarks, handedness)

    # ── Feature extraction (v2 — 126 features) ───────────────────────────────
    # Returns None for degenerate poses (zero scale, wrong landmark count, etc.)
    # so invalid frames are skipped cleanly without a try/except in the caller.
    features = extract_hand_features_v2(hand_landmarks)   # shape (126,) or None

    if features is None:
        log.debug("Feature extraction returned None for: %s", image_path.name)
        return None, "invalid_features"

    # ── Final length guard ────────────────────────────────────────────────────
    # Catches any future version skew between feature_engineering.py and this
    # script before a silently wrong row enters the dataset.
    if features.shape[0] != _EXPECTED_FEATURES:
        log.warning(
            "Feature length %d ≠ %d for %s — skipping.",
            features.shape[0], _EXPECTED_FEATURES, image_path.name,
        )
        return None, "invalid_features"

    return features, "ok"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(data_dir: str, out_dir: str) -> Path:
    """
    Walk the data directory, extract features for every image, and save a CSV.

    Directory structure expected:
        <data_dir>/<LABEL>/<image_files>

    Args:
        data_dir : Root directory containing per-label sub-folders.
        out_dir  : Directory where the CSV will be written.

    Returns:
        Path to the saved CSV file.

    Raises:
        FileNotFoundError : If data_dir does not exist.
        RuntimeError      : If no valid samples were found.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Collect all label sub-folders (sorted for reproducibility)
    label_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not label_dirs:
        raise RuntimeError(f"No sub-folders found inside: {data_path}")

    log.info("Found %d label class(es): %s",
             len(label_dirs), [d.name for d in label_dirs])

    # ── Column names  ─────────────────────────────────────────────────────────
    # label  +  f1, f2, … f126  →  127 columns total
    feature_cols = [f"f{i+1}" for i in range(_EXPECTED_FEATURES)]   # ["f1".."f126"]
    columns      = ["label"] + feature_cols

    # Accumulate rows as plain Python lists for speed; build DataFrame at end
    rows = []

    total_images  = 0
    skipped_load  = 0   # images that couldn't be opened
    skipped_hand  = 0   # images where no hand was detected

    # ── Initialise MediaPipe once for the entire run ──────────────────────────
    with _build_hands_detector() as hands:

        for label_dir in label_dirs:
            label = label_dir.name   # folder name IS the ASL label (e.g. "A")

            # Gather all supported image files in this label folder
            image_files = [
                f for f in sorted(label_dir.iterdir())
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            if not image_files:
                log.warning("No images found in label folder: %s", label_dir)
                continue

            label_ok = 0   # successfully processed images for this label

            for img_path in image_files:
                total_images += 1

                features, reason = process_image(img_path, hands)

                if features is None:

                    if reason == "load_error":
                         skipped_load += 1
                    else:
                        skipped_hand += 1
                    continue

                # Build one CSV row: [label, f1, f2, … f126]
                row = [label] + features.tolist()
                rows.append(row)
                label_ok += 1

            log.info("  %-6s  processed %4d / %4d images",
                     label, label_ok, len(image_files))

    # ── Validate ──────────────────────────────────────────────────────────────
    if not rows:
        raise RuntimeError(
            "No valid samples were generated. "
            "Check that hand images are present and detectable."
        )

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=columns)

    # Enforce types: label stays string, all features are float32
    df["label"] = df["label"].astype(str)
    df[feature_cols] = df[feature_cols].astype(np.float32)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / OUTPUT_FILENAME

    df.to_csv(csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("─" * 55)
    log.info("Dataset generation complete")
    log.info("  Total images found   : %d", total_images)
    log.info("  Skipped (load error) : %d", skipped_load)
    log.info("  Skipped (no hand)    : %d", skipped_hand)
    log.info("  Rows saved           : %d", len(df))
    log.info("  Classes              : %s", sorted(df["label"].unique().tolist()))
    log.info("  CSV shape            : %s  (%d label + %d feature columns)",
             df.shape, 1, _EXPECTED_FEATURES)
    log.info("  Saved to             : %s", csv_path.resolve())
    log.info("─" * 55)

    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 126-feature ASL landmark dataset from images."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    try:
        csv_path = generate_dataset(data_dir=args.data_dir, out_dir=args.out_dir)
        sys.exit(0)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("Dataset generation failed: %s", exc)
        sys.exit(1)