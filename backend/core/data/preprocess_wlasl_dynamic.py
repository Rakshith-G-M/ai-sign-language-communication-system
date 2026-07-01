"""
preprocess_wlasl_dynamic.py
────────────────────────────
Preprocess WLASL videos into the canonical dynamic gesture dataset format.

Pipeline
────────
    WLASL videos
        → MediaPipe Hands
        → orientation normalization
        → extract_hand_features_v2()
        → 134-D engineered features
        → temporal sampling / padding
        → JSONL dataset for train_dynamic_gesture.py

Each output row has the shape contract:
    {"label": str, "frames": List[List[float]]}
where frames is exactly SEQUENCE_LENGTH × TOTAL_FEATURES_V2.

Example
───────
    python -m core.data.preprocess_wlasl_dynamic \\
        --metadata /data/WLASL/WLASL_v0.3.json \\
        --videos_dir /data/WLASL/videos \\
        --output dataset/dynamic_gestures.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import mediapipe as mp
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from core.ml.constants import RANDOM_SEED, SEQUENCE_LENGTH, TOTAL_FEATURES_V2
from core.ml.dataset_validation import (
    report_class_distribution,
    validate_dynamic_jsonl,
    validate_dynamic_sequence,
)
from core.ml.landmark_utils import extract_v2_features_from_results, zero_feature_frame
from core.ml.training_utils import set_deterministic_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MP_HANDS = mp.solutions.hands
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
ZERO_FRAME = zero_feature_frame()


@dataclass(frozen=True)
class WlaslInstance:
    """A single WLASL video sample to preprocess."""

    video_id: str
    label: str
    split: str | None = None


def _normalise_split_filter(splits: Sequence[str] | None) -> set[str] | None:
    """Return a lower-case split filter or None when all splits are allowed."""
    if not splits:
        return None
    return {split.strip().lower() for split in splits if split.strip()}


def load_wlasl_instances(metadata_path: Path, splits: Sequence[str] | None = None) -> list[WlaslInstance]:
    """Load WLASL metadata and return video instances."""
    split_filter = _normalise_split_filter(splits)

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid WLASL metadata JSON: {metadata_path}") from exc

    if not isinstance(metadata, list):
        raise ValueError("WLASL metadata must be a JSON list of gloss entries.")

    instances: list[WlaslInstance] = []
    duplicate_video_ids: list[str] = []
    seen_video_ids: set[str] = set()

    for entry_index, entry in enumerate(metadata, start=1):
        if not isinstance(entry, dict):
            log.warning("Skipping metadata entry %d: expected object.", entry_index)
            continue

        label = str(entry.get("gloss", "")).strip()
        raw_instances = entry.get("instances", [])
        if not label or not isinstance(raw_instances, list):
            log.warning("Skipping metadata entry %d: missing gloss or instances.", entry_index)
            continue

        for raw_instance in raw_instances:
            if not isinstance(raw_instance, dict):
                continue

            video_id = str(raw_instance.get("video_id", "")).strip()
            split = raw_instance.get("split")
            split_text = str(split).strip().lower() if split is not None else None

            if not video_id:
                continue
            if video_id in seen_video_ids:
                duplicate_video_ids.append(video_id)
            seen_video_ids.add(video_id)

            if split_filter is not None and split_text not in split_filter:
                continue

            instances.append(WlaslInstance(video_id=video_id, label=label, split=split_text))

    if duplicate_video_ids:
        log.warning(
            "Duplicate video_id entries in metadata: %d occurrence(s) "
            "(examples: %s). Later entries are still processed.",
            len(duplicate_video_ids),
            sorted(set(duplicate_video_ids))[:5],
        )

    if not instances:
        raise ValueError("No WLASL instances matched the provided metadata and split filter.")

    log.info("Loaded %d WLASL instances from %s.", len(instances), metadata_path)
    return instances


def build_video_index(videos_dir: Path) -> dict[str, Path]:
    """Index video files under videos_dir by stem for fast video_id lookup."""
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    if not videos_dir.is_dir():
        raise NotADirectoryError(f"Videos path is not a directory: {videos_dir}")

    index: dict[str, Path] = {}
    for extension in VIDEO_EXTENSIONS:
        for path in videos_dir.rglob(f"*{extension}"):
            index.setdefault(path.stem, path)

    if not index:
        raise FileNotFoundError(f"No video files found under: {videos_dir}")

    log.info("Indexed %d video files under %s.", len(index), videos_dir)
    return index


def sample_or_pad_sequence(frames: Sequence[Sequence[float]], sequence_length: int) -> list[list[float]]:
    """Uniformly sample or zero-pad frames to a fixed sequence length."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if not frames:
        raise ValueError("Cannot sample or pad an empty sequence.")

    if len(frames) >= sequence_length:
        indices = np.linspace(0, len(frames) - 1, num=sequence_length, dtype=np.int64)
        return [list(frames[index]) for index in indices]

    padded = [list(frame) for frame in frames]
    padded.extend([ZERO_FRAME.copy() for _ in range(sequence_length - len(padded))])
    return padded


def extract_video_features(
    video_path: Path,
    hands: mp.solutions.hands.Hands,
    sequence_length: int,
) -> list[list[float]] | None:
    """Extract a fixed-length canonical feature sequence from one video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        log.warning("Skipping %s: OpenCV could not open video.", video_path)
        return None

    extracted_frames: list[list[float]] = []
    frame_errors = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame is None or frame.size == 0:
                frame_errors += 1
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error:
                frame_errors += 1
                log.warning("Skipping unreadable frame in %s.", video_path)
                continue

            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            features = extract_v2_features_from_results(results)
            if features is None:
                continue

            if features.shape != (TOTAL_FEATURES_V2,):
                log.warning(
                    "Skipping frame in %s: expected %d features, got %s.",
                    video_path, TOTAL_FEATURES_V2, features.shape,
                )
                continue

            extracted_frames.append(features.astype(np.float32).tolist())
    finally:
        capture.release()

    if frame_errors:
        log.debug("%s: skipped %d corrupted frame(s).", video_path.name, frame_errors)

    if not extracted_frames:
        log.warning("Skipping %s: no usable hand frames detected.", video_path)
        return None

    sequence = sample_or_pad_sequence(extracted_frames, sequence_length)
    validate_dynamic_sequence(sequence, sequence_length=sequence_length)
    return sequence


def write_records(
    instances: Iterable[WlaslInstance],
    video_index: dict[str, Path],
    output_path: Path,
    sequence_length: int,
    append: bool,
    max_samples_per_class: int | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> int:
    """Preprocess WLASL instances and write canonical dynamic JSONL records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    written = 0
    skipped_missing = 0
    skipped_failed = 0
    per_class_counts: dict[str, int] = defaultdict(int)

    with MP_HANDS.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands, output_path.open(mode, encoding="utf-8") as output_file:
        for instance in instances:
            if max_samples_per_class is not None and per_class_counts[instance.label] >= max_samples_per_class:
                continue

            video_path = video_index.get(instance.video_id)
            if video_path is None:
                skipped_missing += 1
                log.warning("Missing video for WLASL id %s (%s).", instance.video_id, instance.label)
                continue

            sequence = extract_video_features(video_path, hands, sequence_length)
            if sequence is None:
                skipped_failed += 1
                continue

            record = {"label": instance.label, "frames": sequence}
            output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            written += 1
            per_class_counts[instance.label] += 1

            if written % 25 == 0:
                log.info("Processed %d samples...", written)

    log.info(
        "Done. Wrote %d samples to %s. Missing videos: %d. Failed videos: %d.",
        written, output_path, skipped_missing, skipped_failed,
    )

    if written > 0:
        labels_written = [
            label for label, count in per_class_counts.items() for _ in range(count)
        ]
        report_class_distribution(labels_written, title="Written WLASL samples")
        if not append or output_path.exists():
            try:
                validate_dynamic_jsonl(output_path, min_samples_per_class=1)
                log.info("Output integrity check passed for %s.", output_path)
            except ValueError as exc:
                log.warning("Post-write validation warning: %s", exc)

    return written


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=f"Preprocess WLASL videos into canonical {TOTAL_FEATURES_V2}-D dynamic JSONL."
    )
    parser.add_argument("--metadata", required=True, type=Path, help="Path to WLASL_v*.json metadata.")
    parser.add_argument("--videos_dir", required=True, type=Path, help="Directory containing WLASL video files.")
    parser.add_argument(
        "--output",
        default=Path("dataset/dynamic_gestures.jsonl"),
        type=Path,
        help="Output JSONL path for train_dynamic_gesture.py.",
    )
    parser.add_argument(
        "--sequence_length",
        default=SEQUENCE_LENGTH,
        type=_positive_int,
        help=f"Frames per sample after sampling/padding (default: {SEQUENCE_LENGTH}).",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="WLASL split to include. Repeat for multiple splits. Default: all splits.",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=_positive_int,
        default=None,
        help="Optional cap per gloss label.",
    )
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwriting.")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED}).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    set_deterministic_seeds(args.seed)

    try:
        instances = load_wlasl_instances(args.metadata, args.splits)
        video_index = build_video_index(args.videos_dir)
        written = write_records(
            instances=instances,
            video_index=video_index,
            output_path=args.output,
            sequence_length=args.sequence_length,
            append=args.append,
            max_samples_per_class=args.max_samples_per_class,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    return 0 if written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
