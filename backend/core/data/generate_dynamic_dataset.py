"""
generate_dynamic_dataset.py
────────────────────────────
Records sequences of hand landmarks from a webcam and saves them as a
dataset for training a dynamic sign / whole-word gesture classifier.

Pipeline
────────
    1. User presses SPACE to start a recording window.
    2. MediaPipe Hands captures N frames of landmarks per recording.
    3. Each frame is converted to the canonical 134-D v2 feature vector.
    4. The sequence is saved as a single row in a JSON-lines file:
           {"label": "HELLO", "frames": [[f1, …, f134], …]}
    5. Repeat until the desired sample count per class is reached.

Output
──────
    dataset/dynamic_gestures.jsonl
    Each line: {"label": str, "frames": List[List[float]]}
                where frames[i] is a TOTAL_FEATURES_V2 float list

Usage
─────
    python -m core.data.generate_dynamic_dataset --label HELLO --samples 50
    python -m core.data.generate_dynamic_dataset --label THANK-YOU --samples 50 --out_dir dataset

Controls
────────
    SPACE  —  begin recording
    Q      —  quit / stop collecting
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

from core.ml.constants import RANDOM_SEED, SEQUENCE_LENGTH, TOTAL_FEATURES_V2
from core.ml.dataset_validation import validate_dynamic_record
from core.ml.landmark_utils import extract_v2_features_from_landmarks, handedness_label, zero_feature_frame
from core.ml.training_utils import set_deterministic_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "dataset"
OUTPUT_FILENAME = "dynamic_gestures.jsonl"
RECORDING_WAIT_S = 0.5

MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils
MP_STYLES = mp.solutions.drawing_styles

FONT = cv2.FONT_HERSHEY_DUPLEX
CLR_GREEN = (50, 205, 50)
CLR_AMBER = (0, 191, 255)
CLR_RED = (0, 50, 220)
CLR_WHITE = (255, 255, 255)
CLR_DARK = (30, 30, 30)


def _draw_status(
    frame,
    label: str,
    collected: int,
    target: int,
    recording: bool,
    countdown: int | None,
) -> None:
    """Render a clean status overlay onto the frame (in-place)."""
    height, width = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 56), CLR_DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"Gesture: {label}", (14, 30), FONT, 0.7, CLR_AMBER, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Samples: {collected}/{target}", (14, 50), FONT, 0.55, CLR_WHITE, 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (width - 28, 28), 12, CLR_RED, cv2.FILLED)
        cv2.putText(frame, "REC", (width - 70, 32), FONT, 0.55, CLR_RED, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "SPACE=record  Q=quit", (14, height - 14), FONT, 0.5, CLR_WHITE, 1, cv2.LINE_AA)

    if countdown is not None and not recording:
        text = str(countdown)
        size = cv2.getTextSize(text, FONT, 4.0, 5)[0]
        center_x = (width - size[0]) // 2
        center_y = (height + size[1]) // 2
        cv2.putText(frame, text, (center_x + 3, center_y + 3), FONT, 4.0, CLR_DARK, 7, cv2.LINE_AA)
        cv2.putText(frame, text, (center_x, center_y), FONT, 4.0, CLR_GREEN, 5, cv2.LINE_AA)


def _frame_features(results) -> list[float]:
    """Extract one canonical v2 feature frame from MediaPipe results."""
    if not results.multi_hand_landmarks:
        return zero_feature_frame()

    hand_landmarks = results.multi_hand_landmarks[0]
    features = extract_v2_features_from_landmarks(hand_landmarks, handedness_label(results))
    if features is None:
        return zero_feature_frame()
    return features.tolist()


def record_gesture_sequences(
    label: str,
    target_samples: int,
    out_path: Path,
    camera_index: int = 0,
) -> int:
    """
    Open the webcam and guide the user through recording dynamic gesture samples.

    The recording UX is unchanged: SPACE starts capture, Q quits, and each
    sample stores SEQUENCE_LENGTH frames.  Only the internal representation
    migrated from raw 63-D landmarks to canonical 134-D v2 features.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index {camera_index}).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    log.info("Recording gesture: %s  —  target %d samples", label, target_samples)
    log.info("Press SPACE to record | Q to quit")

    with MP_HANDS.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        recording = False
        sequence: list[list[float]] = []
        start_time = None

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                MP_DRAW.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    MP_HANDS.HAND_CONNECTIONS,
                    MP_STYLES.get_default_hand_landmarks_style(),
                    MP_STYLES.get_default_hand_connections_style(),
                )

            if recording:
                sequence.append(_frame_features(results))

                frames_needed = SEQUENCE_LENGTH
                if len(sequence) >= frames_needed:
                    record = {"label": label, "frames": sequence[:frames_needed]}
                    try:
                        validate_dynamic_record(record)
                    except ValueError as exc:
                        log.warning("Skipping invalid recording: %s", exc)
                        recording = False
                        sequence = []
                        continue

                    with out_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")

                    written += 1
                    recording = False
                    sequence = []
                    log.info("  ✓ Recorded sample %d/%d", written, target_samples)

                    if written >= target_samples:
                        log.info("Target reached — exiting.")
                        break

            _draw_status(frame, label, written, target_samples, recording, countdown=None)
            cv2.imshow("Dynamic Gesture Recorder", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and not recording:
                recording = True
                sequence = []
                start_time = time.time()
                if RECORDING_WAIT_S:
                    time.sleep(RECORDING_WAIT_S)
                log.info("Recording …")

            elif key == ord("q"):
                log.info("Quit — %d samples collected.", written)
                break

    cap.release()
    cv2.destroyAllWindows()
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record dynamic ASL gesture sequences with canonical v2 features."
    )
    parser.add_argument("--label", required=True, help="Gesture class label, e.g. 'HELLO'.")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to collect.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_deterministic_seeds(args.seed)

    out = Path(args.out_dir) / OUTPUT_FILENAME
    try:
        written = record_gesture_sequences(
            label=args.label.upper(),
            target_samples=args.samples,
            out_path=out,
            camera_index=args.camera,
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Done. %d sample(s) written to %s (%d-D features).", written, out, TOTAL_FEATURES_V2)
    sys.exit(0 if written > 0 else 1)
