"""
generate_dynamic_dataset.py
────────────────────────────
Records sequences of hand landmarks from a webcam and saves them as a
dataset for training a dynamic sign / whole-word gesture classifier.

Pipeline
────────
    1. User presses SPACE to start a 2-second recording window.
    2. MediaPipe Hands captures N frames of landmarks per recording.
    3. Each frame's 21 × 3 = 63 normalised coordinates are stored.
    4. The sequence is saved as a single row in a JSON-lines file:
           {"label": "HELLO", "frames": [[x,y,z, …], …]}
    5. Repeat until the desired sample count per class is reached.

Output
──────
    dataset/dynamic_gestures.jsonl
    Each line: {"label": str, "frames": List[List[float]]}
                where frames[i] is a 63-float list  (21 lm × 3 coords)

Usage
─────
    python generate_dynamic_dataset.py --label HELLO --samples 50
    python generate_dynamic_dataset.py --label THANK-YOU --samples 50 --out_dir dataset

Controls
────────
    SPACE  —  begin recording
    Q      —  quit / stop collecting
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

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
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_OUT_DIR   = "dataset"
OUTPUT_FILENAME   = "dynamic_gestures.jsonl"
SEQUENCE_LENGTH   = 30          # frames to capture per gesture recording
RECORDING_WAIT_S  = 0.5         # pause before recording to avoid the press frame

MP_HANDS  = mp.solutions.hands
MP_DRAW   = mp.solutions.drawing_utils
MP_STYLES = mp.solutions.drawing_styles

FONT       = cv2.FONT_HERSHEY_DUPLEX
CLR_GREEN  = (50, 205, 50)
CLR_AMBER  = (0, 191, 255)
CLR_RED    = (0, 50, 220)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = (30, 30, 30)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def landmarks_to_frame(hand_landmarks, handedness: str | None) -> list[float]:
    """
    Convert a MediaPipe NormalizedLandmarkList to a flat 63-float list.
    Left hands are mirror-flipped on X so all samples are orientation-consistent.

    Returns:
        List of 63 floats: [x0,y0,z0, x1,y1,z1, …, x20,y20,z20]
    """
    coords = []
    for lm in hand_landmarks.landmark:
        x = 1.0 - lm.x if handedness == "Left" else lm.x
        coords.extend([x, lm.y, lm.z])
    return coords   # length 63


def _draw_status(frame, label: str, collected: int, target: int,
                 recording: bool, countdown: int | None) -> None:
    """Render a clean status overlay onto the frame (in-place)."""
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), CLR_DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Gesture label
    cv2.putText(frame, f"Gesture: {label}", (14, 30),
                FONT, 0.7, CLR_AMBER, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Samples: {collected}/{target}", (14, 50),
                FONT, 0.55, CLR_WHITE, 1, cv2.LINE_AA)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 28, 28), 12, CLR_RED, cv2.FILLED)
        cv2.putText(frame, "REC", (w - 70, 32), FONT, 0.55, CLR_RED, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "SPACE=record  Q=quit", (14, h - 14),
                    FONT, 0.5, CLR_WHITE, 1, cv2.LINE_AA)

    # Countdown overlay
    if countdown is not None and not recording:
        txt = str(countdown)
        sz  = cv2.getTextSize(txt, FONT, 4.0, 5)[0]
        cx  = (w - sz[0]) // 2
        cy  = (h + sz[1]) // 2
        cv2.putText(frame, txt, (cx + 3, cy + 3), FONT, 4.0, CLR_DARK, 7, cv2.LINE_AA)
        cv2.putText(frame, txt, (cx, cy),         FONT, 4.0, CLR_GREEN, 5, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Core recording loop
# ─────────────────────────────────────────────────────────────────────────────
def record_gesture_sequences(
    label: str,
    target_samples: int,
    out_path: Path,
    camera_index: int = 0,
) -> int:
    """
    Open the webcam and guide the user through recording dynamic gesture samples.

    Args:
        label          : String gesture class, e.g. "HELLO".
        target_samples : Number of samples to collect before exiting.
        out_path       : Path to the JSONL output file (appended, not overwritten).
        camera_index   : OpenCV camera device index.

    Returns:
        Total samples written to `out_path` during this session.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index {camera_index}).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
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
        recording   = False
        sequence    = []
        start_time  = None

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # ── Draw hand skeleton ─────────────────────────────────────────
            if results.multi_hand_landmarks:
                MP_DRAW.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    MP_HANDS.HAND_CONNECTIONS,
                    MP_STYLES.get_default_hand_landmarks_style(),
                    MP_STYLES.get_default_hand_connections_style(),
                )

            # ── Collect frames during recording window ─────────────────────
            if recording:
                if results.multi_hand_landmarks:
                    handedness = None
                    if results.multi_handedness:
                        handedness = results.multi_handedness[0].classification[0].label

                    frame_data = landmarks_to_frame(
                        results.multi_hand_landmarks[0], handedness
                    )
                    sequence.append(frame_data)
                else:
                    # No hand visible — pad with zeros to keep fixed length
                    sequence.append([0.0] * 63)

                # ── Check if recording window is complete ──────────────────
                elapsed = time.time() - start_time
                frames_needed = SEQUENCE_LENGTH
                if len(sequence) >= frames_needed:
                    # Save to JSONL
                    record = {"label": label, "frames": sequence[:frames_needed]}
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                    written  += 1
                    recording = False
                    sequence  = []
                    log.info("  ✓ Recorded sample %d/%d", written, target_samples)

                    if written >= target_samples:
                        log.info("Target reached — exiting.")
                        break

            _draw_status(frame, label, written, target_samples,
                         recording, countdown=None)
            cv2.imshow("Dynamic Gesture Recorder", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and not recording:
                recording  = True
                sequence   = []
                start_time = time.time()
                log.info("Recording …")

            elif key == ord("q"):
                log.info("Quit — %d samples collected.", written)
                break

    cap.release()
    cv2.destroyAllWindows()
    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record dynamic ASL gesture landmark sequences."
    )
    parser.add_argument(
        "--label", required=True,
        help="Gesture class label, e.g. 'HELLO' or 'THANK-YOU'.",
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Number of gesture samples to collect (default: 50).",
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera device index (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    out    = Path(args.out_dir) / OUTPUT_FILENAME
    written = record_gesture_sequences(
        label=args.label.upper(),
        target_samples=args.samples,
        out_path=out,
        camera_index=args.camera,
    )
    log.info("Done. %d sample(s) written to %s", written, out)
    sys.exit(0 if written > 0 else 1)
