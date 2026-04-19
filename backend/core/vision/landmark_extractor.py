"""
landmark_extractor.py
---------------------
Part of the AI Sign Language Communication System.

Converts raw MediaPipe hand landmarks into a flat numeric feature vector
suitable for downstream ML inference or gesture classification.

Location: src/vision/landmark_extractor.py
"""

import sys
import os

import cv2
import numpy as np

# Allow running directly from the project root or via `python src/vision/landmark_extractor.py`
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.vision.hand_detector import HandDetector


# Number of landmarks MediaPipe tracks per hand
NUM_LANDMARKS = 21

# Each landmark carries three coordinates (x, y, z)
COORDS_PER_LANDMARK = 3

# Total length of the resulting feature vector
FEATURE_VECTOR_LENGTH = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63


class LandmarkExtractor:
    """
    Extracts a flat numeric feature vector from MediaPipe hand detection results.

    MediaPipe tracks 21 landmarks per hand. Each landmark provides normalised
    (x, y, z) coordinates relative to the image dimensions. This class flattens
    those coordinates into a single 63-element NumPy array that can be fed
    directly into a machine-learning model.

    Feature vector layout:
        [x1, y1, z1,  x2, y2, z2,  ...  x21, y21, z21]
         ─── lm 0 ─── ─── lm 1 ───      ──── lm 20 ────
    """

    def extract(self, results) -> np.ndarray | None:
        """
        Build a 63-element feature vector from the first detected hand.

        Args:
            results: The detection results object returned by
                     ``HandDetector.detect_hands()``.  Specifically, this is
                     the ``mediapipe.solutions.hands.Hands.process()`` output.

        Returns:
            np.ndarray of shape (63,) and dtype float32 when a hand is
            detected, or ``None`` when no hand is present in the frame.
        """
        # Guard: return None immediately when MediaPipe found no hands
        if not results.multi_hand_landmarks:
            return None

        # Use only the first detected hand (max_num_hands=1 in HandDetector)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Pre-allocate the feature vector for efficiency
        feature_vector = np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)

        for index, landmark in enumerate(hand_landmarks.landmark):
            offset = index * COORDS_PER_LANDMARK
            feature_vector[offset]     = landmark.x  # Normalised horizontal position
            feature_vector[offset + 1] = landmark.y  # Normalised vertical position
            feature_vector[offset + 2] = landmark.z  # Depth relative to wrist

        return feature_vector


# ---------------------------------------------------------------------------
# Stand-alone test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Starting LandmarkExtractor test — press 'q' to quit.\n")

    detector  = HandDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    extractor = LandmarkExtractor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Exiting.")
        sys.exit(1)

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Dropped frame — retrying.")
            continue

        # Mirror the frame for a natural user experience
        frame = cv2.flip(frame, 1)

        # Step 1 — run MediaPipe hand detection
        results = detector.detect_hands(frame)

        # Step 2 — extract the feature vector
        vector = extractor.extract(results)

        # Step 3 — annotate the frame with landmarks
        frame = detector.draw_landmarks(frame, results)

        # Step 4 — console feedback and on-screen overlay
        if vector is not None:
            print(f"[INFO] Feature vector length: {len(vector)} | "
                  f"sample (first 9 values): {vector[:9].round(4)}")

            overlay_text  = f"Vector length: {len(vector)}"
            overlay_color = (0, 200, 0)      # green — hand detected
        else:
            overlay_text  = "No hand detected"
            overlay_color = (0, 0, 220)      # red — no hand

        cv2.putText(
            frame,
            overlay_text,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            overlay_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Landmark Extractor — Sign Language System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit signal received. Shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()