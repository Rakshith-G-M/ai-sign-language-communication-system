"""
hand_detector.py
----------------
Improved version for real-time ASL system.
Fixes detection sensitivity, orientation, and robustness.
"""

import cv2
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.4,   # 🔥 lowered
        min_tracking_confidence: float = 0.4,    # 🔥 lowered
    ) -> None:

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ─────────────────────────────────────────────────────────────
    # Detection
    # ─────────────────────────────────────────────────────────────
    def detect_hands(self, frame):
        """
        Detect hands from BGR frame.
        """

        # 🔥 Flip to match frontend mirror
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Performance optimization
        rgb_frame.flags.writeable = False

        results = self.hands.process(rgb_frame)

        # 🔥 Proper boolean check
        hand_detected = bool(results.multi_hand_landmarks)

        return results, hand_detected, frame

    # ─────────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────────
    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style(),
                )
        return frame

    # ─────────────────────────────────────────────────────────────
    # Debug run (optional)
    # ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[ERROR] Webcam not accessible")
            return

        print("[INFO] Press 'q' to quit")

        while True:
            success, frame = cap.read()
            if not success:
                continue

            results, hand_detected, frame = self.detect_hands(frame)
            frame = self.draw_landmarks(frame, results)

            status = "Hand: Detected" if hand_detected else "Hand: Not Detected"
            color = (0, 200, 0) if hand_detected else (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = HandDetector()
    detector.run()