"""
realtime_asl_predictor.py
──────────────────────────
Real-time ASL sign language recognition from a webcam feed.

Per-frame pipeline:
    BGR frame
      ↓  cv2.flip(frame, 1)                   mirror for natural view
      ↓  BGR → RGB
      ↓  MediaPipe Hands
      ↓  draw_landmarks(original)              true pixel coords on screen
      ↓  normalize_hand_landmarks_copy()       deep-copy + flip x if Left
      ↓  extract_hand_features_v2()            134-feature vector
      ↓  feature length guard                  skip frame on mismatch
      ↓  predict_proba() + confidence gate     threshold = 0.60
      ↓  prediction_buffer.append()            deque(maxlen=10)
      ↓  Counter.most_common(1)               majority vote
      ↓  draw_prediction_overlay()
      ↓  cv2.imshow()

Controls:
    Q  —  quit

Dependencies:
    pip install opencv-python mediapipe xgboost scikit-learn joblib numpy
"""

import sys
import logging
from collections import deque, Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import mediapipe as mp

from src.ml.feature_engineering import extract_hand_features_v2, TOTAL_FEATURES_V2

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
# Paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/asl_xgboost.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────────────
# Rolling window: 10 frames @ ~30 fps ≈ 0.33 s of history.
# Lower → faster sign transitions; higher → more stable display.
BUFFER_SIZE = 7

# Minimum predict_proba score for a prediction to enter the buffer.
# Frames below this are silently dropped — the last stable result stays shown.
# Tuning: 0.5 = permissive, 0.6 = recommended, 0.8 = strict
CONFIDENCE_THRESHOLD = 0.55

# Number of consecutive frames the majority-vote output must agree with the
# candidate letter before the displayed letter is updated.
# Acts as a second stability gate on top of the rolling buffer:
#   buffer   → filters single-frame noise     (short timescale)
#   hysteresis → filters sustained flickering (longer timescale)
# Tuning: 3 = responsive, 4 = recommended, 5–6 = very locked-in
STABILITY_THRESHOLD = 4

# ─────────────────────────────────────────────────────────────────────────────
# Overlay style
# ─────────────────────────────────────────────────────────────────────────────
CLR_WHITE     = (255, 255, 255)
CLR_BLACK     = (  0,   0,   0)
CLR_GREEN     = ( 50, 205,  50)
CLR_DARK_GRAY = ( 40,  40,  40)
CLR_AMBER     = (  0, 191, 255)

FONT           = cv2.FONT_HERSHEY_DUPLEX
FONT_LARGE     = 2.2
FONT_SMALL     = 0.65
THICKNESS_BOLD = 3
THICKNESS_THIN = 1

MP_DRAW   = mp.solutions.drawing_utils
MP_STYLES = mp.solutions.drawing_styles
MP_HANDS  = mp.solutions.hands


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model_artefacts():
    """
    Load the trained XGBClassifier and its LabelEncoder from disk.

    Returns:
        model   : Fitted XGBClassifier.
        encoder : Fitted LabelEncoder (int → ASL letter).

    Raises:
        FileNotFoundError : If either .pkl file is missing.
    """
    for path in (MODEL_PATH, ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {path.resolve()}\n"
                "Run train_asl_xgboost.py first."
            )

    log.info("Loading model   → %s", MODEL_PATH)
    log.info("Loading encoder → %s", ENCODER_PATH)

    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    log.info("Classes recognised: %s", list(encoder.classes_))
    return model, encoder


# ─────────────────────────────────────────────────────────────────────────────
# Orientation normalisation
# ─────────────────────────────────────────────────────────────────────────────
def normalize_hand_landmarks_copy(hand_landmarks, handedness: str | None):
    """
    Return an independent deep copy of hand_landmarks with x-coordinates
    horizontally flipped when handedness == "Left".

    Why a copy:
        MediaPipe's NormalizedLandmarkList is a protobuf object.  Mutating it
        in-place would corrupt the coordinates used by draw_landmarks(), causing
        a mirrored skeleton on screen.  The copy isolates prediction from drawing.

    Transformation for left hands only:
        x_new = 1.0 - x_old   (mirror in normalised [0, 1] space)
        y, z  unchanged

    Args:
        hand_landmarks : Original MediaPipe NormalizedLandmarkList — READ-ONLY.
        handedness     : "Left", "Right", or None (treated as "Right").

    Returns:
        Deep-copied (and possibly flipped) NormalizedLandmarkList.
    """
    from mediapipe.framework.formats import landmark_pb2
    normalised = landmark_pb2.NormalizedLandmarkList()
    normalised.CopyFrom(hand_landmarks)

    if handedness == "Left":
        for lm in normalised.landmark:
            lm.x = 1.0 - lm.x
        log.debug("Left hand — copy flipped for prediction.")

    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based gesture overrides
# ─────────────────────────────────────────────────────────────────────────────
def _lm_dist(lm, i: int, j: int) -> float:
    """
    Euclidean distance between landmarks i and j in raw [0, 1] MediaPipe space.

    Raw coordinates are used intentionally here — the rules were calibrated
    against raw distances, and using them avoids importing the full normalised
    coordinate pipeline into the rule layer.  The thresholds (0.15 / 0.20) are
    expressed as fractions of the normalised image width/height, which are
    consistent across different hand sizes because MediaPipe normalises
    coordinates to the bounding box of the detected hand.
    """
    dx = lm[i].x - lm[j].x
    dy = lm[i].y - lm[j].y
    dz = lm[i].z - lm[j].z
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def apply_gesture_rules(letter: str, hand_landmarks) -> str:
    """
    Apply lightweight rule-based overrides to correct systematic model errors
    on geometrically ambiguous letters.

    Rules are applied AFTER the confidence gate but BEFORE the prediction
    buffer so that corrected letters benefit from the full smoothing pipeline
    downstream.  The model prediction is used as a starting point — rules
    only fire when landmark geometry confirms a specific sign, preventing
    false positives on unrelated gestures.

    Current rules
    ─────────────
    Rule 1 — Detect "T"  (corrects T → A misclassification)
        The T sign tucks the thumb tip between the index and middle fingers.
        Signature: thumb tip is simultaneously close to both index tip AND
        middle tip.  In A, the thumb sits alongside the fist with no such
        proximity to middle.
            CONDITION : dist(thumb_tip, index_tip)  < 0.15
                        dist(thumb_tip, middle_tip) < 0.20
            OVERRIDE  : → "T"

    Rule 2 — Detect "P"  (corrects P → M / N misclassification)
        The P sign points the index finger downward below the knuckle line.
        M and N keep the fingers curled above or level with the MCPs.
            CONDITION : index_tip.y > index_mcp.y   (tip below knuckle in image)
                        dist(thumb_tip, middle_tip) < 0.20
            OVERRIDE  : → "P"

    Args:
        letter         : Model prediction after the confidence gate.
        hand_landmarks : Orientation-normalised NormalizedLandmarkList
                         (same object passed to extract_hand_features_v2).

    Returns:
        Overridden letter string, or the original letter if no rule fires.
    """
    lm = hand_landmarks.landmark

    # Pre-compute the three distances used across both rules — each is O(1)
    d_thumb_index  = _lm_dist(lm, 4,  8)   # THUMB_TIP  ↔ INDEX_TIP
    d_thumb_middle = _lm_dist(lm, 4, 12)   # THUMB_TIP  ↔ MIDDLE_TIP
    # ── Rule 1: T detection (FIXED — stricter + spatial check) ────────────────
    thumb_x  = lm[4].x   # THUMB_TIP
    index_x  = lm[8].x   # INDEX_TIP
    middle_x = lm[12].x  # MIDDLE_TIP

    # Thumb must be BETWEEN index and middle (true T geometry)
    is_between = min(index_x, middle_x) < thumb_x < max(index_x, middle_x)

    if letter in ["A", "S"] and (
        d_thumb_index < 0.12 and
        d_thumb_middle < 0.18 and
        is_between
    ):
        return "T"

    return letter   # no rule fired — return model prediction unchanged


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_sign(hand_landmarks, model, encoder) -> str | None:
    """
    Extract 134 features from orientation-normalised landmarks, apply a
    confidence gate, apply rule-based overrides, and return the final letter.

    Pipeline inside this function:
        extract_hand_features_v2()   →  134-feature vector
        length guard                 →  skip on mismatch
        predict_proba()              →  confidence gate (≥ CONFIDENCE_THRESHOLD)
        apply_gesture_rules()        →  T / P overrides   ← NEW
        return letter

    Args:
        hand_landmarks : Orientation-normalised NormalizedLandmarkList.
        model          : Fitted XGBClassifier.
        encoder        : Fitted LabelEncoder.

    Returns:
        ASL letter string (model output, possibly rule-overridden).
        None for degenerate pose / length mismatch / low confidence.
    """
    # ── Feature extraction ────────────────────────────────────────────────────
    features = extract_hand_features_v2(hand_landmarks)   # (134,) or None

    if features is None:
        log.debug("extract_hand_features_v2 returned None — skipping frame.")
        return None

    # ── Feature length guard ──────────────────────────────────────────────────
    # Catches version skew between feature_engineering.py and the saved model
    # before it causes a shape error inside XGBoost.
    if features.shape[0] != TOTAL_FEATURES_V2:
        log.warning(
            "Feature length %d ≠ %d (TOTAL_FEATURES_V2) — skipping frame.",
            features.shape[0], TOTAL_FEATURES_V2,
        )
        return None

    features_2d = features.reshape(1, -1)   # (1, 134) — sklearn API requires 2-D

    # ── Confidence-filtered inference ─────────────────────────────────────────
    # predict_proba() → (1, n_classes); argmax gives the winning class index
    # and its probability score in a single forward pass — no separate predict().
    proba      = model.predict_proba(features_2d)[0]   # (n_classes,)
    class_idx  = int(np.argmax(proba))
    confidence = float(proba[class_idx])

    letter_candidate = encoder.classes_[class_idx]
    log.debug("Candidate: %s  confidence: %.2f  threshold: %.2f",
              letter_candidate, confidence, CONFIDENCE_THRESHOLD)

    if confidence < CONFIDENCE_THRESHOLD:
        log.debug("Suppressed: %.2f < %.2f", confidence, CONFIDENCE_THRESHOLD)
        return None

    # ── Rule-based overrides ──────────────────────────────────────────────────
    # Applied after the confidence gate so rules only run on predictions the
    # model is already reasonably confident about.  Applied before the buffer
    # so corrected letters are smoothed identically to normal predictions.
    letter_candidate = apply_gesture_rules(letter_candidate, hand_landmarks)

    return str(letter_candidate)


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
def _draw_pill(frame, x: int, y: int, w: int, h: int, colour, alpha: float = 0.6):
    """Semi-transparent filled rectangle for overlay backgrounds."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_prediction_overlay(frame, letter: str | None) -> None:
    """
    Render the majority-vote ASL letter and a status bar onto the frame.

    Layout:
        ┌─────────────────────────────────────────┐
        │  Predicted Sign: A               [LIVE] │  ← top bar
        │                                         │
        │  A                                      │  ← large letter
        └─────────────────────────────────────────┘
    """
    h, w = frame.shape[:2]

    # Status bar
    _draw_pill(frame, 0, 0, w, 48, CLR_DARK_GRAY, alpha=0.7)

    if letter:
        cv2.putText(frame, f"Predicted Sign: {letter}",
                    (16, 33), FONT, FONT_SMALL, CLR_GREEN,
                    THICKNESS_THIN, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No hand detected",
                    (16, 33), FONT, FONT_SMALL, CLR_AMBER,
                    THICKNESS_THIN, cv2.LINE_AA)

    cv2.putText(frame, "LIVE",
                (w - 80, 33), FONT, FONT_SMALL, CLR_AMBER,
                THICKNESS_THIN, cv2.LINE_AA)

    # Large letter display
    if letter:
        lx, ly = 30, h // 2 + 60
        cv2.putText(frame, letter,
                    (lx + 3, ly + 3), FONT, FONT_LARGE, CLR_BLACK,
                    THICKNESS_BOLD + 2, cv2.LINE_AA)
        cv2.putText(frame, letter,
                    (lx, ly), FONT, FONT_LARGE, CLR_GREEN,
                    THICKNESS_BOLD, cv2.LINE_AA)


def draw_landmarks(frame, hand_landmarks) -> None:
    """Render MediaPipe hand skeleton on the BGR frame."""
    MP_DRAW.draw_landmarks(
        frame,
        hand_landmarks,
        MP_HANDS.HAND_CONNECTIONS,
        MP_STYLES.get_default_hand_landmarks_style(),
        MP_STYLES.get_default_hand_connections_style(),
    )


def draw_quit_hint(frame) -> None:
    """Small 'Q — quit' hint in the bottom-right corner."""
    h, w = frame.shape[:2]
    cv2.putText(frame, "Q  quit",
                (w - 110, h - 14), FONT, 0.5, CLR_WHITE,
                THICKNESS_THIN, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run_predictor(camera_index: int = 0) -> None:
    """
    Open the webcam and run the real-time ASL recognition loop.

    Stability layers (outermost → innermost):
        1. Confidence gate     — raw predictions below CONFIDENCE_THRESHOLD
                                 never enter the buffer.
        2. Rolling buffer      — majority vote over the last BUFFER_SIZE
                                 high-confidence frames smooths frame noise.
        3. Hysteresis lock     — the displayed letter only updates after the
                                 new majority-vote candidate has held steady
                                 for STABILITY_THRESHOLD consecutive frames,
                                 eliminating residual flicker from the buffer.

    Other design points:
        • cv2.flip(frame, 1) mirrors the frame for natural camera view.
        • Original landmarks → draw_landmarks()  (real pixel coords).
        • Deep-copied + normalised landmarks → predict_sign() (model coords).
        • All state is reset when no hand is detected.

    Args:
        camera_index : OpenCV camera device index (0 = default webcam).
    """
    model, encoder = load_model_artefacts()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera (index {camera_index}). "
            "Ensure a webcam is connected and not in use by another process."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    log.info(
        "Webcam opened  (buffer=%d, confidence=%.2f, stability=%d) — press Q to quit.",
        BUFFER_SIZE, CONFIDENCE_THRESHOLD, STABILITY_THRESHOLD,
    )

    # ── Layer 2: rolling majority-vote buffer ─────────────────────────────────
    prediction_buffer: deque[str] = deque(maxlen=BUFFER_SIZE)

    # ── Layer 3: hysteresis state ─────────────────────────────────────────────
    # current_stable_letter : the letter currently shown on screen.
    # candidate_letter      : the most recent majority-vote output.
    # stability_counter     : how many consecutive frames candidate has held.
    current_stable_letter: str | None = None
    candidate_letter:      str | None = None
    stability_counter:     int        = 0

    with MP_HANDS.Hands(
        static_image_mode=False,       # tracking mode — faster for video
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Empty frame — retrying …")
                continue

            # ── Mirror frame for natural camera view ──────────────────────────
            # Flipping here means the user sees a mirror image (left/right as
            # expected).  Orientation normalisation upstream handles the model
            # side — this flip is for display only.
            frame = cv2.flip(frame, 1)

            # ── BGR → RGB (MediaPipe requirement) ────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Hand detection ────────────────────────────────────────────────
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Handedness — optional; None treated as Right (no flip)
                handedness = None
                if results.multi_handedness:
                    handedness = (
                        results.multi_handedness[0]
                        .classification[0]
                        .label
                    )   # "Left" | "Right"

                # Draw skeleton from the ORIGINAL (unflipped) landmarks
                # so the skeleton aligns with what's actually on screen
                draw_landmarks(frame, hand_landmarks)

                # Deep copy + orientation flip for prediction only
                normalised_landmarks = normalize_hand_landmarks_copy(
                    hand_landmarks, handedness
                )

                # Inference with confidence gate (Layer 1)
                raw_letter = predict_sign(normalised_landmarks, model, encoder)

                if raw_letter is not None:
                    # Layer 2 — push into rolling buffer, compute majority vote
                    prediction_buffer.append(raw_letter)
                    vote = Counter(prediction_buffer).most_common(1)[0][0]
                else:
                    # Low-confidence / invalid — re-use last buffer vote
                    vote = (
                        Counter(prediction_buffer).most_common(1)[0][0]
                        if prediction_buffer else None
                    )

                # ── Layer 3: hysteresis lock ───────────────────────────────────
                # The displayed letter (current_stable_letter) only updates once
                # the new majority-vote output (vote) has been consistent for
                # STABILITY_THRESHOLD consecutive frames.  This prevents the
                # display from flickering when the buffer majority briefly
                # tips toward a different letter during sign transitions.
                if vote == candidate_letter:
                    # Same candidate as last frame — increment streak counter
                    stability_counter += 1
                else:
                    # New candidate appeared — restart the streak
                    candidate_letter  = vote
                    stability_counter = 1

                if stability_counter >= STABILITY_THRESHOLD:
                    # Candidate has held steady long enough → promote to display
                    current_stable_letter = candidate_letter
                    # Do NOT reset stability_counter here: keeping it ≥ threshold
                    # means the letter stays locked while the same sign holds,
                    # and only resets if a new candidate appears (branch above).

            else:
                # No hand detected — reset all three layers so the next sign
                # starts completely fresh with no stale state carried over
                prediction_buffer.clear()
                candidate_letter      = None
                stability_counter     = 0
                current_stable_letter = None

            # ── Render overlay and display ────────────────────────────────────
            draw_prediction_overlay(frame, current_stable_letter)
            draw_quit_hint(frame)
            cv2.imshow("ASL Recognition — Real-time", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Quit requested — shutting down.")
                break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run_predictor(camera_index=0)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)