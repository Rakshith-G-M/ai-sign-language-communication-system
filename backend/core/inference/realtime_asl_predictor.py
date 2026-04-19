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
import time                                          # ← ADDED
import logging
from collections import deque, Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import mediapipe as mp

from core.ml.feature_engineering import extract_hand_features_v2, TOTAL_FEATURES_V2
from core.inference.text_builder import TextBuilder   # ← ADDED

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
_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH   = _BASE_DIR / "models" / "asl_xgboost.pkl"
ENCODER_PATH = _BASE_DIR / "models" / "label_encoder.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────────────
# Rolling window: 10 frames @ ~30 fps ≈ 0.33 s of history.
# Lower → faster sign transitions; higher → more stable display.
BUFFER_SIZE = 12

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
STABILITY_THRESHOLD = 5

# Grace period: Number of consecutive frames to wait when a hand is missing
# before resetting the stability state. Prevents flickering issues.
HAND_MISSING_THRESHOLD = 6

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

    # ── A/S ambiguity filter ──────────────────────────────────────────────────
    # A and S share very similar fist geometry; the model frequently confuses
    # them at moderate confidence.  Require a stricter threshold for either
    # letter so only high-certainty frames enter the stabilisation buffer.
    # Weak frames are suppressed (None) rather than returned early so that the
    # buffer and hysteresis layers continue operating normally.
    if letter_candidate in ("A", "S") and confidence < 0.75:
        return None

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
# Module-level state — shared across predict_frame() calls
# ─────────────────────────────────────────────────────────────────────────────
# These objects are initialised once at import time so that every call to
# predict_frame() shares the same rolling buffer and hysteresis counter,
# giving the stabilisation layers the same cross-frame memory they have
# in the standalone run_predictor() loop.
#
# _pf_model / _pf_encoder : loaded XGBClassifier + LabelEncoder
# _pf_buffer              : rolling majority-vote deque  (Layer 2)
# _pf_candidate           : current hysteresis candidate (Layer 3)
# _pf_stability           : consecutive-frame streak counter
# _pf_stable              : the last letter that cleared the stability gate
# _pf_hands               : persistent MediaPipe Hands context (tracking mode)
# ─────────────────────────────────────────────────────────────────────────────

def _init_predict_frame_state():
    """Load model artefacts and build all module-level state for predict_frame()."""
    global _pf_model, _pf_encoder, _pf_buffer
    global _pf_candidate, _pf_stability, _pf_stable, _pf_hands
    global _pf_hand_missing_counter

    _pf_model, _pf_encoder = load_model_artefacts()
    _pf_buffer    = deque(maxlen=BUFFER_SIZE)
    _pf_candidate = None
    _pf_stability = 0
    _pf_stable    = None
    _pf_hand_missing_counter = 0
    _pf_hands     = MP_HANDS.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7,
    )
    log.info("predict_frame() state initialised.")

# Initialise eagerly at import time so the first call to predict_frame()
# has no setup latency.  Errors (missing model files) surface immediately.
try:
    _init_predict_frame_state()
except FileNotFoundError as _pf_init_err:
    log.warning("predict_frame() state NOT initialised: %s", _pf_init_err)


# ─────────────────────────────────────────────────────────────────────────────
# predict_frame() — single-frame entry point for external callers (e.g. Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def predict_frame(
    frame,
) -> tuple:
    """
    Process one BGR frame through the full ASL prediction pipeline.

    This is the public entry point for callers that manage their own capture
    loop (e.g. the Streamlit UI).  It is intentionally free of any webcam or
    display logic — those concerns remain in run_predictor().

    Pipeline (mirrors run_predictor() exactly):
        BGR frame (caller-supplied, already flipped if desired)
          ↓  BGR → RGB
          ↓  MediaPipe Hands (persistent tracking context)
          ↓  draw_landmarks() on the input frame
          ↓  normalize_hand_landmarks_copy()
          ↓  predict_sign()          confidence gate + gesture rules + A/S filter
          ↓  prediction_buffer       majority vote          (Layer 2)
          ↓  hysteresis counter      stability gate         (Layer 3)
          ↓  draw_prediction_overlay()
          ↓  return annotated_frame, stable_letter, hand_detected

    State persistence:
        The buffer, candidate, and stability counter live at module level so
        they accumulate correctly across successive calls, exactly as they do
        inside the while-loop of run_predictor().

    Args:
        frame : np.ndarray — BGR image from cv2.VideoCapture or any source.
                The frame is annotated in-place; pass a copy if you need the
                original untouched.

    Returns:
        annotated_frame : np.ndarray — BGR frame with skeleton + letter overlay.
        stable_letter   : str | None — stabilised letter, or None.
        hand_detected   : bool       — True if MediaPipe found a hand.
    """
    global _pf_buffer, _pf_candidate, _pf_stability, _pf_stable
    global _pf_hand_missing_counter

    # ── BGR → RGB ─────────────────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = _pf_hands.process(rgb)
    rgb.flags.writeable = True

    hand_detected = results.multi_hand_landmarks is not None

    if hand_detected:
        # Reset the missing counter since we found a hand
        _pf_hand_missing_counter = 0
        
        hand_landmarks = results.multi_hand_landmarks[0]

        # Handedness (optional — None treated as Right, no x-flip)
        handedness = None
        if results.multi_handedness:
            handedness = (
                results.multi_handedness[0].classification[0].label
            )   # "Left" | "Right"

        # Draw skeleton using the ORIGINAL landmarks (real pixel coords)
        draw_landmarks(frame, hand_landmarks)

        # Deep-copy + orientation-normalise for prediction only
        normalised_landmarks = normalize_hand_landmarks_copy(
            hand_landmarks, handedness
        )

        # ── Layer 1: confidence gate + gesture rules + A/S filter ─────────────
        raw_letter = predict_sign(normalised_landmarks, _pf_model, _pf_encoder)

        # ── Layer 2: rolling majority-vote buffer ─────────────────────────────
        if raw_letter is not None:
            _pf_buffer.append(raw_letter)
            vote = Counter(_pf_buffer).most_common(1)[0][0]
        else:
            vote = (
                Counter(_pf_buffer).most_common(1)[0][0]
                if _pf_buffer else None
            )

        # ── Layer 3: hysteresis lock ───────────────────────────────────────────
        if vote == _pf_candidate:
            _pf_stability += 1
        else:
            _pf_candidate = vote
            _pf_stability = 1

        if _pf_stability >= STABILITY_THRESHOLD:
            _pf_stable = _pf_candidate

    else:
        # ── GRACE PERIOD LOGIC ────────────────────────────────────────────────
        # Instead of resetting immediately, we increment a counter. 
        # Only after HAND_MISSING_THRESHOLD consecutive missing frames 
        # do we actually wipe the stability layers.
        _pf_hand_missing_counter += 1
        
        if _pf_hand_missing_counter >= HAND_MISSING_THRESHOLD:
            _pf_buffer.clear()
            _pf_candidate = None
            _pf_stability = 0
            _pf_stable    = None
            # Log only once on reset to avoid spam
            if _pf_hand_missing_counter == HAND_MISSING_THRESHOLD:
                log.debug("Hand missing threshold reached. Resetting prediction state.")

    # ── Overlay ───────────────────────────────────────────────────────────────
    draw_prediction_overlay(frame, _pf_stable)

    return frame, _pf_stable, hand_detected


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

    # ── TextBuilder: converts stable letters → words → sentences ─────────────
    text_builder = TextBuilder()                             # ← ADDED

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

            # Derived once per frame; used by both the stabilisation logic
            # below and TextBuilder so both always agree on hand presence.
            hand_detected = results.multi_hand_landmarks is not None  # ← ADDED

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

                # ── TextBuilder update (hand present) ─────────────────────────
                # Passes the fully stabilised letter so TextBuilder only ever
                # sees letters that have cleared all three stability layers.
                current_word, sentence = text_builder.update(  # ← ADDED
                    current_stable_letter,                      # ← ADDED
                    hand_detected,                              # ← ADDED
                    time.time(),                                # ← ADDED
                )                                               # ← ADDED
                print("Word:", current_word, "| Sentence:", sentence)  # ← ADDED

            else:
                # No hand detected — reset all three layers so the next sign
                # starts completely fresh with no stale state carried over
                prediction_buffer.clear()
                candidate_letter      = None
                stability_counter     = 0
                current_stable_letter = None

                # ── TextBuilder update (no hand) ──────────────────────────────
                # Drives the space / sentence-finalisation timers even when
                # the stabilisation state has already been wiped above.
                current_word, sentence = text_builder.update(  # ← ADDED
                    None,                                       # ← ADDED
                    False,                                      # ← ADDED
                    time.time(),                                # ← ADDED
                )                                               # ← ADDED
                print("Word:", current_word, "| Sentence:", sentence)  # ← ADDED

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