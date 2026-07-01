"""
landmark_utils.py
─────────────────
Shared MediaPipe landmark preprocessing used by dataset generation scripts
and dynamic inference.  Ensures orientation normalisation and v2 feature
extraction follow identical logic everywhere.
"""

from __future__ import annotations

import numpy as np
from mediapipe.framework.formats import landmark_pb2

from core.ml.constants import TOTAL_FEATURES_V2
from core.ml.feature_engineering import extract_hand_features_v2


def normalize_orientation(hand_landmarks, handedness: str | None):
    """
    Return landmarks in canonical right-hand orientation.

    Left hands are mirrored on the X axis (x = 1.0 - x) in normalised
    coordinate space.  Right hands and unknown handedness are returned
    unchanged.
    """
    if handedness != "Left":
        return hand_landmarks

    normalised = landmark_pb2.NormalizedLandmarkList()
    normalised.CopyFrom(hand_landmarks)

    for lm in normalised.landmark:
        lm.x = 1.0 - lm.x

    return normalised


def copy_landmarks_with_x_flip(hand_landmarks) -> landmark_pb2.NormalizedLandmarkList:
    """Return a deep copy of landmarks with X coordinates mirrored."""
    mirrored = landmark_pb2.NormalizedLandmarkList()
    for landmark in hand_landmarks.landmark:
        copied = mirrored.landmark.add()
        copied.x = 1.0 - landmark.x
        copied.y = landmark.y
        copied.z = landmark.z
        copied.visibility = landmark.visibility
        copied.presence = landmark.presence
    return mirrored


def handedness_label(results) -> str | None:
    """Extract MediaPipe handedness label for the first detected hand."""
    if not results.multi_handedness:
        return None

    try:
        classification = results.multi_handedness[0].classification[0]
    except (IndexError, AttributeError):
        return None
    return classification.label or None


def orientation_normalized_landmarks(results):
    """Return first-hand landmarks in canonical right-hand orientation."""
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    if handedness_label(results) == "Left":
        return copy_landmarks_with_x_flip(hand_landmarks)
    return hand_landmarks


def extract_v2_features_from_landmarks(hand_landmarks, handedness: str | None = None) -> np.ndarray | None:
    """
    Normalise orientation then extract the canonical v2 feature vector.

    Returns:
        float32 array of shape (TOTAL_FEATURES_V2,) or None on failure.
    """
    oriented = normalize_orientation(hand_landmarks, handedness)
    return extract_hand_features_v2(oriented)


def extract_v2_features_from_results(results) -> np.ndarray | None:
    """
    Extract v2 features from a MediaPipe Hands process() result object.

    Returns:
        float32 array of shape (TOTAL_FEATURES_V2,) or None when no hand
        is detected or feature extraction fails.
    """
    landmarks = orientation_normalized_landmarks(results)
    if landmarks is None:
        return None
    return extract_hand_features_v2(landmarks)


def zero_feature_frame() -> list[float]:
    """Return a zero-padded feature frame for temporal padding."""
    return [0.0] * TOTAL_FEATURES_V2
