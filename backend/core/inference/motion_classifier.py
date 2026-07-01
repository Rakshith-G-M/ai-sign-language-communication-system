"""
motion_classifier.py
───────────────────────
Analyzes a history of hand landmarks to determine if the hand is in motion
(dynamic signing) or stationary (static letter spelling).
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

# Standard deviation threshold for hand movement (in normalized coordinate space).
# Values above this indicate the hand is actively moving/signing.
MOTION_THRESHOLD = 0.012

# Minimum history length required to calculate motion confidence
MIN_HISTORY_FRAMES = 5

def is_hand_moving(landmarks_history) -> bool:
    """
    Check if the hand is actively moving based on the trajectory of the wrist.
    
    Args:
        landmarks_history: Deque of MediaPipe NormalizedLandmarkList objects.
        
    Returns:
        bool: True if standard deviation of wrist coords exceeds threshold.
    """
    if len(landmarks_history) < MIN_HISTORY_FRAMES:
        return False

    # Extract WRIST (landmark 0) coordinates across history
    wrist_coords = []
    for lm_list in landmarks_history:
        if lm_list and lm_list.landmark:
            wrist = lm_list.landmark[0]  # WRIST = 0
            wrist_coords.append([wrist.x, wrist.y, wrist.z])

    if len(wrist_coords) < MIN_HISTORY_FRAMES:
        return False

    wrist_arr = np.array(wrist_coords) # shape (N, 3)
    
    # Calculate standard deviation along X and Y axes
    std_x = np.std(wrist_arr[:, 0])
    std_y = np.std(wrist_arr[:, 1])
    
    total_motion = float(std_x + std_y)
    
    is_moving = total_motion > MOTION_THRESHOLD
    log.debug("Hand motion std: %.4f (threshold: %.4f) -> Moving: %s", 
              total_motion, MOTION_THRESHOLD, is_moving)
              
    return is_moving
