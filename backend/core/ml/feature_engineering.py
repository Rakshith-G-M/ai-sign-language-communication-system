"""
feature_engineering.py
───────────────────────
Reusable feature extraction pipeline for AI Sign Language Recognition.

Two public extraction functions:

    extract_hand_features()    — original 82-feature vector (v1)
                                  Kept for backward compatibility with any
                                  already-saved v1 models.

    extract_hand_features_v2() — 134-feature vector (v2)
                                  Retrain the model with this for improved
                                  discrimination of similar gestures.

v1 Feature Groups (82 total):
    Group 1 — Relative Landmarks    : 63  (indices   0 –  62)
    Group 2 — Finger States         :  5  (indices  63 –  67)
    Group 3 — Geometric Distances   :  9  (indices  68 –  76)
    Group 4 — PIP Bend Angles       :  5  (indices  77 –  81)

v2 Feature Groups (134 total):
    Group A — Normalised coordinates        : 63  (indices   0 –  62)
    Group B — Pairwise distances            : 20  (indices  63 –  82)
    Group C — Joint angles                  : 20  (indices  83 – 102)
    Group D — Palm-relative distances       : 20  (indices 103 – 122)
    Group E — Shape descriptors             :  3  (indices 123 – 125)
    Group F — Discriminative features       :  8  (indices 126 – 133)
              F1: thumb→{index,middle,ring,pinky} tip distances  (4)
              F2: adjacent fingertip spread ×3                   (3)
              F3: index/middle crossing binary                   (1)
    ─────────────────────────────────────────────────────────────────
    TOTAL v2                                : 134

    Group F target pairs:
        N vs M vs S  — thumb overlay position (F1)
        T vs A       — thumb-to-index contact (F1[0])
        U vs R       — finger crossing binary + spread (F2[0] + F3)

MediaPipe Landmark Index Reference:
     0  WRIST
     1  THUMB_CMC    2  THUMB_MCP    3  THUMB_IP     4  THUMB_TIP
     5  INDEX_MCP    6  INDEX_PIP    7  INDEX_DIP    8  INDEX_TIP
     9  MIDDLE_MCP  10  MIDDLE_PIP  11  MIDDLE_DIP  12  MIDDLE_TIP
    13  RING_MCP    14  RING_PIP    15  RING_DIP    16  RING_TIP
    17  PINKY_MCP   18  PINKY_PIP   19  PINKY_DIP   20  PINKY_TIP
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Landmark index constants — single source of truth
# ─────────────────────────────────────────────────────────────────────────────
WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP,  THUMB_TIP  =  1,  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  =  5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP =  9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

# Feature-count constants — import these in every consumer script so that a
# single edit here propagates everywhere automatically.
TOTAL_FEATURES    = 82    # v1
TOTAL_FEATURES_V2 = 134   # v2  (126 base + 8 discriminative: F_thumb×4 + F_spread×3 + F_cross×1)


# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _euclidean_distance(lm_a, lm_b) -> float:
    """3-D Euclidean distance between two MediaPipe landmark objects."""
    return float(np.sqrt(
        (lm_a.x - lm_b.x) ** 2 +
        (lm_a.y - lm_b.y) ** 2 +
        (lm_a.z - lm_b.z) ** 2
    ))


def _joint_angle(lm_a, lm_b, lm_c) -> float:
    """
    Interior angle in degrees at landmark B, formed by the vectors B→A and B→C.

    Used to measure finger bend.  The vertex lm_b is the joint being measured:
        lm_a = proximal landmark (e.g. MCP)
        lm_b = joint vertex     (e.g. PIP or DIP)
        lm_c = distal landmark  (e.g. TIP or DIP)

    Returns 0.0 for degenerate (zero-length) vectors.
    """
    vec_ba = np.array([lm_a.x - lm_b.x, lm_a.y - lm_b.y, lm_a.z - lm_b.z])
    vec_bc = np.array([lm_c.x - lm_b.x, lm_c.y - lm_b.y, lm_c.z - lm_b.z])

    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)

    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0

    cos_angle = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(float(np.clip(cos_angle, -1.0, 1.0)))))


def _lm_vec(lm) -> np.ndarray:
    """Return a landmark's (x, y, z) as a float32 NumPy vector."""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _unit_vec(origin_lm, tip_lm) -> np.ndarray:
    """
    Unit vector pointing from origin_lm to tip_lm.
    Returns a zero vector for coincident landmarks (degenerate pose).
    """
    vec  = _lm_vec(tip_lm) - _lm_vec(origin_lm)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 1e-9 else np.zeros(3, dtype=np.float32)


def _palm_normal_vec(lm) -> np.ndarray:
    """
    Unit normal to the palm plane defined by WRIST → INDEX_MCP → PINKY_MCP.

    The cross product of the two edge vectors gives a vector perpendicular to
    the palm surface.  Its direction (toward vs away from camera) encodes palm
    orientation — essential for B vs open-5 and other flat-hand variants.

    Returns a zero vector for degenerate poses.
    """
    wrist     = _lm_vec(lm[WRIST])
    index_mcp = _lm_vec(lm[INDEX_MCP])
    pinky_mcp = _lm_vec(lm[PINKY_MCP])

    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    norm   = np.linalg.norm(normal)
    return (normal / norm).astype(np.float32) if norm > 1e-9 else np.zeros(3, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# v1  —  82-feature extraction (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def extract_hand_features(hand_landmarks) -> np.ndarray:
    """
    Convert MediaPipe Hands landmarks into an 82-dimensional feature vector.

    Deterministic and identical for dataset generation and real-time inference.

    Args:
        hand_landmarks: MediaPipe NormalizedLandmarkList (21 landmarks,
                        each with float .x .y .z attributes in [0, 1]).

    Returns:
        numpy.ndarray shape (82,) dtype float32.
            [  0 –  62]  Relative landmarks  (63)
            [ 63 –  67]  Finger states        ( 5)
            [ 68 –  76]  Euclidean distances  ( 9)
            [ 77 –  81]  PIP bend angles      ( 5)

    Raises:
        ValueError: landmark count ≠ 21.
    """
    lm = hand_landmarks.landmark

    if len(lm) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(lm)}.")

    wrist = lm[WRIST]

    # ── Group 1: Relative Landmarks (63 features) ────────────────────────────
    # All landmarks translated so the wrist is the origin.
    # Makes the vector invariant to absolute hand position in the frame.
    # Order: [x0,y0,z0,  x1,y1,z1,  …  x20,y20,z20]
    relative_landmarks = []
    for i in range(21):
        relative_landmarks.extend([
            lm[i].x - wrist.x,
            lm[i].y - wrist.y,
            lm[i].z - wrist.z,
        ])

    # ── Group 2: Finger State Features (5 features) ──────────────────────────
    # Binary extension flags.
    # Thumb: x-axis comparison (thumb is orthogonal to other fingers).
    # Others: tip.y < pip.y  →  tip is higher in image  →  finger extended.
    thumb_tip_rel_x = lm[THUMB_TIP].x - wrist.x
    thumb_mcp_rel_x = lm[THUMB_MCP].x - wrist.x
    finger_states = [
        1.0 if abs(thumb_tip_rel_x) > abs(thumb_mcp_rel_x) else 0.0,
        1.0 if lm[INDEX_TIP].y  < lm[INDEX_PIP].y  else 0.0,
        1.0 if lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y else 0.0,
        1.0 if lm[RING_TIP].y   < lm[RING_PIP].y   else 0.0,
        1.0 if lm[PINKY_TIP].y  < lm[PINKY_PIP].y  else 0.0,
    ]

    # ── Group 3: Geometric Distance Features (9 features) ────────────────────
    # Key pairwise distances capturing hand shape and posture.
    distances = [
        _euclidean_distance(lm[THUMB_TIP],  lm[INDEX_TIP]),   # pinch aperture
        _euclidean_distance(lm[THUMB_TIP],  lm[MIDDLE_TIP]),  # thumb–middle
        _euclidean_distance(lm[THUMB_TIP],  lm[RING_TIP]),    # thumb–ring
        _euclidean_distance(lm[THUMB_TIP],  lm[PINKY_TIP]),   # thumb–pinky
        _euclidean_distance(lm[INDEX_TIP],  lm[MIDDLE_TIP]),  # index–middle
        _euclidean_distance(lm[MIDDLE_TIP], lm[RING_TIP]),    # middle–ring
        _euclidean_distance(lm[RING_TIP],   lm[PINKY_TIP]),   # ring–pinky
        _euclidean_distance(lm[INDEX_TIP],  lm[WRIST]),       # index reach
        _euclidean_distance(lm[PINKY_TIP],  lm[WRIST]),       # pinky reach
    ]

    # ── Group 4: PIP Bend Angles (5 features) ────────────────────────────────
    # Angle at the PIP (or IP for thumb) joint: 180° = straight, less = curled.
    bend_angles = [
        _joint_angle(lm[THUMB_MCP],  lm[THUMB_IP],   lm[THUMB_TIP]),
        _joint_angle(lm[INDEX_MCP],  lm[INDEX_PIP],  lm[INDEX_TIP]),
        _joint_angle(lm[MIDDLE_MCP], lm[MIDDLE_PIP], lm[MIDDLE_TIP]),
        _joint_angle(lm[RING_MCP],   lm[RING_PIP],   lm[RING_TIP]),
        _joint_angle(lm[PINKY_MCP],  lm[PINKY_PIP],  lm[PINKY_TIP]),
    ]

    feature_vector = np.array(
        relative_landmarks + finger_states + distances + bend_angles,
        dtype=np.float32,
    )

    assert feature_vector.shape == (TOTAL_FEATURES,), (
        f"v1 shape mismatch: expected ({TOTAL_FEATURES},), got {feature_vector.shape}"
    )
    return feature_vector


# ─────────────────────────────────────────────────────────────────────────────
# v2  —  134-feature standalone extraction
# ─────────────────────────────────────────────────────────────────────────────
#
# Feature layout (134 total):
#   A. Normalised coordinates    63   indices   0 –  62
#   B. Pairwise distances        20   indices  63 –  82
#   C. Joint angles              20   indices  83 – 102
#   D. Palm-relative distances   20   indices 103 – 122
#   E. Shape descriptors          3   indices 123 – 125
#   F. Discriminative features    8   indices 126 – 133
#       F1: thumb→{index,middle,ring,pinky} tip distances  (4)
#       F2: adjacent fingertip spread ×3                   (3)
#       F3: index/middle crossing binary                   (1)
#
# Normalisation scheme
# ────────────────────
# Every landmark is first translated so the wrist sits at the origin, then
# divided by the wrist→middle-MCP distance.  This makes the vector invariant
# to both absolute position in the frame AND hand scale, which are the two
# largest sources of inter-subject variance.  Dividing by middle-MCP (rather
# than e.g. bounding-box diagonal) is stable because the wrist→middle-MCP
# segment is rigid and well-detected even in partial views.
# ─────────────────────────────────────────────────────────────────────────────


def extract_hand_features_v2(hand_landmarks) -> np.ndarray | None:

    """
    Convert MediaPipe Hands landmarks into a 134-dimensional feature vector.

    Uses wrist-origin, middle-MCP-scale normalisation for position and scale
    invariance.  Deterministic: identical input always produces identical output
    in the same feature order, making it safe for both dataset generation and
    real-time inference.

    Args:
        hand_landmarks: MediaPipe NormalizedLandmarkList — the object returned
                        by ``results.multi_hand_landmarks[i]``.  Must contain
                        exactly 21 landmarks with .x, .y, .z float attributes.

    Returns:
        numpy.ndarray of shape (134,) and dtype float32 on success.
        None if any error occurs (wrong landmark count, degenerate scale, etc.)
        so the caller can safely skip the frame without a try/except.

    Feature layout:
        [  0 –  62]  A. Normalised coordinates          (63)
        [ 63 –  82]  B. Pairwise distances               (20)
        [ 83 – 102]  C. Joint angles                     (20)
        [103 – 122]  D. Palm-relative distances           (20)
        [123 – 125]  E. Shape descriptors                  (3)
        [126 – 129]  F1. Thumb-to-fingertip distances      (4)  ← NEW
        [130 – 132]  F2. Adjacent fingertip spread         (3)  ← NEW
        [133 – 133]  F3. Index/middle crossing binary      (1)  ← NEW
        ──────────────────────────────────────────────────────
        TOTAL                                            (134)

    Group F discriminates confusable pairs:
        N vs M vs S  — thumb overlay position captured by thumb→{index,middle,ring}
        T vs A       — thumb→index_tip distance separates contact vs non-contact
        U vs R       — index/middle crossing binary + spread distance
    """
    try:
        lm = hand_landmarks.landmark

        # ── Validate ──────────────────────────────────────────────────────────
        if len(lm) != 21:
            return None

        # ── Normalisation basis ───────────────────────────────────────────────
        # Step 1: translate so the wrist is the origin.
        # Step 2: divide by wrist→middle-MCP distance to remove hand scale.
        wx, wy, wz = lm[WRIST].x, lm[WRIST].y, lm[WRIST].z

        hand_scale = float(np.sqrt(
            (lm[MIDDLE_MCP].x - wx) ** 2 +
            (lm[MIDDLE_MCP].y - wy) ** 2 +
            (lm[MIDDLE_MCP].z - wz) ** 2
        ))

        # Guard: degenerate pose where wrist and middle-MCP are coincident.
        if hand_scale < 1e-6:
            return None

        # Build a (21, 3) array of scale-normalised wrist-relative coordinates.
        # coords[i] = (lm[i] - wrist) / hand_scale
        coords = np.array(
            [(lm[i].x - wx, lm[i].y - wy, lm[i].z - wz) for i in range(21)],
            dtype=np.float32
        ) / hand_scale                                   # shape (21, 3)

        # ── Group A: Normalised coordinates (63 features) ─────────────────────
        # Flat row-major traversal: [x0,y0,z0, x1,y1,z1, … x20,y20,z20]
        # coords[0] (wrist) is always (0,0,0) after translation — kept
        # deliberately so that index positions never shift.
        group_a = coords.ravel()                         # shape (63,)

        # ── Convenience: normalised landmark vectors by index ─────────────────
        # Used throughout groups B, C, D so arithmetic stays readable.
        c = coords   # alias; c[i] = [x_norm, y_norm, z_norm] for landmark i

        def _dist(i: int, j: int) -> float:
            """Euclidean distance between normalised landmarks i and j."""
            return float(np.linalg.norm(c[i] - c[j]))

        def _angle(a: int, b: int, cc: int) -> float:
            """
            Angle in degrees at vertex b, formed by vectors b→a and b→cc,
            using normalised coordinates.  Returns 0.0 for degenerate vectors.
            """
            ba = c[a] - c[b]
            bc = c[cc] - c[b]
            n_ba = np.linalg.norm(ba)
            n_bc = np.linalg.norm(bc)
            if n_ba < 1e-9 or n_bc < 1e-9:
                return 0.0
            cos_t = float(np.dot(ba, bc) / (n_ba * n_bc))
            return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

        # ── Group B: Pairwise distances (20 features) ─────────────────────────
        # All computed in normalised space → scale-invariant.
        #
        # Sub-group B1 (10): all C(5,2) fingertip pairs
        tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        tip_pairs = [_dist(tips[i], tips[j])
                     for i in range(5) for j in range(i + 1, 5)]   # 10 values

        # Sub-group B2 (5): each fingertip to wrist (WRIST = index 0, always 0,0,0
        # in normalised space, so this equals ‖c[tip]‖ — tip reach)
        tip_to_wrist = [_dist(t, WRIST) for t in tips]              # 5 values

        # Sub-group B3 (5): each fingertip to middle MCP (palm anchor)
        tip_to_mmcp  = [_dist(t, MIDDLE_MCP) for t in tips]         # 5 values

        group_b = np.array(tip_pairs + tip_to_wrist + tip_to_mmcp,
                           dtype=np.float32)                         # shape (20,)

        # ── Group C: Joint angles (20 features) ───────────────────────────────
        #
        # C1 — PIP-joint angles: MCP→PIP→TIP (one per non-thumb finger) = 4
        pip_angles = [
            _angle(INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
            _angle(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
            _angle(RING_MCP,   RING_PIP,   RING_TIP),
            _angle(PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
        ]

        # C2 — DIP-joint angles: PIP→DIP→TIP (one per non-thumb finger) = 4
        dip_angles = [
            _angle(INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
            _angle(MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            _angle(RING_PIP,   RING_DIP,   RING_TIP),
            _angle(PINKY_PIP,  PINKY_DIP,  PINKY_TIP),
        ]

        # C3 — Thumb two-segment curvature = 2
        thumb_angles = [
            _angle(THUMB_CMC, THUMB_MCP, THUMB_IP),    # CMC→MCP→IP
            _angle(THUMB_MCP, THUMB_IP,  THUMB_TIP),   # MCP→IP→TIP
        ]

        # C4 — MCP knuckle spread: angle at each MCP between its neighbour MCPs.
        # Vertex = this finger's MCP; rays go to adjacent finger's MCPs and wrist.
        # Captures lateral knuckle fan angle.  = 4 angles
        #   index:  wrist→INDEX_MCP→MIDDLE_MCP
        #   middle: INDEX_MCP→MIDDLE_MCP→RING_MCP
        #   ring:   MIDDLE_MCP→RING_MCP→PINKY_MCP
        #   pinky:  RING_MCP→PINKY_MCP→wrist
        knuckle_spread = [
            _angle(WRIST,      INDEX_MCP,  MIDDLE_MCP),
            _angle(INDEX_MCP,  MIDDLE_MCP, RING_MCP),
            _angle(MIDDLE_MCP, RING_MCP,   PINKY_MCP),
            _angle(RING_MCP,   PINKY_MCP,  WRIST),
        ]

        # C5 — Wrist fan angle: INDEX_MCP–WRIST–PINKY_MCP = 1
        wrist_fan = [_angle(INDEX_MCP, WRIST, PINKY_MCP)]

        # C6 — Inter-finger tip angles: angle at each tip between its two
        # immediate neighbour tips.  Only middle three fingers have two
        # neighbours; index and pinky use thumb/ring and ring/nothing
        # → use 5 angles by treating thumb as index's outer neighbour and
        # considering: thumb–index–middle, index–middle–ring,
        #              middle–ring–pinky, ring–pinky–thumb (wrap),
        #              pinky–thumb–index (wrap).  = 5
        tip_angles = [
            _angle(THUMB_TIP,  INDEX_TIP,  MIDDLE_TIP),   # vertex = index
            _angle(INDEX_TIP,  MIDDLE_TIP, RING_TIP),     # vertex = middle
            _angle(MIDDLE_TIP, RING_TIP,   PINKY_TIP),    # vertex = ring
            _angle(RING_TIP,   PINKY_TIP,  THUMB_TIP),    # vertex = pinky (wrap)
            _angle(PINKY_TIP,  THUMB_TIP,  INDEX_TIP),    # vertex = thumb (wrap)
        ]

        group_c = np.array(
            pip_angles + dip_angles + thumb_angles +
            knuckle_spread + wrist_fan + tip_angles,
            dtype=np.float32
        )   # 4+4+2+4+1+5 = 20                            shape (20,)

        # ── Group D: Palm-relative distances (20 features) ────────────────────
        # Palm centroid = mean of the four finger MCPs in normalised space.
        palm_centroid = (
            c[INDEX_MCP] + c[MIDDLE_MCP] + c[RING_MCP] + c[PINKY_MCP]
        ) / 4.0                                          # shape (3,)

        # For each finger, compute distance from each joint to palm centroid.
        # Joint traversal order: MCP, PIP/IP, DIP, TIP (proximal → distal).
        # Thumb uses CMC, MCP, IP, TIP for the same four-joint structure.
        finger_joints = [
            [THUMB_CMC,  THUMB_MCP,  THUMB_IP,   THUMB_TIP],   # thumb
            [INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP],   # index
            [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],  # middle
            [RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP],    # ring
            [PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP],   # pinky
        ]

        palm_dists = [
            float(np.linalg.norm(c[j] - palm_centroid))
            for joints in finger_joints
            for j in joints
        ]   # 5 × 4 = 20

        group_d = np.array(palm_dists, dtype=np.float32)        # shape (20,)

        # ── Group E: Shape descriptors (3 features) ───────────────────────────
        #
        # E1 — Palm aspect ratio (scalar)
        #   width  = distance index_mcp → pinky_mcp  (knuckle row width)
        #   height = distance wrist → middle_mcp     (= 1.0 by definition,
        #            since we normalised by this; kept explicit for clarity)
        #   ratio  = width / height  — encodes how wide / narrow the palm is
        palm_width  = float(np.linalg.norm(c[INDEX_MCP] - c[PINKY_MCP]))
        palm_height = float(np.linalg.norm(c[MIDDLE_MCP]))   # wrist is origin
        palm_aspect = palm_width / palm_height if palm_height > 1e-9 else 0.0

        # E2 — Finger extension ratio (scalar)
        #   Mean distance of the 5 fingertips from the palm centroid,
        #   already in normalised (scale-invariant) space.
        #   High value → fingers extended; low value → fist.
        tip_centroid_dists = [
            float(np.linalg.norm(c[t] - palm_centroid)) for t in tips
        ]
        extension_ratio = float(np.mean(tip_centroid_dists))

        # E3 — Palm normal Z-component (scalar in [-1, 1])
        #   Unit normal to the WRIST–INDEX_MCP–PINKY_MCP plane.
        #   Computed in normalised space; z-component alone is sufficient
        #   because the camera axis is z: +z = palm toward camera, -z = away.
        edge1  = c[INDEX_MCP] - c[WRIST]    # WRIST is (0,0,0) in normed space
        edge2  = c[PINKY_MCP] - c[WRIST]
        normal = np.cross(edge1, edge2).astype(np.float32)
        n_norm = float(np.linalg.norm(normal))
        palm_normal_z = float(normal[2] / n_norm) if n_norm > 1e-9 else 0.0

        group_e = np.array(
            [palm_aspect, extension_ratio, palm_normal_z],
            dtype=np.float32
        )   # shape (3,)

        # ── Group F: Discriminative features for confusable gestures (8) ────────
        #
        # All distances are computed in the same normalised space (wrist-origin,
        # middle-MCP scale) as groups B and D — fully scale-invariant.
        #
        # F1 — Thumb-tip to each fingertip (4 distances) ──────────────────────
        # Primary discriminator for N/M/S (thumb rests on different fingers)
        # and T/A (thumb touches index in T, tucked in A).
        #   index 126: thumb_tip ↔ index_tip
        #   index 127: thumb_tip ↔ middle_tip
        #   index 128: thumb_tip ↔ ring_tip
        #   index 129: thumb_tip ↔ pinky_tip
        thumb_interactions = [
            _dist(THUMB_TIP, INDEX_TIP),    # 126 — T/A separation
            _dist(THUMB_TIP, MIDDLE_TIP),   # 127 — N (thumb over index+middle)
            _dist(THUMB_TIP, RING_TIP),     # 128 — M (thumb over 3 fingers)
            _dist(THUMB_TIP, PINKY_TIP),    # 129 — general thumb reach
        ]

        # F2 — Adjacent fingertip spread (3 distances) ────────────────────────
        # Primary discriminator for U/R (index+middle spread vs crossed).
        # Separates tightly held (U) from crossed (R) finger pairs.
        #   index 130: index_tip ↔ middle_tip
        #   index 131: middle_tip ↔ ring_tip
        #   index 132: ring_tip ↔ pinky_tip
        finger_spread = [
            _dist(INDEX_TIP,  MIDDLE_TIP),  # 130 — U/R crossing gap
            _dist(MIDDLE_TIP, RING_TIP),    # 131
            _dist(RING_TIP,   PINKY_TIP),   # 132
        ]

        # F3 — Index / middle crossing binary (1 feature) ─────────────────────
        # index 133
        # In normalised (wrist-origin) space, c[INDEX_TIP][0] is the
        # scale-normalised x-coordinate.  For a right hand:
        #   R sign: index crosses over middle → index_tip.x < middle_tip.x
        #   U sign: fingers parallel           → index_tip.x > middle_tip.x
        # The comparison uses normalised x so it is hand-size invariant.
        # After left-hand orientation normalisation (x = 1 - x applied upstream),
        # the relative ordering is consistent for both hands.
        crossing_binary = [
            1.0 if c[INDEX_TIP][0] < c[MIDDLE_TIP][0] else 0.0   # 133
        ]

        group_f = np.array(
            thumb_interactions + finger_spread + crossing_binary,
            dtype=np.float32
        )   # 4 + 3 + 1 = 8                                       shape (8,)

        # ── Concatenate: A(63)+B(20)+C(20)+D(20)+E(3)+F(8) = 134 ─────────────
        feature_vector = np.concatenate([group_a, group_b, group_c,
                                         group_d, group_e, group_f])

        # Runtime shape assertion — fires immediately if any group count drifts.
        if feature_vector.shape[0] != TOTAL_FEATURES_V2:
            return None   # never reached with correct code; guard for safety

        return feature_vector.astype(np.float32)

    except Exception:   # noqa: BLE001
        # Return None on any unexpected error so the caller can skip the frame
        # without crashing — critical for real-time robustness.
        return None