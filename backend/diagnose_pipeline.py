"""
diagnose_pipeline.py
────────────────────
Standalone diagnostic that simulates the EXACT web API pipeline:
    Webcam → JPEG encode (like canvas.toBlob) → decode (like cv2.imdecode) → MediaPipe

Run from the backend/ directory with venv activated:
    python diagnose_pipeline.py
"""

import cv2
import numpy as np
import mediapipe as mp
import time


def diagnose():
    print("=" * 60)
    print("ASL Pipeline Diagnostic")
    print("=" * 60)

    # ── Step 1: Open webcam ─────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1)  # let camera warm up

    ret, raw_frame = cap.read()
    cap.release()

    if not ret or raw_frame is None:
        print("[FAIL] Cannot read frame from webcam")
        return

    print(f"[OK] Raw webcam frame: shape={raw_frame.shape}, dtype={raw_frame.dtype}")
    print(f"     Mean pixel value: {raw_frame.mean():.1f} (0=black, 127=mid)")

    # ── Step 2: Mirror (like frontend ctx.scale(-1, 1)) ─────────
    mirrored = cv2.flip(raw_frame, 1)
    print(f"[OK] Mirrored frame: shape={mirrored.shape}")

    # ── Step 3: JPEG encode (like canvas.toBlob quality 0.85) ───
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    success, jpeg_buf = cv2.imencode('.jpg', mirrored, encode_params)
    if not success:
        print("[FAIL] JPEG encoding failed")
        return
    jpeg_bytes = jpeg_buf.tobytes()
    print(f"[OK] JPEG encoded: {len(jpeg_bytes)} bytes")

    # ── Step 4: Decode (like cv2.imdecode in prediction_service) ─
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if decoded is None:
        print("[FAIL] JPEG decoding failed")
        return
    print(f"[OK] Decoded frame: shape={decoded.shape}, dtype={decoded.dtype}")
    print(f"     Mean pixel value: {decoded.mean():.1f}")

    # ── Step 5: BGR → RGB (like predict_frame) ──────────────────
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    print(f"[OK] RGB converted: shape={rgb.shape}")

    # ── Step 6: Test MediaPipe with MULTIPLE confidence levels ──
    print()
    print("-" * 60)
    print("MediaPipe Detection Tests")
    print("-" * 60)

    for conf in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=conf,
        )
        rgb_copy = rgb.copy()
        rgb_copy.flags.writeable = False
        results = hands.process(rgb_copy)
        hands.close()

        detected = results.multi_hand_landmarks is not None
        n_hands = len(results.multi_hand_landmarks) if detected else 0
        status = "DETECTED" if detected else "NOT detected"
        print(f"  confidence={conf:.1f}  ->  {status}  (hands={n_hands})")

    # ── Step 7: Test with RAW frame (no JPEG round-trip) ────────
    print()
    print("-" * 60)
    print("Raw Frame Test (no JPEG compression)")
    print("-" * 60)

    raw_rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    raw_rgb.flags.writeable = False
    results = hands.process(raw_rgb)
    hands.close()

    detected = results.multi_hand_landmarks is not None
    print(f"  confidence=0.5  ->  {'DETECTED' if detected else 'NOT detected'}")

    # ── Step 8: Test without mirror flip ─────────────────────────
    print()
    print("-" * 60)
    print("Unflipped Frame Test")
    print("-" * 60)

    unflipped_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    unflipped_rgb.flags.writeable = False
    results = hands.process(unflipped_rgb)
    hands.close()

    detected = results.multi_hand_landmarks is not None
    print(f"  confidence=0.5  ->  {'DETECTED' if detected else 'NOT detected'}")

    # ── Step 9: Test predict_frame() directly ────────────────────
    print()
    print("-" * 60)
    print("predict_frame() Direct Test")
    print("-" * 60)

    try:
        from core.inference.realtime_asl_predictor import predict_frame
        annotated, letter, hand_det = predict_frame(decoded)
        print(f"  hand_detected={hand_det}, letter={letter}")
        print(f"  annotated frame shape={annotated.shape}")
    except Exception as e:
        print(f"  [ERROR] predict_frame() raised: {type(e).__name__}: {e}")

    # ── Step 10: Check what predict_frame sees internally ────────
    print()
    print("-" * 60)
    print("Internal State Check")
    print("-" * 60)

    try:
        from core.inference import realtime_asl_predictor as rap
        print(f"  _pf_hands exists: {hasattr(rap, '_pf_hands')}")
        if hasattr(rap, '_pf_hands'):
            h = rap._pf_hands
            print(f"  _pf_hands type: {type(h)}")
        print(f"  _pf_model exists: {hasattr(rap, '_pf_model')}")
        print(f"  _pf_encoder exists: {hasattr(rap, '_pf_encoder')}")
        if hasattr(rap, '_pf_model'):
            print(f"  _pf_model type: {type(rap._pf_model)}")
        if hasattr(rap, '_pf_encoder'):
            print(f"  _pf_encoder classes: {list(rap._pf_encoder.classes_)}")
    except Exception as e:
        print(f"  [ERROR] State check failed: {e}")

    print()
    print("=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print()
    print("INSTRUCTIONS: Hold your hand clearly in front of the webcam")
    print("BEFORE running this script (it captures a single frame).")
    print("If all tests say 'NOT detected', the webcam frame itself")
    print("may not contain a visible hand.")


if __name__ == "__main__":
    diagnose()
