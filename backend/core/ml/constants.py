"""
constants.py
────────────
Single source of truth for ML pipeline constants shared across data
generation, training, and inference modules.
"""

# Feature engineering
TOTAL_FEATURES_V2 = 134   # v2 engineered feature vector length

# Dynamic gesture sequences
SEQUENCE_LENGTH = 30      # frames per gesture — must match training and ONNX export

# Reproducibility
RANDOM_SEED = 42

# Model versioning (recorded in training manifests)
STATIC_MODEL_VERSION = "static-xgboost-v2"
DYNAMIC_MODEL_VERSION = "dynamic-lstm-v2"

# Static dataset CSV schema
STATIC_LABEL_COLUMN = "label"
STATIC_FEATURE_PREFIX = "f"

# Dynamic dataset JSONL schema keys
DYNAMIC_LABEL_KEY = "label"
DYNAMIC_FRAMES_KEY = "frames"

# Legacy dynamic feature dimension (raw 21×3 landmarks — deprecated)
LEGACY_DYNAMIC_FEATURE_DIM = 63
