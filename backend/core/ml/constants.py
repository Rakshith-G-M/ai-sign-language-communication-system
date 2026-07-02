"""
constants.py
────────────
Single source of truth for ML pipeline constants shared across data
generation, training, and inference modules.
"""

# Feature engineering
TOTAL_FEATURES_V2 = 134   # v2 engineered feature vector length

# Reproducibility
RANDOM_SEED = 42

# Model versioning (recorded in training manifests)
STATIC_MODEL_VERSION = "static-xgboost-v2"

# Static dataset CSV schema
STATIC_LABEL_COLUMN = "label"
STATIC_FEATURE_PREFIX = "f"

