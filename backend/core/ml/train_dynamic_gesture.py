"""
train_dynamic_gesture.py
─────────────────────────
Trains an LSTM classifier on the canonical dynamic gesture dataset
produced by preprocess_wlasl_dynamic.py, then exports to ONNX.

Pipeline
────────
    1. Load      →  dataset/dynamic_gestures.jsonl
    2. Validate  →  each sequence is SEQUENCE_LENGTH × TOTAL_FEATURES_V2 floats
    3. Encode    →  LabelEncoder  (class strings → integers)
    4. Split     →  stratified 80/20 train-test  (random_state=42)
    5. Normalise →  per-feature StandardScaler fitted on training data
    6. Train     →  LSTM(input=TOTAL_FEATURES_V2, hidden=128, layers=2, classes=N)
    7. Evaluate  →  accuracy + per-class report on test set
    8. Export    →  models/asl_dynamic.onnx  (ONNX opset 11)
                    models/dynamic_label_encoder.pkl
                    models/dynamic_scaler.pkl

Usage
─────
    python train_dynamic_gesture.py
    python train_dynamic_gesture.py \\
        --dataset dataset/dynamic_gestures.jsonl \\
        --model_dir models \\
        --epochs 80 \\
        --hidden 128 \\
        --layers 2

Dependencies
────────────
    pip install torch onnx scikit-learn joblib numpy
    (torch is a training-only dependency; inference uses onnxruntime)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Allow running directly from backend/core/ml or from the backend package root.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from core.ml.feature_engineering import TOTAL_FEATURES_V2

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
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET   = "dataset/dynamic_gestures.jsonl"
DEFAULT_MODEL_DIR = "models"
ONNX_FILENAME     = "asl_dynamic.onnx"
ENCODER_FILENAME  = "dynamic_label_encoder.pkl"
SCALER_FILENAME   = "dynamic_scaler.pkl"

SEQUENCE_LENGTH = 30    # canonical dynamic sequence length
INPUT_SIZE      = TOTAL_FEATURES_V2    # canonical engineered v2 feature count


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(jsonl_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the JSONL dataset into NumPy arrays.

    Returns:
        X : float32 array of shape (n_samples, SEQUENCE_LENGTH, TOTAL_FEATURES_V2)
        y : string array of shape  (n_samples,)
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}\n"
            "Run preprocess_wlasl_dynamic.py to generate the canonical 134-D dataset."
        )

    sequences, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                frames = record["frames"]
                label  = record["label"]
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Skipping malformed line %d: %s", line_no, exc)
                continue

            if not isinstance(label, str) or not label.strip():
                log.warning("Skipping line %d: missing or empty label", line_no)
                continue

            try:
                seq = np.array(frames, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                log.warning("Skipping line %d: frames must be numeric (%s)", line_no, exc)
                continue

            expected_shape = (SEQUENCE_LENGTH, INPUT_SIZE)
            if seq.shape != expected_shape:
                if seq.ndim == 2 and seq.shape[1] == 63:
                    raise ValueError(
                        "Legacy 63-dimensional dynamic dataset detected on "
                        f"line {line_no}. The canonical dynamic dataset format is "
                        f"(sequence_length={SEQUENCE_LENGTH}, "
                        f"feature_dim={TOTAL_FEATURES_V2}). Regenerate the dataset "
                        "with preprocess_wlasl_dynamic.py before training."
                    )
                raise ValueError(
                    f"Invalid dynamic dataset shape on line {line_no}: got {seq.shape}, "
                    f"expected {expected_shape}. Regenerate the dataset with "
                    "preprocess_wlasl_dynamic.py."
                )

            if not np.isfinite(seq).all():
                raise ValueError(
                    f"Invalid dynamic dataset on line {line_no}: sequence contains NaN "
                    "or infinite values. Regenerate or clean the dataset."
                )

            sequences.append(seq)
            labels.append(label.strip())

    if not sequences:
        raise ValueError("No valid sequences found in dataset.")

    X = np.stack(sequences, axis=0)   # (N, T, TOTAL_FEATURES_V2)
    y = np.array(labels)              # (N,)
    log.info("Loaded %d sequences across %d classes.", len(X), len(np.unique(y)))
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorch LSTM Model
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm_model(num_classes: int, hidden: int, layers: int):
    """Construct the LSTM classifier using PyTorch."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for training.\n"
            "Install it via: pip install torch\n"
            "Inference uses onnxruntime (no torch required)."
        )

    class GestureLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=0.3 if layers > 1 else 0.0,
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden // 2, num_classes),
            )

        def forward(self, x):
            # x : (batch, seq_len, input_size)
            out, _ = self.lstm(x)
            # Use only the last time-step's hidden state
            last   = out[:, -1, :]
            return self.classifier(last)

    return GestureLSTM()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden: int,
    layers: int,
):
    """Full training loop with early stopping."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training device: %s", device)

    model = build_lstm_model(num_classes, hidden, layers).to(device)

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).long().to(device)
    X_v = torch.from_numpy(X_val).to(device)
    y_v = torch.from_numpy(y_val).long().to(device)

    train_ds = TensorDataset(X_t, y_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None
    patience      = 15
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_v).argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_v.cpu().numpy(), val_preds)

        if epoch % 10 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  loss=%.4f  val_acc=%.4f",
                     epoch, epochs, total_loss / len(train_dl), val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d (best val_acc=%.4f)", epoch, best_val_acc)
                break

    model.load_state_dict(best_state)
    log.info("Best validation accuracy: %.4f", best_val_acc)
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# 4. ONNX export
# ─────────────────────────────────────────────────────────────────────────────
def export_onnx(model, device, out_path: Path) -> None:
    """Export the trained model to ONNX opset 11."""
    import torch

    model.eval()
    dummy = torch.zeros(1, SEQUENCE_LENGTH, INPUT_SIZE, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["landmarks_sequence"],
        output_names=["class_logits"],
        dynamic_axes={
            "landmarks_sequence": {0: "batch_size"},
            "class_logits":       {0: "batch_size"},
        },
        opset_version=11,
    )
    log.info("ONNX model exported → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)

    # ── Load ──────────────────────────────────────────────────────────────────
    X, y_raw = load_dataset(args.dataset)

    # ── Encode labels ─────────────────────────────────────────────────────────
    encoder    = LabelEncoder()
    y_encoded  = encoder.fit_transform(y_raw).astype(np.int64)
    num_classes = len(encoder.classes_)
    log.info("Classes: %s", list(encoder.classes_))

    # ── Normalise per-feature across the time axis ────────────────────────────
    N, T, F = X.shape
    scaler  = StandardScaler()
    X_flat  = X.reshape(-1, F)
    scaler.fit(X_flat)
    X_norm  = scaler.transform(X_flat).reshape(N, T, F).astype(np.float32)

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_encoded,
        test_size=args.test_size,
        random_state=42,
        stratify=y_encoded,
    )
    log.info("Train: %d  |  Val: %d", len(X_train), len(X_val))

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    model, device = train(
        X_train, y_train, X_val, y_val,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden,
        layers=args.layers,
    )
    log.info("Training time: %.1fs", time.time() - t0)

    # ── Final evaluation ──────────────────────────────────────────────────────
    import torch
    model.eval()
    with torch.no_grad():
        import torch
        X_v_t = torch.from_numpy(X_val).to(device)
        preds  = model(X_v_t).argmax(dim=1).cpu().numpy()
    test_acc = accuracy_score(y_val, preds)
    log.info("Test accuracy: %.4f", test_acc)
    log.info("\n%s", classification_report(
        y_val, preds, target_names=encoder.classes_
    ))

    # ── Save artefacts ────────────────────────────────────────────────────────
    onnx_path = model_dir / ONNX_FILENAME
    export_onnx(model, device, onnx_path)

    enc_path = model_dir / ENCODER_FILENAME
    joblib.dump(encoder, enc_path)
    log.info("Label encoder saved → %s", enc_path)

    scaler_path = model_dir / SCALER_FILENAME
    joblib.dump(scaler, scaler_path)
    log.info("Scaler saved → %s", scaler_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM dynamic gesture classifier and export to ONNX."
    )
    parser.add_argument("--dataset",    default=DEFAULT_DATASET,
                        help="Path to dynamic_gestures.jsonl")
    parser.add_argument("--model_dir",  default=DEFAULT_MODEL_DIR,
                        help="Directory to save model artefacts")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--hidden",     type=int,   default=128,
                        help="LSTM hidden size")
    parser.add_argument("--layers",     type=int,   default=2,
                        help="Number of LSTM layers")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--test_size",  type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(_parse_args())
    except (FileNotFoundError, ValueError, ImportError) as exc:
        log.error("%s", exc)
        sys.exit(1)
