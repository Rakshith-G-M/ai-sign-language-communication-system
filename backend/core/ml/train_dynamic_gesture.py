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
                    models/metadata/dynamic_training_manifest.json

Usage
─────
    python -m core.ml.train_dynamic_gesture
    python -m core.ml.train_dynamic_gesture \\
        --dataset dataset/dynamic_gestures.jsonl \\
        --model_dir models \\
        --epochs 80
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from core.ml.constants import (
    DYNAMIC_MODEL_VERSION,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TOTAL_FEATURES_V2,
)
from core.ml.dataset_validation import validate_dynamic_jsonl
from core.ml.training_utils import (
    set_deterministic_seeds,
    validate_label_encoder,
    validate_onnx_model,
    validate_saved_pickle,
    validate_scaler,
    write_training_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DATASET = "dataset/dynamic_gestures.jsonl"
DEFAULT_MODEL_DIR = "models"
ONNX_FILENAME = "asl_dynamic.onnx"
ENCODER_FILENAME = "dynamic_label_encoder.pkl"
SCALER_FILENAME = "dynamic_scaler.pkl"
MANIFEST_FILENAME = "dynamic_training_manifest.json"

INPUT_SIZE = TOTAL_FEATURES_V2


def load_dataset(jsonl_path: str, *, min_samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the dynamic JSONL dataset."""
    X, y = validate_dynamic_jsonl(jsonl_path, min_samples_per_class=min_samples_per_class)
    log.info("Loaded %d sequences across %d classes.", len(X), len(np.unique(y)))
    return X, y


def build_lstm_model(num_classes: int, hidden: int, layers: int):
    """Construct the LSTM classifier using PyTorch."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training.\n"
            "Install it via: pip install torch\n"
            "Inference uses onnxruntime (no torch required)."
        ) from exc

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
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.classifier(last)

    return GestureLSTM()


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
    seed: int,
):
    """Full training loop with early stopping."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator()
    generator.manual_seed(seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    log.info("Training device: %s", device)

    model = build_lstm_model(num_classes, hidden, layers).to(device)

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).long().to(device)
    X_v = torch.from_numpy(X_val).to(device)
    y_v = torch.from_numpy(y_val).long().to(device)

    train_ds = TensorDataset(X_t, y_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience = 15
    no_improve = 0

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

        model.eval()
        with torch.no_grad():
            val_preds = model(X_v).argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_v.cpu().numpy(), val_preds)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  loss=%.4f  val_acc=%.4f",
                epoch, epochs, total_loss / len(train_dl), val_acc,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d (best val_acc=%.4f)", epoch, best_val_acc)
                break

    model.load_state_dict(best_state)
    log.info("Best validation accuracy: %.4f", best_val_acc)
    return model, device, best_val_acc


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
            "class_logits": {0: "batch_size"},
        },
        opset_version=11,
    )
    log.info("ONNX model exported → %s", out_path)


def run(args: argparse.Namespace) -> None:
    set_deterministic_seeds(args.seed)
    model_dir = Path(args.model_dir)

    X, y_raw = load_dataset(args.dataset, min_samples_per_class=args.min_samples_per_class)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw).astype(np.int64)
    num_classes = len(encoder.classes_)
    log.info("Classes: %s", list(encoder.classes_))

    N, T, F = X.shape
    if T != SEQUENCE_LENGTH or F != TOTAL_FEATURES_V2:
        raise ValueError(
            f"Unexpected dataset shape {X.shape}; expected (N, {SEQUENCE_LENGTH}, {TOTAL_FEATURES_V2})."
        )

    scaler = StandardScaler()
    X_flat = X.reshape(-1, F)
    scaler.fit(X_flat)
    X_norm = scaler.transform(X_flat).reshape(N, T, F).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_encoded,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_encoded,
    )
    log.info("Train: %d  |  Val: %d", len(X_train), len(X_val))

    t0 = time.time()
    model, device, best_val_acc = train(
        X_train, y_train, X_val, y_val,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden,
        layers=args.layers,
        seed=args.seed,
    )
    log.info("Training time: %.1fs", time.time() - t0)

    import torch
    model.eval()
    with torch.no_grad():
        X_v_t = torch.from_numpy(X_val).to(device)
        preds = model(X_v_t).argmax(dim=1).cpu().numpy()
    test_acc = accuracy_score(y_val, preds)
    log.info("Test accuracy: %.4f", test_acc)
    log.info("\n%s", classification_report(y_val, preds, target_names=encoder.classes_))

    onnx_path = model_dir / ONNX_FILENAME
    export_onnx(model, device, onnx_path)

    enc_path = model_dir / ENCODER_FILENAME
    joblib.dump(encoder, enc_path)
    log.info("Label encoder saved → %s", enc_path)

    scaler_path = model_dir / SCALER_FILENAME
    joblib.dump(scaler, scaler_path)
    log.info("Scaler saved → %s", scaler_path)

    loaded_encoder = validate_saved_pickle(enc_path, LabelEncoder, "Dynamic label encoder")
    validate_label_encoder(loaded_encoder, min_classes=1, label="Dynamic label encoder")

    loaded_scaler = validate_saved_pickle(scaler_path, StandardScaler, "Dynamic scaler")
    validate_scaler(loaded_scaler, expected_features=TOTAL_FEATURES_V2, label="Dynamic scaler")

    validate_onnx_model(
        onnx_path,
        input_shape=(1, SEQUENCE_LENGTH, TOTAL_FEATURES_V2),
        output_names=["class_logits"],
    )

    write_training_manifest(
        model_dir,
        MANIFEST_FILENAME,
        {
            "model_version": DYNAMIC_MODEL_VERSION,
            "model_type": "lstm-onnx",
            "dataset_path": str(Path(args.dataset).resolve()),
            "sequence_length": SEQUENCE_LENGTH,
            "feature_dim": TOTAL_FEATURES_V2,
            "n_samples": int(len(X)),
            "n_classes": num_classes,
            "classes": list(encoder.classes_),
            "best_val_accuracy": round(float(best_val_acc), 6),
            "test_accuracy": round(float(test_acc), 6),
            "onnx_opset": 11,
            "artefacts": {
                "onnx_model": ONNX_FILENAME,
                "label_encoder": ENCODER_FILENAME,
                "scaler": SCALER_FILENAME,
            },
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM dynamic gesture classifier and export to ONNX."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--min_samples_per_class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(_parse_args())
    except (FileNotFoundError, ValueError, ImportError, TypeError) as exc:
        log.error("%s", exc)
        sys.exit(1)
