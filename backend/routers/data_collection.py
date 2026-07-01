"""
Data Collection Router
──────────────────────
Provides API endpoints for logging dynamic gesture sequences
from the frontend.  The frontend sends batches of landmarks from
its existing MediaPipe capture loop; the backend writes them to
the same JSONL format consumed by train_dynamic_gesture.py.

Endpoints
─────────
    POST /api/v1/collect/sequence
        Body : {"label": "HELLO", "frames": [[x,y,z,...] × 63] × 30}
        Saves a single gesture sequence to dataset/dynamic_gestures.jsonl.

    GET  /api/v1/collect/stats
        Returns class distribution in the current dataset file.

    DELETE /api/v1/collect/clear?label=HELLO
        Removes all samples for a given label from the dataset.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from fastapi import APIRouter, HTTPException

from core.config import settings
from schemas.data_collection import (
    ClearLabelResponse,
    CollectStatsResponse,
    SequenceRequest,
    SequenceSavedResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/collect", tags=["data-collection"])


def _dataset_path():
    return settings.dynamic_dataset_path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_all_records() -> list[dict]:
    """Read JSONL into a list of dicts.  Returns [] if file absent."""
    path = _dataset_path()
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _write_all_records(records: list[dict]) -> None:
    """Rewrite the whole JSONL file from a list of dicts."""
    path = _dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _pad_or_truncate(frames: list[list[float]]) -> list[list[float]]:
    """Ensure the sequence is exactly SEQUENCE_LENGTH frames long."""
    seq_len = settings.collect_sequence_length
    input_size = settings.collect_input_size
    if len(frames) >= seq_len:
        return frames[-seq_len:]
    padding = [[0.0] * input_size] * (seq_len - len(frames))
    return padding + frames


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/sequence",
    status_code=201,
    response_model=SequenceSavedResponse,
    summary="Save a dynamic gesture sequence",
)
async def save_sequence(request: SequenceRequest) -> dict:
    """
    Accept a single dynamic gesture sequence from the frontend and append
    it to the JSONL dataset file.
    """
    if len(request.frames) < settings.collect_min_frames:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sequence too short ({len(request.frames)} frames). "
                f"Minimum {settings.collect_min_frames} required."
            ),
        )

    frames = _pad_or_truncate(request.frames)
    record = {"label": request.label, "frames": frames}
    path = _dataset_path()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        log.error("Failed to write dataset: %s", exc)
        raise HTTPException(status_code=500, detail="Could not write dataset file.") from exc

    log.info("Collected sequence — label=%s  frames=%d", request.label, len(frames))
    return {"status": "saved", "label": request.label, "frames": len(frames)}


@router.get(
    "/stats",
    response_model=CollectStatsResponse,
    summary="Dataset class distribution",
)
async def get_stats() -> dict:
    """
    Return per-class sample counts from the current dataset.
    Useful for monitoring collection progress from the frontend.
    """
    records = _load_all_records()
    counts = Counter(r.get("label", "UNKNOWN") for r in records)
    return {
        "total_samples": len(records),
        "classes": dict(sorted(counts.items())),
    }


@router.delete(
    "/clear",
    response_model=ClearLabelResponse,
    summary="Remove samples for a label",
)
async def clear_label(label: str) -> dict:
    """
    Remove all samples for a given gesture label.
    Useful when re-recording low-quality data.
    """
    label = label.strip().upper()
    if not label:
        raise HTTPException(status_code=400, detail="label query parameter is required.")

    records = _load_all_records()
    filtered = [r for r in records if r.get("label") != label]
    removed = len(records) - len(filtered)

    _write_all_records(filtered)
    log.info("Removed %d samples for label=%s", removed, label)
    return {"status": "cleared", "label": label, "removed": removed}
