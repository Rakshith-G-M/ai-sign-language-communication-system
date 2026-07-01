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

import json
import logging
from pathlib import Path
from collections import Counter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/collect", tags=["data-collection"])

# ─────────────────────────────────────────────────────────────────────────────
# Dataset path — same file that train_dynamic_gesture.py reads
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
DATASET_PATH = _BASE_DIR / "dataset" / "dynamic_gestures.jsonl"

SEQUENCE_LENGTH = 30
INPUT_SIZE      = 63

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class SequenceRequest(BaseModel):
    label:  str
    frames: list[list[float]]   # shape: (N, 63) — N can vary; we truncate/pad

    @field_validator("label")
    @classmethod
    def label_must_be_nonempty(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("label must be a non-empty string")
        return v

    @field_validator("frames")
    @classmethod
    def frames_must_have_correct_width(cls, v: list[list[float]]) -> list[list[float]]:
        for i, frame in enumerate(v):
            if len(frame) != INPUT_SIZE:
                raise ValueError(
                    f"frame[{i}] has {len(frame)} values; expected {INPUT_SIZE} (21 × 3)"
                )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_all_records() -> list[dict]:
    """Read JSONL into a list of dicts.  Returns [] if file absent."""
    if not DATASET_PATH.exists():
        return []
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
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
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _pad_or_truncate(frames: list[list[float]]) -> list[list[float]]:
    """Ensure the sequence is exactly SEQUENCE_LENGTH frames long."""
    if len(frames) >= SEQUENCE_LENGTH:
        return frames[-SEQUENCE_LENGTH:]
    # Left-pad with zero frames
    padding = [[0.0] * INPUT_SIZE] * (SEQUENCE_LENGTH - len(frames))
    return padding + frames


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/sequence", status_code=201)
async def save_sequence(request: SequenceRequest):
    """
    Accept a single dynamic gesture sequence from the frontend and append
    it to the JSONL dataset file.
    """
    if len(request.frames) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence too short ({len(request.frames)} frames). Minimum 5 required.",
        )

    frames = _pad_or_truncate(request.frames)
    record = {"label": request.label, "frames": frames}

    try:
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        log.error("Failed to write dataset: %s", exc)
        raise HTTPException(status_code=500, detail="Could not write dataset file.")

    log.info("Collected sequence — label=%s  frames=%d", request.label, len(frames))
    return {"status": "saved", "label": request.label, "frames": len(frames)}


@router.get("/stats")
async def get_stats():
    """
    Return per-class sample counts from the current dataset.
    Useful for monitoring collection progress from the frontend.
    """
    records = _load_all_records()
    counts  = Counter(r.get("label", "UNKNOWN") for r in records)
    return {
        "total_samples": len(records),
        "classes":       dict(sorted(counts.items())),
    }


@router.delete("/clear")
async def clear_label(label: str):
    """
    Remove all samples for a given gesture label.
    Useful when re-recording low-quality data.
    """
    label = label.strip().upper()
    if not label:
        raise HTTPException(status_code=400, detail="label query parameter is required.")

    records  = _load_all_records()
    filtered = [r for r in records if r.get("label") != label]
    removed  = len(records) - len(filtered)

    _write_all_records(filtered)
    log.info("Removed %d samples for label=%s", removed, label)
    return {"status": "cleared", "label": label, "removed": removed}
