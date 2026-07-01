"""Data collection API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from core.config import settings


class SequenceRequest(BaseModel):
    """Dynamic gesture sequence submitted from the frontend capture loop."""

    label: str = Field(..., description="Gesture label (e.g. HELLO).")
    frames: list[list[float]] = Field(
        ...,
        description="Landmark frames; each frame must contain 63 values (21 × 3).",
    )

    @field_validator("label")
    @classmethod
    def label_must_be_nonempty(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("label must be a non-empty string")
        if len(v) > 64:
            raise ValueError("label must be at most 64 characters")
        return v

    @field_validator("frames")
    @classmethod
    def frames_must_have_correct_width(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) > settings.max_collect_frames:
            raise ValueError(
                f"Too many frames ({len(v)}). Maximum is {settings.max_collect_frames}."
            )
        input_size = settings.collect_input_size
        for i, frame in enumerate(v):
            if len(frame) != input_size:
                raise ValueError(
                    f"frame[{i}] has {len(frame)} values; expected {input_size} (21 × 3)"
                )
        return v


class SequenceSavedResponse(BaseModel):
    status: str
    label: str
    frames: int


class CollectStatsResponse(BaseModel):
    total_samples: int
    classes: dict[str, int]


class ClearLabelResponse(BaseModel):
    status: str
    label: str
    removed: int
