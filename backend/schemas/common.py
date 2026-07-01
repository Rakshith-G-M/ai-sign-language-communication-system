"""Shared schema utilities."""

from __future__ import annotations

import re

from core.config import settings

_SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def normalize_session_id(session_id: str | None) -> str:
    """
    Normalize a client-supplied session ID.

    Invalid or missing values fall back to ``"default"`` so existing frontend
    behaviour is preserved without rejecting requests.
    """
    if not session_id:
        return "default"

    cleaned = session_id.strip()
    if not cleaned:
        return "default"

    if len(cleaned) > settings.max_session_id_length:
        cleaned = cleaned[: settings.max_session_id_length]

    if not _SESSION_ID_PATTERN.match(cleaned):
        return "default"

    return cleaned
