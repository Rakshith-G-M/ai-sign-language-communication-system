"""Tests for centralized configuration and shared utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from core.config import Settings, get_settings
from schemas.common import normalize_session_id


def test_project_root_resolves_to_repo_root():
    settings = Settings()
    assert settings.project_root.name == "ai-sign-language-communication-system"
    assert (settings.project_root / "backend").is_dir()


def test_dataset_path_under_project_root():
    settings = Settings()
    assert settings.dynamic_dataset_path == settings.project_root / "dataset" / "dynamic_gestures.jsonl"


def test_cors_origin_list_wildcard_default():
    settings = Settings(cors_origins="*")
    assert settings.cors_origin_list == ["*"]


def test_cors_origin_list_parses_comma_separated(monkeypatch):
    settings = Settings(cors_origins="https://a.example.com, https://b.example.com")
    assert settings.cors_origin_list == ["https://a.example.com", "https://b.example.com"]


def test_settings_singleton_cached():
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "default"),
        ("", "default"),
        ("   ", "default"),
        ("valid_session_123", "valid_session_123"),
        ("bad session!", "default"),
        ("a" * 200, "a" * 128),
    ],
)
def test_normalize_session_id(raw, expected):
    assert normalize_session_id(raw) == expected
