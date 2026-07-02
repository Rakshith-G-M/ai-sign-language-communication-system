"""HTTP integration tests for the ASL backend API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from main import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_metadata(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["api_base"] == "/api/v1"


def test_liveness_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_legacy_api_health_preserved(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_returns_structured_checks(client):
    response = client.get("/ready")
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert set(data["checks"].keys()) == {
        "static_model",
        "mediapipe",
        "prediction_service",
        "dynamic_predictor",
    }


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "active_sessions" in data
    assert "total_predictions" in data
    assert "static_predictions" in data
    assert "dynamic_predictions" in data
    assert data["dynamic_predictions"] == 0


def test_predict_invalid_mime_returns_400(client):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("frame.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_invalid_image_returns_200_degraded(client):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("frame.jpg", b"not-a-valid-image", "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["letter"] is None
    assert data["hand_detected"] is False
    assert data["confidence"] == 0.0


def test_predict_empty_file_returns_200_degraded(client):
    response = client.post(
        "/api/v1/predict",
        files={"file": ("frame.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["letter"] is None
    assert data["hand_detected"] is False


def test_tts_empty_text_returns_400(client):
    response = client.post("/api/v1/tts", json={"text": "   "})
    assert response.status_code == 400


def test_tts_valid_text_returns_200_and_audio(client):
    from unittest.mock import patch, MagicMock

    with patch("services.tts_service.gTTS") as mock_gtts:
        mock_instance = MagicMock()
        
        # When write_to_fp is called, write fake audio bytes to the stream
        def mock_write_to_fp(fp):
            fp.write(b"fake_audio_bytes")
            
        mock_instance.write_to_fp.side_effect = mock_write_to_fp
        mock_gtts.return_value = mock_instance
        
        response = client.post("/api/v1/tts", json={"text": "Hello world"})
        assert response.status_code == 200
        assert response.content == b"fake_audio_bytes"
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.headers["content-length"] == str(len(b"fake_audio_bytes"))



def test_reset_and_state_session_isolation(client):
    client.post("/api/v1/reset", params={"session_id": "session_a"})
    client.post("/api/v1/reset", params={"session_id": "session_b"})

    state_a = client.get("/api/v1/state", params={"session_id": "session_a"}).json()
    state_b = client.get("/api/v1/state", params={"session_id": "session_b"}).json()
    assert state_a == {"word": "", "sentence": ""}
    assert state_b == {"word": "", "sentence": ""}


def test_info_lists_endpoints(client):
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    endpoints = response.json()["endpoints"]
    assert "/predict" in endpoints
    assert "/tts" in endpoints
    assert "/health" in endpoints


