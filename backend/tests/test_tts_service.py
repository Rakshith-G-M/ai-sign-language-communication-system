"""Unit tests for the Text-to-Speech Service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from gtts import gTTSError

from services.tts_service import TTSService, TTSServiceError


def test_generate_speech_success():
    service = TTSService()
    with patch("services.tts_service.gTTS") as mock_gtts:
        mock_instance = MagicMock()
        
        def mock_write_to_fp(fp):
            fp.write(b"mock_mp3_data")
            
        mock_instance.write_to_fp.side_effect = mock_write_to_fp
        mock_gtts.return_value = mock_instance

        audio_bytes = service.generate_speech("Hello test")
        assert audio_bytes == b"mock_mp3_data"
        mock_gtts.assert_called_once_with(text="Hello test", lang="en")


def test_generate_speech_empty_text():
    service = TTSService()
    with pytest.raises(TTSServiceError, match="Empty text string"):
        service.generate_speech("")

    with pytest.raises(TTSServiceError, match="Empty text string"):
        service.generate_speech("   ")


def test_generate_speech_gtts_error():
    service = TTSService()
    with patch("services.tts_service.gTTS") as mock_gtts:
        mock_instance = MagicMock()
        mock_instance.write_to_fp.side_effect = gTTSError("Connection failed")
        mock_gtts.return_value = mock_instance

        with pytest.raises(TTSServiceError, match="gTTS API error"):
            service.generate_speech("Hello error")


def test_generate_speech_unexpected_error():
    service = TTSService()
    with patch("services.tts_service.gTTS") as mock_gtts:
        mock_gtts.side_effect = RuntimeError("Something went wrong")

        with pytest.raises(TTSServiceError, match="Unexpected speech synthesis error"):
            service.generate_speech("Hello crash")
