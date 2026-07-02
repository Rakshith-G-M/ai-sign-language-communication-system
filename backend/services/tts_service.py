"""
Text-to-Speech Service
──────────────────────────────────────────────

Provides speech synthesis wrapper around gTTS.
"""

from __future__ import annotations

import io
import logging
from gtts import gTTS, gTTSError

log = logging.getLogger(__name__)


class TTSServiceError(Exception):
    """Raised when speech synthesis fails."""
    pass


class TTSService:
    """
    Thread-safe service to generate browser-playable speech using Google TTS.
    Uses in-memory bytes streams to prevent temporary file creation and cleanup hazards.
    """

    def generate_speech(self, text: str, lang: str = "en") -> bytes:
        """
        Convert text into speech MP3 bytes.

        Args:
            text: The text string to convert.
            lang: Language tag (default: 'en').

        Returns:
            bytes: Synthesised MP3 audio bytes.

        Raises:
            TTSServiceError: If gTTS request fails or returned stream is empty.
        """
        stripped_text = text.strip()
        if not stripped_text:
            raise TTSServiceError("Empty text string provided for synthesis.")

        log.info("Generating speech via gTTS for text: '%s...' (length: %d)",
                 stripped_text[:40], len(stripped_text))

        try:
            tts = gTTS(text=stripped_text, lang=lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()

            if not audio_bytes:
                raise TTSServiceError("Synthesised audio bytes stream was empty.")

            log.info("Speech generated successfully: %d bytes (audio/mpeg)", len(audio_bytes))
            return audio_bytes

        except gTTSError as exc:
            log.error("gTTS API connection failed: %s", exc)
            raise TTSServiceError(f"gTTS API error: {exc}") from exc
        except Exception as exc:
            log.error("Unexpected error during speech synthesis: %s - Type: %s",
                      exc, type(exc).__name__)
            raise TTSServiceError(f"Unexpected speech synthesis error: {exc}") from exc
