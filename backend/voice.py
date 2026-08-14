"""Voice input support: transcribes recorded audio to text using Gemini's
multimodal audio understanding (no separate speech-to-text service needed).

Voice *output* (reading answers aloud) is implemented client-side in the UI
using the browser's built-in SpeechSynthesis API — no backend code required
for that direction, so it isn't duplicated here.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

TRANSCRIBE_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

TRANSCRIBE_PROMPT = (
    "Transcribe the spoken audio verbatim as plain text. "
    "Do not add commentary, punctuation guesses beyond natural speech, "
    "or any preamble — return only the transcription."
)


class TranscriptionError(Exception):
    """Raised when Gemini can't transcribe the provided audio."""


def transcribe_audio(
    client: genai.Client | None,
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    model: str = TRANSCRIBE_MODEL,
) -> str:
    """Send recorded audio to Gemini and return the transcribed text.

    Raises TranscriptionError with a friendly message on failure so the
    caller can surface it in the UI instead of crashing the app.
    """
    if client is None:
        raise TranscriptionError("Gemini API key is not configured, so voice input is unavailable.")
    if not audio_bytes:
        raise TranscriptionError("No audio was recorded.")

    try:
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model=model,
            contents=[TRANSCRIBE_PROMPT, audio_part],
        )
        text = (response.text or "").strip()
        if not text:
            raise TranscriptionError("Gemini returned an empty transcription. Try recording again.")
        return text
    except TranscriptionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Voice transcription failed: {exc}") from exc
