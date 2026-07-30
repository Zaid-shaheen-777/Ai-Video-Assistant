"""
transcriber.py
--------------
This file turns audio chunks into text (transcription).

We support two "engines":
1. Whisper (runs on your own computer) -> used for English audio
2. Gemini (Google's AI, via API) -> used for Hinglish audio
   (Hinglish = Hindi + English mixed speech)

Beginner note: think of this file as a translator between
"audio file" and "text".
"""

import os
import re
import tempfile
from pathlib import Path
from shutil import copy2

from dotenv import load_dotenv
import whisper
from google import genai

load_dotenv()  # loads variables from your .env file (like API keys)

# -----------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------

# Which size of Whisper model to use. Options: tiny, base, small, medium, large
# Bigger = more accurate but slower. "small" is a good balance for beginners.
# Default "base" is Cloud-friendly; use "small" locally via .env if you want.
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Set up the Gemini client only if we have an API key
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("Warning: No GEMINI_API_KEY found. Hinglish transcription won't work.")

# We only want to load the Whisper model once (it's slow to load),
# so we keep it in this variable and reuse it.
_whisper_model = None


def prepare_upload_path(chunk_path: str) -> str:
    """Return a filesystem-safe path for Gemini uploads."""
    path = Path(chunk_path)
    if not path.exists():
        return str(path)

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name)
    safe_name = safe_name.strip("._")
    safe_name = re.sub(r"_+", "_", safe_name)
    if not safe_name:
        safe_name = "audio_file"

    if safe_name != path.name:
        safe_path = path.with_name(safe_name)
        copy2(path, safe_path)
        return str(safe_path)

    return str(path)


def load_whisper_model():
    """Load the Whisper model into memory (only once)."""
    global _whisper_model

    if _whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL_NAME} ...")
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
        print("Whisper model loaded.")

    return _whisper_model


def transcribe_with_whisper(chunk_path: str) -> str:
    """Transcribe one audio chunk using Whisper (good for English)."""
    model = load_whisper_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def transcribe_with_gemini(chunk_path: str) -> str:
    """Transcribe one audio chunk using Gemini (good for Hinglish)."""
    if gemini_client is None:
        raise ValueError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your .env file."
        )

    upload_path = prepare_upload_path(chunk_path)
    uploaded_file = gemini_client.files.upload(file=upload_path)

    instructions = """
Transcribe this audio into English.

Rules:
- Return only the transcript.
- Don't explain anything.
- Don't summarize.
- Don't add extra text.
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, instructions],
    )

    return response.text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Pick the right engine based on the chosen language,
    and transcribe a single audio chunk.
    """
    if language.lower() == "hinglish":
        return transcribe_with_gemini(chunk_path)
    else:
        return transcribe_with_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """
    Transcribe a list of audio chunks and join them into one
    full transcript (a single string).
    """
    engine_name = "Gemini" if language.lower() == "hinglish" else "Whisper"
    print(f"Transcribing using {engine_name}...")

    all_text_pieces = []

    for index, chunk_path in enumerate(chunks):
        print(f"Transcribing chunk {index + 1} of {len(chunks)}...")
        text = transcribe_chunk(chunk_path, language)
        all_text_pieces.append(text)

    full_transcript = " ".join(all_text_pieces).strip()
    print("Transcription complete.")

    return full_transcript