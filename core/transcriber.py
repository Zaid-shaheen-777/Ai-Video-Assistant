"""
transcriber.py
--------------
Turns audio chunks into text (transcription).

Default engine: Gemini API (works on Streamlit Cloud — no local GPU/torch).
Optional engine: Whisper, if installed locally and USE_WHISPER=1.
"""

import os
import re
from pathlib import Path
from shutil import copy2

from dotenv import load_dotenv
from google import genai

load_dotenv()

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

_whisper_model = None
_whisper_module = None
_gemini_client = None


def _get_gemini_api_key() -> str | None:
    """Read the key at call time (after Streamlit Secrets are applied)."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    return key.strip().strip('"').strip("'") or None


def get_gemini_client():
    """Create / reuse the Gemini client using the current env key."""
    global _gemini_client
    api_key = _get_gemini_api_key()
    if not api_key:
        return None
    # Recreate if key changed (e.g. secrets updated + reboot)
    if _gemini_client is None or getattr(_gemini_client, "_api_key", None) != api_key:
        _gemini_client = genai.Client(api_key=api_key)
        _gemini_client._api_key = api_key
    return _gemini_client


def _try_import_whisper():
    """Import openai-whisper only if installed (optional local dependency)."""
    global _whisper_module
    if _whisper_module is not None:
        return _whisper_module
    try:
        import whisper as whisper_mod
        _whisper_module = whisper_mod
        return _whisper_module
    except ImportError:
        return None


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
    whisper_mod = _try_import_whisper()
    if whisper_mod is None:
        raise ImportError(
            "openai-whisper is not installed. "
            "Use Gemini (default) or pip install -r requirements-local.txt"
        )

    if _whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL_NAME} ...")
        _whisper_model = whisper_mod.load_model(WHISPER_MODEL_NAME)
        print("Whisper model loaded.")

    return _whisper_model


def transcribe_with_whisper(chunk_path: str) -> str:
    """Transcribe one audio chunk using Whisper (local, English)."""
    model = load_whisper_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def transcribe_with_gemini(chunk_path: str) -> str:
    """Transcribe one audio chunk using Gemini (Cloud-friendly)."""
    gemini_client = get_gemini_client()
    if gemini_client is None:
        raise ValueError(
            "Gemini API key is missing. In Streamlit Cloud go to "
            "Manage app → Settings → Secrets and paste TOML like:\n"
            'GEMINI_API_KEY = "your_key_here"\n'
            'MISTRAL_API_KEY = "your_key_here"\n'
            "(A .env file is only used when running locally — Cloud ignores it.)"
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


def _should_use_whisper(language: str) -> bool:
    """Whisper only when explicitly enabled and available (local)."""
    if language.lower() == "hinglish":
        return False
    flag = os.getenv("USE_WHISPER", "0").strip().lower()
    if flag not in ("1", "true", "yes"):
        return False
    return _try_import_whisper() is not None


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Pick the engine and transcribe a single audio chunk."""
    if _should_use_whisper(language):
        return transcribe_with_whisper(chunk_path)
    return transcribe_with_gemini(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe all chunks and join into one transcript string."""
    engine_name = "Whisper" if _should_use_whisper(language) else "Gemini"
    print(f"Transcribing using {engine_name}...")

    all_text_pieces = []
    for index, chunk_path in enumerate(chunks):
        print(f"Transcribing chunk {index + 1} of {len(chunks)}...")
        all_text_pieces.append(transcribe_chunk(chunk_path, language))

    full_transcript = " ".join(all_text_pieces).strip()
    print("Transcription complete.")
    return full_transcript
