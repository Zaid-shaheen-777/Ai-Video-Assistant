"""
audio_processor.py
-------------------
This file handles getting audio ready for transcription.

It can:
1. Download audio from a YouTube link
2. Convert any local audio/video file to WAV format
3. Split a long audio file into small chunks (so transcription is faster/easier)
"""

import os
import tempfile

import yt_dlp
from pydub import AudioSegment
from yt_dlp.utils import DownloadError

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class YouTubeDownloadError(Exception):
    """Raised when YouTube audio cannot be downloaded (common on cloud IPs)."""


def download_youtube_audio(url: str) -> str:
    """
    Download the audio from a YouTube video and convert it to WAV.
    Returns the path to the WAV file.
    """
    # Use video id in the filename to avoid OS-unfriendly titles
    output_tmpl = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    ydl_options = {
        # Prefer a simple audio stream; convert to WAV ourselves with pydub
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "outtmpl": output_tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 1,
        # Android/iOS clients often work better than the default web client
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"],
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = ydl.prepare_filename(info)
    except DownloadError as err:
        raise YouTubeDownloadError(
            "YouTube blocked the download from this server (common on Streamlit Cloud). "
            "Please use 'Upload a file' instead — download the video/audio on your computer "
            "and upload the file here."
        ) from err
    except Exception as err:
        raise YouTubeDownloadError(
            f"Could not download YouTube audio: {err}. "
            "Try uploading the file instead."
        ) from err

    if not os.path.exists(downloaded):
        # Sometimes extension differs from prepare_filename guess
        video_id = info.get("id")
        candidates = [
            os.path.join(DOWNLOAD_DIR, name)
            for name in os.listdir(DOWNLOAD_DIR)
            if video_id and name.startswith(video_id)
        ]
        if not candidates:
            raise YouTubeDownloadError(
                "Download finished but the audio file was not found. "
                "Please upload the file instead."
            )
        downloaded = max(candidates, key=os.path.getmtime)

    return convert_to_wav(downloaded)


def convert_to_wav(input_path: str) -> str:
    """
    Convert any local audio or video file into a 16kHz mono WAV file.
    """
    # Write into a temp path so we don't collide with source names on Cloud
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(tempfile.gettempdir(), f"{base}_converted.wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """
    Split a long WAV file into smaller pieces and return their paths.
    """
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = chunk_minutes * 60 * 1000

    chunk_paths = []
    total_length_ms = len(audio)

    start = 0
    chunk_number = 0
    while start < total_length_ms:
        end = start + chunk_length_ms
        chunk = audio[start:end]

        chunk_path = os.path.join(
            tempfile.gettempdir(),
            f"{os.path.basename(wav_path)}_chunk_{chunk_number}.wav",
        )
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

        start = end
        chunk_number += 1

    return chunk_paths


def process_input(source: str) -> list:
    """
    Take a YouTube URL or local file path; return audio chunk paths
    ready for transcription.
    """
    is_url = source.startswith("http://") or source.startswith("https://")

    if is_url:
        print("Detected a YouTube link. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected a local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Splitting audio into chunks...")
    chunks = chunk_audio(wav_path)
    print(f"Done! Created {len(chunks)} chunk(s).")

    return chunks
