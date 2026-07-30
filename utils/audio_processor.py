"""
audio_processor.py
-------------------
This file handles getting audio ready for transcription.

It can:
1. Download audio from a YouTube link
2. Convert any local audio/video file to WAV format
3. Split a long audio file into small chunks (so transcription is faster/easier)

Beginner note: "WAV" is a simple, uncompressed audio format that most
speech-to-text tools (like Whisper) understand well.
"""

import os
import yt_dlp
from pydub import AudioSegment

# Folder where downloaded YouTube audio will be saved
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Download the audio from a YouTube video and save it as a WAV file.
    Returns the path to the downloaded WAV file.
    """
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_options = {
        "format": "bestaudio/best",       # get the best quality audio
        "outtmpl": output_path,           # where/how to name the file
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,  # don't print all of yt-dlp's internal logs
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    # yt-dlp sometimes reports the original extension (webm/m4a) even
    # though we converted to wav — fix the filename so it matches.
    filename = os.path.splitext(filename)[0] + ".wav"
    return filename


def convert_to_wav(input_path: str) -> str:
    """
    Convert any local audio or video file into a 16kHz mono WAV file.
    (16kHz mono is the format Whisper expects.)
    """
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)        # mono (1 channel)
    audio = audio.set_frame_rate(16000)  # 16,000 Hz sample rate

    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """
    Split a long WAV file into smaller pieces (chunks), each
    `chunk_minutes` long. Returns a list of file paths, one per chunk.

    Why chunk? Long audio files can be slow or memory-heavy to
    transcribe all at once. Smaller pieces are easier to process.
    """
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = chunk_minutes * 60 * 1000  # convert minutes -> milliseconds

    chunk_paths = []
    total_length_ms = len(audio)

    start = 0
    chunk_number = 0
    while start < total_length_ms:
        end = start + chunk_length_ms
        chunk = audio[start:end]

        chunk_path = f"{wav_path}_chunk_{chunk_number}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

        start = end
        chunk_number += 1

    return chunk_paths


def process_input(source: str) -> list:
    """
    Main entry point for this file.
    Takes either a YouTube URL or a local file path, and returns
    a list of audio chunk file paths ready for transcription.
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