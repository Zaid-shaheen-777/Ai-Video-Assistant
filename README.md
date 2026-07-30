# Ai-Video-Assistant

# 🎙️ AI Meeting Assistant

Turn any YouTube video or audio/video file into a transcript, summary, action items, and a searchable chat — powered by Whisper, Gemini, and Mistral AI.

## ✨ Features

- 📝 Auto-transcribes YouTube links or uploaded audio/video files
- 📋 Generates a title + summary
- ✅ Extracts action items, key decisions, and open questions
- 💬 Chat with your meeting (ask questions, get answers from the transcript)
- 🌐 Supports English (Whisper) and Hinglish (Gemini)

## 🛠️ Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed on your system
- API keys:
  - `MISTRAL_API_KEY` — for summarizing and chat
  - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) — for Hinglish transcription

## 🚀 Setup

1. **Clone / download this project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API keys**

   Create a `.env` file in the project root:
   ```
   MISTRAL_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   ```

## ▶️ Run

**Web app (recommended):**
```bash
streamlit run app.py
```

**Command line version:**
```bash
python main.py
```

## 📁 Project Structure

```
project/
├── app.py               # Streamlit web app
├── main.py               # Command-line version
├── requirements.txt
├── core/
│   ├── transcriber.py     # Audio → text (Whisper / Gemini)
│   ├── summarizer.py      # Text → summary & title (Mistral)
│   ├── extractor.py       # Finds action items, decisions, questions
│   ├── vector_store.py    # Stores transcript for search (Chroma)
│   └── rag_engine.py      # Chat with your transcript (RAG)
└── utils/
    └── audio_processor.py # Downloads/converts/chunks audio
```

## ☁️ Deploy to Streamlit Community Cloud (safe)

1. Push this repo to GitHub (**never** commit `.env` or real API keys).
2. Go to [share.streamlit.io](https://share.streamlit.io/) → **New app**.
3. Select repo `Zaid-shaheen-777/Ai-Video-Assistant`, branch `main`, main file `app.py`.
4. In **Advanced settings**, prefer **Python 3.11 or 3.12**.
5. Under **Secrets**, paste (TOML):

```toml
MISTRAL_API_KEY = "your_mistral_key_here"
GEMINI_API_KEY = "your_gemini_key_here"
```

6. Click **Deploy**. FFmpeg is installed via `packages.txt`.

Cloud uses **Gemini** for transcription and **Mistral** for summary/chat/embeddings (no heavy torch/Whisper install).

If install fails after a change: **Manage app → Reboot app**. If Python is stuck on 3.14, recreate the app and pick 3.11/3.12.

**Safety checklist**
- API keys live only in Streamlit Secrets (or local `.env`) — not in git.
- Local Whisper is optional: `pip install -r requirements-local.txt` then set `USE_WHISPER=1`.

## ⚠️ Notes

- Long meetings take longer to transcribe and summarize.
- Vector data is stored locally in a `vector_db/` folder.
