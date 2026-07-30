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
4. Under **Advanced settings → Secrets**, paste (TOML):

```toml
MISTRAL_API_KEY = "your_mistral_key_here"
GEMINI_API_KEY = "your_gemini_key_here"
WHISPER_MODEL = "base"
```

5. Click **Deploy**. FFmpeg is installed via `packages.txt`.

**Safety checklist**
- API keys live only in Streamlit Secrets (or local `.env`) — not in git.
- Use `WHISPER_MODEL = "tiny"` or `"base"` on the free Cloud tier (less RAM).
- Whisper + PyTorch are heavy; if the app runs out of memory, switch language to **hinglish** (Gemini API) or set `WHISPER_MODEL = "tiny"`.

## ⚠️ Notes

- First run downloads the Whisper model — this may take a minute.
- Long meetings take longer to transcribe and summarize.
- Vector data is stored locally in a `vector_db/` folder.