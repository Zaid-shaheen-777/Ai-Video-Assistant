"""
app.py
------
A polished Streamlit web app for the AI Meeting Assistant.

Run it with:
    streamlit run app.py

What it does:
1. Lets you enter a YouTube link or upload a local audio/video file
2. Transcribes it, summarizes it, and pulls out action items,
   decisions, and open questions
3. Lets you chat with the transcript (ask questions about it)

Beginner note: Streamlit re-runs this whole script from top to bottom
every time you interact with the page (click a button, type in a box,
etc). That's why we use `st.session_state` to "remember" things
(like the transcript) between those re-runs. All the styling below
is plain CSS injected once at the top — everything else is normal
Streamlit code.
"""

import os
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Local development: load .env (ignored on Streamlit Cloud — use Secrets there)
load_dotenv()


def _apply_streamlit_secrets() -> None:
    """Copy Streamlit Secrets into os.environ for the rest of the app."""
    try:
        secret_map = dict(st.secrets)
    except Exception:
        return

    wanted = (
        "MISTRAL_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "WHISPER_MODEL",
        "USE_WHISPER",
    )
    for key in wanted:
        if key not in secret_map:
            continue
        value = secret_map[key]
        if value is None:
            continue
        text = str(value).strip().strip('"').strip("'")
        if text:
            os.environ[key] = text


_apply_streamlit_secrets()

from utils.audio_processor import process_input, YouTubeDownloadError
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question



# =============================================================
# PAGE SETUP
# =============================================================

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# CUSTOM STYLING
# -------------------------------------------------------------
# Beginner note: this is just a block of CSS wrapped in
# st.markdown(..., unsafe_allow_html=True). Streamlit lets you
# inject raw CSS this way to restyle the default look.
# =============================================================

CUSTOM_CSS = """
<style>
    /* ---------- Import a nicer font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- App background ---------- */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1b1035 0%, #0d0a1f 45%, #08060f 100%);
        color: #ECEAF6;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #150c2e 0%, #0d0a1f 100%);
        border-right: 1px solid rgba(167, 139, 250, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #E9D8FD;
    }

    /* ---------- Hero header ---------- */
    .hero-wrap {
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        margin-bottom: 1.8rem;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.35), rgba(236, 72, 153, 0.20));
        border: 1px solid rgba(196, 181, 253, 0.25);
        box-shadow: 0 20px 60px rgba(124, 58, 237, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(236,72,153,0.35), transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #F5D0FE, #C4B5FD 45%, #93C5FD);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        margin-top: 0.5rem;
        font-size: 1.02rem;
        color: #C9C3E0;
        font-weight: 400;
        max-width: 640px;
    }

    /* ---------- Section / card containers ---------- */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    /* ---------- Metric chips row ---------- */
    .chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.6rem 0 0.2rem 0; }
    .chip {
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.35);
        color: #DCD3FB;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #8B5CF6, #EC4899);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1rem;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(236, 72, 153, 0.45);
        color: white;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.03);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #B9B3D6;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(139,92,246,0.35), rgba(236,72,153,0.25));
        color: #FDF4FF !important;
    }

    /* ---------- Highlight list items ---------- */
    .item-card {
        background: rgba(255,255,255,0.035);
        border-left: 3px solid #A78BFA;
        border-radius: 8px;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
        color: #E4E1F5;
    }
    .item-card.decision { border-left-color: #34D399; }
    .item-card.question { border-left-color: #60A5FA; }
    .empty-note { color: #7B7594; font-style: italic; font-size: 0.9rem; }

    /* ---------- Title badge ---------- */
    .title-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: rgba(139, 92, 246, 0.18);
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #E9D8FD;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* ---------- Chat bubbles polish ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.3rem 0.2rem;
    }

    footer, header {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================
# SESSION STATE (Streamlit's way of remembering data between reruns)
# =============================================================

if "result" not in st.session_state:
    st.session_state.result = None  # will hold transcript, summary, etc.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) pairs


# =============================================================
# HERO HEADER
# =============================================================

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">🎙️ AI Meeting Assistant</div>
        <div class="hero-subtitle">
            Drop in a YouTube link or a recording — get a clean transcript,
            an instant summary, action items, and a chat window to ask
            anything about what was said.
        </div>
        <div class="chip-row">
            <span class="chip">📝 Auto-transcription</span>
            <span class="chip">📋 Smart summary</span>
            <span class="chip">✅ Action items</span>
            <span class="chip">💬 Chat with your meeting</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# SIDEBAR: input options
# =============================================================

with st.sidebar:
    st.markdown("### 1️⃣ Provide your meeting")

    # Helpful Cloud hint if keys are missing
    _has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    _has_mistral = bool(os.getenv("MISTRAL_API_KEY"))
    if not _has_gemini or not _has_mistral:
        missing = []
        if not _has_gemini:
            missing.append("GEMINI_API_KEY")
        if not _has_mistral:
            missing.append("MISTRAL_API_KEY")
        st.warning(
            "Missing secrets: "
            + ", ".join(missing)
            + ". Set them in Manage app → Settings → Secrets (TOML). "
            + "Cloud does **not** read your local `.env` file."
        )

    input_type = st.radio("Choose input type:", ["YouTube URL", "Upload a file"], label_visibility="collapsed")

    source = None  # this will end up holding a URL or a local file path

    if input_type == "YouTube URL":
        youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("Note: YouTube downloads often fail on Streamlit Cloud. Prefer **Upload a file** there.")
        if youtube_url:
            source = youtube_url

    else:
        uploaded_file = st.file_uploader(
            "Upload an audio or video file", type=["mp3", "wav", "m4a", "mp4", "mov"]
        )
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location so our
            # existing functions (which expect a file PATH) can use it.
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = temp_path

    st.markdown("### 2️⃣ Language")
    language = st.selectbox("Language", ["english", "hinglish"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("🚀 Process meeting", type="primary", use_container_width=True)

    if st.session_state.result is not None:
        st.markdown("---")
        st.caption(f"✅ Last processed: {datetime.now().strftime('%b %d, %I:%M %p')}")
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()


# =============================================================
# MAIN PIPELINE: runs when the user clicks the button
# =============================================================

if run_button:
    if not source:
        st.warning("Please provide a YouTube URL or upload a file first.")
    else:
        try:
            with st.status("Processing your meeting...", expanded=True) as status:
                st.write("🔊 Step 1/4 — Preparing audio...")
                chunks = process_input(source)

                st.write("📝 Step 2/4 — Transcribing audio (this can take a while)...")
                transcript = transcribe_all(chunks, language)

                st.write("🧠 Step 3/4 — Generating title, summary, and highlights...")
                title = generate_title(transcript)
                summary = summarize(transcript)
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                st.write("🔎 Step 4/4 — Setting up Q&A engine...")
                rag_chain = build_rag_chain(transcript)

                status.update(label="✨ Done!", state="complete", expanded=False)

            # Save everything so it survives future reruns (e.g. when chatting)
            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.chat_history = []  # reset chat for the new meeting
            st.rerun()

        except YouTubeDownloadError as err:
            st.error(str(err))
            st.info(
                "Tip: On Streamlit Cloud, **Upload a file** is the most reliable option. "
                "Download the YouTube video on your PC (or export audio), then upload mp3/mp4/m4a/wav."
            )
        except Exception as err:
            st.error(f"Something went wrong while processing: {err}")


# =============================================================
# DISPLAY RESULTS
# =============================================================

result = st.session_state.result

if result is None:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding: 3rem 1.5rem;">
            <div style="font-size:2.4rem;">👋</div>
            <div style="font-size:1.15rem; font-weight:700; color:#E9D8FD; margin-top:0.4rem;">
                Nothing processed yet
            </div>
            <div class="empty-note" style="margin-top:0.3rem;">
                Add a YouTube URL or upload a file in the sidebar, then click
                <b style="color:#DCD3FB;">"Process meeting"</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="title-badge">Meeting title</div>', unsafe_allow_html=True)
    st.markdown(f"## 📌 {result['title']}")

    word_count = len(result["transcript"].split())
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("📝 Transcript length", f"{word_count:,} words")
    col_b.metric("✅ Action items", len(result["action_items"]))
    col_c.metric("🔑 Key decisions", len(result["key_decisions"]))

    st.markdown("<br>", unsafe_allow_html=True)

    tab_summary, tab_highlights, tab_transcript, tab_chat = st.tabs(
        ["📋  Summary", "✅  Highlights", "📝  Full Transcript", "💬  Chat"]
    )

    with tab_summary:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(result["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_highlights:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### ✅ Action Items")
            if result["action_items"]:
                for item in result["action_items"]:
                    st.markdown(f'<div class="item-card">{item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty-note">None found.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("#### 🔑 Key Decisions")
            if result["key_decisions"]:
                for item in result["key_decisions"]:
                    st.markdown(f'<div class="item-card decision">{item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty-note">None found.</div>', unsafe_allow_html=True)

        with col3:
            st.markdown("#### ❓ Open Questions")
            if result["open_questions"]:
                for item in result["open_questions"]:
                    st.markdown(f'<div class="item-card question">{item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty-note">None found.</div>', unsafe_allow_html=True)

    with tab_transcript:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.text_area("Full transcript", result["transcript"], height=420, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_chat:
        st.markdown("#### 💬 Chat with your meeting")
        st.caption("Ask anything — answers are grounded only in the transcript.")

        # Show past chat messages
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)

        # Input box for a new question
        new_question = st.chat_input("Ask a question about the meeting...")

        if new_question:
            with st.chat_message("user"):
                st.write(new_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(result["rag_chain"], new_question)
                st.write(answer)

            st.session_state.chat_history.append((new_question, answer))