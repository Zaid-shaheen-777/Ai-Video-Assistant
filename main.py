"""
main.py
-------
This is the command-line version of the AI Meeting Assistant.

It runs the whole pipeline:
1. Get audio (from YouTube or a local file) and split into chunks
2. Transcribe the audio into text
3. Generate a title and summary
4. Extract action items, decisions, and questions
5. Let you chat with the transcript using RAG (Q&A)
"""

from dotenv import load_dotenv
load_dotenv()  # load API keys from .env BEFORE importing anything that needs them

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
    """Run the full pipeline on a video/audio source and return all results."""
    print("Starting AI Meeting Assistant...")

    # Step 1: turn the source (URL or file) into audio chunks
    chunks = process_input(source)

    # Step 2: transcribe the audio chunks into one full transcript
    transcript = transcribe_all(chunks, language)
    print(f"Transcript preview (first 300 characters):\n{transcript[:300]}")

    # Step 3: generate a title and a summary
    title = generate_title(transcript)
    summary = summarize(transcript)

    # Step 4: extract useful info
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    # Step 5: build the Q&A engine so we can chat with the transcript later
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def print_results(result: dict):
    """Nicely print out all the results from run_pipeline()."""
    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)


def chat_loop(rag_chain):
    """Let the user ask questions about the transcript until they type 'exit'."""
    print("\nChat with your meeting (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    result = run_pipeline(source, language)
    print_results(result)
    chat_loop(result["rag_chain"])