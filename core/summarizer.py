"""
summarizer.py
-------------
This file uses an AI model (Mistral) to:
1. Summarize a long meeting transcript
2. Generate a short title for the meeting

Beginner note: Long transcripts are too big to send to the AI all at
once, so we use a "map-reduce" style approach:
  - MAP step:   split the transcript into chunks, summarize each chunk
  - REDUCE step: combine all the small summaries into one final summary
"""

import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():
    """Create a connection to the Mistral AI chat model."""
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,  # lower = more focused/consistent answers
    )


def split_transcript(transcript: str) -> list:
    """Break a long transcript into smaller text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,     # roughly how many characters per chunk
        chunk_overlap=200,   # a little overlap so we don't lose context between chunks
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """
    Summarize a full meeting transcript.
    Step 1: summarize each chunk separately (map step)
    Step 2: combine those summaries into one final summary (reduce step)
    """
    llm = get_llm()

    # ---- Step 1: summarize each chunk ----
    chunk_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])

    chunks = split_transcript(transcript)

    chunk_summaries = []
    for chunk in chunks:
        prompt_value = chunk_summary_prompt.invoke({"text": chunk})
        response = llm.invoke(prompt_value)
        summary_text = StrOutputParser().invoke(response)
        chunk_summaries.append(summary_text)

    combined_summaries = "\n\n".join(chunk_summaries)

    # ---- Step 2: combine all chunk summaries into one final summary ----
    final_summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial "
            "summaries into one final professional meeting summary in "
            "bullet points.",
        ),
        ("human", "{text}"),
    ])

    prompt_value = final_summary_prompt.invoke({"text": combined_summaries})
    response = llm.invoke(prompt_value)
    final_summary = StrOutputParser().invoke(response)

    return final_summary


def generate_title(transcript: str) -> str:
    """Generate a short, professional title for the meeting."""
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the meeting transcript, generate a short professional "
            "meeting title (max 8 words). Only return the title, nothing else.",
        ),
        ("human", "{text}"),
    ])

    # We only need the first part of the transcript to guess a good title
    short_text = transcript[:2000]

    prompt_value = title_prompt.invoke({"text": short_text})
    response = llm.invoke(prompt_value)
    title = StrOutputParser().invoke(response)

    return title