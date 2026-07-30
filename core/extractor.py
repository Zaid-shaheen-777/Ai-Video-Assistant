"""
extractor.py
------------
This file pulls out useful bits of information from a transcript
using simple keyword matching (no AI needed here!):

1. Action items   - things people need to do
2. Key decisions  - things the group agreed on
3. Open questions - things that were asked but maybe not answered

Beginner note: This works by splitting the transcript into sentences,
then checking if each sentence contains certain keywords.
"""

import re
from typing import List


def split_into_sentences(transcript: str) -> List[str]:
    """Split a block of text into a list of individual sentences."""
    if not transcript or not transcript.strip():
        return []

    # Split after ., !, or ? followed by a space
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    return sentences


def find_sentences_with_keywords(transcript: str, keywords: List[str]) -> List[str]:
    """Return every sentence in the transcript that contains at least one keyword."""
    sentences = split_into_sentences(transcript)
    matching_sentences = []

    for sentence in sentences:
        sentence_lowercase = sentence.lower()
        if any(keyword in sentence_lowercase for keyword in keywords):
            matching_sentences.append(sentence.strip())

    return matching_sentences


def extract_action_items(transcript: str) -> List[str]:
    """Find sentences that sound like tasks or to-dos."""
    action_keywords = [
        "action", "next step", "next steps", "follow up", "todo", "to do",
        "should", "need to", "will", "must", "assigned", "responsible",
    ]
    return find_sentences_with_keywords(transcript, action_keywords)


def extract_key_decisions(transcript: str) -> List[str]:
    """Find sentences that sound like decisions the group made."""
    decision_keywords = [
        "decision", "decided", "agreed", "approved", "confirmed", "conclusion",
        "we will", "we'll", "let's", "plan", "go forward",
    ]
    return find_sentences_with_keywords(transcript, decision_keywords)


def extract_questions(transcript: str) -> List[str]:
    """Find every question in the transcript (any text ending in '?')."""
    question_matches = re.findall(r'[^.!?]+\?', transcript)
    questions = [q.strip() for q in question_matches if q.strip()]
    return questions