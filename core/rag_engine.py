"""
rag_engine.py
-------------
This file lets you "chat" with your meeting transcript by asking
questions about it.

This technique is called RAG (Retrieval-Augmented Generation):
1. RETRIEVE: find the most relevant parts of the transcript for
   the question being asked
2. AUGMENT: stick those relevant parts into the AI's prompt as context
3. GENERATE: ask the AI to answer the question using only that context

Beginner note: This is basically "open-book exam" mode for the AI —
instead of guessing from memory, it looks up the exact transcript
parts related to your question before answering.
"""

import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.vector_store import build_vector_store, load_vector_store, get_retriever

SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""


def get_llm():
    """Create a connection to the Mistral AI chat model."""
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs) -> str:
    """Combine a list of retrieved chunks into one text block."""
    return "\n\n".join(doc.page_content for doc in docs)


class RagChain:
    """
    A simple wrapper that stores everything we need to answer
    questions about a transcript: the retriever and the AI model.
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

    def answer(self, question: str) -> str:
        # Step 1: retrieve the most relevant transcript chunks
        relevant_docs = self.retriever.invoke(question)
        context_text = format_docs(relevant_docs)

        # Step 2: build the final prompt with context + question
        prompt_value = self.prompt.invoke({
            "context": context_text,
            "question": question,
        })

        # Step 3: ask the AI to answer
        response = self.llm.invoke(prompt_value)
        answer_text = StrOutputParser().invoke(response)

        return answer_text


def build_rag_chain(transcript: str) -> RagChain:
    """Build a brand-new RAG chain from a fresh transcript."""
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    return RagChain(retriever, llm)


def load_rag_chain() -> RagChain:
    """Load a RAG chain from a previously saved vector store."""
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    return RagChain(retriever, llm)


def ask_question(rag_chain: RagChain, question: str) -> str:
    """Ask a question and get an answer back from the RAG chain."""
    print(f"Question: {question}")
    answer = rag_chain.answer(question)
    print(f"Answer: {answer}")
    return answer