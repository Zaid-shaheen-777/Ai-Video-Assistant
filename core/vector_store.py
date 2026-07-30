"""
vector_store.py
----------------
Turns the transcript into a searchable vector store for Q&A (RAG).

Uses Mistral embeddings over the API so Streamlit Cloud does not need
torch / sentence-transformers.
"""

import os

from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL_NAME = "mistral-embed"


def get_embeddings():
    """Create the embedding model (turns text into numbers via Mistral API)."""
    return MistralAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def build_vector_store(transcript: str) -> Chroma:
    """Split transcript into chunks and store them for similarity search."""
    print("Building vector store from transcript...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    text_chunks = splitter.split_text(transcript)

    documents = []
    for index, chunk in enumerate(text_chunks):
        documents.append(Document(page_content=chunk, metadata={"chunk_index": index}))

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def load_vector_store() -> Chroma:
    """Load a previously saved vector store from disk."""
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """Return a retriever that finds the k most relevant transcript chunks."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
