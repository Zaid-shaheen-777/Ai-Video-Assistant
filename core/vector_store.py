"""
vector_store.py
----------------
This file handles turning the transcript into a "vector store" —
a searchable database of text chunks, used for the Q&A (RAG) feature.

Beginner note: A vector store lets us search a transcript by MEANING,
not just exact words. It works like this:
1. Split the transcript into small chunks of text
2. Convert each chunk into a list of numbers ("embedding") that
   represents its meaning
3. Store those in a database (Chroma)
4. Later, we can search: "which chunks are most similar in meaning
   to this question?"
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Where the vector database is saved on disk
CHROMA_DIR = "vector_db"

# The "collection" is just the name of this specific dataset inside Chroma
COLLECTION_NAME = "meeting_transcript"

# This is a small, fast, free model that converts text into embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embeddings():
    """Create the embedding model (turns text into numbers)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str) -> Chroma:
    """
    Take a transcript, split it into chunks, and store it in a
    new vector database so we can search it later.
    """
    print("Building vector store from transcript...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    text_chunks = splitter.split_text(transcript)

    # Wrap each chunk of text in a "Document" object (LangChain's format)
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
    """
    Create a 'retriever' from the vector store.
    A retriever's job: given a question, find the `k` most relevant
    chunks of the transcript.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )