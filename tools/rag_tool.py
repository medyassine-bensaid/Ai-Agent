"""
rag_tool.py — Lightweight RAG pipeline for DevOps documentation.

Stack:
- ChromaDB (local persistent store, no server needed)
- sentence-transformers all-MiniLM-L6-v2 (fast, free, local)
- Markdown/text ingestion with metadata + source attribution
- Exposed as a LangChain @tool for the agent
- Langfuse span instrumentation for retrieval observability
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.tools import tool
from langfuse import get_client, observe
from sentence_transformers import SentenceTransformer

from utils.logger import logger

# ── Config ──────────────────────────────────────────────────────────────────
CHROMA_PATH = os.getenv(
    "CHROMA_PATH", str(Path(__file__).resolve().parent.parent / "data" / "chroma")
)
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
COLLECTION = "devops_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400  # chars
CHUNK_OVERLAP = 80
TOP_K = 4
# Chunks below this cosine similarity are dropped so off-topic questions do not
# get unrelated context injected into the answer prompt
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))

langfuse = get_client()

# ── Lazy singletons ─────────────────────────────────────────────────────────
_embedder = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Chunking ─────────────────────────────────────────────────────────────────
def _chunk_text(text: str, source: str) -> List[Dict]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            uid = hashlib.md5(f"{source}:{start}".encode()).hexdigest()
            chunks.append(
                {
                    "id": uid,
                    "text": chunk_text,
                    "metadata": {"source": source, "start": start},
                }
            )
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Ingestion ────────────────────────────────────────────────────────────────
def ingest_text(text: str, source: str = "manual") -> int:
    """Ingest raw text into ChromaDB. Returns number of chunks added."""
    col = _get_collection()
    embedder = _get_embedder()
    chunks = _chunk_text(text, source)
    if not chunks:
        return 0

    existing = col.get(ids=[c["id"] for c in chunks], include=["documents"])
    stored = dict(zip(existing["ids"], existing["documents"]))
    # New chunks, plus chunks whose text changed since the last ingestion
    new = [c for c in chunks if stored.get(c["id"]) != c["text"]]
    if not new:
        return 0

    col.upsert(
        ids=[c["id"] for c in new],
        documents=[c["text"] for c in new],
        embeddings=embedder.encode([c["text"] for c in new]).tolist(),
        metadatas=[c["metadata"] for c in new],
    )
    logger.log_action(f"Ingested {len(new)} chunks from '{source}'", "rag_ingest")
    return len(new)


def ingest_file(path: str) -> int:
    """Ingest a markdown/text file."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return ingest_text(text, source=path)


def ingest_builtin_kb():
    """Seed ChromaDB from the existing DEVOPS_KNOWLEDGE dict (one-time)."""
    from tools.knowledge_base import DEVOPS_KNOWLEDGE

    total = 0
    for topic, content in DEVOPS_KNOWLEDGE.items():
        total += ingest_text(content, source=f"kb:{topic}")
    if total:
        logger.log_action(
            f"Seeded RAG with {total} chunks from built-in KB", "rag_seed"
        )


# ── Retrieval ────────────────────────────────────────────────────────────────
@observe(name="rag-retrieval", as_type="retriever")
def _retrieve(query: str) -> List[Dict]:
    col = _get_collection()
    embedder = _get_embedder()

    q_emb = embedder.encode([query]).tolist()
    results = col.query(
        query_embeddings=q_emb,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1 - dist, 3)  # cosine distance -> similarity
        if score < MIN_SCORE:
            continue
        docs.append(
            {
                "text": doc,
                "source": meta.get("source", "?"),
                "chunk_id": meta.get("start"),
                "score": score,
            }
        )

    # Attach retrieval metadata to the current @observe span (SDK v4 has no .trace())
    try:
        langfuse.update_current_span(
            metadata={
                "query": query,
                "num_docs": len(docs),
                "min_score": MIN_SCORE,
                "top_score": docs[0]["score"] if docs else 0,
            },
        )
    except Exception as e:
        logger.log_error(f"Langfuse span update failed: {e}")
    return docs


# ── LangChain Tool ───────────────────────────────────────────────────────────
@tool
def rag_tool(query: str) -> str:
    """
    Search the DevOps documentation knowledge base using semantic retrieval (RAG).
    Use for established DevOps concepts, architecture patterns, and best practices.

    Args:
        query: Natural language question or topic
    Returns:
        Relevant documentation excerpts with source attribution
    """
    logger.log_action(f"RAG retrieval: '{query}'", "rag_tool")

    # Seed on first use if collection is empty
    col = _get_collection()
    if col.count() == 0:
        ingest_builtin_kb()

    docs = _retrieve(query)
    if not docs:
        return (
            f"No relevant documentation found in the knowledge base for '{query}' "
            f"(no chunk scored >= {MIN_SCORE})."
        )

    output = f"Retrieved {len(docs)} relevant chunks:\n\n"
    for i, d in enumerate(docs, 1):
        output += f"[{i}] (score={d['score']}, source={d['source']})\n{d['text']}\n\n"
    return output
