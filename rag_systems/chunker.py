"""
chunker.py — Text chunking utilities shared across all RAG systems.

Decisions: D-004 (512/50), D-005 (256/1024 parent-child)
"""

import uuid
import re
from typing import Optional


def split_text_into_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks by approximate token count.
    Uses word-level splitting (1 word ≈ 1.3 tokens — rough approximation).
    For production, replace with tiktoken for exact token counting.
    """
    # Approximate: convert chunk_size from tokens to words
    words_per_chunk = int(chunk_size / 1.3)
    overlap_words = int(overlap / 1.3)

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 20:   # skip tiny chunks
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap_words

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Chunk a list of documents into overlapping text chunks.
    
    Input doc format:
        {"content": str, "source": str, "metadata": dict}
    
    Output chunk format:
        {"chunk_id": str, "content": str, "source": str, "metadata": dict, "doc_index": int}
    """
    all_chunks = []
    for doc_idx, doc in enumerate(documents):
        text = doc.get("content", "").strip()
        if not text:
            continue

        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_idx}_{i}_{str(uuid.uuid4())[:8]}",
                "content": chunk_text,
                "source": doc.get("source", f"doc_{doc_idx}"),
                "doc_index": doc_idx,
                "chunk_index": i,
                "metadata": doc.get("metadata", {}),
            })
    return all_chunks
