"""
vector_rag.py — Basic Vector RAG using FAISS dense retrieval.

Architecture:
  documents → chunk → embeddings → FAISS flat index → retrieve → LLM

Decisions:
  - FAISS flat index for exact search (D-006: no approximation during benchmark)
  - text-embedding-3-small (D-003)
  - 512-token chunks / 50 overlap (D-004)
  - k=5 retrieval (D-006)

This is the BASELINE. All other architectures are compared against this.
"""

import pickle
import time
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    EMBEDDING_DIMS, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
)
from groq_client import GroqClient
from local_client import LocalEmbedder
from rag_systems.base_rag import BaseRAG, Document
from rag_systems.chunker import chunk_documents


class VectorRAG(BaseRAG):
    """
    Baseline: dense vector search only.
    Pipeline: embed query → cosine similarity against FAISS index → top-k chunks → LLM
    """

    def __init__(self):
        super().__init__("Vector RAG")
        self.client = GroqClient()
        self.embedder = LocalEmbedder()
        self.chunks: list[dict] = []   # stores chunk text + metadata
        self.index_path: Path | None = None

    def index(self, documents: list[dict], cache_path: Path | None = None) -> None:
        """
        Chunk documents, embed, and build FAISS flat index.
        If cache_path is provided, saves index to disk for reuse.
        """
        print(f"[VectorRAG] Indexing {len(documents)} documents...")
        self.chunks = chunk_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        print(f"[VectorRAG] {len(self.chunks)} chunks created")

        # Embed all chunks in batches of 100
        embeddings = self._embed_texts(
            [c["content"] for c in self.chunks],
            batch_size=100,
        )

        # Build FAISS flat L2 index (exact search — D-006)
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)   # cosine similarity via L2 on normalized vectors
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMS)  # inner product = cosine
        self.faiss_index.add(vectors)

        self._is_indexed = True
        print(f"[VectorRAG] FAISS index built: {self.faiss_index.ntotal} vectors")

        if cache_path:
            self._save(cache_path)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        """Embed query → search FAISS → return top-k Documents."""
        q_embedding = self._embed_texts([query])[0]
        q_vec = np.array([q_embedding], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        scores, indices = self.faiss_index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(Document(
                content=chunk["content"],
                source=chunk["source"],
                chunk_id=chunk["chunk_id"],
                score=float(score),
                metadata=chunk.get("metadata", {}),
            ))
        return results

    def generate(self, query: str, documents: list[Document]) -> tuple[str, dict]:
        """Generate answer from retrieved context using frozen LLM (D-001)."""
        prompt = self._build_prompt(query, documents)
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        usage = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "embedding": 0,   # tracked separately in _embed_texts
        }
        return answer, usage

    def _embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embed list of texts via Gemini text-embedding-004 in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self.embedder.embed(batch))
        return all_embeddings

    def _save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(path / "faiss.index"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[VectorRAG] Index saved to {path}")

    def load(self, path: Path) -> None:
        self.faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self._is_indexed = True
        print(f"[VectorRAG] Index loaded from {path} ({self.faiss_index.ntotal} vectors)")
