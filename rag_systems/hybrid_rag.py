"""
hybrid_rag.py — Hybrid RAG: FAISS dense retrieval + BM25 keyword search, linearly fused.

Architecture:
  query → [FAISS dense search] + [BM25 keyword search]
       → normalize scores → linear fusion → rerank → top-k → LLM

Decisions:
  - alpha=0.5 equal weighting (D-007): neutral benchmark starting point
  - Min-max normalization before fusion (D-007): prevents BM25 score dominance
  - Same FAISS index as VectorRAG for fair comparison

Why it matters:
  Vector search misses exact keyword queries. BM25 misses semantic similarity.
  Hybrid captures both. Expected to outperform on technical jargon (arXiv, k8s docs).
"""

import pickle
import time
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    EMBEDDING_DIMS, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, HYBRID_ALPHA,
)
from gemini_client import GeminiClient
from local_client import LocalEmbedder
from rag_systems.base_rag import BaseRAG, Document
from rag_systems.chunker import chunk_documents


class HybridRAG(BaseRAG):
    """
    Hybrid search: vector similarity (FAISS) + keyword search (BM25).
    
    Fusion formula (D-007):
        final_score = alpha * vector_score_normalized + (1 - alpha) * bm25_score_normalized
    
    Both scores are min-max normalized to [0, 1] before fusion.
    """

    def __init__(self, alpha: float = HYBRID_ALPHA):
        super().__init__("Hybrid RAG")
        self.alpha = alpha   # D-007: 0.5 = equal weight
        self.client = GeminiClient()
        self.embedder = LocalEmbedder()
        self.faiss_index = None
        self.bm25 = None
        self.chunks: list[dict] = []

    def index(self, documents: list[dict], cache_path: Path | None = None) -> None:
        print(f"[HybridRAG] Indexing {len(documents)} documents (alpha={self.alpha})...")
        self.chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"[HybridRAG] {len(self.chunks)} chunks created")

        # 1. Build FAISS index (same as VectorRAG — fair comparison)
        embeddings = self._embed_texts([c["content"] for c in self.chunks])
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMS)
        self.faiss_index.add(vectors)

        # 2. Build BM25 index over tokenized chunks
        tokenized = [c["content"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        self._is_indexed = True
        print(f"[HybridRAG] FAISS + BM25 index built: {self.faiss_index.ntotal} vectors")

        if cache_path:
            self._save(cache_path)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        """
        1. Get vector scores from FAISS (full index)
        2. Get BM25 scores for all chunks
        3. Normalize both to [0, 1]
        4. Fuse: alpha * vector + (1 - alpha) * bm25
        5. Return top-k by fused score
        """
        n = len(self.chunks)

        # Vector scores for all chunks
        q_embed = self._embed_texts([query])[0]
        q_vec = np.array([q_embed], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        vector_scores, vector_indices = self.faiss_index.search(q_vec, n)
        vector_scores = vector_scores[0]   # shape: (n,)
        vector_indices = vector_indices[0]

        # Reorder vector_scores to align with chunk order
        aligned_vector = np.zeros(n)
        for score, idx in zip(vector_scores, vector_indices):
            if idx >= 0:
                aligned_vector[idx] = score

        # BM25 scores (already aligned with chunk order)
        bm25_scores = np.array(
            self.bm25.get_scores(query.lower().split()),
            dtype=np.float32,
        )

        # Normalize both to [0, 1] (D-007)
        def minmax(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-9)

        vec_norm = minmax(aligned_vector)
        bm25_norm = minmax(bm25_scores)

        # Fuse
        fused = self.alpha * vec_norm + (1 - self.alpha) * bm25_norm

        # Top-k
        top_indices = np.argsort(fused)[::-1][:k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(Document(
                content=chunk["content"],
                source=chunk["source"],
                chunk_id=chunk["chunk_id"],
                score=float(fused[idx]),
                metadata={
                    **chunk.get("metadata", {}),
                    "vector_score": float(aligned_vector[idx]),
                    "bm25_score": float(bm25_scores[idx]),
                    "alpha": self.alpha,
                },
            ))
        return results

    def generate(self, query: str, documents: list[Document]) -> tuple[str, dict]:
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
            "embedding": 0,
        }
        return answer, usage

    def _embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self.embedder.embed(batch))
        return all_embeddings

    def _save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(path / "faiss.index"))
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: Path) -> None:
        self.faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self._is_indexed = True
