"""
parent_child_rag.py — Parent-Child RAG.

Architecture:
  Index: documents → 1024-token parent chunks → split into 256-token child chunks
         embed child chunks → FAISS index (child vectors)
  Retrieval: query → match child chunks → return PARENT chunks to LLM

Decisions:
  - 256 child / 1024 parent (D-005): 4× ratio, common production standard
  - Embed small, retrieve big: precision of small chunks + context of large parents
  - Each child chunk stores a parent_id pointer

Why it matters:
  Pure small-chunk retrieval has precise matching but LLM gets thin context.
  Pure large-chunk retrieval has rich context but poor embedding precision.
  Parent-Child gets both. Used in production LangChain + LlamaIndex deployments.
"""

import pickle
import time
import uuid
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    EMBEDDING_DIMS,
    CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE,
    TOP_K,
)
from groq_client import GroqClient
from local_client import LocalEmbedder
from rag_systems.base_rag import BaseRAG, Document
from rag_systems.chunker import split_text_into_chunks


class ParentChildRAG(BaseRAG):
    """
    Two-level chunking strategy:
    - CHILD chunks (256 tokens): used for embedding + retrieval
    - PARENT chunks (1024 tokens): passed to LLM for generation

    Retrieval returns parent chunks (not child chunks) to the LLM.
    """

    def __init__(self):
        super().__init__("Parent-Child RAG")
        self.client = GroqClient()
        self.embedder = LocalEmbedder()
        self.faiss_index = None
        self.child_chunks: list[dict] = []    # used for retrieval
        self.parent_chunks: dict[str, dict] = {}   # parent_id → parent chunk

    def index(self, documents: list[dict], cache_path: Path | None = None) -> None:
        print(f"[ParentChildRAG] Indexing {len(documents)} documents...")

        # 1. Create parent chunks (1024 tokens)
        # 2. For each parent, create child chunks (256 tokens) with parent_id pointer
        for doc in documents:
            parents = split_text_into_chunks(
                doc["content"],
                chunk_size=PARENT_CHUNK_SIZE,
                overlap=0,   # no overlap between parents — clean splits
            )
            for parent_text in parents:
                parent_id = str(uuid.uuid4())
                self.parent_chunks[parent_id] = {
                    "parent_id": parent_id,
                    "content": parent_text,
                    "source": doc["source"],
                    "metadata": doc.get("metadata", {}),
                }

                # Create child chunks from this parent
                children = split_text_into_chunks(
                    parent_text,
                    chunk_size=CHILD_CHUNK_SIZE,
                    overlap=20,
                )
                for i, child_text in enumerate(children):
                    self.child_chunks.append({
                        "chunk_id": f"{parent_id}_child_{i}",
                        "parent_id": parent_id,
                        "content": child_text,
                        "source": doc["source"],
                        "metadata": doc.get("metadata", {}),
                    })

        print(f"[ParentChildRAG] {len(self.parent_chunks)} parents, "
              f"{len(self.child_chunks)} children")

        # Embed child chunks
        embeddings = self._embed_texts([c["content"] for c in self.child_chunks])
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMS)
        self.faiss_index.add(vectors)

        self._is_indexed = True
        print(f"[ParentChildRAG] FAISS index built on {self.faiss_index.ntotal} child vectors")

        if cache_path:
            self._save(cache_path)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        """
        1. Embed query
        2. Find top-k CHILD chunks via FAISS
        3. Deduplicate by parent_id (multiple children may share same parent)
        4. Return top-k unique PARENT chunks (rich context for LLM)
        """
        q_embed = self._embed_texts([query])[0]
        q_vec = np.array([q_embed], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        # Retrieve more children than k to handle deduplication
        n_search = min(k * 4, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(q_vec, n_search)

        # Deduplicate: one parent per result (keep best child score per parent)
        seen_parents: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            child = self.child_chunks[idx]
            parent_id = child["parent_id"]
            if parent_id not in seen_parents or score > seen_parents[parent_id]:
                seen_parents[parent_id] = float(score)

        # Sort parents by best child score
        top_parents = sorted(seen_parents.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for parent_id, score in top_parents:
            parent = self.parent_chunks[parent_id]
            results.append(Document(
                content=parent["content"],    # NOTE: returns PARENT content to LLM
                source=parent["source"],
                chunk_id=parent_id,
                score=score,
                metadata={
                    **parent.get("metadata", {}),
                    "retrieved_via": "child_embedding",
                    "parent_size": PARENT_CHUNK_SIZE,
                    "child_size": CHILD_CHUNK_SIZE,
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
        with open(path / "child_chunks.pkl", "wb") as f:
            pickle.dump(self.child_chunks, f)
        with open(path / "parent_chunks.pkl", "wb") as f:
            pickle.dump(self.parent_chunks, f)

    def load(self, path: Path) -> None:
        self.faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "child_chunks.pkl", "rb") as f:
            self.child_chunks = pickle.load(f)
        with open(path / "parent_chunks.pkl", "rb") as f:
            self.parent_chunks = pickle.load(f)
        self._is_indexed = True
