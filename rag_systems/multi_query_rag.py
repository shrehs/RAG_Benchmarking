"""
multi_query_rag.py — Multi-Query RAG: LLM generates sub-queries to improve recall.

Architecture:
  query → LLM generates 3 reformulations → retrieve for each → deduplicate → LLM

Decisions:
  - 3 sub-queries (D-008): cost/recall optimum; diminishing returns after 3
  - Deduplication by chunk_id before generation (D-008): no repeated context
  - Uses VectorRAG retrieval under the hood for fair comparison
  - Cost note (D-008, LIM-005): 3× more LLM tokens for sub-query generation

Why it matters:
  Single queries often miss relevant docs due to vocabulary mismatch.
  "How does Kubernetes handle node failure?" → also searches:
    "What happens when a k8s node goes down?"
    "Kubernetes node recovery mechanisms"
    "Pod rescheduling after node failure"
  Expands recall by covering multiple phrasings.
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
    EMBEDDING_DIMS, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, NUM_SUBQUERIES,
)
from groq_client import GroqClient
from local_client import LocalEmbedder
from rag_systems.base_rag import BaseRAG, Document
from rag_systems.chunker import chunk_documents


SUBQUERY_SYSTEM_PROMPT = """You are an expert at reformulating search queries.
Given a question, generate {n} different versions of it that capture different aspects 
or phrasings. These will be used to retrieve relevant documents from a knowledge base.

Rules:
- Each reformulation should be meaningfully different (different vocabulary, angle, or specificity)
- Keep reformulations concise (1 sentence each)
- Output ONLY the reformulated queries, one per line, no numbering or bullets
- Do not include the original query"""


class MultiQueryRAG(BaseRAG):
    """
    Multi-Query RAG: generates NUM_SUBQUERIES reformulations via LLM,
    retrieves for each, deduplicates, and generates one final answer.
    
    Cost implication (D-008, LIM-005):
    - 1 LLM call to generate sub-queries (~100-200 tokens)
    - NUM_SUBQUERIES FAISS searches (fast, negligible)
    - 1 LLM call to generate answer
    Cost is tracked and reported in results.
    """

    def __init__(self, num_subqueries: int = NUM_SUBQUERIES):
        super().__init__("Multi-Query RAG")
        self.num_subqueries = num_subqueries
        self.client = GroqClient()
        self.embedder = LocalEmbedder()
        self.faiss_index = None
        self.chunks: list[dict] = []
        self._subquery_token_usage: dict = {}   # tracked separately for cost

    def index(self, documents: list[dict], cache_path: Path | None = None) -> None:
        print(f"[MultiQueryRAG] Indexing {len(documents)} documents...")
        self.chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"[MultiQueryRAG] {len(self.chunks)} chunks")

        embeddings = self._embed_texts([c["content"] for c in self.chunks])
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMS)
        self.faiss_index.add(vectors)

        self._is_indexed = True
        print(f"[MultiQueryRAG] FAISS index built: {self.faiss_index.ntotal} vectors")

        if cache_path:
            self._save(cache_path)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        """
        1. Generate NUM_SUBQUERIES reformulations of query
        2. Retrieve k chunks for each sub-query
        3. Deduplicate by chunk_id (keep highest score per chunk)
        4. Return top-k unique chunks
        """
        sub_queries = self._generate_subqueries(query)
        all_queries = [query] + sub_queries   # include original

        # Retrieve for each query
        chunk_best_scores: dict[str, float] = {}
        chunk_objects: dict[str, Document] = {}

        for q in all_queries:
            q_embed = self._embed_texts([q])[0]
            q_vec = np.array([q_embed], dtype=np.float32)
            faiss.normalize_L2(q_vec)
            scores, indices = self.faiss_index.search(q_vec, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                chunk = self.chunks[idx]
                chunk_id = chunk["chunk_id"]
                if chunk_id not in chunk_best_scores or score > chunk_best_scores[chunk_id]:
                    chunk_best_scores[chunk_id] = float(score)
                    chunk_objects[chunk_id] = Document(
                        content=chunk["content"],
                        source=chunk["source"],
                        chunk_id=chunk_id,
                        score=float(score),
                        metadata={
                            **chunk.get("metadata", {}),
                            "sub_queries": sub_queries,
                            "num_subqueries": self.num_subqueries,
                        },
                    )

        # Return top-k by best score across all sub-queries
        top_ids = sorted(chunk_best_scores, key=chunk_best_scores.get, reverse=True)[:k]
        return [chunk_objects[cid] for cid in top_ids]

    def generate(self, query: str, documents: list[Document]) -> tuple[str, dict]:
        prompt = self._build_prompt(query, documents)
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content

        # Include sub-query generation tokens in usage (D-012 cost tracking)
        usage = {
            "prompt": response.usage.prompt_tokens + self._subquery_token_usage.get("prompt", 0),
            "completion": response.usage.completion_tokens + self._subquery_token_usage.get("completion", 0),
            "embedding": 0,
            "subquery_gen_tokens": self._subquery_token_usage,
        }
        return answer, usage

    def _generate_subqueries(self, query: str) -> list[str]:
        """
        Use LLM to generate NUM_SUBQUERIES reformulations of the query.
        Token usage tracked for D-012 cost accounting.
        """
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.7,   # slight temperature for diversity in reformulations
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": SUBQUERY_SYSTEM_PROMPT.format(n=self.num_subqueries),
                },
                {"role": "user", "content": query},
            ],
        )
        # Track for cost reporting
        self._subquery_token_usage = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
        }

        raw = response.choices[0].message.content.strip()
        sub_queries = [q.strip() for q in raw.split("\n") if q.strip()]
        return sub_queries[:self.num_subqueries]

    def _embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
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

    def load(self, path: Path) -> None:
        self.faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self._is_indexed = True
