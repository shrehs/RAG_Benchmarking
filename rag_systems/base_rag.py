"""
base_rag.py — Abstract interface every RAG system must implement.

Decision: All 5 systems share this interface so run_benchmark.py
can call them identically. See D-006 in decisions_log.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import time
import psutil
import os


@dataclass
class Document:
    """A retrieved document chunk with metadata."""
    content: str
    source: str
    chunk_id: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Output of a single retrieve() call with full timing + cost info."""
    query: str
    documents: list[Document]
    answer: str
    latency_s: float
    token_usage: dict      # {"prompt": int, "completion": int, "embedding": int}
    cost_usd: dict         # {"embedding": float, "generation": float}
    architecture: str
    metadata: dict = field(default_factory=dict)


class BaseRAG(ABC):
    """
    All RAG systems implement this interface.
    This ensures run_benchmark.py can call every system identically.
    See decisions_log.md D-006.
    """

    def __init__(self, name: str):
        self.name = name
        self._is_indexed = False

    @abstractmethod
    def index(self, documents: list[dict]) -> None:
        """
        Ingest and index a list of documents.
        Each doc: {"content": str, "source": str, "metadata": dict}
        Must be called once before retrieve().
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """
        Retrieve top-k relevant document chunks for query.
        Returns list of Document objects sorted by relevance (desc).
        """
        pass

    @abstractmethod
    def generate(self, query: str, documents: list[Document]) -> tuple[str, dict]:
        """
        Generate an answer given query + retrieved documents.
        Returns: (answer_text, token_usage_dict)
        """
        pass

    def query(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Full RAG pipeline: retrieve + generate + timing + cost tracking.
        This is what run_benchmark.py calls.
        """
        if not self._is_indexed:
            raise RuntimeError(f"{self.name}: call index() before query()")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        t0 = time.perf_counter()
        docs = self.retrieve(query, k)
        retrieve_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        answer, token_usage = self.generate(query, docs)
        generate_time = time.perf_counter() - t1

        total_latency = retrieve_time + generate_time
        mem_after = process.memory_info().rss / 1024 / 1024

        cost = self._calculate_cost(token_usage)

        return RetrievalResult(
            query=query,
            documents=docs,
            answer=answer,
            latency_s=total_latency,
            token_usage=token_usage,
            cost_usd=cost,
            architecture=self.name,
            metadata={
                "retrieve_time_s": retrieve_time,
                "generate_time_s": generate_time,
                "mem_delta_mb": mem_after - mem_before,
                "num_docs_retrieved": len(docs),
            }
        )

    def _calculate_cost(self, token_usage: dict) -> dict:
        """Calculate USD cost from token usage. D-012."""
        from config import (
            COST_PER_1K_EMBED_TOKENS,
            COST_PER_1K_INPUT_TOKENS,
            COST_PER_1K_OUTPUT_TOKENS,
        )
        embed_tokens = token_usage.get("embedding", 0)
        prompt_tokens = token_usage.get("prompt", 0)
        completion_tokens = token_usage.get("completion", 0)

        embed_cost = (embed_tokens / 1000) * COST_PER_1K_EMBED_TOKENS
        gen_cost = (
            (prompt_tokens / 1000) * COST_PER_1K_INPUT_TOKENS
            + (completion_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
        )
        return {
            "embedding": embed_cost,
            "generation": gen_cost,
            "total": embed_cost + gen_cost,
        }

    def _build_prompt(self, query: str, documents: list[Document]) -> str:
        """Standard RAG prompt. Shared across all architectures for fair comparison."""
        context = "\n\n---\n\n".join([
            f"[Source: {doc.source}]\n{doc.content}"
            for doc in documents
        ])
        return f"""Answer the question based only on the provided context. 
If the context doesn't contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""
