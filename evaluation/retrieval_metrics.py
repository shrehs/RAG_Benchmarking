"""
evaluation/retrieval_metrics.py — Computes Recall@k, Precision@k, MRR.

Decision D-011: k=5 standard across all experiments.

These are computed against ground-truth relevant document IDs,
which are stored in QA pairs as "relevant_chunk_ids".
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K
from rag_systems.base_rag import Document


def recall_at_k(retrieved: list[Document], relevant_ids: set[str], k: int = TOP_K) -> float:
    """
    Recall@k = (# relevant docs in top-k retrieved) / (# total relevant docs)
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc.chunk_id in relevant_ids or doc.source in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved: list[Document], relevant_ids: set[str], k: int = TOP_K) -> float:
    """
    Precision@k = (# relevant docs in top-k retrieved) / k
    """
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc.chunk_id in relevant_ids or doc.source in relevant_ids)
    return hits / min(k, len(top_k))


def mean_reciprocal_rank(retrieved: list[Document], relevant_ids: set[str]) -> float:
    """
    MRR = 1 / (rank of first relevant document)
    If no relevant doc found in retrieved list, returns 0.
    """
    for rank, doc in enumerate(retrieved, start=1):
        if doc.chunk_id in relevant_ids or doc.source in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    rag_system,
    qa_pairs_with_relevance: list[dict],
    k: int = TOP_K,
) -> dict:
    """
    Compute Recall@k, Precision@k, MRR across all QA pairs.
    
    qa_pairs_with_relevance: list of {
        "question": str,
        "relevant_sources": list[str]   # source strings of relevant docs
    }
    
    Returns dict with mean metrics across all pairs.
    """
    recalls, precisions, mrrs = [], [], []

    for pair in qa_pairs_with_relevance:
        question = pair["question"]
        relevant = set(pair.get("relevant_sources", []))

        docs = rag_system.retrieve(question, k=k)

        recalls.append(recall_at_k(docs, relevant, k))
        precisions.append(precision_at_k(docs, relevant, k))
        mrrs.append(mean_reciprocal_rank(docs, relevant))

    n = len(recalls)
    return {
        f"recall_at_{k}": sum(recalls) / n if n > 0 else 0.0,
        f"precision_at_{k}": sum(precisions) / n if n > 0 else 0.0,
        "mrr": sum(mrrs) / n if n > 0 else 0.0,
        "num_queries": n,
    }
