"""
evaluation/embedding_benchmark.py — Benchmark embedding models across quality, speed, and cost.

This is the "Elite Upgrade" module. It runs a controlled experiment:
  SAME documents + SAME queries + SAME RAG architecture (VectorRAG)
  VARY: embedding model × device (CPU / GPU)

Produces:
  - Recall@5 per model
  - Embedding throughput (docs/sec) per model × device
  - Cost per 1M tokens
  - Cost per unit of Recall (efficiency frontier)
  - Storage per model (larger dims = larger indexes)

Decision D-013: Three model tiers compared (budget / standard / premium / local)
Decision D-014: GPU vs CPU measured on every HuggingFace model
Decision D-015: Cost-adjusted recall score = primary comparison metric

Usage:
    python evaluation/embedding_benchmark.py --dataset small
    python evaluation/embedding_benchmark.py --dataset medium --models bge-large text-3-large
"""

import argparse
import json
import time
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K, BENCHMARK_WARMUP_RUNS, BENCHMARK_TIMED_RUNS, RESULTS_DIR
from embedding_registry import (
    EMBEDDING_MODELS, get_embedder, detect_device, EmbeddingModel,
)
from scripts.fetch_datasets import load_dataset
from rag_systems.chunker import chunk_documents
from evaluation.log_results import save_result

import faiss


@dataclass
class EmbeddingBenchmarkResult:
    model_name: str
    display_name: str
    device: str
    dims: int
    provider: str
    cost_per_1m_tokens: float

    # Quality
    recall_at_5: float
    precision_at_5: float
    mrr: float

    # Speed
    embed_throughput_docs_per_sec: float   # during indexing
    query_latency_ms: float                # single query embed + search
    query_p95_ms: float

    # Cost
    indexing_cost_usd: float
    cost_per_query_usd: float
    total_tokens_embedded: int

    # Storage
    index_size_mb: float

    # Efficiency
    recall_per_dollar: float               # D-015: primary cross-model metric
    recall_per_ms: float                   # quality / latency trade-off

    # Device info
    device_info: dict
    dataset: str
    timestamp: str


def build_index(
    chunks: list[dict],
    model_name: str,
    device: str,
) -> tuple[faiss.Index, "BaseEmbedder", float]:
    """
    Embed all chunks with given model+device, build FAISS flat index.
    Returns (index, embedder, throughput_docs_per_sec).
    """
    embedder = get_embedder(model_name, device=device)
    texts = [c["content"] for c in chunks]

    t0 = time.perf_counter()
    vectors = embedder.embed(texts)
    elapsed = time.perf_counter() - t0

    throughput = len(texts) / elapsed if elapsed > 0 else 0

    # Normalize + build FAISS flat inner-product index
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, embedder, throughput


def measure_query_latency(
    query_texts: list[str],
    embedder,
    index: faiss.Index,
    k: int = TOP_K,
    warmup: int = BENCHMARK_WARMUP_RUNS,
    timed: int = BENCHMARK_TIMED_RUNS,
) -> tuple[float, float]:
    """
    Measure per-query latency: embed query + FAISS search.
    Returns (p50_ms, p95_ms).
    """
    def run_query(q):
        vec = embedder.embed([q])
        faiss.normalize_L2(vec)
        index.search(vec, k)

    # Warmup
    for i in range(warmup):
        run_query(query_texts[i % len(query_texts)])

    # Timed
    latencies = []
    for i in range(timed):
        q = query_texts[(warmup + i) % len(query_texts)]
        t0 = time.perf_counter()
        run_query(q)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(0.95 * len(latencies))]
    return p50, p95


def compute_recall(
    query_texts: list[str],
    relevant_sources: list[list[str]],
    embedder,
    index: faiss.Index,
    chunks: list[dict],
    k: int = TOP_K,
) -> tuple[float, float, float]:
    """Compute Recall@k, Precision@k, MRR against ground-truth sources."""
    recalls, precisions, mrrs = [], [], []

    for query, rel_sources in zip(query_texts, relevant_sources):
        rel_set = set(rel_sources)
        vec = embedder.embed([query])
        faiss.normalize_L2(vec)
        scores, indices = index.search(vec, k)

        hits = [
            1 if (chunks[idx]["source"] in rel_set or
                  any(s in chunks[idx]["source"] for s in rel_set))
            else 0
            for idx in indices[0] if idx >= 0
        ]

        recall = sum(hits) / len(rel_set) if rel_set else 0
        precision = sum(hits) / k if k > 0 else 0
        mrr = 0
        for rank, hit in enumerate(hits, 1):
            if hit:
                mrr = 1.0 / rank
                break

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)

    n = len(recalls)
    return (
        sum(recalls) / n if n else 0,
        sum(precisions) / n if n else 0,
        sum(mrrs) / n if n else 0,
    )


def run_embedding_benchmark(
    dataset_name: str,
    model_names: list[str] | None = None,
    devices: list[str] | None = None,
) -> list[EmbeddingBenchmarkResult]:
    """
    Main benchmark loop: for each (model × device) combo:
      1. Build FAISS index
      2. Measure throughput
      3. Measure query latency
      4. Compute Recall@k
      5. Compute cost metrics
      6. Compute efficiency frontier (recall / dollar, recall / ms)
    """
    documents = load_dataset(dataset_name)
    chunks = chunk_documents(documents, chunk_size=512, chunk_overlap=50)

    # Load or generate QA pairs
    qa_cache = RESULTS_DIR / "benchmark_tables" / f"qa_{dataset_name}.json"
    if qa_cache.exists():
        with open(qa_cache) as f:
            qa_pairs = json.load(f)
    else:
        from evaluation.ragas_eval import generate_qa_pairs
        qa_pairs = generate_qa_pairs(documents, dataset_name=dataset_name)
        qa_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(qa_cache, "w") as f:
            json.dump(qa_pairs, f, indent=2)

    queries = [p["question"] for p in qa_pairs]
    relevant_sources = [
        [p.get("source", "")] for p in qa_pairs
    ]

    device_info = detect_device()
    models_to_run = model_names or list(EMBEDDING_MODELS.keys())

    # Determine which device configs to test per model
    if devices:
        device_configs = devices
    else:
        device_configs = ["cpu"]
        if device_info["cuda"]:
            device_configs.append("cuda")
        elif device_info["mps"]:
            device_configs.append("mps")

    results = []

    for model_name in models_to_run:
        model_meta = EMBEDDING_MODELS[model_name]

        # OpenAI models only run "cpu" (API call)
        model_devices = ["cpu"] if model_meta.provider == "openai" else device_configs

        for device in model_devices:
            print(f"\n{'─'*55}")
            print(f"  Model: {model_meta.display_name}")
            print(f"  Device: {device.upper()}")
            print(f"  Dataset: {dataset_name} ({len(chunks)} chunks)")
            print(f"{'─'*55}")

            try:
                # 1. Build index
                t_idx_start = time.perf_counter()
                index, embedder, throughput = build_index(chunks, model_name, device)
                idx_time = time.perf_counter() - t_idx_start

                # 2. Index size (approximate: n_vectors × dims × 4 bytes)
                index_size_mb = (index.ntotal * model_meta.dims * 4) / (1024 ** 2)

                # 3. Query latency
                p50_ms, p95_ms = measure_query_latency(queries, embedder, index)

                # 4. Recall metrics
                recall, precision, mrr = compute_recall(
                    queries, relevant_sources, embedder, index, chunks
                )

                # 5. Cost calculation (D-012, D-013)
                total_tokens = embedder.tokens_used
                indexing_cost = embedder.estimated_cost(total_tokens)

                # Per-query cost: embed one query
                q_embed_start = time.perf_counter()
                q_embedder = get_embedder(model_name, device=device)
                q_embedder.embed([queries[0]])
                query_tokens = q_embedder.tokens_used
                cost_per_query = q_embedder.estimated_cost(query_tokens)

                # 6. Efficiency metrics (D-015)
                # recall_per_dollar: how much recall do you get per $1 of indexing cost?
                # For free models, use a normalized time-cost proxy
                if indexing_cost > 0:
                    recall_per_dollar = recall / indexing_cost if indexing_cost > 0 else float("inf")
                else:
                    # Free model: use time as proxy cost ($0.001 per second of GPU/CPU time)
                    time_cost = idx_time * 0.001
                    recall_per_dollar = recall / time_cost if time_cost > 0 else float("inf")

                recall_per_ms = recall / p50_ms if p50_ms > 0 else 0

                result = EmbeddingBenchmarkResult(
                    model_name=model_name,
                    display_name=model_meta.display_name,
                    device=device,
                    dims=model_meta.dims,
                    provider=model_meta.provider,
                    cost_per_1m_tokens=model_meta.cost_per_1m_tokens,
                    recall_at_5=recall,
                    precision_at_5=precision,
                    mrr=mrr,
                    embed_throughput_docs_per_sec=throughput,
                    query_latency_ms=p50_ms,
                    query_p95_ms=p95_ms,
                    indexing_cost_usd=indexing_cost,
                    cost_per_query_usd=cost_per_query,
                    total_tokens_embedded=total_tokens,
                    index_size_mb=index_size_mb,
                    recall_per_dollar=recall_per_dollar,
                    recall_per_ms=recall_per_ms,
                    device_info=device_info,
                    dataset=dataset_name,
                    timestamp=datetime.utcnow().isoformat(),
                )
                results.append(result)

                print(f"  Recall@5:      {recall:.4f}")
                print(f"  P50 latency:   {p50_ms:.1f}ms")
                print(f"  Throughput:    {throughput:.0f} docs/sec")
                print(f"  Index cost:    ${indexing_cost:.6f}")
                print(f"  Per-query:     ${cost_per_query:.8f}")
                print(f"  Index size:    {index_size_mb:.1f}MB")
                print(f"  Recall/$:      {recall_per_dollar:.2f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    return results


def save_embedding_results(results: list[EmbeddingBenchmarkResult], dataset: str) -> Path:
    """Save results to JSON and append summary to decisions_log.md"""
    out_dir = RESULTS_DIR / "benchmark_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"embedding_benchmark_{dataset}_{timestamp}.json"

    serialized = [asdict(r) for r in results]
    with open(out_path, "w") as f:
        json.dump(serialized, f, indent=2)

    # Append summary to decisions_log.md
    _append_embedding_summary_to_log(results, dataset)

    print(f"\n[embedding_benchmark] Results saved to {out_path}")
    return out_path


def print_summary_table(results: list[EmbeddingBenchmarkResult]) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"\n{'Model':<32} {'Device':<8} {'Dims':<6} {'Recall@5':<10} {'P50(ms)':<9} {'$/1M tok':<10} {'Recall/$':<10} {'Size(MB)':<9}"
    print(header)
    print("─" * len(header))

    for r in sorted(results, key=lambda x: x.recall_at_5, reverse=True):
        cost_str = f"{r.cost_per_1m_tokens:.3f}" if r.cost_per_1m_tokens > 0 else "FREE"
        r_per_d = f"{r.recall_per_dollar:.1f}" if r.recall_per_dollar < 1e6 else "∞ (free)"
        print(
            f"{r.display_name[:31]:<32} "
            f"{r.device:<8} "
            f"{r.dims:<6} "
            f"{r.recall_at_5:<10.4f} "
            f"{r.query_latency_ms:<9.1f} "
            f"{cost_str:<10} "
            f"{r_per_d:<10} "
            f"{r.index_size_mb:<9.1f}"
        )


def _append_embedding_summary_to_log(results: list[EmbeddingBenchmarkResult], dataset: str):
    """Append embedding comparison results to decisions_log.md"""
    decisions_log = Path(__file__).parent.parent / "decisions_log.md"
    if not decisions_log.exists():
        return

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    rows = "\n".join(
        f"| {r.display_name} | {r.device.upper()} | {r.dims} | "
        f"{r.recall_at_5:.4f} | {r.query_latency_ms:.1f}ms | "
        f"${r.cost_per_1m_tokens:.3f} | "
        f"{'FREE' if r.cost_per_1m_tokens == 0 else f'${r.indexing_cost_usd:.4f}'} | "
        f"{r.recall_per_dollar:.1f} |"
        for r in sorted(results, key=lambda x: x.recall_at_5, reverse=True)
    )

    block = f"""
### EMB-RUN — Embedding Model Comparison ({dataset}, {timestamp} UTC)

| Model | Device | Dims | Recall@5 | Query P50 | $/1M tokens | Index cost | Recall/$ |
|-------|--------|------|----------|-----------|-------------|------------|----------|
{rows}

**Key findings**:
- Best Recall: {max(results, key=lambda x: x.recall_at_5).display_name}
- Fastest query: {min(results, key=lambda x: x.query_latency_ms).display_name} ({min(results, key=lambda x: x.query_latency_ms).query_latency_ms:.1f}ms)
- Best free model: {next((r.display_name for r in sorted(results, key=lambda x: x.recall_at_5, reverse=True) if r.cost_per_1m_tokens == 0), 'N/A')}
- GPU speedup: {_calc_gpu_speedup(results)}

---
"""

    with open(decisions_log, "a") as f:
        f.write(block)
    print(f"[embedding_benchmark] Summary appended to {decisions_log}")


def _calc_gpu_speedup(results: list[EmbeddingBenchmarkResult]) -> str:
    """Calculate GPU vs CPU speedup ratio if both exist."""
    for model_name in set(r.model_name for r in results):
        cpu = next((r for r in results if r.model_name == model_name and r.device == "cpu"), None)
        gpu = next((r for r in results if r.model_name == model_name and r.device in ("cuda", "mps")), None)
        if cpu and gpu:
            speedup = cpu.query_latency_ms / gpu.query_latency_ms
            return f"{speedup:.1f}× ({gpu.display_name}, CPU→{gpu.device.upper()})"
    return "N/A (no GPU available)"


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--dataset", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--models", nargs="+", choices=list(EMBEDDING_MODELS.keys()),
                        help="Models to benchmark (default: all)")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda", "mps"],
                        help="Devices to test (default: auto-detect)")
    parser.add_argument("--no-save", action="store_true", help="Print results without saving")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"EMBEDDING MODEL BENCHMARK")
    print(f"Dataset: {args.dataset}")
    print(f"Device info: {detect_device()}")
    print(f"{'='*60}")

    results = run_embedding_benchmark(
        dataset_name=args.dataset,
        model_names=args.models,
        devices=args.devices,
    )

    print_summary_table(results)

    if not args.no_save:
        save_embedding_results(results, args.dataset)


if __name__ == "__main__":
    main()
