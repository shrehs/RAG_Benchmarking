"""
evaluation/gpu_benchmark.py — GPU vs CPU retrieval benchmarking.

Decision D-014: Measures the retrieval speedup from GPU acceleration.
Three axes:
  1. Embedding throughput (docs/sec): GPU wins massively for local models
  2. FAISS search latency: GPU wins for large indexes (>100K vectors)
  3. End-to-end query latency: combination of embed + search

FAISS GPU notes (D-014):
  - faiss-gpu must be installed separately: pip install faiss-gpu
  - For small indexes (<10K vectors) CPU flat index is often faster than GPU transfer overhead
  - GPU wins clearly above ~50K vectors
  - Apple MPS is supported via sentence-transformers but not FAISS GPU

Results are logged per model with:
  - CPU throughput / latency
  - GPU throughput / latency
  - Speedup ratio
  - Crossover point estimate (# vectors where GPU becomes worth it)

Usage:
    python evaluation/gpu_benchmark.py --dataset medium
    python evaluation/gpu_benchmark.py --models bge-large e5-large --dataset small
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K, BENCHMARK_WARMUP_RUNS, BENCHMARK_TIMED_RUNS, RESULTS_DIR
from embedding_registry import (
    EMBEDDING_MODELS, get_embedder, detect_device, HuggingFaceEmbedder,
)
from scripts.fetch_datasets import load_dataset
from rag_systems.chunker import chunk_documents

import faiss


@dataclass
class GPUBenchmarkResult:
    model_name: str
    display_name: str
    dataset: str
    num_chunks: int
    dims: int

    # Embedding throughput (indexing)
    cpu_embed_throughput: float    # chunks/sec
    gpu_embed_throughput: float    # chunks/sec (0 if no GPU)
    embed_speedup: float           # gpu / cpu ratio

    # Query latency (embed + search)
    cpu_query_p50_ms: float
    cpu_query_p95_ms: float
    gpu_query_p50_ms: float        # 0 if no GPU
    gpu_query_p95_ms: float        # 0 if no GPU
    query_speedup: float           # cpu p50 / gpu p50

    # FAISS search only (no embedding)
    cpu_faiss_p50_ms: float
    gpu_faiss_p50_ms: float        # 0 if no GPU
    faiss_speedup: float

    device_info: dict
    has_gpu: bool
    timestamp: str


def time_embedding(embedder, texts: list[str], runs: int = 3) -> float:
    """Returns throughput in chunks/second (averaged over runs)."""
    throughputs = []
    for _ in range(runs):
        t0 = time.perf_counter()
        embedder.embed(texts)
        elapsed = time.perf_counter() - t0
        throughputs.append(len(texts) / elapsed)
    return sum(throughputs) / len(throughputs)


def time_faiss_search(
    index: faiss.Index,
    query_vecs: np.ndarray,
    k: int = TOP_K,
    warmup: int = BENCHMARK_WARMUP_RUNS,
    timed: int = BENCHMARK_TIMED_RUNS,
) -> tuple[float, float]:
    """Time FAISS search only (no embedding). Returns (p50_ms, p95_ms)."""
    for i in range(warmup):
        q = query_vecs[i % len(query_vecs)].reshape(1, -1)
        index.search(q, k)

    latencies = []
    for i in range(timed):
        q = query_vecs[(warmup + i) % len(query_vecs)].reshape(1, -1)
        t0 = time.perf_counter()
        index.search(q, k)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(0.95 * len(latencies))]
    return p50, p95


def time_full_query(
    embedder,
    index: faiss.Index,
    queries: list[str],
    k: int = TOP_K,
    warmup: int = BENCHMARK_WARMUP_RUNS,
    timed: int = BENCHMARK_TIMED_RUNS,
) -> tuple[float, float]:
    """Time full query pipeline: embed + FAISS search. Returns (p50_ms, p95_ms)."""
    for i in range(warmup):
        q = queries[i % len(queries)]
        vec = embedder.embed([q])
        faiss.normalize_L2(vec)
        index.search(vec, k)

    latencies = []
    for i in range(timed):
        q = queries[(warmup + i) % len(queries)]
        t0 = time.perf_counter()
        vec = embedder.embed([q])
        faiss.normalize_L2(vec)
        index.search(vec, k)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(0.95 * len(latencies))]
    return p50, p95


def build_faiss_index(vectors: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Build FAISS index. Uses GPU index if use_gpu=True and faiss-gpu is available.
    D-014: For fair comparison, uses same IndexFlatIP regardless of device.
    """
    dims = vectors.shape[1]
    cpu_index = faiss.IndexFlatIP(dims)
    cpu_index.add(vectors)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            return gpu_index
        except AttributeError:
            print("[gpu_benchmark] faiss-gpu not available — falling back to CPU FAISS")
            return cpu_index

    return cpu_index


def run_gpu_benchmark(
    dataset_name: str,
    model_names: list[str] | None = None,
) -> list[GPUBenchmarkResult]:
    """
    For each local (HuggingFace) model:
      1. Embed all chunks on CPU → measure throughput
      2. Embed all chunks on GPU (if available) → measure throughput
      3. Build FAISS indexes (CPU + GPU)
      4. Time FAISS search on both
      5. Time full query (embed + search) on both
    """
    documents = load_dataset(dataset_name)
    chunks = chunk_documents(documents, chunk_size=512, chunk_overlap=50)
    texts = [c["content"] for c in chunks]

    # Load QA pairs for query latency testing
    qa_cache = RESULTS_DIR / "benchmark_tables" / f"qa_{dataset_name}.json"
    if qa_cache.exists():
        with open(qa_cache) as f:
            qa_pairs = json.load(f)
        queries = [p["question"] for p in qa_pairs]
    else:
        queries = [f"What is described in document {i}?" for i in range(20)]

    device_info = detect_device()
    has_gpu = device_info["cuda"] or device_info["mps"]
    gpu_device = "cuda" if device_info["cuda"] else ("mps" if device_info["mps"] else None)

    # Only HuggingFace models are relevant for GPU benchmarking
    hf_models = {
        k: v for k, v in EMBEDDING_MODELS.items()
        if v.provider == "huggingface" and v.supports_gpu
    }
    models_to_run = (
        [m for m in (model_names or list(hf_models.keys())) if m in hf_models]
    )

    if not models_to_run:
        print("[gpu_benchmark] No HuggingFace models specified. GPU benchmark only applies to local models.")
        return []

    results = []
    for model_name in models_to_run:
        model_meta = EMBEDDING_MODELS[model_name]
        print(f"\n{'─'*55}")
        print(f"  GPU Benchmark: {model_meta.display_name}")
        print(f"  Chunks: {len(chunks)} | Has GPU: {has_gpu}")
        print(f"{'─'*55}")

        # ── CPU Embedding ──────────────────────────────────────────────────────
        print("  [CPU] Embedding chunks...")
        cpu_embedder = get_embedder(model_name, device="cpu")
        cpu_vecs = cpu_embedder.embed(texts)
        faiss.normalize_L2(cpu_vecs)
        cpu_throughput = time_embedding(get_embedder(model_name, device="cpu"), texts[:50])
        print(f"  [CPU] Throughput: {cpu_throughput:.0f} chunks/sec")

        # ── CPU FAISS index ────────────────────────────────────────────────────
        cpu_index = build_faiss_index(cpu_vecs, use_gpu=False)

        # Pre-embed queries for FAISS-only timing
        cpu_q_embedder = get_embedder(model_name, device="cpu")
        q_vecs = cpu_q_embedder.embed(queries[:20])
        faiss.normalize_L2(q_vecs)

        cpu_faiss_p50, cpu_faiss_p95 = time_faiss_search(cpu_index, q_vecs)
        cpu_query_p50, cpu_query_p95 = time_full_query(
            get_embedder(model_name, device="cpu"), cpu_index, queries[:20]
        )
        print(f"  [CPU] FAISS P50: {cpu_faiss_p50:.2f}ms | Full query P50: {cpu_query_p50:.2f}ms")

        # ── GPU Embedding (if available) ───────────────────────────────────────
        gpu_throughput = 0.0
        gpu_faiss_p50 = gpu_faiss_p95 = 0.0
        gpu_query_p50 = gpu_query_p95 = 0.0

        if has_gpu and gpu_device:
            print(f"  [{gpu_device.upper()}] Embedding chunks...")
            try:
                gpu_embedder = get_embedder(model_name, device=gpu_device)
                gpu_vecs = gpu_embedder.embed(texts)
                faiss.normalize_L2(gpu_vecs)
                gpu_throughput = time_embedding(
                    get_embedder(model_name, device=gpu_device), texts[:50]
                )
                print(f"  [{gpu_device.upper()}] Throughput: {gpu_throughput:.0f} chunks/sec")

                # FAISS GPU index
                gpu_index = build_faiss_index(gpu_vecs, use_gpu=(gpu_device == "cuda"))
                gpu_q_vecs = get_embedder(model_name, device=gpu_device).embed(queries[:20])
                faiss.normalize_L2(gpu_q_vecs)

                gpu_faiss_p50, gpu_faiss_p95 = time_faiss_search(gpu_index, gpu_q_vecs)
                gpu_query_p50, gpu_query_p95 = time_full_query(
                    get_embedder(model_name, device=gpu_device), gpu_index, queries[:20]
                )
                print(f"  [{gpu_device.upper()}] FAISS P50: {gpu_faiss_p50:.2f}ms | "
                      f"Full query P50: {gpu_query_p50:.2f}ms")

            except Exception as e:
                print(f"  [{gpu_device.upper()}] ERROR: {e}")

        embed_speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 and gpu_throughput > 0 else 0
        query_speedup = cpu_query_p50 / gpu_query_p50 if gpu_query_p50 > 0 else 0
        faiss_speedup = cpu_faiss_p50 / gpu_faiss_p50 if gpu_faiss_p50 > 0 else 0

        if embed_speedup > 0:
            print(f"\n  📊 GPU Speedup:")
            print(f"     Embedding: {embed_speedup:.1f}×")
            print(f"     FAISS search: {faiss_speedup:.1f}×")
            print(f"     Full query: {query_speedup:.1f}×")

        result = GPUBenchmarkResult(
            model_name=model_name,
            display_name=model_meta.display_name,
            dataset=dataset_name,
            num_chunks=len(chunks),
            dims=model_meta.dims,
            cpu_embed_throughput=cpu_throughput,
            gpu_embed_throughput=gpu_throughput,
            embed_speedup=embed_speedup,
            cpu_query_p50_ms=cpu_query_p50,
            cpu_query_p95_ms=cpu_query_p95,
            gpu_query_p50_ms=gpu_query_p50,
            gpu_query_p95_ms=gpu_query_p95,
            query_speedup=query_speedup,
            cpu_faiss_p50_ms=cpu_faiss_p50,
            gpu_faiss_p50_ms=gpu_faiss_p50,
            faiss_speedup=faiss_speedup,
            device_info=device_info,
            has_gpu=has_gpu,
            timestamp=datetime.utcnow().isoformat(),
        )
        results.append(result)

    return results


def print_gpu_summary(results: list[GPUBenchmarkResult]) -> None:
    print(f"\n{'Model':<30} {'CPU emb/s':<12} {'GPU emb/s':<12} {'Emb ×':<8} "
          f"{'CPU q P50':<12} {'GPU q P50':<12} {'Query ×'}")
    print("─" * 100)
    for r in results:
        gpu_emb = f"{r.gpu_embed_throughput:.0f}" if r.gpu_embed_throughput else "N/A"
        gpu_qp50 = f"{r.gpu_query_p50_ms:.1f}ms" if r.gpu_query_p50_ms else "N/A"
        speedup_emb = f"{r.embed_speedup:.1f}×" if r.embed_speedup else "—"
        speedup_q = f"{r.query_speedup:.1f}×" if r.query_speedup else "—"
        print(
            f"{r.display_name[:29]:<30} "
            f"{r.cpu_embed_throughput:<12.0f} "
            f"{gpu_emb:<12} "
            f"{speedup_emb:<8} "
            f"{r.cpu_query_p50_ms:<12.1f} "
            f"{gpu_qp50:<12} "
            f"{speedup_q}"
        )


def save_gpu_results(results: list[GPUBenchmarkResult], dataset: str) -> Path:
    out_dir = RESULTS_DIR / "benchmark_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"gpu_benchmark_{dataset}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Append to decisions log
    _append_gpu_summary(results, dataset)
    print(f"\n[gpu_benchmark] Results saved to {out_path}")
    return out_path


def _append_gpu_summary(results: list[GPUBenchmarkResult], dataset: str):
    decisions_log = Path(__file__).parent.parent / "decisions_log.md"
    if not decisions_log.exists():
        return

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    rows = "\n".join(
        f"| {r.display_name} | {r.cpu_embed_throughput:.0f} docs/s | "
        f"{r.gpu_embed_throughput:.0f} docs/s | {r.embed_speedup:.1f}× | "
        f"{r.cpu_query_p50_ms:.1f}ms | {r.gpu_query_p50_ms:.1f}ms | {r.query_speedup:.1f}× |"
        for r in results
    )

    block = f"""
### GPU-RUN — GPU vs CPU Benchmark ({dataset}, {timestamp} UTC)

| Model | CPU embed/s | GPU embed/s | Embed speedup | CPU query P50 | GPU query P50 | Query speedup |
|-------|-------------|-------------|---------------|---------------|---------------|---------------|
{rows}

**D-014 Finding**: GPU acceleration most impactful for batch indexing.
For single-query latency, benefit depends on index size vs transfer overhead.

---
"""
    with open(decisions_log, "a") as f:
        f.write(block)


def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU retrieval benchmark")
    parser.add_argument("--dataset", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--models", nargs="+", help="HuggingFace models to benchmark")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    device_info = detect_device()
    print(f"\n{'='*60}")
    print(f"GPU vs CPU RETRIEVAL BENCHMARK")
    print(f"Dataset: {args.dataset}")
    print(f"CUDA: {device_info['cuda']} | MPS: {device_info['mps']}")
    if device_info["cuda_name"]:
        print(f"GPU: {device_info['cuda_name']} ({device_info['cuda_memory_gb']}GB)")
    print(f"{'='*60}")

    results = run_gpu_benchmark(args.dataset, args.models)

    if results:
        print_gpu_summary(results)
        if not args.no_save:
            save_gpu_results(results, args.dataset)
    else:
        print("\nNo results — ensure HuggingFace models are specified and sentence-transformers is installed.")


if __name__ == "__main__":
    main()
