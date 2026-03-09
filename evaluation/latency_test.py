"""
evaluation/latency_test.py — System performance benchmarking.

Measures: P50 latency, P95 latency, throughput (q/s), peak RAM, storage.
Decision D-011: 3 warmup runs discarded, 10 timed runs averaged.

Usage:
    from evaluation.latency_test import measure_latency
    metrics = measure_latency(rag_system, queries, k=5)
"""

import time
import os
import statistics
import psutil
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BENCHMARK_WARMUP_RUNS, BENCHMARK_TIMED_RUNS, TOP_K


@dataclass
class LatencyResult:
    p50_latency_s: float
    p95_latency_s: float
    mean_latency_s: float
    min_latency_s: float
    max_latency_s: float
    throughput_qps: float
    peak_ram_mb: float
    latencies_raw: list[float]


def measure_latency(
    rag_system,
    queries: list[str],
    k: int = TOP_K,
    warmup_runs: int = BENCHMARK_WARMUP_RUNS,
    timed_runs: int = BENCHMARK_TIMED_RUNS,
) -> LatencyResult:
    """
    Benchmark latency for a RAG system.
    
    Protocol (D-011):
    1. Run warmup_runs queries → discard (cold cache / JIT penalty)
    2. Run timed_runs queries → record latencies
    3. Compute P50, P95, mean, throughput
    
    Args:
        rag_system: Any BaseRAG subclass (must be already indexed)
        queries: List of test queries to use
        k: Number of documents to retrieve
    """
    process = psutil.Process(os.getpid())

    # Cycle through queries if we don't have enough
    def get_query(i):
        return queries[i % len(queries)]

    # Warmup runs (D-011: discard cold-cache penalty)
    print(f"[latency] Running {warmup_runs} warmup queries...")
    for i in range(warmup_runs):
        _ = rag_system.retrieve(get_query(i), k=k)

    # Timed runs
    print(f"[latency] Running {timed_runs} timed queries...")
    latencies = []
    peak_ram = 0.0

    for i in range(timed_runs):
        query = get_query(warmup_runs + i)
        mem_before = process.memory_info().rss / 1024 / 1024

        t0 = time.perf_counter()
        _ = rag_system.retrieve(query, k=k)
        elapsed = time.perf_counter() - t0

        mem_after = process.memory_info().rss / 1024 / 1024
        peak_ram = max(peak_ram, mem_after)
        latencies.append(elapsed)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95_idx = int(0.95 * len(latencies))
    p95 = latencies[p95_idx]
    mean = statistics.mean(latencies)
    throughput = 1.0 / mean if mean > 0 else 0

    print(f"[latency] P50={p50:.3f}s  P95={p95:.3f}s  "
          f"Throughput={throughput:.1f}q/s  Peak RAM={peak_ram:.0f}MB")

    return LatencyResult(
        p50_latency_s=p50,
        p95_latency_s=p95,
        mean_latency_s=mean,
        min_latency_s=min(latencies),
        max_latency_s=max(latencies),
        throughput_qps=throughput,
        peak_ram_mb=peak_ram,
        latencies_raw=latencies,
    )


def measure_storage(index_path: Path) -> float:
    """
    Measure disk storage used by a RAG index directory in MB.
    Called after indexing to record storage cost.
    """
    if not index_path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in index_path.rglob("*") if f.is_file())
    return total / 1024 / 1024
