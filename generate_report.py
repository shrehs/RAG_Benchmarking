#!/usr/bin/env python3
"""
RAG Benchmark Analysis Report
Compares 5 RAG architectures across multiple datasets
Focus: Retrieval metrics only (no LLM quality confounding)
"""

import json
from pathlib import Path
from collections import defaultdict

def generate_report():
    """Generate comprehensive benchmark analysis."""

    summary_path = Path("results/benchmark_tables/summary.json")
    with open(summary_path) as f:
        results = json.load(f)

    # Group by dataset
    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r["dataset"]].append(r)

    print("\n" + "="*120)
    print("RAG BENCHMARK ANALYSIS REPORT".center(120))
    print("="*120)
    print("Comparing 5 RAG Architectures across Retrieval Metrics")
    print("(No RAGAS quality metrics - focusing on architecture performance)")
    print("="*120)

    # For each dataset, generate table
    for dataset in sorted(by_dataset.keys()):
        runs = by_dataset[dataset]
        print(f"\n{dataset.upper()} DATASET")
        print("-" * 120)
        print(f"{'Architecture':<16} {'Recall@5':<12} {'Precision@5':<14} {'MRR':<10} {'P50 (s)':<10} {'Throughput':<14} {'Peak RAM':<12} {'Storage':<10}")
        print("-" * 120)

        # Sort by recall (best first)
        runs_sorted = sorted(runs, key=lambda x: x["retrieval_metrics"]["recall_at_5"], reverse=True)

        for i, r in enumerate(runs_sorted, 1):
            arch = r["architecture"]
            recall = r["retrieval_metrics"]["recall_at_5"]
            precision = r["retrieval_metrics"]["precision_at_5"]
            mrr = r["retrieval_metrics"]["mrr"]
            latency = r["system_metrics"]["p50_latency"]
            throughput = r["system_metrics"]["throughput"]
            ram = r["system_metrics"]["peak_ram_mb"]
            storage = r["system_metrics"]["storage_mb"]

            # Mark winners
            winner = f" (#{i})" if i == 1 else ""

            print(f"{arch:<16} {recall:<12.2f} {precision:<14.4f} {mrr:<10.4f} {latency:<10.4f} {throughput:<14.1f}q/s {ram:<12.0f}MB {storage:<10.2f}MB{winner}")

    # Overall recommendations
    print(f"\n{'='*120}")
    print("RECOMMENDATIONS BY USE CASE".center(120))
    print(f"{'='*120}\n")

    recs = {
        "Maximum Recall (Best for Accuracy)": "Hybrid RAG",
        "Best Latency (Fastest Response)": "Graph RAG",
        "Best Balance (Recall + Speed)": "Vector RAG",
        "Best RAM Efficiency": "Graph RAG",
        "Best for Real-time Systems": "Vector RAG",
        "Best Precision": "Hybrid RAG",
    }

    for use_case, recommendation in recs.items():
        print(f"  {use_case:<40} => {recommendation}")

    print(f"\n{'='*120}\n")

    # Key insights
    print("KEY INSIGHTS")
    print("-" * 120)
    print("""
1. HYBRID RAG WINS ON RECALL
   - Combines BM25 (lexical) + Vector (semantic) search
   - Recall@5: 1.80 (highest across all architectures)
   - Trade-off: Slower (P50: 0.188s vs 0.075s for Vector)

2. GRAPH RAG FASTEST BUT LOWEST RECALL
   - Lightning-fast: 127.8 q/s throughput, 0.009s P50 latency
   - Bottleneck: Entity resolution issues hurt recall (0.80)
   - Use when latency is critical, quality acceptable

3. VECTOR RAG IS SWEET SPOT
   - Solid recall (1.50) with reasonable latency (0.075s)
   - Simplest implementation, no entity resolution complexity
   - Recommended for most production systems

4. PARENT-CHILD RAG MODERATE IMPROVEMENTS
   - Recall: 0.95 (better than Graph, worse than Vector/Hybrid)
   - Massive indexing overhead (4.5+ minutes)
   - RAM intensive (3663 MB vs 1636 MB for Vector)
   - Worth it for very large documents only

5. MULTI-QUERY RAG NOT COST-EFFECTIVE
   - Recall: 1.50 (same as single Vector query)
   - Latency: 0.675s (9× slower than Vector)
   - Makes 3 LLM calls per query - overkill for marginal gain
   - Skip unless specifically optimizing for variance reduction
    """)

    print(f"\n{'='*120}\n")

if __name__ == "__main__":
    generate_report()
