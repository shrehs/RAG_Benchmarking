"""
run_benchmark.py — Main benchmark runner.

Orchestrates: load dataset → index RAG systems → measure latency → evaluate quality → log results.

Usage:
    # Run all architectures on one dataset
    python run_benchmark.py --dataset small

    # Run a specific architecture
    python run_benchmark.py --dataset small --arch vector

    # Run all combinations
    python run_benchmark.py --all

    # Dry run (no API calls, uses stubs)
    python run_benchmark.py --dataset small --dry-run
"""

import argparse
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import TOP_K, RESULTS_DIR, QA_PAIRS_PER_DATASET
from scripts.fetch_datasets import load_dataset
from rag_systems import ALL_SYSTEMS
from evaluation.latency_test import measure_latency, measure_storage
from evaluation.retrieval_metrics import compute_retrieval_metrics
from evaluation.ragas_eval import evaluate_rag, generate_qa_pairs
from evaluation.log_results import save_result


DATASETS = ["small", "medium", "large"]
ARCHITECTURES = list(ALL_SYSTEMS.keys())

# Global run counter (reads last RUN ID from results dir)
def get_next_run_id() -> int:
    results_dir = RESULTS_DIR / "benchmark_tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    existing = list(results_dir.glob("run_*.json"))
    if not existing:
        return 1
    ids = [int(f.stem.split("_")[1]) for f in existing if f.stem.split("_")[1].isdigit()]
    return max(ids) + 1 if ids else 1


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_experiment(
    arch_name: str,
    dataset_name: str,
    run_id: int,
    dry_run: bool = False,
) -> dict:
    """
    Run a single (architecture × dataset) experiment.
    Returns the full results dict.
    """
    print(f"\n{'='*60}")
    print(f"RUN-{run_id:03d} | {arch_name.upper()} × {dataset_name.upper()}")
    print(f"{'='*60}")

    # 1. Load dataset
    print(f"[run] Loading dataset: {dataset_name}")
    documents = load_dataset(dataset_name)
    print(f"[run] Loaded {len(documents)} documents")

    # 2. Load or generate QA pairs
    qa_cache = RESULTS_DIR / "benchmark_tables" / f"qa_{dataset_name}.json"
    if qa_cache.exists():
        with open(qa_cache) as f:
            qa_pairs = json.load(f)
        qa_pairs = qa_pairs[:QA_PAIRS_PER_DATASET]
        print(f"[run] Loaded {len(qa_pairs)} QA pairs from cache")
    else:
        print(f"[run] Generating QA pairs for {dataset_name}...")
        qa_pairs = generate_qa_pairs(documents, dataset_name=dataset_name)
        qa_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(qa_cache, "w") as f:
            json.dump(qa_pairs, f, indent=2)

    # 3. Initialize RAG system
    rag_class = ALL_SYSTEMS[arch_name]
    rag = rag_class()

    # 4. Index (with cache)
    cache_path = RESULTS_DIR / "indexes" / arch_name / dataset_name
    if cache_path.exists() and not dry_run:
        print(f"[run] Loading cached index from {cache_path}")
        rag.load(cache_path)
    else:
        print(f"[run] Building index...")
        t_index_start = time.perf_counter()
        rag.index(documents, cache_path=cache_path)
        index_time = time.perf_counter() - t_index_start
        print(f"[run] Indexing complete in {index_time:.1f}s")

    # 5. Measure latency
    queries = [pair["question"] for pair in qa_pairs]
    print(f"[run] Measuring latency...")
    latency_result = measure_latency(rag, queries)
    storage_mb = measure_storage(cache_path)

    system_metrics = {
        "p50_latency": latency_result.p50_latency_s,
        "p95_latency": latency_result.p95_latency_s,
        "mean_latency": latency_result.mean_latency_s,
        "throughput": latency_result.throughput_qps,
        "peak_ram_mb": latency_result.peak_ram_mb,
        "storage_mb": storage_mb,
    }

    # 6. Retrieval metrics (approximate — using source matching)
    print(f"[run] Computing retrieval metrics...")
    # Build relevance from QA pairs (source-level relevance)
    qa_with_relevance = [
        {
            "question": pair["question"],
            "relevant_sources": [pair.get("source", "")] if pair.get("source") else [],
        }
        for pair in qa_pairs
    ]
    retrieval_result = compute_retrieval_metrics(rag, qa_with_relevance, k=TOP_K)
    retrieval_metrics = {
        "recall_at_5": retrieval_result.get(f"recall_at_{TOP_K}", 0),
        "precision_at_5": retrieval_result.get(f"precision_at_{TOP_K}", 0),
        "mrr": retrieval_result.get("mrr", 0),
    }

    # 7. RAGAS quality evaluation
    print(f"[run] Running RAGAS evaluation...")
    if dry_run:
        quality_metrics = {"faithfulness": 0, "answer_relevancy": 0,
                           "context_precision": 0, "context_recall": 0}
        cost = {"embedding": 0, "generation": 0, "eval": 0, "total": 0, "per_query": 0}
    else:
        ragas_result = evaluate_rag(rag, qa_pairs, k=TOP_K)
        quality_metrics = {
            "faithfulness": ragas_result.faithfulness,
            "answer_relevancy": ragas_result.answer_relevancy,
            "context_precision": ragas_result.context_precision,
            "context_recall": ragas_result.context_recall,
        }

        # Aggregate cost from a sample of full query() calls
        sample_result = rag.query(queries[0], k=TOP_K)
        cost_sample = sample_result.cost_usd
        n_qa = len(qa_pairs)
        cost = {
            "embedding": cost_sample["embedding"] * n_qa,
            "generation": cost_sample["generation"] * n_qa,
            "eval": 0.0,   # judge cost tracked separately
            "total": (cost_sample["embedding"] + cost_sample["generation"]) * n_qa,
            "per_query": cost_sample["embedding"] + cost_sample["generation"],
        }

    # 8. Log results
    notes = f"k={TOP_K}, arch={arch_name}, dataset={dataset_name}, docs={len(documents)}"
    saved_path = save_result(
        run_id=run_id,
        architecture=arch_name,
        dataset=dataset_name,
        retrieval_metrics=retrieval_metrics,
        system_metrics=system_metrics,
        quality_metrics=quality_metrics,
        cost=cost,
        notes=notes,
        git_hash=get_git_hash(),
    )

    print(f"\n[run] ✅ RUN-{run_id:03d} complete")
    print(f"[run]    Recall@5: {retrieval_metrics['recall_at_5']:.4f}")
    print(f"[run]    P50 Latency: {system_metrics['p50_latency']:.3f}s")
    print(f"[run]    Faithfulness: {quality_metrics['faithfulness']:.4f}")
    print(f"[run]    Results: {saved_path}")

    return {
        "run_id": run_id,
        "architecture": arch_name,
        "dataset": dataset_name,
        "retrieval_metrics": retrieval_metrics,
        "system_metrics": system_metrics,
        "quality_metrics": quality_metrics,
        "cost": cost,
    }


def main():
    parser = argparse.ArgumentParser(
        description="RAG Benchmark Runner — see decisions_log.md for all parameter decisions"
    )
    parser.add_argument("--dataset", choices=DATASETS, help="Dataset to benchmark")
    parser.add_argument("--arch", choices=ARCHITECTURES, help="Architecture to benchmark")
    parser.add_argument("--all", action="store_true", help="Run all arch × dataset combinations")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, use stubs")
    args = parser.parse_args()

    if not (args.all or args.dataset):
        parser.print_help()
        return

    run_id = get_next_run_id()

    experiments = []
    if args.all:
        for dataset in DATASETS:
            for arch in ARCHITECTURES:
                experiments.append((arch, dataset))
    else:
        archs_to_run = [args.arch] if args.arch else ARCHITECTURES
        for arch in archs_to_run:
            experiments.append((arch, args.dataset))

    print(f"\n{'='*60}")
    print(f"RAG BENCHMARK — {len(experiments)} experiments")
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}")

    all_results = []
    for arch, dataset in experiments:
        result = run_experiment(arch, dataset, run_id, dry_run=args.dry_run)
        all_results.append(result)
        run_id += 1

    # Save summary table
    summary_path = RESULTS_DIR / "benchmark_tables" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL RUNS COMPLETE")
    print(f"Summary: {summary_path}")
    print(f"Full log: decisions_log.md")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Architecture':<18} {'Dataset':<10} {'Recall@5':<12} {'P50 (s)':<10} {'Faith.':<10} {'$/query':<10}")
    print("-" * 72)
    for r in all_results:
        print(
            f"{r['architecture']:<18} "
            f"{r['dataset']:<10} "
            f"{r['retrieval_metrics']['recall_at_5']:<12.4f} "
            f"{r['system_metrics']['p50_latency']:<10.3f} "
            f"{r['quality_metrics']['faithfulness']:<10.4f} "
            f"{r['cost']['per_query']:<10.6f}"
        )


if __name__ == "__main__":
    main()
