"""
fix_qa_pairs.py — Repair QA pairs by linking ground_truth back to document sources.

The problem: RAGAS QA generation creates {question, ground_truth} but doesn't track
which document each pair came from. The retrieval_metrics.py system needs
'relevant_sources' to compute Recall@5, so without it, recall is always 0.0.

Solution: Find which document contains each ground_truth snippet, link it back.

Usage:
    python fix_qa_pairs.py --dataset small
"""

import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher
import sys

sys.path.insert(0, str(Path(__file__).parent))

from scripts.fetch_datasets import load_dataset
from config import RESULTS_DIR, QA_PAIRS_PER_DATASET


def find_source_for_ground_truth(ground_truth: str, documents: list[dict], threshold: float = 0.5) -> str:
    """
    Find which document contains the ground_truth text.
    Uses fuzzy text matching to handle minor differences.

    Returns: source string, or empty string if not found.
    """
    ground_lower = ground_truth.lower()

    # First try exact substring match
    for doc in documents:
        doc_content = doc.get("content", "").lower()
        if ground_lower in doc_content:
            return doc.get("source", "unknown")

    # Fall back to fuzzy matching (for cases where exact match fails)
    best_match_ratio = 0.0
    best_source = ""

    # Use first 100 chars of ground truth for matching
    gt_snippet = ground_truth[:100].lower()

    for doc in documents:
        doc_content = doc.get("content", "").lower()

        # Check if any 100-char window of the doc is similar to ground truth
        for i in range(0, len(doc_content) - 100, 100):
            window = doc_content[i:i+100]
            ratio = SequenceMatcher(None, gt_snippet, window).ratio()

            if ratio > best_match_ratio:
                best_match_ratio = ratio
                best_source = doc.get("source", "unknown")

    if best_match_ratio >= threshold:
        return best_source

    return ""


def fix_qa_pairs(dataset_name: str, dry_run: bool = False) -> dict:
    """
    Fix QA pairs by adding 'source' and 'relevant_sources' fields.

    Returns: stats dict with fix summary.
    """
    print(f"\n{'='*60}")
    print(f"FIXING QA PAIRS: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load documents
    documents = load_dataset(dataset_name)
    print(f"\nLoaded {len(documents)} documents")

    # Load QA pairs
    qa_cache = RESULTS_DIR / "benchmark_tables" / f"qa_{dataset_name}.json"

    if not qa_cache.exists():
        print(f"[ERROR] QA cache not found: {qa_cache}")
        return {"error": "qa_cache_not_found"}

    with open(qa_cache) as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Fix each QA pair
    fixed_pairs = []
    found_count = 0

    for i, pair in enumerate(qa_pairs):
        ground_truth = pair.get("ground_truth", "")

        # Find which document contains this ground truth
        source = find_source_for_ground_truth(ground_truth, documents, threshold=0.5)

        if source:
            found_count += 1

        fixed_pair = {
            "question": pair.get("question", ""),
            "ground_truth": ground_truth,
            "source": source if source else "unknown",
            "relevant_sources": [source] if source else [],
        }

        fixed_pairs.append(fixed_pair)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(qa_pairs)} pairs ({found_count} sources found so far)")

    print(f"\n[RESULT] Successfully linked {found_count}/{len(qa_pairs)} QA pairs to sources")

    # Show examples
    print(f"\nExample fixed pairs:")
    for i, pair in enumerate(fixed_pairs[:3]):
        print(f"\n  Pair {i+1}:")
        print(f"    Question: {pair['question'][:60]}...")
        print(f"    Source: {pair['source']}")
        print(f"    Relevant Sources: {pair['relevant_sources']}")

    # Save fixed pairs (with backup)
    if not dry_run:
        backup_path = qa_cache.with_suffix(".backup.json")
        if qa_cache.exists():
            with open(qa_cache) as f:
                original = json.load(f)
            with open(backup_path, "w") as f:
                json.dump(original, f, indent=2)
            print(f"\n[OK] Backed up original to {backup_path}")

        with open(qa_cache, "w") as f:
            json.dump(fixed_pairs, f, indent=2)

        print(f"[OK] Saved fixed QA pairs to {qa_cache}")
    else:
        print(f"\n[DRY RUN] No changes made")

    return {
        "dataset": dataset_name,
        "total_pairs": len(qa_pairs),
        "pairs_with_source": found_count,
        "success_rate": found_count / len(qa_pairs) if qa_pairs else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Fix QA pairs by adding source links")
    parser.add_argument("--dataset", choices=["small", "medium", "large"], default="small",
                       help="Dataset to fix")
    parser.add_argument("--dry-run", action="store_true", help="Don't save changes")
    parser.add_argument("--all", action="store_true", help="Fix all datasets")

    args = parser.parse_args()

    datasets = ["small", "medium", "large"] if args.all else [args.dataset]

    all_stats = []
    for dataset in datasets:
        try:
            stats = fix_qa_pairs(dataset, dry_run=args.dry_run)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n[ERROR] Failed to fix {dataset}: {e}")
            all_stats.append({"dataset": dataset, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    for stats in all_stats:
        if "error" in stats:
            print(f"\n[ERROR] {stats.get('dataset', 'unknown')}: {stats['error']}")
        else:
            rate = stats.get("success_rate", 0) * 100
            print(f"\n{stats['dataset'].upper()}: {rate:.1f}% of QA pairs linked to sources "
                  f"({stats.get('pairs_with_source', 0)}/{stats.get('total_pairs', 0)})")

    print(f"\n[IMPORTANT] After fixing QA pairs:")
    print(f"  1. Clear result cache: rm -rf results/indexes/")
    print(f"  2. Re-run benchmark: python run_benchmark.py --dataset {args.dataset}")
    print(f"  3. Recall@5 should now be > 0.0")


if __name__ == "__main__":
    main()
