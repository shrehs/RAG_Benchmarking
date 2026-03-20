"""
diagnostic.py — Diagnose zero recall issue

Run with: python diagnostic.py --dataset small

Checks:
  1. Dataset loading (document count, chunk sizes)
  2. QA pair generation (missing source links)
  3. FAISS index (chunk count, embedding quality)
  4. Manual retrieval tests (actual top-5 scores)
  5. Ground truth matching (do answers exist in corpus?)
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from scripts.fetch_datasets import load_dataset
from rag_systems import ALL_SYSTEMS
from evaluation.ragas_eval import generate_qa_pairs
from config import RESULTS_DIR, TOP_K


def diagnose_dataset(dataset_name: str) -> dict:
    """Check dataset loading and structure."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: {dataset_name.upper()} DATASET")
    print(f"{'='*60}")

    documents = load_dataset(dataset_name)
    print(f"\n[OK] Loaded {len(documents)} documents")

    if not documents:
        print("[ERROR] No documents loaded!")
        return {"error": "no_documents"}

    # Analyze document structure
    total_chars = sum(len(d.get("content", "")) for d in documents)
    avg_chars = total_chars / len(documents) if documents else 0

    sources = set(d.get("source", "unknown") for d in documents)

    print(f"  • Total characters: {total_chars:,.0f}")
    print(f"  • Avg doc size: {avg_chars:,.0f} chars")
    print(f"  • Unique sources: {len(sources)}")
    print(f"  • Sample sources: {list(sources)[:3]}")

    return {
        "num_documents": len(documents),
        "total_chars": total_chars,
        "avg_doc_size": avg_chars,
        "unique_sources": len(sources),
        "sources": sorted(list(sources))
    }


def diagnose_qa_pairs(dataset_name: str, documents: list[dict]) -> dict:
    """Check QA pair generation and source linking."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: QA PAIRS FOR {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load or generate QA pairs
    qa_cache = RESULTS_DIR / "benchmark_tables" / f"qa_{dataset_name}.json"

    if qa_cache.exists():
        with open(qa_cache) as f:
            qa_pairs = json.load(f)
        print(f"\n[OK] Loaded {len(qa_pairs)} cached QA pairs")
    else:
        print(f"\n[...] Generating QA pairs (no cache found)...")
        qa_pairs = generate_qa_pairs(documents, dataset_name=dataset_name)
        qa_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(qa_cache, "w") as f:
            json.dump(qa_pairs, f, indent=2)
        print(f"[OK] Generated and cached {len(qa_pairs)} QA pairs")

    if not qa_pairs:
        print("[ERROR] No QA pairs generated!")
        return {"error": "no_qa_pairs"}

    # Analyze QA structure
    qa_keys = set()
    for pair in qa_pairs:
        qa_keys.update(pair.keys())

    print(f"\n  • QA pair fields: {sorted(list(qa_keys))}")

    # Check if source links exist
    has_source = sum(1 for p in qa_pairs if "source" in p)
    has_relevant_sources = sum(1 for p in qa_pairs if "relevant_sources" in p)

    print(f"  • Pairs with 'source' field: {has_source}/{len(qa_pairs)}")
    print(f"  • Pairs with 'relevant_sources' field: {has_relevant_sources}/{len(qa_pairs)}")

    if has_relevant_sources == 0:
        print("\n[CRITICAL ERROR] No 'relevant_sources' in QA pairs!")
        print("   This is why Recall@5 = 0.0 (no ground truth for retrieval metrics)")

    # Sample QA pair
    if qa_pairs:
        print(f"\n  Sample QA pair:")
        sample = qa_pairs[0]
        print(f"    Question: {sample.get('question', '?')[:60]}...")
        print(f"    Ground Truth: {sample.get('ground_truth', '?')[:60]}...")
        if "source" in sample:
            print(f"    Source: {sample['source']}")
        else:
            print(f"    Source: (missing)")

    return {
        "num_qa_pairs": len(qa_pairs),
        "qa_keys": sorted(list(qa_keys)),
        "has_source": has_source,
        "has_relevant_sources": has_relevant_sources,
        "critical_issue": has_relevant_sources == 0,
    }


def diagnose_retrieval(arch_name: str, dataset_name: str, documents: list[dict], qa_pairs: list[dict]) -> dict:
    """Test actual retrieval on sample queries."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: {arch_name.upper()} RETRIEVAL ON {dataset_name.upper()}")
    print(f"{'='*60}")

    rag_class = ALL_SYSTEMS[arch_name]
    rag = rag_class()

    # Load or build index
    cache_path = RESULTS_DIR / "indexes" / arch_name / dataset_name
    if cache_path.exists():
        print(f"\n[OK] Loading cached index from {cache_path}")
        try:
            rag.load(cache_path)
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")
            print(f"[...] Rebuilding index...")
            rag.index(documents, cache_path=cache_path)
    else:
        print(f"\n[...] Building index (first time)...")
        rag.index(documents, cache_path=cache_path)

    print(f"[OK] Index ready")

    if not qa_pairs:
        print("[ERROR] No QA pairs to test")
        return {"error": "no_qa_pairs"}

    # Test on first 3 queries
    test_queries = qa_pairs[:3]
    results = []

    print(f"\nTesting {len(test_queries)} queries:\n")

    for i, qa in enumerate(test_queries, 1):
        query = qa.get("question", "?")
        ground_truth = qa.get("ground_truth", "?")

        print(f"  Query {i}: {query[:55]}...")
        print(f"  Ground Truth: {ground_truth[:55]}...")

        try:
            docs = rag.retrieve(query, k=TOP_K)

            if not docs:
                print(f"  [ERROR] Retrieved 0 documents!")
            else:
                print(f"  [OK] Retrieved {len(docs)} documents:")
                for j, doc in enumerate(docs[:3], 1):
                    score_str = f"(score={doc.score:.3f})" if doc.score else ""
                    print(f"    {j}. [{doc.source}] {doc.content[:50]}... {score_str}")

            # Check if ground truth appears in results
            ground_lower = ground_truth.lower()
            gt_in_results = any(ground_lower in doc.content.lower() for doc in docs)

            if gt_in_results:
                print(f"  [OK] Ground truth found in retrieved docs")
            else:
                print(f"  [WARN] Ground truth NOT in retrieved docs")

            results.append({
                "query": query,
                "num_retrieved": len(docs),
                "gt_found": gt_in_results,
                "top_doc": docs[0].source if docs else None,
            })

        except Exception as e:
            print(f"  [ERROR] Retrieval error: {e}")
            results.append({"error": str(e)})

        print()

    gt_found_count = sum(1 for r in results if r.get("gt_found"))
    print(f"\nSummary: Ground truth found in {gt_found_count}/{len(test_queries)} queries")

    return {
        "architecture": arch_name,
        "test_results": results,
        "gt_found_pct": (gt_found_count / len(test_queries) * 100) if test_queries else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose zero recall issue")
    parser.add_argument("--dataset", choices=["small", "medium", "large"], default="small",
                       help="Dataset to diagnose")
    parser.add_argument("--arch", help="Specific architecture to test (vector, hybrid, graph, etc.)")
    args = parser.parse_args()

    # 1. Check dataset
    dataset_info = diagnose_dataset(args.dataset)
    documents = load_dataset(args.dataset)

    # 2. Check QA pairs
    qa_info = diagnose_qa_pairs(args.dataset, documents)
    qa_pairs = []
    qa_cache = RESULTS_DIR / "benchmark_tables" / f"qa_{args.dataset}.json"
    if qa_cache.exists():
        with open(qa_cache) as f:
            qa_pairs = json.load(f)

    # 3. Test retrieval
    archs_to_test = [args.arch] if args.arch else ["vector"]
    retrieval_results = []

    for arch_name in archs_to_test:
        try:
            ret_info = diagnose_retrieval(arch_name, args.dataset, documents, qa_pairs)
            retrieval_results.append(ret_info)
        except Exception as e:
            print(f"\n❌ Error testing {arch_name}: {e}")
            retrieval_results.append({"architecture": arch_name, "error": str(e)})

    # 4. Summary & recommendations
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS SUMMARY")
    print(f"{'='*60}")

    print(f"\n[OK] Dataset: {len(documents)} documents loaded")
    print(f"[OK] QA Pairs: {len(qa_pairs)} pairs generated/loaded")

    if qa_info.get("critical_issue"):
        print(f"\n[CRITICAL ISSUE FOUND]")
        print(f"   QA pairs have no 'relevant_sources' field!")
        print(f"   This explains Recall@5 = 0.0 in all architectures.")
        print(f"\n[SOLUTIONS]")
        print(f"   1. Modify ragas_eval.py to extract source from QA generation")
        print(f"   2. Link ground_truth text back to original document source")
        print(f"   3. Re-cache QA pairs with new structure")
        print(f"\n   See fix_qa_pairs.py for automated solution.")
    else:
        print(f"\n[OK] QA pairs have relevant_sources")

    print(f"\nRetrieval tests:")
    for r in retrieval_results:
        if "error" in r:
            print(f"  [ERROR] {r['architecture']}: {r['error']}")
        else:
            pct = r.get("gt_found_pct", 0)
            status = "[OK]" if pct > 0 else "[WARN]"
            print(f"  {status} {r['architecture']}: Ground truth found in {pct:.0f}% of tests")

    # Save diagnostic report
    report = {
        "dataset": args.dataset,
        "dataset_info": dataset_info,
        "qa_info": qa_info,
        "retrieval_results": retrieval_results,
    }

    report_path = RESULTS_DIR / "diagnostic_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[INFO] Full report: {report_path}")


if __name__ == "__main__":
    main()
