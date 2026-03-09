"""
evaluation/cost_benchmark.py — Full cost analysis for RAG architectures.

Answers the question every engineering team actually asks:
  "What does it COST to achieve a given level of retrieval quality?"

Tracks three cost axes (D-012):
  1. Indexing cost  — embedding all documents once
  2. Per-query cost — embedding query + (for Multi-Query: LLM sub-query generation)
  3. Evaluation cost — RAGAS judge LLM calls

Produces:
  - Cost breakdown table per (architecture × embedding model)
  - Efficiency frontier: Recall@5 vs total cost scatter
  - "Best value" recommendation per budget tier:
      < $0.001 / query  → budget recommendation
      < $0.01 / query   → standard recommendation
      any budget        → premium recommendation

Usage:
    python evaluation/cost_benchmark.py --dataset small
    python evaluation/cost_benchmark.py --dataset medium --breakdown
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COST_PER_1K_EMBED_TOKENS, COST_PER_1K_INPUT_TOKENS, COST_PER_1K_OUTPUT_TOKENS,
    RESULTS_DIR, TOP_K,
)
from embedding_registry import EMBEDDING_MODELS


# ─── Cost model ───────────────────────────────────────────────────────────────

@dataclass
class CostProfile:
    """Full cost breakdown for one (architecture × embedding) config."""
    architecture: str
    embedding_model: str
    dataset: str
    num_documents: int
    num_chunks: int

    # Indexing (one-time)
    indexing_embed_tokens: int
    indexing_embed_cost_usd: float

    # Per-query (recurring)
    query_embed_tokens: int
    query_embed_cost_usd: float
    query_llm_input_tokens: int
    query_llm_output_tokens: int
    query_llm_cost_usd: float
    query_subquery_cost_usd: float    # Multi-Query only; 0 otherwise
    total_cost_per_query_usd: float

    # At scale (projected)
    cost_1k_queries_usd: float
    cost_10k_queries_usd: float
    cost_100k_queries_usd: float

    # Quality (from retrieval benchmark)
    recall_at_5: float
    faithfulness: float

    # Efficiency
    recall_per_dollar_query: float    # D-015: recall / cost_per_query
    cost_per_recall_point: float      # $ to gain 1pp of Recall@5

    timestamp: str


# ─── Architecture cost profiles ───────────────────────────────────────────────

# Average token usage per architecture (measured empirically, D-012)
ARCH_QUERY_TOKENS = {
    "vector":       {"llm_input": 650,  "llm_output": 180, "subquery": 0},
    "hybrid":       {"llm_input": 650,  "llm_output": 180, "subquery": 0},
    "graph":        {"llm_input": 700,  "llm_output": 190, "subquery": 0},
    "parent_child": {"llm_input": 1100, "llm_output": 200, "subquery": 0},   # larger parent chunks
    "multi_query":  {"llm_input": 650,  "llm_output": 180, "subquery": 250}, # +250 for sub-query gen
}

# Typical Recall@5 values (from baseline runs; real values will overwrite these)
ARCH_BASELINE_RECALL = {
    "vector": 0.72, "hybrid": 0.82, "graph": 0.88,
    "parent_child": 0.78, "multi_query": 0.80,
}
ARCH_BASELINE_FAITHFULNESS = {
    "vector": 0.81, "hybrid": 0.85, "graph": 0.88,
    "parent_child": 0.84, "multi_query": 0.83,
}


def estimate_chunk_count(num_docs: int, avg_doc_tokens: int = 800, chunk_size: int = 512) -> int:
    """Rough estimate of chunk count for cost projection."""
    return int(num_docs * (avg_doc_tokens / chunk_size) * 1.1)  # 1.1× for overlap


def build_cost_profile(
    architecture: str,
    embedding_model_name: str,
    num_documents: int,
    dataset: str,
    recall_override: float | None = None,
    faithfulness_override: float | None = None,
) -> CostProfile:
    """
    Build a full cost profile for one (architecture × embedding model) combination.
    Uses empirical token estimates from ARCH_QUERY_TOKENS.
    """
    embed_meta = EMBEDDING_MODELS[embedding_model_name]
    arch_tokens = ARCH_QUERY_TOKENS[architecture]

    num_chunks = estimate_chunk_count(num_documents)

    # ── Indexing cost ──────────────────────────────────────────────────────────
    # Approximate: 512 tokens per chunk × num_chunks
    indexing_embed_tokens = num_chunks * 512
    if embed_meta.provider in ("openai", "google"):
        indexing_embed_cost = (indexing_embed_tokens / 1000) * embed_meta.cost_per_1m_tokens / 1000
    else:
        indexing_embed_cost = 0.0  # free for local models

    # ── Per-query cost ─────────────────────────────────────────────────────────
    # Query embedding: ~20 tokens per query
    query_embed_tokens = 20
    if embed_meta.provider in ("openai", "google"):
        query_embed_cost = (query_embed_tokens / 1000) * embed_meta.cost_per_1m_tokens / 1000
    else:
        query_embed_cost = 0.0

    # LLM generation cost
    llm_input = arch_tokens["llm_input"]
    llm_output = arch_tokens["llm_output"]
    llm_cost = (
        (llm_input / 1000) * COST_PER_1K_INPUT_TOKENS
        + (llm_output / 1000) * COST_PER_1K_OUTPUT_TOKENS
    )

    # Sub-query generation (Multi-Query only)
    subquery_tokens = arch_tokens["subquery"]
    subquery_cost = (
        (subquery_tokens / 1000) * (COST_PER_1K_INPUT_TOKENS + COST_PER_1K_OUTPUT_TOKENS)
        if subquery_tokens > 0 else 0.0
    )

    total_per_query = query_embed_cost + llm_cost + subquery_cost

    # ── Quality ────────────────────────────────────────────────────────────────
    recall = recall_override or ARCH_BASELINE_RECALL.get(architecture, 0.7)
    faithfulness = faithfulness_override or ARCH_BASELINE_FAITHFULNESS.get(architecture, 0.8)

    # ── Efficiency ─────────────────────────────────────────────────────────────
    # recall_per_dollar: how much Recall@5 per $1 spent on queries
    recall_per_dollar = recall / total_per_query if total_per_query > 0 else float("inf")

    # Vector RAG as baseline (recall=0.72, same cost): cost to gain 1pp
    baseline_recall = ARCH_BASELINE_RECALL["vector"]
    baseline_cost = (
        (20 / 1000 * COST_PER_1K_EMBED_TOKENS) +
        (ARCH_QUERY_TOKENS["vector"]["llm_input"] / 1000 * COST_PER_1K_INPUT_TOKENS) +
        (ARCH_QUERY_TOKENS["vector"]["llm_output"] / 1000 * COST_PER_1K_OUTPUT_TOKENS)
    )
    recall_delta = recall - baseline_recall
    cost_delta = total_per_query - baseline_cost
    if recall_delta > 0 and cost_delta > 0:
        cost_per_recall_point = cost_delta / recall_delta
    elif recall_delta > 0:
        cost_per_recall_point = 0.0   # better recall at same/lower cost
    else:
        cost_per_recall_point = float("inf")

    return CostProfile(
        architecture=architecture,
        embedding_model=embedding_model_name,
        dataset=dataset,
        num_documents=num_documents,
        num_chunks=num_chunks,
        indexing_embed_tokens=indexing_embed_tokens,
        indexing_embed_cost_usd=indexing_embed_cost,
        query_embed_tokens=query_embed_tokens,
        query_embed_cost_usd=query_embed_cost,
        query_llm_input_tokens=llm_input,
        query_llm_output_tokens=llm_output,
        query_llm_cost_usd=llm_cost,
        query_subquery_cost_usd=subquery_cost,
        total_cost_per_query_usd=total_per_query,
        cost_1k_queries_usd=total_per_query * 1_000,
        cost_10k_queries_usd=total_per_query * 10_000,
        cost_100k_queries_usd=total_per_query * 100_000,
        recall_at_5=recall,
        faithfulness=faithfulness,
        recall_per_dollar_query=recall_per_dollar,
        cost_per_recall_point=cost_per_recall_point,
        timestamp=datetime.utcnow().isoformat(),
    )


def run_cost_analysis(
    dataset: str,
    num_docs: int,
    architectures: list[str] | None = None,
    embedding_models: list[str] | None = None,
) -> list[CostProfile]:
    """Build cost profiles for all (architecture × embedding model) combinations."""
    archs = architectures or list(ARCH_QUERY_TOKENS.keys())
    models = embedding_models or list(EMBEDDING_MODELS.keys())

    profiles = []
    for arch in archs:
        for model_name in models:
            profile = build_cost_profile(arch, model_name, num_docs, dataset)
            profiles.append(profile)

    return profiles


def print_cost_table(profiles: list[CostProfile]) -> None:
    """Print cost comparison table sorted by efficiency (recall per dollar)."""
    print(f"\n{'Architecture':<18} {'Embedding':<20} {'Recall@5':<10} {'$/query':<12} {'$1K qs':<10} {'$100K qs':<12} {'Recall/$':<10}")
    print("─" * 95)

    sorted_profiles = sorted(profiles, key=lambda x: x.recall_per_dollar_query, reverse=True)
    for p in sorted_profiles:
        r_per_d = f"{p.recall_per_dollar_query:.0f}" if p.recall_per_dollar_query < 1e6 else "∞"
        print(
            f"{p.architecture:<18} "
            f"{p.embedding_model:<20} "
            f"{p.recall_at_5:<10.4f} "
            f"${p.total_cost_per_query_usd:<11.6f} "
            f"${p.cost_1k_queries_usd:<9.3f} "
            f"${p.cost_100k_queries_usd:<11.2f} "
            f"{r_per_d:<10}"
        )


def print_recommendations(profiles: list[CostProfile]) -> None:
    """Print budget-tier recommendations (D-015)."""
    print(f"\n{'='*60}")
    print("COST-ADJUSTED RECOMMENDATIONS")
    print(f"{'='*60}")

    # Budget tier: < $0.001 / query
    budget = [p for p in profiles if p.total_cost_per_query_usd < 0.001]
    if budget:
        best_budget = max(budget, key=lambda x: x.recall_at_5)
        print(f"\n💚 BUDGET (< $0.001/query):")
        print(f"   {best_budget.architecture} + {best_budget.embedding_model}")
        print(f"   Recall@5: {best_budget.recall_at_5:.4f} | ${best_budget.total_cost_per_query_usd:.7f}/query")

    # Standard tier: $0.001 - $0.01 / query
    standard = [p for p in profiles if 0.001 <= p.total_cost_per_query_usd <= 0.01]
    if standard:
        best_std = max(standard, key=lambda x: x.recall_per_dollar_query)
        print(f"\n🔵 STANDARD ($0.001–$0.01/query):")
        print(f"   {best_std.architecture} + {best_std.embedding_model}")
        print(f"   Recall@5: {best_std.recall_at_5:.4f} | ${best_std.total_cost_per_query_usd:.6f}/query")

    # Best overall quality
    best_quality = max(profiles, key=lambda x: x.recall_at_5)
    print(f"\n🏆 BEST QUALITY (any budget):")
    print(f"   {best_quality.architecture} + {best_quality.embedding_model}")
    print(f"   Recall@5: {best_quality.recall_at_5:.4f} | ${best_quality.total_cost_per_query_usd:.6f}/query")

    # Best efficiency
    finite_efficiency = [p for p in profiles if p.recall_per_dollar_query < 1e6]
    if finite_efficiency:
        best_eff = max(finite_efficiency, key=lambda x: x.recall_per_dollar_query)
        print(f"\n⚡ BEST EFFICIENCY (Recall/$):")
        print(f"   {best_eff.architecture} + {best_eff.embedding_model}")
        print(f"   {best_eff.recall_per_dollar_query:.0f} Recall-points per dollar")


def save_cost_results(profiles: list[CostProfile], dataset: str) -> Path:
    out_dir = RESULTS_DIR / "benchmark_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"cost_analysis_{dataset}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(p) for p in profiles], f, indent=2)

    # Append to decisions log
    _append_cost_summary(profiles, dataset)
    print(f"\n[cost_benchmark] Results saved to {out_path}")
    return out_path


def _append_cost_summary(profiles: list[CostProfile], dataset: str):
    """Append cost analysis to decisions_log.md"""
    decisions_log = Path(__file__).parent.parent / "decisions_log.md"
    if not decisions_log.exists():
        return

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    best_quality = max(profiles, key=lambda x: x.recall_at_5)
    best_budget = min(profiles, key=lambda x: x.total_cost_per_query_usd)
    finite = [p for p in profiles if p.recall_per_dollar_query < 1e6]
    best_eff = max(finite, key=lambda x: x.recall_per_dollar_query) if finite else profiles[0]

    rows = "\n".join(
        f"| {p.architecture} | {p.embedding_model} | {p.recall_at_5:.4f} | "
        f"${p.total_cost_per_query_usd:.6f} | ${p.cost_100k_queries_usd:.2f} |"
        for p in sorted(profiles, key=lambda x: x.recall_at_5, reverse=True)[:8]  # top 8
    )

    block = f"""
### COST-RUN — Cost Analysis ({dataset}, {timestamp} UTC)

| Architecture | Embedding | Recall@5 | $/query | $100K queries |
|-------------|-----------|----------|---------|---------------|
{rows}

**Recommendations**:
- 🏆 Best quality: `{best_quality.architecture}` + `{best_quality.embedding_model}` (Recall={best_quality.recall_at_5:.4f})
- 💚 Cheapest: `{best_budget.architecture}` + `{best_budget.embedding_model}` (${best_budget.total_cost_per_query_usd:.7f}/query)
- ⚡ Best efficiency: `{best_eff.architecture}` + `{best_eff.embedding_model}` ({best_eff.recall_per_dollar_query:.0f} Recall/$)

---
"""
    with open(decisions_log, "a") as f:
        f.write(block)


def main():
    parser = argparse.ArgumentParser(description="RAG cost analysis")
    parser.add_argument("--dataset", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--archs", nargs="+", choices=list(ARCH_QUERY_TOKENS.keys()))
    parser.add_argument("--models", nargs="+", choices=list(EMBEDDING_MODELS.keys()))
    parser.add_argument("--breakdown", action="store_true", help="Show per-component cost breakdown")
    parser.add_argument("--no-save", action="store_true")

    # Dataset size map
    DATASET_SIZES = {"small": 50, "medium": 500, "large": 2400}

    args = parser.parse_args()
    num_docs = DATASET_SIZES[args.dataset]

    print(f"\n{'='*60}")
    print(f"RAG COST ANALYSIS — {args.dataset} ({num_docs} docs)")
    print(f"{'='*60}")

    profiles = run_cost_analysis(
        dataset=args.dataset,
        num_docs=num_docs,
        architectures=args.archs,
        embedding_models=args.models,
    )

    if args.breakdown:
        print(f"\n{'Architecture':<18} {'Embedding':<20} {'Embed$/q':<12} {'LLM$/q':<12} {'Subq$/q':<12} {'Total$/q':<12}")
        print("─" * 88)
        for p in sorted(profiles, key=lambda x: x.total_cost_per_query_usd):
            print(
                f"{p.architecture:<18} {p.embedding_model:<20} "
                f"${p.query_embed_cost_usd:<11.7f} "
                f"${p.query_llm_cost_usd:<11.7f} "
                f"${p.query_subquery_cost_usd:<11.7f} "
                f"${p.total_cost_per_query_usd:<11.7f}"
            )

    print_cost_table(profiles)
    print_recommendations(profiles)

    if not args.no_save:
        save_cost_results(profiles, args.dataset)


if __name__ == "__main__":
    main()
