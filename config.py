"""
config.py — Central configuration for all benchmark parameters.
Every tunable value lives here. See decisions_log.md for WHY each value was chosen.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATASETS_RAW = ROOT / "datasets" / "raw"
DATASETS_PROCESSED = ROOT / "datasets" / "processed"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

# ─── LLM (D-001: Frozen across all architectures) ─────────────────────────────
# Generator: llama-3.1-8b-instant via Groq (fast, free 14K req/day) — D-016
# Judge:     llama-3.3-70b-versatile via Groq (stronger than generator) — D-002
LLM_MODEL = "llama-3.1-8b-instant"        # D-001: frozen generator
LLM_TEMPERATURE = 0                        # D-001: deterministic for reproducibility
LLM_MAX_TOKENS = 512
JUDGE_MODEL = "llama-3.1-8b-instant"      # D-002: same model, 500K TPD vs 100K for 70b
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ─── Embeddings (D-003) ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "bge-large"      # Local: BAAI/bge-large-en-v1.5 via sentence-transformers
EMBEDDING_DIMS = 1024              # bge-large-en-v1.5 output dimensions

# ─── Chunking (D-004) ─────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # D-004: tokens — balances semantic completeness vs granularity
CHUNK_OVERLAP = 50        # D-004: preserves boundary context

# Parent-Child specific (D-005)
CHILD_CHUNK_SIZE = 256    # D-005: small for embedding precision
PARENT_CHUNK_SIZE = 1024  # D-005: large for LLM context richness

# ─── Retrieval (D-006) ────────────────────────────────────────────────────────
TOP_K = 5                 # D-006: k=5 standard — balances recall vs context window
FAISS_INDEX_TYPE = "flat" # D-006: exact search for benchmark fairness (no approximation)

# ─── Hybrid RAG (D-007) ───────────────────────────────────────────────────────
HYBRID_ALPHA = 0.5        # D-007: 0.5 = equal weight BM25 + vector; tune per domain
                          # alpha=1.0 → pure vector; alpha=0.0 → pure BM25

# ─── Multi-Query RAG (D-008) ──────────────────────────────────────────────────
NUM_SUBQUERIES = 3        # D-008: 3 sub-queries; more improves recall but raises cost

# ─── Graph RAG (D-009) ────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"      # D-009: lightweight; swap to en_core_web_trf for quality
GRAPH_HOP_DEPTH = 2                  # D-009: 2-hop traversal captures indirect relationships
MIN_ENTITY_FREQ = 2                  # D-009: entities appearing < 2× are noise, not signal

# ─── Datasets (D-010) ─────────────────────────────────────────────────────────
DATASETS = {
    "small": {
        "name": "Wikipedia (50 articles)",
        "size": 50,
        "domain": "general knowledge",
        "source": "wikipedia-api",
    },
    "medium": {
        "name": "arXiv ML Papers (500)",
        "size": 500,
        "domain": "machine learning / NLP",
        "source": "arxiv-api",
    },
    "large": {
        "name": "Kubernetes Docs (~2400 pages)",
        "size": 2400,
        "domain": "technical documentation",
        "source": "github-kubernetes-website",
    },
}

# ─── Evaluation (D-011) ───────────────────────────────────────────────────────
QA_PAIRS_PER_DATASET = 20    # Gemini judge is fast — 20 pairs balances quality vs API cost
BENCHMARK_WARMUP_RUNS = 1    # 1 warmup discards cold-cache penalty
BENCHMARK_TIMED_RUNS = 5     # 5 timed runs give stable latency estimate

# ─── Cost tracking (D-012) ────────────────────────────────────────────────────
COST_PER_1K_EMBED_TOKENS = 0.00002    # text-embedding-3-small: $0.020/1M tokens (D-003)
COST_PER_1K_INPUT_TOKENS = 0.000075   # gemini-2.0-flash input:  $0.075/1M tokens (D-016)
COST_PER_1K_OUTPUT_TOKENS = 0.0003    # gemini-2.0-flash output: $0.300/1M tokens (D-016)

# ═══════════════════════════════════════════════════════════════════════════════
# ELITE UPGRADES (v1.1+) — Cost, GPU, Multi-Embedding Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Embedding model comparison (D-013) ───────────────────────────────────────
# All models defined in embedding_registry.py
# Run: python evaluation/embedding_benchmark.py --dataset small
EMBEDDING_MODELS_TO_BENCHMARK = [
    "gemini-embedding-001",  # D-013: default Gemini API — 3072-dim (baseline)
    "bge-base",        # D-013: free local, 768-dim, CPU-friendly
    "bge-large",       # D-013: free local, 1024-dim, best open-source English
    "e5-large",        # D-013: free local, 1024-dim, competitive on BEIR
]

# ─── GPU vs CPU (D-014) ───────────────────────────────────────────────────────
# GPU benchmark only applies to HuggingFace local models (bge-*, e5-*)
# OpenAI models are always API calls (no GPU path)
# Run: python evaluation/gpu_benchmark.py --dataset medium
GPU_BENCHMARK_MODELS = ["bge-large", "e5-large"]   # D-014: GPU speedup measured on these
GPU_BENCHMARK_MIN_CHUNKS = 1000    # D-014: GPU overhead not worth it below this chunk count

# ─── Cost analysis (D-015) ────────────────────────────────────────────────────
# Primary metric: Recall per dollar — answers "what quality can I afford?"
# Run: python evaluation/cost_benchmark.py --dataset small --breakdown
COST_EFFICIENCY_METRIC = "recall_per_dollar"   # D-015: primary cross-model comparison
COST_BUDGET_TIERS = {
    "budget":   0.001,   # < $0.001 / query
    "standard": 0.01,    # $0.001 – $0.01 / query
    "premium":  float("inf"),  # any cost
}
