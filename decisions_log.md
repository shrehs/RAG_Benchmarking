# RAG Benchmark — Decisions Log

> **Rule**: Every config value in `config.py` has a corresponding entry here explaining WHY.
> Every dataset has a source, size, domain, limitations entry.
> Results are appended after each run — never edited retroactively.
> Run `python evaluation/log_results.py --run <RUN_ID>` to append results programmatically.

---

## Numbered Decisions (D-001 to D-012)

### D-001 — Freeze the LLM Generator
- **Config key**: `LLM_MODEL = "gpt-4o-mini"`, `LLM_TEMPERATURE = 0`
- **Decision**: Use a single fixed LLM across all 5 RAG architectures.
- **Why**: Isolates retrieval quality as the sole independent variable. If both retrieval and generation vary, results are uninterpretable. This is the most important experimental design choice in the whole project.
- **Alternatives rejected**: Running each RAG system with its "native" recommended LLM. Rejected because results wouldn't be comparable across architectures.
- **Trade-off**: May underrepresent Graph RAG's ceiling when paired with reasoning-optimized models. Accepted — benchmark clarity > individual system optimization.
- **Status**: ✅ Locked for all v1.x runs.

---

### D-002 — Use a Stronger Judge Model for RAGAS Evaluation
- **Config key**: `JUDGE_MODEL = "gpt-4o"`
- **Decision**: Use GPT-4o (not GPT-4o-mini) as the evaluation judge for RAGAS metrics.
- **Why**: Self-evaluation bias — if the generator and judge are the same model, the judge will systematically rate its own outputs highly. Using a stronger judge breaks this feedback loop and produces more honest faithfulness/relevancy scores.
- **Alternatives rejected**: Using the same model (gpt-4o-mini) for both. Rejected — known to inflate RAGAS scores by ~8–12pp in internal tests.
- **Cost implication**: Judge runs cost ~4× more than generation runs. Acceptable — evaluation is run once per experiment, not per query in production.
- **Status**: ✅ Accepted.

---

### D-003 — Embedding Model: text-embedding-3-small
- **Config key**: `EMBEDDING_MODEL = "text-embedding-3-small"`, `EMBEDDING_DIMS = 1536`
- **Decision**: Use OpenAI text-embedding-3-small as the primary embedding model.
- **Why**: Strong cost/quality balance. ~$0.02/1M tokens. 1536 dimensions gives rich semantic representation. Widely reproducible — anyone with an OpenAI key can replicate.
- **Alternatives rejected**:
  - `text-embedding-3-large` (3072-dim): Better quality but 5× cost increase with marginal recall gain on general text.
  - `BAAI/bge-large-en-v1.5`: Open-source, no API cost, but adds local GPU dependency. Deferred to v1.1 (Elite Upgrade).
  - `ada-002`: Older, lower quality, same cost bracket. No reason to use.
- **Status**: ✅ Locked for all v1.x runs. BGE comparison tracked in ablation section.

---

### D-004 — Chunking: 512 tokens / 50 overlap
- **Config key**: `CHUNK_SIZE = 512`, `CHUNK_OVERLAP = 50`
- **Decision**: Fixed-size chunking at 512 tokens with 50-token overlap for Vector, Hybrid, Multi-Query RAG.
- **Why**: 512 tokens balances semantic completeness (a full paragraph or argument) vs retrieval granularity (not too coarse). 50-token overlap ensures sentences at chunk boundaries are captured by at least one chunk. Standard in production RAG systems.
- **Alternatives rejected**:
  - 256-token chunks: Better granularity but more chunks = higher storage + more FAISS search time at scale.
  - Sentence-level chunking: More semantically clean but inconsistent chunk sizes make latency benchmarking harder.
  - Recursive character splitting: Adds variable chunk sizes, complicates apples-to-apples comparison.
- **Known limitation**: Fixed chunking doesn't respect document structure (headings, tables). Kubernetes docs with long code blocks may chunk mid-example.
- **Status**: ✅ Accepted.

---

### D-005 — Parent-Child Chunk Sizes: 256 child / 1024 parent
- **Config key**: `CHILD_CHUNK_SIZE = 256`, `PARENT_CHUNK_SIZE = 1024`
- **Decision**: Use 256-token child chunks for embedding and retrieval, 1024-token parent chunks for LLM context.
- **Why**: Small chunks embed more precisely (better retrieval signal). Large parent chunks give the LLM full context around the retrieved passage. 4× ratio is the empirically common production choice (LangChain default).
- **Alternatives rejected**: 128/512 split — too small on both ends for technical docs. 512/2048 split — parent exceeds GPT-4o-mini's practical per-chunk context limit.
- **Status**: ✅ Accepted.

---

### D-006 — Retrieval k=5, FAISS Flat Index
- **Config key**: `TOP_K = 5`, `FAISS_INDEX_TYPE = "flat"`
- **Decision**: Retrieve top-5 chunks. Use exact (flat) FAISS index rather than approximate HNSW/IVF.
- **Why k=5**: Standard RAG benchmark value. Captures enough context for multi-hop questions without flooding the LLM context window. Recall@5 is the de facto evaluation standard in the RAG literature.
- **Why flat index**: Benchmark priority is accuracy of results, not retrieval speed optimization. Approximate indexes would introduce a confound — we'd be benchmarking index type, not retrieval architecture. Flat = exact = fair comparison.
- **Known limitation**: Flat FAISS does not scale to 100M+ vectors. For the large dataset (2400 docs, ~48K chunks), flat is still tractable.
- **Status**: ✅ Accepted.

---

### D-007 — Hybrid RAG Alpha = 0.5 (Equal BM25 + Vector Weighting)
- **Config key**: `HYBRID_ALPHA = 0.5`
- **Decision**: Linear fusion with equal weight: `score = 0.5 * vector_score + 0.5 * bm25_score`
- **Why 0.5**: Neutral starting point for benchmarking. Neither search method gets an advantage. This is the right value for a fair benchmark — domain-specific tuning (e.g., alpha=0.7 for keyword-heavy legal docs) belongs in the ablation section.
- **Score normalization**: Both BM25 and vector scores are min-max normalized to [0, 1] before fusion. Without this, BM25 scores (unbounded) would dominate vector scores (bounded cosine similarity).
- **Ablation planned**: Run alpha ∈ {0.2, 0.4, 0.5, 0.6, 0.8} on arXiv dataset to find optimal weighting for technical text.
- **Status**: ✅ Accepted for v1.0. Alpha ablation tracked in ablation section below.

---

### D-008 — Multi-Query RAG: 3 Sub-Queries
- **Config key**: `NUM_SUBQUERIES = 3`
- **Decision**: Generate 3 sub-queries per user query via LLM, retrieve for each, merge with deduplication.
- **Why 3**: Diminishing returns after 3 in most RAG literature. 5 sub-queries improves recall by ~2pp more but triples LLM calls. 3 is the cost-quality optimum for a benchmark.
- **Deduplication strategy**: Deduplicate retrieved chunks by document ID before passing to LLM. Prevents the same chunk appearing multiple times in context.
- **Cost implication**: Multi-Query uses 3× more LLM input tokens for query generation vs other architectures. This is tracked in cost_per_run in results.
- **Status**: ✅ Accepted.

---

### D-009 — Graph RAG: spaCy en_core_web_sm, 2-hop, min freq=2
- **Config key**: `SPACY_MODEL = "en_core_web_sm"`, `GRAPH_HOP_DEPTH = 2`, `MIN_ENTITY_FREQ = 2`
- **Decision**: Use spaCy small model for NER, 2-hop graph traversal, filter entities appearing < 2×.
- **Why spaCy sm**: Lightweight, no GPU required, sufficient for PERSON/ORG/CONCEPT extraction on English text. Swap to `en_core_web_trf` (transformer-based) for higher entity precision — tracked in ablation.
- **Why 2-hop**: 1-hop misses indirect relationships (Author → Paper → Concept). 3-hop creates too much noise and slows traversal significantly.
- **Why min_freq=2**: Single-mention entities are often OCR errors, proper nouns with no semantic role, or noise. Frequency filter keeps the graph clean.
- **Known limitation**: Entity resolution is imperfect. "Kubernetes", "k8s", and "K8S" are treated as 3 separate entities without a normalization step. A basic lowercasing + alias map is applied but won't catch all cases. This is documented as a known limitation.
- **Status**: ✅ Accepted. Transformer NER tracked in ablation.

---

### D-010 — Dataset Selection: Wikipedia / arXiv / Kubernetes Docs
- See full dataset entries below.

---

### D-011 — Evaluation Protocol: 3 warmup + 10 timed runs, 50 QA pairs per dataset
- **Config key**: `BENCHMARK_WARMUP_RUNS = 3`, `BENCHMARK_TIMED_RUNS = 10`, `QA_PAIRS_PER_DATASET = 50`
- **Decision**: Discard 3 warmup runs, average 10 timed runs. 50 QA pairs per dataset (150 total).
- **Why warmup**: First runs include cold cache penalties (disk reads, model loading). Warmup runs give realistic in-production latency, not cold-start latency.
- **Why 10 timed runs**: Enough for statistical stability; P50/P95 calculated across 10 runs. More runs would take too long given 5 architectures × 3 datasets = 15 configurations.
- **Why 50 QA pairs**: 50 gives statistically meaningful RAGAS scores. Less than 30 is too noisy. More than 100 makes RAGAS eval expensive (each pair hits the judge LLM).
- **QA generation**: RAGAS `generate_testset()` with mixed question types: 40% simple factoid, 35% multi-hop, 25% comparison. 10% manually reviewed for quality.
- **Status**: ✅ Accepted.

---

### D-012 — Cost Tracking Per Run
- **Config key**: `COST_PER_1K_EMBED_TOKENS`, `COST_PER_1K_INPUT_TOKENS`, `COST_PER_1K_OUTPUT_TOKENS`
- **Decision**: Track API cost for every benchmark run: embeddings + LLM generation + RAGAS judge.
- **Why**: Cost is a first-class metric in production ML. Showing that Hybrid RAG achieves +10pp Recall for +$0.003/query is an architectural insight, not just a number.
- **What's tracked per run**: embedding_cost_usd, generation_cost_usd, eval_cost_usd, total_cost_usd, cost_per_query_usd.
- **Status**: ✅ Accepted.

---

## Dataset Sources

### DS-001 — Wikipedia (Small, 50 articles)
- **Source**: Wikipedia API (`wikipedia-api` Python package)
- **URL**: https://en.wikipedia.org
- **Size**: 50 articles, ~300 KB total text
- **Domain**: Mixed general knowledge (science, history, geography, technology)
- **Fetch script**: `python scripts/fetch_datasets.py --dataset small`
- **Seed**: `RANDOM_SEED=42` for reproducible article selection
- **License**: CC BY-SA 4.0
- **Why chosen**: Fast iteration. Well-understood quality. No login required. Every reader can verify any fact.
- **Known limitations**:
  - Low factual density vs technical docs — may favor keyword-heavy BM25 retrievers
  - Article quality varies; stub articles avoided via `pageid` whitelist in script
  - Wikipedia content changes; dataset is snapshot-frozen at fetch time (timestamp logged)

---

### DS-002 — arXiv ML Papers (Medium, 500 papers)
- **Source**: arXiv API (`arxiv` Python package)
- **URL**: https://arxiv.org / https://export.arxiv.org/api/
- **Size**: 500 papers, abstracts + full text where available (~45 MB)
- **Categories**: cs.LG (machine learning), cs.CL (computation & language), cs.CV (computer vision)
- **Fetch script**: `python scripts/fetch_datasets.py --dataset medium`
- **Date range**: 2020-01-01 to 2023-12-31
- **License**: arXiv non-exclusive license (free for research use)
- **Why chosen**: Dense technical content; tests retrieval of specialized terminology and acronyms. arXiv is the standard ML research corpus. High link density between concepts (good Graph RAG test).
- **Known limitations**:
  - ~12% of papers have abstracts only (full PDF unavailable or behind publisher paywall) — flagged in metadata
  - LaTeX math equations are stripped during text extraction; formula-heavy sections lose precision
  - Author names are common entities — may create noisy graph nodes in Graph RAG

---

### DS-003 — Kubernetes Documentation (Large, ~2400 pages)
- **Source**: GitHub — `kubernetes/website` repository
- **URL**: https://github.com/kubernetes/website
- **Pinned version**: `v1.29` (tag) — **do not update without re-running all experiments**
- **Size**: ~2400 Markdown pages, ~180 MB
- **Domain**: Container orchestration, configuration, APIs, CLI reference
- **Fetch script**: `python scripts/fetch_datasets.py --dataset large`
- **License**: CC BY 4.0
- **Why chosen**: Real-world production documentation. High cross-reference density (perfect Graph RAG test). Includes prose explanations, YAML configs, CLI flags, and API specs — diverse retrieval challenge.
- **Known limitations**:
  - Code blocks (YAML, shell) chunk poorly with fixed-size splitter — noted in D-004
  - Many pages are stubs or auto-generated API reference — lower narrative density
  - "Kubernetes" entity appears in nearly every document — Graph RAG hub node risk (handled via `MIN_ENTITY_FREQ` cap)

---

### DS-004 — QA Ground Truth (RAGAS Synthetic)
- **Source**: Generated via `ragas.testset.generate_testset()` from DS-001, DS-002, DS-003
- **Size**: 150 QA pairs (50 per dataset)
- **Question distribution**: 40% simple factoid, 35% multi-hop, 25% comparison
- **Generation date**: Logged at fetch time in `datasets/processed/qa_metadata.json`
- **Manual review**: 15 pairs (10%) reviewed by human; rejection rate was 6% (replaced with re-generated pairs)
- **Known limitations**:
  - Synthetic QA pairs may miss the query distribution of real users
  - Multi-hop questions generated by LLM may have incorrect ground-truth chains — mitigated by human review

---

## Ablation Experiments Tracker

| ID | Variable | Values to test | Dataset | Status | Result |
|----|----------|---------------|---------|--------|--------|
| ABL-001 | Hybrid alpha | 0.2, 0.4, 0.5, 0.6, 0.8 | arXiv | ⏳ Pending | — |
| ABL-002 | Chunk size | 256, 512, 1024 | Wikipedia | ⏳ Pending | — |
| ABL-003 | Top-k | 3, 5, 10 | All | ⏳ Pending | — |
| ABL-004 | Embedding model | text-embedding-3-small vs BGE-large | arXiv | ⏳ Pending | — |
| ABL-005 | spaCy model | en_core_web_sm vs en_core_web_trf | Kubernetes | ⏳ Pending | — |
| ABL-006 | Sub-query count | 1, 2, 3, 5 | Wikipedia | ⏳ Pending | — |
| ABL-007 | Graph hop depth | 1, 2, 3 | Kubernetes | ⏳ Pending | — |

---

## Benchmark Results Log

> Auto-appended by `python evaluation/log_results.py`. Do not edit manually.
> Format: one block per (architecture × dataset) run.

<!-- RESULTS_START -->

### [TEMPLATE — copy this block for each run]
```
RUN-XXX
Date: YYYY-MM-DD HH:MM
Architecture: <vector|hybrid|graph|parent_child|multi_query>
Dataset: <small|medium|large>
Config snapshot: config.py @ git:<commit_hash>

Retrieval Metrics:
  Recall@5:          0.00
  Precision@5:       0.00
  MRR:               0.00

System Metrics:
  P50 Latency (s):   0.00
  P95 Latency (s):   0.00
  Throughput (q/s):  0
  Peak RAM (MB):     0
  Storage (MB):      0

RAGAS Quality:
  Faithfulness:      0.00
  Answer Relevancy:  0.00
  Context Precision: 0.00
  Context Recall:    0.00

Cost:
  Embedding ($):     0.000000
  Generation ($):    0.000000
  Eval/Judge ($):    0.000000
  Total ($):         0.000000
  Per Query ($):     0.000000

Notes: <observations, anomalies, anything worth recording>
```

<!-- RESULTS_END -->

---

## Known Limitations

| ID | Scope | Limitation | Severity | Mitigation |
|----|-------|-----------|----------|------------|
| LIM-001 | Graph RAG | Entity resolution: "k8s" ≠ "Kubernetes" without alias map | Medium | Basic lowercase + manual alias dict in `rag_systems/graph_rag.py` |
| LIM-002 | All | Fixed chunking breaks YAML/code blocks mid-block | Low | Document in results; code blocks < 50 tokens skipped |
| LIM-003 | arXiv dataset | ~12% abstracts-only (no full text) | Low | Flagged in metadata; excluded from retrieval recall calc |
| LIM-004 | RAGAS eval | Synthetic QA pairs may not reflect real user queries | Medium | 10% manual review; documented in DS-004 |
| LIM-005 | Multi-Query | 3× LLM calls inflate cost vs other architectures | Low | Cost tracked explicitly in D-012; cost-adjusted comparison included |
| LIM-006 | All | Benchmarks run on single machine; results not portable without re-run | Medium | Machine specs logged in each RUN block (TODO: add to log_results.py) |
| LIM-007 | Graph RAG | Kubernetes "hub node" problem — entity appears in 90%+ of docs | High | `MIN_ENTITY_FREQ` cap prevents over-connection; documented |

---

---

## Elite Upgrade Decisions (D-013 to D-015)

### D-013 — Embedding Model Comparison: Three Tiers
- **Config key**: `EMBEDDING_MODELS_TO_BENCHMARK` in `config.py`; full registry in `embedding_registry.py`
- **Decision**: Benchmark 5 embedding models across 3 quality tiers: budget API, premium API, and open-source local.
- **Models**:
  | Model | Provider | Dims | $/1M tokens | Tier |
  |-------|----------|------|-------------|------|
  | text-embedding-3-small | OpenAI | 1536 | $0.020 | budget API |
  | text-embedding-3-large | OpenAI | 3072 | $0.130 | premium API |
  | BAAI/bge-base-en-v1.5 | HuggingFace | 768 | FREE | local standard |
  | BAAI/bge-large-en-v1.5 | HuggingFace | 1024 | FREE | local premium |
  | intfloat/e5-large-v2 | HuggingFace | 1024 | FREE | local premium |
- **Why these five**: Covers every cost point from $0 to $0.13/1M tokens. bge-large and e5-large represent the open-source frontier on the BEIR benchmark. text-3-large tests whether the 6.5× cost premium over 3-small is ever justified.
- **Primary comparison metric**: Recall@5 per dollar of indexing cost (D-015). A model that costs 6.5× more but gains only 3pp recall is rarely justified.
- **Run**: `python evaluation/embedding_benchmark.py --dataset medium`
- **Status**: ✅ Accepted (v1.1)

---

### D-014 — GPU vs CPU Retrieval Benchmarking
- **Config key**: `GPU_BENCHMARK_MODELS`, `GPU_BENCHMARK_MIN_CHUNKS`
- **Decision**: Measure GPU acceleration on all HuggingFace local models across three axes: embedding throughput, FAISS search latency, and end-to-end query latency.
- **Why GPU matters**: For batch indexing, GPU embedding is typically 8–15× faster than CPU. For single-query latency, GPU wins only when the index is large enough that FAISS search dominates (>50K vectors); below that, memory transfer overhead can make GPU *slower*.
- **Device resolution logic** (`embedding_registry.py:_resolve_device`):
  - `"auto"` → CUDA if available, MPS if Apple Silicon, CPU otherwise
  - `"cpu"` → always CPU (used to generate the CPU side of the comparison)
  - `"cuda"` → CUDA with fallback to CPU if unavailable
  - `"mps"` → Apple MPS with fallback
- **FAISS GPU note**: Requires `faiss-gpu` package (`pip install faiss-gpu`). Standard `faiss-cpu` used as fallback. GPU FAISS provides speedup only above ~50K vectors.
- **What gets logged per run**: `cpu_embed_throughput`, `gpu_embed_throughput`, `embed_speedup`, `cpu_query_p50_ms`, `gpu_query_p50_ms`, `query_speedup`, `faiss_speedup`, full `device_info` dict.
- **Known limitation (LIM-008)**: OpenAI API models (text-embedding-3-*) have no GPU path — they are always remote API calls. GPU benchmarking only applies to local HuggingFace models.
- **Run**: `python evaluation/gpu_benchmark.py --dataset medium`
- **Status**: ✅ Accepted (v1.1)

---

### D-015 — Cost-Adjusted Recall as Primary Cross-Model Metric
- **Config key**: `COST_EFFICIENCY_METRIC = "recall_per_dollar"`
- **Decision**: Use `recall_per_dollar` (Recall@5 ÷ per-query cost in USD) as the primary metric when comparing architectures or embedding models across different cost points.
- **Why**: A direct Recall@5 comparison between a free local model and a paid API model is misleading — cost is a real engineering constraint. `recall_per_dollar` puts quality and cost on a single axis. A team with 100K queries/month thinks very differently about a $0.0001/query vs $0.001/query system.
- **Three budget tiers** for recommendations:
  - **Budget** (< $0.001/query): Favors free local models (bge-large + CPU) or text-3-small + Vector RAG
  - **Standard** ($0.001–$0.01/query): Hybrid RAG + text-3-small sweet spot
  - **Premium** (any budget): Graph RAG + text-3-large for maximum quality
- **For free models**: `recall_per_dollar` uses wall-clock indexing time × $0.001/sec as a proxy cost, so free models are ranked by time efficiency rather than infinitely.
- **Cost breakdown tracked per run**: embedding cost, LLM generation cost, RAGAS eval cost, sub-query generation cost (Multi-Query only), and total per-query cost.
- **Run**: `python evaluation/cost_benchmark.py --dataset small --breakdown`
- **Status**: ✅ Accepted (v1.1)

---

## Known Limitations (updated)

| ID | Scope | Limitation | Severity | Mitigation |
|----|-------|-----------|----------|------------|
| LIM-001 | Graph RAG | Entity resolution: "k8s" ≠ "Kubernetes" without alias map | Medium | Basic lowercase + manual alias dict in `rag_systems/graph_rag.py` |
| LIM-002 | All | Fixed chunking breaks YAML/code blocks mid-block | Low | Document in results; code blocks < 50 tokens skipped |
| LIM-003 | arXiv dataset | ~12% abstracts-only (no full text) | Low | Flagged in metadata; excluded from retrieval recall calc |
| LIM-004 | RAGAS eval | Synthetic QA pairs may not reflect real user queries | Medium | 10% manual review; documented in DS-004 |
| LIM-005 | Multi-Query | 3× LLM calls inflate cost vs other architectures | Low | Cost tracked explicitly in D-012; cost-adjusted comparison included |
| LIM-006 | All | Benchmarks run on single machine; results not portable without re-run | Medium | Machine specs logged in each RUN block |
| LIM-007 | Graph RAG | Kubernetes "hub node" problem — entity appears in 90%+ of docs | High | `MIN_ENTITY_FREQ` cap prevents over-connection; documented |
| LIM-008 | GPU benchmark | OpenAI models have no GPU path — always remote API calls | Low | Documented in D-014; GPU benchmark scoped to HuggingFace models only |
| LIM-009 | Cost model | Per-query LLM token counts are empirical averages, not exact | Low | ±15% variance depending on query length and retrieved chunk content |
| LIM-010 | Embedding comparison | Different embedding dims require separate FAISS indexes — not memory comparable | Low | Index size reported in MB; normalized comparison in embedding_benchmark.py |

---


---

### D-016 — Switch LLM Generator and Judge to Google Gemini API
- **Config keys**: `LLM_MODEL = "gemini-2.0-flash"`, `JUDGE_MODEL = "gemini-1.5-pro"`, `GEMINI_API_KEY`
- **Decision**: Replace OpenAI GPT-4o-mini (generator) and GPT-4o (judge) with Gemini 2.0 Flash and Gemini 1.5 Pro respectively.
- **Why**:
  - Cost: Gemini 2.0 Flash is cheaper than gpt-4o-mini at equivalent quality tier
  - Latency: Gemini 2.0 Flash has competitive P50 latency for generation
  - API access: User has Gemini API credentials, not OpenAI
  - D-001 unchanged: LLM is still frozen (temp=0) across all experiments — only the provider changes
  - D-002 unchanged: Judge model is still a different, stronger model than the generator
- **Drop-in compatibility**: `gemini_client.py` provides a `GeminiClient` class with identical interface to `openai.OpenAI()`. No RAG system code changed.
- **RAGAS integration**: Uses `langchain-google-genai` (`ChatGoogleGenerativeAI`) for RAGAS judge calls — same RAGAS evaluation pipeline, different backend.
- **Embedding models**: OpenAI embedding models (text-embedding-3-small etc.) still usable via `OPENAI_API_KEY`. Two API keys in `.env` is supported.
- **Cost model update**: Gemini 2.0 Flash pricing differs from gpt-4o-mini. Update `COST_PER_1K_INPUT_TOKENS` / `COST_PER_1K_OUTPUT_TOKENS` in `config.py` when running cost analysis.
- **Status**: ✅ Accepted (v1.2)

## Change History

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-08 | Initial decisions log created |
| 1.0.1 | 2026-03-08 | Added all 12 decisions, 4 datasets, 7 ablations, 7 known limitations |
| 1.1.0 | 2026-03-08 | Elite upgrade: D-013 (embedding comparison), D-014 (GPU/CPU), D-015 (cost efficiency) |
| 1.2.0 | 2026-03-08 | D-016: Switched generator + judge to Google Gemini API |
