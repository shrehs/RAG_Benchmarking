# RAG Architecture Research & Comparison Guide

Comprehensive technical deep-dive into 5 RAG architectures with diagrams, algorithms, and implementation details.

---

## Table of Contents

1. [Fundamentals](#fundamentals)
2. [Vector RAG](#vector-rag)
3. [Hybrid RAG](#hybrid-rag)
4. [Graph RAG](#graph-rag)
5. [Parent-Child RAG](#parent-child-rag)
6. [Multi-Query RAG](#multi-query-rag)
7. [Comparison Matrix](#comparison-matrix)

---

## Fundamentals

### What is RAG?

RAG (Retrieval-Augmented Generation) augments a language model with external knowledge:

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  User Query                                                   │
│      │                                                        │
│      ├──► [RETRIEVAL]  ──► Find relevant documents           │
│      │                                                        │
│      └──► [GENERATION] ──► LLM reads docs + generates answer │
│                                                               │
│  Why RAG beats fine-tuning:                                  │
│  ✓ No model retraining needed                                │
│  ✓ Knowledge updates automatically                           │
│  ✓ Interpretable (can see source documents)                  │
│  ✓ Low hallucination (grounded in sources)                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                    RAG System Components                      │
└──────────────────────────────────────────────────────────────┘

1. DOCUMENTS (Knowledge Source)
   ┌─────────────────────────┐
   │ Wikipedia articles      │
   │ Company docs            │
   │ Code repositories       │
   │ Research papers         │
   └─────────────────────────┘
           │
           ├──► Split into chunks (512 tokens each)
           │
           ▼

2. INDEXING (Offline)
   ┌─────────────────────────┐
   │ Embed chunks            │
   │ Build search index      │
   │ Cache for fast lookup   │
   └─────────────────────────┘
           │
           ├──► FAISS / Neo4j / BM25 index
           │
           ▼

3. RETRIEVAL (Query Time)
   ┌─────────────────────────┐
   │ Decode question         │
   │ Search index            │
   │ Return top-K documents  │
   └─────────────────────────┘
           │
           ├──► Different strategies per architecture
           │
           ▼

4. GENERATION (Query Time)
   ┌─────────────────────────┐
   │ Build prompt            │
   │ Insert retrieved docs   │
   │ LLM generates answer    │
   └─────────────────────────┘
```

---

## Vector RAG

### Overview

**Vector RAG** uses semantic similarity to find relevant documents. Query and documents are embedded into a high-dimensional space; closest vectors = most similar meaning.

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      VECTOR RAG PIPELINE                       │
└────────────────────────────────────────────────────────────────┘

OFFLINE (Indexing):
┌──────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Documents   │──────│  Embeddings  │──────│  FAISS Index    │
│  (50 chunks) │      │  (1024 dims) │      │  (50 vectors)   │
└──────────────┘      └──────────────┘      └─────────────────┘
                                                     ▲
                                                     │
                         Hugging Face Embedder       │
                    BAAI/bge-large-en-v1.5 ─────────┘


ONLINE (Query Time):
┌──────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Question   │──────│  Embedding   │──────│  FAISS Search   │
│  "What is    │      │  (1024 dims) │      │  (cosine sim)   │
│   cloud?"    │      └──────────────┘      └─────────────────┘
└──────────────┘                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │  Top-5 Chunks   │
                                            │  (sorted by      │
                                            │   similarity)    │
                                            └─────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │  LLM Prompt     │
                                            │  + Documents    │
                                            │  = Answer       │
                                            └─────────────────┘
```

### Algorithm

```python
# INDEXING PHASE
def index_documents(documents):
    chunks = chunk_text(documents, chunk_size=512, overlap=50)
    embeddings = encoder.encode(chunks)  # Dense vectors
    faiss_index.add(embeddings)
    faiss_index.save("vector_index.faiss")

# RETRIEVAL PHASE
def retrieve(query):
    query_embedding = encoder.encode(query)
    distances, indices = faiss_index.search(query_embedding, k=5)
    return [chunks[i] for i in indices]

# GENERATION PHASE
def generate(query, retrieval_results):
    prompt = f"""Given context: {retrieval_results}
                Query: {query}
                Answer:"""
    answer = llm.generate(prompt)
    return answer
```

### Characteristics

| Aspect | Details |
|--------|---------|
| **Recall** | 1.50 (good) |
| **Latency** | 75ms (fast) |
| **Complexity** | Low |
| **Setup Time** | ~10 seconds |
| **RAM Usage** | ~1.6 GB |
| **Dependencies** | FAISS, sentence-transformers |

### Pros & Cons

✅ **Pros**:
- Simple to implement
- Fast (~75ms per query)
- No complex preprocessing
- Works well on general knowledge
- Easy to scale horizontally

❌ **Cons**:
- No semantic structure (just vectors)
- Can miss lexically different relevant docs
- Embedding quality dependent on model
- No reasoning about relationships

### When to Use Vector RAG

✅ Use when:
- You need production-ready RAG
- Query latency < 200ms required
- Documents are English prose
- Limited engineering resources

❌ Skip when:
- Documents are highly structured (code, tables)
- Semantic similarity insufficient
- Entity relationships matter

### Example Implementation

```python
from rag_systems import VectorRAG

# Initialize
rag = VectorRAG()

# Index documents (one-time)
documents = load_documents("wikipedia.json")
rag.index(documents, cache_path="indexes/vector/small")

# Query
query = "What is cloud computing?"
result = rag.query(query, k=5)

print(f"Answer: {result.answer}")
print(f"Latency: {result.latency_ms}ms")
print(f"Top document: {result.documents[0].source}")
```

---

## Hybrid RAG

### Overview

**Hybrid RAG** combines two complementary search strategies:
- **BM25**: Lexical search (exact keywords match)
- **Vector**: Semantic search (meaning match)

Query is scored by BOTH methods; results combined (alpha = 0.5 = equal weight).

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                     HYBRID RAG PIPELINE                        │
└────────────────────────────────────────────────────────────────┘

OFFLINE (Indexing):
┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
│  Documents   │──────┤  BM25 Index  │      │  FAISS Vector    │
│  (50 chunks) │      │  (inverted   │      │  Index           │
│              │      │   term map)  │      │  (embeddings)    │
└──────────────┘      └──────────────┘      └──────────────────┘


ONLINE (Query Time):

Question: "What is cloud computing?"
    │
    ├─────────────────┬────────────────────┐
    │                 │                    │
    ▼                 ▼                    ▼

BM25 SEARCH         VECTOR SEARCH      FUSION
┌──────────────┐    ┌──────────────┐   ┌──────────────┐
│ Lexical match│    │ Semantic sim │   │ Score = α×BM25
│ "cloud"      │    │ Embeddings   │   │      + (1-α)×Vec
│ "computing"  │    │ cosine sim   │   │ α = 0.5      │
│              │    │              │   │ (equal weight)
│ Scores:      │    │ Scores:      │   │              │
│ Doc1: 0.8    │    │ Doc1: 0.9    │   │ Top-5 re-rank│
│ Doc2: 0.6    │    │ Doc2: 0.5    │   │              │
│ Doc3: 0.4    │    │ Doc3: 0.7    │   │ Final ranking│
└──────────────┘    └──────────────┘   └──────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  Top-5 Results   │
                                    │  (both lexical & │
                                    │   semantic)      │
                                    └──────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  LLM Generation  │
                                    │  (with context)  │
                                    └──────────────────┘
```

### Algorithm

```python
# INDEXING
def index_documents(documents):
    chunks = chunk_text(documents, chunk_size=512)

    # BM25 path
    bm25_index = BM25Okapi([chunk.split() for chunk in chunks])

    # Vector path
    embeddings = encoder.encode(chunks)
    faiss_index.add(embeddings)

    return {bm25: bm25_index, faiss: faiss_index, chunks: chunks}

# RETRIEVAL (Dual scoring)
def retrieve(query, alpha=0.5):
    # BM25 scores
    bm25_scores = bm25_index.get_scores(query.split())

    # Vector scores
    query_emb = encoder.encode(query)
    distances, indices = faiss_index.search(query_emb, k=50)
    vector_scores = 1 / (1 + distances)  # normalize to [0, 1]

    # Fusion: Linear combination
    final_scores = {}
    for i, chunk_id in enumerate(indices):
        score = alpha * bm25_scores[chunk_id] + \
                (1 - alpha) * vector_scores[i]
        final_scores[chunk_id] = score

    # Return top-5
    top_5 = sorted(final_scores.items(),
                   key=lambda x: x[1], reverse=True)[:5]
    return [chunks[i] for i, _ in top_5]
```

### Characteristics

| Aspect | Details |
|--------|---------|
| **Recall** | 1.80 (best among all) |
| **Latency** | 188ms (slower) |
| **Complexity** | Medium |
| **Setup Time** | ~15 seconds |
| **RAM Usage** | ~2.4 GB |
| **Dependencies** | FAISS, BM25Okapi, sentence-transformers |

### Why Hybrid Wins on Recall

```
Query: "What are transformers in deep learning?"

BM25 alone:
  ✓ Matches "transformers" keyword
  ✗ Misses "attention is all you need" paper
  ✗ Misses semantic variants

Vector alone:
  ✓ Matches semantic meaning
  ✓ Finds "self-attention mechanisms"
  ✗ May miss exact terminology

HYBRID (Both):
  ✓ Gets keyword matches (BM25)
  ✓ Gets semantic variants (Vector)
  ✓ Best of both worlds → Recall 1.80
```

### Pros & Cons

✅ **Pros**:
- Highest recall (1.80) across all architectures
- Catches both keyword and semantic matches
- No hallucinations—results grounded in both methods
- Scalable to large corpora

❌ **Cons**:
- Slower than Vector (188ms vs 75ms)
- Requires tuning alpha parameter
- More complex to implement
- Higher memory footprint

### When to Use Hybrid RAG

✅ Use when:
- Accuracy critical (e.g., medical, legal)
- Can tolerate 2-3× latency
- Documents are technical (keywords matter)
- Budget allows for extra compute

❌ Skip when:
- Real-time requirements (< 100ms)
- Simple Q&A (Vector suffices)

---

## Graph RAG

### Overview

**Graph RAG** extracts entities and relationships, builds a knowledge graph, then traverses it to find related documents.

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                     GRAPH RAG PIPELINE                         │
└────────────────────────────────────────────────────────────────┘

OFFLINE (Indexing):

Step 1: Entity Extraction (spaCy NER)
┌──────────────────────────┐
│ "Apple Inc. was founded  │
│  by Steve Jobs in 1976"  │
└──────────────────────────┘
         │
         ▼
    Named Entity Recognizer
    PERSON: Steve Jobs
    ORG:    Apple Inc.
    DATE:   1976
         │
         ▼

Step 2: Build Knowledge Graph
┌────────────────────────────────────────────┐
│         KNOWLEDGE GRAPH                    │
├────────────────────────────────────────────┤
│                                            │
│    (Steve Jobs) ──founded──► (Apple Inc.) │
│         │                        │         │
│         │                        │         │
│         └────located────────────┘         │
│                                            │
│    (Apple Inc.) ──released──► (iPhone)    │
│                                            │
└────────────────────────────────────────────┘
         │
         ├──► NetworkX graph
         │
         ├──► Nodes: entities
         │
         └──► Edges: relationships


Step 3: Connect to Documents
┌────────────────────────────────────────────┐
│         DOCUMENT LINKS                     │
├────────────────────────────────────────────┤
│                                            │
│ Wikipedia/Steve_Jobs  ◄──contains──┐      │
│                                     │      │
│                              (Steve Jobs) │
│                                     │      │
│ Wikipedia/Apple_Inc   ◄──contains──┘      │
│                                            │
│ Wikipedia/iPhone      ◄──contains──┐      │
│                                     │      │
│                              (Apple Inc.) │
│                                            │
└────────────────────────────────────────────┘


ONLINE (Query Time):

Query: "Who founded Apple?"
    │
    ├──► Extract entities: {"Steve Jobs", "Apple"}
    │
    ├──► Graph traversal (2 hops)
    │    Start: "Apple" node
    │    Hop 1: Find "founded_by" edge
    │    Hop 2: Land on "Steve Jobs"
    │
    ├──► Collect connected documents
    │    Documents linked to "Steve Jobs"
    │    Documents linked to "Apple"
    │
    ▼
    [Wikipedia/Steve_Jobs, Wikipedia/Apple_Inc, ...]

    │
    ▼
    LLM Generation


ADVANTAGE: Pre-computed relationships → FAST lookup (9ms!)
```

### Algorithm

```python
# INDEXING
def index_documents(documents):
    import spacy
    import networkx as nx

    nlp = spacy.load("en_core_web_sm")
    graph = nx.Graph()
    doc_to_entities = {}

    for doc_id, text in enumerate(documents):
        # Extract entities
        nlp_doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in nlp_doc.ents]
        doc_to_entities[doc_id] = entities

        # Add to graph
        for entity_text, entity_type in entities:
            if entity_text not in graph:
                graph.add_node(entity_text, type=entity_type)
            graph.add_edge(entity_text, f"DOC_{doc_id}")

    return {graph: graph, doc_to_entities: doc_to_entities}

# RETRIEVAL (Graph traversal)
def retrieve(query, graph, doc_to_entities, hops=2):
    import spacy
    nlp = spacy.load("en_core_web_sm")

    # Extract query entities
    nlp_doc = nlp(query)
    query_entities = [ent.text for ent in nlp_doc.ents]

    # BFS from query entities
    visited_docs = set()
    for entity in query_entities:
        if entity in graph:
            # Traverse 2 hops
            for neighbor in graph.neighbors(entity):
                if neighbor.startswith("DOC_"):
                    visited_docs.add(int(neighbor.split("_")[1]))
                else:
                    # 2nd hop
                    for second_neighbor in graph.neighbors(neighbor):
                        if second_neighbor.startswith("DOC_"):
                            visited_docs.add(int(second_neighbor.split("_")[1]))

    return [documents[i] for i in list(visited_docs)[:5]]
```

### Characteristics

| Aspect | Details |
|--------|---------|
| **Recall** | 0.80 (lowest) |
| **Latency** | 9ms (fastest!) |
| **Complexity** | High |
| **Setup Time** | ~60 seconds |
| **RAM Usage** | ~2.5 GB |
| **Dependencies** | spaCy, NetworkX |

### When to Use Graph RAG

✅ Use when:
- Ultra-low latency required (< 10ms)
- Entity relationships critical
- Documents highly interconnected
- Real-time applications (chatbots, search)

❌ Skip when:
- Documents are simple prose (no entities)
- Quality more important than speed
- Limited engineering resources
- Dealing with unstructured text

### Limitations (Critical!)

❌ **Hub Node Problem** (Kubernetes case study):

```
Background: The word "Kubernetes" appears in 90%+ of docs

What happens:
┌──────────────────────────────┐
│  Query: "What is kubectl?"   │
└──────────────────────────────┘
         │
         ├──► Extract entity: "kubectl"
         │
         ├──► Entity not found in graph
         │
         └──► Fallback to keyword search
              Falls back to low recall (0.80)

Why Graph Struggles:
- "Kubernetes" becomes HUB NODE (connects to everything)
- Graph becomes fully connected (no structure)
- Hard to distinguish signal from noise
- Traversal loses specificity
```

---

## Parent-Child RAG

### Overview

**Parent-Child RAG** uses hierarchical chunking:
- **Child chunks**: Small (256 tokens) for precise embedding
- **Parent chunks**: Large (1024 tokens) for LLM context

Query retrieves small chunks, but passes large parent chunks to LLM.

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                  PARENT-CHILD RAG PIPELINE                     │
└────────────────────────────────────────────────────────────────┘

OFFLINE (Indexing - Hierarchical Chunking):

Original Document (2000 tokens):
┌────────────────────────────────────────────┐
│ "Cloud computing is on-demand delivery...  │
│  Three deployment models exist: public,    │
│  private, hybrid. Nine services exist:     │
│  SaaS, PaaS, IaaS. Cloud benefits...       │
│  Security considerations... Cost models... │
│  Migration strategies..."                   │
└────────────────────────────────────────────┘
    │
    ├──► Split into three levels:
    │
    ▼

LEVEL 1: Parent (1024 tokens)
┌────────────────────────────────────────────┐
│ "Cloud computing is on-demand delivery...  │
│  [FULL FIRST HALF OF DOCUMENT]"            │
└────────────────────────────────────────────┘
    │
    ▼

LEVEL 2: Child (256 tokens each)
┌─────────────────────────┬──────────────────┐
│ Child 1:                │ Child 2:          │
│ "Cloud computing is     │ "Three deployment │
│  on-demand delivery     │  models: public,  │
│  of IT resources..."    │  private, hybrid" │
└─────────────────────────┴──────────────────┘
    │
    ├──► Only children get embedded (smaller)
    │
    └──► Parents stored for LLM context


ONLINE (Query Time):

Query: "What are cloud deployment models?"
    │
    ├──► Embed child chunks (256 tokens each)
    │    Fast embeddings, precise matching
    │
    ├──► Search FAISS index
    │    "Three deployment models" (Child 2) ✓ High score
    │
    ├──► Retrieve mapped parent
    │    Parent (1024 tokens) with full context
    │
    ▼
    ┌──────────────────────────────────────┐
    │  LLM Prompt:                         │
    │                                      │
    │  Context (Parent 1024 tokens):       │
    │  "Cloud computing is on-demand...    │
    │   [FULL PARAGRAPH WITH CONTEXT]"     │
    │                                      │
    │  Question: "What are cloud           │
    │  deployment models?"                 │
    │                                      │
    │  ──► Precise answer with context     │
    └──────────────────────────────────────┘


KEY INSIGHT:
Small chunks for retrieval precision
+ Large chunks for LLM reasoning
= Better answers than pure Vector RAG
```

### Algorithm

```python
# INDEXING (Hierarchical)
def index_documents(documents):
    parent_chunks = []
    child_chunks = []
    child_to_parent = {}

    for doc_id, text in enumerate(documents):
        # Split into parent chunks (1024 tokens)
        parent_split = chunk_text(text, chunk_size=1024, overlap=100)

        for parent_idx, parent in enumerate(parent_split):
            parent_id = f"DOC_{doc_id}_PARENT_{parent_idx}"
            parent_chunks.append(parent)

            # Split parent into child chunks (256 tokens)
            child_split = chunk_text(parent,
                                     chunk_size=256,
                                     overlap=20)

            for child_idx, child in enumerate(child_split):
                child_id = f"DOC_{doc_id}_CHILD_{child_idx}"
                child_chunks.append(child)
                child_to_parent[child_id] = parent_id

    # Embed ONLY children (save on compute)
    child_embeddings = encoder.encode(child_chunks)
    faiss_index.add(child_embeddings)

    return {
        faiss_index: faiss_index,
        parent_chunks: parent_chunks,
        child_to_parent: child_to_parent,
        children: child_chunks
    }

# RETRIEVAL (With parent context)
def retrieve(query):
    # Search on children (precise)
    query_emb = encoder.encode(query)
    distances, child_indices = faiss_index.search(query_emb, k=5)

    # Map back to parents (context)
    parent_ids = set()
    for child_idx in child_indices:
        child_id = child_chunks[child_idx]
        parent_id = child_to_parent[child_id]
        parent_ids.add(parent_id)

    # Return parents for LLM
    return [parent_chunks[pid] for pid in parent_ids]
```

### Characteristics

| Aspect | Details |
|--------|---------|
| **Recall** | 0.95 (moderate) |
| **Latency** | 120ms (acceptable) |
| **Complexity** | Medium-High |
| **Setup Time** | 300+ seconds (slow indexing) |
| **RAM Usage** | 3.6 GB (higher) |
| **Dependencies** | FAISS, sentence-transformers |

### When to Use Parent-Child RAG

✅ Use when:
- Documents are LARGE (> 10K tokens)
- Need context preservation
- Can afford slower indexing
- Quality more important than speed

❌ Skip when:
- Documents already short (< 1K tokens)
- Latency critical
- Memory limited
- Parent-child structure not natural

---

## Multi-Query RAG

### Overview

**Multi-Query RAG** generates multiple rephrasing of the user query, retrieves for each, and merges results.

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                   MULTI-QUERY RAG PIPELINE                     │
└────────────────────────────────────────────────────────────────┘

User Query: "What is cloud computing?"
    │
    ├──► LLM EXPANSION PHASE
    │    (Generate 3 alternative phrasings)
    │
    ├──► Query 1: "What is cloud computing?"
    │    Query 2: "Define on-demand IT resources"
    │    Query 3: "Explain cloud-based services"
    │
    ├──► PARALLEL RETRIEVAL (3 searches)
    │
    │    Search 1          Search 2          Search 3
    │    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │    │ Wikipedia:│     │ Wikipedia:│     │ Wikipedia:│
    │    │ Cloud     │     │ On-demand │     │ SaaS      │
    │    │ Computing │     │ Computing │     │           │
    │    │ (score:9) │     │ (score:7) │     │ (score:6) │
    │    │           │     │           │     │           │
    │    │ Wikipedia:│     │ Wikipedia:│     │ Wikipedia:│
    │    │ IT        │     │ Cloud     │     │ IaaS      │
    │    │ Services  │     │ Storage   │     │           │
    │    │ (score:8) │     │ (score:8) │     │ (score:8) │
    │    │           │     │           │     │           │
    │    │ Wikipedia:│     │ Wikipedia:│     │ Wikipedia:│
    │    │ Computing │     │ Grid      │     │ PaaS      │
    │    │ Platform  │     │ Computing │     │           │
    │    │ (score:7) │     │ (score:5) │     │ (score:7) │
    │    └───────────┘     └───────────┘     └───────────┘
    │
    ├──► MERGE & DEDUPLICATE
    │
    └──► Top-5 Results
         (union of all queries)
         │
         ├──► Wikipedia: Cloud Computing (appeared in Q1, Q2)
         ├──► Wikipedia: IT Services (appeared in Q1)
         ├──► Wikipedia: Cloud Storage (appeared in Q2)
         ├──► Wikipedia: On-demand Computing (appeared in Q2)
         └──► Wikipedia: SaaS (appeared in Q3)
              │
              ▼
         LLM Generation
         (High confidence - found in multiple queries)


TOTAL API CALLS: 3 × (embeddings + retrieval searches)
LATENCY: ~675ms (9× slower than Vector)
RECALL: 1.50 (same as Vector! No improvement)
```

### Algorithm

```python
# MULTI-QUERY EXPANSION
def expand_query(query, llm):
    """Generate 3 alternative phrasings of the query"""
    prompt = f"""Generate 2 alternative phrasings of this query
               (plus the original = 3 total):

               Original: "{query}"

               Alternative 1:
               Alternative 2:"""

    response = llm.generate(prompt, max_tokens=200)

    # Parse response to get 3 queries
    queries = [query]  # Original
    queries += parse_alternatives(response)  # 2 new ones
    return queries

# PARALLEL RETRIEVAL
def retrieve_multi_query(query, num_queries=3):
    # Expand query
    expanded_queries = expand_query(query, llm)

    all_docs = set()

    # Retrieve for each query
    for q in expanded_queries:
        query_emb = encoder.encode(q)
        distances, indices = faiss_index.search(query_emb, k=5)

        for idx in indices:
            all_docs.add(idx)

    # Deduplicate and return top-5
    return [documents[i] for i in list(all_docs)[:5]]

# GENERATION
def generate(query):
    docs = retrieve_multi_query(query)
    prompt = f"""Context: {docs}
                Query: {query}
                Answer:"""
    answer = llm.generate(prompt)
    return answer
```

### Characteristics

| Aspect | Details |
|--------|---------|
| **Recall** | 1.50 (same as Vector) |
| **Latency** | 675ms (9× slower!) |
| **Complexity** | Medium |
| **Setup Time** | ~10 seconds |
| **RAM Usage** | ~4.9 GB |
| **API Calls** | 3× per query |

### When to Use Multi-Query RAG

✅ Use when:
- Reducing query variance matters
- Query diversity is goal
- Cost is not a concern
- Batch processing (not real-time)

❌ Skip when:
- Latency < 200ms required
- API costs limited
- Single query sufficient
- Complex queries (expands to 3× work)

### Why Multi-Query Fails on This Dataset

```
The problem with synthetic QA pairs:

Vector RAG already finds most relevant docs
because questions are generated from documents.

Multi-Query expands queries → more docs retrieved
BUT: Same documents found multiple times
Result: No improvement in Recall@5, just 9× slower cost
```

---

## Comparison Matrix

### Performance Metrics

```
┌─────────────────┬────────────┬────────────┬─────────────┬──────────────┬────────────┐
│ Architecture    │ Recall@5   │ Precision  │ Latency     │ Throughput   │ RAM Usage  │
├─────────────────┼────────────┼────────────┼─────────────┼──────────────┼────────────┤
│ Hybrid RAG      │ 1.80 █████ │ 0.36 ████  │ 188ms       │ 5.4 q/s      │ 2,423 MB   │
│ Vector RAG      │ 1.50 ████  │ 0.30 ███   │ 75ms █████  │ 13.0 q/s ███ │ 1,636 MB   │
│ Multi-Query RAG │ 1.50 ████  │ 0.30 ███   │ 675ms       │ 1.5 q/s      │ 4,866 MB   │
│ Parent-Child    │ 0.95 ███   │ 0.19 ██    │ 120ms       │ 8.1 q/s ██   │ 3,663 MB   │
│ Graph RAG       │ 0.80 ██    │ 0.19 ██    │ 9ms ██████  │ 127.8 q/s    │ 2,492 MB   │
└─────────────────┴────────────┴────────────┴─────────────┴──────────────┴────────────┘
```

### Trade-off Visualization

```
              RECALL vs LATENCY TRADE-OFF

Recall        │
  2.0         │
              │  ◆ Hybrid RAG
  1.5         │  ◆ Vector RAG    ◆ Multi-Query
              │  │                 │
  1.0         │  │    ◆ Parent-Child
              │  │     │
  0.5         │  │     │     ◆ Graph RAG
              │  │     │      │
  0.0         └──┴─────┴──────┴────────► Latency (ms)
                 0    100   300  500  700

SWEET SPOT: Vector RAG (1.50 recall, 75ms latency)
```

### Strengths Summary

```
WHAT EACH ARCHITECTURE EXCELS AT:

┌──────────────────────────────────────┐
│ HYBRID RAG: Maximum Accuracy          │
├──────────────────────────────────────┤
│ Strengths:                           │
│ ✓ Highest recall (1.80)              │
│ ✓ Catches keyword + semantic matches │
│ ✓ Best for technical documents      │
│ ✓ No single point of failure        │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ VECTOR RAG: Balanced Default          │
├──────────────────────────────────────┤
│ Strengths:                           │
│ ✓ Good recall (1.50)                 │
│ ✓ Fast (75ms)                        │
│ ✓ Simple to deploy                   │
│ ✓ Lowest memory (1.6GB)              │
│ ✓ Recommended for production         │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ GRAPH RAG: Lightning Fast              │
├──────────────────────────────────────┤
│ Strengths:                           │
│ ✓ Fastest retrieval (9ms)            │
│ ✓ Pre-computed relationships         │
│ ✓ Perfect for real-time              │
│ ✓ Highest throughput (127.8 q/s)    │
│ ⚠️  But: lowest recall (0.80)         │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ PARENT-CHILD RAG: Contextual Richness │
├──────────────────────────────────────┤
│ Strengths:                           │
│ ✓ Preserves paragraph context        │
│ ✓ Better for large documents         │
│ ✓ Hierarchical structure             │
│ ⚠️  But: slow indexing (300s+)        │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ MULTI-QUERY RAG: Diversity            │
├──────────────────────────────────────┤
│ Strengths:                           │
│ ✓ Reduces query variance             │
│ ✓ Finds diverse angles              │
│ ⚠️  But: 9× slower, no recall boost  │
└──────────────────────────────────────┘
```

---

## Decision Tree

```
Choose RAG Architecture:

START
  │
  ├─ Latency < 10ms required?
  │  ├─ YES ──► GRAPH RAG (9ms, 127.8 q/s)
  │  │          ⚠️  Accept 0.80 recall
  │  │
  │  └─ NO
  │
  ├─ Latency < 100ms required?
  │  ├─ YES ──► VECTOR RAG (75ms)
  │  │          ✓ Recommended for production
  │  │
  │  └─ NO
  │
  ├─ Can afford 200ms+ latency?
  │  ├─ YES ──► Accuracy critical?
  │  │          ├─ YES ──► HYBRID RAG (1.80 recall)
  │  │          │          (best for legal, medical)
  │  │          │
  │  │          └─ NO ──► VECTOR RAG (good balance)
  │  │
  │  └─ NO ──► Go back to VECTOR RAG
  │
  ├─ Documents > 10K tokens?
  │  ├─ YES ──► Consider PARENT-CHILD RAG
  │  │
  │  └─ NO ──► VECTOR RAG (standard choice)
  │
  └─ Need query variance reduction?
     ├─ YES & Cost unlimited ──► MULTI-QUERY RAG
     └─ NO or Cost sensitive ──► VECTOR RAG

FINAL RECOMMENDATION: VECTOR RAG (for 80% of use cases)
```

---

## Implementation Checklist

### Choosing Your RAG

```
□ Define latency SLA (milliseconds)
□ Define recall requirement (how many relevant docs matter?)
□ Estimate knowledge base size
□ Check available compute/memory
□ Plan for growth (scalability)
□ Consider team expertise

MOST TEAMS: Choose VECTOR RAG
(Then upgrade to HYBRID if recall insufficient)
```

### Benchmarking Your Choice

```
FOR EACH ARCHITECTURE:

1. Index your documents
   └─ Time -> Setup Duration

2. Run 100 sample queries
   └─ Measure P50, P95 latency
   └─ Measure throughput
   └─ Measure recall on ground truth

3. Compare to baseline
   └─ Acceptable latency?
   └─ Acceptable recall?

4. Iterate
   └─ Tune parameters (chunk size, alpha, etc)
   └─ Re-benchmark
```

---

## References & Further Reading

- **Vector RAG**: [FAISS Paper](https://arxiv.org/abs/1702.08734)
- **Hybrid RAG**: [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- **Graph RAG**: [Knowledge Graphs for QA](https://arxiv.org/abs/2005.00631)
- **Parent-Child**: [Hierarchical Learning](https://arxiv.org/abs/1902.08564)
- **Multi-Query**: [Query Expansion Methods](https://arxiv.org/abs/2305.03653)

---

**Last Updated**: 2026-03-20
**Benchmarked On**: Small Dataset (50 Wikipedia articles, 20 QA pairs)
**See Also**: `README.md`, `BENCHMARK_REPORT.md`, `dashboard.html`
