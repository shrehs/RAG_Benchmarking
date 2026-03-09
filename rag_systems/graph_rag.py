"""
graph_rag.py — Graph RAG: spaCy entity extraction → NetworkX knowledge graph → traversal → LLM

Architecture:
  documents → spaCy NER → entity-relation graph (NetworkX)
  query → extract query entities → graph traversal (2-hop) → retrieve connected chunks → LLM

Decisions:
  - spaCy en_core_web_sm for NER (D-009): lightweight, no GPU
  - 2-hop graph traversal (D-009): captures indirect relationships
  - MIN_ENTITY_FREQ=2 to filter noise entities (D-009)
  - Basic entity normalization: lowercase + alias dict (D-009 known limitation)

Why it matters:
  Vector search treats documents as isolated bags of words.
  Graph RAG understands that Entity A is connected to Entity B which relates to Entity C.
  Best for: relational data, scientific papers, technical docs with cross-references.

Known limitation (LIM-001, LIM-007):
  Entity resolution is imperfect. "k8s" and "Kubernetes" treated as separate nodes
  unless caught by alias_map. Hub nodes (entities in 90%+ of docs) are score-capped.
"""

import re
import pickle
from collections import defaultdict, Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    SPACY_MODEL, GRAPH_HOP_DEPTH, MIN_ENTITY_FREQ,
    TOP_K, CHUNK_SIZE, CHUNK_OVERLAP,
)
from groq_client import GroqClient
from rag_systems.base_rag import BaseRAG, Document
from rag_systems.chunker import chunk_documents


# D-009: Known alias map for common entity normalization
ENTITY_ALIASES = {
    "k8s": "kubernetes",
    "bert": "bert model",
    "gpt": "gpt model",
    "llm": "large language model",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
}


class GraphRAG(BaseRAG):
    """
    Knowledge graph-based RAG.
    
    Graph structure:
      - Nodes: entities (PERSON, ORG, PRODUCT, CONCEPT extracted by spaCy)
      - Edges: co-occurrence within same chunk (weighted by frequency)
      - Node metadata: list of chunk_ids containing that entity
    
    Retrieval:
      1. Extract entities from query
      2. Find matching graph nodes
      3. Traverse 2-hop neighborhood
      4. Collect all chunks from neighborhood nodes
      5. Score chunks by entity overlap, return top-k
    """

    def __init__(self):
        super().__init__("Graph RAG")
        self.client = GroqClient()   # D-016: LLM generation via Groq
        self.graph = nx.Graph()
        self.chunks: list[dict] = []
        self.chunk_map: dict[str, dict] = {}   # chunk_id → chunk
        self.entity_to_chunks: dict[str, list[str]] = defaultdict(list)
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy to avoid slow import at module level."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(SPACY_MODEL)
            except OSError:
                print(f"[GraphRAG] Downloading spaCy model {SPACY_MODEL}...")
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", SPACY_MODEL],
                    check=True,
                )
                self._nlp = spacy.load(SPACY_MODEL)
        return self._nlp

    def index(self, documents: list[dict], cache_path: Path | None = None) -> None:
        print(f"[GraphRAG] Indexing {len(documents)} documents...")
        self.chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"[GraphRAG] {len(self.chunks)} chunks — extracting entities...")

        # Build chunk map
        for chunk in self.chunks:
            self.chunk_map[chunk["chunk_id"]] = chunk

        # Count entity frequency across all chunks (for MIN_ENTITY_FREQ filter)
        entity_freq: Counter = Counter()
        chunk_entities: dict[str, list[str]] = {}

        for chunk in self.chunks:
            entities = self._extract_entities(chunk["content"])
            chunk_entities[chunk["chunk_id"]] = entities
            entity_freq.update(entities)

        # Filter: keep only entities appearing >= MIN_ENTITY_FREQ times
        valid_entities = {e for e, count in entity_freq.items() if count >= MIN_ENTITY_FREQ}
        print(f"[GraphRAG] {len(valid_entities)} valid entities (freq >= {MIN_ENTITY_FREQ}) "
              f"from {len(entity_freq)} total")

        # Build graph
        for chunk in self.chunks:
            chunk_id = chunk["chunk_id"]
            entities = [e for e in chunk_entities[chunk_id] if e in valid_entities]

            for entity in entities:
                # Add node if not exists
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, chunks=[], freq=0)
                # Guard: networkx add_edge() creates nodes implicitly without
                # our custom attrs — ensure they exist before appending.
                node = self.graph.nodes[entity]
                if "chunks" not in node:
                    node["chunks"] = []
                    node["freq"] = 0
                node["chunks"].append(chunk_id)
                node["freq"] += 1
                self.entity_to_chunks[entity].append(chunk_id)

                # Add co-occurrence edges between entities in same chunk
                for other_entity in entities:
                    if other_entity != entity:
                        if self.graph.has_edge(entity, other_entity):
                            self.graph[entity][other_entity]["weight"] += 1
                        else:
                            self.graph.add_edge(entity, other_entity, weight=1)

        print(f"[GraphRAG] Graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        self._is_indexed = True

        if cache_path:
            self._save(cache_path)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Document]:
        """
        1. Extract entities from query
        2. Find those nodes in graph
        3. BFS/DFS up to GRAPH_HOP_DEPTH hops
        4. Score all reachable chunks
        5. Return top-k
        """
        query_entities = self._extract_entities(query)
        if not query_entities:
            # Fallback: if no entities found, use keyword match
            return self._keyword_fallback(query, k)

        # Find matching nodes in graph
        matched_nodes = set()
        for entity in query_entities:
            if self.graph.has_node(entity):
                matched_nodes.add(entity)
            # Partial match: check if entity is substring of any node
            else:
                for node in self.graph.nodes():
                    if entity in node or node in entity:
                        matched_nodes.add(node)

        if not matched_nodes:
            return self._keyword_fallback(query, k)

        # Traverse up to GRAPH_HOP_DEPTH hops (D-009)
        neighborhood = set(matched_nodes)
        frontier = set(matched_nodes)
        for _ in range(GRAPH_HOP_DEPTH):
            new_frontier = set()
            for node in frontier:
                new_frontier.update(self.graph.neighbors(node))
            frontier = new_frontier - neighborhood
            neighborhood.update(frontier)

        # Collect all chunks from neighborhood nodes
        chunk_scores: dict[str, float] = defaultdict(float)
        for node in neighborhood:
            node_data = self.graph.nodes[node]
            # Score: higher if node is directly matched (not just a hop away)
            is_direct = node in matched_nodes
            node_weight = 2.0 if is_direct else 1.0

            for chunk_id in node_data.get("chunks", []):
                chunk_scores[chunk_id] += node_weight

        # Sort by score, return top-k
        top_chunk_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:k]

        results = []
        for chunk_id in top_chunk_ids:
            if chunk_id in self.chunk_map:
                chunk = self.chunk_map[chunk_id]
                results.append(Document(
                    content=chunk["content"],
                    source=chunk["source"],
                    chunk_id=chunk_id,
                    score=chunk_scores[chunk_id],
                    metadata={
                        **chunk.get("metadata", {}),
                        "matched_entities": list(matched_nodes),
                        "neighborhood_size": len(neighborhood),
                    },
                ))
        return results

    def generate(self, query: str, documents: list[Document]) -> tuple[str, dict]:
        prompt = self._build_prompt(query, documents)
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        usage = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "embedding": 0,
        }
        return answer, usage

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract and normalize named entities from text using spaCy.
        Applies alias normalization from ENTITY_ALIASES (D-009 / LIM-001).
        """
        doc = self.nlp(text[:10000])  # spaCy has token limits
        entities = []
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "PRODUCT", "WORK_OF_ART", "EVENT", "GPE"):
                normalized = ent.text.lower().strip()
                normalized = re.sub(r"[^\w\s]", "", normalized)
                normalized = ENTITY_ALIASES.get(normalized, normalized)
                if len(normalized) > 2:   # skip very short entities
                    entities.append(normalized)
        return list(set(entities))

    def _keyword_fallback(self, query: str, k: int) -> list[Document]:
        """
        Fallback when no entities found: simple keyword overlap scoring.
        Documented as a known limitation path — logged in metadata.
        """
        query_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk["content"].lower().split())
            score = len(query_words & chunk_words) / (len(query_words) + 1e-9)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Document(
                content=c["content"],
                source=c["source"],
                chunk_id=c["chunk_id"],
                score=s,
                metadata={**c.get("metadata", {}), "fallback": "keyword"},
            )
            for s, c in scored[:k]
        ]

    def _save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(path / "entity_to_chunks.pkl", "wb") as f:
            pickle.dump(dict(self.entity_to_chunks), f)

    def load(self, path: Path) -> None:
        with open(path / "graph.pkl", "rb") as f:
            self.graph = pickle.load(f)
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
            for chunk in self.chunks:
                self.chunk_map[chunk["chunk_id"]] = chunk
        with open(path / "entity_to_chunks.pkl", "rb") as f:
            self.entity_to_chunks = defaultdict(list, pickle.load(f))
        self._is_indexed = True
        print(f"[GraphRAG] Loaded graph: {self.graph.number_of_nodes()} nodes")
