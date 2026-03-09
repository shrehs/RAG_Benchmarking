from rag_systems.vector_rag import VectorRAG
from rag_systems.hybrid_rag import HybridRAG
from rag_systems.graph_rag import GraphRAG
from rag_systems.parent_child_rag import ParentChildRAG
from rag_systems.multi_query_rag import MultiQueryRAG

ALL_SYSTEMS = {
    "vector": VectorRAG,
    "hybrid": HybridRAG,
    "graph": GraphRAG,
    "parent_child": ParentChildRAG,
    "multi_query": MultiQueryRAG,
}
