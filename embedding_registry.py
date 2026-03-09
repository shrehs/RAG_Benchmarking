"""
embedding_registry.py — Registry of all supported embedding models.

Decisions:
  D-013: Three embedding tiers — cheap API, premium API, open-source local
  D-014: GPU detection logic — auto-select FAISS index type based on device
  D-015: Embedding model comparison is a first-class experiment axis

Each model entry defines:
  - how to embed (API call vs local inference)
  - cost per 1M tokens (0 for local)
  - dimensions
  - expected quality tier
  - whether GPU inference is supported

Usage:
    from embedding_registry import get_embedder, EMBEDDING_MODELS
    embedder = get_embedder("bge-large")
    vecs = embedder.embed(["text one", "text two"])
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import OPENAI_API_KEY


# ─── Model Registry ───────────────────────────────────────────────────────────

@dataclass
class EmbeddingModel:
    name: str                          # short key used in CLI + results
    display_name: str                  # human label
    dims: int                          # output vector dimensions
    provider: str                      # "openai" | "huggingface" | "cohere"
    model_id: str                      # API model ID or HuggingFace repo ID
    cost_per_1m_tokens: float          # USD; 0.0 for local models
    max_batch_size: int = 100
    supports_gpu: bool = False         # whether HF model can use CUDA
    quality_tier: str = "standard"     # "budget" | "standard" | "premium"
    notes: str = ""


# D-013 decision log:
# Three tiers deliberately chosen:
#   budget    → text-embedding-3-small  ($0.02/1M)  — fast, cheap, good enough for most
#   premium   → text-embedding-3-large  ($0.13/1M)  — 5× cost, ~5% recall gain
#   local-std → bge-base-en-v1.5        (free)      — 768-dim, CPU-friendly
#   local-lg  → bge-large-en-v1.5       (free)      — 1024-dim, best open-source English
#   local-gpu → e5-large-v2             (free)      — competitive with 3-small, GPU wins big

EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    "gemini-embedding-001": EmbeddingModel(
        name="gemini-embedding-001",
        display_name="Google gemini-embedding-001",
        dims=3072,
        provider="google",
        model_id="gemini-embedding-001",
        cost_per_1m_tokens=0.025,
        quality_tier="standard",
        notes="D-016: Default embedding model. 3072-dim. $0.025/1M tokens.",
    ),
    "text-3-small": EmbeddingModel(
        name="text-3-small",
        display_name="OpenAI text-embedding-3-small",
        dims=1536,
        provider="openai",
        model_id="text-embedding-3-small",
        cost_per_1m_tokens=0.020,
        quality_tier="standard",
        notes="D-003 primary model. Strong cost/quality balance. Baseline.",
    ),
    "text-3-large": EmbeddingModel(
        name="text-3-large",
        display_name="OpenAI text-embedding-3-large",
        dims=3072,
        provider="openai",
        model_id="text-embedding-3-large",
        cost_per_1m_tokens=0.130,
        quality_tier="premium",
        notes="D-013: 6.5× cost vs 3-small. Tests if quality gap justifies premium.",
    ),
    "ada-002": EmbeddingModel(
        name="ada-002",
        display_name="OpenAI text-embedding-ada-002",
        dims=1536,
        provider="openai",
        model_id="text-embedding-ada-002",
        cost_per_1m_tokens=0.100,
        quality_tier="legacy",
        notes="Legacy model. Included as historical baseline. Worse than 3-small at higher cost.",
    ),
    "bge-base": EmbeddingModel(
        name="bge-base",
        display_name="BAAI/bge-base-en-v1.5",
        dims=768,
        provider="huggingface",
        model_id="BAAI/bge-base-en-v1.5",
        cost_per_1m_tokens=0.0,
        max_batch_size=32,
        supports_gpu=True,
        quality_tier="standard",
        notes="D-013: CPU-runnable local model. 768-dim. Good budget open-source option.",
    ),
    "bge-large": EmbeddingModel(
        name="bge-large",
        display_name="BAAI/bge-large-en-v1.5",
        dims=1024,
        provider="huggingface",
        model_id="BAAI/bge-large-en-v1.5",
        cost_per_1m_tokens=0.0,
        max_batch_size=16,
        supports_gpu=True,
        quality_tier="premium",
        notes="D-013: Best open-source English embedding. GPU gives 8-12× speedup.",
    ),
    "e5-large": EmbeddingModel(
        name="e5-large",
        display_name="intfloat/e5-large-v2",
        dims=1024,
        provider="huggingface",
        model_id="intfloat/e5-large-v2",
        cost_per_1m_tokens=0.0,
        max_batch_size=16,
        supports_gpu=True,
        quality_tier="premium",
        notes="D-013: Strong multilingual model. Competitive with bge-large on BEIR benchmarks.",
    ),
}


# ─── Embedder classes ─────────────────────────────────────────────────────────

class BaseEmbedder:
    """Shared interface for all embedding providers."""

    def __init__(self, model: EmbeddingModel, device: str = "cpu"):
        self.model = model
        self.device = device
        self._total_tokens_used = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed list of texts. Returns (N, dims) float32 array."""
        raise NotImplementedError

    @property
    def tokens_used(self) -> int:
        return self._total_tokens_used

    def estimated_cost(self, n_tokens: int) -> float:
        return (n_tokens / 1_000_000) * self.model.cost_per_1m_tokens


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding API (text-embedding-3-small, 3-large, ada-002)."""

    def __init__(self, model: EmbeddingModel, device: str = "cpu"):
        super().__init__(model, device)
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed(self, texts: list[str], batch_size: int = None) -> np.ndarray:
        bs = batch_size or self.model.max_batch_size
        all_vecs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            response = self.client.embeddings.create(
                model=self.model.model_id,
                input=batch,
            )
            self._total_tokens_used += response.usage.total_tokens
            all_vecs.extend([e.embedding for e in response.data])
        return np.array(all_vecs, dtype=np.float32)


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Local HuggingFace sentence-transformer embedding.
    Supports CPU and CUDA (D-014: GPU vs CPU benchmarking).
    """

    def __init__(self, model: EmbeddingModel, device: str = "auto"):
        # D-014: resolve device
        resolved = _resolve_device(device, model.supports_gpu)
        super().__init__(model, resolved)
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[embedder] Loading {self.model.model_id} on {self.device}...")
                self._encoder = SentenceTransformer(self.model.model_id, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    def embed(self, texts: list[str], batch_size: int = None) -> np.ndarray:
        self._load()
        bs = batch_size or self.model.max_batch_size
        # Rough token estimate: 1 word ≈ 1.3 tokens
        self._total_tokens_used += int(sum(len(t.split()) * 1.3 for t in texts))
        vecs = self._encoder.encode(
            texts,
            batch_size=bs,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalize for cosine similarity
        )
        return vecs.astype(np.float32)


# ─── Device detection (D-014) ─────────────────────────────────────────────────

def _resolve_device(requested: str, model_supports_gpu: bool) -> str:
    """
    D-014: Auto-detect best available device.
    
    Logic:
      "auto"  → CUDA if available + model supports it, else CPU
      "cuda"  → CUDA if available, warn + fallback to CPU if not
      "cpu"   → always CPU (used for GPU vs CPU comparison)
      "mps"   → Apple Silicon MPS if available
    """
    if requested == "cpu":
        return "cpu"

    try:
        import torch
        if requested == "auto":
            if torch.cuda.is_available() and model_supports_gpu:
                return "cuda"
            elif torch.backends.mps.is_available() and model_supports_gpu:
                return "mps"
            return "cpu"
        elif requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print(f"[embedder] WARNING: CUDA requested but not available — falling back to CPU")
            return "cpu"
        elif requested == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
    except ImportError:
        pass
    return "cpu"


def detect_device() -> dict:
    """
    D-014: Return full device info for logging in benchmark results.
    """
    info = {"cpu": True, "cuda": False, "mps": False, "cuda_name": None, "cuda_memory_gb": None}
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = torch.backends.mps.is_available()
        if info["cuda"]:
            info["cuda_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        pass
    return info


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_embedder(model_name: str, device: str = "auto") -> BaseEmbedder:
    """
    Get an embedder instance for the given model name and device.
    
    Args:
        model_name: Key from EMBEDDING_MODELS (e.g. "bge-large", "text-3-small")
        device: "auto" | "cpu" | "cuda" | "mps"
    
    Returns BaseEmbedder subclass ready to call .embed(texts)
    """
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding model: '{model_name}'. "
            f"Available: {list(EMBEDDING_MODELS.keys())}"
        )
    model = EMBEDDING_MODELS[model_name]

    if model.provider == "openai":
        return OpenAIEmbedder(model, device="cpu")   # API always "cpu" side
    elif model.provider == "google":
        from gemini_client import GeminiEmbedder as _GeminiEmbedder

        class _RegistryGeminiEmbedder(BaseEmbedder):
            def __init__(self, m: EmbeddingModel):
                super().__init__(m, "cpu")
                self._gem = _GeminiEmbedder(model=m.model_id)
            def embed(self, texts: list[str]) -> np.ndarray:
                vecs = self._gem.embed(texts)
                self._total_tokens_used += int(sum(len(t.split()) * 1.3 for t in texts))
                return np.array(vecs, dtype=np.float32)

        return _RegistryGeminiEmbedder(model)
    elif model.provider == "huggingface":
        return HuggingFaceEmbedder(model, device=device)
    else:
        raise ValueError(f"Unknown provider: {model.provider}")
