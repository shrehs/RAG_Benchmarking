"""
local_client.py — Fully offline inference stack.

LLM:      Llama 3.2 3B via ollama
Embedder: BAAI/bge-large-en-v1.5 via sentence-transformers (no network needed after first download)

Drop-in replacements for GeminiClient / GeminiEmbedder so no RAG system logic changes.

Usage:
    from local_client import LocalLLMClient, LocalEmbedder
    client  = LocalLLMClient()
    embedder = LocalEmbedder()
    response = client.chat.completions.create(
        model="llama3.2:3b", temperature=0, max_tokens=512,
        messages=[{"role": "user", "content": "hello"}]
    )
    text = response.choices[0].message.content
    vecs = embedder.embed(["text one", "text two"])   # list[list[float]]
"""

import os

# Hide all GPUs from torch/sentence-transformers — forces CPU-only mode and
# prevents torch from probing CUDA/ROCm drivers on import (avoids hangs on
# systems with Intel integrated graphics or unsupported GPU drivers).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


# ── Response shims — identical shape to GeminiClient / OpenAI ────────────────

@dataclass
class _Message:
    content: str
    role: str = "assistant"

@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"
    index: int = 0

@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class _ChatResponse:
    choices: list
    usage: _Usage
    model: str


# ── LLM via ollama ────────────────────────────────────────────────────────────

class _Completions:
    def create(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 512,
        **kwargs,
    ) -> _ChatResponse:
        """
        Drop-in for openai/gemini chat.completions.create().
        Converts OpenAI-style messages list directly to ollama format.
        """
        import ollama as _ollama
        response = _ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        text = response.message.content
        prompt_tokens = int(sum(len(m["content"].split()) * 1.3 for m in messages))
        completion_tokens = int(len(text.split()) * 1.3)
        return _ChatResponse(
            choices=[_Choice(message=_Message(content=text))],
            usage=_Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=model,
        )


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class LocalLLMClient:
    """
    Drop-in replacement for GeminiClient.

    Swap:
        # Before:
        from gemini_client import GeminiClient
        client = GeminiClient()

        # After:
        from local_client import LocalLLMClient
        client = LocalLLMClient()

    Interface is identical: client.chat.completions.create(model=..., messages=[...])
    Requires ollama running locally with the target model pulled.
    """
    def __init__(self):
        self.chat = _Chat()


# ── Embedder via sentence-transformers ────────────────────────────────────────

class LocalEmbedder:
    """
    Drop-in replacement for GeminiEmbedder.
    Uses BAAI/bge-large-en-v1.5 (1024-dim, L2-normalised).

    embed()       → list[list[float]]   (N texts → N vectors)
    embed_query() → list[float]         (1 text  → 1 vector)

    Model is downloaded from HuggingFace on first use, then cached locally.
    """

    def __init__(self, model_key: str = "bge-large"):
        from embedding_registry import EMBEDDING_MODELS, HuggingFaceEmbedder
        model_cfg = EMBEDDING_MODELS[model_key]
        # Use CUDA when visible (e.g. Colab T4), fall back to CPU otherwise.
        # CUDA_VISIBLE_DEVICES="" means no GPU; anything else (e.g. "0") means GPU available.
        _cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        device = "cpu" if _cuda == "" else "cuda"
        self._hf = HuggingFaceEmbedder(model_cfg, device=device)

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._hf.embed(texts)   # np.ndarray (N, 1024), already L2-normalised
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vecs = self._hf.embed([text])  # np.ndarray (1, 1024)
        return vecs[0].tolist()


# ── LangChain wrappers for RAGAS ──────────────────────────────────────────────

def get_langchain_llm(model: str = None):
    """
    Returns a LangChain-compatible local LLM for RAGAS evaluation.
    Uses ChatOllama — no API key required.
    timeout=None disables the client-level timeout; RAGAS RunConfig controls it.
    """
    from config import LLM_MODEL
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model or LLM_MODEL, temperature=0, timeout=None)


def get_langchain_embeddings(model_name: str = "BAAI/bge-large-en-v1.5"):
    """
    Returns a LangChain-compatible embeddings object for RAGAS.
    Uses GPU automatically on Colab (CUDA_VISIBLE_DEVICES set to device id),
    falls back to CPU locally.
    """
    _cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if _cuda == "" else "cuda"
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
