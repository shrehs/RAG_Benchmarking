"""
gemini_client.py — Google Gemini API wrapper.

Decision D-016: Switched from OpenAI GPT to Google Gemini.
This module provides a drop-in client that matches the OpenAI chat interface
used in all 5 RAG systems, so no RAG system code needs to change.

Usage (same as OpenAI client in all RAG systems):
    from gemini_client import GeminiClient
    client = GeminiClient()
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=512,
        messages=[{"role": "user", "content": "..."}]
    )
    text = response.choices[0].message.content
    usage = response.usage  # .prompt_tokens, .completion_tokens

D-016 decision rationale:
  - Gemini 2.0 Flash: comparable quality to gpt-4o-mini, lower cost, faster
  - Gemini 1.5 Pro: used as judge model (stronger than flash, avoids self-eval bias D-002)
  - RAGAS compatibility: works with LangChain ChatGoogleGenerativeAI wrapper
  - Uses google-genai SDK (google.generativeai is deprecated)
"""

import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

from config import GEMINI_API_KEY


# ── Response shims — match OpenAI response shape ──────────────────────────────

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


# ── Shared rate limiter (free tier: 15 RPM) ───────────────────────────────────

class _RateLimiter:
    """Enforces a minimum gap between successive API calls (thread-safe)."""
    def __init__(self, min_interval: float = 2.1):
        # 2.1 s → ≤28.6 RPM, safely under the 30 RPM limit
        self._min_interval = min_interval
        self._last_call: float = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()

_rate_limiter = _RateLimiter()


def get_rate_limiter() -> _RateLimiter:
    """Return the shared rate limiter so external callers join the same budget."""
    return _rate_limiter


# LangChain-compatible wrapper around _rate_limiter (used by get_langchain_llm)
try:
    import asyncio
    from langchain_core.rate_limiters import BaseRateLimiter as _BaseRateLimiter

    class _LCRateLimiter(_BaseRateLimiter):
        """Bridges _RateLimiter to LangChain's BaseRateLimiter interface."""

        def acquire(self, *, blocking: bool = True) -> bool:
            if blocking:
                _rate_limiter.wait()
            return True

        async def aacquire(self, *, blocking: bool = True) -> bool:
            if blocking:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _rate_limiter.wait)
            return True

    _lc_rate_limiter: _BaseRateLimiter | None = _LCRateLimiter()
except ImportError:
    _lc_rate_limiter = None


# ── Gemini client ──────────────────────────────────────────────────────────────

class _Completions:
    def __init__(self, api_key: str):
        if not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai not installed.\n"
                "Run: pip install google-genai"
            )
        self._client = genai.Client(
            api_key=api_key,
        )

    def create(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 512,
        **kwargs,
    ) -> _ChatResponse:
        """
        Drop-in replacement for openai.chat.completions.create().

        Converts OpenAI-style messages list to google.genai format:
          - "system" role → system_instruction in GenerateContentConfig
          - "user" / "assistant" → Content(role="user" / "model")
        """
        # Separate system messages from conversation
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system_instruction = "\n".join(system_parts) if system_parts else None

        contents = []
        for m in messages:
            if m["role"] == "user":
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=m["content"])],
                    )
                )
            elif m["role"] == "assistant":
                contents.append(
                    genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text=m["content"])],
                    )
                )

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        # Simplify to plain string for single-turn (no history)
        send_contents = (
            contents[0].parts[0].text if len(contents) == 1 else contents
        )

        _rate_limiter.wait()
        response = self._client.models.generate_content(
            model=model,
            contents=send_contents,
            config=config,
        )

        output_text = response.text

        # Token counts from usage metadata
        try:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
        except (AttributeError, TypeError):
            prompt_tokens = int(len(" ".join(m["content"] for m in messages).split()) * 1.3)
            completion_tokens = int(len(output_text.split()) * 1.3)

        return _ChatResponse(
            choices=[_Choice(message=_Message(content=output_text))],
            usage=_Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=model,
        )


class _Chat:
    def __init__(self, api_key: str):
        self.completions = _Completions(api_key)


class GeminiClient:
    """
    Drop-in replacement for openai.OpenAI() client.

    Swap usage in RAG systems:
        # Before (OpenAI):
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # After (Gemini):
        from gemini_client import GeminiClient
        client = GeminiClient()

    Interface is identical: client.chat.completions.create(...)
    """
    def __init__(self, api_key: str = None):
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not set. Add to .env:\n"
                "GEMINI_API_KEY=your-key-here\n"
                "Get one at: https://aistudio.google.com/app/apikey"
            )
        self.chat = _Chat(api_key=key)


class GeminiEmbedder:
    """
    Google gemini-embedding-001 embedder. Replaces OpenAI embeddings (D-016).

    Model specs:
      - Model: gemini-embedding-001
      - Dimensions: 3072
      - Pricing: ~$0.000025 / 1K tokens
      - task_type RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for queries

    Usage:
        embedder = GeminiEmbedder()
        vecs = embedder.embed(["text one", "text two"])          # list[list[float]]
        qvec = embedder.embed_query("what is kubernetes?")       # list[float]
    """

    def __init__(self, model: str = "gemini-embedding-001"):
        if not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai not installed.\n"
                "Run: pip install google-genai"
            )
        self._client = genai.Client(
            api_key=GEMINI_API_KEY,
        )
        self.model = model

    def _call_with_backoff(self, contents, task_type: str, max_retries: int = 3):
        """Call embed_content with exponential backoff on 429 / quota errors."""
        for attempt in range(max_retries):
            try:
                _rate_limiter.wait()
                return self._client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=genai_types.EmbedContentConfig(task_type=task_type),
                )
            except Exception as e:
                if "429" not in str(e) or attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[embedder] Rate limited (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait:.1f}s…")
                time.sleep(wait)

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """
        Embed a list of texts. Returns list[list[float]].
        Retries with exponential backoff on 429 quota errors.
        """
        if not texts:
            return []
        result = self._call_with_backoff(texts, task_type)
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (RETRIEVAL_QUERY for better query performance)."""
        result = self._call_with_backoff(text, "RETRIEVAL_QUERY")
        return result.embeddings[0].values


def get_llm_client() -> GeminiClient:
    """Factory function — returns GeminiClient. Import this in RAG systems."""
    return GeminiClient()


def get_langchain_llm(model: str = None):
    """
    Returns a LangChain-compatible Gemini LLM for RAGAS evaluation (D-002).
    RAGAS uses LangChain under the hood.
    """
    from config import JUDGE_MODEL
    model = model or JUDGE_MODEL
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(model=model, google_api_key=GEMINI_API_KEY, temperature=0)
        if _lc_rate_limiter is not None:
            kwargs["rate_limiter"] = _lc_rate_limiter
        return ChatGoogleGenerativeAI(**kwargs)
    except ImportError:
        raise ImportError(
            "langchain-google-genai not installed.\n"
            "Run: pip install langchain-google-genai"
        )
