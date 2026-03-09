"""
groq_client.py — Groq API LLM client (drop-in for GeminiClient).

Uses Groq's OpenAI-compatible interface, so the response shape is identical:
  response.choices[0].message.content
  response.usage.prompt_tokens / completion_tokens

Free tier limits (as of 2025):
  llama-3.1-8b-instant:     14,400 req/day, 20,000 tokens/min  ← generator
  llama-3.3-70b-versatile:  14,400 req/day,  6,000 tokens/min  ← judge

Get a free API key at https://console.groq.com

Usage:
    from groq_client import GroqClient
    client = GroqClient()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
        max_tokens=512,
    )
    text = response.choices[0].message.content
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


class GroqClient:
    """
    Drop-in replacement for GeminiClient using Groq's OpenAI-compatible API.

    Swap:
        # Before:
        from gemini_client import GeminiClient
        client = GeminiClient()

        # After:
        from groq_client import GroqClient
        client = GroqClient()

    Interface is identical: client.chat.completions.create(model=..., messages=[...])
    Requires GROQ_API_KEY environment variable.
    Get a free key at https://console.groq.com
    """

    def __init__(self, api_key: str = None):
        from groq import Groq
        self._client = Groq(api_key=api_key or GROQ_API_KEY)
        self.chat = self._client.chat  # groq.chat.completions is OpenAI-compatible


def get_langchain_llm(model: str = None):
    """
    Returns a LangChain-compatible Groq LLM for RAGAS evaluation.
    Requires langchain-groq: pip install langchain-groq

    Groq only allows n=1 per request. RAGAS faithfulness passes n=k (number of
    NLI statements) to get k verdicts in one shot. We handle this by making k
    sequential calls and combining the results into a single LLMResult — same
    shape as if n=k had worked.
    """
    from config import LLM_MODEL
    from langchain_groq import ChatGroq

    class _GroqNoN(ChatGroq):
        """Simulate n > 1 via sequential calls — Groq only allows n=1."""

        @property
        def _default_params(self):
            params = super()._default_params
            params.pop("n", None)
            return params

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.pop("n", None)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.pop("n", None)
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        async def agenerate(self, messages, stop=None, callbacks=None, **kwargs):
            from langchain_core.outputs import LLMResult
            import asyncio

            n = kwargs.pop("n", 1)
            if n <= 1:
                return await super().agenerate(
                    messages, stop=stop, callbacks=callbacks, **kwargs
                )

            # Groq rejects n > 1 — simulate with n sequential single calls
            results = []
            for _ in range(n):
                r = await super().agenerate(
                    messages, stop=stop, callbacks=None, **kwargs
                )
                results.append(r)
                await asyncio.sleep(0.1)   # small pause to respect rate limits

            # Combine: generations[i] should be a list of n ChatGenerations
            combined = [
                [r.generations[i][0] for r in results]
                for i in range(len(messages))
            ]
            return LLMResult(generations=combined)

    return _GroqNoN(
        model=model or LLM_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
        n=1,
    )
