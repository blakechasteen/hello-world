"""
Ollama Infer Client
====================

Direct adapter to a local Ollama instance. Implements the same interface
as DaemonInferClient so the Weaverlet loop can swap backends freely.

Uses the /api/chat endpoint with think=false to disable thinking mode
for thinking models (Qwen3.5, etc.), making inference fast and predictable.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from hololoom.weaverlet.shuttle.http_client import (
    DaemonUnavailableError,
    InferResponse,
)

logger = logging.getLogger(__name__)

# Default model — matches what's running on the dev machine.
DEFAULT_MODEL = "qwen3.5:9b"


class OllamaInferClient:
    """
    Drop-in replacement for DaemonInferClient that talks directly
    to a local Ollama instance.

    Disables thinking mode by default for consistent, fast responses.
    Set think=True in constructor to allow model reasoning (slower,
    needs higher max_tokens).

    Usage:
        async with OllamaInferClient() as client:
            resp = await client.infer("Hello world")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        think: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.think = think
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Inference ─────────────────────────────────────────────────────

    async def infer(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> InferResponse:
        """
        Inference via Ollama /api/chat.

        Uses think=false by default to skip internal reasoning, which
        makes thinking models (Qwen3.5) fast and token-efficient.
        """
        session = await self._get_session()
        use_model = model or self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        t0 = time.monotonic()

        try:
            async with session.post(
                f"{self.base_url}/api/chat", json=payload
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise DaemonUnavailableError(
                        f"Ollama /api/chat returned {resp.status}: {text}"
                    )
                data = await resp.json()
        except aiohttp.ClientError as e:
            raise DaemonUnavailableError(
                f"Cannot reach Ollama at {self.base_url}: {e}"
            )

        latency_ms = (time.monotonic() - t0) * 1000

        # Extract content from chat response
        message = data.get("message", {})
        content = message.get("content", "")

        tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return InferResponse(
            content=content,
            tokens_used=tokens_used,
            model=data.get("model", use_model),
            latency_ms=latency_ms,
        )

    # ── Health ────────────────────────────────────────────────────────

    async def health(self) -> bool:
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/tags") as resp:
                return resp.status == 200
        except aiohttp.ClientError:
            return False

    # ── Memory (no-op for Ollama) ─────────────────────────────────────

    async def store_memory(
        self, content: str, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """No-op — Ollama has no memory endpoint."""
        logger.debug("store_memory called on OllamaInferClient (no-op)")
        return {"status": "skipped", "reason": "ollama_direct"}

    async def recall_memory(
        self, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """No-op — Ollama has no memory endpoint."""
        return []

    # ── Context manager ───────────────────────────────────────────────

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
