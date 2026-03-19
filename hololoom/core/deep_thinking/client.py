"""
Deep Client — Async OpenAI-Compatible Inference Client
=======================================================

Talks to AirLLM GPU servers and Ollama via the same interface.
Long timeouts (AirLLM thinks in minutes). Retries on connection
errors but not on timeouts — if it's thinking, let it think.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Result from a chat completion request."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    raw: dict  # full response for debugging


@dataclass
class HealthStatus:
    """Health probe result from a GPU server."""
    alive: bool
    gpu_id: int | None = None
    status: str | None = None  # "ready" | "busy" | "loading" | "error"
    model: str | None = None
    uptime_s: float | None = None
    error: str | None = None


class DeepClient:
    """
    Async client for GPU servers and Ollama.

    Both expose OpenAI-compatible /v1/chat/completions.
    Ollama natively supports this at /v1/chat/completions.
    GPU servers expose it via our FastAPI wrapper.
    """

    def __init__(self, timeout_s: float = 900.0, retries: int = 1):
        self._timeout = timeout_s
        self._retries = retries

    async def chat(
        self,
        endpoint: str,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        timeout_s: float | None = None,
    ) -> ChatResult:
        """
        Send a chat completion request.

        Args:
            endpoint: Base URL (e.g., "http://RIG:8001" or "http://localhost:11434")
            model: Model name
            messages: List of {"role": ..., "content": ...} dicts
            max_tokens: Max output tokens
            temperature: Sampling temperature
            timeout_s: Override default timeout

        Returns:
            ChatResult with generated content and metadata
        """
        url = f"{endpoint.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        timeout = timeout_s or self._timeout

        for attempt in range(1 + self._retries):
            try:
                t0 = time.time()

                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()

                elapsed = time.time() - t0
                data = resp.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                logger.info(
                    f"DeepClient: {model}@{endpoint} → "
                    f"{usage.get('completion_tokens', '?')} tokens in {elapsed:.1f}s"
                )

                return ChatResult(
                    content=content,
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    elapsed_s=elapsed,
                    raw=data,
                )

            except httpx.ConnectError as e:
                # Connection failed — retry if we have attempts left
                if attempt < self._retries:
                    logger.warning(
                        f"DeepClient: Connect error to {endpoint}, "
                        f"retrying ({attempt + 1}/{self._retries}): {e}"
                    )
                    continue
                raise ConnectionError(
                    f"Cannot reach {endpoint} after {1 + self._retries} attempts: {e}"
                ) from e

            except httpx.TimeoutException:
                # Timeout — do NOT retry. AirLLM is slow; if it timed out
                # at 15 min, retrying won't help.
                raise TimeoutError(
                    f"Inference timed out after {timeout}s at {endpoint}"
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RuntimeError(f"GPU busy at {endpoint}") from e
                raise

    async def health(self, endpoint: str) -> HealthStatus:
        """Probe a server's health endpoint."""
        url = f"{endpoint.rstrip('/')}/health"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

                return HealthStatus(
                    alive=True,
                    gpu_id=data.get("gpu_id"),
                    status=data.get("status"),
                    model=data.get("model"),
                    uptime_s=data.get("uptime_s"),
                )

        except Exception as e:
            return HealthStatus(alive=False, error=str(e))

    async def is_alive(self, endpoint: str) -> bool:
        """Quick liveness check."""
        status = await self.health(endpoint)
        return status.alive and status.status in ("ready", "busy")
