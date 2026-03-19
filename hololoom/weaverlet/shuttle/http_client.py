"""
Daemon Infer Client
====================

Async HTTP client to the HoloLoom daemon API.

Endpoints:
    POST /infer          -> model inference (routed by daemon)
    GET  /health         -> daemon status
    POST /memory/store   -> persist memory item
    POST /memory/recall  -> query memory
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class DaemonUnavailableError(Exception):
    """Daemon is not reachable."""
    pass


@dataclass
class InferResponse:
    content: str
    tokens_used: int
    model: str
    latency_ms: float


class DaemonInferClient:

    def __init__(
        self,
        base_url: str = "http://localhost:4243",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
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
        """POST /infer — daemon routes to best available model."""
        session = await self._get_session()
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            payload["model"] = model
        if system_prompt:
            # Prepend system prompt to prompt (daemon API is prompt-only)
            payload["prompt"] = f"{system_prompt}\n\n{prompt}"

        try:
            async with session.post(
                f"{self.base_url}/infer", json=payload
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise DaemonUnavailableError(
                        f"Daemon /infer returned {resp.status}: {text}"
                    )
                data = await resp.json()
                return InferResponse(
                    content=data.get("response", data.get("content", "")),
                    tokens_used=data.get("tokens_used", 0),
                    model=data.get("model", "unknown"),
                    latency_ms=data.get("latency_ms", 0.0),
                )
        except aiohttp.ClientError as e:
            raise DaemonUnavailableError(f"Cannot reach daemon at {self.base_url}: {e}")

    # ── Health ────────────────────────────────────────────────────────

    async def health(self) -> bool:
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except aiohttp.ClientError:
            return False

    # ── Memory ────────────────────────────────────────────────────────

    async def store_memory(
        self, content: str, tags: list[str] | None = None
    ) -> dict[str, Any]:
        session = await self._get_session()
        payload = {"content": content}
        if tags:
            payload["tags"] = tags
        try:
            async with session.post(
                f"{self.base_url}/memory/store", json=payload
            ) as resp:
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.warning(f"Memory store failed: {e}")
            return {"error": str(e)}

    async def recall_memory(
        self, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/memory/recall",
                json={"query": query, "limit": limit},
            ) as resp:
                data = await resp.json()
                return data.get("memories", data.get("results", []))
        except aiohttp.ClientError as e:
            logger.warning(f"Memory recall failed: {e}")
            return []

    # ── Context manager ───────────────────────────────────────────────

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
