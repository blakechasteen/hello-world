"""
LLM backend factories — Ollama + router-aware dispatch.

Each factory returns an object conforming to the LLMBackend protocol:
an async .call(messages, **kwargs) -> dict that returns an Ollama-shaped response.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol
# ============================================================================

@runtime_checkable
class LLMBackend(Protocol):
    """LLM backend protocol.

    call() must return a dict with this shape (Ollama-compatible):
        {
            "message": {"content": "response text"},
            "model": "model-name",           # optional, defaults to "unknown"
            "eval_count": 123,               # optional, output tokens
            "prompt_eval_count": 456,        # optional, input tokens
        }

    extract_content() and extract_tokens() parse this shape.
    Non-Ollama backends should normalize their response to match.
    """
    async def call(self, messages: list[dict[str, str]], **kwargs) -> dict: ...


# ============================================================================
# Ollama backend
# ============================================================================

def _resolve_ollama_url(url: str | None = None) -> str:
    """Resolve Ollama URL from arg, env, or default."""
    if url:
        return url if url.startswith("http") else f"http://{url}"
    raw = os.environ.get("PROMPTLY_OLLAMA_URL", "").strip()
    if not raw:
        raw = "http://127.0.0.1:11434"
    return raw if raw.startswith("http") else f"http://{raw}"


class OllamaBackend:
    """Direct Ollama /api/chat backend."""

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        temperature: float = 0.7,
        url: str | None = None,
        timeout: int = 120,
        max_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.url = _resolve_ollama_url(url)
        self.timeout = timeout
        self.max_tokens = max_tokens

    async def call(self, messages: list[dict[str, str]], **kwargs) -> dict:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Inject /no_think into last user message — Qwen ignores API-level
        # think:false and system prompt placement
        patched = _inject_no_think(messages)
        body = {
            "model": self.model,
            "messages": patched,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.url}/api/chat", json=body)

        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Ollama returned {resp.status_code}: {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )

        data = resp.json()

        # Log if model produces thinking tokens despite think:false
        thinking = data.get("message", {}).get("thinking", "")
        if thinking:
            logger.warning(
                "Model %s produced %d thinking chars despite think:false — stripping",
                self.model, len(thinking),
            )

        return data


def ollama(
    model: str = "qwen3.5:9b",
    temperature: float = 0.7,
    url: str | None = None,
    timeout: int = 120,
    max_tokens: int = 8192,
) -> OllamaBackend:
    """Factory: create an Ollama LLM backend."""
    return OllamaBackend(
        model=model, temperature=temperature,
        url=url, timeout=timeout, max_tokens=max_tokens,
    )


# ============================================================================
# Helpers
# ============================================================================

def _inject_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Append /no_think to the last user message to suppress Qwen extended thinking."""
    if not messages:
        return messages
    patched = [m.copy() for m in messages]
    for m in reversed(patched):
        if m.get("role") == "user" and "/no_think" not in m.get("content", ""):
            m["content"] = m["content"].rstrip() + "\n/no_think"
            break
    return patched


def extract_content(data: dict) -> str:
    """Extract text from an Ollama/OpenAI-compat response dict."""
    msg = data.get("message", {})
    content = msg.get("content", "")
    # Strip thinking tokens (closed and unclosed)
    if content:
        content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content)
        content = re.sub(r"<think>[\s\S]*$", "", content)
        content = content.strip()
    return content or msg.get("thinking", "")


def extract_tokens(data: dict) -> int:
    """Extract total token count from response dict."""
    return data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
