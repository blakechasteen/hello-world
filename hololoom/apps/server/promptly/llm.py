"""
LLM client — Ollama + AirLLM, router-aware.

Provides call_model() which dispatches to the right backend based on
the ModelSpec from the model router.
"""
from __future__ import annotations

import logging
import re

import httpx
from fastapi import HTTPException

from .config import DEFAULT_MAX_TOKENS, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL

logger = logging.getLogger(__name__)

# Late import to avoid circular — model_router is a sibling in the server package
from ..model_router import Backend, ModelSpec

# ============================================================================
# Public API
# ============================================================================

async def call_model(
    messages: list[dict[str, str]],
    model_spec: ModelSpec | None = None,
    temperature: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Call an LLM. Dispatches to Ollama or AirLLM based on model_spec.backend.
    Falls back to default Ollama model if no spec provided.
    """
    # Suppress Qwen extended thinking across ALL backends
    messages = _inject_no_think(messages)
    backend_name = model_spec.backend.value if model_spec else "ollama(default)"
    model_name = model_spec.id if model_spec else OLLAMA_MODEL
    logger.info("call_model: backend=%s model=%s", backend_name, model_name)

    if model_spec is None:
        return await _call_ollama(messages, OLLAMA_MODEL, temperature, max_tokens)

    if model_spec.backend == Backend.OLLAMA:
        return await _call_ollama(
            messages, model_spec.id, temperature, max_tokens,
            base_url=model_spec.base_url,
        )
    elif model_spec.backend in (Backend.AIRLLM, Backend.OPENAI, Backend.LLAMACPP):
        return await _call_openai_compat(
            messages, model_spec, temperature, max_tokens,
        )
    else:
        return await _call_ollama(messages, OLLAMA_MODEL, temperature, max_tokens)


# ============================================================================
# Backends
# ============================================================================

async def _call_ollama(
    messages: list[dict[str, str]],
    model: str = OLLAMA_MODEL,
    temperature: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str | None = None,
) -> dict:
    """Call Ollama /api/chat directly."""
    url = base_url or OLLAMA_URL
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(f"{url}/api/chat", json=body)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned {resp.status_code}: {resp.text[:200]}",
        )

    data = resp.json()

    # Log thinking diagnostics
    thinking = data.get("message", {}).get("thinking", "")
    content = data.get("message", {}).get("content", "")
    think_in_content = bool(re.search(r"<think>", content))
    if thinking or think_in_content:
        logger.warning(
            "THINKING DETECTED model=%s thinking_field=%d content_has_think_tags=%s "
            "content_len=%d eval_count=%d",
            model, len(thinking), think_in_content, len(content),
            data.get("eval_count", 0),
        )

    return data


async def _call_openai_compat(
    messages: list[dict[str, str]],
    model_spec: ModelSpec,
    temperature: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Call an OpenAI-compatible endpoint (AirLLM GPU servers)."""
    # messages already have /no_think injected by call_model()
    body = {
        "model": model_spec.id,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if model_spec.api_key:
        headers["Authorization"] = f"Bearer {model_spec.api_key}"

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(
            f"{model_spec.base_url}/v1/chat/completions",
            json=body,
            headers=headers,
        )

    if resp.status_code == 429:
        raise HTTPException(status_code=503, detail=f"GPU busy: {model_spec.id}")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"AirLLM returned {resp.status_code}: {resp.text[:200]}",
        )

    data = resp.json()
    # Normalize to Ollama response shape for downstream compat
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return {
        "message": {"role": "assistant", "content": content},
        "eval_count": usage.get("completion_tokens", 0),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }


# ============================================================================
# Helpers
# ============================================================================

def _inject_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Append /no_think to the last user message to suppress Qwen extended thinking.

    Must be on its own line at the end of the message for Qwen to parse it.
    """
    if not messages:
        return messages
    patched = [m.copy() for m in messages]
    for m in reversed(patched):
        if m.get("role") == "user" and "/no_think" not in m.get("content", ""):
            m["content"] = m["content"].rstrip() + "\n/no_think"
            logger.debug("Injected /no_think into user message (%d chars)",
                         len(m["content"]))
            break
    return patched


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks, including unclosed <think> tags."""
    # Closed tags first
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    # Unclosed <think> — model ran out of tokens mid-thought
    text = re.sub(r"<think>[\s\S]*$", "", text)
    return text.strip()


def _classify_query_type(text: str) -> str:
    """Classify query into a type for MRF strategy selection."""
    lower = text.lower()
    if any(k in lower for k in ("```", "def ", "class ", "code", "bug", "error")):
        return "code"
    if any(k in lower for k in ("explain", "what is", "how does", "why")):
        return "factual"
    if any(k in lower for k in ("compare", "difference", "tradeoff", "vs")):
        return "analytical"
    if any(k in lower for k in ("design", "architect", "plan", "implement")):
        return "creative"
    return "general"


def _get_mrf():
    """Lazy-init MRF RefinementEngine + StrategySelector. Returns None if unavailable."""
    global _mrf_engine, _mrf_available
    if _mrf_available is False:
        return None
    if _mrf_engine is not None:
        return _mrf_engine
    try:
        from hololoom.prompting.unified_mrf import (
            RefinementEngine,
            RefinementStrategyType,
            StrategySelector,
        )
        _mrf_engine = {
            "refinement": RefinementEngine(),
            "selector": StrategySelector(),
            "strategies": RefinementStrategyType,
        }
        _mrf_available = True
        logger.info("MRF prompt refinement stack loaded")
        return _mrf_engine
    except Exception as e:
        _mrf_available = False
        logger.info("MRF unavailable, using fallback refinement: %s", e)
        return None


_mrf_engine = None
_mrf_available: bool | None = None
