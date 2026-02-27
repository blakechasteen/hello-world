"""
Jenny LLM Client - Production-Grade Async LLM Integration
==========================================================
Elegant, tastefully concurrent LLM client with connection pooling and retry logic.

Date: December 2025
Status: Phase M2 Implementation

Features:
- True async HTTP clients (httpx)
- Connection pooling for performance
- Exponential backoff with jitter
- Configurable timeouts
- Multi-provider support (Ollama, Anthropic, OpenAI)
- Graceful degradation on failures

Philosophy:
> "Make LLM calls as reliable as database queries."

Usage:
    from HoloLoom.visualization.jenny_llm_client import (
        AsyncLLMClient,
        create_llm_client,
    )

    async with create_llm_client("ollama", "llama3.2:3b") as client:
        response = await client.generate("What is the best panel type?")
"""

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, AsyncIterator, Type

from .jenny_config_base import BaseConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LLMClientConfig(BaseConfig):
    """Configuration for LLM client."""
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 10.0
    jitter_factor: float = 0.1
    max_tokens: int = 500
    temperature: float = 0.7
    # Connection pool settings
    max_connections: int = 10
    keepalive_expiry: float = 30.0
    # Provider-specific settings
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    @classmethod
    def fast(cls) -> 'LLMClientConfig':
        """Fast config for real-time UI."""
        return cls(
            timeout_seconds=5.0,
            max_retries=1,
            max_tokens=200,
        )

    @classmethod
    def reliable(cls) -> 'LLMClientConfig':
        """Reliable config with retries."""
        return cls(
            timeout_seconds=30.0,
            max_retries=3,
            max_tokens=500,
        )


# ============================================================================
# Response Types
# ============================================================================

@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    model: str
    provider: str
    latency_ms: float
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    retries: int = 0


@dataclass
class LLMError:
    """LLM error details."""
    error_type: str
    message: str
    provider: str
    retriable: bool = True
    status_code: Optional[int] = None


# ============================================================================
# Base Client
# ============================================================================

class AsyncLLMClientBase(ABC):
    """
    Abstract base class for async LLM clients.

    Implements:
    - Connection pooling via httpx.AsyncClient
    - Exponential backoff with jitter
    - Timeout management
    - Statistics tracking
    """

    def __init__(self, config: LLMClientConfig):
        self.config = config
        self._client = None
        self._available = False
        self._stats = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "total_latency_ms": 0.0,
        }

    @property
    def available(self) -> bool:
        return self._available

    @property
    def stats(self) -> Dict[str, Any]:
        calls = max(self._stats["calls"], 1)
        return {
            **self._stats,
            "success_rate": self._stats["successes"] / calls,
            "avg_latency_ms": self._stats["total_latency_ms"] / calls,
        }

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @abstractmethod
    async def connect(self) -> bool:
        """Initialize connection. Returns True if available."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup."""
        ...

    @abstractmethod
    async def _generate_impl(self, prompt: str) -> LLMResponse:
        """Provider-specific generation implementation."""
        ...

    async def generate(self, prompt: str) -> Optional[LLMResponse]:
        """
        Generate response with retry logic.

        Returns None on failure after all retries.
        """
        if not self._available:
            logger.warning(f"{self.__class__.__name__} not available")
            return None

        self._stats["calls"] += 1
        start_time = time.monotonic()
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._generate_impl(prompt),
                    timeout=self.config.timeout_seconds
                )

                # Success
                latency_ms = (time.monotonic() - start_time) * 1000
                response.latency_ms = latency_ms
                response.retries = attempt

                self._stats["successes"] += 1
                self._stats["total_latency_ms"] += latency_ms

                logger.debug(
                    f"LLM response: {len(response.content)} chars, "
                    f"{latency_ms:.0f}ms, {attempt} retries"
                )
                return response

            except asyncio.TimeoutError:
                last_error = LLMError(
                    error_type="timeout",
                    message=f"Timeout after {self.config.timeout_seconds}s",
                    provider=self.config.provider,
                    retriable=True,
                )
            except Exception as e:
                last_error = LLMError(
                    error_type=type(e).__name__,
                    message=str(e),
                    provider=self.config.provider,
                    retriable=self._is_retriable(e),
                )

            # Check if we should retry
            if attempt < self.config.max_retries and last_error.retriable:
                self._stats["retries"] += 1
                delay = self._calculate_backoff(attempt)
                logger.warning(
                    f"LLM attempt {attempt + 1} failed: {last_error.message}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
            else:
                break

        # All retries exhausted
        self._stats["failures"] += 1
        logger.error(f"LLM generation failed after {self.config.max_retries + 1} attempts")
        return None

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff delay with exponential backoff and jitter."""
        base = self.config.base_delay_seconds * (2 ** attempt)
        capped = min(base, self.config.max_delay_seconds)
        jitter = capped * self.config.jitter_factor * random.random()
        return capped + jitter

    def _is_retriable(self, error: Exception) -> bool:
        """Check if error is retriable."""
        # Network errors are retriable
        retriable_types = (
            ConnectionError, TimeoutError, asyncio.TimeoutError,
        )
        if isinstance(error, retriable_types):
            return True

        # Check for rate limiting (usually retriable)
        error_str = str(error).lower()
        if "rate" in error_str or "429" in error_str:
            return True

        # Server errors are retriable
        if "500" in error_str or "502" in error_str or "503" in error_str:
            return True

        return False


# ============================================================================
# Ollama Client
# ============================================================================

class OllamaClient(AsyncLLMClientBase):
    """
    Async Ollama client using httpx.

    Ollama runs locally, so connection pooling helps with
    repeated requests to the same server.
    """

    async def connect(self) -> bool:
        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.config.ollama_base_url,
                timeout=httpx.Timeout(self.config.timeout_seconds),
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_connections,
                    keepalive_expiry=self.config.keepalive_expiry,
                ),
            )

            # Test connection by listing models
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if any(self.config.model in name for name in model_names):
                    self._available = True
                    logger.info(f"Ollama connected, model {self.config.model} available")
                    return True
                else:
                    logger.warning(f"Ollama model {self.config.model} not found")
                    # Still mark as available if Ollama is running
                    self._available = True
                    return True
            else:
                logger.warning(f"Ollama returned {response.status_code}")

        except ImportError:
            logger.warning("httpx not installed - Ollama client unavailable")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")

        return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _generate_impl(self, prompt: str) -> LLMResponse:
        response = await self._client.post(
            "/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data.get("response", ""),
            model=data.get("model", self.config.model),
            provider="ollama",
            latency_ms=0,  # Will be filled by caller
            tokens_used=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
            raw_response=data,
        )


# ============================================================================
# Anthropic Client
# ============================================================================

class AnthropicClient(AsyncLLMClientBase):
    """
    Async Anthropic (Claude) client using httpx.
    """

    async def connect(self) -> bool:
        try:
            import httpx
            import os

            api_key = self.config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found")
                return False

            self._client = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_connections,
                ),
            )

            self._available = True
            logger.info(f"Anthropic client initialized with model {self.config.model}")
            return True

        except ImportError:
            logger.warning("httpx not installed - Anthropic client unavailable")
        except Exception as e:
            logger.warning(f"Anthropic connection failed: {e}")

        return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _generate_impl(self, prompt: str) -> LLMResponse:
        model = self.config.model or "claude-3-5-sonnet-20241022"

        response = await self._client.post(
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

        content = ""
        if data.get("content"):
            content = data["content"][0].get("text", "")

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            provider="anthropic",
            latency_ms=0,
            tokens_used=data.get("usage", {}).get("output_tokens"),
            finish_reason=data.get("stop_reason"),
            raw_response=data,
        )


# ============================================================================
# OpenAI Client
# ============================================================================

class OpenAIClient(AsyncLLMClientBase):
    """
    Async OpenAI client using httpx.
    """

    async def connect(self) -> bool:
        try:
            import httpx
            import os

            api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found")
                return False

            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com",
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_connections,
                ),
            )

            self._available = True
            logger.info(f"OpenAI client initialized with model {self.config.model}")
            return True

        except ImportError:
            logger.warning("httpx not installed - OpenAI client unavailable")
        except Exception as e:
            logger.warning(f"OpenAI connection failed: {e}")

        return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _generate_impl(self, prompt: str) -> LLMResponse:
        model = self.config.model or "gpt-4"

        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            provider="openai",
            latency_ms=0,
            tokens_used=data.get("usage", {}).get("completion_tokens"),
            finish_reason=data["choices"][0].get("finish_reason") if data.get("choices") else None,
            raw_response=data,
        )


# ============================================================================
# Factory Functions
# ============================================================================

async def create_llm_client(
    provider: str = "ollama",
    model: Optional[str] = None,
    config: Optional[LLMClientConfig] = None,
) -> AsyncLLMClientBase:
    """
    Create and connect an async LLM client.

    Args:
        provider: LLM provider ("ollama", "anthropic", "openai")
        model: Model name (uses provider default if None)
        config: Full configuration (overrides provider/model)

    Returns:
        Connected LLM client

    Example:
        async with await create_llm_client("ollama", "llama3.2:3b") as client:
            response = await client.generate("Hello!")
    """
    if config is None:
        config = LLMClientConfig(provider=provider)
        if model:
            config.model = model
        else:
            # Provider defaults
            defaults = {
                "ollama": "llama3.2:3b",
                "anthropic": "claude-3-5-sonnet-20241022",
                "openai": "gpt-4",
            }
            config.model = defaults.get(provider, "llama3.2:3b")

    # Create appropriate client
    clients = {
        "ollama": OllamaClient,
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
    }

    client_class = clients.get(provider, OllamaClient)
    client = client_class(config)
    await client.connect()

    return client


async def create_best_available_client(
    prefer_local: bool = True,
    config: Optional[LLMClientConfig] = None,
) -> Optional[AsyncLLMClientBase]:
    """
    Create the best available LLM client.

    Tries providers in order:
    - If prefer_local: Ollama -> Anthropic -> OpenAI
    - Otherwise: Anthropic -> OpenAI -> Ollama

    Args:
        prefer_local: Prefer local Ollama over cloud providers
        config: Base configuration to use

    Returns:
        Best available client or None if none available
    """
    config = config or LLMClientConfig()

    providers = (
        ["ollama", "anthropic", "openai"]
        if prefer_local
        else ["anthropic", "openai", "ollama"]
    )

    for provider in providers:
        try:
            config.provider = provider
            client = await create_llm_client(provider=provider, config=config)
            if client.available:
                logger.info(f"Using {provider} as best available LLM provider")
                return client
            await client.close()
        except Exception as e:
            logger.debug(f"Provider {provider} not available: {e}")

    logger.warning("No LLM provider available")
    return None


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Config
    'LLMClientConfig',
    # Response types
    'LLMResponse',
    'LLMError',
    # Base class
    'AsyncLLMClientBase',
    # Implementations
    'OllamaClient',
    'AnthropicClient',
    'OpenAIClient',
    # Factory
    'create_llm_client',
    'create_best_available_client',
]
