#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Provider Abstraction Layer

Unified interface for multiple LLM providers:
- Anthropic (Claude)
- OpenAI (GPT-4)
- Ollama (local models)
- Google (Gemini) - future
- Cohere - future

Design Philosophy:
- Provider-agnostic API
- Graceful degradation (if one provider fails, try another)
- Cost tracking (token usage per provider)
- Performance tracking (latency, quality)
- Zero-config defaults (works out of the box)

Usage:
    provider = get_llm_provider("anthropic")
    response = await provider.generate(
        prompt="Explain quantum tunneling",
        max_tokens=500,
        temperature=0.7
    )
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Protocol
from dataclasses import dataclass, field
from enum import Enum
import time
import os
import asyncio
from abc import ABC, abstractmethod


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"  # Future
    COHERE = "cohere"  # Future


@dataclass
class LLMResponse:
    """Unified LLM response"""

    # Generated content
    text: str

    # Metadata
    provider: str
    model: str

    # Token usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Performance
    latency_ms: float

    # Quality indicators (if available)
    finish_reason: str  # "stop", "length", "content_filter", etc.

    # Raw response (for debugging)
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def cost_estimate_usd(self) -> float:
        """
        Estimate cost in USD based on provider pricing.

        Approximate pricing (as of Nov 2025):
        - Anthropic Claude 3.5 Sonnet: $3/$15 per 1M tokens (in/out)
        - OpenAI GPT-4 Turbo: $10/$30 per 1M tokens
        - Ollama: Free (local)
        """
        if self.provider == "ollama":
            return 0.0

        # Anthropic pricing
        if self.provider == "anthropic":
            input_cost = (self.prompt_tokens / 1_000_000) * 3.0
            output_cost = (self.completion_tokens / 1_000_000) * 15.0
            return input_cost + output_cost

        # OpenAI pricing
        if self.provider == "openai":
            input_cost = (self.prompt_tokens / 1_000_000) * 10.0
            output_cost = (self.completion_tokens / 1_000_000) * 30.0
            return input_cost + output_cost

        return 0.0


@dataclass
class LLMGenerationConfig:
    """Configuration for LLM generation"""

    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None

    # Stop sequences
    stop_sequences: List[str] = field(default_factory=list)

    # Provider-specific options
    extra_options: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize provider.

        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            api_key: API key (optional, can use env var)
            base_url: Base URL for API (optional)
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        # Performance tracking
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost_usd = 0.0
        self._total_latency_ms = 0.0

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[LLMGenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from prompt"""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            "provider": self.__class__.__name__,
            "model": self.model,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost_usd,
            "avg_latency_ms": (
                self._total_latency_ms / self._total_requests
                if self._total_requests > 0 else 0.0
            )
        }


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider"""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None
    ):
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        # Lazy import
        self._client = None

    async def generate(
        self,
        prompt: str,
        config: Optional[LLMGenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using Anthropic API"""

        # Default config
        if config is None:
            config = LLMGenerationConfig()

        # Lazy import and initialize
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )

        start_time = time.time()

        # Call Anthropic API
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_sequences=config.stop_sequences,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **config.extra_options
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            text = response.content[0].text

            # Build unified response
            llm_response = LLMResponse(
                text=text,
                provider="anthropic",
                model=self.model,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=latency_ms,
                finish_reason=response.stop_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )

            # Update statistics
            self._total_requests += 1
            self._total_tokens += llm_response.total_tokens
            self._total_cost_usd += llm_response.cost_estimate_usd
            self._total_latency_ms += latency_ms

            return llm_response

        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None
    ):
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

        # Lazy import
        self._client = None

    async def generate(
        self,
        prompt: str,
        config: Optional[LLMGenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using OpenAI API"""

        # Default config
        if config is None:
            config = LLMGenerationConfig()

        # Lazy import and initialize
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )

        start_time = time.time()

        # Call OpenAI API
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences if config.stop_sequences else None,
                **config.extra_options
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            text = response.choices[0].message.content

            # Build unified response
            llm_response = LLMResponse(
                text=text,
                provider="openai",
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )

            # Update statistics
            self._total_requests += 1
            self._total_tokens += llm_response.total_tokens
            self._total_cost_usd += llm_response.cost_estimate_usd
            self._total_latency_ms += latency_ms

            return llm_response

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider"""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434"
    ):
        super().__init__(
            model=model,
            base_url=base_url
        )

        # Lazy import
        self._client = None

    async def generate(
        self,
        prompt: str,
        config: Optional[LLMGenerationConfig] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using Ollama API"""

        # Default config
        if config is None:
            config = LLMGenerationConfig()

        # Lazy import and initialize
        if self._client is None:
            try:
                from ollama import AsyncClient
                self._client = AsyncClient(host=self.base_url)
            except ImportError:
                raise ImportError(
                    "ollama package not installed. "
                    "Install with: pip install ollama"
                )

        start_time = time.time()

        # Call Ollama API
        try:
            response = await self._client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k if config.top_k else 40,
                    "stop": config.stop_sequences,
                    **config.extra_options
                }
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            text = response['response']

            # Estimate tokens (Ollama doesn't provide exact counts)
            # Rough estimate: 1 token ≈ 4 characters
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(text) // 4

            # Build unified response
            llm_response = LLMResponse(
                text=text,
                provider="ollama",
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                finish_reason="stop",  # Ollama doesn't provide this
                raw_response=response
            )

            # Update statistics
            self._total_requests += 1
            self._total_tokens += llm_response.total_tokens
            self._total_latency_ms += latency_ms

            return llm_response

        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


# Provider factory
def get_llm_provider(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> BaseLLMProvider:
    """
    Factory function to get LLM provider.

    Args:
        provider: Provider name ("anthropic", "openai", "ollama")
        model: Model identifier (optional, uses default)
        api_key: API key (optional, uses env var)
        base_url: Base URL (optional, for Ollama)

    Returns:
        BaseLLMProvider instance

    Examples:
        # Anthropic
        provider = get_llm_provider("anthropic")
        provider = get_llm_provider("anthropic", model="claude-3-5-sonnet-20241022")

        # OpenAI
        provider = get_llm_provider("openai")
        provider = get_llm_provider("openai", model="gpt-4-turbo-preview")

        # Ollama (local)
        provider = get_llm_provider("ollama")
        provider = get_llm_provider("ollama", model="llama3.2:3b")
    """
    provider_lower = provider.lower()

    if provider_lower == "anthropic":
        return AnthropicProvider(
            model=model or "claude-3-5-sonnet-20241022",
            api_key=api_key
        )

    elif provider_lower == "openai":
        return OpenAIProvider(
            model=model or "gpt-4-turbo-preview",
            api_key=api_key
        )

    elif provider_lower == "ollama":
        return OllamaProvider(
            model=model or "llama3.2:3b",
            base_url=base_url or "http://localhost:11434"
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: anthropic, openai, ollama"
        )


# Graceful fallback chain
async def generate_with_fallback(
    prompt: str,
    providers: List[str] = ["ollama", "anthropic", "openai"],
    config: Optional[LLMGenerationConfig] = None,
    **kwargs
) -> LLMResponse:
    """
    Try multiple providers in order until one succeeds.

    Default order: Ollama (free) → Anthropic → OpenAI

    Args:
        prompt: Prompt to generate from
        providers: List of provider names to try in order
        config: Generation config
        **kwargs: Additional provider-specific kwargs

    Returns:
        LLMResponse from first successful provider

    Raises:
        RuntimeError: If all providers fail
    """
    errors = []

    for provider_name in providers:
        try:
            provider = get_llm_provider(provider_name, **kwargs)
            return await provider.generate(prompt, config)
        except Exception as e:
            errors.append(f"{provider_name}: {e}")
            continue

    # All providers failed
    raise RuntimeError(
        f"All LLM providers failed. Errors:\n" +
        "\n".join(f"- {err}" for err in errors)
    )
