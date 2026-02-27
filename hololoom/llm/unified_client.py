"""
Unified LLM Client
==================

Single interface for multiple LLM providers with automatic fallback,
cost tracking, and A/B testing capabilities.

Features:
- Unified interface across providers (Ollama, Anthropic, OpenAI, Google)
- Automatic fallback on failure
- Cost tracking and budget limits
- Model comparison for A/B testing
- Graceful degradation

Created: 2025-01-20
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

from hololoom.llm.cost_tracker import CostTracker, CostEstimate

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"  # Gemini


@dataclass
class LLMConfig:
    """
    Configuration for an LLM model.

    Args:
        provider: Provider name (ollama/anthropic/openai/google)
        model: Model name (e.g., "claude-3-5-sonnet-20241022")
        api_key: Optional API key (uses env var if not provided)
        base_url: Optional custom base URL
        timeout: Request timeout in seconds
    """
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0

    def __post_init__(self):
        """Validate and normalize configuration"""
        self.provider = self.provider.lower()

        # Get API key from environment if not provided
        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "google":
                self.api_key = os.getenv("GOOGLE_API_KEY")

    @property
    def full_name(self) -> str:
        """Get full model name (provider/model)"""
        return f"{self.provider}/{self.model}"


@dataclass
class LLMResponse:
    """
    Response from LLM generation.

    Args:
        content: Generated text
        model: Model used
        provider: Provider used
        input_tokens: Input token count
        output_tokens: Output token count
        cost_estimate: Cost estimate
        metadata: Additional metadata
    """
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_estimate: Optional[CostEstimate] = None
    metadata: Optional[Dict[str, Any]] = None


class UnifiedLLMClient:
    """
    Unified LLM client supporting multiple providers.

    Handles automatic fallback, cost tracking, and model comparison.

    Usage:
        # Simple usage
        client = UnifiedLLMClient.create_default()
        response = await client.complete("What is recursion?")

        # With custom config
        primary = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        fallback = LLMConfig(provider="ollama", model="llama3.2:3b")
        client = UnifiedLLMClient(primary=primary, fallbacks=[fallback])

        # Compare models
        results = await client.compare_models(
            "Explain Thompson Sampling",
            models=["ollama/llama3.2:3b", "anthropic/claude-3-5-sonnet"]
        )
    """

    def __init__(
        self,
        primary: LLMConfig,
        fallbacks: Optional[List[LLMConfig]] = None,
        enable_cost_tracking: bool = True,
        max_cost_per_query: Optional[float] = None
    ):
        """
        Initialize unified LLM client.

        Args:
            primary: Primary LLM configuration
            fallbacks: Optional fallback LLMs (tried in order)
            enable_cost_tracking: Enable cost tracking
            max_cost_per_query: Maximum cost per query (USD)
        """
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.enable_cost_tracking = enable_cost_tracking
        self.max_cost_per_query = max_cost_per_query

        # Initialize cost tracker
        self.cost_tracker = CostTracker() if enable_cost_tracking else None

        # Initialize provider clients
        self._clients: Dict[str, Any] = {}

        logger.info(
            f"UnifiedLLMClient initialized: primary={primary.full_name}, "
            f"fallbacks={[f.full_name for f in self.fallbacks]}"
        )

    @classmethod
    def create_default(cls) -> "UnifiedLLMClient":
        """
        Create client with default configuration.

        Tries Ollama first (free), falls back to Claude if API key available.
        """
        primary = LLMConfig(provider="ollama", model="llama3.2:3b")
        fallbacks = []

        # Add Claude fallback if API key available
        if os.getenv("ANTHROPIC_API_KEY"):
            fallbacks.append(
                LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
            )

        return cls(primary=primary, fallbacks=fallbacks)

    def _get_client(self, config: LLMConfig) -> Optional[Any]:
        """
        Get or create provider client.

        Args:
            config: LLM configuration

        Returns:
            Provider client or None if unavailable
        """
        key = config.full_name
        if key in self._clients:
            return self._clients[key]

        # Import and create provider client
        try:
            if config.provider == "ollama":
                from hololoom.llm.providers.ollama_provider import OllamaProvider
                client = OllamaProvider(config)
            elif config.provider == "anthropic":
                from hololoom.llm.providers.anthropic_provider import AnthropicProvider
                client = AnthropicProvider(config)
            elif config.provider == "openai":
                from hololoom.llm.providers.openai_provider import OpenAIProvider
                client = OpenAIProvider(config)
            elif config.provider == "google":
                from hololoom.llm.providers.gemini_provider import GeminiProvider
                client = GeminiProvider(config)
            else:
                logger.error(f"Unknown provider: {config.provider}")
                return None

            # Check availability
            if not client.is_available():
                logger.warning(f"Provider {config.full_name} not available")
                return None

            self._clients[key] = client
            logger.info(f"✓ Initialized {config.full_name}")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize {config.full_name}: {e}")
            return None

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        model_override: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from prompt.

        Tries primary model first, falls back on failure.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            model_override: Override primary model (e.g., "anthropic/claude-3-haiku")
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with generated content

        Raises:
            RuntimeError: If all models fail
        """
        # Parse model override
        configs = [self.primary] + self.fallbacks
        if model_override:
            # Parse "provider/model" format
            parts = model_override.split("/", 1)
            if len(parts) == 2:
                override_config = LLMConfig(provider=parts[0], model=parts[1])
                configs = [override_config] + configs

        # Check cost limit before calling
        if self.max_cost_per_query and self.cost_tracker:
            for config in configs:
                estimate = self.cost_tracker.estimate_query_cost(
                    query=prompt,
                    model=config.model,
                    max_output_tokens=max_tokens
                )
                if estimate.total_cost_usd > self.max_cost_per_query:
                    logger.warning(
                        f"Skipping {config.full_name}: estimated cost "
                        f"${estimate.total_cost_usd:.4f} exceeds limit "
                        f"${self.max_cost_per_query:.4f}"
                    )
                    continue

        # Try each config in order
        last_error = None
        for config in configs:
            try:
                client = self._get_client(config)
                if client is None:
                    continue

                logger.info(f"Trying {config.full_name}...")
                response = await client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )

                # Track cost
                if self.cost_tracker:
                    cost_estimate = self.cost_tracker.track_call(
                        model=config.model,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens
                    )
                    response.cost_estimate = cost_estimate

                logger.info(
                    f"✓ {config.full_name}: {response.output_tokens} tokens, "
                    f"${response.cost_estimate.total_cost_usd:.4f}" if response.cost_estimate else ""
                )

                return response

            except Exception as e:
                logger.warning(f"✗ {config.full_name} failed: {e}")
                last_error = e
                continue

        # All models failed
        raise RuntimeError(
            f"All LLM models failed. Last error: {last_error}"
        )

    async def compare_models(
        self,
        prompt: str,
        models: List[str],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, LLMResponse]:
        """
        Run same prompt on multiple models for A/B testing.

        Args:
            prompt: Input prompt
            models: List of models (format: "provider/model")
            max_tokens: Maximum tokens per model
            temperature: Sampling temperature

        Returns:
            Dict mapping model name to response

        Example:
            results = await client.compare_models(
                "Explain recursion",
                models=["ollama/llama3.2:3b", "anthropic/claude-3-5-sonnet"]
            )
            for model, response in results.items():
                print(f"{model}: {response.content[:100]}...")
        """
        tasks = []
        model_names = []

        for model_spec in models:
            # Parse "provider/model" format
            parts = model_spec.split("/", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid model spec: {model_spec}")
                continue

            provider, model = parts
            config = LLMConfig(provider=provider, model=model)
            model_names.append(model_spec)

            # Create task
            async def _generate_with_config(cfg: LLMConfig):
                client = self._get_client(cfg)
                if client is None:
                    raise RuntimeError(f"Client {cfg.full_name} not available")
                return await client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

            tasks.append(_generate_with_config(config))

        # Run all models in parallel
        logger.info(f"Comparing {len(tasks)} models...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dict
        comparison = {}
        for model_name, result in zip(model_names, results):
            if isinstance(result, Exception):
                logger.error(f"{model_name} failed: {result}")
                continue

            # Track cost
            if self.cost_tracker:
                cost_estimate = self.cost_tracker.track_call(
                    model=result.model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens
                )
                result.cost_estimate = cost_estimate

            comparison[model_name] = result

        logger.info(f"✓ Compared {len(comparison)}/{len(models)} models")
        return comparison

    def get_cost_statistics(self) -> Optional[Dict]:
        """
        Get cost tracking statistics.

        Returns:
            Dict with cost breakdown or None if tracking disabled
        """
        if self.cost_tracker:
            return self.cost_tracker.get_statistics()
        return None

    def reset_cost_tracking(self):
        """Reset cost tracking statistics"""
        if self.cost_tracker:
            self.cost_tracker.reset()

    def list_available_models(self) -> List[str]:
        """
        List all available models.

        Returns:
            List of model names (format: "provider/model")
        """
        available = []

        # Try primary
        if self._get_client(self.primary):
            available.append(self.primary.full_name)

        # Try fallbacks
        for fallback in self.fallbacks:
            if self._get_client(fallback):
                available.append(fallback.full_name)

        return available
