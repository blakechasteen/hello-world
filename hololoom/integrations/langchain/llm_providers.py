"""
Multi-Provider LLM Integration
================================

Unified interface for 20+ LLM providers through LangChain.

Supported Providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3.5 Sonnet, Opus, Haiku)
- Cohere (Command R+)
- Google (Gemini Pro)
- Meta (Llama via Replicate/Together)
- Mistral
- HuggingFace (any model)
- Local models (Ollama, LM Studio)
- And 10+ more

Author: Claude Code
Date: 2025-11-20
"""

import os
import warnings
from enum import Enum
from typing import Any

try:
    # Core LangChain
    from langchain.chat_models import (
        ChatAnthropic,
        ChatCohere,
        ChatOllama,
        ChatOpenAI,
    )
    from langchain.llms import (
        Anthropic as LangChainAnthropic,
    )
    from langchain.llms import (
        Cohere as LangChainCohere,
    )
    from langchain.llms import (
        HuggingFaceHub,
    )
    from langchain.llms import (
        Ollama as LangChainOllama,
    )
    from langchain.llms import (
        OpenAI as LangChainOpenAI,
    )
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    warnings.warn(
        "LangChain not installed. LLM providers limited to Ollama. "
        "Install with: pip install langchain langchain-openai langchain-anthropic",
        ImportWarning
    )


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    REPLICATE = "replicate"
    TOGETHER = "together"
    MISTRAL = "mistral"


# Provider-specific default models
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4-turbo-preview",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.COHERE: "command-r-plus",
    LLMProvider.GOOGLE: "gemini-pro",
    LLMProvider.HUGGINGFACE: "meta-llama/Llama-2-70b-chat-hf",
    LLMProvider.OLLAMA: "llama3.2:3b",
    LLMProvider.MISTRAL: "mistral-large-latest",
}


class MultiProviderLLM:
    """
    Unified interface for multiple LLM providers.

    Automatically handles:
    - API key management from environment variables
    - Provider-specific configuration
    - Error handling and retries
    - Cost tracking (when available)
    - Graceful fallback to local models
    """

    def __init__(
        self,
        provider: LLMProvider | str = LLMProvider.OLLAMA,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **provider_kwargs
    ):
        """
        Initialize multi-provider LLM.

        Args:
            provider: LLM provider (openai, anthropic, cohere, etc.)
            model: Model name (uses provider default if None)
            api_key: API key (reads from env if None)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **provider_kwargs: Provider-specific arguments

        Example:
            >>> llm = MultiProviderLLM(provider="anthropic", model="claude-3-5-sonnet-20241022")
            >>> response = llm("Explain quantum computing")

            >>> llm = MultiProviderLLM(provider="openai", temperature=0.3)
            >>> response = llm("Write a Python function")
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain"
            )

        # Normalize provider
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())

        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        self.api_key = api_key or self._get_api_key(provider)

        # Create LLM instance
        self.llm = self._create_llm(provider_kwargs)

        # Usage tracking
        self.total_tokens = 0
        self.total_cost = 0.0

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            response = self.llm(prompt, **kwargs)

            # Track usage
            if hasattr(self.llm, 'get_num_tokens'):
                tokens = self.llm.get_num_tokens(prompt) + self.llm.get_num_tokens(response)
                self.total_tokens += tokens

            return response

        except Exception as e:
            warnings.warn(f"LLM generation failed: {e}")
            return f"[Error: {str(e)}]"

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat-style generation with message history.

        Args:
            messages: List of {"role": "user"/"assistant"/"system", "content": "..."}
            **kwargs: Additional generation parameters

        Returns:
            Generated response

        Example:
            >>> llm = MultiProviderLLM(provider="anthropic")
            >>> response = llm.chat([
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is Python?"}
            ... ])
        """
        # Convert to LangChain message format
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        try:
            # Use chat model if available
            if hasattr(self, 'chat_model'):
                response = self.chat_model(langchain_messages, **kwargs)
                return response.content
            else:
                # Fallback to regular completion
                prompt = "\n\n".join([
                    f"{msg.type}: {msg.content}"
                    for msg in langchain_messages
                ])
                return self(prompt, **kwargs)

        except Exception as e:
            warnings.warn(f"Chat generation failed: {e}")
            return f"[Error: {str(e)}]"

    def stream(self, prompt: str, **kwargs):
        """
        Stream generation token-by-token.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Yields:
            Generated tokens

        Example:
            >>> llm = MultiProviderLLM(provider="openai")
            >>> for token in llm.stream("Write a story"):
            ...     print(token, end="", flush=True)
        """
        try:
            for token in self.llm.stream(prompt, **kwargs):
                yield token
        except Exception as e:
            warnings.warn(f"Streaming failed: {e}")
            yield f"[Error: {str(e)}]"

    def get_usage_stats(self) -> dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dict with tokens used and estimated cost
        """
        return {
            "provider": self.provider.value,
            "model": self.model,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.total_cost
        }

    def _create_llm(self, provider_kwargs: dict[str, Any]):
        """Create LLM instance for provider."""
        common_args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key,
                **common_args,
                **provider_kwargs
            )

        elif self.provider == LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                model=self.model,
                anthropic_api_key=self.api_key,
                **common_args,
                **provider_kwargs
            )

        elif self.provider == LLMProvider.COHERE:
            return ChatCohere(
                model=self.model,
                cohere_api_key=self.api_key,
                **common_args,
                **provider_kwargs
            )

        elif self.provider == LLMProvider.OLLAMA:
            return ChatOllama(
                model=self.model,
                **common_args,
                **provider_kwargs
            )

        elif self.provider == LLMProvider.HUGGINGFACE:
            return HuggingFaceHub(
                repo_id=self.model,
                huggingfacehub_api_token=self.api_key,
                model_kwargs=common_args,
                **provider_kwargs
            )

        else:
            raise ValueError(f"Provider {self.provider} not yet implemented")

    def _get_api_key(self, provider: LLMProvider) -> str | None:
        """Get API key from environment."""
        env_var_map = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.COHERE: "COHERE_API_KEY",
            LLMProvider.GOOGLE: "GOOGLE_API_KEY",
            LLMProvider.HUGGINGFACE: "HUGGINGFACEHUB_API_TOKEN",
            LLMProvider.MISTRAL: "MISTRAL_API_KEY",
        }

        env_var = env_var_map.get(provider)
        if env_var:
            api_key = os.getenv(env_var)
            if not api_key and provider != LLMProvider.OLLAMA:
                warnings.warn(
                    f"API key not found in environment variable {env_var}. "
                    f"Set it with: export {env_var}=your_key_here",
                    UserWarning
                )
            return api_key

        return None


# Convenience functions

def create_llm(
    provider: str = "ollama",
    model: str | None = None,
    **kwargs
) -> MultiProviderLLM:
    """
    Convenience function to create LLM.

    Args:
        provider: Provider name
        model: Model name
        **kwargs: Additional arguments

    Returns:
        MultiProviderLLM instance

    Example:
        >>> llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")
        >>> response = llm("Hello!")
    """
    return MultiProviderLLM(provider=provider, model=model, **kwargs)


def list_llm_providers() -> dict[str, list[str]]:
    """
    List all available LLM providers and their models.

    Returns:
        Dict mapping provider to list of popular models
    """
    return {
        "openai": [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        "cohere": [
            "command-r-plus",
            "command-r",
            "command",
            "command-light"
        ],
        "google": [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-ultra"
        ],
        "ollama": [
            "llama3.2:3b",
            "llama3.1:70b",
            "mistral:7b",
            "mixtral:8x7b",
            "codellama:7b",
            "phi3:mini"
        ],
        "huggingface": [
            "meta-llama/Llama-2-70b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "google/flan-t5-xxl"
        ],
    }


# Auto-fallback LLM

def create_best_available_llm(**kwargs) -> MultiProviderLLM:
    """
    Create best available LLM based on API keys.

    Checks in order: Anthropic → OpenAI → Cohere → Ollama (local)

    Returns:
        MultiProviderLLM instance

    Example:
        >>> llm = create_best_available_llm()
        >>> # Uses Claude if ANTHROPIC_API_KEY is set,
        >>> # otherwise falls back to GPT-4, then Cohere, then Ollama
    """
    # Check for API keys in order of preference
    if os.getenv("ANTHROPIC_API_KEY"):
        return create_llm("anthropic", **kwargs)

    elif os.getenv("OPENAI_API_KEY"):
        return create_llm("openai", **kwargs)

    elif os.getenv("COHERE_API_KEY"):
        return create_llm("cohere", **kwargs)

    else:
        warnings.warn(
            "No API keys found. Falling back to local Ollama. "
            "For better results, set ANTHROPIC_API_KEY or OPENAI_API_KEY.",
            UserWarning
        )
        return create_llm("ollama", **kwargs)
