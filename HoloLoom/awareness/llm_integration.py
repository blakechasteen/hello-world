"""
LLM Integration Layer for Awareness-Guided Generation
======================================================

Connects awareness context to actual LLMs (Ollama, Anthropic, OpenAI).

Architecture:
- Protocol-based design (swappable LLM backends)
- Graceful degradation (falls back to templates if LLM unavailable)
- Streaming support (for real-time generation)
- Awareness context injection (prompts include compositional signals)

Usage:
    from HoloLoom.awareness.llm_integration import OllamaLLM

    llm = OllamaLLM(model="llama3.2:3b")
    response = await llm.generate(
        prompt="What is Thompson Sampling?",
        system_prompt="You are a helpful AI assistant.",
        max_tokens=500
    )
"""

from typing import Protocol, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import json


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMResponse:
    """Response from LLM generation"""
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, int]] = None  # Token counts
    metadata: Optional[Dict[str, Any]] = None


class LLMProtocol(Protocol):
    """
    Protocol for LLM integrations.

    All LLM implementations must follow this interface.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from prompt"""
        ...

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream generation token by token"""
        ...

    def is_available(self) -> bool:
        """Check if LLM is available"""
        ...


class OllamaLLM:
    """
    Ollama integration for local LLM inference.

    Requires:
        pip install ollama

    Or download from: https://ollama.ai

    Models:
        - llama3.2:3b (fast, good for most tasks)
        - llama3.1:8b (slower, better quality)
        - mistral:7b (alternative)
        - phi3:3.8b (very fast, decent quality)
    """

    def __init__(self, model: str = "llama3.2:3b", base_url: Optional[str] = None):
        """
        Initialize Ollama LLM.

        Args:
            model: Ollama model name (e.g., "llama3.2:3b")
            base_url: Optional custom Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.provider = LLMProvider.OLLAMA

        try:
            import ollama
            self.client = ollama
            self._available = True
        except ImportError:
            self.client = None
            self._available = False

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        if not self._available:
            return False

        # Try to connect to Ollama
        try:
            # List models to verify connection
            self.client.list()
            return True
        except:
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        format: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            format: Optional format (e.g. "json")

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError(
                "Ollama not available. Install with: pip install ollama\n"
                "Or download from: https://ollama.ai"
            )

        # Build messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            # Call Ollama
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                format=format
            )

            content = response['message']['content']

            # Extract usage stats if available
            usage = None
            if 'eval_count' in response:
                usage = {
                    'prompt_tokens': response.get('prompt_eval_count', 0),
                    'completion_tokens': response.get('eval_count', 0),
                    'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                }

            return LLMResponse(
                content=content,
                provider=self.provider,
                model=self.model,
                usage=usage,
                metadata={'response': response}
            )

        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generation from Ollama token by token.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated tokens as they arrive
        """
        if not self.is_available():
            raise RuntimeError("Ollama not available")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            # Stream from Ollama
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            )

            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']

        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {e}")


class AnthropicLLM:
    """
    Anthropic Claude integration.

    Requires:
        pip install anthropic
        export ANTHROPIC_API_KEY="your-key"

    Models:
        - claude-3-5-sonnet-20241022 (balanced speed/quality)
        - claude-3-opus-20240229 (highest quality)
        - claude-3-haiku-20240307 (fastest)
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        """
        Initialize Anthropic LLM.

        Args:
            model: Anthropic model name
            api_key: Optional API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key
        self.provider = LLMProvider.ANTHROPIC
        self.client = None
        self._available = False

        try:
            import anthropic
            import os
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                self.client = anthropic.Anthropic(api_key=key)
                self._available = True
        except ImportError:
            self.client = None
            self._available = False

    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        return self._available and self.client is not None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from Anthropic Claude.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError(
                "Anthropic not available. Install with: pip install anthropic\n"
                "Set API key: export ANTHROPIC_API_KEY='your-key'"
            )

        try:
            # Build message
            messages = [{"role": "user", "content": prompt}]

            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else "",
                messages=messages
            )

            content = response.content[0].text if response.content else ""

            # Extract usage stats
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                }

            return LLMResponse(
                content=content,
                provider=self.provider,
                model=self.model,
                usage=usage,
                metadata={
                    'stop_reason': response.stop_reason,
                    'model': response.model
                }
            )

        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generation from Anthropic Claude token by token.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated tokens as they arrive
        """
        if not self.is_available():
            raise RuntimeError("Anthropic not available")

        try:
            messages = [{"role": "user", "content": prompt}]

            # Stream from Anthropic
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else "",
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise RuntimeError(f"Anthropic streaming failed: {e}")


class OpenAILLM:
    """
    OpenAI integration (GPT-4, GPT-3.5).

    TODO: Implement when needed.

    Requires:
        pip install openai
        export OPENAI_API_KEY="your-key"
    """

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.provider = LLMProvider.OPENAI
        self._available = False

        # TODO: Initialize OpenAI client when implemented

    def is_available(self) -> bool:
        return self._available

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError("OpenAI integration coming soon!")

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError("OpenAI streaming coming soon!")


def create_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    **kwargs
) -> LLMProtocol:
    """
    Factory function to create LLM instances.

    Args:
        provider: "ollama", "anthropic", or "openai"
        model: Optional model override
        **kwargs: Provider-specific arguments

    Returns:
        LLM instance

    Examples:
        # Ollama (local)
        llm = create_llm("ollama", model="llama3.2:3b")

        # Anthropic (when available)
        llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022", api_key="...")

        # OpenAI (when available)
        llm = create_llm("openai", model="gpt-4", api_key="...")
    """
    provider_lower = provider.lower()

    if provider_lower == "ollama":
        model = model or "llama3.2:3b"
        return OllamaLLM(model=model, **kwargs)
    elif provider_lower == "anthropic":
        model = model or "claude-3-5-sonnet-20241022"
        return AnthropicLLM(model=model, **kwargs)
    elif provider_lower == "openai":
        model = model or "gpt-4"
        return OpenAILLM(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama', 'anthropic', or 'openai'")
