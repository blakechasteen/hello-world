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
    from hololoom.awareness.llm_integration import OllamaLLM

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
    GEMINI = "gemini"
    GROK = "grok"


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
        except Exception:
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

    Requires:
        pip install openai
        export OPENAI_API_KEY="your-key"

    Models:
        - gpt-4 (highest quality, slower)
        - gpt-4-turbo (faster, good quality)
        - gpt-3.5-turbo (fastest, lower cost)
    """

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize OpenAI LLM.

        Args:
            model: OpenAI model name
            api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key
        self.provider = LLMProvider.OPENAI
        self.client = None
        self._available = False

        try:
            from openai import OpenAI
            import os
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key:
                self.client = OpenAI(api_key=key)
                self._available = True
        except ImportError:
            self.client = None
            self._available = False

    def is_available(self) -> bool:
        """Check if OpenAI is available"""
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
        Generate completion from OpenAI.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError(
                "OpenAI not available. Install with: pip install openai\n"
                "Set API key: export OPENAI_API_KEY='your-key'"
            )

        try:
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

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            content = response.choices[0].message.content if response.choices else ""

            # Extract usage stats
            usage = None
            if response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }

            return LLMResponse(
                content=content,
                provider=self.provider,
                model=self.model,
                usage=usage,
                metadata={
                    'finish_reason': response.choices[0].finish_reason if response.choices else None,
                    'model': response.model
                }
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generation from OpenAI token by token.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated tokens as they arrive
        """
        if not self.is_available():
            raise RuntimeError("OpenAI not available")

        try:
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

            # Stream from OpenAI
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise RuntimeError(f"OpenAI streaming failed: {e}")


class GeminiLLM:
    """
    Google Gemini integration.

    Requires:
        pip install google-genai
        export GEMINI_API_KEY="your-key"

    Models:
        - gemini-2.0-flash (fast, recommended)
        - gemini-2.0-flash-lite (fastest)
        - gemini-1.5-pro (highest quality)
    """

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        """
        Initialize Gemini LLM.

        Args:
            model: Gemini model name
            api_key: Optional API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key
        self.provider = LLMProvider.GEMINI
        self.client = None
        self._available = False

        try:
            from google import genai
            import os
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if key:
                self.client = genai.Client(api_key=key)
                self._available = True
        except ImportError:
            self.client = None
            self._available = False

    def is_available(self) -> bool:
        """Check if Gemini is available"""
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
        Generate completion from Google Gemini.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError(
                "Gemini not available. Install with: pip install google-genai\n"
                "Set API key: export GEMINI_API_KEY='your-key'"
            )

        try:
            # Build content with system instruction if provided
            contents = prompt
            config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_prompt:
                config["system_instruction"] = system_prompt

            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            content = response.text if hasattr(response, 'text') else ""

            # Extract usage stats if available
            usage = None
            if hasattr(response, 'usage_metadata'):
                usage = {
                    'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                    'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                    'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
                }

            return LLMResponse(
                content=content,
                provider=self.provider,
                model=self.model,
                usage=usage,
                metadata={'response': str(response)}
            )

        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generation from Gemini token by token.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated tokens as they arrive
        """
        if not self.is_available():
            raise RuntimeError("Gemini not available")

        try:
            contents = prompt
            config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_prompt:
                config["system_instruction"] = system_prompt

            # Stream from Gemini
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config
            ):
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text

        except Exception as e:
            raise RuntimeError(f"Gemini streaming failed: {e}")


class GrokLLM:
    """
    xAI Grok integration.

    Requires:
        pip install xai-sdk
        export XAI_API_KEY="your-key"

    Models:
        - grok-4-1-fast (fastest, recommended)
        - grok-4-1 (balanced)
        - grok-4 (highest quality)
    """

    def __init__(self, model: str = "grok-4-1-fast", api_key: Optional[str] = None):
        """
        Initialize Grok LLM.

        Args:
            model: Grok model name
            api_key: Optional API key (defaults to XAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key
        self.provider = LLMProvider.GROK
        self.client = None
        self._available = False

        try:
            from xai_sdk import Client
            import os
            key = api_key or os.environ.get("XAI_API_KEY")
            if key:
                self.client = Client(api_key=key)
                self._available = True
        except ImportError:
            self.client = None
            self._available = False

    def is_available(self) -> bool:
        """Check if Grok is available"""
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
        Generate completion from xAI Grok.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError(
                "Grok not available. Install with: pip install xai-sdk\n"
                "Set API key: export XAI_API_KEY='your-key'"
            )

        try:
            # Create chat session
            chat = await self.client.chat.create(
                model=self.model,
                system_prompt=system_prompt if system_prompt else None,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Sample response
            response, _ = await chat.sample(prompt)

            content = response if isinstance(response, str) else str(response)

            return LLMResponse(
                content=content,
                provider=self.provider,
                model=self.model,
                usage=None,  # Grok SDK may not expose token counts
                metadata={}
            )

        except Exception as e:
            raise RuntimeError(f"Grok generation failed: {e}")

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generation from Grok token by token.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated tokens as they arrive
        """
        if not self.is_available():
            raise RuntimeError("Grok not available")

        try:
            # Create chat session
            chat = await self.client.chat.create(
                model=self.model,
                system_prompt=system_prompt if system_prompt else None,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Stream from Grok - returns tuple (response, token)
            async for _, token in chat.stream(prompt):
                if token:
                    yield token

        except Exception as e:
            raise RuntimeError(f"Grok streaming failed: {e}")


def create_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    **kwargs
) -> LLMProtocol:
    """
    Factory function to create LLM instances.

    Args:
        provider: "ollama", "anthropic", "openai", "gemini", or "grok"
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

        # Gemini (when available)
        llm = create_llm("gemini", model="gemini-2.0-flash", api_key="...")

        # Grok (when available)
        llm = create_llm("grok", model="grok-4-1-fast", api_key="...")
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
    elif provider_lower == "gemini":
        model = model or "gemini-2.0-flash"
        return GeminiLLM(model=model, **kwargs)
    elif provider_lower == "grok":
        model = model or "grok-4-1-fast"
        return GrokLLM(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama', 'anthropic', 'openai', 'gemini', or 'grok'")
