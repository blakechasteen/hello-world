"""LLM client interface for calling different providers."""

from typing import Protocol, Optional, Dict, Any
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LOCAL = "local"  # Alias for ollama


class LLMClient(Protocol):
    """
    Interface for calling LLMs.
    
    Different implementations for different providers,
    but same contract for Elle Core.
    """
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Get completion from LLM.
        
        Args:
            prompt: Full prompt text
            temperature: Sampling temperature (0-1)
            max_tokens: Optional token limit
            **kwargs: Provider-specific options
        
        Returns:
            Raw text response from LLM
        """
        ...


class OpenAIClient:
    """OpenAI implementation of LLMClient."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, timeout=timeout)
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Call OpenAI API."""
        from openai import OpenAIError
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class AnthropicClient:
    """Anthropic implementation of LLMClient."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250514",
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Call Anthropic API."""
        import anthropic
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 1000,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract text from response
            return message.content[0].text
            
        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}")


class OllamaClient:
    """
    Ollama LLM client for local model inference.

    Connects to Ollama server (default: http://127.0.0.1:11434).
    Supports any model available in Ollama (qwen3.5, llama3.2, etc).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:9b",
        timeout: int = 120,
    ):
        import httpx

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

        # Verify ollama is available
        try:
            response = self._client.get("/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not responding at {self.base_url}")
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: ollama serve. Error: {e}"
            )

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Call Ollama generate API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        for key, value in kwargs.items():
            if key not in ("model", "prompt", "stream"):
                payload["options"][key] = value

        try:
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Chat completion with message history."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # Disable thinking mode (qwen3.5 puts all output there)
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            msg = response.json().get("message", {})
            # Prefer content, fall back to thinking field
            return msg.get("content", "") or msg.get("thinking", "")
        except Exception as e:
            raise RuntimeError(f"Ollama chat API error: {e}")


# Alias for backward compatibility
LocalClient = OllamaClient


def create_llm_client(
    provider: LLMProvider,
    config: Dict[str, Any]
) -> LLMClient:
    """
    Factory for creating LLM clients.

    Args:
        provider: Which LLM provider to use
        config: Provider-specific configuration

    Returns:
        Configured LLM client

    Example:
        # Ollama (local)
        client = create_llm_client(LLMProvider.OLLAMA, {"model": "llama3.2:3b"})

        # Anthropic
        client = create_llm_client(LLMProvider.ANTHROPIC, {"api_key": "..."})

        # OpenAI
        client = create_llm_client(LLMProvider.OPENAI, {"api_key": "..."})
    """

    if provider == LLMProvider.OPENAI:
        return OpenAIClient(**config)
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(**config)
    elif provider in (LLMProvider.OLLAMA, LLMProvider.LOCAL):
        return OllamaClient(**config)
    else:
        raise ValueError(f"Unknown provider: {provider}")
