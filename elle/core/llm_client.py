"""LLM client interface for calling different providers."""

from typing import Protocol, Optional, Dict, Any
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


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
        model: str = "claude-3-5-sonnet-20241022",
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


class LocalClient:
    """Local LLM implementation (e.g., via Ollama)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: int = 60,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Call local LLM."""
        # TODO: Implement actual local call
        raise NotImplementedError("Local client not yet implemented")


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
    """
    
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(**config)
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(**config)
    elif provider == LLMProvider.LOCAL:
        return LocalClient(**config)
    else:
        raise ValueError(f"Unknown provider: {provider}")
