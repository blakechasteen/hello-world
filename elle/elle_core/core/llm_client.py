"""
Elle Core - LLM Client Abstraction

Provides a unified interface for calling different LLM providers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str  # The raw text response
    raw_response: Dict[str, Any]  # Provider-specific response data
    model: str  # Which model was used
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Implementations must support:
    - Sending prompts
    - Getting structured (JSON) responses
    - Handling errors gracefully
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with the completion
        """
        pass

    async def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.

        Convenience method that calls complete() and parses JSON.
        """
        response = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            # Try to extract JSON from markdown code blocks if present
            content = response.content.strip()
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON: {e}\n\nResponse: {response.content}")


class ClaudeClient(LLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Call Claude API."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        client = AsyncAnthropic(api_key=self.api_key)

        messages = [{"role": "user", "content": prompt}]

        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages,
        )

        return LLMResponse(
            content=response.content[0].text,
            raw_response=response.model_dump(),
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Call OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = AsyncOpenAI(api_key=self.api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            raw_response=response.model_dump(),
            model=self.model,
            tokens_used=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason,
        )


class OllamaClient(LLMClient):
    """Client for local Ollama models."""

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Call Ollama API."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp not installed. Run: pip install aiohttp")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama API error: {response.status}")
                data = await response.json()

        return LLMResponse(
            content=data["message"]["content"],
            raw_response=data,
            model=self.model,
            finish_reason=data.get("done_reason"),
        )


def create_llm_client(
    provider: str = "claude",
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create the appropriate LLM client.

    Args:
        provider: "claude", "openai", or "ollama"
        model: Optional model name (uses defaults if not specified)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLMClient instance
    """
    provider = provider.lower()

    if provider == "claude":
        return ClaudeClient(model=model or "claude-3-5-sonnet-20241022", **kwargs)
    elif provider == "openai":
        return OpenAIClient(model=model or "gpt-4o", **kwargs)
    elif provider == "ollama":
        return OllamaClient(model=model or "llama3.2:3b", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'claude', 'openai', or 'ollama'")
