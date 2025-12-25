"""
LLM Backend Integrations
========================

Provides OllamaBackend and ClaudeBackend for prompt execution and evaluation.
"""

import os
from typing import Optional
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Generate completion for prompt."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama
            except ImportError:
                raise ImportError("Ollama not installed. Run: pip install ollama")
        return self._client

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            client = self._get_client()
            result = client.list()
            # Handle both old dict and new typed response
            if hasattr(result, 'models'):
                models_list = result.models
            else:
                models_list = result.get('models', [])

            # Check if any models are available
            return len(models_list) > 0
        except Exception:
            return False

    async def complete(self, prompt: str) -> str:
        """Generate completion using Ollama."""
        client = self._get_client()
        try:
            response = client.generate(model=self.model, prompt=prompt)
            # Handle both old dict and new typed response
            if hasattr(response, 'response'):
                return response.response
            return response.get('response', '')
        except Exception as e:
            raise RuntimeError(f"Ollama completion failed: {e}")


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Anthropic not installed. Run: pip install anthropic")
        return self._client

    async def is_available(self) -> bool:
        """Check if Claude API is configured."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    async def complete(self, prompt: str) -> str:
        """Generate completion using Claude API."""
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Claude completion failed: {e}")


class MockBackend(LLMBackend):
    """Mock backend for testing without LLM."""

    async def is_available(self) -> bool:
        return True

    async def complete(self, prompt: str) -> str:
        return f"[Mock Response] Prompt received ({len(prompt)} chars)"
