#!/usr/bin/env python3
"""
LLM Provider Abstraction
========================
Unified interface for Ollama (local), Anthropic, and OpenAI LLMs.
Provides graceful fallback and automatic provider selection.

Author: Claude Code
Date: November 16, 2025
"""

import os
import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000


class LLMClient:
    """
    Unified LLM client with multi-provider support.

    Automatically selects best available provider:
    1. Anthropic (if API key set) - Best for code
    2. OpenAI (if API key set) - Good fallback
    3. Ollama (local) - Always available
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client with auto-provider selection"""
        if config:
            self.config = config
        else:
            self.config = self._auto_select_provider()

        self._client = None
        self._initialize_client()

    def _auto_select_provider(self) -> LLMConfig:
        """
        Auto-select best available provider from environment.

        Priority:
        1. ANTHROPIC_API_KEY → claude-3-5-sonnet (best for code)
        2. OPENAI_API_KEY → gpt-4 (good fallback)
        3. Ollama → qwen2.5-coder:latest (local, always works)
        """
        # Check for Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            logger.info("Using Anthropic Claude 3.5 Sonnet (cloud)")
            return LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-5-sonnet-20241022",
                api_key=anthropic_key,
                temperature=0.7,
                max_tokens=4000
            )

        # Check for OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            logger.info("Using OpenAI GPT-4 (cloud)")
            return LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                api_key=openai_key,
                temperature=0.7,
                max_tokens=4000
            )

        # Fallback to Ollama (local)
        logger.info("Using Ollama qwen2.5-coder (local)")
        return LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="qwen2.5-coder:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=4000
        )

    def _initialize_client(self):
        """Initialize provider-specific client"""
        if self.config.provider == LLMProvider.OLLAMA:
            self._initialize_ollama()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            self._initialize_anthropic()
        elif self.config.provider == LLMProvider.OPENAI:
            self._initialize_openai()

    def _initialize_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self._client = ollama
            logger.info(f"Ollama initialized: {self.config.model}")
        except ImportError:
            logger.warning("ollama package not installed. Install with: pip install ollama")
            self._client = None

    def _initialize_anthropic(self):
        """Initialize Anthropic client"""
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.config.api_key)
            logger.info(f"Anthropic initialized: {self.config.model}")
        except ImportError:
            logger.warning("anthropic package not installed. Install with: pip install anthropic")
            self._client = None

    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.config.api_key)
            logger.info(f"OpenAI initialized: {self.config.model}")
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            self._client = None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using configured LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (instructions)
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text
        """
        if not self._client:
            raise RuntimeError(f"LLM client not initialized for {self.config.provider.value}")

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.provider == LLMProvider.OLLAMA:
            return await self._generate_ollama(prompt, system_prompt, temp, max_tok)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic(prompt, system_prompt, temp, max_tok)
        elif self.config.provider == LLMProvider.OPENAI:
            return await self._generate_openai(prompt, system_prompt, temp, max_tok)

    async def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate with Ollama"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate with Anthropic"""
        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise

    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate with OpenAI"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about active provider"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "available": self._client is not None
        }
