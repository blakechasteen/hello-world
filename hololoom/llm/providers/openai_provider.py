"""
OpenAI Provider
===============

GPT integration via OpenAI API.

Requires:
- pip install openai
- export OPENAI_API_KEY="your-key"

Created: 2025-01-20
"""

import logging
from typing import Optional
from HoloLoom.llm.unified_client import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    OpenAI GPT provider.

    Supported models:
    - gpt-4 (best quality)
    - gpt-4-turbo (faster, cheaper)
    - gpt-4o (multimodal)
    - gpt-4o-mini (very cheap)
    - gpt-3.5-turbo (fast, cheap)
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration with API key
        """
        self.config = config
        self.client = None
        self._available = False

        if not config.api_key:
            logger.warning("⚠ OPENAI_API_KEY not set")
            return

        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=config.api_key)
            self._available = True
            logger.info(f"✓ OpenAI client initialized")
        except ImportError:
            logger.warning("⚠ OpenAI not installed. Install with: pip install openai")

    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self._available and self.client is not None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from GPT.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional OpenAI options

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError("OpenAI not available")

        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            # Extract content
            content = response.choices[0].message.content if response.choices else ""

            # Extract token counts
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider="openai",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={
                    "id": response.id,
                    "finish_reason": response.choices[0].finish_reason if response.choices else None,
                }
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")
