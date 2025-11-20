"""
Anthropic Provider
==================

Claude integration via Anthropic API.

Requires:
- pip install anthropic
- export ANTHROPIC_API_KEY="your-key"

Created: 2025-01-20
"""

import logging
from typing import Optional
from HoloLoom.llm.unified_client import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """
    Anthropic Claude provider.

    Supported models:
    - claude-3-5-sonnet-20241022 (best quality)
    - claude-3-opus (highest intelligence)
    - claude-3-sonnet (balanced)
    - claude-3-haiku (fast, cheap)
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize Anthropic provider.

        Args:
            config: LLM configuration with API key
        """
        self.config = config
        self.client = None
        self._available = False

        if not config.api_key:
            logger.warning("⚠ ANTHROPIC_API_KEY not set")
            return

        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
            self._available = True
            logger.info(f"✓ Anthropic client initialized")
        except ImportError:
            logger.warning("⚠ Anthropic not installed. Install with: pip install anthropic")

    def is_available(self) -> bool:
        """Check if Anthropic is available"""
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
        Generate completion from Claude.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional Anthropic options

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError("Anthropic not available")

        try:
            # Build request
            request_kwargs = {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs
            }

            # Add system prompt if provided
            if system_prompt:
                request_kwargs["system"] = system_prompt

            # Call Anthropic API
            response = await self.client.messages.create(**request_kwargs)

            # Extract content
            content = response.content[0].text if response.content else ""

            # Extract token counts
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider="anthropic",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={
                    "id": response.id,
                    "stop_reason": response.stop_reason,
                }
            )

        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")
