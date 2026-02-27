"""
Google Gemini Provider
======================

Gemini integration via Google Generative AI API.

Requires:
- pip install google-generativeai
- export GOOGLE_API_KEY="your-key"

Created: 2025-01-20
"""

import logging
from typing import Optional
from hololoom.llm.unified_client import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider:
    """
    Google Gemini provider.

    Supported models:
    - gemini-1.5-pro (best quality)
    - gemini-1.5-flash (fast, cheap)
    - gemini-pro (older version)
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize Gemini provider.

        Args:
            config: LLM configuration with API key
        """
        self.config = config
        self.client = None
        self.model = None
        self._available = False

        if not config.api_key:
            logger.warning("⚠ GOOGLE_API_KEY not set")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            self.client = genai
            self.model = genai.GenerativeModel(config.model)
            self._available = True
            logger.info(f"✓ Gemini client initialized")
        except ImportError:
            logger.warning(
                "⚠ Google Generative AI not installed. "
                "Install with: pip install google-generativeai"
            )

    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self._available and self.model is not None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from Gemini.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional Gemini options

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError("Gemini not available")

        try:
            # Build prompt (Gemini doesn't have separate system prompt)
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Configure generation
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }

            # Call Gemini API (async)
            response = await self.model.generate_content_async(
                full_prompt,
                generation_config=generation_config
            )

            # Extract content
            content = response.text if hasattr(response, 'text') else ""

            # Gemini doesn't return token counts in standard way
            # Estimate tokens (rough approximation)
            input_tokens = len(full_prompt) // 4
            output_tokens = len(content) // 4

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider="google",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={
                    "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                }
            )

        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")
