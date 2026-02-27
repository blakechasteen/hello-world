"""
Ollama Provider
===============

Local LLM inference via Ollama.

Requires: pip install ollama
Or download from: https://ollama.ai

Created: 2025-01-20
"""

import logging
from typing import Optional
from HoloLoom.llm.unified_client import LLMConfig, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Ollama provider for local LLM inference.

    Supported models:
    - llama3.2:3b (fast, 3B parameters)
    - llama3.1:8b (slower, better quality)
    - mistral:7b (alternative)
    - phi3:3.8b (very fast)
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = None
        self._available = False

        try:
            import ollama
            self.client = ollama
            self._available = True
            logger.info(f"✓ Ollama client loaded")
        except ImportError:
            logger.warning("⚠ Ollama not installed. Install with: pip install ollama")

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        if not self._available:
            return False

        try:
            # Try to list models
            self.client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama not running: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from Ollama.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional Ollama options

        Returns:
            LLMResponse with generated content
        """
        if not self.is_available():
            raise RuntimeError("Ollama not available")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call Ollama
        try:
            response = self.client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
            )

            content = response['message']['content']

            # Extract token counts
            input_tokens = response.get('prompt_eval_count', 0)
            output_tokens = response.get('eval_count', 0)

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider="ollama",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={"response": response}
            )

        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
