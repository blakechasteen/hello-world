"""
LLM-Integrated Weaving Orchestrator
====================================
Drop-in replacement for WeavingOrchestrator with actual LLM calls.

This module extends the base orchestrator to call real LLMs instead of stubs.
Simply swap imports to enable full LLM integration.

Usage:
    # Before (stubs):
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

    # After (real LLMs):
    from HoloLoom.weaving_orchestrator_llm import WeavingOrchestrator

Installation:
    pip install ollama anthropic openai
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from HoloLoom.documentation.types import Query, Context
from HoloLoom.weaving_orchestrator import WeavingOrchestrator as BaseOrchestrator, ToolExecutor as BaseToolExecutor

# LLM imports
try:
    from HoloLoom.awareness.llm_integration import OllamaLLM, AnthropicLLM, LLMProvider
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


logger = logging.getLogger(__name__)


class LLMToolExecutor(BaseToolExecutor):
    """
    Tool executor with actual LLM calls.

    Extends base ToolExecutor to replace stubs with real LLM generation.
    """

    def __init__(self, llm: Optional[Any] = None, llm_provider: str = "ollama"):
        """
        Initialize LLM tool executor.

        Args:
            llm: Pre-initialized LLM instance (optional)
            llm_provider: Provider to use if llm not provided ("ollama", "anthropic", "openai")
        """
        super().__init__()

        self.llm = llm
        self.llm_provider = llm_provider

        # Initialize LLM if not provided
        if self.llm is None and LLM_AVAILABLE:
            self._init_llm()

    def _init_llm(self):
        """Initialize LLM based on provider."""
        try:
            if self.llm_provider == "ollama":
                self.llm = OllamaLLM(model="llama3.2:3b")
                if self.llm.is_available():
                    logger.info("✓ Initialized Ollama LLM (llama3.2:3b)")
                else:
                    logger.warning("⚠ Ollama not available - using fallback")
                    self.llm = None

            elif self.llm_provider == "anthropic":
                import os
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.llm = AnthropicLLM(api_key=api_key, model="claude-3-5-sonnet-20241022")
                    logger.info("✓ Initialized Anthropic LLM (Claude 3.5 Sonnet)")
                else:
                    logger.warning("⚠ ANTHROPIC_API_KEY not set - using fallback")
                    self.llm = None

            else:
                logger.warning(f"⚠ Unknown LLM provider: {self.llm_provider}")
                self.llm = None

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    async def _handle_answer(self, query: Query, context: Context) -> Dict:
        """
        Generate an answer using actual LLM.

        Overrides base implementation to call real LLMs instead of stubs.
        """

        # Build context from retrieved shards
        if context and hasattr(context, 'shard_texts'):
            context_texts = context.shard_texts[:5]  # Top 5 shards
            context_str = "\n\n".join(f"[{i+1}] {text}" for i, text in enumerate(context_texts))
            source_count = len(context.shards) if hasattr(context, 'shards') else len(context_texts)
        else:
            context_str = "(No context available)"
            source_count = 0

        # Build prompts
        system_prompt = """You are a helpful AI assistant integrated with HoloLoom's knowledge system.
Answer questions based on the provided context. Be concise and accurate.
If the context doesn't contain relevant information, say so."""

        user_prompt = f"""Context:
{context_str}

Question: {query.text}

Answer (be concise):"""

        # Try LLM generation
        if self.llm and (not hasattr(self.llm, 'is_available') or self.llm.is_available()):
            try:
                # Add timeout protection (default 30s)
                llm_timeout = getattr(self.cfg, 'llm_timeout', 30.0)
                response = await asyncio.wait_for(
                    self.llm.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_tokens=500,
                        temperature=0.7
                    ),
                    timeout=llm_timeout
                )

                return {
                    "tool": "answer",
                    "result": response.content,  # ✅ Real LLM response!
                    "confidence": 0.85,
                    "sources": source_count,
                    "llm_provider": response.provider.value if hasattr(response, 'provider') else "unknown",
                    "llm_model": response.model if hasattr(response, 'model') else "unknown",
                    "usage": response.usage if hasattr(response, 'usage') else None,
                    "context_preview": context_str[:200] + "..." if len(context_str) > 200 else context_str
                }

            except asyncio.TimeoutError:
                logger.error(f"LLM generation timed out after {llm_timeout}s", exc_info=True)
                # Fall through to fallback
            except Exception as e:
                logger.error(f"LLM generation failed: {e}", exc_info=True)
                # Fall through to fallback

        # Fallback: Return base implementation (stubs)
        logger.info("Using fallback (no LLM available)")
        return await super()._handle_answer(query, context)


class WeavingOrchestrator(BaseOrchestrator):
    """
    LLM-integrated weaving orchestrator.

    Drop-in replacement for base WeavingOrchestrator with real LLM calls.
    All other functionality remains identical.
    """

    def __init__(self, *args, llm=None, llm_provider="ollama", **kwargs):
        """
        Initialize orchestrator with LLM integration.

        Args:
            llm: Pre-initialized LLM instance (optional)
            llm_provider: Provider to use ("ollama", "anthropic", "openai")
            *args, **kwargs: Passed to base orchestrator
        """
        super().__init__(*args, **kwargs)

        # Replace tool executor with LLM-enabled version
        self.tool_executor = LLMToolExecutor(llm=llm, llm_provider=llm_provider)

        logger.info(f"LLM-integrated orchestrator initialized (provider: {llm_provider})")


# Export for easy importing
__all__ = ["WeavingOrchestrator", "LLMToolExecutor"]
