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

import logging
import re
from typing import Dict, Optional, Any
from HoloLoom.protocols.types import Query, Context
from HoloLoom.weaving_orchestrator import WeavingOrchestrator as BaseOrchestrator, ToolExecutor as BaseToolExecutor


# =============================================================================
# SECURITY: Prompt Injection Protection
# =============================================================================

def sanitize_prompt(user_input: str) -> str:
    """
    Remove potential prompt injection patterns from user input.

    SECURITY: Prevents adversarial manipulation of LLM behavior through
    instruction override attempts embedded in user queries.

    Args:
        user_input: Raw user input text

    Returns:
        Sanitized text with injection patterns filtered
    """
    if not user_input:
        return ""

    # Patterns that indicate prompt injection attempts
    # Each pattern is replaced with [FILTERED] to neutralize the attack
    injection_patterns = [
        r"ignore\s+(previous|all|above|prior)\s+instructions?",
        r"disregard\s+(previous|all|above|prior)\s+instructions?",
        r"forget\s+(previous|all|above|prior)\s+instructions?",
        r"you\s+are\s+now\s+a?\s*",
        r"new\s+instructions?:",
        r"system\s*:",
        r"<\s*system\s*>",
        r"<\s*/?\s*instructions?\s*>",
        r"override\s+mode",
        r"admin\s+mode",
        r"developer\s+mode",
        r"jailbreak",
        r"do\s+anything\s+now",
        r"pretend\s+you\s+are",
        r"act\s+as\s+if",
        r"roleplay\s+as",
    ]

    sanitized = user_input
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)

    return sanitized


def mask_sensitive(text: str) -> str:
    """
    Mask API keys and secrets in error messages.

    SECURITY: Prevents accidental exposure of credentials in logs or error messages.

    Args:
        text: Text that may contain sensitive information

    Returns:
        Text with sensitive patterns masked
    """
    if not text:
        return ""

    text_str = str(text)

    # Patterns for various API key formats
    sensitive_patterns = [
        # Anthropic API keys
        (r'sk-ant-[a-zA-Z0-9\-_]{32,}', 'sk-ant-***'),
        # OpenAI API keys
        (r'sk-[a-zA-Z0-9]{32,}', 'sk-***'),
        # Generic API keys in key=value format
        (r'(api[_-]?key\s*[=:]\s*)[a-zA-Z0-9\-_]{16,}', r'\1***'),
        # Authorization headers
        (r'(Bearer\s+)[a-zA-Z0-9\-_.]{20,}', r'\1***'),
        # Password patterns
        (r'(password\s*[=:]\s*)\S+', r'\1***'),
        # Token patterns
        (r'(token\s*[=:]\s*)[a-zA-Z0-9\-_.]{16,}', r'\1***'),
    ]

    masked = text_str
    for pattern, replacement in sensitive_patterns:
        masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)

    return masked

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
            # SECURITY: Mask any API keys that might be in the error message
            logger.error(f"Failed to initialize LLM: {mask_sensitive(str(e))}")
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

        # SECURITY: Sanitize user input to prevent prompt injection attacks
        # Context is from internal retrieval, but sanitize for defense-in-depth
        sanitized_context = sanitize_prompt(context_str)
        sanitized_query = sanitize_prompt(query.text)

        user_prompt = f"""Context:
{sanitized_context}

Question: {sanitized_query}

Answer (be concise):"""

        # Try LLM generation
        if self.llm and (not hasattr(self.llm, 'is_available') or self.llm.is_available()):
            try:
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.7
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

            except Exception as e:
                # SECURITY: Mask any API keys that might be in the error message
                logger.error(f"LLM generation failed: {mask_sensitive(str(e))}")
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
