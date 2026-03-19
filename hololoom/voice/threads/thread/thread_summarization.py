"""
Thread Summarization

Enables automatic summarization of conversation threads with multiple styles.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class SummaryStyle(Enum):
    """Styles for thread summarization"""
    BULLET_POINTS = auto()  # Key points as bullets
    NARRATIVE = auto()      # Flowing narrative summary
    DECISIONS = auto()      # Decisions and action items
    QUESTIONS = auto()      # Unanswered questions
    TIMELINE = auto()       # Chronological timeline


@dataclass
class ThreadSummary:
    """Result of a thread summarization"""
    thread_id: str
    thread_name: str
    style: SummaryStyle
    summary: str
    message_count: int
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'thread_id': self.thread_id,
            'thread_name': self.thread_name,
            'style': self.style.name,
            'summary': self.summary,
            'message_count': self.message_count,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class ThreadSummarizer:
    """
    Handles thread summarization with multiple styles and caching.

    Features:
    - 5 summary styles (BULLET_POINTS, NARRATIVE, DECISIONS, QUESTIONS, TIMELINE)
    - Metaprompt-enhanced LLM prompts
    - Summary caching (5-minute TTL)
    - Graceful fallback without LLM
    """

    def __init__(
        self,
        thread_manager: Any,  # elle.ThreadManager
        llm_client: Any | None = None,  # LLM client for synthesis
        enable_caching: bool = True,
        cache_ttl_seconds: float = 300.0,  # 5 minutes
        default_style: SummaryStyle = SummaryStyle.BULLET_POINTS
    ):
        """
        Initialize thread summarizer.

        Args:
            thread_manager: Elle ThreadManager instance
            llm_client: Optional LLM client for synthesis
            enable_caching: Enable summary caching
            cache_ttl_seconds: Cache TTL in seconds (default: 300 = 5 min)
            default_style: Default summary style
        """
        self.thread_manager = thread_manager
        self.llm_client = llm_client
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.default_style = default_style

        # Cache: {cache_key: (summary, timestamp)}
        self._cache: dict[str, tuple[ThreadSummary, float]] = {}

        logger.info(
            f"ThreadSummarizer initialized: "
            f"default_style={default_style.name}, "
            f"caching={enable_caching}, "
            f"ttl={cache_ttl_seconds}s, "
            f"llm_available={llm_client is not None}"
        )

    async def summarize_thread(
        self,
        thread_id: str,
        style: SummaryStyle | None = None,
        custom_prompt: str | None = None,
        force_refresh: bool = False
    ) -> ThreadSummary:
        """
        Summarize a conversation thread.

        Algorithm:
        1. Check cache (if enabled and not force_refresh)
        2. Validate thread exists
        3. Build style-specific prompt
        4. Enhance with metaprompt (if available)
        5. Generate summary with LLM
        6. Cache result (if caching enabled)
        7. Return summary

        Args:
            thread_id: ID of thread to summarize
            style: Summary style (default: self.default_style)
            custom_prompt: Optional custom LLM prompt
            force_refresh: Skip cache and regenerate

        Returns:
            ThreadSummary with summary text and metadata

        Raises:
            ValueError: If thread not found
        """
        style = style or self.default_style

        logger.info(f"Summarizing thread '{thread_id}' with {style.name} style")

        # Step 1: Check cache
        if self.enable_caching and not force_refresh:
            cached = self._get_cached_summary(thread_id, style)
            if cached is not None:
                logger.debug(f"Cache hit for thread '{thread_id}' ({style.name})")
                return cached

        # Step 2: Validate thread
        thread = self.thread_manager.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"Thread not found: {thread_id}")

        if len(thread.messages) == 0:
            # Empty thread - return simple summary
            return ThreadSummary(
                thread_id=thread_id,
                thread_name=thread.name,
                style=style,
                summary="(No messages in this thread yet)",
                message_count=0,
                metadata={'empty': True}
            )

        # Step 3: Build prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_summary_prompt(thread, style)

        # Step 4: Enhance with metaprompt
        prompt = await self._enhance_prompt(prompt)

        # Step 5: Generate summary
        if self.llm_client is None:
            summary_text = self._fallback_summary(thread, style)
            logger.warning("No LLM client, using fallback summary")
        else:
            summary_text = await self.llm_client.generate(prompt)

        # Step 6: Create result
        summary = ThreadSummary(
            thread_id=thread_id,
            thread_name=thread.name,
            style=style,
            summary=summary_text,
            message_count=len(thread.messages),
            metadata={
                'llm_generated': self.llm_client is not None,
                'prompt_enhanced': True
            }
        )

        # Step 7: Cache result
        if self.enable_caching:
            self._cache_summary(thread_id, style, summary)

        logger.info(
            f"Summary complete: {len(thread.messages)} messages → "
            f"{len(summary_text)} chars"
        )

        return summary

    def _build_summary_prompt(
        self,
        thread: Any,
        style: SummaryStyle
    ) -> str:
        """
        Build LLM prompt for summary.

        Args:
            thread: Thread object
            style: Summary style

        Returns:
            LLM prompt string
        """
        # Style-specific instructions
        style_instructions = {
            SummaryStyle.BULLET_POINTS: """
Format your summary as bullet points:
- Start each point with a dash (-)
- Keep each point concise (1-2 lines)
- Capture key facts, insights, and decisions
- Order by importance (most important first)
- Aim for 5-10 bullet points
""",
            SummaryStyle.NARRATIVE: """
Format your summary as a flowing narrative:
- Write 2-4 concise paragraphs
- Tell the story of the conversation
- Connect ideas and show progression
- Use clear, natural language
- Focus on the main thread of discussion
""",
            SummaryStyle.DECISIONS: """
Format your summary focused on decisions and actions:
- List specific decisions made
- Include action items (who/what/when if mentioned)
- Note any commitments or agreements
- Highlight next steps
- If no decisions, state "No explicit decisions made"
""",
            SummaryStyle.QUESTIONS: """
Format your summary focused on questions:
- List unanswered questions from the conversation
- Include both explicit questions and implied uncertainties
- Categorize if helpful (technical, strategic, etc.)
- Note any questions that were answered
- If no open questions, state "All questions resolved"
""",
            SummaryStyle.TIMELINE: """
Format your summary as a chronological timeline:
- Show key events/topics in time order
- Use timestamp indicators (beginning, middle, end)
- Note topic transitions
- Keep each timeline entry brief (1 line)
- Aim for 5-8 timeline entries
"""
        }

        instruction = style_instructions.get(style, style_instructions[SummaryStyle.BULLET_POINTS])

        # Build prompt
        prompt = f"""Summarize this conversation thread.

Thread: {thread.name}
Messages: {len(thread.messages)}
Style: {style.name}

{instruction}

Conversation:
"""

        # Add messages (show all for short threads, sample for long)
        if len(thread.messages) <= 20:
            # Show all messages
            for msg in thread.messages:
                role_indicator = "User" if msg.role == "user" else "Assistant"
                content_preview = msg.content[:200]
                if len(msg.content) > 200:
                    content_preview += "..."
                prompt += f"\n{role_indicator}: {content_preview}"
        else:
            # Show first 10 and last 10
            for msg in thread.messages[:10]:
                role_indicator = "User" if msg.role == "user" else "Assistant"
                content_preview = msg.content[:200]
                prompt += f"\n{role_indicator}: {content_preview}"

            prompt += f"\n\n[... {len(thread.messages) - 20} messages omitted ...]\n"

            for msg in thread.messages[-10:]:
                role_indicator = "User" if msg.role == "user" else "Assistant"
                content_preview = msg.content[:200]
                prompt += f"\n{role_indicator}: {content_preview}"

        prompt += "\n\nSummary:"

        return prompt

    async def _enhance_prompt(self, prompt: str) -> str:
        """
        Enhance prompt with metaprompt system.

        Args:
            prompt: Basic prompt

        Returns:
            Enhanced prompt
        """
        try:
            from hololoom.config import Config
            from hololoom.prompting import create_metaprompt

            config = Config.fast()
            config.llm_provider = "anthropic"
            enhanced_prompt = create_metaprompt(prompt, config=config)
            logger.debug("Enhanced prompt with metaprompt")
            return enhanced_prompt
        except ImportError:
            logger.debug("Metaprompt not available, using basic prompt")
            return prompt

    def _fallback_summary(
        self,
        thread: Any,
        style: SummaryStyle
    ) -> str:
        """
        Generate fallback summary without LLM.

        Args:
            thread: Thread object
            style: Summary style

        Returns:
            Basic summary string
        """
        if style == SummaryStyle.BULLET_POINTS:
            # Extract first sentence from each message
            points = []
            for msg in thread.messages[:10]:
                first_sentence = msg.content.split('.')[0]
                if first_sentence:
                    points.append(f"- {first_sentence}")
            return "\n".join(points) if points else "- No summary available"

        elif style == SummaryStyle.TIMELINE:
            # Show message count progression
            return f"Timeline: {len(thread.messages)} messages exchanged"

        else:
            # Generic fallback
            return f"Thread '{thread.name}' contains {len(thread.messages)} messages."

    def _get_cache_key(self, thread_id: str, style: SummaryStyle) -> str:
        """
        Generate cache key for thread + style.

        Args:
            thread_id: Thread ID
            style: Summary style

        Returns:
            Cache key string
        """
        key_str = f"{thread_id}:{style.name}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_summary(
        self,
        thread_id: str,
        style: SummaryStyle
    ) -> ThreadSummary | None:
        """
        Get cached summary if valid.

        Args:
            thread_id: Thread ID
            style: Summary style

        Returns:
            Cached ThreadSummary or None
        """
        cache_key = self._get_cache_key(thread_id, style)

        if cache_key not in self._cache:
            return None

        summary, timestamp = self._cache[cache_key]
        now = datetime.now().timestamp()

        # Check TTL
        if now - timestamp > self.cache_ttl_seconds:
            # Expired
            del self._cache[cache_key]
            logger.debug(f"Cache expired for thread '{thread_id}' ({style.name})")
            return None

        return summary

    def _cache_summary(
        self,
        thread_id: str,
        style: SummaryStyle,
        summary: ThreadSummary
    ) -> None:
        """
        Cache summary result.

        Args:
            thread_id: Thread ID
            style: Summary style
            summary: ThreadSummary to cache
        """
        cache_key = self._get_cache_key(thread_id, style)
        self._cache[cache_key] = (summary, datetime.now().timestamp())
        logger.debug(f"Cached summary for thread '{thread_id}' ({style.name})")

    def clear_cache(self, thread_id: str | None = None) -> int:
        """
        Clear summary cache.

        Args:
            thread_id: Optional thread ID to clear (None = clear all)

        Returns:
            Number of cache entries cleared
        """
        if thread_id is None:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared all cache ({count} entries)")
            return count

        # Clear specific thread (all styles)
        cleared = 0
        keys_to_delete = []
        for cache_key in self._cache.keys():
            summary, _ = self._cache[cache_key]
            if summary.thread_id == thread_id:
                keys_to_delete.append(cache_key)

        for key in keys_to_delete:
            del self._cache[key]
            cleared += 1

        logger.info(f"Cleared cache for thread '{thread_id}' ({cleared} entries)")
        return cleared

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        now = datetime.now().timestamp()
        expired = 0

        for cache_key, (summary, timestamp) in self._cache.items():
            if now - timestamp > self.cache_ttl_seconds:
                expired += 1

        return {
            'total_entries': len(self._cache),
            'expired_entries': expired,
            'valid_entries': len(self._cache) - expired,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'enabled': self.enable_caching
        }
