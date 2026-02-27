"""
Thread Summarization Tests

Unit tests for ThreadSummarizer with 10 test cases.

Run:
    pytest hololoom/voice_first/thread/tests/test_thread_summarization.py -v
"""

import pytest
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# ============================================================================
# Mock Objects
# ============================================================================

@dataclass
class Message:
    """Mock Message object"""
    role: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class Thread:
    """Mock Thread object"""
    thread_id: str
    name: str
    messages: List[Message]
    metadata: Dict[str, Any]


class MockThreadManager:
    """Mock ThreadManager for testing"""

    def __init__(self):
        self.threads: Dict[str, Thread] = {}
        self._next_id = 1

    def create_thread(self, name: str) -> str:
        """Create a new thread"""
        thread_id = f"thread_{self._next_id}"
        self._next_id += 1

        self.threads[thread_id] = Thread(
            thread_id=thread_id,
            name=name,
            messages=[],
            metadata={}
        )
        return thread_id

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID"""
        return self.threads.get(thread_id)


class MockLLMClient:
    """Mock LLM client for testing"""

    async def generate(self, prompt: str) -> str:
        """Mock LLM generation"""
        # Return different summaries based on style in prompt
        if "bullet points" in prompt.lower():
            return "- First key point\n- Second key point\n- Third key point"
        elif "narrative" in prompt.lower():
            return "This is a narrative summary. It tells the story of the conversation."
        elif "decisions" in prompt.lower():
            return "Decision: Proceed with plan A\nAction: User to review by Friday"
        elif "questions" in prompt.lower():
            return "Unanswered: What is the timeline?\nResolved: Budget was approved"
        elif "timeline" in prompt.lower():
            return "Beginning: Discussion started\nMiddle: Options explored\nEnd: Conclusion reached"
        else:
            return "Generic summary of the thread."


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_thread_manager():
    """Create mock thread manager with sample thread"""
    manager = MockThreadManager()

    # Create thread with messages
    thread_id = manager.create_thread("farming discussion")
    thread = manager.get_thread(thread_id)
    now = datetime.now().timestamp()

    thread.messages = [
        Message(
            role="user",
            content="How should I start with permaculture?",
            timestamp=now - 200,
            metadata={}
        ),
        Message(
            role="assistant",
            content="Start with soil assessment and water management.",
            timestamp=now - 195,
            metadata={}
        ),
        Message(
            role="user",
            content="What about companion planting?",
            timestamp=now - 180,
            metadata={}
        ),
        Message(
            role="assistant",
            content="Companion planting works great with permaculture. Plant tomatoes with basil.",
            timestamp=now - 175,
            metadata={}
        ),
        Message(
            role="user",
            content="Should I use raised beds?",
            timestamp=now - 160,
            metadata={}
        ),
        Message(
            role="assistant",
            content="Raised beds are good for drainage but not required.",
            timestamp=now - 155,
            metadata={}
        ),
    ]

    return manager


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client"""
    return MockLLMClient()


# ============================================================================
# Tests
# ============================================================================

class TestThreadSummarizerBasics:
    """Basic functionality tests (1-4)"""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_thread_manager):
        """Test 1: ThreadSummarizer initialization"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        summarizer = ThreadSummarizer(
            thread_manager=mock_thread_manager,
            enable_caching=True,
            cache_ttl_seconds=300.0,
            default_style=SummaryStyle.BULLET_POINTS
        )

        assert summarizer.thread_manager is mock_thread_manager
        assert summarizer.enable_caching is True
        assert summarizer.cache_ttl_seconds == 300.0
        assert summarizer.default_style == SummaryStyle.BULLET_POINTS

    @pytest.mark.asyncio
    async def test_basic_summarization(self, mock_thread_manager, mock_llm_client):
        """Test 2: Basic thread summarization"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        summarizer = ThreadSummarizer(
            thread_manager=mock_thread_manager,
            llm_client=mock_llm_client,
            enable_caching=False  # Disable for deterministic test
        )

        result = await summarizer.summarize_thread(
            thread_id="thread_1",
            style=SummaryStyle.BULLET_POINTS
        )

        assert result.thread_id == "thread_1"
        assert result.thread_name == "farming discussion"
        assert result.style == SummaryStyle.BULLET_POINTS
        assert result.message_count == 6
        assert len(result.summary) > 0

    @pytest.mark.asyncio
    async def test_thread_not_found(self, mock_thread_manager):
        """Test 3: Error when thread not found"""
        from hololoom.voice.threads.thread import ThreadSummarizer

        summarizer = ThreadSummarizer(thread_manager=mock_thread_manager)

        with pytest.raises(ValueError, match="Thread not found"):
            await summarizer.summarize_thread(thread_id="nonexistent")

    @pytest.mark.asyncio
    async def test_empty_thread(self, mock_thread_manager):
        """Test 4: Summarize empty thread"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        # Create empty thread
        thread_id = mock_thread_manager.create_thread("empty thread")

        summarizer = ThreadSummarizer(thread_manager=mock_thread_manager)

        result = await summarizer.summarize_thread(thread_id=thread_id)

        assert result.thread_id == thread_id
        assert result.message_count == 0
        assert "No messages" in result.summary
        assert result.metadata.get('empty') is True


class TestSummaryStyles:
    """Summary style tests (5-7)"""

    @pytest.mark.asyncio
    async def test_bullet_points_style(self, mock_thread_manager, mock_llm_client):
        """Test 5: BULLET_POINTS style"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        summarizer = ThreadSummarizer(
            thread_manager=mock_thread_manager,
            llm_client=mock_llm_client,
            enable_caching=False
        )

        result = await summarizer.summarize_thread(
            thread_id="thread_1",
            style=SummaryStyle.BULLET_POINTS
        )

        assert result.style == SummaryStyle.BULLET_POINTS
        assert "-" in result.summary  # Bullet points use dashes

    @pytest.mark.asyncio
    async def test_narrative_style(self, mock_thread_manager, mock_llm_client):
        """Test 6: NARRATIVE style"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        summarizer = ThreadSummarizer(
            thread_manager=mock_thread_manager,
            llm_client=mock_llm_client,
            enable_caching=False
        )

        result = await summarizer.summarize_thread(
            thread_id="thread_1",
            style=SummaryStyle.NARRATIVE
        )

        assert result.style == SummaryStyle.NARRATIVE
        assert "narrative" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_all_five_styles(self, mock_thread_manager, mock_llm_client):
        """Test 7: All 5 summary styles work"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        summarizer = ThreadSummarizer(
            thread_manager=mock_thread_manager,
            llm_client=mock_llm_client,
            enable_caching=False
        )

        styles = [
            SummaryStyle.BULLET_POINTS,
            SummaryStyle.NARRATIVE,
            SummaryStyle.DECISIONS,
            SummaryStyle.QUESTIONS,
            SummaryStyle.TIMELINE
        ]

        for style in styles:
            result = await summarizer.summarize_thread(
                thread_id="thread_1",
                style=style
            )
            assert result.style == style
            assert len(result.summary) > 0


class TestCaching:
    """Cache functionality tests (8-9)"""

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_thread_manager, mock_llm_client):
        """Test 8: Cache hit on repeated query"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        summarizer = ThreadSummarizer(
            thread_manager=mock_thread_manager,
            llm_client=mock_llm_client,
            enable_caching=True,
            cache_ttl_seconds=300.0
        )

        # First call (cache miss)
        result1 = await summarizer.summarize_thread(
            thread_id="thread_1",
            style=SummaryStyle.BULLET_POINTS
        )

        # Second call (should hit cache)
        result2 = await summarizer.summarize_thread(
            thread_id="thread_1",
            style=SummaryStyle.BULLET_POINTS
        )

        # Should return same summary
        assert result1.summary == result2.summary
        assert result1.timestamp == result2.timestamp

        # Cache stats should show 1 entry
        stats = summarizer.get_cache_stats()
        assert stats['total_entries'] >= 1

    @pytest.mark.asyncio
    async def test_cache_management(self, mock_thread_manager, mock_llm_client):
        """Test 9: Cache clearing and stats"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        summarizer = ThreadSummarizer(
            thread_manager=mock_thread_manager,
            llm_client=mock_llm_client,
            enable_caching=True
        )

        # Summarize with different styles
        await summarizer.summarize_thread("thread_1", SummaryStyle.BULLET_POINTS)
        await summarizer.summarize_thread("thread_1", SummaryStyle.NARRATIVE)

        # Check stats
        stats = summarizer.get_cache_stats()
        assert stats['total_entries'] == 2

        # Clear cache for specific thread
        cleared = summarizer.clear_cache(thread_id="thread_1")
        assert cleared == 2

        # Stats should show 0 entries
        stats = summarizer.get_cache_stats()
        assert stats['total_entries'] == 0


class TestGracefulDegradation:
    """Graceful degradation tests (10)"""

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, mock_thread_manager):
        """Test 10: Fallback summary without LLM"""
        from hololoom.voice.threads.thread import ThreadSummarizer, SummaryStyle

        summarizer = ThreadSummarizer(
            thread_manager=mock_thread_manager,
            llm_client=None,  # No LLM
            enable_caching=False
        )

        result = await summarizer.summarize_thread(
            thread_id="thread_1",
            style=SummaryStyle.BULLET_POINTS
        )

        # Should still return a summary (fallback mode)
        assert result.thread_id == "thread_1"
        assert len(result.summary) > 0
        assert result.metadata.get('llm_generated') is False


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
