"""
Thread Merging Tests

Unit tests for ThreadMerger with 12 test cases.

Run:
    pytest HoloLoom/voice_first/thread/tests/test_thread_merging.py -v
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


class MockYarnGraph:
    """Mock YarnGraph for testing"""

    def __init__(self):
        self.edges: List[Dict[str, Any]] = []

    def add_edge(self, edge: Any) -> None:
        """Add edge to graph"""
        self.edges.append({
            'source': edge.src,
            'target': edge.dst,
            'relation': edge.type,
            'weight': edge.weight,
            'metadata': edge.metadata
        })


class MockLLMClient:
    """Mock LLM client for testing"""

    async def generate(self, prompt: str) -> str:
        """Mock LLM generation"""
        return f"Synthesis: This is a synthesized response based on the prompt."


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_thread_manager():
    """Create mock thread manager with sample threads"""
    manager = MockThreadManager()

    # Create target thread
    target_id = manager.create_thread("orchard planning")
    target = manager.get_thread(target_id)
    now = datetime.now().timestamp()

    target.messages = [
        Message(
            role="user",
            content="How should I plan my orchard?",
            timestamp=now - 100,
            metadata={}
        ),
        Message(
            role="assistant",
            content="Consider soil type and spacing.",
            timestamp=now - 95,
            metadata={}
        ),
    ]

    # Create source thread 1
    source1_id = manager.create_thread("biochar production")
    source1 = manager.get_thread(source1_id)

    source1.messages = [
        Message(
            role="user",
            content="What temperature for biochar?",
            timestamp=now - 80,
            metadata={}
        ),
        Message(
            role="assistant",
            content="400-700°C works best.",
            timestamp=now - 75,
            metadata={}
        ),
    ]

    # Create source thread 2
    source2_id = manager.create_thread("composting methods")
    source2 = manager.get_thread(source2_id)

    source2.messages = [
        Message(
            role="user",
            content="Hot vs cold composting?",
            timestamp=now - 60,
            metadata={}
        ),
        Message(
            role="assistant",
            content="Hot composting is faster.",
            timestamp=now - 55,
            metadata={}
        ),
    ]

    return manager


@pytest.fixture
def mock_yarn_graph():
    """Create mock yarn graph"""
    return MockYarnGraph()


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client"""
    return MockLLMClient()


# ============================================================================
# Tests
# ============================================================================

class TestThreadMergerBasics:
    """Basic functionality tests (1-4)"""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_thread_manager):
        """Test 1: ThreadMerger initialization"""
        from HoloLoom.voice_first.thread import ThreadMerger, MergeStrategy

        merger = ThreadMerger(
            thread_manager=mock_thread_manager,
            default_strategy=MergeStrategy.APPEND
        )

        assert merger.thread_manager is mock_thread_manager
        assert merger.default_strategy == MergeStrategy.APPEND

    @pytest.mark.asyncio
    async def test_basic_merge(self, mock_thread_manager):
        """Test 2: Basic thread merging"""
        from HoloLoom.voice_first.thread import ThreadMerger, MergeStrategy

        merger = ThreadMerger(
            thread_manager=mock_thread_manager,
            default_strategy=MergeStrategy.APPEND
        )

        result = await merger.merge_threads(
            target_thread_id="thread_1",  # orchard planning
            source_thread_ids=["thread_2"]  # biochar production
        )

        assert result.target_thread_id == "thread_1"
        assert result.source_thread_ids == ["thread_2"]
        assert result.strategy == MergeStrategy.APPEND
        assert result.messages_merged > 0

    @pytest.mark.asyncio
    async def test_target_not_found(self, mock_thread_manager):
        """Test 3: Error when target thread not found"""
        from HoloLoom.voice_first.thread import ThreadMerger

        merger = ThreadMerger(thread_manager=mock_thread_manager)

        with pytest.raises(ValueError, match="Target thread not found"):
            await merger.merge_threads(
                target_thread_id="nonexistent",
                source_thread_ids=["thread_2"]
            )

    @pytest.mark.asyncio
    async def test_source_not_found(self, mock_thread_manager):
        """Test 4: Error when source thread not found"""
        from HoloLoom.voice_first.thread import ThreadMerger

        merger = ThreadMerger(thread_manager=mock_thread_manager)

        with pytest.raises(ValueError, match="Source thread not found"):
            await merger.merge_threads(
                target_thread_id="thread_1",
                source_thread_ids=["nonexistent"]
            )


class TestMergeStrategies:
    """Merge strategy tests (5-8)"""

    @pytest.mark.asyncio
    async def test_append_strategy(self, mock_thread_manager):
        """Test 5: APPEND strategy - chronological concatenation"""
        from HoloLoom.voice_first.thread import ThreadMerger, MergeStrategy

        merger = ThreadMerger(thread_manager=mock_thread_manager)

        # Get initial message count
        target = mock_thread_manager.get_thread("thread_1")
        initial_count = len(target.messages)

        result = await merger.merge_threads(
            target_thread_id="thread_1",
            source_thread_ids=["thread_2"],
            strategy=MergeStrategy.APPEND
        )

        # Should have merged all messages chronologically
        final_count = len(target.messages)
        assert final_count > initial_count
        assert result.strategy == MergeStrategy.APPEND

    @pytest.mark.asyncio
    async def test_synthesize_strategy(
        self,
        mock_thread_manager,
        mock_llm_client
    ):
        """Test 6: SYNTHESIZE strategy - LLM synthesis"""
        from HoloLoom.voice_first.thread import ThreadMerger, MergeStrategy

        merger = ThreadMerger(
            thread_manager=mock_thread_manager,
            llm_client=mock_llm_client
        )

        result = await merger.merge_threads(
            target_thread_id="thread_1",
            source_thread_ids=["thread_2"],
            strategy=MergeStrategy.SYNTHESIZE
        )

        # Should have synthesis
        assert result.strategy == MergeStrategy.SYNTHESIZE
        assert result.synthesis is not None
        assert "Synthesis" in result.synthesis

    @pytest.mark.asyncio
    async def test_synthesize_without_llm(self, mock_thread_manager):
        """Test 7: SYNTHESIZE fallback without LLM"""
        from HoloLoom.voice_first.thread import ThreadMerger, MergeStrategy

        merger = ThreadMerger(
            thread_manager=mock_thread_manager,
            llm_client=None  # No LLM
        )

        result = await merger.merge_threads(
            target_thread_id="thread_1",
            source_thread_ids=["thread_2"],
            strategy=MergeStrategy.SYNTHESIZE
        )

        # Should fall back to APPEND
        assert result.synthesis is not None
        assert "No synthesis available" in result.synthesis

    @pytest.mark.asyncio
    async def test_preserve_all_strategy(self, mock_thread_manager):
        """Test 8: PRESERVE_ALL strategy - thread markers"""
        from HoloLoom.voice_first.thread import ThreadMerger, MergeStrategy

        merger = ThreadMerger(thread_manager=mock_thread_manager)

        result = await merger.merge_threads(
            target_thread_id="thread_1",
            source_thread_ids=["thread_2"],
            strategy=MergeStrategy.PRESERVE_ALL
        )

        # Check that thread markers are added
        target = mock_thread_manager.get_thread("thread_1")
        assert result.strategy == MergeStrategy.PRESERVE_ALL

        # Should have thread markers in content
        has_markers = any(
            '[' in msg.content and ']' in msg.content
            for msg in target.messages
        )
        assert has_markers


class TestMultipleThreads:
    """Multiple source thread tests (9-10)"""

    @pytest.mark.asyncio
    async def test_merge_multiple_sources(self, mock_thread_manager):
        """Test 9: Merge multiple source threads"""
        from HoloLoom.voice_first.thread import ThreadMerger

        merger = ThreadMerger(thread_manager=mock_thread_manager)

        result = await merger.merge_threads(
            target_thread_id="thread_1",
            source_thread_ids=["thread_2", "thread_3"]  # 2 sources
        )

        assert len(result.source_thread_ids) == 2
        assert result.messages_merged > 0

    @pytest.mark.asyncio
    async def test_target_in_sources_error(self, mock_thread_manager):
        """Test 10: Error when target is in sources"""
        from HoloLoom.voice_first.thread import ThreadMerger

        merger = ThreadMerger(thread_manager=mock_thread_manager)

        with pytest.raises(ValueError, match="Target thread cannot be in source"):
            await merger.merge_threads(
                target_thread_id="thread_1",
                source_thread_ids=["thread_1", "thread_2"]  # target in sources
            )


class TestYarnGraphIntegration:
    """YarnGraph integration tests (11-12)"""

    @pytest.mark.asyncio
    async def test_merge_edges_created(
        self,
        mock_thread_manager,
        mock_yarn_graph
    ):
        """Test 11: MERGED_INTO edges created in YarnGraph"""
        from HoloLoom.voice_first.thread import ThreadMerger

        merger = ThreadMerger(
            thread_manager=mock_thread_manager,
            yarn_graph=mock_yarn_graph
        )

        await merger.merge_threads(
            target_thread_id="thread_1",
            source_thread_ids=["thread_2", "thread_3"]
        )

        # Should have created 2 edges (one per source)
        assert len(mock_yarn_graph.edges) == 2

        # Check edge properties
        for edge in mock_yarn_graph.edges:
            assert edge['target'] == "thread_1"
            assert edge['source'] in ["thread_2", "thread_3"]
            assert edge['relation'] == "MERGED_INTO"

    @pytest.mark.asyncio
    async def test_no_yarngraph_graceful(self, mock_thread_manager):
        """Test 12: Graceful degradation without YarnGraph"""
        from HoloLoom.voice_first.thread import ThreadMerger

        merger = ThreadMerger(
            thread_manager=mock_thread_manager,
            yarn_graph=None  # No YarnGraph
        )

        # Should still work without YarnGraph
        result = await merger.merge_threads(
            target_thread_id="thread_1",
            source_thread_ids=["thread_2"]
        )

        assert result.target_thread_id == "thread_1"


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
