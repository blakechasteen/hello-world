"""
Thread Branching Tests

Unit tests for ThreadBrancher with 15 test cases.

Run:
    pytest HoloLoom/voice_first/thread/tests/test_thread_branching.py -v
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


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_thread_manager():
    """Create mock thread manager"""
    return MockThreadManager()


@pytest.fixture
def mock_yarn_graph():
    """Create mock yarn graph"""
    return MockYarnGraph()


@pytest.fixture
def parent_thread(mock_thread_manager):
    """Create parent thread with messages"""
    thread_id = mock_thread_manager.create_thread("orchard planning")
    thread = mock_thread_manager.get_thread(thread_id)

    now = datetime.now().timestamp()

    # Add messages spanning 60 seconds
    messages = [
        Message(
            role="user",
            content="How should I plan my orchard?",
            timestamp=now - 55,
            metadata={}
        ),
        Message(
            role="assistant",
            content="Consider soil type and Apple tree spacing.",
            timestamp=now - 50,
            metadata={}
        ),
        Message(
            role="user",
            content="What about biochar for Soil amendment?",
            timestamp=now - 20,
            metadata={}
        ),
        Message(
            role="assistant",
            content="Biochar can improve Soil structure.",
            timestamp=now - 15,
            metadata={}
        ),
        Message(
            role="user",
            content="Should I use Thompson Sampling for experiments?",
            timestamp=now - 5,
            metadata={}
        ),
    ]

    thread.messages = messages
    return thread


# ============================================================================
# Tests
# ============================================================================

class TestThreadBrancherBasics:
    """Basic functionality tests (1-5)"""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_thread_manager):
        """Test 1: ThreadBrancher initialization"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(
            thread_manager=mock_thread_manager,
            default_lookback_seconds=30.0,
            enable_entity_extraction=True
        )

        assert brancher.thread_manager is mock_thread_manager
        assert brancher.default_lookback_seconds == 30.0
        assert brancher.enable_entity_extraction is True

    @pytest.mark.asyncio
    async def test_basic_fork(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 2: Basic thread forking"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(
            thread_manager=mock_thread_manager,
            default_lookback_seconds=30.0
        )

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="biochar experiments"
        )

        assert branch.branch_id is not None
        assert branch.branch_name == "biochar experiments"
        assert branch.parent_thread_id == parent_thread.thread_id
        assert branch.parent_thread_name == "orchard planning"

    @pytest.mark.asyncio
    async def test_parent_not_found(self, mock_thread_manager):
        """Test 3: Error when parent thread not found"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(thread_manager=mock_thread_manager)

        with pytest.raises(ValueError, match="Parent thread not found"):
            await brancher.fork_thread(
                parent_thread_id="nonexistent",
                branch_name="test"
            )

    @pytest.mark.asyncio
    async def test_branch_created_in_manager(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 4: Branch thread created in ThreadManager"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(thread_manager=mock_thread_manager)

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="biochar experiments"
        )

        # Verify thread exists in manager
        created_thread = mock_thread_manager.get_thread(branch.branch_id)
        assert created_thread is not None
        assert created_thread.name == "biochar experiments"

    @pytest.mark.asyncio
    async def test_branch_metadata(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 5: Branch includes correct metadata"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(thread_manager=mock_thread_manager)

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="biochar experiments"
        )

        assert 'lookback_seconds' in branch.metadata
        assert 'message_count' in branch.metadata
        assert 'entity_count' in branch.metadata
        assert branch.metadata['lookback_seconds'] == 30.0


class TestContextInheritance:
    """Context inheritance tests (6-10)"""

    @pytest.mark.asyncio
    async def test_default_context_window(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 6: Default 30-second context window"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(
            thread_manager=mock_thread_manager,
            default_lookback_seconds=30.0
        )

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="test"
        )

        # Should include messages from last 30 seconds
        # Parent has messages at: -55, -50, -20, -15, -5 seconds
        # Last 30s should include: -20, -15, -5 (3 messages)
        assert len(branch.context.messages) == 3

    @pytest.mark.asyncio
    async def test_custom_context_window(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 7: Custom context window"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(thread_manager=mock_thread_manager)

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="test",
            lookback_seconds=60.0  # Custom: 60 seconds
        )

        # Should include all messages (all within 60s)
        assert len(branch.context.messages) == 5

    @pytest.mark.asyncio
    async def test_inherited_messages(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 8: Inherited messages added to new thread"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(
            thread_manager=mock_thread_manager,
            default_lookback_seconds=30.0
        )

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="test"
        )

        # Get created thread
        thread = mock_thread_manager.get_thread(branch.branch_id)

        # Should have inherited messages
        assert len(thread.messages) == 3
        assert thread.messages[0].content == "What about biochar for Soil amendment?"

    @pytest.mark.asyncio
    async def test_entity_extraction(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 9: Entity extraction from context"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(
            thread_manager=mock_thread_manager,
            default_lookback_seconds=30.0,
            enable_entity_extraction=True
        )

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="test"
        )

        # Should extract entities: Apple, Soil, Biochar, Thompson, Sampling
        # (capitalized words from last 30s)
        entities = branch.context.entities
        assert len(entities) > 0
        assert 'Biochar' in entities
        assert 'Soil' in entities

    @pytest.mark.asyncio
    async def test_custom_context_override(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 10: Custom context override"""
        from hololoom.voice.threads.thread import (
            ThreadBrancher,
            BranchContext
        )

        brancher = ThreadBrancher(thread_manager=mock_thread_manager)

        custom_context = BranchContext(
            messages=[
                {
                    'role': 'user',
                    'content': 'Custom message',
                    'timestamp': datetime.now().timestamp(),
                    'metadata': {}
                }
            ],
            entities=['Custom', 'Entity'],
            timestamp=datetime.now().timestamp(),
            lookback_seconds=10.0
        )

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="test",
            custom_context=custom_context
        )

        # Should use custom context
        assert len(branch.context.messages) == 1
        assert branch.context.messages[0]['content'] == 'Custom message'
        assert 'Custom' in branch.context.entities


class TestYarnGraphIntegration:
    """YarnGraph integration tests (11-13)"""

    @pytest.mark.asyncio
    async def test_branch_edge_created(
        self,
        mock_thread_manager,
        mock_yarn_graph,
        parent_thread
    ):
        """Test 11: BRANCHED_FROM edge created in YarnGraph"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(
            thread_manager=mock_thread_manager,
            yarn_graph=mock_yarn_graph
        )

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="test"
        )

        # Should have created edge
        assert len(mock_yarn_graph.edges) == 1
        edge = mock_yarn_graph.edges[0]
        assert edge['source'] == branch.branch_id
        assert edge['target'] == parent_thread.thread_id
        assert edge['relation'] == 'BRANCHED_FROM'

    @pytest.mark.asyncio
    async def test_edge_metadata(
        self,
        mock_thread_manager,
        mock_yarn_graph,
        parent_thread
    ):
        """Test 12: Edge includes branch metadata"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(
            thread_manager=mock_thread_manager,
            yarn_graph=mock_yarn_graph
        )

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="biochar experiments"
        )

        edge = mock_yarn_graph.edges[0]
        assert 'branch_name' in edge['metadata']
        assert edge['metadata']['branch_name'] == "biochar experiments"
        assert 'timestamp' in edge['metadata']

    @pytest.mark.asyncio
    async def test_no_yarngraph_graceful(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 13: Graceful degradation without YarnGraph"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(
            thread_manager=mock_thread_manager,
            yarn_graph=None  # No YarnGraph
        )

        # Should still work without YarnGraph
        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="test"
        )

        assert branch.branch_id is not None


class TestEdgeCases:
    """Edge cases and error handling (14-15)"""

    @pytest.mark.asyncio
    async def test_empty_parent_thread(self, mock_thread_manager):
        """Test 14: Fork from empty parent thread"""
        from hololoom.voice.threads.thread import ThreadBrancher

        thread_id = mock_thread_manager.create_thread("empty thread")
        brancher = ThreadBrancher(thread_manager=mock_thread_manager)

        branch = await brancher.fork_thread(
            parent_thread_id=thread_id,
            branch_name="test"
        )

        # Should work with no messages
        assert len(branch.context.messages) == 0
        assert len(branch.context.entities) == 0

    @pytest.mark.asyncio
    async def test_branch_serialization(
        self,
        mock_thread_manager,
        parent_thread
    ):
        """Test 15: Branch serialization to dict"""
        from hololoom.voice.threads.thread import ThreadBrancher

        brancher = ThreadBrancher(thread_manager=mock_thread_manager)

        branch = await brancher.fork_thread(
            parent_thread_id=parent_thread.thread_id,
            branch_name="test"
        )

        # Should serialize to dict
        data = branch.to_dict()
        assert isinstance(data, dict)
        assert data['branch_name'] == "test"
        assert data['parent_thread_id'] == parent_thread.thread_id
        assert 'context' in data
        assert 'messages' in data['context']


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
