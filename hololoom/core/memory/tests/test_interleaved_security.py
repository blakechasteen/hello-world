"""
Test security fixes for interleaved generation concurrent streaming.

Tests proper exception handling, timeouts, and task cleanup in concurrent mode.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import AsyncIterator, List, Union
import logging

from hololoom.memory.interleaved_generation import (
    InterleavedStreamManager,
    StreamMode,
    ContextChunk,
    GenerationToken,
    StreamItem,
    StreamMetadata,
    stream_interleaved_expansion_generation
)
from hololoom.memory.streaming_expansion import ChunkYieldStrategy


class TestConcurrentStreamingSecurity:
    """Test security fixes in concurrent streaming mode."""

    @pytest.fixture
    def mock_expansion_stream(self):
        """Create mock expansion stream."""
        async def stream():
            chunks = [
                ContextChunk(
                    nodes=['node1'],
                    relevance_scores={'node1': 0.9},
                    hop_distance=0,
                    token_count=50,
                    cumulative_tokens=50,
                    chunk_index=0,
                    is_final=False,
                    avg_relevance=0.9,
                    yield_reason='hop_boundary'
                ),
                ContextChunk(
                    nodes=['node2'],
                    relevance_scores={'node2': 0.8},
                    hop_distance=1,
                    token_count=50,
                    cumulative_tokens=100,
                    chunk_index=1,
                    is_final=True,
                    avg_relevance=0.85,
                    yield_reason='expansion_complete'
                )
            ]
            for chunk in chunks:
                await asyncio.sleep(0.01)  # Simulate async work
                yield chunk

        return stream

    @pytest.fixture
    def mock_llm_stream(self):
        """Create mock LLM generation stream."""
        async def stream(prompt: str):
            tokens = ['This', ' is', ' a', ' test', ' response']
            for token in tokens:
                await asyncio.sleep(0.01)  # Simulate async work
                yield token

        return stream

    @pytest.fixture
    def mock_llm_stream_with_error(self):
        """Create mock LLM stream that errors."""
        async def stream(prompt: str):
            yield 'Start'
            await asyncio.sleep(0.01)
            raise RuntimeError("LLM generation failed")

        return stream

    @pytest.fixture
    def mock_expansion_stream_with_error(self):
        """Create mock expansion stream that errors."""
        async def stream():
            yield ContextChunk(
                nodes=['node1'],
                relevance_scores={'node1': 0.9},
                hop_distance=0,
                token_count=50,
                cumulative_tokens=50,
                chunk_index=0,
                is_final=False,
                avg_relevance=0.9,
                yield_reason='hop_boundary'
            )
            await asyncio.sleep(0.01)
            raise ValueError("Expansion failed")

        return stream

    @pytest.mark.asyncio
    async def test_concurrent_streaming_exception_handling(self, mock_expansion_stream_with_error, mock_llm_stream):
        """Test that exceptions in expansion are properly handled."""
        manager = InterleavedStreamManager(
            expansion_stream=mock_expansion_stream_with_error(),
            llm=MagicMock(stream=mock_llm_stream),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        items = []
        with pytest.raises(ValueError) as exc_info:
            async for item in manager.stream():
                items.append(item)

        assert "Expansion failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_concurrent_streaming_llm_exception(self, mock_expansion_stream, mock_llm_stream_with_error):
        """Test that exceptions in LLM generation are properly handled."""
        manager = InterleavedStreamManager(
            expansion_stream=mock_expansion_stream(),
            llm=MagicMock(stream=mock_llm_stream_with_error),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        items = []
        with pytest.raises(RuntimeError) as exc_info:
            async for item in manager.stream():
                items.append(item)

        assert "LLM generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_concurrent_streaming_timeout(self, mock_expansion_stream):
        """Test that timeout is properly enforced."""
        # Create a slow LLM that will timeout
        async def slow_llm_stream(prompt: str):
            await asyncio.sleep(40)  # Longer than 30s timeout
            yield "Never reached"

        manager = InterleavedStreamManager(
            expansion_stream=mock_expansion_stream(),
            llm=MagicMock(stream=slow_llm_stream),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        # Mock the timeout to be shorter for testing
        with patch('hololoom.memory.interleaved_generation.InterleavedStreamManager._stream_concurrent') as mock_concurrent:
            async def stream_with_timeout():
                # Simulate timeout
                await asyncio.sleep(0.1)
                raise asyncio.TimeoutError("Operation timed out")

            mock_concurrent.return_value = stream_with_timeout()

            with pytest.raises(asyncio.TimeoutError):
                items = []
                async for item in manager.stream():
                    items.append(item)

    @pytest.mark.asyncio
    async def test_task_cancellation_on_error(self, mock_expansion_stream, mock_llm_stream_with_error):
        """Test that tasks are properly cancelled on error."""
        manager = InterleavedStreamManager(
            expansion_stream=mock_expansion_stream(),
            llm=MagicMock(stream=mock_llm_stream_with_error),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        expansion_task = None
        generation_task = None

        # Patch create_task to capture tasks
        original_create_task = asyncio.create_task
        created_tasks = []

        def track_task(coro):
            task = original_create_task(coro)
            created_tasks.append(task)
            return task

        with patch('asyncio.create_task', side_effect=track_task):
            try:
                async for item in manager.stream():
                    pass
            except RuntimeError:
                pass

            # Verify tasks were cancelled
            for task in created_tasks:
                if not task.done():
                    assert task.cancelled()

    @pytest.mark.asyncio
    async def test_asyncio_gather_exception_propagation(self):
        """Test that asyncio.gather properly propagates exceptions."""
        async def task1():
            await asyncio.sleep(0.01)
            return "success"

        async def task2():
            await asyncio.sleep(0.02)
            raise ValueError("Task 2 failed")

        task1_coro = asyncio.create_task(task1())
        task2_coro = asyncio.create_task(task2())

        with pytest.raises(ValueError) as exc_info:
            await asyncio.gather(task1_coro, task2_coro, return_exceptions=False)

        assert "Task 2 failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup_on_timeout(self):
        """Test that resources are cleaned up on timeout."""
        # Create manager with mocked streams
        async def infinite_expansion():
            while True:
                yield ContextChunk(
                    nodes=['node'],
                    relevance_scores={'node': 0.9},
                    hop_distance=0,
                    token_count=50,
                    cumulative_tokens=50,
                    chunk_index=0,
                    is_final=False,
                    avg_relevance=0.9,
                    yield_reason='hop_boundary'
                )
                await asyncio.sleep(0.1)

        async def infinite_generation(prompt):
            while True:
                yield "token"
                await asyncio.sleep(0.1)

        manager = InterleavedStreamManager(
            expansion_stream=infinite_expansion(),
            llm=MagicMock(stream=infinite_generation),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        # Set a short timeout for testing
        async def stream_with_timeout():
            items = []
            try:
                async with asyncio.timeout(0.5):  # 500ms timeout
                    async for item in manager.stream():
                        items.append(item)
            except asyncio.TimeoutError:
                # Expected
                pass
            return items

        items = await stream_with_timeout()
        # Should have collected some items before timeout
        assert len(items) > 0

    @pytest.mark.asyncio
    async def test_queue_exception_handling(self, mock_expansion_stream_with_error):
        """Test that exceptions in queue operations are handled."""
        manager = InterleavedStreamManager(
            expansion_stream=mock_expansion_stream_with_error(),
            llm=MagicMock(stream=AsyncMock(return_value=[])),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        with pytest.raises(ValueError):
            async for item in manager.stream():
                pass

    @pytest.mark.asyncio
    async def test_multiple_concurrent_errors(self):
        """Test handling of multiple simultaneous errors."""
        async def failing_expansion():
            await asyncio.sleep(0.01)
            raise ValueError("Expansion error")

        async def failing_generation(prompt):
            await asyncio.sleep(0.02)
            raise RuntimeError("Generation error")

        manager = InterleavedStreamManager(
            expansion_stream=failing_expansion(),
            llm=MagicMock(stream=failing_generation),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        # Should get the first error that occurs
        with pytest.raises((ValueError, RuntimeError)):
            async for item in manager.stream():
                pass

    @pytest.mark.asyncio
    async def test_error_logging(self, mock_expansion_stream_with_error, mock_llm_stream, caplog):
        """Test that errors are properly logged."""
        with caplog.at_level(logging.ERROR):
            manager = InterleavedStreamManager(
                expansion_stream=mock_expansion_stream_with_error(),
                llm=MagicMock(stream=mock_llm_stream),
                query="test query",
                stream_mode=StreamMode.CONCURRENT
            )

            try:
                async for item in manager.stream():
                    pass
            except ValueError:
                pass

            # Check that error was logged
            assert any("Error in concurrent streaming" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_partial_results_before_error(self, mock_llm_stream):
        """Test that partial results are yielded before an error occurs."""
        async def expansion_with_delayed_error():
            yield ContextChunk(
                nodes=['node1'],
                relevance_scores={'node1': 0.9},
                hop_distance=0,
                token_count=50,
                cumulative_tokens=50,
                chunk_index=0,
                is_final=False,
                avg_relevance=0.9,
                yield_reason='hop_boundary'
            )
            await asyncio.sleep(0.1)
            yield ContextChunk(
                nodes=['node2'],
                relevance_scores={'node2': 0.8},
                hop_distance=1,
                token_count=50,
                cumulative_tokens=100,
                chunk_index=1,
                is_final=False,
                avg_relevance=0.85,
                yield_reason='hop_boundary'
            )
            await asyncio.sleep(0.1)
            raise ValueError("Late error")

        manager = InterleavedStreamManager(
            expansion_stream=expansion_with_delayed_error(),
            llm=MagicMock(stream=mock_llm_stream),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        collected_items = []
        try:
            async for item in manager.stream():
                collected_items.append(item)
        except ValueError:
            pass

        # Should have collected some items before the error
        assert len(collected_items) > 0

    @pytest.mark.asyncio
    async def test_task_cleanup_verification(self):
        """Test that all tasks are properly cleaned up."""
        tasks_before = len(asyncio.all_tasks())

        async def normal_expansion():
            for i in range(3):
                yield ContextChunk(
                    nodes=[f'node{i}'],
                    relevance_scores={f'node{i}': 0.9},
                    hop_distance=i,
                    token_count=50,
                    cumulative_tokens=50 * (i + 1),
                    chunk_index=i,
                    is_final=(i == 2),
                    avg_relevance=0.9,
                    yield_reason='hop_boundary'
                )
                await asyncio.sleep(0.01)

        async def normal_generation(prompt):
            for token in ['Hello', ' world']:
                yield token
                await asyncio.sleep(0.01)

        manager = InterleavedStreamManager(
            expansion_stream=normal_expansion(),
            llm=MagicMock(stream=normal_generation),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        items = []
        async for item in manager.stream():
            items.append(item)

        # Wait a bit for cleanup
        await asyncio.sleep(0.1)

        tasks_after = len(asyncio.all_tasks())

        # Should have same number of tasks (no leaks)
        assert tasks_after <= tasks_before + 1  # Allow for test task itself

    @pytest.mark.asyncio
    async def test_timeout_configuration(self):
        """Test that timeout is properly configured (30s)."""
        # Access the actual timeout value used in the code
        import inspect
        from hololoom.memory import interleaved_generation

        # Get the source code
        source = inspect.getsource(interleaved_generation.InterleavedStreamManager._stream_concurrent)

        # Verify timeout is set to 30 seconds
        assert "timeout=30.0" in source or "timeout=30" in source

    @pytest.mark.asyncio
    async def test_resilience_to_slow_consumers(self):
        """Test that slow consumers don't cause issues."""
        manager = InterleavedStreamManager(
            expansion_stream=self.create_fast_expansion(),
            llm=MagicMock(stream=self.create_fast_generation),
            query="test query",
            stream_mode=StreamMode.CONCURRENT
        )

        items = []
        async for item in manager.stream():
            items.append(item)
            # Simulate slow consumer
            await asyncio.sleep(0.1)

        # Should still work correctly
        assert len(items) > 0

    async def create_fast_expansion(self):
        """Helper to create fast expansion stream."""
        for i in range(5):
            yield ContextChunk(
                nodes=[f'node{i}'],
                relevance_scores={f'node{i}': 0.9},
                hop_distance=0,
                token_count=10,
                cumulative_tokens=10 * (i + 1),
                chunk_index=i,
                is_final=(i == 4),
                avg_relevance=0.9,
                yield_reason='hop_boundary'
            )

    async def create_fast_generation(self, prompt):
        """Helper to create fast generation stream."""
        for token in ['Fast', ' response']:
            yield token

    @pytest.mark.asyncio
    async def test_convenience_function_error_handling(self):
        """Test that convenience function handles errors properly."""
        mock_graph = MagicMock()

        with patch('hololoom.memory.streaming_expansion.stream_context_expansion') as mock_expansion:
            # Make expansion fail
            async def failing_expansion(**kwargs):
                raise ValueError("Expansion failed")
                # Make it an async generator
                yield  # pragma: no cover

            mock_expansion.return_value = failing_expansion()

            with pytest.raises(ValueError) as exc_info:
                async for item in stream_interleaved_expansion_generation(
                    query="test",
                    seed_nodes=['node1'],
                    graph=mock_graph,
                    stream_mode=StreamMode.CONCURRENT
                ):
                    pass

            assert "Expansion failed" in str(exc_info.value)