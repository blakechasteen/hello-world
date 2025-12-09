"""
Unit tests for Portal Job Queue.

Tests JobQueue and QueueDispatcher functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from ..job_queue import (
    JobQueue,
    QueueDispatcher,
    QueuedJob,
    QueuedJobStatus,
    create_job_queue_system,
)


def make_job(
    job_id: str = "test-1",
    module_id: str = "test-module",
    priority: int = 0,
) -> QueuedJob:
    """Create a test job."""
    return QueuedJob(
        job_id=job_id,
        module_id=module_id,
        input_json={"test": True},
        priority=priority,
    )


class TestJobQueue:
    """Tests for JobQueue class."""

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self):
        """Basic enqueue and dequeue works."""
        queue = JobQueue(max_size=100)

        job = make_job("job-1")
        assert await queue.enqueue(job)

        retrieved = await queue.dequeue()
        assert retrieved is not None
        assert retrieved.job_id == "job-1"
        assert retrieved.status == QueuedJobStatus.DISPATCHING

    @pytest.mark.asyncio
    async def test_fifo_ordering(self):
        """Jobs are dequeued in FIFO order (same priority)."""
        queue = JobQueue(max_size=100)

        await queue.enqueue(make_job("first"))
        await queue.enqueue(make_job("second"))
        await queue.enqueue(make_job("third"))

        assert (await queue.dequeue()).job_id == "first"
        assert (await queue.dequeue()).job_id == "second"
        assert (await queue.dequeue()).job_id == "third"

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Higher priority jobs are dequeued first."""
        queue = JobQueue(max_size=100)

        # Add low priority first
        await queue.enqueue(make_job("low", priority=0))
        # Add high priority second
        await queue.enqueue(make_job("high", priority=10))
        # Add medium priority third
        await queue.enqueue(make_job("medium", priority=5))

        # Should come out in priority order (high, medium, low)
        assert (await queue.dequeue()).job_id == "high"
        assert (await queue.dequeue()).job_id == "medium"
        assert (await queue.dequeue()).job_id == "low"

    @pytest.mark.asyncio
    async def test_max_size_enforced(self):
        """Queue rejects jobs when full."""
        queue = JobQueue(max_size=2)

        assert await queue.enqueue(make_job("1"))
        assert await queue.enqueue(make_job("2"))
        assert not await queue.enqueue(make_job("3"))  # Should fail

        assert queue.is_full()
        stats = queue.get_stats()
        assert stats["current_depth"] == 2
        assert stats["utilization_percent"] == 100.0

    @pytest.mark.asyncio
    async def test_empty_dequeue_returns_none(self):
        """Dequeue from empty queue returns None."""
        queue = JobQueue(max_size=100)
        assert await queue.dequeue() is None

    @pytest.mark.asyncio
    async def test_peek_does_not_remove(self):
        """Peek returns job without removing it."""
        queue = JobQueue(max_size=100)
        await queue.enqueue(make_job("test"))

        # Peek twice
        peeked1 = await queue.peek()
        peeked2 = await queue.peek()

        assert peeked1 is not None
        assert peeked2 is not None
        assert peeked1.job_id == peeked2.job_id

        # Job still in queue
        dequeued = await queue.dequeue()
        assert dequeued.job_id == "test"

    @pytest.mark.asyncio
    async def test_mark_dispatched(self):
        """Marking as dispatched removes from tracking."""
        queue = JobQueue(max_size=100)
        await queue.enqueue(make_job("test"))

        job = await queue.dequeue()
        await queue.mark_dispatched(job.job_id)

        # Should not be findable anymore
        status = await queue.get_job_status("test")
        assert status is None

        stats = queue.get_stats()
        assert stats["total_dispatched"] == 1

    @pytest.mark.asyncio
    async def test_mark_failed_retries(self):
        """Failed jobs are re-queued with retry."""
        queue = JobQueue(max_size=100)
        job = make_job("test")
        job.max_attempts = 3
        await queue.enqueue(job)

        # First attempt fails
        dequeued = await queue.dequeue()
        await queue.mark_failed(dequeued.job_id, "Connection error")

        # Should be re-queued
        assert not queue.is_empty()
        re_dequeued = await queue.dequeue()
        assert re_dequeued.job_id == "test"
        assert re_dequeued.attempts == 1
        assert re_dequeued.last_error == "Connection error"

    @pytest.mark.asyncio
    async def test_mark_failed_max_attempts(self):
        """Jobs are removed after max attempts."""
        queue = JobQueue(max_size=100)
        job = make_job("test")
        job.max_attempts = 2
        await queue.enqueue(job)

        # First attempt
        dequeued = await queue.dequeue()
        await queue.mark_failed(dequeued.job_id, "Error 1")

        # Second attempt (final)
        dequeued = await queue.dequeue()
        await queue.mark_failed(dequeued.job_id, "Error 2")

        # Should be permanently failed
        assert queue.is_empty()
        status = await queue.get_job_status("test")
        assert status is None

        stats = queue.get_stats()
        assert stats["total_failed"] == 1

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self):
        """Pending jobs can be cancelled."""
        queue = JobQueue(max_size=100)
        await queue.enqueue(make_job("to-cancel"))
        await queue.enqueue(make_job("keep"))

        assert await queue.cancel_job("to-cancel")

        # Only one job left
        dequeued = await queue.dequeue()
        assert dequeued.job_id == "keep"
        assert await queue.dequeue() is None

    @pytest.mark.asyncio
    async def test_cancel_dispatching_job_fails(self):
        """Can't cancel job that's already dispatching."""
        queue = JobQueue(max_size=100)
        await queue.enqueue(make_job("test"))

        # Start dispatching
        await queue.dequeue()

        # Cancel should fail
        assert not await queue.cancel_job("test")

    @pytest.mark.asyncio
    async def test_wait_time_tracking(self):
        """Average wait time is tracked."""
        queue = JobQueue(max_size=100)

        # Add job and immediately dequeue
        await queue.enqueue(make_job("1"))
        await queue.dequeue()

        stats = queue.get_stats()
        assert stats["avg_wait_time_ms"] >= 0  # Should have some value

    @pytest.mark.asyncio
    async def test_stats_accuracy(self):
        """Stats accurately reflect queue state."""
        queue = JobQueue(max_size=100)

        await queue.enqueue(make_job("1"))
        await queue.enqueue(make_job("2"))

        stats = queue.get_stats()
        assert stats["total_enqueued"] == 2
        assert stats["current_depth"] == 2
        assert stats["total_dispatched"] == 0

        job = await queue.dequeue()
        await queue.mark_dispatched(job.job_id)

        stats = queue.get_stats()
        assert stats["current_depth"] == 1
        assert stats["total_dispatched"] == 1


class TestQueueDispatcher:
    """Tests for QueueDispatcher class."""

    @pytest.mark.asyncio
    async def test_dispatcher_starts_stops(self):
        """Dispatcher can start and stop cleanly."""
        queue = JobQueue(max_size=100)
        scheduler = MagicMock()

        dispatcher = QueueDispatcher(queue, scheduler, check_interval=0.1)

        await dispatcher.start()
        assert dispatcher.is_running

        await dispatcher.stop()
        assert not dispatcher.is_running

    @pytest.mark.asyncio
    async def test_dispatcher_dispatches_job(self):
        """Dispatcher picks up and dispatches queued jobs."""
        queue = JobQueue(max_size=100)

        # Mock scheduler
        mock_node = MagicMock()
        mock_node.node_id = "test-node"

        scheduler = MagicMock()
        scheduler.pick_node = AsyncMock(return_value=mock_node)
        scheduler.dispatch_job_async = AsyncMock(
            return_value={"status": "running", "job_id": "test"}
        )

        dispatcher = QueueDispatcher(queue, scheduler, check_interval=0.05)

        # Add job before starting
        await queue.enqueue(make_job("test"))

        # Start and let it process
        await dispatcher.start()
        await asyncio.sleep(0.15)
        await dispatcher.stop()

        # Job should have been dispatched
        scheduler.pick_node.assert_called()
        scheduler.dispatch_job_async.assert_called()

        stats = dispatcher.get_stats()
        assert stats["dispatched_count"] == 1

    @pytest.mark.asyncio
    async def test_dispatcher_handles_no_nodes(self):
        """Dispatcher re-queues when no nodes available."""
        queue = JobQueue(max_size=100)

        scheduler = MagicMock()
        scheduler.pick_node = AsyncMock(return_value=None)  # No nodes

        dispatcher = QueueDispatcher(queue, scheduler, check_interval=0.05)

        job = make_job("test")
        job.max_attempts = 2
        await queue.enqueue(job)

        await dispatcher.start()
        await asyncio.sleep(0.2)
        await dispatcher.stop()

        # Should have failed
        stats = dispatcher.get_stats()
        assert stats["failed_count"] >= 1

    @pytest.mark.asyncio
    async def test_dispatcher_handles_dispatch_failure(self):
        """Dispatcher handles failed dispatch gracefully."""
        queue = JobQueue(max_size=100)

        mock_node = MagicMock()
        scheduler = MagicMock()
        scheduler.pick_node = AsyncMock(return_value=mock_node)
        scheduler.dispatch_job_async = AsyncMock(
            return_value={"status": "failed", "error": "Node unreachable"}
        )

        dispatcher = QueueDispatcher(queue, scheduler, check_interval=0.05)

        job = make_job("test")
        job.max_attempts = 1
        await queue.enqueue(job)

        await dispatcher.start()
        await asyncio.sleep(0.15)
        await dispatcher.stop()

        stats = dispatcher.get_stats()
        assert stats["failed_count"] == 1


class TestCreateJobQueueSystem:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_factory_creates_system(self):
        """Factory creates queue and dispatcher."""
        scheduler = MagicMock()

        queue, dispatcher = await create_job_queue_system(
            scheduler=scheduler,
            max_queue_size=500,
            check_interval=1.0,
            auto_start=False,
        )

        assert queue.max_size == 500
        assert dispatcher.check_interval == 1.0
        assert not dispatcher.is_running

    @pytest.mark.asyncio
    async def test_factory_auto_starts(self):
        """Factory auto-starts dispatcher when requested."""
        scheduler = MagicMock()

        queue, dispatcher = await create_job_queue_system(
            scheduler=scheduler,
            auto_start=True,
        )

        try:
            assert dispatcher.is_running
        finally:
            await dispatcher.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
