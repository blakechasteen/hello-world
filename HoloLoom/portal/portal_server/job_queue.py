"""
Central Job Queue - Queue jobs when all nodes busy.

Jobs are queued if no nodes available, then dispatched
by background task when capacity frees up.

Usage:
    queue = JobQueue(max_size=10000)
    dispatcher = QueueDispatcher(queue, scheduler, check_interval=5.0)

    # Queue a job
    job = QueuedJob(job_id="abc", module_id="test", input_json={"x": 1})
    await queue.enqueue(job)

    # Start background dispatcher
    await dispatcher.start()

    # Stop when done
    await dispatcher.stop()
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Deque, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..shuttle_bot.scheduler import SimpleScheduler


class QueuedJobStatus(str, Enum):
    """Job status in the queue lifecycle."""
    PENDING = "pending"           # Waiting in queue
    DISPATCHING = "dispatching"   # Being dispatched to node
    DISPATCHED = "dispatched"     # Successfully sent to node
    FAILED = "failed"             # Dispatch failed after max attempts


@dataclass
class QueuedJob:
    """
    A job waiting in the queue.

    Attributes:
        job_id: Unique job identifier
        module_id: WASM module to execute
        input_json: Input data for the job
        job_requirements: Optional requirements (gpu, min_memory_mb, etc.)
        timeout_seconds: Job execution timeout
        priority: Higher = more important (0 is lowest)
        queued_at: Timestamp when job was queued
        status: Current job status
        attempts: Number of dispatch attempts
        max_attempts: Maximum dispatch attempts before failure
        last_error: Most recent error message
    """
    job_id: str
    module_id: str
    input_json: Dict[str, Any]
    job_requirements: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 60
    priority: int = 0
    queued_at: float = field(default_factory=time.time)
    status: QueuedJobStatus = QueuedJobStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None


@dataclass
class QueueStats:
    """Queue metrics for monitoring."""
    total_enqueued: int = 0
    total_dispatched: int = 0
    total_failed: int = 0
    current_depth: int = 0
    avg_wait_time_ms: float = 0.0


class JobQueue:
    """
    In-memory job queue with priority support.

    Jobs are stored in a deque and dispatched FIFO by default.
    Higher priority jobs (priority > 0) are inserted ahead of
    lower priority jobs.

    Thread-safe via asyncio.Lock for concurrent access.

    Args:
        max_size: Maximum queue size (rejects new jobs when full)
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: Deque[QueuedJob] = deque()
        self._jobs_by_id: Dict[str, QueuedJob] = {}
        self._stats = QueueStats()
        self._lock = asyncio.Lock()
        self._wait_times: List[float] = []  # Rolling window for avg calculation

    async def enqueue(self, job: QueuedJob) -> bool:
        """
        Add job to queue.

        Args:
            job: Job to enqueue

        Returns:
            True if enqueued, False if queue is full
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                return False

            # Insert by priority (higher priority first)
            inserted = False
            for i, existing in enumerate(self._queue):
                if job.priority > existing.priority:
                    self._queue.insert(i, job)
                    inserted = True
                    break

            if not inserted:
                self._queue.append(job)

            self._jobs_by_id[job.job_id] = job
            self._stats.total_enqueued += 1
            self._stats.current_depth = len(self._queue)
            return True

    async def dequeue(self) -> Optional[QueuedJob]:
        """
        Get next job from queue.

        Returns:
            Next job, or None if queue is empty
        """
        async with self._lock:
            if not self._queue:
                return None

            job = self._queue.popleft()
            job.status = QueuedJobStatus.DISPATCHING
            self._stats.current_depth = len(self._queue)

            # Track wait time (rolling window of 100)
            wait_ms = (time.time() - job.queued_at) * 1000
            self._wait_times.append(wait_ms)
            if len(self._wait_times) > 100:
                self._wait_times.pop(0)
            self._stats.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)

            return job

    async def peek(self) -> Optional[QueuedJob]:
        """
        Peek at next job without removing it.

        Returns:
            Next job, or None if queue is empty
        """
        async with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    async def mark_dispatched(self, job_id: str) -> None:
        """
        Mark job as successfully dispatched.

        Args:
            job_id: Job to mark as dispatched
        """
        async with self._lock:
            if job_id in self._jobs_by_id:
                self._jobs_by_id[job_id].status = QueuedJobStatus.DISPATCHED
                self._stats.total_dispatched += 1
                del self._jobs_by_id[job_id]

    async def mark_failed(self, job_id: str, error: str) -> None:
        """
        Mark job as failed. Re-queue if attempts remain.

        Args:
            job_id: Job that failed
            error: Error message
        """
        async with self._lock:
            if job_id not in self._jobs_by_id:
                return

            job = self._jobs_by_id[job_id]
            job.attempts += 1
            job.last_error = error

            if job.attempts < job.max_attempts:
                # Re-queue with lower priority (demote on failure)
                job.status = QueuedJobStatus.PENDING
                job.priority = max(0, job.priority - 1)
                self._queue.append(job)
                self._stats.current_depth = len(self._queue)
            else:
                # Max attempts reached - permanent failure
                job.status = QueuedJobStatus.FAILED
                self._stats.total_failed += 1
                del self._jobs_by_id[job_id]

    async def get_job_status(self, job_id: str) -> Optional[QueuedJob]:
        """
        Get job by ID (for status polling).

        Args:
            job_id: Job ID to look up

        Returns:
            Job if found, None otherwise
        """
        async with self._lock:
            return self._jobs_by_id.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: Job to cancel

        Returns:
            True if cancelled, False if not found or already dispatching
        """
        async with self._lock:
            if job_id not in self._jobs_by_id:
                return False

            job = self._jobs_by_id[job_id]
            if job.status != QueuedJobStatus.PENDING:
                return False  # Can't cancel if already dispatching

            # Remove from queue
            try:
                self._queue.remove(job)
            except ValueError:
                pass  # Already removed

            del self._jobs_by_id[job_id]
            self._stats.current_depth = len(self._queue)
            return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get queue metrics for monitoring.

        Returns:
            Dict with queue statistics
        """
        return {
            "total_enqueued": self._stats.total_enqueued,
            "total_dispatched": self._stats.total_dispatched,
            "total_failed": self._stats.total_failed,
            "current_depth": self._stats.current_depth,
            "avg_wait_time_ms": round(self._stats.avg_wait_time_ms, 2),
            "max_size": self.max_size,
            "utilization_percent": round(
                (self._stats.current_depth / self.max_size) * 100, 1
            ) if self.max_size > 0 else 0.0,
        }

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return len(self._queue) >= self.max_size


class QueueDispatcher:
    """
    Background task that dispatches queued jobs when nodes are available.

    Continuously polls the queue and attempts to dispatch jobs.
    On failure, jobs are re-queued with retry logic.

    Args:
        queue: JobQueue instance
        scheduler: SimpleScheduler for node selection and dispatch
        check_interval: Seconds between queue checks when empty
    """

    def __init__(
        self,
        queue: JobQueue,
        scheduler: "SimpleScheduler",
        check_interval: float = 5.0,
    ):
        self.queue = queue
        self.scheduler = scheduler
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._dispatched_count = 0
        self._failed_count = 0

    async def start(self) -> None:
        """Start background dispatcher."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """Stop background dispatcher gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def is_running(self) -> bool:
        """Check if dispatcher is running."""
        return self._running

    async def _dispatch_loop(self) -> None:
        """Main dispatch loop - runs continuously in background."""
        while self._running:
            try:
                job = await self.queue.dequeue()
                if job:
                    await self._dispatch_job(job)
                else:
                    # Queue empty, wait before checking again
                    await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                raise  # Propagate cancellation
            except Exception:
                # Log but don't crash - keep dispatcher running
                await asyncio.sleep(self.check_interval)

    async def _dispatch_job(self, job: QueuedJob) -> None:
        """
        Attempt to dispatch a single job.

        Args:
            job: Job to dispatch
        """
        try:
            # Pick a node (respecting job requirements)
            node = await self.scheduler.pick_node(job.job_requirements)

            if not node:
                # No nodes available - re-queue
                await self.queue.mark_failed(job.job_id, "No nodes available")
                self._failed_count += 1
                return

            # Dispatch the job
            result = await self.scheduler.dispatch_job_async(
                node=node,
                module_id=job.module_id,
                input_json=job.input_json,
                job_id=job.job_id,
                timeout_seconds=job.timeout_seconds,
            )

            if result.get("status") == "failed":
                await self.queue.mark_failed(
                    job.job_id,
                    result.get("error", "Unknown dispatch error"),
                )
                self._failed_count += 1
            else:
                await self.queue.mark_dispatched(job.job_id)
                self._dispatched_count += 1

        except Exception as e:
            await self.queue.mark_failed(job.job_id, str(e))
            self._failed_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dispatcher statistics.

        Returns:
            Dict with dispatcher stats
        """
        return {
            "running": self._running,
            "dispatched_count": self._dispatched_count,
            "failed_count": self._failed_count,
            "check_interval_seconds": self.check_interval,
        }


async def create_job_queue_system(
    scheduler: "SimpleScheduler",
    max_queue_size: int = 10000,
    check_interval: float = 5.0,
    auto_start: bool = True,
) -> tuple[JobQueue, QueueDispatcher]:
    """
    Factory function to create a complete job queue system.

    Args:
        scheduler: SimpleScheduler for node selection
        max_queue_size: Maximum queue capacity
        check_interval: Seconds between queue checks when empty
        auto_start: Start dispatcher immediately

    Returns:
        Tuple of (JobQueue, QueueDispatcher)
    """
    queue = JobQueue(max_size=max_queue_size)
    dispatcher = QueueDispatcher(
        queue=queue,
        scheduler=scheduler,
        check_interval=check_interval,
    )

    if auto_start:
        await dispatcher.start()

    return queue, dispatcher
