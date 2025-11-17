"""
HoloLoom Task Coordinator

Coordinates distributed task execution across workers with:
- Load balancing
- Task prioritization
- Result aggregation
- Failure handling and retries
"""

import os
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .queue import QueueManager, QueuePriority
from .cache import CacheManager

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """Task coordination strategies"""
    ROUND_ROBIN = 'round_robin'
    LEAST_LOADED = 'least_loaded'
    PRIORITY_BASED = 'priority_based'
    RANDOM = 'random'


@dataclass
class TaskResult:
    """Task result wrapper"""
    task_id: str
    status: str  # 'pending', 'running', 'success', 'failure'
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskCoordinator:
    """
    Coordinates distributed task execution across workers.

    Features:
    - Parallel task execution
    - Load balancing strategies
    - Result aggregation
    - Automatic retry on failure
    - Progress tracking
    """

    def __init__(
        self,
        queue_manager: Optional[QueueManager] = None,
        cache_manager: Optional[CacheManager] = None,
        strategy: CoordinationStrategy = CoordinationStrategy.ROUND_ROBIN,
        max_retries: int = 3,
        retry_delay: int = 60,
    ):
        """
        Initialize task coordinator.

        Args:
            queue_manager: Queue manager instance
            cache_manager: Cache manager instance
            strategy: Coordination strategy
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
        """
        self.queue = queue_manager or QueueManager()
        self.cache = cache_manager or CacheManager()
        self.strategy = strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Task tracking
        self.active_tasks: Dict[str, TaskResult] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}

        logger.info(f"Task coordinator initialized with strategy: {strategy.value}")

    def submit_query(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> str:
        """
        Submit a single query for processing.

        Args:
            query: Query text
            config: Optional configuration
            priority: Task priority

        Returns:
            Task ID
        """
        # Check cache first
        cache_key = f"query:{hash(query)}"
        cached_result = self.cache.get(cache_key)

        if cached_result is not None:
            logger.info(f"Query result found in cache: {query[:50]}")
            # Create synthetic task result
            task_id = f"cached-{hash(query)}"
            result = TaskResult(
                task_id=task_id,
                status='success',
                result=cached_result,
                completed_at=time.time(),
            )
            self.completed_tasks[task_id] = result
            return task_id

        # Enqueue task
        task_id = self.queue.enqueue_query(query, config, priority)

        # Track task
        self.active_tasks[task_id] = TaskResult(
            task_id=task_id,
            status='pending',
            started_at=time.time(),
        )

        return task_id

    def submit_batch(
        self,
        queries: List[str],
        config: Optional[Dict[str, Any]] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        parallel: bool = True,
    ) -> List[str]:
        """
        Submit a batch of queries for processing.

        Args:
            queries: List of query strings
            config: Optional configuration
            priority: Task priority
            parallel: Process queries in parallel

        Returns:
            List of task IDs
        """
        if parallel:
            # Submit each query as separate task
            task_ids = []
            for query in queries:
                task_id = self.submit_query(query, config, priority)
                task_ids.append(task_id)

            logger.info(f"Submitted {len(task_ids)} parallel tasks")
            return task_ids
        else:
            # Submit as single batch task
            task_id = self.queue.enqueue_batch(queries, config, priority)

            self.active_tasks[task_id] = TaskResult(
                task_id=task_id,
                status='pending',
                started_at=time.time(),
                metadata={'batch_size': len(queries)},
            )

            logger.info(f"Submitted batch task with {len(queries)} queries")
            return [task_id]

    def get_result(
        self,
        task_id: str,
        timeout: Optional[float] = None,
        cache_result: bool = True,
    ) -> Optional[TaskResult]:
        """
        Get task result (blocking).

        Args:
            task_id: Task ID
            timeout: Maximum wait time
            cache_result: Cache the result for future queries

        Returns:
            TaskResult or None
        """
        # Check if already completed
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        # Check if cached
        if task_id.startswith('cached-'):
            return self.completed_tasks.get(task_id)

        # Wait for result
        result_data = self.queue.get_result(task_id, timeout)

        if result_data is None:
            # Not ready yet
            return None

        # Create task result
        task_result = self.active_tasks.get(task_id)
        if task_result is None:
            task_result = TaskResult(task_id=task_id, status='unknown')

        task_result.completed_at = time.time()
        if task_result.started_at:
            task_result.duration = task_result.completed_at - task_result.started_at

        if result_data.get('status') == 'success':
            task_result.status = 'success'
            task_result.result = result_data.get('result')

            # Cache result if requested
            if cache_result and 'query' in result_data:
                cache_key = f"query:{hash(result_data['query'])}"
                self.cache.set(cache_key, task_result.result, ttl=3600)

        else:
            task_result.status = 'failure'
            task_result.error = str(result_data.get('error'))

        # Move to completed
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        self.completed_tasks[task_id] = task_result

        return task_result

    async def get_result_async(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> Optional[TaskResult]:
        """
        Get task result asynchronously.

        Args:
            task_id: Task ID
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time

        Returns:
            TaskResult or None
        """
        start_time = time.time()

        while True:
            result = self.get_result(task_id, timeout=0)

            if result is not None:
                return result

            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                logger.warning(f"Timeout waiting for task {task_id}")
                return None

            # Wait before polling again
            await asyncio.sleep(poll_interval)

    def wait_all(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
    ) -> List[TaskResult]:
        """
        Wait for all tasks to complete.

        Args:
            task_ids: List of task IDs
            timeout: Maximum wait time

        Returns:
            List of TaskResults
        """
        start_time = time.time()
        results = []

        for task_id in task_ids:
            # Calculate remaining timeout
            remaining_timeout = None
            if timeout:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)

            result = self.get_result(task_id, timeout=remaining_timeout)
            results.append(result)

            # Check if we've exceeded timeout
            if timeout and (time.time() - start_time) >= timeout:
                logger.warning(f"Timeout waiting for all tasks")
                break

        return results

    async def wait_all_async(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
    ) -> List[TaskResult]:
        """
        Wait for all tasks to complete asynchronously.

        Args:
            task_ids: List of task IDs
            timeout: Maximum wait time

        Returns:
            List of TaskResults
        """
        tasks = [
            self.get_result_async(task_id, timeout=timeout)
            for task_id in task_ids
        ]

        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    def get_progress(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get progress of tasks.

        Args:
            task_ids: Optional list of task IDs to check (None = all)

        Returns:
            Dict with progress information
        """
        if task_ids is None:
            # Check all tracked tasks
            task_ids = list(self.active_tasks.keys())

        pending = 0
        running = 0
        completed = 0
        failed = 0

        for task_id in task_ids:
            status = self.queue.get_status(task_id)

            if status['state'] == 'PENDING':
                pending += 1
            elif status['state'] in ('STARTED', 'RETRY'):
                running += 1
            elif status['state'] == 'SUCCESS':
                completed += 1
            elif status['state'] in ('FAILURE', 'REVOKED'):
                failed += 1

        total = len(task_ids)
        progress = (completed + failed) / total if total > 0 else 0.0

        return {
            'total': total,
            'pending': pending,
            'running': running,
            'completed': completed,
            'failed': failed,
            'progress': progress,
        }

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled
        """
        success = self.queue.cancel_task(task_id)

        if success and task_id in self.active_tasks:
            task_result = self.active_tasks[task_id]
            task_result.status = 'cancelled'
            task_result.completed_at = time.time()

            del self.active_tasks[task_id]
            self.completed_tasks[task_id] = task_result

        return success

    def cancel_all(self, task_ids: List[str]) -> int:
        """
        Cancel multiple tasks.

        Args:
            task_ids: List of task IDs

        Returns:
            Number of tasks cancelled
        """
        count = 0
        for task_id in task_ids:
            if self.cancel_task(task_id):
                count += 1

        logger.info(f"Cancelled {count}/{len(task_ids)} tasks")
        return count

    def retry_failed(self) -> List[str]:
        """
        Retry all failed tasks.

        Returns:
            List of new task IDs
        """
        new_task_ids = []

        for task_id, result in list(self.completed_tasks.items()):
            if result.status == 'failure':
                # Extract original query from metadata
                query = result.metadata.get('query')
                config = result.metadata.get('config')

                if query:
                    new_task_id = self.submit_query(query, config)
                    new_task_ids.append(new_task_id)

        logger.info(f"Retrying {len(new_task_ids)} failed tasks")
        return new_task_ids

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get coordination statistics.

        Returns:
            Dict with statistics
        """
        # Queue stats
        queue_stats = self.queue.get_queue_stats()

        # Cache stats
        cache_stats = self.cache.get_stats()

        # Task stats
        total_tasks = len(self.active_tasks) + len(self.completed_tasks)
        successful = sum(1 for r in self.completed_tasks.values() if r.status == 'success')
        failed = sum(1 for r in self.completed_tasks.values() if r.status == 'failure')

        avg_duration = 0.0
        if successful > 0:
            durations = [r.duration for r in self.completed_tasks.values()
                        if r.status == 'success' and r.duration]
            if durations:
                avg_duration = sum(durations) / len(durations)

        return {
            'queue': queue_stats,
            'cache': cache_stats,
            'tasks': {
                'total': total_tasks,
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total_tasks if total_tasks > 0 else 0.0,
                'avg_duration': avg_duration,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of coordinator and dependencies.

        Returns:
            Dict with health status
        """
        queue_health = self.queue.health_check()
        cache_health = self.cache.health_check()

        overall_status = 'healthy'
        if queue_health['status'] != 'healthy' or cache_health['status'] != 'healthy':
            overall_status = 'degraded'

        return {
            'status': overall_status,
            'queue': queue_health,
            'cache': cache_health,
            'active_tasks': len(self.active_tasks),
        }
