"""
HoloLoom Queue Manager

Abstraction layer for message queue operations with RabbitMQ/Redis backend.
Provides enqueue, dequeue, priority queues, and monitoring capabilities.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
from celery import Celery
from celery.result import AsyncResult

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Queue priority levels"""
    LOW = 0
    NORMAL = 5
    HIGH = 9


@dataclass
class QueueMessage:
    """Queue message wrapper"""
    id: str
    payload: Dict[str, Any]
    priority: QueuePriority
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class QueueManager:
    """
    Message queue manager for distributed task processing.

    Provides high-level interface for:
    - Enqueueing tasks with priority
    - Monitoring queue depth
    - Task result retrieval
    - Queue statistics
    """

    def __init__(
        self,
        broker_url: Optional[str] = None,
        backend_url: Optional[str] = None,
    ):
        """
        Initialize queue manager.

        Args:
            broker_url: Message broker URL (RabbitMQ)
            backend_url: Result backend URL (Redis)
        """
        self.broker_url = broker_url or os.getenv(
            'CELERY_BROKER_URL',
            'amqp://guest:guest@localhost:5672//'
        )
        self.backend_url = backend_url or os.getenv(
            'CELERY_RESULT_BACKEND',
            'redis://localhost:6379/1'
        )

        # Import worker app
        from .worker import app as celery_app
        self.celery_app = celery_app

        logger.info(f"Queue manager initialized with broker: {self.broker_url}")

    def enqueue_query(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        countdown: Optional[int] = None,
    ) -> str:
        """
        Enqueue a single query for processing.

        Args:
            query: Query text
            config: Optional configuration overrides
            priority: Task priority
            countdown: Delay before execution (seconds)

        Returns:
            Task ID
        """
        from .worker import process_query, priority_task

        if priority == QueuePriority.HIGH:
            # Use priority queue
            task = priority_task.apply_async(
                args=[
                    'query',
                    {'query': query, 'config': config}
                ],
                countdown=countdown,
            )
        else:
            # Use normal queue
            task = process_query.apply_async(
                args=[query, config],
                countdown=countdown,
                priority=priority.value,
            )

        logger.info(f"Enqueued query task: {task.id}")
        return task.id

    def enqueue_batch(
        self,
        queries: List[str],
        config: Optional[Dict[str, Any]] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> str:
        """
        Enqueue a batch of queries for processing.

        Args:
            queries: List of query strings
            config: Optional configuration overrides
            priority: Task priority

        Returns:
            Task ID
        """
        from .worker import process_batch, priority_task

        if priority == QueuePriority.HIGH:
            task = priority_task.apply_async(
                args=[
                    'batch',
                    {'queries': queries, 'config': config}
                ],
            )
        else:
            task = process_batch.apply_async(
                args=[queries, config],
                priority=priority.value,
            )

        logger.info(f"Enqueued batch task: {task.id} ({len(queries)} queries)")
        return task.id

    def get_result(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get task result (blocking).

        Args:
            task_id: Task ID
            timeout: Maximum wait time in seconds

        Returns:
            Task result or None if not ready
        """
        result = AsyncResult(task_id, app=self.celery_app)

        try:
            if timeout:
                return result.get(timeout=timeout)
            else:
                if result.ready():
                    return result.get()
                return None
        except Exception as exc:
            logger.error(f"Error getting result for task {task_id}: {exc}")
            return None

    def get_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status.

        Args:
            task_id: Task ID

        Returns:
            Dict with task status information
        """
        result = AsyncResult(task_id, app=self.celery_app)

        return {
            'task_id': task_id,
            'state': result.state,
            'ready': result.ready(),
            'successful': result.successful() if result.ready() else None,
            'failed': result.failed() if result.ready() else None,
            'info': result.info,
        }

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled successfully
        """
        result = AsyncResult(task_id, app=self.celery_app)

        try:
            result.revoke(terminate=True)
            logger.info(f"Cancelled task: {task_id}")
            return True
        except Exception as exc:
            logger.error(f"Error cancelling task {task_id}: {exc}")
            return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dict with queue statistics
        """
        inspect = self.celery_app.control.inspect()

        try:
            # Get active tasks
            active = inspect.active()
            active_count = sum(len(tasks) for tasks in (active or {}).values())

            # Get scheduled tasks
            scheduled = inspect.scheduled()
            scheduled_count = sum(len(tasks) for tasks in (scheduled or {}).values())

            # Get reserved tasks
            reserved = inspect.reserved()
            reserved_count = sum(len(tasks) for tasks in (reserved or {}).values())

            # Get worker stats
            stats = inspect.stats()
            worker_count = len(stats or {})

            return {
                'active_tasks': active_count,
                'scheduled_tasks': scheduled_count,
                'reserved_tasks': reserved_count,
                'total_pending': active_count + scheduled_count + reserved_count,
                'workers': worker_count,
                'stats': stats,
            }
        except Exception as exc:
            logger.error(f"Error getting queue stats: {exc}")
            return {
                'active_tasks': 0,
                'scheduled_tasks': 0,
                'reserved_tasks': 0,
                'total_pending': 0,
                'workers': 0,
                'error': str(exc),
            }

    def purge_queue(self, queue_name: str = 'hololoom.tasks.default') -> int:
        """
        Purge all messages from a queue.

        Args:
            queue_name: Queue to purge

        Returns:
            Number of messages purged
        """
        try:
            count = self.celery_app.control.purge()
            logger.warning(f"Purged {count} messages from queue: {queue_name}")
            return count
        except Exception as exc:
            logger.error(f"Error purging queue {queue_name}: {exc}")
            return 0

    def health_check(self) -> Dict[str, Any]:
        """
        Check queue health.

        Returns:
            Dict with health status
        """
        try:
            # Ping workers
            inspect = self.celery_app.control.inspect()
            ping_result = inspect.ping()

            workers_alive = len(ping_result or {})

            return {
                'status': 'healthy' if workers_alive > 0 else 'unhealthy',
                'workers_alive': workers_alive,
                'workers': list((ping_result or {}).keys()),
            }
        except Exception as exc:
            logger.error(f"Queue health check failed: {exc}")
            return {
                'status': 'unhealthy',
                'workers_alive': 0,
                'error': str(exc),
            }

    def broadcast_task(
        self,
        task_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Broadcast task to all workers.

        Args:
            task_name: Task name
            args: Task arguments
            kwargs: Task keyword arguments

        Returns:
            List of task IDs
        """
        from celery import group

        task = self.celery_app.tasks[task_name]
        job = group(task.s(*(args or []), **(kwargs or {})))

        result = job.apply_async()
        task_ids = [r.id for r in result.results]

        logger.info(f"Broadcast task {task_name} to {len(task_ids)} workers")
        return task_ids

    def register_callback(
        self,
        task_id: str,
        callback: Callable[[Dict[str, Any]], None],
    ):
        """
        Register callback for task completion.

        Args:
            task_id: Task ID
            callback: Callback function to execute on completion
        """
        result = AsyncResult(task_id, app=self.celery_app)

        def on_success(retval):
            callback({'status': 'success', 'result': retval})

        def on_failure(exc):
            callback({'status': 'failure', 'error': str(exc)})

        result.then(on_success, on_error=on_failure)
        logger.info(f"Registered callback for task: {task_id}")

    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """
        Get detailed task information.

        Args:
            task_id: Task ID

        Returns:
            Dict with detailed task information
        """
        result = AsyncResult(task_id, app=self.celery_app)

        info = {
            'task_id': task_id,
            'state': result.state,
            'ready': result.ready(),
            'successful': result.successful() if result.ready() else None,
            'failed': result.failed() if result.ready() else None,
        }

        if result.ready():
            if result.successful():
                info['result'] = result.result
            elif result.failed():
                info['error'] = str(result.info)
                info['traceback'] = result.traceback

        return info
