"""
HoloLoom Celery Worker

Distributed task processing using Celery with RabbitMQ broker and Redis backend.
Supports async execution of HoloLoom queries, batch processing, and background tasks.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from celery import Celery, Task
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun
from prometheus_client import Counter, Histogram, Gauge
import time

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
BROKER_URL = os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@localhost:5672//')
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
TASK_TRACK_STARTED = os.getenv('CELERY_TASK_TRACK_STARTED', 'true').lower() == 'true'
TASK_TIME_LIMIT = int(os.getenv('CELERY_TASK_TIME_LIMIT', '3600'))
TASK_SOFT_TIME_LIMIT = int(os.getenv('CELERY_TASK_SOFT_TIME_LIMIT', '3300'))

# Prometheus metrics
TASK_COUNTER = Counter(
    'hololoom_worker_tasks_total',
    'Total number of tasks processed',
    ['task_name', 'status']
)
TASK_DURATION = Histogram(
    'hololoom_worker_task_duration_seconds',
    'Task execution duration',
    ['task_name']
)
ACTIVE_TASKS = Gauge(
    'hololoom_worker_active_tasks',
    'Number of currently active tasks'
)
QUEUE_DEPTH = Gauge(
    'hololoom_worker_queue_depth',
    'Depth of task queue',
    ['queue_name']
)

# Create Celery app
app = Celery(
    'hololoom',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=TASK_TRACK_STARTED,
    task_time_limit=TASK_TIME_LIMIT,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    task_acks_late=True,
    worker_prefetch_multiplier=int(os.getenv('WORKER_PREFETCH_MULTIPLIER', '4')),
    worker_max_tasks_per_child=int(os.getenv('WORKER_MAX_TASKS_PER_CHILD', '1000')),
    worker_max_memory_per_child=int(os.getenv('WORKER_MAX_MEMORY_PER_CHILD', '200000')),
    result_expires=3600,
    task_default_queue='hololoom.tasks.default',
    task_default_exchange='hololoom',
    task_default_routing_key='hololoom.default',
    task_routes={
        'hololoom.distributed.worker.process_query': {
            'queue': 'hololoom.tasks.default',
            'routing_key': 'hololoom.query',
        },
        'hololoom.distributed.worker.process_batch': {
            'queue': 'hololoom.tasks.default',
            'routing_key': 'hololoom.batch',
        },
        'hololoom.distributed.worker.priority_task': {
            'queue': 'hololoom.tasks.priority',
            'routing_key': 'hololoom.priority',
        },
    },
)


class InstrumentedTask(Task):
    """Base task class with Prometheus instrumentation"""

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        ACTIVE_TASKS.inc()

        try:
            result = super().__call__(*args, **kwargs)
            TASK_COUNTER.labels(task_name=self.name, status='success').inc()
            return result
        except Exception as exc:
            TASK_COUNTER.labels(task_name=self.name, status='failure').inc()
            raise
        finally:
            duration = time.time() - start_time
            TASK_DURATION.labels(task_name=self.name).observe(duration)
            ACTIVE_TASKS.dec()


# Set base task class
app.Task = InstrumentedTask


@app.task(name='hololoom.distributed.worker.process_query', bind=True)
def process_query(self, query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a single HoloLoom query asynchronously.

    Args:
        query: The query text to process
        config: Optional configuration overrides

    Returns:
        Dict containing the query result and metadata
    """
    logger.info(f"Processing query: {query[:50]}...")

    try:
        # Import here to avoid circular dependencies
        from holoLoom.orchestrator import HoloLoomOrchestrator
        from holoLoom.config import Config

        # Get configuration
        if config:
            cfg = Config(**config)
        else:
            cfg = Config.fused()

        # Create orchestrator instance
        orchestrator = HoloLoomOrchestrator(config=cfg)

        # Process query
        # Note: Since orchestrator.process is async, we need to handle it properly
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(orchestrator.process(query))
        finally:
            loop.close()

        logger.info(f"Query processed successfully: {query[:50]}")

        return {
            'status': 'success',
            'query': query,
            'result': result,
            'task_id': self.request.id,
        }

    except Exception as exc:
        logger.error(f"Error processing query: {exc}")
        self.retry(exc=exc, countdown=60, max_retries=3)


@app.task(name='hololoom.distributed.worker.process_batch', bind=True)
def process_batch(self, queries: List[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a batch of queries in parallel.

    Args:
        queries: List of query strings
        config: Optional configuration overrides

    Returns:
        Dict containing batch results and metadata
    """
    logger.info(f"Processing batch of {len(queries)} queries")

    try:
        # Import here to avoid circular dependencies
        from holoLoom.orchestrator import HoloLoomOrchestrator
        from holoLoom.config import Config

        # Get configuration
        if config:
            cfg = Config(**config)
        else:
            cfg = Config.fused()

        # Create orchestrator instance
        orchestrator = HoloLoomOrchestrator(config=cfg)

        # Process queries
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Process all queries concurrently
            async def process_all():
                tasks = [orchestrator.process(q) for q in queries]
                return await asyncio.gather(*tasks, return_exceptions=True)

            results = loop.run_until_complete(process_all())
        finally:
            loop.close()

        # Separate successes and failures
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        logger.info(f"Batch processed: {len(successes)} succeeded, {len(failures)} failed")

        return {
            'status': 'success',
            'total': len(queries),
            'succeeded': len(successes),
            'failed': len(failures),
            'results': successes,
            'errors': [str(f) for f in failures],
            'task_id': self.request.id,
        }

    except Exception as exc:
        logger.error(f"Error processing batch: {exc}")
        self.retry(exc=exc, countdown=60, max_retries=3)


@app.task(name='hololoom.distributed.worker.priority_task', bind=True, priority=9)
def priority_task(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-priority task for urgent processing.

    Args:
        task_type: Type of priority task
        payload: Task payload

    Returns:
        Dict containing task result
    """
    logger.info(f"Processing priority task: {task_type}")

    try:
        if task_type == 'query':
            return process_query(payload.get('query'), payload.get('config'))
        elif task_type == 'batch':
            return process_batch(payload.get('queries'), payload.get('config'))
        else:
            raise ValueError(f"Unknown priority task type: {task_type}")

    except Exception as exc:
        logger.error(f"Error processing priority task: {exc}")
        raise


@app.task(name='hololoom.distributed.worker.health_check')
def health_check() -> Dict[str, Any]:
    """
    Health check task for monitoring.

    Returns:
        Dict with health status
    """
    return {
        'status': 'healthy',
        'worker': True,
        'timestamp': time.time(),
    }


# Signal handlers
@worker_ready.connect
def worker_ready_handler(sender, **kwargs):
    """Handle worker ready event"""
    logger.info("Worker is ready and accepting tasks")


@worker_shutdown.connect
def worker_shutdown_handler(sender, **kwargs):
    """Handle worker shutdown event"""
    logger.info("Worker is shutting down")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Handle task pre-run event"""
    logger.debug(f"Starting task: {task.name} [{task_id}]")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None,
                        retval=None, state=None, **extra):
    """Handle task post-run event"""
    logger.debug(f"Completed task: {task.name} [{task_id}] - State: {state}")


# Periodic tasks
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks"""
    # Health check every 60 seconds
    sender.add_periodic_task(60.0, health_check.s(), name='health-check')

    # Update queue depth metrics every 30 seconds
    sender.add_periodic_task(30.0, update_queue_metrics.s(), name='queue-metrics')


@app.task(name='hololoom.distributed.worker.update_queue_metrics')
def update_queue_metrics():
    """Update queue depth metrics for Prometheus"""
    try:
        # Get queue stats from RabbitMQ
        inspect = app.control.inspect()
        stats = inspect.stats()

        if stats:
            for worker, worker_stats in stats.items():
                # Update metrics based on worker stats
                logger.debug(f"Worker {worker} stats: {worker_stats}")

    except Exception as exc:
        logger.error(f"Error updating queue metrics: {exc}")


if __name__ == '__main__':
    # Start worker
    app.start()
