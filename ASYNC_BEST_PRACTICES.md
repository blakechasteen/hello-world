# HoloLoom Async/Await Best Practices Guide

**Date**: 2025-11-16
**Purpose**: Guidelines for writing safe, maintainable async code in HoloLoom
**Audience**: All contributors to the HoloLoom codebase

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Async Context Managers](#async-context-managers)
3. [Background Task Management](#background-task-management)
4. [Resource Cleanup](#resource-cleanup)
5. [Error Handling](#error-handling)
6. [Common Patterns](#common-patterns)
7. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
8. [Testing Async Code](#testing-async-code)

---

## Core Principles

### 1. **Always Await Async Calls**

The most fundamental rule: if a function is `async def`, you MUST `await` it.

```python
# ❌ WRONG - Returns coroutine, doesn't execute
result = some_async_function()

# ✅ CORRECT - Awaits execution
result = await some_async_function()
```

### 2. **Use Async Context Managers for Resource Management**

Always implement `__aenter__` and `__aexit__` for classes that manage resources.

```python
# ✅ CORRECT
class MyService:
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False  # Don't suppress exceptions

# Usage
async with MyService() as service:
    await service.do_work()
    # Cleanup happens automatically
```

### 3. **Track All Background Tasks**

Every `asyncio.create_task()` call must be tracked for proper cleanup.

```python
# ✅ CORRECT
class MyService:
    def __init__(self):
        self._background_tasks: List[asyncio.Task] = []

    def spawn_task(self, coro):
        """Spawn and track background task."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)

        # Auto-cleanup on completion
        def cleanup(t):
            try:
                self._background_tasks.remove(t)
            except ValueError:
                pass
        task.add_done_callback(cleanup)
        return task

    async def close(self):
        """Cancel all background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.wait(
                self._background_tasks,
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED
            )
        self._background_tasks.clear()
```

### 4. **Make Cleanup Idempotent**

Cleanup methods should be safe to call multiple times.

```python
# ✅ CORRECT
class MyService:
    def __init__(self):
        self._closed = False

    async def close(self):
        """Idempotent cleanup."""
        if self._closed:
            return  # Already closed, safe to return

        # Perform cleanup
        await self.cleanup_resources()
        self._closed = True
```

---

## Async Context Managers

### Standard Pattern

Every class that manages resources should implement async context manager:

```python
class DatabaseConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
        self._closed = False

    async def __aenter__(self):
        """Initialize resources."""
        self.connection = await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        await self.close()
        return False  # Don't suppress exceptions

    async def connect(self):
        """Establish connection."""
        # ... connection logic ...
        return connection

    async def close(self):
        """Close connection (idempotent)."""
        if self._closed:
            return

        if self.connection:
            await self.connection.close()
            self.connection = None

        self._closed = True

# Usage
async with DatabaseConnection("postgresql://...") as db:
    result = await db.query("SELECT * FROM users")
    # Connection automatically closed on exit
```

### Nested Cleanup

When composing multiple services, cleanup in reverse order:

```python
class CompositeService:
    async def __aenter__(self):
        # Initialize in order
        self.database = DatabaseConnection("...")
        await self.database.__aenter__()

        self.cache = CacheService()
        await self.cache.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup in reverse order
        if self.cache:
            await self.cache.__aexit__(exc_type, exc_val, exc_tb)

        if self.database:
            await self.database.__aexit__(exc_type, exc_val, exc_tb)

        return False
```

---

## Background Task Management

### The Task Tracker Pattern

Standard pattern for managing background tasks:

```python
class ServiceWithBackgroundTasks:
    def __init__(self):
        self._background_tasks: List[asyncio.Task] = []
        self._running = False

    async def start(self):
        """Start background services."""
        self._running = True
        task = self.spawn_task(self._background_worker())
        logger.info("Background worker started")

    def spawn_task(self, coro) -> asyncio.Task:
        """
        Spawn and track background task.

        Args:
            coro: Coroutine to run in background

        Returns:
            asyncio.Task object
        """
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)

        # Auto-cleanup completed tasks
        def cleanup(t):
            try:
                self._background_tasks.remove(t)
            except ValueError:
                pass
        task.add_done_callback(cleanup)

        return task

    async def _background_worker(self):
        """Background worker loop."""
        while self._running:
            try:
                await asyncio.sleep(60)
                await self._do_work()
            except asyncio.CancelledError:
                logger.info("Background worker cancelled")
                break
            except Exception as e:
                logger.error(f"Background worker error: {e}", exc_info=True)
                # Continue running despite errors

    async def stop(self):
        """Stop all background tasks."""
        self._running = False

        # Cancel all tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation with timeout
        if self._background_tasks:
            try:
                await asyncio.wait(
                    self._background_tasks,
                    timeout=5.0,
                    return_when=asyncio.ALL_COMPLETED
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within timeout")

        self._background_tasks.clear()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False
```

### Long-Running Background Tasks

For tasks that should run for the lifetime of the service:

```python
class MonitoringService:
    async def __aenter__(self):
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        return self

    async def _monitor_loop(self):
        """Long-running monitoring loop."""
        while True:
            try:
                await asyncio.sleep(10)
                await self._collect_metrics()
            except asyncio.CancelledError:
                logger.info("Monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)
                # Continue despite errors

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        return False
```

---

## Resource Cleanup

### Database Connections

```python
class DatabaseBackend:
    def __init__(self, connection_string: str):
        self.engine = None
        self._pool = None

    async def __aenter__(self):
        self.engine = create_async_engine(self.connection_string)
        self._pool = await self.engine.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._pool:
            await self._pool.close()

        if self.engine:
            await self.engine.dispose()  # Close all connections in pool

        return False
```

### File Handles

```python
class AsyncFileProcessor:
    async def __aenter__(self):
        self.file = await aiofiles.open('data.txt', 'r')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            await self.file.close()
        return False
```

### Network Connections

```python
class WebSocketClient:
    async def __aenter__(self):
        self.ws = await websockets.connect('ws://localhost:8000')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.ws:
            await self.ws.close()
        return False
```

---

## Error Handling

### Exception Handling in Async Code

```python
# ✅ CORRECT - Comprehensive error handling
async def process_query(query: str):
    try:
        result = await dangerous_operation(query)
        return result
    except asyncio.TimeoutError:
        logger.error(f"Query timed out: {query}")
        raise
    except ValueError as e:
        logger.error(f"Invalid query: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
```

### Cleanup on Exception

Always use try/finally for cleanup:

```python
# ✅ CORRECT
async def process_with_cleanup():
    resource = await acquire_resource()
    try:
        result = await process(resource)
        return result
    finally:
        await release_resource(resource)
```

### Error Handling in Background Tasks

```python
async def background_worker():
    while running:
        try:
            await do_work()
        except asyncio.CancelledError:
            logger.info("Worker cancelled")
            break  # Exit cleanly on cancellation
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            # Continue running despite errors
            await asyncio.sleep(5)  # Backoff on error
```

---

## Common Patterns

### 1. Parallel Execution with asyncio.gather()

```python
# ✅ CORRECT - Parallel execution with error handling
async def fetch_multiple_sources(queries: List[str]):
    tasks = [fetch_source(q) for q in queries]

    try:
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True  # Don't fail entire batch on single error
        )

        # Process results, checking for exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Query {queries[i]} failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    except Exception as e:
        # Cancel pending tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
```

### 2. Timeout on Operations

```python
# ✅ CORRECT - Add timeout to prevent hanging
async def query_with_timeout(query: str, timeout: float = 30.0):
    try:
        result = await asyncio.wait_for(
            slow_operation(query),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout}s")
        raise
```

### 3. Retry with Exponential Backoff

```python
# ✅ CORRECT - Retry pattern
async def retry_with_backoff(
    coro,
    max_retries: int = 3,
    initial_delay: float = 1.0
):
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return await coro
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Last attempt, give up

            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay}s..."
            )
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
```

### 4. Semaphore for Concurrency Control

```python
# ✅ CORRECT - Limit concurrent operations
class RateLimitedService:
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def process(self, item):
        async with self._semaphore:
            # Only max_concurrent operations run at once
            return await self._do_process(item)
```

---

## Anti-Patterns to Avoid

### ❌ Fire-and-Forget Tasks

```python
# ❌ WRONG - Task not tracked
asyncio.create_task(background_work())

# ✅ CORRECT - Track task for cleanup
task = self.spawn_task(background_work())
```

### ❌ Swallowing Exceptions

```python
# ❌ WRONG - Exception lost
try:
    await operation()
except Exception:
    pass  # Silent failure

# ✅ CORRECT - Log exception
try:
    await operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise  # Re-raise or handle appropriately
```

### ❌ Missing await on Async Methods

```python
# ❌ WRONG - Returns coroutine, doesn't execute
result = async_function()

# ✅ CORRECT - Await execution
result = await async_function()
```

### ❌ Blocking Operations in Async Code

```python
# ❌ WRONG - Blocks event loop
import time
async def slow_function():
    time.sleep(10)  # Blocks entire event loop!

# ✅ CORRECT - Use async sleep
async def slow_function():
    await asyncio.sleep(10)  # Yields to event loop

# ✅ CORRECT - Run blocking code in executor
async def slow_function():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, blocking_function)
```

### ❌ Not Closing Resources

```python
# ❌ WRONG - Resource leak
async def process():
    conn = await connect_db()
    result = await conn.query("SELECT * FROM users")
    return result  # Connection never closed!

# ✅ CORRECT - Use context manager
async def process():
    async with connect_db() as conn:
        result = await conn.query("SELECT * FROM users")
        return result
```

### ❌ Race Conditions in Cleanup

```python
# ❌ WRONG - Race condition
def cleanup_callback(task):
    self._background_tasks.remove(task)  # Can race with close()

# ✅ CORRECT - Safe cleanup
def cleanup_callback(task):
    try:
        self._background_tasks.remove(task)
    except ValueError:
        pass  # Already removed, ignore
```

---

## Testing Async Code

### Basic Async Test

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result == expected_value
```

### Testing Context Managers

```python
@pytest.mark.asyncio
async def test_context_manager_cleanup():
    obj = MyService()

    async with obj:
        # Verify initialization
        assert obj._initialized is True

    # Verify cleanup after exit
    assert obj._closed is True
```

### Testing Exception Handling

```python
@pytest.mark.asyncio
async def test_cleanup_on_exception():
    obj = MyService()

    try:
        async with obj:
            raise ValueError("Test error")
    except ValueError:
        pass

    # Verify cleanup happened despite exception
    assert obj._closed is True
```

### Testing Background Tasks

```python
@pytest.mark.asyncio
async def test_background_task_cleanup():
    async with ServiceWithBackgroundTasks() as service:
        task = service.spawn_task(some_coro())

        # Verify task is tracked
        assert task in service._background_tasks

    # After exit, task should be cancelled
    assert task.cancelled()
```

### Testing Timeouts

```python
@pytest.mark.asyncio
async def test_operation_timeout():
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            slow_operation(),
            timeout=0.1
        )
```

---

## Checklist for Code Review

When reviewing async code, check:

- [ ] All async functions are awaited
- [ ] All resources are cleaned up in `__aexit__`
- [ ] All background tasks are tracked
- [ ] Background tasks are cancelled on shutdown
- [ ] Exceptions are logged with `exc_info=True`
- [ ] Cleanup is idempotent (safe to call multiple times)
- [ ] No blocking operations (time.sleep, requests.get, etc.)
- [ ] Timeouts are set for external I/O
- [ ] Race conditions are avoided
- [ ] Tests verify cleanup behavior

---

## Quick Reference

### Creating Async Context Manager
```python
async def __aenter__(self):
    await self.initialize()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.cleanup()
    return False
```

### Spawning Background Task
```python
task = asyncio.create_task(coro)
self._background_tasks.append(task)
```

### Cancelling Background Tasks
```python
for task in self._background_tasks:
    if not task.done():
        task.cancel()

await asyncio.wait(
    self._background_tasks,
    timeout=5.0,
    return_when=asyncio.ALL_COMPLETED
)
```

### Parallel Execution
```python
results = await asyncio.gather(
    task1(), task2(), task3(),
    return_exceptions=True
)
```

### Timeout
```python
result = await asyncio.wait_for(
    slow_operation(),
    timeout=30.0
)
```

---

## Resources

- Python asyncio documentation: https://docs.python.org/3/library/asyncio.html
- Python async/await tutorial: https://realpython.com/async-io-python/
- HoloLoom weaving_orchestrator.py - Reference implementation

---

**Remember**: When in doubt, follow the patterns in `weaving_orchestrator.py` - it demonstrates proper lifecycle management, background task tracking, and resource cleanup.
