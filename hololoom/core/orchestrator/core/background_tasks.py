#!/usr/bin/env python3
"""
Background Task Management
==========================

Background task spawning and lifecycle management.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 2520-2662 (~142 lines total, 2 methods)

This module handles:
- Background consolidation loop startup (statistical mechanics)
- Background task spawning with automatic cleanup
- Task lifecycle tracking and graceful shutdown

Author: Claude Code (Elegance Pass Refactoring - Phase 3)
Date: 2025-11-22
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator


logger = logging.getLogger(__name__)


def start_background_consolidation(orchestrator: WeavingOrchestrator) -> None:
    """
    Start background consolidation loop.

    Should be called after initialization if statistical mechanics is enabled.
    Creates a background task that periodically consolidates memory using
    statistical mechanics (Gibbs distribution, phase transitions).

    Args:
        orchestrator: The WeavingOrchestrator instance

    Example:
        >>> orchestrator = WeavingOrchestrator(cfg=config, shards=shards)
        >>> if orchestrator.enable_statistical_mechanics:
        ...     start_background_consolidation(orchestrator)
        >>> # Background consolidation now running every N seconds

    Side Effects:
        - Creates background asyncio task
        - Adds task to orchestrator._background_tasks list
        - Sets orchestrator._consolidation_task

    Note:
        - Safe to call multiple times (checks if already running)
        - Logs warning if statistical mechanics not enabled
        - Task runs until orchestrator.close() is called
    """
    if not orchestrator.enable_statistical_mechanics or not orchestrator.stat_mech_engine:
        logger.warning("Statistical mechanics not enabled, cannot start consolidation")
        return

    if orchestrator._consolidation_task is not None:
        logger.warning("Background consolidation already running")
        return

    # Create background task
    orchestrator._consolidation_task = asyncio.create_task(
        orchestrator._background_consolidation_loop()
    )
    orchestrator._background_tasks.append(orchestrator._consolidation_task)
    logger.info("Background consolidation started")


def spawn_background_task(
    orchestrator: WeavingOrchestrator,
    coro
) -> asyncio.Task:
    """
    Spawn a background task and track it for cleanup.

    Creates an asyncio Task from the given coroutine and adds it to the
    orchestrator's tracked background tasks. The task is automatically
    cleaned up from the tracking list when it completes.

    NOTE: While this method is synchronous, _background_tasks access is
    protected by _bg_lock in async contexts. Since asyncio is single-threaded,
    synchronous appends are safe, but removal via callback needs protection.

    Args:
        orchestrator: The WeavingOrchestrator instance
        coro: Coroutine to run in background

    Returns:
        asyncio.Task object that can be awaited or cancelled

    Example:
        >>> async def some_async_work():
        ...     await asyncio.sleep(5)
        ...     print("Background work done")
        >>>
        >>> orchestrator = WeavingOrchestrator(cfg=config, shards=shards)
        >>> task = spawn_background_task(orchestrator, some_async_work())
        >>> # Task now running in background
        >>> # Will be automatically cancelled when orchestrator.close() is called

    Side Effects:
        - Creates asyncio.Task
        - Adds task to orchestrator._background_tasks
        - Registers cleanup callback for task completion

    Thread Safety:
        - Safe in single-threaded asyncio event loop
        - List append is atomic in CPython
        - Callback-based removal is thread-safe
    """
    task = asyncio.create_task(coro)

    # Append is atomic in Python, safe in single-threaded asyncio
    orchestrator._background_tasks.append(task)

    # Clean up completed tasks (callback is synchronous, but list operations are atomic)
    def cleanup_callback(t):
        try:
            orchestrator._background_tasks.remove(t)
        except ValueError:
            pass  # Already removed

    task.add_done_callback(cleanup_callback)

    return task
