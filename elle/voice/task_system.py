"""
Task Delegation System

Provides protocol-based task execution with:
- Multi-turn conversations
- State persistence
- Progress tracking
- Plugin architecture

Created: 2025-11-22
Part of: Voice UX Milestone 2 Phase 3
"""

import asyncio
import uuid
import json
from pathlib import Path
from typing import (
    Protocol, AsyncIterator, Any, Dict, List, Optional,
    Type, Callable, Awaitable, Tuple
)
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


# ============================================================================
# Core Protocols and Types
# ============================================================================

class TaskState(Enum):
    """Task lifecycle states"""
    PENDING = "pending"        # Not started
    RUNNING = "running"        # Actively executing
    PAUSED = "paused"          # Temporarily suspended
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Error occurred
    CANCELLED = "cancelled"    # User cancelled


@dataclass
class TaskProgress:
    """Progress snapshot for a task"""
    task_id: str
    state: TaskState
    progress_pct: float        # 0.0-1.0
    current_step: str
    total_steps: int
    completed_steps: int
    status_message: str
    timestamp: str = ""        # ISO format
    error: Optional[str] = None

    def __post_init__(self):
        """Set timestamp if not provided"""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['state'] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskProgress':
        """Create from dictionary"""
        data['state'] = TaskState(data['state'])
        return cls(**data)


class TaskProtocol(Protocol):
    """Protocol for executable tasks"""

    @property
    def task_id(self) -> str:
        """Unique task identifier"""
        ...

    @property
    def name(self) -> str:
        """Human-readable task name"""
        ...

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """
        Execute task with progress updates

        Yields progress updates as task runs.
        Final yield should have state=COMPLETED or FAILED.

        Args:
            context: Execution context (thread info, user data, etc.)

        Yields:
            TaskProgress objects with current state
        """
        ...

    async def pause(self) -> bool:
        """
        Pause execution

        Returns:
            True if paused successfully, False otherwise
        """
        ...

    async def resume(self) -> bool:
        """
        Resume from pause

        Returns:
            True if resumed successfully, False otherwise
        """
        ...

    async def cancel(self) -> bool:
        """
        Cancel execution

        Returns:
            True if cancelled successfully, False otherwise
        """
        ...


# ============================================================================
# Task Registry (Plugin System)
# ============================================================================

class TaskRegistry:
    """
    Plugin registry for task types

    Tasks self-register at import time using decorator:
    @register_task("analyze")
    """

    _tasks: Dict[str, Type[TaskProtocol]] = {}

    @classmethod
    def register(cls, name: str, task_class: Type[TaskProtocol]):
        """
        Register a task type

        Args:
            name: Task name (e.g., "analyze", "search")
            task_class: Class implementing TaskProtocol
        """
        cls._tasks[name] = task_class

    @classmethod
    def create(cls, name: str, **kwargs) -> TaskProtocol:
        """
        Create task instance by name

        Args:
            name: Task name
            **kwargs: Arguments to pass to task constructor

        Returns:
            Task instance

        Raises:
            ValueError: If task name not registered
        """
        if name not in cls._tasks:
            available = ', '.join(cls._tasks.keys()) or 'none'
            raise ValueError(
                f"Unknown task: '{name}'. Available tasks: {available}"
            )

        return cls._tasks[name](**kwargs)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """
        List all registered task names

        Returns:
            List of task names
        """
        return sorted(cls._tasks.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if task is registered"""
        return name in cls._tasks


def register_task(name: str):
    """
    Decorator to register a task type

    Usage:
        @register_task("analyze")
        class AnalyzeTask:
            ...
    """
    def decorator(task_class: Type[TaskProtocol]):
        TaskRegistry.register(name, task_class)
        return task_class
    return decorator


# ============================================================================
# Task Manager (State Management)
# ============================================================================

class TaskManager:
    """
    Manages task lifecycle and state

    Responsibilities:
    - Track running tasks
    - Persist task state
    - Handle multi-turn conversations
    - Report progress
    """

    def __init__(self, persistence_path: str = "elle/data/tasks"):
        """
        Initialize task manager

        Args:
            persistence_path: Directory for persisting task state
        """
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Active tasks: {task_id: (task, asyncio.Task)}
        self._active_tasks: Dict[str, Tuple[TaskProtocol, asyncio.Task]] = {}

        # Task history: {task_id: TaskProgress}
        self._task_history: Dict[str, TaskProgress] = {}

        # Load persisted tasks
        self._load_tasks()

    async def start_task(
        self,
        task_name: str,
        context: dict,
        **kwargs
    ) -> str:
        """
        Start a new task

        Args:
            task_name: Name of registered task
            context: Execution context
            **kwargs: Arguments for task constructor

        Returns:
            task_id: Unique identifier for task

        Raises:
            ValueError: If task name not registered
        """
        # Create task instance
        task = TaskRegistry.create(task_name, **kwargs)

        # Start execution in background
        async_task = asyncio.create_task(
            self._execute_task(task, context)
        )

        # Track active task
        self._active_tasks[task.task_id] = (task, async_task)

        return task.task_id

    async def _execute_task(
        self,
        task: TaskProtocol,
        context: dict
    ):
        """
        Execute task and track progress

        Args:
            task: Task to execute
            context: Execution context
        """
        try:
            async for progress in task.execute(context):
                # Update history
                self._task_history[task.task_id] = progress

                # Persist state
                self._persist_task(task.task_id, progress)

                # Note: Voice feedback is handled by VoiceFeedbackManager (Phase 4)

        except Exception as e:
            # Handle failure
            progress = TaskProgress(
                task_id=task.task_id,
                state=TaskState.FAILED,
                progress_pct=0.0,
                current_step="Error",
                total_steps=0,
                completed_steps=0,
                status_message=f"Task failed",
                error=str(e)
            )
            self._task_history[task.task_id] = progress
            self._persist_task(task.task_id, progress)

        finally:
            # Cleanup active tasks
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]

    async def pause_task(self, task_id: str) -> bool:
        """
        Pause running task

        Args:
            task_id: Task identifier

        Returns:
            True if paused successfully, False otherwise
        """
        if task_id not in self._active_tasks:
            return False

        task, _ = self._active_tasks[task_id]
        return await task.pause()

    async def resume_task(self, task_id: str) -> bool:
        """
        Resume paused task

        Args:
            task_id: Task identifier

        Returns:
            True if resumed successfully, False otherwise
        """
        if task_id not in self._active_tasks:
            return False

        task, _ = self._active_tasks[task_id]
        return await task.resume()

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel running task

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled successfully, False otherwise
        """
        if task_id not in self._active_tasks:
            return False

        task, async_task = self._active_tasks[task_id]

        # Cancel task
        success = await task.cancel()

        # Cancel asyncio task
        async_task.cancel()

        # Remove from active tasks
        del self._active_tasks[task_id]

        return success

    def get_task_status(self, task_id: str) -> Optional[TaskProgress]:
        """
        Get current task status

        Args:
            task_id: Task identifier

        Returns:
            TaskProgress if task exists, None otherwise
        """
        return self._task_history.get(task_id)

    def list_active_tasks(self) -> List[TaskProgress]:
        """
        List all active tasks

        Returns:
            List of TaskProgress for active tasks
        """
        return [
            self._task_history[task_id]
            for task_id in self._active_tasks.keys()
            if task_id in self._task_history
        ]

    def get_current_task(self) -> Optional[TaskProgress]:
        """
        Get most recent active task

        Returns:
            TaskProgress of most recent task, or None if no active tasks
        """
        active = self.list_active_tasks()
        return active[-1] if active else None

    def _persist_task(self, task_id: str, progress: TaskProgress):
        """
        Persist task state to disk

        Args:
            task_id: Task identifier
            progress: Current progress snapshot
        """
        path = self.persistence_path / f"{task_id}.json"

        try:
            with open(path, 'w') as f:
                json.dump(progress.to_dict(), f, indent=2)
        except Exception as e:
            # Don't crash on persistence errors
            print(f"[WARNING] Failed to persist task {task_id}: {e}")

    def _load_tasks(self):
        """Load persisted tasks from disk"""
        if not self.persistence_path.exists():
            return

        for json_file in self.persistence_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    progress = TaskProgress.from_dict(data)
                    self._task_history[progress.task_id] = progress
            except Exception as e:
                print(f"[WARNING] Failed to load task from {json_file}: {e}")


# ============================================================================
# Unified Task System (High-Level Interface)
# ============================================================================

class TaskSystem:
    """
    Unified task delegation system

    High-level interface that combines:
    - TaskManager (state management)
    - TaskRegistry (plugin system)

    Usage:
        task_system = TaskSystem()
        await task_system.run("analyze", data_source="logs")
        await task_system.pause()
        status = await task_system.status()
    """

    def __init__(self):
        """Initialize task system"""
        self.task_manager = TaskManager()

    async def run(self, task_name: str, context: Optional[dict] = None, **kwargs) -> str:
        """
        Run task

        Args:
            task_name: Name of registered task
            context: Execution context (optional)
            **kwargs: Arguments for task constructor

        Returns:
            Brief confirmation message

        Raises:
            ValueError: If task name not registered
        """
        if context is None:
            context = {}

        # Start task
        task_id = await self.task_manager.start_task(
            task_name,
            context,
            **kwargs
        )

        # Get task for name
        task = None
        for tid, (t, _) in self.task_manager._active_tasks.items():
            if tid == task_id:
                task = t
                break

        name = task.name if task else task_name

        return f"Running {name}"

    async def pause(self) -> str:
        """
        Pause current task

        Returns:
            Brief status message
        """
        current = self.task_manager.get_current_task()

        if not current:
            return "No tasks running"

        success = await self.task_manager.pause_task(current.task_id)

        if success:
            return "Paused"
        else:
            return "Cannot pause"

    async def resume(self) -> str:
        """
        Resume paused task

        Returns:
            Brief status message
        """
        current = self.task_manager.get_current_task()

        if not current:
            return "No tasks to resume"

        if current.state != TaskState.PAUSED:
            return "Task not paused"

        success = await self.task_manager.resume_task(current.task_id)

        if success:
            return "Resumed"
        else:
            return "Cannot resume"

    async def stop(self) -> str:
        """
        Stop (cancel) current task

        Returns:
            Brief status message
        """
        current = self.task_manager.get_current_task()

        if not current:
            return "No tasks running"

        success = await self.task_manager.cancel_task(current.task_id)

        if success:
            return "Stopped"
        else:
            return "Cannot stop"

    async def status(self) -> str:
        """
        Get status of current task

        Returns:
            Brief status message
        """
        current = self.task_manager.get_current_task()

        if not current:
            return "No tasks running"

        # Format brief status message
        pct = int(current.progress_pct * 100)

        if current.state == TaskState.COMPLETED:
            return f"{current.status_message}"
        elif current.state == TaskState.FAILED:
            return f"Task failed: {current.error or 'Unknown error'}"
        elif current.state == TaskState.PAUSED:
            return f"Paused at {current.current_step}: {pct}%"
        else:
            return f"{current.current_step}: {pct}% complete"

    def list_available_tasks(self) -> List[str]:
        """
        List all registered task types

        Returns:
            List of task names
        """
        return TaskRegistry.list_tasks()


# ============================================================================
# Example Base Class (for convenience)
# ============================================================================

class BaseTask:
    """
    Base class for tasks (optional convenience)

    Provides default implementations of pause/resume/cancel.
    Subclasses only need to implement execute().
    """

    def __init__(self, name: str):
        """
        Initialize base task

        Args:
            name: Human-readable task name
        """
        self.task_id = f"task_{uuid.uuid4().hex[:8]}"
        self.name = name
        self._state = TaskState.PENDING
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default
        self._cancelled = False

    async def pause(self) -> bool:
        """Pause execution"""
        if self._state != TaskState.RUNNING:
            return False

        self._state = TaskState.PAUSED
        self._pause_event.clear()
        return True

    async def resume(self) -> bool:
        """Resume from pause"""
        if self._state != TaskState.PAUSED:
            return False

        self._state = TaskState.RUNNING
        self._pause_event.set()
        return True

    async def cancel(self) -> bool:
        """Cancel execution"""
        if self._state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
            return False

        self._cancelled = True
        self._state = TaskState.CANCELLED
        self._pause_event.set()  # Unblock if paused
        return True

    async def _check_pause(self):
        """Check if task should pause (call in execute loop)"""
        await self._pause_event.wait()

    def _check_cancelled(self) -> bool:
        """Check if task was cancelled"""
        return self._cancelled
