"""
Task System Unit Tests

Tests:
- TaskState transitions
- TaskRegistry plugin system
- TaskManager lifecycle
- State persistence and recovery
- Example task execution
- Pause/resume/cancel functionality

Created: 2025-11-22
Part of: Voice UX Milestone 2 Phase 3
"""

import asyncio
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from typing import AsyncIterator

# Import task system
from task_system import (
    TaskState, TaskProgress, TaskProtocol,
    TaskRegistry, TaskManager, TaskSystem,
    BaseTask, register_task
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for persistence tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def task_manager(temp_dir):
    """Create TaskManager with temp persistence path"""
    return TaskManager(persistence_path=temp_dir)


# ============================================================================
# Mock Tasks for Testing
# ============================================================================

class MockTask(BaseTask):
    """Simple mock task for testing"""

    def __init__(self, name: str = "Mock Task", steps: int = 3):
        super().__init__(name)
        self.total_steps = steps
        self.execution_log = []

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """Execute mock task"""
        self._state = TaskState.RUNNING
        self.execution_log.append("started")

        for step in range(self.total_steps):
            # Check cancellation
            if self._check_cancelled():
                self.execution_log.append("cancelled")
                yield TaskProgress(
                    task_id=self.task_id,
                    state=TaskState.CANCELLED,
                    progress_pct=step / self.total_steps,
                    current_step=f"Step {step+1}",
                    total_steps=self.total_steps,
                    completed_steps=step,
                    status_message="Task cancelled"
                )
                return

            # Check pause
            await self._check_pause()

            # Yield progress
            yield TaskProgress(
                task_id=self.task_id,
                state=TaskState.RUNNING,
                progress_pct=(step + 1) / self.total_steps,
                current_step=f"Step {step+1}",
                total_steps=self.total_steps,
                completed_steps=step + 1,
                status_message=f"Executing step {step+1}"
            )

            self.execution_log.append(f"step_{step+1}")
            await asyncio.sleep(0.1)  # Simulate work

        # Completed
        self._state = TaskState.COMPLETED
        self.execution_log.append("completed")
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            progress_pct=1.0,
            current_step="Complete",
            total_steps=self.total_steps,
            completed_steps=self.total_steps,
            status_message="Task complete"
        )


# ============================================================================
# Test TaskState and TaskProgress
# ============================================================================

def test_task_state_enum():
    """Test TaskState enum values"""
    assert TaskState.PENDING.value == "pending"
    assert TaskState.RUNNING.value == "running"
    assert TaskState.PAUSED.value == "paused"
    assert TaskState.COMPLETED.value == "completed"
    assert TaskState.FAILED.value == "failed"
    assert TaskState.CANCELLED.value == "cancelled"


def test_task_progress_serialization():
    """Test TaskProgress to_dict() and from_dict()"""
    # Create progress
    progress = TaskProgress(
        task_id="test_123",
        state=TaskState.RUNNING,
        progress_pct=0.5,
        current_step="Processing",
        total_steps=10,
        completed_steps=5,
        status_message="Halfway there"
    )

    # Serialize
    data = progress.to_dict()
    assert data["task_id"] == "test_123"
    assert data["state"] == "running"
    assert data["progress_pct"] == 0.5
    assert data["current_step"] == "Processing"

    # Deserialize
    restored = TaskProgress.from_dict(data)
    assert restored.task_id == progress.task_id
    assert restored.state == progress.state
    assert restored.progress_pct == progress.progress_pct
    assert restored.current_step == progress.current_step


def test_task_progress_timestamp():
    """Test automatic timestamp generation"""
    progress = TaskProgress(
        task_id="test_123",
        state=TaskState.RUNNING,
        progress_pct=0.0,
        current_step="Start",
        total_steps=1,
        completed_steps=0,
        status_message="Starting"
    )

    # Timestamp should be auto-generated
    assert progress.timestamp != ""
    assert len(progress.timestamp) > 0


# ============================================================================
# Test BaseTask
# ============================================================================

@pytest.mark.asyncio
async def test_base_task_execution():
    """Test BaseTask execute() method"""
    task = MockTask(name="Test Task", steps=3)

    # Task starts in PENDING state
    assert task._state == TaskState.PENDING

    # Execute task
    context = {"test": "data"}
    progress_updates = []

    async for progress in task.execute(context):
        progress_updates.append(progress)

    # Verify execution
    assert len(progress_updates) == 4  # 3 steps + 1 completion
    assert progress_updates[0].progress_pct == pytest.approx(0.33, 0.1)
    assert progress_updates[1].progress_pct == pytest.approx(0.67, 0.1)
    assert progress_updates[2].progress_pct == 1.0
    assert progress_updates[3].state == TaskState.COMPLETED

    # Verify execution log
    assert task.execution_log == ["started", "step_1", "step_2", "step_3", "completed"]


@pytest.mark.asyncio
async def test_base_task_pause_resume():
    """Test pause/resume functionality"""
    task = MockTask(steps=5)

    # Start execution
    progress_gen = task.execute({})

    # Get first progress update
    progress = await progress_gen.__anext__()
    assert progress.state == TaskState.RUNNING

    # Pause task
    success = await task.pause()
    assert success is True
    assert task._state == TaskState.PAUSED

    # Cannot pause already paused task
    success = await task.pause()
    assert success is False

    # Resume task
    success = await task.resume()
    assert success is True
    assert task._state == TaskState.RUNNING

    # Cannot resume running task
    success = await task.resume()
    assert success is False

    # Complete execution
    async for _ in progress_gen:
        pass

    assert task._state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_base_task_cancel():
    """Test cancel functionality"""
    task = MockTask(steps=5)

    # Start execution
    progress_gen = task.execute({})

    # Get first 2 progress updates
    await progress_gen.__anext__()
    await progress_gen.__anext__()

    # Cancel task
    success = await task.cancel()
    assert success is True
    assert task._state == TaskState.CANCELLED
    assert task._cancelled is True

    # Next progress should be CANCELLED
    progress = await progress_gen.__anext__()
    assert progress.state == TaskState.CANCELLED

    # Execution should stop
    with pytest.raises(StopAsyncIteration):
        await progress_gen.__anext__()

    # Verify execution log
    assert "cancelled" in task.execution_log
    assert "completed" not in task.execution_log


# ============================================================================
# Test TaskRegistry
# ============================================================================

def test_task_registry_registration():
    """Test task registration"""
    # Clear registry for test
    TaskRegistry._tasks.clear()

    # Register task
    @register_task("test_task")
    class TestTask(BaseTask):
        def __init__(self):
            super().__init__("Test")

        async def execute(self, context):
            yield TaskProgress(
                task_id=self.task_id,
                state=TaskState.COMPLETED,
                progress_pct=1.0,
                current_step="Done",
                total_steps=1,
                completed_steps=1,
                status_message="Done"
            )

    # Verify registration
    assert TaskRegistry.is_registered("test_task")
    assert "test_task" in TaskRegistry.list_tasks()

    # Create instance
    task = TaskRegistry.create("test_task")
    assert isinstance(task, TestTask)
    assert task.name == "Test"


def test_task_registry_unknown_task():
    """Test creating unknown task raises error"""
    with pytest.raises(ValueError) as exc_info:
        TaskRegistry.create("nonexistent_task")

    assert "Unknown task" in str(exc_info.value)


def test_task_registry_list_tasks():
    """Test listing registered tasks"""
    TaskRegistry._tasks.clear()

    @register_task("task_a")
    class TaskA(BaseTask):
        def __init__(self):
            super().__init__("A")

    @register_task("task_b")
    class TaskB(BaseTask):
        def __init__(self):
            super().__init__("B")

    tasks = TaskRegistry.list_tasks()
    assert tasks == ["task_a", "task_b"]  # Sorted


# ============================================================================
# Test TaskManager
# ============================================================================

@pytest.mark.asyncio
async def test_task_manager_start_task(task_manager):
    """Test starting a task"""
    # Register mock task
    TaskRegistry._tasks.clear()
    TaskRegistry.register("mock", MockTask)

    # Start task
    task_id = await task_manager.start_task("mock", context={}, name="Test")

    # Verify task is active
    assert task_id in task_manager._active_tasks

    # Wait for completion
    await asyncio.sleep(0.5)

    # Verify task completed
    status = task_manager.get_task_status(task_id)
    assert status is not None
    assert status.state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_task_manager_persistence(temp_dir):
    """Test task state persistence"""
    # Create manager
    manager = TaskManager(persistence_path=temp_dir)

    # Register and start task
    TaskRegistry._tasks.clear()
    TaskRegistry.register("mock", MockTask)

    task_id = await manager.start_task("mock", context={})

    # Wait for first progress update
    await asyncio.sleep(0.15)

    # Check persistence file exists
    persist_file = Path(temp_dir) / f"{task_id}.json"
    assert persist_file.exists()

    # Verify persisted data
    with open(persist_file, 'r') as f:
        data = json.load(f)

    assert data["task_id"] == task_id
    assert data["state"] in ["running", "completed"]


@pytest.mark.asyncio
async def test_task_manager_pause_resume(task_manager):
    """Test TaskManager pause/resume operations"""
    # Register mock task
    TaskRegistry._tasks.clear()
    TaskRegistry.register("mock", MockTask)

    # Start task
    task_id = await task_manager.start_task("mock", context={}, steps=10)

    # Wait for task to start
    await asyncio.sleep(0.15)

    # Pause task
    success = await task_manager.pause_task(task_id)
    assert success is True

    # Get current status
    status = task_manager.get_task_status(task_id)
    assert status.state == TaskState.PAUSED

    # Resume task
    success = await task_manager.resume_task(task_id)
    assert success is True

    # Wait for completion
    await asyncio.sleep(1.2)

    # Verify completed
    status = task_manager.get_task_status(task_id)
    assert status.state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_task_manager_cancel(task_manager):
    """Test TaskManager cancel operation"""
    # Register mock task
    TaskRegistry._tasks.clear()
    TaskRegistry.register("mock", MockTask)

    # Start task
    task_id = await task_manager.start_task("mock", context={}, steps=10)

    # Wait for task to start
    await asyncio.sleep(0.15)

    # Cancel task
    success = await task_manager.cancel_task(task_id)
    assert success is True

    # Wait for cancellation to propagate
    await asyncio.sleep(0.2)

    # Verify cancelled
    status = task_manager.get_task_status(task_id)
    assert status.state == TaskState.CANCELLED


@pytest.mark.asyncio
async def test_task_manager_list_active_tasks(task_manager):
    """Test listing active tasks"""
    # Register mock task
    TaskRegistry._tasks.clear()
    TaskRegistry.register("mock", MockTask)

    # Start multiple tasks
    task_ids = []
    for i in range(3):
        task_id = await task_manager.start_task("mock", context={}, steps=10)
        task_ids.append(task_id)

    # Wait for tasks to start
    await asyncio.sleep(0.15)

    # List active tasks
    active = task_manager.list_active_tasks()
    assert len(active) == 3

    # All should be RUNNING
    for progress in active:
        assert progress.state == TaskState.RUNNING


# ============================================================================
# Test TaskSystem (High-Level Interface)
# ============================================================================

@pytest.mark.asyncio
async def test_task_system_run():
    """Test TaskSystem.run() method"""
    # Register mock task
    TaskRegistry._tasks.clear()
    TaskRegistry.register("mock", MockTask)

    # Create TaskSystem
    system = TaskSystem()

    # Run task
    message = await system.run("mock", steps=3)
    assert "Running" in message

    # Wait for completion
    await asyncio.sleep(0.5)

    # Check status
    status_msg = await system.status()
    assert "complete" in status_msg.lower() or "no tasks" in status_msg.lower()


@pytest.mark.asyncio
async def test_task_system_pause_resume():
    """Test TaskSystem pause/resume"""
    TaskRegistry._tasks.clear()
    TaskRegistry.register("mock", MockTask)

    system = TaskSystem()

    # Start task
    await system.run("mock", steps=10)
    await asyncio.sleep(0.15)

    # Pause
    msg = await system.pause()
    assert msg == "Paused"

    # Resume
    msg = await system.resume()
    assert msg == "Resumed"


@pytest.mark.asyncio
async def test_task_system_stop():
    """Test TaskSystem.stop() method"""
    TaskRegistry._tasks.clear()
    TaskRegistry.register("mock", MockTask)

    system = TaskSystem()

    # Start task
    await system.run("mock", steps=10)
    await asyncio.sleep(0.15)

    # Stop
    msg = await system.stop()
    assert msg == "Stopped"

    # Wait for cancellation
    await asyncio.sleep(0.2)

    # Verify no active tasks
    msg = await system.status()
    assert "No tasks" in msg


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
