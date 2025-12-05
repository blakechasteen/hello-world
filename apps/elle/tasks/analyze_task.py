"""
Analyze Task - Example Task Implementation

Demonstrates:
- Async generator progress updates
- Pause/resume/cancel functionality
- Multi-step analysis process
- Integration with task system

Created: 2025-11-22
Part of: Voice UX Milestone 2 Phase 3
"""

import asyncio
from typing import AsyncIterator, Any
from pathlib import Path

# Import task system components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from voice.task_system import (
    TaskProgress, TaskState, BaseTask, register_task
)


@register_task("analyze")
class AnalyzeTask(BaseTask):
    """
    Example task: Analyze data source

    Demonstrates multi-step analysis with progress tracking:
    1. Load data source
    2. Extract entities
    3. Compute statistics
    4. Generate insights
    5. Create summary

    Usage:
        task = AnalyzeTask(data_source="logs")
        async for progress in task.execute(context):
            print(f"{progress.current_step}: {progress.progress_pct:.0%}")
    """

    def __init__(self, data_source: str = "unknown"):
        """
        Initialize analyze task

        Args:
            data_source: Data source identifier (e.g., "logs", "code", "docs")
        """
        super().__init__(name=f"Analyze {data_source}")
        self.data_source = data_source
        self.total_steps = 5

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """
        Execute analysis with progress updates

        Args:
            context: Execution context (thread info, user data, etc.)

        Yields:
            TaskProgress objects with current state
        """
        self._state = TaskState.RUNNING

        steps = [
            ("Loading data", 0.2, 1.5),      # (step_name, progress, duration_seconds)
            ("Extracting entities", 0.4, 2.0),
            ("Computing statistics", 0.6, 1.0),
            ("Generating insights", 0.8, 1.5),
            ("Creating summary", 1.0, 0.5),
        ]

        for idx, (step_name, progress_pct, duration) in enumerate(steps):
            # Check if cancelled
            if self._check_cancelled():
                yield TaskProgress(
                    task_id=self.task_id,
                    state=TaskState.CANCELLED,
                    progress_pct=progress_pct,
                    current_step=step_name,
                    total_steps=self.total_steps,
                    completed_steps=idx,
                    status_message=f"Analysis cancelled at {step_name}"
                )
                return

            # Check if paused
            await self._check_pause()

            # Yield progress update
            yield TaskProgress(
                task_id=self.task_id,
                state=TaskState.RUNNING,
                progress_pct=progress_pct,
                current_step=step_name,
                total_steps=self.total_steps,
                completed_steps=idx,
                status_message=f"Analyzing {self.data_source}: {step_name}"
            )

            # Simulate work (in real implementation, do actual analysis)
            await asyncio.sleep(duration)

        # Task completed successfully
        self._state = TaskState.COMPLETED
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            progress_pct=1.0,
            current_step="Complete",
            total_steps=self.total_steps,
            completed_steps=self.total_steps,
            status_message=f"Analysis of {self.data_source} complete"
        )


@register_task("analyze_sentiment")
class AnalyzeSentimentTask(BaseTask):
    """
    Example task: Sentiment analysis

    Demonstrates:
    - Different analysis type
    - Variable step count
    - Error handling
    """

    def __init__(self, text_source: str = "unknown"):
        super().__init__(name=f"Analyze sentiment: {text_source}")
        self.text_source = text_source
        self.total_steps = 3

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """Execute sentiment analysis"""
        self._state = TaskState.RUNNING

        steps = [
            ("Tokenizing text", 0.33, 0.5),
            ("Computing sentiment scores", 0.67, 1.0),
            ("Generating report", 1.0, 0.5),
        ]

        for idx, (step_name, progress_pct, duration) in enumerate(steps):
            if self._check_cancelled():
                yield TaskProgress(
                    task_id=self.task_id,
                    state=TaskState.CANCELLED,
                    progress_pct=progress_pct,
                    current_step=step_name,
                    total_steps=self.total_steps,
                    completed_steps=idx,
                    status_message="Sentiment analysis cancelled"
                )
                return

            await self._check_pause()

            yield TaskProgress(
                task_id=self.task_id,
                state=TaskState.RUNNING,
                progress_pct=progress_pct,
                current_step=step_name,
                total_steps=self.total_steps,
                completed_steps=idx,
                status_message=f"Sentiment analysis: {step_name}"
            )

            await asyncio.sleep(duration)

        # Completed
        self._state = TaskState.COMPLETED
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            progress_pct=1.0,
            current_step="Complete",
            total_steps=self.total_steps,
            completed_steps=self.total_steps,
            status_message=f"Sentiment analysis of {self.text_source} complete"
        )


# Example usage
async def demo_analyze_task():
    """Demo AnalyzeTask execution"""
    print("\n" + "="*60)
    print("ANALYZE TASK DEMO")
    print("="*60)

    # Create task
    task = AnalyzeTask(data_source="conversation_logs")

    print(f"\n[Task Created] {task.name} (ID: {task.task_id})")
    print(f"Total steps: {task.total_steps}")

    # Execute with progress tracking
    print("\n[Execution Started]")
    context = {
        "thread_id": "thread_1",
        "user": "demo_user"
    }

    async for progress in task.execute(context):
        # Format progress bar
        bar_length = 30
        filled = int(bar_length * progress.progress_pct)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"  [{bar}] {progress.progress_pct:>5.0%} | "
              f"{progress.current_step} ({progress.completed_steps}/{progress.total_steps})")

    print(f"\n[Task Complete] Final state: {task._state.value}")


async def demo_pause_resume():
    """Demo pause/resume functionality"""
    print("\n" + "="*60)
    print("PAUSE/RESUME DEMO")
    print("="*60)

    task = AnalyzeTask(data_source="system_metrics")
    print(f"\n[Task Created] {task.name}")

    context = {}

    # Start task
    progress_gen = task.execute(context)

    # Process first 2 steps
    for i in range(2):
        progress = await progress_gen.__anext__()
        print(f"  Step {i+1}: {progress.current_step} ({progress.progress_pct:.0%})")

    # Pause
    print("\n[Pausing task...]")
    await task.pause()
    print(f"  Task state: {task._state.value}")

    # Try to get next progress (will block until resumed)
    print("\n[Attempting to continue while paused...]")
    print("  (Task is paused, waiting...)")

    # Resume after 1 second
    async def resume_after_delay():
        await asyncio.sleep(1)
        print("\n[Resuming task...]")
        await task.resume()
        print(f"  Task state: {task._state.value}")

    resume_task = asyncio.create_task(resume_after_delay())

    # Continue execution
    progress = await progress_gen.__anext__()
    print(f"  Resumed at: {progress.current_step} ({progress.progress_pct:.0%})")

    await resume_task

    # Complete remaining steps
    async for progress in progress_gen:
        print(f"  Step: {progress.current_step} ({progress.progress_pct:.0%})")

    print(f"\n[Task Complete] Final state: {task._state.value}")


async def demo_cancel():
    """Demo cancel functionality"""
    print("\n" + "="*60)
    print("CANCEL DEMO")
    print("="*60)

    task = AnalyzeTask(data_source="error_logs")
    print(f"\n[Task Created] {task.name}")

    context = {}
    progress_gen = task.execute(context)

    # Process first 2 steps
    for i in range(2):
        progress = await progress_gen.__anext__()
        print(f"  Step {i+1}: {progress.current_step} ({progress.progress_pct:.0%})")

    # Cancel
    print("\n[Cancelling task...]")
    await task.cancel()
    print(f"  Task state: {task._state.value}")

    # Try to continue (should get CANCELLED state)
    progress = await progress_gen.__anext__()
    print(f"\n[Task Cancelled] State: {progress.state.value}")
    print(f"  Message: {progress.status_message}")


if __name__ == "__main__":
    print("AnalyzeTask Example Implementation")
    print("Demonstrates task system with progress tracking\n")

    # Run demos
    asyncio.run(demo_analyze_task())
    asyncio.run(demo_pause_resume())
    asyncio.run(demo_cancel())

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE")
    print("="*60)
