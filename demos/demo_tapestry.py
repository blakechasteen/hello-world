"""
Demo: Tapestry Session Continuity System

Demonstrates the complete Tapestry workflow:
1. Start a new session with goal decomposition
2. Weave threads with holistic verification
3. Resume existing sessions
4. Handle tangled threads

Usage:
    PYTHONPATH=. python demos/demo_tapestry.py

Created: December 2025
"""

import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.tapestry import (
    LoomKeeper,
    Tapestry,
    Thread,
    ThreadStatus,
    FabricInspector,
    SignalRegistry,
    FabricCheckResult,
    SignalResult
)
from HoloLoom.tapestry.backends.json_backend import JsonTapestryBackend
from HoloLoom.tapestry.git import GitIntegration


# ─────────────────────────────────────────────────────────────────
# Demo Helpers
# ─────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_thread(thread: Thread, indent: str = "") -> None:
    """Print thread details."""
    status_icons = {
        ThreadStatus.UNWOVEN: "[ ]",
        ThreadStatus.WEAVING: "[~]",
        ThreadStatus.WOVEN: "[x]",
        ThreadStatus.TANGLED: "[!]",
        ThreadStatus.UNRAVELED: "[<]"
    }
    icon = status_icons.get(thread.status, "?")
    print(f"{indent}{icon} [{thread.id}] {thread.description[:50]}")
    print(f"{indent}  Status: {thread.status.value}")
    if thread.commit_hash:
        print(f"{indent}  Commit: {thread.commit_hash}")
    if thread.fabric_check:
        print(f"{indent}  Confidence: {thread.fabric_check.confidence:.1%}")


def print_tapestry(tapestry: Tapestry) -> None:
    """Print tapestry overview."""
    print(f"Loom ID: {tapestry.loom_id}")
    print(f"Goal: {tapestry.goal}")
    print(f"Threads: {len(tapestry.threads)}")
    print(f"Status: {tapestry.get_status_summary()}")
    print("\nThreads:")
    for thread in tapestry.threads:
        print_thread(thread, indent="  ")


# ─────────────────────────────────────────────────────────────────
# Mock Executor (simulates actual work)
# ─────────────────────────────────────────────────────────────────

class MockExecutor:
    """
    Simulates thread execution.

    In production, this would be your actual coding agent
    that implements the thread's task.
    """

    def __init__(self, success_rate: float = 0.8):
        self.success_rate = success_rate
        self.execution_count = 0

    async def execute(self, thread: Thread) -> dict:
        """Execute a thread (mock implementation)."""
        import random

        self.execution_count += 1
        print(f"    Executing: {thread.description[:40]}...")

        # Simulate work
        await asyncio.sleep(0.1)

        # Simulate success/failure
        success = random.random() < self.success_rate

        if success:
            print(f"    [OK] Execution successful")
            return {
                "success": True,
                "changes": ["file1.py", "file2.py"],
                "lines_added": random.randint(10, 100),
                "lines_removed": random.randint(0, 20)
            }
        else:
            print(f"    [FAIL] Execution failed")
            raise RuntimeError(f"Failed to execute: {thread.description}")


# ─────────────────────────────────────────────────────────────────
# Mock Signal (for demo without full HoloLoom integration)
# ─────────────────────────────────────────────────────────────────

class MockSignal:
    """Simple mock signal for demo purposes."""

    def __init__(self, name: str, weight: float, pass_rate: float = 0.9):
        self.name = name
        self.weight = weight
        self.pass_rate = pass_rate

    async def check(self, thread: Thread, context: dict) -> SignalResult:
        """Run mock verification."""
        import random

        passed = random.random() < self.pass_rate
        score = random.uniform(0.7, 1.0) if passed else random.uniform(0.2, 0.5)

        return SignalResult(
            signal_name=self.name,
            passed=passed,
            score=score,
            details={"mock": True},
            is_blocker=self.name == "tests"  # Tests are blockers
        )


# ─────────────────────────────────────────────────────────────────
# Demo 1: Basic Session Workflow
# ─────────────────────────────────────────────────────────────────

async def demo_basic_workflow(temp_dir: str) -> None:
    """
    Demonstrate basic tapestry workflow:
    1. Start new session
    2. Weave each thread
    3. Complete session
    """
    print_header("Demo 1: Basic Session Workflow")

    # Setup
    tapestry_path = os.path.join(temp_dir, "tapestry.json")
    backend = JsonTapestryBackend(tapestry_path)

    # Create mock signals for demo
    mock_signals = [
        MockSignal("tests", 0.25, pass_rate=0.95),
        MockSignal("quality", 0.20, pass_rate=0.9),
        MockSignal("docs", 0.15, pass_rate=0.95),
        MockSignal("arch", 0.15, pass_rate=0.9),
        MockSignal("align", 0.15, pass_rate=0.95),
        MockSignal("integration", 0.10, pass_rate=0.9)
    ]
    inspector = FabricInspector(signals=mock_signals)

    # Create keeper without git (for demo)
    keeper = LoomKeeper(
        backend=backend,
        inspector=inspector,
        git=None,
        path=tapestry_path
    )

    executor = MockExecutor(success_rate=0.9)

    print("Starting new session...")
    print()

    # Start session with explicit threads
    goal = "Implement user authentication feature"
    threads = [
        "Research authentication patterns",
        "Implement password hashing",
        "Create login endpoint",
        "Add session management",
        "Write unit tests"
    ]

    async with keeper:
        # Start new session
        tapestry = await keeper.start(goal, threads=threads)

        print("Initial tapestry created:")
        print_tapestry(tapestry)
        print()

        # Weave each thread
        print("Weaving threads...")
        print()

        while True:
            next_thread = tapestry.next_unwoven()
            if not next_thread:
                break

            print(f"  Weaving thread {next_thread.id}:")

            try:
                tapestry = await keeper.weave_thread(
                    tapestry,
                    next_thread,
                    executor.execute
                )

                # Check result
                updated_thread = tapestry._get_thread(next_thread.id)
                if updated_thread.status == ThreadStatus.WOVEN:
                    print(f"    -> Woven successfully!")
                    if updated_thread.fabric_check:
                        print(f"    -> Confidence: {updated_thread.fabric_check.confidence:.1%}")
                else:
                    print(f"    -> Tangled: {updated_thread.fabric_check.blockers if updated_thread.fabric_check else 'Unknown'}")

            except Exception as e:
                print(f"    -> Error: {e}")

            print()

        print("Final tapestry state:")
        print_tapestry(tapestry)

        print()
        print(f"Executions: {executor.execution_count}")
        print(f"Complete: {tapestry.is_complete()}")


# ─────────────────────────────────────────────────────────────────
# Demo 2: Session Resume
# ─────────────────────────────────────────────────────────────────

async def demo_session_resume(temp_dir: str) -> None:
    """
    Demonstrate session persistence and resume:
    1. Start session, weave some threads
    2. "Crash" (exit keeper)
    3. Resume session, continue weaving
    """
    print_header("Demo 2: Session Resume")

    tapestry_path = os.path.join(temp_dir, "resume_tapestry.json")

    # Create mock signals
    mock_signals = [
        MockSignal("tests", 0.25, pass_rate=1.0),  # Always pass for demo
        MockSignal("quality", 0.25, pass_rate=1.0),
        MockSignal("docs", 0.25, pass_rate=1.0),
        MockSignal("arch", 0.25, pass_rate=1.0)
    ]
    inspector = FabricInspector(signals=mock_signals)

    executor = MockExecutor(success_rate=1.0)  # Always succeed

    # Phase 1: Start session, weave 2 threads, then "crash"
    print("Phase 1: Start session, weave 2 threads, then exit...")
    print()

    backend = JsonTapestryBackend(tapestry_path)
    keeper = LoomKeeper(backend=backend, inspector=inspector, git=None)

    async with keeper:
        tapestry = await keeper.start(
            "Build REST API",
            threads=[
                "Design API schema",
                "Implement endpoints",
                "Add authentication",
                "Write documentation"
            ]
        )

        # Weave first 2 threads
        for i in range(2):
            thread = tapestry.next_unwoven()
            if thread:
                tapestry = await keeper.weave_thread(
                    tapestry, thread, executor.execute
                )

        print("Status after Phase 1:")
        print_tapestry(tapestry)
        print()

    # Simulate "crash" - keeper is now closed
    print("--- Session 'crashed' (keeper closed) ---")
    print()

    # Phase 2: Resume session and continue
    print("Phase 2: Resume session and continue...")
    print()

    # Create NEW keeper instance (simulates new process)
    backend2 = JsonTapestryBackend(tapestry_path)
    keeper2 = LoomKeeper(backend=backend2, inspector=inspector, git=None)

    async with keeper2:
        result = await keeper2.resume()

        if result:
            tapestry, next_thread = result
            print("Session resumed!")
            print(f"  Goal: {tapestry.goal}")
            print(f"  Status: {tapestry.get_status_summary()}")
            if next_thread:
                print(f"  Next thread: {next_thread.description}")
            print()

            # Continue weaving remaining threads
            while True:
                thread = tapestry.next_unwoven()
                if not thread:
                    break

                print(f"  Weaving: {thread.description}")
                tapestry = await keeper2.weave_thread(
                    tapestry, thread, executor.execute
                )

            print()
            print("Final status after resume:")
            print_tapestry(tapestry)
        else:
            print("No session to resume!")


# ─────────────────────────────────────────────────────────────────
# Demo 3: Handling Tangled Threads
# ─────────────────────────────────────────────────────────────────

async def demo_tangled_threads(temp_dir: str) -> None:
    """
    Demonstrate handling of failed (tangled) threads:
    1. Thread execution fails
    2. Thread is marked as TANGLED
    3. Recommendations are provided
    """
    print_header("Demo 3: Handling Tangled Threads")

    tapestry_path = os.path.join(temp_dir, "tangled_tapestry.json")

    # Create signals that will sometimes fail
    mock_signals = [
        MockSignal("tests", 0.30, pass_rate=0.5),  # 50% pass rate
        MockSignal("quality", 0.30, pass_rate=0.8),
        MockSignal("security", 0.40, pass_rate=0.9)
    ]
    # Make tests a blocker
    mock_signals[0].is_blocker = True

    inspector = FabricInspector(signals=mock_signals)

    # Executor that sometimes fails
    executor = MockExecutor(success_rate=0.7)

    backend = JsonTapestryBackend(tapestry_path)
    keeper = LoomKeeper(backend=backend, inspector=inspector, git=None)

    async with keeper:
        tapestry = await keeper.start(
            "Risky refactoring task",
            threads=[
                "Update database schema",
                "Migrate existing data",
                "Update API handlers"
            ]
        )

        print("Attempting to weave threads (may fail)...")
        print()

        woven_count = 0
        tangled_count = 0

        for _ in range(len(tapestry.threads)):
            thread = tapestry.next_unwoven()
            if not thread:
                break

            print(f"Weaving: {thread.description}")

            tapestry = await keeper.weave_thread(
                tapestry, thread, executor.execute
            )

            updated = tapestry._get_thread(thread.id)

            if updated.status == ThreadStatus.WOVEN:
                woven_count += 1
                print(f"  -> Woven! Confidence: {updated.fabric_check.confidence:.1%}")
            elif updated.status == ThreadStatus.TANGLED:
                tangled_count += 1
                print(f"  -> Tangled!")
                if updated.fabric_check:
                    if updated.fabric_check.blockers:
                        print(f"    Blockers: {updated.fabric_check.blockers}")
                    if updated.fabric_check.recommendations:
                        print(f"    Recommendations:")
                        for rec in updated.fabric_check.recommendations[:3]:
                            print(f"      - {rec}")
            print()

        print("Final Summary:")
        print(f"  Woven: {woven_count}")
        print(f"  Tangled: {tangled_count}")
        print()
        print_tapestry(tapestry)


# ─────────────────────────────────────────────────────────────────
# Demo 4: Session Context Manager
# ─────────────────────────────────────────────────────────────────

async def demo_session_context(temp_dir: str) -> None:
    """
    Demonstrate the session() context manager for scoped sessions.
    """
    print_header("Demo 4: Session Context Manager")

    tapestry_path = os.path.join(temp_dir, "context_tapestry.json")

    mock_signals = [
        MockSignal("tests", 0.5, pass_rate=1.0),
        MockSignal("quality", 0.5, pass_rate=1.0)
    ]
    inspector = FabricInspector(signals=mock_signals)
    executor = MockExecutor(success_rate=1.0)

    backend = JsonTapestryBackend(tapestry_path)
    keeper = LoomKeeper(backend=backend, inspector=inspector, git=None)

    print("Using session() context manager...")
    print()

    async with keeper:
        async with keeper.session("Quick task with 3 steps") as ctx:
            print(f"Session started!")
            print(f"  Goal: {ctx.tapestry.goal}")
            print()

            # Use the simplified ctx.weave() API
            while ctx.next_thread:
                thread = ctx.next_thread
                print(f"Weaving: {thread.description}")

                success, check = await ctx.weave(executor.execute)

                if success:
                    print(f"  -> Success! Confidence: {check.confidence:.1%}")
                else:
                    print(f"  -> Failed")
                print()

            print("Session complete!")
            print(ctx.status_summary())


# ─────────────────────────────────────────────────────────────────
# Demo 5: FabricInspector Details
# ─────────────────────────────────────────────────────────────────

async def demo_inspector_details(temp_dir: str) -> None:
    """
    Demonstrate FabricInspector's weighted signal aggregation.
    """
    print_header("Demo 5: FabricInspector Signal Aggregation")

    # Create signals with varying pass rates
    signals = [
        MockSignal("tests", 0.20, pass_rate=1.0),      # Always pass
        MockSignal("quality", 0.20, pass_rate=0.8),    # Usually pass
        MockSignal("docs", 0.15, pass_rate=0.9),       # Usually pass
        MockSignal("arch", 0.15, pass_rate=0.7),       # Sometimes fail
        MockSignal("align", 0.15, pass_rate=1.0),      # Always pass
        MockSignal("integration", 0.15, pass_rate=0.5) # 50/50
    ]

    inspector = FabricInspector(signals=signals)

    print("Inspector configuration:")
    print(inspector.describe())
    print()

    # Create a mock thread for testing
    from datetime import datetime
    mock_thread = Thread(
        id="test-1",
        description="Test thread for inspection",
        status=ThreadStatus.WEAVING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    print("Running inspection 5 times to show aggregation variability...")
    print()

    for i in range(5):
        result = await inspector.inspect(mock_thread, {"test": True})

        print(f"Inspection {i+1}:")
        print(f"  Passed: {result.passed}")
        print(f"  Confidence: {result.confidence:.1%}")

        # Show individual signal scores
        print("  Signal scores:")
        for name, sig_result in result.signals.items():
            status = "[OK]" if sig_result.passed else "[X]"
            print(f"    {status} {name}: {sig_result.score:.1%}")

        if result.blockers:
            print(f"  Blockers: {result.blockers}")
        if result.recommendations:
            print(f"  Recommendations: {result.recommendations[:2]}")
        print()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  TAPESTRY DEMO: Session Continuity for HoloLoom")
    print("=" * 60)

    # Create temp directory for demos
    temp_dir = tempfile.mkdtemp(prefix="tapestry_demo_")
    print(f"\nUsing temp directory: {temp_dir}")

    try:
        await demo_basic_workflow(temp_dir)
        await demo_session_resume(temp_dir)
        await demo_tangled_threads(temp_dir)
        await demo_session_context(temp_dir)
        await demo_inspector_details(temp_dir)

        print("\n" + "=" * 60)
        print("  ALL DEMOS COMPLETE")
        print("=" * 60 + "\n")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    asyncio.run(main())
