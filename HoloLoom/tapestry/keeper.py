"""
LoomKeeper - Session Orchestrator

Maintains loom state across sessions.

The LoomKeeper:
1. Starts new weaving sessions
2. Resumes existing sessions
3. Weaves individual threads with verification
4. Commits successful threads
5. Handles failures gracefully

Usage:
    async with LoomKeeper() as keeper:
        async with keeper.session("Implement feature X") as ctx:
            while thread := ctx.next_thread:
                await ctx.weave(thread, my_executor)

Created: December 2025
"""

import logging
from typing import Optional, Tuple, Callable, Awaitable, Any, List
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from HoloLoom.tapestry.protocol import (
    Tapestry,
    Thread,
    ThreadStatus,
    TapestryBackend,
    FabricCheckResult,
    NoTapestryError,
    VerificationFailedError
)
from HoloLoom.tapestry.backends.json_backend import JsonTapestryBackend
from HoloLoom.tapestry.inspector import FabricInspector
from HoloLoom.tapestry.warper import Warper
from HoloLoom.tapestry.git import GitIntegration

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """
    Context for an active weaving session.

    Provides:
    - next_thread: Get next unwoven thread
    - weave(): Execute and verify a thread
    - tapestry: Current tapestry state
    - status_summary(): Human-readable status
    """
    keeper: 'LoomKeeper'
    tapestry: Tapestry
    _current_thread: Optional[Thread] = field(default=None, repr=False)

    @property
    def next_thread(self) -> Optional[Thread]:
        """Get next unwoven thread (respecting dependencies)."""
        return self.tapestry.next_unwoven()

    async def weave(
        self,
        executor: Callable[[Thread], Awaitable[Any]],
        thread: Optional[Thread] = None
    ) -> Tuple[bool, Optional[FabricCheckResult]]:
        """
        Execute and verify a thread.

        Args:
            executor: Async function that executes the thread
            thread: Thread to weave (None = next_thread)

        Returns:
            (success, fabric_check_result)
        """
        thread = thread or self.next_thread
        if not thread:
            logger.info("No more threads to weave")
            return True, None

        self.tapestry = await self.keeper.weave_thread(
            self.tapestry,
            thread,
            executor
        )
        return (
            self.tapestry._get_thread(thread.id).status == ThreadStatus.WOVEN,
            self.tapestry._get_thread(thread.id).fabric_check
        )

    def status_summary(self) -> str:
        """Get human-readable status summary."""
        status = self.tapestry.get_status_summary()
        total = len(self.tapestry.threads)
        woven = status.get('woven', 0)

        lines = [
            f"Goal: {self.tapestry.goal}",
            f"Progress: {woven}/{total} threads woven",
            f"Status: {status}"
        ]

        next_t = self.next_thread
        if next_t:
            lines.append(f"Next: {next_t.description}")

        return "\n".join(lines)

    def is_complete(self) -> bool:
        """Check if all threads are woven."""
        return self.tapestry.is_complete()


class LoomKeeper:
    """
    Maintains loom state across sessions.

    Entry point for session-based workflows.

    Features:
    - Start new sessions with goal decomposition
    - Resume existing sessions
    - Weave threads with holistic verification
    - Automatic git commits
    - Graceful error handling
    """

    def __init__(
        self,
        backend: Optional[TapestryBackend] = None,
        inspector: Optional[FabricInspector] = None,
        git: Optional[GitIntegration] = None,
        path: str = ".hololoom/tapestry.json"
    ):
        """
        Initialize LoomKeeper.

        Args:
            backend: TapestryBackend for persistence (default: JSON)
            inspector: FabricInspector for verification (default: all signals)
            git: GitIntegration for version control
            path: Path to tapestry file (used if backend is None)
        """
        self.backend = backend or JsonTapestryBackend(path)
        self.inspector = inspector or FabricInspector()
        self.git = git or GitIntegration()
        self.warper = Warper(self.backend, self.git)

    async def __aenter__(self) -> 'LoomKeeper':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Nothing to clean up for now
        pass

    async def start(
        self,
        goal: str,
        threads: Optional[List[str]] = None
    ) -> Tapestry:
        """
        Start new weaving session.

        Args:
            goal: Overall goal for this session
            threads: Optional explicit thread descriptions

        Returns:
            Created Tapestry
        """
        logger.info(f"Starting new session: {goal[:50]}...")
        return await self.warper.setup(goal, threads)

    async def resume(self) -> Optional[Tuple[Tapestry, Optional[Thread]]]:
        """
        Resume existing session.

        Returns:
            (Tapestry, next_thread) or None if no tapestry exists
        """
        tapestry = await self.warper.resume()
        if not tapestry:
            return None

        next_thread = tapestry.next_unwoven()
        return (tapestry, next_thread)

    async def weave_thread(
        self,
        tapestry: Tapestry,
        thread: Thread,
        executor: Callable[[Thread], Awaitable[Any]]
    ) -> Tapestry:
        """
        Execute single thread with verification.

        Process:
        1. Mark thread as weaving
        2. Execute work
        3. Verify with FabricInspector
        4. Commit if passed
        5. Update tapestry

        Args:
            tapestry: Current tapestry
            thread: Thread to weave
            executor: Async function that executes the thread

        Returns:
            Updated Tapestry
        """
        logger.info(f"Weaving thread {thread.id}: {thread.description[:50]}...")

        # Step 1: Mark weaving
        tapestry.update_thread(thread.id, ThreadStatus.WEAVING)
        await self.backend.save(tapestry)

        try:
            # Step 2: Execute
            result = await executor(thread)

            # Step 3: Get changed files for verification
            changed_files = await self.git.get_changed_files()

            # Step 4: Verify (holistic)
            context = {
                "result": result,
                "files": changed_files,
                "thread": thread,
                "tapestry": tapestry
            }
            check = await self.inspector.inspect(thread, context)

            if check.passed:
                # Step 5a: Commit on success
                commit_hash = await self.git.commit(
                    f"tapestry: weave thread {thread.id} - {thread.description[:50]}"
                )
                tapestry.update_thread(
                    thread.id,
                    ThreadStatus.WOVEN,
                    commit_hash=commit_hash,
                    fabric_check=check
                )
                tapestry.current_commit = commit_hash
                logger.info(
                    f"Thread {thread.id} woven successfully "
                    f"(confidence: {check.confidence:.1%})"
                )
            else:
                # Step 5b: Mark tangled on verification failure
                tapestry.update_thread(
                    thread.id,
                    ThreadStatus.TANGLED,
                    fabric_check=check
                )
                logger.warning(
                    f"Thread {thread.id} tangled: {check.blockers}"
                )

        except Exception as e:
            # Step 5c: Mark tangled on execution failure
            logger.error(f"Thread {thread.id} failed: {e}")
            error_check = FabricCheckResult(
                passed=False,
                confidence=0.0,
                blockers=[str(e)],
                recommendations=["Fix the error and retry"]
            )
            tapestry.update_thread(
                thread.id,
                ThreadStatus.TANGLED,
                fabric_check=error_check
            )

        # Always save updated state
        await self.backend.save(tapestry)
        return tapestry

    async def unravel_thread(
        self,
        tapestry: Tapestry,
        thread_id: str
    ) -> Tapestry:
        """
        Unravel (rollback) a thread.

        Uses git to rollback to the previous commit.

        Args:
            tapestry: Current tapestry
            thread_id: Thread to unravel

        Returns:
            Updated Tapestry
        """
        thread = tapestry._get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")

        if thread.status != ThreadStatus.WOVEN:
            logger.warning(f"Thread {thread_id} is not woven, nothing to unravel")
            return tapestry

        # Find previous commit
        if thread.commit_hash and tapestry.initial_commit:
            # Rollback git
            await self.git.rollback(tapestry.initial_commit)

        # Mark as unraveled
        tapestry.update_thread(thread_id, ThreadStatus.UNRAVELED)
        await self.backend.save(tapestry)

        logger.info(f"Unraveled thread {thread_id}")
        return tapestry

    @asynccontextmanager
    async def session(self, goal_or_resume: Optional[str] = None):
        """
        Context manager for scoped sessions.

        Usage:
            # New session
            async with keeper.session("Implement feature X") as ctx:
                while ctx.next_thread:
                    await ctx.weave(my_executor)

            # Resume existing
            async with keeper.session() as ctx:
                while ctx.next_thread:
                    await ctx.weave(my_executor)

        Args:
            goal_or_resume: Goal string for new session, None to resume

        Yields:
            SessionContext with weaving controls
        """
        if goal_or_resume:
            # Start new session
            tapestry = await self.start(goal_or_resume)
        else:
            # Resume existing
            result = await self.resume()
            if not result:
                raise NoTapestryError("No tapestry to resume")
            tapestry, _ = result

        ctx = SessionContext(keeper=self, tapestry=tapestry)

        try:
            yield ctx
        finally:
            # Save final state
            await self.backend.save(ctx.tapestry)
            logger.info(f"Session complete: {ctx.tapestry.get_status_summary()}")

    async def get_status(self) -> Optional[dict]:
        """
        Get current session status.

        Returns:
            Status dict or None if no tapestry
        """
        tapestry = await self.backend.load()
        if not tapestry:
            return None

        return {
            "loom_id": tapestry.loom_id,
            "goal": tapestry.goal,
            "status": tapestry.get_status_summary(),
            "is_complete": tapestry.is_complete(),
            "threads": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.value,
                    "commit": t.commit_hash
                }
                for t in tapestry.threads
            ]
        }

    async def clear(self) -> None:
        """Delete existing tapestry."""
        await self.warper.clear()

    def describe(self) -> str:
        """Get description of keeper configuration."""
        return (
            f"LoomKeeper(\n"
            f"  backend={type(self.backend).__name__},\n"
            f"  inspector={self.inspector.describe()},\n"
            f"  git={self.git.describe()}\n"
            f")"
        )


# Convenience function
async def create_keeper(
    path: str = ".hololoom/tapestry.json",
    enable_git: bool = True
) -> LoomKeeper:
    """
    Create a LoomKeeper with default configuration.

    Args:
        path: Path to tapestry file
        enable_git: Whether to enable git integration

    Returns:
        Configured LoomKeeper
    """
    backend = JsonTapestryBackend(path)
    git = GitIntegration() if enable_git else None
    inspector = FabricInspector()

    return LoomKeeper(
        backend=backend,
        inspector=inspector,
        git=git
    )
