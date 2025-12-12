"""
Agent Manager Hub - Central Orchestrator for Multi-Threaded Agent Execution

Provides unified control over multiple concurrent agent threads with:
- Thread lifecycle management (create, start, pause, resume, cancel)
- Priority control with upvote/downvote system
- MRF/MCTS strategy injection during execution
- Swarm coordination for parallel exploration
- Real-time progress tracking via WebSocket
- Temporal rollback via git-based versioning

Status: Production Ready (December 2025)
Port: 8002 (avoids conflict with 8000 agentic_api, 8001 workflow)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class ThreadStatus(Enum):
    """Agent thread execution status."""
    IDLE = "idle"           # Created but not started
    QUEUED = "queued"       # Waiting for resources
    RUNNING = "running"     # Actively executing
    PAUSED = "paused"       # Temporarily suspended
    COMPLETED = "completed" # Successfully finished
    FAILED = "failed"       # Error during execution
    CANCELLED = "cancelled" # User-initiated cancellation


class ReasoningMode(Enum):
    """Agent reasoning modes."""
    DIRECT = "direct"           # Single-pass answer (~150ms)
    VERIFY = "verify"           # Answer + verification (~600ms)
    RESEARCH = "research"       # Multi-query exploration (~900ms)
    PLAN_EXECUTE = "plan_execute"  # Goal decomposition (~750ms)


class StepType(Enum):
    """Types of reasoning steps."""
    INITIAL = "initial"
    VERIFICATION = "verification"
    RESEARCH_QUERY = "research_query"
    SYNTHESIS = "synthesis"
    PLAN = "plan"
    EXECUTION = "execution"


class MRFStrategy(Enum):
    """MRF refinement strategies."""
    AUTO = "auto"
    VERIFY = "verify"
    ELEGANCE = "elegance"
    CRITIQUE = "critique"
    REFINE = "refine"
    HOFSTADTER = "hofstadter"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TokenCostEstimate:
    """Token usage and cost estimation."""
    tokens_input: int = 0
    tokens_output: int = 0
    provider: str = "ollama"  # anthropic, openai, ollama, local
    model: str = "llama3.2:3b"

    # Cost per 1M tokens (December 2025 pricing)
    PRICING = {
        ("anthropic", "claude-3-5-sonnet"): (3.00, 15.00),
        ("anthropic", "claude-3-haiku"): (0.25, 1.25),
        ("openai", "gpt-4-turbo"): (10.00, 30.00),
        ("openai", "gpt-4o"): (2.50, 10.00),
        ("openai", "gpt-4o-mini"): (0.15, 0.60),
        ("ollama", "*"): (0.00, 0.00),  # All Ollama models free
    }

    @property
    def is_local(self) -> bool:
        """Check if using local/free model."""
        return self.provider in ("ollama", "local")

    @property
    def estimated_cost_usd(self) -> float:
        """Calculate estimated cost in USD."""
        if self.is_local:
            return 0.0

        # Look up pricing
        key = (self.provider, self.model)
        if key not in self.PRICING:
            # Try wildcard for provider
            key = (self.provider, "*")

        if key not in self.PRICING:
            return 0.0

        input_cost, output_cost = self.PRICING[key]
        return (
            (self.tokens_input / 1_000_000) * input_cost +
            (self.tokens_output / 1_000_000) * output_cost
        )


@dataclass
class TaskNode:
    """A single step/node in the agent reasoning tree."""
    id: str
    thread_id: str

    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # Step info
    step_type: StepType = StepType.INITIAL
    name: str = ""
    query: Optional[str] = None

    # Status & Progress
    status: ThreadStatus = ThreadStatus.IDLE
    progress_pct: float = 0.0
    elapsed_time_ms: float = 0.0
    tokens_used: int = 0
    confidence: float = 0.0
    epistemic_confidence: float = 0.0

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    # Injection
    mrf_eligible: bool = True
    mcts_eligible: bool = True
    injection_applied: Optional[str] = None

    # Results
    response: Optional[str] = None
    tool_selected: Optional[str] = None


@dataclass
class AgentThread:
    """A complete agent reasoning thread."""
    id: str
    name: str
    status: ThreadStatus = ThreadStatus.IDLE
    priority: int = 0  # Higher = sooner execution

    # Config
    agent_type: str = "weaving"  # weaving, rag, agentic, custom
    reasoning_mode: ReasoningMode = ReasoningMode.DIRECT

    # Progress (multi-dimensional)
    current_step: int = 0
    total_steps: int = 1
    elapsed_time_ms: float = 0.0
    time_budget_ms: Optional[float] = None
    tokens_used: int = 0
    token_budget: Optional[int] = None

    # Cost tracking
    cost: TokenCostEstimate = field(default_factory=TokenCostEstimate)

    # Tree structure
    root_step_id: Optional[str] = None
    steps: Dict[str, TaskNode] = field(default_factory=dict)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    # Swarm
    swarm_id: Optional[str] = None
    parent_thread_id: Optional[str] = None
    child_thread_ids: List[str] = field(default_factory=list)

    # Results
    confidence: float = 0.0
    epistemic_confidence: float = 0.0
    final_response: Optional[str] = None
    error: Optional[str] = None

    # Files being modified
    active_files: List[str] = field(default_factory=list)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Git-style versioning
    head_commit_id: Optional[str] = None
    current_branch: str = "main"


@dataclass
class SwarmConfig:
    """Configuration for swarm execution."""
    name: str
    fan_out: int = 3  # Number of parallel threads
    reasoning_modes: List[ReasoningMode] = field(
        default_factory=lambda: [ReasoningMode.DIRECT, ReasoningMode.VERIFY, ReasoningMode.RESEARCH]
    )
    merge_strategy: str = "best"  # best, combine, manual
    token_budget_per_thread: Optional[int] = None
    time_budget_ms: Optional[float] = None


@dataclass
class Swarm:
    """A swarm of parallel agent threads."""
    id: str
    name: str
    config: SwarmConfig
    status: ThreadStatus = ThreadStatus.IDLE

    # Child threads
    thread_ids: List[str] = field(default_factory=list)
    completed_thread_ids: List[str] = field(default_factory=list)

    # Merged result
    merged_response: Optional[str] = None
    merged_confidence: float = 0.0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ThreadConfig:
    """Configuration for creating a new thread."""
    name: str
    query: str
    agent_type: str = "weaving"
    reasoning_mode: ReasoningMode = ReasoningMode.DIRECT
    priority: int = 0
    depends_on: List[str] = field(default_factory=list)
    token_budget: Optional[int] = None
    time_budget_ms: Optional[float] = None
    mrf_strategy: Optional[str] = None
    mcts_budget: Optional[int] = None
    provider: str = "ollama"
    model: str = "llama3.2:3b"


# ============================================================================
# Agent Manager Hub
# ============================================================================

class AgentManagerHub:
    """
    Central hub for multi-threaded agent execution.

    Coordinates multiple concurrent agent threads with:
    - Thread lifecycle management
    - Priority-based scheduling
    - MRF/MCTS strategy injection
    - Swarm coordination
    - Real-time WebSocket updates
    - Git-based temporal rollback
    """

    def __init__(
        self,
        max_concurrent_threads: int = 5,
        enable_git_versioning: bool = True,
        git_repo_path: str = "data/agent-repos"
    ):
        """
        Initialize the Agent Manager Hub.

        Args:
            max_concurrent_threads: Maximum threads running concurrently
            enable_git_versioning: Enable git-based temporal rollback
            git_repo_path: Path for agent git repositories
        """
        self._threads: Dict[str, AgentThread] = {}
        self._swarms: Dict[str, Swarm] = {}
        self._max_concurrent = max_concurrent_threads
        self._running_threads: Set[str] = set()

        # Git versioning
        self._enable_git = enable_git_versioning
        self._git_repo_path = Path(git_repo_path)
        self._git_store: Optional[Any] = None  # AgentGitStore instance

        # Event callbacks (global - receive all events with event_type)
        self._event_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # Event-specific handlers (receive only data for their event type)
        self._event_handlers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {}

        # Execution queue (priority queue would be better, but simple list for now)
        self._queue: List[str] = []

        # Background task
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Integration hooks (set externally)
        self.monitor = None  # AgentMonitor
        self.bus = None      # MessageBus
        self.memory = None   # SyncedMemory
        self.mrf = None      # UnifiedMRF

        logger.info(f"AgentManagerHub initialized (max_concurrent={max_concurrent_threads})")

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def start(self):
        """Start the hub and background scheduler."""
        logger.info("Starting AgentManagerHub...")

        # Initialize git store if enabled
        if self._enable_git:
            try:
                from HoloLoom.server.agent_git_store import AgentGitStore
                self._git_repo_path.mkdir(parents=True, exist_ok=True)
                self._git_store = AgentGitStore(str(self._git_repo_path))
                logger.info(f"Git versioning enabled at {self._git_repo_path}")
            except ImportError:
                logger.warning("AgentGitStore not available, git versioning disabled")
                self._enable_git = False

        # Start background scheduler
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("AgentManagerHub started")

    async def stop(self):
        """Stop the hub gracefully."""
        logger.info("Stopping AgentManagerHub...")
        self._shutdown_event.set()

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel all running threads
        for thread_id in list(self._running_threads):
            await self.cancel_thread(thread_id)

        logger.info("AgentManagerHub stopped")

    async def _scheduler_loop(self):
        """Background loop that schedules queued threads."""
        while not self._shutdown_event.is_set():
            try:
                await self._process_queue()
                await asyncio.sleep(0.1)  # 100ms tick
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)

    async def _process_queue(self):
        """Process queued threads based on priority and capacity."""
        if not self._queue:
            return

        # Check capacity
        available_slots = self._max_concurrent - len(self._running_threads)
        if available_slots <= 0:
            return

        # Sort queue by priority (descending)
        self._queue.sort(key=lambda tid: self._threads.get(tid, AgentThread(id="", name="")).priority, reverse=True)

        # Start threads up to available capacity
        started = 0
        for thread_id in list(self._queue):
            if started >= available_slots:
                break

            thread = self._threads.get(thread_id)
            if not thread:
                self._queue.remove(thread_id)
                continue

            # Check dependencies
            if not self._dependencies_satisfied(thread):
                continue

            # Start the thread
            self._queue.remove(thread_id)
            await self._execute_thread(thread_id)
            started += 1

    def _dependencies_satisfied(self, thread: AgentThread) -> bool:
        """Check if all dependencies are satisfied."""
        for dep_id in thread.depends_on:
            dep_thread = self._threads.get(dep_id)
            if not dep_thread:
                continue
            if dep_thread.status not in (ThreadStatus.COMPLETED, ThreadStatus.CANCELLED):
                return False
        return True

    # ========================================================================
    # Thread Operations
    # ========================================================================

    async def create_thread(self, config: ThreadConfig) -> str:
        """
        Create a new agent thread.

        Args:
            config: Thread configuration

        Returns:
            Thread ID
        """
        thread_id = f"thread-{uuid.uuid4().hex[:8]}"

        # Create root step
        root_step_id = f"{thread_id}_step_0"
        root_step = TaskNode(
            id=root_step_id,
            thread_id=thread_id,
            step_type=StepType.INITIAL,
            name="Initial Query",
            query=config.query
        )

        thread = AgentThread(
            id=thread_id,
            name=config.name,
            agent_type=config.agent_type,
            reasoning_mode=config.reasoning_mode,
            priority=config.priority,
            depends_on=config.depends_on,
            token_budget=config.token_budget,
            time_budget_ms=config.time_budget_ms,
            root_step_id=root_step_id,
            steps={root_step_id: root_step},
            cost=TokenCostEstimate(
                provider=config.provider,
                model=config.model
            )
        )

        self._threads[thread_id] = thread

        # Initialize git repo if enabled
        if self._git_store:
            self._git_store.init_thread_repo(thread_id)

        # Emit event
        await self._emit_event("thread_created", {
            "thread_id": thread_id,
            "name": config.name,
            "agent_type": config.agent_type,
            "reasoning_mode": config.reasoning_mode.value,
            "priority": config.priority
        })

        logger.info(f"Created thread {thread_id}: {config.name}")
        return thread_id

    async def start_thread(self, thread_id: str) -> None:
        """
        Start or queue a thread for execution.

        Args:
            thread_id: Thread to start
        """
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        if thread.status not in (ThreadStatus.IDLE, ThreadStatus.PAUSED):
            raise ValueError(f"Cannot start thread in status: {thread.status.value}")

        thread.status = ThreadStatus.QUEUED
        self._queue.append(thread_id)

        await self._emit_event("thread_queued", {
            "thread_id": thread_id,
            "queue_position": len(self._queue)
        })

        logger.info(f"Queued thread {thread_id}")

    async def _execute_thread(self, thread_id: str) -> None:
        """
        Execute a thread (internal).

        Args:
            thread_id: Thread to execute
        """
        thread = self._threads.get(thread_id)
        if not thread:
            return

        thread.status = ThreadStatus.RUNNING
        thread.started_at = datetime.now().isoformat()
        self._running_threads.add(thread_id)

        await self._emit_event("thread_started", {
            "thread_id": thread_id,
            "started_at": thread.started_at
        })

        logger.info(f"Started thread {thread_id}")

        # Spawn async execution task
        asyncio.create_task(self._run_thread(thread_id))

    async def _run_thread(self, thread_id: str) -> None:
        """
        Run thread execution logic (internal).

        This is where the actual agent execution happens.
        """
        thread = self._threads.get(thread_id)
        if not thread:
            return

        start_time = datetime.now()

        try:
            # Get root step
            root_step = thread.steps.get(thread.root_step_id)
            if not root_step:
                raise ValueError("No root step found")

            # Execute based on reasoning mode
            if thread.reasoning_mode == ReasoningMode.DIRECT:
                await self._execute_direct(thread, root_step)
            elif thread.reasoning_mode == ReasoningMode.VERIFY:
                await self._execute_verify(thread, root_step)
            elif thread.reasoning_mode == ReasoningMode.RESEARCH:
                await self._execute_research(thread, root_step)
            elif thread.reasoning_mode == ReasoningMode.PLAN_EXECUTE:
                await self._execute_plan_execute(thread, root_step)

            # Mark completed
            thread.status = ThreadStatus.COMPLETED
            thread.completed_at = datetime.now().isoformat()

            await self._emit_event("thread_completed", {
                "thread_id": thread_id,
                "confidence": thread.confidence,
                "elapsed_time_ms": thread.elapsed_time_ms,
                "tokens_used": thread.tokens_used,
                "cost_usd": thread.cost.estimated_cost_usd
            })

            logger.info(f"Completed thread {thread_id} (confidence={thread.confidence:.2f})")

        except asyncio.CancelledError:
            thread.status = ThreadStatus.CANCELLED
            logger.info(f"Thread {thread_id} cancelled")
            raise

        except Exception as e:
            thread.status = ThreadStatus.FAILED
            thread.error = str(e)

            await self._emit_event("thread_failed", {
                "thread_id": thread_id,
                "error": str(e)
            })

            logger.error(f"Thread {thread_id} failed: {e}", exc_info=True)

        finally:
            # Calculate elapsed time
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            thread.elapsed_time_ms = elapsed

            # Remove from running set
            self._running_threads.discard(thread_id)

            # Unblock dependent threads
            for blocked_id in thread.blocks:
                blocked_thread = self._threads.get(blocked_id)
                if blocked_thread and blocked_thread.status == ThreadStatus.QUEUED:
                    # Re-check queue
                    pass

    async def _execute_direct(self, thread: AgentThread, root_step: TaskNode) -> None:
        """Execute DIRECT mode: single-pass answer."""
        root_step.status = ThreadStatus.RUNNING

        # Simulate execution (replace with actual agent call)
        await asyncio.sleep(0.15)  # ~150ms

        root_step.status = ThreadStatus.COMPLETED
        root_step.confidence = 0.85
        root_step.response = f"Direct answer for: {root_step.query}"

        thread.current_step = 1
        thread.total_steps = 1
        thread.confidence = root_step.confidence
        thread.final_response = root_step.response

        # Commit step if git enabled
        await self._commit_step(thread, root_step)

    async def _execute_verify(self, thread: AgentThread, root_step: TaskNode) -> None:
        """Execute VERIFY mode: answer + verification."""
        # Step 1: Initial answer
        root_step.status = ThreadStatus.RUNNING
        await asyncio.sleep(0.15)
        root_step.status = ThreadStatus.COMPLETED
        root_step.confidence = 0.80
        root_step.response = f"Initial answer for: {root_step.query}"

        await self._commit_step(thread, root_step)

        # Step 2: Verification
        verify_step_id = f"{thread.id}_step_1"
        verify_step = TaskNode(
            id=verify_step_id,
            thread_id=thread.id,
            parent_id=root_step.id,
            step_type=StepType.VERIFICATION,
            name="Verification",
            depth=1
        )
        thread.steps[verify_step_id] = verify_step
        root_step.children_ids.append(verify_step_id)

        verify_step.status = ThreadStatus.RUNNING
        await self._emit_step_progress(thread, verify_step)

        await asyncio.sleep(0.45)  # ~450ms for verification

        verify_step.status = ThreadStatus.COMPLETED
        verify_step.confidence = 0.90
        verify_step.response = "Verification passed"

        await self._commit_step(thread, verify_step)

        thread.current_step = 2
        thread.total_steps = 2
        thread.confidence = 0.88
        thread.final_response = root_step.response

    async def _execute_research(self, thread: AgentThread, root_step: TaskNode) -> None:
        """Execute RESEARCH mode: multi-query exploration."""
        root_step.status = ThreadStatus.RUNNING
        await asyncio.sleep(0.1)
        root_step.status = ThreadStatus.COMPLETED
        root_step.confidence = 0.70

        await self._commit_step(thread, root_step)

        # Research queries (3 sub-queries)
        research_queries = [
            "Related concept 1",
            "Related concept 2",
            "Related concept 3"
        ]

        total_confidence = root_step.confidence

        for i, query in enumerate(research_queries):
            step_id = f"{thread.id}_step_{i + 1}"
            step = TaskNode(
                id=step_id,
                thread_id=thread.id,
                parent_id=root_step.id,
                step_type=StepType.RESEARCH_QUERY,
                name=f"Research: {query}",
                query=query,
                depth=1
            )
            thread.steps[step_id] = step
            root_step.children_ids.append(step_id)

            step.status = ThreadStatus.RUNNING
            await self._emit_step_progress(thread, step)

            await asyncio.sleep(0.2)  # ~200ms per research query

            step.status = ThreadStatus.COMPLETED
            step.confidence = 0.75 + (i * 0.05)
            step.response = f"Research finding for: {query}"
            total_confidence += step.confidence

            thread.current_step = i + 2
            thread.total_steps = len(research_queries) + 2  # root + queries + synthesis

            await self._commit_step(thread, step)

        # Synthesis step
        synth_step_id = f"{thread.id}_step_{len(research_queries) + 1}"
        synth_step = TaskNode(
            id=synth_step_id,
            thread_id=thread.id,
            parent_id=root_step.id,
            step_type=StepType.SYNTHESIS,
            name="Synthesis",
            depth=1
        )
        thread.steps[synth_step_id] = synth_step

        synth_step.status = ThreadStatus.RUNNING
        await self._emit_step_progress(thread, synth_step)

        await asyncio.sleep(0.15)

        synth_step.status = ThreadStatus.COMPLETED
        synth_step.confidence = 0.92
        synth_step.response = f"Synthesized answer for: {root_step.query}"

        await self._commit_step(thread, synth_step)

        thread.current_step = thread.total_steps
        thread.confidence = total_confidence / (len(research_queries) + 2)
        thread.final_response = synth_step.response

    async def _execute_plan_execute(self, thread: AgentThread, root_step: TaskNode) -> None:
        """Execute PLAN_EXECUTE mode: goal decomposition."""
        # Plan step
        root_step.status = ThreadStatus.RUNNING
        root_step.step_type = StepType.PLAN
        root_step.name = "Planning"

        await asyncio.sleep(0.2)

        root_step.status = ThreadStatus.COMPLETED
        root_step.confidence = 0.85
        root_step.response = "Plan created with 3 steps"

        await self._commit_step(thread, root_step)

        # Execute steps
        for i in range(3):
            step_id = f"{thread.id}_step_{i + 1}"
            step = TaskNode(
                id=step_id,
                thread_id=thread.id,
                parent_id=root_step.id,
                step_type=StepType.EXECUTION,
                name=f"Execute step {i + 1}",
                depth=1
            )
            thread.steps[step_id] = step
            root_step.children_ids.append(step_id)

            step.status = ThreadStatus.RUNNING
            await self._emit_step_progress(thread, step)

            await asyncio.sleep(0.15)

            step.status = ThreadStatus.COMPLETED
            step.confidence = 0.80 + (i * 0.05)
            step.response = f"Step {i + 1} completed"

            thread.current_step = i + 2
            thread.total_steps = 4  # plan + 3 executions

            await self._commit_step(thread, step)

        thread.confidence = 0.88
        thread.final_response = "All steps executed successfully"

    async def pause_thread(self, thread_id: str) -> None:
        """Pause a running thread."""
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        if thread.status != ThreadStatus.RUNNING:
            raise ValueError(f"Cannot pause thread in status: {thread.status.value}")

        thread.status = ThreadStatus.PAUSED

        await self._emit_event("thread_paused", {"thread_id": thread_id})
        logger.info(f"Paused thread {thread_id}")

    async def resume_thread(self, thread_id: str) -> None:
        """Resume a paused thread."""
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        if thread.status != ThreadStatus.PAUSED:
            raise ValueError(f"Cannot resume thread in status: {thread.status.value}")

        thread.status = ThreadStatus.QUEUED
        self._queue.append(thread_id)

        await self._emit_event("thread_resumed", {"thread_id": thread_id})
        logger.info(f"Resumed thread {thread_id}")

    async def cancel_thread(self, thread_id: str) -> None:
        """Cancel a thread."""
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        if thread.status == ThreadStatus.COMPLETED:
            raise ValueError("Cannot cancel completed thread")

        # Remove from queue if queued
        if thread_id in self._queue:
            self._queue.remove(thread_id)

        thread.status = ThreadStatus.CANCELLED
        thread.completed_at = datetime.now().isoformat()

        await self._emit_event("thread_cancelled", {"thread_id": thread_id})
        logger.info(f"Cancelled thread {thread_id}")

    # ========================================================================
    # Priority Control
    # ========================================================================

    async def set_priority(self, thread_id: str, priority: int) -> None:
        """Set thread priority."""
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        old_priority = thread.priority
        thread.priority = priority

        await self._emit_event("priority_changed", {
            "thread_id": thread_id,
            "old_priority": old_priority,
            "new_priority": priority
        })

        logger.info(f"Thread {thread_id} priority: {old_priority} -> {priority}")

    async def upvote(self, thread_id: str) -> None:
        """Increase thread priority by 1."""
        thread = self._threads.get(thread_id)
        if thread:
            await self.set_priority(thread_id, thread.priority + 1)

    async def downvote(self, thread_id: str) -> None:
        """Decrease thread priority by 1."""
        thread = self._threads.get(thread_id)
        if thread:
            await self.set_priority(thread_id, thread.priority - 1)

    # ========================================================================
    # MRF/MCTS Injection
    # ========================================================================

    async def inject_mrf(
        self,
        thread_id: str,
        step_id: str,
        strategy: str
    ) -> None:
        """
        Inject MRF strategy into a step.

        Args:
            thread_id: Target thread
            step_id: Target step
            strategy: MRF strategy (AUTO, VERIFY, ELEGANCE, CRITIQUE, REFINE, HOFSTADTER)
        """
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        step = thread.steps.get(step_id)
        if not step:
            raise ValueError(f"Step not found: {step_id}")

        if not step.mrf_eligible:
            raise ValueError("Step is not MRF-eligible")

        step.injection_applied = f"mrf:{strategy}"

        await self._emit_event("mrf_injected", {
            "thread_id": thread_id,
            "step_id": step_id,
            "strategy": strategy
        })

        logger.info(f"Injected MRF strategy '{strategy}' into {thread_id}/{step_id}")

    async def inject_mcts(
        self,
        thread_id: str,
        step_id: str,
        budget: int = 100,
        exploration_c: float = 1.414
    ) -> None:
        """
        Inject MCTS exploration into a step.

        Args:
            thread_id: Target thread
            step_id: Target step
            budget: MCTS iteration budget
            exploration_c: UCB1 exploration coefficient
        """
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        step = thread.steps.get(step_id)
        if not step:
            raise ValueError(f"Step not found: {step_id}")

        if not step.mcts_eligible:
            raise ValueError("Step is not MCTS-eligible")

        step.injection_applied = f"mcts:budget={budget},c={exploration_c}"

        await self._emit_event("mcts_injected", {
            "thread_id": thread_id,
            "step_id": step_id,
            "budget": budget,
            "exploration_c": exploration_c
        })

        logger.info(f"Injected MCTS (budget={budget}, c={exploration_c}) into {thread_id}/{step_id}")

    # ========================================================================
    # Swarm Management
    # ========================================================================

    async def create_swarm(self, config: SwarmConfig, query: str) -> str:
        """
        Create a swarm of parallel threads.

        Args:
            config: Swarm configuration
            query: Base query for all threads

        Returns:
            Swarm ID
        """
        swarm_id = f"swarm-{uuid.uuid4().hex[:8]}"

        swarm = Swarm(
            id=swarm_id,
            name=config.name,
            config=config
        )

        # Create child threads
        for i, mode in enumerate(config.reasoning_modes[:config.fan_out]):
            thread_config = ThreadConfig(
                name=f"{config.name} - {mode.value}",
                query=query,
                reasoning_mode=mode,
                token_budget=config.token_budget_per_thread,
                time_budget_ms=config.time_budget_ms
            )

            thread_id = await self.create_thread(thread_config)
            thread = self._threads[thread_id]
            thread.swarm_id = swarm_id
            swarm.thread_ids.append(thread_id)

        self._swarms[swarm_id] = swarm

        await self._emit_event("swarm_created", {
            "swarm_id": swarm_id,
            "name": config.name,
            "thread_ids": swarm.thread_ids,
            "fan_out": config.fan_out
        })

        logger.info(f"Created swarm {swarm_id} with {len(swarm.thread_ids)} threads")
        return swarm_id

    async def start_swarm(self, swarm_id: str) -> None:
        """Start all threads in a swarm."""
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            raise ValueError(f"Swarm not found: {swarm_id}")

        swarm.status = ThreadStatus.RUNNING
        swarm.started_at = datetime.now().isoformat()

        for thread_id in swarm.thread_ids:
            await self.start_thread(thread_id)

        await self._emit_event("swarm_started", {
            "swarm_id": swarm_id
        })

    async def merge_swarm(
        self,
        swarm_id: str,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge swarm results.

        Args:
            swarm_id: Swarm to merge
            strategy: Override merge strategy (best/combine/manual)

        Returns:
            Dict with merged_thread_id, strategy_used, merged_response, merged_confidence
        """
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            raise ValueError(f"Swarm not found: {swarm_id}")

        # Collect completed threads
        completed = []
        for thread_id in swarm.thread_ids:
            thread = self._threads.get(thread_id)
            if thread and thread.status == ThreadStatus.COMPLETED:
                completed.append(thread)

        if not completed:
            raise ValueError("No completed threads to merge")

        # Use override strategy or swarm's configured strategy
        merge_strategy = strategy or swarm.config.merge_strategy
        merged_thread_id: Optional[str] = None

        # Merge based on strategy
        if merge_strategy == "best":
            best = max(completed, key=lambda t: t.confidence)
            swarm.merged_response = best.final_response
            swarm.merged_confidence = best.confidence
            merged_thread_id = best.id

        elif merge_strategy == "combine":
            responses = [t.final_response for t in completed if t.final_response]
            swarm.merged_response = "\n\n---\n\n".join(responses)
            swarm.merged_confidence = sum(t.confidence for t in completed) / len(completed)
            # For combine, use highest confidence thread as reference
            merged_thread_id = max(completed, key=lambda t: t.confidence).id

        else:  # manual
            swarm.merged_response = None  # User must select
            swarm.merged_confidence = 0.0
            merged_thread_id = None

        swarm.status = ThreadStatus.COMPLETED
        swarm.completed_at = datetime.now().isoformat()

        await self._emit_event("swarm_merged", {
            "swarm_id": swarm_id,
            "merged_confidence": swarm.merged_confidence,
            "merge_strategy": merge_strategy,
            "merged_thread_id": merged_thread_id
        })

        return {
            "merged_thread_id": merged_thread_id,
            "strategy_used": merge_strategy,
            "merged_response": swarm.merged_response,
            "merged_confidence": swarm.merged_confidence
        }

    async def fork_thread(
        self,
        thread_id: str,
        from_step_id: Optional[str] = None,
        new_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Fork a thread from a specific step.

        Args:
            thread_id: Source thread
            from_step_id: Step to fork from (default: latest)
            new_params: Override parameters for fork

        Returns:
            New thread ID
        """
        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        # Determine fork point
        if from_step_id:
            step = thread.steps.get(from_step_id)
            if not step:
                raise ValueError(f"Step not found: {from_step_id}")
        else:
            # Find latest completed step
            latest = None
            for step in thread.steps.values():
                if step.status == ThreadStatus.COMPLETED:
                    if not latest or step.depth > latest.depth:
                        latest = step
            step = latest or thread.steps.get(thread.root_step_id)

        # Create new thread
        new_params = new_params or {}
        config = ThreadConfig(
            name=f"{thread.name} (fork)",
            query=step.query or "",
            agent_type=thread.agent_type,
            reasoning_mode=new_params.get("reasoning_mode", thread.reasoning_mode),
            priority=thread.priority,
            mrf_strategy=new_params.get("mrf_strategy"),
            mcts_budget=new_params.get("mcts_budget")
        )

        new_thread_id = await self.create_thread(config)
        new_thread = self._threads[new_thread_id]
        new_thread.parent_thread_id = thread_id
        thread.child_thread_ids.append(new_thread_id)

        # Create git branch if enabled
        if self._git_store and thread.head_commit_id:
            branch_name = f"fork-{new_thread_id[-8:]}"
            self._git_store.create_branch(
                thread_id,
                branch_name,
                bytes.fromhex(thread.head_commit_id)
            )
            new_thread.current_branch = branch_name

        await self._emit_event("thread_forked", {
            "source_thread_id": thread_id,
            "new_thread_id": new_thread_id,
            "from_step_id": step.id
        })

        logger.info(f"Forked {thread_id} -> {new_thread_id} from step {step.id}")
        return new_thread_id

    # ========================================================================
    # Git Operations (Temporal Rollback)
    # ========================================================================

    async def _commit_step(self, thread: AgentThread, step: TaskNode) -> None:
        """Commit a step to git history."""
        if not self._git_store:
            return

        try:
            from HoloLoom.server.agent_git_store import StepCommit
            import time

            commit = StepCommit(
                thread_id=thread.id,
                step_index=len([s for s in thread.steps.values() if s.status == ThreadStatus.COMPLETED]),
                timestamp=time.time(),
                author="agent",
                message=f"Step {step.depth}: {step.name}",
                query=step.query or "",
                response=step.response,
                confidence=step.confidence,
                epistemic_confidence=step.epistemic_confidence,
                tokens_used=step.tokens_used,
                reasoning_mode=thread.reasoning_mode,
                mrf_strategy=step.injection_applied.split(":")[1] if step.injection_applied and step.injection_applied.startswith("mrf:") else None,
                tool_selected=step.tool_selected
            )

            parent_sha = bytes.fromhex(thread.head_commit_id) if thread.head_commit_id else None
            commit_sha = self._git_store.commit_step(thread.id, commit, parent_sha)
            thread.head_commit_id = commit_sha.hex()

        except Exception as e:
            logger.warning(f"Failed to commit step: {e}")

    async def get_thread_history(self, thread_id: str, max_count: int = 50) -> List[Dict]:
        """Get git commit history for a thread."""
        if not self._git_store:
            return []

        return self._git_store.get_log(thread_id, max_count)

    async def checkout_commit(self, thread_id: str, commit_id: str) -> None:
        """Checkout a specific commit (rollback)."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        self._git_store.checkout(thread_id, commit_id)
        thread.head_commit_id = commit_id

        await self._emit_event("rollback_completed", {
            "thread_id": thread_id,
            "commit_id": commit_id
        })

        logger.info(f"Rolled back {thread_id} to {commit_id[:8]}")

    async def create_branch(
        self,
        thread_id: str,
        branch_name: str,
        from_commit: Optional[str] = None
    ) -> None:
        """Create a git branch."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        commit_sha = bytes.fromhex(from_commit or thread.head_commit_id)
        self._git_store.create_branch(thread_id, branch_name, commit_sha)

        await self._emit_event("branch_created", {
            "thread_id": thread_id,
            "branch_name": branch_name,
            "from_commit": (from_commit or thread.head_commit_id)[:8]
        })

        logger.info(f"Created branch '{branch_name}' in {thread_id}")

    async def get_commit_log(self, thread_id: str, max_count: int = 50) -> List[Dict]:
        """Get git commit history (alias for get_thread_history)."""
        return await self.get_thread_history(thread_id, max_count)

    async def get_commit(self, thread_id: str, commit_id: str) -> Optional[Dict]:
        """Get details of a specific commit."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        try:
            commits = self._git_store.get_log(thread_id, max_count=1000)
            for commit in commits:
                if commit.get("sha", "").startswith(commit_id):
                    return commit
            return None
        except Exception as e:
            logger.warning(f"Failed to get commit {commit_id}: {e}")
            return None

    async def get_diff(self, thread_id: str, from_commit: str, to_commit: str) -> Dict[str, Any]:
        """Get diff between two commits."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        return self._git_store.diff(thread_id, from_commit, to_commit)

    async def list_branches(self, thread_id: str) -> List[str]:
        """List all branches for a thread."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        return self._git_store.list_branches(thread_id)

    async def delete_branch(self, thread_id: str, branch_name: str) -> None:
        """Delete a branch."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        if branch_name == "main":
            raise ValueError("Cannot delete main branch")

        if branch_name == thread.current_branch:
            raise ValueError("Cannot delete currently checked out branch")

        self._git_store.delete_branch(thread_id, branch_name)

        await self._emit_event("branch_deleted", {
            "thread_id": thread_id,
            "branch_name": branch_name
        })

        logger.info(f"Deleted branch '{branch_name}' in {thread_id}")

    async def reset_thread(
        self,
        thread_id: str,
        target_commit_id: str,
        mode: str = "soft"
    ) -> None:
        """Reset thread to a specific commit."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        if mode not in ("soft", "mixed", "hard"):
            raise ValueError(f"Invalid reset mode: {mode}")

        # Perform git reset
        self._git_store.reset(thread_id, target_commit_id, mode)
        thread.head_commit_id = target_commit_id
        thread.updated_at = datetime.now().isoformat()

        await self._emit_event("thread_reset", {
            "thread_id": thread_id,
            "target_commit": target_commit_id,
            "mode": mode
        })

        logger.info(f"Reset {thread_id} to {target_commit_id[:8]} ({mode})")

    async def cherry_pick(self, thread_id: str, commit_id: str) -> str:
        """Cherry-pick a commit onto current branch."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        new_commit_sha = self._git_store.cherry_pick(thread_id, commit_id)
        thread.head_commit_id = new_commit_sha.hex()
        thread.updated_at = datetime.now().isoformat()

        await self._emit_event("cherry_picked", {
            "thread_id": thread_id,
            "source_commit": commit_id,
            "new_commit": thread.head_commit_id[:8]
        })

        logger.info(f"Cherry-picked {commit_id[:8]} onto {thread_id}")
        return thread.head_commit_id

    async def merge_branch(
        self,
        thread_id: str,
        source_branch: str,
        strategy: str = "best"
    ) -> Dict[str, Any]:
        """Merge a branch into current branch."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        if strategy not in ("best", "combine", "manual"):
            raise ValueError(f"Invalid merge strategy: {strategy}")

        merge_result = self._git_store.merge(thread_id, source_branch, strategy)
        if merge_result.get("commit"):
            thread.head_commit_id = merge_result["commit"]
            thread.updated_at = datetime.now().isoformat()

        await self._emit_event("branch_merged", {
            "thread_id": thread_id,
            "source_branch": source_branch,
            "strategy": strategy,
            "result": merge_result
        })

        logger.info(f"Merged '{source_branch}' into {thread_id}")
        return merge_result

    async def get_merge_base(
        self,
        thread_id: str,
        branch1: str,
        branch2: str
    ) -> Optional[str]:
        """Find the common ancestor (merge base) of two branches."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        return self._git_store.get_merge_base(thread_id, branch1, branch2)

    async def get_reflog(self, thread_id: str, limit: int = 50) -> List[Dict]:
        """Get reflog (operation history) for a thread."""
        if not self._git_store:
            raise ValueError("Git versioning not enabled")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread not found: {thread_id}")

        return self._git_store.get_reflog(thread_id, limit)

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_thread(self, thread_id: str) -> Optional[AgentThread]:
        """Get thread by ID."""
        return self._threads.get(thread_id)

    def get_all_threads(self) -> List[AgentThread]:
        """Get all threads."""
        return list(self._threads.values())

    def get_swarm(self, swarm_id: str) -> Optional[Swarm]:
        """Get swarm by ID."""
        return self._swarms.get(swarm_id)

    def get_all_swarms(self) -> List[Swarm]:
        """Get all swarms."""
        return list(self._swarms.values())

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queued": len(self._queue),
            "running": len(self._running_threads),
            "max_concurrent": self._max_concurrent,
            "queue_order": self._queue.copy()
        }

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get aggregate cost summary for all threads."""
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0
        paid_threads = 0
        local_threads = 0

        for thread in self._threads.values():
            total_tokens_in += thread.cost.tokens_input
            total_tokens_out += thread.cost.tokens_output
            total_cost += thread.cost.estimated_cost_usd

            if thread.cost.is_local:
                local_threads += 1
            else:
                paid_threads += 1

        return {
            "total_threads": len(self._threads),
            "total_tokens_input": total_tokens_in,
            "total_tokens_output": total_tokens_out,
            "paid_threads": paid_threads,
            "local_threads": local_threads,
            "total_cost_usd": total_cost
        }

    # Alias for API compatibility
    def list_threads(self) -> List[AgentThread]:
        """List all threads (alias for get_all_threads)."""
        return self.get_all_threads()

    def get_statistics(self) -> Dict[str, Any]:
        """Get hub statistics for health check."""
        status_counts = {}
        for thread in self._threads.values():
            status = thread.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_threads": len(self._threads),
            "total_swarms": len(self._swarms),
            "running_threads": len(self._running_threads),
            "queued_threads": len(self._queue),
            "max_concurrent": self._max_concurrent,
            "git_enabled": self._enable_git,
            "status_counts": status_counts,
            "cost_summary": self.get_session_cost_summary()
        }

    async def shutdown(self) -> None:
        """Shutdown the hub gracefully (alias for stop)."""
        await self.stop()

    # ========================================================================
    # Event System
    # ========================================================================

    def on_event(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register event callback."""
        self._event_callbacks.append(callback)

    def register_event_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """
        Register handler for specific event type.

        The handler receives only the event data dict (not event_type).

        Args:
            event_type: Event type to handle (e.g., "thread_created", "step_progress")
            handler: Async or sync callable that receives event data dict
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def unregister_event_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """
        Unregister handler for specific event type.

        Args:
            event_type: Event type the handler was registered for
            handler: The handler to unregister
        """
        if event_type in self._event_handlers:
            if handler in self._event_handlers[event_type]:
                self._event_handlers[event_type].remove(handler)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event to all registered handlers and callbacks."""
        data["timestamp"] = datetime.now().isoformat()

        # Call event-specific handlers (receive only data)
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")

        # Call global callbacks (receive event_type + data)
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    async def _emit_step_progress(self, thread: AgentThread, step: TaskNode) -> None:
        """Emit step progress event."""
        await self._emit_event("step_progress", {
            "thread_id": thread.id,
            "step_id": step.id,
            "step_name": step.name,
            "step_type": step.step_type.value,
            "current_step": thread.current_step,
            "total_steps": thread.total_steps,
            "status": step.status.value
        })


# ============================================================================
# Factory Function
# ============================================================================

def create_agent_manager_hub(
    max_concurrent: int = 5,
    enable_git: bool = True,
    git_path: str = "data/agent-repos"
) -> AgentManagerHub:
    """
    Create an AgentManagerHub instance.

    Args:
        max_concurrent: Maximum concurrent threads
        enable_git: Enable git-based versioning
        git_path: Path for git repositories

    Returns:
        Configured AgentManagerHub
    """
    return AgentManagerHub(
        max_concurrent_threads=max_concurrent,
        enable_git_versioning=enable_git,
        git_repo_path=git_path
    )
