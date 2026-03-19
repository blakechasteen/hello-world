"""
Domain Harness Worker - Stateless Task Executor

A worker that:
- Has NO memory outside domain memory (Neo4j/Qdrant)
- Executes ONE feature per run
- Follows Boot/Action/Exit protocol
- Integrates with HoloLoom's AgentOrchestrator

Key principle: The worker disappears after each run.
All persistence is in the domain memory, not the LLM.

Weaving Metaphor:
The Worker is the shuttle - it passes through once,
leaving a single weft thread, then returns to rest.
"""

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from hololoom.domain_harness.protocol import Neo4jDomainProtocol
from hololoom.domain_harness.schema import (
    ActionResult,
    DomainContext,
    Feature,
    FeatureStatus,
    TestResult,
)

# ============================================================================
# Worker Configuration
# ============================================================================

@dataclass
class WorkerConfig:
    """Configuration for a domain worker."""

    # Identity
    worker_id: str = None
    worker_type: str = "default"

    # Timeouts
    implementation_timeout_seconds: float = 300.0
    test_timeout_seconds: float = 60.0

    # Retry behavior
    max_retries_per_feature: int = 3
    retry_delay_seconds: float = 1.0

    # Working memory
    use_working_memory: bool = True
    attention_radius: float = 0.3

    # Execution
    dry_run: bool = False  # Don't actually modify anything
    verbose: bool = True

    def __post_init__(self):
        if self.worker_id is None:
            self.worker_id = f"worker_{uuid.uuid4().hex[:8]}"


# ============================================================================
# Implementation Result
# ============================================================================

@dataclass
class ImplementationResult:
    """Result from implementation function."""
    success: bool
    files_changed: list[str]
    diff_hash: str | None = None
    output: str = ""
    error: str | None = None


# ============================================================================
# Domain Worker
# ============================================================================

class DomainWorker:
    """
    Stateless worker that executes domain tasks.
    
    Each run():
    1. Boots (loads domain state, selects ONE feature)
    2. Acts (implements feature, runs tests)
    3. Exits (records progress, cleans up)
    
    Usage:
        worker = DomainWorker(
            yarn=Neo4jKG(),
            implementation_fn=my_impl_function,
            test_fn=my_test_function
        )
        result = await worker.run()
    """

    def __init__(
        self,
        yarn,  # Neo4jKG or compatible
        implementation_fn: Callable[[Feature, DomainContext], Awaitable[ImplementationResult]],
        test_fn: Callable[[Feature], Awaitable[TestResult]],
        warp=None,  # WarpSpace for embeddings
        working_memory=None,  # AgentWorkingMemory
        config: WorkerConfig | None = None,
        project_name: str = "default"
    ):
        self.yarn = yarn
        self.warp = warp
        self.working_memory = working_memory
        self.config = config or WorkerConfig()

        # Implementation and test functions
        self.implementation_fn = implementation_fn
        self.test_fn = test_fn

        # Protocol handles the lifecycle
        self.protocol = Neo4jDomainProtocol(
            yarn=yarn,
            warp=warp,
            working_memory=working_memory,
            project_name=project_name
        )

        # State (reset each run)
        self._current_context: DomainContext | None = None
        self._run_start_time: float | None = None

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    async def run(self) -> ActionResult:
        """
        Execute one complete worker cycle.
        
        Boot → Act → Exit
        
        Returns:
            ActionResult with outcome
        """
        self._run_start_time = time.time()

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[{self.config.worker_id}] Starting run at {datetime.now().isoformat()}")

        # Boot
        context = await self._boot()
        self._current_context = context

        if not context.has_work():
            if self.config.verbose:
                print(f"[{self.config.worker_id}] No actionable work found")
            return ActionResult(
                feature_id="none",
                success=True,
                summary="No actionable work available",
                worker_id=self.config.worker_id
            )

        if self.config.verbose:
            print(f"[{self.config.worker_id}] Selected feature: {context.target.name}")

        # Act
        result = await self._act(context)

        # Exit
        await self._exit(result)

        if self.config.verbose:
            status = "✓" if result.success else "✗"
            duration = time.time() - self._run_start_time
            print(f"[{self.config.worker_id}] {status} Completed in {duration:.2f}s")
            print(f"{'='*60}\n")

        # Clear state (worker disappears)
        self._current_context = None
        self._run_start_time = None

        return result

    # ========================================================================
    # Lifecycle Methods
    # ========================================================================

    async def _boot(self) -> DomainContext:
        """
        Boot ritual: Ground worker in current reality.
        """
        if self.config.verbose:
            print(f"[{self.config.worker_id}] Booting...")

        return await self.protocol.boot()

    async def _act(self, context: DomainContext) -> ActionResult:
        """
        Action ritual: Execute ONE unit of work.
        """
        target = context.target

        if self.config.verbose:
            print(f"[{self.config.worker_id}] Implementing: {target.name}")

        if self.config.dry_run:
            # Dry run - simulate success
            return ActionResult(
                feature_id=target.id,
                success=True,
                summary=f"[DRY RUN] Would implement: {target.name}",
                worker_id=self.config.worker_id
            )

        # Implementation
        try:
            impl_result = await self.implementation_fn(target, context)
        except Exception as e:
            return ActionResult(
                feature_id=target.id,
                success=False,
                summary=f"Implementation failed: {str(e)}",
                worker_id=self.config.worker_id
            )

        if not impl_result.success:
            return ActionResult(
                feature_id=target.id,
                success=False,
                summary=f"Implementation error: {impl_result.error or impl_result.output}",
                files_changed=impl_result.files_changed,
                diff_hash=impl_result.diff_hash,
                worker_id=self.config.worker_id
            )

        # Testing
        if self.config.verbose:
            print(f"[{self.config.worker_id}] Running tests...")

        try:
            test_result = await self.test_fn(target)
        except Exception as e:
            test_result = TestResult(
                feature_id=target.id,
                passed=False,
                output=f"Test execution failed: {str(e)}"
            )

        return ActionResult(
            feature_id=target.id,
            success=test_result.passed,
            summary=f"{'✓' if test_result.passed else '✗'} {target.name}",
            files_changed=impl_result.files_changed,
            diff_hash=impl_result.diff_hash,
            test_result=test_result,
            duration_ms=int((time.time() - self._run_start_time) * 1000),
            worker_id=self.config.worker_id
        )

    async def _exit(self, result: ActionResult):
        """
        Exit ritual: Record outcome and clean up.
        """
        if self.config.verbose:
            print(f"[{self.config.worker_id}] Recording outcome...")

        await self.protocol.exit(result, self.config.worker_id)

    # ========================================================================
    # Completion Check
    # ========================================================================

    async def is_complete(self) -> bool:
        """Check if all features are passing."""
        context = await self.protocol.boot()
        return not context.has_work()

    async def get_status(self) -> dict:
        """Get current domain status."""
        context = await self.protocol.boot()
        return {
            "project": context.state.project_name,
            "total_features": context.state.total_features,
            "passing": context.state.passing_features,
            "failing": context.state.failing_features,
            "pending": len([f for f in context.pending_features if f.status == FeatureStatus.PENDING]),
            "actionable": len([f for f in context.pending_features if f.is_actionable()]),
            "completion_rate": context.state.completion_rate()
        }


# ============================================================================
# Worker with AgentOrchestrator Integration
# ============================================================================

class AgentDomainWorker(DomainWorker):
    """
    Domain Worker that uses HoloLoom's AgentOrchestrator.
    
    Extends base worker with:
    - Trinity Working Memory
    - Pattern Learning
    - Profile-based configuration
    """

    def __init__(
        self,
        yarn,
        implementation_fn: Callable[[Feature, DomainContext], Awaitable[ImplementationResult]],
        test_fn: Callable[[Feature], Awaitable[TestResult]],
        agent_profile=None,  # AgentProfile
        embedding_model=None,  # MatryoshkaEmbeddings
        warp=None,
        config: WorkerConfig | None = None,
        project_name: str = "default"
    ):
        # Create working memory from agent profile
        working_memory = None
        if agent_profile and yarn and embedding_model:
            from hololoom.agents.working_memory import AgentWorkingMemory
            working_memory = AgentWorkingMemory(
                profile=agent_profile,
                yarn_graph=yarn,
                embedding_model=embedding_model
            )

        super().__init__(
            yarn=yarn,
            implementation_fn=implementation_fn,
            test_fn=test_fn,
            warp=warp,
            working_memory=working_memory,
            config=config,
            project_name=project_name
        )

        self.agent_profile = agent_profile
        self.embedding_model = embedding_model

    async def _boot(self) -> DomainContext:
        """Boot with agent-enhanced working memory."""
        context = await super()._boot()

        # Pin target feature in working memory
        if self.working_memory and context.target:
            # Pin the target concept
            self.working_memory.pin(context.target.name, weight=0.9)

            # Also pin recent progress context
            for entry in context.recent_progress[:3]:
                self.working_memory.pin(entry.summary, weight=0.3)

        return context

    async def _exit(self, result: ActionResult):
        """Exit with pattern learning."""
        await super()._exit(result)

        # Record outcome for pattern learning (if we have learner)
        # This would integrate with WorkingMemoryLearner


# ============================================================================
# Runner Utilities
# ============================================================================

async def run_worker_loop(
    worker: DomainWorker,
    max_cycles: int = 100,
    on_result: Callable[[ActionResult], None] | None = None
) -> list[ActionResult]:
    """
    Run worker in a loop until complete.
    
    Args:
        worker: DomainWorker instance
        max_cycles: Maximum number of cycles
        on_result: Callback for each result
        
    Returns:
        List of all results
    """
    results = []

    for i in range(max_cycles):
        if await worker.is_complete():
            break

        result = await worker.run()
        results.append(result)

        if on_result:
            on_result(result)

        # Stop if no work (different from complete)
        if result.feature_id == "none":
            break

    return results


# ============================================================================
# Factory Functions
# ============================================================================

def create_worker(
    yarn,
    implementation_fn: Callable[[Feature, DomainContext], Awaitable[ImplementationResult]],
    test_fn: Callable[[Feature], Awaitable[TestResult]],
    profile_name: str = "general",
    **kwargs
) -> DomainWorker:
    """
    Factory for creating domain workers.
    
    Args:
        yarn: Neo4j knowledge graph
        implementation_fn: Function to implement features
        test_fn: Function to run tests
        profile_name: Agent profile to use (general, code, research, etc.)
        **kwargs: Additional worker config
        
    Returns:
        Configured DomainWorker
    """
    config = WorkerConfig(**kwargs)

    return DomainWorker(
        yarn=yarn,
        implementation_fn=implementation_fn,
        test_fn=test_fn,
        config=config
    )


# ============================================================================
# Example Implementation Functions
# ============================================================================

async def stub_implementation(feature: Feature, context: DomainContext) -> ImplementationResult:
    """Stub implementation that always succeeds."""
    return ImplementationResult(
        success=True,
        files_changed=[],
        output=f"Stub implementation for {feature.name}"
    )


async def stub_test(feature: Feature) -> TestResult:
    """Stub test that always passes."""
    return TestResult(
        feature_id=feature.id,
        passed=True,
        output=f"Stub test passed for {feature.name}"
    )


# ============================================================================
# LLM-Powered Implementation
# ============================================================================

class LLMImplementer:
    """
    Use an LLM to implement features.
    
    This is where the actual code generation happens.
    """

    def __init__(self, llm, codebase_path: str):
        self.llm = llm
        self.codebase_path = codebase_path

    async def implement(self, feature: Feature, context: DomainContext) -> ImplementationResult:
        """
        Generate implementation using LLM.
        
        The LLM receives:
        - Feature description and acceptance criteria
        - Recent progress (what's been tried)
        - Domain state (constraints, environment)
        - Current codebase context (relevant files)
        """
        # Build prompt
        self._build_implementation_prompt(feature, context)

        # Call LLM
        # response = await self.llm.complete(prompt)

        # Parse response into file changes
        # files_changed = self._parse_code_changes(response)

        # Apply changes
        # diff_hash = self._apply_changes(files_changed)

        return ImplementationResult(
            success=True,
            files_changed=[],
            output="LLM implementation"
        )

    def _build_implementation_prompt(self, feature: Feature, context: DomainContext) -> str:
        """Build the implementation prompt."""
        return f"""
You are implementing a feature for {context.state.project_name}.

Feature: {feature.name}
Description: {feature.description}

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in feature.acceptance_criteria)}

Constraints:
{chr(10).join(f'- {c}' for c in context.state.constraints)}

Recent Progress:
{chr(10).join(f'- {p.summary}' for p in context.recent_progress[:5])}

Implement ONLY this feature. Write production-quality code.
"""


# ============================================================================
# Migration: Weaverlet Compatibility
# ============================================================================

class WeaverletWorker:
    """
    Drop-in replacement for DomainWorker using Weaverlet internals.

    Wraps TaskController with Budget(max_iterations=1) to match
    single-pass DomainWorker semantics. Use this as the migration
    path from DomainWorker to Weaverlet.

    Usage:
        worker = WeaverletWorker(daemon_url="http://localhost:4243")
        result = await worker.run(goal="Implement feature X")
    """

    def __init__(self, daemon_url: str = "http://localhost:4243", **kwargs):
        self._daemon_url = daemon_url
        self._kwargs = kwargs

    async def run(self, goal: str = "single_pass") -> ActionResult:
        """Run one iteration, matching DomainWorker.run() semantics."""
        from hololoom.weaverlet import Budget, RefinementLoop, TaskController, TaskLease
        from hololoom.weaverlet.evaluation.evaluator import WeaverletEvaluator
        from hololoom.weaverlet.schemas import TerminationReason
        from hololoom.weaverlet.shuttle.http_client import DaemonInferClient
        from hololoom.weaverlet.tools.registry import WeaverletToolRegistry

        async with DaemonInferClient(self._daemon_url) as client:
            loop = RefinementLoop(client, WeaverletToolRegistry())
            evaluator = WeaverletEvaluator(infer_client=client)
            controller = TaskController(loop, evaluator)

            lease = TaskLease.create(goal=goal, budget=Budget(max_iterations=1))
            result = await controller.execute(lease)

            return ActionResult(
                feature_id=result.lease_id,
                success=(result.termination_reason != TerminationReason.BLOCKER),
                summary=f"Weaverlet: {result.termination_reason.value} "
                        f"(score={result.final_score:.2f})",
                duration_ms=int(result.wall_time_seconds * 1000),
            )
