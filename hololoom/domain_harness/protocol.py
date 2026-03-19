"""
Domain Harness Protocol - Worker Lifecycle Rituals

Implements the Boot/Action/Exit pattern for stateless workers:
- Boot: Ground worker in current reality
- Action: Execute one unit of work
- Exit: Record outcome and clean up

The protocol ensures workers have no hidden state - 
all knowledge comes from the domain memory (Neo4j/Qdrant).

Weaving Metaphor:
Boot = Tension the relevant threads
Action = Pass the shuttle once
Exit = Record the weft, release tension
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from hololoom.domain_harness.schema import (
    ActionResult,
    DomainContext,
    DomainState,
    Feature,
    FeatureStatus,
    ProgressAction,
    ProgressEntry,
    TestResult,
)

# ============================================================================
# Protocol Interface
# ============================================================================

class DomainProtocol:
    """
    Orchestrates the Boot/Action/Exit lifecycle.
    
    Uses Neo4j (Yarn) for graph storage and optionally
    Qdrant (Warp) for semantic embedding.
    """

    def __init__(
        self,
        yarn,  # Neo4jKG or compatible
        warp=None,  # WarpSpace for embeddings (optional)
        working_memory=None,  # AgentWorkingMemory (optional)
        project_name: str = "default"
    ):
        self.yarn = yarn
        self.warp = warp
        self.working_memory = working_memory
        self.project_name = project_name

        # Selection strategy (can be overridden)
        self.feature_selector = self._default_feature_selector

    # ========================================================================
    # Boot Ritual
    # ========================================================================

    async def boot(self) -> DomainContext:
        """
        Ground worker in current reality.
        
        Steps:
        1. Load/create domain state
        2. Load actionable features
        3. Select ONE target feature
        4. Load recent progress
        5. Tension working memory (if available)
        
        Returns:
            DomainContext with everything worker needs
        """
        # 1. Get or create domain state
        state = await self._get_or_create_state()

        # 2. Load actionable features (not blocked by dependencies)
        pending_features = await self._get_actionable_features()

        # 3. Select target feature
        target = self.feature_selector(pending_features, state)

        # 4. Load recent progress
        recent_progress = await self._get_recent_progress(limit=10)

        # 5. Identify blocked features
        blocked = [f.id for f in pending_features if not f.is_actionable()]

        # 6. Tension working memory if available
        if self.working_memory and target:
            await self._tension_working_memory(target, recent_progress)

        # 7. Update state with current target
        if target:
            await self._update_state(current_feature_id=target.id)

        return DomainContext(
            state=state,
            target=target,
            pending_features=pending_features,
            recent_progress=recent_progress,
            blocked_features=blocked
        )

    # ========================================================================
    # Action Ritual
    # ========================================================================

    async def act(
        self,
        context: DomainContext,
        implementation_fn: Callable[[Feature], Awaitable[Any]],
        test_fn: Callable[[Feature], Awaitable[TestResult]]
    ) -> ActionResult:
        """
        Execute ONE unit of work.
        
        Steps:
        1. Call implementation function
        2. Run tests
        3. Update feature status
        
        Args:
            context: From boot()
            implementation_fn: Async function that implements the feature
            test_fn: Async function that runs tests
            
        Returns:
            ActionResult with outcome
        """
        if not context.has_work():
            return ActionResult(
                feature_id="none",
                success=False,
                summary="No actionable work available"
            )

        target = context.target
        start_time = time.time()

        # 1. Mark feature as in progress
        await self._update_feature_status(target.id, FeatureStatus.IN_PROGRESS)

        try:
            # 2. Implement
            impl_result = await implementation_fn(target)

            # 3. Run tests
            test_result = await test_fn(target)

            # 4. Update status based on test result
            new_status = FeatureStatus.PASSING if test_result.passed else FeatureStatus.FAILING
            await self._update_feature_status(target.id, new_status)

            duration_ms = int((time.time() - start_time) * 1000)

            return ActionResult(
                feature_id=target.id,
                success=test_result.passed,
                summary=f"{'✓' if test_result.passed else '✗'} {target.name}: {test_result.output[:200]}",
                test_result=test_result,
                duration_ms=duration_ms,
                files_changed=getattr(impl_result, 'files_changed', []),
                diff_hash=getattr(impl_result, 'diff_hash', None)
            )

        except Exception as e:
            # Mark as failing on exception
            await self._update_feature_status(target.id, FeatureStatus.FAILING)

            return ActionResult(
                feature_id=target.id,
                success=False,
                summary=f"Exception: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            )

    # ========================================================================
    # Exit Ritual
    # ========================================================================

    async def exit(
        self,
        result: ActionResult,
        worker_id: str | None = None
    ) -> ProgressEntry:
        """
        Record outcome and clean up.
        
        Steps:
        1. Create progress entry
        2. Save to Neo4j
        3. Embed in Qdrant (if available)
        4. Update domain state
        5. Clear working memory
        
        Returns:
            The created ProgressEntry
        """
        # 1. Create progress entry
        action = ProgressAction.COMPLETED if result.success else ProgressAction.FAILED
        entry = ProgressEntry(
            feature_id=result.feature_id,
            action=action,
            summary=result.summary,
            diff_hash=result.diff_hash,
            files_changed=result.files_changed,
            confidence=1.0 if result.success else 0.0,
            duration_ms=result.duration_ms,
            worker_id=worker_id
        )

        # 2. Save to Neo4j
        await self._save_progress_entry(entry)

        # 3. Save test result if present
        if result.test_result:
            await self._save_test_result(result.test_result)

        # 4. Embed in Qdrant if available
        if self.warp:
            await self._embed_progress(entry)

        # 5. Update domain state
        await self._update_state(
            current_feature_id=None,
            last_worker_run=datetime.now()
        )

        # 6. Clear working memory
        if self.working_memory:
            await self.working_memory.relax()

        return entry

    # ========================================================================
    # Helper: Feature Selection
    # ========================================================================

    def _default_feature_selector(
        self,
        features: list[Feature],
        state: DomainState
    ) -> Feature | None:
        """
        Default strategy: highest priority actionable feature.
        
        Can be overridden for:
        - MCTS-based selection
        - Thompson Sampling
        - Dependency-aware ordering
        """
        actionable = [f for f in features if f.is_actionable()]
        if not actionable:
            return None

        # Sort by priority (desc), then created_at (asc)
        actionable.sort(key=lambda f: (-f.priority, f.created_at))
        return actionable[0]

    def set_feature_selector(
        self,
        selector: Callable[[list[Feature], DomainState], Feature | None]
    ):
        """Override the feature selection strategy."""
        self.feature_selector = selector

    # ========================================================================
    # Helper: Neo4j Operations
    # ========================================================================

    async def _get_or_create_state(self) -> DomainState:
        """Load or create domain state singleton."""
        # This would use the actual Neo4j driver
        # For now, return a default state
        return DomainState(project_name=self.project_name)

    async def _get_actionable_features(self) -> list[Feature]:
        """Load features that can be worked on."""
        # Would execute DomainCypher.GET_ACTIONABLE_FEATURES
        return []

    async def _get_recent_progress(self, limit: int = 10) -> list[ProgressEntry]:
        """Load recent progress entries."""
        # Would execute DomainCypher.GET_RECENT_PROGRESS
        return []

    async def _update_feature_status(self, feature_id: str, status: FeatureStatus):
        """Update feature status in Neo4j."""
        # Would execute DomainCypher.UPDATE_FEATURE_STATUS
        pass

    async def _save_progress_entry(self, entry: ProgressEntry):
        """Save progress entry to Neo4j."""
        # Would create :ProgressEntry node
        pass

    async def _save_test_result(self, result: TestResult):
        """Save test result to Neo4j."""
        # Would create :TestResult node
        pass

    async def _update_state(self, **kwargs):
        """Update domain state properties."""
        # Would execute DomainCypher.UPDATE_STATE
        pass

    # ========================================================================
    # Helper: Warp Operations
    # ========================================================================

    async def _embed_progress(self, entry: ProgressEntry):
        """Embed progress entry in Qdrant for semantic search."""
        if not self.warp:
            return

        # Would use warp.embed() or similar
        entry.to_embedding_text()
        # await self.warp.add(text, metadata=entry.to_cypher_properties())

    # ========================================================================
    # Helper: Working Memory Operations
    # ========================================================================

    async def _tension_working_memory(
        self,
        target: Feature,
        progress: list[ProgressEntry]
    ):
        """Prepare working memory for the target feature."""
        if not self.working_memory:
            return

        # Pin the target feature
        # await self.working_memory.pin(target.name, weight=0.9)

        # Attend to relevant context
        # query = Query(text=target.description)
        # await self.working_memory.attend_to(query)


# ============================================================================
# Completion Check
# ============================================================================

class CompletionChecker:
    """Check if domain work is complete."""

    def __init__(self, protocol: DomainProtocol):
        self.protocol = protocol

    async def is_complete(self) -> bool:
        """Are all features passing?"""
        features = await self.protocol._get_actionable_features()
        return len(features) == 0

    async def get_status(self) -> dict:
        """Get completion status."""
        # Would query Neo4j for feature counts
        return {
            "total": 0,
            "passing": 0,
            "failing": 0,
            "pending": 0,
            "completion_rate": 0.0
        }


# ============================================================================
# Runner
# ============================================================================

class WorkerRunner:
    """
    Run worker cycles until completion.
    
    Usage:
        runner = WorkerRunner(protocol, implementation_fn, test_fn)
        results = await runner.run_until_complete(max_cycles=100)
    """

    def __init__(
        self,
        protocol: DomainProtocol,
        implementation_fn: Callable[[Feature], Awaitable[Any]],
        test_fn: Callable[[Feature], Awaitable[TestResult]],
        worker_id: str | None = None
    ):
        self.protocol = protocol
        self.implementation_fn = implementation_fn
        self.test_fn = test_fn
        self.worker_id = worker_id
        self.checker = CompletionChecker(protocol)

    async def run_once(self) -> ActionResult:
        """Run one complete worker cycle."""
        context = await self.protocol.boot()
        result = await self.protocol.act(
            context,
            self.implementation_fn,
            self.test_fn
        )
        await self.protocol.exit(result, self.worker_id)
        return result

    async def run_until_complete(
        self,
        max_cycles: int = 100,
        delay_between_cycles: float = 0.0
    ) -> list[ActionResult]:
        """
        Run worker cycles until all features pass or max_cycles reached.
        
        Returns:
            List of ActionResults from each cycle
        """
        results = []

        for i in range(max_cycles):
            if await self.checker.is_complete():
                break

            result = await self.run_once()
            results.append(result)

            # No work available
            if result.feature_id == "none":
                break

            if delay_between_cycles > 0:
                await asyncio.sleep(delay_between_cycles)

        return results


# ============================================================================
# Protocol with Neo4j Integration
# ============================================================================

class Neo4jDomainProtocol(DomainProtocol):
    """
    Full implementation with actual Neo4j operations.
    
    Requires a Neo4jKG instance from hololoom.memory.neo4j_graph
    """

    def __init__(self, yarn, warp=None, working_memory=None, project_name: str = "default"):
        super().__init__(yarn, warp, working_memory, project_name)
        self._initialized = False

    async def initialize(self):
        """Set up Neo4j schema (constraints, indexes)."""
        if self._initialized:
            return

        # Run schema setup
        # await self.yarn.execute(DomainCypher.CREATE_CONSTRAINTS)
        # await self.yarn.execute(DomainCypher.CREATE_INDEXES)

        self._initialized = True

    async def _get_or_create_state(self) -> DomainState:
        """Load or create domain state using Neo4j."""
        # query = DomainCypher.GET_OR_CREATE_STATE
        # result = await self.yarn.execute(query, {"project_name": self.project_name})
        # if result:
        #     return DomainState.from_cypher_record(result[0])
        return DomainState(project_name=self.project_name)

    async def _get_actionable_features(self) -> list[Feature]:
        """Load actionable features from Neo4j."""
        # result = await self.yarn.execute(DomainCypher.GET_ACTIONABLE_FEATURES)
        # return [Feature.from_cypher_record(r) for r in result]
        return []

    async def _get_recent_progress(self, limit: int = 10) -> list[ProgressEntry]:
        """Load recent progress from Neo4j."""
        # result = await self.yarn.execute(
        #     DomainCypher.GET_RECENT_PROGRESS,
        #     {"limit": limit}
        # )
        # return [ProgressEntry.from_cypher_record(r) for r in result]
        return []

    async def _update_feature_status(self, feature_id: str, status: FeatureStatus):
        """Update feature status in Neo4j."""
        # await self.yarn.execute(
        #     DomainCypher.UPDATE_FEATURE_STATUS,
        #     {"feature_id": feature_id, "status": status.value}
        # )
        pass

    async def _save_progress_entry(self, entry: ProgressEntry):
        """Save progress entry to Neo4j."""
        entry.to_cypher_properties()
        # await self.yarn.execute(
        #     "CREATE (p:ProgressEntry $props)",
        #     {"props": props}
        # )
        pass

    async def _save_test_result(self, result: TestResult):
        """Save test result to Neo4j."""
        result.to_cypher_properties()
        # await self.yarn.execute(
        #     "CREATE (t:TestResult $props)",
        #     {"props": props}
        # )
        pass

    async def _update_state(self, **kwargs):
        """Update domain state in Neo4j."""
        # Convert datetime to ISO string
        props = {}
        for k, v in kwargs.items():
            if isinstance(v, datetime):
                props[k] = v.isoformat()
            else:
                props[k] = v

        # await self.yarn.execute(
        #     DomainCypher.UPDATE_STATE,
        #     {"properties": props}
        # )
        pass
