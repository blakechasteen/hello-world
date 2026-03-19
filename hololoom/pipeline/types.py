"""
Pipeline Types — Stage[C] and RunPolicy[C] protocols.

These are the core abstractions. Everything else in the pipeline
package is downstream of these two protocols.

Stage[C]: declares reads/writes on a typed context, executes logic.
RunPolicy[C]: shapes runtime behavior (skip, abort, adapt, observe).
"""

from typing import ClassVar, Protocol, TypeVar

C = TypeVar('C')


class Stage(Protocol[C]):
    """
    A single step in a pipeline.

    Stages declare their data dependencies via reads/writes frozensets.
    These declarations drive:
    - Ordering (resolver)
    - Parallelism (runner)
    - Caching (runner)
    - Safety validation (resolver)
    - CI verification (ReadWriteTracker)

    reads/writes reference field names on the typed context dataclass.
    Components (backends, services) are injected at __init__, not declared here.
    """

    reads: ClassVar[frozenset[str]]
    """Context field names this stage reads."""

    writes: ClassVar[frozenset[str]]
    """Context field names this stage produces."""

    cacheable: ClassVar[bool]
    """Whether this stage's outputs can be cached. False for side-effect stages."""

    async def execute(self, ctx: C) -> None:
        """
        Execute the stage, mutating ctx in place.

        Reads from ctx fields declared in `reads`.
        Writes to ctx fields declared in `writes`.
        """
        ...


class RunPolicy(Protocol[C]):
    """
    Runtime behavior hooks injected into PipelineRunner.

    5 methods: 3 decisions + 2 lifecycle.
    Default no-ops via DefaultPolicy.
    """

    # --- Decisions ---

    def should_skip(self, stage_name: str, ctx: C) -> bool:
        """Return True to skip this stage at runtime."""
        ...

    def should_abort_on_failure(self, stage_name: str) -> bool:
        """Return True if this stage's failure should stop the pipeline."""
        ...

    def on_level_complete(
        self, level: list[str], ctx: C
    ) -> frozenset[str] | None:
        """
        Called after each level completes.

        Return a new set of active stage names to re-resolve remaining
        levels (for dynamic recipe adaptation), or None to continue normally.
        """
        ...

    # --- Lifecycle ---

    def on_stage_start(self, stage_name: str, ctx: C) -> None:
        """Called before a stage executes."""
        ...

    def on_stage_complete(
        self,
        stage_name: str,
        ctx: C,
        duration_ms: float,
        error: Exception | None,
    ) -> None:
        """Called after a stage completes (success or failure)."""
        ...


class DefaultPolicy:
    """
    No-op policy. No skips, always abort on failure, no adaptation.

    Used for pipelines that don't need runtime behavior shaping,
    or as a base for testing.
    """

    def should_skip(self, stage_name: str, ctx) -> bool:
        return False

    def should_abort_on_failure(self, stage_name: str) -> bool:
        return True

    def on_level_complete(self, level, ctx):
        return None

    def on_stage_start(self, stage_name: str, ctx) -> None:
        pass

    def on_stage_complete(self, stage_name, ctx, duration_ms, error) -> None:
        pass
