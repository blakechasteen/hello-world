"""
Weaving Policy Base
===================

Shared lifecycle hooks for all weaving policies.
Events, logging, timing — all live here, not in the runner.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class WeavingPolicyBase:
    """
    Base weaving policy — default no-skip, abort-on-failure behavior.

    Used directly for LITE and FULL tiers where no special runtime
    logic is needed. Subclassed by FastPolicy and ResearchPolicy.
    """

    def __init__(
        self,
        event_emitter: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.emitter = event_emitter
        self.logger = logger or logging.getLogger('hololoom.weaving')

    # ----- Decisions -----

    def should_skip(self, stage_name: str, ctx: 'WeavingContext') -> bool:
        """Return True to skip this stage. Default: never skip."""
        return False

    def should_abort_on_failure(self, stage_name: str) -> bool:
        """Return True to abort pipeline on this stage's failure. Default: always abort."""
        return True

    def on_level_complete(
        self, level: list[str], ctx: 'WeavingContext',
    ) -> Optional[frozenset[str]]:
        """Called after each level completes.

        Return a new active stage set to re-resolve remaining levels,
        or None to continue normally.
        """
        return None

    # ----- Lifecycle -----

    def on_stage_start(self, stage_name: str, ctx: 'WeavingContext') -> None:
        """Called before each stage executes."""
        self.logger.info(f"  [{stage_name}] starting")
        if self.emitter:
            self.emitter('stage_start', stage_name)

    def on_stage_complete(
        self,
        stage_name: str,
        ctx: 'WeavingContext',
        duration_ms: float,
        error: Optional[Exception] = None,
    ) -> None:
        """Called after each stage completes (success or failure)."""
        ctx.record_timing(stage_name, duration_ms)
        status = 'FAIL' if error else 'OK'
        self.logger.info(f"  [{stage_name}] {status} ({duration_ms:.1f}ms)")
        if self.emitter:
            self.emitter('stage_complete', stage_name, duration_ms, error)
