"""
Research Policy
===============

RESEARCH mode: run everything, abort on quality failures,
adapt dynamically via recipe_override.
"""

from typing import TYPE_CHECKING

from .base import WeavingPolicyBase

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ResearchPolicy(WeavingPolicyBase):
    """
    RESEARCH mode policy.

    - Never skips stages (maximum quality)
    - Always aborts on failure (fail-fast for debugging)
    - Supports dynamic recipe adaptation via ctx.recipe_override
    """

    def should_skip(self, stage_name: str, ctx: 'WeavingContext') -> bool:
        return False

    def should_abort_on_failure(self, stage_name: str) -> bool:
        return True

    def on_level_complete(
        self, level: list[str], ctx: 'WeavingContext',
    ) -> frozenset[str] | None:
        # Dynamic adaptation: if smart_routing set a recipe override,
        # return the new stage set for the runner to re-resolve.
        if ctx.recipe_override is not None:
            override = ctx.recipe_override
            ctx.recipe_override = None
            # Extract stage set from Recipe or frozenset
            if hasattr(override, 'stages'):
                return override.stages
            return frozenset(override)
        return None
