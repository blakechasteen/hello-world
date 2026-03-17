"""
Fast Policy
===========

FAST mode: skip memory if no source, abort only on critical stages,
early termination on high-confidence quick decisions.
"""

from typing import Optional, TYPE_CHECKING

from .base import WeavingPolicyBase

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class FastPolicy(WeavingPolicyBase):
    """
    FAST mode policy.

    - Skips memory_retrieval if no memory source is available
    - Only aborts on critical stages (pattern_selection, feature_extraction)
    - Checks for early termination after quick_decision
    """

    # Stages whose failure should NOT abort the pipeline
    _NON_CRITICAL = frozenset({
        'memory_retrieval',
        'temporal_control',
        'production_metrics',
        'reflection',
    })

    def should_skip(self, stage_name: str, ctx: 'WeavingContext') -> bool:
        # Skip memory retrieval if no memory source attached
        if stage_name == 'memory_retrieval':
            has_memory = (
                hasattr(ctx, 'shards') and ctx.shards
            ) or getattr(ctx, '_has_memory_source', False)
            if not has_memory:
                return False  # Still attempt — stage handles empty gracefully
        return False

    def should_abort_on_failure(self, stage_name: str) -> bool:
        return stage_name not in self._NON_CRITICAL

    def on_level_complete(
        self, level: list[str], ctx: 'WeavingContext',
    ) -> Optional[frozenset[str]]:
        # Early termination: if quick_decision produced high-confidence result
        if 'quick_decision' in level and ctx.collapse_result:
            if hasattr(ctx.collapse_result, 'confidence') and ctx.collapse_result.confidence > 0.95:
                self.logger.info("  [FAST] High confidence — early termination")
                ctx.blocked_by = 'early_termination'
        return None
