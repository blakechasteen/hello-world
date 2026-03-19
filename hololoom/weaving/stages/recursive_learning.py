"""
Recursive Learning Stage
========================
Post-pipeline learning updates via Thompson Sampling.

Updates bandit priors based on execution outcomes.
Side-effect stage — no context writes.
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class RecursiveLearningStage:
    """
    Recursive Learning Stage (Thompson Sampling Updates).

    Updates policy priors based on execution outcomes and confidence.
    Side-effect stage — writes to learning backend, not context.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'spacetime', 'collapse_result'})
    writes: ClassVar[frozenset[str]] = frozenset()  # Side-effect only
    cacheable: ClassVar[bool] = False  # Side-effect stage

    def __init__(
        self,
        learning_loop: Any | None = None,
        semantic_bandit: Any | None = None,
        success_threshold: float = 0.5,
        logger: logging.Logger | None = None,
    ):
        self.learning_loop = learning_loop
        self.semantic_bandit = semantic_bandit
        self.success_threshold = success_threshold
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Update learning systems with execution outcome."""
        collapse_result = ctx.collapse_result
        if collapse_result is None:
            return

        # Update semantic bandit
        if self.semantic_bandit:
            try:
                success = collapse_result.confidence >= self.success_threshold
                self.semantic_bandit.update(
                    tool=collapse_result.tool,
                    success=success,
                    confidence=collapse_result.confidence,
                )
                self.logger.info(
                    f"  [RL] Bandit updated: tool={collapse_result.tool}, "
                    f"success={success}, conf={collapse_result.confidence:.3f}"
                )
            except Exception as e:
                self.logger.warning(f"  [RL] Semantic bandit update failed: {e}")

        # Update learning loop
        if self.learning_loop and ctx.spacetime:
            try:
                await self.learning_loop.record_outcome(
                    spacetime=ctx.spacetime,
                    collapse_result=collapse_result,
                )
                self.logger.info("  [RL] Learning loop updated")
            except Exception as e:
                self.logger.warning(f"  [RL] Learning loop update failed: {e}")

    def get_stage_name(self) -> str:
        return "recursive_learning"

    def get_required_context_keys(self) -> list[str]:
        return ["collapse_result"]

    def get_output_context_keys(self) -> list[str]:
        return []
