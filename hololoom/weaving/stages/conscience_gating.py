"""
Conscience Gating Stage
=======================
Ethical evaluation of the planned action before execution.

Evaluates the collapse result against conscience/ethics constraints,
potentially blocking actions that violate safety policies.
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ConscienceGatingStage:
    """
    Conscience Gating Stage (Ethical Evaluation).

    Evaluates the planned tool action against ethical constraints.
    Can block execution by setting conscience_blocked on the context.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'query', 'collapse_result'})
    writes: ClassVar[frozenset[str]] = frozenset({'conscience_decision', 'conscience_blocked'})
    cacheable: ClassVar[bool] = False  # Ethical decisions shouldn't be cached

    def __init__(
        self,
        evaluator: Any | None = None,
        logger: logging.Logger | None = None,
    ):
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Evaluate ethics constraints, writing conscience_decision to ctx."""
        if self.evaluator is None:
            ctx.conscience_decision = None
            ctx.conscience_blocked = False
            return

        try:
            decision = await self.evaluator.evaluate(
                query=ctx.query,
                collapse_result=ctx.collapse_result,
            )

            ctx.conscience_decision = decision
            ctx.conscience_blocked = getattr(decision, 'blocked', False)

            if ctx.conscience_blocked:
                self.logger.warning(
                    f"  [CG] Conscience BLOCKED: {getattr(decision, 'reason', 'unknown')}"
                )
            else:
                self.logger.info("  [CG] Conscience gate passed")

        except Exception as e:
            self.logger.warning(f"  [CG] Conscience gating failed (non-blocking): {e}")
            ctx.conscience_decision = None
            ctx.conscience_blocked = False

    def get_stage_name(self) -> str:
        return "conscience_gating"

    def get_required_context_keys(self) -> list[str]:
        return ["collapse_result"]

    def get_output_context_keys(self) -> list[str]:
        return ["conscience_decision", "conscience_blocked"]
