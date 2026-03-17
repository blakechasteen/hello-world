"""
Smart Routing Stage
===================
Pre-pipeline query classification and fast-path routing.

Classifies query complexity and optionally triggers recipe override
for dynamic pipeline adaptation.
"""

import logging
from typing import Any, ClassVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class SmartRoutingStage:
    """
    Smart Routing Stage (Query Classification).

    Classifies the query to determine complexity level and whether
    a fast path can be used. May set recipe_override for dynamic
    pipeline re-resolution.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'query'})
    writes: ClassVar[frozenset[str]] = frozenset({'query_classification', 'fast_path_used', 'recipe_override'})
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        classifier: Optional[Any] = None,
        fast_path_recipes: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.classifier = classifier
        self.fast_path_recipes = fast_path_recipes or {}
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Classify query and set routing decisions."""
        ctx.fast_path_used = False
        ctx.recipe_override = None

        if self.classifier is None:
            ctx.query_classification = None
            return

        try:
            classification = self.classifier.classify(ctx.query.text)
            ctx.query_classification = classification

            # Check if classification triggers a fast path
            level = getattr(classification, 'level', None) or classification
            level_name = level.name if hasattr(level, 'name') else str(level)

            if level_name in self.fast_path_recipes:
                ctx.recipe_override = self.fast_path_recipes[level_name]
                ctx.fast_path_used = True
                self.logger.info(
                    f"  [R] Smart routing: {level_name} -> fast path"
                )
            else:
                self.logger.info(f"  [R] Smart routing: {level_name} -> full pipeline")

        except (TypeError, AttributeError, ValueError, RuntimeError) as e:
            self.logger.warning(f"  [R] Smart routing failed: {e}. Using default pipeline.")
            ctx.query_classification = None

    def get_stage_name(self) -> str:
        return "smart_routing"

    def get_required_context_keys(self) -> list[str]:
        return []

    def get_output_context_keys(self) -> list[str]:
        return ["query_classification", "fast_path_used", "recipe_override"]
