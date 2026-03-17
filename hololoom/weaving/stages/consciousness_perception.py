"""
Consciousness Perception Stage
===============================
Epistemic awareness and activation tracking.

Computes awareness context including activation levels, coherence,
and semantic shift detection from the awareness graph.
"""

import logging
from typing import Any, ClassVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ConsciousnessPerceptionStage:
    """
    Consciousness Perception Stage (Awareness Integration).

    Computes epistemic awareness context from the awareness graph,
    including activation levels, coherence metrics, and shift detection.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'query', 'features'})
    writes: ClassVar[frozenset[str]] = frozenset({'awareness_context'})
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        awareness_graph: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.awareness_graph = awareness_graph
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Compute awareness context, writing awareness_context to ctx."""
        if self.awareness_graph is None:
            ctx.awareness_context = None
            return

        try:
            perception = await self.awareness_graph.perceive(
                query_text=ctx.query.text,
                features=ctx.features,
            )

            ctx.awareness_context = {
                'activation_level': getattr(perception, 'activation_level', 0.0),
                'coherence': getattr(perception, 'coherence', 0.0),
                'active_nodes': getattr(perception, 'active_nodes', 0),
                'shift_detected': getattr(perception, 'shift_detected', False),
                'semantic_position': getattr(perception, 'semantic_position', None),
                'perception_time_ms': getattr(perception, 'perception_time_ms', 0.0),
            }

            self.logger.info(
                f"  [A] Awareness: activation={ctx.awareness_context['activation_level']:.3f}, "
                f"coherence={ctx.awareness_context['coherence']:.3f}"
            )

        except (TypeError, AttributeError, ValueError, RuntimeError) as e:
            self.logger.warning(f"  [A] Consciousness perception failed: {e}")
            ctx.awareness_context = None

    def get_stage_name(self) -> str:
        return "consciousness_perception"

    def get_required_context_keys(self) -> list[str]:
        return []

    def get_output_context_keys(self) -> list[str]:
        return ["awareness_context"]
