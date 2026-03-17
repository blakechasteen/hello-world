"""Awareness perception gate — enrichment only, never short-circuits."""

from __future__ import annotations

import time
from typing import Any, Optional, TYPE_CHECKING

from hololoom.orchestrator.protocols.stage import GateExecutor

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class AwarenessExecutor(GateExecutor):
    """Perceive query through awareness layer for epistemic consciousness."""

    stage_id: int = -4
    stage_name: str = "Awareness Perception"
    timing_key: str = "awareness_perception"

    def __init__(self, awareness_layer: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.awareness_layer = awareness_layer

    async def evaluate(self, ctx: 'WeavingContext') -> Optional[Any]:
        if not self.awareness_layer:
            return None

        awareness_perception = await self.awareness_layer.perceive(ctx.current_query_text)

        awareness_context = {
            'semantic_position': awareness_perception.get('position'),
            'activation_level': awareness_perception.get('activation', 0.0),
            'perception_time_ms': 0.0,  # filled by base timing
        }

        awareness_metrics = self.awareness_layer.get_metrics()
        awareness_context.update({
            'active_nodes': awareness_metrics.get('active_nodes', 0),
            'coherence': awareness_metrics.get('coherence', {}).get('global_coherence', 0.0),
            'shift_detected': awareness_metrics.get('shift_detected', False),
        })

        self.logger.info(
            f"[AWARENESS] Perception complete: "
            f"activation={awareness_context['activation_level']:.3f}, "
            f"coherence={awareness_context['coherence']:.3f}"
        )

        ctx.awareness_context = awareness_context
        return None  # Never short-circuits
