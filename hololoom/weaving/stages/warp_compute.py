"""
Warp Compute Stage
==================
Step 5.5 of the weaving cycle: Tensor operations in continuous manifold.

Performs after warp tensioning:
- Query embedding extraction from DotPlasma
- WarpSpace compute (spectral features, attention, weighted context)
- Semantic state computation (topic shift detection, policy features)
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class WarpComputeStage:
    """
    Warp Compute Stage (Step 5.5: Tensor Operations).

    Computes spectral features, attention entropy, and semantic state
    from the tensioned warp space and DotPlasma embeddings.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'dot_plasma', 'warp_operations'})
    writes: ClassVar[frozenset[str]] = frozenset({'warp_compute_results'})
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        warp_space: Any,
        semantic_state_class: Any | None = None,
        semantic_tool_selector: Any | None = None,
        logger: logging.Logger | None = None,
    ):
        self.warp_space = warp_space
        self.semantic_state_class = semantic_state_class
        self.semantic_tool_selector = semantic_tool_selector
        self.logger = logger or logging.getLogger(__name__)
        self._previous_semantic_state: Any | None = None
        self._query_index: int = 0

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Compute warp tensor operations, writing warp_compute_results to ctx."""
        dot_plasma = ctx.dot_plasma or {}

        query_embedding = self._extract_query_embedding(dot_plasma)

        try:
            warp_results = self.warp_space.compute(
                query_embedding=query_embedding,
                compute_spectral=True,
            )

            # Append compute operation to warp_operations log
            if ctx.warp_operations is not None:
                ctx.warp_operations.append((
                    __import__('datetime').datetime.now().isoformat(),
                    "compute",
                    {
                        'attention_entropy': warp_results.get('attention_entropy', 0.0),
                        'spectral_computed': warp_results.get('metadata', {}).get('spectral_computed', False),
                    }
                ))

            self.logger.info(
                f"  [5.5] Warp Space compute: "
                f"attention_entropy={warp_results.get('attention_entropy', 0.0):.3f}, "
                f"spectral={warp_results.get('metadata', {}).get('spectral_computed', False)}"
            )

            # Compute semantic state if available
            semantic_state = self._compute_semantic_state(query_embedding)
            if semantic_state is not None:
                warp_results['semantic_state'] = semantic_state

            ctx.warp_compute_results = warp_results

        except (TypeError, AttributeError, ValueError, RuntimeError) as e:
            self.logger.warning(f"  [5.5] Warp Space compute failed: {e}. Continuing without warp features.")
            ctx.warp_compute_results = None

    def _extract_query_embedding(self, dot_plasma: dict[str, Any]) -> np.ndarray:
        """Extract query embedding from DotPlasma psi field."""
        psi_raw = dot_plasma.get('psi', [])
        if isinstance(psi_raw, dict):
            query_embedding = psi_raw[max(psi_raw.keys())]
        else:
            query_embedding = psi_raw

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        return query_embedding

    def _compute_semantic_state(self, query_embedding: np.ndarray) -> dict[str, Any] | None:
        """Compute semantic state for topic shift detection."""
        if self.semantic_state_class is None:
            return None

        try:
            semantic_state = self.semantic_state_class.from_embedding(
                embedding=query_embedding,
                previous_state=self._previous_semantic_state,
                query_index=self._query_index,
            )

            self._previous_semantic_state = semantic_state
            self._query_index += 1

            result = {
                'position': semantic_state.position.tolist() if semantic_state.position is not None else None,
                'velocity': semantic_state.velocity.tolist() if semantic_state.velocity is not None else None,
                'category_scores': semantic_state.category_scores,
                'dominant_category': semantic_state.dominant_category,
                'topic_shift_detected': semantic_state.topic_shift_detected,
                'shift_magnitude': semantic_state.shift_magnitude,
                'query_index': semantic_state.query_index,
                'policy_features': semantic_state.to_feature_vector().tolist(),
            }

            if semantic_state.topic_shift_detected:
                self.logger.info(
                    f"  [5.6] Topic shift detected: "
                    f"magnitude={semantic_state.shift_magnitude:.3f}, "
                    f"dominant_category={semantic_state.dominant_category}"
                )

            if self.semantic_tool_selector is not None:
                suggested_tool = self.semantic_tool_selector.suggest_tool(semantic_state)
                result['suggested_tool'] = suggested_tool

            return result

        except (TypeError, AttributeError, ValueError, RuntimeError) as e:
            self.logger.warning(f"  [5.6] Semantic state computation failed: {e}")
            return None

    def get_stage_name(self) -> str:
        return "warp_compute"

    def get_required_context_keys(self) -> list[str]:
        return ["dot_plasma", "warp_operations"]

    def get_output_context_keys(self) -> list[str]:
        return ["warp_compute_results"]
