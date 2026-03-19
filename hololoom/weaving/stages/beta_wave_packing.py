from __future__ import annotations
"""
Beta Wave Packing Stage
=======================
Step 6.5 of the weaving cycle: Physics-based context optimization.

Uses activation spreading through spring engine to:
- Prioritize most relevant context elements
- Compress lower-activation shards
- Respect token budget constraints
- Achieve ~50% token reduction with <1ms overhead
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class BetaWavePackingStage:
    """
    Beta Wave Packing Stage (Step 6.5: Context Optimization).

    Physics-based context packing using spring engine activation spreading.
    Optional stage — requires memory backend with spring_engine.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'dot_plasma', 'shards'})
    writes: ClassVar[frozenset[str]] = frozenset({'packed_context'})
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        spring_engine: Any,
        token_budget: int = 4096,
        query_reserve: int = 512,
        response_reserve: int = 1024,
        activation_threshold: float = 0.1,
        compression_threshold: float = 0.3,
        logger: logging.Logger | None = None,
    ):
        self.spring_engine = spring_engine
        self.token_budget = token_budget
        self.query_reserve = query_reserve
        self.response_reserve = response_reserve
        self.activation_threshold = activation_threshold
        self.compression_threshold = compression_threshold
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Pack context using physics-based activation spreading."""
        dot_plasma = ctx.dot_plasma or {}
        shards = ctx.shards or []

        if not shards:
            ctx.packed_context = None
            return

        try:
            from hololoom.awareness.beta_wave_packer import BetaWaveContextPacker, TokenBudget

            budget = TokenBudget(
                total=self.token_budget,
                reserved_for_query=self.query_reserve,
                reserved_for_response=self.response_reserve,
            )

            packer = BetaWaveContextPacker(
                spring_engine=self.spring_engine,
                token_budget=budget,
                activation_threshold=self.activation_threshold,
                compression_threshold=self.compression_threshold,
            )

            query_embedding = self._extract_query_embedding(dot_plasma)

            packed = await packer.pack_context(
                query_text=ctx.query.text if ctx.query else "",
                query_embedding=query_embedding,
                awareness_context=None,
                top_k=len(shards),
            )

            ctx.packed_context = packed
            self.logger.info(
                f"  [6.5] Beta wave packing: {packed.elements_included} included, "
                f"{packed.elements_compressed} compressed, "
                f"{packed.elements_excluded} excluded "
                f"({packed.total_tokens}/{budget.available_for_context} tokens, "
                f"avg_activation={packed.avg_activation:.3f})"
            )

        except ImportError:
            self.logger.warning("  [6.5] Beta wave packer not available")
            ctx.packed_context = None
        except (TypeError, AttributeError, ValueError, RuntimeError) as e:
            self.logger.warning(f"  [6.5] Beta wave packing failed: {e}. Falling back to raw shards.")
            ctx.packed_context = None

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

    def get_stage_name(self) -> str:
        return "beta_wave_packing"

    def get_required_context_keys(self) -> list[str]:
        return ["dot_plasma", "shards"]

    def get_output_context_keys(self) -> list[str]:
        return ["packed_context"]
