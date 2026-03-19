"""
Quick Decision Stage
====================
Lightweight decision stage for LITE and FAST recipes.

Combines simplified feature routing and tool selection into a single
stage, bypassing the full convergence engine.
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class QuickDecisionStage:
    """
    Quick Decision Stage (LITE/FAST Variant).

    Simplified decision-making using epsilon-greedy or rule-based routing
    instead of the full Bayesian convergence engine.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'features', 'query'})
    writes: ClassVar[frozenset[str]] = frozenset({'collapse_result', 'tool_result'})
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        tool_executor: Any,
        routing_table: dict | None = None,
        default_tool: str = "knowledge_base",
        logger: logging.Logger | None = None,
    ):
        self.tool_executor = tool_executor
        self.routing_table = routing_table or {}
        self.default_tool = default_tool
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Route query to tool and execute in single step."""
        from types import SimpleNamespace

        # Simple routing based on features
        tool = self._route(ctx)

        ctx.collapse_result = SimpleNamespace(
            tool=tool,
            confidence=0.8,
            strategy=SimpleNamespace(value="quick_decision"),
            bandit_stats=None,
        )

        # Execute immediately
        context = SimpleNamespace(
            memories=ctx.shards or [],
            shards=ctx.shards or [],
            shard_texts=ctx.shard_texts or [],
            query=ctx.query,
            features=ctx.features,
            metadata={},
        )

        ctx.tool_result = await self.tool_executor.execute(
            tool, ctx.query, context,
        )

        self.logger.info(f"  [QD] Quick decision: {tool}")

    def _route(self, ctx: 'WeavingContext') -> str:
        """Route query to appropriate tool."""
        if ctx.features and ctx.features.motifs:
            for motif in ctx.features.motifs:
                pattern = motif.pattern if hasattr(motif, 'pattern') else str(motif)
                if pattern in self.routing_table:
                    return self.routing_table[pattern]
        return self.default_tool

    def get_stage_name(self) -> str:
        return "quick_decision"

    def get_required_context_keys(self) -> list[str]:
        return ["features"]

    def get_output_context_keys(self) -> list[str]:
        return ["collapse_result", "tool_result"]
