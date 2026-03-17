"""
ReasoningMiddleware — orchestrator.reason() invocation.

Extracted from routers/query.py lines 266-291.
"""

import logging
from typing import Callable, Awaitable, Any

from ..context import GovernanceContext
from ..protocol import NextFn

logger = logging.getLogger(__name__)


class ReasoningMiddleware:
    """Calls the agentic orchestrator and stores the result on the context.

    Constructor takes a coroutine factory that returns the orchestrator
    (e.g. ``get_orchestrator``).
    """

    name: str = "reasoning"

    def __init__(self, get_orchestrator: Callable[[], Awaitable[Any]]) -> None:
        self._get_orchestrator = get_orchestrator

    async def __call__(self, ctx: GovernanceContext, next: NextFn) -> GovernanceContext:
        from hololoom.agentic import ReasoningMode
        from hololoom.protocols.types import Query

        mode_map = {
            "direct": ReasoningMode.DIRECT,
            "verify": ReasoningMode.VERIFY,
            "research": ReasoningMode.RESEARCH,
            "plan_execute": ReasoningMode.PLAN_EXECUTE,
        }
        mode = mode_map.get(ctx.mode, ReasoningMode.VERIFY)

        query = Query(text=ctx.query)
        # Carry forward any metadata (code context, safety eval, etc.)
        if ctx.request_metadata.get("code_context"):
            query.metadata = {
                "code_context": ctx.request_metadata["code_context"],
                **({k: v for k, v in ctx.request_metadata.items() if k != "code_context"}),
            }
        if ctx.safety_evaluation:
            if not query.metadata:
                query.metadata = {}
            query.metadata["safety_evaluation"] = ctx.safety_evaluation

        orchestrator = await self._get_orchestrator()
        max_steps = ctx.request_metadata.get("max_steps", 5)

        logger.info("Query: %s... (mode=%s)", ctx.query[:100], ctx.mode)
        result = await orchestrator.reason(query, mode=mode, max_steps=max_steps)

        response_text = result.spacetime.metadata.get("response", "")
        if not response_text:
            response_text = f"Processed query with {result.reasoning_mode.value} mode."

        ctx.reasoning_result = result
        ctx.response = response_text
        ctx.annotations["query_obj"] = query  # downstream middleware may need it

        return await next(ctx)
