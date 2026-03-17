"""
AuditTrailMiddleware — post-reasoning audit logging.

Extracted from routers/query.py lines 293-319.
"""

import logging

from ..context import GovernanceContext
from ..protocol import NextFn

logger = logging.getLogger(__name__)


class AuditTrailMiddleware:
    """Wraps ``next()`` and logs the completed reasoning result to the audit trail.

    This is a *post-reasoning* middleware: it calls ``next`` first (which may
    include the reasoning step), then logs the outcome.
    """

    name: str = "audit_trail"

    def __init__(self, audit_trail) -> None:
        self._trail = audit_trail

    async def __call__(self, ctx: GovernanceContext, next: NextFn) -> GovernanceContext:
        # Let downstream middleware (including reasoning) run first
        ctx = await next(ctx)

        result = ctx.reasoning_result
        if result is None:
            return ctx

        try:
            from hololoom.alignment.audit_trail import DecisionType, OutcomeType

            safety_eval = ctx.safety_evaluation or {}

            self._trail.log_decision(
                decision_type=DecisionType.TOOL_SELECTION,
                outcome=OutcomeType.APPROVED,
                reason=f"Agentic reasoning completed successfully in {ctx.mode} mode",
                query_text=ctx.query,
                action_description=f"agentic_reasoning_{ctx.mode}",
                risk_level=safety_eval.get("risk_level"),
                confidence=result.spacetime.confidence,
                metadata={
                    "code_context": ctx.request_metadata.get("code_context"),
                    "max_steps": ctx.request_metadata.get("max_steps"),
                    "reasoning_mode": result.reasoning_mode.value,
                    "steps_taken": len(result.steps_taken),
                    "total_queries": result.total_queries,
                    "timestamp": ctx.request_metadata.get("timestamp", ""),
                    "query_id": result.spacetime.query_id,
                    "latency_ms": ctx.elapsed_ms,
                    "verification": result.verification is not None,
                    "safety_score": safety_eval.get("safety_score"),
                },
            )
            logger.debug("Logged to audit trail: query_id=%s", result.spacetime.query_id)
        except Exception as e:
            logger.error("Failed to log to audit trail: %s", e)

        return ctx
