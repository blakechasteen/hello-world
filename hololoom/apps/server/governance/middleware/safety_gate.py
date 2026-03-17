"""
SafetyGateMiddleware — SafetyGuardrails evaluation.

Extracted from routers/query.py lines 214-260.
"""

import logging

from fastapi import HTTPException

from ..context import GovernanceContext
from ..protocol import NextFn

logger = logging.getLogger(__name__)


class SafetyGateMiddleware:
    """Evaluates the request against SafetyGuardrails.

    If blocked, raises HTTPException(403).
    Otherwise stores the evaluation in ctx.safety_evaluation and proceeds.
    """

    name: str = "safety_gate"

    def __init__(self, safety_guardrails) -> None:
        self._guardrails = safety_guardrails

    async def __call__(self, ctx: GovernanceContext, next: NextFn) -> GovernanceContext:
        from hololoom.alignment.safety_guardrails import ActionRequest, ActionCategory

        has_code_context = bool(ctx.request_metadata.get("has_code_context"))

        action_request = ActionRequest(
            action="code_analysis" if has_code_context else "text_query",
            category=ActionCategory.CODE_EXECUTION if has_code_context else ActionCategory.QUERY,
            context={
                "query": ctx.query,
                "mode": ctx.mode,
                "max_steps": ctx.request_metadata.get("max_steps"),
                "has_code_context": has_code_context,
                "source": "vscode_extension",
                "timestamp": ctx.request_metadata.get("timestamp", ""),
            },
        )

        gate_result = self._guardrails.evaluate(action_request)

        logger.info(
            "Safety Gate: %s risk (score=%.2f, allowed=%s)",
            gate_result.risk_level.value,
            gate_result.safety_score,
            gate_result.allowed,
        )

        if not gate_result.allowed:
            error_msg = (
                f"Query blocked by safety guardrails: {gate_result.reason}. "
                f"Risk level: {gate_result.risk_level.value} "
                f"(safety score: {gate_result.safety_score:.2f})"
            )
            logger.warning(error_msg)
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "safety_guardrail_blocked",
                    "reason": gate_result.reason,
                    "risk_level": gate_result.risk_level.value,
                    "safety_score": gate_result.safety_score,
                    "message": error_msg,
                },
            )

        ctx.safety_evaluation = {
            "risk_level": gate_result.risk_level.value,
            "safety_score": gate_result.safety_score,
            "allowed": gate_result.allowed,
        }
        return await next(ctx)
