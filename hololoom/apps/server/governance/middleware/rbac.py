"""
RBACMiddleware — RBAC / PolicyGovernance gate.

Extracted from routers/query.py lines 183-213.
"""

import logging

from fastapi import HTTPException

from ..context import GovernanceContext
from ..protocol import NextFn

logger = logging.getLogger(__name__)


class RBACMiddleware:
    """Evaluates governance policy on the incoming request.

    On DENY, raises HTTPException(403).
    On ESCALATE, logs and proceeds (audit trail picks it up later).
    On ALLOW, proceeds silently.
    """

    name: str = "rbac"

    def __init__(self, governance_engine) -> None:
        self._engine = governance_engine

    async def __call__(self, ctx: GovernanceContext, next: NextFn) -> GovernanceContext:
        try:
            from hololoom.agents.policy_governance import (
                CommunicationRequest as GovRequest,
                Priority as GovPriority,
                PolicyDecision,
            )
        except ImportError:
            # Governance module not installed — skip silently
            return await next(ctx)

        gov_request = GovRequest(
            from_agent=ctx.request_metadata.get("agent_id", "anonymous"),
            to_agent="hololoom_server",
            message_type="query",
            topic=ctx.mode,
            content=ctx.query,
            priority=GovPriority.MEDIUM,
            metadata={
                "mode": ctx.mode,
                "max_steps": ctx.request_metadata.get("max_steps"),
            },
        )

        decision, reason = self._engine.evaluate_communication_request(gov_request)

        if decision == PolicyDecision.DENY:
            logger.warning("RBAC DENY: %s", reason)
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "governance_denied",
                    "reason": reason,
                    "message": f"Request denied by governance policy: {reason}",
                },
            )

        if decision == PolicyDecision.ESCALATE:
            logger.info("RBAC ESCALATE: %s (proceeding with audit)", reason)

        ctx.rbac_decision = decision.name if hasattr(decision, "name") else str(decision)
        return await next(ctx)
