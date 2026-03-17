"""
DeceptionMonitorMiddleware — post-reasoning deception probe.

Extracted from routers/query.py lines 321-343.
"""

import logging

from ..context import GovernanceContext
from ..protocol import NextFn

logger = logging.getLogger(__name__)


class DeceptionMonitorMiddleware:
    """Wraps ``next()`` and runs a deception probe on the response.

    This is a *post-reasoning* middleware: it calls ``next`` first, then
    evaluates the response for deception and optionally logs to an audit trail.
    """

    name: str = "deception_monitor"

    def __init__(self, deception_detector, audit_trail=None) -> None:
        self._detector = deception_detector
        self._trail = audit_trail

    async def __call__(self, ctx: GovernanceContext, next: NextFn) -> GovernanceContext:
        # Let downstream middleware (including reasoning) run first
        ctx = await next(ctx)

        result = ctx.reasoning_result
        response_text = ctx.response
        if result is None or not response_text:
            return ctx

        try:
            from hololoom.alignment.deception_detection import BehavioralProbe, ProbeType
            from hololoom.alignment.audit_trail import DecisionType, OutcomeType

            probe = BehavioralProbe(
                probe_type=ProbeType.HONESTY,
                scenario=f"Response to query: {ctx.query[:100]}",
                expected_behavior=ctx.query,
            )
            _passed, deception_score = self._detector.run_probe(
                probe, actual_behavior=response_text
            )

            ctx.annotations["deception_score"] = deception_score

            if self._trail:
                self._trail.log_decision(
                    decision_type=DecisionType.DECEPTION_CHECK,
                    outcome=OutcomeType.APPROVED,
                    reason=f"Deception monitoring (Jaccard score={deception_score:.2f})",
                    query_text=ctx.query,
                    action_description="deception_monitoring",
                    confidence=1.0 - deception_score,
                    metadata={"deception_score": deception_score},
                )
        except Exception as e:
            logger.debug("Deception monitoring skipped: %s", e)

        return ctx
