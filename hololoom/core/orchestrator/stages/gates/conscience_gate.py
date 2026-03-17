"""Conscience gate — blocks queries that fail ethical reasoning."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from hololoom.orchestrator.protocols.stage import GateExecutor

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ConscienceGateExecutor(GateExecutor):
    """Per-query conscience check. Returns blocked Spacetime if rejected."""

    stage_id: int = -3
    stage_name: str = "Conscience Gate"
    timing_key: str = "conscience_gate"

    def __init__(self, conscience_adapter: Any, audit_trail: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.conscience_adapter = conscience_adapter
        self.audit_trail = audit_trail

    async def evaluate(self, ctx: 'WeavingContext') -> Optional[Any]:
        if not self.conscience_adapter:
            return None

        conscience_decision = await self.conscience_adapter.gate_reasoning(
            query_text=ctx.current_query_text,
            context={
                'awareness': ctx.awareness_context,
                'complexity': ctx.complexity.name if ctx.complexity else 'auto',
                'source': 'weaving_orchestrator',
            }
        )

        if not conscience_decision:
            return None

        # Log to audit trail
        if self.audit_trail:
            try:
                from hololoom.alignment.audit_trail import DecisionType, OutcomeType
                self.audit_trail.log_decision(
                    decision_type=DecisionType.QUERY_EVALUATION,
                    outcome=OutcomeType.APPROVED if conscience_decision.allowed else OutcomeType.REJECTED,
                    reason=f"Conscience gate: {conscience_decision.reason}",
                    query_text=ctx.current_query_text,
                    metadata={
                        'conscience_voice': conscience_decision.voice,
                        'risk_level': conscience_decision.risk_level,
                        'concerns': conscience_decision.concerns,
                        'guidance': conscience_decision.guidance,
                    }
                )
            except ImportError:
                pass

        if not conscience_decision.allowed:
            self.logger.warning(
                f"[CONSCIENCE] Query blocked: {conscience_decision.reason} "
                f"(voice={conscience_decision.voice}, risk={conscience_decision.risk_level})"
            )
            ctx.conscience_blocked = True

            from hololoom.fabric.spacetime import Spacetime, WeavingTrace
            return Spacetime(
                response=conscience_decision.guidance or "I cannot process this query at this time.",
                confidence=0.0,
                trace=WeavingTrace(),
                metadata={
                    'blocked_by_conscience': True,
                    'conscience_decision': {
                        'allowed': False,
                        'reason': conscience_decision.reason,
                        'voice': conscience_decision.voice,
                        'risk_level': conscience_decision.risk_level,
                        'concerns': conscience_decision.concerns,
                        'guidance': conscience_decision.guidance,
                    },
                    'query': ctx.current_query_text,
                }
            )

        self.logger.info(
            f"[CONSCIENCE] Query approved: {conscience_decision.reason} "
            f"(voice={conscience_decision.voice}, risk={conscience_decision.risk_level})"
        )
        return None
