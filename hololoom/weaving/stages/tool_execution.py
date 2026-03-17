"""
Tool Execution Stage
====================
Step 8 of the weaving cycle: Execute selected tool with safety gating.

This stage:
- Evaluates action safety via guardrails (if available)
- Logs to audit trail
- Executes the selected tool
- Handles safety-blocked scenarios gracefully
"""

import logging
from typing import Any, ClassVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ToolExecutionStage:
    """
    Tool Execution Stage (Step 8: Safety Gating + Execution).

    Evaluates safety constraints before executing the tool selected
    by the convergence engine. Blocks execution if safety thresholds
    are exceeded.
    """

    # Note: also checks ctx.conscience_blocked at runtime as a defensive guard.
    # Proper ordering (conscience_gating -> tool_execution) is enforced by the
    # factory computing dynamic pre_satisfied in A-8.
    reads: ClassVar[frozenset[str]] = frozenset({'query', 'collapse_result', 'complexity'})
    writes: ClassVar[frozenset[str]] = frozenset({'tool_result', 'safety_decision', 'safety_blocked'})
    cacheable: ClassVar[bool] = False  # Tool execution has side effects

    def __init__(
        self,
        tool_executor: Any,
        guardrails: Optional[Any] = None,
        audit_trail: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.tool_executor = tool_executor
        self.guardrails = guardrails
        self.audit_trail = audit_trail
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Execute tool with safety gating, writing tool_result to ctx."""
        collapse_result = ctx.collapse_result

        # 8a. Check conscience gate
        if ctx.conscience_blocked:
            self.logger.warning("  [8] Skipped — conscience gate blocked execution")
            ctx.safety_blocked = True
            ctx.tool_result = {
                'result': '[Action blocked by conscience gating]',
                'artifacts': [],
                'blocked': True,
            }
            return

        # 8b. Safety gating via guardrails
        if self.guardrails:
            blocked = await self._evaluate_safety(ctx, collapse_result)
            if blocked:
                return

        # 8b. Execute the selected tool
        ctx.tool_result = await self.tool_executor.execute(
            collapse_result.tool,
            ctx.query,
            self._build_tool_context(ctx),
        )
        self.logger.info(f"  [8] Tool executed: {collapse_result.tool}")

    async def _evaluate_safety(self, ctx: 'WeavingContext', collapse_result: Any) -> bool:
        """Evaluate safety constraints. Returns True if blocked."""
        try:
            from uuid import uuid4
            from hololoom.alignment.safety_guardrails import ActionRequest

            action_request = ActionRequest(
                action_id=str(uuid4()),
                category=self._categorize_tool(collapse_result.tool),
                description=f"Execute '{collapse_result.tool}' for query: {ctx.query.text[:100]}",
                context={
                    "query": ctx.query.text,
                    "tool": collapse_result.tool,
                    "confidence": collapse_result.confidence,
                    "complexity": ctx.complexity.value if ctx.complexity else "unknown",
                },
            )

            safety_decision = self.guardrails.evaluate(
                action_request,
                text_input=ctx.query.text,
            )
            ctx.safety_decision = safety_decision

            # Log to audit trail
            if self.audit_trail:
                self._log_audit(action_request, collapse_result, safety_decision, ctx)

            if not safety_decision.allowed:
                self.logger.warning(
                    f"  [8] Tool execution BLOCKED by safety gating: {safety_decision.reason}"
                )
                ctx.safety_blocked = True
                ctx.tool_result = {
                    'result': f"[Action blocked by safety guardrails: {safety_decision.reason}]",
                    'artifacts': [],
                    'blocked': True,
                }
                return True

        except Exception as e:
            self.logger.warning(f"Safety gating failed (non-blocking): {e}")

        return False

    def _log_audit(
        self, action_request: Any, collapse_result: Any,
        safety_decision: Any, ctx: 'WeavingContext',
    ) -> None:
        """Log safety decision to audit trail."""
        try:
            from hololoom.alignment.safety_guardrails import DecisionType, OutcomeType

            self.audit_trail.log_decision(
                decision_id=action_request.action_id,
                decision_type=DecisionType.TOOL_SELECTION,
                action=collapse_result.tool,
                outcome=OutcomeType.ALLOWED if safety_decision.allowed else OutcomeType.BLOCKED,
                metadata={
                    "query": ctx.query.text[:200],
                    "confidence": collapse_result.confidence,
                    "risk_level": (
                        safety_decision.risk_level.value
                        if hasattr(safety_decision.risk_level, 'value')
                        else str(safety_decision.risk_level)
                    ),
                    "safety_score": safety_decision.safety_score,
                },
            )
        except Exception as e:
            self.logger.warning(f"Audit trail logging failed: {e}")

    def _build_tool_context(self, ctx: 'WeavingContext') -> Any:
        """Build context object for tool executor."""
        from types import SimpleNamespace
        return SimpleNamespace(
            memories=ctx.shards,
            shards=ctx.shards,
            hits=ctx.hits,
            shard_texts=ctx.shard_texts,
            query=ctx.query,
            features=ctx.features,
            metadata={},
        )

    @staticmethod
    def _categorize_tool(tool_name: str) -> str:
        """Categorize tool for safety evaluation."""
        high_risk = {'execute', 'run', 'shell', 'system', 'delete', 'modify'}
        medium_risk = {'write', 'create', 'update', 'send'}
        for keyword in high_risk:
            if keyword in tool_name.lower():
                return "high_risk"
        for keyword in medium_risk:
            if keyword in tool_name.lower():
                return "medium_risk"
        return "low_risk"

    def get_stage_name(self) -> str:
        return "tool_execution"

    def get_required_context_keys(self) -> list[str]:
        return ["collapse_result"]

    def get_output_context_keys(self) -> list[str]:
        return ["tool_result", "safety_decision", "safety_blocked"]
