"""
Coding rituals — pre-task health checks and post-task validation.

pre_code_health_check: Check code health before modification tasks.
If health < 9.5, signal refactor_required. Otherwise, proceed.
"""

from ..hooks import HookEvent
from ..policies import FailurePolicy
from ..types import Ritual, RitualBudget, RitualContext, RitualResult


async def _pre_code_health_check_body(ctx: RitualContext) -> RitualResult:
    """
    Evaluate code health for the target task.

    Wire to Code Health MCP server or local analysis tool.
    If health_score < 9.5: return refactor_required signal.
    If health_score >= 9.5: proceed, write score to context.
    """
    # Wire to Code Health MCP or local analysis
    # For now, placeholder — will connect to MCP tool
    health_score = 10.0

    if health_score < 9.5:
        return RitualResult(
            success=False,
            produced={
                "HealthScoreBobbin": {
                    "score": health_score,
                    "status": "refactor_required",
                },
            },
            confidence_delta=-0.05,
            notes=f"Code health {health_score:.1f} < 9.5. Refactor before proceeding.",
        )

    return RitualResult(
        success=True,
        produced={
            "HealthScoreBobbin": {
                "score": health_score,
                "status": "ok",
            },
        },
        confidence_delta=+0.03,
        notes=f"Code health {health_score:.1f} — proceeding.",
    )


PRE_CODE_HEALTH_CHECK = Ritual(
    id="pre_code_health_check",
    version="1.0.0",
    domain="coding",
    intent=(
        "Check code health score before any modification task. "
        "Agents perform best on healthy, modular code. "
        "If health < 9.5, halt and return refactor_required."
    ),
    author="system",
    consumes=[],
    produces=["HealthScoreBobbin"],
    failure_policy=FailurePolicy.LOG_AND_CONTINUE,
    budget=RitualBudget(max_tokens=300, max_seconds=30.0, priority_class="standard"),
    skip_condition=lambda ctx: (
        ctx.ritual_outputs.get("HealthScoreBobbin") is not None
        or ctx.task_complexity == "trivial"
        or ctx.token_budget.remaining < 500
    ),
    body=_pre_code_health_check_body,
)

PRE_CODE_HEALTH_CHECK_BINDING_CONFIG = {
    "trigger": HookEvent.TASK_RECEIVED,
    "condition": lambda ctx: ctx.task_type in ("code_modify", "refactor", "feature_add"),
    "priority": "BLOCKING",
}
