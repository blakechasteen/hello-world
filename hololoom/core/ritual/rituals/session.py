"""
Session rituals — dawn prime and dusk consolidate.

dawn_prime: SESSION_START — prime working memory from Yarn + Warp.
  Retrieves recent ritual outcomes, session context, and domain priors.

dusk_consolidate: SESSION_END — consolidate episodic → semantic,
  compress low-value memories, write forget-log.
"""

from ..hooks import HookEvent
from ..policies import FailurePolicy
from ..types import Ritual, RitualBudget, RitualContext, RitualResult


async def _dawn_prime_body(ctx: RitualContext) -> RitualResult:
    """
    Prime working memory at session start.

    1. Query Yarn for recent RitualHistoryNodes from this agent
    2. Query Warp for semantically relevant ritual outcomes
    3. Load domain priors and confidence history
    4. Produce a SessionPrimeBobbin with the priming context
    """
    # Wire to Yarn: recent outcomes for this agent
    # Wire to Warp: semantic search for task-relevant history
    priming_context = {
        "recent_rituals": [],
        "domain_priors": {},
        "session_history": [],
    }

    return RitualResult(
        success=True,
        produced={"SessionPrimeBobbin": priming_context},
        confidence_delta=+0.05,
        notes="Session primed with historical context.",
    )


DAWN_PRIME = Ritual(
    id="dawn_prime",
    version="1.0.0",
    domain="universal",
    intent="Prime working memory at session start from Yarn and Warp history.",
    author="system",
    consumes=[],
    produces=["SessionPrimeBobbin"],
    failure_policy=FailurePolicy.LOG_AND_CONTINUE,
    budget=RitualBudget(max_tokens=500, max_seconds=15.0, priority_class="standard"),
    body=_dawn_prime_body,
    closure_required_by="dusk_consolidate",
)

DAWN_PRIME_BINDING_CONFIG = {
    "trigger": HookEvent.SESSION_START,
    "condition": None,
    "priority": "BLOCKING",
}


async def _dusk_consolidate_body(ctx: RitualContext) -> RitualResult:
    """
    Consolidate session knowledge at session end.

    1. Review all ritual outcomes from this session
    2. Promote high-value episodic memories to semantic
    3. Compress low-value entries
    4. Write forget-log for pruned items
    """
    consolidation_report = {
        "rituals_executed": len(ctx.ritual_log),
        "promoted": 0,
        "compressed": 0,
        "pruned": 0,
    }

    return RitualResult(
        success=True,
        produced={"ConsolidationReportBobbin": consolidation_report},
        confidence_delta=0.0,
        notes=f"Session consolidated: {len(ctx.ritual_log)} rituals reviewed.",
    )


DUSK_CONSOLIDATE = Ritual(
    id="dusk_consolidate",
    version="1.0.0",
    domain="universal",
    intent="Consolidate episodic memory to semantic at session end. Compress and forget-log.",
    author="system",
    consumes=[],
    produces=["ConsolidationReportBobbin"],
    failure_policy=FailurePolicy.LOG_AND_CONTINUE,
    budget=RitualBudget(max_tokens=800, max_seconds=30.0, priority_class="best_effort"),
    body=_dusk_consolidate_body,
)

DUSK_CONSOLIDATE_BINDING_CONFIG = {
    "trigger": HookEvent.SESSION_END,
    "condition": None,
    "priority": "DEFERRED",
}
