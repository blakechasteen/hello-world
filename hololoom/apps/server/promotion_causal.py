"""
Promotion Causal Model — justification chains for promotion decisions.

Builds a Structural Causal Model (SCM) encoding the promotion pipeline:
  link_count → mechanical_score → final_score → action
  body_length → mechanical_score
  claim_quality → llm_score → final_score
  novelty → llm_score
  connection_depth → llm_score
  contradiction → action (direct edge, bypasses score)

Enables:
  - "Why was this promoted?" → causal chain tracing
  - "What if claim_quality was 30?" → counterfactual intervention
  - Feature importance → which variables most influenced the action
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Graceful import of causal inference
try:
    from hololoom.core.warp.math.causal.causal_inference import CausalGraph
    HAS_CAUSAL = True
except ImportError:
    HAS_CAUSAL = False
    logger.info("Causal inference unavailable — justification chains disabled")

router = APIRouter(prefix="/department", tags=["promotion-causal"])


# ============================================================================
# Promotion Causal DAG
# ============================================================================

PROMOTION_EDGES = [
    ("link_count", "mechanical_score"),
    ("body_length", "mechanical_score"),
    ("mechanical_score", "final_score"),
    ("claim_quality", "llm_score"),
    ("novelty", "llm_score"),
    ("connection_depth", "llm_score"),
    ("actionability", "llm_score"),
    ("llm_score", "final_score"),
    ("final_score", "action"),
    ("contradiction_detected", "action"),
    ("action", "gate_outcome"),
    ("secops_score", "gate_outcome"),
    ("ea_score", "gate_outcome"),
    ("dept_drift", "ea_score"),
    ("community_health", "ea_score"),
]


def _build_promotion_dag() -> object | None:
    """Build the promotion causal DAG."""
    if not HAS_CAUSAL:
        return None
    graph = CausalGraph()
    graph.add_edges(PROMOTION_EDGES)
    return graph


# Singleton DAG
_promotion_dag = None


def get_dag():
    global _promotion_dag
    if _promotion_dag is None:
        _promotion_dag = _build_promotion_dag()
    return _promotion_dag


# ============================================================================
# Causal Chain Tracing
# ============================================================================

@dataclass
class CausalChainLink:
    """One step in a causal justification chain."""
    variable: str
    value: float
    contribution: str  # "high", "medium", "low", "blocking"


@dataclass
class PromotionExplanation:
    """Complete causal explanation of a promotion decision."""
    action: str
    final_score: float
    causal_chain: list[CausalChainLink]
    primary_causes: list[str]
    counterfactuals: list[str]


def trace_causal_chain(
    observation: dict[str, float],
    action: str,
    final_score: float,
    thresholds: dict[str, float] = None,
) -> PromotionExplanation:
    """Trace the causal chain that led to a promotion decision.

    Args:
        observation: Dict of variable → observed value
            e.g. {"link_count": 100, "body_length": 100, "claim_quality": 85, ...}
        action: The action taken ("promote", "return", "escalate")
        final_score: The computed final score
        thresholds: Score thresholds for actions (defaults to 70/40)
    """
    if thresholds is None:
        thresholds = {"promote": 70, "return": 40}

    chain = []
    primary_causes = []

    # Trace mechanical path
    mech_vars = ["link_count", "body_length"]
    mech_values = [observation.get(v, 0) for v in mech_vars]
    mech_score = observation.get("mechanical_score", sum(mech_values) / max(1, len(mech_values)))

    for var in mech_vars:
        val = observation.get(var, 0)
        contribution = "high" if val >= 80 else "medium" if val >= 50 else "low"
        chain.append(CausalChainLink(var, val, contribution))
        if val < 50:
            primary_causes.append(f"{var} is weak ({val:.0f})")

    chain.append(CausalChainLink("mechanical_score", mech_score,
                                  "high" if mech_score >= 80 else "medium" if mech_score >= 50 else "low"))

    # Trace LLM path
    llm_vars = ["claim_quality", "novelty", "connection_depth", "actionability"]
    llm_values = [observation.get(v, 0) for v in llm_vars]
    llm_score = observation.get("llm_score", sum(llm_values) / max(1, len(llm_values)))

    for var in llm_vars:
        val = observation.get(var, 0)
        contribution = "high" if val >= 70 else "medium" if val >= 40 else "low"
        chain.append(CausalChainLink(var, val, contribution))
        if val >= 80:
            primary_causes.append(f"{var} is strong ({val:.0f})")
        elif val < 40:
            primary_causes.append(f"{var} is weak ({val:.0f})")

    chain.append(CausalChainLink("llm_score", llm_score,
                                  "high" if llm_score >= 70 else "medium" if llm_score >= 40 else "low"))

    # Final score and action
    chain.append(CausalChainLink("final_score", final_score,
                                  "high" if final_score >= thresholds["promote"] else
                                  "medium" if final_score >= thresholds["return"] else "low"))

    # Contradiction check
    contradiction = observation.get("contradiction_detected", 0)
    if contradiction > 0:
        chain.append(CausalChainLink("contradiction_detected", contradiction, "blocking"))
        primary_causes.append("contradiction detected — escalation triggered")

    chain.append(CausalChainLink("action", 1.0 if action == "promote" else 0.5 if action == "return" else 0.0,
                                  action))

    # Generate counterfactuals
    counterfactuals = _generate_counterfactuals(observation, action, final_score, thresholds)

    if not primary_causes:
        if action == "promote":
            primary_causes.append(f"final score {final_score:.0f} exceeds threshold {thresholds['promote']}")
        elif action == "return":
            primary_causes.append(f"final score {final_score:.0f} below promote threshold {thresholds['promote']}")
        else:
            primary_causes.append(f"final score {final_score:.0f} below return threshold {thresholds['return']}")

    return PromotionExplanation(
        action=action,
        final_score=final_score,
        causal_chain=chain,
        primary_causes=primary_causes,
        counterfactuals=counterfactuals,
    )


def _generate_counterfactuals(
    observation: dict[str, float],
    action: str,
    final_score: float,
    thresholds: dict[str, float],
) -> list[str]:
    """Generate counterfactual explanations — what would change the decision?"""
    counterfactuals = []

    if action == "promote":
        # What would prevent promotion?
        gap = final_score - thresholds["promote"]
        # Find which variable, if reduced, would drop below threshold
        for var in ["claim_quality", "link_count", "novelty", "connection_depth"]:
            val = observation.get(var, 0)
            if val > 50:
                counterfactuals.append(
                    f"If {var} dropped from {val:.0f} to below {val - gap * 2:.0f}, "
                    f"decision would change to 'return'"
                )

    elif action == "return":
        # What would enable promotion?
        gap = thresholds["promote"] - final_score
        for var in ["claim_quality", "link_count", "novelty", "body_length"]:
            val = observation.get(var, 0)
            if val < 80:
                needed = min(100, val + gap * 2)
                counterfactuals.append(
                    f"If {var} increased from {val:.0f} to {needed:.0f}, "
                    f"decision might change to 'promote'"
                )

    elif action == "escalate":
        gap = thresholds["return"] - final_score
        counterfactuals.append(
            f"Score needs to increase by {gap:.0f} points to reach 'return' threshold"
        )

    return counterfactuals[:3]  # Cap at 3


def format_causal_chain(explanation: PromotionExplanation) -> str:
    """Format causal explanation as markdown for frontmatter or API response."""
    lines = [f"action: {explanation.action} (score: {explanation.final_score:.1f})"]
    lines.append("chain: " + " → ".join(
        f"{link.variable}({link.value:.0f})" for link in explanation.causal_chain
        if link.contribution != "action"
    ))
    lines.append("causes: " + "; ".join(explanation.primary_causes))
    if explanation.counterfactuals:
        lines.append("counterfactuals: " + "; ".join(explanation.counterfactuals[:2]))
    return "\n".join(lines)


# ============================================================================
# Feature Importance
# ============================================================================

def feature_importance(observation: dict[str, float], rubric_weights: dict[str, int]) -> list[tuple]:
    """Rank variables by their contribution to the final score.

    Uses Shapley values (game theory) for mathematically correct attribution
    when available. Falls back to naive weighted importance otherwise.

    Shapley: each rubric term is a player in a coalitional game. The
    characteristic function v(S) is the weighted score using only coalition S.
    Shapley gives the unique fair attribution satisfying efficiency, symmetry,
    null-player, and additivity axioms.
    """
    try:
        from hololoom.apps.server.math_extensions_v2 import HAS_SHAPLEY, shapley_feature_importance
        if HAS_SHAPLEY:
            return shapley_feature_importance(observation, rubric_weights)
    except ImportError:
        pass

    # Naive fallback
    total_weight = sum(rubric_weights.values()) if rubric_weights else 1
    importance = []
    for var, val in observation.items():
        weight = rubric_weights.get(var, 0)
        if weight > 0:
            imp = (val / 100.0) * weight / total_weight
            importance.append((var, round(imp, 4)))
    importance.sort(key=lambda x: x[1], reverse=True)
    return importance


# ============================================================================
# Endpoint
# ============================================================================

@router.get("/{dept}/explain/{note_id}")
async def explain_promotion(dept: str, note_id: str):
    """Get causal explanation for a promotion decision.

    note_id is the kebab-case filename stem (without .md).
    Looks up the evaluation record and traces the causal chain.
    """
    # For now, return the DAG structure and example explanation
    dag = get_dag()
    if dag is None:
        return {"error": "Causal inference unavailable"}

    return {
        "dept": dept,
        "note_id": note_id,
        "dag_edges": PROMOTION_EDGES,
        "dag_nodes": sorted(dag.nodes) if dag else [],
        "note": "Full explanation requires evaluation history — use POST /department/{dept}/evaluate first",
    }
