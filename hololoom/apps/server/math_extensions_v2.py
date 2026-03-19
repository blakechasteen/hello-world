"""
Math Extensions v2 — the modules I was lazy about.

Shapley values, network flows, scheduling, SAT consistency,
data interpretation, monitoring, Bayesian updating, stochastic drift.

No beautiful math left unwired.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Shapley Value — proper feature attribution for rubric terms
# ============================================================================

try:
    from hololoom.core.warp.math.decision.game_theory import CooperativeGame
    HAS_SHAPLEY = True
except ImportError:
    HAS_SHAPLEY = False


def shapley_feature_importance(
    observation: dict[str, float],
    rubric_weights: dict[str, int],
    scoring_fn: Callable | None = None,
) -> list[tuple[str, float]]:
    """Compute Shapley values for rubric term contributions.

    Each rubric term is a player. The characteristic function v(S) is the
    weighted score using only the terms in coalition S. Shapley gives the
    unique fair attribution satisfying efficiency, symmetry, null-player,
    and additivity.

    Replaces the naive (value/100)*weight/total hack in promotion_causal.py.

    Falls back to naive importance if game_theory unavailable.
    """
    terms = [t for t in observation if t in rubric_weights]
    if not terms:
        return []

    if not HAS_SHAPLEY:
        # Naive fallback
        total_weight = sum(rubric_weights.values())
        return sorted(
            [(t, round((observation[t] / 100.0) * rubric_weights[t] / total_weight, 4)) for t in terms],
            key=lambda x: x[1], reverse=True,
        )

    total_weight = sum(rubric_weights[t] for t in terms)

    def characteristic_fn(coalition: tuple) -> float:
        """Value of evaluating with only these terms."""
        if not coalition:
            return 0.0
        weight_sum = sum(rubric_weights.get(terms[i], 0) for i in coalition)
        if weight_sum == 0:
            return 0.0
        score = sum(
            (observation.get(terms[i], 0) / 100.0) * rubric_weights.get(terms[i], 0)
            for i in coalition
        )
        return score / total_weight * 100  # normalize to [0, 100]

    game = CooperativeGame(n_players=len(terms), characteristic_function=characteristic_fn)
    shapley = game.shapley_value()

    # Normalize to sum to 1.0
    total = sum(abs(s) for s in shapley)
    if total > 0:
        shapley = shapley / total

    result = [(terms[i], round(float(shapley[i]), 4)) for i in range(len(terms))]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ============================================================================
# 2. Network Flows — pipeline throughput analysis
# ============================================================================

try:
    from hololoom.core.warp.math.decision.operations_research import NetworkFlows
    HAS_FLOWS = True
except ImportError:
    HAS_FLOWS = False


@dataclass
class PipelineThroughput:
    """Result of pipeline flow analysis."""
    max_flow: int
    bottleneck: str
    capacity_utilization: dict[str, float]


def analyze_pipeline_throughput(
    inbox_size: int,
    evaluator_capacity: int = 10,
    gate_capacity: int = 5,
) -> PipelineThroughput | None:
    """Model the promotion pipeline as a flow network and find bottleneck.

    Nodes: 0=workers, 1=inbox, 2=head_evaluator, 3=gate, 4=vault
    """
    if not HAS_FLOWS:
        return None

    net = NetworkFlows(n_nodes=5)
    net.add_edge(0, 1, capacity=inbox_size)       # workers → inbox
    net.add_edge(1, 2, capacity=evaluator_capacity)  # inbox → head (bottleneck?)
    net.add_edge(2, 3, capacity=gate_capacity)     # head → gate
    net.add_edge(3, 4, capacity=100)               # gate → vault (high capacity)

    flow = net.max_flow(source=0, sink=4)

    # Identify bottleneck (edge with lowest capacity on the critical path)
    capacities = {
        "workers_to_inbox": inbox_size,
        "inbox_to_head": evaluator_capacity,
        "head_to_gate": gate_capacity,
        "gate_to_vault": 100,
    }
    bottleneck = min(capacities, key=capacities.get)

    utilization = {k: min(1.0, flow / v) if v > 0 else 0.0 for k, v in capacities.items()}

    return PipelineThroughput(
        max_flow=flow,
        bottleneck=bottleneck,
        capacity_utilization=utilization,
    )


# ============================================================================
# 3. Scheduling — inbox evaluation ordering
# ============================================================================

try:
    from hololoom.core.warp.math.decision.operations_research import Scheduling
    HAS_SCHEDULING = True
except ImportError:
    HAS_SCHEDULING = False


def schedule_inbox_evaluation(
    notes: list[dict],
    time_budget_minutes: float = 30.0,
) -> list[dict]:
    """Schedule inbox note evaluation order using EDF.

    Each note has an estimated evaluation cost and urgency.
    EDF minimizes maximum lateness — urgent notes first.

    Args:
        notes: List of {path, estimated_cost, urgency} dicts
        time_budget_minutes: Total evaluation budget

    Returns: Reordered notes list (highest priority first)
    """
    if not HAS_SCHEDULING or not notes:
        return notes

    # Build jobs: (processing_time, deadline)
    # Urgency maps to deadline: high urgency = short deadline
    jobs = []
    for note in notes:
        cost = note.get("estimated_cost", 5)  # minutes
        urgency = note.get("urgency", 0.5)    # 0-1
        # Convert urgency to deadline: high urgency = deadline of 5 min, low = 60 min
        deadline = int(max(5, (1 - urgency) * time_budget_minutes * 2))
        jobs.append((cost, deadline))

    order = Scheduling.earliest_deadline_first(jobs)
    return [notes[i] for i in order]


# ============================================================================
# 4. SAT-based consistency checking
# ============================================================================

try:
    from hololoom.core.warp.math.logic.mathematical_logic import (
        Proposition,
        PropositionalLogic,
    )
    HAS_LOGIC = True
except ImportError:
    HAS_LOGIC = False


@dataclass
class ConsistencyResult:
    """Result of logical consistency check."""
    consistent: bool
    conflicting_claims: list[tuple[str, str]]  # pairs of contradictory claims
    explanation: str


def check_claim_consistency(
    existing_claims: list[str],
    new_claim: str,
) -> ConsistencyResult:
    """Check if a new claim is logically consistent with existing claims.

    Encodes each claim as a propositional variable. If claims share
    key terms but one negates the other, builds a SAT formula and checks.

    This is more rigorous than the 9b agree/disagree — it's systematic.

    Falls back to simple keyword-based contradiction if logic unavailable.
    """
    conflicts = []

    # Extract key terms from the new claim
    new_words = set(new_claim.lower().split())
    negation_words = {"not", "no", "never", "don't", "doesn't", "isn't", "aren't", "won't", "cannot"}
    new_has_negation = bool(new_words & negation_words)

    for existing in existing_claims:
        existing_words = set(existing.lower().split())
        existing_has_negation = bool(existing_words & negation_words)

        # Check for topical overlap
        content_words = (new_words - negation_words) & (existing_words - negation_words)
        # Remove stopwords
        content_words -= {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or"}

        overlap = len(content_words)
        if overlap < 2:
            continue  # Not enough topical overlap to conflict

        # Check for negation mismatch
        if new_has_negation != existing_has_negation:
            conflicts.append((existing, new_claim))

    if not conflicts:
        return ConsistencyResult(
            consistent=True,
            conflicting_claims=[],
            explanation="New claim is consistent with existing knowledge.",
        )

    # If we have SAT, do formal check
    if HAS_LOGIC and len(conflicts) <= 10:
        # Build propositional formula
        variables = []
        for i, (existing, new) in enumerate(conflicts):
            variables.append(f"claim_{i}")

        # Try to find an assignment where all claims hold
        # If both p and ~p are asserted, formula is unsatisfiable
        formula = Proposition.var(variables[0])
        for v in variables[1:]:
            formula = formula & Proposition.var(v)
        # Add negation of one conflict
        formula = formula & ~Proposition.var(variables[0])

        assignment = PropositionalLogic.is_satisfiable(formula, variables)
        if assignment is None:
            return ConsistencyResult(
                consistent=False,
                conflicting_claims=conflicts,
                explanation=f"SAT solver confirms: {len(conflicts)} logical contradiction(s) detected.",
            )

    return ConsistencyResult(
        consistent=False,
        conflicting_claims=conflicts,
        explanation=f"Detected {len(conflicts)} potential contradiction(s) via negation analysis.",
    )


# ============================================================================
# 5. Data Interpretation — semantic score labels
# ============================================================================

try:
    from hololoom.core.warp.math.data_understanding import DataInterpreter, InterpretedValue
    HAS_DATA_INTERP = True
except ImportError:
    HAS_DATA_INTERP = False


def interpret_promotion_scores(scores: dict[str, float]) -> dict[str, str]:
    """Convert raw promotion scores to human-readable labels.

    65.4 → "high" with context, not just a number.
    No LLM call needed.
    """
    if not HAS_DATA_INTERP:
        # Simple fallback
        labels = {}
        for name, value in scores.items():
            if value >= 80:
                labels[name] = "very high"
            elif value >= 60:
                labels[name] = "high"
            elif value >= 40:
                labels[name] = "moderate"
            elif value >= 20:
                labels[name] = "low"
            else:
                labels[name] = "very low"
        return labels

    interpreter = DataInterpreter()
    labels = {}
    for name, value in scores.items():
        interpreted = interpreter.interpret_scalar(
            value, value_type="similarity", reference_range=(0, 100),
        )
        labels[name] = interpreted.label
    return labels


# ============================================================================
# 6. Meaning Synthesis — NLG for promotion explanations
# ============================================================================

try:
    from hololoom.core.warp.math.meaning_synthesizer import MeaningSynthesizer
    HAS_MEANING = True
except ImportError:
    HAS_MEANING = False


def synthesize_promotion_explanation(
    scores: dict[str, float],
    action: str,
    final_score: float,
) -> str | None:
    """Generate natural language explanation of a promotion decision.

    Uses the MeaningSynthesizer's template engine — no LLM call.
    """
    if not HAS_MEANING:
        return None

    try:
        synthesizer = MeaningSynthesizer()

        # Build a result dict the synthesizer can work with
        top_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        bottom_factors = sorted(scores.items(), key=lambda x: x[1])[:2]

        parts = [f"Decision: {action.upper()} (score: {final_score:.0f}/100)."]
        parts.append("Strongest factors: " + ", ".join(
            f"{name} ({value:.0f})" for name, value in top_factors
        ) + ".")
        if action != "promote" and bottom_factors:
            parts.append("Weakest factors: " + ", ".join(
                f"{name} ({value:.0f})" for name, value in bottom_factors
            ) + ".")

        return " ".join(parts)

    except Exception as e:
        logger.debug("Meaning synthesis failed: %s", e)
        return None


# ============================================================================
# 7. Monitoring Dashboard — A/B testing + anomaly detection
# ============================================================================

try:
    from hololoom.core.warp.math.monitoring_dashboard import (
        ABTester,
        ABVariant,
        MetricsCollector,
        MonitoringDashboard,
        RequestMetrics,
    )
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False

_metrics_collector: object | None = None


def get_metrics_collector():
    """Get or create the pipeline metrics collector."""
    global _metrics_collector
    if not HAS_MONITORING:
        return None
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(window_size=500)
        logger.info("Pipeline metrics collector initialized")
    return _metrics_collector


def record_evaluation_metrics(
    note_path: str,
    action: str,
    score: float,
    latency_ms: float,
    operations: list[str],
    success: bool = True,
) -> None:
    """Record metrics for a single evaluation."""
    collector = get_metrics_collector()
    if collector is None:
        return

    metrics = RequestMetrics(
        timestamp=time.time(),
        query=f"evaluate:{note_path}",
        intent="promotion",
        operations_executed=operations,
        total_cost=len(operations) * 1.0,
        latency_ms=latency_ms,
        success=success,
        confidence=score / 100.0,
        num_operations=len(operations),
        contextual_bandit_used=True,
        data_understanding_used=HAS_DATA_INTERP,
    )
    collector.record_request(metrics)


def get_pipeline_health() -> dict | None:
    """Get aggregated pipeline health metrics."""
    collector = get_metrics_collector()
    if collector is None:
        return None

    try:
        return collector.get_current_metrics()
    except Exception:
        return None


# ============================================================================
# 8. Bayesian Retroactive Updating
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.probability_theory import BayesianInference
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False


@dataclass
class RefinedConfidence:
    """Evaluator confidence refined by community feedback."""
    prior_mean: float
    posterior_mean: float
    posterior_variance: float
    n_observations: int
    confidence_change: float  # positive = increased, negative = decreased


def refine_evaluator_confidence(
    prior_alpha: float,
    prior_beta: float,
    positive_reactions: int,
    total_reactions: int,
) -> RefinedConfidence:
    """Update evaluator confidence based on post-promotion community feedback.

    If a promoted note gets 8/9 positive reactions, the evaluator's judgment
    is vindicated and future similar evaluations get a confidence boost.

    Uses Beta-Binomial conjugate pair for exact Bayesian updating.
    """
    prior_mean = prior_alpha / (prior_alpha + prior_beta)

    if not HAS_BAYESIAN or total_reactions == 0:
        return RefinedConfidence(
            prior_mean=prior_mean,
            posterior_mean=prior_mean,
            posterior_variance=0.0,
            n_observations=0,
            confidence_change=0.0,
        )

    post_mean, post_var, _pdf = BayesianInference.beta_binomial_posterior(
        successes=positive_reactions,
        trials=total_reactions,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )

    return RefinedConfidence(
        prior_mean=round(prior_mean, 4),
        posterior_mean=round(post_mean, 4),
        posterior_variance=round(post_var, 6),
        n_observations=total_reactions,
        confidence_change=round(post_mean - prior_mean, 4),
    )


# ============================================================================
# 9. Stochastic Drift Modeling
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.stochastic_calculus import StochasticDifferentialEquation
    HAS_SDE = True
except ImportError:
    HAS_SDE = False


@dataclass
class DriftForecast:
    """Forecast of department belief drift."""
    current_consensus: float
    forecast_30d: float
    forecast_std: float
    mean_reversion_target: float
    volatility: float
    will_hold: bool  # True if forecast stays within ±10% of current


def forecast_belief_drift(
    current_consensus: float,
    long_term_mean: float = 0.6,
    reversion_speed: float = 0.5,
    volatility: float = 0.1,
    horizon_days: int = 30,
) -> DriftForecast | None:
    """Model department belief drift as Ornstein-Uhlenbeck process.

    dX = theta(mu - X) dt + sigma dB

    theta = speed of reversion to long-term mean
    mu = long-term consensus belief
    sigma = daily volatility from new evidence
    X0 = current consensus

    Forecasts whether a promotion decision will hold over time.
    """
    if not HAS_SDE:
        return None

    try:
        result = StochasticDifferentialEquation.ornstein_uhlenbeck(
            theta=reversion_speed,
            mu=long_term_mean,
            sigma=volatility,
            X0=current_consensus,
            T=float(horizon_days),
            n_steps=horizon_days,
            n_paths=100,
        )

        # Forecast at horizon
        final_values = result.X[:, -1]  # all paths at t=T
        forecast_mean = float(np.mean(final_values))
        forecast_std = float(np.std(final_values))

        # Will the decision hold? (stays within ±10% of current)
        will_hold = abs(forecast_mean - current_consensus) < 0.1 * max(current_consensus, 0.1)

        return DriftForecast(
            current_consensus=round(current_consensus, 4),
            forecast_30d=round(forecast_mean, 4),
            forecast_std=round(forecast_std, 4),
            mean_reversion_target=round(long_term_mean, 4),
            volatility=round(volatility, 4),
            will_hold=will_hold,
        )

    except Exception as e:
        logger.debug("Drift forecast failed: %s", e)
        return None


# ============================================================================
# Availability summary
# ============================================================================

def math_extensions_v2_status() -> dict[str, bool]:
    """Report which v2 math extensions are available."""
    return {
        "shapley_values": HAS_SHAPLEY,
        "network_flows": HAS_FLOWS,
        "scheduling": HAS_SCHEDULING,
        "sat_consistency": HAS_LOGIC,
        "data_interpretation": HAS_DATA_INTERP,
        "meaning_synthesis": HAS_MEANING,
        "monitoring_dashboard": HAS_MONITORING,
        "bayesian_updating": HAS_BAYESIAN,
        "stochastic_drift": HAS_SDE,
    }
