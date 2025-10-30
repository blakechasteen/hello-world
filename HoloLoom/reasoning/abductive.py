"""
Abductive Reasoning Engine

Layer 3 of Cognitive Architecture - Inference to best explanation.

Abduction: Given observations, find the most likely explanation.
- Medical diagnosis: symptoms → disease
- Debugging: bug behavior → root cause
- Science: experimental data → theory

Core Algorithm:
1. Generate candidate hypotheses
2. Score each hypothesis:
   score(H | O) = P(O | H) × P(H) / complexity(H)
              = likelihood × prior / parsimony
3. Rank and return top-k explanations

Research Alignment:
- Peirce (1878): "Deduction, Induction, and Hypothesis" (origin of abduction)
- Josephson & Josephson (1996): "Abductive Inference"
- Pearl (2000): Causality (causal explanation)
- Hobbs et al. (1993): "Interpretation as Abduction"

Public API:
    Hypothesis: Candidate explanation with scoring components
    AbductiveReasoner: Main reasoning engine
    generate_hypotheses: Helper for hypothesis generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from enum import Enum
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Hypothesis:
    """
    Candidate explanation for observations.

    A hypothesis is a set of variable assignments that explains observations.
    Scored by likelihood (how well it explains), prior (how plausible),
    and complexity (simplicity via Occam's razor).

    Attributes:
        explanation: Variable assignments (e.g., {"disease": "flu"})
        likelihood: P(observations | hypothesis) - how well it explains
        prior: P(hypothesis) - base rate plausibility
        complexity: Complexity penalty (number of assumptions)
        observations_explained: Which observations this hypothesis explains
        supporting_evidence: Evidence supporting this hypothesis
    """
    explanation: Dict[str, Any]
    likelihood: float
    prior: float
    complexity: float
    observations_explained: Set[str] = field(default_factory=set)
    supporting_evidence: List[str] = field(default_factory=list)

    def score(self) -> float:
        """
        Combined explanation quality score.

        Formula: (likelihood × prior) / (1 + complexity)

        Higher score = better explanation:
        - High likelihood: explains observations well
        - High prior: plausible a priori
        - Low complexity: simpler (Occam's razor)

        Returns:
            Composite score in [0, 1]
        """
        # Prevent division by zero, penalize high complexity
        return (self.likelihood * self.prior) / (1.0 + self.complexity)

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (f"Hypothesis({self.explanation}, "
                f"score={self.score():.3f}, "
                f"L={self.likelihood:.3f}, "
                f"P={self.prior:.3f}, "
                f"C={self.complexity:.1f})")


@dataclass
class Observation:
    """
    Single observation (evidence).

    Attributes:
        variable: What was observed (e.g., "symptom")
        value: Observed value (e.g., "fever")
        confidence: Observation reliability [0, 1]
        timestamp: When observed (optional)
    """
    variable: str
    value: Any
    confidence: float = 1.0
    timestamp: Optional[float] = None

    def __hash__(self):
        """Make observations hashable for set operations."""
        return hash((self.variable, str(self.value)))

    def __eq__(self, other):
        """Equality based on variable and value."""
        return (isinstance(other, Observation) and
                self.variable == other.variable and
                self.value == other.value)

    def __repr__(self) -> str:
        return f"Observation({self.variable}={self.value}, conf={self.confidence:.2f})"


@dataclass
class CausalRule:
    """
    Causal rule for hypothesis generation.

    Represents: cause → effect with strength.
    E.g., "flu → fever" with strength 0.9

    Attributes:
        cause_variable: Cause variable name
        cause_value: Cause value
        effect_variable: Effect variable name
        effect_value: Effect value
        strength: P(effect | cause) - how reliably cause produces effect
    """
    cause_variable: str
    cause_value: Any
    effect_variable: str
    effect_value: Any
    strength: float  # P(effect | cause)

    def __repr__(self) -> str:
        return (f"{self.cause_variable}={self.cause_value} → "
                f"{self.effect_variable}={self.effect_value} "
                f"[{self.strength:.2f}]")


# ============================================================================
# Hypothesis Generation
# ============================================================================

class HypothesisGenerator:
    """
    Generates candidate explanations from observations.

    Uses causal rules to work backward from observations to possible causes.
    Multiple strategies: single-cause, multi-cause, composite hypotheses.
    """

    def __init__(self, causal_rules: List[CausalRule]):
        """
        Initialize generator with causal knowledge.

        Args:
            causal_rules: Domain causal rules (cause → effect)
        """
        self.causal_rules = causal_rules

        # Index: effect → causes for fast lookup
        self.effect_to_causes: Dict[Tuple[str, Any], List[CausalRule]] = defaultdict(list)
        for rule in causal_rules:
            key = (rule.effect_variable, rule.effect_value)
            self.effect_to_causes[key].append(rule)

    def generate(self,
                observations: List[Observation],
                max_hypotheses: int = 20,
                allow_multi_cause: bool = True) -> List[Dict[str, Any]]:
        """
        Generate candidate hypotheses from observations.

        Strategy:
        1. For each observation, find possible causes (backward chaining)
        2. Generate single-cause hypotheses
        3. If allow_multi_cause, generate composite hypotheses
        4. Limit to max_hypotheses

        Args:
            observations: Observed evidence
            max_hypotheses: Maximum hypotheses to generate
            allow_multi_cause: Allow multiple causes in one hypothesis

        Returns:
            List of hypothesis dicts (variable assignments)
        """
        logger.info(f"Generating hypotheses for {len(observations)} observations")

        # Single-cause hypotheses (each observation → one cause)
        single_hypotheses = self._generate_single_cause(observations)

        if not allow_multi_cause:
            return single_hypotheses[:max_hypotheses]

        # Multi-cause hypotheses (multiple causes together)
        multi_hypotheses = self._generate_multi_cause(observations, single_hypotheses)

        # Combine and deduplicate
        all_hypotheses = single_hypotheses + multi_hypotheses
        unique_hypotheses = self._deduplicate(all_hypotheses)

        logger.info(f"Generated {len(unique_hypotheses)} unique hypotheses")
        return unique_hypotheses[:max_hypotheses]

    def _generate_single_cause(self, observations: List[Observation]) -> List[Dict[str, Any]]:
        """Generate hypotheses with single cause explaining all observations."""
        hypotheses = []

        # For each observation, find possible causes
        for obs in observations:
            key = (obs.variable, obs.value)
            possible_causes = self.effect_to_causes.get(key, [])

            for rule in possible_causes:
                hypothesis = {rule.cause_variable: rule.cause_value}
                hypotheses.append(hypothesis)

        return hypotheses

    def _generate_multi_cause(self,
                             observations: List[Observation],
                             single_hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses with multiple causes (conjunctions)."""
        multi_hypotheses = []

        # Combine pairs of single-cause hypotheses
        for i, h1 in enumerate(single_hypotheses):
            for h2 in single_hypotheses[i+1:]:
                # Merge if no conflicting assignments
                if self._compatible(h1, h2):
                    merged = {**h1, **h2}
                    multi_hypotheses.append(merged)

        return multi_hypotheses

    def _compatible(self, h1: Dict[str, Any], h2: Dict[str, Any]) -> bool:
        """Check if two hypotheses can be merged (no conflicts)."""
        for var in h1:
            if var in h2 and h1[var] != h2[var]:
                return False
        return True

    def _deduplicate(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate hypotheses."""
        seen = set()
        unique = []
        for h in hypotheses:
            # Convert dict to hashable representation
            key = tuple(sorted(h.items()))
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique


# ============================================================================
# Hypothesis Scoring
# ============================================================================

class HypothesisScorer:
    """
    Scores hypotheses using Bayesian inference.

    Score = P(observations | hypothesis) × P(hypothesis) / complexity(hypothesis)
          = likelihood × prior / parsimony
    """

    def __init__(self,
                 causal_rules: List[CausalRule],
                 prior_probabilities: Optional[Dict[str, Dict[Any, float]]] = None):
        """
        Initialize scorer with causal model and priors.

        Args:
            causal_rules: Causal rules for likelihood calculation
            prior_probabilities: P(variable=value) for each variable
        """
        self.causal_rules = causal_rules
        self.prior_probabilities = prior_probabilities or {}

        # Index for fast lookup
        self.cause_to_effects: Dict[Tuple[str, Any], List[CausalRule]] = defaultdict(list)
        for rule in causal_rules:
            key = (rule.cause_variable, rule.cause_value)
            self.cause_to_effects[key].append(rule)

    def score(self,
             hypothesis_dict: Dict[str, Any],
             observations: List[Observation]) -> Hypothesis:
        """
        Score a single hypothesis.

        Computes:
        - Likelihood: P(observations | hypothesis)
        - Prior: P(hypothesis)
        - Complexity: Number of assumptions

        Args:
            hypothesis_dict: Variable assignments
            observations: Observed evidence

        Returns:
            Scored Hypothesis object
        """
        likelihood = self._calculate_likelihood(hypothesis_dict, observations)
        prior = self._calculate_prior(hypothesis_dict)
        complexity = self._calculate_complexity(hypothesis_dict)

        # Track which observations this hypothesis explains
        explained = self._find_explained_observations(hypothesis_dict, observations)
        evidence = self._find_supporting_evidence(hypothesis_dict, observations)

        return Hypothesis(
            explanation=hypothesis_dict,
            likelihood=likelihood,
            prior=prior,
            complexity=complexity,
            observations_explained=explained,
            supporting_evidence=evidence
        )

    def _calculate_likelihood(self,
                            hypothesis: Dict[str, Any],
                            observations: List[Observation]) -> float:
        """
        Calculate P(observations | hypothesis).

        Uses causal rules to predict observations from hypothesis.
        For each observation, check if hypothesis predicts it.

        Likelihood = product of P(obs_i | hypothesis) for all observations

        Args:
            hypothesis: Cause assignments
            observations: Observed effects

        Returns:
            Likelihood in [0, 1]
        """
        if not observations:
            return 1.0

        likelihoods = []

        for obs in observations:
            # Find if hypothesis causes this observation
            prob = self._probability_of_observation(hypothesis, obs)
            # Weight by observation confidence
            weighted_prob = prob * obs.confidence + (1 - obs.confidence) * 0.5
            likelihoods.append(weighted_prob)

        # Combine likelihoods (geometric mean to prevent too small values)
        if likelihoods:
            geometric_mean = math.exp(sum(math.log(p + 1e-10) for p in likelihoods) / len(likelihoods))
            return geometric_mean

        return 0.5  # Neutral

    def _probability_of_observation(self,
                                   hypothesis: Dict[str, Any],
                                   observation: Observation) -> float:
        """
        Calculate P(observation | hypothesis).

        Checks if hypothesis causes observation via causal rules.

        Returns:
            Probability in [0, 1]
        """
        # Check all variables in hypothesis
        for var, value in hypothesis.items():
            key = (var, value)
            effects = self.cause_to_effects.get(key, [])

            # Find rule that produces this observation
            for rule in effects:
                if (rule.effect_variable == observation.variable and
                    rule.effect_value == observation.value):
                    return rule.strength

        # No rule found → observation unlikely given hypothesis
        return 0.1  # Small probability (not impossible, but unlikely)

    def _calculate_prior(self, hypothesis: Dict[str, Any]) -> float:
        """
        Calculate P(hypothesis) - base rate probability.

        Assumes independence: P(H) = ∏ P(var=value)

        Args:
            hypothesis: Variable assignments

        Returns:
            Prior probability in [0, 1]
        """
        if not hypothesis:
            return 1.0

        priors = []
        for var, value in hypothesis.items():
            if var in self.prior_probabilities:
                prob = self.prior_probabilities[var].get(value, 0.1)
            else:
                prob = 0.5  # Uniform prior if unknown
            priors.append(prob)

        # Combine priors (geometric mean)
        if priors:
            return math.exp(sum(math.log(p + 1e-10) for p in priors) / len(priors))

        return 0.5

    def _calculate_complexity(self, hypothesis: Dict[str, Any]) -> float:
        """
        Calculate complexity penalty (Occam's razor).

        Simpler explanations preferred. Complexity = number of assumptions.

        Args:
            hypothesis: Variable assignments

        Returns:
            Complexity (0 = simplest)
        """
        return float(len(hypothesis))

    def _find_explained_observations(self,
                                    hypothesis: Dict[str, Any],
                                    observations: List[Observation]) -> Set[str]:
        """Find which observations this hypothesis explains."""
        explained = set()
        for obs in observations:
            if self._probability_of_observation(hypothesis, obs) > 0.3:
                explained.add(obs.variable)
        return explained

    def _find_supporting_evidence(self,
                                 hypothesis: Dict[str, Any],
                                 observations: List[Observation]) -> List[str]:
        """Find evidence supporting this hypothesis."""
        evidence = []
        for obs in observations:
            prob = self._probability_of_observation(hypothesis, obs)
            if prob > 0.5:
                evidence.append(f"{obs.variable}={obs.value} (p={prob:.2f})")
        return evidence


# ============================================================================
# Abductive Reasoner
# ============================================================================

class AbductiveReasoner:
    """
    Main abductive reasoning engine.

    Given observations, generates and scores hypotheses to find
    the best explanation.

    Usage:
        reasoner = AbductiveReasoner(causal_rules, priors)
        explanations = reasoner.explain(observations, max_hypotheses=10)
        best = explanations[0]  # Highest scored
    """

    def __init__(self,
                 causal_rules: List[CausalRule],
                 prior_probabilities: Optional[Dict[str, Dict[Any, float]]] = None):
        """
        Initialize reasoner with causal knowledge.

        Args:
            causal_rules: Domain causal rules
            prior_probabilities: P(variable=value) priors
        """
        self.generator = HypothesisGenerator(causal_rules)
        self.scorer = HypothesisScorer(causal_rules, prior_probabilities)
        self.causal_rules = causal_rules

    def explain(self,
               observations: List[Observation],
               max_hypotheses: int = 10,
               allow_multi_cause: bool = True,
               min_score: float = 0.0) -> List[Hypothesis]:
        """
        Generate best explanations for observations.

        Main entry point for abductive reasoning.

        Algorithm:
        1. Generate candidate hypotheses (backward from observations)
        2. Score each hypothesis (likelihood × prior / complexity)
        3. Rank by score
        4. Return top-k

        Args:
            observations: Observed evidence to explain
            max_hypotheses: Maximum explanations to return
            allow_multi_cause: Allow multiple causes in explanations
            min_score: Minimum score threshold

        Returns:
            Ranked list of hypotheses (best first)
        """
        logger.info(f"Abductive reasoning: Explaining {len(observations)} observations")

        # 1. Generate candidates
        candidate_dicts = self.generator.generate(
            observations,
            max_hypotheses=max_hypotheses * 2,  # Generate extras for filtering
            allow_multi_cause=allow_multi_cause
        )

        logger.info(f"Generated {len(candidate_dicts)} candidate hypotheses")

        # 2. Score each candidate
        hypotheses = []
        for h_dict in candidate_dicts:
            hypothesis = self.scorer.score(h_dict, observations)
            if hypothesis.score() >= min_score:
                hypotheses.append(hypothesis)

        # 3. Rank by score (best first)
        hypotheses.sort(key=lambda h: h.score(), reverse=True)

        logger.info(f"Returning top {min(max_hypotheses, len(hypotheses))} explanations")

        return hypotheses[:max_hypotheses]

    def explain_single(self,
                      observation: Observation,
                      max_hypotheses: int = 5) -> List[Hypothesis]:
        """
        Explain a single observation.

        Convenience method for single observation.

        Args:
            observation: Single observation to explain
            max_hypotheses: Maximum explanations

        Returns:
            Ranked explanations
        """
        return self.explain([observation], max_hypotheses=max_hypotheses)

    def best_explanation(self, observations: List[Observation]) -> Optional[Hypothesis]:
        """
        Get single best explanation.

        Args:
            observations: Evidence to explain

        Returns:
            Best hypothesis or None if no valid explanations
        """
        explanations = self.explain(observations, max_hypotheses=1)
        return explanations[0] if explanations else None

    def compare_hypotheses(self,
                          h1: Dict[str, Any],
                          h2: Dict[str, Any],
                          observations: List[Observation]) -> Tuple[Hypothesis, Hypothesis]:
        """
        Compare two specific hypotheses.

        Useful for hypothesis testing or model comparison.

        Args:
            h1: First hypothesis to compare
            h2: Second hypothesis to compare
            observations: Evidence to evaluate against

        Returns:
            Tuple of scored (h1, h2)
        """
        scored_h1 = self.scorer.score(h1, observations)
        scored_h2 = self.scorer.score(h2, observations)
        return (scored_h1, scored_h2)

    def explain_with_confidence(self,
                               observations: List[Observation],
                               confidence_threshold: float = 0.7) -> Optional[Hypothesis]:
        """
        Get best explanation only if confidence is high enough.

        Args:
            observations: Evidence to explain
            confidence_threshold: Minimum score required

        Returns:
            Best hypothesis if score > threshold, else None
        """
        best = self.best_explanation(observations)
        if best and best.score() >= confidence_threshold:
            return best
        return None


# ============================================================================
# Helper Functions
# ============================================================================

def create_causal_rule(cause: str, cause_val: Any,
                      effect: str, effect_val: Any,
                      strength: float = 0.9) -> CausalRule:
    """
    Convenience function to create causal rule.

    Args:
        cause: Cause variable name
        cause_val: Cause value
        effect: Effect variable name
        effect_val: Effect value
        strength: P(effect | cause)

    Returns:
        CausalRule object
    """
    return CausalRule(
        cause_variable=cause,
        cause_value=cause_val,
        effect_variable=effect,
        effect_value=effect_val,
        strength=strength
    )


def create_observation(variable: str, value: Any, confidence: float = 1.0) -> Observation:
    """
    Convenience function to create observation.

    Args:
        variable: Observed variable
        value: Observed value
        confidence: Observation reliability [0, 1]

    Returns:
        Observation object
    """
    return Observation(variable=variable, value=value, confidence=confidence)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core classes
    'Hypothesis',
    'Observation',
    'CausalRule',
    'HypothesisGenerator',
    'HypothesisScorer',
    'AbductiveReasoner',

    # Helper functions
    'create_causal_rule',
    'create_observation',
]
