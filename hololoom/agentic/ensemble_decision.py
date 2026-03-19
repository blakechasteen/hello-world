"""
Ensemble Decision - Phase 7.2
=============================

Multi-agent result aggregation for ensemble decisions.
Combines outputs from multiple agents using configurable strategies.

Key Features:
- 4 aggregation strategies (MAJORITY_VOTE, CONFIDENCE_WEIGHTED, BEST_CONFIDENCE, UNANIMOUS)
- Agreement scoring to measure consensus
- Flexible response handling (text, numeric, structured)
- Complete aggregation metadata

Author: Claude Code
Date: 2025-12-09
"""

import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EnsembleStrategy(Enum):
    """Aggregation strategies for ensemble decisions."""
    MAJORITY_VOTE = "majority"         # Most common answer wins
    CONFIDENCE_WEIGHTED = "weighted"   # Average weighted by confidence
    BEST_CONFIDENCE = "best"           # Take highest-confidence result
    UNANIMOUS = "unanimous"            # All must agree (fallback to weighted)


@dataclass
class AgentResponse:
    """
    Response from a single agent in the ensemble.

    Attributes:
        agent_id: ID of the responding agent
        response: The agent's response (text, number, or structured data)
        confidence: Agent's confidence in response (0.0-1.0)
        latency_ms: Response latency in milliseconds
        metadata: Additional response metadata
    """
    agent_id: str
    response: Any
    confidence: float
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate confidence is in range."""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class EnsembleResult:
    """
    Result of ensemble aggregation.

    Contains the final response, aggregation details, and all
    individual agent responses for transparency.
    """
    final_response: Any
    strategy_used: EnsembleStrategy
    agreement_score: float  # 0.0-1.0 (how much agents agreed)
    individual_responses: list[AgentResponse]
    aggregation_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_count(self) -> int:
        """Number of agents that responded."""
        return len(self.individual_responses)

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all agents."""
        if not self.individual_responses:
            return 0.0
        return sum(r.confidence for r in self.individual_responses) / len(self.individual_responses)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all agents."""
        if not self.individual_responses:
            return 0.0
        return sum(r.latency_ms for r in self.individual_responses) / len(self.individual_responses)

    @property
    def unanimous(self) -> bool:
        """Whether all agents gave the same response."""
        return self.agreement_score >= 0.99

    def summary(self) -> str:
        """Human-readable summary of the ensemble result."""
        return (
            f"EnsembleResult: strategy={self.strategy_used.value}, "
            f"agents={self.agent_count}, agreement={self.agreement_score:.2f}, "
            f"avg_confidence={self.avg_confidence:.2f}"
        )


class EnsembleAggregator:
    """
    Aggregates results from multiple agents using configurable strategies.

    Supports text responses (majority vote), numeric responses (weighted average),
    and structured responses (confidence-based selection).

    Example:
        aggregator = EnsembleAggregator()

        responses = [
            AgentResponse("agent_1", "Thompson Sampling", 0.9, 150.0),
            AgentResponse("agent_2", "Thompson Sampling", 0.85, 120.0),
            AgentResponse("agent_3", "UCB Algorithm", 0.7, 100.0),
        ]

        result = aggregator.aggregate(responses, EnsembleStrategy.MAJORITY_VOTE)
        print(result.final_response)  # "Thompson Sampling"
        print(result.agreement_score)  # 0.667 (2/3 agree)
    """

    def __init__(
        self,
        min_responses: int = 1,
        unanimous_fallback: EnsembleStrategy = EnsembleStrategy.CONFIDENCE_WEIGHTED
    ):
        """
        Initialize ensemble aggregator.

        Args:
            min_responses: Minimum responses required for aggregation
            unanimous_fallback: Strategy to use if UNANIMOUS fails
        """
        self._min_responses = min_responses
        self._unanimous_fallback = unanimous_fallback

    def aggregate(
        self,
        responses: list[AgentResponse],
        strategy: EnsembleStrategy = EnsembleStrategy.CONFIDENCE_WEIGHTED
    ) -> EnsembleResult:
        """
        Aggregate responses using specified strategy.

        Args:
            responses: List of agent responses
            strategy: Aggregation strategy to use

        Returns:
            EnsembleResult with final response and metadata
        """
        start_time = datetime.now()

        if not responses:
            return EnsembleResult(
                final_response=None,
                strategy_used=strategy,
                agreement_score=0.0,
                individual_responses=[],
                metadata={"error": "No responses to aggregate"}
            )

        if len(responses) < self._min_responses:
            return EnsembleResult(
                final_response=responses[0].response if responses else None,
                strategy_used=strategy,
                agreement_score=1.0 if responses else 0.0,
                individual_responses=responses,
                metadata={"error": f"Below minimum responses ({self._min_responses})"}
            )

        # Compute agreement score first
        agreement = self._compute_agreement(responses)

        # Aggregate based on strategy
        if strategy == EnsembleStrategy.MAJORITY_VOTE:
            final_response = self._majority_vote(responses)
        elif strategy == EnsembleStrategy.CONFIDENCE_WEIGHTED:
            final_response = self._confidence_weighted(responses)
        elif strategy == EnsembleStrategy.BEST_CONFIDENCE:
            final_response = self._best_confidence(responses)
        elif strategy == EnsembleStrategy.UNANIMOUS:
            final_response = self._unanimous(responses)
            if final_response is None:
                # Fallback to alternative strategy
                strategy = self._unanimous_fallback
                final_response = self._confidence_weighted(responses)
        else:
            final_response = self._confidence_weighted(responses)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return EnsembleResult(
            final_response=final_response,
            strategy_used=strategy,
            agreement_score=agreement,
            individual_responses=responses,
            aggregation_time_ms=elapsed_ms,
            metadata={
                "response_count": len(responses),
                "confidence_range": (
                    min(r.confidence for r in responses),
                    max(r.confidence for r in responses)
                ),
                "latency_range": (
                    min(r.latency_ms for r in responses),
                    max(r.latency_ms for r in responses)
                )
            }
        )

    def _compute_agreement(self, responses: list[AgentResponse]) -> float:
        """
        Compute agreement score (how much agents agree).

        For text responses: fraction with most common answer
        For numeric responses: 1 - normalized std deviation
        """
        if len(responses) <= 1:
            return 1.0

        # Try numeric agreement first
        try:
            values = [float(r.response) for r in responses]
            if all(v == values[0] for v in values):
                return 1.0

            mean = statistics.mean(values)
            if mean == 0:
                return 0.0

            std = statistics.stdev(values)
            # Normalize: lower std relative to mean = higher agreement
            cv = std / abs(mean)  # Coefficient of variation
            return max(0.0, 1.0 - min(cv, 1.0))
        except (ValueError, TypeError):
            pass

        # Fall back to text/categorical agreement
        response_strs = [str(r.response).strip().lower() for r in responses]
        counter = Counter(response_strs)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(responses)

    def _majority_vote(self, responses: list[AgentResponse]) -> Any:
        """
        Select response with most votes.

        For ties, prefer the response with higher average confidence.
        """
        # Group by response value (normalized for text)
        response_groups: dict[str, list[AgentResponse]] = {}

        for r in responses:
            key = str(r.response).strip().lower()
            if key not in response_groups:
                response_groups[key] = []
            response_groups[key].append(r)

        # Find most common (with confidence tie-breaker)
        best_key = None
        best_count = 0
        best_confidence = 0.0

        for key, group in response_groups.items():
            count = len(group)
            avg_conf = sum(r.confidence for r in group) / len(group)

            if count > best_count or (count == best_count and avg_conf > best_confidence):
                best_key = key
                best_count = count
                best_confidence = avg_conf

        # Return original response (not normalized key)
        if best_key:
            for r in responses:
                if str(r.response).strip().lower() == best_key:
                    return r.response

        return responses[0].response

    def _confidence_weighted(self, responses: list[AgentResponse]) -> Any:
        """
        Compute weighted average/selection by confidence.

        For numeric responses: weighted average
        For text responses: highest total confidence weight
        """
        # Try numeric weighted average first
        try:
            values = [(float(r.response), r.confidence) for r in responses]
            total_weight = sum(conf for _, conf in values)
            if total_weight == 0:
                return responses[0].response

            weighted_sum = sum(val * conf for val, conf in values)
            return weighted_sum / total_weight
        except (ValueError, TypeError):
            pass

        # For text: group and sum confidence
        confidence_by_response: dict[str, float] = {}
        original_by_key: dict[str, Any] = {}

        for r in responses:
            key = str(r.response).strip().lower()
            if key not in confidence_by_response:
                confidence_by_response[key] = 0.0
                original_by_key[key] = r.response
            confidence_by_response[key] += r.confidence

        # Return response with highest total confidence
        best_key = max(confidence_by_response, key=confidence_by_response.get)
        return original_by_key[best_key]

    def _best_confidence(self, responses: list[AgentResponse]) -> Any:
        """
        Select response with highest confidence.
        """
        best = max(responses, key=lambda r: r.confidence)
        return best.response

    def _unanimous(self, responses: list[AgentResponse]) -> Any | None:
        """
        Return response only if all agents agree.

        Returns None if not unanimous.
        """
        if not responses:
            return None

        # Normalize responses for comparison
        first = str(responses[0].response).strip().lower()
        for r in responses[1:]:
            if str(r.response).strip().lower() != first:
                return None

        return responses[0].response

    def compute_agreement(self, responses: list[AgentResponse]) -> float:
        """
        Public method to compute agreement score.

        Args:
            responses: List of agent responses

        Returns:
            Agreement score (0.0-1.0)
        """
        return self._compute_agreement(responses)


class NumericEnsemble(EnsembleAggregator):
    """
    Specialized ensemble for numeric responses.

    Provides additional statistics and outlier handling.
    """

    def __init__(
        self,
        outlier_threshold: float = 2.0,
        remove_outliers: bool = False
    ):
        """
        Initialize numeric ensemble.

        Args:
            outlier_threshold: Standard deviations for outlier detection
            remove_outliers: Whether to remove outliers before aggregation
        """
        super().__init__()
        self._outlier_threshold = outlier_threshold
        self._remove_outliers = remove_outliers

    def aggregate_numeric(
        self,
        responses: list[AgentResponse],
        strategy: EnsembleStrategy = EnsembleStrategy.CONFIDENCE_WEIGHTED
    ) -> EnsembleResult:
        """
        Aggregate numeric responses with outlier handling.
        """
        if self._remove_outliers and len(responses) >= 3:
            responses = self._filter_outliers(responses)

        result = self.aggregate(responses, strategy)

        # Add numeric-specific metadata
        try:
            values = [float(r.response) for r in responses]
            result.metadata.update({
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
                "range": (min(values), max(values))
            })
        except (ValueError, TypeError):
            pass

        return result

    def _filter_outliers(self, responses: list[AgentResponse]) -> list[AgentResponse]:
        """Remove outlier responses based on z-score."""
        try:
            values = [float(r.response) for r in responses]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) >= 2 else 0.0

            if std == 0:
                return responses

            # Keep responses within threshold
            filtered = []
            for r in responses:
                z = abs(float(r.response) - mean) / std
                if z <= self._outlier_threshold:
                    filtered.append(r)

            return filtered if filtered else responses
        except (ValueError, TypeError):
            return responses


class TextEnsemble(EnsembleAggregator):
    """
    Specialized ensemble for text responses.

    Provides semantic similarity option (if embedder available).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        normalize_whitespace: bool = True
    ):
        """
        Initialize text ensemble.

        Args:
            similarity_threshold: Threshold for considering responses "similar"
            normalize_whitespace: Whether to normalize whitespace before comparison
        """
        super().__init__()
        self._similarity_threshold = similarity_threshold
        self._normalize_whitespace = normalize_whitespace

    def aggregate_text(
        self,
        responses: list[AgentResponse],
        strategy: EnsembleStrategy = EnsembleStrategy.MAJORITY_VOTE
    ) -> EnsembleResult:
        """
        Aggregate text responses with optional normalization.
        """
        # Normalize responses if configured
        if self._normalize_whitespace:
            for r in responses:
                if isinstance(r.response, str):
                    r.response = " ".join(r.response.split())

        result = self.aggregate(responses, strategy)

        # Add text-specific metadata
        unique_responses = set(str(r.response) for r in responses)
        result.metadata.update({
            "unique_responses": len(unique_responses),
            "responses_sample": list(unique_responses)[:5]
        })

        return result


# Convenience functions
def aggregate_responses(
    responses: list[AgentResponse],
    strategy: EnsembleStrategy = EnsembleStrategy.CONFIDENCE_WEIGHTED
) -> EnsembleResult:
    """
    Convenience function to aggregate agent responses.

    Args:
        responses: List of agent responses
        strategy: Aggregation strategy

    Returns:
        EnsembleResult with aggregated response
    """
    aggregator = EnsembleAggregator()
    return aggregator.aggregate(responses, strategy)


def create_agent_response(
    agent_id: str,
    response: Any,
    confidence: float,
    latency_ms: float = 0.0
) -> AgentResponse:
    """
    Convenience function to create an agent response.

    Args:
        agent_id: Agent identifier
        response: The response value
        confidence: Confidence level (0.0-1.0)
        latency_ms: Response latency

    Returns:
        AgentResponse object
    """
    return AgentResponse(
        agent_id=agent_id,
        response=response,
        confidence=confidence,
        latency_ms=latency_ms
    )
