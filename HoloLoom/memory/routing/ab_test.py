"""
A/B Test Router - Compare Multiple Routing Strategies
====================================================

Test multiple routing strategies simultaneously and determine winner.
"""

import random
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from .protocol import (
    RoutingStrategy,
    ExperimentalStrategy,
    RoutingDecision,
    RoutingOutcome,
    BackendType
)


@dataclass
class StrategyVariant:
    """A strategy variant in an A/B test."""
    name: str
    strategy: RoutingStrategy
    weight: float = 1.0
    outcomes: List[RoutingOutcome] = field(default_factory=list)

    def record_outcome(self, outcome: RoutingOutcome):
        """Record outcome for this variant."""
        self.outcomes.append(outcome)
        self.strategy.record_outcome(outcome)

    def get_metrics(self) -> Dict[str, float]:
        """Compute performance metrics."""
        if not self.outcomes:
            return {
                'avg_relevance': 0.0,
                'avg_latency': 0.0,
                'success_rate': 0.0,
                'total_queries': 0
            }

        relevances = [o.avg_relevance for o in self.outcomes]
        latencies = [o.latency_ms for o in self.outcomes]
        successes = sum(1 for o in self.outcomes if o.avg_relevance > 0.7)

        return {
            'avg_relevance': sum(relevances) / len(relevances),
            'avg_latency': sum(latencies) / len(latencies),
            'success_rate': successes / len(self.outcomes),
            'total_queries': len(self.outcomes)
        }


class ABTestRouter:
    """
    A/B test multiple routing strategies.

    Routes queries to different strategies with specified weights,
    tracks outcomes, and determines statistical winner.
    """

    def __init__(self):
        self.variants: Dict[str, StrategyVariant] = {}
        self.query_count = 0

    def add_strategy(
        self,
        name: str,
        strategy: RoutingStrategy,
        weight: float = 1.0
    ):
        """Add strategy variant to experiment."""
        self.variants[name] = StrategyVariant(
            name=name,
            strategy=strategy,
            weight=weight
        )

    def select_backend(
        self,
        query: str,
        available_backends: List[BackendType],
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Select backend using weighted random variant selection.

        Returns decision with metadata indicating which variant was used.
        """
        if not self.variants:
            raise RuntimeError("No strategy variants added to experiment")

        # Weighted random selection of variant
        variant = self._select_variant()

        # Get decision from selected variant
        decision = variant.strategy.select_backend(
            query,
            available_backends,
            context
        )

        # Tag decision with variant name
        decision.metadata['ab_variant'] = variant.name

        self.query_count += 1

        return decision

    def _select_variant(self) -> StrategyVariant:
        """Select variant using weighted random selection."""
        total_weight = sum(v.weight for v in self.variants.values())
        rand = random.uniform(0, total_weight)

        cumulative = 0
        for variant in self.variants.values():
            cumulative += variant.weight
            if rand <= cumulative:
                return variant

        # Fallback to first variant
        return list(self.variants.values())[0]

    def record_outcome(self, outcome: RoutingOutcome):
        """Record outcome to appropriate variant."""
        variant_name = outcome.decision.metadata.get('ab_variant')

        if variant_name and variant_name in self.variants:
            self.variants[variant_name].record_outcome(outcome)

    def get_winner(self) -> str:
        """
        Determine winning strategy based on metrics.

        Winner = highest success rate (relevance > 0.7) with minimum 10 queries.
        """
        qualified = [
            (name, variant)
            for name, variant in self.variants.items()
            if len(variant.outcomes) >= 10
        ]

        if not qualified:
            return "insufficient_data"

        # Sort by success rate
        ranked = sorted(
            qualified,
            key=lambda x: x[1].get_metrics()['success_rate'],
            reverse=True
        )

        return ranked[0][0]

    def get_statistics(self) -> Dict[str, Any]:
        """Get A/B test statistics."""
        return {
            'total_queries': self.query_count,
            'variants': {
                name: variant.get_metrics()
                for name, variant in self.variants.items()
            },
            'winner': self.get_winner()
        }


class RoutingExperiment:
    """
    High-level experiment manager for routing strategies.

    Usage:
        experiment = RoutingExperiment()
        experiment.add_baseline(rule_based_router)
        experiment.add_challenger("learned_v1", learned_router)

        # Run experiment
        for query in test_queries:
            decision = experiment.route(query, available_backends)
            # ... execute query ...
            experiment.record(outcome)

        # Get results
        winner = experiment.get_winner()
        report = experiment.generate_report()
    """

    def __init__(self):
        self.router = ABTestRouter()
        self.baseline_name: Optional[str] = None

    def add_baseline(
        self,
        strategy: RoutingStrategy,
        name: str = "baseline"
    ):
        """Add baseline strategy (usually rule-based)."""
        self.router.add_strategy(name, strategy, weight=0.5)
        self.baseline_name = name

    def add_challenger(
        self,
        name: str,
        strategy: RoutingStrategy,
        weight: float = 0.5
    ):
        """Add challenger strategy to test against baseline."""
        self.router.add_strategy(name, strategy, weight=weight)

    def route(
        self,
        query: str,
        available_backends: List[BackendType],
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Route query using experiment."""
        return self.router.select_backend(query, available_backends, context)

    def record(self, outcome: RoutingOutcome):
        """Record outcome."""
        self.router.record_outcome(outcome)

    def get_winner(self) -> str:
        """Get winning strategy."""
        return self.router.get_winner()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        stats = self.router.get_statistics()
        variants = stats['variants']

        # Compute lift over baseline
        if self.baseline_name and self.baseline_name in variants:
            baseline_success = variants[self.baseline_name]['success_rate']

            for name, metrics in variants.items():
                if name != self.baseline_name:
                    lift = (
                        (metrics['success_rate'] - baseline_success) / baseline_success * 100
                        if baseline_success > 0 else 0
                    )
                    metrics['lift_over_baseline'] = lift

        return {
            'experiment_stats': stats,
            'winner': stats['winner'],
            'baseline': self.baseline_name,
            'recommendation': self._generate_recommendation(stats)
        }

    def _generate_recommendation(self, stats: Dict[str, Any]) -> str:
        """Generate human-readable recommendation."""
        winner = stats['winner']

        if winner == "insufficient_data":
            return "Need more data to determine winner (minimum 10 queries per variant)"

        if winner == self.baseline_name:
            return f"Baseline ({self.baseline_name}) remains best strategy"

        variants = stats['variants']
        winner_metrics = variants[winner]
        baseline_metrics = variants.get(self.baseline_name, {})

        if baseline_metrics:
            lift = winner_metrics.get('lift_over_baseline', 0)
            return (
                f"Strategy '{winner}' wins with {lift:.1f}% lift over baseline. "
                f"Recommend rollout to production."
            )

        return f"Strategy '{winner}' wins with {winner_metrics['success_rate']:.1%} success rate"