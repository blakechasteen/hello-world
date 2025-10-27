"""
Learned Router - ML-Based Backend Selection
==========================================

Uses Thompson Sampling to learn optimal routing from outcomes.
Adapts to actual performance rather than hardcoded rules.
"""

import json
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path

from .protocol import (
    RoutingStrategy,
    LearnableStrategy,
    RoutingDecision,
    RoutingOutcome,
    BackendType,
    QueryType
)


class ThompsonBandit:
    """
    Multi-armed bandit for backend selection using Thompson Sampling.

    Each backend is an "arm" - we learn which arm gives best results
    for different query types.
    """

    def __init__(self, backends: List[BackendType]):
        self.backends = backends

        # Beta distribution parameters for each backend
        # successes (alpha) and failures (beta)
        self.alpha = {backend: 1.0 for backend in backends}
        self.beta = {backend: 1.0 for backend in backends}

    def select(self) -> BackendType:
        """Sample from posterior distributions and select best."""
        import random

        samples = {}
        for backend in self.backends:
            # Sample from Beta(alpha, beta) distribution
            sample = random.betavariate(self.alpha[backend], self.beta[backend])
            samples[backend] = sample

        # Return backend with highest sample
        return max(samples, key=samples.get)

    def update(self, backend: BackendType, success: bool):
        """Update posterior based on outcome."""
        if success:
            self.alpha[backend] += 1
        else:
            self.beta[backend] += 1

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get bandit statistics."""
        stats = {}
        for backend in self.backends:
            total = self.alpha[backend] + self.beta[backend]
            mean = self.alpha[backend] / total
            stats[backend.value] = {
                'mean_reward': mean,
                'alpha': self.alpha[backend],
                'beta': self.beta[backend],
                'total_trials': total - 2  # Subtract initial priors
            }
        return stats


class LearnedRouter:
    """
    Learned routing strategy using Thompson Sampling.

    Features:
    - Separate bandit for each query type
    - Learns optimal backend per query type
    - Adapts to actual performance over time
    - Can save/load learned parameters
    """

    def __init__(self, backends: Optional[List[BackendType]] = None):
        self.backends = backends or [
            BackendType.NEO4J,
            BackendType.QDRANT,
            BackendType.MEM0,
            BackendType.INMEMORY,
        ]

        # One bandit per query type
        self.bandits: Dict[QueryType, ThompsonBandit] = {
            query_type: ThompsonBandit(self.backends)
            for query_type in QueryType
        }

        self.outcomes: List[RoutingOutcome] = []

        # Simple query classifier (can be replaced with ML model)
        self.rule_classifier = self._create_rule_classifier()

    def _create_rule_classifier(self):
        """Create simple rule-based query classifier."""
        from .rule_based import RuleBasedRouter
        router = RuleBasedRouter()
        return router._classify_query

    def select_backend(
        self,
        query: str,
        available_backends: List[BackendType],
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Select backend using Thompson Sampling."""

        # Classify query type
        query_type = self.rule_classifier(query)

        # Get bandit for this query type
        bandit = self.bandits[query_type]

        # Filter bandit to only available backends
        available_in_bandit = [b for b in bandit.backends if b in available_backends]

        if not available_in_bandit:
            # Fallback to first available
            backend = available_backends[0]
            confidence = 0.3
        else:
            # Use temporary bandit with only available backends
            temp_bandit = ThompsonBandit(available_in_bandit)
            temp_bandit.alpha = {b: bandit.alpha[b] for b in available_in_bandit}
            temp_bandit.beta = {b: bandit.beta[b] for b in available_in_bandit}

            backend = temp_bandit.select()

            # Confidence = mean reward of selected backend
            total = bandit.alpha[backend] + bandit.beta[backend]
            confidence = bandit.alpha[backend] / total if total > 0 else 0.5

        # Build alternatives (other backends sorted by mean reward)
        bandit_stats = bandit.get_stats()
        alternatives = sorted(
            [b for b in available_backends if b != backend],
            key=lambda b: bandit_stats[b.value]['mean_reward'],
            reverse=True
        )

        return RoutingDecision(
            backend_type=backend,
            confidence=confidence,
            query_type=query_type,
            reasoning=f"Thompson Sampling for {query_type.value} queries (learned)",
            alternatives=alternatives,
            metadata={
                'strategy': 'learned_thompson',
                'bandit_stats': bandit_stats,
                'trials': bandit_stats[backend.value]['total_trials']
            }
        )

    def record_outcome(self, outcome: RoutingOutcome):
        """Record outcome and update bandit."""
        self.outcomes.append(outcome)

        # Define success: good relevance and reasonable latency
        success = outcome.avg_relevance > 0.7 and outcome.latency_ms < 2000

        # Update corresponding bandit
        query_type = outcome.decision.query_type
        backend = outcome.decision.backend_type

        self.bandits[query_type].update(backend, success)

    def get_statistics(self) -> Dict[str, Any]:
        """Get learned routing statistics."""
        stats = {
            'total_outcomes': len(self.outcomes),
            'bandits': {
                query_type.value: bandit.get_stats()
                for query_type, bandit in self.bandits.items()
            }
        }

        # Overall accuracy
        if self.outcomes:
            successful = sum(1 for o in self.outcomes if o.avg_relevance > 0.7)
            stats['overall_accuracy'] = successful / len(self.outcomes)

        return stats

    def train(self, outcomes: List[RoutingOutcome]):
        """Train from batch of historical outcomes."""
        for outcome in outcomes:
            self.record_outcome(outcome)

    def save(self, path: str):
        """Save learned bandit parameters."""
        data = {
            'bandits': {
                query_type.value: {
                    'alpha': {b.value: bandit.alpha[b] for b in bandit.backends},
                    'beta': {b.value: bandit.beta[b] for b in bandit.backends}
                }
                for query_type, bandit in self.bandits.items()
            },
            'total_outcomes': len(self.outcomes)
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load learned bandit parameters."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Restore bandit parameters
        for query_type_str, bandit_data in data['bandits'].items():
            query_type = QueryType(query_type_str)
            bandit = self.bandits[query_type]

            # Restore alpha and beta
            for backend_str, alpha in bandit_data['alpha'].items():
                backend = BackendType(backend_str)
                bandit.alpha[backend] = alpha

            for backend_str, beta in bandit_data['beta'].items():
                backend = BackendType(backend_str)
                bandit.beta[backend] = beta