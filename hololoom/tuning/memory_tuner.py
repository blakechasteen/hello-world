"""
Memory Tuner - Agent 4

Automatically optimizes memory retrieval parameters using Thompson Sampling.

Strategy:
1. Measure retrieval quality (precision@k, recall@k, context usefulness)
2. Thompson Sampling learns optimal retrieval_k per query type
3. Balance retrieval quality vs latency
4. Separate bandits for different query complexity levels

Eliminates Parameters:
- retrieval_k (number of memories to retrieve)
- max_memories (maximum memory limit)
- context_window_size (context size for decision making)
"""

from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import numpy as np
from hololoom.tuning.base import TuningAgent, ThompsonBandit, SafeParameter

# Retrieval k values (5 arms)
RETRIEVAL_K_VALUES = [3, 5, 10, 20, 50]  # 5 discrete values

# Query complexity categories
class QueryComplexity(Enum):
    SIMPLE = "simple"      # Factual, direct queries
    MODERATE = "moderate"  # Multi-step reasoning
    COMPLEX = "complex"    # Research, open-ended

# Baseline retrieval parameters
BASELINE_RETRIEVAL_K = {
    QueryComplexity.SIMPLE: 5,
    QueryComplexity.MODERATE: 10,
    QueryComplexity.COMPLEX: 20,
}

# Safe ranges
SAFE_RETRIEVAL_RANGES = {
    'retrieval_k': (1, 100),
    'max_memories': (100, 10000),
    'context_window': (512, 8192),
}


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality evaluation."""
    relevant_retrieved: int = 0   # Relevant memories retrieved
    total_retrieved: int = 0      # Total memories retrieved
    total_relevant: int = 0       # Total relevant memories available
    retrieval_latency_ms: float = 0.0
    context_usefulness: float = 0.0  # 0.0-1.0, subjective quality

    @property
    def precision_at_k(self) -> float:
        """Precision@k: relevant_retrieved / total_retrieved."""
        return self.relevant_retrieved / self.total_retrieved if self.total_retrieved > 0 else 0.0

    @property
    def recall_at_k(self) -> float:
        """Recall@k: relevant_retrieved / total_relevant."""
        return self.relevant_retrieved / self.total_relevant if self.total_relevant > 0 else 0.0

    @property
    def f1_at_k(self) -> float:
        """F1@k: harmonic mean of precision and recall."""
        p = self.precision_at_k
        r = self.recall_at_k
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def efficiency(self) -> float:
        """Retrieval efficiency: precision / latency."""
        return self.precision_at_k / (self.retrieval_latency_ms + 1.0)  # +1 to avoid div by 0

    def quality_score(self) -> float:
        """
        Multi-objective quality score.

        Formula:
        quality = 0.5 * f1@k + 0.3 * context_usefulness + 0.2 * efficiency

        Balances:
        - 50% retrieval accuracy (F1 score)
        - 30% context usefulness (how helpful is the context)
        - 20% efficiency (fast + accurate)
        """
        return (
            0.5 * self.f1_at_k +
            0.3 * self.context_usefulness +
            0.2 * min(self.efficiency, 1.0)  # Cap efficiency at 1.0
        )


class MemoryTuner(TuningAgent):
    """
    Agent 4: Memory Tuner

    Optimizes retrieval parameters using Thompson Sampling.
    Eliminates manual retrieval configuration.
    """

    def __init__(self):
        super().__init__('memory_tuner')

        # Metrics history (last 1000 measurements per complexity level)
        self.metrics_history: Dict[QueryComplexity, deque] = {
            complexity: deque(maxlen=1000)
            for complexity in QueryComplexity
        }

        # Thompson Sampling bandits (one per complexity level)
        self.bandits: Dict[QueryComplexity, ThompsonBandit] = {
            complexity: ThompsonBandit(n_arms=len(RETRIEVAL_K_VALUES))
            for complexity in QueryComplexity
        }

        # Current retrieval_k indices (selected arm per complexity)
        self.current_k_indices: Dict[QueryComplexity, int] = {
            complexity: RETRIEVAL_K_VALUES.index(BASELINE_RETRIEVAL_K[complexity])
            for complexity in QueryComplexity
        }

        # Safe parameters
        self.safe_params: Dict[str, SafeParameter] = {
            'simple_k': SafeParameter(
                name='simple_retrieval_k',
                current_value=float(BASELINE_RETRIEVAL_K[QueryComplexity.SIMPLE]),
                min_value=1.0,
                max_value=100.0,
                max_change_percent=0.5,  # Allow 50% change (retrieval k can vary more)
            ),
            'moderate_k': SafeParameter(
                name='moderate_retrieval_k',
                current_value=float(BASELINE_RETRIEVAL_K[QueryComplexity.MODERATE]),
                min_value=1.0,
                max_value=100.0,
                max_change_percent=0.5,
            ),
            'complex_k': SafeParameter(
                name='complex_retrieval_k',
                current_value=float(BASELINE_RETRIEVAL_K[QueryComplexity.COMPLEX]),
                min_value=1.0,
                max_value=100.0,
                max_change_percent=0.5,
            ),
        }

        # Query complexity distribution (for deciding which to tune)
        self.complexity_distribution = defaultdict(int)
        self.current_tuning_complexity = QueryComplexity.SIMPLE

    def classify_query_complexity(self, query_text: str) -> QueryComplexity:
        """
        Classify query complexity (simple heuristic).

        Args:
            query_text: Query text

        Returns:
            QueryComplexity enum
        """
        # Simple heuristic based on query length and keywords
        length = len(query_text.split())

        if length < 10:
            return QueryComplexity.SIMPLE
        elif length < 20:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX

    def get_adaptive_retrieval_k(self, complexity: QueryComplexity) -> int:
        """
        Get current retrieval_k for given complexity level.

        Args:
            complexity: QueryComplexity enum

        Returns:
            Current retrieval_k value
        """
        arm_idx = self.current_k_indices[complexity]
        return RETRIEVAL_K_VALUES[arm_idx]

    async def measure_performance(self) -> Dict[str, Any]:
        """
        Measure retrieval performance metrics.

        Returns:
            Dict with current retrieval metrics per complexity level
        """
        metrics = {}

        for complexity in QueryComplexity:
            param_name = f'{complexity.value}_k'

            if len(self.metrics_history[complexity]) > 0:
                recent_metrics = list(self.metrics_history[complexity])[-100:]  # Last 100

                # Aggregate metrics
                total_relevant = sum(m.relevant_retrieved for m in recent_metrics)
                total_retrieved = sum(m.total_retrieved for m in recent_metrics)
                total_available = sum(m.total_relevant for m in recent_metrics)
                avg_latency = np.mean([m.retrieval_latency_ms for m in recent_metrics])
                avg_usefulness = np.mean([m.context_usefulness for m in recent_metrics])

                agg = RetrievalMetrics(
                    relevant_retrieved=total_relevant,
                    total_retrieved=total_retrieved,
                    total_relevant=total_available,
                    retrieval_latency_ms=avg_latency,
                    context_usefulness=avg_usefulness,
                )

                metrics[param_name] = {
                    'precision_at_k': agg.precision_at_k,
                    'recall_at_k': agg.recall_at_k,
                    'f1_at_k': agg.f1_at_k,
                    'efficiency': agg.efficiency,
                    'quality': agg.quality_score(),
                    'current_k': int(self.safe_params[param_name].current_value),
                    'avg_latency_ms': avg_latency,
                }
            else:
                # No data yet
                metrics[param_name] = {
                    'precision_at_k': 0.0,
                    'recall_at_k': 0.0,
                    'f1_at_k': 0.0,
                    'efficiency': 0.0,
                    'quality': 0.0,
                    'current_k': int(self.safe_params[param_name].current_value),
                    'avg_latency_ms': 0.0,
                }

        return metrics

    async def tune_parameters(self) -> Dict[str, Any]:
        """
        Tune retrieval_k based on Thompson Sampling.

        Strategy: Tune one complexity level at a time (round-robin).

        Returns:
            Dict with proposed retrieval_k changes
        """
        proposals = {}

        # Select which complexity level to tune (focus on most common)
        if sum(self.complexity_distribution.values()) > 0:
            # Pick most frequent complexity
            self.current_tuning_complexity = max(
                self.complexity_distribution.items(),
                key=lambda x: x[1]
            )[0]
        else:
            # Round-robin if no distribution yet
            complexities = list(QueryComplexity)
            idx = complexities.index(self.current_tuning_complexity)
            self.current_tuning_complexity = complexities[(idx + 1) % len(complexities)]

        complexity = self.current_tuning_complexity
        param_name = f'{complexity.value}_k'

        # Sample from Thompson Sampling bandit
        arm_idx = self.bandits[complexity].sample()
        proposed_k = RETRIEVAL_K_VALUES[arm_idx]

        # Apply safety checks
        safe_k = self.safe_params[param_name].propose(float(proposed_k))

        proposals[param_name] = {
            'complexity': complexity.value,
            'old_k': int(self.safe_params[param_name].current_value),
            'proposed_k': int(safe_k),
            'arm_selected': arm_idx,
        }

        # Update current index (for feedback)
        self.current_k_indices[complexity] = arm_idx

        # Update parameter value
        self.safe_params[param_name].current_value = safe_k

        return proposals

    def record_retrieval_metrics(self, query_text: str, metrics: RetrievalMetrics):
        """
        Record retrieval metrics for tuning.

        Args:
            query_text: Query text (for complexity classification)
            metrics: RetrievalMetrics object
        """
        complexity = self.classify_query_complexity(query_text)
        self.metrics_history[complexity].append(metrics)
        self.complexity_distribution[complexity] += 1

    def update_bandits(self, baseline_metrics: Dict[str, Any], new_metrics: Dict[str, Any]):
        """
        Update Thompson Sampling bandits based on quality change.

        Args:
            baseline_metrics: Metrics before tuning
            new_metrics: Metrics after tuning
        """
        for complexity in QueryComplexity:
            param_name = f'{complexity.value}_k'

            baseline_quality = baseline_metrics.get(param_name, {}).get('quality', 0.0)
            new_quality = new_metrics.get(param_name, {}).get('quality', 0.0)

            # Calculate improvement
            improvement = new_quality - baseline_quality

            # Update bandit (success if improvement > 0)
            arm_idx = self.current_k_indices[complexity]
            success = improvement > 0
            confidence = abs(improvement)

            self.bandits[complexity].update(arm_idx, success, confidence)

    async def run_tuning_cycle(self) -> Dict[str, Any]:
        """
        Run complete tuning cycle for retrieval parameters.

        Returns:
            Dict with tuning results
        """
        # Measure baseline performance
        baseline_metrics = await self.measure_performance()

        # Tune parameters (one complexity level at a time)
        proposals = await self.tune_parameters()

        # Measure new performance (simulate - would need real measurements)
        new_metrics = await self.measure_performance()

        # Update bandits based on results
        self.update_bandits(baseline_metrics, new_metrics)

        # Calculate impact
        impact = self._calculate_impact(baseline_metrics, new_metrics)

        return {
            'agent': self.name,
            'baseline': baseline_metrics,
            'proposals': proposals,
            'new_metrics': new_metrics,
            'impact': impact,
            'bandit_stats': self.get_bandit_stats(),
        }

    def _calculate_impact(self, baseline: Dict[str, Any], new: Dict[str, Any]) -> float:
        """
        Calculate impact of tuning changes.

        Args:
            baseline: Baseline metrics
            new: New metrics after tuning

        Returns:
            Scalar impact score (average quality improvement)
        """
        quality_deltas = []

        for complexity in QueryComplexity:
            param_name = f'{complexity.value}_k'

            baseline_quality = baseline.get(param_name, {}).get('quality', 0.0)
            new_quality = new.get(param_name, {}).get('quality', 0.0)

            quality_deltas.append(new_quality - baseline_quality)

        # Overall impact (average quality improvement)
        return float(np.mean(quality_deltas)) if quality_deltas else 0.0

    def get_bandit_stats(self) -> Dict[str, Any]:
        """
        Get Thompson Sampling bandit statistics.

        Returns:
            Dict with bandit stats per complexity level
        """
        stats = {}

        for complexity, bandit in self.bandits.items():
            arm_stats = []
            for i, k_value in enumerate(RETRIEVAL_K_VALUES):
                expected_reward = bandit.alpha[i] / (bandit.alpha[i] + bandit.beta[i])
                n_pulls = int(bandit.alpha[i] + bandit.beta[i] - 2)

                arm_stats.append({
                    'retrieval_k': k_value,
                    'expected_reward': expected_reward,
                    'n_pulls': n_pulls,
                    'alpha': float(bandit.alpha[i]),
                    'beta': float(bandit.beta[i]),
                })

            stats[complexity.value] = arm_stats

        return stats

    async def persist_state(self):
        """
        Persist tuning state (handled by TuningStateManager in coordinator).
        """
        pass  # State persistence handled by coordinator

    def get_state(self) -> Dict[str, Any]:
        """Serialize agent state for persistence."""
        return {
            'total_tuning_cycles': self.total_tuning_cycles,
            'successful_tunings': self.successful_tunings,
            'safe_params': {
                param_name: {
                    'current_value': param.current_value,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                }
                for param_name, param in self.safe_params.items()
            },
            'bandits': {
                complexity.value: bandit.get_state()
                for complexity, bandit in self.bandits.items()
            },
            'current_k_indices': {
                complexity.value: idx
                for complexity, idx in self.current_k_indices.items()
            },
            'complexity_distribution': dict(self.complexity_distribution),
            'current_tuning_complexity': self.current_tuning_complexity.value,
            'metrics_samples': {
                complexity.value: [
                    {
                        'relevant_retrieved': m.relevant_retrieved,
                        'total_retrieved': m.total_retrieved,
                        'total_relevant': m.total_relevant,
                        'retrieval_latency_ms': m.retrieval_latency_ms,
                        'context_usefulness': m.context_usefulness,
                    }
                    for m in list(history)[-100:]  # Save last 100 samples
                ]
                for complexity, history in self.metrics_history.items()
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """Restore agent from serialized state."""
        self.total_tuning_cycles = state.get('total_tuning_cycles', 0)
        self.successful_tunings = state.get('successful_tunings', 0)

        # Restore safe parameters
        for param_name, param_state in state.get('safe_params', {}).items():
            if param_name in self.safe_params:
                self.safe_params[param_name].current_value = param_state['current_value']

        # Restore bandits
        for complexity_str, bandit_state in state.get('bandits', {}).items():
            complexity = QueryComplexity(complexity_str)
            if complexity in self.bandits:
                self.bandits[complexity] = ThompsonBandit.from_state(bandit_state)

        # Restore current k indices
        for complexity_str, idx in state.get('current_k_indices', {}).items():
            complexity = QueryComplexity(complexity_str)
            self.current_k_indices[complexity] = idx

        # Restore complexity distribution
        self.complexity_distribution = defaultdict(
            int,
            {QueryComplexity(k): v for k, v in state.get('complexity_distribution', {}).items()}
        )

        # Restore current tuning complexity
        complexity_str = state.get('current_tuning_complexity', 'simple')
        self.current_tuning_complexity = QueryComplexity(complexity_str)

        # Restore metrics samples
        for complexity_str, samples in state.get('metrics_samples', {}).items():
            complexity = QueryComplexity(complexity_str)
            if complexity in self.metrics_history:
                self.metrics_history[complexity] = deque(
                    [RetrievalMetrics(**s) for s in samples],
                    maxlen=1000
                )

        print(f"[MemoryTuner] Loaded state: {self.total_tuning_cycles} cycles, "
              f"{self.successful_tunings} successful")


def create_memory_tuner() -> MemoryTuner:
    """
    Factory function to create MemoryTuner.

    Returns:
        Initialized MemoryTuner instance
    """
    return MemoryTuner()
