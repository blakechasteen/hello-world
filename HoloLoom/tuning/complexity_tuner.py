"""
Complexity Tuner - Agent 5

Automatically selects optimal execution mode (BARE/FAST/FUSED) using Thompson Sampling.

Strategy:
1. Measure query characteristics (length, keywords, etc.)
2. Thompson Sampling learns optimal mode per query type
3. Balance latency vs quality tradeoff
4. Separate bandits for different query complexity levels

Eliminates Parameters:
- execution_mode (BARE/FAST/FUSED selection)
- complexity_threshold (when to switch modes)
- auto_mode_selection (boolean flag)
"""

from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import numpy as np
from HoloLoom.tuning.base import TuningAgent, ThompsonBandit, SafeParameter

# Execution modes (3 arms)
class ExecutionMode(Enum):
    BARE = "bare"      # Minimal processing (~50ms)
    FAST = "fast"      # Balanced (~150ms)
    FUSED = "fused"    # Full processing (~300ms)

# Mode mapping to arm index
MODE_TO_ARM = {
    ExecutionMode.BARE: 0,
    ExecutionMode.FAST: 1,
    ExecutionMode.FUSED: 2,
}

ARM_TO_MODE = {
    0: ExecutionMode.BARE,
    1: ExecutionMode.FAST,
    2: ExecutionMode.FUSED,
}

# Baseline modes per query complexity
BASELINE_MODES = {
    'simple': ExecutionMode.FAST,      # Simple queries default to FAST
    'moderate': ExecutionMode.FAST,    # Moderate queries default to FAST
    'complex': ExecutionMode.FUSED,    # Complex queries default to FUSED
}

# Expected latencies (for normalization)
EXPECTED_LATENCY = {
    ExecutionMode.BARE: 50.0,
    ExecutionMode.FAST: 150.0,
    ExecutionMode.FUSED: 300.0,
}

# Query complexity categories (reuse from memory_tuner)
class QueryComplexity(Enum):
    SIMPLE = "simple"      # Factual, direct queries
    MODERATE = "moderate"  # Multi-step reasoning
    COMPLEX = "complex"    # Research, open-ended


@dataclass
class ComplexityMetrics:
    """Metrics for execution mode evaluation."""
    latency_ms: float = 0.0       # Query latency
    quality_score: float = 0.0    # Quality/confidence (0.0-1.0)
    mode: ExecutionMode = ExecutionMode.FAST
    query_complexity: QueryComplexity = QueryComplexity.MODERATE

    @property
    def efficiency(self) -> float:
        """Efficiency: quality / latency."""
        return self.quality_score / (self.latency_ms + 1.0) if self.latency_ms > 0 else 0.0

    @property
    def latency_ratio(self) -> float:
        """Latency ratio: actual / expected."""
        expected = EXPECTED_LATENCY[self.mode]
        return self.latency_ms / expected if expected > 0 else 1.0

    def quality_metric(self) -> float:
        """
        Multi-objective quality metric.

        Formula:
        quality = 0.5 * quality_score + 0.3 * efficiency + 0.2 * (1 - latency_ratio)

        Balances:
        - 50% quality (accuracy, confidence)
        - 30% efficiency (quality per unit time)
        - 20% latency (prefer faster modes)
        """
        return (
            0.5 * self.quality_score +
            0.3 * min(self.efficiency, 1.0) +
            0.2 * max(0.0, 1.0 - self.latency_ratio)
        )


class ComplexityTuner(TuningAgent):
    """
    Agent 5: Complexity Tuner

    Optimizes execution mode selection using Thompson Sampling.
    Eliminates manual BARE/FAST/FUSED configuration.
    """

    def __init__(self):
        super().__init__('complexity_tuner')

        # Metrics history (last 1000 measurements per complexity level)
        self.metrics_history: Dict[QueryComplexity, deque] = {
            complexity: deque(maxlen=1000)
            for complexity in QueryComplexity
        }

        # Thompson Sampling bandits (one per complexity level)
        self.bandits: Dict[QueryComplexity, ThompsonBandit] = {
            complexity: ThompsonBandit(n_arms=3)  # BARE, FAST, FUSED
            for complexity in QueryComplexity
        }

        # Current mode indices (selected arm per complexity)
        self.current_modes: Dict[QueryComplexity, int] = {
            QueryComplexity.SIMPLE: MODE_TO_ARM[ExecutionMode.FAST],
            QueryComplexity.MODERATE: MODE_TO_ARM[ExecutionMode.FAST],
            QueryComplexity.COMPLEX: MODE_TO_ARM[ExecutionMode.FUSED],
        }

        # Safe parameters (mode selection is discrete, but we track as float for consistency)
        self.safe_params: Dict[str, SafeParameter] = {
            'simple_mode': SafeParameter(
                name='simple_mode',
                current_value=float(MODE_TO_ARM[ExecutionMode.FAST]),
                min_value=0.0,
                max_value=2.0,
                max_change_percent=1.0,  # Allow full mode switches
            ),
            'moderate_mode': SafeParameter(
                name='moderate_mode',
                current_value=float(MODE_TO_ARM[ExecutionMode.FAST]),
                min_value=0.0,
                max_value=2.0,
                max_change_percent=1.0,
            ),
            'complex_mode': SafeParameter(
                name='complex_mode',
                current_value=float(MODE_TO_ARM[ExecutionMode.FUSED]),
                min_value=0.0,
                max_value=2.0,
                max_change_percent=1.0,
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

        # Check for complexity keywords
        complex_keywords = ['explain', 'compare', 'analyze', 'evaluate', 'discuss', 'research']
        has_complex_keyword = any(kw in query_text.lower() for kw in complex_keywords)

        if length < 10 and not has_complex_keyword:
            return QueryComplexity.SIMPLE
        elif length < 20 or (length < 30 and not has_complex_keyword):
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX

    def get_adaptive_mode(self, complexity: QueryComplexity) -> ExecutionMode:
        """
        Get current execution mode for given complexity level.

        Args:
            complexity: QueryComplexity enum

        Returns:
            ExecutionMode enum
        """
        arm_idx = self.current_modes[complexity]
        return ARM_TO_MODE[arm_idx]

    async def measure_performance(self) -> Dict[str, Any]:
        """
        Measure execution mode performance metrics.

        Returns:
            Dict with current metrics per complexity level
        """
        metrics = {}

        for complexity in QueryComplexity:
            param_name = f'{complexity.value}_mode'

            if len(self.metrics_history[complexity]) > 0:
                recent_metrics = list(self.metrics_history[complexity])[-100:]  # Last 100

                # Aggregate metrics
                avg_latency = np.mean([m.latency_ms for m in recent_metrics])
                avg_quality = np.mean([m.quality_score for m in recent_metrics])
                avg_efficiency = np.mean([m.efficiency for m in recent_metrics])
                avg_quality_metric = np.mean([m.quality_metric() for m in recent_metrics])

                # Mode distribution
                mode_counts = defaultdict(int)
                for m in recent_metrics:
                    mode_counts[m.mode.value] += 1

                metrics[param_name] = {
                    'avg_latency_ms': avg_latency,
                    'avg_quality': avg_quality,
                    'avg_efficiency': avg_efficiency,
                    'quality_metric': avg_quality_metric,
                    'current_mode': self.get_adaptive_mode(complexity).value,
                    'mode_distribution': dict(mode_counts),
                    'sample_count': len(recent_metrics),
                }
            else:
                # No data yet, use baseline
                metrics[param_name] = {
                    'avg_latency_ms': 0.0,
                    'avg_quality': 0.0,
                    'avg_efficiency': 0.0,
                    'quality_metric': 0.0,
                    'current_mode': self.get_adaptive_mode(complexity).value,
                    'mode_distribution': {},
                    'sample_count': 0,
                }

        return metrics

    async def tune_parameters(self) -> Dict[str, Any]:
        """
        Propose new execution modes using Thompson Sampling.

        Returns:
            Dict with proposed mode changes
        """
        proposals = {}

        # Select complexity level to tune (round-robin based on distribution)
        total_queries = sum(self.complexity_distribution.values())
        if total_queries == 0:
            # No queries yet, start with SIMPLE
            complexity_to_tune = QueryComplexity.SIMPLE
        else:
            # Pick the most frequent complexity level
            complexity_to_tune = max(
                self.complexity_distribution.items(),
                key=lambda x: x[1]
            )[0]

        # Thompson Sampling: select mode
        bandit = self.bandits[complexity_to_tune]
        arm_idx = bandit.select()

        # Map to mode
        proposed_mode = ARM_TO_MODE[arm_idx]
        param_name = f'{complexity_to_tune.value}_mode'

        # Update safe parameter (discrete, so we just set it)
        self.safe_params[param_name].current_value = float(arm_idx)

        proposals[param_name] = {
            'complexity': complexity_to_tune.value,
            'old_mode': self.get_adaptive_mode(complexity_to_tune).value,
            'proposed_mode': proposed_mode.value,
            'arm_selected': arm_idx,
        }

        # Update current mode (for feedback)
        self.current_modes[complexity_to_tune] = arm_idx

        return proposals

    def record_complexity_metrics(self, query_text: str, metrics: ComplexityMetrics):
        """
        Record complexity metrics for tuning.

        Args:
            query_text: Query text (for complexity classification)
            metrics: ComplexityMetrics object
        """
        complexity = self.classify_query_complexity(query_text)
        metrics.query_complexity = complexity
        self.metrics_history[complexity].append(metrics)
        self.complexity_distribution[complexity] += 1

    def update_bandits(self, baseline_metrics: Dict[str, Any], new_metrics: Dict[str, Any]):
        """
        Update Thompson Sampling bandits based on quality change.

        Args:
            baseline_metrics: Baseline metrics before tuning
            new_metrics: New metrics after tuning
        """
        for complexity in QueryComplexity:
            param_name = f'{complexity.value}_mode'

            baseline_quality = baseline_metrics.get(param_name, {}).get('quality_metric', 0.0)
            new_quality = new_metrics.get(param_name, {}).get('quality_metric', 0.0)

            # Calculate improvement
            quality_delta = new_quality - baseline_quality

            # Update bandit (success if quality improved)
            success = quality_delta > 0.0
            confidence = abs(quality_delta)

            arm_idx = self.current_modes[complexity]
            self.bandits[complexity].update(arm_idx, success=success, confidence=confidence)

    async def run_tuning_cycle(self) -> Dict[str, Any]:
        """
        Run one tuning cycle.

        Returns:
            Dict with tuning results
        """
        # Measure baseline performance
        baseline_metrics = await self.measure_performance()

        # Propose new modes
        proposals = await self.tune_parameters()

        # In production, the system would run queries with the new modes
        # and measure actual performance. For now, we just return proposals.

        # Measure new performance (would be done after running queries)
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
            param_name = f'{complexity.value}_mode'

            baseline_quality = baseline.get(param_name, {}).get('quality_metric', 0.0)
            new_quality = new.get(param_name, {}).get('quality_metric', 0.0)

            quality_deltas.append(new_quality - baseline_quality)

        # Overall impact (average quality improvement)
        return float(np.mean(quality_deltas)) if quality_deltas else 0.0

    def get_bandit_stats(self) -> Dict[str, Any]:
        """
        Get Thompson Sampling bandit statistics.

        Returns:
            Dict with bandit statistics per complexity level
        """
        stats = {}

        for complexity in QueryComplexity:
            bandit = self.bandits[complexity]
            arm_stats = []

            for arm_idx in range(3):  # BARE, FAST, FUSED
                alpha, beta = bandit.alpha[arm_idx], bandit.beta[arm_idx]
                expected_reward = alpha / (alpha + beta)
                mode_name = ARM_TO_MODE[arm_idx].value

                arm_stats.append({
                    'mode': mode_name,
                    'alpha': alpha,
                    'beta': beta,
                    'expected_reward': expected_reward,
                    'pulls': bandit.pulls[arm_idx],
                })

            stats[complexity.value] = {
                'current_mode': self.get_adaptive_mode(complexity).value,
                'arms': arm_stats,
            }

        return stats

    async def persist_state(self):
        """Persist tuning state (handled by coordinator)."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Get tuning state for persistence.

        Returns:
            Dict with tuning state
        """
        return {
            'name': self.name,
            'current_modes': {
                complexity.value: arm_idx
                for complexity, arm_idx in self.current_modes.items()
            },
            'bandits': {
                complexity.value: {
                    'alpha': self.bandits[complexity].alpha.tolist(),
                    'beta': self.bandits[complexity].beta.tolist(),
                    'pulls': self.bandits[complexity].pulls.tolist(),
                }
                for complexity in QueryComplexity
            },
            'safe_params': {
                name: {
                    'current_value': param.current_value,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                    'max_change_percent': param.max_change_percent,
                }
                for name, param in self.safe_params.items()
            },
            'complexity_distribution': dict(self.complexity_distribution),
        }

    def load_state(self, state: Dict[str, Any]):
        """
        Load tuning state from persistence.

        Args:
            state: Dict with saved tuning state
        """
        # Load current modes
        for complexity_str, arm_idx in state.get('current_modes', {}).items():
            complexity = QueryComplexity(complexity_str)
            self.current_modes[complexity] = arm_idx

        # Load bandits
        for complexity_str, bandit_state in state.get('bandits', {}).items():
            complexity = QueryComplexity(complexity_str)
            self.bandits[complexity].alpha = np.array(bandit_state['alpha'])
            self.bandits[complexity].beta = np.array(bandit_state['beta'])
            self.bandits[complexity].pulls = np.array(bandit_state['pulls'])

        # Load safe parameters
        for name, param_state in state.get('safe_params', {}).items():
            if name in self.safe_params:
                self.safe_params[name].current_value = param_state['current_value']
                self.safe_params[name].min_value = param_state['min_value']
                self.safe_params[name].max_value = param_state['max_value']
                self.safe_params[name].max_change_percent = param_state['max_change_percent']

        # Load complexity distribution
        for complexity_str, count in state.get('complexity_distribution', {}).items():
            complexity = QueryComplexity(complexity_str)
            self.complexity_distribution[complexity] = count


def create_complexity_tuner() -> ComplexityTuner:
    """
    Factory function to create ComplexityTuner.

    Returns:
        ComplexityTuner instance
    """
    return ComplexityTuner()
