"""
Threshold Tuner - Agent 3

Automatically optimizes similarity and activation thresholds using Thompson Sampling.

Strategy:
1. Measure precision, recall, F1 score for different threshold combinations
2. Thompson Sampling learns optimal threshold values per parameter
3. Independent bandits per threshold (tune one at a time for efficiency)
4. Multi-objective: maximize F1 score (harmonic mean of precision/recall)

Eliminates Parameters:
- motif_similarity_threshold (motif matching)
- phrase_similarity_threshold (phrase matching)
- activation_threshold (graph node activation)
- prefilter_similarity_threshold (linguistic filtering)
- cache_similarity_threshold (cache matching)
- retrieval_similarity_threshold (retrieval filtering)
- context_relevance_threshold (context selection)
- confidence_threshold (decision confidence)
"""

from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass
import numpy as np
from HoloLoom.tuning.base import TuningAgent, ThompsonBandit, SafeParameter

# Threshold value options (5 arms per threshold)
THRESHOLD_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]  # 5 discrete levels

# Default baseline thresholds (conservative defaults)
BASELINE_THRESHOLDS = {
    'motif_similarity': 0.7,
    'phrase_similarity': 0.8,
    'activation': 0.5,
    'prefilter_similarity': 0.3,
    'cache_similarity': 0.85,
    'retrieval_similarity': 0.6,
    'context_relevance': 0.6,
    'confidence': 0.75,
}

# Safe ranges for thresholds (all 0.0-1.0 but with practical bounds)
SAFE_THRESHOLD_RANGES = {
    'motif_similarity': (0.3, 0.95),      # Too low = noise, too high = miss matches
    'phrase_similarity': (0.5, 0.95),     # Phrases should be fairly similar
    'activation': (0.3, 0.9),             # Graph node activation
    'prefilter_similarity': (0.2, 0.8),   # Pre-filter can be loose
    'cache_similarity': (0.7, 0.99),      # Cache needs high precision
    'retrieval_similarity': (0.4, 0.9),   # Retrieval balance
    'context_relevance': (0.4, 0.9),      # Context selection
    'confidence': (0.5, 0.95),            # Decision confidence
}


@dataclass
class QualityMetrics:
    """Quality metrics for threshold evaluation."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision (positive predictive value)."""
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall (sensitivity)."""
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        p = self.precision
        r = self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0

    def quality_score(self) -> float:
        """
        Multi-objective quality score.

        Formula:
        quality = 0.7 * f1_score + 0.2 * precision + 0.1 * recall

        Prioritizes:
        - 70% F1 score (balanced precision/recall)
        - 20% precision (avoid false positives)
        - 10% recall (capture true positives)
        """
        return 0.7 * self.f1_score + 0.2 * self.precision + 0.1 * self.recall


class ThresholdTuner(TuningAgent):
    """
    Agent 3: Threshold Tuner

    Optimizes similarity and activation thresholds using Thompson Sampling.
    Eliminates manual threshold configuration.
    """

    def __init__(self):
        super().__init__('threshold_tuner')

        # Metrics history (last 1000 measurements per threshold)
        self.metrics_history: Dict[str, deque] = {
            threshold_name: deque(maxlen=1000)
            for threshold_name in BASELINE_THRESHOLDS.keys()
        }

        # Thompson Sampling bandits (one per threshold parameter)
        self.bandits: Dict[str, ThompsonBandit] = {
            threshold_name: ThompsonBandit(n_arms=len(THRESHOLD_VALUES))
            for threshold_name in BASELINE_THRESHOLDS.keys()
        }

        # Current threshold value indices (selected arm per threshold)
        self.current_thresholds: Dict[str, int] = {
            threshold_name: THRESHOLD_VALUES.index(BASELINE_THRESHOLDS[threshold_name])
            if BASELINE_THRESHOLDS[threshold_name] in THRESHOLD_VALUES
            else 2  # Default to middle value (0.7)
            for threshold_name in BASELINE_THRESHOLDS.keys()
        }

        # Safe parameters for each threshold
        self.safe_params: Dict[str, SafeParameter] = {}
        for threshold_name, baseline_value in BASELINE_THRESHOLDS.items():
            min_val, max_val = SAFE_THRESHOLD_RANGES[threshold_name]
            self.safe_params[threshold_name] = SafeParameter(
                name=f'{threshold_name}_threshold',
                current_value=baseline_value,
                min_value=min_val,
                max_value=max_val,
                max_change_percent=0.2,  # Allow 20% change per cycle
            )

        # Track which threshold to tune next (round-robin)
        self.threshold_names = list(BASELINE_THRESHOLDS.keys())
        self.current_tuning_index = 0

    async def measure_performance(self) -> Dict[str, Any]:
        """
        Measure threshold performance metrics.

        Returns:
            Dict with current quality metrics per threshold
        """
        metrics = {}

        for threshold_name in self.threshold_names:
            if len(self.metrics_history[threshold_name]) > 0:
                recent_metrics = list(self.metrics_history[threshold_name])[-100:]  # Last 100

                # Aggregate metrics
                total_tp = sum(m.true_positives for m in recent_metrics)
                total_fp = sum(m.false_positives for m in recent_metrics)
                total_tn = sum(m.true_negatives for m in recent_metrics)
                total_fn = sum(m.false_negatives for m in recent_metrics)

                agg = QualityMetrics(
                    true_positives=total_tp,
                    false_positives=total_fp,
                    true_negatives=total_tn,
                    false_negatives=total_fn,
                )

                metrics[threshold_name] = {
                    'precision': agg.precision,
                    'recall': agg.recall,
                    'f1_score': agg.f1_score,
                    'accuracy': agg.accuracy,
                    'quality': agg.quality_score(),
                    'current_value': self.safe_params[threshold_name].current_value,
                }
            else:
                # No data yet
                metrics[threshold_name] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'accuracy': 0.0,
                    'quality': 0.0,
                    'current_value': self.safe_params[threshold_name].current_value,
                }

        return metrics

    async def tune_parameters(self) -> Dict[str, Any]:
        """
        Tune thresholds based on Thompson Sampling.

        Strategy: Tune one threshold at a time (round-robin) for efficiency.

        Returns:
            Dict with proposed threshold changes
        """
        proposals = {}

        # Select which threshold to tune (round-robin)
        threshold_name = self.threshold_names[self.current_tuning_index]
        self.current_tuning_index = (self.current_tuning_index + 1) % len(self.threshold_names)

        # Sample from Thompson Sampling bandit
        arm_idx = self.bandits[threshold_name].sample()
        proposed_value = THRESHOLD_VALUES[arm_idx]

        # Apply safety checks
        safe_value = self.safe_params[threshold_name].propose(proposed_value)

        proposals[threshold_name] = {
            'old_value': self.safe_params[threshold_name].current_value,
            'proposed_value': safe_value,
            'arm_selected': arm_idx,
        }

        # Update current threshold index (for feedback)
        self.current_thresholds[threshold_name] = arm_idx

        # Update parameter value
        self.safe_params[threshold_name].current_value = safe_value

        return proposals

    def record_quality_metrics(self, threshold_name: str, metrics: QualityMetrics):
        """
        Record quality metrics for tuning.

        Args:
            threshold_name: Name of threshold (e.g., 'motif_similarity')
            metrics: QualityMetrics object
        """
        if threshold_name in self.metrics_history:
            self.metrics_history[threshold_name].append(metrics)

    def update_bandits(self, baseline_metrics: Dict[str, Any], new_metrics: Dict[str, Any]):
        """
        Update Thompson Sampling bandits based on quality change.

        Args:
            baseline_metrics: Metrics before tuning
            new_metrics: Metrics after tuning
        """
        for threshold_name in self.threshold_names:
            baseline_quality = baseline_metrics.get(threshold_name, {}).get('quality', 0.0)
            new_quality = new_metrics.get(threshold_name, {}).get('quality', 0.0)

            # Calculate improvement
            improvement = new_quality - baseline_quality

            # Update bandit (success if improvement > 0)
            arm_idx = self.current_thresholds[threshold_name]
            success = improvement > 0
            confidence = abs(improvement)

            self.bandits[threshold_name].update(arm_idx, success, confidence)

    async def run_tuning_cycle(self) -> Dict[str, Any]:
        """
        Run complete tuning cycle for thresholds.

        Returns:
            Dict with tuning results
        """
        # Measure baseline performance
        baseline_metrics = await self.measure_performance()

        # Tune parameters (one threshold at a time)
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

    def _calculate_impact(self, baseline: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate impact of tuning changes.

        Args:
            baseline: Baseline metrics
            new: New metrics after tuning

        Returns:
            Dict with impact metrics
        """
        impacts = {}

        for threshold_name in self.threshold_names:
            baseline_quality = baseline.get(threshold_name, {}).get('quality', 0.0)
            new_quality = new.get(threshold_name, {}).get('quality', 0.0)

            baseline_f1 = baseline.get(threshold_name, {}).get('f1_score', 0.0)
            new_f1 = new.get(threshold_name, {}).get('f1_score', 0.0)

            baseline_value = baseline.get(threshold_name, {}).get('current_value', 0.0)
            new_value = new.get(threshold_name, {}).get('current_value', 0.0)

            impacts[threshold_name] = {
                'quality_delta': new_quality - baseline_quality,
                'f1_delta': new_f1 - baseline_f1,
                'value_delta': new_value - baseline_value,
                'value_delta_percent': ((new_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0.0,
            }

        # Overall impact (average quality improvement)
        avg_quality_delta = np.mean([impacts[name]['quality_delta'] for name in self.threshold_names])
        impacts['overall'] = avg_quality_delta

        return impacts

    def get_bandit_stats(self) -> Dict[str, Any]:
        """
        Get Thompson Sampling bandit statistics.

        Returns:
            Dict with bandit stats per threshold
        """
        stats = {}

        for threshold_name, bandit in self.bandits.items():
            arm_stats = []
            for i, threshold_value in enumerate(THRESHOLD_VALUES):
                expected_reward = bandit.alpha[i] / (bandit.alpha[i] + bandit.beta[i])
                n_pulls = int(bandit.alpha[i] + bandit.beta[i] - 2)

                arm_stats.append({
                    'threshold_value': threshold_value,
                    'expected_reward': expected_reward,
                    'n_pulls': n_pulls,
                    'alpha': float(bandit.alpha[i]),
                    'beta': float(bandit.beta[i]),
                })

            stats[threshold_name] = arm_stats

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
                threshold_name: {
                    'current_value': param.current_value,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                }
                for threshold_name, param in self.safe_params.items()
            },
            'bandits': {
                threshold_name: bandit.get_state()
                for threshold_name, bandit in self.bandits.items()
            },
            'current_thresholds': self.current_thresholds.copy(),
            'current_tuning_index': self.current_tuning_index,
            'metrics_samples': {
                threshold_name: [
                    {
                        'true_positives': m.true_positives,
                        'false_positives': m.false_positives,
                        'true_negatives': m.true_negatives,
                        'false_negatives': m.false_negatives,
                    }
                    for m in list(history)[-100:]  # Save last 100 samples
                ]
                for threshold_name, history in self.metrics_history.items()
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """Restore agent from serialized state."""
        self.total_tuning_cycles = state.get('total_tuning_cycles', 0)
        self.successful_tunings = state.get('successful_tunings', 0)

        # Restore safe parameters
        for threshold_name, param_state in state.get('safe_params', {}).items():
            if threshold_name in self.safe_params:
                self.safe_params[threshold_name].current_value = param_state['current_value']

        # Restore bandits
        for threshold_name, bandit_state in state.get('bandits', {}).items():
            if threshold_name in self.bandits:
                self.bandits[threshold_name] = ThompsonBandit.from_state(bandit_state)

        # Restore current thresholds
        self.current_thresholds = state.get('current_thresholds', {
            name: 2 for name in self.threshold_names
        })

        # Restore tuning index
        self.current_tuning_index = state.get('current_tuning_index', 0)

        # Restore metrics samples
        for threshold_name, samples in state.get('metrics_samples', {}).items():
            if threshold_name in self.metrics_history:
                self.metrics_history[threshold_name] = deque(
                    [QualityMetrics(**s) for s in samples],
                    maxlen=1000
                )

        print(f"[ThresholdTuner] Loaded state: {self.total_tuning_cycles} cycles, "
              f"{self.successful_tunings} successful")


def create_threshold_tuner() -> ThresholdTuner:
    """
    Factory function to create ThresholdTuner.

    Returns:
        Initialized ThresholdTuner instance
    """
    return ThresholdTuner()
