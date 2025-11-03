"""
Cache Size Tuner - Agent 2

Automatically optimizes cache sizes using Thompson Sampling.

Strategy:
1. Measure hit rate, eviction rate, memory usage per cache type
2. Thompson Sampling learns optimal cache size multipliers
3. Multi-objective: maximize hit rate, minimize memory usage
4. Separate bandits for different cache types (query, parse, merge)

Eliminates Parameters:
- cache_size (query cache)
- parse_cache_size (Phase 5 compositional cache)
- merge_cache_size (Phase 5 compositional cache)
"""

from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass
import numpy as np
from HoloLoom.tuning.base import TuningAgent, ThompsonBandit, SafeParameter

# Cache size multipliers (relative to baseline)
CACHE_SIZE_MULTIPLIERS = [0.5, 1.0, 2.0, 5.0, 10.0]  # 5 arms

# Baseline cache sizes (conservative defaults)
BASELINE_CACHE_SIZES = {
    'query': 1000,      # Query result cache
    'parse': 5000,      # Phase 5 parse cache
    'merge': 10000,     # Phase 5 merge cache
}

# Safe ranges for cache sizes
SAFE_CACHE_RANGES = {
    'query': (100, 10000),
    'parse': (1000, 50000),
    'merge': (5000, 100000),
}


@dataclass
class CacheMetrics:
    """Metrics for a single cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate (0.0-1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def eviction_rate(self) -> float:
        """Calculate eviction rate (evictions per access)."""
        total = self.hits + self.misses
        return self.evictions / total if total > 0 else 0.0

    @property
    def utilization(self) -> float:
        """Calculate cache utilization (0.0-1.0)."""
        return self.size / self.max_size if self.max_size > 0 else 0.0

    def quality_score(self) -> float:
        """
        Multi-objective quality score.

        Formula:
        quality = 0.6 * hit_rate + 0.2 * (1 - eviction_rate) + 0.2 * (1 - utilization)

        Balances:
        - 60% hit rate (higher is better)
        - 20% eviction avoidance (lower evictions is better)
        - 20% memory efficiency (lower utilization is better)
        """
        return (
            0.6 * self.hit_rate +
            0.2 * (1.0 - min(self.eviction_rate, 1.0)) +
            0.2 * (1.0 - self.utilization)
        )


class CacheTuner(TuningAgent):
    """
    Agent 2: Cache Size Tuner

    Optimizes cache sizes using Thompson Sampling.
    Eliminates manual cache size configuration.
    """

    def __init__(self):
        super().__init__('cache_tuner')

        # Cache metrics history (last 1000 measurements per cache)
        self.metrics_history: Dict[str, deque] = {
            cache_type: deque(maxlen=1000)
            for cache_type in ['query', 'parse', 'merge']
        }

        # Thompson Sampling bandits (one per cache type)
        self.bandits: Dict[str, ThompsonBandit] = {
            cache_type: ThompsonBandit(n_arms=len(CACHE_SIZE_MULTIPLIERS))
            for cache_type in ['query', 'parse', 'merge']
        }

        # Current multiplier indices (selected arm per cache)
        self.current_multipliers: Dict[str, int] = {
            'query': 1,   # Start with 1.0x (baseline)
            'parse': 1,
            'merge': 1,
        }

        # Safe parameters for each cache
        self.safe_params: Dict[str, SafeParameter] = {}
        for cache_type, baseline_size in BASELINE_CACHE_SIZES.items():
            min_size, max_size = SAFE_CACHE_RANGES[cache_type]
            self.safe_params[cache_type] = SafeParameter(
                name=f'{cache_type}_cache_size',
                current_value=float(baseline_size),
                min_value=float(min_size),
                max_value=float(max_size),
                max_change_percent=0.3,  # Allow 30% change per cycle
            )

    async def measure_performance(self) -> Dict[str, Any]:
        """
        Measure cache performance metrics.

        Returns:
            Dict with current cache metrics
        """
        metrics = {}

        for cache_type in ['query', 'parse', 'merge']:
            if len(self.metrics_history[cache_type]) > 0:
                recent_metrics = list(self.metrics_history[cache_type])[-100:]  # Last 100

                # Aggregate metrics
                total_hits = sum(m.hits for m in recent_metrics)
                total_misses = sum(m.misses for m in recent_metrics)
                total_evictions = sum(m.evictions for m in recent_metrics)
                avg_size = np.mean([m.size for m in recent_metrics])
                avg_max_size = np.mean([m.max_size for m in recent_metrics])

                agg = CacheMetrics(
                    hits=total_hits,
                    misses=total_misses,
                    evictions=total_evictions,
                    size=int(avg_size),
                    max_size=int(avg_max_size),
                )

                metrics[cache_type] = {
                    'hit_rate': agg.hit_rate,
                    'eviction_rate': agg.eviction_rate,
                    'utilization': agg.utilization,
                    'quality': agg.quality_score(),
                    'current_size': self.safe_params[cache_type].current_value,
                }
            else:
                # No data yet
                metrics[cache_type] = {
                    'hit_rate': 0.0,
                    'eviction_rate': 0.0,
                    'utilization': 0.0,
                    'quality': 0.0,
                    'current_size': self.safe_params[cache_type].current_value,
                }

        return metrics

    async def tune_parameters(self) -> Dict[str, Any]:
        """
        Tune cache sizes based on Thompson Sampling.

        Returns:
            Dict with proposed cache size changes
        """
        proposals = {}

        for cache_type in ['query', 'parse', 'merge']:
            # Sample from Thompson Sampling bandit
            arm_idx = self.bandits[cache_type].sample()
            multiplier = CACHE_SIZE_MULTIPLIERS[arm_idx]

            # Calculate new cache size
            baseline = BASELINE_CACHE_SIZES[cache_type]
            proposed_size = baseline * multiplier

            # Apply safety checks
            safe_size = self.safe_params[cache_type].propose(proposed_size)

            proposals[cache_type] = {
                'old_size': int(self.safe_params[cache_type].current_value),
                'proposed_size': int(safe_size),
                'multiplier': safe_size / baseline,
                'arm_selected': arm_idx,
            }

            # Update current multiplier (for feedback)
            self.current_multipliers[cache_type] = arm_idx

        return proposals

    def record_cache_metrics(self, cache_type: str, metrics: CacheMetrics):
        """
        Record cache metrics for tuning.

        Args:
            cache_type: One of 'query', 'parse', 'merge'
            metrics: CacheMetrics object
        """
        if cache_type in self.metrics_history:
            self.metrics_history[cache_type].append(metrics)

    def update_bandits(self, baseline_metrics: Dict[str, Any], new_metrics: Dict[str, Any]):
        """
        Update Thompson Sampling bandits based on performance change.

        Args:
            baseline_metrics: Metrics before tuning
            new_metrics: Metrics after tuning
        """
        for cache_type in ['query', 'parse', 'merge']:
            baseline_quality = baseline_metrics.get(cache_type, {}).get('quality', 0.0)
            new_quality = new_metrics.get(cache_type, {}).get('quality', 0.0)

            # Calculate improvement
            improvement = new_quality - baseline_quality

            # Update bandit (success if improvement > 0)
            arm_idx = self.current_multipliers[cache_type]
            success = improvement > 0
            confidence = abs(improvement)  # Magnitude of improvement

            self.bandits[cache_type].update(arm_idx, success, confidence)

    async def run_tuning_cycle(self) -> Dict[str, Any]:
        """
        Run complete tuning cycle for cache sizes.

        Returns:
            Dict with tuning results
        """
        # Measure baseline performance
        baseline_metrics = await self.measure_performance()

        # Tune parameters
        proposals = await self.tune_parameters()

        # Apply proposed changes (simulate - actual application happens in orchestrator)
        for cache_type, proposal in proposals.items():
            self.safe_params[cache_type].current_value = float(proposal['proposed_size'])

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

        for cache_type in ['query', 'parse', 'merge']:
            baseline_quality = baseline.get(cache_type, {}).get('quality', 0.0)
            new_quality = new.get(cache_type, {}).get('quality', 0.0)

            baseline_hit_rate = baseline.get(cache_type, {}).get('hit_rate', 0.0)
            new_hit_rate = new.get(cache_type, {}).get('hit_rate', 0.0)

            baseline_size = baseline.get(cache_type, {}).get('current_size', 0.0)
            new_size = new.get(cache_type, {}).get('current_size', 0.0)

            impacts[cache_type] = {
                'quality_delta': new_quality - baseline_quality,
                'hit_rate_delta': new_hit_rate - baseline_hit_rate,
                'size_delta': new_size - baseline_size,
                'size_delta_percent': ((new_size - baseline_size) / baseline_size * 100) if baseline_size > 0 else 0.0,
            }

        # Overall impact (average quality improvement)
        avg_quality_delta = np.mean([impacts[ct]['quality_delta'] for ct in ['query', 'parse', 'merge']])
        impacts['overall'] = avg_quality_delta

        return impacts

    def get_bandit_stats(self) -> Dict[str, Any]:
        """
        Get Thompson Sampling bandit statistics.

        Returns:
            Dict with bandit stats per cache type
        """
        stats = {}

        for cache_type, bandit in self.bandits.items():
            arm_stats = []
            for i, multiplier in enumerate(CACHE_SIZE_MULTIPLIERS):
                expected_reward = bandit.alpha[i] / (bandit.alpha[i] + bandit.beta[i])
                n_pulls = int(bandit.alpha[i] + bandit.beta[i] - 2)  # -2 for priors

                arm_stats.append({
                    'multiplier': multiplier,
                    'expected_reward': expected_reward,
                    'n_pulls': n_pulls,
                    'alpha': float(bandit.alpha[i]),
                    'beta': float(bandit.beta[i]),
                })

            stats[cache_type] = arm_stats

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
                cache_type: {
                    'current_value': param.current_value,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                }
                for cache_type, param in self.safe_params.items()
            },
            'bandits': {
                cache_type: bandit.get_state()
                for cache_type, bandit in self.bandits.items()
            },
            'current_multipliers': self.current_multipliers.copy(),
            'metrics_samples': {
                cache_type: [
                    {
                        'hits': m.hits,
                        'misses': m.misses,
                        'evictions': m.evictions,
                        'size': m.size,
                        'max_size': m.max_size,
                    }
                    for m in list(history)[-100:]  # Save last 100 samples
                ]
                for cache_type, history in self.metrics_history.items()
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """Restore agent from serialized state."""
        self.total_tuning_cycles = state.get('total_tuning_cycles', 0)
        self.successful_tunings = state.get('successful_tunings', 0)

        # Restore safe parameters
        for cache_type, param_state in state.get('safe_params', {}).items():
            if cache_type in self.safe_params:
                self.safe_params[cache_type].current_value = param_state['current_value']

        # Restore bandits
        for cache_type, bandit_state in state.get('bandits', {}).items():
            if cache_type in self.bandits:
                self.bandits[cache_type] = ThompsonBandit.from_state(bandit_state)

        # Restore current multipliers
        self.current_multipliers = state.get('current_multipliers', {
            'query': 1,
            'parse': 1,
            'merge': 1,
        })

        # Restore metrics samples
        for cache_type, samples in state.get('metrics_samples', {}).items():
            if cache_type in self.metrics_history:
                self.metrics_history[cache_type] = deque(
                    [CacheMetrics(**s) for s in samples],
                    maxlen=1000
                )

        print(f"[CacheTuner] Loaded state: {self.total_tuning_cycles} cycles, "
              f"{self.successful_tunings} successful")


def create_cache_tuner() -> CacheTuner:
    """
    Factory function to create CacheTuner.

    Returns:
        Initialized CacheTuner instance
    """
    return CacheTuner()
