"""
TimeoutTuner: Adaptive timeout management.

Tunes:
- pipeline_timeout
- retrieval_timeout
- stage_timeouts (features, retrieval, decision, execution)

Strategy:
- Measure p50, p95, p99 latencies per stage
- Set timeout = p95 × safety_margin
- Learn optimal safety margin via Thompson Sampling
- Adapt to system load (CPU usage)
"""

import asyncio
import logging
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from hololoom.tuning.base import TuningAgent, ThompsonBandit, SafeParameter, create_safe_parameter

logger = logging.getLogger(__name__)


# Safety margin candidates for Thompson Sampling
SAFETY_MARGINS = [1.2, 1.5, 2.0, 2.5, 3.0]

# Default timeouts (fallback if no data)
DEFAULT_TIMEOUTS = {
    'pipeline': 5.0,
    'retrieval': 0.2,
    'features': 2.0,
    'decision': 2.0,
    'execution': 3.0,
}


@dataclass
class LoadState:
    """System load state for adaptive timeouts."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0

    @property
    def category(self) -> str:
        """Categorize load: low, medium, high."""
        if self.cpu_percent < 50.0:
            return 'low_load'
        elif self.cpu_percent < 80.0:
            return 'medium_load'
        else:
            return 'high_load'


class TimeoutTuner(TuningAgent):
    """
    Adaptive timeout tuning agent.

    Learns optimal timeouts based on:
    - Actual stage latencies (p95)
    - System load
    - Timeout success/failure rates
    """

    def __init__(self):
        super().__init__('timeout_tuner')

        # Latency measurements per stage (last 1000 samples)
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Thompson Sampling bandits per load state
        self.bandits: Dict[str, ThompsonBandit] = {
            'low_load': ThompsonBandit(n_arms=len(SAFETY_MARGINS)),
            'medium_load': ThompsonBandit(n_arms=len(SAFETY_MARGINS)),
            'high_load': ThompsonBandit(n_arms=len(SAFETY_MARGINS)),
        }

        # Current load state
        self.load_state = LoadState()

        # Timeout outcomes tracking
        self.timeout_outcomes: Dict[str, List[bool]] = defaultdict(lambda: deque(maxlen=100))

        # Safe parameters
        self.parameters: Dict[str, SafeParameter] = {
            'pipeline': create_safe_parameter('pipeline_timeout', DEFAULT_TIMEOUTS['pipeline']),
            'retrieval': create_safe_parameter('retrieval_timeout', DEFAULT_TIMEOUTS['retrieval']),
            'features': create_safe_parameter('stage_timeout', DEFAULT_TIMEOUTS['features']),
            'decision': create_safe_parameter('stage_timeout', DEFAULT_TIMEOUTS['decision']),
            'execution': create_safe_parameter('stage_timeout', DEFAULT_TIMEOUTS['execution']),
        }

        # Last margin used per stage (for feedback)
        self.last_margin_used: Dict[str, int] = {}
        self.last_timeout_used: Dict[str, float] = {}

    async def measure_performance(self) -> Dict[str, float]:
        """
        Measure current timeout performance.

        Returns:
            Metrics: timeout_false_positive_rate, avg_headroom, etc.
        """
        metrics = {}

        # Update system load
        self._update_load_state()

        # Calculate timeout false positive rate
        false_positives = []
        for stage, outcomes in self.timeout_outcomes.items():
            if outcomes:
                fps = sum(1 for timed_out in outcomes if timed_out) / len(outcomes)
                false_positives.append(fps)

        metrics['timeout_false_positive_rate'] = np.mean(false_positives) if false_positives else 0.0

        # Calculate average headroom (how much slack in timeouts)
        headrooms = []
        for stage, latencies in self.latencies.items():
            if len(latencies) > 10 and stage in self.last_timeout_used:
                actual_latencies = list(latencies)
                timeout = self.last_timeout_used[stage] * 1000  # Convert to ms

                for latency in actual_latencies[-20:]:  # Last 20 samples
                    if latency < timeout:
                        headroom = (timeout - latency) / timeout
                        headrooms.append(headroom)

        metrics['avg_headroom'] = np.mean(headrooms) if headrooms else 0.5

        # Per-stage p50, p95, p99
        for stage, latencies in self.latencies.items():
            if len(latencies) >= 20:
                samples = list(latencies)
                metrics[f'{stage}_p50'] = np.percentile(samples, 50)
                metrics[f'{stage}_p95'] = np.percentile(samples, 95)
                metrics[f'{stage}_p99'] = np.percentile(samples, 99)

        # CPU/memory load
        metrics['cpu_percent'] = self.load_state.cpu_percent
        metrics['memory_percent'] = self.load_state.memory_percent

        return metrics

    async def tune_parameters(self) -> Dict[str, Any]:
        """
        Tune timeouts based on current performance.

        Returns:
            Dict describing tuning actions
        """
        actions = {}

        # Get current load category
        load_category = self.load_state.category

        # Tune each stage
        for stage in ['features', 'retrieval', 'decision', 'execution']:
            if len(self.latencies[stage]) < 20:
                # Not enough data yet
                continue

            # Calculate p95 latency
            samples = list(self.latencies[stage])
            p95 = np.percentile(samples, 95)

            # Sample safety margin from bandit
            bandit = self.bandits[load_category]
            margin_idx = bandit.sample()
            margin = SAFETY_MARGINS[margin_idx]

            # Calculate new timeout
            new_timeout = (p95 * margin) / 1000.0  # Convert to seconds

            # Apply with safety checks
            param = self.parameters.get(stage)
            if param:
                old_timeout = param.current_value
                safe_timeout = param.update(new_timeout)

                if abs(safe_timeout - old_timeout) > 0.01:
                    actions[f'{stage}_timeout'] = {
                        'old': old_timeout,
                        'new': safe_timeout,
                        'p95_latency_ms': p95,
                        'safety_margin': margin,
                        'load_category': load_category,
                    }

                # Track for feedback
                self.last_margin_used[stage] = margin_idx
                self.last_timeout_used[stage] = safe_timeout

        # Tune pipeline timeout (sum of stages + buffer)
        if all(stage in self.parameters for stage in ['features', 'retrieval', 'decision', 'execution']):
            total_stages = sum(self.parameters[s].current_value
                             for s in ['features', 'retrieval', 'decision', 'execution'])
            new_pipeline = total_stages * 1.2  # 20% buffer

            param = self.parameters['pipeline']
            old_pipeline = param.current_value
            safe_pipeline = param.update(new_pipeline)

            if abs(safe_pipeline - old_pipeline) > 0.1:
                actions['pipeline_timeout'] = {
                    'old': old_pipeline,
                    'new': safe_pipeline,
                    'basis': 'sum_of_stages_plus_buffer',
                }

        return actions

    def record_stage_completion(self, stage: str, duration_ms: float, timed_out: bool = False):
        """
        Record stage completion for learning.

        Args:
            stage: Stage name (features, retrieval, decision, execution)
            duration_ms: Actual duration in milliseconds
            timed_out: Whether stage timed out
        """
        self.latencies[stage].append(duration_ms)
        self.timeout_outcomes[stage].append(timed_out)

        # Update bandit if we have enough data
        if len(self.latencies[stage]) >= 20 and stage in self.last_margin_used:
            self._update_bandit_from_outcome(stage, duration_ms, timed_out)

    def _update_bandit_from_outcome(self, stage: str, actual_duration: float, timed_out: bool):
        """Update Thompson Sampling bandit based on outcome."""
        load_category = self.load_state.category
        bandit = self.bandits[load_category]
        margin_idx = self.last_margin_used.get(stage)

        if margin_idx is None:
            return

        if timed_out:
            # Timeout too aggressive
            bandit.update(margin_idx, success=False, confidence=0.0)
            self.logger.debug(f"{stage}: Timeout too aggressive (margin={SAFETY_MARGINS[margin_idx]})")
        else:
            # Success - evaluate headroom quality
            timeout = self.last_timeout_used[stage] * 1000  # Convert to ms
            headroom = (timeout - actual_duration) / timeout if timeout > 0 else 0.5

            # Ideal headroom: 30-50%
            if 0.3 <= headroom <= 0.5:
                confidence = 1.0  # Perfect
            elif headroom < 0.3:
                confidence = 0.7  # Cutting it close
            elif headroom > 0.7:
                confidence = 0.4  # Too much slack
            else:
                confidence = 0.6  # Acceptable

            bandit.update(margin_idx, success=True, confidence=confidence)
            self.logger.debug(f"{stage}: Headroom={headroom:.1%}, confidence={confidence:.2f}")

    def _update_load_state(self):
        """Update current system load state."""
        if not PSUTIL_AVAILABLE:
            self.load_state = LoadState(cpu_percent=50.0, memory_percent=50.0)
            return

        try:
            self.load_state = LoadState(
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
            )
        except Exception as e:
            self.logger.warning(f"Failed to get system load: {e}")

    def get_adaptive_timeout(self, stage: str) -> float:
        """
        Get current optimal timeout for stage.

        Args:
            stage: Stage name

        Returns:
            Timeout in seconds
        """
        param = self.parameters.get(stage)
        if param:
            return param.current_value

        return DEFAULT_TIMEOUTS.get(stage, 5.0)

    def get_state(self) -> Dict[str, Any]:
        """Serialize agent state for persistence."""
        return {
            'total_tuning_cycles': self.total_tuning_cycles,
            'successful_tunings': self.successful_tunings,
            'parameters': {
                name: {
                    'current_value': param.current_value,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                }
                for name, param in self.parameters.items()
            },
            'bandits': {
                load_cat: bandit.get_state()
                for load_cat, bandit in self.bandits.items()
            },
            'latency_samples': {
                stage: list(latencies)[-100:]  # Save last 100 samples
                for stage, latencies in self.latencies.items()
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """Restore agent from serialized state."""
        self.total_tuning_cycles = state.get('total_tuning_cycles', 0)
        self.successful_tunings = state.get('successful_tunings', 0)

        # Restore parameters
        for name, param_state in state.get('parameters', {}).items():
            if name in self.parameters:
                self.parameters[name].current_value = param_state['current_value']

        # Restore bandits
        for load_cat, bandit_state in state.get('bandits', {}).items():
            if load_cat in self.bandits:
                self.bandits[load_cat] = ThompsonBandit.from_state(bandit_state)

        # Restore latency samples
        for stage, samples in state.get('latency_samples', {}).items():
            self.latencies[stage] = deque(samples, maxlen=1000)

        self.logger.info(f"Loaded state: {self.total_tuning_cycles} cycles, "
                        f"{self.successful_tunings} successful")
