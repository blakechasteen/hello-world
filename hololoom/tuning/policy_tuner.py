"""
Policy Tuner - Agent 6

Automatically optimizes policy exploration/exploitation parameters using Thompson Sampling.

Strategy:
1. Measure decision quality (accuracy, diversity, regret)
2. Thompson Sampling learns optimal exploration parameters
3. Balance exploitation (use best tools) vs exploration (try new tools)
4. Separate bandits for epsilon and bayesian_blend_weight

Eliminates Parameters:
- epsilon (exploration rate for epsilon-greedy)
- bayesian_blend_weight (weight for Bayesian blending)
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from hololoom.tuning.base import SafeParameter, ThompsonBandit, TuningAgent

# Epsilon values (exploration rate) - 5 arms
EPSILON_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]

# Bayesian blend weight values - 5 arms
BAYESIAN_BLEND_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]

# Baseline values
BASELINE_EPSILON = 0.10
BASELINE_BAYESIAN_BLEND = 0.30

# Safe ranges
SAFE_RANGES = {
    'epsilon': (0.01, 0.50),
    'bayesian_blend': (0.0, 0.7),
}


@dataclass
class PolicyMetrics:
    """Metrics for policy performance evaluation."""
    decision_accuracy: float = 0.0     # 0.0-1.0, how often correct tool selected
    tool_diversity: float = 0.0        # 0.0-1.0, variety of tools used
    regret: float = 0.0                # 0.0-1.0, how much better could we have done
    confidence: float = 0.0            # 0.0-1.0, confidence in decisions

    def quality_score(self) -> float:
        """
        Multi-objective quality score.

        Formula:
        quality = 0.5 * accuracy + 0.2 * confidence + 0.2 * (1-regret) + 0.1 * diversity

        Balances:
        - 50% decision accuracy (correct tool selection)
        - 20% confidence (high confidence decisions)
        - 20% low regret (near-optimal decisions)
        - 10% diversity (explore different tools)
        """
        return (
            0.5 * self.decision_accuracy +
            0.2 * self.confidence +
            0.2 * max(0.0, 1.0 - self.regret) +
            0.1 * self.tool_diversity
        )


class PolicyTuner(TuningAgent):
    """
    Agent 6: Policy Tuner

    Optimizes exploration/exploitation parameters using Thompson Sampling.
    Eliminates manual epsilon and bayesian_blend_weight configuration.
    """

    def __init__(self):
        super().__init__('policy_tuner')

        # Metrics history (last 1000 measurements)
        self.metrics_history: deque = deque(maxlen=1000)

        # Thompson Sampling bandits (one per parameter)
        self.epsilon_bandit = ThompsonBandit(n_arms=len(EPSILON_VALUES))
        self.bayesian_bandit = ThompsonBandit(n_arms=len(BAYESIAN_BLEND_VALUES))

        # Current parameter indices (selected arms)
        self.current_epsilon_idx = EPSILON_VALUES.index(BASELINE_EPSILON)
        self.current_bayesian_idx = BAYESIAN_BLEND_VALUES.index(BASELINE_BAYESIAN_BLEND)

        # Safe parameters
        self.safe_params: dict[str, SafeParameter] = {
            'epsilon': SafeParameter(
                name='epsilon',
                current_value=BASELINE_EPSILON,
                min_value=SAFE_RANGES['epsilon'][0],
                max_value=SAFE_RANGES['epsilon'][1],
                max_change_percent=0.30,  # Allow 30% change per cycle
            ),
            'bayesian_blend': SafeParameter(
                name='bayesian_blend_weight',
                current_value=BASELINE_BAYESIAN_BLEND,
                min_value=SAFE_RANGES['bayesian_blend'][0],
                max_value=SAFE_RANGES['bayesian_blend'][1],
                max_change_percent=0.30,
            ),
        }

        # Track which parameter to tune next (alternate)
        self.current_tuning_param = 'epsilon'

    def get_adaptive_epsilon(self) -> float:
        """
        Get current epsilon value.

        Returns:
            Current epsilon (exploration rate)
        """
        return self.safe_params['epsilon'].current_value

    def get_adaptive_bayesian_blend(self) -> float:
        """
        Get current bayesian blend weight.

        Returns:
            Current bayesian_blend_weight
        """
        return self.safe_params['bayesian_blend'].current_value

    async def measure_performance(self) -> dict[str, Any]:
        """
        Measure policy performance metrics.

        Returns:
            Dict with current policy metrics
        """
        metrics = {}

        if len(self.metrics_history) > 0:
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100

            # Aggregate metrics
            avg_accuracy = np.mean([m.decision_accuracy for m in recent_metrics])
            avg_diversity = np.mean([m.tool_diversity for m in recent_metrics])
            avg_regret = np.mean([m.regret for m in recent_metrics])
            avg_confidence = np.mean([m.confidence for m in recent_metrics])
            avg_quality = np.mean([m.quality_score() for m in recent_metrics])

            metrics['policy'] = {
                'avg_accuracy': avg_accuracy,
                'avg_diversity': avg_diversity,
                'avg_regret': avg_regret,
                'avg_confidence': avg_confidence,
                'quality_score': avg_quality,
                'current_epsilon': self.get_adaptive_epsilon(),
                'current_bayesian_blend': self.get_adaptive_bayesian_blend(),
                'sample_count': len(recent_metrics),
            }
        else:
            # No data yet, use baseline
            metrics['policy'] = {
                'avg_accuracy': 0.0,
                'avg_diversity': 0.0,
                'avg_regret': 0.0,
                'avg_confidence': 0.0,
                'quality_score': 0.0,
                'current_epsilon': self.get_adaptive_epsilon(),
                'current_bayesian_blend': self.get_adaptive_bayesian_blend(),
                'sample_count': 0,
            }

        return metrics

    async def tune_parameters(self) -> dict[str, Any]:
        """
        Propose new policy parameters using Thompson Sampling.

        Returns:
            Dict with proposed parameter changes
        """
        proposals = {}

        # Alternate between tuning epsilon and bayesian_blend
        if self.current_tuning_param == 'epsilon':
            # Tune epsilon
            arm_idx = self.epsilon_bandit.select()
            proposed_value = EPSILON_VALUES[arm_idx]

            # Apply safe parameter constraints
            safe_value = self.safe_params['epsilon'].propose(proposed_value)

            proposals['epsilon'] = {
                'old_value': self.safe_params['epsilon'].current_value,
                'proposed_value': safe_value,
                'arm_selected': arm_idx,
            }

            # Update parameter
            self.safe_params['epsilon'].current_value = safe_value
            self.current_epsilon_idx = arm_idx

            # Next cycle, tune bayesian_blend
            self.current_tuning_param = 'bayesian_blend'

        else:
            # Tune bayesian_blend
            arm_idx = self.bayesian_bandit.select()
            proposed_value = BAYESIAN_BLEND_VALUES[arm_idx]

            # Apply safe parameter constraints
            safe_value = self.safe_params['bayesian_blend'].propose(proposed_value)

            proposals['bayesian_blend'] = {
                'old_value': self.safe_params['bayesian_blend'].current_value,
                'proposed_value': safe_value,
                'arm_selected': arm_idx,
            }

            # Update parameter
            self.safe_params['bayesian_blend'].current_value = safe_value
            self.current_bayesian_idx = arm_idx

            # Next cycle, tune epsilon
            self.current_tuning_param = 'epsilon'

        return proposals

    def record_policy_metrics(self, metrics: PolicyMetrics):
        """
        Record policy metrics for tuning.

        Args:
            metrics: PolicyMetrics object
        """
        self.metrics_history.append(metrics)

    def update_bandits(self, baseline_metrics: dict[str, Any], new_metrics: dict[str, Any]):
        """
        Update Thompson Sampling bandits based on quality change.

        Args:
            baseline_metrics: Baseline metrics before tuning
            new_metrics: New metrics after tuning
        """
        baseline_quality = baseline_metrics.get('policy', {}).get('quality_score', 0.0)
        new_quality = new_metrics.get('policy', {}).get('quality_score', 0.0)

        # Calculate improvement
        quality_delta = new_quality - baseline_quality

        # Update bandit (success if quality improved)
        success = quality_delta > 0.0
        confidence = abs(quality_delta)

        # Update the bandit that was just used
        if self.current_tuning_param == 'bayesian_blend':
            # Last cycle tuned epsilon
            self.epsilon_bandit.update(self.current_epsilon_idx, success=success, confidence=confidence)
        else:
            # Last cycle tuned bayesian_blend
            self.bayesian_bandit.update(self.current_bayesian_idx, success=success, confidence=confidence)

    async def run_tuning_cycle(self) -> dict[str, Any]:
        """
        Run one tuning cycle.

        Returns:
            Dict with tuning results
        """
        # Measure baseline performance
        baseline_metrics = await self.measure_performance()

        # Propose new parameters
        proposals = await self.tune_parameters()

        # In production, the system would run queries with the new parameters
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

    def _calculate_impact(self, baseline: dict[str, Any], new: dict[str, Any]) -> float:
        """
        Calculate impact of tuning changes.

        Args:
            baseline: Baseline metrics
            new: New metrics after tuning

        Returns:
            Scalar impact score (quality improvement)
        """
        baseline_quality = baseline.get('policy', {}).get('quality_score', 0.0)
        new_quality = new.get('policy', {}).get('quality_score', 0.0)

        return new_quality - baseline_quality

    def get_bandit_stats(self) -> dict[str, Any]:
        """
        Get Thompson Sampling bandit statistics.

        Returns:
            Dict with bandit statistics for both parameters
        """
        stats = {}

        # Epsilon bandit stats
        epsilon_arms = []
        for arm_idx in range(len(EPSILON_VALUES)):
            alpha, beta = self.epsilon_bandit.alpha[arm_idx], self.epsilon_bandit.beta[arm_idx]
            expected_reward = alpha / (alpha + beta)

            epsilon_arms.append({
                'value': EPSILON_VALUES[arm_idx],
                'alpha': alpha,
                'beta': beta,
                'expected_reward': expected_reward,
                'pulls': self.epsilon_bandit.pulls[arm_idx],
            })

        stats['epsilon'] = {
            'current_value': self.get_adaptive_epsilon(),
            'arms': epsilon_arms,
        }

        # Bayesian blend bandit stats
        bayesian_arms = []
        for arm_idx in range(len(BAYESIAN_BLEND_VALUES)):
            alpha, beta = self.bayesian_bandit.alpha[arm_idx], self.bayesian_bandit.beta[arm_idx]
            expected_reward = alpha / (alpha + beta)

            bayesian_arms.append({
                'value': BAYESIAN_BLEND_VALUES[arm_idx],
                'alpha': alpha,
                'beta': beta,
                'expected_reward': expected_reward,
                'pulls': self.bayesian_bandit.pulls[arm_idx],
            })

        stats['bayesian_blend'] = {
            'current_value': self.get_adaptive_bayesian_blend(),
            'arms': bayesian_arms,
        }

        return stats

    async def persist_state(self):
        """Persist tuning state (handled by coordinator)."""
        pass

    def get_state(self) -> dict[str, Any]:
        """
        Get tuning state for persistence.

        Returns:
            Dict with tuning state
        """
        return {
            'name': self.name,
            'current_epsilon_idx': self.current_epsilon_idx,
            'current_bayesian_idx': self.current_bayesian_idx,
            'current_tuning_param': self.current_tuning_param,
            'epsilon_bandit': {
                'alpha': self.epsilon_bandit.alpha.tolist(),
                'beta': self.epsilon_bandit.beta.tolist(),
                'pulls': self.epsilon_bandit.pulls.tolist(),
            },
            'bayesian_bandit': {
                'alpha': self.bayesian_bandit.alpha.tolist(),
                'beta': self.bayesian_bandit.beta.tolist(),
                'pulls': self.bayesian_bandit.pulls.tolist(),
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
        }

    def load_state(self, state: dict[str, Any]):
        """
        Load tuning state from persistence.

        Args:
            state: Dict with saved tuning state
        """
        # Load current indices
        self.current_epsilon_idx = state.get('current_epsilon_idx', self.current_epsilon_idx)
        self.current_bayesian_idx = state.get('current_bayesian_idx', self.current_bayesian_idx)
        self.current_tuning_param = state.get('current_tuning_param', 'epsilon')

        # Load epsilon bandit
        if 'epsilon_bandit' in state:
            self.epsilon_bandit.alpha = np.array(state['epsilon_bandit']['alpha'])
            self.epsilon_bandit.beta = np.array(state['epsilon_bandit']['beta'])
            self.epsilon_bandit.pulls = np.array(state['epsilon_bandit']['pulls'])

        # Load bayesian bandit
        if 'bayesian_bandit' in state:
            self.bayesian_bandit.alpha = np.array(state['bayesian_bandit']['alpha'])
            self.bayesian_bandit.beta = np.array(state['bayesian_bandit']['beta'])
            self.bayesian_bandit.pulls = np.array(state['bayesian_bandit']['pulls'])

        # Load safe parameters
        for name, param_state in state.get('safe_params', {}).items():
            if name in self.safe_params:
                self.safe_params[name].current_value = param_state['current_value']
                self.safe_params[name].min_value = param_state['min_value']
                self.safe_params[name].max_value = param_state['max_value']
                self.safe_params[name].max_change_percent = param_state['max_change_percent']


def create_policy_tuner() -> PolicyTuner:
    """
    Factory function to create PolicyTuner.

    Returns:
        PolicyTuner instance
    """
    return PolicyTuner()
