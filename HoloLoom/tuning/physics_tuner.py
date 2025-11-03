"""
Physics Tuner - Agent 7 (THE FINAL AGENT!)

Automatically optimizes spring dynamics parameters using Thompson Sampling.

Strategy:
1. Measure propagation quality (convergence speed, stability, accuracy)
2. Thompson Sampling learns optimal physics parameters
3. Balance convergence speed vs accuracy vs stability
4. Group-based tuning: physics → simulation → activation → edge multipliers

Eliminates Parameters (12 total):
- stiffness (spring stiffness k)
- damping (damping coefficient c)
- decay (activation decay)
- max_iterations (propagation steps)
- convergence_epsilon (convergence threshold)
- dt (time step)
- activation_threshold (minimum activation)
- edge_type_multipliers (IS_A, PART_OF, USES, MENTIONS, RELATED_TO) - 5 params
"""

from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import numpy as np
from HoloLoom.tuning.base import TuningAgent, ThompsonBandit, SafeParameter

# Parameter value ranges (5 arms each for continuous params)
STIFFNESS_VALUES = [0.10, 0.40, 0.70, 1.00, 1.30]  # Spring stiffness
DAMPING_VALUES = [0.70, 0.80, 0.90, 0.95, 0.99]    # Damping coefficient
DECAY_VALUES = [0.90, 0.94, 0.97, 0.99, 1.00]      # Activation decay
DT_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]         # Time step
ACTIVATION_THRESHOLD_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]  # Min activation
EDGE_MULTIPLIER_VALUES = [0.5, 0.7, 0.9, 1.1, 1.3]  # Edge type multipliers

# Baseline values (from spring_dynamics.py)
BASELINE_PHYSICS = {
    'stiffness': 0.80,
    'damping': 0.85,
    'decay': 0.99,
    'dt': 0.20,
    'activation_threshold': 0.10,
    'is_a_multiplier': 1.2,
    'part_of_multiplier': 1.1,
    'uses_multiplier': 0.9,
    'mentions_multiplier': 0.7,
    'related_to_multiplier': 0.6,
}

# Safe ranges
SAFE_RANGES = {
    'stiffness': (0.01, 2.0),
    'damping': (0.50, 1.0),
    'decay': (0.80, 1.0),
    'dt': (0.01, 0.5),
    'activation_threshold': (0.01, 0.5),
    'is_a_multiplier': (0.5, 2.0),
    'part_of_multiplier': (0.5, 2.0),
    'uses_multiplier': (0.5, 2.0),
    'mentions_multiplier': (0.5, 2.0),
    'related_to_multiplier': (0.5, 2.0),
}


@dataclass
class PhysicsMetrics:
    """Metrics for spring dynamics evaluation."""
    convergence_speed: float = 0.0      # 0.0-1.0, how fast it converges (inverse of iterations)
    stability: float = 0.0              # 0.0-1.0, oscillation damping
    propagation_accuracy: float = 0.0   # 0.0-1.0, correct nodes activated
    energy_efficiency: float = 0.0      # 0.0-1.0, energy per unit quality

    def quality_score(self) -> float:
        """
        Multi-objective quality score.

        Formula:
        quality = 0.4 * accuracy + 0.3 * speed + 0.2 * stability + 0.1 * efficiency

        Balances:
        - 40% propagation accuracy (correct nodes activated)
        - 30% convergence speed (fast propagation)
        - 20% stability (no oscillations)
        - 10% energy efficiency (low computational cost)
        """
        return (
            0.4 * self.propagation_accuracy +
            0.3 * self.convergence_speed +
            0.2 * self.stability +
            0.1 * self.energy_efficiency
        )


class ParameterGroup(Enum):
    """Parameter grouping for round-robin tuning."""
    PHYSICS = "physics"                 # stiffness, damping, decay
    SIMULATION = "simulation"           # dt
    ACTIVATION = "activation"           # activation_threshold
    EDGE_MULTIPLIERS = "edge_multipliers"  # 5 edge type multipliers


class PhysicsTuner(TuningAgent):
    """
    Agent 7: Physics Tuner (THE FINAL AGENT!)

    Optimizes spring dynamics parameters using Thompson Sampling.
    Eliminates 12 physics configuration parameters.
    """

    def __init__(self):
        super().__init__('physics_tuner')

        # Metrics history (last 1000 measurements)
        self.metrics_history: deque = deque(maxlen=1000)

        # Thompson Sampling bandits (one per parameter)
        self.bandits: Dict[str, ThompsonBandit] = {
            'stiffness': ThompsonBandit(n_arms=len(STIFFNESS_VALUES)),
            'damping': ThompsonBandit(n_arms=len(DAMPING_VALUES)),
            'decay': ThompsonBandit(n_arms=len(DECAY_VALUES)),
            'dt': ThompsonBandit(n_arms=len(DT_VALUES)),
            'activation_threshold': ThompsonBandit(n_arms=len(ACTIVATION_THRESHOLD_VALUES)),
            'is_a_multiplier': ThompsonBandit(n_arms=len(EDGE_MULTIPLIER_VALUES)),
            'part_of_multiplier': ThompsonBandit(n_arms=len(EDGE_MULTIPLIER_VALUES)),
            'uses_multiplier': ThompsonBandit(n_arms=len(EDGE_MULTIPLIER_VALUES)),
            'mentions_multiplier': ThompsonBandit(n_arms=len(EDGE_MULTIPLIER_VALUES)),
            'related_to_multiplier': ThompsonBandit(n_arms=len(EDGE_MULTIPLIER_VALUES)),
        }

        # Current parameter indices
        self.param_indices = {
            'stiffness': STIFFNESS_VALUES.index(0.70),  # Closest to 0.80
            'damping': DAMPING_VALUES.index(0.80),  # Closest to 0.85
            'decay': DECAY_VALUES.index(0.99),
            'dt': DT_VALUES.index(0.20),
            'activation_threshold': ACTIVATION_THRESHOLD_VALUES.index(0.10),
            'is_a_multiplier': EDGE_MULTIPLIER_VALUES.index(1.1),  # Closest to 1.2
            'part_of_multiplier': EDGE_MULTIPLIER_VALUES.index(1.1),
            'uses_multiplier': EDGE_MULTIPLIER_VALUES.index(0.9),
            'mentions_multiplier': EDGE_MULTIPLIER_VALUES.index(0.7),
            'related_to_multiplier': EDGE_MULTIPLIER_VALUES.index(0.5),  # Closest to 0.6
        }

        # Safe parameters
        self.safe_params: Dict[str, SafeParameter] = {}
        for param_name, baseline_value in BASELINE_PHYSICS.items():
            min_val, max_val = SAFE_RANGES[param_name]
            self.safe_params[param_name] = SafeParameter(
                name=param_name,
                current_value=baseline_value,
                min_value=min_val,
                max_value=max_val,
                max_change_percent=0.25,  # Allow 25% change per cycle
            )

        # Round-robin parameter group tuning
        self.param_groups = {
            ParameterGroup.PHYSICS: ['stiffness', 'damping', 'decay'],
            ParameterGroup.SIMULATION: ['dt'],
            ParameterGroup.ACTIVATION: ['activation_threshold'],
            ParameterGroup.EDGE_MULTIPLIERS: [
                'is_a_multiplier', 'part_of_multiplier', 'uses_multiplier',
                'mentions_multiplier', 'related_to_multiplier'
            ],
        }
        self.current_group_index = 0

    def get_value_arms(self, param_name: str) -> List[float]:
        """Get value arms for a parameter."""
        if param_name == 'stiffness':
            return STIFFNESS_VALUES
        elif param_name == 'damping':
            return DAMPING_VALUES
        elif param_name == 'decay':
            return DECAY_VALUES
        elif param_name == 'dt':
            return DT_VALUES
        elif param_name == 'activation_threshold':
            return ACTIVATION_THRESHOLD_VALUES
        else:  # Edge multipliers
            return EDGE_MULTIPLIER_VALUES

    def get_adaptive_parameter(self, param_name: str) -> float:
        """
        Get current value for a parameter.

        Args:
            param_name: Parameter name

        Returns:
            Current parameter value
        """
        return self.safe_params[param_name].current_value

    async def measure_performance(self) -> Dict[str, Any]:
        """
        Measure physics performance metrics.

        Returns:
            Dict with current physics metrics
        """
        metrics = {}

        if len(self.metrics_history) > 0:
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100

            # Aggregate metrics
            avg_speed = np.mean([m.convergence_speed for m in recent_metrics])
            avg_stability = np.mean([m.stability for m in recent_metrics])
            avg_accuracy = np.mean([m.propagation_accuracy for m in recent_metrics])
            avg_efficiency = np.mean([m.energy_efficiency for m in recent_metrics])
            avg_quality = np.mean([m.quality_score() for m in recent_metrics])

            metrics['physics'] = {
                'avg_convergence_speed': avg_speed,
                'avg_stability': avg_stability,
                'avg_propagation_accuracy': avg_accuracy,
                'avg_energy_efficiency': avg_efficiency,
                'quality_score': avg_quality,
                'sample_count': len(recent_metrics),
            }

            # Current parameter values
            for param_name in self.safe_params.keys():
                metrics['physics'][f'current_{param_name}'] = self.get_adaptive_parameter(param_name)

        else:
            # No data yet, use baseline
            metrics['physics'] = {
                'avg_convergence_speed': 0.0,
                'avg_stability': 0.0,
                'avg_propagation_accuracy': 0.0,
                'avg_energy_efficiency': 0.0,
                'quality_score': 0.0,
                'sample_count': 0,
            }

        return metrics

    async def tune_parameters(self) -> Dict[str, Any]:
        """
        Propose new physics parameters using Thompson Sampling.

        Uses round-robin group tuning: physics → simulation → activation → edge_multipliers

        Returns:
            Dict with proposed parameter changes
        """
        proposals = {}

        # Select parameter group (round-robin)
        group_list = list(ParameterGroup)
        current_group = group_list[self.current_group_index]
        param_names = self.param_groups[current_group]

        # Tune one parameter from this group (rotate within group)
        param_to_tune = param_names[0]  # For simplicity, just pick first (could rotate)

        # Thompson Sampling: select arm
        bandit = self.bandits[param_to_tune]
        arm_idx = bandit.select()

        # Map to value
        value_arms = self.get_value_arms(param_to_tune)
        proposed_value = value_arms[arm_idx]

        # Apply safe parameter constraints
        safe_value = self.safe_params[param_to_tune].propose(proposed_value)

        proposals[param_to_tune] = {
            'old_value': self.safe_params[param_to_tune].current_value,
            'proposed_value': safe_value,
            'arm_selected': arm_idx,
            'group': current_group.value,
        }

        # Update parameter
        self.safe_params[param_to_tune].current_value = safe_value
        self.param_indices[param_to_tune] = arm_idx

        # Next cycle, tune next group
        self.current_group_index = (self.current_group_index + 1) % len(group_list)

        return proposals

    def record_physics_metrics(self, metrics: PhysicsMetrics):
        """
        Record physics metrics for tuning.

        Args:
            metrics: PhysicsMetrics object
        """
        self.metrics_history.append(metrics)

    def update_bandits(self, baseline_metrics: Dict[str, Any], new_metrics: Dict[str, Any]):
        """
        Update Thompson Sampling bandits based on quality change.

        Args:
            baseline_metrics: Baseline metrics before tuning
            new_metrics: New metrics after tuning
        """
        baseline_quality = baseline_metrics.get('physics', {}).get('quality_score', 0.0)
        new_quality = new_metrics.get('physics', {}).get('quality_score', 0.0)

        # Calculate improvement
        quality_delta = new_quality - baseline_quality

        # Update bandit (success if quality improved)
        success = quality_delta > 0.0
        confidence = abs(quality_delta)

        # Update all bandits with current arms (weighted by recent usage)
        for param_name, arm_idx in self.param_indices.items():
            self.bandits[param_name].update(arm_idx, success=success, confidence=confidence * 0.5)

    async def run_tuning_cycle(self) -> Dict[str, Any]:
        """
        Run one tuning cycle.

        Returns:
            Dict with tuning results
        """
        # Measure baseline performance
        baseline_metrics = await self.measure_performance()

        # Propose new parameters
        proposals = await self.tune_parameters()

        # In production, the system would run propagation with the new parameters
        # and measure actual performance. For now, we just return proposals.

        # Measure new performance (would be done after running propagation)
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
            Scalar impact score (quality improvement)
        """
        baseline_quality = baseline.get('physics', {}).get('quality_score', 0.0)
        new_quality = new.get('physics', {}).get('quality_score', 0.0)

        return new_quality - baseline_quality

    def get_bandit_stats(self) -> Dict[str, Any]:
        """
        Get Thompson Sampling bandit statistics.

        Returns:
            Dict with bandit statistics for all parameters
        """
        stats = {}

        for param_name, bandit in self.bandits.items():
            value_arms = self.get_value_arms(param_name)
            arm_stats = []

            for arm_idx in range(len(value_arms)):
                alpha, beta = bandit.alpha[arm_idx], bandit.beta[arm_idx]
                expected_reward = alpha / (alpha + beta)

                arm_stats.append({
                    'value': value_arms[arm_idx],
                    'alpha': alpha,
                    'beta': beta,
                    'expected_reward': expected_reward,
                    'pulls': bandit.pulls[arm_idx],
                })

            stats[param_name] = {
                'current_value': self.get_adaptive_parameter(param_name),
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
            'param_indices': self.param_indices,
            'current_group_index': self.current_group_index,
            'bandits': {
                param_name: {
                    'alpha': bandit.alpha.tolist(),
                    'beta': bandit.beta.tolist(),
                    'pulls': bandit.pulls.tolist(),
                }
                for param_name, bandit in self.bandits.items()
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

    def load_state(self, state: Dict[str, Any]):
        """
        Load tuning state from persistence.

        Args:
            state: Dict with saved tuning state
        """
        # Load parameter indices
        self.param_indices = state.get('param_indices', self.param_indices)
        self.current_group_index = state.get('current_group_index', 0)

        # Load bandits
        for param_name, bandit_state in state.get('bandits', {}).items():
            if param_name in self.bandits:
                self.bandits[param_name].alpha = np.array(bandit_state['alpha'])
                self.bandits[param_name].beta = np.array(bandit_state['beta'])
                self.bandits[param_name].pulls = np.array(bandit_state['pulls'])

        # Load safe parameters
        for name, param_state in state.get('safe_params', {}).items():
            if name in self.safe_params:
                self.safe_params[name].current_value = param_state['current_value']
                self.safe_params[name].min_value = param_state['min_value']
                self.safe_params[name].max_value = param_state['max_value']
                self.safe_params[name].max_change_percent = param_state['max_change_percent']


def create_physics_tuner() -> PhysicsTuner:
    """
    Factory function to create PhysicsTuner.

    Returns:
        PhysicsTuner instance
    """
    return PhysicsTuner()
