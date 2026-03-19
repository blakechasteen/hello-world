"""
Hierarchical Learning System for CARTS
======================================

Multi-level learning across strategy, payload, and mutation layers.

Architecture:
    Level 1: STRATEGY (12 attack strategies)
        └── Level 2: PAYLOAD (per-strategy payload variants)
            └── Level 3: MUTATION (10 mutation types)

Each level maintains its own Thompson Sampling bandits,
with information flowing up and down the hierarchy:
- Top-down: Strategy selection guides payload selection
- Bottom-up: Mutation success improves payload, which improves strategy

Philosophy: "Learn at the right level of abstraction."

Author: CARTS Team
Date: 2025-12-03
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hololoom.bandits.beta_arm import BetaArm

from ..mutator import MutationType
from ..strategies import AttackStrategy
from .learning_protocols import (
    LearningLevel,
)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HierarchicalArm:
    """
    A single arm at any level of the hierarchy.

    Delegates Beta math to BetaArm (canonical implementation).
    """

    level: LearningLevel
    arm_id: str  # Strategy name, payload ID, or mutation type
    parent_id: str | None = None  # Link to parent level
    _arm: BetaArm = field(default_factory=BetaArm)
    total_pulls: int = 0
    total_rewards: float = 0.0
    children: set[str] = field(default_factory=set)
    last_updated: str | None = None

    @property
    def alpha(self) -> float:
        return self._arm.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._arm.alpha = value

    @property
    def beta(self) -> float:
        return self._arm.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._arm.beta = value

    def sample(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)."""
        return self._arm.sample()

    def update(self, success: bool, reward: float = 1.0):
        """Update arm based on result."""
        self.total_pulls += 1

        if success:
            self._arm.update(success=True, weight=reward)
            self.total_rewards += reward
        else:
            self._arm.update(success=False, weight=1.0)

        self.last_updated = datetime.now().isoformat()

    @property
    def expected_reward(self) -> float:
        """Expected reward (mean of Beta distribution)."""
        return self._arm.expected_value()

    @property
    def success_rate(self) -> float:
        """Empirical success rate."""
        if self.total_pulls == 0:
            return 0.0
        return self.total_rewards / self.total_pulls

    @property
    def uncertainty(self) -> float:
        """Uncertainty (variance of Beta distribution)."""
        return self._arm.variance()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'level': self.level.value,
            'arm_id': self.arm_id,
            'parent_id': self.parent_id,
            'alpha': self.alpha,
            'beta': self.beta,
            'total_pulls': self.total_pulls,
            'total_rewards': self.total_rewards,
            'children': list(self.children),
            'last_updated': self.last_updated,
            'expected_reward': self.expected_reward,
            'success_rate': self.success_rate,
            'uncertainty': self.uncertainty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'HierarchicalArm':
        """Create from dictionary."""
        return cls(
            level=LearningLevel(data['level']),
            arm_id=data['arm_id'],
            parent_id=data.get('parent_id'),
            _arm=BetaArm(alpha=data['alpha'], beta=data['beta']),
            total_pulls=data['total_pulls'],
            total_rewards=data['total_rewards'],
            children=set(data.get('children', [])),
            last_updated=data.get('last_updated'),
        )


@dataclass
class HierarchicalSelection:
    """Result of a hierarchical selection."""

    strategy: AttackStrategy
    payload_id: str | None = None
    mutation_type: MutationType | None = None

    strategy_arm: HierarchicalArm | None = None
    payload_arm: HierarchicalArm | None = None
    mutation_arm: HierarchicalArm | None = None

    sampled_values: dict[str, float] = field(default_factory=dict)


@dataclass
class HierarchicalUpdate:
    """Update to propagate through hierarchy."""

    strategy: AttackStrategy
    payload_id: str | None
    mutation_type: MutationType | None
    success: bool
    severity: float = 0.0

    # Propagation weights
    strategy_weight: float = 0.5  # How much success flows to strategy
    payload_weight: float = 0.3   # How much flows to payload
    mutation_weight: float = 0.2  # How much flows to mutation


# =============================================================================
# Main Hierarchical Learner Class
# =============================================================================

class HierarchicalLearner:
    """
    Hierarchical Thompson Sampling learner for CARTS.

    Maintains three levels of bandits:
    - Level 1 (STRATEGY): Which attack type works best?
    - Level 2 (PAYLOAD): Which payload variants succeed?
    - Level 3 (MUTATION): Which mutations improve payloads?

    Features:
    - Top-down selection: Strategy guides payload guides mutation
    - Bottom-up credit assignment: Success propagates upward
    - Cross-level transfer: Hot payloads seed new strategies
    - Adaptive weighting: Learn optimal credit assignment

    Example:
        learner = HierarchicalLearner()

        # Full hierarchical selection
        selection = learner.select_full_hierarchy()
        print(f"Strategy: {selection.strategy.value}")
        print(f"Payload: {selection.payload_id}")
        print(f"Mutation: {selection.mutation_type.value}")

        # Execute attack...

        # Update with result
        learner.update_hierarchy(HierarchicalUpdate(
            strategy=selection.strategy,
            payload_id=selection.payload_id,
            mutation_type=selection.mutation_type,
            success=True,
            severity=0.85
        ))
    """

    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        propagation_decay: float = 0.8,  # Decay when propagating up
        min_samples_for_child: int = 3,  # Min samples before using child-level data
    ):
        """
        Initialize hierarchical learner.

        Args:
            initial_alpha: Initial alpha for all arms
            initial_beta: Initial beta for all arms
            propagation_decay: Decay factor when propagating credit upward
            min_samples_for_child: Min samples before trusting child data
        """
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.propagation_decay = propagation_decay
        self.min_samples_for_child = min_samples_for_child

        # Level 1: Strategy arms (12 strategies)
        self.strategy_arms: dict[AttackStrategy, HierarchicalArm] = {}

        # Level 2: Payload arms (per strategy)
        # strategy -> payload_id -> arm
        # NOTE: Must be initialized BEFORE _init_strategy_arms() which populates it
        self.payload_arms: dict[AttackStrategy, dict[str, HierarchicalArm]] = {}

        # Level 3: Mutation arms (per payload)
        # payload_id -> mutation_type -> arm
        self.mutation_arms: dict[str, dict[MutationType, HierarchicalArm]] = {}

        # Initialize strategy arms (must be after payload_arms is defined)
        self._init_strategy_arms()

        # Track known entities
        self._known_payloads: set[str] = set()
        self._known_mutations: set[tuple[str, MutationType]] = set()

        # Selection history
        self.selection_history: list[HierarchicalSelection] = []

    def _init_strategy_arms(self):
        """Initialize strategy-level arms."""
        for strategy in AttackStrategy:
            self.strategy_arms[strategy] = HierarchicalArm(
                level=LearningLevel.STRATEGY,
                arm_id=strategy.value,
                parent_id=None,
                alpha=self.initial_alpha,
                beta=self.initial_beta,
            )
            self.payload_arms[strategy] = {}

    def _get_or_create_payload_arm(
        self,
        strategy: AttackStrategy,
        payload_id: str
    ) -> HierarchicalArm:
        """Get or create a payload arm."""
        if payload_id not in self.payload_arms[strategy]:
            arm = HierarchicalArm(
                level=LearningLevel.PAYLOAD,
                arm_id=payload_id,
                parent_id=strategy.value,
                alpha=self.initial_alpha,
                beta=self.initial_beta,
            )
            self.payload_arms[strategy][payload_id] = arm
            self.strategy_arms[strategy].children.add(payload_id)
            self._known_payloads.add(payload_id)
            self.mutation_arms[payload_id] = {}

        return self.payload_arms[strategy][payload_id]

    def _get_or_create_mutation_arm(
        self,
        payload_id: str,
        mutation_type: MutationType
    ) -> HierarchicalArm:
        """Get or create a mutation arm."""
        if payload_id not in self.mutation_arms:
            self.mutation_arms[payload_id] = {}

        if mutation_type not in self.mutation_arms[payload_id]:
            arm = HierarchicalArm(
                level=LearningLevel.MUTATION,
                arm_id=mutation_type.value,
                parent_id=payload_id,
                alpha=self.initial_alpha,
                beta=self.initial_beta,
            )
            self.mutation_arms[payload_id][mutation_type] = arm
            self._known_mutations.add((payload_id, mutation_type))

            # Link to parent payload
            if payload_id in self._known_payloads:
                for strategy_payloads in self.payload_arms.values():
                    if payload_id in strategy_payloads:
                        strategy_payloads[payload_id].children.add(mutation_type.value)
                        break

        return self.mutation_arms[payload_id][mutation_type]

    # =========================================================================
    # Selection Methods
    # =========================================================================

    def select_strategy(self) -> AttackStrategy:
        """
        Select attack strategy using Thompson Sampling.

        Returns:
            Selected AttackStrategy
        """
        samples = {
            strategy: arm.sample()
            for strategy, arm in self.strategy_arms.items()
        }

        return max(samples, key=samples.get)

    def select_payload(
        self,
        strategy: AttackStrategy,
        available_payloads: list[str] | None = None
    ) -> str | None:
        """
        Select payload for given strategy.

        Args:
            strategy: Selected strategy
            available_payloads: If provided, only select from these payloads

        Returns:
            Selected payload ID or None if no payloads
        """
        strategy_payloads = self.payload_arms.get(strategy, {})

        if not strategy_payloads:
            return None

        if available_payloads:
            strategy_payloads = {
                pid: arm for pid, arm in strategy_payloads.items()
                if pid in available_payloads
            }

        if not strategy_payloads:
            return None

        samples = {
            payload_id: arm.sample()
            for payload_id, arm in strategy_payloads.items()
        }

        return max(samples, key=samples.get)

    def select_mutation(
        self,
        payload_id: str,
        available_mutations: list[MutationType] | None = None
    ) -> MutationType | None:
        """
        Select mutation type for given payload.

        Args:
            payload_id: Selected payload
            available_mutations: If provided, only select from these mutations

        Returns:
            Selected MutationType or None if no mutations
        """
        payload_mutations = self.mutation_arms.get(payload_id, {})

        if not payload_mutations:
            # Fall back to global mutation priors (from all payloads)
            payload_mutations = self._get_global_mutation_priors()

        if available_mutations:
            payload_mutations = {
                mut: arm for mut, arm in payload_mutations.items()
                if mut in available_mutations
            }

        if not payload_mutations:
            return None

        samples = {
            mutation: arm.sample()
            for mutation, arm in payload_mutations.items()
        }

        return max(samples, key=samples.get)

    def select_full_hierarchy(
        self,
        available_payloads: dict[AttackStrategy, list[str]] | None = None,
        available_mutations: list[MutationType] | None = None
    ) -> HierarchicalSelection:
        """
        Full hierarchical selection: strategy → payload → mutation.

        Args:
            available_payloads: Map of strategy -> available payload IDs
            available_mutations: Available mutation types

        Returns:
            HierarchicalSelection with all levels
        """
        sampled_values = {}

        # Level 1: Select strategy
        strategy_samples = {
            strategy: arm.sample()
            for strategy, arm in self.strategy_arms.items()
        }
        strategy = max(strategy_samples, key=strategy_samples.get)
        sampled_values['strategy'] = strategy_samples[strategy]

        # Level 2: Select payload
        strategy_available = (
            available_payloads.get(strategy, []) if available_payloads else None
        )
        payload_id = self.select_payload(strategy, strategy_available)
        if payload_id:
            sampled_values['payload'] = self.payload_arms[strategy][payload_id].sample()

        # Level 3: Select mutation
        mutation_type = None
        if payload_id:
            mutation_type = self.select_mutation(payload_id, available_mutations)
            if mutation_type and payload_id in self.mutation_arms:
                if mutation_type in self.mutation_arms[payload_id]:
                    sampled_values['mutation'] = self.mutation_arms[payload_id][mutation_type].sample()

        selection = HierarchicalSelection(
            strategy=strategy,
            payload_id=payload_id,
            mutation_type=mutation_type,
            strategy_arm=self.strategy_arms[strategy],
            payload_arm=self.payload_arms[strategy].get(payload_id) if payload_id else None,
            mutation_arm=(
                self.mutation_arms.get(payload_id, {}).get(mutation_type)
                if payload_id and mutation_type else None
            ),
            sampled_values=sampled_values,
        )

        self.selection_history.append(selection)
        return selection

    def _get_global_mutation_priors(self) -> dict[MutationType, HierarchicalArm]:
        """
        Get aggregated mutation priors across all payloads.

        Used when a payload has no mutation history.
        """
        global_priors: dict[MutationType, tuple[float, float, int]] = {}

        for payload_mutations in self.mutation_arms.values():
            for mutation_type, arm in payload_mutations.items():
                if mutation_type not in global_priors:
                    global_priors[mutation_type] = (0.0, 0.0, 0)

                alpha, beta, count = global_priors[mutation_type]
                global_priors[mutation_type] = (
                    alpha + arm.alpha - 1.0,
                    beta + arm.beta - 1.0,
                    count + arm.total_pulls
                )

        # Create temporary arms from aggregated priors
        result: dict[MutationType, HierarchicalArm] = {}

        for mutation_type in MutationType:
            if mutation_type in global_priors:
                alpha, beta, count = global_priors[mutation_type]
                arm = HierarchicalArm(
                    level=LearningLevel.MUTATION,
                    arm_id=mutation_type.value,
                    alpha=max(1.0, alpha + 1.0),
                    beta=max(1.0, beta + 1.0),
                    total_pulls=count,
                )
            else:
                arm = HierarchicalArm(
                    level=LearningLevel.MUTATION,
                    arm_id=mutation_type.value,
                    alpha=self.initial_alpha,
                    beta=self.initial_beta,
                )
            result[mutation_type] = arm

        return result

    # =========================================================================
    # Update Methods
    # =========================================================================

    def update_strategy(
        self,
        strategy: AttackStrategy,
        success: bool,
        severity: float = 1.0
    ):
        """Update strategy-level arm."""
        if strategy in self.strategy_arms:
            self.strategy_arms[strategy].update(success, severity if success else 0.0)

    def update_payload(
        self,
        strategy: AttackStrategy,
        payload_id: str,
        success: bool,
        severity: float = 1.0
    ):
        """Update payload-level arm."""
        arm = self._get_or_create_payload_arm(strategy, payload_id)
        arm.update(success, severity if success else 0.0)

    def update_mutation(
        self,
        payload_id: str,
        mutation_type: MutationType,
        success: bool,
        severity: float = 1.0
    ):
        """Update mutation-level arm."""
        arm = self._get_or_create_mutation_arm(payload_id, mutation_type)
        arm.update(success, severity if success else 0.0)

    def update_hierarchy(self, update: HierarchicalUpdate):
        """
        Update entire hierarchy with propagation.

        Credit flows from bottom to top with decay.

        Args:
            update: HierarchicalUpdate with all levels and result
        """
        reward = update.severity if update.success else 0.0

        # Level 3: Update mutation (if provided)
        if update.mutation_type and update.payload_id:
            mutation_reward = reward * update.mutation_weight
            self.update_mutation(
                update.payload_id,
                update.mutation_type,
                update.success,
                mutation_reward
            )

        # Level 2: Update payload (if provided)
        if update.payload_id:
            payload_reward = reward * update.payload_weight
            # Add propagated credit from mutation
            if update.mutation_type:
                payload_reward += reward * update.mutation_weight * self.propagation_decay

            self.update_payload(
                update.strategy,
                update.payload_id,
                update.success,
                payload_reward
            )

        # Level 1: Update strategy
        strategy_reward = reward * update.strategy_weight
        # Add propagated credit from payload
        if update.payload_id:
            strategy_reward += reward * update.payload_weight * self.propagation_decay
            # Add doubly-propagated credit from mutation
            if update.mutation_type:
                strategy_reward += reward * update.mutation_weight * (self.propagation_decay ** 2)

        self.update_strategy(
            update.strategy,
            update.success,
            strategy_reward
        )

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get statistics for all strategies."""
        total_pulls = sum(arm.total_pulls for arm in self.strategy_arms.values())
        total_rewards = sum(arm.total_rewards for arm in self.strategy_arms.values())

        best_arm = max(self.strategy_arms.values(), key=lambda a: a.expected_reward)

        return {
            'total_pulls': total_pulls,
            'total_rewards': total_rewards,
            'overall_success_rate': total_rewards / max(1, total_pulls),
            'best_strategy': best_arm.arm_id,
            'best_expected_reward': best_arm.expected_reward,
            'arms': {
                strategy.value: arm.to_dict()
                for strategy, arm in self.strategy_arms.items()
            }
        }

    def get_payload_stats(
        self,
        strategy: AttackStrategy | None = None
    ) -> dict[str, Any]:
        """
        Get statistics for payloads.

        Args:
            strategy: If provided, only return payloads for this strategy
        """
        if strategy:
            payloads = self.payload_arms.get(strategy, {})
            return {
                'strategy': strategy.value,
                'total_payloads': len(payloads),
                'payloads': {
                    pid: arm.to_dict() for pid, arm in payloads.items()
                }
            }

        # All payloads
        all_payloads = {}
        for strat, payloads in self.payload_arms.items():
            for pid, arm in payloads.items():
                all_payloads[pid] = arm.to_dict()

        return {
            'total_payloads': len(all_payloads),
            'payloads': all_payloads
        }

    def get_mutation_stats(
        self,
        payload_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get statistics for mutations.

        Args:
            payload_id: If provided, only return mutations for this payload
        """
        if payload_id:
            mutations = self.mutation_arms.get(payload_id, {})
            return {
                'payload_id': payload_id,
                'total_mutations': len(mutations),
                'mutations': {
                    mut.value: arm.to_dict()
                    for mut, arm in mutations.items()
                }
            }

        # Global mutation stats
        global_priors = self._get_global_mutation_priors()
        return {
            'total_mutation_types': len(global_priors),
            'mutations': {
                mut.value: arm.to_dict()
                for mut, arm in global_priors.items()
            }
        }

    def get_hierarchy_summary(self) -> dict[str, Any]:
        """Get complete hierarchy summary."""
        return {
            'strategy_level': self.get_strategy_stats(),
            'payload_level': self.get_payload_stats(),
            'mutation_level': self.get_mutation_stats(),
            'total_known_payloads': len(self._known_payloads),
            'total_known_mutations': len(self._known_mutations),
        }

    def get_best_path(self, strategy: AttackStrategy) -> dict[str, Any]:
        """
        Get best path through hierarchy for a strategy.

        Returns the highest expected reward path from strategy → payload → mutation.
        """
        result = {
            'strategy': strategy.value,
            'strategy_reward': self.strategy_arms[strategy].expected_reward,
            'best_payload': None,
            'best_mutation': None,
        }

        # Find best payload
        payloads = self.payload_arms.get(strategy, {})
        if payloads:
            best_payload = max(payloads.values(), key=lambda a: a.expected_reward)
            result['best_payload'] = {
                'id': best_payload.arm_id,
                'expected_reward': best_payload.expected_reward,
            }

            # Find best mutation for this payload
            mutations = self.mutation_arms.get(best_payload.arm_id, {})
            if mutations:
                best_mutation = max(mutations.values(), key=lambda a: a.expected_reward)
                result['best_mutation'] = {
                    'type': best_mutation.arm_id,
                    'expected_reward': best_mutation.expected_reward,
                }

        return result

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Path):
        """Save learner state to file."""
        state = {
            'initial_alpha': self.initial_alpha,
            'initial_beta': self.initial_beta,
            'propagation_decay': self.propagation_decay,
            'min_samples_for_child': self.min_samples_for_child,
            'strategy_arms': {
                s.value: arm.to_dict() for s, arm in self.strategy_arms.items()
            },
            'payload_arms': {},
            'mutation_arms': {},
            'saved_at': datetime.now().isoformat(),
        }

        # Save payload arms
        for strategy, payloads in self.payload_arms.items():
            state['payload_arms'][strategy.value] = {
                pid: arm.to_dict() for pid, arm in payloads.items()
            }

        # Save mutation arms
        for payload_id, mutations in self.mutation_arms.items():
            state['mutation_arms'][payload_id] = {
                mut.value: arm.to_dict() for mut, arm in mutations.items()
            }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: Path):
        """Load learner state from file."""
        path = Path(path)
        if not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        self.initial_alpha = state.get('initial_alpha', 1.0)
        self.initial_beta = state.get('initial_beta', 1.0)
        self.propagation_decay = state.get('propagation_decay', 0.8)
        self.min_samples_for_child = state.get('min_samples_for_child', 3)

        # Load strategy arms
        for strategy_value, arm_data in state.get('strategy_arms', {}).items():
            try:
                strategy = AttackStrategy(strategy_value)
                self.strategy_arms[strategy] = HierarchicalArm.from_dict(arm_data)
            except ValueError:
                pass

        # Load payload arms
        for strategy_value, payloads in state.get('payload_arms', {}).items():
            try:
                strategy = AttackStrategy(strategy_value)
                if strategy not in self.payload_arms:
                    self.payload_arms[strategy] = {}

                for pid, arm_data in payloads.items():
                    self.payload_arms[strategy][pid] = HierarchicalArm.from_dict(arm_data)
                    self._known_payloads.add(pid)
            except ValueError:
                pass

        # Load mutation arms
        for payload_id, mutations in state.get('mutation_arms', {}).items():
            self.mutation_arms[payload_id] = {}

            for mut_value, arm_data in mutations.items():
                try:
                    mutation = MutationType(mut_value)
                    self.mutation_arms[payload_id][mutation] = HierarchicalArm.from_dict(arm_data)
                    self._known_mutations.add((payload_id, mutation))
                except ValueError:
                    pass

    def reset(self):
        """Reset learner to initial state."""
        for arm in self.strategy_arms.values():
            arm.alpha = self.initial_alpha
            arm.beta = self.initial_beta
            arm.total_pulls = 0
            arm.total_rewards = 0.0
            arm.children.clear()
            arm.last_updated = None

        self.payload_arms = {s: {} for s in AttackStrategy}
        self.mutation_arms = {}
        self._known_payloads.clear()
        self._known_mutations.clear()
        self.selection_history.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_hierarchical_learner(
    initial_alpha: float = 1.0,
    initial_beta: float = 1.0,
    propagation_decay: float = 0.8,
    min_samples_for_child: int = 3,
    state_path: Path | None = None
) -> HierarchicalLearner:
    """
    Create a HierarchicalLearner, optionally loading saved state.

    Args:
        initial_alpha: Initial alpha for all arms
        initial_beta: Initial beta for all arms
        propagation_decay: Decay factor when propagating credit upward
        min_samples_for_child: Min samples before trusting child data
        state_path: Path to load saved state from

    Returns:
        Configured HierarchicalLearner
    """
    learner = HierarchicalLearner(
        initial_alpha=initial_alpha,
        initial_beta=initial_beta,
        propagation_decay=propagation_decay,
        min_samples_for_child=min_samples_for_child,
    )

    if state_path and Path(state_path).exists():
        learner.load(state_path)

    return learner


def create_hierarchical_update(
    strategy: AttackStrategy,
    success: bool,
    severity: float = 0.0,
    payload_id: str | None = None,
    mutation_type: MutationType | None = None,
    strategy_weight: float = 0.5,
    payload_weight: float = 0.3,
    mutation_weight: float = 0.2,
) -> HierarchicalUpdate:
    """
    Create a HierarchicalUpdate for updating the learner.

    Args:
        strategy: Attack strategy used
        success: Whether attack succeeded
        severity: Severity of vulnerability (0.0-1.0)
        payload_id: Payload ID (optional)
        mutation_type: Mutation type used (optional)
        strategy_weight: Credit weight for strategy
        payload_weight: Credit weight for payload
        mutation_weight: Credit weight for mutation

    Returns:
        HierarchicalUpdate instance
    """
    return HierarchicalUpdate(
        strategy=strategy,
        payload_id=payload_id,
        mutation_type=mutation_type,
        success=success,
        severity=severity,
        strategy_weight=strategy_weight,
        payload_weight=payload_weight,
        mutation_weight=mutation_weight,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data Classes
    'HierarchicalArm',
    'HierarchicalSelection',
    'HierarchicalUpdate',

    # Main Class
    'HierarchicalLearner',

    # Convenience Functions
    'create_hierarchical_learner',
    'create_hierarchical_update',
]
