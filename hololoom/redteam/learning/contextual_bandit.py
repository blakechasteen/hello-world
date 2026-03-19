"""
Contextual Thompson Sampling Bandit for CARTS
==============================================

Context-aware attack strategy selection that learns per-context
effectiveness patterns. Different target types, vulnerability classes,
and defense levels may respond differently to attack strategies.

Architecture:
- Flat bandit: One arm per strategy (12 arms)
- Contextual bandit: One bandit per context (N contexts × 12 arms)

Context Key:
    (target_type, vuln_class, defense_level)

Example contexts:
- ("llm", "prompt_injection", "basic")
- ("api", "encoding_bypass", "advanced")
- ("web_app", "authority_escalation", "none")

Philosophy: "What works against GPT may not work against Claude."

Author: CARTS Team
Date: 2025-12-03
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hololoom.bandits.beta_arm import BetaArm

from ..strategies import AttackStrategy
from .learning_protocols import (
    ContextKey,
)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ContextualArm:
    """
    A single arm in a contextual bandit.

    Delegates Beta math to BetaArm (canonical implementation).
    Tracks strategy effectiveness for a specific context.
    """

    strategy: AttackStrategy
    context: ContextKey
    _arm: BetaArm = field(default_factory=BetaArm)
    total_pulls: int = 0
    total_rewards: float = 0.0
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
        """
        Update arm based on result.

        Args:
            success: True if attack found vulnerability
            reward: Severity of vulnerability (0.0-1.0)
        """
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
            'strategy': self.strategy.value,
            'context': {
                'target_type': self.context.target_type,
                'vuln_class': self.context.vuln_class,
                'defense_level': self.context.defense_level,
            },
            'alpha': self.alpha,
            'beta': self.beta,
            'total_pulls': self.total_pulls,
            'total_rewards': self.total_rewards,
            'last_updated': self.last_updated,
            'expected_reward': self.expected_reward,
            'success_rate': self.success_rate,
            'uncertainty': self.uncertainty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ContextualArm':
        """Create from dictionary."""
        context_data = data.get('context', {})
        return cls(
            strategy=AttackStrategy(data['strategy']),
            context=ContextKey(
                target_type=context_data.get('target_type', 'unknown'),
                vuln_class=context_data.get('vuln_class', 'unknown'),
                defense_level=context_data.get('defense_level', 'unknown'),
            ),
            _arm=BetaArm(alpha=data['alpha'], beta=data['beta']),
            total_pulls=data['total_pulls'],
            total_rewards=data['total_rewards'],
            last_updated=data.get('last_updated'),
        )


@dataclass
class ContextualSelectionResult:
    """Result of a contextual strategy selection."""

    context: ContextKey
    strategy: AttackStrategy
    sampled_value: float
    all_samples: dict[AttackStrategy, float]
    arm: ContextualArm
    used_global_fallback: bool = False  # True if context had no data


# =============================================================================
# Main Contextual Bandit Class
# =============================================================================

class ContextualRedTeamBandit:
    """
    Contextual Thompson Sampling bandit for attack strategy selection.

    Maintains separate Beta priors per (target_type, vuln_class, defense_level)
    context, enabling specialized learning for different attack scenarios.

    Features:
    - Per-context learning: Different targets/defenses need different strategies
    - Global fallback: Uses aggregated priors when context is unseen
    - Context transfer: Hot contexts can seed cold contexts
    - Decay support: For non-stationary environments

    Example:
        bandit = ContextualRedTeamBandit()

        # Define context
        context = ContextKey(
            target_type="llm",
            vuln_class="prompt_injection",
            defense_level="basic"
        )

        # Select strategy for this context
        strategy = bandit.select(context)

        # Execute attack
        result = await executor.execute_attack(strategy, payload, context)

        # Update bandit for this context
        bandit.update(context, strategy, result.bypassed, result.severity)

        # Over time, bandit learns what works for different contexts
        stats = bandit.get_context_stats(context)
        print(f"Best for {context}: {stats['best_strategy']}")
    """

    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        decay_rate: float = 0.0,
        global_weight: float = 0.3,  # Weight for global prior when context unseen
        min_context_samples: int = 5,  # Min samples before trusting context-specific data
    ):
        """
        Initialize contextual bandit.

        Args:
            initial_alpha: Initial alpha for all arms
            initial_beta: Initial beta for all arms
            decay_rate: Decay rate for priors (0 = no decay)
            global_weight: Weight given to global prior for unseen contexts
            min_context_samples: Minimum samples before trusting context data
        """
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.decay_rate = decay_rate
        self.global_weight = global_weight
        self.min_context_samples = min_context_samples

        # Context → Strategy → Arm mapping
        self.context_arms: dict[ContextKey, dict[AttackStrategy, ContextualArm]] = {}

        # Global arms (aggregated across all contexts)
        self.global_arms: dict[AttackStrategy, ContextualArm] = {}
        self._init_global_arms()

        # Selection history
        self.selection_history: list[ContextualSelectionResult] = []

        # Tracked contexts
        self._known_contexts: set[ContextKey] = set()

    def _init_global_arms(self):
        """Initialize global arms for fallback."""
        global_context = ContextKey(
            target_type="global",
            vuln_class="global",
            defense_level="global"
        )
        for strategy in AttackStrategy:
            self.global_arms[strategy] = ContextualArm(
                strategy=strategy,
                context=global_context,
                alpha=self.initial_alpha,
                beta=self.initial_beta,
            )

    def _get_or_create_context_arms(
        self,
        context: ContextKey
    ) -> dict[AttackStrategy, ContextualArm]:
        """Get or create arms for a context."""
        if context not in self.context_arms:
            self.context_arms[context] = {}
            for strategy in AttackStrategy:
                self.context_arms[context][strategy] = ContextualArm(
                    strategy=strategy,
                    context=context,
                    alpha=self.initial_alpha,
                    beta=self.initial_beta,
                )
            self._known_contexts.add(context)

        return self.context_arms[context]

    def _get_context_sample_count(self, context: ContextKey) -> int:
        """Get total samples for a context."""
        if context not in self.context_arms:
            return 0
        return sum(arm.total_pulls for arm in self.context_arms[context].values())

    def select(self, context: ContextKey) -> AttackStrategy:
        """
        Select attack strategy for given context using Thompson Sampling.

        If context has insufficient samples, blends with global prior.

        Args:
            context: Context key for this selection

        Returns:
            Selected AttackStrategy
        """
        # Apply decay if configured
        if self.decay_rate > 0:
            self._apply_decay(context)

        # Get context-specific arms
        context_arms = self._get_or_create_context_arms(context)
        context_samples = self._get_context_sample_count(context)

        # Determine blending weights
        use_global_fallback = context_samples < self.min_context_samples

        # Sample from each strategy
        samples: dict[AttackStrategy, float] = {}

        for strategy in AttackStrategy:
            context_sample = context_arms[strategy].sample()

            if use_global_fallback:
                # Blend with global prior
                global_sample = self.global_arms[strategy].sample()
                blend_weight = min(context_samples / self.min_context_samples, 1.0)
                samples[strategy] = (
                    blend_weight * context_sample +
                    (1 - blend_weight) * global_sample
                )
            else:
                samples[strategy] = context_sample

        # Select strategy with highest sample
        selected = max(samples, key=samples.get)

        # Record selection
        self.selection_history.append(ContextualSelectionResult(
            context=context,
            strategy=selected,
            sampled_value=samples[selected],
            all_samples=samples,
            arm=context_arms[selected],
            used_global_fallback=use_global_fallback,
        ))

        return selected

    def select_top_k(self, context: ContextKey, k: int = 3) -> list[AttackStrategy]:
        """
        Select top-k strategies for given context.

        Args:
            context: Context key
            k: Number of strategies to select

        Returns:
            List of top-k strategies
        """
        context_arms = self._get_or_create_context_arms(context)
        context_samples = self._get_context_sample_count(context)
        use_global_fallback = context_samples < self.min_context_samples

        samples: dict[AttackStrategy, float] = {}

        for strategy in AttackStrategy:
            context_sample = context_arms[strategy].sample()

            if use_global_fallback:
                global_sample = self.global_arms[strategy].sample()
                blend_weight = min(context_samples / self.min_context_samples, 1.0)
                samples[strategy] = (
                    blend_weight * context_sample +
                    (1 - blend_weight) * global_sample
                )
            else:
                samples[strategy] = context_sample

        sorted_strategies = sorted(samples, key=samples.get, reverse=True)
        return sorted_strategies[:k]

    def update(
        self,
        context: ContextKey,
        strategy: AttackStrategy,
        success: bool,
        reward: float = 1.0
    ):
        """
        Update bandit for given context.

        Updates both context-specific and global arms.

        Args:
            context: Context key
            strategy: Strategy that was used
            success: True if attack found vulnerability
            reward: Severity of vulnerability (0.0-1.0)
        """
        # Update context-specific arm
        context_arms = self._get_or_create_context_arms(context)
        if strategy in context_arms:
            context_arms[strategy].update(success, reward if success else 0.0)

        # Update global arm (for fallback learning)
        if strategy in self.global_arms:
            self.global_arms[strategy].update(success, reward if success else 0.0)

    def update_batch(
        self,
        results: list[tuple[ContextKey, AttackStrategy, bool, float]]
    ):
        """
        Update bandit with multiple results.

        Args:
            results: List of (context, strategy, success, severity) tuples
        """
        for context, strategy, success, severity in results:
            self.update(context, strategy, success, severity)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_context_stats(self, context: ContextKey) -> dict[str, Any]:
        """Get statistics for a specific context."""
        if context not in self.context_arms:
            return {
                'context': {
                    'target_type': context.target_type,
                    'vuln_class': context.vuln_class,
                    'defense_level': context.defense_level,
                },
                'total_pulls': 0,
                'total_rewards': 0.0,
                'best_strategy': None,
                'arms': {},
            }

        arms = self.context_arms[context]
        total_pulls = sum(arm.total_pulls for arm in arms.values())
        total_rewards = sum(arm.total_rewards for arm in arms.values())

        best_arm = max(arms.values(), key=lambda a: a.expected_reward)

        return {
            'context': {
                'target_type': context.target_type,
                'vuln_class': context.vuln_class,
                'defense_level': context.defense_level,
            },
            'total_pulls': total_pulls,
            'total_rewards': total_rewards,
            'overall_success_rate': total_rewards / max(1, total_pulls),
            'best_strategy': best_arm.strategy.value,
            'best_expected_reward': best_arm.expected_reward,
            'arms': {
                strategy.value: arm.to_dict()
                for strategy, arm in arms.items()
            }
        }

    def get_all_contexts(self) -> list[ContextKey]:
        """Get all known contexts."""
        return list(self._known_contexts)

    def get_expected_rewards(
        self,
        context: ContextKey | None = None
    ) -> dict[AttackStrategy, float]:
        """
        Get expected rewards for all strategies.

        Args:
            context: If provided, returns context-specific rewards.
                    If None, returns global rewards.
        """
        if context is None:
            return {
                strategy: arm.expected_reward
                for strategy, arm in self.global_arms.items()
            }

        arms = self._get_or_create_context_arms(context)
        return {
            strategy: arm.expected_reward
            for strategy, arm in arms.items()
        }

    def get_best_strategy(
        self,
        context: ContextKey | None = None
    ) -> AttackStrategy:
        """Get strategy with highest expected reward."""
        rewards = self.get_expected_rewards(context)
        return max(rewards, key=rewards.get)

    def get_exploration_candidates(
        self,
        context: ContextKey | None = None,
        uncertainty_threshold: float = 0.1
    ) -> list[AttackStrategy]:
        """
        Get strategies with high uncertainty (worth exploring).

        Args:
            context: If provided, uses context-specific uncertainty
            uncertainty_threshold: Minimum uncertainty to include

        Returns:
            List of high-uncertainty strategies
        """
        if context is None:
            arms = self.global_arms
        else:
            arms = self._get_or_create_context_arms(context)

        return [
            strategy
            for strategy, arm in arms.items()
            if arm.uncertainty > uncertainty_threshold
        ]

    def get_global_stats(self) -> dict[str, Any]:
        """Get global (aggregated) statistics."""
        total_pulls = sum(arm.total_pulls for arm in self.global_arms.values())
        total_rewards = sum(arm.total_rewards for arm in self.global_arms.values())

        best_arm = max(self.global_arms.values(), key=lambda a: a.expected_reward)

        return {
            'total_contexts': len(self._known_contexts),
            'total_pulls': total_pulls,
            'total_rewards': total_rewards,
            'overall_success_rate': total_rewards / max(1, total_pulls),
            'best_strategy': best_arm.strategy.value,
            'best_expected_reward': best_arm.expected_reward,
            'arms': {
                strategy.value: arm.to_dict()
                for strategy, arm in self.global_arms.items()
            }
        }

    def get_context_comparison(self) -> list[dict[str, Any]]:
        """
        Compare best strategies across all contexts.

        Returns:
            List of context summaries sorted by performance
        """
        summaries = []

        for context in self._known_contexts:
            stats = self.get_context_stats(context)
            summaries.append({
                'context': f"{context.target_type}/{context.vuln_class}/{context.defense_level}",
                'target_type': context.target_type,
                'vuln_class': context.vuln_class,
                'defense_level': context.defense_level,
                'total_pulls': stats['total_pulls'],
                'success_rate': stats['overall_success_rate'],
                'best_strategy': stats['best_strategy'],
                'best_reward': stats['best_expected_reward'],
            })

        return sorted(summaries, key=lambda x: x['success_rate'], reverse=True)

    # =========================================================================
    # Context Transfer
    # =========================================================================

    def transfer_context(
        self,
        from_context: ContextKey,
        to_context: ContextKey,
        transfer_rate: float = 0.5
    ):
        """
        Transfer learned priors from one context to another.

        Useful for seeding new contexts with knowledge from similar contexts.

        Args:
            from_context: Source context
            to_context: Destination context
            transfer_rate: How much to transfer (0.0 = none, 1.0 = full copy)
        """
        if from_context not in self.context_arms:
            return

        source_arms = self.context_arms[from_context]
        dest_arms = self._get_or_create_context_arms(to_context)

        for strategy in AttackStrategy:
            src = source_arms[strategy]
            dst = dest_arms[strategy]

            # Blend priors
            dst.alpha = dst.alpha + transfer_rate * (src.alpha - self.initial_alpha)
            dst.beta = dst.beta + transfer_rate * (src.beta - self.initial_beta)

    def find_similar_context(
        self,
        context: ContextKey
    ) -> ContextKey | None:
        """
        Find the most similar known context.

        Similarity based on shared attributes.

        Args:
            context: Context to find similar to

        Returns:
            Most similar known context or None
        """
        if not self._known_contexts:
            return None

        best_match = None
        best_score = -1

        for known in self._known_contexts:
            score = 0
            if known.target_type == context.target_type:
                score += 2
            if known.vuln_class == context.vuln_class:
                score += 2
            if known.defense_level == context.defense_level:
                score += 1

            if score > best_score:
                best_score = score
                best_match = known

        return best_match if best_score > 0 else None

    # =========================================================================
    # Decay
    # =========================================================================

    def _apply_decay(self, context: ContextKey | None = None):
        """Apply decay to priors (for non-stationary environments)."""
        # Decay global arms
        for arm in self.global_arms.values():
            arm.alpha = 1.0 + (arm.alpha - 1.0) * (1.0 - self.decay_rate)
            arm.beta = 1.0 + (arm.beta - 1.0) * (1.0 - self.decay_rate)

        # Decay context-specific arms
        if context and context in self.context_arms:
            for arm in self.context_arms[context].values():
                arm.alpha = 1.0 + (arm.alpha - 1.0) * (1.0 - self.decay_rate)
                arm.beta = 1.0 + (arm.beta - 1.0) * (1.0 - self.decay_rate)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Path):
        """Save bandit state to file."""
        state = {
            'initial_alpha': self.initial_alpha,
            'initial_beta': self.initial_beta,
            'decay_rate': self.decay_rate,
            'global_weight': self.global_weight,
            'min_context_samples': self.min_context_samples,
            'global_arms': {
                strategy.value: arm.to_dict()
                for strategy, arm in self.global_arms.items()
            },
            'contexts': [],
            'saved_at': datetime.now().isoformat(),
        }

        # Save context arms
        for context, arms in self.context_arms.items():
            context_data = {
                'context': {
                    'target_type': context.target_type,
                    'vuln_class': context.vuln_class,
                    'defense_level': context.defense_level,
                },
                'arms': {
                    strategy.value: arm.to_dict()
                    for strategy, arm in arms.items()
                }
            }
            state['contexts'].append(context_data)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: Path):
        """Load bandit state from file."""
        path = Path(path)
        if not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        self.initial_alpha = state.get('initial_alpha', 1.0)
        self.initial_beta = state.get('initial_beta', 1.0)
        self.decay_rate = state.get('decay_rate', 0.0)
        self.global_weight = state.get('global_weight', 0.3)
        self.min_context_samples = state.get('min_context_samples', 5)

        # Load global arms
        for strategy_value, arm_data in state.get('global_arms', {}).items():
            try:
                strategy = AttackStrategy(strategy_value)
                self.global_arms[strategy] = ContextualArm.from_dict(arm_data)
            except ValueError:
                pass

        # Load context arms
        for context_data in state.get('contexts', []):
            ctx_info = context_data.get('context', {})
            context = ContextKey(
                target_type=ctx_info.get('target_type', 'unknown'),
                vuln_class=ctx_info.get('vuln_class', 'unknown'),
                defense_level=ctx_info.get('defense_level', 'unknown'),
            )

            self.context_arms[context] = {}
            self._known_contexts.add(context)

            for strategy_value, arm_data in context_data.get('arms', {}).items():
                try:
                    strategy = AttackStrategy(strategy_value)
                    self.context_arms[context][strategy] = ContextualArm.from_dict(arm_data)
                except ValueError:
                    pass

    def reset(self, context: ContextKey | None = None):
        """
        Reset bandit to initial state.

        Args:
            context: If provided, only reset this context. Otherwise reset all.
        """
        if context:
            if context in self.context_arms:
                for arm in self.context_arms[context].values():
                    arm.alpha = self.initial_alpha
                    arm.beta = self.initial_beta
                    arm.total_pulls = 0
                    arm.total_rewards = 0.0
                    arm.last_updated = None
        else:
            # Reset global
            for arm in self.global_arms.values():
                arm.alpha = self.initial_alpha
                arm.beta = self.initial_beta
                arm.total_pulls = 0
                arm.total_rewards = 0.0
                arm.last_updated = None

            # Reset all contexts
            for arms in self.context_arms.values():
                for arm in arms.values():
                    arm.alpha = self.initial_alpha
                    arm.beta = self.initial_beta
                    arm.total_pulls = 0
                    arm.total_rewards = 0.0
                    arm.last_updated = None

            self.selection_history.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_contextual_bandit(
    initial_alpha: float = 1.0,
    initial_beta: float = 1.0,
    decay_rate: float = 0.0,
    global_weight: float = 0.3,
    min_context_samples: int = 5,
    state_path: Path | None = None
) -> ContextualRedTeamBandit:
    """
    Create a ContextualRedTeamBandit, optionally loading saved state.

    Args:
        initial_alpha: Initial alpha for new bandits
        initial_beta: Initial beta for new bandits
        decay_rate: Decay rate for non-stationary environments
        global_weight: Weight for global prior when context unseen
        min_context_samples: Minimum samples before trusting context data
        state_path: Path to load saved state from

    Returns:
        Configured ContextualRedTeamBandit
    """
    bandit = ContextualRedTeamBandit(
        initial_alpha=initial_alpha,
        initial_beta=initial_beta,
        decay_rate=decay_rate,
        global_weight=global_weight,
        min_context_samples=min_context_samples,
    )

    if state_path and Path(state_path).exists():
        bandit.load(state_path)

    return bandit


def create_context_key(
    target_type: str = "unknown",
    vuln_class: str = "unknown",
    defense_level: str = "unknown"
) -> ContextKey:
    """
    Create a ContextKey for contextual bandit.

    Args:
        target_type: Type of target (llm, api, web_app, etc.)
        vuln_class: Vulnerability classification
        defense_level: Level of defensive measures (none, basic, advanced)

    Returns:
        ContextKey instance
    """
    return ContextKey(
        target_type=target_type,
        vuln_class=vuln_class,
        defense_level=defense_level,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data Classes
    'ContextualArm',
    'ContextualSelectionResult',

    # Main Class
    'ContextualRedTeamBandit',

    # Convenience Functions
    'create_contextual_bandit',
    'create_context_key',
]
