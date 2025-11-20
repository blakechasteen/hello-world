#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested Learning Pattern
========================

Multi-timescale learning pattern inspired by Google's Nested Learning
and Continuum Memory research.

Treats different components as "nested learners" with different update
frequencies (horizons):

- bandit_module: FAST horizon (per-episode reward updates)
- graph_module: MEDIUM horizon (periodic structural/heuristic updates)
- policy_module: SLOW horizon (infrequent meta-strategy updates)

This enables continual learning without modifying core logic.

Created: 2025-01-20
Based on: Google Research Nested Learning / Continuum Memory
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Protocol
import logging

from hololoom.core.types import WeaveResult
from hololoom.patterns.base import LearningPattern


logger = logging.getLogger(__name__)


# Module Protocols (minimal interfaces for decoupling)


class SupportsBanditModule(Protocol):
    """
    Minimal interface for a module that can learn from scalar rewards.

    Real implementations might be:
    - Wrapper around PolicyBandit
    - Thompson Sampling updater
    - UCB bandit
    - Contextual bandit
    """

    def learn_from_reward(self, reward: float) -> None:
        """Update bandit based on scalar reward."""
        ...


class SupportsGraphModule(Protocol):
    """
    Minimal interface for a module that can refine graph-related heuristics.

    Real implementations might adjust:
    - Node expansion priorities
    - Edge scoring functions
    - Graph traversal strategies
    - Yarn retrieval heuristics
    """

    def refine_heuristics(self) -> None:
        """Refine graph-related heuristics based on experience."""
        ...


class SupportsPolicyModule(Protocol):
    """
    Minimal interface for a module that can update high-level meta-strategy.

    Real implementations might adjust:
    - Available policy set
    - Policy selection priors
    - Meta-learning parameters
    - High-level strategy preferences
    """

    def update_meta_strategy(self) -> None:
        """Update meta-strategy based on long-term patterns."""
        ...


# Main Pattern Implementation


@dataclass
class NestedLearningPattern(LearningPattern):
    """
    Nested Learning / Continuum Memory-style pattern.

    Treats different components as "nested learners" with different
    update frequencies (horizons):

    - bandit_module: Fast / per-episode reward updates
    - graph_module: Medium horizon, periodic structural/heuristic updates
    - policy_module: Slow horizon, infrequent meta-strategy updates

    Design:
    - Observes WeaveResult after each episode
    - Extracts reward from metadata
    - Updates modules at different frequencies
    - No coupling to core logic

    Example:
        from hololoom.patterns.nested_learning import (
            NestedLearningPattern,
            NoOpBanditModule,
            NoOpGraphModule,
            NoOpPolicyModule
        )

        pattern = NestedLearningPattern(
            bandit_module=NoOpBanditModule(),
            graph_module=NoOpGraphModule(),
            policy_module=NoOpPolicyModule(),
            freq={'bandit': 1, 'graph': 100, 'policy': 1000}
        )

        # Use with WeaveRunner
        runner = WeaveRunner(shuttle=shuttle, patterns=[pattern])
        result = runner.run_query("my query", warp_results)
    """

    bandit_module: SupportsBanditModule
    """Fast-horizon learner (per-episode reward updates)"""

    graph_module: SupportsGraphModule
    """Medium-horizon learner (periodic structural updates)"""

    policy_module: SupportsPolicyModule
    """Slow-horizon learner (infrequent meta-strategy updates)"""

    # Internal counters for when to trigger learning
    counters: Dict[str, int] = field(default_factory=lambda: {
        "bandit": 0,
        "graph": 0,
        "policy": 0,
    })

    # Update frequencies in units of episodes
    freq: Dict[str, int] = field(default_factory=lambda: {
        "bandit": 1,     # update every episode (fast horizon)
        "graph": 100,    # update every 100 episodes (medium horizon)
        "policy": 1000,  # update every 1000 episodes (slow horizon)
    })

    # Statistics (for monitoring)
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "total_episodes": 0,
        "bandit_updates": 0,
        "graph_updates": 0,
        "policy_updates": 0,
        "last_reward": 0.0,
    })

    def on_episode_end(self, result: WeaveResult) -> None:
        """
        Called after each weave completes.

        Extracts reward and updates modules at their respective frequencies.

        Args:
            result: WeaveResult from completed weave
        """
        # Extract reward from metadata
        reward = result.get_reward(default=0.0)

        # Update statistics
        self.stats["total_episodes"] += 1
        self.stats["last_reward"] = reward

        # Log if reward is unusually high/low
        if reward > 0.9:
            logger.info(f"High reward: {reward:.3f} for query: {result.query[:50]}...")
        elif reward < 0.1:
            logger.debug(f"Low reward: {reward:.3f} for query: {result.query[:50]}...")

        # Tick each module at its frequency
        self._tick("bandit", reward)
        self._tick("graph", reward)
        self._tick("policy", reward)

    def _extract_reward(self, result: WeaveResult) -> float:
        """
        Extract scalar reward from weave result.

        Tries multiple fallbacks:
        1. result.metadata['reward']
        2. result.metadata['confidence']
        3. 0.5 (neutral default)

        Args:
            result: WeaveResult to extract from

        Returns:
            Scalar reward (0.0-1.0)
        """
        # Try direct reward first
        if 'reward' in result.metadata:
            return result.get_reward()

        # Fall back to confidence
        if 'confidence' in result.metadata:
            return result.get_confidence()

        # Neutral default
        return 0.5

    def _tick(self, name: str, reward: float) -> None:
        """
        Increment counter and trigger learning if frequency reached.

        Args:
            name: Module name ('bandit', 'graph', 'policy')
            reward: Scalar reward for this episode
        """
        self.counters[name] += 1

        if self.counters[name] >= self.freq[name]:
            self._learn(name, reward)
            self.counters[name] = 0  # Reset counter

    def _learn(self, name: str, reward: float) -> None:
        """
        Trigger learning for specified module.

        Args:
            name: Module name ('bandit', 'graph', 'policy')
            reward: Scalar reward for this episode
        """
        try:
            if name == "bandit":
                self.bandit_module.learn_from_reward(reward)
                self.stats["bandit_updates"] += 1
                logger.debug(f"Bandit update #{self.stats['bandit_updates']}: reward={reward:.3f}")

            elif name == "graph":
                self.graph_module.refine_heuristics()
                self.stats["graph_updates"] += 1
                logger.info(f"Graph heuristics refined (update #{self.stats['graph_updates']})")

            elif name == "policy":
                self.policy_module.update_meta_strategy()
                self.stats["policy_updates"] += 1
                logger.info(f"Meta-strategy updated (update #{self.stats['policy_updates']})")

        except Exception as e:
            # Patterns should not crash the runner
            logger.error(f"Error in {name} module learning: {e}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current pattern statistics.

        Returns:
            Dictionary with:
            - total_episodes: Total episodes observed
            - bandit_updates: Number of bandit updates
            - graph_updates: Number of graph updates
            - policy_updates: Number of policy updates
            - last_reward: Most recent reward value
        """
        return dict(self.stats)

    def reset_stats(self) -> None:
        """Reset all statistics and counters."""
        self.counters = {"bandit": 0, "graph": 0, "policy": 0}
        self.stats = {
            "total_episodes": 0,
            "bandit_updates": 0,
            "graph_updates": 0,
            "policy_updates": 0,
            "last_reward": 0.0,
        }


__all__ = [
    'NestedLearningPattern',
    'SupportsBanditModule',
    'SupportsGraphModule',
    'SupportsPolicyModule',
]
