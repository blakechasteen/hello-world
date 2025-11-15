#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Mode Bandit
======================
Thompson Sampling for adaptive reasoning mode selection.

Author: Claude Code
Date: 2025-11-15
Phase: 3 (DEEP Mode)
"""

import logging
import random
from typing import Dict, Optional
from dataclasses import dataclass, field
import json

from HoloLoom.reasoning.types import ReasoningMode


logger = logging.getLogger(__name__)


# ============================================================================
# Thompson Sampling Prior
# ============================================================================

@dataclass
class ThompsonPrior:
    """
    Beta distribution prior for Thompson Sampling.

    Attributes:
        alpha: Success count (alpha parameter)
        beta: Failure count (beta parameter)
    """
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self) -> float:
        """
        Sample from beta distribution.

        Returns:
            Sampled value [0.0, 1.0]
        """
        # Use Python's random.betavariate for sampling
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool, confidence: float = 1.0):
        """
        Update prior based on outcome.

        Args:
            success: Whether outcome was successful
            confidence: Confidence in outcome [0.0, 1.0]
        """
        if success:
            self.alpha += confidence
        else:
            self.beta += (1.0 - confidence)

    def expected_reward(self) -> float:
        """
        Calculate expected reward (mean of beta distribution).

        Returns:
            Expected reward [0.0, 1.0]
        """
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "alpha": self.alpha,
            "beta": self.beta
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ThompsonPrior':
        """Deserialize from dict."""
        return cls(alpha=data["alpha"], beta=data["beta"])


# ============================================================================
# Reasoning Mode Statistics
# ============================================================================

@dataclass
class ModeStatistics:
    """
    Statistics for a reasoning mode.

    Attributes:
        mode: Reasoning mode
        prior: Thompson Sampling prior
        total_uses: Total times mode was used
        successes: Number of successful outcomes
        total_confidence: Sum of all confidences
        avg_duration_ms: Average duration in milliseconds
    """
    mode: ReasoningMode
    prior: ThompsonPrior = field(default_factory=ThompsonPrior)
    total_uses: int = 0
    successes: int = 0
    total_confidence: float = 0.0
    avg_duration_ms: float = 0.0

    def update(self, success: bool, confidence: float, duration_ms: float):
        """
        Update statistics based on outcome.

        Args:
            success: Whether outcome was successful
            confidence: Outcome confidence
            duration_ms: Duration in milliseconds
        """
        self.total_uses += 1
        if success:
            self.successes += 1

        self.total_confidence += confidence

        # Update average duration (running average)
        if self.avg_duration_ms == 0:
            self.avg_duration_ms = duration_ms
        else:
            # Exponential moving average
            alpha = 0.3  # Weight for new value
            self.avg_duration_ms = alpha * duration_ms + (1 - alpha) * self.avg_duration_ms

        # Update prior
        self.prior.update(success, confidence)

    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / self.total_uses if self.total_uses > 0 else 0.0

    def avg_confidence(self) -> float:
        """Calculate average confidence."""
        return self.total_confidence / self.total_uses if self.total_uses > 0 else 0.0


# ============================================================================
# Reasoning Mode Bandit
# ============================================================================

class ReasoningModeBandit:
    """
    Thompson Sampling for reasoning mode selection.

    Learns which reasoning modes work best for different query types.
    """

    def __init__(
        self,
        initial_priors: Optional[Dict[ReasoningMode, ThompsonPrior]] = None,
        exploration_bonus: float = 0.1
    ):
        """
        Initialize reasoning mode bandit.

        Args:
            initial_priors: Optional initial priors for each mode
            exploration_bonus: Bonus for exploration (0.0 = pure exploitation)
        """
        self.exploration_bonus = exploration_bonus
        self.logger = logging.getLogger(f"{__name__}.ReasoningModeBandit")

        # Initialize statistics for each mode
        self.stats: Dict[ReasoningMode, ModeStatistics] = {}

        for mode in ReasoningMode:
            if initial_priors and mode in initial_priors:
                prior = initial_priors[mode]
            else:
                # Default priors (slightly favor FAST, then STANDARD)
                if mode == ReasoningMode.FAST:
                    prior = ThompsonPrior(alpha=10, beta=2)  # Initially optimistic
                elif mode == ReasoningMode.STANDARD:
                    prior = ThompsonPrior(alpha=15, beta=5)  # Balanced
                else:  # DEEP
                    prior = ThompsonPrior(alpha=5, beta=5)   # Neutral (expensive)

            self.stats[mode] = ModeStatistics(mode=mode, prior=prior)

    def select_mode(
        self,
        query_complexity: float,
        context_quality: float = 0.5
    ) -> ReasoningMode:
        """
        Select reasoning mode using Thompson Sampling + query complexity.

        Args:
            query_complexity: Estimated query complexity [0.0, 1.0]
            context_quality: Quality of retrieved context [0.0, 1.0]

        Returns:
            Selected ReasoningMode
        """
        samples = {}

        for mode in ReasoningMode:
            # Sample from Thompson prior
            sample = self.stats[mode].prior.sample()

            # Apply complexity weighting
            if mode == ReasoningMode.DEEP and query_complexity > 0.7:
                # Boost DEEP for complex queries
                sample *= (1.0 + 0.5 * query_complexity)
            elif mode == ReasoningMode.FAST and query_complexity < 0.3:
                # Boost FAST for simple queries
                sample *= (1.0 + 0.3 * (1.0 - query_complexity))

            # Apply context quality weighting
            if mode == ReasoningMode.FAST and context_quality > 0.8:
                # FAST works well with high-quality context
                sample *= (1.0 + 0.2 * context_quality)
            elif mode == ReasoningMode.DEEP and context_quality < 0.4:
                # DEEP needed when context is poor
                sample *= (1.0 + 0.3 * (1.0 - context_quality))

            # Add exploration bonus
            if self.exploration_bonus > 0:
                # Encourage exploration of less-used modes
                uses = self.stats[mode].total_uses
                total_uses = sum(s.total_uses for s in self.stats.values())

                if total_uses > 0:
                    exploration_factor = 1.0 + self.exploration_bonus * (1.0 - uses / total_uses)
                    sample *= exploration_factor

            samples[mode] = sample

        # Select mode with highest weighted sample
        selected_mode = max(samples.items(), key=lambda x: x[1])[0]

        self.logger.debug(
            f"Mode selection: complexity={query_complexity:.2f}, "
            f"quality={context_quality:.2f}, selected={selected_mode.value}, "
            f"samples={{{', '.join(f'{k.value}:{v:.3f}' for k, v in samples.items())}}}"
        )

        return selected_mode

    def update(
        self,
        mode: ReasoningMode,
        success: bool,
        confidence: float,
        duration_ms: float
    ):
        """
        Update bandit statistics based on outcome.

        Args:
            mode: Mode that was used
            success: Whether reasoning was successful (typically confidence > threshold)
            confidence: Outcome confidence [0.0, 1.0]
            duration_ms: Duration in milliseconds
        """
        if mode not in self.stats:
            self.logger.warning(f"Unknown mode: {mode}")
            return

        self.stats[mode].update(success, confidence, duration_ms)

        self.logger.info(
            f"Updated {mode.value}: "
            f"success_rate={self.stats[mode].success_rate():.2f}, "
            f"avg_confidence={self.stats[mode].avg_confidence():.2f}, "
            f"avg_duration={self.stats[mode].avg_duration_ms:.1f}ms, "
            f"uses={self.stats[mode].total_uses}"
        )

    def get_stats(self) -> Dict[str, Dict]:
        """
        Get current statistics for all modes.

        Returns:
            Dict mapping mode names to statistics
        """
        return {
            mode.value: {
                "total_uses": stats.total_uses,
                "success_rate": stats.success_rate(),
                "avg_confidence": stats.avg_confidence(),
                "avg_duration_ms": stats.avg_duration_ms,
                "expected_reward": stats.prior.expected_reward(),
                "alpha": stats.prior.alpha,
                "beta": stats.prior.beta
            }
            for mode, stats in self.stats.items()
        }

    def save(self, filepath: str):
        """
        Save bandit state to file.

        Args:
            filepath: Path to save file
        """
        data = {
            "exploration_bonus": self.exploration_bonus,
            "stats": {
                mode.value: {
                    "prior": stats.prior.to_dict(),
                    "total_uses": stats.total_uses,
                    "successes": stats.successes,
                    "total_confidence": stats.total_confidence,
                    "avg_duration_ms": stats.avg_duration_ms
                }
                for mode, stats in self.stats.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Bandit state saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ReasoningModeBandit':
        """
        Load bandit state from file.

        Args:
            filepath: Path to load file

        Returns:
            Loaded ReasoningModeBandit
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        exploration_bonus = data.get("exploration_bonus", 0.1)

        # Reconstruct priors
        initial_priors = {}
        for mode_str, stats_data in data["stats"].items():
            mode = ReasoningMode(mode_str)
            initial_priors[mode] = ThompsonPrior.from_dict(stats_data["prior"])

        # Create bandit
        bandit = cls(initial_priors=initial_priors, exploration_bonus=exploration_bonus)

        # Restore statistics
        for mode_str, stats_data in data["stats"].items():
            mode = ReasoningMode(mode_str)
            bandit.stats[mode].total_uses = stats_data["total_uses"]
            bandit.stats[mode].successes = stats_data["successes"]
            bandit.stats[mode].total_confidence = stats_data["total_confidence"]
            bandit.stats[mode].avg_duration_ms = stats_data["avg_duration_ms"]

        logger.info(f"Bandit state loaded from {filepath}")

        return bandit

    def recommend_mode(
        self,
        query_complexity: float,
        context_quality: float = 0.5,
        time_budget_ms: Optional[float] = None
    ) -> ReasoningMode:
        """
        Recommend mode based on constraints (without Thompson sampling).

        Args:
            query_complexity: Query complexity [0.0, 1.0]
            context_quality: Context quality [0.0, 1.0]
            time_budget_ms: Optional time budget in milliseconds

        Returns:
            Recommended ReasoningMode
        """
        # If time budget is very tight, prefer FAST
        if time_budget_ms and time_budget_ms < 100:
            return ReasoningMode.FAST

        # If query is very complex, use DEEP
        if query_complexity > 0.8:
            return ReasoningMode.DEEP

        # If context is excellent and query is simple, use FAST
        if context_quality > 0.85 and query_complexity < 0.3:
            return ReasoningMode.FAST

        # Default to STANDARD for most cases
        return ReasoningMode.STANDARD
