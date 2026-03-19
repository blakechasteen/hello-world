"""
MRF Thompson Sampling Learning Module
======================================
Continuous learning system that adapts MRF strategy selection based on
production outcomes using Thompson Sampling.

**Phase 2 - November 2025**: Production learning for automatic strategy
optimization across all integrated systems.

Features:
- Thompson Sampling for strategy selection (α/β Bayesian updates)
- Per-system and per-query-type learning
- Automatic strategy weight adjustment
- Quality-based reward signals
- Learning state persistence and recovery

Usage:
    from hololoom.prompting.analytics import ThompsonLearner

    # Create learner
    learner = ThompsonLearner()

    # Select best strategy (exploration/exploitation)
    strategy = learner.select_strategy(
        query_type="factual",
        system="agentic"
    )

    # Update from outcome
    learner.update_from_outcome(
        query_type="factual",
        system="agentic",
        strategy="verify",
        quality_improvement=0.25
    )

    # Get learning statistics
    stats = learner.get_learning_stats()
"""

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    """Thompson Sampling statistics for a strategy."""
    strategy_name: str
    alpha: float = 1.0  # Success count (Bayesian prior)
    beta: float = 1.0   # Failure count (Bayesian prior)
    total_uses: int = 0
    total_successes: int = 0
    avg_quality_improvement: float = 0.0
    last_used: float = 0.0

    @property
    def expected_reward(self) -> float:
        """Expected reward: E[X] = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def success_rate(self) -> float:
        """Observed success rate."""
        return self.total_successes / max(self.total_uses, 1)

    def sample_thompson(self) -> float:
        """Sample from Beta(α, β) distribution."""
        # Simple approximation: expected + Gaussian noise
        expected = self.expected_reward
        # Uncertainty decreases with more observations
        uncertainty = 1.0 / (self.alpha + self.beta)
        noise = random.gauss(0, uncertainty * 0.5)
        return max(0.0, min(1.0, expected + noise))


@dataclass
class QueryTypeProfile:
    """Learning profile for a query type."""
    query_type: str
    system: str
    strategies: dict[str, StrategyStats] = field(default_factory=dict)
    query_count: int = 0
    last_updated: float = 0.0

    def get_or_create_strategy(self, strategy: str) -> StrategyStats:
        """Get or create strategy stats."""
        if strategy not in self.strategies:
            self.strategies[strategy] = StrategyStats(strategy_name=strategy)
        return self.strategies[strategy]


class ThompsonLearner:
    """
    Thompson Sampling learner for MRF strategy selection.

    Learns which refinement strategies work best for different query types
    and systems through Bayesian online learning.
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        quality_threshold: float = 0.15,
        exploration_bonus: float = 0.1
    ):
        """
        Initialize Thompson learner.

        Args:
            persist_path: Optional path to persist learning state
            quality_threshold: Quality improvement threshold for success (default: 15%)
            exploration_bonus: Bonus for rarely-used strategies (default: 0.1)
        """
        self.persist_path = persist_path or Path("./mrf_learning")
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.quality_threshold = quality_threshold
        self.exploration_bonus = exploration_bonus

        # Learning profiles: (query_type, system) -> QueryTypeProfile
        self.profiles: dict[tuple[str, str], QueryTypeProfile] = {}

        # Selection history
        self.selection_history: list[dict] = []

        # Load existing state
        self._load_state()

    def select_strategy(
        self,
        query_type: str,
        system: str,
        available_strategies: list[str] | None = None
    ) -> str:
        """
        Select best strategy using Thompson Sampling.

        Args:
            query_type: Type of query ("factual", "analytical", "creative", etc.)
            system: System name ("agentic", "rag", "alignment", etc.)
            available_strategies: Available strategies (defaults to all known)

        Returns:
            Selected strategy name
        """
        key = (query_type, system)

        # Get or create profile
        if key not in self.profiles:
            self.profiles[key] = QueryTypeProfile(
                query_type=query_type,
                system=system
            )

        profile = self.profiles[key]
        profile.query_count += 1
        profile.last_updated = time.time()

        # Default strategies if not specified
        if available_strategies is None:
            available_strategies = ["verify", "refine", "critique", "elegance", "hofstadter"]

        # Ensure all strategies have stats
        for strategy in available_strategies:
            profile.get_or_create_strategy(strategy)

        # Thompson Sampling: sample from Beta(α, β) for each strategy
        best_strategy = None
        best_sample = -1.0

        for strategy in available_strategies:
            stats = profile.strategies[strategy]

            # Sample from Thompson distribution
            sample = stats.sample_thompson()

            # Add exploration bonus for rarely-used strategies
            if stats.total_uses < 5:
                sample += self.exploration_bonus

            if sample > best_sample:
                best_sample = sample
                best_strategy = strategy

        # Log selection
        self.selection_history.append({
            "timestamp": time.time(),
            "query_type": query_type,
            "system": system,
            "selected_strategy": best_strategy,
            "sample_value": best_sample
        })

        logger.debug(
            f"[Thompson] {system}/{query_type}: selected '{best_strategy}' "
            f"(sample={best_sample:.3f})"
        )

        return best_strategy or available_strategies[0]

    def update_from_outcome(
        self,
        query_type: str,
        system: str,
        strategy: str,
        quality_improvement: float,
        execution_time_ms: float | None = None
    ):
        """
        Update Thompson Sampling statistics from outcome.

        Args:
            query_type: Type of query
            system: System name
            strategy: Strategy that was used
            quality_improvement: Quality improvement achieved (0.0-1.0)
            execution_time_ms: Optional execution time
        """
        key = (query_type, system)

        # Get profile
        if key not in self.profiles:
            self.profiles[key] = QueryTypeProfile(
                query_type=query_type,
                system=system
            )

        profile = self.profiles[key]
        stats = profile.get_or_create_strategy(strategy)

        # Update usage stats
        stats.total_uses += 1
        stats.last_used = time.time()

        # Determine if success (quality improvement above threshold)
        is_success = quality_improvement >= self.quality_threshold

        # Update Thompson Sampling priors (α, β)
        if is_success:
            # Success: increase α
            stats.alpha += quality_improvement
            stats.total_successes += 1
            logger.debug(
                f"[Thompson] SUCCESS: {system}/{query_type}/{strategy} | "
                f"improvement={quality_improvement:.2f} | α={stats.alpha:.2f}"
            )
        else:
            # Failure: increase β
            stats.beta += (self.quality_threshold - quality_improvement)
            logger.debug(
                f"[Thompson] FAILURE: {system}/{query_type}/{strategy} | "
                f"improvement={quality_improvement:.2f} | β={stats.beta:.2f}"
            )

        # Update average quality improvement (running average)
        n = stats.total_uses
        stats.avg_quality_improvement = (
            (stats.avg_quality_improvement * (n - 1) + quality_improvement) / n
        )

        # Persist state
        self._save_state()

    def get_learning_stats(
        self,
        system: str | None = None,
        query_type: str | None = None
    ) -> dict:
        """
        Get learning statistics.

        Args:
            system: Optional system filter
            query_type: Optional query type filter

        Returns:
            Statistics dictionary
        """
        # Filter profiles
        profiles = [
            p for (qt, sys), p in self.profiles.items()
            if (system is None or sys == system) and
               (query_type is None or qt == query_type)
        ]

        if not profiles:
            return {"error": "No learning data available"}

        # Aggregate statistics
        total_queries = sum(p.query_count for p in profiles)
        total_strategies = sum(len(p.strategies) for p in profiles)

        # Best strategies per (system, query_type)
        best_strategies = {}
        for (qt, sys), profile in self.profiles.items():
            if (system is None or sys == system) and \
               (query_type is None or qt == query_type):
                # Find strategy with highest expected reward
                best = max(
                    profile.strategies.values(),
                    key=lambda s: s.expected_reward,
                    default=None
                )
                if best:
                    best_strategies[f"{sys}/{qt}"] = {
                        "strategy": best.strategy_name,
                        "expected_reward": best.expected_reward,
                        "success_rate": best.success_rate,
                        "total_uses": best.total_uses
                    }

        # Strategy effectiveness ranking (global)
        strategy_stats = defaultdict(lambda: {"uses": 0, "successes": 0, "avg_quality": 0.0})
        for profile in profiles:
            for strategy, stats in profile.strategies.items():
                strategy_stats[strategy]["uses"] += stats.total_uses
                strategy_stats[strategy]["successes"] += stats.total_successes
                strategy_stats[strategy]["avg_quality"] += stats.avg_quality_improvement * stats.total_uses

        # Compute averages
        for strategy, data in strategy_stats.items():
            if data["uses"] > 0:
                data["success_rate"] = data["successes"] / data["uses"]
                data["avg_quality"] = data["avg_quality"] / data["uses"]

        # Rank by success rate
        strategy_ranking = sorted(
            strategy_stats.items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True
        )

        return {
            "total_queries": total_queries,
            "profiles_count": len(profiles),
            "total_strategies_tracked": total_strategies,
            "best_strategies_per_context": best_strategies,
            "strategy_effectiveness_ranking": [
                {
                    "strategy": name,
                    "success_rate": data["success_rate"],
                    "avg_quality_improvement": data["avg_quality"],
                    "total_uses": data["uses"]
                }
                for name, data in strategy_ranking
            ],
            "selection_history_count": len(self.selection_history)
        }

    def get_strategy_recommendation(
        self,
        query_type: str,
        system: str
    ) -> dict[str, Any]:
        """
        Get strategy recommendation with confidence.

        Args:
            query_type: Type of query
            system: System name

        Returns:
            Recommendation dict with strategy, confidence, and alternatives
        """
        key = (query_type, system)

        if key not in self.profiles:
            return {
                "recommended_strategy": "verify",  # Default
                "confidence": 0.0,
                "reason": "No learning data available",
                "alternatives": []
            }

        profile = self.profiles[key]

        # Rank strategies by expected reward
        ranked = sorted(
            profile.strategies.values(),
            key=lambda s: s.expected_reward,
            reverse=True
        )

        if not ranked:
            return {
                "recommended_strategy": "verify",
                "confidence": 0.0,
                "reason": "No strategies available",
                "alternatives": []
            }

        best = ranked[0]

        # Confidence based on number of observations and success rate
        confidence = min(1.0, best.total_uses / 20.0) * best.success_rate

        return {
            "recommended_strategy": best.strategy_name,
            "confidence": confidence,
            "expected_reward": best.expected_reward,
            "success_rate": best.success_rate,
            "total_uses": best.total_uses,
            "avg_quality_improvement": best.avg_quality_improvement,
            "alternatives": [
                {
                    "strategy": s.strategy_name,
                    "expected_reward": s.expected_reward,
                    "success_rate": s.success_rate
                }
                for s in ranked[1:4]  # Top 3 alternatives
            ]
        }

    def reset_learning(self, system: str | None = None, query_type: str | None = None):
        """
        Reset learning state (use with caution!).

        Args:
            system: Optional system filter (resets only this system)
            query_type: Optional query type filter
        """
        if system is None and query_type is None:
            # Reset everything
            self.profiles.clear()
            self.selection_history.clear()
            logger.warning("[Thompson] Reset ALL learning state")
        else:
            # Reset specific profiles
            keys_to_remove = [
                key for key in self.profiles.keys()
                if (system is None or key[1] == system) and
                   (query_type is None or key[0] == query_type)
            ]
            for key in keys_to_remove:
                del self.profiles[key]
            logger.warning(
                f"[Thompson] Reset learning for system={system}, query_type={query_type}"
            )

        self._save_state()

    def _save_state(self):
        """Persist learning state to disk."""
        state = {
            "profiles": {
                f"{qt}::{sys}": {
                    "query_type": profile.query_type,
                    "system": profile.system,
                    "query_count": profile.query_count,
                    "last_updated": profile.last_updated,
                    "strategies": {
                        name: asdict(stats)
                        for name, stats in profile.strategies.items()
                    }
                }
                for (qt, sys), profile in self.profiles.items()
            },
            "selection_history": self.selection_history[-1000:]  # Keep last 1000
        }

        state_file = self.persist_path / "thompson_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        logger.debug(f"[Thompson] Saved learning state to {state_file}")

    def _load_state(self):
        """Load learning state from disk."""
        state_file = self.persist_path / "thompson_state.json"

        if not state_file.exists():
            logger.info("[Thompson] No existing learning state found")
            return

        try:
            with open(state_file, encoding="utf-8") as f:
                state = json.load(f)

            # Restore profiles
            for key_str, profile_data in state.get("profiles", {}).items():
                qt, sys = key_str.split("::")
                profile = QueryTypeProfile(
                    query_type=profile_data["query_type"],
                    system=profile_data["system"],
                    query_count=profile_data["query_count"],
                    last_updated=profile_data["last_updated"]
                )

                # Restore strategy stats
                for name, stats_data in profile_data.get("strategies", {}).items():
                    profile.strategies[name] = StrategyStats(**stats_data)

                self.profiles[(qt, sys)] = profile

            # Restore selection history
            self.selection_history = state.get("selection_history", [])

            logger.info(
                f"[Thompson] Loaded learning state: "
                f"{len(self.profiles)} profiles, "
                f"{len(self.selection_history)} selections"
            )

        except Exception as e:
            logger.error(f"[Thompson] Error loading learning state: {e}")


# Convenience functions
def create_learner(persist_path: Path | None = None) -> ThompsonLearner:
    """
    Create Thompson Sampling learner.

    Args:
        persist_path: Optional path to persist learning state

    Returns:
        ThompsonLearner instance
    """
    return ThompsonLearner(persist_path=persist_path)


__all__ = [
    "ThompsonLearner",
    "StrategyStats",
    "QueryTypeProfile",
    "create_learner"
]
