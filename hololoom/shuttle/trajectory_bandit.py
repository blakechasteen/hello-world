"""
HoloLoom Shuttle - Trajectory Bandit

Uses Thompson Sampling to select which TrajectoryStrategy to apply for a given query.
Maintains Beta distributions for each trajectory and updates them based on reward feedback.

NOTE: Renamed from "PolicyBandit" to "TrajectoryBandit" to avoid collision with
HoloLoom's existing policy system (tool selection policies in policy/unified.py).
"""

from dataclasses import dataclass
from typing import Dict, List
import random
import json
from pathlib import Path


# ============================================================================
# Bandit State
# ============================================================================

@dataclass
class TrajectoryStats:
    """
    Statistics for one trajectory arm.
    Tracks successes and failures for Thompson Sampling with Beta distribution.
    """
    successes: int = 0
    failures: int = 0

    def to_dict(self) -> Dict:
        """Serialize to dict for persistence."""
        return {
            "successes": self.successes,
            "failures": self.failures,
            "mean_reward": self.mean_reward,
            "total_pulls": self.total_pulls,
        }

    @property
    def mean_reward(self) -> float:
        """Expected value of the Beta distribution."""
        total = self.successes + self.failures
        if total == 0:
            return 0.5  # Prior mean
        return self.successes / total

    @property
    def total_pulls(self) -> int:
        """Total number of times this trajectory was selected."""
        return self.successes + self.failures


class TrajectoryBandit:
    """
    Thompson Sampling bandit over a fixed set of trajectory strategies.

    Maintains Beta(successes+1, failures+1) distributions for each trajectory.
    Samples from these distributions to balance exploration and exploitation.
    """

    def __init__(self, trajectory_names: List[str]):
        """
        Initialize bandit with given trajectory names.

        Args:
            trajectory_names: List of trajectory names to track
        """
        self.trajectory_stats: Dict[str, TrajectoryStats] = {
            name: TrajectoryStats() for name in trajectory_names
        }

    def choose_trajectory(self) -> str:
        """
        Select a trajectory using Thompson Sampling.

        Samples from Beta(successes+1, failures+1) for each trajectory
        and picks the one with the highest sample.

        Uses Gamma sampling for numerical stability:
        Beta(α, β) = Gamma(α, 1) / (Gamma(α, 1) + Gamma(β, 1))

        Returns:
            Selected trajectory name
        """
        best_name = None
        best_sample = -1.0

        for name, stats in self.trajectory_stats.items():
            alpha = stats.successes + 1
            beta = stats.failures + 1

            # Sample from Beta(alpha, beta) using Gamma sampling
            # This is more numerically stable than direct Beta sampling
            x = random.gammavariate(alpha, 1.0)
            y = random.gammavariate(beta, 1.0)
            sample = x / (x + y + 1e-9)

            if sample > best_sample:
                best_sample = sample
                best_name = name

        assert best_name is not None, "No trajectories available"
        return best_name

    def update(self, trajectory_name: str, reward: float):
        """
        Update trajectory statistics based on observed reward.

        Simple binary threshold:
        - reward > 0.5 → success
        - reward ≤ 0.5 → failure

        Can be made more nuanced later (e.g., fractional updates).

        Args:
            trajectory_name: The trajectory that was used
            reward: Observed reward in [0, 1]
        """
        if trajectory_name not in self.trajectory_stats:
            raise ValueError(f"Unknown trajectory: {trajectory_name}")

        stats = self.trajectory_stats[trajectory_name]

        if reward > 0.5:
            stats.successes += 1
        else:
            stats.failures += 1

    def get_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all trajectories."""
        return {
            name: stats.to_dict()
            for name, stats in self.trajectory_stats.items()
        }

    def save(self, filepath: str):
        """Save bandit state to JSON file."""
        data = {
            "trajectories": {
                name: {
                    "successes": stats.successes,
                    "failures": stats.failures,
                }
                for name, stats in self.trajectory_stats.items()
            }
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str, trajectory_names: List[str]) -> "TrajectoryBandit":
        """
        Load bandit state from JSON file.

        Args:
            filepath: Path to saved state
            trajectory_names: List of trajectory names (for validation)

        Returns:
            Loaded TrajectoryBandit instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        bandit = cls(trajectory_names)

        for name, stats_data in data.get("trajectories", {}).items():
            if name in bandit.trajectory_stats:
                bandit.trajectory_stats[name].successes = stats_data["successes"]
                bandit.trajectory_stats[name].failures = stats_data["failures"]

        return bandit


# ============================================================================
# Convenience Interface
# ============================================================================

class TrajectorySelector:
    """
    High-level interface for trajectory selection using Thompson Sampling.
    Wraps TrajectoryBandit with a simpler API.
    """

    def __init__(self, trajectory_names: List[str]):
        """
        Args:
            trajectory_names: List of available trajectory names
        """
        self.bandit = TrajectoryBandit(trajectory_names)

    def select(self, trajectory_names: List[str]) -> str:
        """
        Select a trajectory for the given query.

        Args:
            trajectory_names: List of available trajectory names

        Returns:
            Selected trajectory name
        """
        # For now, just use all trajectories
        # In future, could filter based on query context
        return self.bandit.choose_trajectory()

    def update(self, trajectory_name: str, reward: float):
        """Update the bandit based on observed reward."""
        self.bandit.update(trajectory_name, reward)

    def get_best_trajectories(self, top_k: int = 3) -> List[tuple[str, float]]:
        """
        Get the top-k trajectories by mean reward.

        Returns:
            List of (trajectory_name, mean_reward) tuples
        """
        trajectories = [
            (name, stats.mean_reward)
            for name, stats in self.bandit.trajectory_stats.items()
        ]
        trajectories.sort(key=lambda x: x[1], reverse=True)
        return trajectories[:top_k]

    def get_statistics(self) -> Dict[str, Dict]:
        """Get detailed statistics for all trajectories."""
        return self.bandit.get_statistics()

    def save(self, filepath: str):
        """Save bandit state to file."""
        self.bandit.save(filepath)

    @classmethod
    def load(cls, filepath: str, trajectory_names: List[str]) -> "TrajectorySelector":
        """Load bandit state from file."""
        bandit = TrajectoryBandit.load(filepath, trajectory_names)
        selector = cls(trajectory_names)
        selector.bandit = bandit
        return selector


# ============================================================================
# Reward Calculation Utilities
# ============================================================================

class RewardCalculator:
    """
    Utilities for calculating rewards from answer quality metrics.
    Reward should be in [0, 1], where 1 = perfect result, 0 = failure.
    """

    @staticmethod
    def from_answer_quality(
        answer_generated: bool,
        answer_length: int,
        retrieval_count: int,
        user_feedback: float | None = None,
    ) -> float:
        """
        Calculate reward from answer quality metrics.

        Args:
            answer_generated: Whether an answer was generated
            answer_length: Length of generated answer (chars)
            retrieval_count: Number of relevant nodes/edges retrieved
            user_feedback: Optional explicit user feedback in [0, 1]

        Returns:
            Reward in [0, 1]
        """
        if user_feedback is not None:
            # If we have explicit feedback, weight it heavily
            implicit = RewardCalculator._implicit_reward(
                answer_generated, answer_length, retrieval_count
            )
            return 0.3 * implicit + 0.7 * user_feedback

        return RewardCalculator._implicit_reward(
            answer_generated, answer_length, retrieval_count
        )

    @staticmethod
    def _implicit_reward(
        answer_generated: bool,
        answer_length: int,
        retrieval_count: int,
    ) -> float:
        """Calculate implicit reward from observable metrics."""
        if not answer_generated:
            return 0.0

        # Reward components
        r_answer = 0.4  # Base reward for generating an answer

        # Reward for reasonable answer length
        if answer_length < 50:
            r_length = 0.0
        elif answer_length < 200:
            r_length = 0.2
        else:
            r_length = 0.3

        # Reward for retrieving relevant context
        if retrieval_count == 0:
            r_retrieval = 0.0
        elif retrieval_count < 5:
            r_retrieval = 0.1
        elif retrieval_count < 20:
            r_retrieval = 0.2
        else:
            r_retrieval = 0.3

        return r_answer + r_length + r_retrieval

    @staticmethod
    def from_user_feedback(feedback: str) -> float:
        """
        Convert user feedback to reward.

        Args:
            feedback: "thumbs_up", "thumbs_down", or "neutral"

        Returns:
            Reward in [0, 1]
        """
        mapping = {
            "thumbs_up": 1.0,
            "neutral": 0.5,
            "thumbs_down": 0.0,
        }
        return mapping.get(feedback, 0.5)
