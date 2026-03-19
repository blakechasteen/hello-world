"""
Evaluation metrics for contextual bandits.

Implements key metrics for bandit performance analysis:
- Expected Calibration Error (ECE): Uncertainty calibration
- Regret proxy: Suboptimality vs. oracle
- Reward tracking: Online performance monitoring
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class CalibrationBin:
    """Single bin for calibration analysis."""
    predicted_mean: float
    actual_mean: float
    count: int
    lower_bound: float
    upper_bound: float


class BanditEvaluator:
    """
    Evaluator for bandit performance metrics.

    Tracks online performance and computes offline metrics like ECE and regret.

    Attributes:
        history: List of (predicted_reward, actual_reward) tuples
        n_bins: Number of bins for ECE computation

    Example:
        >>> evaluator = BanditEvaluator(n_bins=10)
        >>>
        >>> # After each decision
        >>> evaluator.record(predicted=0.8, actual=0.92)
        >>> evaluator.record(predicted=0.6, actual=0.55)
        >>>
        >>> # Compute metrics
        >>> metrics = evaluator.compute_metrics()
        >>> print(f"ECE: {metrics['ece']:.4f}")
        >>> print(f"Mean reward: {metrics['mean_reward']:.4f}")
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize evaluator.

        Args:
            n_bins: Number of bins for ECE (typically 10)

        Raises:
            ValueError: If n_bins <= 0
        """
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")

        self.n_bins = n_bins
        self.history: list[tuple[float, float]] = []

    def record(self, predicted: float, actual: float) -> None:
        """
        Record a prediction and outcome.

        Args:
            predicted: Predicted reward (model output)
            actual: Actual reward (observed)

        Example:
            >>> evaluator.record(predicted=0.75, actual=0.82)
        """
        self.history.append((predicted, actual))

    def compute_ece(self) -> float:
        """
        Compute Expected Calibration Error.

        ECE measures how well predicted rewards match actual rewards.
        Lower is better (0 = perfect calibration).

        Returns:
            ECE score in [0, 1] (typically < 0.1 is good)

        Algorithm:
            1. Bin predictions by predicted value
            2. For each bin: compute |mean_predicted - mean_actual|
            3. Weighted average across bins

        Example:
            >>> ece = evaluator.compute_ece()
            >>> if ece < 0.08:
            ...     print("Well calibrated!")

        Reference:
            Naeini et al. "Obtaining Well Calibrated Probabilities Using
            Bayesian Binning" (AAAI 2015)
        """
        if not self.history:
            return 0.0

        predictions = np.array([p for p, _ in self.history])
        actuals = np.array([a for _, a in self.history])

        # Create bins
        bin_edges = np.linspace(
            predictions.min(),
            predictions.max(),
            self.n_bins + 1,
        )

        ece = 0.0
        total_count = len(predictions)

        for i in range(self.n_bins):
            # Find predictions in this bin
            in_bin = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])

            # Handle last bin edge inclusively
            if i == self.n_bins - 1:
                in_bin = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])

            bin_count = in_bin.sum()

            if bin_count == 0:
                continue

            # Mean predicted and actual in this bin
            mean_predicted = predictions[in_bin].mean()
            mean_actual = actuals[in_bin].mean()

            # Weighted calibration error
            bin_weight = bin_count / total_count
            ece += bin_weight * abs(mean_predicted - mean_actual)

        return float(ece)

    def get_calibration_bins(self) -> list[CalibrationBin]:
        """
        Get detailed calibration bin statistics.

        Returns:
            List of CalibrationBin objects with predicted vs actual means

        Example:
            >>> bins = evaluator.get_calibration_bins()
            >>> for bin in bins:
            ...     print(f"[{bin.lower_bound:.2f}, {bin.upper_bound:.2f}]: "
            ...           f"predicted={bin.predicted_mean:.3f}, "
            ...           f"actual={bin.actual_mean:.3f}, n={bin.count}")
        """
        if not self.history:
            return []

        predictions = np.array([p for p, _ in self.history])
        actuals = np.array([a for _, a in self.history])

        bin_edges = np.linspace(
            predictions.min(),
            predictions.max(),
            self.n_bins + 1,
        )

        bins = []

        for i in range(self.n_bins):
            in_bin = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])

            if i == self.n_bins - 1:
                in_bin = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])

            bin_count = int(in_bin.sum())

            if bin_count == 0:
                continue

            mean_predicted = float(predictions[in_bin].mean())
            mean_actual = float(actuals[in_bin].mean())

            bins.append(
                CalibrationBin(
                    predicted_mean=mean_predicted,
                    actual_mean=mean_actual,
                    count=bin_count,
                    lower_bound=float(bin_edges[i]),
                    upper_bound=float(bin_edges[i + 1]),
                )
            )

        return bins

    def compute_regret_proxy(
        self,
        oracle_rewards: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """
        Compute regret proxy (suboptimality vs. oracle).

        Regret = Σ(oracle_reward - actual_reward) / n
        Lower is better (0 = always picks best action).

        Args:
            oracle_rewards: Best possible reward for each decision
                           If None, uses max(actual_rewards) as proxy

        Returns:
            Mean regret (0 = optimal, higher = more suboptimal)

        Note:
            True regret requires knowing the oracle (best action per context).
            Without oracle, we use max observed reward as approximation.

        Example:
            >>> # With oracle (offline evaluation)
            >>> regret = evaluator.compute_regret_proxy(oracle_rewards=best_rewards)
            >>>
            >>> # Without oracle (proxy)
            >>> regret = evaluator.compute_regret_proxy()  # Uses max(actuals)
        """
        if not self.history:
            return 0.0

        actuals = np.array([a for _, a in self.history])

        if oracle_rewards is None:
            # Proxy: assume max observed is oracle
            oracle_rewards = np.full_like(actuals, actuals.max())
        else:
            if len(oracle_rewards) != len(actuals):
                raise ValueError(
                    f"oracle_rewards length ({len(oracle_rewards)}) doesn't match "
                    f"history length ({len(actuals)})"
                )

        regrets = oracle_rewards - actuals
        mean_regret = float(regrets.mean())

        return mean_regret

    def compute_metrics(self) -> dict[str, float]:
        """
        Compute all evaluation metrics.

        Returns:
            Dict with:
                - ece: Expected Calibration Error
                - regret_proxy: Mean regret vs. oracle
                - mean_reward: Mean actual reward
                - std_reward: Std dev of rewards
                - mean_predicted: Mean predicted reward
                - mae: Mean absolute error (predicted vs actual)
                - count: Number of observations

        Example:
            >>> metrics = evaluator.compute_metrics()
            >>> print(f"ECE: {metrics['ece']:.4f}")
            >>> print(f"Regret: {metrics['regret_proxy']:.4f}")
            >>> print(f"Mean reward: {metrics['mean_reward']:.4f}")
        """
        if not self.history:
            return {
                "ece": 0.0,
                "regret_proxy": 0.0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_predicted": 0.0,
                "mae": 0.0,
                "count": 0,
            }

        predictions = np.array([p for p, _ in self.history])
        actuals = np.array([a for _, a in self.history])

        ece = self.compute_ece()
        regret = self.compute_regret_proxy()
        mae = float(np.abs(predictions - actuals).mean())

        return {
            "ece": ece,
            "regret_proxy": regret,
            "mean_reward": float(actuals.mean()),
            "std_reward": float(actuals.std()),
            "mean_predicted": float(predictions.mean()),
            "mae": mae,
            "count": len(self.history),
        }

    def reset(self) -> None:
        """Clear history (for new evaluation period)."""
        self.history.clear()

    def __len__(self) -> int:
        """Number of recorded observations."""
        return len(self.history)

    def __repr__(self) -> str:
        metrics = self.compute_metrics()
        return (
            f"BanditEvaluator(n={metrics['count']}, "
            f"ece={metrics['ece']:.4f}, "
            f"mean_reward={metrics['mean_reward']:.4f})"
        )


def compute_cumulative_regret(
    rewards: npt.NDArray[np.floating],
    oracle_rewards: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute cumulative regret over time.

    Useful for visualizing learning progress.

    Args:
        rewards: Actual rewards received, shape [T]
        oracle_rewards: Oracle rewards (best possible), shape [T]

    Returns:
        Cumulative regret, shape [T]

    Example:
        >>> rewards = np.array([0.5, 0.7, 0.8, 0.9])
        >>> oracle = np.array([0.9, 0.9, 0.9, 0.9])
        >>> cum_regret = compute_cumulative_regret(rewards, oracle)
        >>> print(cum_regret)  # [0.4, 0.6, 0.7, 0.7]
    """
    regrets = oracle_rewards - rewards
    cumulative_regret = np.cumsum(regrets)
    return cumulative_regret


def compute_moving_average_reward(
    rewards: npt.NDArray[np.floating],
    window: int = 100,
) -> npt.NDArray[np.floating]:
    """
    Compute moving average reward for smoothed performance tracking.

    Args:
        rewards: Actual rewards, shape [T]
        window: Window size for moving average

    Returns:
        Moving average rewards, shape [T]

    Example:
        >>> rewards = np.random.randn(1000)
        >>> smooth = compute_moving_average_reward(rewards, window=100)
    """
    if window > len(rewards):
        window = len(rewards)

    cumsum = np.cumsum(rewards)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]

    moving_avg = cumsum.copy()
    moving_avg[:window] = cumsum[:window] / np.arange(1, window + 1)
    moving_avg[window:] = cumsum[window:] / window

    return moving_avg
