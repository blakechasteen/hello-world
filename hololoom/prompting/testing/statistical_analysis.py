"""Statistical analysis for metrics time series.

Provides extended statistics beyond min/max/avg:
- Standard deviation and variance
- Pearson correlation coefficient
- Rolling statistics (mean, std)
- Trend detection (increasing, decreasing, stable)
- Z-score based anomaly detection
"""

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class StatisticalSummary:
    """Extended statistics beyond min/max/avg."""

    mean: float
    std_dev: float
    variance: float
    median: float
    percentiles: dict[int, float]
    trend: str  # "increasing", "decreasing", "stable"
    anomalies: list[int]  # Indices of anomalous values


class StatisticalAnalyzer:
    """Statistical analysis for metrics time series."""

    def __init__(self, anomaly_threshold: float = 2.0):
        """Initialize analyzer.

        Args:
            anomaly_threshold: Z-score threshold for anomaly detection (default: 2.0)
        """
        self.anomaly_threshold = anomaly_threshold

    def analyze(self, values: list[float]) -> StatisticalSummary:
        """Compute extended statistics on values.

        Args:
            values: List of numeric values to analyze

        Returns:
            StatisticalSummary with mean, std_dev, variance, median,
            percentiles, trend, and anomaly indices
        """
        if not values:
            return self._empty_summary()

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        sorted_vals = sorted(values)

        return StatisticalSummary(
            mean=mean,
            std_dev=std_dev,
            variance=variance,
            median=self._median(sorted_vals),
            percentiles=self._percentiles(sorted_vals),
            trend=self._detect_trend(values),
            anomalies=self._detect_anomalies(values, mean, std_dev)
        )

    def correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient.

        Args:
            x: First series of values
            y: Second series of values (must be same length as x)

        Returns:
            Correlation coefficient between -1.0 and 1.0
            Returns 0.0 if series are too short or invalid
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(a * b for a, b in zip(x, y))
        sum_x2 = sum(a ** 2 for a in x)
        sum_y2 = sum(b ** 2 for b in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt(
            (n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)
        )

        return numerator / denominator if denominator != 0 else 0.0

    def rolling_stats(
        self,
        values: list[float],
        window_size: int = 10
    ) -> list[dict[str, float]]:
        """Calculate rolling mean and standard deviation.

        Args:
            values: Series of values
            window_size: Rolling window size (default: 10)

        Returns:
            List of dicts with index, rolling_mean, rolling_std
        """
        results = []
        window: deque = deque(maxlen=window_size)

        for i, v in enumerate(values):
            window.append(v)
            if len(window) >= 2:
                mean = sum(window) / len(window)
                std = math.sqrt(
                    sum((x - mean) ** 2 for x in window) / len(window)
                )
                results.append({
                    "index": i,
                    "rolling_mean": mean,
                    "rolling_std": std
                })

        return results

    def _median(self, sorted_values: list[float]) -> float:
        """Calculate median from sorted values."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return sorted_values[n // 2]
        else:
            mid = n // 2
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2

    def _detect_trend(self, values: list[float]) -> str:
        """Simple linear trend detection.

        Compares average of first third to average of last third.

        Args:
            values: Series of values

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(values) < 3:
            return "stable"

        n = len(values)
        third = n // 3
        if third == 0:
            return "stable"

        first_avg = sum(values[:third]) / third
        last_avg = sum(values[-third:]) / third

        diff = last_avg - first_avg
        value_range = max(values) - min(values)
        threshold = value_range * 0.1 if value_range > 0 else 0.01

        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        return "stable"

    def _detect_anomalies(
        self,
        values: list[float],
        mean: float,
        std_dev: float
    ) -> list[int]:
        """Z-score based anomaly detection.

        Args:
            values: Series of values
            mean: Pre-computed mean
            std_dev: Pre-computed standard deviation

        Returns:
            List of indices where values are anomalous (|z-score| > threshold)
        """
        if std_dev == 0:
            return []

        anomalies = []
        for i, v in enumerate(values):
            z_score = abs(v - mean) / std_dev
            if z_score > self.anomaly_threshold:
                anomalies.append(i)
        return anomalies

    def _percentiles(self, sorted_values: list[float]) -> dict[int, float]:
        """Calculate standard percentiles.

        Args:
            sorted_values: Pre-sorted list of values

        Returns:
            Dict mapping percentile (25, 50, 75, 90, 95, 99) to value
        """
        if not sorted_values:
            return {}

        result = {}
        n = len(sorted_values)
        for p in [25, 50, 75, 90, 95, 99]:
            idx = int((p / 100) * n)
            idx = min(idx, n - 1)
            result[p] = sorted_values[idx]
        return result

    def _empty_summary(self) -> StatisticalSummary:
        """Return empty summary for edge cases."""
        return StatisticalSummary(
            mean=0.0,
            std_dev=0.0,
            variance=0.0,
            median=0.0,
            percentiles={},
            trend="stable",
            anomalies=[]
        )


def create_statistical_analyzer(
    anomaly_threshold: float = 2.0
) -> StatisticalAnalyzer:
    """Factory function for StatisticalAnalyzer.

    Args:
        anomaly_threshold: Z-score threshold for anomaly detection

    Returns:
        New StatisticalAnalyzer instance
    """
    return StatisticalAnalyzer(anomaly_threshold=anomaly_threshold)
