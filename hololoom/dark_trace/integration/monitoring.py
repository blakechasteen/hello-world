"""
Dark Trace Production Monitoring

Real-time feature monitoring and drift detection for production deployments.
Tracks feature statistics, detects distribution shifts, and alerts on anomalies.

Key Components:
- FeatureStatistics: Running statistics for feature activations
- DriftDetector: Detects when features drift from baseline
- InterpretabilityMonitor: Main production monitoring class
- MonitorConfig: Configuration for monitoring behavior

Design Philosophy:
- Production monitoring should be lightweight (<1ms overhead)
- Drift detection enables early warning of model degradation
- Statistics enable understanding of feature behavior over time
- Alerts should be actionable, not noisy

Usage:
    from hololoom.dark_trace.integration.monitoring import (
        InterpretabilityMonitor,
        create_monitor,
    )

    # Create monitor
    monitor = create_monitor(config)

    # Record activations during inference
    monitor.record(activations, metadata={"query_id": "q123"})

    # Check for drift
    drift_report = monitor.check_drift()
    if drift_report.has_drift:
        print(f"Drift detected in features: {drift_report.drifted_features}")

    # Get statistics
    stats = monitor.get_statistics()
    print(f"Mean activation: {stats.global_mean:.3f}")
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine


class DriftType(Enum):
    """Types of feature drift."""
    MEAN_SHIFT = "mean_shift"           # Mean activation changed significantly
    VARIANCE_CHANGE = "variance_change"  # Variance changed significantly
    DISTRIBUTION_SHIFT = "distribution_shift"  # Overall distribution changed
    SPARSITY_CHANGE = "sparsity_change"  # Sparsity pattern changed
    CORRELATION_CHANGE = "correlation_change"  # Feature correlations changed


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FeatureStatistics:
    """Running statistics for a single feature."""

    feature_id: str
    n_observations: int = 0

    # Running statistics (Welford's algorithm)
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from mean

    # Extrema
    min_activation: float = float('inf')
    max_activation: float = float('-inf')

    # Sparsity tracking
    n_zero: int = 0
    n_positive: int = 0

    # Recent values for drift detection
    recent_values: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Baseline statistics (for drift comparison)
    baseline_mean: float | None = None
    baseline_std: float | None = None
    baseline_sparsity: float | None = None

    @property
    def variance(self) -> float:
        """Compute variance from running statistics."""
        if self.n_observations < 2:
            return 0.0
        return self.m2 / (self.n_observations - 1)

    @property
    def std(self) -> float:
        """Standard deviation."""
        return np.sqrt(self.variance)

    @property
    def sparsity(self) -> float:
        """Fraction of zero activations."""
        if self.n_observations == 0:
            return 0.0
        return self.n_zero / self.n_observations

    def update(self, value: float) -> None:
        """Update statistics with new observation (Welford's algorithm)."""
        self.n_observations += 1

        # Update running mean and variance
        delta = value - self.mean
        self.mean += delta / self.n_observations
        delta2 = value - self.mean
        self.m2 += delta * delta2

        # Update extrema
        self.min_activation = min(self.min_activation, value)
        self.max_activation = max(self.max_activation, value)

        # Update sparsity tracking
        if abs(value) < 1e-8:
            self.n_zero += 1
        elif value > 0:
            self.n_positive += 1

        # Store recent value for drift detection
        self.recent_values.append(value)

    def set_baseline(self) -> None:
        """Set current statistics as baseline for drift detection."""
        self.baseline_mean = self.mean
        self.baseline_std = self.std
        self.baseline_sparsity = self.sparsity

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "feature_id": self.feature_id,
            "n_observations": self.n_observations,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_activation if self.min_activation != float('inf') else None,
            "max": self.max_activation if self.max_activation != float('-inf') else None,
            "sparsity": self.sparsity,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
        }


@dataclass
class DriftReport:
    """Report of detected drift."""

    has_drift: bool = False
    drifted_features: list[str] = field(default_factory=list)
    drift_details: dict[str, list[DriftType]] = field(default_factory=dict)
    drift_scores: dict[str, float] = field(default_factory=dict)
    alert_level: AlertLevel = AlertLevel.INFO
    timestamp: float = field(default_factory=time.time)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "has_drift": self.has_drift,
            "drifted_features": self.drifted_features,
            "drift_details": {
                k: [d.value for d in v]
                for k, v in self.drift_details.items()
            },
            "drift_scores": self.drift_scores,
            "alert_level": self.alert_level.value,
            "timestamp": self.timestamp,
            "recommendations": self.recommendations,
        }


@dataclass
class Alert:
    """Monitoring alert."""

    level: AlertLevel
    message: str
    feature_id: str | None = None
    drift_type: DriftType | None = None
    value: float | None = None
    threshold: float | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorConfig:
    """Configuration for interpretability monitoring."""

    # Statistics settings
    track_per_feature: bool = True
    max_features_tracked: int = 1000
    recent_window_size: int = 1000

    # Drift detection settings
    enable_drift_detection: bool = True
    drift_check_interval: int = 100  # Check every N observations
    mean_drift_threshold: float = 2.0  # Standard deviations
    variance_drift_threshold: float = 0.5  # Relative change
    sparsity_drift_threshold: float = 0.1  # Absolute change

    # Alerting settings
    enable_alerting: bool = True
    critical_drift_threshold: float = 3.0  # Standard deviations for critical
    max_alerts_per_hour: int = 10

    # Performance settings
    async_updates: bool = False
    batch_size: int = 100


class DriftDetector:
    """
    Detects feature drift from baseline distributions.

    Implements multiple drift detection methods:
    - Mean shift detection (z-score based)
    - Variance change detection (F-test based)
    - Sparsity change detection (absolute difference)
    """

    def __init__(self, config: MonitorConfig | None = None):
        """
        Initialize drift detector.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitorConfig()

    def detect_drift(
        self,
        statistics: dict[str, FeatureStatistics],
        features_to_check: set[str] | None = None,
    ) -> DriftReport:
        """
        Check for drift in feature statistics.

        Args:
            statistics: Current feature statistics
            features_to_check: Optional subset of features to check

        Returns:
            DriftReport with detected drifts
        """
        report = DriftReport()
        features = features_to_check or set(statistics.keys())

        for feature_id in features:
            if feature_id not in statistics:
                continue

            stats = statistics[feature_id]

            # Skip if no baseline set
            if stats.baseline_mean is None:
                continue

            # Check for various drift types
            drift_types = []
            drift_score = 0.0

            # Mean shift detection
            if stats.baseline_std and stats.baseline_std > 0:
                z_score = abs(stats.mean - stats.baseline_mean) / stats.baseline_std
                if z_score > self.config.mean_drift_threshold:
                    drift_types.append(DriftType.MEAN_SHIFT)
                    drift_score = max(drift_score, z_score / self.config.mean_drift_threshold)

            # Variance change detection
            if stats.baseline_std and stats.baseline_std > 0:
                variance_ratio = stats.std / stats.baseline_std
                if abs(variance_ratio - 1.0) > self.config.variance_drift_threshold:
                    drift_types.append(DriftType.VARIANCE_CHANGE)
                    drift_score = max(
                        drift_score,
                        abs(variance_ratio - 1.0) / self.config.variance_drift_threshold
                    )

            # Sparsity change detection
            if stats.baseline_sparsity is not None:
                sparsity_diff = abs(stats.sparsity - stats.baseline_sparsity)
                if sparsity_diff > self.config.sparsity_drift_threshold:
                    drift_types.append(DriftType.SPARSITY_CHANGE)
                    drift_score = max(
                        drift_score,
                        sparsity_diff / self.config.sparsity_drift_threshold
                    )

            # Record if any drift detected
            if drift_types:
                report.has_drift = True
                report.drifted_features.append(feature_id)
                report.drift_details[feature_id] = drift_types
                report.drift_scores[feature_id] = drift_score

        # Determine alert level
        if report.has_drift:
            max_score = max(report.drift_scores.values()) if report.drift_scores else 0
            if max_score >= self.config.critical_drift_threshold:
                report.alert_level = AlertLevel.CRITICAL
            elif max_score >= self.config.mean_drift_threshold:
                report.alert_level = AlertLevel.WARNING
            else:
                report.alert_level = AlertLevel.INFO

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: DriftReport) -> list[str]:
        """Generate actionable recommendations based on drift."""
        recommendations = []

        # Count drift types
        drift_type_counts: dict[DriftType, int] = {}
        for drift_types in report.drift_details.values():
            for dt in drift_types:
                drift_type_counts[dt] = drift_type_counts.get(dt, 0) + 1

        # Generate recommendations based on drift types
        if DriftType.MEAN_SHIFT in drift_type_counts:
            count = drift_type_counts[DriftType.MEAN_SHIFT]
            recommendations.append(
                f"Mean shift detected in {count} features. "
                "Consider retraining SAE or updating baseline."
            )

        if DriftType.VARIANCE_CHANGE in drift_type_counts:
            count = drift_type_counts[DriftType.VARIANCE_CHANGE]
            recommendations.append(
                f"Variance change in {count} features. "
                "May indicate distribution shift in input data."
            )

        if DriftType.SPARSITY_CHANGE in drift_type_counts:
            count = drift_type_counts[DriftType.SPARSITY_CHANGE]
            recommendations.append(
                f"Sparsity pattern changed in {count} features. "
                "Check for dead features or activation collapse."
            )

        if report.alert_level == AlertLevel.CRITICAL:
            recommendations.append(
                "CRITICAL: Consider pausing deployment and investigating immediately."
            )

        return recommendations


class InterpretabilityMonitor:
    """
    Production monitoring for Dark Trace interpretability.

    Tracks feature statistics over time, detects drift from baseline,
    and generates alerts for anomalous behavior.
    """

    def __init__(
        self,
        engine: Optional["DarkTraceEngine"] = None,
        config: MonitorConfig | None = None,
    ):
        """
        Initialize interpretability monitor.

        Args:
            engine: Optional DarkTraceEngine for integrated monitoring
            config: Monitoring configuration
        """
        self.engine = engine
        self.config = config or MonitorConfig()

        # Statistics tracking
        self._statistics: dict[str, FeatureStatistics] = {}
        self._global_stats = FeatureStatistics(feature_id="__global__")

        # Drift detection
        self._drift_detector = DriftDetector(self.config)
        self._observation_count = 0
        self._last_drift_check = 0

        # Alerting
        self._alerts: list[Alert] = []
        self._alerts_this_hour: int = 0
        self._hour_start: float = time.time()

        # Performance tracking
        self._total_record_time_ms: float = 0.0
        self._total_records: int = 0

    def record(
        self,
        activations: np.ndarray,
        feature_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record activations for monitoring.

        Args:
            activations: Activation values (1D array or batch)
            feature_ids: Optional feature IDs (default: use indices)
            metadata: Optional metadata for this observation
        """
        start_time = time.time()

        # Flatten if necessary
        if activations.ndim > 1:
            activations = activations.flatten()

        # Generate feature IDs if not provided
        if feature_ids is None:
            feature_ids = [f"feature_{i}" for i in range(len(activations))]

        # Update global statistics
        global_mean = float(np.mean(activations))
        self._global_stats.update(global_mean)

        # Update per-feature statistics
        if self.config.track_per_feature:
            for i, (fid, value) in enumerate(zip(feature_ids, activations)):
                if i >= self.config.max_features_tracked:
                    break

                if fid not in self._statistics:
                    self._statistics[fid] = FeatureStatistics(
                        feature_id=fid,
                        recent_values=deque(maxlen=self.config.recent_window_size),
                    )

                self._statistics[fid].update(float(value))

        # Update observation count
        self._observation_count += 1

        # Check for drift periodically
        if (
            self.config.enable_drift_detection
            and self._observation_count - self._last_drift_check
                >= self.config.drift_check_interval
        ):
            self._check_and_alert_drift()
            self._last_drift_check = self._observation_count

        # Track performance
        elapsed_ms = (time.time() - start_time) * 1000
        self._total_record_time_ms += elapsed_ms
        self._total_records += 1

    def record_from_trace(
        self,
        trace_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record from a TraceResult object.

        Args:
            trace_result: TraceResult from DarkTraceEngine
            metadata: Optional additional metadata
        """
        # Extract activations from trace result
        if hasattr(trace_result, 'activations') and trace_result.activations is not None:
            self.record(trace_result.activations, metadata=metadata)

        # Also record active feature activations
        if hasattr(trace_result, 'active_features'):
            feature_ids = []
            activations = []
            for feature in trace_result.active_features:
                fid = getattr(feature, 'id', str(feature))
                activation = getattr(feature, 'activation', 0.5)
                feature_ids.append(fid)
                activations.append(activation)

            if activations:
                self.record(
                    np.array(activations),
                    feature_ids=feature_ids,
                    metadata=metadata,
                )

    def set_baseline(self, feature_ids: list[str] | None = None) -> None:
        """
        Set current statistics as baseline for drift detection.

        Args:
            feature_ids: Optional subset of features to set baseline for
        """
        features = feature_ids or list(self._statistics.keys())

        for fid in features:
            if fid in self._statistics:
                self._statistics[fid].set_baseline()

        self._global_stats.set_baseline()

    def check_drift(
        self,
        features_to_check: set[str] | None = None,
    ) -> DriftReport:
        """
        Check for feature drift.

        Args:
            features_to_check: Optional subset of features to check

        Returns:
            DriftReport with detected drifts
        """
        return self._drift_detector.detect_drift(
            self._statistics,
            features_to_check,
        )

    def _check_and_alert_drift(self) -> None:
        """Check for drift and generate alerts if needed."""
        report = self.check_drift()

        if report.has_drift and self.config.enable_alerting:
            # Reset hourly counter if needed
            current_time = time.time()
            if current_time - self._hour_start >= 3600:
                self._alerts_this_hour = 0
                self._hour_start = current_time

            # Generate alerts if under limit
            if self._alerts_this_hour < self.config.max_alerts_per_hour:
                alert = Alert(
                    level=report.alert_level,
                    message=f"Drift detected in {len(report.drifted_features)} features",
                    metadata={
                        "drifted_features": report.drifted_features,
                        "recommendations": report.recommendations,
                    },
                )
                self._alerts.append(alert)
                self._alerts_this_hour += 1

    def get_statistics(
        self,
        feature_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get current statistics.

        Args:
            feature_ids: Optional subset of features to get stats for

        Returns:
            Dictionary of statistics
        """
        result = {
            "global": self._global_stats.to_dict(),
            "observation_count": self._observation_count,
            "features_tracked": len(self._statistics),
            "avg_record_time_ms": (
                self._total_record_time_ms / self._total_records
                if self._total_records > 0 else 0
            ),
        }

        if feature_ids:
            result["features"] = {
                fid: self._statistics[fid].to_dict()
                for fid in feature_ids
                if fid in self._statistics
            }
        else:
            # Return top features by observation count
            sorted_features = sorted(
                self._statistics.items(),
                key=lambda x: x[1].n_observations,
                reverse=True,
            )[:100]
            result["features"] = {
                fid: stats.to_dict()
                for fid, stats in sorted_features
            }

        return result

    def get_feature_statistics(self, feature_id: str) -> FeatureStatistics | None:
        """Get statistics for a specific feature."""
        return self._statistics.get(feature_id)

    def get_alerts(
        self,
        since: float | None = None,
        level: AlertLevel | None = None,
    ) -> list[Alert]:
        """
        Get alerts.

        Args:
            since: Only return alerts after this timestamp
            level: Filter by alert level

        Returns:
            List of matching alerts
        """
        alerts = self._alerts

        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]

        if level is not None:
            alerts = [a for a in alerts if a.level == level]

        return alerts

    def clear_alerts(self) -> int:
        """Clear all alerts. Returns count cleared."""
        count = len(self._alerts)
        self._alerts = []
        return count

    def get_health_summary(self) -> dict[str, Any]:
        """
        Get overall health summary.

        Returns:
            Dictionary with health indicators
        """
        drift_report = self.check_drift()

        return {
            "status": "healthy" if not drift_report.has_drift else "degraded",
            "observation_count": self._observation_count,
            "features_tracked": len(self._statistics),
            "global_mean": self._global_stats.mean,
            "global_std": self._global_stats.std,
            "drift_detected": drift_report.has_drift,
            "drifted_feature_count": len(drift_report.drifted_features),
            "alert_level": drift_report.alert_level.value,
            "pending_alerts": len(self._alerts),
            "avg_record_latency_ms": (
                self._total_record_time_ms / self._total_records
                if self._total_records > 0 else 0
            ),
        }

    def reset(self) -> None:
        """Reset all statistics and alerts."""
        self._statistics.clear()
        self._global_stats = FeatureStatistics(feature_id="__global__")
        self._alerts.clear()
        self._observation_count = 0
        self._last_drift_check = 0
        self._total_record_time_ms = 0.0
        self._total_records = 0


def create_monitor(
    engine: Optional["DarkTraceEngine"] = None,
    config: MonitorConfig | None = None,
    enable_drift_detection: bool = True,
    enable_alerting: bool = True,
) -> InterpretabilityMonitor:
    """
    Create an interpretability monitor.

    Args:
        engine: Optional DarkTraceEngine for integrated monitoring
        config: Optional configuration (will be created if not provided)
        enable_drift_detection: Enable drift detection
        enable_alerting: Enable alerting

    Returns:
        Configured InterpretabilityMonitor
    """
    if config is None:
        config = MonitorConfig(
            enable_drift_detection=enable_drift_detection,
            enable_alerting=enable_alerting,
        )

    return InterpretabilityMonitor(engine=engine, config=config)
