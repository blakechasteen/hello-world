"""
Dark Trace Multilayer: Cross-Layer Correlation Tracking

Tracks how features correlate across layers using memory-efficient
streaming algorithms (Welford's online algorithm).

Key Capabilities:
- Per-layer feature correlation tracking
- Cross-layer correlation matrices
- Memory-efficient streaming updates
- Temporal correlation analysis

Research Basis:
- Scaling Monosemanticity (Anthropic 2024): Multi-layer SAE analysis
- Welford's Online Algorithm: Numerical stability for streaming stats

Created: 2025-12-28
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import math
import time


# =============================================================================
# Configuration
# =============================================================================

class CorrelationMethod(Enum):
    """Methods for computing correlations."""
    PEARSON = "pearson"           # Linear correlation
    SPEARMAN = "spearman"         # Rank correlation (more robust)
    COSINE = "cosine"             # Cosine similarity
    MUTUAL_INFO = "mutual_info"   # Mutual information estimate


@dataclass
class CorrelationConfig:
    """Configuration for correlation tracking."""

    # Correlation computation
    method: CorrelationMethod = CorrelationMethod.PEARSON
    min_samples: int = 30          # Minimum samples before computing correlation

    # Memory management
    max_features_per_layer: int = 1000   # Maximum features to track per layer
    max_correlations_stored: int = 10000  # Maximum correlation pairs to store
    correlation_threshold: float = 0.3    # Minimum |correlation| to store

    # Temporal settings
    decay_factor: float = 0.99     # Exponential decay for temporal weighting
    window_size: int = 1000        # Rolling window size for recent correlations

    # Update settings
    update_frequency: int = 10     # Update correlations every N samples
    async_updates: bool = True     # Enable async correlation updates

    @classmethod
    def default(cls) -> "CorrelationConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def high_precision(cls) -> "CorrelationConfig":
        """High precision configuration for research."""
        return cls(
            min_samples=100,
            max_features_per_layer=5000,
            max_correlations_stored=50000,
            correlation_threshold=0.1,
            window_size=5000,
            update_frequency=1
        )

    @classmethod
    def fast(cls) -> "CorrelationConfig":
        """Fast configuration with lower precision."""
        return cls(
            min_samples=10,
            max_features_per_layer=500,
            max_correlations_stored=5000,
            correlation_threshold=0.5,
            window_size=500,
            update_frequency=50
        )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CorrelationPair:
    """A single correlation between two features."""
    source_feature: str        # Feature ID (e.g., "layer_5.sae_42")
    target_feature: str        # Feature ID (e.g., "layer_6.sae_17")
    correlation: float         # Correlation value [-1, 1]
    n_samples: int             # Number of samples used
    p_value: Optional[float] = None   # Statistical significance
    confidence: float = 0.0    # Confidence in correlation estimate
    last_updated: float = 0.0  # Timestamp of last update

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant."""
        if self.p_value is None:
            return self.n_samples >= 30  # Default to sample size check
        return self.p_value < alpha

    def __hash__(self):
        return hash((self.source_feature, self.target_feature))


@dataclass
class LayerCorrelation:
    """Correlations between features in two layers."""
    source_layer: int
    target_layer: int
    correlations: List[CorrelationPair] = field(default_factory=list)
    mean_correlation: float = 0.0
    max_correlation: float = 0.0
    n_significant: int = 0
    timestamp: float = field(default_factory=time.time)

    def get_top_correlations(self, k: int = 10) -> List[CorrelationPair]:
        """Get top-k correlations by absolute value."""
        sorted_corrs = sorted(
            self.correlations,
            key=lambda c: abs(c.correlation),
            reverse=True
        )
        return sorted_corrs[:k]

    def get_correlations_for_feature(
        self,
        feature_id: str
    ) -> List[CorrelationPair]:
        """Get all correlations involving a specific feature."""
        return [
            c for c in self.correlations
            if c.source_feature == feature_id or c.target_feature == feature_id
        ]

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.correlations:
            return {
                "source_layer": self.source_layer,
                "target_layer": self.target_layer,
                "n_correlations": 0,
                "mean_correlation": 0.0,
                "max_correlation": 0.0,
                "n_significant": 0
            }

        abs_corrs = [abs(c.correlation) for c in self.correlations]
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "n_correlations": len(self.correlations),
            "mean_correlation": sum(abs_corrs) / len(abs_corrs),
            "max_correlation": max(abs_corrs),
            "min_correlation": min(abs_corrs),
            "n_significant": self.n_significant,
            "n_positive": sum(1 for c in self.correlations if c.correlation > 0),
            "n_negative": sum(1 for c in self.correlations if c.correlation < 0)
        }


@dataclass
class CorrelationSnapshot:
    """Snapshot of all cross-layer correlations at a point in time."""
    timestamp: float
    layer_correlations: Dict[Tuple[int, int], LayerCorrelation]
    n_total_pairs: int
    n_significant_pairs: int
    global_mean_correlation: float

    def get_correlation_matrix(
        self,
        source_layer: int,
        target_layer: int
    ) -> Optional[LayerCorrelation]:
        """Get correlation data between two specific layers."""
        return self.layer_correlations.get((source_layer, target_layer))

    def get_feature_trajectory(
        self,
        feature_id: str
    ) -> List[CorrelationPair]:
        """Get all correlations for a feature across all layer pairs."""
        trajectory = []
        for layer_corr in self.layer_correlations.values():
            for corr in layer_corr.correlations:
                if feature_id in (corr.source_feature, corr.target_feature):
                    trajectory.append(corr)
        return trajectory


# =============================================================================
# Welford's Online Algorithm for Streaming Correlation
# =============================================================================

class WelfordCorrelator:
    """
    Memory-efficient streaming correlation using Welford's algorithm.

    Computes Pearson correlation coefficient incrementally without
    storing all data points.

    Algorithm:
        For each new pair (x, y):
        1. Update count: n += 1
        2. Update means: mean_x += (x - mean_x) / n
        3. Update covariance: C_xy += (x - mean_x) * (y - mean_y_old)
        4. Update variances: M2_x += (x - mean_x_old) * (x - mean_x)

        Correlation = C_xy / sqrt(M2_x * M2_y)

    Usage:
        correlator = WelfordCorrelator()
        for x, y in data_stream:
            correlator.update(x, y)
        print(f"Correlation: {correlator.correlation()}")
    """

    def __init__(self, decay_factor: Optional[float] = None):
        """
        Initialize correlator.

        Args:
            decay_factor: If set, use exponential weighted correlation
        """
        self.n: int = 0
        self.mean_x: float = 0.0
        self.mean_y: float = 0.0
        self.M2_x: float = 0.0     # Sum of squared deviations for x
        self.M2_y: float = 0.0     # Sum of squared deviations for y
        self.C_xy: float = 0.0     # Co-moment (unnormalized covariance)
        self.decay_factor = decay_factor
        self._last_update = time.time()

    def update(self, x: float, y: float) -> None:
        """
        Update statistics with a new data point.

        Uses Welford's online algorithm for numerical stability.

        Args:
            x: Value for first variable
            y: Value for second variable
        """
        self.n += 1

        if self.decay_factor is not None and self.n > 1:
            # Exponential weighted update
            alpha = 1.0 - self.decay_factor

            delta_x = x - self.mean_x
            delta_y = y - self.mean_y

            self.mean_x += alpha * delta_x
            self.mean_y += alpha * delta_y

            self.M2_x = self.decay_factor * self.M2_x + alpha * delta_x * (x - self.mean_x)
            self.M2_y = self.decay_factor * self.M2_y + alpha * delta_y * (y - self.mean_y)
            self.C_xy = self.decay_factor * self.C_xy + alpha * delta_x * (y - self.mean_y)
        else:
            # Standard Welford's algorithm
            delta_x = x - self.mean_x
            self.mean_x += delta_x / self.n
            delta_x2 = x - self.mean_x
            self.M2_x += delta_x * delta_x2

            delta_y = y - self.mean_y
            mean_y_old = self.mean_y
            self.mean_y += delta_y / self.n
            delta_y2 = y - self.mean_y
            self.M2_y += delta_y * delta_y2

            # Cross-moment uses old mean_y
            self.C_xy += delta_x * (y - mean_y_old)

        self._last_update = time.time()

    def correlation(self) -> float:
        """
        Compute current correlation coefficient.

        Returns:
            Pearson correlation coefficient [-1, 1], or 0 if insufficient data
        """
        if self.n < 2:
            return 0.0

        if self.decay_factor is not None:
            # For exponential weighted, use accumulated values directly
            var_x = self.M2_x
            var_y = self.M2_y
            cov_xy = self.C_xy
        else:
            var_x = self.M2_x / (self.n - 1)
            var_y = self.M2_y / (self.n - 1)
            cov_xy = self.C_xy / (self.n - 1)

        if var_x <= 0 or var_y <= 0:
            return 0.0

        correlation = cov_xy / math.sqrt(var_x * var_y)

        # Clamp to [-1, 1] due to numerical precision
        return max(-1.0, min(1.0, correlation))

    def variance_x(self) -> float:
        """Get variance of x."""
        if self.n < 2:
            return 0.0
        return self.M2_x / (self.n - 1)

    def variance_y(self) -> float:
        """Get variance of y."""
        if self.n < 2:
            return 0.0
        return self.M2_y / (self.n - 1)

    def covariance(self) -> float:
        """Get covariance of x and y."""
        if self.n < 2:
            return 0.0
        return self.C_xy / (self.n - 1)

    def p_value(self) -> Optional[float]:
        """
        Estimate p-value for correlation using t-distribution approximation.

        Returns:
            Approximate p-value, or None if insufficient data
        """
        if self.n < 4:  # Need at least 4 points for meaningful p-value
            return None

        r = self.correlation()
        if abs(r) >= 1.0:
            return 0.0 if r != 0 else 1.0

        # t-statistic: t = r * sqrt((n-2) / (1-r^2))
        df = self.n - 2
        t_stat = r * math.sqrt(df / (1.0 - r * r))

        # Approximate p-value using normal distribution for large n
        # For small n, this is an approximation
        if self.n > 30:
            # Standard normal approximation
            p = 2.0 * (1.0 - self._normal_cdf(abs(t_stat)))
        else:
            # Rough t-distribution approximation
            p = 2.0 * (1.0 - self._t_cdf(abs(t_stat), df))

        return p

    def confidence(self) -> float:
        """
        Get confidence in correlation estimate.

        Based on sample size and variance.

        Returns:
            Confidence score [0, 1]
        """
        if self.n < 2:
            return 0.0

        # Base confidence on sample size (logarithmic scaling)
        size_confidence = min(1.0, math.log10(self.n + 1) / 3.0)

        # Adjust for correlation strength
        r = abs(self.correlation())
        strength_factor = 0.5 + 0.5 * r

        return size_confidence * strength_factor

    def reset(self) -> None:
        """Reset all statistics."""
        self.n = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.M2_x = 0.0
        self.M2_y = 0.0
        self.C_xy = 0.0

    def merge(self, other: "WelfordCorrelator") -> "WelfordCorrelator":
        """
        Merge two correlators (parallel computation).

        Uses Chan's parallel algorithm for combining statistics.

        Args:
            other: Another WelfordCorrelator

        Returns:
            New merged correlator
        """
        if other.n == 0:
            return self
        if self.n == 0:
            result = WelfordCorrelator(self.decay_factor)
            result.n = other.n
            result.mean_x = other.mean_x
            result.mean_y = other.mean_y
            result.M2_x = other.M2_x
            result.M2_y = other.M2_y
            result.C_xy = other.C_xy
            return result

        result = WelfordCorrelator(self.decay_factor)
        result.n = self.n + other.n

        delta_x = other.mean_x - self.mean_x
        delta_y = other.mean_y - self.mean_y

        result.mean_x = (self.n * self.mean_x + other.n * other.mean_x) / result.n
        result.mean_y = (self.n * self.mean_y + other.n * other.mean_y) / result.n

        result.M2_x = self.M2_x + other.M2_x + delta_x * delta_x * self.n * other.n / result.n
        result.M2_y = self.M2_y + other.M2_y + delta_y * delta_y * self.n * other.n / result.n
        result.C_xy = self.C_xy + other.C_xy + delta_x * delta_y * self.n * other.n / result.n

        return result

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _t_cdf(t: float, df: int) -> float:
        """
        Approximate t-distribution CDF.

        Uses normal approximation with adjustment for degrees of freedom.
        """
        # Approximation: t-distribution approaches normal as df increases
        adjustment = 1.0 + 1.0 / (4.0 * df)
        return WelfordCorrelator._normal_cdf(t / adjustment)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "n": self.n,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "M2_x": self.M2_x,
            "M2_y": self.M2_y,
            "C_xy": self.C_xy,
            "decay_factor": self.decay_factor,
            "correlation": self.correlation(),
            "p_value": self.p_value(),
            "confidence": self.confidence()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WelfordCorrelator":
        """Deserialize from dictionary."""
        correlator = cls(decay_factor=data.get("decay_factor"))
        correlator.n = data["n"]
        correlator.mean_x = data["mean_x"]
        correlator.mean_y = data["mean_y"]
        correlator.M2_x = data["M2_x"]
        correlator.M2_y = data["M2_y"]
        correlator.C_xy = data["C_xy"]
        return correlator


# =============================================================================
# Main Correlation Tracker
# =============================================================================

class CorrelationTracker:
    """
    Tracks cross-layer feature correlations using streaming algorithms.

    Maintains correlation estimates between features in adjacent layers,
    updating incrementally as new activations are observed.

    Usage:
        tracker = CorrelationTracker(n_layers=12, config=CorrelationConfig.default())

        # Update with activations from each layer
        for batch in data:
            layer_activations = model.get_all_layer_activations(batch)
            tracker.update(layer_activations)

        # Get correlations between layers 5 and 6
        corr = tracker.get_layer_correlation(5, 6)
        print(f"Mean correlation: {corr.mean_correlation:.3f}")

        # Get feature propagation path
        path = tracker.trace_feature_propagation("layer_5.sae_42")
    """

    def __init__(
        self,
        n_layers: int,
        config: Optional[CorrelationConfig] = None,
        feature_ids: Optional[Dict[int, List[str]]] = None
    ):
        """
        Initialize correlation tracker.

        Args:
            n_layers: Number of layers in the model
            config: Correlation tracking configuration
            feature_ids: Pre-defined feature IDs per layer
        """
        self.n_layers = n_layers
        self.config = config or CorrelationConfig.default()

        # Initialize feature ID mappings
        self.feature_ids: Dict[int, List[str]] = feature_ids or {}

        # Correlators for each layer pair
        # Key: (source_layer, target_layer, source_feature, target_feature)
        self._correlators: Dict[Tuple[int, int, str, str], WelfordCorrelator] = {}

        # Cache for computed correlations
        self._correlation_cache: Dict[Tuple[int, int], LayerCorrelation] = {}
        self._cache_valid: Dict[Tuple[int, int], bool] = {}

        # Statistics
        self._n_updates = 0
        self._last_snapshot_time = 0.0

        # Active feature tracking (for memory efficiency)
        self._active_features: Dict[int, Set[str]] = {i: set() for i in range(n_layers)}

    def update(
        self,
        layer_activations: Dict[int, Dict[str, float]],
        update_cache: bool = True
    ) -> None:
        """
        Update correlations with new layer activations.

        Args:
            layer_activations: Dict mapping layer_idx -> {feature_id: activation}
            update_cache: Whether to update cached correlations
        """
        self._n_updates += 1

        # Get sorted layer indices
        layers = sorted(layer_activations.keys())

        # Update correlations for adjacent layer pairs
        for i in range(len(layers) - 1):
            source_layer = layers[i]
            target_layer = layers[i + 1]

            source_acts = layer_activations[source_layer]
            target_acts = layer_activations[target_layer]

            self._update_layer_pair(
                source_layer, target_layer,
                source_acts, target_acts
            )

        # Invalidate cache
        if update_cache:
            for key in self._cache_valid:
                self._cache_valid[key] = False

    def _update_layer_pair(
        self,
        source_layer: int,
        target_layer: int,
        source_acts: Dict[str, float],
        target_acts: Dict[str, float]
    ) -> None:
        """Update correlations for a single layer pair."""
        # Track active features
        for feat_id in source_acts:
            self._active_features[source_layer].add(feat_id)
        for feat_id in target_acts:
            self._active_features[target_layer].add(feat_id)

        # Limit features if needed
        source_features = self._limit_features(
            source_acts,
            self.config.max_features_per_layer
        )
        target_features = self._limit_features(
            target_acts,
            self.config.max_features_per_layer
        )

        # Update correlators for each pair
        for source_feat, source_val in source_features.items():
            for target_feat, target_val in target_features.items():
                key = (source_layer, target_layer, source_feat, target_feat)

                if key not in self._correlators:
                    # Check if we've hit the storage limit
                    if len(self._correlators) >= self.config.max_correlations_stored:
                        self._prune_correlators()

                    self._correlators[key] = WelfordCorrelator(
                        decay_factor=self.config.decay_factor if self.config.decay_factor < 1.0 else None
                    )

                self._correlators[key].update(source_val, target_val)

    def _limit_features(
        self,
        activations: Dict[str, float],
        max_features: int
    ) -> Dict[str, float]:
        """Limit features to top-k by activation magnitude."""
        if len(activations) <= max_features:
            return activations

        # Keep top features by absolute activation
        sorted_items = sorted(
            activations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return dict(sorted_items[:max_features])

    def _prune_correlators(self) -> None:
        """Remove low-value correlators to stay within memory limits."""
        if len(self._correlators) < self.config.max_correlations_stored * 0.9:
            return

        # Score correlators by importance
        scored = []
        for key, correlator in self._correlators.items():
            score = correlator.confidence() * abs(correlator.correlation())
            scored.append((key, score))

        # Keep top correlators
        scored.sort(key=lambda x: x[1], reverse=True)
        keep_count = int(self.config.max_correlations_stored * 0.8)

        keys_to_keep = {item[0] for item in scored[:keep_count]}
        self._correlators = {
            k: v for k, v in self._correlators.items()
            if k in keys_to_keep
        }

    def get_layer_correlation(
        self,
        source_layer: int,
        target_layer: int,
        force_recompute: bool = False
    ) -> LayerCorrelation:
        """
        Get correlations between two layers.

        Args:
            source_layer: Source layer index
            target_layer: Target layer index
            force_recompute: Force recomputation even if cached

        Returns:
            LayerCorrelation with all significant correlations
        """
        cache_key = (source_layer, target_layer)

        # Check cache
        if not force_recompute and cache_key in self._correlation_cache:
            if self._cache_valid.get(cache_key, False):
                return self._correlation_cache[cache_key]

        # Compute correlations
        correlations = []

        for key, correlator in self._correlators.items():
            src_layer, tgt_layer, src_feat, tgt_feat = key

            if src_layer != source_layer or tgt_layer != target_layer:
                continue

            if correlator.n < self.config.min_samples:
                continue

            corr_value = correlator.correlation()

            if abs(corr_value) < self.config.correlation_threshold:
                continue

            correlations.append(CorrelationPair(
                source_feature=src_feat,
                target_feature=tgt_feat,
                correlation=corr_value,
                n_samples=correlator.n,
                p_value=correlator.p_value(),
                confidence=correlator.confidence(),
                last_updated=correlator._last_update
            ))

        # Sort by absolute correlation
        correlations.sort(key=lambda c: abs(c.correlation), reverse=True)

        # Compute summary statistics
        if correlations:
            abs_corrs = [abs(c.correlation) for c in correlations]
            mean_corr = sum(abs_corrs) / len(abs_corrs)
            max_corr = max(abs_corrs)
            n_sig = sum(1 for c in correlations if c.is_significant())
        else:
            mean_corr = 0.0
            max_corr = 0.0
            n_sig = 0

        layer_corr = LayerCorrelation(
            source_layer=source_layer,
            target_layer=target_layer,
            correlations=correlations,
            mean_correlation=mean_corr,
            max_correlation=max_corr,
            n_significant=n_sig
        )

        # Update cache
        self._correlation_cache[cache_key] = layer_corr
        self._cache_valid[cache_key] = True

        return layer_corr

    def get_all_layer_correlations(self) -> Dict[Tuple[int, int], LayerCorrelation]:
        """Get correlations for all layer pairs."""
        result = {}
        for i in range(self.n_layers - 1):
            result[(i, i + 1)] = self.get_layer_correlation(i, i + 1)
        return result

    def trace_feature_propagation(
        self,
        feature_id: str,
        min_correlation: float = 0.3
    ) -> List[CorrelationPair]:
        """
        Trace how a feature propagates through layers.

        Args:
            feature_id: Feature ID to trace (e.g., "layer_5.sae_42")
            min_correlation: Minimum correlation to include in trace

        Returns:
            List of correlations showing feature propagation
        """
        trace = []

        # Extract layer from feature ID if present
        parts = feature_id.split(".")
        if len(parts) >= 2 and parts[0].startswith("layer_"):
            try:
                start_layer = int(parts[0].replace("layer_", ""))
            except ValueError:
                start_layer = 0
        else:
            start_layer = 0

        # Forward propagation
        current_feature = feature_id
        for target_layer in range(start_layer + 1, self.n_layers):
            source_layer = target_layer - 1

            best_corr = None
            best_val = 0.0

            for key, correlator in self._correlators.items():
                src_layer, tgt_layer, src_feat, tgt_feat = key

                if src_layer != source_layer or tgt_layer != target_layer:
                    continue

                if src_feat != current_feature:
                    continue

                corr_val = correlator.correlation()
                if abs(corr_val) > abs(best_val) and abs(corr_val) >= min_correlation:
                    best_val = corr_val
                    best_corr = CorrelationPair(
                        source_feature=src_feat,
                        target_feature=tgt_feat,
                        correlation=corr_val,
                        n_samples=correlator.n,
                        confidence=correlator.confidence()
                    )

            if best_corr:
                trace.append(best_corr)
                current_feature = best_corr.target_feature
            else:
                break  # No strong propagation found

        return trace

    def get_feature_correlations(
        self,
        feature_id: str
    ) -> List[CorrelationPair]:
        """Get all correlations involving a specific feature."""
        correlations = []

        for key, correlator in self._correlators.items():
            src_layer, tgt_layer, src_feat, tgt_feat = key

            if src_feat != feature_id and tgt_feat != feature_id:
                continue

            if correlator.n < self.config.min_samples:
                continue

            corr_val = correlator.correlation()

            correlations.append(CorrelationPair(
                source_feature=src_feat,
                target_feature=tgt_feat,
                correlation=corr_val,
                n_samples=correlator.n,
                p_value=correlator.p_value(),
                confidence=correlator.confidence()
            ))

        correlations.sort(key=lambda c: abs(c.correlation), reverse=True)
        return correlations

    def create_snapshot(self) -> CorrelationSnapshot:
        """Create a snapshot of current correlation state."""
        layer_correlations = self.get_all_layer_correlations()

        total_pairs = sum(len(lc.correlations) for lc in layer_correlations.values())
        sig_pairs = sum(lc.n_significant for lc in layer_correlations.values())

        if layer_correlations:
            mean_corrs = [lc.mean_correlation for lc in layer_correlations.values() if lc.correlations]
            global_mean = sum(mean_corrs) / len(mean_corrs) if mean_corrs else 0.0
        else:
            global_mean = 0.0

        snapshot = CorrelationSnapshot(
            timestamp=time.time(),
            layer_correlations=layer_correlations,
            n_total_pairs=total_pairs,
            n_significant_pairs=sig_pairs,
            global_mean_correlation=global_mean
        )

        self._last_snapshot_time = snapshot.timestamp
        return snapshot

    def statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "n_updates": self._n_updates,
            "n_correlators": len(self._correlators),
            "n_layers": self.n_layers,
            "active_features_per_layer": {
                layer: len(features)
                for layer, features in self._active_features.items()
            },
            "config": {
                "method": self.config.method.value,
                "min_samples": self.config.min_samples,
                "correlation_threshold": self.config.correlation_threshold
            }
        }

    def reset(self) -> None:
        """Reset all correlation tracking."""
        self._correlators.clear()
        self._correlation_cache.clear()
        self._cache_valid.clear()
        self._n_updates = 0
        self._active_features = {i: set() for i in range(self.n_layers)}

    def save_state(self) -> Dict[str, Any]:
        """Save tracker state for persistence."""
        return {
            "n_layers": self.n_layers,
            "config": {
                "method": self.config.method.value,
                "min_samples": self.config.min_samples,
                "max_features_per_layer": self.config.max_features_per_layer,
                "max_correlations_stored": self.config.max_correlations_stored,
                "correlation_threshold": self.config.correlation_threshold,
                "decay_factor": self.config.decay_factor,
                "window_size": self.config.window_size
            },
            "correlators": {
                f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v.to_dict()
                for k, v in self._correlators.items()
            },
            "n_updates": self._n_updates
        }

    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> "CorrelationTracker":
        """Load tracker from saved state."""
        config = CorrelationConfig(
            method=CorrelationMethod(state["config"]["method"]),
            min_samples=state["config"]["min_samples"],
            max_features_per_layer=state["config"]["max_features_per_layer"],
            max_correlations_stored=state["config"]["max_correlations_stored"],
            correlation_threshold=state["config"]["correlation_threshold"],
            decay_factor=state["config"]["decay_factor"],
            window_size=state["config"]["window_size"]
        )

        tracker = cls(n_layers=state["n_layers"], config=config)
        tracker._n_updates = state["n_updates"]

        # Restore correlators
        for key_str, corr_data in state.get("correlators", {}).items():
            parts = key_str.split("_", 3)
            if len(parts) >= 4:
                key = (int(parts[0]), int(parts[1]), parts[2], parts[3])
                tracker._correlators[key] = WelfordCorrelator.from_dict(corr_data)

        return tracker


# =============================================================================
# Factory Functions
# =============================================================================

def create_correlation_tracker(
    n_layers: int,
    preset: str = "default"
) -> CorrelationTracker:
    """
    Create a correlation tracker with preset configuration.

    Args:
        n_layers: Number of layers in model
        preset: Configuration preset ("default", "fast", "high_precision")

    Returns:
        Configured CorrelationTracker
    """
    configs = {
        "default": CorrelationConfig.default(),
        "fast": CorrelationConfig.fast(),
        "high_precision": CorrelationConfig.high_precision()
    }

    config = configs.get(preset, CorrelationConfig.default())
    return CorrelationTracker(n_layers=n_layers, config=config)
