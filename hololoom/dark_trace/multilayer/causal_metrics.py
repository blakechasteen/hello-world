"""
Dark Trace Multilayer: Causal Metrics

Implements causal inference metrics for understanding directional
information flow between features across layers.

Key Capabilities:
- Pearl's do-calculus based causal strength estimation
- Transfer entropy for directional information flow
- Intervention effect estimation
- Confounding detection

Research Basis:
- Pearl (2000): Causality: Models, Reasoning, and Inference
- Schreiber (2000): Transfer Entropy
- ACDC (Conmy et al. 2023): Causal circuit discovery

Created: 2025-12-28
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
import time


# =============================================================================
# Data Structures
# =============================================================================

class CausalRelationType(Enum):
    """Types of causal relationships."""
    DIRECT = "direct"           # X directly causes Y
    INDIRECT = "indirect"       # X causes Y through mediators
    CONFOUNDED = "confounded"   # Common cause of both
    NONE = "none"               # No causal relationship


@dataclass
class CausalStrength:
    """Causal strength estimate between two features."""
    source_feature: str
    target_feature: str
    strength: float             # Estimated causal effect [-1, 1]
    confidence: float           # Confidence in estimate [0, 1]
    relation_type: CausalRelationType = CausalRelationType.DIRECT
    n_samples: int = 0
    intervention_effect: Optional[float] = None
    confounding_score: float = 0.0

    def is_significant(self, threshold: float = 0.1) -> bool:
        """Check if causal strength is significant."""
        return abs(self.strength) >= threshold and self.confidence >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_feature": self.source_feature,
            "target_feature": self.target_feature,
            "strength": self.strength,
            "confidence": self.confidence,
            "relation_type": self.relation_type.value,
            "n_samples": self.n_samples,
            "intervention_effect": self.intervention_effect,
            "confounding_score": self.confounding_score
        }


@dataclass
class TransferEntropyResult:
    """Transfer entropy measurement result."""
    source_feature: str
    target_feature: str
    transfer_entropy: float     # TE(source → target) in bits
    reverse_entropy: float      # TE(target → source) in bits
    net_information_flow: float  # TE_forward - TE_reverse
    significance: float         # Statistical significance
    n_samples: int = 0

    @property
    def direction(self) -> str:
        """Dominant direction of information flow."""
        if abs(self.net_information_flow) < 0.01:
            return "bidirectional"
        return "forward" if self.net_information_flow > 0 else "reverse"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_feature": self.source_feature,
            "target_feature": self.target_feature,
            "transfer_entropy": self.transfer_entropy,
            "reverse_entropy": self.reverse_entropy,
            "net_information_flow": self.net_information_flow,
            "direction": self.direction,
            "significance": self.significance,
            "n_samples": self.n_samples
        }


# =============================================================================
# Causal Strength Estimator
# =============================================================================

class CausalStrengthEstimator:
    """
    Estimates causal strength between features using Pearl's framework.

    Implements approximate causal inference using:
    1. Observational correlation as baseline
    2. Adjustment for confounders
    3. Simulated intervention effects

    Note: True causal inference requires interventional data (do-calculus).
    This estimator provides approximations from observational data.

    Usage:
        estimator = CausalStrengthEstimator()

        # Add observations
        for batch in data:
            estimator.observe(
                source_activations=source_acts,
                target_activations=target_acts,
                confounders=confounder_acts  # Optional
            )

        # Get causal strength estimate
        strength = estimator.estimate("source_feat", "target_feat")
        print(f"Causal effect: {strength.strength:.3f}")
    """

    def __init__(
        self,
        adjustment_method: str = "regression",
        min_samples: int = 50
    ):
        """
        Initialize causal strength estimator.

        Args:
            adjustment_method: Method for confounder adjustment
                              ("regression", "propensity", "matching")
            min_samples: Minimum samples for estimation
        """
        self.adjustment_method = adjustment_method
        self.min_samples = min_samples

        # Storage for observations
        self._observations: List[Dict[str, Dict[str, float]]] = []
        self._source_stats: Dict[str, Dict[str, float]] = {}
        self._target_stats: Dict[str, Dict[str, float]] = {}
        self._joint_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    def observe(
        self,
        source_activations: Dict[str, float],
        target_activations: Dict[str, float],
        confounders: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add an observation for causal estimation.

        Args:
            source_activations: {feature_id: activation} for source layer
            target_activations: {feature_id: activation} for target layer
            confounders: Optional confounder activations
        """
        observation = {
            "source": source_activations,
            "target": target_activations,
            "confounders": confounders or {}
        }
        self._observations.append(observation)

        # Update running statistics
        self._update_stats(source_activations, target_activations)

    def _update_stats(
        self,
        source_acts: Dict[str, float],
        target_acts: Dict[str, float]
    ) -> None:
        """Update running statistics for features."""
        n = len(self._observations)

        # Update source stats
        for feat, val in source_acts.items():
            if feat not in self._source_stats:
                self._source_stats[feat] = {"sum": 0.0, "sum_sq": 0.0, "n": 0}
            stats = self._source_stats[feat]
            stats["sum"] += val
            stats["sum_sq"] += val * val
            stats["n"] += 1

        # Update target stats
        for feat, val in target_acts.items():
            if feat not in self._target_stats:
                self._target_stats[feat] = {"sum": 0.0, "sum_sq": 0.0, "n": 0}
            stats = self._target_stats[feat]
            stats["sum"] += val
            stats["sum_sq"] += val * val
            stats["n"] += 1

        # Update joint stats
        for src_feat, src_val in source_acts.items():
            for tgt_feat, tgt_val in target_acts.items():
                key = (src_feat, tgt_feat)
                if key not in self._joint_stats:
                    self._joint_stats[key] = {"sum_xy": 0.0, "n": 0}
                stats = self._joint_stats[key]
                stats["sum_xy"] += src_val * tgt_val
                stats["n"] += 1

    def estimate(
        self,
        source_feature: str,
        target_feature: str,
        adjust_confounders: bool = True
    ) -> CausalStrength:
        """
        Estimate causal strength between two features.

        Args:
            source_feature: Source feature ID
            target_feature: Target feature ID
            adjust_confounders: Whether to adjust for confounders

        Returns:
            CausalStrength estimate
        """
        n = len(self._observations)

        if n < self.min_samples:
            return CausalStrength(
                source_feature=source_feature,
                target_feature=target_feature,
                strength=0.0,
                confidence=0.0,
                relation_type=CausalRelationType.NONE,
                n_samples=n
            )

        # Compute baseline correlation
        correlation = self._compute_correlation(source_feature, target_feature)

        # Adjust for confounders if requested
        if adjust_confounders:
            adjusted_effect, confounding_score = self._adjust_for_confounders(
                source_feature, target_feature, correlation
            )
        else:
            adjusted_effect = correlation
            confounding_score = 0.0

        # Estimate intervention effect (approximate)
        intervention_effect = self._estimate_intervention_effect(
            source_feature, target_feature
        )

        # Determine relationship type
        if abs(correlation) < 0.1:
            relation_type = CausalRelationType.NONE
        elif confounding_score > 0.5:
            relation_type = CausalRelationType.CONFOUNDED
        elif abs(correlation - adjusted_effect) > 0.2:
            relation_type = CausalRelationType.INDIRECT
        else:
            relation_type = CausalRelationType.DIRECT

        # Compute confidence
        confidence = self._compute_confidence(n, correlation, confounding_score)

        return CausalStrength(
            source_feature=source_feature,
            target_feature=target_feature,
            strength=adjusted_effect,
            confidence=confidence,
            relation_type=relation_type,
            n_samples=n,
            intervention_effect=intervention_effect,
            confounding_score=confounding_score
        )

    def _compute_correlation(
        self,
        source_feature: str,
        target_feature: str
    ) -> float:
        """Compute Pearson correlation from running statistics."""
        key = (source_feature, target_feature)

        if key not in self._joint_stats:
            return 0.0

        joint = self._joint_stats[key]
        src_stats = self._source_stats.get(source_feature, {})
        tgt_stats = self._target_stats.get(target_feature, {})

        n = joint["n"]
        if n < 2:
            return 0.0

        # Compute correlation
        mean_x = src_stats.get("sum", 0.0) / n
        mean_y = tgt_stats.get("sum", 0.0) / n
        mean_xy = joint["sum_xy"] / n

        var_x = src_stats.get("sum_sq", 0.0) / n - mean_x * mean_x
        var_y = tgt_stats.get("sum_sq", 0.0) / n - mean_y * mean_y

        if var_x <= 0 or var_y <= 0:
            return 0.0

        cov_xy = mean_xy - mean_x * mean_y
        correlation = cov_xy / math.sqrt(var_x * var_y)

        return max(-1.0, min(1.0, correlation))

    def _adjust_for_confounders(
        self,
        source_feature: str,
        target_feature: str,
        raw_correlation: float
    ) -> Tuple[float, float]:
        """
        Adjust correlation for potential confounders.

        Uses a simple regression-based adjustment.

        Returns:
            (adjusted_effect, confounding_score)
        """
        # Find potential confounders (features correlated with both)
        confounders = []

        for obs in self._observations[-min(100, len(self._observations)):]:
            conf = obs.get("confounders", {})
            for conf_feat, conf_val in conf.items():
                if conf_feat not in [f[0] for f in confounders]:
                    # Check correlation with source and target
                    src_corr = self._quick_correlation(
                        conf_feat, source_feature, obs
                    )
                    tgt_corr = self._quick_correlation(
                        conf_feat, target_feature, obs
                    )

                    if abs(src_corr) > 0.3 and abs(tgt_corr) > 0.3:
                        confounders.append((conf_feat, src_corr, tgt_corr))

        if not confounders:
            return raw_correlation, 0.0

        # Simple confounding adjustment: partial correlation
        # Approximate partial correlation by subtracting confounding effect
        confounding_effect = 0.0
        for conf_feat, src_corr, tgt_corr in confounders:
            confounding_effect += src_corr * tgt_corr

        confounding_effect /= len(confounders)
        adjusted_effect = raw_correlation - confounding_effect

        # Confounding score: how much the correlation changed
        if abs(raw_correlation) > 0.01:
            confounding_score = abs(raw_correlation - adjusted_effect) / abs(raw_correlation)
        else:
            confounding_score = 0.0

        return adjusted_effect, min(1.0, confounding_score)

    def _quick_correlation(
        self,
        feat1: str,
        feat2: str,
        obs: Dict
    ) -> float:
        """Quick correlation estimate from a single observation."""
        # This is a placeholder - real implementation would use
        # accumulated statistics
        return 0.0

    def _estimate_intervention_effect(
        self,
        source_feature: str,
        target_feature: str
    ) -> float:
        """
        Estimate intervention effect P(Y | do(X)).

        Uses natural experiments when source has high/low values.
        This is an approximation of true interventional effects.
        """
        if len(self._observations) < self.min_samples:
            return 0.0

        # Find observations with extreme source values
        source_vals = []
        target_vals_high = []
        target_vals_low = []

        for obs in self._observations:
            src_val = obs["source"].get(source_feature, 0.0)
            tgt_val = obs["target"].get(target_feature, 0.0)
            source_vals.append(src_val)

        if not source_vals:
            return 0.0

        # Define high/low thresholds
        sorted_vals = sorted(source_vals)
        n = len(sorted_vals)
        low_threshold = sorted_vals[n // 4]
        high_threshold = sorted_vals[3 * n // 4]

        # Collect target values for high/low source
        for obs in self._observations:
            src_val = obs["source"].get(source_feature, 0.0)
            tgt_val = obs["target"].get(target_feature, 0.0)

            if src_val <= low_threshold:
                target_vals_low.append(tgt_val)
            elif src_val >= high_threshold:
                target_vals_high.append(tgt_val)

        if not target_vals_high or not target_vals_low:
            return 0.0

        # Intervention effect = difference in target means
        mean_high = sum(target_vals_high) / len(target_vals_high)
        mean_low = sum(target_vals_low) / len(target_vals_low)

        return mean_high - mean_low

    def _compute_confidence(
        self,
        n_samples: int,
        correlation: float,
        confounding_score: float
    ) -> float:
        """Compute confidence in causal estimate."""
        # Base confidence on sample size
        sample_confidence = min(1.0, n_samples / (2 * self.min_samples))

        # Adjust for correlation strength
        strength_factor = min(1.0, abs(correlation) + 0.3)

        # Penalize for confounding
        confounding_penalty = 1.0 - 0.5 * confounding_score

        return sample_confidence * strength_factor * confounding_penalty

    def batch_estimate(
        self,
        feature_pairs: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], CausalStrength]:
        """Estimate causal strength for multiple feature pairs."""
        results = {}
        for source, target in feature_pairs:
            results[(source, target)] = self.estimate(source, target)
        return results

    def clear(self) -> None:
        """Clear all observations."""
        self._observations.clear()
        self._source_stats.clear()
        self._target_stats.clear()
        self._joint_stats.clear()


# =============================================================================
# Transfer Entropy Estimator
# =============================================================================

class TransferEntropyEstimator:
    """
    Estimates transfer entropy for directional information flow.

    Transfer Entropy TE(X→Y) quantifies the reduction in uncertainty
    of Y's future given the past of X, beyond what Y's own past provides.

    TE(X→Y) = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-1)

    Higher TE(X→Y) indicates X provides information about Y's future
    that Y's own past doesn't capture.

    Usage:
        estimator = TransferEntropyEstimator()

        # Add time series observations
        for t in range(len(data)):
            estimator.observe(
                source_val=source_series[t],
                target_val=target_series[t]
            )

        # Get transfer entropy
        result = estimator.estimate("source", "target")
        print(f"TE(source→target): {result.transfer_entropy:.3f} bits")
        print(f"Direction: {result.direction}")
    """

    def __init__(
        self,
        n_bins: int = 10,
        lag: int = 1,
        min_samples: int = 100
    ):
        """
        Initialize transfer entropy estimator.

        Args:
            n_bins: Number of bins for discretization
            lag: Time lag for transfer entropy
            min_samples: Minimum samples for estimation
        """
        self.n_bins = n_bins
        self.lag = lag
        self.min_samples = min_samples

        # Time series storage
        self._series: Dict[str, List[float]] = {}

    def observe(
        self,
        feature_id: str,
        value: float
    ) -> None:
        """
        Add a time series observation.

        Args:
            feature_id: Feature identifier
            value: Observed value at current time
        """
        if feature_id not in self._series:
            self._series[feature_id] = []
        self._series[feature_id].append(value)

    def observe_batch(
        self,
        observations: Dict[str, float]
    ) -> None:
        """Add observations for multiple features at once."""
        for feat_id, value in observations.items():
            self.observe(feat_id, value)

    def estimate(
        self,
        source_feature: str,
        target_feature: str
    ) -> TransferEntropyResult:
        """
        Estimate transfer entropy between two features.

        Args:
            source_feature: Source feature ID
            target_feature: Target feature ID

        Returns:
            TransferEntropyResult with bidirectional TE
        """
        source_series = self._series.get(source_feature, [])
        target_series = self._series.get(target_feature, [])

        n = min(len(source_series), len(target_series))

        if n < self.min_samples:
            return TransferEntropyResult(
                source_feature=source_feature,
                target_feature=target_feature,
                transfer_entropy=0.0,
                reverse_entropy=0.0,
                net_information_flow=0.0,
                significance=0.0,
                n_samples=n
            )

        # Align and truncate series
        source = source_series[:n]
        target = target_series[:n]

        # Discretize values
        source_disc = self._discretize(source)
        target_disc = self._discretize(target)

        # Compute transfer entropy in both directions
        te_forward = self._compute_transfer_entropy(source_disc, target_disc)
        te_reverse = self._compute_transfer_entropy(target_disc, source_disc)

        # Compute significance via shuffling
        significance = self._compute_significance(
            source_disc, target_disc, te_forward
        )

        return TransferEntropyResult(
            source_feature=source_feature,
            target_feature=target_feature,
            transfer_entropy=te_forward,
            reverse_entropy=te_reverse,
            net_information_flow=te_forward - te_reverse,
            significance=significance,
            n_samples=n
        )

    def _discretize(self, series: List[float]) -> List[int]:
        """Discretize continuous values into bins."""
        if not series:
            return []

        min_val = min(series)
        max_val = max(series)

        if max_val == min_val:
            return [0] * len(series)

        bin_width = (max_val - min_val) / self.n_bins

        discretized = []
        for val in series:
            bin_idx = int((val - min_val) / bin_width)
            bin_idx = min(bin_idx, self.n_bins - 1)
            discretized.append(bin_idx)

        return discretized

    def _compute_transfer_entropy(
        self,
        source: List[int],
        target: List[int]
    ) -> float:
        """
        Compute transfer entropy TE(source → target).

        TE(X→Y) = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-1)
        """
        n = len(source)
        lag = self.lag

        if n < lag + 1:
            return 0.0

        # Count joint and marginal frequencies
        joint_counts: Dict[Tuple[int, int, int], int] = {}   # (y_past, x_past, y_now)
        target_cond_counts: Dict[Tuple[int, int], int] = {}  # (y_past, y_now)
        source_cond_counts: Dict[Tuple[int, int, int], int] = {}  # (y_past, x_past, y_now) marginal

        for t in range(lag, n):
            y_past = target[t - lag]
            x_past = source[t - lag]
            y_now = target[t]

            key_joint = (y_past, x_past, y_now)
            key_target = (y_past, y_now)

            joint_counts[key_joint] = joint_counts.get(key_joint, 0) + 1
            target_cond_counts[key_target] = target_cond_counts.get(key_target, 0) + 1

        total = n - lag

        # Compute conditional entropies
        # H(Y_t | Y_t-1)
        h_y_given_ypast = 0.0
        ypast_counts: Dict[int, int] = {}
        for (y_past, y_now), count in target_cond_counts.items():
            ypast_counts[y_past] = ypast_counts.get(y_past, 0) + count

        for (y_past, y_now), count in target_cond_counts.items():
            p_joint = count / total
            p_ypast = ypast_counts[y_past] / total
            if p_joint > 0 and p_ypast > 0:
                p_cond = p_joint / p_ypast
                h_y_given_ypast -= p_joint * math.log2(p_cond + 1e-10)

        # H(Y_t | Y_t-1, X_t-1)
        h_y_given_yxpast = 0.0
        yxpast_counts: Dict[Tuple[int, int], int] = {}
        for (y_past, x_past, y_now), count in joint_counts.items():
            key = (y_past, x_past)
            yxpast_counts[key] = yxpast_counts.get(key, 0) + count

        for (y_past, x_past, y_now), count in joint_counts.items():
            p_joint = count / total
            p_yxpast = yxpast_counts[(y_past, x_past)] / total
            if p_joint > 0 and p_yxpast > 0:
                p_cond = p_joint / p_yxpast
                h_y_given_yxpast -= p_joint * math.log2(p_cond + 1e-10)

        # Transfer entropy
        te = max(0.0, h_y_given_ypast - h_y_given_yxpast)

        return te

    def _compute_significance(
        self,
        source: List[int],
        target: List[int],
        observed_te: float,
        n_shuffles: int = 100
    ) -> float:
        """
        Compute significance via permutation test.

        Returns proportion of shuffled TEs less than observed.
        """
        import random

        n_lower = 0

        for _ in range(n_shuffles):
            # Shuffle source series
            shuffled_source = source.copy()
            random.shuffle(shuffled_source)

            # Compute TE with shuffled source
            shuffled_te = self._compute_transfer_entropy(shuffled_source, target)

            if shuffled_te < observed_te:
                n_lower += 1

        return n_lower / n_shuffles

    def get_information_flow_matrix(
        self,
        feature_ids: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise net information flow for multiple features.

        Returns matrix of net TE (positive = forward flow).
        """
        matrix = {}

        for i, src in enumerate(feature_ids):
            for j, tgt in enumerate(feature_ids):
                if i != j:
                    result = self.estimate(src, tgt)
                    matrix[(src, tgt)] = result.net_information_flow

        return matrix

    def clear(self) -> None:
        """Clear all time series data."""
        self._series.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def create_causal_estimator(
    method: str = "regression",
    min_samples: int = 50
) -> CausalStrengthEstimator:
    """Create a causal strength estimator."""
    return CausalStrengthEstimator(
        adjustment_method=method,
        min_samples=min_samples
    )


def create_transfer_entropy_estimator(
    n_bins: int = 10,
    lag: int = 1,
    min_samples: int = 100
) -> TransferEntropyEstimator:
    """Create a transfer entropy estimator."""
    return TransferEntropyEstimator(
        n_bins=n_bins,
        lag=lag,
        min_samples=min_samples
    )
