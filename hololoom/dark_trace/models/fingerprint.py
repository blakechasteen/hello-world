"""
Cross-Model Feature Fingerprinting

This module enables comparison of learned features across different models,
identifying universal features (shared across architectures) and model-specific
features (unique to particular models).

Key Components:
- FeatureFingerprint: Representation of a feature's activation pattern
- ModelFingerprinter: Extracts fingerprints from any model adapter
- FingerprintComparison: Results of comparing two fingerprints
- Universal/Model-Specific feature discovery

Use Cases:
- Compare SAE features across different model sizes
- Identify universal concepts (e.g., "code", "safety") vs architecture artifacts
- Transfer interpretability insights between models
- Validate feature consistency across fine-tuning

Design Notes:
- Fingerprints are model-agnostic representations
- Comparison uses multiple similarity metrics (cosine, correlation, JS divergence)
- Threshold-based classification of universal vs specific features
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hololoom.dark_trace.models.adapter import ModelAdapter


class FingerprintMethod(Enum):
    """Methods for computing feature fingerprints."""
    MEAN_ACTIVATION = "mean_activation"  # Average activation pattern
    MAX_ACTIVATION = "max_activation"    # Maximum activation pattern
    ACTIVATION_HISTOGRAM = "histogram"   # Distribution of activations
    COVARIANCE = "covariance"           # Feature covariance structure
    GRADIENT = "gradient"               # Gradient-based fingerprint


class SimilarityMetric(Enum):
    """Metrics for comparing fingerprints."""
    COSINE = "cosine"
    CORRELATION = "correlation"
    JS_DIVERGENCE = "js_divergence"
    EUCLIDEAN = "euclidean"
    MUTUAL_INFO = "mutual_info"


@dataclass
class FingerprintConfig:
    """Configuration for fingerprint extraction."""

    method: FingerprintMethod = FingerprintMethod.MEAN_ACTIVATION
    n_samples: int = 1000  # Number of samples for fingerprinting
    normalize: bool = True  # Normalize fingerprints
    include_statistics: bool = True  # Include mean, std, etc.
    histogram_bins: int = 50  # Bins for histogram method

    # Comparison settings
    similarity_metrics: list[SimilarityMetric] = field(
        default_factory=lambda: [
            SimilarityMetric.COSINE,
            SimilarityMetric.CORRELATION,
        ]
    )

    # Universal feature thresholds
    universal_threshold: float = 0.8  # Min similarity for universal
    specific_threshold: float = 0.3   # Max similarity for model-specific


@dataclass
class FeatureFingerprint:
    """
    Fingerprint representation of a feature.

    Contains activation statistics and patterns that characterize
    how a feature responds across diverse inputs.
    """

    feature_id: str
    model_id: str
    layer: str

    # Core fingerprint data
    pattern: np.ndarray  # Main fingerprint vector
    method: FingerprintMethod = FingerprintMethod.MEAN_ACTIVATION

    # Statistics (if computed)
    mean: float | None = None
    std: float | None = None
    sparsity: float | None = None  # Fraction of zero activations
    max_activation: float | None = None
    activation_histogram: np.ndarray | None = None

    # Metadata
    n_samples: int = 0
    label: str | None = None  # Human-readable label if available
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure pattern is numpy array."""
        if not isinstance(self.pattern, np.ndarray):
            self.pattern = np.array(self.pattern, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Dimensionality of fingerprint."""
        return self.pattern.shape[0] if len(self.pattern.shape) > 0 else 0

    def normalize(self) -> "FeatureFingerprint":
        """Return normalized fingerprint."""
        norm = np.linalg.norm(self.pattern)
        if norm > 1e-8:
            normalized_pattern = self.pattern / norm
        else:
            normalized_pattern = self.pattern

        return FeatureFingerprint(
            feature_id=self.feature_id,
            model_id=self.model_id,
            layer=self.layer,
            pattern=normalized_pattern,
            method=self.method,
            mean=self.mean,
            std=self.std,
            sparsity=self.sparsity,
            max_activation=self.max_activation,
            activation_histogram=self.activation_histogram,
            n_samples=self.n_samples,
            label=self.label,
            metadata={**self.metadata, "normalized": True},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "feature_id": self.feature_id,
            "model_id": self.model_id,
            "layer": self.layer,
            "pattern": self.pattern.tolist(),
            "method": self.method.value,
            "mean": self.mean,
            "std": self.std,
            "sparsity": self.sparsity,
            "max_activation": self.max_activation,
            "n_samples": self.n_samples,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureFingerprint":
        """Deserialize from dictionary."""
        return cls(
            feature_id=data["feature_id"],
            model_id=data["model_id"],
            layer=data["layer"],
            pattern=np.array(data["pattern"], dtype=np.float32),
            method=FingerprintMethod(data["method"]),
            mean=data.get("mean"),
            std=data.get("std"),
            sparsity=data.get("sparsity"),
            max_activation=data.get("max_activation"),
            n_samples=data.get("n_samples", 0),
            label=data.get("label"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FingerprintComparison:
    """Results of comparing two fingerprints."""

    fingerprint_a: FeatureFingerprint
    fingerprint_b: FeatureFingerprint

    # Similarity scores by metric
    similarities: dict[SimilarityMetric, float] = field(default_factory=dict)

    # Classification
    is_universal: bool = False
    is_model_specific: bool = False
    confidence: float = 0.0

    # Additional analysis
    alignment_score: float | None = None  # How well aligned are patterns
    shared_variance: float | None = None  # Common variance explained

    @property
    def primary_similarity(self) -> float:
        """Get primary similarity score (cosine or first available)."""
        if SimilarityMetric.COSINE in self.similarities:
            return self.similarities[SimilarityMetric.COSINE]
        elif self.similarities:
            return next(iter(self.similarities.values()))
        return 0.0

    @property
    def classification(self) -> str:
        """Return classification as string."""
        if self.is_universal:
            return "universal"
        elif self.is_model_specific:
            return "model_specific"
        else:
            return "intermediate"

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Fingerprint Comparison: {self.fingerprint_a.feature_id} vs {self.fingerprint_b.feature_id}",
            f"  Models: {self.fingerprint_a.model_id} vs {self.fingerprint_b.model_id}",
            f"  Classification: {self.classification} (confidence: {self.confidence:.2f})",
            "  Similarities:",
        ]
        for metric, score in self.similarities.items():
            lines.append(f"    {metric.value}: {score:.4f}")
        return "\n".join(lines)


class ModelFingerprinter:
    """
    Extracts feature fingerprints from model adapters.

    Supports multiple fingerprinting methods and can process
    entire models or specific layers.
    """

    def __init__(
        self,
        adapter: "ModelAdapter",
        model_id: str,
        config: FingerprintConfig | None = None,
    ):
        """
        Initialize fingerprinter.

        Args:
            adapter: Model adapter for activation extraction
            model_id: Unique identifier for the model
            config: Fingerprinting configuration
        """
        self.adapter = adapter
        self.model_id = model_id
        self.config = config or FingerprintConfig()

        # Cache of computed fingerprints
        self._fingerprint_cache: dict[str, FeatureFingerprint] = {}

    def fingerprint_layer(
        self,
        layer: str,
        inputs: list[Any],
        feature_indices: list[int] | None = None,
    ) -> dict[str, FeatureFingerprint]:
        """
        Compute fingerprints for features in a layer.

        Args:
            layer: Layer name to fingerprint
            inputs: List of inputs to process
            feature_indices: Specific feature indices (None = all)

        Returns:
            Dict mapping feature_id to fingerprint
        """
        # Collect activations across inputs
        all_activations = []
        for inp in inputs[:self.config.n_samples]:
            try:
                acts = self.adapter.get_activations(inp, layer)
                all_activations.append(acts)
            except Exception:
                continue

        if not all_activations:
            return {}

        # Stack activations: (n_samples, ...)
        stacked = np.stack(all_activations, axis=0)

        # Reshape to (n_samples, n_features) if needed
        if len(stacked.shape) > 2:
            # Flatten spatial dimensions
            n_samples = stacked.shape[0]
            stacked = stacked.reshape(n_samples, -1)

        n_features = stacked.shape[-1]

        # Determine which features to fingerprint
        if feature_indices is None:
            feature_indices = list(range(n_features))

        fingerprints = {}
        for idx in feature_indices:
            if idx >= n_features:
                continue

            # Extract feature activations
            feature_acts = stacked[:, idx]

            # Compute fingerprint based on method
            fingerprint = self._compute_fingerprint(
                feature_acts,
                feature_id=f"{layer}.{idx}",
                layer=layer,
            )

            fingerprints[fingerprint.feature_id] = fingerprint

        return fingerprints

    def fingerprint_model(
        self,
        inputs: list[Any],
        layers: list[str] | None = None,
    ) -> dict[str, FeatureFingerprint]:
        """
        Compute fingerprints for entire model.

        Args:
            inputs: List of inputs to process
            layers: Specific layers (None = all hookable layers)

        Returns:
            Dict mapping feature_id to fingerprint
        """
        if layers is None:
            layers = self.adapter.get_layer_names()

        all_fingerprints = {}
        for layer in layers:
            layer_fps = self.fingerprint_layer(layer, inputs)
            all_fingerprints.update(layer_fps)

        return all_fingerprints

    def _compute_fingerprint(
        self,
        activations: np.ndarray,
        feature_id: str,
        layer: str,
    ) -> FeatureFingerprint:
        """
        Compute fingerprint from activations using configured method.

        Args:
            activations: Feature activations (n_samples,) or (n_samples, dim)
            feature_id: Feature identifier
            layer: Layer name

        Returns:
            Computed fingerprint
        """
        method = self.config.method

        if method == FingerprintMethod.MEAN_ACTIVATION:
            pattern = np.mean(activations, axis=0)
            if np.isscalar(pattern):
                pattern = np.array([pattern])

        elif method == FingerprintMethod.MAX_ACTIVATION:
            pattern = np.max(activations, axis=0)
            if np.isscalar(pattern):
                pattern = np.array([pattern])

        elif method == FingerprintMethod.ACTIVATION_HISTOGRAM:
            hist, _ = np.histogram(
                activations.flatten(),
                bins=self.config.histogram_bins,
                density=True,
            )
            pattern = hist.astype(np.float32)

        elif method == FingerprintMethod.COVARIANCE:
            if len(activations.shape) == 1:
                pattern = np.array([np.var(activations)])
            else:
                pattern = np.cov(activations, rowvar=False).flatten()

        elif method == FingerprintMethod.GRADIENT:
            # Use gradient of activations as fingerprint
            pattern = np.gradient(activations.mean(axis=0))
            if np.isscalar(pattern):
                pattern = np.array([pattern])

        else:
            pattern = np.mean(activations, axis=0)
            if np.isscalar(pattern):
                pattern = np.array([pattern])

        # Compute statistics if requested
        mean = float(np.mean(activations)) if self.config.include_statistics else None
        std = float(np.std(activations)) if self.config.include_statistics else None
        sparsity = float(np.mean(activations == 0)) if self.config.include_statistics else None
        max_act = float(np.max(activations)) if self.config.include_statistics else None

        # Compute histogram for all methods
        hist = None
        if self.config.include_statistics:
            hist, _ = np.histogram(
                activations.flatten(),
                bins=self.config.histogram_bins,
                density=True,
            )

        fingerprint = FeatureFingerprint(
            feature_id=feature_id,
            model_id=self.model_id,
            layer=layer,
            pattern=pattern.astype(np.float32),
            method=method,
            mean=mean,
            std=std,
            sparsity=sparsity,
            max_activation=max_act,
            activation_histogram=hist,
            n_samples=len(activations),
        )

        if self.config.normalize:
            fingerprint = fingerprint.normalize()

        # Cache fingerprint
        self._fingerprint_cache[feature_id] = fingerprint

        return fingerprint

    def get_cached_fingerprint(self, feature_id: str) -> FeatureFingerprint | None:
        """Get cached fingerprint if available."""
        return self._fingerprint_cache.get(feature_id)

    def clear_cache(self) -> None:
        """Clear fingerprint cache."""
        self._fingerprint_cache.clear()


def compute_similarity(
    fp_a: FeatureFingerprint,
    fp_b: FeatureFingerprint,
    metric: SimilarityMetric,
) -> float:
    """
    Compute similarity between two fingerprints.

    Args:
        fp_a: First fingerprint
        fp_b: Second fingerprint
        metric: Similarity metric to use

    Returns:
        Similarity score (typically 0-1, higher = more similar)
    """
    # Ensure patterns are same size
    pattern_a = fp_a.pattern.flatten()
    pattern_b = fp_b.pattern.flatten()

    # Handle size mismatch
    if pattern_a.shape != pattern_b.shape:
        # Truncate to shorter
        min_len = min(len(pattern_a), len(pattern_b))
        pattern_a = pattern_a[:min_len]
        pattern_b = pattern_b[:min_len]

    if metric == SimilarityMetric.COSINE:
        norm_a = np.linalg.norm(pattern_a)
        norm_b = np.linalg.norm(pattern_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(pattern_a, pattern_b) / (norm_a * norm_b))

    elif metric == SimilarityMetric.CORRELATION:
        if np.std(pattern_a) < 1e-8 or np.std(pattern_b) < 1e-8:
            return 0.0
        corr = np.corrcoef(pattern_a, pattern_b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    elif metric == SimilarityMetric.EUCLIDEAN:
        # Convert to similarity (inverse of distance)
        dist = np.linalg.norm(pattern_a - pattern_b)
        return float(1.0 / (1.0 + dist))

    elif metric == SimilarityMetric.JS_DIVERGENCE:
        # Jensen-Shannon divergence (convert to similarity)
        # Ensure non-negative (treat as distributions)
        p = np.abs(pattern_a) + 1e-10
        q = np.abs(pattern_b) + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
        return float(1.0 - min(js, 1.0))  # Convert to similarity

    elif metric == SimilarityMetric.MUTUAL_INFO:
        # Simplified mutual information estimate
        # Use histogram-based estimation
        n_bins = 20
        hist_a, _ = np.histogram(pattern_a, bins=n_bins, density=True)
        hist_b, _ = np.histogram(pattern_b, bins=n_bins, density=True)

        # Normalize
        hist_a = hist_a / (hist_a.sum() + 1e-10)
        hist_b = hist_b / (hist_b.sum() + 1e-10)

        # Joint histogram
        joint, _, _ = np.histogram2d(pattern_a, pattern_b, bins=n_bins, density=True)
        joint = joint / (joint.sum() + 1e-10)

        # Compute MI
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 1e-10 and hist_a[i] > 1e-10 and hist_b[j] > 1e-10:
                    mi += joint[i, j] * np.log(joint[i, j] / (hist_a[i] * hist_b[j]))

        # Normalize MI to [0, 1]
        max_mi = np.log(n_bins)
        return float(min(mi / max_mi, 1.0)) if max_mi > 0 else 0.0

    return 0.0


def compare_fingerprints(
    fp_a: FeatureFingerprint,
    fp_b: FeatureFingerprint,
    config: FingerprintConfig | None = None,
) -> FingerprintComparison:
    """
    Compare two fingerprints and classify relationship.

    Args:
        fp_a: First fingerprint
        fp_b: Second fingerprint
        config: Configuration for comparison

    Returns:
        Comparison result with similarity scores and classification
    """
    config = config or FingerprintConfig()

    # Compute similarities for all requested metrics
    similarities = {}
    for metric in config.similarity_metrics:
        similarities[metric] = compute_similarity(fp_a, fp_b, metric)

    # Compute average similarity for classification
    avg_similarity = np.mean(list(similarities.values())) if similarities else 0.0

    # Classify relationship
    is_universal = avg_similarity >= config.universal_threshold
    is_model_specific = avg_similarity <= config.specific_threshold

    # Compute confidence based on how clearly it falls into a category
    if is_universal:
        confidence = min(1.0, (avg_similarity - config.universal_threshold) /
                        (1.0 - config.universal_threshold) + 0.5)
    elif is_model_specific:
        confidence = min(1.0, (config.specific_threshold - avg_similarity) /
                        config.specific_threshold + 0.5)
    else:
        # Intermediate zone - lower confidence
        mid = (config.universal_threshold + config.specific_threshold) / 2
        distance_from_mid = abs(avg_similarity - mid)
        confidence = 0.5 - distance_from_mid

    return FingerprintComparison(
        fingerprint_a=fp_a,
        fingerprint_b=fp_b,
        similarities=similarities,
        is_universal=is_universal,
        is_model_specific=is_model_specific,
        confidence=float(confidence),
        alignment_score=similarities.get(SimilarityMetric.COSINE),
        shared_variance=similarities.get(SimilarityMetric.CORRELATION),
    )


def find_universal_features(
    fingerprints_by_model: dict[str, dict[str, FeatureFingerprint]],
    config: FingerprintConfig | None = None,
) -> list[tuple[str, list[FeatureFingerprint], float]]:
    """
    Find features that appear universally across models.

    Args:
        fingerprints_by_model: Dict mapping model_id to feature fingerprints
        config: Fingerprinting configuration

    Returns:
        List of (feature_concept, matching_fingerprints, avg_similarity)
    """
    config = config or FingerprintConfig()

    if len(fingerprints_by_model) < 2:
        return []

    model_ids = list(fingerprints_by_model.keys())
    first_model = model_ids[0]
    first_fingerprints = fingerprints_by_model[first_model]

    universal_features = []

    for feature_id, fp in first_fingerprints.items():
        # Find best matches in other models
        matches = [fp]
        similarities = []

        for other_model in model_ids[1:]:
            other_fps = fingerprints_by_model[other_model]

            # Find best matching feature
            best_match = None
            best_similarity = -1.0

            for other_fp in other_fps.values():
                comparison = compare_fingerprints(fp, other_fp, config)
                if comparison.primary_similarity > best_similarity:
                    best_similarity = comparison.primary_similarity
                    best_match = other_fp

            if best_match and best_similarity >= config.universal_threshold:
                matches.append(best_match)
                similarities.append(best_similarity)

        # Feature is universal if it matches across all models
        if len(matches) == len(model_ids):
            avg_sim = float(np.mean(similarities))
            universal_features.append((feature_id, matches, avg_sim))

    # Sort by average similarity
    universal_features.sort(key=lambda x: x[2], reverse=True)

    return universal_features


def find_model_specific_features(
    fingerprints_by_model: dict[str, dict[str, FeatureFingerprint]],
    config: FingerprintConfig | None = None,
) -> dict[str, list[tuple[str, FeatureFingerprint, float]]]:
    """
    Find features unique to each model.

    Args:
        fingerprints_by_model: Dict mapping model_id to feature fingerprints
        config: Fingerprinting configuration

    Returns:
        Dict mapping model_id to list of (feature_id, fingerprint, specificity_score)
    """
    config = config or FingerprintConfig()

    model_specific = defaultdict(list)

    for model_id, fingerprints in fingerprints_by_model.items():
        other_models = [m for m in fingerprints_by_model.keys() if m != model_id]

        for feature_id, fp in fingerprints.items():
            # Check similarity with all other models
            max_similarity_to_others = 0.0

            for other_model in other_models:
                other_fps = fingerprints_by_model[other_model]

                for other_fp in other_fps.values():
                    comparison = compare_fingerprints(fp, other_fp, config)
                    max_similarity_to_others = max(
                        max_similarity_to_others,
                        comparison.primary_similarity
                    )

            # Feature is model-specific if max similarity is below threshold
            if max_similarity_to_others <= config.specific_threshold:
                specificity_score = 1.0 - max_similarity_to_others
                model_specific[model_id].append((feature_id, fp, specificity_score))

        # Sort by specificity score
        model_specific[model_id].sort(key=lambda x: x[2], reverse=True)

    return dict(model_specific)


@dataclass
class ModelComparisonReport:
    """Comprehensive report comparing multiple models."""

    model_ids: list[str]
    universal_features: list[tuple[str, list[FeatureFingerprint], float]]
    model_specific_features: dict[str, list[tuple[str, FeatureFingerprint, float]]]

    # Statistics
    total_features_per_model: dict[str, int] = field(default_factory=dict)
    universal_count: int = 0
    specific_count_per_model: dict[str, int] = field(default_factory=dict)

    # Pairwise similarities
    pairwise_similarities: dict[tuple[str, str], float] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "Model Comparison Report",
            "=" * 60,
            f"\nModels compared: {', '.join(self.model_ids)}",
            f"\nUniversal features found: {len(self.universal_features)}",
        ]

        if self.universal_features:
            lines.append("\nTop 5 universal features:")
            for i, (fid, fps, sim) in enumerate(self.universal_features[:5]):
                labels = [fp.label or "unlabeled" for fp in fps]
                lines.append(f"  {i+1}. {fid} (similarity: {sim:.3f})")
                lines.append(f"      Labels: {', '.join(labels[:3])}")

        lines.append("\nModel-specific features:")
        for model_id, features in self.model_specific_features.items():
            lines.append(f"  {model_id}: {len(features)} unique features")
            if features:
                lines.append(f"    Most specific: {features[0][0]} (score: {features[0][2]:.3f})")

        if self.pairwise_similarities:
            lines.append("\nPairwise model similarities:")
            for (m1, m2), sim in sorted(self.pairwise_similarities.items()):
                lines.append(f"  {m1} ↔ {m2}: {sim:.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)


def compare_models(
    fingerprints_by_model: dict[str, dict[str, FeatureFingerprint]],
    config: FingerprintConfig | None = None,
) -> ModelComparisonReport:
    """
    Generate comprehensive comparison report for multiple models.

    Args:
        fingerprints_by_model: Dict mapping model_id to feature fingerprints
        config: Fingerprinting configuration

    Returns:
        Complete comparison report
    """
    config = config or FingerprintConfig()

    model_ids = list(fingerprints_by_model.keys())

    # Find universal and model-specific features
    universal = find_universal_features(fingerprints_by_model, config)
    specific = find_model_specific_features(fingerprints_by_model, config)

    # Compute pairwise similarities
    pairwise = {}
    for i, m1 in enumerate(model_ids):
        for m2 in model_ids[i+1:]:
            # Average similarity across matching features
            sims = []
            for fid, fp1 in fingerprints_by_model[m1].items():
                for fp2 in fingerprints_by_model[m2].values():
                    comparison = compare_fingerprints(fp1, fp2, config)
                    sims.append(comparison.primary_similarity)

            if sims:
                pairwise[(m1, m2)] = float(np.mean(sorted(sims)[-10:]))  # Top 10 matches

    # Build report
    report = ModelComparisonReport(
        model_ids=model_ids,
        universal_features=universal,
        model_specific_features=specific,
        total_features_per_model={
            m: len(fps) for m, fps in fingerprints_by_model.items()
        },
        universal_count=len(universal),
        specific_count_per_model={
            m: len(feats) for m, feats in specific.items()
        },
        pairwise_similarities=pairwise,
    )

    return report


def create_fingerprinter(
    adapter: "ModelAdapter",
    model_id: str,
    config: FingerprintConfig | None = None,
) -> ModelFingerprinter:
    """
    Create a fingerprinter for a model adapter.

    Args:
        adapter: Model adapter for activation extraction
        model_id: Unique identifier for the model
        config: Optional configuration

    Returns:
        Configured ModelFingerprinter
    """
    return ModelFingerprinter(adapter, model_id, config)
