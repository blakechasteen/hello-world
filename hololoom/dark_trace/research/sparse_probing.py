"""
Sparse Probing for Dark Trace

Techniques for probing what specific SAE features represent using
sparse linear probes and activation analysis.

Based on:
- "Sparse Autoencoders Find Highly Interpretable Features" (Cunningham et al., 2023)
- "Scaling Monosemanticity" (Anthropic, 2024)

Author: HoloLoom Team
Created: December 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ProbeType(Enum):
    """Types of probes for feature analysis."""
    LINEAR = "linear"               # Simple linear probe
    LOGISTIC = "logistic"           # Logistic regression
    SPARSE = "sparse"               # L1-regularized (sparse)
    RIDGE = "ridge"                 # L2-regularized
    CONTRAST = "contrast"           # Contrastive probe


@dataclass
class ProbeConfig:
    """Configuration for sparse probing."""
    probe_type: ProbeType = ProbeType.SPARSE
    regularization: float = 0.01        # L1/L2 regularization strength
    max_iterations: int = 1000
    tolerance: float = 1e-4
    batch_size: int = 32
    learning_rate: float = 0.01
    class_weight: str | None = "balanced"  # Handle class imbalance


@dataclass
class ProbeResult:
    """Result from a probing experiment."""
    feature_idx: int
    probe_type: ProbeType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    coefficients: np.ndarray
    intercept: float
    top_positive_tokens: list[tuple[str, float]]    # Tokens that activate
    top_negative_tokens: list[tuple[str, float]]    # Tokens that suppress
    interpretation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_idx": self.feature_idx,
            "probe_type": self.probe_type.value,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "top_positive_tokens": self.top_positive_tokens,
            "top_negative_tokens": self.top_negative_tokens,
            "interpretation": self.interpretation,
            "metadata": self.metadata,
        }


@dataclass
class ActivationDataset:
    """Dataset of activations for probing."""
    activations: np.ndarray         # Shape: (n_samples, n_features)
    labels: np.ndarray              # Shape: (n_samples,) or (n_samples, n_classes)
    tokens: list[str]               # Token strings for interpretation
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.activations.shape[0]

    @property
    def n_features(self) -> int:
        return self.activations.shape[1]


class SparseProbe:
    """
    Sparse linear probe for analyzing feature representations.

    Uses L1 regularization to find sparse, interpretable mappings
    from features to concepts.
    """

    def __init__(self, config: ProbeConfig | None = None):
        self.config = config or ProbeConfig()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None
    ) -> "SparseProbe":
        """
        Fit the sparse probe to data.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            sample_weight: Optional sample weights

        Returns:
            self
        """
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Handle class weights
        if self.config.class_weight == "balanced":
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_weights = n_samples / (len(unique_classes) * class_counts)
            sample_weight = np.array([class_weights[int(yi)] for yi in y])
        elif sample_weight is None:
            sample_weight = np.ones(n_samples)

        # Gradient descent with L1 regularization (proximal gradient)
        for iteration in range(self.config.max_iterations):
            # Forward pass
            logits = X @ self.weights + self.bias

            if self.config.probe_type == ProbeType.LOGISTIC:
                # Logistic regression
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                grad = X.T @ ((probs - y) * sample_weight) / n_samples
                grad_bias = np.sum((probs - y) * sample_weight) / n_samples
            else:
                # Linear regression
                residuals = logits - y
                grad = X.T @ (residuals * sample_weight) / n_samples
                grad_bias = np.sum(residuals * sample_weight) / n_samples

            # Update with L1 proximal operator
            self.weights -= self.config.learning_rate * grad

            if self.config.probe_type in [ProbeType.SPARSE, ProbeType.LINEAR]:
                # Soft thresholding for L1
                threshold = self.config.regularization * self.config.learning_rate
                self.weights = np.sign(self.weights) * np.maximum(
                    np.abs(self.weights) - threshold, 0
                )
            elif self.config.probe_type == ProbeType.RIDGE:
                # L2 regularization
                self.weights *= (1 - self.config.regularization * self.config.learning_rate)

            self.bias -= self.config.learning_rate * grad_bias

            # Check convergence
            if np.max(np.abs(grad)) < self.config.tolerance:
                break

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for features."""
        if not self._is_fitted:
            raise ValueError("Probe not fitted")

        logits = X @ self.weights + self.bias

        if self.config.probe_type == ProbeType.LOGISTIC:
            return (logits > 0).astype(int)
        else:
            return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for features."""
        if not self._is_fitted:
            raise ValueError("Probe not fitted")

        logits = X @ self.weights + self.bias
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        return np.column_stack([1 - probs, probs])

    def get_top_features(
        self,
        feature_names: list[str] | None = None,
        k: int = 10
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Get top positive and negative features."""
        if not self._is_fitted:
            raise ValueError("Probe not fitted")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.weights))]

        # Sort by absolute weight
        indices_pos = np.argsort(self.weights)[::-1][:k]
        indices_neg = np.argsort(self.weights)[:k]

        top_positive = [
            (feature_names[i], float(self.weights[i]))
            for i in indices_pos if self.weights[i] > 0
        ]
        top_negative = [
            (feature_names[i], float(self.weights[i]))
            for i in indices_neg if self.weights[i] < 0
        ]

        return top_positive, top_negative


class FeatureProber:
    """
    Probe SAE features to understand what they represent.

    Uses contrastive and correlational probing to build
    interpretations of feature functions.
    """

    def __init__(self, config: ProbeConfig | None = None):
        self.config = config or ProbeConfig()
        self._probes: dict[int, SparseProbe] = {}
        self._results: dict[int, ProbeResult] = {}

    def probe_feature(
        self,
        feature_idx: int,
        dataset: ActivationDataset,
        concept_labels: np.ndarray,
        concept_name: str = "concept",
    ) -> ProbeResult:
        """
        Probe a specific feature for a concept.

        Args:
            feature_idx: Index of SAE feature to probe
            dataset: Dataset of activations
            concept_labels: Binary labels for concept presence
            concept_name: Name of the concept being probed

        Returns:
            ProbeResult with interpretation
        """
        # Extract feature activations
        feature_acts = dataset.activations[:, feature_idx]

        # Create and fit probe
        probe = SparseProbe(self.config)

        # Use feature as input to predict concept
        X = feature_acts.reshape(-1, 1)
        probe.fit(X, concept_labels)

        # Evaluate
        predictions = probe.predict(X)
        accuracy = np.mean(predictions == concept_labels)

        # Calculate metrics
        tp = np.sum((predictions == 1) & (concept_labels == 1))
        fp = np.sum((predictions == 1) & (concept_labels == 0))
        fn = np.sum((predictions == 0) & (concept_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Find top activating tokens
        sorted_indices = np.argsort(feature_acts)[::-1]
        top_positive_tokens = [
            (dataset.tokens[i], float(feature_acts[i]))
            for i in sorted_indices[:10]
            if feature_acts[i] > 0
        ]
        top_negative_tokens = [
            (dataset.tokens[i], float(feature_acts[i]))
            for i in sorted_indices[-10:][::-1]
            if feature_acts[i] <= 0
        ]

        # Generate interpretation
        interpretation = self._generate_interpretation(
            feature_idx,
            concept_name,
            accuracy,
            f1,
            top_positive_tokens,
            probe.weights[0] if len(probe.weights) > 0 else 0.0
        )

        result = ProbeResult(
            feature_idx=feature_idx,
            probe_type=self.config.probe_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            coefficients=probe.weights,
            intercept=probe.bias,
            top_positive_tokens=top_positive_tokens,
            top_negative_tokens=top_negative_tokens,
            interpretation=interpretation,
            metadata={
                "concept_name": concept_name,
                "n_samples": dataset.n_samples,
                "n_positive": int(np.sum(concept_labels)),
                "n_negative": int(np.sum(1 - concept_labels)),
            }
        )

        self._probes[feature_idx] = probe
        self._results[feature_idx] = result

        return result

    def probe_all_features(
        self,
        dataset: ActivationDataset,
        concept_labels: np.ndarray,
        concept_name: str = "concept",
        min_activation: float = 0.1,
    ) -> list[ProbeResult]:
        """
        Probe all features for a concept.

        Args:
            dataset: Dataset of activations
            concept_labels: Binary labels for concept presence
            concept_name: Name of the concept
            min_activation: Minimum mean activation to consider

        Returns:
            List of ProbeResults sorted by F1 score
        """
        results = []

        for feature_idx in range(dataset.n_features):
            # Skip features with very low activation
            mean_act = np.mean(np.abs(dataset.activations[:, feature_idx]))
            if mean_act < min_activation:
                continue

            result = self.probe_feature(
                feature_idx,
                dataset,
                concept_labels,
                concept_name
            )
            results.append(result)

        # Sort by F1 score
        results.sort(key=lambda r: r.f1_score, reverse=True)

        return results

    def find_concept_features(
        self,
        dataset: ActivationDataset,
        concept_labels: np.ndarray,
        concept_name: str = "concept",
        min_f1: float = 0.5,
        max_features: int = 20,
    ) -> list[ProbeResult]:
        """
        Find features that represent a specific concept.

        Args:
            dataset: Dataset of activations
            concept_labels: Binary labels for concept presence
            concept_name: Name of the concept
            min_f1: Minimum F1 score to consider
            max_features: Maximum number of features to return

        Returns:
            List of features that represent the concept
        """
        all_results = self.probe_all_features(
            dataset,
            concept_labels,
            concept_name
        )

        # Filter by F1 score
        filtered = [r for r in all_results if r.f1_score >= min_f1]

        return filtered[:max_features]

    def _generate_interpretation(
        self,
        feature_idx: int,
        concept_name: str,
        accuracy: float,
        f1: float,
        top_tokens: list[tuple[str, float]],
        weight: float
    ) -> str:
        """Generate human-readable interpretation."""
        if f1 < 0.3:
            strength = "weakly"
        elif f1 < 0.6:
            strength = "moderately"
        else:
            strength = "strongly"

        direction = "positively" if weight > 0 else "negatively"

        tokens_str = ", ".join([f"'{t[0]}'" for t in top_tokens[:3]])

        return (
            f"Feature {feature_idx} {strength} correlates ({direction}) "
            f"with '{concept_name}' (F1={f1:.2f}). "
            f"Highest activations on: {tokens_str}."
        )

    def get_result(self, feature_idx: int) -> ProbeResult | None:
        """Get cached probe result for a feature."""
        return self._results.get(feature_idx)

    def export_results(self) -> dict[int, dict]:
        """Export all probe results as dictionary."""
        return {
            idx: result.to_dict()
            for idx, result in self._results.items()
        }


class ContrastiveProber:
    """
    Contrastive probing to find what distinguishes feature activations.

    Compares positive and negative examples to find discriminative patterns.
    """

    def __init__(self, config: ProbeConfig | None = None):
        self.config = config or ProbeConfig(probe_type=ProbeType.CONTRAST)

    def probe_contrast(
        self,
        positive_acts: np.ndarray,
        negative_acts: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Find what distinguishes positive from negative examples.

        Args:
            positive_acts: Activations for positive examples (n_pos, n_features)
            negative_acts: Activations for negative examples (n_neg, n_features)
            feature_names: Optional feature names

        Returns:
            Contrastive analysis results
        """
        n_features = positive_acts.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Mean difference
        pos_mean = np.mean(positive_acts, axis=0)
        neg_mean = np.mean(negative_acts, axis=0)
        diff = pos_mean - neg_mean

        # Variance
        pos_var = np.var(positive_acts, axis=0)
        neg_var = np.var(negative_acts, axis=0)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((pos_var + neg_var) / 2)
        pooled_std = np.where(pooled_std < 1e-8, 1e-8, pooled_std)
        effect_size = diff / pooled_std

        # Find most discriminative features
        sorted_indices = np.argsort(np.abs(effect_size))[::-1]

        discriminative_features = []
        for i in sorted_indices[:20]:
            discriminative_features.append({
                "feature_idx": int(i),
                "feature_name": feature_names[i],
                "effect_size": float(effect_size[i]),
                "pos_mean": float(pos_mean[i]),
                "neg_mean": float(neg_mean[i]),
                "direction": "positive" if diff[i] > 0 else "negative",
            })

        return {
            "discriminative_features": discriminative_features,
            "mean_effect_size": float(np.mean(np.abs(effect_size))),
            "max_effect_size": float(np.max(np.abs(effect_size))),
            "n_positive": positive_acts.shape[0],
            "n_negative": negative_acts.shape[0],
        }


def create_probe(
    probe_type: ProbeType = ProbeType.SPARSE,
    regularization: float = 0.01,
) -> SparseProbe:
    """Factory function to create probes."""
    config = ProbeConfig(
        probe_type=probe_type,
        regularization=regularization
    )
    return SparseProbe(config)


def create_feature_prober(
    probe_type: ProbeType = ProbeType.SPARSE,
) -> FeatureProber:
    """Factory function to create feature prober."""
    config = ProbeConfig(probe_type=probe_type)
    return FeatureProber(config)
