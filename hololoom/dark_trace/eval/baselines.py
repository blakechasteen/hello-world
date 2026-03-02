"""
Dark Trace Evaluation: Baseline Methods

Provides baseline comparison methods for validating SAE superiority.
Essential for demonstrating that interpretability claims are meaningful.

Baseline Types:
- RandomBaseline: Random feature directions
- MeanBaseline: Mean activation patterns
- PCABaseline: Principal components
- SupervisedProbeBaseline: Supervised linear probes

Research Basis:
- Standard ML practice: Compare to simple baselines
- Interpretability benchmarking: Ensure methods beat random
- Ablation methodology: Isolate contribution of each component
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np


class BaselineType(Enum):
    """Types of baseline methods."""

    RANDOM = "random"
    MEAN = "mean"
    PCA = "pca"
    SUPERVISED_PROBE = "supervised_probe"


@dataclass
class BaselineResult:
    """Result from baseline evaluation.

    Attributes:
        baseline_type: Type of baseline
        name: Baseline name
        metrics: Evaluation metrics
        duration_seconds: Computation time
        metadata: Additional information
    """

    baseline_type: BaselineType
    name: str
    metrics: Dict[str, float]
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Baseline: {self.name}",
            f"Duration: {self.duration_seconds:.2f}s",
            "",
            "Metrics:",
        ]
        for key, value in sorted(self.metrics.items()):
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "baseline_type": self.baseline_type.value,
            "name": self.name,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class SAEProtocol(Protocol):
    """Protocol for SAE models."""

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent features."""
        ...

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent features to reconstruction."""
        ...

    @property
    def n_features(self) -> int:
        """Number of latent features."""
        ...


class BaselineMethod(ABC):
    """Abstract base class for baseline methods."""

    def __init__(self, n_features: int, seed: Optional[int] = 42):
        """Initialize baseline method.

        Args:
            n_features: Number of features to learn/use
            seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self._rng = np.random.default_rng(seed)
        self._fitted = False

    @abstractmethod
    def fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        """Fit the baseline to data.

        Args:
            data: Training data [n_samples, dim]
            labels: Optional labels for supervised methods
        """
        pass

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to features.

        Args:
            x: Input data [n_samples, dim]

        Returns:
            Feature activations [n_samples, n_features]
        """
        pass

    @abstractmethod
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode features to reconstruction.

        Args:
            z: Feature activations [n_samples, n_features]

        Returns:
            Reconstructed data [n_samples, dim]
        """
        pass

    def evaluate(
        self,
        test_data: np.ndarray,
        test_labels: Optional[np.ndarray] = None,
    ) -> BaselineResult:
        """Evaluate the baseline on test data.

        Args:
            test_data: Test data
            test_labels: Optional test labels

        Returns:
            Evaluation result
        """
        start_time = time.time()

        # Encode and decode
        encoded = self.encode(test_data)
        decoded = self.decode(encoded)

        # Compute metrics
        mse = np.mean((test_data - decoded) ** 2)
        rmse = np.sqrt(mse)

        # Cosine similarity
        norms_orig = np.linalg.norm(test_data, axis=1, keepdims=True) + 1e-10
        norms_recon = np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-10
        cos_sim = np.mean(
            np.sum(test_data * decoded, axis=1) /
            (norms_orig.squeeze() * norms_recon.squeeze())
        )

        # Explained variance
        var_orig = np.var(test_data)
        var_residual = np.var(test_data - decoded)
        explained_var = 1.0 - var_residual / (var_orig + 1e-10)

        # Sparsity
        sparsity = np.mean(encoded > 0)
        mean_l0 = np.mean(np.sum(encoded > 0, axis=1))

        duration = time.time() - start_time

        return BaselineResult(
            baseline_type=self._baseline_type,
            name=self._name,
            metrics={
                "mse": mse,
                "rmse": rmse,
                "cosine_similarity": cos_sim,
                "explained_variance": explained_var,
                "sparsity": sparsity,
                "mean_l0": mean_l0,
            },
            duration_seconds=duration,
            metadata={
                "n_features": self.n_features,
                "n_test_samples": len(test_data),
            },
        )

    @property
    @abstractmethod
    def _baseline_type(self) -> BaselineType:
        """Return baseline type."""
        pass

    @property
    @abstractmethod
    def _name(self) -> str:
        """Return baseline name."""
        pass


class RandomBaseline(BaselineMethod):
    """Random feature directions baseline.

    Creates random orthogonal directions and projects data.
    This is the "null hypothesis" - SAE should beat random.

    Projection: z = ReLU(W @ x)
    Reconstruction: x_hat = z @ W.T
    """

    def __init__(
        self,
        n_features: int,
        seed: Optional[int] = 42,
        sparsity: float = 0.1,
    ):
        """Initialize random baseline.

        Args:
            n_features: Number of random features
            seed: Random seed
            sparsity: Target sparsity (ReLU threshold percentile)
        """
        super().__init__(n_features, seed)
        self.sparsity = sparsity
        self._weights: Optional[np.ndarray] = None
        self._threshold: float = 0.0

    def fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        """Fit random directions.

        Args:
            data: Training data [n_samples, dim]
            labels: Ignored (unsupervised)
        """
        dim = data.shape[1]

        # Generate random orthogonal directions
        self._weights = self._rng.standard_normal((self.n_features, dim))

        # Orthogonalize using QR decomposition
        if self.n_features <= dim:
            q, _ = np.linalg.qr(self._weights.T)
            self._weights = q.T[:self.n_features]
        else:
            # More features than dimensions - normalize only
            self._weights = self._weights / np.linalg.norm(
                self._weights, axis=1, keepdims=True
            )

        # Compute threshold for sparsity
        projections = data @ self._weights.T
        self._threshold = np.percentile(projections, (1 - self.sparsity) * 100)

        self._fitted = True

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode with random directions.

        Args:
            x: Input data [n_samples, dim]

        Returns:
            Sparse activations [n_samples, n_features]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode()")

        projections = x @ self._weights.T
        # Apply ReLU with threshold
        return np.maximum(0, projections - self._threshold)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from random features.

        Args:
            z: Activations [n_samples, n_features]

        Returns:
            Reconstruction [n_samples, dim]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decode()")

        return z @ self._weights

    @property
    def _baseline_type(self) -> BaselineType:
        return BaselineType.RANDOM

    @property
    def _name(self) -> str:
        return f"Random ({self.n_features} features)"


class MeanBaseline(BaselineMethod):
    """Mean activation baseline.

    Simply returns mean of training data.
    Tests whether SAE captures more than central tendency.

    Projection: z = constant
    Reconstruction: x_hat = mean(x_train)
    """

    def __init__(self, n_features: int = 1, seed: Optional[int] = 42):
        """Initialize mean baseline.

        Args:
            n_features: Ignored (mean has 1 effective feature)
            seed: Random seed
        """
        super().__init__(1, seed)  # Mean is 1 feature
        self._mean: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        """Compute mean of training data.

        Args:
            data: Training data
            labels: Ignored
        """
        self._mean = np.mean(data, axis=0)
        self._fitted = True

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode (returns ones - constant activation).

        Args:
            x: Input data

        Returns:
            Constant activations [n_samples, 1]
        """
        return np.ones((len(x), 1))

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode (returns mean for all inputs).

        Args:
            z: Activations (ignored)

        Returns:
            Mean reconstruction [n_samples, dim]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decode()")

        return np.tile(self._mean, (len(z), 1))

    @property
    def _baseline_type(self) -> BaselineType:
        return BaselineType.MEAN

    @property
    def _name(self) -> str:
        return "Mean Baseline"


class PCABaseline(BaselineMethod):
    """Principal Component Analysis baseline.

    Projects to top-k principal components.
    Standard dimensionality reduction - SAE should do better
    at finding interpretable features.

    Projection: z = (x - mean) @ V[:k]
    Reconstruction: x_hat = z @ V[:k].T + mean
    """

    def __init__(
        self,
        n_features: int,
        seed: Optional[int] = 42,
        whiten: bool = False,
    ):
        """Initialize PCA baseline.

        Args:
            n_features: Number of principal components
            seed: Random seed
            whiten: Whether to whiten components
        """
        super().__init__(n_features, seed)
        self.whiten = whiten
        self._components: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._singular_values: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        """Fit PCA to data.

        Args:
            data: Training data
            labels: Ignored
        """
        # Center data
        self._mean = np.mean(data, axis=0)
        centered = data - self._mean

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Keep top-k components
        k = min(self.n_features, len(S))
        self._components = Vt[:k]
        self._singular_values = S[:k]

        self._fitted = True

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Project to principal components.

        Args:
            x: Input data

        Returns:
            PCA projections [n_samples, n_features]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode()")

        centered = x - self._mean
        projections = centered @ self._components.T

        if self.whiten:
            projections = projections / (self._singular_values + 1e-10)

        return projections

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct from principal components.

        Args:
            z: PCA projections

        Returns:
            Reconstruction [n_samples, dim]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decode()")

        if self.whiten:
            z = z * self._singular_values

        return z @ self._components + self._mean

    @property
    def _baseline_type(self) -> BaselineType:
        return BaselineType.PCA

    @property
    def _name(self) -> str:
        return f"PCA ({self.n_features} components)"


class SupervisedProbeBaseline(BaselineMethod):
    """Supervised linear probe baseline.

    Trains linear classifiers for known concepts.
    Tests whether SAE features are better than supervised features.

    For each label:
    Projection: z_i = sigmoid(w_i @ x + b_i)
    Reconstruction: x_hat = Z @ W
    """

    def __init__(
        self,
        n_features: int,
        seed: Optional[int] = 42,
        l2_reg: float = 0.01,
        max_iter: int = 100,
    ):
        """Initialize supervised probe baseline.

        Args:
            n_features: Number of probe features (should match n_labels)
            seed: Random seed
            l2_reg: L2 regularization strength
            max_iter: Maximum training iterations
        """
        super().__init__(n_features, seed)
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self._weights: Optional[np.ndarray] = None
        self._biases: Optional[np.ndarray] = None
        self._decode_weights: Optional[np.ndarray] = None

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        """Fit linear probes to data with labels.

        Args:
            data: Training data [n_samples, dim]
            labels: Labels [n_samples] or [n_samples, n_labels]
        """
        n_samples, dim = data.shape

        # Handle labels
        if labels is None:
            # Create synthetic labels based on data clusters
            # K-means style: assign to nearest random centroid
            centroids = data[self._rng.choice(n_samples, self.n_features, replace=False)]
            distances = np.linalg.norm(
                data[:, np.newaxis] - centroids, axis=2
            )
            labels = np.argmin(distances, axis=1)

        # One-hot encode if needed
        if labels.ndim == 1:
            n_classes = min(self.n_features, len(np.unique(labels)))
            labels_oh = np.zeros((n_samples, n_classes))
            for i, l in enumerate(labels):
                if l < n_classes:
                    labels_oh[i, l] = 1
            labels = labels_oh

        n_probes = labels.shape[1]

        # Initialize weights
        self._weights = self._rng.standard_normal((n_probes, dim)) * 0.01
        self._biases = np.zeros(n_probes)

        # Train each probe with gradient descent
        lr = 0.1

        for iteration in range(self.max_iter):
            # Forward pass
            logits = data @ self._weights.T + self._biases
            probs = self._sigmoid(logits)

            # Binary cross-entropy gradient
            grad_logits = probs - labels  # [n_samples, n_probes]

            # Weight gradients
            grad_w = grad_logits.T @ data / n_samples + self.l2_reg * self._weights
            grad_b = np.mean(grad_logits, axis=0)

            # Update
            self._weights -= lr * grad_w
            self._biases -= lr * grad_b

            # Decay learning rate
            lr *= 0.99

        # Fit decoder using pseudo-inverse
        encoded = self._sigmoid(data @ self._weights.T + self._biases)
        self._decode_weights = np.linalg.lstsq(encoded, data, rcond=None)[0]

        self._fitted = True

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode using linear probes.

        Args:
            x: Input data

        Returns:
            Probe activations [n_samples, n_probes]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode()")

        logits = x @ self._weights.T + self._biases
        return self._sigmoid(logits)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from probe activations.

        Args:
            z: Probe activations

        Returns:
            Reconstruction [n_samples, dim]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decode()")

        return z @ self._decode_weights

    @property
    def _baseline_type(self) -> BaselineType:
        return BaselineType.SUPERVISED_PROBE

    @property
    def _name(self) -> str:
        return f"Supervised Probe ({self.n_features} probes)"


@dataclass
class ComparisonResult:
    """Result from comparing SAE to baselines.

    Attributes:
        sae_result: SAE evaluation metrics
        baseline_results: Dict of baseline results
        improvements: SAE improvement over each baseline
        best_baseline: Name of best performing baseline
        sae_beats_all: Whether SAE beats all baselines
    """

    sae_result: Dict[str, float]
    baseline_results: Dict[str, BaselineResult]
    improvements: Dict[str, Dict[str, float]]
    best_baseline: str
    sae_beats_all: bool

    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            "=" * 60,
            "SAE vs BASELINES COMPARISON",
            "=" * 60,
            "",
            "SAE Metrics:",
        ]

        for key, value in sorted(self.sae_result.items()):
            lines.append(f"  {key}: {value:.4f}")

        lines.extend(["", "-" * 60, "Baseline Comparisons:", ""])

        for name, result in self.baseline_results.items():
            improvement = self.improvements.get(name, {})
            lines.append(f"{name}:")
            for metric, imp in improvement.items():
                status = "↑" if imp > 0 else "↓"
                lines.append(f"  {metric}: {status} {abs(imp):.4f}")
            lines.append("")

        lines.extend([
            "-" * 60,
            f"Best baseline: {self.best_baseline}",
            f"SAE beats all baselines: {'YES' if self.sae_beats_all else 'NO'}",
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sae_result": self.sae_result,
            "baseline_results": {
                name: result.to_dict()
                for name, result in self.baseline_results.items()
            },
            "improvements": self.improvements,
            "best_baseline": self.best_baseline,
            "sae_beats_all": self.sae_beats_all,
        }


class BaselineComparison:
    """Compare SAE to multiple baseline methods.

    Example:
        comparison = BaselineComparison(sae, test_data)
        comparison.add_baseline(RandomBaseline(n_features=100))
        comparison.add_baseline(PCABaseline(n_components=100))
        comparison.add_baseline(SupervisedProbeBaseline(n_features=10))

        results = comparison.evaluate()
        print(results.summary())
    """

    def __init__(
        self,
        sae: SAEProtocol,
        test_data: np.ndarray,
        test_labels: Optional[np.ndarray] = None,
        primary_metric: str = "explained_variance",
    ):
        """Initialize comparison.

        Args:
            sae: SAE model to evaluate
            test_data: Test data
            test_labels: Optional test labels
            primary_metric: Metric for determining "best" baseline
        """
        self.sae = sae
        self.test_data = test_data
        self.test_labels = test_labels
        self.primary_metric = primary_metric
        self.baselines: List[BaselineMethod] = []
        self._train_data: Optional[np.ndarray] = None

    def set_training_data(self, train_data: np.ndarray) -> "BaselineComparison":
        """Set training data for baselines.

        Args:
            train_data: Training data

        Returns:
            Self for chaining
        """
        self._train_data = train_data
        return self

    def add_baseline(self, baseline: BaselineMethod) -> "BaselineComparison":
        """Add a baseline method.

        Args:
            baseline: Baseline to add

        Returns:
            Self for chaining
        """
        self.baselines.append(baseline)
        return self

    def add_all_defaults(self, n_features: int) -> "BaselineComparison":
        """Add all default baselines.

        Args:
            n_features: Number of features for baselines

        Returns:
            Self for chaining
        """
        self.add_baseline(RandomBaseline(n_features))
        self.add_baseline(MeanBaseline())
        self.add_baseline(PCABaseline(n_features))
        self.add_baseline(SupervisedProbeBaseline(min(n_features, 10)))
        return self

    def _evaluate_sae(self) -> Dict[str, float]:
        """Evaluate SAE on test data.

        Returns:
            SAE metrics
        """
        encoded = self.sae.encode(self.test_data)
        decoded = self.sae.decode(encoded)

        mse = np.mean((self.test_data - decoded) ** 2)
        rmse = np.sqrt(mse)

        norms_orig = np.linalg.norm(self.test_data, axis=1, keepdims=True) + 1e-10
        norms_recon = np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-10
        cos_sim = np.mean(
            np.sum(self.test_data * decoded, axis=1) /
            (norms_orig.squeeze() * norms_recon.squeeze())
        )

        var_orig = np.var(self.test_data)
        var_residual = np.var(self.test_data - decoded)
        explained_var = 1.0 - var_residual / (var_orig + 1e-10)

        sparsity = np.mean(encoded > 0)
        mean_l0 = np.mean(np.sum(encoded > 0, axis=1))

        return {
            "mse": mse,
            "rmse": rmse,
            "cosine_similarity": cos_sim,
            "explained_variance": explained_var,
            "sparsity": sparsity,
            "mean_l0": mean_l0,
        }

    def evaluate(self) -> ComparisonResult:
        """Run comparison of SAE vs all baselines.

        Returns:
            Comparison result
        """
        # Use test data as training if not set
        train_data = self._train_data if self._train_data is not None else self.test_data

        # Evaluate SAE
        sae_metrics = self._evaluate_sae()

        # Evaluate baselines
        baseline_results: Dict[str, BaselineResult] = {}
        for baseline in self.baselines:
            # Fit baseline
            baseline.fit(train_data, self.test_labels)

            # Evaluate
            result = baseline.evaluate(self.test_data, self.test_labels)
            baseline_results[result.name] = result

        # Compute improvements
        improvements: Dict[str, Dict[str, float]] = {}
        for name, result in baseline_results.items():
            improvements[name] = {}
            for metric in sae_metrics:
                if metric in result.metrics:
                    # Higher is better for most metrics, lower for MSE/RMSE
                    if metric in ("mse", "rmse"):
                        imp = result.metrics[metric] - sae_metrics[metric]
                    else:
                        imp = sae_metrics[metric] - result.metrics[metric]
                    improvements[name][metric] = imp

        # Find best baseline
        best_score = -float("inf")
        best_baseline = ""
        for name, result in baseline_results.items():
            score = result.metrics.get(self.primary_metric, 0)
            if self.primary_metric in ("mse", "rmse"):
                score = -score  # Lower is better
            if score > best_score:
                best_score = score
                best_baseline = name

        # Check if SAE beats all baselines on primary metric
        sae_primary = sae_metrics.get(self.primary_metric, 0)
        if self.primary_metric in ("mse", "rmse"):
            sae_primary = -sae_primary
        sae_beats_all = sae_primary > best_score

        return ComparisonResult(
            sae_result=sae_metrics,
            baseline_results=baseline_results,
            improvements=improvements,
            best_baseline=best_baseline,
            sae_beats_all=sae_beats_all,
        )


# Convenience functions

def create_default_baselines(n_features: int) -> List[BaselineMethod]:
    """Create default set of baselines.

    Args:
        n_features: Number of features

    Returns:
        List of baseline methods
    """
    return [
        RandomBaseline(n_features),
        MeanBaseline(),
        PCABaseline(n_features),
        SupervisedProbeBaseline(min(n_features, 10)),
    ]


def quick_baseline_comparison(
    sae: SAEProtocol,
    test_data: np.ndarray,
    n_features: int = 100,
) -> ComparisonResult:
    """Run quick baseline comparison.

    Args:
        sae: SAE model
        test_data: Test data
        n_features: Number of features for baselines

    Returns:
        Comparison result
    """
    comparison = BaselineComparison(sae, test_data)
    comparison.add_all_defaults(n_features)
    return comparison.evaluate()
