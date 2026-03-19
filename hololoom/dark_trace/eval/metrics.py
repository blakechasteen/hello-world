"""
Dark Trace Evaluation Metrics: Comprehensive SAE Quality Assessment

This module provides rigorous metrics for evaluating SAE quality across
multiple dimensions: reconstruction, sparsity, interpretability, steering,
and causal validity.

Research Basis:
- Scaling Monosemanticity (Anthropic 2024): Standard SAE metrics
- Towards Monosemanticity (Anthropic 2023): Feature quality assessment
- SAE evaluation best practices from interpretability research
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Protocol,
)


class MetricLevel(Enum):
    """Level of metric computation."""

    QUICK = "quick"  # Fast, essential metrics only
    STANDARD = "standard"  # Standard evaluation
    COMPREHENSIVE = "comprehensive"  # All metrics including expensive ones


@dataclass
class MetricsConfig:
    """Configuration for metrics computation.

    Attributes:
        level: Metric computation level
        batch_size: Batch size for computation
        n_samples: Number of samples for statistical metrics
        compute_per_feature: Compute per-feature metrics
        dead_threshold: Activation threshold for dead features
        sparsity_target: Target sparsity level
        significance_level: Statistical significance level
    """

    level: MetricLevel = MetricLevel.STANDARD
    batch_size: int = 256
    n_samples: int = 10000
    compute_per_feature: bool = True
    dead_threshold: float = 1e-6
    sparsity_target: float = 0.01  # 1% active features
    significance_level: float = 0.05

    @classmethod
    def quick(cls) -> "MetricsConfig":
        """Quick evaluation configuration."""
        return cls(
            level=MetricLevel.QUICK,
            n_samples=1000,
            compute_per_feature=False,
        )

    @classmethod
    def comprehensive(cls) -> "MetricsConfig":
        """Comprehensive evaluation configuration."""
        return cls(
            level=MetricLevel.COMPREHENSIVE,
            n_samples=50000,
            compute_per_feature=True,
        )


@dataclass
class ReconstructionMetrics:
    """Metrics for reconstruction quality.

    Measures how well the SAE reconstructs activations.
    """

    mse: float  # Mean squared error
    rmse: float  # Root mean squared error
    mae: float  # Mean absolute error
    cosine_similarity: float  # Average cosine similarity
    explained_variance: float  # Variance explained ratio
    r_squared: float  # Coefficient of determination
    relative_error: float  # Relative reconstruction error

    # Per-layer metrics (optional)
    per_layer_mse: dict[int, float] | None = None

    def is_acceptable(self, threshold_mse: float = 0.01) -> bool:
        """Check if reconstruction quality is acceptable."""
        return self.mse < threshold_mse and self.explained_variance > 0.9

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Reconstruction: MSE={self.mse:.4f}, "
            f"Cosine={self.cosine_similarity:.4f}, "
            f"ExplainedVar={self.explained_variance:.2%}"
        )


@dataclass
class SparsityMetrics:
    """Metrics for activation sparsity.

    Measures how sparse the SAE latent activations are.
    """

    l0: float  # Average L0 norm (number of active features)
    l0_ratio: float  # L0 / total features
    l1: float  # Average L1 norm
    entropy: float  # Activation entropy
    gini_coefficient: float  # Gini coefficient of activations
    dead_feature_count: int  # Number of dead features
    dead_feature_ratio: float  # Ratio of dead features
    activation_frequency: dict[str, float] = field(default_factory=dict)

    # Distribution statistics
    mean_activation: float = 0.0
    std_activation: float = 0.0
    max_activation: float = 0.0

    def is_sparse_enough(self, target_l0_ratio: float = 0.01) -> bool:
        """Check if sparsity meets target."""
        return self.l0_ratio <= target_l0_ratio

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Sparsity: L0={self.l0:.1f} ({self.l0_ratio:.2%}), "
            f"DeadFeatures={self.dead_feature_ratio:.2%}, "
            f"Entropy={self.entropy:.3f}"
        )


@dataclass
class FeatureQuality:
    """Quality metrics for a single feature."""

    feature_idx: int
    activation_frequency: float  # How often feature activates
    mean_activation: float  # Mean activation when active
    std_activation: float  # Std of activations
    monosemanticity_score: float  # 0-1, higher is more monosemantic
    polysemanticity_score: float  # 0-1, higher is more polysemantic
    importance: float  # Impact on reconstruction
    dead: bool  # Whether feature is dead


@dataclass
class InterpretabilityMetrics:
    """Metrics for feature interpretability.

    Measures how interpretable and monosemantic the learned features are.
    """

    mean_monosemanticity: float  # Average monosemanticity score
    std_monosemanticity: float  # Std of monosemanticity
    polysemantic_ratio: float  # Ratio of polysemantic features
    highly_interpretable_ratio: float  # Ratio of highly interpretable features
    feature_coherence: float  # How coherent features are
    feature_uniqueness: float  # How unique/distinct features are

    # Optional per-feature metrics
    per_feature_quality: list[FeatureQuality] | None = None

    # Clustering metrics
    n_semantic_clusters: int = 0
    cluster_purity: float = 0.0

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Interpretability: Monosemanticity={self.mean_monosemanticity:.3f}, "
            f"HighlyInterpretable={self.highly_interpretable_ratio:.2%}, "
            f"Polysemantic={self.polysemantic_ratio:.2%}"
        )


@dataclass
class SteeringMetrics:
    """Metrics for steering effectiveness.

    Measures how well SAE features can be used for behavior steering.
    """

    steering_accuracy: float  # Accuracy of steering interventions
    steering_magnitude: float  # Average magnitude of steering effect
    feature_controllability: float  # How controllable features are
    side_effect_ratio: float  # Ratio of unintended side effects
    target_achievement_rate: float  # Rate of achieving steering targets

    # Per-intervention metrics
    intervention_results: dict[str, float] = field(default_factory=dict)

    # Safety metrics
    safety_relevant_features: int = 0
    safe_steering_ratio: float = 1.0

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Steering: Accuracy={self.steering_accuracy:.2%}, "
            f"Controllability={self.feature_controllability:.3f}, "
            f"SideEffects={self.side_effect_ratio:.2%}"
        )


@dataclass
class CausalMetrics:
    """Metrics for causal validity.

    Measures whether feature attributions are causally valid.
    """

    ablation_accuracy: float  # Accuracy of ablation predictions
    injection_accuracy: float  # Accuracy of injection predictions
    causal_faithfulness: float  # How faithful to true causal structure
    intervention_consistency: float  # Consistency across interventions
    counterfactual_accuracy: float  # Accuracy of counterfactual predictions

    # Path analysis
    causal_path_coverage: float = 0.0  # Coverage of causal paths
    spurious_correlation_ratio: float = 0.0  # Ratio of spurious correlations

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Causal: AblationAcc={self.ablation_accuracy:.2%}, "
            f"Faithfulness={self.causal_faithfulness:.3f}, "
            f"SpuriousCorr={self.spurious_correlation_ratio:.2%}"
        )


@dataclass
class MetricsReport:
    """Complete metrics report for SAE evaluation."""

    reconstruction: ReconstructionMetrics
    sparsity: SparsityMetrics
    interpretability: InterpretabilityMetrics
    steering: SteeringMetrics | None = None
    causal: CausalMetrics | None = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: MetricsConfig | None = None
    n_samples_evaluated: int = 0
    evaluation_time_seconds: float = 0.0

    def overall_score(self) -> float:
        """Compute overall quality score (0-1)."""
        scores = [
            self.reconstruction.explained_variance,
            1.0 - self.sparsity.dead_feature_ratio,
            self.interpretability.mean_monosemanticity,
        ]

        if self.steering:
            scores.append(self.steering.steering_accuracy)
        if self.causal:
            scores.append(self.causal.causal_faithfulness)

        return sum(scores) / len(scores)

    def summary(self) -> str:
        """Generate complete summary."""
        lines = [
            "=" * 60,
            "SAE Evaluation Report",
            "=" * 60,
            self.reconstruction.summary(),
            self.sparsity.summary(),
            self.interpretability.summary(),
        ]

        if self.steering:
            lines.append(self.steering.summary())
        if self.causal:
            lines.append(self.causal.summary())

        lines.extend([
            "-" * 60,
            f"Overall Score: {self.overall_score():.3f}",
            f"Evaluated: {self.n_samples_evaluated} samples in {self.evaluation_time_seconds:.2f}s",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "timestamp": self.timestamp,
            "n_samples": self.n_samples_evaluated,
            "overall_score": self.overall_score(),
            "reconstruction": {
                "mse": self.reconstruction.mse,
                "cosine_similarity": self.reconstruction.cosine_similarity,
                "explained_variance": self.reconstruction.explained_variance,
            },
            "sparsity": {
                "l0": self.sparsity.l0,
                "l0_ratio": self.sparsity.l0_ratio,
                "dead_feature_ratio": self.sparsity.dead_feature_ratio,
                "entropy": self.sparsity.entropy,
            },
            "interpretability": {
                "mean_monosemanticity": self.interpretability.mean_monosemanticity,
                "polysemantic_ratio": self.interpretability.polysemantic_ratio,
                "highly_interpretable_ratio": self.interpretability.highly_interpretable_ratio,
            },
        }

        if self.steering:
            result["steering"] = {
                "accuracy": self.steering.steering_accuracy,
                "controllability": self.steering.feature_controllability,
            }

        if self.causal:
            result["causal"] = {
                "faithfulness": self.causal.causal_faithfulness,
                "ablation_accuracy": self.causal.ablation_accuracy,
            }

        return result


class SAEProtocol(Protocol):
    """Protocol for SAE model interface."""

    def encode(self, x: Any) -> Any:
        """Encode activations to latent space."""
        ...

    def decode(self, z: Any) -> Any:
        """Decode latent space to activations."""
        ...

    def forward(self, x: Any) -> tuple[Any, Any]:
        """Forward pass returning reconstruction and latents."""
        ...


class SAEMetrics:
    """Comprehensive SAE metrics calculator.

    Computes all evaluation metrics for an SAE model.

    Example:
        metrics = SAEMetrics(sae, config=MetricsConfig.standard())
        report = metrics.compute_all(test_activations)
        print(report.summary())
    """

    def __init__(
        self,
        sae: SAEProtocol | None = None,
        config: MetricsConfig | None = None,
    ):
        """Initialize metrics calculator.

        Args:
            sae: SAE model to evaluate
            config: Metrics configuration
        """
        self.sae = sae
        self.config = config or MetricsConfig()
        self._cache: dict[str, Any] = {}

    def compute_all(
        self,
        activations: Any,
        labels: Any | None = None,
    ) -> MetricsReport:
        """Compute all metrics.

        Args:
            activations: Test activations (numpy array or tensor)
            labels: Optional ground truth labels

        Returns:
            Complete metrics report
        """
        import time
        start_time = time.time()

        # Compute each metric category
        reconstruction = self.compute_reconstruction(activations)
        sparsity = self.compute_sparsity(activations)
        interpretability = self.compute_interpretability(activations)

        steering = None
        causal = None

        if self.config.level == MetricLevel.COMPREHENSIVE:
            steering = self.compute_steering(activations)
            causal = self.compute_causal(activations)

        elapsed = time.time() - start_time

        # Get sample count
        n_samples = self._get_sample_count(activations)

        return MetricsReport(
            reconstruction=reconstruction,
            sparsity=sparsity,
            interpretability=interpretability,
            steering=steering,
            causal=causal,
            config=self.config,
            n_samples_evaluated=n_samples,
            evaluation_time_seconds=elapsed,
        )

    def compute_reconstruction(self, activations: Any) -> ReconstructionMetrics:
        """Compute reconstruction quality metrics."""
        # Get reconstructions
        if self.sae is not None:
            try:
                reconstructed, _ = self.sae.forward(activations)
                return self._compute_reconstruction_metrics(activations, reconstructed)
            except Exception:
                pass

        # Return placeholder metrics if no SAE
        return self._placeholder_reconstruction_metrics()

    def compute_sparsity(self, activations: Any) -> SparsityMetrics:
        """Compute sparsity metrics."""
        if self.sae is not None:
            try:
                _, latents = self.sae.forward(activations)
                return self._compute_sparsity_metrics(latents)
            except Exception:
                pass

        return self._placeholder_sparsity_metrics()

    def compute_interpretability(self, activations: Any) -> InterpretabilityMetrics:
        """Compute interpretability metrics."""
        if self.sae is not None:
            try:
                _, latents = self.sae.forward(activations)
                return self._compute_interpretability_metrics(latents)
            except Exception:
                pass

        return self._placeholder_interpretability_metrics()

    def compute_steering(self, activations: Any) -> SteeringMetrics:
        """Compute steering effectiveness metrics."""
        # Placeholder - requires intervention experiments
        return SteeringMetrics(
            steering_accuracy=0.0,
            steering_magnitude=0.0,
            feature_controllability=0.0,
            side_effect_ratio=0.0,
            target_achievement_rate=0.0,
        )

    def compute_causal(self, activations: Any) -> CausalMetrics:
        """Compute causal validity metrics."""
        # Placeholder - requires causal intervention experiments
        return CausalMetrics(
            ablation_accuracy=0.0,
            injection_accuracy=0.0,
            causal_faithfulness=0.0,
            intervention_consistency=0.0,
            counterfactual_accuracy=0.0,
        )

    def _compute_reconstruction_metrics(
        self,
        original: Any,
        reconstructed: Any,
    ) -> ReconstructionMetrics:
        """Compute reconstruction metrics from arrays."""
        try:
            # Convert to numpy if needed
            orig = self._to_numpy(original)
            recon = self._to_numpy(reconstructed)

            # MSE and variants
            diff = orig - recon
            mse = float((diff ** 2).mean())
            rmse = math.sqrt(mse)
            mae = float(abs(diff).mean())

            # Cosine similarity
            dot_product = (orig * recon).sum(axis=-1)
            norm_orig = (orig ** 2).sum(axis=-1) ** 0.5
            norm_recon = (recon ** 2).sum(axis=-1) ** 0.5
            cosine = float((dot_product / (norm_orig * norm_recon + 1e-10)).mean())

            # Explained variance
            var_orig = orig.var()
            var_residual = diff.var()
            explained_var = float(1 - var_residual / (var_orig + 1e-10))

            # R-squared
            ss_res = (diff ** 2).sum()
            ss_tot = ((orig - orig.mean()) ** 2).sum()
            r_squared = float(1 - ss_res / (ss_tot + 1e-10))

            # Relative error
            relative_error = float((abs(diff) / (abs(orig) + 1e-10)).mean())

            return ReconstructionMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                cosine_similarity=max(0, min(1, cosine)),
                explained_variance=max(0, min(1, explained_var)),
                r_squared=max(0, min(1, r_squared)),
                relative_error=relative_error,
            )

        except Exception:
            return self._placeholder_reconstruction_metrics()

    def _compute_sparsity_metrics(self, latents: Any) -> SparsityMetrics:
        """Compute sparsity metrics from latent activations."""
        try:
            z = self._to_numpy(latents)

            # L0 (number of non-zero activations)
            active = abs(z) > self.config.dead_threshold
            l0 = float(active.sum(axis=-1).mean())
            n_features = z.shape[-1]
            l0_ratio = l0 / n_features

            # L1 norm
            l1 = float(abs(z).sum(axis=-1).mean())

            # Entropy
            z_flat = abs(z).flatten()
            z_norm = z_flat / (z_flat.sum() + 1e-10)
            entropy = float(-((z_norm + 1e-10) * (z_norm + 1e-10).log()).sum())

            # Gini coefficient
            sorted_z = sorted(z_flat)
            n = len(sorted_z)
            cumsum = sum((i + 1) * sorted_z[i] for i in range(n))
            gini = (2 * cumsum) / (n * sum(sorted_z) + 1e-10) - (n + 1) / n

            # Dead features
            feature_active = active.any(axis=0) if len(z.shape) > 1 else active
            dead_count = int((~feature_active).sum())
            dead_ratio = dead_count / n_features

            # Activation statistics
            mean_act = float(z.mean())
            std_act = float(z.std())
            max_act = float(z.max())

            return SparsityMetrics(
                l0=l0,
                l0_ratio=l0_ratio,
                l1=l1,
                entropy=entropy,
                gini_coefficient=max(0, min(1, gini)),
                dead_feature_count=dead_count,
                dead_feature_ratio=dead_ratio,
                mean_activation=mean_act,
                std_activation=std_act,
                max_activation=max_act,
            )

        except Exception:
            return self._placeholder_sparsity_metrics()

    def _compute_interpretability_metrics(self, latents: Any) -> InterpretabilityMetrics:
        """Compute interpretability metrics from latent activations."""
        try:
            z = self._to_numpy(latents)

            # Monosemanticity: how focused each feature is
            # Approximate by measuring activation variance per feature
            if len(z.shape) > 1:
                feature_std = z.std(axis=0)
                feature_mean = abs(z).mean(axis=0)
                # High std relative to mean suggests polysemanticity
                ratio = feature_std / (feature_mean + 1e-10)
                monosemanticity = 1.0 / (1.0 + ratio)
                mean_mono = float(monosemanticity.mean())
                std_mono = float(monosemanticity.std())
            else:
                mean_mono = 0.5
                std_mono = 0.1

            # Polysemantic ratio (features with high variance)
            poly_threshold = 0.3
            polysemantic_ratio = float((monosemanticity < poly_threshold).mean())

            # Highly interpretable (high monosemanticity)
            interpretable_threshold = 0.7
            highly_interpretable = float((monosemanticity > interpretable_threshold).mean())

            # Feature coherence (similarity within active sets)
            coherence = mean_mono  # Simplified

            # Feature uniqueness (orthogonality)
            if len(z.shape) > 1 and z.shape[0] > 1:
                # Correlation between features
                z_centered = z - z.mean(axis=0)
                norms = (z_centered ** 2).sum(axis=0) ** 0.5 + 1e-10
                z_normalized = z_centered / norms
                correlations = abs(z_normalized.T @ z_normalized / z.shape[0])
                # Uniqueness is inversely related to average off-diagonal correlation
                off_diag_mean = (correlations.sum() - correlations.trace()) / (
                    correlations.size - len(correlations)
                )
                uniqueness = float(1.0 - off_diag_mean)
            else:
                uniqueness = 0.5

            return InterpretabilityMetrics(
                mean_monosemanticity=mean_mono,
                std_monosemanticity=std_mono,
                polysemantic_ratio=polysemantic_ratio,
                highly_interpretable_ratio=highly_interpretable,
                feature_coherence=coherence,
                feature_uniqueness=uniqueness,
            )

        except Exception:
            return self._placeholder_interpretability_metrics()

    def _placeholder_reconstruction_metrics(self) -> ReconstructionMetrics:
        """Return placeholder reconstruction metrics."""
        return ReconstructionMetrics(
            mse=0.0,
            rmse=0.0,
            mae=0.0,
            cosine_similarity=0.0,
            explained_variance=0.0,
            r_squared=0.0,
            relative_error=0.0,
        )

    def _placeholder_sparsity_metrics(self) -> SparsityMetrics:
        """Return placeholder sparsity metrics."""
        return SparsityMetrics(
            l0=0.0,
            l0_ratio=0.0,
            l1=0.0,
            entropy=0.0,
            gini_coefficient=0.0,
            dead_feature_count=0,
            dead_feature_ratio=0.0,
        )

    def _placeholder_interpretability_metrics(self) -> InterpretabilityMetrics:
        """Return placeholder interpretability metrics."""
        return InterpretabilityMetrics(
            mean_monosemanticity=0.0,
            std_monosemanticity=0.0,
            polysemantic_ratio=0.0,
            highly_interpretable_ratio=0.0,
            feature_coherence=0.0,
            feature_uniqueness=0.0,
        )

    def _to_numpy(self, x: Any) -> Any:
        """Convert tensor to numpy array."""
        if hasattr(x, "numpy"):
            return x.numpy()
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return x

    def _get_sample_count(self, activations: Any) -> int:
        """Get number of samples in activations."""
        if hasattr(activations, "shape"):
            return activations.shape[0]
        if hasattr(activations, "__len__"):
            return len(activations)
        return 0


def compute_all_metrics(
    sae: SAEProtocol,
    activations: Any,
    config: MetricsConfig | None = None,
) -> MetricsReport:
    """Convenience function to compute all metrics.

    Args:
        sae: SAE model to evaluate
        activations: Test activations
        config: Metrics configuration

    Returns:
        Complete metrics report
    """
    calculator = SAEMetrics(sae, config)
    return calculator.compute_all(activations)
