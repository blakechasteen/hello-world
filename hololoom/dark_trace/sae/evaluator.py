"""
SAE Evaluation Metrics

Comprehensive evaluation suite for Sparse Autoencoders:
- Reconstruction quality (MSE, cosine similarity, explained variance)
- Sparsity metrics (L0, L1, activation entropy)
- Feature health (dead features, utilization, polysemanticity)
- Overall quality scoring

These metrics help assess whether an SAE has successfully learned
monosemantic, interpretable features.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import math

from hololoom.dark_trace.sae.core import ResearchSAE


@dataclass
class ReconstructionMetrics:
    """
    Metrics measuring how well the SAE reconstructs inputs.

    Good reconstruction means the SAE preserves information
    while achieving sparsity in the latent space.
    """
    mse: float = 0.0                    # Mean squared error
    rmse: float = 0.0                   # Root mean squared error
    cosine_similarity: float = 0.0       # Average cosine sim between x and x_hat
    explained_variance: float = 0.0      # 1 - Var(x - x_hat) / Var(x)
    relative_error: float = 0.0          # ||x - x_hat|| / ||x||
    n_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "cosine_similarity": self.cosine_similarity,
            "explained_variance": self.explained_variance,
            "relative_error": self.relative_error,
            "n_samples": self.n_samples,
        }


@dataclass
class SparsityMetrics:
    """
    Metrics measuring sparsity of latent activations.

    Good SAEs achieve high sparsity (few features active per input)
    while maintaining reconstruction quality.
    """
    l0_sparsity: float = 0.0             # Fraction of features active (>threshold)
    l0_count: float = 0.0                # Average number of active features
    l1_norm: float = 0.0                 # Average L1 norm of activations
    activation_entropy: float = 0.0       # Entropy of activation distribution
    max_activation: float = 0.0           # Maximum activation seen
    mean_activation: float = 0.0          # Mean activation (when active)
    activation_std: float = 0.0           # Std of activations (when active)
    n_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "l0_sparsity": self.l0_sparsity,
            "l0_count": self.l0_count,
            "l1_norm": self.l1_norm,
            "activation_entropy": self.activation_entropy,
            "max_activation": self.max_activation,
            "mean_activation": self.mean_activation,
            "activation_std": self.activation_std,
            "n_samples": self.n_samples,
        }


@dataclass
class HealthMetrics:
    """
    Metrics measuring feature health and utilization.

    Healthy SAEs have:
    - Few dead features (most features activate sometimes)
    - Good utilization (features activate on distinct inputs)
    - Low polysemanticity (features represent single concepts)
    """
    n_features: int = 0
    n_dead: int = 0                       # Features that never activate
    n_alive: int = 0                      # Features that activate sometimes
    dead_fraction: float = 0.0            # n_dead / n_features
    mean_activation_frequency: float = 0.0  # Average frequency per feature
    feature_utilization: float = 0.0       # Fraction of features used per sample
    polysemanticity_score: float = 0.0     # Co-activation measure (lower = better)
    feature_orthogonality: float = 0.0     # Decoder column orthogonality

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_features": self.n_features,
            "n_dead": self.n_dead,
            "n_alive": self.n_alive,
            "dead_fraction": self.dead_fraction,
            "mean_activation_frequency": self.mean_activation_frequency,
            "feature_utilization": self.feature_utilization,
            "polysemanticity_score": self.polysemanticity_score,
            "feature_orthogonality": self.feature_orthogonality,
        }


@dataclass
class SAEEvaluationReport:
    """
    Complete SAE evaluation report.

    Combines all metrics with an overall quality score.
    """
    reconstruction: ReconstructionMetrics = field(default_factory=ReconstructionMetrics)
    sparsity: SparsityMetrics = field(default_factory=SparsityMetrics)
    health: HealthMetrics = field(default_factory=HealthMetrics)

    # Overall scores (0-1, higher is better)
    reconstruction_score: float = 0.0     # Based on explained variance + cosine sim
    sparsity_score: float = 0.0           # Based on L0 and entropy
    health_score: float = 0.0             # Based on dead fraction and utilization
    overall_score: float = 0.0            # Weighted combination

    # Metadata
    n_samples_evaluated: int = 0
    evaluation_time_ms: float = 0.0
    model_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconstruction": self.reconstruction.to_dict(),
            "sparsity": self.sparsity.to_dict(),
            "health": self.health.to_dict(),
            "scores": {
                "reconstruction": self.reconstruction_score,
                "sparsity": self.sparsity_score,
                "health": self.health_score,
                "overall": self.overall_score,
            },
            "metadata": {
                "n_samples_evaluated": self.n_samples_evaluated,
                "evaluation_time_ms": self.evaluation_time_ms,
                "model_config": self.model_config,
            },
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "SAE Evaluation Report",
            "=" * 50,
            "",
            f"Overall Score: {self.overall_score:.2f}/1.00",
            f"  Reconstruction: {self.reconstruction_score:.2f}",
            f"  Sparsity:       {self.sparsity_score:.2f}",
            f"  Health:         {self.health_score:.2f}",
            "",
            "Reconstruction Metrics:",
            f"  MSE:              {self.reconstruction.mse:.6f}",
            f"  Cosine Similarity: {self.reconstruction.cosine_similarity:.4f}",
            f"  Explained Var:    {self.reconstruction.explained_variance:.4f}",
            "",
            "Sparsity Metrics:",
            f"  L0 (active frac): {self.sparsity.l0_sparsity:.4f}",
            f"  L0 (count):       {self.sparsity.l0_count:.1f} features",
            f"  Activation Entropy: {self.sparsity.activation_entropy:.4f}",
            "",
            "Health Metrics:",
            f"  Dead Features:    {self.health.n_dead}/{self.health.n_features} ({self.health.dead_fraction:.1%})",
            f"  Utilization:      {self.health.feature_utilization:.4f}",
            f"  Polysemanticity:  {self.health.polysemanticity_score:.4f}",
            f"  Orthogonality:    {self.health.feature_orthogonality:.4f}",
            "",
            f"Samples Evaluated: {self.n_samples_evaluated}",
            f"Evaluation Time:   {self.evaluation_time_ms:.1f}ms",
            "=" * 50,
        ]
        return "\n".join(lines)


class SAEEvaluator:
    """
    Evaluates SAE quality across multiple dimensions.

    Usage:
        evaluator = SAEEvaluator(activation_threshold=1e-5)
        report = evaluator.evaluate(sae, test_activations)
        print(report.summary())
    """

    def __init__(
        self,
        activation_threshold: float = 1e-5,
        reconstruction_weight: float = 0.4,
        sparsity_weight: float = 0.3,
        health_weight: float = 0.3,
    ):
        """
        Args:
            activation_threshold: Threshold for "active" features
            reconstruction_weight: Weight for reconstruction in overall score
            sparsity_weight: Weight for sparsity in overall score
            health_weight: Weight for health in overall score
        """
        self.activation_threshold = activation_threshold
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.health_weight = health_weight

    def evaluate(
        self,
        sae: ResearchSAE,
        activations: torch.Tensor,
        batch_size: int = 128,
    ) -> SAEEvaluationReport:
        """
        Evaluate SAE on a dataset of activations.

        Args:
            sae: The SAE model to evaluate
            activations: [n_samples, input_dim] activation dataset
            batch_size: Batch size for evaluation

        Returns:
            Complete evaluation report
        """
        import time
        start_time = time.time()

        sae.eval()
        n_samples = activations.size(0)
        device = next(sae.parameters()).device

        # Accumulators
        all_recons = []
        all_latents = []
        all_originals = []

        # Process in batches
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = activations[i:i + batch_size].to(device)
                recon, latents, _ = sae.forward(batch)

                all_recons.append(recon.cpu())
                all_latents.append(latents.cpu())
                all_originals.append(batch.cpu())

        # Concatenate
        all_recons = torch.cat(all_recons, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        all_originals = torch.cat(all_originals, dim=0)

        # Compute metrics
        recon_metrics = self._compute_reconstruction_metrics(
            all_originals, all_recons
        )
        sparsity_metrics = self._compute_sparsity_metrics(all_latents)
        health_metrics = self._compute_health_metrics(sae, all_latents)

        # Compute scores
        recon_score = self._score_reconstruction(recon_metrics)
        sparsity_score = self._score_sparsity(sparsity_metrics, sae.latent_dim)
        health_score = self._score_health(health_metrics)

        overall_score = (
            self.reconstruction_weight * recon_score +
            self.sparsity_weight * sparsity_score +
            self.health_weight * health_score
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return SAEEvaluationReport(
            reconstruction=recon_metrics,
            sparsity=sparsity_metrics,
            health=health_metrics,
            reconstruction_score=recon_score,
            sparsity_score=sparsity_score,
            health_score=health_score,
            overall_score=overall_score,
            n_samples_evaluated=n_samples,
            evaluation_time_ms=elapsed_ms,
            model_config={
                "input_dim": sae.input_dim,
                "latent_dim": sae.latent_dim,
                "expansion_factor": sae.expansion_factor,
                "l1_coefficient": sae.l1_coefficient,
            },
        )

    def _compute_reconstruction_metrics(
        self,
        originals: torch.Tensor,
        reconstructions: torch.Tensor,
    ) -> ReconstructionMetrics:
        """Compute reconstruction quality metrics."""
        # MSE and RMSE
        mse = F.mse_loss(reconstructions, originals).item()
        rmse = math.sqrt(mse)

        # Cosine similarity
        orig_flat = originals.view(originals.size(0), -1)
        recon_flat = reconstructions.view(reconstructions.size(0), -1)
        cos_sim = F.cosine_similarity(orig_flat, recon_flat, dim=1).mean().item()

        # Explained variance: 1 - Var(residual) / Var(original)
        residual = originals - reconstructions
        var_residual = residual.var().item()
        var_original = originals.var().item()
        explained_var = 1 - (var_residual / (var_original + 1e-8))
        explained_var = max(0.0, min(1.0, explained_var))  # Clip to [0, 1]

        # Relative error
        orig_norm = originals.norm(dim=-1).mean().item()
        residual_norm = residual.norm(dim=-1).mean().item()
        relative_error = residual_norm / (orig_norm + 1e-8)

        return ReconstructionMetrics(
            mse=mse,
            rmse=rmse,
            cosine_similarity=cos_sim,
            explained_variance=explained_var,
            relative_error=relative_error,
            n_samples=originals.size(0),
        )

    def _compute_sparsity_metrics(
        self,
        latents: torch.Tensor,
    ) -> SparsityMetrics:
        """Compute sparsity metrics on latent activations."""
        n_samples, n_features = latents.shape

        # L0: fraction of active features
        active_mask = latents > self.activation_threshold
        l0_per_sample = active_mask.float().mean(dim=1)
        l0_sparsity = l0_per_sample.mean().item()
        l0_count = active_mask.float().sum(dim=1).mean().item()

        # L1 norm
        l1_norm = latents.sum(dim=1).mean().item()

        # Activation statistics (when active)
        active_values = latents[active_mask]
        if active_values.numel() > 0:
            max_activation = latents.max().item()
            mean_activation = active_values.mean().item()
            activation_std = active_values.std().item()
        else:
            max_activation = 0.0
            mean_activation = 0.0
            activation_std = 0.0

        # Activation entropy: how uniformly distributed are activations?
        # Normalized per-feature activation frequencies
        feature_activation_freq = active_mask.float().mean(dim=0)  # [n_features]
        freq_normalized = feature_activation_freq / (feature_activation_freq.sum() + 1e-8)
        entropy = -(freq_normalized * torch.log(freq_normalized + 1e-8)).sum().item()
        # Normalize by max entropy (log(n_features))
        max_entropy = math.log(n_features)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return SparsityMetrics(
            l0_sparsity=l0_sparsity,
            l0_count=l0_count,
            l1_norm=l1_norm,
            activation_entropy=normalized_entropy,
            max_activation=max_activation,
            mean_activation=mean_activation,
            activation_std=activation_std,
            n_samples=n_samples,
        )

    def _compute_health_metrics(
        self,
        sae: ResearchSAE,
        latents: torch.Tensor,
    ) -> HealthMetrics:
        """Compute feature health metrics."""
        n_samples, n_features = latents.shape

        # Dead/alive features
        active_mask = latents > self.activation_threshold
        feature_ever_active = active_mask.any(dim=0)  # [n_features]
        n_alive = feature_ever_active.sum().item()
        n_dead = n_features - n_alive
        dead_fraction = n_dead / n_features

        # Activation frequency per feature
        activation_freq = active_mask.float().mean(dim=0)  # [n_features]
        mean_activation_frequency = activation_freq.mean().item()

        # Feature utilization: average fraction of features used per sample
        feature_utilization = active_mask.float().mean().item()

        # Polysemanticity: measure of co-activation
        # High co-activation suggests features aren't monosemantic
        # Compute mean pairwise co-activation rate
        polysemanticity = self._compute_polysemanticity(active_mask)

        # Decoder orthogonality: how orthogonal are decoder columns?
        orthogonality = self._compute_decoder_orthogonality(sae)

        return HealthMetrics(
            n_features=n_features,
            n_dead=n_dead,
            n_alive=n_alive,
            dead_fraction=dead_fraction,
            mean_activation_frequency=mean_activation_frequency,
            feature_utilization=feature_utilization,
            polysemanticity_score=polysemanticity,
            feature_orthogonality=orthogonality,
        )

    def _compute_polysemanticity(
        self,
        active_mask: torch.Tensor,
        max_pairs: int = 1000,
    ) -> float:
        """
        Compute polysemanticity score based on feature co-activation.

        Lower is better (features activate independently).
        """
        n_samples, n_features = active_mask.shape

        # For efficiency, sample pairs of features
        if n_features > 100:
            # Sample random pairs
            indices = torch.randperm(n_features)[:100]
            active_mask = active_mask[:, indices]
            n_features = 100

        # Compute co-activation matrix
        # co_activation[i,j] = P(feature_i and feature_j both active)
        active_float = active_mask.float()
        co_activation = (active_float.T @ active_float) / n_samples

        # Remove diagonal (self-activation)
        mask = ~torch.eye(n_features, dtype=torch.bool, device=co_activation.device)
        off_diagonal = co_activation[mask]

        # Polysemanticity = mean off-diagonal co-activation
        return off_diagonal.mean().item()

    def _compute_decoder_orthogonality(self, sae: ResearchSAE) -> float:
        """
        Compute orthogonality of decoder columns.

        Perfect orthogonality = 1.0, complete overlap = 0.0.
        """
        with torch.no_grad():
            # Decoder weight: [input_dim, latent_dim]
            W = sae.decoder.weight.data

            # Normalize columns
            W_normalized = F.normalize(W, p=2, dim=0)

            # Compute gram matrix (dot products between columns)
            gram = W_normalized.T @ W_normalized  # [latent_dim, latent_dim]

            # Off-diagonal elements measure non-orthogonality
            n = gram.size(0)
            mask = ~torch.eye(n, dtype=torch.bool, device=gram.device)
            off_diagonal = gram[mask].abs()

            # Mean off-diagonal (0 = perfectly orthogonal)
            mean_overlap = off_diagonal.mean().item()

            # Convert to orthogonality score (1 = perfect)
            orthogonality = 1.0 - mean_overlap

        return orthogonality

    def _score_reconstruction(self, metrics: ReconstructionMetrics) -> float:
        """Convert reconstruction metrics to 0-1 score."""
        # Combine explained variance and cosine similarity
        ev_score = max(0.0, metrics.explained_variance)  # Already 0-1
        cos_score = (metrics.cosine_similarity + 1) / 2  # Map [-1, 1] to [0, 1]

        return 0.6 * ev_score + 0.4 * cos_score

    def _score_sparsity(self, metrics: SparsityMetrics, n_features: int) -> float:
        """Convert sparsity metrics to 0-1 score."""
        # Target: low L0 (few features active), high entropy (uniform usage)

        # L0 score: want L0 around 0.01-0.05 (1-5% of features)
        # Score decreases if too sparse (nothing active) or too dense
        target_l0 = 0.03
        l0_deviation = abs(metrics.l0_sparsity - target_l0)
        l0_score = max(0.0, 1.0 - l0_deviation * 10)  # Penalize deviation

        # Entropy score: higher is better (uniform feature usage)
        entropy_score = metrics.activation_entropy  # Already normalized to [0, 1]

        return 0.6 * l0_score + 0.4 * entropy_score

    def _score_health(self, metrics: HealthMetrics) -> float:
        """Convert health metrics to 0-1 score."""
        # Dead fraction: want low (most features alive)
        alive_score = 1.0 - metrics.dead_fraction

        # Polysemanticity: want low (features are monosemantic)
        poly_score = 1.0 - min(1.0, metrics.polysemanticity_score * 10)

        # Orthogonality: want high
        orth_score = metrics.feature_orthogonality

        return 0.4 * alive_score + 0.3 * poly_score + 0.3 * orth_score
