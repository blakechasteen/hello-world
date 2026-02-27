"""
Research-Grade SAE Core Architecture

Enhanced Sparse Autoencoder with:
- Ghost gradients for dead feature recovery
- Per-feature activation tracking
- Feature health monitoring
- Decoder normalization (Anthropic standard)

Based on:
- "Towards Monosemanticity" (Anthropic, 2023)
- "Scaling Monosemanticity" (Anthropic, 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time


@dataclass
class FeatureHealth:
    """
    Health statistics for a single SAE feature.

    Tracks activation patterns to detect dead features and
    provide insights into feature utilization.
    """
    feature_idx: int
    activation_count: int = 0           # Total times activated (above threshold)
    total_samples: int = 0              # Total samples seen
    mean_activation: float = 0.0        # Mean activation when active
    max_activation: float = 0.0         # Maximum activation seen
    last_activation_step: int = -1      # Step when last activated
    is_dead: bool = False               # Marked dead if no activation for N steps
    consecutive_dead_batches: int = 0   # Batches without activation

    @property
    def activation_frequency(self) -> float:
        """Fraction of samples where feature activated."""
        if self.total_samples == 0:
            return 0.0
        return self.activation_count / self.total_samples

    @property
    def sparsity(self) -> float:
        """Sparsity = 1 - activation_frequency (higher = sparser)."""
        return 1.0 - self.activation_frequency

    def update(
        self,
        activated: bool,
        activation_value: float,
        current_step: int,
        dead_threshold_batches: int = 100
    ):
        """Update health stats after seeing this feature's activation."""
        self.total_samples += 1

        if activated:
            self.activation_count += 1
            self.last_activation_step = current_step
            self.consecutive_dead_batches = 0
            self.is_dead = False

            # Update running mean (Welford's algorithm)
            delta = activation_value - self.mean_activation
            self.mean_activation += delta / self.activation_count

            if activation_value > self.max_activation:
                self.max_activation = activation_value
        else:
            self.consecutive_dead_batches += 1
            if self.consecutive_dead_batches >= dead_threshold_batches:
                self.is_dead = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "feature_idx": self.feature_idx,
            "activation_count": self.activation_count,
            "total_samples": self.total_samples,
            "activation_frequency": self.activation_frequency,
            "mean_activation": self.mean_activation,
            "max_activation": self.max_activation,
            "last_activation_step": self.last_activation_step,
            "is_dead": self.is_dead,
            "consecutive_dead_batches": self.consecutive_dead_batches,
            "sparsity": self.sparsity,
        }


class ActivationTracker:
    """
    Tracks activation patterns across the SAE's latent features.

    Maintains:
    - Per-feature health statistics
    - Recent activation history (rolling buffer)
    - Dead feature detection
    - Activation distribution analysis
    """

    def __init__(
        self,
        n_features: int,
        activation_threshold: float = 1e-5,
        dead_threshold_batches: int = 100,
        history_size: int = 1000,
    ):
        """
        Args:
            n_features: Number of SAE latent features
            activation_threshold: Minimum activation to count as "active"
            dead_threshold_batches: Batches without activation to mark dead
            history_size: Size of rolling activation history
        """
        self.n_features = n_features
        self.activation_threshold = activation_threshold
        self.dead_threshold_batches = dead_threshold_batches
        self.history_size = history_size

        # Per-feature health tracking
        self.feature_health: List[FeatureHealth] = [
            FeatureHealth(feature_idx=i) for i in range(n_features)
        ]

        # Rolling history of activation counts per batch
        self.activation_history: deque = deque(maxlen=history_size)

        # Global counters
        self.current_step = 0
        self.total_batches = 0

    def update(self, latents: torch.Tensor) -> Dict[str, Any]:
        """
        Update tracking with a batch of latent activations.

        Args:
            latents: [batch_size, n_features] tensor of activations

        Returns:
            Summary statistics for this batch
        """
        self.current_step += 1
        self.total_batches += 1
        batch_size = latents.size(0)

        # Compute per-feature activation stats for this batch
        with torch.no_grad():
            # Which features activated in this batch?
            max_per_feature = latents.max(dim=0).values  # [n_features]
            mean_per_feature = latents.mean(dim=0)       # [n_features]
            active_mask = max_per_feature > self.activation_threshold

            # Count activations per feature across batch
            activations_per_feature = (latents > self.activation_threshold).sum(dim=0)

            # Update per-feature health
            for i in range(self.n_features):
                activated = active_mask[i].item()
                activation_value = mean_per_feature[i].item() if activated else 0.0
                self.feature_health[i].update(
                    activated=activated,
                    activation_value=activation_value,
                    current_step=self.current_step,
                    dead_threshold_batches=self.dead_threshold_batches,
                )

            # Record batch statistics
            n_active = active_mask.sum().item()
            batch_stats = {
                "step": self.current_step,
                "n_active_features": n_active,
                "active_fraction": n_active / self.n_features,
                "mean_activation": latents.mean().item(),
                "max_activation": latents.max().item(),
                "l0_sparsity": (latents > self.activation_threshold).float().mean().item(),
            }
            self.activation_history.append(batch_stats)

            return batch_stats

    def get_dead_features(self) -> List[int]:
        """Return indices of features marked as dead."""
        return [h.feature_idx for h in self.feature_health if h.is_dead]

    def get_alive_features(self) -> List[int]:
        """Return indices of features that are still active."""
        return [h.feature_idx for h in self.feature_health if not h.is_dead]

    def get_feature_health(self, feature_idx: int) -> FeatureHealth:
        """Get health stats for a specific feature."""
        return self.feature_health[feature_idx]

    def get_summary(self) -> Dict[str, Any]:
        """Get overall tracking summary."""
        dead_features = self.get_dead_features()
        alive_features = self.get_alive_features()

        # Compute distribution stats
        frequencies = [h.activation_frequency for h in self.feature_health]
        mean_activations = [h.mean_activation for h in self.feature_health if h.activation_count > 0]

        return {
            "n_features": self.n_features,
            "n_dead": len(dead_features),
            "n_alive": len(alive_features),
            "dead_fraction": len(dead_features) / self.n_features if self.n_features > 0 else 0.0,
            "mean_activation_frequency": sum(frequencies) / len(frequencies) if frequencies else 0.0,
            "mean_activation_magnitude": sum(mean_activations) / len(mean_activations) if mean_activations else 0.0,
            "total_batches": self.total_batches,
            "dead_feature_indices": dead_features[:10],  # First 10 for brevity
        }


class ResearchSAE(nn.Module):
    """
    Research-Grade Sparse Autoencoder.

    Extends basic SAE with:
    - Ghost gradients: Allows dead features to receive gradient signals
    - Activation tracking: Per-feature health monitoring
    - Auxk loss: Additional loss term for dead feature recovery
    - Decoder normalization: Per Anthropic's methodology

    Architecture:
        x -> Encoder -> ReLU (+ ghost) -> Latents -> Decoder -> x_reconstructed

    Ghost Gradient Mechanism:
        For features marked as dead, we add a small gradient signal:
        activation = ReLU(pre_act) + ghost_coef * max(0, -pre_act) * dead_mask

        This allows dead features to receive gradients during backprop,
        giving them a chance to "wake up" and become useful again.
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 16,
        l1_coefficient: float = 3e-4,
        ghost_coefficient: float = 0.01,
        activation_threshold: float = 1e-5,
        dead_threshold_batches: int = 100,
        enable_ghost_gradients: bool = True,
        normalize_decoder: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input activations
            expansion_factor: Latent dim = input_dim * expansion_factor
            l1_coefficient: L1 sparsity penalty coefficient
            ghost_coefficient: Coefficient for ghost gradient term (α in paper)
            activation_threshold: Minimum activation to count as "active"
            dead_threshold_batches: Batches without activation to mark dead
            enable_ghost_gradients: Whether to use ghost gradient mechanism
            normalize_decoder: Whether to normalize decoder columns after updates
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        self.expansion_factor = expansion_factor
        self.l1_coefficient = l1_coefficient
        self.ghost_coefficient = ghost_coefficient
        self.activation_threshold = activation_threshold
        self.enable_ghost_gradients = enable_ghost_gradients
        self.normalize_decoder = normalize_decoder

        # Encoder: Projects model activations into sparse space
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

        # Decoder: Projects sparse features back to activation space
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)

        # Initialize decoder with normalized columns
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

        # Activation tracker for feature health monitoring
        self.tracker = ActivationTracker(
            n_features=self.latent_dim,
            activation_threshold=activation_threshold,
            dead_threshold_batches=dead_threshold_batches,
        )

        # Track dead features for ghost gradients
        self._dead_feature_mask: Optional[torch.Tensor] = None

    def _update_dead_mask(self):
        """Update the dead feature mask from tracker."""
        dead_indices = self.tracker.get_dead_features()
        device = self.encoder.weight.device

        self._dead_feature_mask = torch.zeros(self.latent_dim, device=device)
        if dead_indices:
            self._dead_feature_mask[dead_indices] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        return_pre_activations: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional ghost gradients.

        Args:
            x: [batch_size, input_dim] input activations
            return_pre_activations: Whether to return pre-ReLU activations

        Returns:
            reconstruction: [batch_size, input_dim] reconstructed activations
            latents: [batch_size, latent_dim] sparse latent activations
            latents_pre_act: [batch_size, latent_dim] pre-ReLU (if requested)
        """
        # Encode
        latents_pre_act = self.encoder(x)

        # ReLU activation
        latents = F.relu(latents_pre_act)

        # Ghost gradient mechanism for dead features
        if self.enable_ghost_gradients and self._dead_feature_mask is not None:
            # For dead features, add small gradient-enabling term
            # ghost_term = α * max(0, -pre_act) * dead_mask
            ghost_term = self.ghost_coefficient * F.relu(-latents_pre_act)
            ghost_term = ghost_term * self._dead_feature_mask.unsqueeze(0)
            latents = latents + ghost_term

        # Decode
        reconstruction = self.decoder(latents)

        if return_pre_activations:
            return reconstruction, latents, latents_pre_act
        return reconstruction, latents, None

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latents: torch.Tensor,
        l1_coefficient: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            x: Original input
            reconstruction: Reconstructed input
            latents: Sparse latent activations
            l1_coefficient: Override L1 coefficient (for annealing)

        Returns:
            Dictionary of loss components
        """
        l1_coef = l1_coefficient if l1_coefficient is not None else self.l1_coefficient

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)

        # Sparsity loss (L1 on activations)
        # Sum over features, mean over batch
        sparsity_loss = l1_coef * latents.sum(dim=-1).mean()

        # Total loss
        total_loss = recon_loss + sparsity_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "l1_coefficient": torch.tensor(l1_coef),
        }

    def compute_auxk_loss(
        self,
        latents: torch.Tensor,
        k: int = 32,
    ) -> torch.Tensor:
        """
        Compute auxiliary top-k loss for dead feature recovery.

        This loss encourages the top-k dead features to activate,
        providing gradient signal to help them recover.

        Args:
            latents: [batch_size, latent_dim] sparse activations
            k: Number of top dead features to encourage

        Returns:
            Auxiliary loss scalar
        """
        if self._dead_feature_mask is None or self._dead_feature_mask.sum() == 0:
            return torch.tensor(0.0, device=latents.device)

        # Get activations of dead features only
        dead_mask = self._dead_feature_mask.bool()
        dead_activations = latents[:, dead_mask]

        if dead_activations.numel() == 0:
            return torch.tensor(0.0, device=latents.device)

        # Encourage top-k dead features to activate
        n_dead = dead_activations.size(1)
        k_actual = min(k, n_dead)

        if k_actual == 0:
            return torch.tensor(0.0, device=latents.device)

        # Take mean of top-k dead feature activations and negate
        # (we want to maximize activation, so minimize negative)
        topk_values, _ = dead_activations.topk(k_actual, dim=1)
        auxk_loss = -topk_values.mean()

        return auxk_loss

    def update_tracking(self, latents: torch.Tensor) -> Dict[str, Any]:
        """
        Update activation tracking with a batch of latents.

        Call this after each forward pass during training.
        """
        stats = self.tracker.update(latents.detach())
        self._update_dead_mask()
        return stats

    def normalize_decoder_weights(self):
        """Normalize decoder columns to unit norm (Anthropic standard)."""
        if self.normalize_decoder:
            with torch.no_grad():
                self.decoder.weight.data = F.normalize(
                    self.decoder.weight.data, p=2, dim=0
                )

    @torch.no_grad()
    def get_active_features(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Get active features for an input.

        Args:
            x: [batch_size, input_dim] or [input_dim] input
            threshold: Activation threshold (default: self.activation_threshold)
            top_k: If set, return only top-k features

        Returns:
            Dictionary of {feature_idx: activation_value}
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        _, latents, _ = self.forward(x)

        # Average over batch if multiple samples
        if latents.size(0) > 1:
            latents = latents.mean(dim=0)
        else:
            latents = latents.squeeze(0)

        threshold = threshold if threshold is not None else self.activation_threshold

        # Find active features
        active_mask = latents > threshold
        active_indices = torch.nonzero(active_mask).squeeze(-1)

        if active_indices.numel() == 0:
            return {}

        # Build result dictionary
        results = {}
        for idx in active_indices:
            idx_item = idx.item()
            results[idx_item] = latents[idx_item].item()

        # Optionally limit to top-k
        if top_k is not None and len(results) > top_k:
            sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
            results = dict(sorted_items[:top_k])

        return results

    def get_feature_vectors(self, feature_indices: List[int]) -> torch.Tensor:
        """
        Get decoder vectors for specified features.

        These vectors represent the "direction" each feature encodes
        in the original activation space.

        Args:
            feature_indices: List of feature indices

        Returns:
            [len(feature_indices), input_dim] tensor of feature vectors
        """
        # Decoder weight is [input_dim, latent_dim]
        # Each column is a feature vector
        return self.decoder.weight[:, feature_indices].T

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of feature health statistics."""
        return self.tracker.get_summary()

    def state_dict_extended(self) -> Dict[str, Any]:
        """Get extended state dict including tracker state."""
        return {
            "model_state_dict": self.state_dict(),
            "tracker_summary": self.tracker.get_summary(),
            "feature_health": [h.to_dict() for h in self.tracker.feature_health],
            "config": {
                "input_dim": self.input_dim,
                "expansion_factor": self.expansion_factor,
                "l1_coefficient": self.l1_coefficient,
                "ghost_coefficient": self.ghost_coefficient,
                "activation_threshold": self.activation_threshold,
                "enable_ghost_gradients": self.enable_ghost_gradients,
            }
        }
