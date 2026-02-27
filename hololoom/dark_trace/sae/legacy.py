"""
Dark Trace SAE: Legacy SAE Implementation

This module contains the original basic SAE implementation for backward compatibility.
For research-grade SAE with L1 annealing, dead feature detection, and ghost gradients,
use ResearchSAE from the core module instead.

Migration Guide:
    # Old (basic SAE):
    from hololoom.dark_trace.sae import SparseAutoEncoder, DarkSaeTrainer
    sae = SparseAutoEncoder(input_dim=384)
    trainer = DarkSaeTrainer(sae)

    # New (research-grade SAE):
    from hololoom.dark_trace.sae import ResearchSAE, ResearchTrainer, TrainingConfig
    sae = ResearchSAE(input_dim=384, expansion_factor=16)
    config = TrainingConfig(l1_warmup_steps=1000, enable_ghost_gradients=True)
    trainer = ResearchTrainer(sae, config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Tuple


class SparseAutoEncoder(nn.Module):
    """
    SAE: Sparse Autoencoder for extracting interpretable features from LLM activations.

    This is the original basic implementation. For research-grade features including
    L1 annealing, dead feature detection, and ghost gradients, use ResearchSAE instead.

    Architecture:
    x -> Encoder -> ReLU -> Latents -> Decoder -> x_reconstructed

    Objective:
    - Minimize ||x - x_rec||^2 (Reconstruction)
    - Minimize L1(Latents) (Sparsity)

    Args:
        input_dim: Dimension of input activations
        expansion_factor: Multiplier for latent dimension (default: 16x)
        l1_coefficient: Sparsity regularization strength (default: 3e-4)

    Example:
        >>> sae = SparseAutoEncoder(input_dim=384)
        >>> x = torch.randn(32, 384)
        >>> recon, latents, pre_act = sae(x)
        >>> losses = sae.compute_loss(x, recon, latents)
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 16,
        l1_coefficient: float = 3e-4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        self.l1_coefficient = l1_coefficient

        # Encoder: Projects model activations into a higher-dimensional sparse space
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

        # Decoder: Projects sparse features back to model activation space
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)

        # Initialize decoder weights to be normalized
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations [batch, input_dim]

        Returns:
            Tuple of (reconstruction, latents, latents_pre_activation)
        """
        # Encode & apply ReLU for sparsity
        latents_pre_act = self.encoder(x)
        latents = F.relu(latents_pre_act)

        # Decode
        reconstruction = self.decoder(latents)

        return reconstruction, latents, latents_pre_act

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latents: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction and sparsity losses.

        Args:
            x: Original input
            reconstruction: Reconstructed output from decoder
            latents: Sparse latent activations

        Returns:
            Dict with 'loss', 'recon_loss', 'sparsity_loss'
        """
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)

        # Sparsity Loss (L1)
        sparsity_loss = self.l1_coefficient * latents.sum(dim=-1).mean()

        total_loss = recon_loss + sparsity_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss
        }

    @torch.no_grad()
    def get_active_features(
        self,
        x: torch.Tensor,
        threshold: float = 0.0
    ) -> Dict[int, float]:
        """
        Get active features above threshold for given input.

        Use this to 'read the mind' of the model for a specific token.

        Args:
            x: Input activations [batch, input_dim] or [input_dim]
            threshold: Minimum activation value to consider active

        Returns:
            Dict mapping feature_idx -> activation_value
        """
        _, latents, _ = self.forward(x)

        # Handle batched input by taking mean
        if x.dim() > 1 and x.size(0) > 1:
            latents = latents.mean(dim=0)
        else:
            latents = latents.squeeze(0)

        active_indices = torch.nonzero(latents > threshold).squeeze(-1)
        if active_indices.numel() == 0:
            return {}

        if active_indices.dim() == 0:
            # Single active feature
            return {active_indices.item(): latents[active_indices].item()}

        return {
            idx.item(): latents[idx].item()
            for idx in active_indices
        }


class DarkSaeTrainer:
    """
    Basic trainer for SAE on streaming activations.

    This is the original basic implementation. For research-grade training with
    L1 annealing, dead feature detection, and ghost gradients, use ResearchTrainer.

    Args:
        sae: SparseAutoEncoder instance to train
        lr: Learning rate (default: 1e-3)

    Example:
        >>> sae = SparseAutoEncoder(input_dim=384)
        >>> trainer = DarkSaeTrainer(sae, lr=1e-3)
        >>> losses = trainer.step(activations)
    """

    def __init__(self, sae: SparseAutoEncoder, lr: float = 1e-3):
        self.sae = sae
        self.optimizer = optim.Adam(sae.parameters(), lr=lr)

    def step(self, activations: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            activations: Batch of activations [batch, input_dim]

        Returns:
            Dict of loss values (detached as floats)
        """
        self.optimizer.zero_grad()

        recon, latents, _ = self.sae(activations)
        losses = self.sae.compute_loss(activations, recon, latents)

        losses["loss"].backward()
        self.optimizer.step()

        # Enforce unit norm on decoder weights (prevents scaling collapse)
        with torch.no_grad():
            self.sae.decoder.weight.data = F.normalize(
                self.sae.decoder.weight.data, p=2, dim=0
            )

        return {k: v.item() for k, v in losses.items()}
