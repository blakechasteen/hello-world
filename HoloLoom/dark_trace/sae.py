import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Tuple

class SparseAutoEncoder(nn.Module):
    """
    SAE: Sparse Autoencoder for extracting interpretable features from LLM activations.
    
    Architecture:
    x -> Encoder -> ReLU -> Latents -> Decoder -> x_reconstructed
    
    Objective:
    - Minimize ||x - x_rec||^2 (Reconstruction)
    - Minimize L1(Latents) (Sparsity)
    """
    def __init__(self, input_dim: int, expansion_factor: int = 16, l1_coefficient: float = 3e-4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        self.l1_coefficient = l1_coefficient
        
        # Encoder: Projects model activations into a higher-dimensional sparse space
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)
        
        # Decoder: Projects sparse features back to model activation space
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)
        
        # Tie weights (optional, but effectively commonly used initialized as tied)
        # Here we keep them separate for flexibility.
        
        # Initialize decoder weights to be normalized
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [Batch, Input_Dim]
        
        # 1. Encode & Sparsity
        # Remove mean or bias? Usually raw activations are fine.
        latents_pre_act = self.encoder(x)
        latents = F.relu(latents_pre_act)
        
        # 2. Decode
        reconstruction = self.decoder(latents)
        
        return reconstruction, latents, latents_pre_act

    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)
        
        # Sparsity Loss (L1)
        # We want meaningful, rare features.
        sparsity_loss = self.l1_coefficient * latents.sum(dim=-1).mean()
        
        # Decoder Norm Constraint (kept during optimization steps usually, or added as penalty)
        # Here we just compute the core losses
        
        total_loss = recon_loss + sparsity_loss
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss
        }
        
    @torch.no_grad()
    def get_active_features(self, x: torch.Tensor, threshold: float = 0.0) -> Dict[int, float]:
        """
        Returns a dictionary of {feature_idx: activation_value} for the most active features specific input.
        Use this to 'read the mind' of the model for a specific token.
        """
        _, latents, _ = self.forward(x)
        # Take the max over the batch or expect single input
        if x.size(0) > 1:
            latents = latents.mean(dim=0)
        else:
            latents = latents.squeeze(0)
            
        active_indices = torch.nonzero(latents > threshold).squeeze()
        if active_indices.numel() == 0:
            return {}
            
        return {idx.item(): latents[idx].item() for idx in active_indices if idx.numel() > 0} 

class DarkSaeTrainer:
    """
    Manages the training of an SAE on streaming activations.
    """
    def __init__(self, sae: SparseAutoEncoder, lr: float = 1e-3):
        self.sae = sae
        self.optimizer = optim.Adam(sae.parameters(), lr=lr)
        
    def step(self, activations: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()
        
        recon, latents, _ = self.sae(activations)
        losses = self.sae.compute_loss(activations, recon, latents)
        
        losses["loss"].backward()
        self.optimizer.step()
        
        # Enforce unit norm on decoder weights (common SAE trick to prevent scaling collapse)
        with torch.no_grad():
             self.sae.decoder.weight.data = F.normalize(self.sae.decoder.weight.data, p=2, dim=0)
             
        return {k: v.item() for k, v in losses.items()}
