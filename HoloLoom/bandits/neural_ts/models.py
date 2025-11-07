"""
Neural network models for contextual bandit reward prediction.

This module provides the core MLP architecture used in both Bootstrap Ensemble
and MC-Dropout backends. Models are kept small (<200k params/head) for fast
inference and training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class MLP(nn.Module):
    """
    Multi-layer perceptron for reward prediction.

    Simple feedforward network that maps (context, action) features to
    expected reward. Supports dropout for MC-Dropout posterior, GELU/ReLU
    activation, and configurable depth.

    Architecture:
        Input → [Linear → Activation → Dropout]* → Linear → Output

    Attributes:
        net: Sequential container of layers
        in_dim: Input dimension (context_dim + action_dim)
        hidden: List of hidden layer sizes
        dropout_p: Dropout probability (0.0 disables dropout)
        activation: Activation function type

    Example:
        >>> mlp = MLP(in_dim=512, hidden=[256, 128], out_dim=1, dropout_p=0.1)
        >>> x = torch.randn(32, 512)  # batch of 32 contexts
        >>> rewards = mlp(x)  # shape: [32]
    """

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        out_dim: int = 1,
        dropout_p: float = 0.0,
        activation: Literal["relu", "gelu"] = "gelu",
    ):
        """
        Initialize MLP.

        Args:
            in_dim: Input feature dimension (context_dim + action_dim)
            hidden: List of hidden layer sizes (e.g., [256, 128])
            out_dim: Output dimension (1 for scalar reward)
            dropout_p: Dropout probability (0.0-1.0)
            activation: Activation function ("relu" or "gelu")

        Raises:
            ValueError: If in_dim <= 0 or hidden is empty
        """
        super().__init__()

        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if not hidden:
            raise ValueError("hidden must contain at least one layer size")

        self.in_dim = in_dim
        self.hidden = hidden
        self.dropout_p = dropout_p
        self.activation = activation

        # Build layers
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]
        layers = []
        d = in_dim

        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(act_fn())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            d = h

        # Output layer (no activation, raw reward prediction)
        layers.append(nn.Linear(d, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features, shape [batch_size, in_dim]

        Returns:
            Predicted rewards, shape [batch_size] (squeezed from [batch_size, 1])

        Raises:
            ValueError: If input dimension doesn't match in_dim
        """
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Input dimension {x.shape[-1]} doesn't match model in_dim {self.in_dim}"
            )

        out = self.net(x)  # [batch_size, out_dim]
        return out.squeeze(-1)  # [batch_size]

    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Human-readable model summary."""
        n_params = self.num_parameters()
        return (
            f"MLP(in_dim={self.in_dim}, hidden={self.hidden}, "
            f"dropout_p={self.dropout_p}, activation={self.activation}, "
            f"params={n_params:,})"
        )


def create_mlp_ensemble(
    n_models: int,
    in_dim: int,
    hidden: list[int],
    dropout_p: float = 0.0,
    activation: Literal["relu", "gelu"] = "gelu",
    device: str = "cpu",
) -> list[MLP]:
    """
    Create ensemble of independent MLPs for Bootstrap posterior.

    Each model is initialized with different random weights.

    Args:
        n_models: Number of models in ensemble
        in_dim: Input dimension
        hidden: Hidden layer sizes
        dropout_p: Dropout probability (typically 0.0 for Bootstrap)
        activation: Activation function
        device: Device to place models on

    Returns:
        List of independent MLP models

    Example:
        >>> ensemble = create_mlp_ensemble(
        ...     n_models=7,
        ...     in_dim=512,
        ...     hidden=[256, 128],
        ...     device="cuda"
        ... )
        >>> print(len(ensemble))  # 7
        >>> print(ensemble[0].num_parameters())  # ~165k params
    """
    models = []
    for _ in range(n_models):
        mlp = MLP(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=1,
            dropout_p=dropout_p,
            activation=activation,
        )
        mlp = mlp.to(device)
        models.append(mlp)

    return models


def estimate_model_size(in_dim: int, hidden: list[int]) -> dict[str, int]:
    """
    Estimate model size without instantiating.

    Useful for config validation before creating large ensembles.

    Args:
        in_dim: Input dimension
        hidden: Hidden layer sizes

    Returns:
        Dict with 'params' (total params) and 'memory_mb' (approx memory)

    Example:
        >>> size = estimate_model_size(in_dim=512, hidden=[256, 128])
        >>> print(f"Params: {size['params']:,}, Memory: {size['memory_mb']:.1f} MB")
        Params: 165,121, Memory: 0.6 MB
    """
    total_params = 0

    d = in_dim
    for h in hidden:
        total_params += (d + 1) * h  # Linear: d*h weights + h biases
        d = h

    total_params += (d + 1) * 1  # Output layer

    # Estimate memory (4 bytes per float32 param, plus overhead)
    memory_bytes = total_params * 4 * 2  # params + gradients
    memory_mb = memory_bytes / (1024 * 1024)

    return {"params": total_params, "memory_mb": memory_mb}
