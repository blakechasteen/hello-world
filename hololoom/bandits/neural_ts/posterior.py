"""
Posterior distributions for Thompson Sampling.

This module provides uncertainty quantification backends:
- BootstrapPosterior: Ensemble of MLPs trained on bootstrap resamples
- MCDropoutPosterior: Single MLP with dropout-based uncertainty

Both implement the Posterior protocol for Thompson Sampling.
"""

import random
import torch
import torch.nn as nn
from typing import Literal
from HoloLoom.bandits.neural_ts.models import MLP


class BootstrapPosterior:
    """
    Bootstrap ensemble posterior for Thompson Sampling.

    Maintains N independent MLPs, each trained on bootstrap resamples of the
    replay buffer. Thompson Sampling samples one model uniformly at random.

    This is the **default backend** - simple, robust, well-calibrated.

    Attributes:
        models: List of N independent MLPs
        device: Device models are on

    Example:
        >>> from HoloLoom.bandits.neural_ts.models import create_mlp_ensemble
        >>> models = create_mlp_ensemble(n_models=7, in_dim=512, hidden=[256, 128])
        >>> posterior = BootstrapPosterior(models)
        >>>
        >>> # Thompson Sampling: sample one model
        >>> f_theta = posterior.sample_fn()
        >>> x = torch.randn(10, 512)  # 10 candidates
        >>> rewards = f_theta(x)  # Greedy prediction with sampled model
        >>> best_idx = rewards.argmax()
    """

    def __init__(self, models: list[MLP]):
        """
        Initialize Bootstrap posterior.

        Args:
            models: List of independent MLP models (all same architecture)

        Raises:
            ValueError: If models list is empty
        """
        if not models:
            raise ValueError("models list cannot be empty")

        self.models = models
        self.device = next(models[0].parameters()).device

    def sample_fn(self) -> nn.Module:
        """
        Sample one model uniformly for Thompson Sampling.

        Returns:
            One MLP from the ensemble in eval mode

        Note:
            Each call returns a potentially different model, providing
            exploration through posterior uncertainty.
        """
        model = random.choice(self.models)
        model.eval()
        return model

    def ensemble_predict(
        self,
        x: torch.Tensor,
        mode: Literal["mean", "samples"] = "mean"
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Predict with full ensemble (for evaluation, not Thompson Sampling).

        Args:
            x: Input features, shape [batch_size, in_dim]
            mode: "mean" returns ensemble average, "samples" returns all predictions

        Returns:
            If mode="mean": shape [batch_size]
            If mode="samples": list of N tensors, each [batch_size]

        Example:
            >>> x = torch.randn(5, 512)
            >>> mean_pred = posterior.ensemble_predict(x, mode="mean")
            >>> print(mean_pred.shape)  # [5]
            >>>
            >>> samples = posterior.ensemble_predict(x, mode="samples")
            >>> print(len(samples))  # N ensemble members
        """
        with torch.no_grad():
            predictions = []
            for model in self.models:
                model.eval()
                pred = model(x)
                predictions.append(pred)

        if mode == "mean":
            return torch.stack(predictions).mean(dim=0)
        else:
            return predictions

    def epistemic_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic uncertainty (ensemble disagreement).

        Args:
            x: Input features, shape [batch_size, in_dim]

        Returns:
            Standard deviation across ensemble, shape [batch_size]

        Note:
            High uncertainty indicates areas where the model is unsure,
            useful for active learning and exploration diagnostics.

        Example:
            >>> x = torch.randn(5, 512)
            >>> uncertainty = posterior.epistemic_uncertainty(x)
            >>> print(uncertainty)  # Higher values = more disagreement
        """
        with torch.no_grad():
            predictions = []
            for model in self.models:
                model.eval()
                pred = model(x)
                predictions.append(pred)

            stacked = torch.stack(predictions)  # [N, batch_size]
            std = stacked.std(dim=0)  # [batch_size]

        return std

    def __len__(self) -> int:
        """Number of models in ensemble."""
        return len(self.models)

    def __repr__(self) -> str:
        n = len(self.models)
        params = self.models[0].num_parameters()
        return f"BootstrapPosterior(n_models={n}, params_per_model={params:,})"


class MCDropoutPosterior:
    """
    MC-Dropout posterior for Thompson Sampling.

    Single MLP with dropout enabled at inference time. Each forward pass
    samples a different function (dropout mask), providing uncertainty
    through stochastic predictions.

    More parameter-efficient than Bootstrap (1 model vs N), but may be
    less calibrated.

    Attributes:
        model: Single MLP with dropout layers
        n_samples: Number of forward passes for uncertainty estimation

    Example:
        >>> from HoloLoom.bandits.neural_ts.models import MLP
        >>> model = MLP(in_dim=512, hidden=[256, 128], dropout_p=0.1)
        >>> posterior = MCDropoutPosterior(model, n_samples=10)
        >>>
        >>> # Thompson Sampling: one stochastic forward
        >>> f_theta = posterior.sample_fn()
        >>> x = torch.randn(10, 512)
        >>> rewards = f_theta(x)  # Dropout ON, stochastic
        >>> best_idx = rewards.argmax()
    """

    def __init__(self, model: MLP, n_samples: int = 10):
        """
        Initialize MC-Dropout posterior.

        Args:
            model: MLP with dropout_p > 0
            n_samples: Number of forward passes for uncertainty estimation

        Raises:
            ValueError: If model has no dropout layers
        """
        if model.dropout_p == 0.0:
            raise ValueError(
                f"Model must have dropout_p > 0 for MC-Dropout, got {model.dropout_p}"
            )

        self.model = model
        self.n_samples = n_samples
        self.device = next(model.parameters()).device

    def sample_fn(self) -> nn.Module:
        """
        Return model in train mode (dropout ON) for Thompson Sampling.

        Returns:
            The model with dropout enabled

        Note:
            This is NOT a typical eval() call - we want stochastic behavior!
            Each forward pass samples a different dropout mask.
        """
        self.model.train()  # Dropout ON
        return self.model

    def mc_predict(
        self,
        x: torch.Tensor,
        n_samples: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo prediction with multiple forward passes.

        Args:
            x: Input features, shape [batch_size, in_dim]
            n_samples: Number of MC samples (defaults to self.n_samples)

        Returns:
            Tuple of (mean, std), both shape [batch_size]

        Example:
            >>> x = torch.randn(5, 512)
            >>> mean, std = posterior.mc_predict(x, n_samples=20)
            >>> print(mean.shape, std.shape)  # both [5]
        """
        if n_samples is None:
            n_samples = self.n_samples

        self.model.train()  # Dropout ON

        with torch.no_grad():
            predictions = []
            for _ in range(n_samples):
                pred = self.model(x)
                predictions.append(pred)

            stacked = torch.stack(predictions)  # [n_samples, batch_size]
            mean = stacked.mean(dim=0)
            std = stacked.std(dim=0)

        return mean, std

    def epistemic_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic uncertainty via MC-Dropout.

        Args:
            x: Input features, shape [batch_size, in_dim]

        Returns:
            Standard deviation across MC samples, shape [batch_size]
        """
        _, std = self.mc_predict(x)
        return std

    def __repr__(self) -> str:
        params = self.model.num_parameters()
        return (
            f"MCDropoutPosterior(params={params:,}, "
            f"dropout_p={self.model.dropout_p}, n_samples={self.n_samples})"
        )


def create_bootstrap_posterior(
    n_models: int,
    in_dim: int,
    hidden: list[int],
    activation: Literal["relu", "gelu"] = "gelu",
    device: str = "cpu",
) -> BootstrapPosterior:
    """
    Factory for Bootstrap posterior.

    Args:
        n_models: Number of ensemble members
        in_dim: Input dimension
        hidden: Hidden layer sizes
        activation: Activation function
        device: Device to place models on

    Returns:
        Initialized BootstrapPosterior

    Example:
        >>> posterior = create_bootstrap_posterior(
        ...     n_models=7,
        ...     in_dim=512,
        ...     hidden=[256, 128],
        ...     device="cuda"
        ... )
    """
    from HoloLoom.bandits.neural_ts.models import create_mlp_ensemble

    models = create_mlp_ensemble(
        n_models=n_models,
        in_dim=in_dim,
        hidden=hidden,
        dropout_p=0.0,  # Bootstrap doesn't use dropout
        activation=activation,
        device=device,
    )

    return BootstrapPosterior(models)


def create_mc_dropout_posterior(
    in_dim: int,
    hidden: list[int],
    dropout_p: float = 0.1,
    activation: Literal["relu", "gelu"] = "gelu",
    n_samples: int = 10,
    device: str = "cpu",
) -> MCDropoutPosterior:
    """
    Factory for MC-Dropout posterior.

    Args:
        in_dim: Input dimension
        hidden: Hidden layer sizes
        dropout_p: Dropout probability (typically 0.1)
        activation: Activation function
        n_samples: MC samples for uncertainty estimation
        device: Device to place model on

    Returns:
        Initialized MCDropoutPosterior

    Example:
        >>> posterior = create_mc_dropout_posterior(
        ...     in_dim=512,
        ...     hidden=[256, 128],
        ...     dropout_p=0.1,
        ...     device="cuda"
        ... )
    """
    model = MLP(
        in_dim=in_dim,
        hidden=hidden,
        out_dim=1,
        dropout_p=dropout_p,
        activation=activation,
    )
    model = model.to(device)

    return MCDropoutPosterior(model, n_samples=n_samples)
