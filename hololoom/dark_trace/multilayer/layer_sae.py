"""
Dark Trace Multilayer: Per-Layer SAE Management

Manages Sparse Autoencoders for each layer of a model, enabling
multi-layer interpretability analysis.

Research Basis:
- Scaling Monosemanticity (Anthropic 2024): SAEs at multiple layers
- Residual stream analysis: Layer-specific representations
- Logit Lens: Understanding layer-wise predictions
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class TrainingStrategy(Enum):
    """Strategy for multi-layer SAE training."""
    SEQUENTIAL = "sequential"  # Train layer by layer
    PARALLEL = "parallel"  # Train all layers simultaneously
    CURRICULUM = "curriculum"  # Early layers first, then later
    JOINT = "joint"  # Joint optimization across layers


@dataclass
class LayerSAEConfig:
    """Configuration for per-layer SAE."""

    # SAE architecture
    expansion_factor: int = 4  # Hidden dim = input_dim * expansion
    n_features: int | None = None  # Override expansion_factor

    # Training parameters
    l1_coefficient: float = 3e-4
    learning_rate: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 10

    # Layer-specific adjustments
    layer_l1_scaling: bool = True  # Scale L1 by layer depth
    early_layer_l1_multiplier: float = 1.5  # More sparsity in early layers
    late_layer_l1_multiplier: float = 0.7  # Less sparsity in late layers

    # Training strategy
    training_strategy: TrainingStrategy = TrainingStrategy.SEQUENTIAL

    # Resource management
    max_parallel_layers: int = 4  # Limit GPU memory usage
    checkpoint_frequency: int = 1000  # Steps between checkpoints

    # Quality thresholds
    min_alive_features: float = 0.9  # Minimum fraction of active features
    max_reconstruction_loss: float = 0.1  # Maximum acceptable reconstruction

    @classmethod
    def default(cls) -> "LayerSAEConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def efficient(cls) -> "LayerSAEConfig":
        """Memory-efficient configuration."""
        return cls(
            expansion_factor=2,
            batch_size=32,
            max_parallel_layers=2,
            n_epochs=5,
        )

    @classmethod
    def research(cls) -> "LayerSAEConfig":
        """High-quality research configuration."""
        return cls(
            expansion_factor=8,
            n_epochs=20,
            l1_coefficient=1e-4,
            min_alive_features=0.95,
        )


@dataclass
class LayerActivations:
    """Activations from a single layer."""

    layer_idx: int
    activations: torch.Tensor  # [batch, seq, dim]
    residual_stream: torch.Tensor | None = None  # Before layer
    attention_out: torch.Tensor | None = None  # Attention contribution
    mlp_out: torch.Tensor | None = None  # MLP contribution

    @property
    def shape(self) -> tuple[int, ...]:
        """Get activation shape."""
        return tuple(self.activations.shape)

    @property
    def dim(self) -> int:
        """Get activation dimension."""
        return self.activations.shape[-1]


@dataclass
class MultiLayerFeatures:
    """Features extracted from multiple layers."""

    # Per-layer feature activations
    layer_features: dict[int, torch.Tensor]  # layer_idx -> [batch, seq, n_features]

    # Cross-layer statistics
    feature_correlations: torch.Tensor | None = None  # [n_layers, n_layers]
    shared_features: list[tuple[int, int, float]] | None = None  # (layer1, layer2, correlation)

    # Metadata
    n_layers: int = 0
    total_features: int = 0

    def get_layer(self, layer_idx: int) -> torch.Tensor | None:
        """Get features for a specific layer."""
        return self.layer_features.get(layer_idx)

    def get_active_features(
        self,
        layer_idx: int,
        threshold: float = 0.1,
    ) -> list[int]:
        """Get indices of active features at a layer."""
        features = self.layer_features.get(layer_idx)
        if features is None:
            return []

        # Mean activation across batch and sequence
        mean_act = features.mean(dim=(0, 1))
        active = torch.where(mean_act > threshold)[0]
        return active.tolist()


class LayerSAE(nn.Module):
    """
    Sparse Autoencoder for a single layer.

    Based on Anthropic's SAE architecture with layer-specific
    optimizations for multi-layer training.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        layer_idx: int,
        config: LayerSAEConfig | None = None,
    ):
        """
        Initialize layer SAE.

        Args:
            input_dim: Dimension of layer activations
            n_features: Number of SAE features
            layer_idx: Index of this layer in the model
            config: Configuration options
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_features = n_features
        self.layer_idx = layer_idx
        self.config = config or LayerSAEConfig.default()

        # Encoder: input -> features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)

        # Decoder: features -> reconstruction
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

        # Feature statistics
        self._feature_activations = torch.zeros(n_features)
        self._feature_counts = torch.zeros(n_features)
        self._total_samples = 0

        # Training state
        self._trained = False
        self._training_history: list[dict[str, float]] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse features.

        Args:
            x: Input activations [batch, seq, dim] or [batch, dim]

        Returns:
            Sparse feature activations
        """
        # Flatten if needed
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq, dim = x.shape
            x = x.reshape(-1, dim)
        else:
            batch, dim = x.shape
            seq = None

        # Encode with ReLU for sparsity
        features = torch.relu(self.encoder(x))

        # Restore shape
        if seq is not None:
            features = features.reshape(batch, seq, -1)

        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to activation space.

        Args:
            features: Sparse feature activations

        Returns:
            Reconstructed activations
        """
        original_shape = features.shape
        if features.dim() == 3:
            batch, seq, n_feat = features.shape
            features = features.reshape(-1, n_feat)
        else:
            seq = None

        # Decode
        reconstructed = self.decoder(features)

        # Restore shape
        if seq is not None:
            reconstructed = reconstructed.reshape(batch, seq, -1)

        return reconstructed

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Args:
            x: Input activations
            return_features: Whether to return feature activations

        Returns:
            Reconstructed activations, optionally with features
        """
        features = self.encode(x)
        reconstructed = self.decode(features)

        if return_features:
            return reconstructed, features
        return reconstructed

    def compute_loss(
        self,
        x: torch.Tensor,
        update_stats: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            x: Input activations
            update_stats: Whether to update feature statistics

        Returns:
            Dictionary with loss components
        """
        reconstructed, features = self.forward(x, return_features=True)

        # Reconstruction loss (MSE)
        reconstruction_loss = torch.mean((x - reconstructed) ** 2)

        # L1 sparsity loss
        # Apply layer-specific scaling
        l1_coef = self.config.l1_coefficient
        if self.config.layer_l1_scaling:
            # Assume 12 layers, adjust L1 based on position
            layer_fraction = self.layer_idx / 12.0
            multiplier = (
                self.config.early_layer_l1_multiplier * (1 - layer_fraction) +
                self.config.late_layer_l1_multiplier * layer_fraction
            )
            l1_coef = l1_coef * multiplier

        # Flatten features for L1
        if features.dim() == 3:
            features_flat = features.reshape(-1, features.shape[-1])
        else:
            features_flat = features

        l1_loss = l1_coef * torch.mean(torch.abs(features_flat))

        # Total loss
        total_loss = reconstruction_loss + l1_loss

        # Update statistics
        if update_stats:
            with torch.no_grad():
                # Track which features activated
                active = (features_flat > 0).float().mean(dim=0)
                self._feature_activations = (
                    0.99 * self._feature_activations.to(active.device) +
                    0.01 * active
                )
                self._feature_counts += (features_flat > 0).sum(dim=0).cpu()
                self._total_samples += features_flat.shape[0]

        return {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "l1": l1_loss,
            "l0": (features_flat > 0).float().mean(),  # Average active features
        }

    def get_feature_stats(self) -> dict[str, Any]:
        """Get feature activation statistics."""
        if self._total_samples == 0:
            return {"alive_fraction": 0.0, "mean_activation": 0.0}

        alive = (self._feature_counts > 0).sum().item()
        alive_fraction = alive / self.n_features

        mean_activation = self._feature_activations.mean().item()

        return {
            "alive_fraction": alive_fraction,
            "alive_count": alive,
            "dead_count": self.n_features - alive,
            "mean_activation": mean_activation,
            "total_samples": self._total_samples,
        }

    def get_decoder_norms(self) -> torch.Tensor:
        """Get L2 norms of decoder columns (feature directions)."""
        return torch.norm(self.decoder.weight, dim=0)

    def normalize_decoder(self) -> None:
        """Normalize decoder columns to unit norm."""
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )


class LayerSAEManager:
    """
    Manages SAEs across all layers of a model.

    Coordinates training, provides unified interface for
    multi-layer feature extraction.
    """

    def __init__(
        self,
        model: Any,
        n_layers: int,
        layer_dim: int,
        config: LayerSAEConfig | None = None,
        activation_hook: Callable | None = None,
    ):
        """
        Initialize layer SAE manager.

        Args:
            model: The model to interpret
            n_layers: Number of layers in the model
            layer_dim: Dimension of layer activations
            config: Configuration for all layer SAEs
            activation_hook: Custom function to extract layer activations
        """
        self.model = model
        self.n_layers = n_layers
        self.layer_dim = layer_dim
        self.config = config or LayerSAEConfig.default()
        self.activation_hook = activation_hook

        # Compute number of features per layer
        if self.config.n_features is not None:
            n_features = self.config.n_features
        else:
            n_features = layer_dim * self.config.expansion_factor

        # Create SAE for each layer
        self.layer_saes: dict[int, LayerSAE] = {}
        for layer_idx in range(n_layers):
            self.layer_saes[layer_idx] = LayerSAE(
                input_dim=layer_dim,
                n_features=n_features,
                layer_idx=layer_idx,
                config=self.config,
            )

        # Training state
        self._trained_layers: set = set()
        self._training_stats: dict[int, list[dict]] = defaultdict(list)

    def get_layer_sae(self, layer_idx: int) -> LayerSAE | None:
        """Get SAE for a specific layer."""
        return self.layer_saes.get(layer_idx)

    def extract_activations(
        self,
        inputs: Any,
        layers: list[int] | None = None,
    ) -> dict[int, LayerActivations]:
        """
        Extract activations from specified layers.

        Args:
            inputs: Model inputs
            layers: Which layers to extract (None = all)

        Returns:
            Dictionary mapping layer index to activations
        """
        if layers is None:
            layers = list(range(self.n_layers))

        if self.activation_hook is not None:
            # Use custom hook
            return self.activation_hook(self.model, inputs, layers)

        # Default: try to use model's internal hooks
        activations = {}

        # This is model-specific - provide placeholder
        # Real implementation depends on model architecture
        for layer_idx in layers:
            # Placeholder - real implementation would hook into model
            placeholder = torch.randn(1, 1, self.layer_dim)
            activations[layer_idx] = LayerActivations(
                layer_idx=layer_idx,
                activations=placeholder,
            )

        return activations

    def encode_all_layers(
        self,
        inputs: Any,
        layers: list[int] | None = None,
    ) -> MultiLayerFeatures:
        """
        Extract SAE features from all (or specified) layers.

        Args:
            inputs: Model inputs
            layers: Which layers to process (None = all trained)

        Returns:
            MultiLayerFeatures with per-layer feature activations
        """
        if layers is None:
            layers = list(self._trained_layers)

        # Get activations
        activations = self.extract_activations(inputs, layers)

        # Encode through each layer's SAE
        layer_features = {}
        for layer_idx, layer_act in activations.items():
            sae = self.layer_saes.get(layer_idx)
            if sae is not None and layer_idx in self._trained_layers:
                features = sae.encode(layer_act.activations)
                layer_features[layer_idx] = features

        return MultiLayerFeatures(
            layer_features=layer_features,
            n_layers=len(layer_features),
            total_features=sum(f.shape[-1] for f in layer_features.values()),
        )

    async def train_layer(
        self,
        layer_idx: int,
        training_data: Any,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Train SAE for a single layer.

        Args:
            layer_idx: Which layer to train
            training_data: Training data iterator
            n_steps: Number of training steps (None = use config)

        Returns:
            Training statistics
        """
        sae = self.layer_saes[layer_idx]
        optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=self.config.learning_rate,
        )

        if n_steps is None:
            n_steps = self.config.n_epochs * 1000  # Approximate

        stats = {
            "layer": layer_idx,
            "total_loss": [],
            "reconstruction_loss": [],
            "l1_loss": [],
            "l0": [],
        }

        for step in range(n_steps):
            # Get batch of activations
            # (Placeholder - real implementation uses training_data)
            batch = torch.randn(
                self.config.batch_size,
                self.layer_dim,
            )

            # Forward pass
            losses = sae.compute_loss(batch)

            # Backward pass
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            # Normalize decoder
            sae.normalize_decoder()

            # Record stats
            stats["total_loss"].append(losses["total"].item())
            stats["reconstruction_loss"].append(losses["reconstruction"].item())
            stats["l1_loss"].append(losses["l1"].item())
            stats["l0"].append(losses["l0"].item())

            # Check quality
            if step > 0 and step % self.config.checkpoint_frequency == 0:
                feature_stats = sae.get_feature_stats()
                if feature_stats["alive_fraction"] < self.config.min_alive_features:
                    # Too many dead features - could trigger resampling
                    pass

        self._trained_layers.add(layer_idx)
        self._training_stats[layer_idx] = stats

        return {
            "layer": layer_idx,
            "final_loss": stats["total_loss"][-1] if stats["total_loss"] else 0,
            "final_l0": stats["l0"][-1] if stats["l0"] else 0,
            "feature_stats": sae.get_feature_stats(),
        }

    async def train_all(
        self,
        training_data: Any,
        layers: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Train SAEs for all (or specified) layers.

        Args:
            training_data: Training data
            layers: Which layers to train (None = all)

        Returns:
            Aggregated training statistics
        """
        if layers is None:
            layers = list(range(self.n_layers))

        strategy = self.config.training_strategy
        results = {}

        if strategy == TrainingStrategy.SEQUENTIAL:
            # Train one layer at a time
            for layer_idx in layers:
                result = await self.train_layer(layer_idx, training_data)
                results[layer_idx] = result

        elif strategy == TrainingStrategy.CURRICULUM:
            # Train early layers first (they tend to learn simpler features)
            sorted_layers = sorted(layers)
            for layer_idx in sorted_layers:
                result = await self.train_layer(layer_idx, training_data)
                results[layer_idx] = result

        elif strategy in (TrainingStrategy.PARALLEL, TrainingStrategy.JOINT):
            # Train multiple layers in parallel (memory permitting)
            # For now, treat as sequential with batching
            for i in range(0, len(layers), self.config.max_parallel_layers):
                batch_layers = layers[i:i + self.config.max_parallel_layers]
                for layer_idx in batch_layers:
                    result = await self.train_layer(layer_idx, training_data)
                    results[layer_idx] = result

        return {
            "strategy": strategy.value,
            "layers_trained": len(results),
            "results": results,
        }

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of training across all layers."""
        summaries = {}

        for layer_idx, sae in self.layer_saes.items():
            if layer_idx in self._trained_layers:
                stats = sae.get_feature_stats()
                summaries[layer_idx] = {
                    "alive_fraction": stats["alive_fraction"],
                    "mean_activation": stats["mean_activation"],
                    "trained": True,
                }
            else:
                summaries[layer_idx] = {"trained": False}

        return {
            "n_layers": self.n_layers,
            "trained_layers": len(self._trained_layers),
            "layer_summaries": summaries,
        }

    def save(self, path: str) -> None:
        """Save all layer SAEs to disk."""
        state = {
            "config": self.config,
            "n_layers": self.n_layers,
            "layer_dim": self.layer_dim,
            "trained_layers": list(self._trained_layers),
            "layer_saes": {
                idx: sae.state_dict()
                for idx, sae in self.layer_saes.items()
            },
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load layer SAEs from disk."""
        state = torch.load(path)

        self._trained_layers = set(state["trained_layers"])

        for idx, sae_state in state["layer_saes"].items():
            if idx in self.layer_saes:
                self.layer_saes[idx].load_state_dict(sae_state)


def create_layer_sae_manager(
    model: Any,
    n_layers: int,
    layer_dim: int,
    config: LayerSAEConfig | None = None,
) -> LayerSAEManager:
    """
    Factory function to create LayerSAEManager.

    Args:
        model: Model to interpret
        n_layers: Number of layers
        layer_dim: Dimension of layer activations
        config: Optional configuration

    Returns:
        Configured LayerSAEManager
    """
    return LayerSAEManager(
        model=model,
        n_layers=n_layers,
        layer_dim=layer_dim,
        config=config,
    )
