"""
Model Adapter Protocol: Unified Interface for Any Model

This module defines the protocol for model adapters, enabling consistent
activation extraction, hook management, and steering across model types.

Key Components:
- ModelAdapter: Abstract base class for model wrappers
- ActivationHook: Callable for capturing/modifying activations
- HookHandle: Reference for managing registered hooks
- SteeringConfig: Configuration for steering vector injection
- LayerInfo: Metadata about model layers

Design Philosophy:
- Models are black boxes until we hook them
- Hooks should be lightweight and removable
- Steering should be controllable and reversible
- Layer enumeration enables systematic analysis

Usage:
    class MyModelAdapter(ModelAdapter):
        def get_activations(self, inputs, layer):
            # Extract activations from layer
            ...

        def inject_steering(self, vector, layer, scale):
            # Modify activations during forward pass
            ...

    adapter = MyModelAdapter(model)
    acts = adapter.get_activations(inputs, "layer.5")
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class LayerType(Enum):
    """Types of layers that can be hooked."""
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    MLP = "mlp"
    NORMALIZATION = "normalization"
    OUTPUT = "output"
    RESIDUAL = "residual"
    UNKNOWN = "unknown"


@dataclass
class LayerInfo:
    """Metadata about a model layer."""

    name: str
    layer_type: LayerType
    index: int
    input_dim: int | None = None
    output_dim: int | None = None
    num_heads: int | None = None  # For attention layers
    is_hookable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"LayerInfo({self.name}, type={self.layer_type.value}, idx={self.index})"


@dataclass
class ModelCapabilities:
    """Capabilities of a model adapter."""

    can_extract_activations: bool = True
    can_inject_steering: bool = True
    can_ablate: bool = True
    can_patch: bool = True
    supports_batching: bool = True
    supports_gradients: bool = True
    max_batch_size: int | None = None
    supported_dtypes: list[str] = field(default_factory=lambda: ["float32"])


@dataclass
class SteeringConfig:
    """Configuration for steering vector injection."""

    layer: str
    vector: np.ndarray
    scale: float = 1.0
    method: str = "add"  # "add", "replace", "multiply"
    position: int | None = None  # Token position, None = all
    enabled: bool = True

    def apply(self, activations: np.ndarray) -> np.ndarray:
        """Apply steering to activations."""
        if not self.enabled:
            return activations

        steering = self.vector * self.scale

        # Handle position-specific steering
        if self.position is not None and activations.ndim >= 2:
            result = activations.copy()
            if self.method == "add":
                result[:, self.position] = result[:, self.position] + steering
            elif self.method == "replace":
                result[:, self.position] = steering
            elif self.method == "multiply":
                result[:, self.position] = result[:, self.position] * steering
            return result

        # Apply to all positions
        if self.method == "add":
            return activations + steering
        elif self.method == "replace":
            return np.broadcast_to(steering, activations.shape).copy()
        elif self.method == "multiply":
            return activations * steering
        else:
            raise ValueError(f"Unknown steering method: {self.method}")


class HookHandle:
    """Handle for managing a registered hook."""

    def __init__(
        self,
        hook_id: str,
        layer: str,
        remove_fn: Callable[[], None],
    ):
        self.hook_id = hook_id
        self.layer = layer
        self._remove_fn = remove_fn
        self._active = True

    def remove(self) -> None:
        """Remove this hook."""
        if self._active:
            self._remove_fn()
            self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def __repr__(self) -> str:
        status = "active" if self._active else "removed"
        return f"HookHandle({self.hook_id}, layer={self.layer}, {status})"


# Type alias for activation hooks
ActivationHook = Callable[[np.ndarray, str], np.ndarray | None]


def create_hook(
    callback: Callable[[np.ndarray], np.ndarray | None],
    layer_filter: str | None = None,
) -> ActivationHook:
    """
    Create an activation hook from a callback.

    Args:
        callback: Function that receives activations and optionally returns modified ones
        layer_filter: Optional layer name filter (exact match or prefix)

    Returns:
        ActivationHook that can be registered with an adapter
    """
    def hook(activations: np.ndarray, layer_name: str) -> np.ndarray | None:
        if layer_filter is not None:
            if not layer_name.startswith(layer_filter):
                return None
        return callback(activations)

    return hook


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    Provides unified interface for:
    - Activation extraction
    - Hook management
    - Steering injection
    - Layer enumeration
    """

    def __init__(self, model: Any):
        """
        Initialize adapter with a model.

        Args:
            model: The model to wrap (any type)
        """
        self._model = model
        self._hooks: dict[str, tuple[ActivationHook, str]] = {}
        self._steering_configs: dict[str, SteeringConfig] = {}
        self._hook_counter = 0

    @property
    def model(self) -> Any:
        """Access the underlying model."""
        return self._model

    @abstractmethod
    def get_layer_names(self) -> list[str]:
        """Get names of all hookable layers."""
        pass

    @abstractmethod
    def get_layer_info(self, layer_name: str) -> LayerInfo:
        """Get metadata about a specific layer."""
        pass

    def get_all_layer_info(self) -> list[LayerInfo]:
        """Get metadata about all layers."""
        return [self.get_layer_info(name) for name in self.get_layer_names()]

    @abstractmethod
    def get_activations(
        self,
        inputs: Any,
        layer: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract activations from a layer.

        Args:
            inputs: Model inputs (format depends on model type)
            layer: Layer name to extract from
            **kwargs: Additional arguments for forward pass

        Returns:
            Activations as numpy array
        """
        pass

    def get_activations_batch(
        self,
        inputs_batch: list[Any],
        layer: str,
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Extract activations for a batch of inputs.

        Default implementation processes sequentially.
        Override for parallel processing.
        """
        return [self.get_activations(inp, layer, **kwargs) for inp in inputs_batch]

    def get_multi_layer_activations(
        self,
        inputs: Any,
        layers: list[str],
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Extract activations from multiple layers in a single forward pass.

        Default implementation makes multiple passes.
        Override for efficient single-pass extraction.
        """
        return {layer: self.get_activations(inputs, layer, **kwargs) for layer in layers}

    @abstractmethod
    def inject_steering(
        self,
        vector: np.ndarray,
        layer: str,
        scale: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Inject a steering vector at a layer.

        Args:
            vector: Steering direction
            layer: Target layer
            scale: Scaling factor
            **kwargs: Additional steering options

        Returns:
            Steering config ID for later removal
        """
        pass

    def remove_steering(self, config_id: str) -> bool:
        """
        Remove a steering configuration.

        Args:
            config_id: ID returned by inject_steering

        Returns:
            True if removed, False if not found
        """
        if config_id in self._steering_configs:
            del self._steering_configs[config_id]
            return True
        return False

    def clear_steering(self) -> int:
        """Remove all steering configurations. Returns count removed."""
        count = len(self._steering_configs)
        self._steering_configs.clear()
        return count

    def register_hook(
        self,
        hook: ActivationHook,
        layer: str,
    ) -> HookHandle:
        """
        Register an activation hook.

        Args:
            hook: Callback function for activations
            layer: Layer to hook

        Returns:
            Handle for managing the hook
        """
        hook_id = f"hook_{self._hook_counter}"
        self._hook_counter += 1
        self._hooks[hook_id] = (hook, layer)

        def remove_fn():
            if hook_id in self._hooks:
                del self._hooks[hook_id]

        return HookHandle(hook_id, layer, remove_fn)

    def clear_hooks(self) -> int:
        """Remove all registered hooks. Returns count removed."""
        count = len(self._hooks)
        self._hooks.clear()
        return count

    def _apply_hooks(
        self,
        activations: np.ndarray,
        layer_name: str,
    ) -> np.ndarray:
        """Apply all registered hooks to activations."""
        result = activations
        for hook_id, (hook, target_layer) in self._hooks.items():
            if target_layer == layer_name or target_layer == "*":
                modified = hook(result, layer_name)
                if modified is not None:
                    result = modified
        return result

    def _apply_steering(
        self,
        activations: np.ndarray,
        layer_name: str,
    ) -> np.ndarray:
        """Apply all active steering configs to activations."""
        result = activations
        for config in self._steering_configs.values():
            if config.layer == layer_name and config.enabled:
                result = config.apply(result)
        return result

    @abstractmethod
    def forward(self, inputs: Any, **kwargs) -> Any:
        """
        Run forward pass through the model.

        Args:
            inputs: Model inputs
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        pass

    def get_capabilities(self) -> ModelCapabilities:
        """Get capabilities of this adapter."""
        return ModelCapabilities()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={type(self._model).__name__})"


class DummyAdapter(ModelAdapter):
    """
    Dummy adapter for testing.

    Returns random activations and accepts any operations.
    """

    def __init__(self, n_layers: int = 12, hidden_dim: int = 768):
        super().__init__(model=None)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def get_layer_names(self) -> list[str]:
        return [f"layer.{i}" for i in range(self.n_layers)]

    def get_layer_info(self, layer_name: str) -> LayerInfo:
        try:
            idx = int(layer_name.split(".")[-1])
        except (ValueError, IndexError):
            idx = 0

        return LayerInfo(
            name=layer_name,
            layer_type=LayerType.MLP,
            index=idx,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def get_activations(
        self,
        inputs: Any,
        layer: str,
        **kwargs,
    ) -> np.ndarray:
        # Return random activations
        batch_size = kwargs.get("batch_size", 1)
        seq_len = kwargs.get("seq_len", 10)
        return np.random.randn(batch_size, seq_len, self.hidden_dim).astype(np.float32)

    def inject_steering(
        self,
        vector: np.ndarray,
        layer: str,
        scale: float = 1.0,
        **kwargs,
    ) -> str:
        config_id = f"steering_{len(self._steering_configs)}"
        self._steering_configs[config_id] = SteeringConfig(
            layer=layer,
            vector=vector,
            scale=scale,
            method=kwargs.get("method", "add"),
            position=kwargs.get("position"),
        )
        return config_id

    def forward(self, inputs: Any, **kwargs) -> Any:
        return {"output": np.random.randn(1, self.hidden_dim)}


class ActivationCache:
    """
    Cache for storing activations during analysis.

    Enables efficient multi-pass analysis without re-running forward passes.
    """

    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, np.ndarray] = {}
        self._max_size = max_size
        self._access_order: list[str] = []

    def get(self, key: str) -> np.ndarray | None:
        """Get cached activations."""
        if key in self._cache:
            # Update access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, activations: np.ndarray) -> None:
        """Cache activations."""
        if len(self._cache) >= self._max_size:
            # Evict least recently used
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = activations
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


def make_cache_key(inputs: Any, layer: str) -> str:
    """Generate a cache key for inputs and layer."""
    # Simple hash-based key
    import hashlib
    input_str = str(inputs)
    key = f"{layer}:{hashlib.md5(input_str.encode()).hexdigest()[:16]}"
    return key
