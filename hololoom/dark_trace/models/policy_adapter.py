from __future__ import annotations
"""
Policy Adapter: Interpretability Interface for HoloLoom Policy Networks

This module adapts HoloLoom's neural policy engine for interpretability,
enabling activation extraction, hook management, and steering.

The policy network has this structure:
- NeuralCore with learnable latent queries
- TinyTransformerBlock layers (cross-attention + motif-gated MHA + LoRA FFN)
- Tool selection head
- Thompson Sampling bandit for exploration

Layer naming convention:
- latent: Learnable query tokens
- block.{i}.cross: Cross-attention to context
- block.{i}.mha: Motif-gated self-attention
- block.{i}.ffn: LoRA feed-forward network
- readout: Pooling projection
- tool_head: Tool selection logits

Usage:
    from hololoom.policy.unified import create_policy
    from hololoom.dark_trace.models import PolicyAdapter

    policy = create_policy(...)
    adapter = PolicyAdapter(policy.net)

    # Extract activations
    acts = adapter.get_activations(inputs, "block.0.mha")

    # Register hook for activation capture
    captured = {}
    def capture_hook(acts, layer):
        captured[layer] = acts
    handle = adapter.register_hook(capture_hook, "block.0.ffn")

    # Run forward pass (hooks fire)
    output = adapter.forward(mem, ctrl, adapter_idx)

    # Inject steering vector
    steering_id = adapter.inject_steering(
        vector=steering_direction,
        layer="block.1.mha",
        scale=0.5
    )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from hololoom.dark_trace.models.adapter import (
    LayerInfo,
    LayerType,
    ModelAdapter,
    ModelCapabilities,
    SteeringConfig,
)


class PolicyLayerType(Enum):
    """Types of layers in HoloLoom policy network."""
    LATENT = "latent"
    CROSS_ATTENTION = "cross_attention"
    SELF_ATTENTION = "self_attention"
    FFN = "ffn"
    READOUT = "readout"
    TOOL_HEAD = "tool_head"
    LAYER_NORM = "layer_norm"


@dataclass
class PolicyLayerInfo:
    """Extended layer info for policy network."""

    name: str
    layer_type: PolicyLayerType
    block_index: int | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    n_heads: int | None = None
    has_adapter: bool = False
    adapter_rank: int | None = None

    def to_layer_info(self) -> LayerInfo:
        """Convert to generic LayerInfo."""
        # Map PolicyLayerType to LayerType
        type_map = {
            PolicyLayerType.LATENT: LayerType.EMBEDDING,
            PolicyLayerType.CROSS_ATTENTION: LayerType.ATTENTION,
            PolicyLayerType.SELF_ATTENTION: LayerType.ATTENTION,
            PolicyLayerType.FFN: LayerType.MLP,
            PolicyLayerType.READOUT: LayerType.OUTPUT,
            PolicyLayerType.TOOL_HEAD: LayerType.OUTPUT,
            PolicyLayerType.LAYER_NORM: LayerType.NORMALIZATION,
        }

        return LayerInfo(
            name=self.name,
            layer_type=type_map.get(self.layer_type, LayerType.UNKNOWN),
            index=self.block_index or 0,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_heads=self.n_heads,
            is_hookable=True,
            metadata={
                "policy_layer_type": self.layer_type.value,
                "has_adapter": self.has_adapter,
                "adapter_rank": self.adapter_rank,
            }
        )


class PolicyAdapter(ModelAdapter):
    """
    Model adapter for HoloLoom's policy network (NeuralCore).

    Enables interpretability features:
    - Activation extraction from any layer
    - Hook registration for capture/modification
    - Steering vector injection
    - Layer enumeration and metadata

    Args:
        neural_core: NeuralCore instance from policy.unified
        device: torch device (auto-detected if not provided)
    """

    def __init__(self, neural_core: Any, device: str | None = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for PolicyAdapter")

        super().__init__(model=neural_core)

        # Detect device
        if device is None:
            if hasattr(neural_core, 'latent'):
                self._device = neural_core.latent.device
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Cache layer info
        self._layer_info_cache: dict[str, PolicyLayerInfo] = {}
        self._build_layer_cache()

        # Storage for activations during forward pass
        self._activation_storage: dict[str, torch.Tensor] = {}
        self._native_hooks: dict[str, Any] = {}  # PyTorch hook handles

    def _build_layer_cache(self) -> None:
        """Build cache of layer information."""
        net = self._model

        # Get model dimensions
        d_model = net.latent.size(-1) if hasattr(net, 'latent') else 384
        n_tools = net.tool_head.out_features if hasattr(net, 'tool_head') else 4

        # Latent tokens
        self._layer_info_cache["latent"] = PolicyLayerInfo(
            name="latent",
            layer_type=PolicyLayerType.LATENT,
            input_dim=d_model,
            output_dim=d_model,
        )

        # Transformer blocks
        if hasattr(net, 'blocks'):
            for i, block in enumerate(net.blocks):
                # Cross-attention
                self._layer_info_cache[f"block.{i}.cross"] = PolicyLayerInfo(
                    name=f"block.{i}.cross",
                    layer_type=PolicyLayerType.CROSS_ATTENTION,
                    block_index=i,
                    input_dim=d_model,
                    output_dim=d_model,
                    n_heads=block.cross.n_heads if hasattr(block.cross, 'n_heads') else 4,
                )

                # Motif-gated self-attention
                self._layer_info_cache[f"block.{i}.mha"] = PolicyLayerInfo(
                    name=f"block.{i}.mha",
                    layer_type=PolicyLayerType.SELF_ATTENTION,
                    block_index=i,
                    input_dim=d_model,
                    output_dim=d_model,
                    n_heads=block.mha.mha.n_heads if hasattr(block.mha, 'mha') else 4,
                )

                # LoRA FFN
                self._layer_info_cache[f"block.{i}.ffn"] = PolicyLayerInfo(
                    name=f"block.{i}.ffn",
                    layer_type=PolicyLayerType.FFN,
                    block_index=i,
                    input_dim=d_model,
                    output_dim=d_model,
                    has_adapter=True,
                    adapter_rank=16,  # Default LoRA rank
                )

                # Layer norms
                for ln_name in ['ln1', 'ln2', 'ln3']:
                    self._layer_info_cache[f"block.{i}.{ln_name}"] = PolicyLayerInfo(
                        name=f"block.{i}.{ln_name}",
                        layer_type=PolicyLayerType.LAYER_NORM,
                        block_index=i,
                        input_dim=d_model,
                        output_dim=d_model,
                    )

        # Output layers
        self._layer_info_cache["readout"] = PolicyLayerInfo(
            name="readout",
            layer_type=PolicyLayerType.READOUT,
            input_dim=d_model,
            output_dim=d_model,
        )

        self._layer_info_cache["tool_head"] = PolicyLayerInfo(
            name="tool_head",
            layer_type=PolicyLayerType.TOOL_HEAD,
            input_dim=d_model,
            output_dim=n_tools,
        )

    def get_layer_names(self) -> list[str]:
        """Get names of all hookable layers."""
        return list(self._layer_info_cache.keys())

    def get_layer_info(self, layer_name: str) -> LayerInfo:
        """Get metadata about a specific layer."""
        if layer_name not in self._layer_info_cache:
            raise ValueError(f"Unknown layer: {layer_name}")
        return self._layer_info_cache[layer_name].to_layer_info()

    def get_policy_layer_info(self, layer_name: str) -> PolicyLayerInfo:
        """Get policy-specific layer info."""
        if layer_name not in self._layer_info_cache:
            raise ValueError(f"Unknown layer: {layer_name}")
        return self._layer_info_cache[layer_name]

    def _get_module(self, layer_name: str) -> nn.Module:
        """Get the actual PyTorch module for a layer name."""
        net = self._model

        if layer_name == "latent":
            # Latent is a Parameter, not a Module
            return None

        if layer_name == "readout":
            return net.readout

        if layer_name == "tool_head":
            return net.tool_head

        if layer_name.startswith("block."):
            parts = layer_name.split(".")
            block_idx = int(parts[1])
            component = parts[2]

            block = net.blocks[block_idx]

            if component == "cross":
                return block.cross
            elif component == "mha":
                return block.mha
            elif component == "ffn":
                return block.ffn
            elif component in ["ln1", "ln2", "ln3"]:
                return getattr(block, component)

        return None

    def get_activations(
        self,
        inputs: dict[str, Any],
        layer: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract activations from a layer.

        Args:
            inputs: Dict with keys:
                - mem: Context memory tensor [B, M, D]
                - ctrl: Motif control vector [B, n_motifs]
                - adapter_idx: Which adapter to use
            layer: Layer name to extract from
            **kwargs: Additional arguments

        Returns:
            Activations as numpy array
        """
        if layer not in self._layer_info_cache:
            raise ValueError(f"Unknown layer: {layer}")

        # Extract inputs
        mem = inputs.get("mem")
        ctrl = inputs.get("ctrl")
        adapter_idx = inputs.get("adapter_idx", 0)

        # Convert to tensors if needed
        if isinstance(mem, np.ndarray):
            mem = torch.tensor(mem, device=self._device, dtype=torch.float32)
        if isinstance(ctrl, np.ndarray):
            ctrl = torch.tensor(ctrl, device=self._device, dtype=torch.float32)

        # Storage for captured activation
        captured = {}

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                captured["activation"] = output[0].detach().cpu().numpy()
            else:
                captured["activation"] = output.detach().cpu().numpy()

        # Register hook
        module = self._get_module(layer)

        if module is not None:
            handle = module.register_forward_hook(capture_hook)

            # Run forward pass
            with torch.no_grad():
                self._model.eval()
                _ = self._forward_impl(mem, ctrl, adapter_idx)

            handle.remove()

            if "activation" in captured:
                return captured["activation"]

        # Handle latent specially
        if layer == "latent":
            return self._model.latent.detach().cpu().numpy()

        # Fallback to empty array
        info = self._layer_info_cache[layer]
        return np.zeros((1, info.output_dim or 384), dtype=np.float32)

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
            **kwargs: Additional options (method, position)

        Returns:
            Steering config ID
        """
        if layer not in self._layer_info_cache:
            raise ValueError(f"Unknown layer: {layer}")

        config_id = f"steering_{len(self._steering_configs)}"
        self._steering_configs[config_id] = SteeringConfig(
            layer=layer,
            vector=vector,
            scale=scale,
            method=kwargs.get("method", "add"),
            position=kwargs.get("position"),
        )

        return config_id

    def _forward_impl(
        self,
        mem: "torch.Tensor",
        ctrl: "torch.Tensor",
        adapter_idx: int,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Internal forward pass implementation."""
        net = self._model
        B = mem.size(0)

        # Expand latent query for batch
        x = net.latent.expand(B, -1, -1)

        # Apply steering to latent if configured
        x = self._apply_torch_steering(x, "latent")

        # Process through transformer blocks
        for i, blk in enumerate(net.blocks):
            # Cross-attention
            x_cross = x + blk.cross(blk.ln1(x), mem)
            x_cross = self._apply_torch_steering(x_cross, f"block.{i}.cross")

            # Motif-gated self-attention
            mha_out, _ = blk.mha(blk.ln2(x_cross), ctrl)
            x_mha = x_cross + mha_out
            x_mha = self._apply_torch_steering(x_mha, f"block.{i}.mha")

            # FFN with adapter
            x_ffn = x_mha + blk.ffn(blk.ln3(x_mha), adapter_idx)
            x_ffn = self._apply_torch_steering(x_ffn, f"block.{i}.ffn")

            x = x_ffn

        # Pool and readout
        pooled = x.mean(dim=1)
        pooled = self._apply_torch_steering(pooled.unsqueeze(1), "readout").squeeze(1)

        # Tool selection
        logits = net.tool_head(net.readout(pooled))
        logits = self._apply_torch_steering(logits.unsqueeze(1), "tool_head").squeeze(1)

        return logits, pooled

    def _apply_torch_steering(
        self,
        activations: "torch.Tensor",
        layer_name: str,
    ) -> "torch.Tensor":
        """Apply steering configs to torch tensor."""
        result = activations

        for config in self._steering_configs.values():
            if config.layer == layer_name and config.enabled:
                steering = torch.tensor(
                    config.vector * config.scale,
                    device=activations.device,
                    dtype=activations.dtype,
                )

                # Broadcast steering to match activation shape
                while steering.dim() < result.dim():
                    steering = steering.unsqueeze(0)

                if config.method == "add":
                    result = result + steering
                elif config.method == "replace":
                    result = steering.expand_as(result)
                elif config.method == "multiply":
                    result = result * steering

        return result

    def forward(self, inputs: Any, **kwargs) -> Any:
        """
        Run forward pass through the model.

        Args:
            inputs: Dict with mem, ctrl, adapter_idx
            **kwargs: Additional arguments

        Returns:
            Dict with logits and pooled features
        """
        mem = inputs.get("mem")
        ctrl = inputs.get("ctrl")
        adapter_idx = inputs.get("adapter_idx", 0)

        # Convert to tensors if needed
        if isinstance(mem, np.ndarray):
            mem = torch.tensor(mem, device=self._device, dtype=torch.float32)
        if isinstance(ctrl, np.ndarray):
            ctrl = torch.tensor(ctrl, device=self._device, dtype=torch.float32)

        # Run forward with steering
        logits, pooled = self._forward_impl(mem, ctrl, adapter_idx)

        return {
            "logits": logits.detach().cpu().numpy(),
            "pooled": pooled.detach().cpu().numpy(),
        }

    def get_capabilities(self) -> ModelCapabilities:
        """Get capabilities of this adapter."""
        return ModelCapabilities(
            can_extract_activations=True,
            can_inject_steering=True,
            can_ablate=True,
            can_patch=True,
            supports_batching=True,
            supports_gradients=TORCH_AVAILABLE,
            max_batch_size=32,
            supported_dtypes=["float32", "float16"],
        )

    def get_tool_names(self) -> list[str]:
        """Get names of tools the policy selects from."""
        if hasattr(self._model, 'tools'):
            return self._model.tools
        return ["answer", "search", "notion_write", "calc"]

    def get_attention_patterns(
        self,
        inputs: dict[str, Any],
        layer: str,
    ) -> np.ndarray | None:
        """
        Extract attention patterns from an attention layer.

        Args:
            inputs: Dict with mem, ctrl, adapter_idx
            layer: Layer name (must be attention layer)

        Returns:
            Attention weights [B, H, T, S] or None if not attention layer
        """
        info = self._layer_info_cache.get(layer)
        if info is None or info.layer_type not in [
            PolicyLayerType.CROSS_ATTENTION,
            PolicyLayerType.SELF_ATTENTION,
        ]:
            return None

        # Extract inputs
        mem = inputs.get("mem")
        ctrl = inputs.get("ctrl")
        adapter_idx = inputs.get("adapter_idx", 0)

        if isinstance(mem, np.ndarray):
            mem = torch.tensor(mem, device=self._device, dtype=torch.float32)
        if isinstance(ctrl, np.ndarray):
            ctrl = torch.tensor(ctrl, device=self._device, dtype=torch.float32)

        # Capture attention weights
        captured = {}

        def capture_attention(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                captured["attention"] = output[1].detach().cpu().numpy()

        module = self._get_module(layer)
        if module is not None:
            handle = module.register_forward_hook(capture_attention)

            with torch.no_grad():
                self._model.eval()
                _ = self._forward_impl(mem, ctrl, adapter_idx)

            handle.remove()

            return captured.get("attention")

        return None


def create_policy_adapter(
    policy: Any,
    device: str | None = None,
) -> PolicyAdapter:
    """
    Create a PolicyAdapter from a UnifiedPolicy.

    Args:
        policy: UnifiedPolicy instance
        device: Optional device string

    Returns:
        PolicyAdapter wrapping the neural core
    """
    # Extract neural core from unified policy
    if hasattr(policy, 'net'):
        neural_core = policy.net
    elif hasattr(policy, 'neural_core'):
        neural_core = policy.neural_core
    else:
        neural_core = policy

    return PolicyAdapter(neural_core, device=device)
