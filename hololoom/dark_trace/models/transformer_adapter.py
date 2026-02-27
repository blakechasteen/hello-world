"""
Transformer Adapter: Interpretability Interface for External Transformers

This module provides adapters for external transformer models (GPT, BERT, etc.)
enabling activation extraction, hook management, and steering.

Supports:
- HuggingFace Transformers (GPT-2, BERT, Llama, etc.)
- OpenAI models via API (limited - no internal activations)
- Custom transformer implementations

Layer naming convention (HuggingFace):
- embed_tokens: Token embedding layer
- layer.{i}.attention: Self-attention block
- layer.{i}.mlp: MLP/FFN block
- layer.{i}.ln1, layer.{i}.ln2: Layer norms
- lm_head: Output projection (for causal LMs)

Usage:
    from transformers import AutoModel
    from HoloLoom.dark_trace.models import TransformerAdapter

    model = AutoModel.from_pretrained("gpt2")
    adapter = TransformerAdapter(model)

    # Get layer names
    layers = adapter.get_layer_names()

    # Extract activations
    inputs = {"input_ids": tokenizer.encode("Hello world", return_tensors="pt")}
    acts = adapter.get_activations(inputs, "layer.5.attention")

    # Register hooks for multiple layers
    captured = {}
    for layer in ["layer.0.attention", "layer.5.mlp", "layer.11.attention"]:
        adapter.register_hook(lambda a, l: captured.update({l: a}), layer)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from HoloLoom.dark_trace.models.adapter import (
    ModelAdapter,
    LayerInfo,
    LayerType,
    ModelCapabilities,
    SteeringConfig,
    HookHandle,
    ActivationHook,
)


class TransformerLayerType(Enum):
    """Types of layers in transformer architectures."""
    EMBEDDING = "embedding"
    POSITION_EMBEDDING = "position_embedding"
    ATTENTION = "attention"
    ATTENTION_OUTPUT = "attention_output"
    MLP = "mlp"
    MLP_GATE = "mlp_gate"  # For gated architectures like Llama
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    OUTPUT_HEAD = "output_head"
    POOLER = "pooler"


@dataclass
class TransformerLayerInfo:
    """Extended layer info for transformers."""

    name: str
    layer_type: TransformerLayerType
    layer_index: Optional[int] = None
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    module_path: Optional[str] = None  # Full module path for hook registration

    def to_layer_info(self) -> LayerInfo:
        """Convert to generic LayerInfo."""
        type_map = {
            TransformerLayerType.EMBEDDING: LayerType.EMBEDDING,
            TransformerLayerType.POSITION_EMBEDDING: LayerType.EMBEDDING,
            TransformerLayerType.ATTENTION: LayerType.ATTENTION,
            TransformerLayerType.ATTENTION_OUTPUT: LayerType.ATTENTION,
            TransformerLayerType.MLP: LayerType.MLP,
            TransformerLayerType.MLP_GATE: LayerType.MLP,
            TransformerLayerType.LAYER_NORM: LayerType.NORMALIZATION,
            TransformerLayerType.RMS_NORM: LayerType.NORMALIZATION,
            TransformerLayerType.OUTPUT_HEAD: LayerType.OUTPUT,
            TransformerLayerType.POOLER: LayerType.OUTPUT,
        }

        return LayerInfo(
            name=self.name,
            layer_type=type_map.get(self.layer_type, LayerType.UNKNOWN),
            index=self.layer_index or 0,
            input_dim=self.hidden_size,
            output_dim=self.hidden_size,
            num_heads=self.num_heads,
            is_hookable=True,
            metadata={
                "transformer_layer_type": self.layer_type.value,
                "head_dim": self.head_dim,
                "intermediate_size": self.intermediate_size,
                "module_path": self.module_path,
            }
        )


class TransformerAdapter(ModelAdapter):
    """
    Model adapter for external transformer models.

    Automatically detects model architecture and provides unified
    interface for activation extraction and steering.

    Supported architectures:
    - GPT-2 family
    - BERT family
    - Llama family
    - Other HuggingFace transformers

    Args:
        model: Transformer model (HuggingFace or similar)
        tokenizer: Optional tokenizer for text inputs
        device: torch device
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        device: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for TransformerAdapter")

        super().__init__(model=model)
        self._tokenizer = tokenizer

        # Detect device
        if device is None:
            try:
                self._device = next(model.parameters()).device
            except StopIteration:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Detect architecture
        self._architecture = self._detect_architecture()
        self._config = self._get_model_config()

        # Build layer cache
        self._layer_info_cache: Dict[str, TransformerLayerInfo] = {}
        self._module_map: Dict[str, nn.Module] = {}
        self._build_layer_cache()

        # Active hooks
        self._torch_hooks: Dict[str, Any] = {}

    def _detect_architecture(self) -> str:
        """Detect transformer architecture type."""
        model_class = type(self._model).__name__.lower()

        if "gpt2" in model_class or "gpt" in model_class:
            return "gpt2"
        elif "bert" in model_class:
            return "bert"
        elif "llama" in model_class:
            return "llama"
        elif "mistral" in model_class:
            return "mistral"
        elif "opt" in model_class:
            return "opt"
        elif "t5" in model_class:
            return "t5"
        else:
            return "generic"

    def _get_model_config(self) -> Dict[str, Any]:
        """Extract model configuration."""
        config = {}

        if hasattr(self._model, 'config'):
            model_config = self._model.config

            # Common attributes
            config['hidden_size'] = getattr(model_config, 'hidden_size',
                                           getattr(model_config, 'n_embd', 768))
            config['num_layers'] = getattr(model_config, 'num_hidden_layers',
                                          getattr(model_config, 'n_layer', 12))
            config['num_heads'] = getattr(model_config, 'num_attention_heads',
                                         getattr(model_config, 'n_head', 12))
            config['intermediate_size'] = getattr(model_config, 'intermediate_size',
                                                 getattr(model_config, 'n_inner',
                                                        config['hidden_size'] * 4))
            config['vocab_size'] = getattr(model_config, 'vocab_size', 50257)
        else:
            # Defaults
            config = {
                'hidden_size': 768,
                'num_layers': 12,
                'num_heads': 12,
                'intermediate_size': 3072,
                'vocab_size': 50257,
            }

        config['head_dim'] = config['hidden_size'] // config['num_heads']

        return config

    def _build_layer_cache(self) -> None:
        """Build cache of layer information."""
        # Get transformer block container
        blocks = self._get_transformer_blocks()
        if blocks is None:
            # Fallback: try to enumerate all modules
            self._enumerate_all_modules()
            return

        # Embedding layers
        embed_module = self._get_embedding_module()
        if embed_module is not None:
            self._layer_info_cache["embed_tokens"] = TransformerLayerInfo(
                name="embed_tokens",
                layer_type=TransformerLayerType.EMBEDDING,
                hidden_size=self._config['hidden_size'],
                module_path=self._get_module_path(embed_module),
            )
            self._module_map["embed_tokens"] = embed_module

        # Position embeddings
        pos_embed = self._get_position_embedding()
        if pos_embed is not None:
            self._layer_info_cache["position_embedding"] = TransformerLayerInfo(
                name="position_embedding",
                layer_type=TransformerLayerType.POSITION_EMBEDDING,
                hidden_size=self._config['hidden_size'],
            )
            self._module_map["position_embedding"] = pos_embed

        # Transformer layers
        for i, block in enumerate(blocks):
            self._add_block_layers(i, block)

        # Output head
        lm_head = self._get_lm_head()
        if lm_head is not None:
            self._layer_info_cache["lm_head"] = TransformerLayerInfo(
                name="lm_head",
                layer_type=TransformerLayerType.OUTPUT_HEAD,
                hidden_size=self._config['hidden_size'],
            )
            self._module_map["lm_head"] = lm_head

    def _get_transformer_blocks(self) -> Optional[nn.ModuleList]:
        """Get the list of transformer blocks."""
        # Try common paths
        paths = [
            'transformer.h',  # GPT-2
            'encoder.layer',  # BERT
            'decoder.layers',  # Llama, Mistral
            'model.layers',  # Some models
            'layers',  # Generic
        ]

        for path in paths:
            parts = path.split('.')
            module = self._model
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    module = None
                    break
            if module is not None and isinstance(module, (nn.ModuleList, list)):
                return module

        return None

    def _get_embedding_module(self) -> Optional[nn.Module]:
        """Get token embedding module."""
        paths = [
            'transformer.wte',  # GPT-2
            'embeddings.word_embeddings',  # BERT
            'model.embed_tokens',  # Llama
            'embed_tokens',  # Generic
        ]

        for path in paths:
            module = self._get_module_by_path(path)
            if module is not None:
                return module

        return None

    def _get_position_embedding(self) -> Optional[nn.Module]:
        """Get position embedding module."""
        paths = [
            'transformer.wpe',  # GPT-2
            'embeddings.position_embeddings',  # BERT
        ]

        for path in paths:
            module = self._get_module_by_path(path)
            if module is not None:
                return module

        return None

    def _get_lm_head(self) -> Optional[nn.Module]:
        """Get language model head."""
        paths = [
            'lm_head',
            'cls',
            'classifier',
        ]

        for path in paths:
            if hasattr(self._model, path):
                return getattr(self._model, path)

        return None

    def _get_module_by_path(self, path: str) -> Optional[nn.Module]:
        """Get module by dot-separated path."""
        parts = path.split('.')
        module = self._model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _get_module_path(self, module: nn.Module) -> Optional[str]:
        """Get the full path to a module."""
        for name, mod in self._model.named_modules():
            if mod is module:
                return name
        return None

    def _add_block_layers(self, idx: int, block: nn.Module) -> None:
        """Add layer info for a transformer block."""
        hidden = self._config['hidden_size']
        heads = self._config['num_heads']
        head_dim = self._config['head_dim']
        inter = self._config['intermediate_size']

        # Attention layer
        attn = self._get_attention_module(block)
        if attn is not None:
            name = f"layer.{idx}.attention"
            self._layer_info_cache[name] = TransformerLayerInfo(
                name=name,
                layer_type=TransformerLayerType.ATTENTION,
                layer_index=idx,
                hidden_size=hidden,
                num_heads=heads,
                head_dim=head_dim,
            )
            self._module_map[name] = attn

        # MLP layer
        mlp = self._get_mlp_module(block)
        if mlp is not None:
            name = f"layer.{idx}.mlp"
            self._layer_info_cache[name] = TransformerLayerInfo(
                name=name,
                layer_type=TransformerLayerType.MLP,
                layer_index=idx,
                hidden_size=hidden,
                intermediate_size=inter,
            )
            self._module_map[name] = mlp

        # Layer norms
        ln1 = self._get_first_norm(block)
        if ln1 is not None:
            name = f"layer.{idx}.ln1"
            norm_type = TransformerLayerType.RMS_NORM if 'rms' in type(ln1).__name__.lower() else TransformerLayerType.LAYER_NORM
            self._layer_info_cache[name] = TransformerLayerInfo(
                name=name,
                layer_type=norm_type,
                layer_index=idx,
                hidden_size=hidden,
            )
            self._module_map[name] = ln1

        ln2 = self._get_second_norm(block)
        if ln2 is not None:
            name = f"layer.{idx}.ln2"
            norm_type = TransformerLayerType.RMS_NORM if 'rms' in type(ln2).__name__.lower() else TransformerLayerType.LAYER_NORM
            self._layer_info_cache[name] = TransformerLayerInfo(
                name=name,
                layer_type=norm_type,
                layer_index=idx,
                hidden_size=hidden,
            )
            self._module_map[name] = ln2

    def _get_attention_module(self, block: nn.Module) -> Optional[nn.Module]:
        """Get attention module from a block."""
        paths = ['attn', 'attention', 'self_attn', 'self_attention']
        for path in paths:
            if hasattr(block, path):
                return getattr(block, path)
        return None

    def _get_mlp_module(self, block: nn.Module) -> Optional[nn.Module]:
        """Get MLP module from a block."""
        paths = ['mlp', 'ffn', 'feed_forward', 'fc', 'intermediate']
        for path in paths:
            if hasattr(block, path):
                return getattr(block, path)
        return None

    def _get_first_norm(self, block: nn.Module) -> Optional[nn.Module]:
        """Get first layer norm from a block."""
        paths = ['ln_1', 'ln1', 'input_layernorm', 'norm1', 'attention_norm']
        for path in paths:
            if hasattr(block, path):
                return getattr(block, path)
        return None

    def _get_second_norm(self, block: nn.Module) -> Optional[nn.Module]:
        """Get second layer norm from a block."""
        paths = ['ln_2', 'ln2', 'post_attention_layernorm', 'norm2', 'ffn_norm']
        for path in paths:
            if hasattr(block, path):
                return getattr(block, path)
        return None

    def _enumerate_all_modules(self) -> None:
        """Fallback: enumerate all named modules."""
        for name, module in self._model.named_modules():
            if name == '':
                continue

            # Determine layer type from module type
            module_type = type(module).__name__.lower()

            if 'embed' in module_type:
                layer_type = TransformerLayerType.EMBEDDING
            elif 'attention' in module_type:
                layer_type = TransformerLayerType.ATTENTION
            elif 'mlp' in module_type or 'ffn' in module_type:
                layer_type = TransformerLayerType.MLP
            elif 'layernorm' in module_type:
                layer_type = TransformerLayerType.LAYER_NORM
            elif 'rmsnorm' in module_type:
                layer_type = TransformerLayerType.RMS_NORM
            else:
                continue  # Skip unrecognized modules

            # Extract layer index if present
            parts = name.split('.')
            layer_idx = None
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break

            self._layer_info_cache[name] = TransformerLayerInfo(
                name=name,
                layer_type=layer_type,
                layer_index=layer_idx,
                hidden_size=self._config['hidden_size'],
                module_path=name,
            )
            self._module_map[name] = module

    def get_layer_names(self) -> List[str]:
        """Get names of all hookable layers."""
        return list(self._layer_info_cache.keys())

    def get_layer_info(self, layer_name: str) -> LayerInfo:
        """Get metadata about a specific layer."""
        if layer_name not in self._layer_info_cache:
            raise ValueError(f"Unknown layer: {layer_name}")
        return self._layer_info_cache[layer_name].to_layer_info()

    def get_transformer_layer_info(self, layer_name: str) -> TransformerLayerInfo:
        """Get transformer-specific layer info."""
        if layer_name not in self._layer_info_cache:
            raise ValueError(f"Unknown layer: {layer_name}")
        return self._layer_info_cache[layer_name]

    def get_activations(
        self,
        inputs: Any,
        layer: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract activations from a layer.

        Args:
            inputs: Dict with input_ids, attention_mask, etc.
                   Or string text (if tokenizer provided)
            layer: Layer name to extract from
            **kwargs: Additional forward arguments

        Returns:
            Activations as numpy array
        """
        if layer not in self._layer_info_cache:
            raise ValueError(f"Unknown layer: {layer}")

        # Handle text input
        if isinstance(inputs, str) and self._tokenizer is not None:
            inputs = self._tokenizer(inputs, return_tensors="pt")

        # Move inputs to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        # Get module
        module = self._module_map.get(layer)
        if module is None:
            return np.zeros((1, self._config['hidden_size']), dtype=np.float32)

        # Capture activation
        captured = {}

        def capture_hook(mod, inp, out):
            if isinstance(out, tuple):
                captured["activation"] = out[0].detach().cpu().numpy()
            else:
                captured["activation"] = out.detach().cpu().numpy()

        handle = module.register_forward_hook(capture_hook)

        with torch.no_grad():
            self._model.eval()
            if isinstance(inputs, dict):
                _ = self._model(**inputs)
            else:
                _ = self._model(inputs)

        handle.remove()

        return captured.get("activation", np.zeros((1, self._config['hidden_size']), dtype=np.float32))

    def inject_steering(
        self,
        vector: np.ndarray,
        layer: str,
        scale: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Inject a steering vector at a layer.

        Note: For transformers, steering is typically applied via hooks
        that modify activations during the forward pass.

        Args:
            vector: Steering direction
            layer: Target layer
            scale: Scaling factor
            **kwargs: method, position

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

        # Register steering hook
        self._register_steering_hook(layer, config_id)

        return config_id

    def _register_steering_hook(self, layer: str, config_id: str) -> None:
        """Register a hook to apply steering."""
        module = self._module_map.get(layer)
        if module is None:
            return

        config = self._steering_configs[config_id]

        def steering_hook(mod, inp, out):
            if not config.enabled:
                return out

            steering = torch.tensor(
                config.vector * config.scale,
                device=out[0].device if isinstance(out, tuple) else out.device,
                dtype=out[0].dtype if isinstance(out, tuple) else out.dtype,
            )

            if isinstance(out, tuple):
                modified = out[0]
            else:
                modified = out

            # Broadcast steering
            while steering.dim() < modified.dim():
                steering = steering.unsqueeze(0)

            if config.position is not None:
                # Position-specific steering
                result = modified.clone()
                if config.method == "add":
                    result[:, config.position] = result[:, config.position] + steering
                elif config.method == "replace":
                    result[:, config.position] = steering
            else:
                # Global steering
                if config.method == "add":
                    result = modified + steering
                elif config.method == "replace":
                    result = steering.expand_as(modified)
                elif config.method == "multiply":
                    result = modified * steering
                else:
                    result = modified

            if isinstance(out, tuple):
                return (result,) + out[1:]
            return result

        handle = module.register_forward_hook(steering_hook)
        self._torch_hooks[config_id] = handle

    def remove_steering(self, config_id: str) -> bool:
        """Remove a steering configuration."""
        if config_id in self._torch_hooks:
            self._torch_hooks[config_id].remove()
            del self._torch_hooks[config_id]

        return super().remove_steering(config_id)

    def forward(self, inputs: Any, **kwargs) -> Any:
        """
        Run forward pass through the model.

        Args:
            inputs: Dict with input_ids, etc. or text string

        Returns:
            Model outputs
        """
        # Handle text input
        if isinstance(inputs, str) and self._tokenizer is not None:
            inputs = self._tokenizer(inputs, return_tensors="pt")

        # Move to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        with torch.no_grad():
            self._model.eval()
            if isinstance(inputs, dict):
                outputs = self._model(**inputs, **kwargs)
            else:
                outputs = self._model(inputs, **kwargs)

        return outputs

    def get_capabilities(self) -> ModelCapabilities:
        """Get capabilities of this adapter."""
        return ModelCapabilities(
            can_extract_activations=True,
            can_inject_steering=True,
            can_ablate=True,
            can_patch=True,
            supports_batching=True,
            supports_gradients=TORCH_AVAILABLE,
            max_batch_size=16,
            supported_dtypes=["float32", "float16", "bfloat16"],
        )

    @property
    def architecture(self) -> str:
        """Get detected architecture type."""
        return self._architecture

    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.copy()


def get_layer_names(model: Any) -> List[str]:
    """
    Convenience function to get layer names from any transformer.

    Args:
        model: Transformer model

    Returns:
        List of hookable layer names
    """
    adapter = TransformerAdapter(model)
    return adapter.get_layer_names()


def create_transformer_adapter(
    model: Any,
    tokenizer: Any = None,
    device: Optional[str] = None,
) -> TransformerAdapter:
    """
    Create a TransformerAdapter for a model.

    Args:
        model: Transformer model
        tokenizer: Optional tokenizer
        device: Optional device string

    Returns:
        TransformerAdapter instance
    """
    return TransformerAdapter(model, tokenizer=tokenizer, device=device)
