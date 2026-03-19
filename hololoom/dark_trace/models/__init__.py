"""
Dark Trace Models: Multi-Model Interpretability Support

This module provides adapters for extracting activations, applying steering,
and comparing features across different model architectures.

Key Capabilities:
- Unified adapter protocol for any model type
- Activation extraction with hook management
- Steering vector injection
- Cross-model feature fingerprinting
- Universal vs model-specific feature analysis

Supported Model Types:
- HoloLoom policy networks (internal)
- Transformer models (via hooks)
- Small/custom models (extensible)

Design Philosophy:
- Models are black boxes until we hook them
- Same interpretability tools should work across models
- Feature fingerprints enable model comparison
- Universal features are the holy grail

Research Basis:
- Activation patching (Anthropic, 2024)
- Cross-model feature comparison
- Universal feature hypothesis

Usage:
    from hololoom.dark_trace.models import (
        # Adapter protocol
        ModelAdapter,
        ActivationHook,
        SteeringConfig,
        # Specific adapters
        PolicyAdapter,
        TransformerAdapter,
        # Fingerprinting
        FeatureFingerprint,
        ModelFingerprinter,
        compare_fingerprints,
    )

    # Wrap a model
    adapter = TransformerAdapter(model)

    # Extract activations
    activations = adapter.get_activations(inputs, layer="layer.12")

    # Apply steering
    adapter.inject_steering(steering_vector, layer="layer.12", scale=0.5)

    # Generate fingerprint
    fingerprinter = ModelFingerprinter(adapter)
    fingerprint = fingerprinter.generate(probe_inputs)

    # Compare models
    similarity = compare_fingerprints(fp1, fp2)
"""

from hololoom.dark_trace.models.adapter import (
    ActivationCache,
    ActivationHook,
    DummyAdapter,
    HookHandle,
    LayerInfo,
    LayerType,
    ModelAdapter,
    ModelCapabilities,
    SteeringConfig,
    create_hook,
    make_cache_key,
)
from hololoom.dark_trace.models.fingerprint import (
    FeatureFingerprint,
    FingerprintComparison,
    FingerprintConfig,
    ModelFingerprinter,
    compare_fingerprints,
    find_model_specific_features,
    find_universal_features,
)
from hololoom.dark_trace.models.policy_adapter import (
    PolicyAdapter,
    PolicyLayerType,
)
from hololoom.dark_trace.models.transformer_adapter import (
    TransformerAdapter,
    TransformerLayerType,
    get_layer_names,
)

__all__ = [
    # Adapter protocol
    "ModelAdapter",
    "ActivationHook",
    "HookHandle",
    "SteeringConfig",
    "LayerInfo",
    "LayerType",
    "ModelCapabilities",
    "create_hook",
    "DummyAdapter",
    "ActivationCache",
    "make_cache_key",
    # Policy adapter
    "PolicyAdapter",
    "PolicyLayerType",
    # Transformer adapter
    "TransformerAdapter",
    "TransformerLayerType",
    "get_layer_names",
    # Fingerprinting
    "FeatureFingerprint",
    "ModelFingerprinter",
    "FingerprintConfig",
    "FingerprintComparison",
    "compare_fingerprints",
    "find_universal_features",
    "find_model_specific_features",
]
