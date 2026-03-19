"""
HoloLoom Awareness Layer
========================

Compositional AI consciousness: The system becomes aware of its own
linguistic knowledge, confidence levels, and epistemic boundaries.

Three-layer architecture:
1. Compositional Awareness - Real-time linguistic intelligence
2. Dual-Stream Generation - Internal reasoning + External response
3. Meta-Awareness - Recursive self-reflection

Usage:
    from hololoom.awareness import (
        CompositionalAwarenessLayer,
        DualStreamGenerator,
        MetaAwarenessLayer
    )

    # Initialize stack
    awareness = CompositionalAwarenessLayer()
    generator = DualStreamGenerator(awareness)
    meta = MetaAwarenessLayer(awareness)

    # Generate awareness-guided response
    dual_stream = await generator.generate("What is Thompson Sampling?")

    # Recursive self-reflection
    reflection = await meta.recursive_self_reflection(
        query="...",
        response=dual_stream.external_stream,
        awareness_context=dual_stream.awareness_context
    )
"""

# Use relative imports (moved to HoloLoom/memory/awareness/ in Dec 2025)
# Additional awareness modules
from .beta_wave_packer import BetaWaveContextPacker
from .compositional_awareness import (
    CompositionalAwarenessLayer,
    CompositionalPatterns,
    ConfidenceSignals,
    ExternalStreamGuidance,
    InternalStreamGuidance,
    StructuralAwareness,
    UnifiedAwarenessContext,
    format_awareness_for_prompt,
)
from .context_packer import SmartContextPacker
from .dual_stream import (
    DualStreamGenerator,
    DualStreamResponse,
    build_external_prompt,
    build_internal_prompt,
)
from .meta_awareness import (
    AdversarialProbe,
    KnowledgeGapHypothesis,
    MetaAwarenessLayer,
    MetaConfidence,
    SelfReflectionResult,
    UncertaintyDecomposition,
    UncertaintyType,
)

# Alias for backward compatibility
ContextPacker = SmartContextPacker
from .memory_fusion import MemoryFusion

# Alias for backward compatibility
BetaWavePacker = BetaWaveContextPacker

# LLM Integration
try:
    from .llm_integration import (
        AnthropicLLM,
        LLMProtocol,
        LLMProvider,
        LLMResponse,
        OllamaLLM,
        OpenAILLM,
        create_llm,
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

__all__ = [
    # Compositional Awareness
    "CompositionalAwarenessLayer",
    "UnifiedAwarenessContext",
    "StructuralAwareness",
    "CompositionalPatterns",
    "ConfidenceSignals",
    "InternalStreamGuidance",
    "ExternalStreamGuidance",
    "format_awareness_for_prompt",

    # Dual-Stream Generation
    "DualStreamGenerator",
    "DualStreamResponse",
    "build_internal_prompt",
    "build_external_prompt",

    # Meta-Awareness
    "MetaAwarenessLayer",
    "SelfReflectionResult",
    "UncertaintyDecomposition",
    "MetaConfidence",
    "KnowledgeGapHypothesis",
    "AdversarialProbe",
    "UncertaintyType",

    # Additional Awareness Modules
    "BetaWaveContextPacker",
    "BetaWavePacker",  # Backward compat alias
    "SmartContextPacker",
    "ContextPacker",  # Backward compat alias
    "MemoryFusion",

    # LLM Integration (if available)
    "LLM_AVAILABLE",
]

# Add LLM exports if available
if LLM_AVAILABLE:
    __all__.extend([
        "LLMProtocol",
        "LLMResponse",
        "LLMProvider",
        "OllamaLLM",
        "AnthropicLLM",
        "OpenAILLM",
        "create_llm",
    ])
