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
    from HoloLoom.awareness import (
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

from HoloLoom.awareness.compositional_awareness import (
    CompositionalAwarenessLayer,
    UnifiedAwarenessContext,
    StructuralAwareness,
    CompositionalPatterns,
    ConfidenceSignals,
    InternalStreamGuidance,
    ExternalStreamGuidance,
    format_awareness_for_prompt
)

from HoloLoom.awareness.dual_stream import (
    DualStreamGenerator,
    DualStreamResponse,
    build_internal_prompt,
    build_external_prompt
)

from HoloLoom.awareness.meta_awareness import (
    MetaAwarenessLayer,
    SelfReflectionResult,
    UncertaintyDecomposition,
    MetaConfidence,
    KnowledgeGapHypothesis,
    AdversarialProbe,
    UncertaintyType
)

# LLM Integration
try:
    from HoloLoom.awareness.llm_integration import (
        LLMProtocol,
        LLMResponse,
        LLMProvider,
        OllamaLLM,
        AnthropicLLM,
        OpenAILLM,
        create_llm
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
