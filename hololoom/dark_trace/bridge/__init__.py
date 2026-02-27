"""
Dark Trace Bridge: SAE ↔ Semantic Connection

This module provides the bridge between learned SAE features and interpretable
semantic axes from hololoom's semantic_calculus module.

Key Features:
- Correlation matrix learning between SAE features and 244 semantic axes
- SemanticLens implementing TraceLens protocol for unified interpretability
- Bidirectional mapping (SAE → Semantic, Semantic → SAE)
- Bridged explanations ("Feature 42 = Warmth (0.85) + Formality (0.62)")

Research Foundation:
- SAE: Anthropic's "Towards Monosemanticity" (2023, 2024)
- Semantic Axes: Conjugate exemplar projection (HoloLoom)

Usage:
    from hololoom.dark_trace.bridge import (
        SAESemanticCorrelator,
        SemanticLens,
        FeatureSemanticMapper,
        BridgedExplainer,
    )

    # Learn correlations between SAE and semantic features
    correlator = SAESemanticCorrelator(sae_lens, semantic_lens)
    correlator.update_correlations(activations)

    # Get semantic meaning of SAE feature
    mapper = FeatureSemanticMapper(correlator)
    meanings = mapper.get_semantic_meanings("sae.42")
    # Returns: [("Warmth", 0.85), ("Formality", -0.62), ...]

    # Generate bridged explanation
    explainer = BridgedExplainer(sae_lens, semantic_lens, mapper)
    explanation = explainer.explain(activations)
    # Returns: "Feature sae.42 (Warmth + Formality) active at 0.75..."
"""

# SAE ↔ Semantic Correlation Learning
from hololoom.dark_trace.bridge.correlator import (
    SAESemanticCorrelator,
    CorrelationConfig,
    CorrelationResult,
)

# Semantic Lens (TraceLens implementation)
from hololoom.dark_trace.bridge.semantic_lens import (
    SemanticLens,
)

# Bidirectional Feature ↔ Semantic Mapping
from hololoom.dark_trace.bridge.mapper import (
    FeatureSemanticMapper,
    SemanticMeaning,
    FeatureMapping,
)

# Bridged Explanations
from hololoom.dark_trace.bridge.explainer import (
    BridgedExplainer,
    BridgedExplanation,
)

__all__ = [
    # Correlator
    "SAESemanticCorrelator",
    "CorrelationConfig",
    "CorrelationResult",
    # Semantic Lens
    "SemanticLens",
    # Mapper
    "FeatureSemanticMapper",
    "SemanticMeaning",
    "FeatureMapping",
    # Explainer
    "BridgedExplainer",
    "BridgedExplanation",
]
