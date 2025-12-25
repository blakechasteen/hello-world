"""
Dark Trace Research Extensions

Advanced research techniques for interpretability analysis:

- Sparse Probing: Linear probes for feature interpretation
- Concept Erasure: Remove concepts from representations
- Adversarial Discovery: Find inputs that manipulate features

These techniques are based on cutting-edge research from:
- Anthropic's "Scaling Monosemanticity" (2024)
- "Sparse Autoencoders Find Highly Interpretable Features" (Cunningham et al., 2023)
- "LEACE: Perfect Linear Concept Erasure" (Belrose et al., 2023)

Usage:
    from HoloLoom.dark_trace.research import (
        # Sparse Probing
        SparseProbe,
        FeatureProber,
        ContrastiveProber,
        create_probe,
        create_feature_prober,

        # Concept Erasure
        ConceptEraser,
        LEACEEraser,
        ConceptSurgery,
        create_eraser,
        create_leace_eraser,

        # Adversarial Discovery
        AdversarialAttacker,
        AdversarialDiscoverer,
        FeatureSensitivityAnalyzer,
        create_attacker,
        create_discoverer,
    )

Author: HoloLoom Team
Created: December 2025
"""

from HoloLoom.dark_trace.research.sparse_probing import (
    # Enums
    ProbeType,
    # Config and Results
    ProbeConfig,
    ProbeResult,
    ActivationDataset,
    # Classes
    SparseProbe,
    FeatureProber,
    ContrastiveProber,
    # Factory functions
    create_probe,
    create_feature_prober,
)

from HoloLoom.dark_trace.research.concept_erasure import (
    # Enums
    ErasureMethod,
    # Config and Results
    ErasureConfig,
    ErasureResult,
    # Classes
    ConceptEraser,
    LEACEEraser,
    ConceptSurgery,
    # Factory functions
    create_eraser,
    create_leace_eraser,
    create_concept_surgery,
)

from HoloLoom.dark_trace.research.adversarial_discovery import (
    # Enums
    AttackMethod,
    TargetType,
    # Config and Results
    AttackConfig,
    AdversarialExample,
    DiscoveryResult,
    # Classes
    AdversarialAttacker,
    FeatureSensitivityAnalyzer,
    AdversarialDiscoverer,
    # Factory functions
    create_attacker,
    create_discoverer,
    create_sensitivity_analyzer,
)

__all__ = [
    # === Sparse Probing ===
    # Enums
    "ProbeType",
    # Config and Results
    "ProbeConfig",
    "ProbeResult",
    "ActivationDataset",
    # Classes
    "SparseProbe",
    "FeatureProber",
    "ContrastiveProber",
    # Factory functions
    "create_probe",
    "create_feature_prober",

    # === Concept Erasure ===
    # Enums
    "ErasureMethod",
    # Config and Results
    "ErasureConfig",
    "ErasureResult",
    # Classes
    "ConceptEraser",
    "LEACEEraser",
    "ConceptSurgery",
    # Factory functions
    "create_eraser",
    "create_leace_eraser",
    "create_concept_surgery",

    # === Adversarial Discovery ===
    # Enums
    "AttackMethod",
    "TargetType",
    # Config and Results
    "AttackConfig",
    "AdversarialExample",
    "DiscoveryResult",
    # Classes
    "AdversarialAttacker",
    "FeatureSensitivityAnalyzer",
    "AdversarialDiscoverer",
    # Factory functions
    "create_attacker",
    "create_discoverer",
    "create_sensitivity_analyzer",
]
