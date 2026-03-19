"""
Dark Trace Research Extensions

Advanced research techniques for interpretability analysis:

- Sparse Probing: Linear probes for feature interpretation
- Concept Erasure: Remove concepts from representations
- Adversarial Discovery: Find inputs that manipulate features
- Adversarial Probing (Phase 2): Systematic vulnerability testing for SAE features
- Genetic Probing (Phase 2): Gradient-free adversarial search
- Vulnerability Catalog (Phase 2): Persistence and tracking of vulnerabilities

These techniques are based on cutting-edge research from:
- Anthropic's "Scaling Monosemanticity" (2024)
- "Sparse Autoencoders Find Highly Interpretable Features" (Cunningham et al., 2023)
- "LEACE: Perfect Linear Concept Erasure" (Belrose et al., 2023)
- FGSM and PGD adversarial attacks (Goodfellow et al., 2014; Madry et al., 2018)

Usage:
    from hololoom.dark_trace.research import (
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

        # Adversarial Discovery (Phase 1)
        AdversarialAttacker,
        AdversarialDiscoverer,
        FeatureSensitivityAnalyzer,
        create_attacker,
        create_discoverer,

        # Adversarial Probing (Phase 2)
        AdversarialProber,
        GradientProber,
        ProbeConfig,
        ProbeResult,
        VulnerabilityLevel,
        VulnerabilityReport,
        create_prober,

        # Genetic Probing (Phase 2)
        GeneticAdversarialSearch,
        GeneticConfig,
        GeneticOperators,
        Individual,
        EvolutionResult,
        create_genetic_search,

        # Vulnerability Catalog (Phase 2)
        AdversarialCatalog,
        CatalogedVulnerability,
        VulnerabilityStatus,
        CatalogSummary,
        RegressionReport,
        create_catalog,
    )

Author: HoloLoom Team
Created: December 2025
"""

# Phase 2: Adversarial Catalog
from hololoom.dark_trace.research.adversarial_catalog import (
    # Classes
    AdversarialCatalog,
    # Data Structures
    CatalogedVulnerability,
    CatalogSummary,
    RegressionReport,
    # Enums
    VulnerabilityStatus,
    # Factory functions
    create_catalog,
)
from hololoom.dark_trace.research.adversarial_discovery import (
    # Classes
    AdversarialAttacker,
    AdversarialDiscoverer,
    AdversarialExample,
    # Config and Results
    AttackConfig,
    # Enums
    AttackMethod,
    DiscoveryResult,
    FeatureSensitivityAnalyzer,
    TargetType,
    # Factory functions
    create_attacker,
    create_discoverer,
    # Phase 2 integration functions
    create_prober_from_discoverer,
    create_sensitivity_analyzer,
    probe_feature_with_catalog,
)

# Phase 2: Adversarial Probing Suite
from hololoom.dark_trace.research.adversarial_prober import (
    AdversarialProber,
    # Classes
    GradientProber,
    # Config and Results
    ProbeConfig,
    ProbeMethod,
    ProbeResult,
    # Enums
    VulnerabilityLevel,
    VulnerabilityReport,
    # Factory functions
    create_prober,
)
from hololoom.dark_trace.research.concept_erasure import (
    # Classes
    ConceptEraser,
    ConceptSurgery,
    # Config and Results
    ErasureConfig,
    # Enums
    ErasureMethod,
    ErasureResult,
    LEACEEraser,
    create_concept_surgery,
    # Factory functions
    create_eraser,
    create_leace_eraser,
)

# Phase 2: Genetic Adversarial Search
from hololoom.dark_trace.research.genetic_prober import (
    CrossoverMethod,
    EvolutionResult,
    GeneticAdversarialSearch,
    # Config and Results
    GeneticConfig,
    # Classes
    GeneticOperators,
    Individual,
    MutationMethod,
    # Enums
    SelectionMethod,
    # Factory functions
    create_genetic_search,
)
from hololoom.dark_trace.research.sparse_probing import (
    ActivationDataset,
    ContrastiveProber,
    FeatureProber,
    # Config and Results
    ProbeConfig,
    ProbeResult,
    # Enums
    ProbeType,
    # Classes
    SparseProbe,
    create_feature_prober,
    # Factory functions
    create_probe,
)

__all__ = [
    # === Sparse Probing ===
    # Enums
    "ProbeType",
    # Config and Results (note: ProbeConfig also used by adversarial_prober)
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

    # === Adversarial Discovery (Phase 1) ===
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
    # Phase 2 integration functions
    "create_prober_from_discoverer",
    "probe_feature_with_catalog",

    # === Adversarial Probing (Phase 2) ===
    # Enums
    "VulnerabilityLevel",
    "ProbeMethod",
    # Config and Results
    "ProbeConfig",
    "ProbeResult",
    "VulnerabilityReport",
    # Classes
    "GradientProber",
    "AdversarialProber",
    # Factory functions
    "create_prober",

    # === Genetic Probing (Phase 2) ===
    # Enums
    "SelectionMethod",
    "CrossoverMethod",
    "MutationMethod",
    # Config and Results
    "GeneticConfig",
    "Individual",
    "EvolutionResult",
    # Classes
    "GeneticOperators",
    "GeneticAdversarialSearch",
    # Factory functions
    "create_genetic_search",

    # === Vulnerability Catalog (Phase 2) ===
    # Enums
    "VulnerabilityStatus",
    # Data Structures
    "CatalogedVulnerability",
    "CatalogSummary",
    "RegressionReport",
    # Classes
    "AdversarialCatalog",
    # Factory functions
    "create_catalog",
]
