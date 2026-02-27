"""
Dark Trace Causal: Causal Validation for Interpretability Claims

This module provides causal validation infrastructure for proving
interpretability claims with rigorous empirical evidence.

Key Capabilities:
- Ablation testing: Remove features → measure impact on outputs
- Injection testing: Activate features → measure behavioral effects
- Activation patching: Transfer activations between contexts
- Hypothesis testing: Formal statistical validation of claims

Research Basis:
- Causal Tracing (Meng et al., 2022): Locating factual knowledge
- Activation Patching: Isolating causal mechanisms
- ROME/MEMIT: Precise model editing via causal analysis
- EAP-IG: Integrated gradient-based causal attribution

Usage:
    from HoloLoom.dark_trace.causal import (
        FeatureAblator,
        AblationResult,
        FeatureInjector,
        InjectionResult,
        ActivationPatcher,
        PatchResult,
        CausalValidator,
        CausalHypothesis,
        ValidationResult,
    )

    # Ablation: Remove feature and measure impact
    ablator = FeatureAblator(model, probe)
    result = ablator.ablate_feature("sae.42", test_inputs)
    print(f"Output change: {result.output_delta:.3f}")

    # Injection: Force-activate feature and measure effect
    injector = FeatureInjector(model, probe)
    result = injector.inject_feature("sae.42", strength=0.8, test_inputs)
    print(f"Behavioral shift: {result.behavioral_shift}")

    # Patching: Transfer activations between contexts
    patcher = ActivationPatcher(model, probe)
    result = patcher.patch("context_a", "context_b", features=["sae.42"])
    print(f"Counterfactual output: {result.patched_output}")

    # Validation: Test causal hypothesis
    validator = CausalValidator(ablator, injector, patcher)
    hypothesis = CausalHypothesis(
        feature="sae.42",
        claim="Controls response warmth",
        expected_effect={"warmth": 0.5}
    )
    result = validator.validate(hypothesis, test_suite)
    print(f"Hypothesis {'supported' if result.supported else 'rejected'}")
"""

# Ablation testing
from HoloLoom.dark_trace.causal.ablation import (
    FeatureAblator,
    AblationConfig,
    AblationResult,
    AblationEffect,
    BatchAblationResult,
)

# Injection testing
from HoloLoom.dark_trace.causal.injection import (
    FeatureInjector,
    InjectionConfig,
    InjectionResult,
    DoseResponseCurve,
    InteractionEffect,
)

# Activation patching
from HoloLoom.dark_trace.causal.patching import (
    ActivationPatcher,
    PatchConfig,
    PatchResult,
    CausalTrace,
    InformationFlow,
)

# Causal validation
from HoloLoom.dark_trace.causal.validator import (
    CausalValidator,
    CausalHypothesis,
    ValidationResult,
    ValidationSuite,
    GroundTruthComparison,
)

__all__ = [
    # Ablation
    "FeatureAblator",
    "AblationConfig",
    "AblationResult",
    "AblationEffect",
    "BatchAblationResult",
    # Injection
    "FeatureInjector",
    "InjectionConfig",
    "InjectionResult",
    "DoseResponseCurve",
    "InteractionEffect",
    # Patching
    "ActivationPatcher",
    "PatchConfig",
    "PatchResult",
    "CausalTrace",
    "InformationFlow",
    # Validation
    "CausalValidator",
    "CausalHypothesis",
    "ValidationResult",
    "ValidationSuite",
    "GroundTruthComparison",
]
