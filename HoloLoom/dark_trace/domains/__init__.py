"""
Dark Trace Domains: Custom Interpretability for Any Domain

This module provides extensibility for domain-specific interpretability,
enabling custom semantic axes, learned dimensions, and domain templates.

Key Capabilities:
- Data-driven axis discovery (PCA, contrastive, clustering)
- YAML-configured domain lenses for rapid customization
- Pre-built templates for common domains (medical, legal, code, safety)
- Lens composition for multi-domain analysis
- Auto-labeling of discovered axes

Design Philosophy:
- Interpretability should adapt to the domain
- Custom axes should be as valid as learned features
- Configuration should be simple (YAML), code optional
- Composition enables complex multi-domain scenarios

Research Basis:
- Domain adaptation in interpretability (Anthropic methodology)
- Concept bottleneck models for interpretable representations
- Contrastive learning for semantic axis discovery

Usage:
    from HoloLoom.dark_trace.domains import (
        # Learned axis discovery
        LearnedLens,
        PCAAxisDiscovery,
        ContrastiveAxisDiscovery,
        ClusteringAxisDiscovery,
        AutoLabeler,
        DiscoveredAxis,
        # Config-based lenses
        ConfigLens,
        DomainConfig,
        AxisSpec,
        ExemplarSet,
        load_domain_config,
        validate_domain_config,
        # Templates
        load_template,
        list_templates,
        MedicalDomain,
        LegalDomain,
        CodeDomain,
        SafetyDomain,
        # Composition
        LensComposer,
        ComposedLens,
        CompositionStrategy,
        DomainRouter,
    )

    # Data-driven axis discovery
    discovery = PCAAxisDiscovery(n_components=50)
    discovery.fit(activations)
    axes = discovery.get_axes()

    # Auto-label discovered axes
    labeler = AutoLabeler(llm_client=llm)
    for axis in axes:
        label = labeler.label(axis, top_activating_examples)
        print(f"Axis {axis.index}: {label}")

    # Load domain from YAML
    config = load_domain_config("medical.yaml")
    lens = ConfigLens(config)
    projection = lens.project(activations)

    # Use pre-built template
    medical = MedicalDomain()
    risk_score = medical.assess_safety_relevance(features)

    # Compose multiple domains
    composer = LensComposer()
    composer.add(medical, weight=0.6)
    composer.add(SafetyDomain(), weight=0.4)
    composed = composer.compose()
    projection = composed.project(activations)
"""

# Learned axis discovery
from HoloLoom.dark_trace.domains.learned_lens import (
    LearnedLens,
    PCAAxisDiscovery,
    ContrastiveAxisDiscovery,
    ClusteringAxisDiscovery,
    AutoLabeler,
    DiscoveredAxis,
    AxisDiscoveryConfig,
    ContrastivePair,
    LabelingResult,
)

# Config-based lenses
from HoloLoom.dark_trace.domains.config_lens import (
    ConfigLens,
    DomainConfig,
    AxisSpec,
    ExemplarSet,
    ConfigValidationError,
    load_domain_config,
    validate_domain_config,
    create_lens_from_yaml,
)

# Composition
from HoloLoom.dark_trace.domains.composer import (
    LensComposer,
    ComposedLens,
    CompositionStrategy,
    DomainRouter,
    RoutingRule,
    CompositionResult,
)

# Template domains (lazy-loaded to avoid import overhead)
def load_template(name: str) -> "ConfigLens":
    """Load a pre-built domain template by name."""
    import os
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    template_path = os.path.join(template_dir, f"{name}.yaml")

    if not os.path.exists(template_path):
        available = list_templates()
        raise ValueError(
            f"Template '{name}' not found. Available: {available}"
        )

    return create_lens_from_yaml(template_path)


def list_templates() -> list:
    """List available domain templates."""
    import os
    template_dir = os.path.join(os.path.dirname(__file__), "templates")

    if not os.path.exists(template_dir):
        return []

    templates = []
    for f in os.listdir(template_dir):
        if f.endswith(".yaml") or f.endswith(".yml"):
            templates.append(f.rsplit(".", 1)[0])

    return sorted(templates)


# Convenience domain classes
class MedicalDomain:
    """Pre-configured lens for medical/healthcare domain."""

    def __init__(self):
        self._lens = None

    @property
    def lens(self) -> "ConfigLens":
        if self._lens is None:
            try:
                self._lens = load_template("medical")
            except ValueError:
                # Create minimal fallback
                from HoloLoom.dark_trace.domains.config_lens import (
                    DomainConfig, AxisSpec, ConfigLens
                )
                config = DomainConfig(
                    name="medical",
                    description="Medical domain (fallback)",
                    axes=[
                        AxisSpec(name="severity", description="Medical severity"),
                        AxisSpec(name="urgency", description="Medical urgency"),
                        AxisSpec(name="risk", description="Patient risk level"),
                    ]
                )
                self._lens = ConfigLens(config)
        return self._lens

    def project(self, activations):
        """Project activations onto medical semantic space."""
        return self.lens.project(activations)

    def assess_safety_relevance(self, features) -> float:
        """Assess safety relevance for medical applications."""
        projection = self.project(features)
        # Weight safety-critical dimensions higher
        safety_dims = ["severity", "urgency", "risk"]
        score = 0.0
        count = 0
        for dim in safety_dims:
            if dim in projection:
                score += abs(projection[dim])
                count += 1
        return score / max(count, 1)


class LegalDomain:
    """Pre-configured lens for legal/compliance domain."""

    def __init__(self):
        self._lens = None

    @property
    def lens(self) -> "ConfigLens":
        if self._lens is None:
            try:
                self._lens = load_template("legal")
            except ValueError:
                from HoloLoom.dark_trace.domains.config_lens import (
                    DomainConfig, AxisSpec, ConfigLens
                )
                config = DomainConfig(
                    name="legal",
                    description="Legal domain (fallback)",
                    axes=[
                        AxisSpec(name="liability", description="Legal liability"),
                        AxisSpec(name="compliance", description="Regulatory compliance"),
                        AxisSpec(name="precedent", description="Legal precedent relevance"),
                    ]
                )
                self._lens = ConfigLens(config)
        return self._lens

    def project(self, activations):
        """Project activations onto legal semantic space."""
        return self.lens.project(activations)


class CodeDomain:
    """Pre-configured lens for code/programming domain."""

    def __init__(self):
        self._lens = None

    @property
    def lens(self) -> "ConfigLens":
        if self._lens is None:
            try:
                self._lens = load_template("code")
            except ValueError:
                from HoloLoom.dark_trace.domains.config_lens import (
                    DomainConfig, AxisSpec, ConfigLens
                )
                config = DomainConfig(
                    name="code",
                    description="Code domain (fallback)",
                    axes=[
                        AxisSpec(name="complexity", description="Code complexity"),
                        AxisSpec(name="security", description="Security relevance"),
                        AxisSpec(name="performance", description="Performance impact"),
                    ]
                )
                self._lens = ConfigLens(config)
        return self._lens

    def project(self, activations):
        """Project activations onto code semantic space."""
        return self.lens.project(activations)


class SafetyDomain:
    """Pre-configured lens for AI safety domain."""

    def __init__(self):
        self._lens = None

    @property
    def lens(self) -> "ConfigLens":
        if self._lens is None:
            try:
                self._lens = load_template("safety")
            except ValueError:
                from HoloLoom.dark_trace.domains.config_lens import (
                    DomainConfig, AxisSpec, ConfigLens
                )
                config = DomainConfig(
                    name="safety",
                    description="AI Safety domain (fallback)",
                    axes=[
                        AxisSpec(name="deception", description="Deceptive content"),
                        AxisSpec(name="manipulation", description="Manipulative intent"),
                        AxisSpec(name="harm", description="Potential for harm"),
                        AxisSpec(name="bias", description="Unfair bias"),
                    ]
                )
                self._lens = ConfigLens(config)
        return self._lens

    def project(self, activations):
        """Project activations onto safety semantic space."""
        return self.lens.project(activations)

    def assess_risk(self, features) -> dict:
        """Assess safety risks from features."""
        projection = self.project(features)
        return {
            "overall_risk": sum(abs(v) for v in projection.values()) / len(projection),
            "dimension_risks": projection,
            "high_risk_dimensions": [
                k for k, v in projection.items() if abs(v) > 0.7
            ]
        }


__all__ = [
    # Learned axis discovery
    "LearnedLens",
    "PCAAxisDiscovery",
    "ContrastiveAxisDiscovery",
    "ClusteringAxisDiscovery",
    "AutoLabeler",
    "DiscoveredAxis",
    "AxisDiscoveryConfig",
    "ContrastivePair",
    "LabelingResult",
    # Config-based lenses
    "ConfigLens",
    "DomainConfig",
    "AxisSpec",
    "ExemplarSet",
    "ConfigValidationError",
    "load_domain_config",
    "validate_domain_config",
    "create_lens_from_yaml",
    # Composition
    "LensComposer",
    "ComposedLens",
    "CompositionStrategy",
    "DomainRouter",
    "RoutingRule",
    "CompositionResult",
    # Templates
    "load_template",
    "list_templates",
    "MedicalDomain",
    "LegalDomain",
    "CodeDomain",
    "SafetyDomain",
]
