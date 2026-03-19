"""
Dark Trace Protocol: TraceLens Protocol for Unified Interpretability

The TraceLens protocol provides a unified interface for multiple interpretability
methods (SAE, Semantic Axes, Domain-Specific). All lenses implement this protocol,
enabling the DarkTraceEngine to orchestrate multiple perspectives.

Research Foundation:
- SAE: Anthropic's "Towards Monosemanticity" (2023)
- Semantic Axes: Conjugate exemplar projection (HoloLoom)
- Causal Validation: ACDC (NeurIPS 2023), Activation Patching

Dark Trace Mission:
- Responsibility: We take responsibility for understanding what our systems learn
- Interpretability: Every decision, activation, and representation is transparent
- Alignment: Safety-relevant features are identified and controlled
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import torch

# =============================================================================
# Core Enums and Types
# =============================================================================

class LensType(Enum):
    """Types of interpretability lenses available in Dark Trace."""
    SAE = auto()          # Sparse Autoencoder (learned features)
    SEMANTIC = auto()     # Semantic Axes (hand-crafted interpretable dimensions)
    DOMAIN = auto()       # Domain-specific (learned or configured)
    CIRCUIT = auto()      # Circuit-level (mechanistic)
    CAUSAL = auto()       # Causal validation lens


class FeatureSource(Enum):
    """Source of a feature identifier."""
    SAE = "sae"           # e.g., "sae.42"
    SEMANTIC = "semantic" # e.g., "semantic.Warmth"
    DOMAIN = "domain"     # e.g., "domain.medical.Severity"
    CIRCUIT = "circuit"   # e.g., "circuit.attention_head_7"


class SafetyFlag(Enum):
    """Safety-relevant feature flags."""
    SAFE = auto()         # Normal feature
    MONITOR = auto()      # Should be monitored
    SENSITIVE = auto()    # Requires extra scrutiny
    DANGEROUS = auto()    # Potential safety concern (deception, manipulation)
    BLOCKED = auto()      # Feature activation should be suppressed


# =============================================================================
# Feature Representation
# =============================================================================

@dataclass
class Feature:
    """
    Represents a single interpretable feature from any lens.

    The unified feature representation enables cross-lens analysis
    and a single namespace for all features (sae.42, semantic.Warmth, etc.)
    """
    # Identification
    id: str                              # Unique identifier (e.g., "sae.42", "semantic.Warmth")
    source: FeatureSource                # Which lens provided this feature
    index: int                           # Numeric index within the source

    # Interpretation
    label: str | None = None          # Human-readable label (auto-generated or manual)
    description: str | None = None    # Detailed description
    exemplars: list[str] = field(default_factory=list)  # Example activations

    # Statistics
    activation_mean: float = 0.0         # Mean activation across dataset
    activation_std: float = 0.0          # Std dev of activation
    sparsity: float = 0.0                # How often this feature fires (0-1)

    # Safety
    safety_flag: SafetyFlag = SafetyFlag.SAFE

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize feature ID."""
        if not self.id:
            self.id = f"{self.source.value}.{self.index}"

    @classmethod
    def from_sae(cls, index: int, activation_value: float = 0.0,
                 label: str | None = None) -> "Feature":
        """Create a feature from SAE index."""
        return cls(
            id=f"sae.{index}",
            source=FeatureSource.SAE,
            index=index,
            label=label or f"SAE Feature {index}",
            metadata={"initial_activation": activation_value}
        )

    @classmethod
    def from_semantic(cls, name: str, index: int,
                      positive_exemplars: list[str] = None,
                      negative_exemplars: list[str] = None) -> "Feature":
        """Create a feature from semantic axis."""
        return cls(
            id=f"semantic.{name}",
            source=FeatureSource.SEMANTIC,
            index=index,
            label=name,
            exemplars=positive_exemplars or [],
            metadata={
                "positive_exemplars": positive_exemplars or [],
                "negative_exemplars": negative_exemplars or []
            }
        )


@dataclass
class FeatureActivation:
    """A feature activation at a specific point."""
    feature: Feature
    activation: float           # Activation strength
    confidence: float = 1.0     # Confidence in this activation (for uncertainty)

    # Context
    position: int | None = None      # Token position (if applicable)
    layer: str | None = None         # Layer name (if applicable)

    def __lt__(self, other: "FeatureActivation"):
        """Enable sorting by activation strength."""
        return self.activation < other.activation


@dataclass
class SteeringVector:
    """
    A vector for steering model behavior.

    Computed by combining feature directions from SAE decoder weights
    or semantic axis projections.
    """
    vector: torch.Tensor                 # The actual steering vector [dim]
    features: dict[str, float]           # Feature contributions {feature_id: strength}
    source: LensType                     # Which lens generated this

    # Metadata
    target_description: str | None = None  # What behavior we're steering toward
    safety_validated: bool = False       # Has this been validated for safety?

    def __post_init__(self):
        """Ensure vector is the right shape."""
        if self.vector.dim() != 1:
            raise ValueError(f"SteeringVector must be 1D, got shape {self.vector.shape}")


# =============================================================================
# TraceLens Protocol
# =============================================================================

@runtime_checkable
class TraceLens(Protocol):
    """
    Protocol for interpretability lenses in Dark Trace.

    All interpretability methods (SAE, Semantic Axes, Domain-Specific)
    implement this protocol, enabling unified orchestration.

    Methods:
        project: Map activations to feature space
        get_active: Get active features above threshold
        steer: Generate steering vector for goals
        explain: Generate human-readable explanation
    """

    @property
    def lens_type(self) -> LensType:
        """Return the type of this lens."""
        ...

    @property
    def feature_dim(self) -> int:
        """Return the number of features in this lens."""
        ...

    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Project activations into feature space.

        Args:
            activations: Model activations [batch, seq, dim] or [batch, dim] or [dim]

        Returns:
            Feature activations [batch, seq, n_features] or matching input shape
        """
        ...

    def get_active(
        self,
        activations: torch.Tensor,
        threshold: float = 0.0,
        top_k: int | None = None
    ) -> list[FeatureActivation]:
        """
        Get active features above threshold.

        Args:
            activations: Model activations
            threshold: Minimum activation threshold
            top_k: Return only top-k features (optional)

        Returns:
            List of FeatureActivation objects, sorted by activation (descending)
        """
        ...

    def steer(
        self,
        goals: dict[str, float],
        input_dim: int | None = None
    ) -> SteeringVector:
        """
        Generate steering vector for goals.

        Args:
            goals: Feature goals {feature_id: target_strength}
            input_dim: Dimension of activation space (if different from lens default)

        Returns:
            SteeringVector for applying to activations
        """
        ...

    def explain(
        self,
        activations: torch.Tensor,
        verbosity: int = 1,
        include_features: list[str] | None = None
    ) -> str:
        """
        Generate human-readable explanation of activations.

        Args:
            activations: Model activations to explain
            verbosity: Level of detail (1=brief, 2=moderate, 3=detailed)
            include_features: Only explain these features (optional)

        Returns:
            Human-readable explanation string
        """
        ...


# =============================================================================
# Abstract Base Lens
# =============================================================================

class BaseLens(ABC):
    """
    Abstract base class for TraceLens implementations.

    Provides common functionality and enforces the protocol.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        lens_type: LensType,
        name: str | None = None
    ):
        self._input_dim = input_dim
        self._n_features = n_features
        self._lens_type = lens_type
        self._name = name or f"{lens_type.name}Lens"

        # Feature registry (populated by subclasses)
        self._features: dict[str, Feature] = {}

    @property
    def lens_type(self) -> LensType:
        return self._lens_type

    @property
    def feature_dim(self) -> int:
        return self._n_features

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def name(self) -> str:
        return self._name

    def get_feature(self, feature_id: str) -> Feature | None:
        """Get feature by ID."""
        return self._features.get(feature_id)

    def register_feature(self, feature: Feature) -> None:
        """Register a feature in this lens."""
        self._features[feature.id] = feature

    def list_features(self) -> list[Feature]:
        """List all registered features."""
        return list(self._features.values())

    @abstractmethod
    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """Project activations to feature space."""
        pass

    @abstractmethod
    def get_active(
        self,
        activations: torch.Tensor,
        threshold: float = 0.0,
        top_k: int | None = None
    ) -> list[FeatureActivation]:
        """Get active features."""
        pass

    @abstractmethod
    def steer(
        self,
        goals: dict[str, float],
        input_dim: int | None = None
    ) -> SteeringVector:
        """Generate steering vector."""
        pass

    def explain(
        self,
        activations: torch.Tensor,
        verbosity: int = 1,
        include_features: list[str] | None = None
    ) -> str:
        """
        Default explanation implementation.

        Subclasses can override for lens-specific explanations.
        """
        active = self.get_active(activations, threshold=0.1, top_k=10)

        if include_features:
            active = [a for a in active if a.feature.id in include_features]

        if not active:
            return f"{self.name}: No significant features detected."

        lines = [f"{self.name} Analysis:"]

        if verbosity >= 1:
            # Brief: just top features
            for act in active[:5]:
                label = act.feature.label or act.feature.id
                lines.append(f"  • {label}: {act.activation:.3f}")

        if verbosity >= 2:
            # Moderate: add descriptions
            lines.append("\nFeature Details:")
            for act in active[:5]:
                desc = act.feature.description or "No description"
                lines.append(f"  {act.feature.id}: {desc}")

        if verbosity >= 3:
            # Detailed: add exemplars and metadata
            lines.append("\nExemplars and Statistics:")
            for act in active[:5]:
                lines.append(f"  {act.feature.id}:")
                if act.feature.exemplars:
                    lines.append(f"    Exemplars: {', '.join(act.feature.exemplars[:3])}")
                lines.append(f"    Sparsity: {act.feature.sparsity:.3f}")
                lines.append(f"    Safety: {act.feature.safety_flag.name}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self._lens_type.name}, features={self._n_features})"


# =============================================================================
# Validation Protocols
# =============================================================================

@runtime_checkable
class CausalValidator(Protocol):
    """
    Protocol for causal validation of interpretability claims.

    Implements ablation, injection, and patching for ground truth verification.
    """

    def ablate(
        self,
        activations: torch.Tensor,
        feature_ids: list[str]
    ) -> torch.Tensor:
        """
        Remove specific features from activations.

        Args:
            activations: Original activations
            feature_ids: Features to ablate

        Returns:
            Modified activations with features removed
        """
        ...

    def inject(
        self,
        activations: torch.Tensor,
        feature_values: dict[str, float]
    ) -> torch.Tensor:
        """
        Inject specific feature values into activations.

        Args:
            activations: Original activations
            feature_values: {feature_id: value} to inject

        Returns:
            Modified activations with features injected
        """
        ...

    def patch(
        self,
        source_activations: torch.Tensor,
        target_activations: torch.Tensor,
        feature_ids: list[str]
    ) -> torch.Tensor:
        """
        Patch features from source to target activations.

        Args:
            source_activations: Source of feature values
            target_activations: Target to patch into
            feature_ids: Features to transfer

        Returns:
            Target activations with patched features
        """
        ...


# =============================================================================
# Type Aliases for Convenience
# =============================================================================

# Activation tensor types
Activations = torch.Tensor  # [batch, seq, dim] or [batch, dim] or [dim]

# Feature mapping types
FeatureMap = dict[str, float]  # {feature_id: activation}
GoalMap = dict[str, float]     # {feature_id: target_strength}

# Result types
ActiveFeatures = list[FeatureActivation]
