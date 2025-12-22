"""
Dark Trace Bridge: SemanticLens - TraceLens for Semantic Axes

Implements the TraceLens protocol for HoloLoom's 244 semantic dimensions,
enabling unified interpretability with SAE features.

The SemanticLens provides:
- Projection onto 244 interpretable axes (Warmth, Valence, Formality, etc.)
- Steering via semantic goals
- Human-readable explanations

Research Basis:
- Semantic axes learned from conjugate exemplar pairs
- 16 standard dimensions + 228 extended (narrative, emotional, archetypal, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import torch

from HoloLoom.dark_trace.protocol import (
    TraceLens,
    BaseLens,
    LensType,
    Feature,
    FeatureSource,
    FeatureActivation,
    SteeringVector,
    SafetyFlag,
)

# Import semantic dimensions
try:
    from HoloLoom.semantic_calculus.dimensions import (
        SemanticDimension,
        SemanticSpectrum,
        STANDARD_DIMENSIONS,
        EXTENDED_244_DIMENSIONS,
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    STANDARD_DIMENSIONS = []
    EXTENDED_244_DIMENSIONS = []


@dataclass
class SemanticLensConfig:
    """Configuration for SemanticLens."""

    # Which dimensions to use
    use_extended: bool = False  # Use 244 extended dims or 16 standard

    # Projection settings
    normalize_projections: bool = True  # Normalize to [-1, 1]

    # Feature registration
    register_features: bool = True

    # Minimum projection to consider "active"
    activation_threshold: float = 0.1

    @classmethod
    def standard(cls) -> "SemanticLensConfig":
        """Standard 16-dimension configuration."""
        return cls(use_extended=False)

    @classmethod
    def extended(cls) -> "SemanticLensConfig":
        """Extended 244-dimension configuration."""
        return cls(use_extended=True)


class SemanticLens(BaseLens):
    """
    Semantic interpretability lens implementing TraceLens protocol.

    Projects activations onto interpretable semantic axes learned from
    conjugate exemplar pairs (e.g., warm/cold, formal/casual).

    Usage:
        # Create lens with embedding function
        lens = SemanticLens(embed_fn, input_dim=384)

        # Project activations
        semantic_features = lens.project(activations)

        # Get active semantic dimensions
        active = lens.get_active(activations, top_k=10)

        # Steer toward semantic goals
        steering = lens.steer({"semantic.Warmth": 0.8, "semantic.Formality": -0.5})

        # Explain in human terms
        explanation = lens.explain(activations, verbosity=2)
    """

    def __init__(
        self,
        embed_fn: Optional[Callable] = None,
        input_dim: int = 384,
        config: Optional[SemanticLensConfig] = None,
        name: str = "SemanticLens",
    ):
        """
        Initialize SemanticLens.

        Args:
            embed_fn: Function to embed words (required for axis learning)
            input_dim: Dimension of input activations
            config: Configuration options
            name: Name for this lens
        """
        self.config = config or SemanticLensConfig.standard()

        # Select dimensions based on config
        if not SEMANTIC_AVAILABLE:
            raise ImportError(
                "semantic_calculus module not available. "
                "Cannot create SemanticLens without semantic dimensions."
            )

        self._dimensions = (
            EXTENDED_244_DIMENSIONS if self.config.use_extended
            else STANDARD_DIMENSIONS
        )

        n_features = len(self._dimensions)

        super().__init__(
            input_dim=input_dim,
            n_features=n_features,
            lens_type=LensType.SEMANTIC,
            name=name,
        )

        self.embed_fn = embed_fn
        self._spectrum = SemanticSpectrum(self._dimensions)
        self._axes_learned = False

        # Projection matrix (learned from axes)
        self._projection_matrix: Optional[torch.Tensor] = None

        # Register semantic features
        if self.config.register_features:
            self._register_semantic_features()

    def _register_semantic_features(self) -> None:
        """Register all semantic dimensions as features."""
        for idx, dim in enumerate(self._dimensions):
            feature = Feature.from_semantic(
                name=dim.name,
                index=idx,
                positive_exemplars=dim.positive_exemplars,
                negative_exemplars=dim.negative_exemplars,
            )
            self.register_feature(feature)

    def learn_axes(self, embed_fn: Optional[Callable] = None) -> None:
        """
        Learn semantic axes from exemplar words.

        Args:
            embed_fn: Optional new embedding function (overrides constructor)
        """
        if embed_fn is not None:
            self.embed_fn = embed_fn

        if self.embed_fn is None:
            raise ValueError(
                "No embedding function provided. "
                "Pass embed_fn to constructor or learn_axes()."
            )

        # Learn axes for each dimension
        self._spectrum.learn_axes(self.embed_fn)

        # Build projection matrix from learned axes
        axes = []
        for dim in self._dimensions:
            if dim.axis is not None:
                axes.append(dim.axis)
            else:
                # Fallback: zero axis if learning failed
                axes.append(np.zeros(self._input_dim))

        self._projection_matrix = torch.tensor(
            np.stack(axes, axis=0),  # [n_dims, input_dim]
            dtype=torch.float32
        )

        self._axes_learned = True

    # =========================================================================
    # TraceLens Protocol Implementation
    # =========================================================================

    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Project activations onto semantic dimensions.

        Args:
            activations: Input activations [batch, seq, dim] or [batch, dim] or [dim]

        Returns:
            Semantic projections with same batch structure
        """
        if not self._axes_learned:
            raise ValueError(
                "Semantic axes not learned. Call learn_axes() first."
            )

        # Handle different input shapes
        original_shape = activations.shape
        original_dim = activations.dim()

        # Flatten to [N, input_dim]
        if original_dim == 1:
            flat = activations.unsqueeze(0)  # [1, dim]
        elif original_dim == 2:
            flat = activations  # [batch, dim]
        elif original_dim == 3:
            batch, seq, dim = activations.shape
            flat = activations.reshape(-1, dim)  # [batch*seq, dim]
        else:
            raise ValueError(f"Unsupported activation shape: {original_shape}")

        # Move projection matrix to same device
        proj_matrix = self._projection_matrix.to(flat.device)

        # Project: [N, input_dim] @ [input_dim, n_dims] -> [N, n_dims]
        projections = flat @ proj_matrix.T

        # Normalize if configured
        if self.config.normalize_projections:
            # Soft normalization to roughly [-1, 1]
            projections = torch.tanh(projections)

        # Restore shape
        if original_dim == 1:
            return projections.squeeze(0)  # [n_dims]
        elif original_dim == 2:
            return projections  # [batch, n_dims]
        else:
            return projections.reshape(batch, seq, -1)  # [batch, seq, n_dims]

    def get_active(
        self,
        activations: torch.Tensor,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> List[FeatureActivation]:
        """
        Get active semantic dimensions above threshold.

        Args:
            activations: Input activations
            threshold: Minimum absolute projection to consider active
            top_k: Return only top-k dimensions (optional)

        Returns:
            List of FeatureActivation sorted by absolute projection (descending)
        """
        if threshold == 0.0:
            threshold = self.config.activation_threshold

        # Project to semantic space
        semantic_acts = self.project(activations)

        # Handle batched input by taking mean
        if semantic_acts.dim() > 1:
            semantic_acts = semantic_acts.mean(dim=0)  # [n_dims]

        # Filter by absolute threshold
        abs_acts = torch.abs(semantic_acts)
        active_mask = abs_acts > threshold
        active_indices = torch.where(active_mask)[0]
        active_values = semantic_acts[active_indices]
        active_abs = abs_acts[active_indices]

        # Sort by absolute value (descending)
        sorted_order = torch.argsort(active_abs, descending=True)
        active_indices = active_indices[sorted_order]
        active_values = active_values[sorted_order]

        # Limit to top_k
        if top_k is not None:
            active_indices = active_indices[:top_k]
            active_values = active_values[:top_k]

        # Build FeatureActivation list
        result = []
        for idx, val in zip(active_indices.tolist(), active_values.tolist()):
            dim = self._dimensions[idx]
            feature = self.get_feature(f"semantic.{dim.name}")

            if feature is None:
                feature = Feature.from_semantic(
                    name=dim.name,
                    index=idx,
                    positive_exemplars=dim.positive_exemplars,
                    negative_exemplars=dim.negative_exemplars,
                )

            result.append(FeatureActivation(
                feature=feature,
                activation=val,
                confidence=1.0,
            ))

        return result

    def steer(
        self,
        goals: Dict[str, float],
        input_dim: Optional[int] = None,
    ) -> SteeringVector:
        """
        Generate steering vector from semantic goals.

        Uses semantic axes: steering = sum(goal[i] * axis[i])

        Args:
            goals: Semantic goals {"semantic.Warmth": 0.8, "semantic.Formality": -0.5}
            input_dim: Unused (determined by lens)

        Returns:
            SteeringVector for adding to activations
        """
        if not self._axes_learned:
            raise ValueError(
                "Semantic axes not learned. Call learn_axes() first."
            )

        # Build steering vector
        steering = torch.zeros(self._input_dim, dtype=torch.float32)

        for feature_id, target_strength in goals.items():
            # Parse feature ID
            if not feature_id.startswith("semantic."):
                continue

            dim_name = feature_id.replace("semantic.", "")

            # Find dimension by name
            dim_idx = None
            for idx, dim in enumerate(self._dimensions):
                if dim.name == dim_name:
                    dim_idx = idx
                    break

            if dim_idx is None:
                continue

            # Add axis scaled by target strength
            axis = self._projection_matrix[dim_idx]
            steering += target_strength * axis

        return SteeringVector(
            vector=steering,
            features=goals,
            source=LensType.SEMANTIC,
            target_description=f"Steer {len(goals)} semantic dimensions",
            safety_validated=False,
        )

    def explain(
        self,
        activations: torch.Tensor,
        verbosity: int = 1,
        include_features: Optional[List[str]] = None,
    ) -> str:
        """
        Generate human-readable explanation of semantic activations.

        Args:
            activations: Input activations to explain
            verbosity: Detail level (1=brief, 2=moderate, 3=detailed)
            include_features: Only explain these features (optional)

        Returns:
            Human-readable explanation string
        """
        active = self.get_active(activations, threshold=0.1, top_k=10)

        if include_features:
            active = [a for a in active if a.feature.id in include_features]

        if not active:
            return f"{self.name}: No significant semantic dimensions detected."

        lines = [f"{self.name} Analysis ({self._n_features} dimensions):"]

        # Brief: top dimensions with direction
        if verbosity >= 1:
            lines.append("\nTop Active Dimensions:")
            for act in active[:5]:
                direction = "+" if act.activation > 0 else "-"
                pole = (
                    act.feature.metadata.get("positive_exemplars", ["positive"])[0]
                    if act.activation > 0
                    else act.feature.metadata.get("negative_exemplars", ["negative"])[0]
                )
                lines.append(
                    f"  • {act.feature.label}: {direction}{abs(act.activation):.3f} "
                    f"(toward '{pole}')"
                )

        # Moderate: add exemplar pairs
        if verbosity >= 2:
            lines.append("\nSemantic Poles:")
            for act in active[:5]:
                pos = act.feature.metadata.get("positive_exemplars", [])
                neg = act.feature.metadata.get("negative_exemplars", [])
                pos_str = ", ".join(pos[:3]) if pos else "?"
                neg_str = ", ".join(neg[:3]) if neg else "?"
                lines.append(f"  {act.feature.label}: [{neg_str}] ↔ [{pos_str}]")

        # Detailed: interpretation
        if verbosity >= 3:
            lines.append("\nInterpretation:")
            for act in active[:5]:
                direction = "positive" if act.activation > 0 else "negative"
                strength = "strongly" if abs(act.activation) > 0.5 else "slightly"
                lines.append(
                    f"  {act.feature.label}: {strength} {direction} "
                    f"(projection: {act.activation:.4f})"
                )

        return "\n".join(lines)

    # =========================================================================
    # Additional Methods
    # =========================================================================

    def get_dimension_names(self) -> List[str]:
        """Get list of all dimension names."""
        return [dim.name for dim in self._dimensions]

    def get_dimension(self, name: str) -> Optional[SemanticDimension]:
        """Get a semantic dimension by name."""
        for dim in self._dimensions:
            if dim.name == name:
                return dim
        return None

    def project_to_dict(self, activations: torch.Tensor) -> Dict[str, float]:
        """
        Project activations and return as dimension-name dict.

        Args:
            activations: Input activations

        Returns:
            Dict mapping dimension name to projection value
        """
        projections = self.project(activations)

        # Handle batched by taking mean
        if projections.dim() > 1:
            projections = projections.mean(dim=0)

        return {
            dim.name: projections[idx].item()
            for idx, dim in enumerate(self._dimensions)
        }

    def compare_projections(
        self,
        activations1: torch.Tensor,
        activations2: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare semantic projections of two activations.

        Useful for understanding how responses differ semantically.

        Returns:
            Dict with 'projection1', 'projection2', and 'difference'
        """
        proj1 = self.project_to_dict(activations1)
        proj2 = self.project_to_dict(activations2)

        diff = {
            name: proj2[name] - proj1[name]
            for name in proj1
        }

        return {
            "projection1": proj1,
            "projection2": proj2,
            "difference": diff,
        }

    def get_axes_tensor(self) -> Optional[torch.Tensor]:
        """Get the learned axes as a tensor [n_dims, input_dim]."""
        return self._projection_matrix

    def save(self, path: str) -> None:
        """Save learned axes to file."""
        if not self._axes_learned:
            raise ValueError("No axes to save. Call learn_axes() first.")

        torch.save({
            "projection_matrix": self._projection_matrix,
            "dimension_names": [d.name for d in self._dimensions],
            "input_dim": self._input_dim,
            "n_features": self._n_features,
        }, path)

    def load(self, path: str) -> None:
        """Load learned axes from file."""
        state = torch.load(path)
        self._projection_matrix = state["projection_matrix"]
        self._axes_learned = True


def create_semantic_lens(
    embed_fn: Optional[Callable] = None,
    input_dim: int = 384,
    use_extended: bool = False,
) -> SemanticLens:
    """
    Factory function to create SemanticLens.

    Args:
        embed_fn: Embedding function for axis learning
        input_dim: Input activation dimension
        use_extended: Use 244 extended dimensions (vs 16 standard)

    Returns:
        Configured SemanticLens
    """
    config = (
        SemanticLensConfig.extended() if use_extended
        else SemanticLensConfig.standard()
    )

    lens = SemanticLens(
        embed_fn=embed_fn,
        input_dim=input_dim,
        config=config,
    )

    if embed_fn is not None:
        lens.learn_axes()

    return lens
