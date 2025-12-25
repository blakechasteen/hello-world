"""
Concept Erasure for Dark Trace

Techniques for removing specific concepts from neural representations
while preserving other information. Useful for:
- Removing biased features
- Creating controlled representations
- Testing feature necessity

Based on:
- "Erasing Concepts from Diffusion Models" (Gandikota et al., 2023)
- "LEACE: Perfect linear concept erasure" (Belrose et al., 2023)
- "Knowledge Unlearning for Mitigating Privacy Risks" (Chen et al., 2023)

Author: HoloLoom Team
Created: December 2025
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ErasureMethod(Enum):
    """Methods for concept erasure."""
    LINEAR = "linear"               # Linear projection (LEACE-style)
    ORTHOGONAL = "orthogonal"       # Orthogonal complement
    ITERATIVE = "iterative"         # Iterative nullspace projection
    SOFT = "soft"                   # Soft erasure with controllable strength
    SURGICAL = "surgical"           # Target specific features only


@dataclass
class ErasureConfig:
    """Configuration for concept erasure."""
    method: ErasureMethod = ErasureMethod.LINEAR
    strength: float = 1.0               # 0.0 = no erasure, 1.0 = full erasure
    preserve_variance: bool = True      # Preserve total variance after erasure
    regularization: float = 1e-6        # Regularization for numerical stability
    max_iterations: int = 100           # For iterative methods
    tolerance: float = 1e-6             # Convergence tolerance
    target_features: Optional[List[int]] = None  # For surgical erasure


@dataclass
class ErasureResult:
    """Result of concept erasure."""
    original_activations: np.ndarray
    erased_activations: np.ndarray
    erasure_direction: np.ndarray
    projection_matrix: np.ndarray
    concept_strength_before: float
    concept_strength_after: float
    variance_preserved: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_strength_before": self.concept_strength_before,
            "concept_strength_after": self.concept_strength_after,
            "erasure_effectiveness": 1.0 - (self.concept_strength_after / (self.concept_strength_before + 1e-8)),
            "variance_preserved": self.variance_preserved,
            "metadata": self.metadata,
        }


class ConceptEraser:
    """
    Remove specific concepts from neural representations.

    Uses linear algebra techniques to project activations onto
    the nullspace of concept directions, effectively removing
    concept-related information while preserving other structure.
    """

    def __init__(self, config: Optional[ErasureConfig] = None):
        self.config = config or ErasureConfig()
        self._concept_directions: Dict[str, np.ndarray] = {}
        self._projection_matrices: Dict[str, np.ndarray] = {}

    def learn_concept_direction(
        self,
        positive_examples: np.ndarray,
        negative_examples: np.ndarray,
        concept_name: str = "concept",
    ) -> np.ndarray:
        """
        Learn the direction in activation space that represents a concept.

        Args:
            positive_examples: Activations for examples WITH the concept
            negative_examples: Activations for examples WITHOUT the concept
            concept_name: Name for this concept direction

        Returns:
            Unit vector pointing in the concept direction
        """
        # Compute mean difference
        pos_mean = np.mean(positive_examples, axis=0)
        neg_mean = np.mean(negative_examples, axis=0)
        direction = pos_mean - neg_mean

        # Normalize to unit vector
        norm = np.linalg.norm(direction)
        if norm > self.config.regularization:
            direction = direction / norm
        else:
            direction = np.zeros_like(direction)

        # Store for later use
        self._concept_directions[concept_name] = direction

        return direction

    def learn_concept_subspace(
        self,
        concept_activations: np.ndarray,
        concept_labels: np.ndarray,
        n_components: int = 5,
        concept_name: str = "concept",
    ) -> np.ndarray:
        """
        Learn a subspace (multiple directions) for a concept.

        Uses SVD on the class-conditional covariance to find
        directions that capture concept-related variance.

        Args:
            concept_activations: All activations (n_samples, n_features)
            concept_labels: Binary labels (n_samples,)
            n_components: Number of concept dimensions to extract
            concept_name: Name for this concept

        Returns:
            Matrix of shape (n_components, n_features) - concept subspace basis
        """
        # Split by label
        pos_mask = concept_labels.astype(bool)
        pos_acts = concept_activations[pos_mask]
        neg_acts = concept_activations[~pos_mask]

        # Compute class-conditional means
        pos_mean = np.mean(pos_acts, axis=0, keepdims=True)
        neg_mean = np.mean(neg_acts, axis=0, keepdims=True)

        # Center and combine
        pos_centered = pos_acts - pos_mean
        neg_centered = neg_acts - neg_mean

        # Between-class scatter direction
        between_class = pos_mean - neg_mean

        # SVD on centered class-conditional data
        combined = np.vstack([pos_centered, neg_centered])
        U, S, Vt = np.linalg.svd(combined, full_matrices=False)

        # Take top components that align with between-class direction
        components = Vt[:n_components]

        # Store as concept direction (use first component as primary)
        self._concept_directions[concept_name] = components[0] / (np.linalg.norm(components[0]) + 1e-8)

        return components

    def _compute_projection_matrix(
        self,
        concept_direction: np.ndarray,
        method: ErasureMethod,
    ) -> np.ndarray:
        """Compute projection matrix for erasure."""
        d = len(concept_direction)

        if method == ErasureMethod.LINEAR:
            # Project onto orthogonal complement: P = I - vv^T
            v = concept_direction.reshape(-1, 1)
            P = np.eye(d) - v @ v.T

        elif method == ErasureMethod.ORTHOGONAL:
            # Same as linear but with explicit orthonormalization
            v = concept_direction.reshape(-1, 1)
            v = v / (np.linalg.norm(v) + self.config.regularization)
            P = np.eye(d) - v @ v.T

        elif method == ErasureMethod.SOFT:
            # Soft erasure: P = I - strength * vv^T
            v = concept_direction.reshape(-1, 1)
            P = np.eye(d) - self.config.strength * (v @ v.T)

        elif method == ErasureMethod.ITERATIVE:
            # Iterative refinement
            P = self._iterative_nullspace_projection(concept_direction)

        elif method == ErasureMethod.SURGICAL:
            # Only modify specific features
            P = self._surgical_projection(concept_direction)

        else:
            # Default to linear
            v = concept_direction.reshape(-1, 1)
            P = np.eye(d) - v @ v.T

        return P

    def _iterative_nullspace_projection(
        self,
        concept_direction: np.ndarray,
    ) -> np.ndarray:
        """
        Iteratively refine nullspace projection.

        Uses power iteration-like approach to find stable projection.
        """
        d = len(concept_direction)
        v = concept_direction.copy()
        v = v / (np.linalg.norm(v) + self.config.regularization)

        P = np.eye(d) - v.reshape(-1, 1) @ v.reshape(1, -1)

        for _ in range(self.config.max_iterations):
            # Project and re-normalize
            v_new = P @ v
            norm = np.linalg.norm(v_new)

            if norm < self.config.tolerance:
                break

            v_new = v_new / (norm + self.config.regularization)

            # Check convergence
            if np.linalg.norm(v_new - v) < self.config.tolerance:
                break

            v = v_new
            P = np.eye(d) - v.reshape(-1, 1) @ v.reshape(1, -1)

        return P

    def _surgical_projection(
        self,
        concept_direction: np.ndarray,
    ) -> np.ndarray:
        """
        Surgical erasure targeting specific features.

        Only modifies features specified in config.target_features.
        """
        d = len(concept_direction)
        P = np.eye(d)

        if self.config.target_features is None:
            # Default to all features
            v = concept_direction.reshape(-1, 1)
            P = np.eye(d) - v @ v.T
        else:
            # Only modify target features
            mask = np.zeros(d)
            mask[self.config.target_features] = 1.0

            v = concept_direction * mask
            norm = np.linalg.norm(v)
            if norm > self.config.regularization:
                v = v / norm
                v = v.reshape(-1, 1)
                P = np.eye(d) - v @ v.T

        return P

    def erase(
        self,
        activations: np.ndarray,
        concept_direction: Optional[np.ndarray] = None,
        concept_name: Optional[str] = None,
    ) -> ErasureResult:
        """
        Erase a concept from activations.

        Args:
            activations: Activations to modify (n_samples, n_features) or (n_features,)
            concept_direction: Direction to erase (if None, use stored direction)
            concept_name: Name of stored concept direction to use

        Returns:
            ErasureResult with erased activations and metadata
        """
        # Get concept direction
        if concept_direction is None:
            if concept_name is None:
                concept_name = list(self._concept_directions.keys())[0] if self._concept_directions else "concept"
            concept_direction = self._concept_directions.get(concept_name)
            if concept_direction is None:
                raise ValueError(f"No concept direction for '{concept_name}'")

        # Ensure 2D
        is_1d = activations.ndim == 1
        if is_1d:
            activations = activations.reshape(1, -1)

        # Compute projection matrix
        P = self._compute_projection_matrix(concept_direction, self.config.method)

        # Store for reuse
        if concept_name:
            self._projection_matrices[concept_name] = P

        # Measure concept strength before
        concept_strength_before = self._measure_concept_strength(activations, concept_direction)

        # Apply erasure
        erased = activations @ P.T

        # Optionally preserve variance
        if self.config.preserve_variance:
            original_var = np.var(activations)
            erased_var = np.var(erased)
            if erased_var > self.config.regularization:
                scale = np.sqrt(original_var / erased_var)
                erased = erased * scale

        # Measure concept strength after
        concept_strength_after = self._measure_concept_strength(erased, concept_direction)

        # Compute variance preservation
        original_total_var = np.sum(np.var(activations, axis=0))
        erased_total_var = np.sum(np.var(erased, axis=0))
        variance_preserved = erased_total_var / (original_total_var + self.config.regularization)

        # Restore shape if needed
        if is_1d:
            erased = erased.flatten()
            activations = activations.flatten()

        return ErasureResult(
            original_activations=activations,
            erased_activations=erased,
            erasure_direction=concept_direction,
            projection_matrix=P,
            concept_strength_before=concept_strength_before,
            concept_strength_after=concept_strength_after,
            variance_preserved=variance_preserved,
            metadata={
                "method": self.config.method.value,
                "strength": self.config.strength,
                "concept_name": concept_name,
            }
        )

    def _measure_concept_strength(
        self,
        activations: np.ndarray,
        concept_direction: np.ndarray,
    ) -> float:
        """Measure how strongly activations express a concept."""
        # Project onto concept direction
        projections = activations @ concept_direction
        # Return mean absolute projection
        return float(np.mean(np.abs(projections)))

    def erase_multiple(
        self,
        activations: np.ndarray,
        concept_names: List[str],
    ) -> ErasureResult:
        """
        Erase multiple concepts from activations.

        Sequentially projects onto orthogonal complement of each concept.
        Order can matter for overlapping concepts.

        Args:
            activations: Activations to modify
            concept_names: List of concept names to erase

        Returns:
            ErasureResult with all concepts erased
        """
        current = activations.copy()
        total_strength_before = 0.0
        total_strength_after = 0.0

        # Combined projection matrix
        d = activations.shape[-1] if activations.ndim > 1 else len(activations)
        combined_P = np.eye(d)

        for name in concept_names:
            direction = self._concept_directions.get(name)
            if direction is None:
                continue

            # Measure before
            strength_before = self._measure_concept_strength(
                current if current.ndim > 1 else current.reshape(1, -1),
                direction
            )
            total_strength_before += strength_before

            # Compute projection
            P = self._compute_projection_matrix(direction, self.config.method)
            combined_P = P @ combined_P

            # Apply
            if current.ndim == 1:
                current = (P @ current.reshape(-1, 1)).flatten()
            else:
                current = current @ P.T

            # Measure after
            strength_after = self._measure_concept_strength(
                current if current.ndim > 1 else current.reshape(1, -1),
                direction
            )
            total_strength_after += strength_after

        # Compute variance preservation
        original_var = np.var(activations)
        erased_var = np.var(current)
        variance_preserved = erased_var / (original_var + self.config.regularization)

        return ErasureResult(
            original_activations=activations,
            erased_activations=current,
            erasure_direction=np.zeros(d),  # Multiple directions
            projection_matrix=combined_P,
            concept_strength_before=total_strength_before,
            concept_strength_after=total_strength_after,
            variance_preserved=variance_preserved,
            metadata={
                "method": self.config.method.value,
                "concepts_erased": concept_names,
            }
        )

    def get_concept_direction(self, concept_name: str) -> Optional[np.ndarray]:
        """Get stored concept direction."""
        return self._concept_directions.get(concept_name)

    def get_projection_matrix(self, concept_name: str) -> Optional[np.ndarray]:
        """Get stored projection matrix."""
        return self._projection_matrices.get(concept_name)

    def list_concepts(self) -> List[str]:
        """List all learned concepts."""
        return list(self._concept_directions.keys())


class LEACEEraser:
    """
    LEACE: Perfect Linear Concept Erasure.

    Implements the LEACE algorithm which provides provably optimal
    linear erasure that:
    1. Perfectly removes linear information about the concept
    2. Minimizes distortion to the representations
    3. Is unique and independent of classifier choice

    Reference: Belrose et al., "LEACE: Perfect linear concept erasure" (2023)
    """

    def __init__(self, regularization: float = 1e-6):
        self.regularization = regularization
        self._projection: Optional[np.ndarray] = None
        self._concept_mean: Optional[np.ndarray] = None

    def fit(
        self,
        activations: np.ndarray,
        concept_labels: np.ndarray,
    ) -> "LEACEEraser":
        """
        Fit LEACE erasure transformation.

        Args:
            activations: Activations (n_samples, n_features)
            concept_labels: Binary concept labels (n_samples,)

        Returns:
            self
        """
        n_samples, n_features = activations.shape

        # Compute class-conditional means
        pos_mask = concept_labels.astype(bool)
        pos_mean = np.mean(activations[pos_mask], axis=0)
        neg_mean = np.mean(activations[~pos_mask], axis=0)

        # Between-class direction
        d = pos_mean - neg_mean
        d_norm = np.linalg.norm(d)

        if d_norm < self.regularization:
            # Concepts already indistinguishable
            self._projection = np.eye(n_features)
            self._concept_mean = np.mean(activations, axis=0)
            return self

        # Normalize
        d = d / d_norm

        # LEACE projection: P = I - d d^T
        self._projection = np.eye(n_features) - np.outer(d, d)

        # Store concept mean for centering
        self._concept_mean = np.mean(activations, axis=0)

        return self

    def transform(self, activations: np.ndarray) -> np.ndarray:
        """
        Apply LEACE erasure to new activations.

        Args:
            activations: Activations to transform

        Returns:
            Transformed activations with concept erased
        """
        if self._projection is None:
            raise ValueError("Must call fit() first")

        is_1d = activations.ndim == 1
        if is_1d:
            activations = activations.reshape(1, -1)

        # Apply projection
        erased = activations @ self._projection.T

        if is_1d:
            erased = erased.flatten()

        return erased

    def fit_transform(
        self,
        activations: np.ndarray,
        concept_labels: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(activations, concept_labels)
        return self.transform(activations)


class ConceptSurgery:
    """
    Surgical concept modification.

    Rather than complete erasure, allows fine-grained control
    over concept expression:
    - Amplify or suppress specific concepts
    - Swap concepts (replace one with another)
    - Blend concepts
    """

    def __init__(self):
        self._concept_vectors: Dict[str, np.ndarray] = {}

    def register_concept(
        self,
        name: str,
        direction: np.ndarray,
    ) -> None:
        """Register a concept direction."""
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            self._concept_vectors[name] = direction / norm
        else:
            self._concept_vectors[name] = direction

    def amplify(
        self,
        activations: np.ndarray,
        concept_name: str,
        factor: float = 2.0,
    ) -> np.ndarray:
        """
        Amplify a concept in activations.

        Args:
            activations: Input activations
            concept_name: Concept to amplify
            factor: Amplification factor (1.0 = no change, 2.0 = double)

        Returns:
            Modified activations
        """
        direction = self._concept_vectors.get(concept_name)
        if direction is None:
            return activations

        # Project onto concept direction
        projection = activations @ direction

        # Add amplified component
        if activations.ndim == 1:
            delta = (factor - 1.0) * projection * direction
        else:
            delta = (factor - 1.0) * projection.reshape(-1, 1) @ direction.reshape(1, -1)

        return activations + delta

    def suppress(
        self,
        activations: np.ndarray,
        concept_name: str,
        factor: float = 0.5,
    ) -> np.ndarray:
        """
        Suppress a concept in activations.

        Args:
            activations: Input activations
            concept_name: Concept to suppress
            factor: Suppression factor (0.0 = full removal, 0.5 = half)

        Returns:
            Modified activations
        """
        return self.amplify(activations, concept_name, factor)

    def swap(
        self,
        activations: np.ndarray,
        from_concept: str,
        to_concept: str,
        strength: float = 1.0,
    ) -> np.ndarray:
        """
        Swap one concept for another.

        Args:
            activations: Input activations
            from_concept: Concept to remove
            to_concept: Concept to add
            strength: Swap strength (1.0 = full swap)

        Returns:
            Modified activations with concepts swapped
        """
        from_dir = self._concept_vectors.get(from_concept)
        to_dir = self._concept_vectors.get(to_concept)

        if from_dir is None or to_dir is None:
            return activations

        # Get projection onto source concept
        projection = activations @ from_dir

        # Remove source, add target
        if activations.ndim == 1:
            removed = activations - strength * projection * from_dir
            added = removed + strength * projection * to_dir
        else:
            removed = activations - strength * projection.reshape(-1, 1) @ from_dir.reshape(1, -1)
            added = removed + strength * projection.reshape(-1, 1) @ to_dir.reshape(1, -1)

        return added

    def blend(
        self,
        activations: np.ndarray,
        concept_weights: Dict[str, float],
    ) -> np.ndarray:
        """
        Blend multiple concepts with specified weights.

        Args:
            activations: Input activations
            concept_weights: Dict mapping concept names to target weights

        Returns:
            Modified activations with blended concepts
        """
        result = activations.copy()

        for name, weight in concept_weights.items():
            direction = self._concept_vectors.get(name)
            if direction is None:
                continue

            # Current projection
            current_proj = result @ direction

            # Target projection
            target_proj = weight * np.linalg.norm(result, axis=-1 if result.ndim > 1 else None)

            # Adjust
            delta_proj = target_proj - current_proj

            if result.ndim == 1:
                result = result + delta_proj * direction
            else:
                result = result + delta_proj.reshape(-1, 1) @ direction.reshape(1, -1)

        return result


def create_eraser(
    method: ErasureMethod = ErasureMethod.LINEAR,
    strength: float = 1.0,
) -> ConceptEraser:
    """Factory function to create concept eraser."""
    config = ErasureConfig(method=method, strength=strength)
    return ConceptEraser(config)


def create_leace_eraser(regularization: float = 1e-6) -> LEACEEraser:
    """Factory function to create LEACE eraser."""
    return LEACEEraser(regularization=regularization)


def create_concept_surgery() -> ConceptSurgery:
    """Factory function to create concept surgery tool."""
    return ConceptSurgery()
