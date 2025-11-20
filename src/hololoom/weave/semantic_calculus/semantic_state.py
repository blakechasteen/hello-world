#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic State for Policy Integration
======================================

Converts high-dimensional semantic snapshots (244D) into compact feature vectors (8D)
that can be consumed by the policy network for semantic-aware decision making.

Architecture:
    MatryoshkaSnapshot (244D) → SemanticState (8D) → Policy (Neural Network)

The 8D feature vector captures:
1. Momentum (0-1): How aligned are semantic changes across scales?
2. Complexity (0-1): How divergent are different scales?
3. Top 5 dominant dimensions: Which semantic dimensions are most active?
4. Velocity magnitude: How fast is semantic meaning changing?

This enables the policy to:
- Detect topic shifts (low momentum, high complexity)
- Classify conversation purpose (dominant dimensions)
- Select appropriate tools based on semantic dynamics
- Suggest thread branching when topics diverge

Philosophy:
    "The policy doesn't need all 244 dimensions. It needs 8 numbers that tell a story."
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import from existing semantic calculus
try:
    from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSnapshot
    from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum
    SEMANTIC_CALCULUS_AVAILABLE = True
except ImportError:
    SEMANTIC_CALCULUS_AVAILABLE = False
    MatryoshkaSnapshot = None  # Type hint fallback


@dataclass
class SemanticState:
    """
    Compact semantic state for policy integration.

    Provides an 8D feature vector that captures the essential dynamics
    of semantic flow for decision-making.

    Attributes:
        position: Current position in 244D semantic space (full detail)
        velocity: Semantic velocity vector (rate of change)
        acceleration: Semantic acceleration vector (change of velocity)

        momentum: How aligned are semantic changes? (0-1)
                  High = consistent direction, Low = chaotic/diverging
        complexity: How complex is the semantic state? (0-1)
                    High = many active dimensions, Low = focused

        dominant_dimensions: Top 5 most active semantic dimensions
        dimension_values: Values for top 5 dimensions (for interpretability)

        topic_shift_detected: Boolean flag for significant topic change
        shift_magnitude: Magnitude of topic shift (0-1)

    Usage:
        state = SemanticState.from_snapshot(snapshot)
        feature_vector = state.to_feature_vector()  # 8D for policy
    """

    # Full semantic state (244D)
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None

    # Aggregate metrics (scalars)
    momentum: float = 0.0  # 0-1, alignment of semantic changes
    complexity: float = 0.0  # 0-1, diversity of active dimensions

    # Interpretable dimensions
    dominant_dimensions: List[str] = field(default_factory=list)
    dimension_values: List[float] = field(default_factory=list)

    # Topic shift detection
    topic_shift_detected: bool = False
    shift_magnitude: float = 0.0

    # Metadata
    timestamp: float = 0.0
    word_count: int = 0

    @classmethod
    def from_snapshot(
        cls,
        snapshot: 'MatryoshkaSnapshot',
        spectrum: Optional['SemanticSpectrum'] = None,
        shift_threshold: float = 0.6
    ) -> 'SemanticState':
        """
        Create SemanticState from MatryoshkaSnapshot.

        Args:
            snapshot: Multi-scale semantic snapshot
            spectrum: Semantic spectrum for dimension names (optional)
            shift_threshold: Threshold for detecting topic shifts (0-1)

        Returns:
            SemanticState instance with computed metrics
        """
        if not SEMANTIC_CALCULUS_AVAILABLE:
            raise ImportError(
                "Semantic calculus not available. "
                "MatryoshkaSnapshot requires semantic_calculus module."
            )

        # Extract position (use finest scale: paragraph-level 244D projection)
        position = snapshot.paragraph_projection

        # Extract velocity and acceleration if available
        velocity = snapshot.paragraph_velocity if hasattr(snapshot, 'paragraph_velocity') else None
        acceleration = snapshot.paragraph_acceleration if hasattr(snapshot, 'paragraph_acceleration') else None

        # Compute momentum (alignment of semantic changes)
        momentum = cls._compute_momentum(snapshot)

        # Compute complexity (diversity of active dimensions)
        complexity = cls._compute_complexity(position)

        # Extract dominant dimensions
        dominant_dims, dim_values = cls._extract_dominant_dimensions(
            position, spectrum, top_k=5
        )

        # Detect topic shift
        topic_shift_detected = False
        shift_magnitude = 0.0

        if velocity is not None:
            shift_magnitude = np.linalg.norm(velocity)
            topic_shift_detected = shift_magnitude > shift_threshold

        return cls(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            momentum=momentum,
            complexity=complexity,
            dominant_dimensions=dominant_dims,
            dimension_values=dim_values,
            topic_shift_detected=topic_shift_detected,
            shift_magnitude=shift_magnitude,
            timestamp=snapshot.timestamp,
            word_count=snapshot.word_count
        )

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to 8D feature vector for policy.

        Format:
        [0] momentum (0-1)
        [1] complexity (0-1)
        [2-6] top 5 dimension values (normalized)
        [7] velocity magnitude (normalized)

        Returns:
            8D numpy array suitable for policy input
        """
        # Top 5 dimension values (already 0-1 from projection)
        top_5_values = self.dimension_values[:5] if len(self.dimension_values) >= 5 else [0.0] * 5

        # Velocity magnitude (normalize to 0-1)
        velocity_mag = 0.0
        if self.velocity is not None:
            velocity_mag = min(np.linalg.norm(self.velocity), 1.0)

        # Construct 8D vector
        return np.array([
            self.momentum,
            self.complexity,
            *top_5_values,
            velocity_mag
        ], dtype=np.float32)

    @staticmethod
    def _compute_momentum(snapshot: 'MatryoshkaSnapshot') -> float:
        """
        Compute semantic momentum (alignment across scales).

        High momentum = all scales moving in same direction (focused)
        Low momentum = scales diverging (topic shift, confusion)

        Method: Cosine similarity between velocity vectors at different scales

        Returns:
            Momentum score (0-1)
        """
        # Get velocities at different scales
        velocities = []

        if hasattr(snapshot, 'word_velocity') and snapshot.word_velocity is not None:
            velocities.append(snapshot.word_velocity[:16])  # Truncate to common size
        if hasattr(snapshot, 'phrase_velocity') and snapshot.phrase_velocity is not None:
            velocities.append(snapshot.phrase_velocity[:16])
        if hasattr(snapshot, 'sentence_velocity') and snapshot.sentence_velocity is not None:
            velocities.append(snapshot.sentence_velocity[:16])
        if hasattr(snapshot, 'paragraph_velocity') and snapshot.paragraph_velocity is not None:
            velocities.append(snapshot.paragraph_velocity[:16])

        if len(velocities) < 2:
            return 0.5  # Neutral momentum if insufficient data

        # Compute average pairwise cosine similarity
        similarities = []
        for i in range(len(velocities)):
            for j in range(i + 1, len(velocities)):
                v1 = velocities[i]
                v2 = velocities[j]

                # Cosine similarity
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 1e-8 and norm2 > 1e-8:
                    sim = np.dot(v1, v2) / (norm1 * norm2)
                    similarities.append(sim)

        if not similarities:
            return 0.5

        # Average similarity, map from [-1, 1] to [0, 1]
        avg_sim = np.mean(similarities)
        momentum = (avg_sim + 1.0) / 2.0

        return float(np.clip(momentum, 0.0, 1.0))

    @staticmethod
    def _compute_complexity(position: np.ndarray) -> float:
        """
        Compute semantic complexity (diversity of active dimensions).

        High complexity = many dimensions active (rich, nuanced)
        Low complexity = few dimensions active (simple, focused)

        Method: Normalized entropy of dimension activations

        Returns:
            Complexity score (0-1)
        """
        # Compute normalized entropy
        # Treat position values as pseudo-probabilities (normalized)

        # Normalize to [0, 1] and sum to 1
        pos_normalized = np.abs(position)
        pos_sum = np.sum(pos_normalized)

        if pos_sum < 1e-8:
            return 0.0  # No complexity if all zeros

        probs = pos_normalized / pos_sum

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(position))

        if max_entropy < 1e-8:
            return 0.0

        complexity = entropy / max_entropy

        return float(np.clip(complexity, 0.0, 1.0))

    @staticmethod
    def _extract_dominant_dimensions(
        position: np.ndarray,
        spectrum: Optional['SemanticSpectrum'],
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Extract top K dominant semantic dimensions.

        Args:
            position: Semantic position vector (244D)
            spectrum: Semantic spectrum with dimension names (optional)
            top_k: Number of top dimensions to extract

        Returns:
            (dimension_names, dimension_values)
        """
        # Get top K indices by absolute value
        abs_position = np.abs(position)
        top_indices = np.argsort(abs_position)[-top_k:][::-1]

        # Extract values
        top_values = [float(position[i]) for i in top_indices]

        # Extract names if spectrum available
        if spectrum and hasattr(spectrum, 'dimensions'):
            dimension_names = [
                spectrum.dimensions[i].name
                for i in top_indices
                if i < len(spectrum.dimensions)
            ]
        else:
            dimension_names = [f"dim_{i}" for i in top_indices]

        return dimension_names, top_values

    def __repr__(self) -> str:
        """Human-readable representation."""
        shift_str = f" [SHIFT: {self.shift_magnitude:.2f}]" if self.topic_shift_detected else ""
        dims_str = ", ".join(f"{dim}={val:.2f}" for dim, val in zip(
            self.dominant_dimensions[:3], self.dimension_values[:3]
        ))

        return (
            f"SemanticState("
            f"momentum={self.momentum:.2f}, "
            f"complexity={self.complexity:.2f}, "
            f"top_dims=[{dims_str}]"
            f"{shift_str}"
            f")"
        )


class SemanticToolSelector:
    """
    Use semantic state to guide tool selection.

    Maps dominant semantic dimensions to appropriate tools:
    - Confusion/Uncertainty → explain, clarify
    - Curiosity/Learning → explore, search
    - Problem/Conflict → analyze, solve
    - Creative/Transformation → brainstorm, generate

    Also detects when to suggest branching threads based on topic shifts.
    """

    def __init__(self):
        """Initialize semantic tool selector with dimension→tool mappings."""

        # Map semantic dimensions to tools
        self.dimension_to_tool = {
            # Emotional states
            'Confusion': 'explain',
            'Curiosity': 'explore',
            'Fear': 'reassure',
            'Joy': 'celebrate',
            'Anger': 'mediate',

            # Cognitive states
            'Analysis': 'analyze',
            'Synthesis': 'synthesize',
            'Evaluation': 'evaluate',
            'Creation': 'generate',

            # Action states
            'Transformation': 'guide',
            'Conflict': 'mediate',
            'Discovery': 'search',
            'Learning': 'teach',

            # Meta-cognitive
            'Reflection': 'reflect',
            'Planning': 'plan',
            'Execution': 'execute',
        }

    def suggest_tool(self, semantic_state: SemanticState) -> str:
        """
        Suggest tool based on semantic dynamics.

        Args:
            semantic_state: Current semantic state

        Returns:
            Tool name suggestion
        """
        # Check for topic shift → suggest branching
        if semantic_state.topic_shift_detected:
            return 'branch_thread'

        # Check dominant dimensions
        for dim in semantic_state.dominant_dimensions:
            if dim in self.dimension_to_tool:
                return self.dimension_to_tool[dim]

        # Check momentum
        if semantic_state.momentum < 0.3:
            return 'clarify'  # Low momentum = confused user

        # Check complexity
        if semantic_state.complexity > 0.7:
            return 'simplify'  # High complexity = need to focus

        # Default: continue conversation
        return 'answer'

    def should_branch_thread(self, semantic_state: SemanticState) -> bool:
        """
        Determine if a new thread should be branched.

        Criteria:
        - Topic shift detected (shift_magnitude > threshold)
        - Low momentum (scales diverging)
        - High complexity change (sudden increase in dimensions)

        Returns:
            True if branching recommended
        """
        return (
            semantic_state.topic_shift_detected or
            semantic_state.momentum < 0.25 or
            (semantic_state.complexity > 0.8 and semantic_state.shift_magnitude > 0.5)
        )


# Convenience function
def semantic_state_from_text(
    text: str,
    embedder,
    spectrum: Optional['SemanticSpectrum'] = None
) -> SemanticState:
    """
    Create SemanticState directly from text (simplified API).

    This is a convenience function for quick semantic analysis without
    needing to set up the full streaming pipeline.

    Args:
        text: Input text
        embedder: Matryoshka embedder
        spectrum: Semantic spectrum (optional)

    Returns:
        SemanticState instance
    """
    if not SEMANTIC_CALCULUS_AVAILABLE:
        raise ImportError("Semantic calculus not available")

    from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus

    # Create streaming calculus
    calculus = MatryoshkaSemanticCalculus(
        matryoshka_embedder=embedder,
        snapshot_interval=1.0
    )

    # Analyze text (simple tokenization)
    async def text_stream():
        for word in text.split():
            yield word

    # Get final snapshot (synchronous wrapper)
    import asyncio
    snapshot = None

    async def analyze():
        nonlocal snapshot
        async for snap in calculus.stream_analyze(text_stream()):
            snapshot = snap

    asyncio.run(analyze())

    if snapshot is None:
        raise ValueError("Failed to analyze text")

    # Convert to semantic state
    return SemanticState.from_snapshot(snapshot, spectrum)
