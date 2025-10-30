"""
Awareness Types - Minimal, elegant data structures for living memory.

Design principles:
1. Immutable memories (content is ground truth)
2. Position/topology/activation are indices (recomputable)
3. Minimal new abstractions (compose existing types)
4. All types are serializable
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
from enum import Enum
import numpy as np


# =============================================================================
# Semantic Perception (Output of semantic calculus)
# =============================================================================

@dataclass
class SemanticPerception:
    """
    Result of perceiving input through semantic calculus.

    This is what awareness "sees" when it processes text.
    """
    # Where in 244D semantic space
    position: np.ndarray  # 244D

    # How meaning is changing
    velocity: Optional[np.ndarray] = None  # 244D, if trajectory exists

    # Interpretable semantics
    dominant_dimensions: List[str] = field(default_factory=list)  # ['Heroism', 'Tension', ...]

    # Aggregate metrics
    momentum: float = 0.0      # 0-1, scale alignment
    complexity: float = 0.0    # 0-1, scale divergence

    # Shift detection
    shift_magnitude: float = 0.0
    shift_detected: bool = False

    @classmethod
    def from_snapshot(cls, snapshot, prev_position: Optional[np.ndarray] = None) -> 'SemanticPerception':
        """
        Create from MatryoshkaSnapshot (semantic calculus output).

        Args:
            snapshot: MatryoshkaSnapshot from semantic calculus
            prev_position: Previous position for velocity/shift calculation
        """
        from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaScale

        # Extract paragraph-level state (richest)
        para_state = snapshot.states_by_scale.get(MatryoshkaScale.PARAGRAPH, {})
        position = para_state.get('position')

        if position is None:
            # Fallback to zeros if no position (228D = EXTENDED_244_DIMENSIONS actual size)
            position = np.zeros(228)

        velocity = para_state.get('velocity')

        # Detect shift if we have previous position
        shift_magnitude = 0.0
        shift_detected = False
        if prev_position is not None:
            shift_magnitude = float(np.linalg.norm(position - prev_position))
            shift_detected = shift_magnitude > 0.5

        return cls(
            position=position,
            velocity=velocity,
            dominant_dimensions=snapshot.dominant_dimensions_by_scale.get(
                MatryoshkaScale.PARAGRAPH, []
            ),
            momentum=snapshot.narrative_momentum,
            complexity=snapshot.complexity_index,
            shift_magnitude=shift_magnitude,
            shift_detected=shift_detected
        )

    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'dominant_dimensions': self.dominant_dimensions,
            'momentum': float(self.momentum),
            'complexity': float(self.complexity),
            'shift_magnitude': float(self.shift_magnitude),
            'shift_detected': self.shift_detected
        }


# =============================================================================
# Activation Control (Budget and Strategy)
# =============================================================================

class ActivationStrategy(Enum):
    """
    Different activation patterns for different needs.

    Elegant mapping: retrieval strategy → activation pattern
    """
    PRECISE = "precise"        # Narrow, high confidence (topic shift detection)
    BALANCED = "balanced"      # Standard retrieval (general queries)
    EXPLORATORY = "exploratory" # Broad, discovery (research mode)
    DEEP = "deep"              # Follow connections deeply (complex reasoning)


@dataclass
class ActivationBudget:
    """
    Budget for activation resources.

    Elegant mapping: LLM context window → activation constraints
    """
    max_memories: int           # Hard limit on returned memories
    semantic_radius: float      # How far to reach in 244D space (0-1)
    spread_iterations: int      # How many hops through graph connections
    activation_threshold: float # Minimum activation level to include (0-1)

    @classmethod
    def for_context_window(cls, tokens: int) -> 'ActivationBudget':
        """
        Create budget based on available context window.

        Maps LLM token constraints to activation parameters.
        Assumes ~200 tokens per memory on average.
        """
        max_memories = max(3, tokens // 200)  # At least 3, scale with tokens

        if tokens <= 2000:  # Small context (Claude Instant)
            return cls(
                max_memories=min(5, max_memories),
                semantic_radius=0.8,  # Narrow (adjusted for 244D space)
                spread_iterations=1,
                activation_threshold=0.5
            )
        elif tokens <= 8000:  # Standard (GPT-3.5, Claude)
            return cls(
                max_memories=min(20, max_memories),
                semantic_radius=1.2,  # Adjusted for 244D normalized distances
                spread_iterations=2,
                activation_threshold=0.3
            )
        elif tokens <= 32000:  # Large (GPT-4 Turbo)
            return cls(
                max_memories=min(50, max_memories),
                semantic_radius=1.6,  # Broader (adjusted for 244D)
                spread_iterations=3,
                activation_threshold=0.2
            )
        else:  # Very large (GPT-4 128k, Claude 200k)
            return cls(
                max_memories=min(100, max_memories),
                semantic_radius=2.0,  # Very broad (adjusted for 244D)
                spread_iterations=4,
                activation_threshold=0.15
            )

    @classmethod
    def for_strategy(cls, strategy: ActivationStrategy) -> 'ActivationBudget':
        """
        Create budget based on retrieval strategy.

        Different strategies = different activation patterns.
        """
        if strategy == ActivationStrategy.PRECISE:
            # Topic shift detection: Need HIGH precision
            return cls(
                max_memories=3,
                semantic_radius=0.8,  # Very narrow (228D-adjusted for topic shift detection)
                spread_iterations=0,    # No spreading
                activation_threshold=0.5 # Moderate confidence (allows partial activations)
            )

        elif strategy == ActivationStrategy.BALANCED:
            # Standard queries: Balance precision/recall
            return cls(
                max_memories=10,
                semantic_radius=1.2,  # Balanced (244D-adjusted)
                spread_iterations=2,
                activation_threshold=0.4
            )

        elif strategy == ActivationStrategy.EXPLORATORY:
            # Research mode: Broad exploration
            return cls(
                max_memories=30,
                semantic_radius=1.8,  # Wide net (244D-adjusted)
                spread_iterations=3,
                activation_threshold=0.2
            )

        elif strategy == ActivationStrategy.DEEP:
            # Complex reasoning: Follow connections
            return cls(
                max_memories=20,
                semantic_radius=1.0,  # Moderate (244D-adjusted)
                spread_iterations=5,  # Many hops
                activation_threshold=0.3
            )

        else:
            # Default to balanced
            return cls.for_strategy(ActivationStrategy.BALANCED)


# =============================================================================
# Awareness Metrics (What policy reads)
# =============================================================================

@dataclass
class AwarenessMetrics:
    """
    Current state of awareness system.

    Simple dict-like metrics for policy consumption.
    """
    # Semantic position (sampled)
    current_position: np.ndarray  # First 64D of 244D (for efficiency)

    # Shift detection
    shift_magnitude: float
    shift_detected: bool

    # Graph topology
    n_memories: int
    n_connections: int
    avg_resonance: float

    # Activation state
    n_active: int
    activation_density: float  # n_active / n_memories

    # Recent trajectory
    trajectory_length: int

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to compact feature vector for policy.

        Policy neural network needs fixed-size input.
        """
        # Sample position (64D) + aggregate metrics (6D) = 70D total
        metrics = np.array([
            self.shift_magnitude,
            float(self.shift_detected),
            self.avg_resonance,
            self.activation_density,
            float(self.n_active) / (self.n_memories + 1),
            float(self.trajectory_length) / 100.0  # Normalize
        ])

        return np.concatenate([self.current_position, metrics])

    def to_dict(self) -> Dict:
        """Simple dict for logging/debugging."""
        return {
            'shift_magnitude': float(self.shift_magnitude),
            'shift_detected': self.shift_detected,
            'n_memories': self.n_memories,
            'n_connections': self.n_connections,
            'avg_resonance': float(self.avg_resonance),
            'n_active': self.n_active,
            'activation_density': float(self.activation_density),
            'trajectory_length': self.trajectory_length
        }


# =============================================================================
# Edge Types (Graph topology)
# =============================================================================

class EdgeType(Enum):
    """
    Types of connections in awareness graph.

    One graph, typed edges - brings them all together.
    """
    TEMPORAL = "temporal"           # Happened before/after (linear time)
    SEMANTIC_RESONANCE = "semantic" # Conceptually related (non-linear)
    CAUSAL = "causal"              # Caused by tool execution
    REFERENCE = "reference"        # Explicit reference/citation


@dataclass
class EdgeMetadata:
    """Metadata for graph edges."""
    edge_type: EdgeType
    strength: float = 1.0  # For SEMANTIC_RESONANCE edges
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'type': self.edge_type.value,
            'strength': float(self.strength),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }