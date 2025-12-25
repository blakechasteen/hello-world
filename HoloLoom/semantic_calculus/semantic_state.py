#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic State for Policy Integration
======================================

Converts high-dimensional semantic snapshots (244D) into compact feature vectors (8D)
that can be consumed by the policy network for semantic-aware decision making.

Architecture:
    MatryoshkaSnapshot (244D) → SemanticState (8D) → Policy (Neural Network)
    Embedding (384D) → SemanticState (8D) → Policy (Neural Network)

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

Integration Points (Phase 8 - December 2025):
- WeavingOrchestrator: Computes state from query embedding via from_embedding()
- NeuralCore: Receives 8D policy features via to_feature_vector()
- SemanticToolSelector: Uses category_scores for tool affinity
- Thompson Sampling: Adjusts priors based on dominant_category

Philosophy:
    "The policy doesn't need all 244 dimensions. It needs 8 numbers that tell a story."
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Import from existing semantic calculus
try:
    from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSnapshot
    from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum
    SEMANTIC_CALCULUS_AVAILABLE = True
except ImportError:
    SEMANTIC_CALCULUS_AVAILABLE = False
    MatryoshkaSnapshot = None  # Type hint fallback

# Import 16 semantic categories from semantic_nudging
try:
    from HoloLoom.policy.semantic_nudging import (
        SEMANTIC_CATEGORIES,
        DIM_TO_CATEGORY,
        aggregate_by_category,
    )
    SEMANTIC_NUDGING_AVAILABLE = True
except ImportError:
    SEMANTIC_NUDGING_AVAILABLE = False
    # Fallback definitions
    SEMANTIC_CATEGORIES = {
        'Core': list(range(0, 16)),
        'Narrative': list(range(16, 32)),
        'Emotional': list(range(32, 48)),
        'Relational': list(range(48, 64)),
        'Archetypal': list(range(64, 80)),
        'Philosophical': list(range(80, 96)),
        'Transformation': list(range(96, 112)),
        'Ethical': list(range(112, 128)),
        'Creative': list(range(128, 144)),
        'Cognitive': list(range(144, 160)),
        'Temporal': list(range(160, 176)),
        'Spatial': list(range(176, 188)),
        'Character': list(range(188, 200)),
        'Plot': list(range(200, 212)),
        'Theme': list(range(212, 224)),
        'Style': list(range(224, 244))
    }
    DIM_TO_CATEGORY = {}
    for cat, indices in SEMANTIC_CATEGORIES.items():
        for idx in indices:
            if idx < 244:
                DIM_TO_CATEGORY[idx] = cat

# Topic shift thresholds
TOPIC_SHIFT_THRESHOLD = 0.3
SIGNIFICANT_SHIFT_THRESHOLD = 0.5


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_category_scores(position: np.ndarray) -> Dict[str, float]:
    """
    Aggregate 244D position into 16 semantic categories.

    Each category corresponds to a range of dimensions from SEMANTIC_CATEGORIES.
    The score for each category is the L2 norm of those dimensions, normalized.

    Args:
        position: 244D semantic position vector

    Returns:
        Dictionary mapping category name to score (0-1)
    """
    if position is None or len(position) == 0:
        return {cat: 0.0 for cat in SEMANTIC_CATEGORIES}

    # Ensure position is at least 244D (pad if needed)
    if len(position) < 244:
        padded = np.zeros(244, dtype=np.float32)
        padded[:len(position)] = position
        position = padded

    category_scores = {}
    max_score = 0.0

    for category, indices in SEMANTIC_CATEGORIES.items():
        # Get values for this category's dimensions
        valid_indices = [i for i in indices if i < len(position)]
        if valid_indices:
            values = position[valid_indices]
            # Use L2 norm as the category score
            score = float(np.linalg.norm(values))
            category_scores[category] = score
            max_score = max(max_score, score)
        else:
            category_scores[category] = 0.0

    # Normalize scores to 0-1 range
    if max_score > 1e-8:
        category_scores = {k: v / max_score for k, v in category_scores.items()}

    return category_scores


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad or truncate array to target length.

    Args:
        arr: Input numpy array
        target_len: Desired length

    Returns:
        Array of exactly target_len
    """
    arr = np.asarray(arr, dtype=np.float32)

    if len(arr) == target_len:
        return arr
    elif len(arr) > target_len:
        return arr[:target_len]
    else:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(arr)] = arr
        return padded


def _dict_to_position_array(projection: Dict[str, float]) -> np.ndarray:
    """
    Convert projection dictionary to 244D position array.

    The SemanticSpectrum.project_vector() may return a dict of dimension names
    to values. This converts it to a proper numpy array.

    Args:
        projection: Dictionary from dimension name to value

    Returns:
        244D numpy array
    """
    position = np.zeros(244, dtype=np.float32)

    # If projection has numeric keys (indices)
    if projection and isinstance(next(iter(projection.keys())), int):
        for idx, val in projection.items():
            if 0 <= idx < 244:
                position[idx] = val
    else:
        # Projection has string keys - fill in order
        for i, (name, val) in enumerate(projection.items()):
            if i < 244:
                position[i] = val

    return position


# =============================================================================
# SemanticState Dataclass
# =============================================================================

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

    # Category aggregation (16 categories from semantic_nudging)
    category_scores: Dict[str, float] = field(default_factory=dict)
    dominant_category: str = "Core"

    # Metadata
    timestamp: float = 0.0
    word_count: int = 0
    query_index: int = 0

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

        # Compute category scores (16 categories)
        category_scores = _compute_category_scores(position)
        dominant_category = max(
            category_scores.items(), key=lambda x: x[1]
        )[0] if category_scores else "Core"

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
            category_scores=category_scores,
            dominant_category=dominant_category,
            timestamp=snapshot.timestamp,
            word_count=snapshot.word_count
        )

    @classmethod
    def from_embedding(
        cls,
        embedding: np.ndarray,
        previous_state: Optional['SemanticState'] = None,
        semantic_spectrum: Optional['SemanticSpectrum'] = None,
        query_index: int = 0,
        timestamp: Optional[float] = None
    ) -> 'SemanticState':
        """
        Create SemanticState directly from query embedding.

        This is the primary factory method for WeavingOrchestrator integration.
        It takes a raw embedding and computes all semantic metrics.

        Args:
            embedding: Query embedding vector (384D or 244D)
            previous_state: Previous state for velocity/momentum computation
            semantic_spectrum: SemanticSpectrum for projection (optional)
            query_index: Index in conversation (for tracking)
            timestamp: Optional timestamp (uses time.time() if None)

        Returns:
            SemanticState with computed velocity, categories, and shift detection

        Example:
            >>> state = SemanticState.from_embedding(
            ...     embedding=query_embedding,
            ...     previous_state=last_state,
            ...     semantic_spectrum=spectrum
            ... )
            >>> features = state.to_feature_vector()  # 8D for policy
        """
        if timestamp is None:
            timestamp = time.time()

        # Project embedding to 244D semantic space if needed
        if semantic_spectrum is not None and len(embedding) != 244:
            try:
                projection = semantic_spectrum.project_vector(embedding)
                if isinstance(projection, dict):
                    position = _dict_to_position_array(projection)
                else:
                    position = np.array(projection, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Semantic projection failed: {e}, using raw embedding")
                position = _pad_or_truncate(embedding, 244)
        else:
            position = _pad_or_truncate(embedding, 244)

        # Compute velocity from previous state
        if previous_state is not None and previous_state.position is not None:
            velocity = position - previous_state.position
        else:
            velocity = np.zeros(244, dtype=np.float32)

        # Detect topic shift
        shift_magnitude = float(np.linalg.norm(velocity))
        topic_shift_detected = shift_magnitude > TOPIC_SHIFT_THRESHOLD

        # Compute category scores (16 categories)
        category_scores = _compute_category_scores(position)
        dominant_category = max(
            category_scores.items(), key=lambda x: x[1]
        )[0] if category_scores else "Core"

        # Compute momentum (requires velocity history - simplified for single embedding)
        if previous_state is not None and previous_state.velocity is not None:
            # Cosine similarity between current and previous velocity
            prev_vel = previous_state.velocity
            vel_norm = np.linalg.norm(velocity)
            prev_norm = np.linalg.norm(prev_vel)
            if vel_norm > 1e-8 and prev_norm > 1e-8:
                momentum = (np.dot(velocity, prev_vel) / (vel_norm * prev_norm) + 1.0) / 2.0
            else:
                momentum = 0.5
        else:
            momentum = 0.5  # Neutral momentum for first query

        # Compute complexity
        complexity = cls._compute_complexity(position)

        # Extract dominant dimensions
        dominant_dims, dim_values = cls._extract_dominant_dimensions(
            position, semantic_spectrum, top_k=5
        )

        return cls(
            position=position,
            velocity=velocity,
            acceleration=None,  # Not computed from single embedding
            momentum=float(np.clip(momentum, 0.0, 1.0)),
            complexity=complexity,
            dominant_dimensions=dominant_dims,
            dimension_values=dim_values,
            topic_shift_detected=topic_shift_detected,
            shift_magnitude=shift_magnitude,
            category_scores=category_scores,
            dominant_category=dominant_category,
            timestamp=timestamp,
            word_count=0,
            query_index=query_index
        )

    @classmethod
    def empty(cls) -> 'SemanticState':
        """Create an empty/zero state (useful for initialization)."""
        return cls(
            position=np.zeros(244, dtype=np.float32),
            velocity=np.zeros(244, dtype=np.float32),
            acceleration=None,
            momentum=0.5,
            complexity=0.0,
            dominant_dimensions=[],
            dimension_values=[],
            topic_shift_detected=False,
            shift_magnitude=0.0,
            category_scores={cat: 0.0 for cat in SEMANTIC_CATEGORIES},
            dominant_category="Core",
            timestamp=0.0,
            word_count=0,
            query_index=0
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


class SemanticAwareBandit:
    """
    Thompson Sampling bandit that adjusts priors based on semantic state.

    Maps semantic categories to tool affinities, allowing the bandit to
    boost/reduce exploration based on semantic understanding of the query.

    Integration (Phase 8 - December 2025):
        - Receives SemanticState from WeavingOrchestrator
        - Adjusts α/β priors before Thompson Sampling
        - Returns adjusted tool distribution

    Category → Tool Affinities:
        - Cognitive (Analysis, Synthesis) → search, analyze
        - Creative (Transformation, Creative) → generate, brainstorm
        - Emotional (Relational, Character) → answer, empathize
        - Technical (Core, Spatial) → calc, execute
        - Narrative (Narrative, Plot, Theme) → explain, tell
    """

    # Standard HoloLoom tools
    DEFAULT_TOOLS = ["answer", "search", "notion_write", "calc"]

    # Map semantic categories to tool boost factors
    # Values > 1.0 boost the tool's α prior, < 1.0 reduce it
    CATEGORY_TOOL_AFFINITIES: Dict[str, Dict[str, float]] = {
        # Core categories (technical queries)
        "Core": {"calc": 1.5, "search": 1.2, "answer": 1.0, "notion_write": 0.8},
        "Cognitive": {"search": 1.5, "calc": 1.3, "answer": 1.0, "notion_write": 0.9},
        "Spatial": {"calc": 1.4, "search": 1.2, "answer": 1.0, "notion_write": 0.8},

        # Creative categories
        "Creative": {"notion_write": 1.5, "answer": 1.2, "search": 1.0, "calc": 0.7},
        "Transformation": {"notion_write": 1.4, "answer": 1.3, "search": 1.0, "calc": 0.8},
        "Style": {"notion_write": 1.5, "answer": 1.3, "search": 0.9, "calc": 0.6},

        # Narrative categories (explanatory queries)
        "Narrative": {"answer": 1.5, "search": 1.2, "notion_write": 1.1, "calc": 0.7},
        "Plot": {"answer": 1.4, "notion_write": 1.3, "search": 1.0, "calc": 0.6},
        "Theme": {"answer": 1.5, "search": 1.3, "notion_write": 1.0, "calc": 0.6},

        # Emotional/relational categories
        "Emotional": {"answer": 1.6, "notion_write": 1.1, "search": 0.9, "calc": 0.5},
        "Relational": {"answer": 1.5, "notion_write": 1.2, "search": 1.0, "calc": 0.6},
        "Character": {"answer": 1.4, "notion_write": 1.3, "search": 0.9, "calc": 0.5},

        # Philosophical/ethical categories
        "Philosophical": {"answer": 1.5, "search": 1.3, "notion_write": 1.0, "calc": 0.6},
        "Ethical": {"answer": 1.5, "search": 1.2, "notion_write": 1.1, "calc": 0.5},
        "Archetypal": {"answer": 1.4, "notion_write": 1.2, "search": 1.1, "calc": 0.6},

        # Temporal categories
        "Temporal": {"search": 1.4, "answer": 1.2, "calc": 1.1, "notion_write": 0.9},
    }

    def __init__(
        self,
        tools: Optional[List[str]] = None,
        base_alpha: float = 1.0,
        base_beta: float = 1.0,
        semantic_weight: float = 0.3
    ):
        """
        Initialize semantic-aware bandit.

        Args:
            tools: List of tool names (defaults to DEFAULT_TOOLS)
            base_alpha: Base α prior for Beta distribution
            base_beta: Base β prior for Beta distribution
            semantic_weight: How much semantic state affects priors (0-1)
        """
        self.tools = tools or self.DEFAULT_TOOLS
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.semantic_weight = semantic_weight

        # Initialize priors for each tool
        self.alphas = {tool: base_alpha for tool in self.tools}
        self.betas = {tool: base_beta for tool in self.tools}

        # Track semantic adjustments for debugging
        self._last_adjustment = {}

    def adjust_priors(
        self,
        semantic_state: SemanticState
    ) -> Dict[str, Tuple[float, float]]:
        """
        Adjust Thompson Sampling priors based on semantic state.

        Uses dominant_category and category_scores to boost/reduce
        tool priors before sampling.

        Args:
            semantic_state: Current semantic state from query

        Returns:
            Dict mapping tool name to (adjusted_α, adjusted_β)
        """
        adjusted = {}
        self._last_adjustment = {}

        # Get dominant category and its affinity map
        dominant_cat = semantic_state.dominant_category
        affinities = self.CATEGORY_TOOL_AFFINITIES.get(
            dominant_cat,
            {tool: 1.0 for tool in self.tools}  # Neutral if unknown
        )

        # Also consider secondary categories from category_scores
        secondary_boost = {}
        for cat, score in semantic_state.category_scores.items():
            if score > 0.5 and cat != dominant_cat:  # Significant secondary categories
                cat_affinities = self.CATEGORY_TOOL_AFFINITIES.get(cat, {})
                for tool, affinity in cat_affinities.items():
                    if tool not in secondary_boost:
                        secondary_boost[tool] = []
                    secondary_boost[tool].append((affinity - 1.0) * score * 0.5)

        # Adjust each tool's prior
        for tool in self.tools:
            base_α = self.alphas[tool]
            base_β = self.betas[tool]

            # Primary category boost
            primary_boost = affinities.get(tool, 1.0)

            # Secondary category contributions
            secondary = 0.0
            if tool in secondary_boost:
                secondary = sum(secondary_boost[tool])

            # Combined boost factor
            boost = primary_boost + secondary

            # Apply semantic weight (blend between base and adjusted)
            # boost > 1.0 increases α (more likely to select)
            # boost < 1.0 increases β (less likely to select)
            if boost >= 1.0:
                α_adj = base_α * (1.0 + (boost - 1.0) * self.semantic_weight)
                β_adj = base_β
            else:
                α_adj = base_α
                β_adj = base_β * (1.0 + (1.0 - boost) * self.semantic_weight)

            adjusted[tool] = (α_adj, β_adj)
            self._last_adjustment[tool] = {
                'primary_boost': primary_boost,
                'secondary': secondary,
                'final_boost': boost,
                'α': α_adj,
                'β': β_adj
            }

        return adjusted

    def sample(
        self,
        semantic_state: Optional[SemanticState] = None,
        temperature: float = 1.0
    ) -> str:
        """
        Sample a tool using Thompson Sampling with semantic adjustments.

        Args:
            semantic_state: Optional semantic state for adjustment
            temperature: Sampling temperature (higher = more random)

        Returns:
            Selected tool name
        """
        # Adjust priors if semantic state provided
        if semantic_state is not None:
            adjusted = self.adjust_priors(semantic_state)
        else:
            adjusted = {tool: (self.alphas[tool], self.betas[tool]) for tool in self.tools}

        # Sample from Beta distribution for each tool
        samples = {}
        for tool, (α, β) in adjusted.items():
            # Apply temperature to the Beta parameters
            α_temp = max(0.1, α / temperature)
            β_temp = max(0.1, β / temperature)
            samples[tool] = np.random.beta(α_temp, β_temp)

        # Select tool with highest sample
        return max(samples, key=samples.get)

    def update(
        self,
        tool: str,
        success: bool,
        confidence: float = 1.0
    ):
        """
        Update bandit priors based on outcome.

        Args:
            tool: Tool that was used
            success: Whether the outcome was successful
            confidence: Confidence in the outcome (0-1)
        """
        if tool not in self.tools:
            return

        if success:
            self.alphas[tool] += confidence
        else:
            self.betas[tool] += (1.0 - confidence)

    def get_expected_rewards(
        self,
        semantic_state: Optional[SemanticState] = None
    ) -> Dict[str, float]:
        """
        Get expected reward (selection probability) for each tool.

        Args:
            semantic_state: Optional semantic state for adjustment

        Returns:
            Dict mapping tool name to expected reward E[X] = α/(α+β)
        """
        if semantic_state is not None:
            adjusted = self.adjust_priors(semantic_state)
        else:
            adjusted = {tool: (self.alphas[tool], self.betas[tool]) for tool in self.tools}

        return {
            tool: α / (α + β)
            for tool, (α, β) in adjusted.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics for debugging."""
        return {
            'tools': self.tools,
            'alphas': dict(self.alphas),
            'betas': dict(self.betas),
            'expected_rewards': {
                tool: self.alphas[tool] / (self.alphas[tool] + self.betas[tool])
                for tool in self.tools
            },
            'last_adjustment': self._last_adjustment,
            'semantic_weight': self.semantic_weight
        }


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
