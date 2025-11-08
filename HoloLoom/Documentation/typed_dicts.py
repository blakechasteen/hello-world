"""
TypedDict Definitions - Improves type safety for complex return types.

This module defines TypedDict classes for complex data structures,
replacing loose Dict[str, Any] types with concrete, type-checked structures.

Benefits:
- IDE autocomplete for dictionary keys
- Type checking catches errors at development time
- Self-documenting code (keys and types explicit)
- Better refactoring support

Usage:
    from HoloLoom.documentation.typed_dicts import DotPlasmaDict, ActionPlanDict

    # Type-checked dictionary
    plasma: DotPlasmaDict = {
        "motifs": ["pattern1", "pattern2"],
        "embeddings": {"96": [0.1, 0.2]},
        "spectral": {"eigenvalues": [1.0, 0.5]},
        "confidence": 0.85
    }
"""

from typing import TypedDict, List, Dict, Optional, Any
# Removed numpy.typing import to fix import timeout issues
# Using List[float] instead of npt.NDArray for simplicity


# ============================================================================
# Core Feature Types
# ============================================================================

class DotPlasmaDict(TypedDict, total=False):
    """
    DotPlasma - Flowing feature representation.

    The "feature fluid" that flows between extraction and decision stages.
    Contains motifs (symbolic), embeddings (continuous), and spectral (topological).
    """
    motifs: List[str]  # Detected motifs/patterns
    embeddings: Dict[str, List[float]]  # Multi-scale embeddings {scale: vector}
    spectral: Dict[str, Any]  # Spectral features (eigenvalues, topics, etc.)
    confidence: float  # Overall feature extraction confidence
    extraction_time_ms: float  # Time taken for feature extraction
    scales: List[int]  # Scales used for embedding


class ActionPlanDict(TypedDict, total=False):
    """
    ActionPlan - Decision output from policy engine.

    Specifies which tool to execute and how confident the decision is.
    """
    tool: str  # Tool name to execute
    confidence: float  # Decision confidence [0, 1]
    tool_probs: Dict[str, float]  # Probability distribution over all tools
    adapter: str  # Which LoRA adapter was used (bare/fast/fused)
    metadata: Dict[str, Any]  # Additional decision metadata
    reasoning: str  # Explanation of decision (optional)


class SpacetimeDict(TypedDict, total=False):
    """
    Spacetime - Complete woven output with provenance.

    4-dimensional output: 3D semantic space + 1D temporal trace.
    Contains full computational provenance for debugging and learning.
    """
    response: str  # Final response text
    tool_used: str  # Which tool was executed
    confidence: float  # Overall confidence
    sources: int  # Number of source memories used
    trace: 'WeavingTraceDict'  # Complete execution trace
    metadata: Dict[str, Any]  # Additional metadata


class WeavingTraceDict(TypedDict, total=False):
    """
    WeavingTrace - Complete execution provenance.

    Tracks every step of the weaving cycle for debugging and reflection.
    """
    stage_durations: Dict[str, float]  # Duration of each stage (ms)
    pattern_card: str  # Which pattern was used (BARE/FAST/FUSED)
    temporal_window: 'TemporalWindowDict'  # Temporal constraints
    features_extracted: DotPlasmaDict  # Extracted features
    decision_made: ActionPlanDict  # Policy decision
    memories_activated: List[str]  # Which memories were activated
    bottlenecks: List[str]  # Detected performance bottlenecks


# ============================================================================
# Weaving Cycle Types
# ============================================================================

class TemporalWindowDict(TypedDict, total=False):
    """
    TemporalWindow - Time-based constraints for weaving.

    Controls when threads activate and execution timing.
    """
    start_time: float  # Window start (epoch seconds)
    end_time: float  # Window end (epoch seconds)
    duration_budget_ms: float  # Maximum duration allowed
    rhythm: str  # Temporal rhythm (fast/medium/slow)
    halt_conditions: List[str]  # When to stop early


class PatternCardDict(TypedDict, total=False):
    """
    PatternCard - Execution template selection.

    Determines which warp threads to lift and how densely to weave.
    """
    mode: str  # BARE, FAST, or FUSED
    scales: List[int]  # Which embedding scales to use
    features: List[str]  # Which features to extract
    timeout_ms: float  # Maximum execution time
    memory_limit: int  # Maximum memories to retrieve


class ResonanceDict(TypedDict, total=False):
    """
    Resonance - Feature interference zone.

    Where different feature extraction threads combine.
    """
    motif_features: List[str]  # Symbolic motifs detected
    embedding_features: Dict[str, List[float]]  # Continuous embeddings
    spectral_features: Dict[str, float]  # Topological features
    fusion_method: str  # How features were combined
    interference_pattern: str  # Type of interference (constructive/destructive)


# ============================================================================
# Memory Types
# ============================================================================

class MemoryShardDict(TypedDict, total=False):
    """
    MemoryShard - Unit of retrievable memory.

    Atomic unit we retrieve and compose into context.
    """
    id: str  # Unique shard identifier
    text: str  # Shard content
    episode: str  # Which episode/session this belongs to
    entities: List[str]  # Extracted entities
    motifs: List[str]  # Detected motifs
    scales: Dict[str, List[float]]  # Pre-computed embeddings
    metadata: Dict[str, Any]  # Additional metadata


class ContextDict(TypedDict, total=False):
    """
    Context - Retrieved memories for a query.

    Contains all relevant memories and their metadata.
    """
    hits: List[tuple]  # (shard, score) tuples
    kg_sub: Any  # Knowledge graph subgraph (NetworkX MultiDiGraph)
    shard_texts: List[str]  # Text content of shards
    relevance: float  # Overall relevance score
    cache_hit: bool  # Whether this was a cache hit
    retrieval_time_ms: float  # Time taken to retrieve


# ============================================================================
# Thompson Sampling Types
# ============================================================================

class ThompsonStatsDict(TypedDict, total=False):
    """
    Thompson Sampling bandit statistics.

    Tracks successes/failures for each tool to guide exploration.
    """
    tool: str  # Tool name
    alpha: float  # Success parameter (successes + 1)
    beta: float  # Failure parameter (failures + 1)
    total_pulls: int  # Total times this tool was selected
    successes: int  # Number of successful outcomes
    failures: int  # Number of failed outcomes
    expected_reward: float  # E[reward] = alpha / (alpha + beta)


# ============================================================================
# Reflection Types
# ============================================================================

class ReflectionDict(TypedDict, total=False):
    """
    Reflection - Learning signal from outcomes.

    Captures feedback for system improvement.
    """
    query: str  # Original query
    outcome: SpacetimeDict  # Execution outcome
    feedback: Dict[str, Any]  # User feedback
    confidence_delta: float  # Change in confidence
    lessons_learned: List[str]  # Extracted lessons
    timestamp: float  # When reflection occurred


class LearningSignalDict(TypedDict, total=False):
    """
    Learning signal for PPO or reflection learning.

    Contains everything needed to update the policy.
    """
    state: List[float]  # Input state (features)
    action: int  # Action taken (tool index)
    reward: float  # Reward received
    next_state: List[float]  # Resulting state
    done: bool  # Whether episode ended
    metadata: Dict[str, Any]  # Additional context


# ============================================================================
# Stage Timing Types
# ============================================================================

class StageTimingDict(TypedDict, total=False):
    """
    Timing information for a pipeline stage.

    Tracks duration and identifies bottlenecks.
    """
    stage_name: str  # Name of stage
    duration_ms: float  # Duration in milliseconds
    start_time: float  # Start time (epoch)
    end_time: float  # End time (epoch)
    is_bottleneck: bool  # Whether this stage is a bottleneck
    percentage_of_total: float  # Percentage of total pipeline time


# ============================================================================
# Configuration Types
# ============================================================================

class ConfigDict(TypedDict, total=False):
    """
    System configuration dictionary.

    Comprehensive config for all HoloLoom components.
    """
    mode: str  # BARE, FAST, or FUSED
    scales: List[int]  # Embedding scales to use
    memory_backend: str  # INMEMORY, HYBRID, or HYPERSPACE
    enable_semantic_calculus: bool  # Whether to use 244D projection
    enable_alignment: bool  # Whether to use alignment framework
    enable_linguistic_gate: bool  # Whether to use Phase 5 linguistic features
    use_compositional_cache: bool  # Whether to use compositional caching
    timeout_ms: float  # Maximum execution time
    memory_limit: int  # Maximum memories to retrieve


# ============================================================================
# Visualization Types
# ============================================================================

class TuftePlotDict(TypedDict, total=False):
    """
    Tufte-style visualization configuration.

    For creating high data-density visualizations.
    """
    title: str  # Plot title
    subtitle: str  # Plot subtitle
    data_points: List[Any]  # Data to visualize
    sparklines: bool  # Whether to include sparklines
    show_anomalies: bool  # Whether to highlight anomalies
    layout: str  # grid or horizontal
    max_columns: int  # Maximum columns (for grid layout)


# ============================================================================
# Awareness Graph Types
# ============================================================================

class AwarenessMetricsDict(TypedDict, total=False):
    """
    Awareness graph metrics.

    Tracks activation patterns in memory graph.
    """
    active_nodes: int  # Number of activated nodes
    total_nodes: int  # Total nodes in graph
    activation_density: float  # active / total
    avg_activation: float  # Average activation level
    coherence_score: float  # How coherent the activation pattern is
    temporal_focus: str  # Which time period is most active


# Example usage
if __name__ == "__main__":
    # Type-checked dictionaries
    plasma: DotPlasmaDict = {
        "motifs": ["pattern_a", "pattern_b"],
        "embeddings": {"96": [0.1, 0.2, 0.3]},
        "spectral": {"eigenvalues": [1.0, 0.8]},
        "confidence": 0.92
    }

    action_plan: ActionPlanDict = {
        "tool": "answer",
        "confidence": 0.85,
        "tool_probs": {"answer": 0.85, "search": 0.15},
        "adapter": "fused"
    }

    # IDE will autocomplete keys and type-check values
    print(f"Motifs detected: {plasma['motifs']}")
    print(f"Tool selected: {action_plan['tool']}")
