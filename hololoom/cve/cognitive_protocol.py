"""
Cognitive Rendering Protocol (CRP)
==================================

Separates WHAT to visualize (cognitive processes) from HOW to render (platform-specific).

15 CognitiveVizTypes:
1. ACTIVATION_SPREAD - Spreading activation waves across memory graph
2. PROBABILITY_MANIFOLD - Thompson Sampling Beta distributions
3. SEMANTIC_POSITION - 228D semantic space projection (16 interpretable axes)
4. SEMANTIC_TRAJECTORY - Movement through semantic space over time
5. KNOWLEDGE_GRAPH - 3D navigable entity relationships
6. TOKEN_STREAM - Real-time generation visualization
7. DECISION_COLLAPSE - Probability → discrete choice animation
8. ATTENTION_FLOW - Cross-attention pattern visualization
9. REASONING_CHAIN - Multi-step reasoning path
10. MEMORY_RETRIEVAL - Memory recall visualization
11. CONFIDENCE_FLOW - Confidence trajectory over time
12. TOOL_SELECTION - Tool choice with probability breakdown
13. CONTEXT_EXPANSION - Context window growth/contraction
14. WEAVING_CYCLE - Full 9-stage weaving cycle state
15. REFLECTION_LOOP - Learning/improvement feedback

7 Native Cognitive Primitives (don't exist in game engines):
- ActivationField: Spreading activation waves
- ProbabilityManifold: Beta distributions from Thompson Sampling
- SemanticTopology: 228D semantic space projection
- KnowledgeGraph3D: 3D navigable entity relationships
- TokenStream: Real-time token generation
- DecisionCollapse: Probability → discrete animation
- AttentionFlow: Attention pattern visualization

Created: 2025-11-30
Author: HoloLoom Team
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Protocol, AsyncIterator, Union, Tuple
from enum import Enum, auto
from datetime import datetime
import uuid


# ============================================================================
# Cognitive Visualization Types (15 Types)
# ============================================================================

class CognitiveVizType(str, Enum):
    """
    15 cognitive visualization types.

    These describe WHAT to visualize (cognitive meaning),
    not HOW to render (platform-specific).
    """

    # Memory & Activation (1-4)
    ACTIVATION_SPREAD = "activation_spread"
    MEMORY_RETRIEVAL = "memory_retrieval"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    CONTEXT_EXPANSION = "context_expansion"

    # Semantic Space (5-6)
    SEMANTIC_POSITION = "semantic_position"
    SEMANTIC_TRAJECTORY = "semantic_trajectory"

    # Decision Making (7-10)
    PROBABILITY_MANIFOLD = "probability_manifold"
    DECISION_COLLAPSE = "decision_collapse"
    TOOL_SELECTION = "tool_selection"
    CONFIDENCE_FLOW = "confidence_flow"

    # Reasoning (11-13)
    REASONING_CHAIN = "reasoning_chain"
    ATTENTION_FLOW = "attention_flow"
    TOKEN_STREAM = "token_stream"

    # System-Level (14-15)
    WEAVING_CYCLE = "weaving_cycle"
    REFLECTION_LOOP = "reflection_loop"


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class CognitiveEvent:
    """
    Single cognitive visualization event.

    The atomic unit of the CRP - describes a single cognitive
    state or change that should be rendered.
    """
    id: str
    viz_type: CognitiveVizType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Lifecycle hints
    duration_ms: Optional[float] = None  # How long to display
    priority: int = 5  # 1 (highest) to 10 (lowest)
    replaces: Optional[str] = None  # ID of event this replaces

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class CognitiveFrame:
    """
    Complete cognitive state at a point in time.

    Contains all events that should be rendered together.
    """
    timestamp: datetime
    events: List[CognitiveEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Weaving context
    query_id: Optional[str] = None
    weaving_stage: Optional[str] = None  # "retrieval", "decision", "synthesis", etc.
    confidence: float = 0.0

    @property
    def stage(self) -> Optional[str]:
        """Alias for weaving_stage for convenience."""
        return self.weaving_stage

    @property
    def title(self) -> Optional[str]:
        """Get title from metadata."""
        return self.metadata.get("title")

    def add_event(self, event: CognitiveEvent) -> "CognitiveFrame":
        """Add event (builder pattern)."""
        self.events.append(event)
        return self

    def get_by_type(self, viz_type: CognitiveVizType) -> List[CognitiveEvent]:
        """Get events of a specific type."""
        return [e for e in self.events if e.viz_type == viz_type]


class CognitiveStream:
    """
    Async stream of cognitive frames.

    Enables real-time visualization of cognitive processes.
    """

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.started_at = datetime.now()
        self._frames: List[CognitiveFrame] = []

    async def emit(self, frame: CognitiveFrame) -> None:
        """Emit a frame (for producers)."""
        self._frames.append(frame)

    async def __aiter__(self) -> AsyncIterator[CognitiveFrame]:
        """Iterate over frames (for consumers)."""
        for frame in self._frames:
            yield frame


# ============================================================================
# Cognitive Primitives (7 Native Types)
# ============================================================================

@dataclass
class ActivationField:
    """
    Spreading activation wave across memory graph.

    Data Source: AwarenessGraph.activation_field

    Visualized as: Glowing waves propagating from source nodes,
    with intensity representing activation level.
    """
    source_nodes: List[str]  # Origin points of activation
    activation_levels: Dict[str, float]  # node_id → activation (0.0-1.0)
    wave_front: List[str]  # Currently propagating nodes
    propagation_step: int = 0  # Time step in propagation
    decay_rate: float = 0.15  # Activation decay per step

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"activation_{self.propagation_step}",
            viz_type=CognitiveVizType.ACTIVATION_SPREAD,
            timestamp=datetime.now(),
            data={
                "source_nodes": self.source_nodes,
                "activation_levels": self.activation_levels,
                "wave_front": self.wave_front,
                "propagation_step": self.propagation_step,
                "decay_rate": self.decay_rate,
            }
        )


@dataclass
class ProbabilityManifold:
    """
    Thompson Sampling Beta distributions for tool selection.

    Data Source: TSBandit.success/fail counts

    Visualized as: Probability density clouds, with brighter regions
    indicating higher probability of selection.
    """
    tool_names: List[str]
    alpha_values: List[float]  # Success counts + 1 (Beta prior)
    beta_values: List[float]  # Failure counts + 1 (Beta prior)
    sampled_values: Optional[List[float]] = None  # Current samples
    selected_tool: Optional[str] = None  # Tool chosen this step

    @property
    def expected_values(self) -> List[float]:
        """Expected value for each tool: alpha / (alpha + beta)."""
        return [a / (a + b) for a, b in zip(self.alpha_values, self.beta_values)]

    @property
    def variances(self) -> List[float]:
        """Variance for each tool."""
        return [
            (a * b) / ((a + b) ** 2 * (a + b + 1))
            for a, b in zip(self.alpha_values, self.beta_values)
        ]

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"prob_manifold_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.PROBABILITY_MANIFOLD,
            timestamp=datetime.now(),
            data={
                "tools": self.tool_names,
                "alphas": self.alpha_values,
                "betas": self.beta_values,
                "expected": self.expected_values,
                "variances": self.variances,
                "sampled": self.sampled_values,
                "selected": self.selected_tool,
            }
        )


@dataclass
class SemanticPosition:
    """
    Position in 228D semantic space.

    Data Source: MatryoshkaSemanticCalculus

    The first 16 dimensions are human-interpretable axes:
    sentiment, formality, technicality, certainty, urgency,
    abstraction, specificity, temporality, objectivity, complexity,
    scope, directness, emotionality, actionability, novelty, controversy

    Visualized as: Point in 3D projected space (t-SNE/UMAP/PCA),
    with color encoding dominant semantic dimension.
    """
    position: List[float]  # 228D vector (or projected)
    dominant_axes: List[Tuple[str, float]]  # Top semantic axes with values
    query_text: str = ""

    # 16 Interpretable Axes
    INTERPRETABLE_AXES = [
        "sentiment", "formality", "technicality", "certainty",
        "urgency", "abstraction", "specificity", "temporality",
        "objectivity", "complexity", "scope", "directness",
        "emotionality", "actionability", "novelty", "controversy"
    ]

    def get_interpretable(self) -> Dict[str, float]:
        """Get first 16 dimensions as interpretable axes."""
        if len(self.position) >= 16:
            return dict(zip(self.INTERPRETABLE_AXES, self.position[:16]))
        return {}

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"semantic_pos_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.SEMANTIC_POSITION,
            timestamp=datetime.now(),
            data={
                "position": self.position[:16] if len(self.position) >= 16 else self.position,
                "full_position": self.position,
                "dominant_axes": self.dominant_axes,
                "interpretable": self.get_interpretable(),
                "query_text": self.query_text,
            }
        )


@dataclass
class SemanticTrajectory:
    """
    Movement through semantic space over time.

    Data Source: AwarenessGraph.trajectory

    Visualized as: Path/ribbon in 3D space showing semantic journey,
    with color encoding velocity/shift magnitude.
    """
    positions: List[SemanticPosition]
    velocities: List[float] = field(default_factory=list)  # Speed at each step
    shift_points: List[int] = field(default_factory=list)  # Major shift indices

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"semantic_traj_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.SEMANTIC_TRAJECTORY,
            timestamp=datetime.now(),
            data={
                "positions": [p.position for p in self.positions],
                "velocities": self.velocities,
                "shift_points": self.shift_points,
                "length": len(self.positions),
            }
        )


@dataclass
class KnowledgeGraph3D:
    """
    3D navigable entity relationships.

    Data Source: HoloLoom.memory.graph.KG (NetworkX MultiDiGraph)

    Visualized as: Force-directed 3D graph with typed edges,
    node sizing by importance, edge coloring by relationship type.
    """
    nodes: List[Dict[str, Any]]  # [{id, label, type, importance, position, ...}]
    edges: List[Dict[str, Any]]  # [{src, dst, type, weight, ...}]
    layout: str = "force_directed"  # force_directed, spherical, hierarchical, radial
    highlighted_path: Optional[List[str]] = None  # Path to highlight

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"kg3d_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.KNOWLEDGE_GRAPH,
            timestamp=datetime.now(),
            data={
                "nodes": self.nodes,
                "edges": self.edges,
                "layout": self.layout,
                "highlighted_path": self.highlighted_path,
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            }
        )


@dataclass
class TokenStream:
    """
    Real-time token generation visualization.

    Data Source: GenerationToken from interleaved_generation.py

    Visualized as: Tokens appearing in 3D space, flowing from
    generation point with semantic coloring.
    """
    tokens: List[str]
    token_indices: List[int]
    cumulative_text: str = ""
    is_final: bool = False
    generation_speed: float = 0.0  # tokens/sec

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"tokens_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.TOKEN_STREAM,
            timestamp=datetime.now(),
            data={
                "tokens": self.tokens,
                "indices": self.token_indices,
                "cumulative": self.cumulative_text,
                "is_final": self.is_final,
                "speed": self.generation_speed,
            }
        )


@dataclass
class DecisionCollapse:
    """
    Probability → discrete choice animation.

    Data Source: ConvergenceEngine collapse

    Visualized as: Probability distribution "collapsing" to selected option,
    like quantum measurement visualization.
    """
    options: List[str]
    probabilities: List[float]
    selected: str
    collapse_strategy: str = "bayesian_blend"  # argmax, epsilon_greedy, bayesian_blend, pure_thompson
    collapse_confidence: float = 0.0

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"collapse_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.DECISION_COLLAPSE,
            timestamp=datetime.now(),
            data={
                "options": self.options,
                "probabilities": self.probabilities,
                "selected": self.selected,
                "strategy": self.collapse_strategy,
                "confidence": self.collapse_confidence,
            }
        )


@dataclass
class AttentionFlow:
    """
    Cross-attention pattern visualization.

    Data Source: Policy engine attention matrices

    Visualized as: Flowing connections between query tokens and
    memory elements, with width encoding attention weight.
    """
    query_tokens: List[str]
    memory_keys: List[str]
    attention_weights: List[List[float]]  # [query_len, key_len]
    head_index: int = 0  # For multi-head attention

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"attention_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.ATTENTION_FLOW,
            timestamp=datetime.now(),
            data={
                "query_tokens": self.query_tokens,
                "memory_keys": self.memory_keys,
                "weights": self.attention_weights,
                "head_index": self.head_index,
            }
        )


@dataclass
class ReasoningChain:
    """
    Multi-step reasoning path.

    Data Source: Agentic reasoning steps

    Visualized as: Connected steps showing reasoning flow,
    with each step showing query → result → confidence.
    """
    steps: List[Dict[str, Any]]  # [{query, result, confidence, reasoning_mode, ...}]
    current_step: int = 0
    total_confidence: float = 0.0
    reasoning_mode: str = "direct"  # direct, verify, research, plan_execute

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"reasoning_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.REASONING_CHAIN,
            timestamp=datetime.now(),
            data={
                "steps": self.steps,
                "current": self.current_step,
                "confidence": self.total_confidence,
                "mode": self.reasoning_mode,
            }
        )


@dataclass
class MemoryRetrieval:
    """
    Memory recall visualization.

    Data Source: Memory recall results

    Visualized as: Memories "lighting up" in the knowledge graph,
    with relevance encoding brightness.
    """
    query: str
    retrieved_ids: List[str]
    relevance_scores: List[float]
    retrieval_method: str = "hybrid"  # vector, bm25, hybrid, graph
    total_candidates: int = 0

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"retrieval_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.MEMORY_RETRIEVAL,
            timestamp=datetime.now(),
            data={
                "query": self.query,
                "retrieved": self.retrieved_ids,
                "scores": self.relevance_scores,
                "method": self.retrieval_method,
                "total_candidates": self.total_candidates,
            }
        )


@dataclass
class ConfidenceFlow:
    """
    Confidence trajectory over time.

    Data Source: Weaving cycle confidence values

    Visualized as: Line chart showing confidence evolution,
    with anomaly detection and cache markers.
    """
    confidences: List[float]
    timestamps: List[datetime]
    cache_hits: List[bool] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"confidence_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.CONFIDENCE_FLOW,
            timestamp=datetime.now(),
            data={
                "confidences": self.confidences,
                "timestamps": [t.isoformat() for t in self.timestamps],
                "cache_hits": self.cache_hits,
                "anomalies": self.anomalies,
            }
        )


@dataclass
class ToolSelection:
    """
    Tool choice with probability breakdown.

    Data Source: Policy engine tool selection

    Visualized as: Horizontal bar chart of tool probabilities,
    with selected tool highlighted.
    """
    tools: List[str]
    probabilities: List[float]
    selected: str
    bandit_stats: Optional[Dict[str, Any]] = None  # Thompson Sampling stats

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"tool_sel_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.TOOL_SELECTION,
            timestamp=datetime.now(),
            data={
                "tools": self.tools,
                "probabilities": self.probabilities,
                "selected": self.selected,
                "bandit_stats": self.bandit_stats,
            }
        )


@dataclass
class ContextExpansion:
    """
    Context window growth/contraction.

    Data Source: Adaptive graph expansion

    Visualized as: Expanding circle/sphere showing context boundary,
    with nodes entering/leaving visible.
    """
    seed_nodes: List[str]
    expanded_nodes: List[str]
    token_budget: int
    tokens_used: int
    expansion_strategy: str = "adaptive"  # fixed_depth, adaptive, streaming

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"context_exp_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.CONTEXT_EXPANSION,
            timestamp=datetime.now(),
            data={
                "seeds": self.seed_nodes,
                "expanded": self.expanded_nodes,
                "budget": self.token_budget,
                "used": self.tokens_used,
                "strategy": self.expansion_strategy,
            }
        )


@dataclass
class WeavingCycle:
    """
    Full 9-stage weaving cycle state.

    Data Source: WeavingOrchestrator stages

    Visualized as: 9-node circular diagram showing current stage,
    with completed stages filled and upcoming stages outlined.

    Stages:
    1. Loom Command - Pattern Card selection
    2. Chrono Trigger - Temporal window
    3. Yarn Graph - Thread selection
    4. Resonance Shed - Feature extraction
    5. Warp Space - Continuous manifold
    6. Convergence Engine - Decision collapse
    7. Tool Execution - Action
    8. Spacetime Fabric - Output weaving
    9. Reflection Buffer - Learning
    """
    current_stage: int  # 1-9
    stage_names: List[str] = field(default_factory=lambda: [
        "Loom Command", "Chrono Trigger", "Yarn Graph",
        "Resonance Shed", "Warp Space", "Convergence Engine",
        "Tool Execution", "Spacetime Fabric", "Reflection Buffer"
    ])
    stage_statuses: List[str] = field(default_factory=lambda: ["pending"] * 9)  # pending, active, complete
    stage_durations: List[float] = field(default_factory=lambda: [0.0] * 9)  # ms per stage
    pattern_card: str = "FAST"  # BARE, FAST, FUSED

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"weaving_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.WEAVING_CYCLE,
            timestamp=datetime.now(),
            data={
                "current_stage": self.current_stage,
                "stage_names": self.stage_names,
                "statuses": self.stage_statuses,
                "durations": self.stage_durations,
                "pattern_card": self.pattern_card,
            }
        )


@dataclass
class ReflectionLoop:
    """
    Learning/improvement feedback.

    Data Source: ReflectionBuffer, recursive learning

    Visualized as: Feedback loop showing outcome → learning → improvement.
    """
    query: str
    outcome: str  # success, failure, partial
    confidence: float
    feedback: Dict[str, Any]
    pattern_learned: Optional[str] = None
    policy_updated: bool = False

    def to_event(self) -> CognitiveEvent:
        """Convert to CRP event."""
        return CognitiveEvent(
            id=f"reflection_{datetime.now().timestamp()}",
            viz_type=CognitiveVizType.REFLECTION_LOOP,
            timestamp=datetime.now(),
            data={
                "query": self.query,
                "outcome": self.outcome,
                "confidence": self.confidence,
                "feedback": self.feedback,
                "pattern_learned": self.pattern_learned,
                "policy_updated": self.policy_updated,
            }
        )


# ============================================================================
# Renderer Protocol
# ============================================================================

class CognitiveRenderer(Protocol):
    """
    Protocol for cognitive visualization renderers.

    Implementers must handle all 15 CognitiveVizTypes.
    """

    def render_event(self, event: CognitiveEvent) -> str:
        """Render single event to output format (HTML, WebGL commands, etc.)."""
        ...

    def render_frame(self, frame: CognitiveFrame) -> str:
        """Render complete frame."""
        ...

    async def stream_render(self, stream: CognitiveStream) -> AsyncIterator[str]:
        """Stream render frames as they arrive."""
        ...

    def supports_viz_type(self, viz_type: CognitiveVizType) -> bool:
        """Check if renderer supports a visualization type."""
        ...

    def get_supported_types(self) -> List[CognitiveVizType]:
        """Get all supported visualization types."""
        ...


# ============================================================================
# Factory Functions
# ============================================================================

def create_event(
    viz_type: CognitiveVizType,
    data: Dict[str, Any],
    **kwargs
) -> CognitiveEvent:
    """Create a cognitive event."""
    return CognitiveEvent(
        id=kwargs.get("id", str(uuid.uuid4())[:8]),
        viz_type=viz_type,
        timestamp=kwargs.get("timestamp", datetime.now()),
        data=data,
        metadata=kwargs.get("metadata", {}),
        duration_ms=kwargs.get("duration_ms"),
        priority=kwargs.get("priority", 5),
        replaces=kwargs.get("replaces"),
    )


def create_frame(
    events: List[CognitiveEvent],
    **kwargs
) -> CognitiveFrame:
    """Create a cognitive frame.

    Args:
        events: List of cognitive events in this frame
        **kwargs: Optional parameters:
            - timestamp: Frame timestamp (default: now)
            - metadata: Additional metadata dict
            - query_id: Query identifier
            - stage or weaving_stage: Current weaving stage (e.g., "decision")
            - confidence: Confidence value (0.0-1.0)
            - title: Frame title (stored in metadata)
    """
    # Accept either 'stage' or 'weaving_stage' for convenience
    weaving_stage = kwargs.get("weaving_stage") or kwargs.get("stage")

    # Store title in metadata if provided
    metadata = kwargs.get("metadata", {})
    if "title" in kwargs:
        metadata["title"] = kwargs["title"]

    return CognitiveFrame(
        timestamp=kwargs.get("timestamp", datetime.now()),
        events=events,
        metadata=metadata,
        query_id=kwargs.get("query_id"),
        weaving_stage=weaving_stage,
        confidence=kwargs.get("confidence", 0.0),
    )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("=== Cognitive Rendering Protocol Demo ===\n")

    # Create sample cognitive events
    activation = ActivationField(
        source_nodes=["thompson_sampling"],
        activation_levels={
            "thompson_sampling": 1.0,
            "exploration": 0.85,
            "exploitation": 0.85,
            "bayesian": 0.7,
            "bandit": 0.6,
        },
        wave_front=["exploration", "exploitation"],
        propagation_step=1,
    )
    print(f"1. ActivationField event: {activation.to_event().viz_type.value}")

    manifold = ProbabilityManifold(
        tool_names=["answer", "research", "explore", "clarify"],
        alpha_values=[10.0, 5.0, 3.0, 2.0],
        beta_values=[2.0, 3.0, 5.0, 8.0],
        selected_tool="answer",
    )
    print(f"2. ProbabilityManifold event: expected values = {manifold.expected_values}")

    position = SemanticPosition(
        position=list(range(228)),  # Mock 228D vector
        dominant_axes=[("technicality", 0.9), ("certainty", 0.8)],
        query_text="What is Thompson Sampling?",
    )
    print(f"3. SemanticPosition event: interpretable axes = {list(position.get_interpretable().keys())[:4]}...")

    # Create frame with multiple events
    frame = create_frame(
        events=[
            activation.to_event(),
            manifold.to_event(),
            position.to_event(),
        ],
        query_id="demo_query",
        weaving_stage="decision",
        confidence=0.85,
    )
    print(f"\n4. CognitiveFrame: {len(frame.events)} events at {frame.weaving_stage} stage")

    # Show all viz types
    print(f"\n5. All {len(CognitiveVizType)} CognitiveVizTypes:")
    for i, vt in enumerate(CognitiveVizType, 1):
        print(f"   {i:2d}. {vt.value}")

    print("\n✓ CRP protocol ready for renderer implementations!")
