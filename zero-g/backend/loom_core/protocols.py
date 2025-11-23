"""
Loom Core Protocol Definitions

This module defines the core interfaces for Zero-G's Loom-based reasoning engine.
All Loom components implement these protocols for modularity and testability.

Metaphor Guide:
- WarpSpace: Semantic retrieval layer (vector search, embeddings)
- YarnGraph: Structured knowledge graph (entities, relationships)
- ResonanceShed: Multimodal fusion bus (audio, visual, text, sensors)
- ConvergenceEngine: Decision planner (MCTS, Thompson Sampling)
- Rift: Tool/API invocation layer (external integrations)
- SpacetimeFabric: Provenance and execution trace logging
- ReflectionBuffer: Learning and adaptation memory (RL, feedback)
- ThreadSpinner: Hot/cold memory paging (MemGPT-like)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# ==================== Core Data Types ====================

class MemoryType(Enum):
    """Memory classification for ThreadSpinner paging"""
    HOT = "hot"      # Actively used, keep in fast memory
    WARM = "warm"    # Recently used, may page soon
    COLD = "cold"    # Archive, page to disk/remote


@dataclass
class Thread:
    """A single thread of memory/knowledge in the Loom"""
    id: str
    content: Any
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    memory_type: MemoryType = MemoryType.WARM
    created_at: datetime = None
    last_accessed: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class YarnNode:
    """A node in the YarnGraph knowledge structure"""
    id: str
    entity_type: str
    properties: Dict[str, Any]
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class YarnEdge:
    """A relationship edge in the YarnGraph"""
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class SpacetimeEvent:
    """A single event in the provenance trace"""
    timestamp: datetime
    event_type: str
    component: str
    data: Dict[str, Any]
    parent_event_id: Optional[str] = None
    event_id: Optional[str] = None

    def __post_init__(self):
        if self.event_id is None:
            self.event_id = f"{self.component}_{self.timestamp.timestamp()}"


@dataclass
class RiftAction:
    """An action to be executed through the Rift (tool invocation)"""
    tool_name: str
    parameters: Dict[str, Any]
    timeout_seconds: float = 30.0
    retry_policy: Optional[Dict[str, Any]] = None


@dataclass
class DecisionContext:
    """Context for ConvergenceEngine decision making"""
    query: str
    threads: List[Thread]
    yarn_nodes: List[YarnNode]
    available_tools: List[str]
    constraints: Dict[str, Any] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}


# ==================== Protocol Definitions ====================

class WarpSpaceProtocol(Protocol):
    """
    Semantic retrieval layer - vector search and embedding operations

    WarpSpace is responsible for:
    - Embedding text/multimodal content into vector space
    - Semantic similarity search
    - Hybrid search (keyword + semantic)
    - Multi-scale embeddings (Matryoshka-style)
    """

    async def embed(self, content: str, modality: str = "text") -> List[float]:
        """Embed content into vector space"""
        ...

    async def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Thread]:
        """Semantic search for relevant threads"""
        ...

    async def index_thread(self, thread: Thread) -> None:
        """Index a thread for future retrieval"""
        ...


class YarnGraphProtocol(Protocol):
    """
    Structured knowledge graph - entities and relationships

    YarnGraph provides:
    - Entity storage and retrieval
    - Relationship traversal
    - Subgraph extraction
    - Path finding and graph queries
    """

    async def add_node(self, node: YarnNode) -> None:
        """Add a node to the graph"""
        ...

    async def add_edge(self, edge: YarnEdge) -> None:
        """Add an edge to the graph"""
        ...

    async def get_node(self, node_id: str) -> Optional[YarnNode]:
        """Retrieve a node by ID"""
        ...

    async def find_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[YarnNode]:
        """Find neighboring nodes"""
        ...

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 5
    ) -> Optional[List[YarnEdge]]:
        """Find shortest path between nodes"""
        ...


class ResonanceShedProtocol(Protocol):
    """
    Multimodal fusion bus - combines audio, visual, text, sensor data

    ResonanceShed handles:
    - Multimodal input ingestion
    - Cross-modal alignment
    - Fusion strategies (early, late, hybrid)
    - Temporal synchronization
    """

    async def fuse_inputs(
        self,
        inputs: Dict[str, Any],  # {modality: data}
        fusion_strategy: str = "hybrid"
    ) -> Thread:
        """Fuse multimodal inputs into a unified thread"""
        ...

    async def align_temporal(
        self,
        streams: Dict[str, List[Any]],  # {modality: time series}
        alignment_method: str = "dtw"  # Dynamic Time Warping
    ) -> Dict[str, List[Any]]:
        """Align temporal streams across modalities"""
        ...


class ConvergenceEngineProtocol(Protocol):
    """
    Decision planner and action selector

    ConvergenceEngine provides:
    - Monte Carlo Tree Search (MCTS)
    - Thompson Sampling for exploration
    - Multi-criteria decision making
    - Plan generation and execution
    """

    async def decide(
        self,
        context: DecisionContext,
        strategy: str = "thompson_sampling"
    ) -> RiftAction:
        """Make a decision given context"""
        ...

    async def plan(
        self,
        goal: str,
        context: DecisionContext,
        max_steps: int = 10
    ) -> List[RiftAction]:
        """Generate a plan to achieve a goal"""
        ...


class RiftProtocol(Protocol):
    """
    Tool and API invocation layer

    Rift handles:
    - External API calls
    - Tool execution (local or remote)
    - Safety checks and sandboxing
    - Retry and error handling
    """

    async def invoke(self, action: RiftAction) -> Dict[str, Any]:
        """Execute an action through the Rift"""
        ...

    async def register_tool(
        self,
        tool_name: str,
        executor: Any,  # Callable
        schema: Dict[str, Any]
    ) -> None:
        """Register a new tool for invocation"""
        ...


class SpacetimeFabricProtocol(Protocol):
    """
    Provenance and execution trace logging

    SpacetimeFabric provides:
    - Complete execution traces
    - Causal event chains
    - Audit logging
    - Replay and debugging capabilities
    """

    async def log_event(self, event: SpacetimeEvent) -> None:
        """Log a single event to the fabric"""
        ...

    async def get_trace(
        self,
        start_time: datetime,
        end_time: datetime,
        component: Optional[str] = None
    ) -> List[SpacetimeEvent]:
        """Retrieve events within a time window"""
        ...

    async def get_causal_chain(
        self,
        event_id: str
    ) -> List[SpacetimeEvent]:
        """Get the causal chain leading to an event"""
        ...


class ReflectionBufferProtocol(Protocol):
    """
    Learning and adaptation memory

    ReflectionBuffer handles:
    - Experience replay for RL
    - Feedback incorporation
    - Performance metrics
    - Continuous improvement
    """

    async def store_experience(
        self,
        state: Dict[str, Any],
        action: RiftAction,
        reward: float,
        next_state: Dict[str, Any]
    ) -> None:
        """Store an experience for later learning"""
        ...

    async def sample_batch(
        self,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Sample a batch of experiences for training"""
        ...

    async def update_metrics(
        self,
        metrics: Dict[str, float]
    ) -> None:
        """Update performance metrics"""
        ...


class ThreadSpinnerProtocol(Protocol):
    """
    Hot/cold memory paging manager (MemGPT-like)

    ThreadSpinner provides:
    - Automatic hot/cold classification
    - Paging between fast/slow storage
    - Access pattern tracking
    - Memory pressure management
    """

    async def page_in(self, thread_id: str) -> Thread:
        """Load a thread into hot memory"""
        ...

    async def page_out(self, thread_id: str) -> None:
        """Move a thread to cold storage"""
        ...

    async def classify_memory(
        self,
        thread_id: str,
        access_pattern: Dict[str, Any]
    ) -> MemoryType:
        """Classify memory as hot/warm/cold based on access pattern"""
        ...

    async def get_hot_threads(
        self,
        limit: int = 100
    ) -> List[Thread]:
        """Get currently hot (actively used) threads"""
        ...


# ==================== Loom Core Interface ====================

class LoomCore(ABC):
    """
    Main Loom Core interface - orchestrates all subsystems

    LoomCore is the central reasoning engine that coordinates:
    - WarpSpace (semantic retrieval)
    - YarnGraph (knowledge structure)
    - ResonanceShed (multimodal fusion)
    - ConvergenceEngine (decision making)
    - Rift (tool execution)
    - SpacetimeFabric (provenance)
    - ReflectionBuffer (learning)
    - ThreadSpinner (memory management)
    """

    def __init__(
        self,
        warp_space: WarpSpaceProtocol,
        yarn_graph: YarnGraphProtocol,
        resonance_shed: ResonanceShedProtocol,
        convergence_engine: ConvergenceEngineProtocol,
        rift: RiftProtocol,
        spacetime_fabric: SpacetimeFabricProtocol,
        reflection_buffer: ReflectionBufferProtocol,
        thread_spinner: ThreadSpinnerProtocol
    ):
        self.warp_space = warp_space
        self.yarn_graph = yarn_graph
        self.resonance_shed = resonance_shed
        self.convergence_engine = convergence_engine
        self.rift = rift
        self.spacetime_fabric = spacetime_fabric
        self.reflection_buffer = reflection_buffer
        self.thread_spinner = thread_spinner

    @abstractmethod
    async def weave(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main reasoning loop - the "weaving" process

        Weaving coordinates:
        1. Semantic retrieval (WarpSpace)
        2. Graph traversal (YarnGraph)
        3. Decision making (ConvergenceEngine)
        4. Action execution (Rift)
        5. Provenance logging (SpacetimeFabric)
        6. Learning (ReflectionBuffer)
        """
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize all Loom subsystems"""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown all Loom subsystems"""
        ...
