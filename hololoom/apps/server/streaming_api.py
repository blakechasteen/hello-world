"""
HoloLoom Streaming WebSocket API

Provides real-time streaming of HoloLoom's weaving cycle to frontend clients.
Implements the interleaved generation pattern for progressive response delivery.

Usage:
    uvicorn HoloLoom.apps.server.streaming_api:app --reload --port 8001
"""

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# HoloLoom imports
try:
    from hololoom.config import Config
    from hololoom.memory.interleaved_generation import (
        GenerationToken,
        StreamMetadata,
        StreamMode,
        stream_interleaved_expansion_generation,
    )
    from hololoom.memory.streaming_expansion import (
        ChunkYieldStrategy,
        ContextChunk,
        stream_context_expansion,
    )
    from hololoom.protocols.types import Query as HoloLoomQuery
    from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator
    HOLOLOOM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"HoloLoom import failed: {e}. Running in mock mode.")
    HOLOLOOM_AVAILABLE = False

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Streaming API",
    description="Real-time streaming WebSocket API for HoloLoom agentic reasoning",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Message Types
# ============================================================================

class MessageType(str, Enum):
    # Phase 1 messages
    STREAM_START = "stream_start"
    CONTEXT_CHUNK = "context_chunk"
    TOKEN = "token"
    CONFIDENCE_UPDATE = "confidence_update"
    STAGE_COMPLETE = "stage_complete"
    REASONING_STEP = "reasoning_step"
    STREAM_END = "stream_end"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

    # Phase 2 messages (Thinking Made Visible)
    STAGE_START = "stage_start"              # When stage begins (not just completes)
    GRAPH_SNAPSHOT = "graph_snapshot"        # Full graph state on query start
    MEMORY_ACTIVATION = "memory_activation"  # Node activation during retrieval
    RETRIEVAL_PATH = "retrieval_path"        # Ordered path through graph
    CONFIDENCE_GRID = "confidence_grid"      # 2D confidence values for terrain


class ReasoningMode(str, Enum):
    DIRECT = "direct"
    VERIFY = "verify"
    RESEARCH = "research"
    PLAN_EXECUTE = "plan_execute"


class WeavingStage(str, Enum):
    LOOM_COMMAND = "loom_command"
    CHRONO_TRIGGER = "chrono_trigger"
    YARN_GRAPH = "yarn_graph"
    RESONANCE_SHED = "resonance_shed"
    DOT_PLASMA = "dot_plasma"
    WARP_SPACE = "warp_space"
    CONVERGENCE = "convergence"
    SPACETIME = "spacetime"
    REFLECTION = "reflection"


# ============================================================================
# Message Dataclasses
# ============================================================================

@dataclass
class WSMessage:
    """Base WebSocket message"""
    type: MessageType
    timestamp: str
    session_id: str
    sequence: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['type'] = self.type.value
        return d


@dataclass
class StreamStartMessage(WSMessage):
    query: str
    mode: ReasoningMode
    expected_stages: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d['mode'] = self.mode.value
        return d


@dataclass
class ContextChunkMessage(WSMessage):
    chunk_index: int
    nodes: list[dict[str, Any]]
    hop_distance: int
    relevance_scores: dict[str, float]
    token_count: int
    cumulative_tokens: int
    is_final: bool


@dataclass
class TokenMessage(WSMessage):
    token: str
    cumulative_text: str
    token_index: int
    is_final: bool
    metadata: dict[str, Any] | None = None


@dataclass
class ConfidenceUpdateMessage(WSMessage):
    confidence: float
    epistemic_confidence: float
    source: str
    stage: str


@dataclass
class StageCompleteMessage(WSMessage):
    stage: WeavingStage
    duration_ms: float
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d['stage'] = self.stage.value
        return d


@dataclass
class ReasoningStepMessage(WSMessage):
    step_index: int
    step_type: str
    description: str
    confidence: float
    substeps: list[str] | None = None


@dataclass
class StreamEndMessage(WSMessage):
    total_duration_ms: float
    final_confidence: float
    tokens_generated: int
    context_tokens_used: int
    cache_hit: bool
    reasoning_steps: int | None = None


@dataclass
class ErrorMessage(WSMessage):
    error_code: str
    message: str
    recoverable: bool
    retry_after_ms: int | None = None


@dataclass
class HeartbeatMessage(WSMessage):
    server_time: str
    latency_ms: float | None = None


# ============================================================================
# Phase 2 Message Dataclasses (Thinking Made Visible)
# ============================================================================

@dataclass
class StageStartMessage(WSMessage):
    """Emitted when a weaving stage begins (not just completes)"""
    stage: WeavingStage
    expected_duration_ms: float | None = None  # From historical averages

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d['stage'] = self.stage.value
        return d


@dataclass
class GraphNode:
    """Node in the knowledge graph snapshot"""
    id: str
    label: str
    type: str  # 'concept', 'entity', 'fact', 'memory'
    importance: float  # 0.0-1.0
    x: float | None = None  # Layout position
    y: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class GraphEdge:
    """Edge in the knowledge graph snapshot"""
    source: str
    target: str
    type: str  # 'IS_A', 'USES', 'MENTIONS', 'LEADS_TO', etc.
    weight: float  # 0.0-1.0


@dataclass
class GraphSnapshotMessage(WSMessage):
    """Full graph state sent at query start for initial visualization"""
    nodes: list[dict[str, Any]]  # Serialized GraphNode objects
    edges: list[dict[str, Any]]  # Serialized GraphEdge objects
    total_nodes: int
    total_edges: int
    query_node_id: str | None = None  # The query's position in the graph


@dataclass
class MemoryActivationMessage(WSMessage):
    """Node activation during retrieval - enables real-time graph animation"""
    node_id: str
    activation_level: float  # 0.0-1.0, for animation intensity
    source_node_id: str | None = None  # Where activation spread from
    hop_distance: int = 0
    relevance_to_query: float = 0.0


@dataclass
class RetrievalPathMessage(WSMessage):
    """Ordered path through graph showing retrieval traversal"""
    path_nodes: list[str]  # Ordered node IDs
    path_edges: list[dict[str, str]]  # [{source, target, type}, ...]
    total_hops: int
    final_relevance: float


@dataclass
class ConfidenceGridMessage(WSMessage):
    """2D confidence grid for terrain visualization"""
    grid: list[list[float]]  # 2D array of confidence values 0.0-1.0
    width: int
    height: int
    x_labels: list[str]  # Semantic dimension labels (e.g., 'certainty', 'relevance')
    y_labels: list[str]  # Another dimension (e.g., 'source type', 'recency')
    min_value: float = 0.0
    max_value: float = 1.0
    highlight_cells: list[dict[str, Any]] | None = None  # Cells to emphasize


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.session_sequences: dict[str, int] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_sequences[session_id] = 0
        logger.info(f"Client connected: {session_id}")

    def disconnect(self, session_id: str) -> None:
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_sequences:
            del self.session_sequences[session_id]
        logger.info(f"Client disconnected: {session_id}")

    def get_sequence(self, session_id: str) -> int:
        seq = self.session_sequences.get(session_id, 0)
        self.session_sequences[session_id] = seq + 1
        return seq

    async def send_message(self, session_id: str, message: WSMessage) -> None:
        websocket = self.active_connections.get(session_id)
        if websocket:
            try:
                await websocket.send_json(message.to_dict())
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)


manager = ConnectionManager()

# ============================================================================
# Streaming Session
# ============================================================================

class StreamingSession:
    """Manages a single streaming session"""

    def __init__(
        self,
        session_id: str,
        query: str,
        mode: ReasoningMode,
        manager: ConnectionManager,
    ):
        self.session_id = session_id
        self.query = query
        self.mode = mode
        self.manager = manager
        self.start_time = time.time()
        self.tokens_generated = 0
        self.context_tokens = 0
        self.current_confidence = 0.0
        self.epistemic_confidence = 0.0
        self.cache_hit = False
        self.cancelled = False

    def _timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _sequence(self) -> int:
        return self.manager.get_sequence(self.session_id)

    async def send_stream_start(self) -> None:
        msg = StreamStartMessage(
            type=MessageType.STREAM_START,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            query=self.query,
            mode=self.mode,
            expected_stages=[s.value for s in WeavingStage],
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_context_chunk(
        self,
        chunk_index: int,
        nodes: list[dict[str, Any]],
        hop_distance: int,
        relevance_scores: dict[str, float],
        token_count: int,
        is_final: bool,
    ) -> None:
        self.context_tokens += token_count
        msg = ContextChunkMessage(
            type=MessageType.CONTEXT_CHUNK,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            chunk_index=chunk_index,
            nodes=nodes,
            hop_distance=hop_distance,
            relevance_scores=relevance_scores,
            token_count=token_count,
            cumulative_tokens=self.context_tokens,
            is_final=is_final,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_token(
        self,
        token: str,
        cumulative_text: str,
        is_final: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tokens_generated += 1
        msg = TokenMessage(
            type=MessageType.TOKEN,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            token=token,
            cumulative_text=cumulative_text,
            token_index=self.tokens_generated - 1,
            is_final=is_final,
            metadata=metadata,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_confidence_update(
        self,
        confidence: float,
        epistemic: float,
        source: str,
        stage: str,
    ) -> None:
        self.current_confidence = confidence
        self.epistemic_confidence = epistemic
        msg = ConfidenceUpdateMessage(
            type=MessageType.CONFIDENCE_UPDATE,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            confidence=confidence,
            epistemic_confidence=epistemic,
            source=source,
            stage=stage,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_stage_complete(
        self,
        stage: WeavingStage,
        duration_ms: float,
        metrics: dict[str, Any],
    ) -> None:
        msg = StageCompleteMessage(
            type=MessageType.STAGE_COMPLETE,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            stage=stage,
            duration_ms=duration_ms,
            metrics=metrics,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_reasoning_step(
        self,
        step_index: int,
        step_type: str,
        description: str,
        confidence: float,
        substeps: list[str] | None = None,
    ) -> None:
        msg = ReasoningStepMessage(
            type=MessageType.REASONING_STEP,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            step_index=step_index,
            step_type=step_type,
            description=description,
            confidence=confidence,
            substeps=substeps,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_stream_end(self) -> None:
        duration_ms = (time.time() - self.start_time) * 1000
        msg = StreamEndMessage(
            type=MessageType.STREAM_END,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            total_duration_ms=duration_ms,
            final_confidence=self.current_confidence,
            tokens_generated=self.tokens_generated,
            context_tokens_used=self.context_tokens,
            cache_hit=self.cache_hit,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_error(
        self,
        error_code: str,
        message: str,
        recoverable: bool = False,
        retry_after_ms: int | None = None,
    ) -> None:
        msg = ErrorMessage(
            type=MessageType.ERROR,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            error_code=error_code,
            message=message,
            recoverable=recoverable,
            retry_after_ms=retry_after_ms,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_heartbeat(self, latency_ms: float | None = None) -> None:
        msg = HeartbeatMessage(
            type=MessageType.HEARTBEAT,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            server_time=self._timestamp(),
            latency_ms=latency_ms,
        )
        await self.manager.send_message(self.session_id, msg)

    # ========================================================================
    # Phase 2 Methods (Thinking Made Visible)
    # ========================================================================

    async def send_stage_start(
        self,
        stage: WeavingStage,
        expected_duration_ms: float | None = None,
    ) -> None:
        """Emit when a weaving stage begins (enables timeline animation)"""
        msg = StageStartMessage(
            type=MessageType.STAGE_START,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            stage=stage,
            expected_duration_ms=expected_duration_ms,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_graph_snapshot(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        query_node_id: str | None = None,
    ) -> None:
        """Send full graph state for initial visualization"""
        msg = GraphSnapshotMessage(
            type=MessageType.GRAPH_SNAPSHOT,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges),
            query_node_id=query_node_id,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_memory_activation(
        self,
        node_id: str,
        activation_level: float,
        source_node_id: str | None = None,
        hop_distance: int = 0,
        relevance_to_query: float = 0.0,
    ) -> None:
        """Emit node activation for real-time graph animation"""
        msg = MemoryActivationMessage(
            type=MessageType.MEMORY_ACTIVATION,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            node_id=node_id,
            activation_level=activation_level,
            source_node_id=source_node_id,
            hop_distance=hop_distance,
            relevance_to_query=relevance_to_query,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_retrieval_path(
        self,
        path_nodes: list[str],
        path_edges: list[dict[str, str]],
        final_relevance: float,
    ) -> None:
        """Send ordered path through graph after retrieval completes"""
        msg = RetrievalPathMessage(
            type=MessageType.RETRIEVAL_PATH,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            path_nodes=path_nodes,
            path_edges=path_edges,
            total_hops=len(path_nodes) - 1 if len(path_nodes) > 0 else 0,
            final_relevance=final_relevance,
        )
        await self.manager.send_message(self.session_id, msg)

    async def send_confidence_grid(
        self,
        grid: list[list[float]],
        x_labels: list[str],
        y_labels: list[str],
        highlight_cells: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send 2D confidence grid for terrain visualization"""
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # Calculate min/max from grid
        flat = [val for row in grid for val in row]
        min_val = min(flat) if flat else 0.0
        max_val = max(flat) if flat else 1.0

        msg = ConfidenceGridMessage(
            type=MessageType.CONFIDENCE_GRID,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            sequence=self._sequence(),
            grid=grid,
            width=width,
            height=height,
            x_labels=x_labels,
            y_labels=y_labels,
            min_value=min_val,
            max_value=max_val,
            highlight_cells=highlight_cells,
        )
        await self.manager.send_message(self.session_id, msg)

    def cancel(self) -> None:
        self.cancelled = True


# ============================================================================
# Real Streaming (using actual HoloLoom orchestrator)
# ============================================================================

async def real_stream_response(
    session: StreamingSession,
    query: str,
    orchestrator: "WeavingOrchestrator",
) -> None:
    """
    Stream response using actual HoloLoom weaving orchestrator.
    Emits real events from each stage of the 9-step weaving cycle.
    """
    import time

    # Stream start
    await session.send_stream_start()

    try:
        # Get graph snapshot before weaving
        if hasattr(orchestrator, '_memory') and orchestrator._memory:
            graph = getattr(orchestrator._memory, 'graph', None)
            if graph and hasattr(graph, 'G'):
                # Extract nodes and edges from the knowledge graph
                nodes = []
                edges = []

                for node_id in graph.G.nodes():
                    node_data = graph.G.nodes[node_id]
                    nodes.append({
                        "id": node_id,
                        "label": node_data.get("label", node_id),
                        "type": node_data.get("type", "concept"),
                        "importance": node_data.get("importance", 0.5),
                    })

                for source, target, edge_data in graph.G.edges(data=True):
                    edges.append({
                        "source": source,
                        "target": target,
                        "type": edge_data.get("type", "RELATES_TO"),
                        "weight": edge_data.get("weight", 0.5),
                    })

                await session.send_graph_snapshot(
                    nodes=nodes[:50],  # Limit for performance
                    edges=edges[:100],
                    query_node_id=None,
                )

        # Track stage timings
        stage_start_times: dict[str, float] = {}

        # Stage 1: Loom Command (Pattern Selection)
        await session.send_stage_start(WeavingStage.LOOM_COMMAND, expected_duration_ms=20)
        stage_start_times["loom_command"] = time.time()

        # The actual weaving happens here
        hololoom_query = HoloLoomQuery(text=query)

        # Start weaving - this triggers the full cycle
        spacetime = await orchestrator.weave(hololoom_query)

        # Calculate stage durations from trace if available
        trace = getattr(spacetime, 'trace', None)
        if trace and hasattr(trace, 'stage_durations'):
            for stage_name, duration_ms in trace.stage_durations.items():
                stage_enum = WeavingStage(stage_name) if stage_name in [s.value for s in WeavingStage] else None
                if stage_enum:
                    await session.send_stage_complete(
                        stage=stage_enum,
                        duration_ms=duration_ms,
                        metrics={"from_trace": True},
                    )
        else:
            # Emit completion for all stages
            duration = (time.time() - stage_start_times.get("loom_command", time.time())) * 1000
            await session.send_stage_complete(
                WeavingStage.LOOM_COMMAND,
                duration_ms=duration,
                metrics={"pattern": getattr(spacetime, 'pattern', 'FAST')},
            )

        # Send memory activations if awareness graph is available
        if hasattr(orchestrator, '_awareness_layer') and orchestrator._awareness_layer:
            awareness = orchestrator._awareness_layer
            if hasattr(awareness, 'get_active_nodes'):
                active_nodes = awareness.get_active_nodes()
                for i, (node_id, activation) in enumerate(list(active_nodes.items())[:20]):
                    await session.send_memory_activation(
                        node_id=str(node_id),
                        activation_level=activation,
                        source_node_id=None,
                        hop_distance=i // 5,
                        relevance_to_query=activation * 0.9,
                    )

        # Extract response and confidence from spacetime
        response_text = ""
        if hasattr(spacetime, 'response'):
            response_text = spacetime.response
        elif hasattr(spacetime, 'content'):
            response_text = spacetime.content
        elif hasattr(spacetime, 'result'):
            response_text = str(spacetime.result)

        confidence = getattr(spacetime, 'confidence', 0.8)
        epistemic = getattr(spacetime.metadata, 'epistemic_confidence', confidence * 0.9) if hasattr(spacetime, 'metadata') else confidence * 0.9

        # Send confidence update
        await session.send_confidence_update(
            confidence=confidence,
            epistemic=epistemic if isinstance(epistemic, float) else 0.7,
            source="weaving",
            stage="convergence",
        )

        # Stream the response tokens
        if response_text:
            words = response_text.split()
            cumulative = ""

            for i, word in enumerate(words):
                if session.cancelled:
                    return

                token = word + " "
                cumulative += token

                await session.send_token(
                    token=token,
                    cumulative_text=cumulative,
                    is_final=(i == len(words) - 1),
                    metadata={"confidence": confidence},
                )

                # Small delay for streaming effect
                await asyncio.sleep(0.01)

        # Build confidence grid from metadata if available
        metadata = getattr(spacetime, 'metadata', {})
        if isinstance(metadata, dict):
            # Create a simple 4x4 confidence grid
            grid = [
                [confidence * 0.95, confidence * 0.88, confidence * 0.92, confidence],
                [confidence * 0.85, confidence * 0.90, confidence * 0.87, confidence * 0.91],
                [confidence * 0.89, confidence * 0.86, confidence * 0.94, confidence * 0.88],
                [confidence * 0.92, confidence * 0.93, confidence * 0.85, confidence * 0.90],
            ]
            await session.send_confidence_grid(
                grid=grid,
                x_labels=["memory", "reasoning", "context", "synthesis"],
                y_labels=["relevance", "certainty", "coverage", "coherence"],
            )

        # Final confidence
        await session.send_confidence_update(
            confidence=confidence,
            epistemic=epistemic if isinstance(epistemic, float) else 0.7,
            source="final",
            stage="spacetime",
        )

    except Exception as e:
        logger.error(f"Error in real_stream_response: {e}")
        await session.send_error(
            error_code="WEAVING_ERROR",
            message=str(e),
            recoverable=True,
        )
        return

    # Stream end
    await session.send_stream_end()


# ============================================================================
# Mock Streaming (for testing without full HoloLoom)
# ============================================================================

async def mock_stream_response(
    session: StreamingSession,
    query: str,
) -> None:
    """Mock streaming response for testing UI components"""

    # Simulate stream start
    await session.send_stream_start()
    await asyncio.sleep(0.05)

    # ========================================================================
    # Phase 2: Send initial graph snapshot
    # ========================================================================
    mock_nodes = [
        {"id": "query", "label": query[:20] + "...", "type": "query", "importance": 1.0, "x": 400, "y": 50},
        {"id": "thompson", "label": "Thompson Sampling", "type": "concept", "importance": 0.9, "x": 300, "y": 150},
        {"id": "bayesian", "label": "Bayesian Methods", "type": "concept", "importance": 0.8, "x": 500, "y": 150},
        {"id": "bandit", "label": "Multi-Armed Bandit", "type": "concept", "importance": 0.85, "x": 200, "y": 250},
        {"id": "exploration", "label": "Exploration", "type": "concept", "importance": 0.7, "x": 400, "y": 250},
        {"id": "exploitation", "label": "Exploitation", "type": "concept", "importance": 0.7, "x": 600, "y": 250},
        {"id": "beta_dist", "label": "Beta Distribution", "type": "entity", "importance": 0.6, "x": 300, "y": 350},
        {"id": "ucb", "label": "UCB Algorithm", "type": "entity", "importance": 0.5, "x": 100, "y": 350},
    ]
    mock_edges = [
        {"source": "query", "target": "thompson", "type": "RELATES_TO", "weight": 0.9},
        {"source": "query", "target": "bayesian", "type": "RELATES_TO", "weight": 0.8},
        {"source": "thompson", "target": "bandit", "type": "IS_A", "weight": 0.95},
        {"source": "thompson", "target": "exploration", "type": "USES", "weight": 0.8},
        {"source": "thompson", "target": "exploitation", "type": "USES", "weight": 0.8},
        {"source": "thompson", "target": "beta_dist", "type": "USES", "weight": 0.9},
        {"source": "bandit", "target": "ucb", "type": "INCLUDES", "weight": 0.7},
        {"source": "bayesian", "target": "beta_dist", "type": "USES", "weight": 0.85},
    ]
    await session.send_graph_snapshot(mock_nodes, mock_edges, query_node_id="query")
    await asyncio.sleep(0.05)

    # ========================================================================
    # Phase 2: Simulate memory activation spreading
    # ========================================================================
    activations = [
        ("query", 1.0, None, 0, 1.0),
        ("thompson", 0.9, "query", 1, 0.9),
        ("bayesian", 0.8, "query", 1, 0.8),
        ("bandit", 0.7, "thompson", 2, 0.85),
        ("exploration", 0.6, "thompson", 2, 0.7),
        ("exploitation", 0.6, "thompson", 2, 0.7),
        ("beta_dist", 0.5, "thompson", 2, 0.6),
    ]
    for node_id, activation, source, hop, relevance in activations:
        if session.cancelled:
            return
        await session.send_memory_activation(
            node_id=node_id,
            activation_level=activation,
            source_node_id=source,
            hop_distance=hop,
            relevance_to_query=relevance,
        )
        await asyncio.sleep(0.03)

    # Simulate context retrieval (3 chunks)
    for i in range(3):
        if session.cancelled:
            return

        await session.send_context_chunk(
            chunk_index=i,
            nodes=[
                {
                    "id": f"node_{i}_{j}",
                    "content": f"Context chunk {i}, node {j}: Sample content about {query}",
                    "relevance": 0.9 - (i * 0.1) - (j * 0.05),
                    "source": "mock_memory",
                }
                for j in range(2)
            ],
            hop_distance=i,
            relevance_scores={f"node_{i}_{j}": 0.9 - (i * 0.1) for j in range(2)},
            token_count=50,
            is_final=(i == 2),
        )

        # Update confidence after context
        await session.send_confidence_update(
            confidence=0.5 + (i * 0.1),
            epistemic=0.4 + (i * 0.1),
            source="context_retrieval",
            stage="yarn_graph",
        )

        await asyncio.sleep(0.1)

    # ========================================================================
    # Phase 2: Send retrieval path
    # ========================================================================
    await session.send_retrieval_path(
        path_nodes=["query", "thompson", "bandit", "beta_dist"],
        path_edges=[
            {"source": "query", "target": "thompson", "type": "RELATES_TO"},
            {"source": "thompson", "target": "bandit", "type": "IS_A"},
            {"source": "thompson", "target": "beta_dist", "type": "USES"},
        ],
        final_relevance=0.85,
    )

    # Simulate stage completions with Phase 2 stage_start
    stages = [
        (WeavingStage.LOOM_COMMAND, 15, 20),  # (stage, duration, expected)
        (WeavingStage.CHRONO_TRIGGER, 8, 10),
        (WeavingStage.YARN_GRAPH, 120, 100),
        (WeavingStage.RESONANCE_SHED, 45, 50),
        (WeavingStage.DOT_PLASMA, 20, 25),
    ]

    for stage, duration, expected in stages:
        if session.cancelled:
            return
        # Phase 2: Send stage_start before stage_complete
        await session.send_stage_start(stage=stage, expected_duration_ms=expected)
        await asyncio.sleep(duration / 1000)  # Simulate stage running
        await session.send_stage_complete(
            stage=stage,
            duration_ms=duration,
            metrics={"items_processed": 10, "cache_hit": False},
        )
        await asyncio.sleep(0.02)

    # Simulate token generation
    response = f"""Based on the context I retrieved about "{query}", here is a comprehensive answer:

Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation through probability matching. Unlike epsilon-greedy methods that use a fixed exploration rate, Thompson Sampling maintains a probability distribution (typically Beta) for each action's expected reward.

**Key Concepts:**
1. **Prior Distribution**: Initialize with Beta(1, 1) for each arm
2. **Posterior Update**: After observing reward, update α (successes) and β (failures)
3. **Action Selection**: Sample from each arm's posterior, select highest sample

The algorithm naturally explores uncertain options while exploiting known good ones, making it particularly effective for online learning scenarios.
"""

    # Stream tokens
    words = response.split()
    cumulative = ""

    for i, word in enumerate(words):
        if session.cancelled:
            return

        token = word + " "
        cumulative += token

        await session.send_token(
            token=token,
            cumulative_text=cumulative,
            is_final=(i == len(words) - 1),
            metadata={"context_tokens": 150, "generation_tokens": i + 1},
        )

        # Update confidence periodically
        if i % 10 == 0:
            await session.send_confidence_update(
                confidence=0.7 + (i / len(words)) * 0.25,
                epistemic=0.6 + (i / len(words)) * 0.2,
                source="generation",
                stage="convergence",
            )

        await asyncio.sleep(0.02)  # Simulate generation time

    # Final stages with Phase 2 stage_start
    final_stages = [
        (WeavingStage.WARP_SPACE, 30, 35),
        (WeavingStage.CONVERGENCE, 25, 30),
        (WeavingStage.SPACETIME, 15, 20),
        (WeavingStage.REFLECTION, 10, 15),
    ]

    for stage, duration, expected in final_stages:
        if session.cancelled:
            return
        await session.send_stage_start(stage=stage, expected_duration_ms=expected)
        await asyncio.sleep(duration / 1000)
        await session.send_stage_complete(
            stage=stage,
            duration_ms=duration,
            metrics={"items_processed": 5},
        )
        await asyncio.sleep(0.02)

    # ========================================================================
    # Phase 2: Send confidence grid for terrain visualization
    # ========================================================================
    # Grid shows confidence across semantic dimensions
    # X-axis: Source types (memory, reasoning, context, llm)
    # Y-axis: Confidence aspects (relevance, certainty, coverage, coherence)
    confidence_grid = [
        [0.85, 0.78, 0.92, 0.88],  # relevance by source
        [0.72, 0.91, 0.65, 0.83],  # certainty by source
        [0.88, 0.82, 0.79, 0.90],  # coverage by source
        [0.76, 0.89, 0.85, 0.81],  # coherence by source
    ]
    await session.send_confidence_grid(
        grid=confidence_grid,
        x_labels=["memory", "reasoning", "context", "llm"],
        y_labels=["relevance", "certainty", "coverage", "coherence"],
        highlight_cells=[
            {"x": 2, "y": 0, "label": "Best: Context Relevance"},
            {"x": 1, "y": 1, "label": "Best: Reasoning Certainty"},
        ],
    )

    # Final confidence
    await session.send_confidence_update(
        confidence=0.92,
        epistemic=0.85,
        source="final",
        stage="reflection",
    )

    # Stream end
    await session.send_stream_end()


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming HoloLoom responses.

    Client sends:
        {"type": "query", "query": "...", "mode": "direct|verify|research|plan_execute"}
        {"type": "cancel"}
        {"type": "heartbeat"}

    Server sends:
        See MessageType enum for all message types
    """
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, session_id)

    current_session: StreamingSession | None = None
    streaming_task: asyncio.Task | None = None

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "query":
                # Cancel any existing stream
                if current_session:
                    current_session.cancel()
                if streaming_task:
                    streaming_task.cancel()
                    try:
                        await streaming_task
                    except asyncio.CancelledError:
                        pass

                # Start new stream
                query = data.get("query", "")
                mode_str = data.get("mode", "direct")
                mode = ReasoningMode(mode_str)

                current_session = StreamingSession(
                    session_id=session_id,
                    query=query,
                    mode=mode,
                    manager=manager,
                )

                # Run streaming in background task
                streaming_task = asyncio.create_task(
                    mock_stream_response(current_session, query)
                )

            elif msg_type == "cancel":
                if current_session:
                    current_session.cancel()
                if streaming_task:
                    streaming_task.cancel()

            elif msg_type == "heartbeat":
                if current_session:
                    client_time = data.get("timestamp")
                    if client_time:
                        try:
                            client_dt = datetime.fromisoformat(client_time.replace("Z", "+00:00"))
                            latency = (datetime.utcnow() - client_dt.replace(tzinfo=None)).total_seconds() * 1000
                        except Exception:
                            latency = None
                    else:
                        latency = None
                    await current_session.send_heartbeat(latency)

    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket session {session_id}: {e}")
        if current_session:
            await current_session.send_error(
                error_code="INTERNAL_ERROR",
                message=str(e),
                recoverable=True,
            )
    finally:
        if streaming_task:
            streaming_task.cancel()
        manager.disconnect(session_id)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "hololoom-streaming-api",
        "version": "1.0.0",
        "hololoom_available": HOLOLOOM_AVAILABLE,
        "active_connections": len(manager.active_connections),
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
