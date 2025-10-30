"""
Awareness Graph - Living memory with semantic topology.

Key design:
1. Composes existing backends (graph + vector stores)
2. Memory is immutable ground truth
3. Position/topology/activation are indices (recomputable)
4. One graph, typed edges (TEMPORAL | SEMANTIC | CAUSAL)
5. Activation as process (field-based)
6. Simple policy interface (memories + metrics)
"""

from typing import List, Dict, Optional, Set, Any, Union
from collections import deque
from datetime import datetime
import numpy as np
import networkx as nx
import uuid

from HoloLoom.memory.protocol import Memory
from HoloLoom.memory.awareness_types import (
    SemanticPerception,
    ActivationStrategy,
    ActivationBudget,
    AwarenessMetrics,
    EdgeType,
    EdgeMetadata
)
from HoloLoom.memory.activation_field import ActivationField

# Multi-modal input support (graceful degradation if not available)
try:
    from HoloLoom.input.protocol import ProcessedInput
    MULTIMODAL_AVAILABLE = True
except ImportError:
    ProcessedInput = None  # type: ignore
    MULTIMODAL_AVAILABLE = False


class AwarenessGraph:
    """
    Living memory with semantic topology.

    Elegant composition:
    - Graph backend (Neo4j/NetworkX) handles topology
    - Vector store (Qdrant/FAISS) handles fast semantic search
    - Semantic calculus handles perception
    - Activation field handles dynamic retrieval
    """

    def __init__(
        self,
        graph_backend: nx.MultiDiGraph,  # Can be Neo4j or NetworkX
        semantic_calculus,                # MatryoshkaSemanticCalculus
        vector_store: Optional[Any] = None  # Optional Qdrant/FAISS
    ):
        """
        Initialize awareness graph.

        Args:
            graph_backend: NetworkX or Neo4j graph for topology
            semantic_calculus: Semantic analyzer for perception
            vector_store: Optional vector DB for fast search
        """
        # Backend composition (no duplication!)
        self.graph = graph_backend
        self.semantic = semantic_calculus
        self.vectors = vector_store

        # Semantic index (lightweight: node_id → position)
        self.semantic_positions: Dict[str, np.ndarray] = {}

        # Activation field (dynamic process)
        self.activation_field = ActivationField()

        # Trajectory tracking
        self.trajectory: deque = deque(maxlen=100)

        # Resonance cache (for metrics)
        self.resonance_cache: Dict[tuple, float] = {}

    # =========================================================================
    # Core Operations: Perceive, Remember, Activate
    # =========================================================================

    def _align_embedding_to_228d(self, embedding: np.ndarray) -> np.ndarray:
        """
        Align any-sized embedding to 228D semantic space.

        Strategies:
        - embedding < 228D: Pad with zeros
        - embedding == 228D: Use directly
        - embedding > 228D: Project via PCA-like truncation

        Falls forward: Pre-computed embeddings slot right in!
        """
        current_dim = embedding.shape[0]
        target_dim = 228

        if current_dim == target_dim:
            # Perfect match - use directly
            return embedding

        elif current_dim < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:current_dim] = embedding
            return padded

        else:  # current_dim > target_dim
            # Truncate to 228D (simple projection)
            # Could use PCA, but truncation preserves leading dimensions
            return embedding[:target_dim]

    async def perceive(
        self,
        content: Union[str, 'ProcessedInput']
    ) -> SemanticPerception:
        """
        Analyze input through semantic calculus OR accept pre-computed embeddings.

        Multimodal awareness: Falls forward naturally!

        Args:
            content: Either text string OR ProcessedInput with pre-computed embedding

        Returns:
            SemanticPerception with 228D position

        Examples:
            # Text (streaming semantic calculus)
            perception = await awareness.perceive("Thompson Sampling")

            # Multimodal (pre-computed embedding)
            from HoloLoom.input import InputRouter
            router = InputRouter()
            processed = await router.process({"data": "structured"})
            perception = await awareness.perceive(processed)
        """
        # Case 1: Pre-computed embedding (multimodal input)
        if MULTIMODAL_AVAILABLE and isinstance(content, ProcessedInput):
            # Use pre-computed embedding as position
            position = self._align_embedding_to_228d(content.embedding)

            # Get previous position for shift detection
            prev_position = self.trajectory[-1] if self.trajectory else None

            # Detect shift
            shift_magnitude = 0.0
            shift_detected = False
            if prev_position is not None:
                shift_magnitude = float(np.linalg.norm(position - prev_position))
                shift_detected = shift_magnitude > 0.5

            # Create perception directly
            perception = SemanticPerception(
                position=position,
                velocity=None,  # No velocity for static embeddings
                dominant_dimensions=[],  # Could extract from features
                momentum=content.confidence,  # Use confidence as momentum proxy
                complexity=1.0,  # Default complexity
                shift_magnitude=shift_magnitude,
                shift_detected=shift_detected
            )

            # Track trajectory
            self.trajectory.append(perception.position)

            return perception

        # Case 2: Text string (streaming semantic calculus)
        elif isinstance(content, str):
            # Stream analyze through semantic calculus
            async def word_stream():
                for word in content.split():
                    yield word

            # Get final snapshot
            final_snapshot = None
            async for snapshot in self.semantic.stream_analyze(word_stream()):
                final_snapshot = snapshot

            # Get previous position for shift detection
            prev_position = self.trajectory[-1] if self.trajectory else None

            # Convert to SemanticPerception
            perception = SemanticPerception.from_snapshot(final_snapshot, prev_position)

            # Track trajectory
            self.trajectory.append(perception.position)

            return perception

        else:
            raise TypeError(
                f"Content must be str or ProcessedInput, got {type(content)}"
            )

    async def remember(
        self,
        content: Union[str, 'ProcessedInput'],
        perception: SemanticPerception,
        context: Optional[Dict] = None
    ) -> str:
        """
        Store memory with semantic integration.

        Multimodal: Accepts text OR ProcessedInput.

        Weaves memory into:
        1. Graph backend (topology)
        2. Vector store (fast retrieval)
        3. Semantic index (local lookup)
        4. Activation field (spatial index)

        Returns: memory_id
        """
        # Extract text representation
        if MULTIMODAL_AVAILABLE and isinstance(content, ProcessedInput):
            text = content.content  # Human-readable description
            # Merge context with modality info
            if context is None:
                context = {}
            context['modality'] = content.modality.value
            context['source'] = content.source
            context['confidence'] = content.confidence
        elif isinstance(content, str):
            text = content
        else:
            raise TypeError(f"Content must be str or ProcessedInput, got {type(content)}")

        # Create immutable Memory (content only)
        memory_id = str(uuid.uuid4())
        memory = Memory(
            id=memory_id,
            text=text,
            timestamp=datetime.now(),
            context=context or {},
            metadata={
                'dominant_dimensions': perception.dominant_dimensions,
                'momentum': perception.momentum,
                'complexity': perception.complexity
            }
        )

        # 1. Store in graph backend
        self.graph.add_node(
            memory_id,
            **memory.to_dict()
        )

        # 2. Store in vector store (if available)
        if self.vectors is not None:
            try:
                await self.vectors.add(memory_id, perception.position)
            except:
                pass  # Graceful degradation

        # 3. Update semantic index
        self.semantic_positions[memory_id] = perception.position

        # 4. Update activation field spatial index
        self.activation_field.update_spatial_index(memory_id, perception.position)

        # 5. Weave temporal connections (most recent memories)
        await self._weave_temporal(memory_id)

        # 6. Weave semantic connections (resonant memories)
        await self._weave_semantic(memory_id, perception.position)

        return memory_id

    async def activate(
        self,
        perception: SemanticPerception,
        budget: Optional[ActivationBudget] = None,
        strategy: ActivationStrategy = ActivationStrategy.BALANCED
    ) -> List[Memory]:
        """
        Activate relevant memories based on semantic proximity.

        Args:
            perception: Query semantic state
            budget: Resource constraints (optional)
            strategy: Activation pattern

        Returns: List of activated Memory objects
        """
        # Default budget from strategy
        if budget is None:
            budget = ActivationBudget.for_strategy(strategy)

        # Clear previous activation
        self.activation_field.clear()

        # 1. Fast semantic search (vector store if available)
        if self.vectors is not None:
            try:
                nearby_ids = await self.vectors.search(
                    perception.position,
                    radius=budget.semantic_radius,
                    k=budget.max_memories * 2
                )
            except:
                nearby_ids = self._brute_force_search(
                    perception.position,
                    budget.semantic_radius,
                    budget.max_memories * 2
                )
        else:
            nearby_ids = self._brute_force_search(
                perception.position,
                budget.semantic_radius,
                budget.max_memories * 2
            )

        # 2. Activate region
        self.activation_field.activate_region(
            center=perception.position,
            radius=budget.semantic_radius,
            node_ids=nearby_ids
        )

        # 3. Spread activation through graph
        if budget.spread_iterations > 0:
            self.activation_field.spread_via_graph(
                self.graph,
                iterations=budget.spread_iterations
            )

        # 4. Get activated nodes above threshold
        activated_ids = self.activation_field.above_threshold(
            budget.activation_threshold
        )

        # 5. Respect max limit
        top_ids = activated_ids[:budget.max_memories]

        # 6. Fetch full Memory objects
        memories = []
        for node_id in top_ids:
            node_data = self.graph.nodes.get(node_id)
            if node_data:
                memory = Memory.from_dict(node_data)
                memories.append(memory)

        return memories

    # =========================================================================
    # Connection Weaving
    # =========================================================================

    async def _weave_temporal(self, new_node_id: str, window: int = 10):
        """Connect to recent memories (temporal sequence)."""
        recent_nodes = list(self.graph.nodes())[-window:]

        if len(recent_nodes) > 1:
            prev_node = recent_nodes[-2]
            self.graph.add_edge(
                prev_node,
                new_node_id,
                type=EdgeType.TEMPORAL.value,
                timestamp=datetime.now().isoformat()
            )

    async def _weave_semantic(
        self,
        new_node_id: str,
        position: np.ndarray,
        threshold: float = 0.7
    ):
        """Connect to semantically resonant memories."""
        for existing_id, existing_pos in self.semantic_positions.items():
            if existing_id == new_node_id:
                continue

            resonance = self._compute_resonance(position, existing_pos)

            if resonance > threshold:
                self.graph.add_edge(
                    new_node_id,
                    existing_id,
                    type=EdgeType.SEMANTIC_RESONANCE.value,
                    strength=float(resonance)
                )
                self.graph.add_edge(
                    existing_id,
                    new_node_id,
                    type=EdgeType.SEMANTIC_RESONANCE.value,
                    strength=float(resonance)
                )

                self.resonance_cache[(new_node_id, existing_id)] = resonance

    def add_causal_edge(self, source_id: str, target_id: str, tool: str):
        """Add causal edge (query caused result via tool)."""
        self.graph.add_edge(
            source_id,
            target_id,
            type=EdgeType.CAUSAL.value,
            tool=tool,
            timestamp=datetime.now().isoformat()
        )

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> AwarenessMetrics:
        """Get current awareness state metrics for policy."""
        if self.trajectory:
            current_position = np.mean(list(self.trajectory)[-5:], axis=0)
        else:
            current_position = np.zeros(228)  # 228D = EXTENDED_244_DIMENSIONS actual size

        position_sample = current_position[:64]

        shift_magnitude = 0.0
        if len(self.trajectory) >= 2:
            shift_magnitude = float(np.linalg.norm(
                self.trajectory[-1] - self.trajectory[-2]
            ))

        n_nodes = len(self.graph.nodes)
        n_edges = len(self.graph.edges)

        avg_resonance = float(np.mean(list(self.resonance_cache.values()))) if self.resonance_cache else 0.0

        return AwarenessMetrics(
            current_position=position_sample,
            shift_magnitude=shift_magnitude,
            shift_detected=shift_magnitude > 0.5,
            n_memories=n_nodes,
            n_connections=n_edges,
            avg_resonance=avg_resonance,
            n_active=self.activation_field.n_active(),
            activation_density=self.activation_field.density(),
            trajectory_length=len(self.trajectory)
        )

    # =========================================================================
    # Utilities
    # =========================================================================

    def _compute_resonance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute semantic resonance (cosine similarity)."""
        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(pos1, pos2) / (norm1 * norm2))

    def _brute_force_search(
        self,
        query_pos: np.ndarray,
        radius: float,
        k: int
    ) -> List[str]:
        """Brute force semantic search (fallback)."""
        distances = []

        for node_id, node_pos in self.semantic_positions.items():
            distance = float(np.linalg.norm(query_pos - node_pos))
            if distance < radius:
                distances.append((node_id, distance))

        distances.sort(key=lambda x: x[1])
        return [node_id for node_id, _ in distances[:k]]
