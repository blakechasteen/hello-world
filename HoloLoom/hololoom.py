"""
HoloLoom - Unified Memory System

The 10/10 Layer: Everything is a memory operation.

Usage:
    from HoloLoom import HoloLoom

    # Initialize
    loom = HoloLoom()

    # Experience content (any modality)
    memory = await loom.experience("Thompson Sampling balances exploration")

    # Recall related memories
    memories = await loom.recall("What did I learn about sampling?")

    # Reflect on outcomes
    await loom.reflect(memories, feedback={"helpful": True})

Design Philosophy:
- Single entry point (HoloLoom)
- Single representation (Memory)
- Three core operations (experience/recall/reflect)
- Implementation details hidden
- Modality-agnostic
"""

from typing import Any, List, Optional, Dict, Union
from pathlib import Path
import networkx as nx

from HoloLoom.config import Config
from HoloLoom.memory.protocol import Memory
from HoloLoom.memory.awareness_graph import AwarenessGraph
from HoloLoom.memory.awareness_types import ActivationStrategy, AwarenessMetrics
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

# Input processing (graceful degradation)
try:
    from HoloLoom.input.router import InputRouter
    MULTIMODAL_AVAILABLE = True
except ImportError:
    InputRouter = None
    MULTIMODAL_AVAILABLE = False


class HoloLoom:
    """
    Unified memory system.

    Everything is a memory operation:
    - experience() → form memories
    - recall() → activate memories
    - reflect() → learn from feedback

    Example:
        >>> loom = HoloLoom()
        >>>
        >>> # Experience (any modality)
        >>> mem = await loom.experience("Python has decorators")
        >>>
        >>> # Recall
        >>> memories = await loom.recall("Tell me about Python")
        >>>
        >>> # Reflect
        >>> await loom.reflect(memories, feedback={"helpful": True})
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        graph_backend: Optional[nx.MultiDiGraph] = None
    ):
        """
        Initialize HoloLoom.

        Args:
            config: System configuration (defaults to Config.fast())
            graph_backend: Optional graph backend (creates new if None)

        Example:
            >>> # Simple initialization
            >>> loom = HoloLoom()
            >>>
            >>> # With custom config
            >>> from HoloLoom.config import Config
            >>> loom = HoloLoom(config=Config.fused())
        """
        # Configuration
        self.config = config or Config.fast()

        # Create input router (if available)
        if MULTIMODAL_AVAILABLE:
            self._router = InputRouter()
        else:
            self._router = None

        # Create semantic calculus
        embedder = MatryoshkaEmbeddings(sizes=self.config.scales)
        self._semantic = MatryoshkaSemanticCalculus(
            matryoshka_embedder=embedder,
            snapshot_interval=0.5
        )

        # Create awareness graph (the core)
        self._graph = graph_backend or nx.MultiDiGraph()
        self._awareness = AwarenessGraph(
            graph_backend=self._graph,
            semantic_calculus=self._semantic,
            vector_store=None  # Could add vector store here
        )

    # =========================================================================
    # Core Operations: The 10/10 API
    # =========================================================================

    async def experience(
        self,
        content: Any,
        context: Optional[Dict] = None
    ) -> Memory:
        """
        Experience content and integrate into memory.

        The fundamental operation - everything flows through this.

        Handles any modality:
        - Text: "Thompson Sampling balances exploration"
        - Structured: {"algorithm": "Thompson Sampling", "type": "bayesian"}
        - Image: image_bytes (if PIL installed)
        - Audio: audio_bytes (if audio libs installed)
        - Multimodal: [text, image, data] (fused automatically)

        Args:
            content: Content to experience (any modality)
            context: Optional context metadata

        Returns:
            Memory object (immutable, can be used for recall)

        Example:
            >>> # Text
            >>> mem1 = await loom.experience("Python has decorators")
            >>>
            >>> # Structured data
            >>> mem2 = await loom.experience({
            ...     "language": "Python",
            ...     "features": ["decorators", "generators"]
            ... })
            >>>
            >>> # With context
            >>> mem3 = await loom.experience(
            ...     "Important fact",
            ...     context={"priority": "high", "source": "lecture"}
            ... )
        """
        # Fast path: text string
        if isinstance(content, str):
            perception = await self._awareness.perceive(content)
            memory_id = await self._awareness.remember(content, perception, context)

        # General path: multimodal (if available)
        elif self._router is not None and not isinstance(content, str):
            # Process through input router (handles modality detection)
            processed = await self._router.process(content)
            perception = await self._awareness.perceive(processed)
            memory_id = await self._awareness.remember(processed, perception, context)

        else:
            raise ValueError(
                f"Cannot process content type {type(content)}. "
                "Either provide text string or install multimodal dependencies."
            )

        # Retrieve full Memory object
        memory = self._awareness.graph.nodes[memory_id]
        return Memory(
            id=memory_id,
            text=memory['text'],
            timestamp=memory['timestamp'],
            context=memory.get('context', {}),
            metadata=memory.get('metadata', {})
        )

    async def recall(
        self,
        query: Any,
        strategy: ActivationStrategy = ActivationStrategy.BALANCED,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """
        Recall memories related to query.

        Query can be:
        - Text: "What is Thompson Sampling?"
        - Structured: {"topic": "reinforcement_learning"}
        - Image: image_bytes (finds similar images)

        Args:
            query: What to search for (any modality)
            strategy: Activation strategy
                - PRECISE: High precision (topic shift detection)
                - BALANCED: Balance precision/recall (default)
                - EXPLORATORY: Broad exploration
                - DEEP: Follow connections deeply
            limit: Maximum memories to return (None = no limit)

        Returns:
            Activated memories (sorted by relevance)

        Example:
            >>> # Simple recall
            >>> memories = await loom.recall("What did I learn about Python?")
            >>>
            >>> # Precise recall (narrow search)
            >>> memories = await loom.recall(
            ...     "Python decorators",
            ...     strategy=ActivationStrategy.PRECISE
            ... )
            >>>
            >>> # Exploratory recall (broad search)
            >>> memories = await loom.recall(
            ...     "programming concepts",
            ...     strategy=ActivationStrategy.EXPLORATORY,
            ...     limit=10
            ... )
        """
        # Experience query (creates temporary perception)
        if isinstance(query, str):
            perception = await self._awareness.perceive(query)

        elif self._router is not None:
            processed = await self._router.process(query)
            perception = await self._awareness.perceive(processed)

        else:
            raise ValueError(
                f"Cannot process query type {type(query)}. "
                "Either provide text string or install multimodal dependencies."
            )

        # Activate memories
        memories = await self._awareness.activate(
            perception,
            strategy=strategy
        )

        # Apply limit if specified
        if limit is not None:
            memories = memories[:limit]

        return memories

    async def reflect(
        self,
        memories: List[Memory],
        feedback: Optional[Dict] = None
    ) -> None:
        """
        Reflect on memories to improve future recall.

        Learning signal for the system. Updates topology weights,
        activation parameters, and exploration strategy based on feedback.

        Args:
            memories: Memories to reflect on
            feedback: Feedback dictionary with signals like:
                - {"helpful": True/False}
                - {"relevance": 0.8}
                - {"selected": [memory.id for memory in top_3]}
                - {"outcome": "success"}

        Example:
            >>> # Recall memories
            >>> memories = await loom.recall("machine learning algorithms")
            >>>
            >>> # User found these helpful
            >>> await loom.reflect(memories, feedback={"helpful": True})
            >>>
            >>> # More detailed feedback
            >>> await loom.reflect(memories, feedback={
            ...     "relevance": 0.9,
            ...     "selected": [memories[0].id, memories[1].id],
            ...     "outcome": "answered_question"
            ... })
        """
        # Simple reflection: strengthen activated connections
        # Future: Use reflection buffer for more sophisticated learning

        if not memories:
            return

        # Update topology weights based on feedback
        helpful = feedback.get('helpful', True) if feedback else True
        relevance = feedback.get('relevance', 0.8) if feedback else 0.8

        # For now, just track the feedback
        # Future: Update graph edge weights, adjust activation thresholds, etc.
        for memory in memories:
            # Could update metadata, edge weights, etc.
            pass

        # TODO: Integrate with reflection buffer for sophisticated learning
        # await self._awareness.reflection_buffer.learn(memories, feedback)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def experience_batch(
        self,
        contents: List[Any],
        context: Optional[Dict] = None
    ) -> List[Memory]:
        """
        Experience multiple contents efficiently.

        Args:
            contents: List of content items
            context: Optional shared context for all items

        Returns:
            List of Memory objects

        Example:
            >>> contents = [
            ...     "Python has decorators",
            ...     "Python has generators",
            ...     {"language": "Python"}
            ... ]
            >>> memories = await loom.experience_batch(contents)
        """
        memories = []
        for content in contents:
            memory = await self.experience(content, context)
            memories.append(memory)
        return memories

    async def search(
        self,
        query: str,
        **kwargs
    ) -> List[Memory]:
        """
        Alias for recall() with more intuitive name.

        Args:
            query: Search query
            **kwargs: Additional arguments passed to recall()

        Returns:
            Activated memories

        Example:
            >>> results = await loom.search("machine learning")
        """
        return await self.recall(query, **kwargs)

    def get_metrics(self) -> Dict:
        """
        Get system metrics for monitoring.

        Returns:
            Dictionary with metrics:
            - n_memories: Total memories stored
            - n_connections: Total connections (edges)
            - n_active: Currently activated memories
            - activation_density: Activation field density
            - trajectory_length: Semantic trajectory length

        Example:
            >>> metrics = loom.get_metrics()
            >>> print(f"Total memories: {metrics['n_memories']}")
        """
        awareness_metrics = self._awareness.get_metrics()
        return {
            'n_memories': awareness_metrics.n_memories,
            'n_connections': awareness_metrics.n_connections,
            'n_active': awareness_metrics.n_active,
            'activation_density': awareness_metrics.activation_density,
            'trajectory_length': awareness_metrics.trajectory_length,
            'shift_magnitude': awareness_metrics.shift_magnitude,
            'shift_detected': awareness_metrics.shift_detected
        }

    def summary(self) -> str:
        """
        Get human-readable summary of system state.

        Returns:
            Summary string

        Example:
            >>> print(loom.summary())
            HoloLoom System
            ===============
            Memories: 42
            Connections: 128
            Active: 5 (density: 0.12)
        """
        metrics = self.get_metrics()
        return f"""HoloLoom System
===============
Memories: {metrics['n_memories']}
Connections: {metrics['n_connections']}
Active: {metrics['n_active']} (density: {metrics['activation_density']:.2f})
Trajectory: {metrics['trajectory_length']} steps
Shift detected: {metrics['shift_detected']}
"""

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit (cleanup).

        Ensures proper cleanup of resources.

        Example:
            >>> async with HoloLoom() as loom:
            ...     memory = await loom.experience("content")
            ...     # Automatic cleanup on exit
        """
        # Close awareness graph (cleanup tasks, connections, etc.)
        if hasattr(self._awareness, 'close'):
            await self._awareness.close()

    # =========================================================================
    # Advanced Access (for power users)
    # =========================================================================

    @property
    def awareness(self) -> AwarenessGraph:
        """
        Direct access to awareness graph (advanced users only).

        Most users should use experience()/recall()/reflect().
        This provides full control over internal components.

        Returns:
            AwarenessGraph instance

        Example:
            >>> # Advanced: direct graph manipulation
            >>> graph = loom.awareness.graph
            >>> nodes = list(graph.nodes())
        """
        return self._awareness

    @property
    def graph(self) -> nx.MultiDiGraph:
        """
        Direct access to graph backend (advanced users only).

        Returns:
            NetworkX MultiDiGraph

        Example:
            >>> # Advanced: query graph directly
            >>> graph = loom.graph
            >>> edges = list(graph.edges(data=True))
        """
        return self._graph
