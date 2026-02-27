"""
HoloLoom Lite Memory - Pure Memory Operations
==============================================

Memory does one thing: store, search, learn, navigate.
No reasoning. No generation. Just memory.

Usage:
    from hololoom.lite import Memory

    async with Memory() as mem:
        await mem.store("Thompson Sampling balances exploration")
        results = await mem.search("exploration strategies")
        await mem.feedback(results, helpful=True)
        neighbors = await mem.related(results[0].id)

Date: December 2025
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import warnings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.config import Config


@dataclass
class MemoryItem:
    """A single memory."""
    id: str
    text: str
    timestamp: float = 0.0
    context: Dict[str, Any] = None
    relevance: float = 0.0  # Set during search

    def __post_init__(self):
        if self.context is None:
            self.context = {}

    def __str__(self) -> str:
        return self.text


class Memory:
    """
    Pure memory operations.

    4 methods:
        store(content) - Add memory
        search(query) - Find memories
        feedback(items, helpful) - Learn from usage
        related(memory_id) - Navigate graph

    This class knows nothing about reasoning or generation.
    It just holds things.
    """

    def __init__(
        self,
        config: Optional['Config'] = None,
        persist: bool = False,
    ):
        """
        Initialize Memory.

        Args:
            config: Optional config (defaults to Config.lite())
            persist: Use persistent storage (Neo4j + Qdrant)
        """
        from hololoom.config import Config, MemoryBackend

        self.config = config or Config.lite()
        self._persist = persist

        if persist:
            self.config.memory_backend = MemoryBackend.HYBRID

        self._graph = None
        self._embedder = None
        self._backend = None
        self._initialized = False

    async def _initialize(self):
        """Initialize on first use."""
        if self._initialized:
            return

        import networkx as nx
        from hololoom.memory.awareness_graph import AwarenessGraph
        from hololoom.embedding.spectral import MatryoshkaEmbeddings
        from hololoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus

        self._embedder = MatryoshkaEmbeddings(sizes=self.config.scales)

        semantic = MatryoshkaSemanticCalculus(
            matryoshka_embedder=self._embedder,
            snapshot_interval=0.5
        )

        if self._persist:
            try:
                from hololoom.memory.backend_factory import create_memory_backend
                self._backend = await create_memory_backend(self.config)

                backend_type = type(self._backend).__name__
                if 'InMemory' in backend_type or 'NetworkX' in backend_type:
                    warnings.warn(
                        "Persistent storage unavailable. Using in-memory. "
                        "Start Docker: docker-compose -f docker-compose.lite.yml up -d"
                    )

                graph_backend = getattr(self._backend, 'graph', nx.MultiDiGraph())
                vector_store = getattr(self._backend, 'vector_store', None)

            except Exception as e:
                warnings.warn(f"Persistent backend failed: {e}. Using in-memory.")
                graph_backend = nx.MultiDiGraph()
                vector_store = None
        else:
            graph_backend = nx.MultiDiGraph()
            vector_store = None

        self._graph = AwarenessGraph(
            graph_backend=graph_backend,
            semantic_calculus=semantic,
            vector_store=vector_store
        )

        self._initialized = True

    # =========================================================================
    # 4 Core Methods
    # =========================================================================

    async def store(
        self,
        content: str,
        context: Optional[Dict] = None
    ) -> MemoryItem:
        """
        Store content in memory.

        Args:
            content: Text to remember
            context: Optional metadata

        Returns:
            MemoryItem with id and text

        Example:
            >>> item = await mem.store("Python uses indentation")
            >>> print(item.id)
        """
        await self._initialize()

        perception = await self._graph.perceive(content)
        memory_id = await self._graph.remember(content, perception, context)

        node_data = self._graph.graph.nodes[memory_id]
        return MemoryItem(
            id=memory_id,
            text=node_data['text'],
            timestamp=node_data['timestamp'],
            context=node_data.get('context', {}),
        )

    async def search(
        self,
        query: str,
        limit: int = 5
    ) -> List[MemoryItem]:
        """
        Search for relevant memories.

        Args:
            query: What to search for
            limit: Max results

        Returns:
            List of MemoryItems, most relevant first

        Example:
            >>> results = await mem.search("Python")
            >>> for item in results:
            ...     print(item.text)
        """
        await self._initialize()

        from hololoom.memory.awareness_types import ActivationStrategy

        perception = await self._graph.perceive(query)
        memories = await self._graph.activate(
            perception,
            strategy=ActivationStrategy.BALANCED
        )

        items = []
        for mem in memories[:limit]:
            items.append(MemoryItem(
                id=mem.id,
                text=mem.text,
                timestamp=getattr(mem, 'timestamp', 0.0),
                context=getattr(mem, 'context', {}),
                relevance=getattr(mem, 'relevance', 0.0),
            ))
        return items

    async def feedback(
        self,
        items: List[MemoryItem],
        helpful: bool = True,
        relevance: float = 0.8
    ) -> None:
        """
        Learn from feedback on searched items.

        Strengthens helpful memories, weakens unhelpful ones.

        Args:
            items: Memory items to rate
            helpful: Were these helpful?
            relevance: How relevant (0.0-1.0)?

        Example:
            >>> results = await mem.search("Python")
            >>> await mem.feedback(results, helpful=True)
        """
        await self._initialize()

        if not items:
            return

        # Update activation weights based on feedback
        for item in items:
            if item.id in self._graph.graph.nodes:
                node = self._graph.graph.nodes[item.id]

                # Increase/decrease activation based on feedback
                current = node.get('activation', 0.5)
                delta = 0.1 if helpful else -0.1
                new_activation = max(0.0, min(1.0, current + delta * relevance))
                node['activation'] = new_activation

                # Track feedback count
                node['feedback_count'] = node.get('feedback_count', 0) + 1
                node['helpful_count'] = node.get('helpful_count', 0) + (1 if helpful else 0)

    async def related(
        self,
        memory_id: str,
        hops: int = 1,
        direction: str = "both"
    ) -> List[MemoryItem]:
        """
        Get memories connected to a given memory.

        Args:
            memory_id: Starting point
            hops: Graph distance (1 = neighbors, 2 = neighbors of neighbors)
            direction: "out", "in", or "both"

        Returns:
            Connected MemoryItems

        Example:
            >>> item = await mem.store("Thompson Sampling")
            >>> neighbors = await mem.related(item.id)
        """
        await self._initialize()

        graph = self._graph.graph

        if memory_id not in graph:
            return []

        visited = {memory_id}
        current = {memory_id}
        neighbors = set()

        for _ in range(hops):
            next_level = set()
            for node in current:
                if direction in ("out", "both"):
                    next_level.update(n for n in graph.successors(node) if n not in visited)
                if direction in ("in", "both"):
                    next_level.update(n for n in graph.predecessors(node) if n not in visited)

            neighbors.update(next_level)
            visited.update(next_level)
            current = next_level

            if not current:
                break

        items = []
        for node_id in neighbors:
            if node_id in graph.nodes:
                data = graph.nodes[node_id]
                if 'text' in data:
                    items.append(MemoryItem(
                        id=node_id,
                        text=data['text'],
                        timestamp=data.get('timestamp', 0.0),
                        context=data.get('context', {}),
                    ))

        return items

    # =========================================================================
    # Convenience
    # =========================================================================

    @property
    def count(self) -> int:
        """Number of memories stored."""
        if not self._initialized or self._graph is None:
            return 0
        return len([n for n in self._graph.graph.nodes()
                   if 'text' in self._graph.graph.nodes[n]])

    def __len__(self) -> int:
        return self.count

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self):
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._backend is not None and hasattr(self._backend, 'close'):
            await self._backend.close()
