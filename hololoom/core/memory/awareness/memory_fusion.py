"""
Memory Fusion Module - Multipass Graph Crawling Integration

Integrates advanced memory retrieval with context packing:
- Recursive gated multipass crawling
- Graph traversal following entity relationships  
- Temporal weighting (recency + relevance)
- Importance-based depth control
- Result fusion with composite scoring

This enhances SmartContextPacker with intelligent memory exploration.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryNode:
    """Single memory item with graph relationships"""

    id: str
    content: str
    relevance: float  # Initial relevance score (0.0-1.0)
    timestamp: datetime | None = None

    # Graph relationships
    related_ids: list[str] = field(default_factory=list)

    # Fusion metadata
    retrieval_depth: int = 0  # How many hops from initial query
    composite_score: float = 0.0  # Relevance + temporal + graph
    source_path: list[str] = field(default_factory=list)  # Path to this node

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultipassConfig:
    """Configuration for multipass memory crawling"""

    max_passes: int = 3  # Number of crawl passes
    max_depth: int = 3  # Maximum graph traversal depth

    # Matryoshka importance gating (increasing thresholds by depth)
    importance_thresholds: list[float] = field(
        default_factory=lambda: [0.6, 0.75, 0.85]
    )

    # Result limits per pass
    items_per_pass: list[int] = field(
        default_factory=lambda: [10, 8, 5]
    )

    # Temporal weighting
    temporal_decay_hours: float = 24.0  # Hours for 50% temporal weight
    temporal_weight: float = 0.3  # Weight of temporal component

    # Graph weighting
    graph_decay_per_hop: float = 0.15  # Score decay per graph hop
    graph_weight: float = 0.4  # Weight of graph proximity

    # Relevance weighting
    relevance_weight: float = 0.3  # Weight of original relevance

    @classmethod
    def for_complexity(cls, complexity: str) -> 'MultipassConfig':
        """Create config based on complexity level"""

        if complexity == "LITE":
            return cls(
                max_passes=1,
                max_depth=1,
                importance_thresholds=[0.7],
                items_per_pass=[5]
            )
        elif complexity == "FAST":
            return cls(
                max_passes=2,
                max_depth=2,
                importance_thresholds=[0.6, 0.75],
                items_per_pass=[8, 5]
            )
        elif complexity == "FULL":
            return cls(
                max_passes=3,
                max_depth=3,
                importance_thresholds=[0.6, 0.75, 0.85],
                items_per_pass=[10, 8, 5]
            )
        elif complexity == "RESEARCH":
            return cls(
                max_passes=4,
                max_depth=4,
                importance_thresholds=[0.5, 0.65, 0.8, 0.9],
                items_per_pass=[15, 12, 8, 5]
            )
        else:
            return cls()  # Default FULL


class MemoryFusion:
    """
    Advanced memory retrieval with multipass graph crawling.
    
    Strategy:
    1. Initial retrieval (Pass 0): Broad search above threshold
    2. Graph expansion (Pass 1+): Follow relationships from high-scoring items
    3. Temporal weighting: Boost recent memories
    4. Composite scoring: Combine relevance + temporal + graph proximity
    5. Deduplication: Keep best path to each unique item
    """

    def __init__(
        self,
        config: MultipassConfig | None = None,
        memory_backend = None
    ):
        """
        Initialize memory fusion.
        
        Args:
            config: Multipass crawling configuration
            memory_backend: Backend supporting retrieve() and get_related()
        """
        self.config = config or MultipassConfig()
        self.backend = memory_backend

    async def retrieve_with_fusion(
        self,
        query: str,
        max_results: int = 20,
        reference_time: datetime | None = None
    ) -> list[MemoryNode]:
        """
        Retrieve memories with multipass fusion.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            reference_time: Reference time for temporal weighting (default: now)
            
        Returns:
            List of MemoryNode with composite scores
        """

        if reference_time is None:
            reference_time = datetime.now()

        start_time = time.time()

        # Track all discovered nodes (deduplicated by id)
        discovered: dict[str, MemoryNode] = {}

        # Track which nodes to expand in next pass
        expansion_frontier: set[str] = set()

        # Pass 0: Initial broad retrieval
        initial_results = await self._initial_retrieval(
            query,
            threshold=self.config.importance_thresholds[0],
            limit=self.config.items_per_pass[0]
        )

        for result in initial_results:
            node = self._result_to_node(result, depth=0, source_path=[])
            discovered[node.id] = node
            expansion_frontier.add(node.id)

        print(f"  Pass 0: Retrieved {len(initial_results)} initial items (threshold: {self.config.importance_thresholds[0]:.2f})")

        # Multipass graph expansion
        for pass_num in range(1, min(self.config.max_passes, len(self.config.importance_thresholds))):

            threshold = self.config.importance_thresholds[pass_num]
            limit = self.config.items_per_pass[pass_num] if pass_num < len(self.config.items_per_pass) else 5

            # Expand from frontier
            new_nodes = await self._expand_frontier(
                expansion_frontier,
                discovered,
                depth=pass_num,
                threshold=threshold,
                limit=limit
            )

            print(f"  Pass {pass_num}: Expanded {len(new_nodes)} new items (threshold: {threshold:.2f}, depth: {pass_num})")

            # Update frontier for next pass
            expansion_frontier = {node.id for node in new_nodes}

            if not expansion_frontier:
                print(f"  Pass {pass_num}: No new items, stopping early")
                break

        # Calculate composite scores
        scored_nodes = self._calculate_composite_scores(
            list(discovered.values()),
            reference_time
        )

        # Sort by composite score and limit
        scored_nodes.sort(key=lambda n: n.composite_score, reverse=True)
        final_results = scored_nodes[:max_results]

        retrieval_time = (time.time() - start_time) * 1000

        print(f"  ✅ Retrieved {len(final_results)} memories in {retrieval_time:.2f}ms")
        if final_results:
            print(f"     Avg composite score: {sum(n.composite_score for n in final_results) / len(final_results):.2f}")
            print(f"     Max depth reached: {max(n.retrieval_depth for n in final_results)}")
        else:
            print(f"     No results matched query")

        return final_results

    async def _initial_retrieval(
        self,
        query: str,
        threshold: float,
        limit: int
    ) -> list[dict[str, Any]]:
        """Initial broad retrieval"""

        if self.backend and hasattr(self.backend, 'retrieve_with_threshold'):
            return await self.backend.retrieve_with_threshold(query, threshold, limit)
        elif self.backend and hasattr(self.backend, 'retrieve'):
            # Fallback to basic retrieve
            results = await self.backend.retrieve(query, limit=limit)
            # Filter by threshold if relevance available
            return [r for r in results if r.get('relevance', 0) >= threshold]
        else:
            # No backend - return empty
            return []

    async def _expand_frontier(
        self,
        frontier: set[str],
        discovered: dict[str, MemoryNode],
        depth: int,
        threshold: float,
        limit: int
    ) -> list[MemoryNode]:
        """Expand from frontier nodes via graph relationships"""

        new_nodes = []

        for node_id in frontier:
            if node_id not in discovered:
                continue

            parent_node = discovered[node_id]

            # Get related items
            related = await self._get_related_items(node_id, limit)

            for related_item in related:
                related_id = related_item.get('id', related_item.get('content', str(related_item))[:50])

                # Skip if already discovered
                if related_id in discovered:
                    continue

                # Check if meets threshold
                relevance = related_item.get('relevance', related_item.get('score', 0.5))
                if relevance < threshold:
                    continue

                # Create node
                node = self._result_to_node(
                    related_item,
                    depth=depth,
                    source_path=parent_node.source_path + [parent_node.id]
                )

                discovered[node.id] = node
                new_nodes.append(node)

                if len(new_nodes) >= limit:
                    break

            if len(new_nodes) >= limit:
                break

        return new_nodes

    async def _get_related_items(
        self,
        item_id: str,
        limit: int
    ) -> list[dict[str, Any]]:
        """Get items related to given item"""

        if self.backend and hasattr(self.backend, 'get_related'):
            return await self.backend.get_related(item_id, limit=limit)
        else:
            return []

    def _result_to_node(
        self,
        result: dict[str, Any],
        depth: int,
        source_path: list[str]
    ) -> MemoryNode:
        """Convert retrieval result to MemoryNode"""

        # Extract fields (handle different formats)
        node_id = result.get('id', result.get('content', str(result))[:50])
        content = result.get('content', result.get('text', str(result)))
        relevance = result.get('relevance', result.get('score', 0.5))

        # Parse timestamp
        timestamp = None
        if 'timestamp' in result:
            ts = result['timestamp']
            if isinstance(ts, datetime):
                timestamp = ts
            elif isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts)
                except ValueError:
                    pass

        # Extract related IDs
        related_ids = result.get('related', result.get('related_ids', []))

        return MemoryNode(
            id=node_id,
            content=content,
            relevance=relevance,
            timestamp=timestamp,
            related_ids=related_ids,
            retrieval_depth=depth,
            source_path=source_path,
            metadata=result.get('metadata', {})
        )

    def _calculate_composite_scores(
        self,
        nodes: list[MemoryNode],
        reference_time: datetime
    ) -> list[MemoryNode]:
        """
        Calculate composite scores combining relevance + temporal + graph.
        
        Score = (relevance_weight * relevance) +
                (temporal_weight * temporal_score) +
                (graph_weight * graph_proximity_score)
        """

        for node in nodes:
            # Relevance component
            relevance_component = node.relevance * self.config.relevance_weight

            # Temporal component
            temporal_component = self._calculate_temporal_score(
                node.timestamp,
                reference_time
            ) * self.config.temporal_weight

            # Graph proximity component
            graph_component = self._calculate_graph_score(
                node.retrieval_depth
            ) * self.config.graph_weight

            # Composite score
            node.composite_score = relevance_component + temporal_component + graph_component

        return nodes

    def _calculate_temporal_score(
        self,
        timestamp: datetime | None,
        reference_time: datetime
    ) -> float:
        """Calculate temporal score (1.0 = very recent, 0.0 = very old)"""

        if timestamp is None:
            return 0.5  # Neutral if no timestamp

        # Calculate age in hours
        age_hours = (reference_time - timestamp).total_seconds() / 3600

        # Exponential decay (half-life = temporal_decay_hours)
        decay_rate = 0.693 / self.config.temporal_decay_hours
        temporal_score = 2 ** (-decay_rate * age_hours)

        return max(0.0, min(1.0, temporal_score))

    def _calculate_graph_score(self, depth: int) -> float:
        """Calculate graph proximity score (1.0 = direct, lower = more hops)"""

        # Linear decay per hop
        score = 1.0 - (depth * self.config.graph_decay_per_hop)

        return max(0.0, min(1.0, score))
