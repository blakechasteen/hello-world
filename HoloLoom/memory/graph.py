"""
HoloLoom Knowledge Graph Store
===============================
Graph-based knowledge representation and traversal.

This is a "warp thread" module - independent graph storage.

Architecture:
- Protocol-based design (KGStore)
- NetworkX MultiDiGraph backend
- Entity-centric retrieval
- Weighted edges with metadata
- Zero dependencies on other HoloLoom modules (except types)

Philosophy:
The KG is the "structural memory" - how concepts relate to each other.
Unlike vector memory (which captures similarity), the graph captures
explicit relationships, hierarchies, and dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Protocol, Set, Tuple, Any
from enum import Enum
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx

from HoloLoom.utils.time_bucket import TimeInput, time_bucket, to_utc_datetime


# ============================================================================
# Eviction Strategy Enum (Phase 1: Bounded Growth - December 2025)
# ============================================================================

class EvictionStrategy(Enum):
    """
    Strategy for evicting nodes when KG reaches hard limits.

    - LRU: Least Recently Used (evict nodes accessed longest ago)
    - LFU: Least Frequently Used (evict nodes with fewest accesses)
    - IMPORTANCE_WEIGHTED: Multi-signal scoring (recency + access + centrality + connections)
    - SCOPE_AWARE: Considers lifecycle scope (EPHEMERAL → TEMPORARY → SEMANTIC)
    """
    LRU = "lru"
    LFU = "lfu"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    SCOPE_AWARE = "scope_aware"


class LifecycleScope(Enum):
    """
    Lifecycle scope for memory nodes.

    - PERMANENT: Never evict (user-pinned, critical knowledge)
    - SEMANTIC: Long-lived semantic knowledge (high bar for eviction)
    - WORKING: Session-level working memory (medium TTL)
    - TEMPORARY: Short-term memories (can be evicted easily)
    - EPHEMERAL: Very short-lived (evict first)
    """
    PERMANENT = "permanent"
    SEMANTIC = "semantic"
    WORKING = "working"
    TEMPORARY = "temporary"
    EPHEMERAL = "ephemeral"


@dataclass
class NodeAccessStats:
    """
    Tracks access statistics for a node (for eviction scoring).
    """
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    lifecycle_scope: LifecycleScope = LifecycleScope.TEMPORARY

    def record_access(self):
        """Record a node access."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class EvictionResult:
    """
    Result of an eviction operation.
    """
    nodes_evicted: int
    edges_removed: int
    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int
    strategy_used: EvictionStrategy
    eviction_time_ms: float
    evicted_node_ids: List[str] = field(default_factory=list)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class KGEdge:
    """
    A directed edge in the knowledge graph with bi-temporal support.

    Represents a typed relationship between two entities with temporal tracking.

    Bi-Temporal Model (from Graphiti research):
    - event_time: When the event/relationship occurred in reality
    - ingestion_time: When we learned about it
    - valid_from: When this edge became valid (default: ingestion_time)
    - valid_to: When this edge was invalidated (None = still valid)

    Examples:
    - ("Python", "programming_language", IS_A)
    - ("attention", "transformer", USES)
    - ("cause", "effect", LEADS_TO)

    Temporal Edge Invalidation:
    When info changes, we don't delete old edges. Instead:
    1. Mark old edge as invalid (set valid_to timestamp)
    2. Add new edge with new valid_from timestamp

    This enables point-in-time queries: "What did we know on Oct 12?"
    """
    src: str                    # Source entity
    dst: str                    # Destination entity
    type: str                   # Relationship type (e.g., IS_A, USES, MENTIONS)
    weight: float = 1.0         # Edge weight/confidence
    span_id: Optional[str] = None  # Optional: link to source span/shard
    metadata: Dict = field(default_factory=dict)  # Additional properties

    # Bi-temporal fields (from Graphiti research)
    event_time: Optional['datetime'] = None      # When event occurred
    ingestion_time: Optional['datetime'] = None  # When we learned about it
    valid_from: Optional['datetime'] = None      # When edge became valid
    valid_to: Optional['datetime'] = None        # When edge was invalidated (None = still valid)

    def __post_init__(self):
        """Initialize temporal fields with defaults."""
        from datetime import datetime

        now = datetime.now()

        # Default: if not specified, assume event happened when ingested
        if self.ingestion_time is None:
            self.ingestion_time = now

        if self.event_time is None:
            self.event_time = self.ingestion_time

        # Default: edge is valid from when ingested
        if self.valid_from is None:
            self.valid_from = self.ingestion_time

        # valid_to = None means edge is still valid
    
    def to_dict(self) -> Dict:
        """Serialize edge for persistence (includes bi-temporal fields)."""
        result = {
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "weight": self.weight,
            "span_id": self.span_id,
            "metadata": self.metadata
        }

        # Serialize temporal fields (ISO format)
        if self.event_time:
            result["event_time"] = self.event_time.isoformat()
        if self.ingestion_time:
            result["ingestion_time"] = self.ingestion_time.isoformat()
        if self.valid_from:
            result["valid_from"] = self.valid_from.isoformat()
        if self.valid_to:
            result["valid_to"] = self.valid_to.isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'KGEdge':
        """Deserialize edge from storage (includes bi-temporal fields)."""
        from datetime import datetime

        # Parse temporal fields
        event_time = None
        ingestion_time = None
        valid_from = None
        valid_to = None

        if "event_time" in data:
            event_time = datetime.fromisoformat(data["event_time"])
        if "ingestion_time" in data:
            ingestion_time = datetime.fromisoformat(data["ingestion_time"])
        if "valid_from" in data:
            valid_from = datetime.fromisoformat(data["valid_from"])
        if "valid_to" in data:
            valid_to = datetime.fromisoformat(data["valid_to"])

        return cls(
            src=data["src"],
            dst=data["dst"],
            type=data["type"],
            weight=data.get("weight", 1.0),
            span_id=data.get("span_id"),
            metadata=data.get("metadata", {}),
            event_time=event_time,
            ingestion_time=ingestion_time,
            valid_from=valid_from,
            valid_to=valid_to
        )

    def is_valid_at(self, timestamp: 'datetime') -> bool:
        """
        Check if edge is valid at given timestamp.

        Use case: Point-in-time queries ("What did we know on Oct 12?")
        """
        # Edge is valid if:
        # 1. valid_from <= timestamp
        # 2. valid_to is None OR valid_to > timestamp
        if self.valid_from and timestamp < self.valid_from:
            return False

        if self.valid_to and timestamp >= self.valid_to:
            return False

        return True

    def invalidate(self, timestamp: Optional['datetime'] = None) -> None:
        """
        Mark edge as invalid (for temporal edge invalidation).

        From Graphiti research: Instead of deleting edges when info changes,
        we invalidate old edges and add new ones.

        Args:
            timestamp: When edge became invalid (default: now)
        """
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now()

        self.valid_to = timestamp


# ============================================================================
# Protocol
# ============================================================================

class KGStore(Protocol):
    """Protocol for knowledge graph implementations."""
    
    def add_edge(self, edge: KGEdge) -> None:
        """Add an edge to the graph."""
        ...
    
    def subgraph_for_entities(self, entities: List[str]) -> nx.MultiDiGraph:
        """Get subgraph containing entities and their neighborhoods."""
        ...
    
    def get_neighbors(self, entity: str, direction: str = "both") -> List[str]:
        """Get neighboring entities."""
        ...


# ============================================================================
# Knowledge Graph Implementation
# ============================================================================

class KG:
    """
    Knowledge graph using NetworkX MultiDiGraph.

    Features:
    - Typed, weighted edges
    - Multi-edges (multiple relationships between same entities)
    - Efficient neighborhood queries
    - Subgraph extraction
    - Persistence to/from disk
    - **Hard limits with eviction** (Phase 1: Bounded Growth - December 2025)

    Use Cases:
    - Entity relationship tracking
    - Context expansion (find related entities)
    - Reasoning over structured knowledge
    - Spectral analysis of knowledge structure

    Bounded Growth (December 2025):
    - Hard limits on max_nodes and max_edges
    - 4 eviction strategies: LRU, LFU, importance-weighted, scope-aware
    - Automatic eviction when limits reached
    - Node access tracking for intelligent eviction
    """

    def __init__(self, config=None):
        """
        Initialize knowledge graph.

        Args:
            config: Optional Config object with KG limit settings.
                    If None, uses unlimited defaults (backward compatible).
        """
        self.G = nx.MultiDiGraph()
        self._entity_index: Dict[str, Set[str]] = {}  # Fast neighbor lookup

        # =====================================================================
        # Bounded Growth Configuration (Phase 1 - December 2025)
        # =====================================================================
        self._config = config

        # Hard limits (0 = unlimited for backward compatibility)
        self._max_nodes = getattr(config, 'kg_max_nodes', 0) if config else 0
        self._max_edges = getattr(config, 'kg_max_edges', 0) if config else 0
        self._hard_limit_enforce = getattr(config, 'kg_hard_limit_enforce', True) if config else False
        self._hard_limit_fail_mode = getattr(config, 'kg_hard_limit_fail_mode', 'evict') if config else 'evict'

        # Growth monitoring
        self._growth_alert_threshold = getattr(config, 'kg_growth_alert_threshold', 0.8) if config else 0.8
        self._enable_growth_monitoring = getattr(config, 'kg_enable_growth_monitoring', True) if config else False

        # Eviction configuration
        self._eviction_strategy = EvictionStrategy(
            getattr(config, 'kg_eviction_strategy', 'importance_weighted') if config else 'importance_weighted'
        )
        self._eviction_batch_size = getattr(config, 'kg_eviction_batch_size', 1000) if config else 1000
        self._eviction_target_ratio = getattr(config, 'kg_eviction_target_ratio', 0.75) if config else 0.75
        self._eviction_min_age_hours = getattr(config, 'kg_eviction_min_age_hours', 24) if config else 24
        self._eviction_preserve_permanent = getattr(config, 'kg_eviction_preserve_permanent', True) if config else True

        # Importance scoring weights
        self._importance_weights = {
            'recency': getattr(config, 'kg_importance_recency_weight', 0.25) if config else 0.25,
            'access': getattr(config, 'kg_importance_access_weight', 0.20) if config else 0.20,
            'centrality': getattr(config, 'kg_importance_centrality_weight', 0.25) if config else 0.25,
            'connection': getattr(config, 'kg_importance_connection_weight', 0.15) if config else 0.15,
            'lifecycle': getattr(config, 'kg_importance_lifecycle_weight', 0.15) if config else 0.15,
        }

        # Node access tracking (for eviction scoring)
        self._node_stats: Dict[str, NodeAccessStats] = {}

        # Eviction history (for monitoring)
        self._eviction_history: List[EvictionResult] = []
        self._total_evictions = 0

        # Logger
        self._logger = logging.getLogger(__name__)

    # =========================================================================
    # Public Configuration API (Phase 1: Bounded Growth - December 2025)
    # =========================================================================

    def configure_limits(
        self,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        eviction_strategy: Optional[EvictionStrategy] = None,
        enforce: bool = True,
    ) -> 'KG':
        """
        Configure bounded growth limits (fluent API).

        This is the recommended way to set KG limits instead of
        accessing private attributes directly.

        Args:
            max_nodes: Maximum number of nodes (None = unchanged)
            max_edges: Maximum number of edges (None = unchanged)
            eviction_strategy: Strategy for eviction when limits reached
            enforce: Whether to enforce hard limits (triggers eviction)

        Returns:
            self (for method chaining)

        Example:
            kg.configure_limits(
                max_nodes=10000,
                max_edges=50000,
                eviction_strategy=EvictionStrategy.LRU,
                enforce=True
            )
        """
        if max_nodes is not None:
            self._max_nodes = max_nodes
        if max_edges is not None:
            self._max_edges = max_edges
        if eviction_strategy is not None:
            self._eviction_strategy = eviction_strategy
        self._hard_limit_enforce = enforce

        self._logger.info(
            f"KG limits configured: max_nodes={self._max_nodes}, "
            f"max_edges={self._max_edges}, strategy={self._eviction_strategy.value}, "
            f"enforce={self._hard_limit_enforce}"
        )

        return self

    @property
    def limits(self) -> Dict[str, Any]:
        """
        Get current limit configuration (read-only).

        Returns:
            Dict with max_nodes, max_edges, strategy, enforce, current usage
        """
        return {
            'max_nodes': self._max_nodes,
            'max_edges': self._max_edges,
            'eviction_strategy': self._eviction_strategy.value,
            'enforce': self._hard_limit_enforce,
            'current_nodes': self.G.number_of_nodes(),
            'current_edges': self.G.number_of_edges(),
            'node_usage_pct': (
                self.G.number_of_nodes() / self._max_nodes * 100
                if self._max_nodes > 0 else 0
            ),
            'edge_usage_pct': (
                self.G.number_of_edges() / self._max_edges * 100
                if self._max_edges > 0 else 0
            ),
        }

    def add_edge(self, edge: KGEdge) -> bool:
        """
        Add an edge to the knowledge graph.

        Automatically creates nodes if they don't exist.
        Supports multiple edges between the same entities.

        **Bounded Growth (December 2025):**
        - Checks hard limits before adding
        - Triggers eviction if limits reached (configurable)
        - Tracks node access statistics

        Args:
            edge: KGEdge to add

        Returns:
            True if edge was added, False if rejected due to limits
        """
        # =====================================================================
        # Check hard limits (Phase 1: Bounded Growth)
        # =====================================================================
        if self._hard_limit_enforce and self._max_nodes > 0:
            # Check if we're at node limit (only matters for new nodes)
            new_nodes_needed = 0
            if edge.src not in self.G:
                new_nodes_needed += 1
            if edge.dst not in self.G:
                new_nodes_needed += 1

            current_nodes = self.G.number_of_nodes()
            if current_nodes + new_nodes_needed > self._max_nodes:
                if self._hard_limit_fail_mode == 'evict':
                    # Trigger eviction
                    self._logger.info(
                        f"KG node limit reached ({current_nodes}/{self._max_nodes}). "
                        f"Triggering {self._eviction_strategy.value} eviction..."
                    )
                    eviction_result = self.evict_to_capacity()
                    if eviction_result.nodes_evicted == 0:
                        self._logger.warning("Eviction failed to free nodes. Rejecting edge.")
                        return False
                elif self._hard_limit_fail_mode == 'reject':
                    self._logger.warning(f"KG node limit reached. Rejecting edge: {edge.src} → {edge.dst}")
                    return False
                elif self._hard_limit_fail_mode == 'warn':
                    self._logger.warning(f"KG node limit exceeded ({current_nodes}/{self._max_nodes}). Allowing edge.")

        if self._hard_limit_enforce and self._max_edges > 0:
            current_edges = self.G.number_of_edges()
            if current_edges >= self._max_edges:
                if self._hard_limit_fail_mode == 'evict':
                    self._logger.info(
                        f"KG edge limit reached ({current_edges}/{self._max_edges}). "
                        f"Triggering eviction..."
                    )
                    eviction_result = self.evict_to_capacity()
                    if eviction_result.edges_removed == 0:
                        self._logger.warning("Eviction failed to free edges. Rejecting edge.")
                        return False
                elif self._hard_limit_fail_mode == 'reject':
                    self._logger.warning(f"KG edge limit reached. Rejecting edge: {edge.src} → {edge.dst}")
                    return False
                elif self._hard_limit_fail_mode == 'warn':
                    self._logger.warning(f"KG edge limit exceeded ({current_edges}/{self._max_edges}). Allowing edge.")

        # Growth monitoring alerts
        if self._enable_growth_monitoring and self._max_nodes > 0:
            current_nodes = self.G.number_of_nodes()
            usage_ratio = current_nodes / self._max_nodes
            if usage_ratio >= self._growth_alert_threshold:
                self._logger.warning(
                    f"KG growth alert: {usage_ratio:.1%} of node limit "
                    f"({current_nodes}/{self._max_nodes})"
                )

        # =====================================================================
        # Create nodes and track access stats
        # =====================================================================
        if edge.src not in self.G:
            self.G.add_node(edge.src)
            # Initialize access stats for new node
            lifecycle = edge.metadata.get('lifecycle_scope', LifecycleScope.TEMPORARY)
            if isinstance(lifecycle, str):
                try:
                    lifecycle = LifecycleScope(lifecycle)
                except ValueError:
                    lifecycle = LifecycleScope.TEMPORARY
            self._node_stats[edge.src] = NodeAccessStats(lifecycle_scope=lifecycle)

        if edge.dst not in self.G:
            self.G.add_node(edge.dst)
            lifecycle = edge.metadata.get('lifecycle_scope', LifecycleScope.TEMPORARY)
            if isinstance(lifecycle, str):
                try:
                    lifecycle = LifecycleScope(lifecycle)
                except ValueError:
                    lifecycle = LifecycleScope.TEMPORARY
            self._node_stats[edge.dst] = NodeAccessStats(lifecycle_scope=lifecycle)

        # Record access for both nodes
        if edge.src in self._node_stats:
            self._node_stats[edge.src].record_access()
        if edge.dst in self._node_stats:
            self._node_stats[edge.dst].record_access()

        # =====================================================================
        # Add edge with all metadata
        # =====================================================================
        # Note: NetworkX uses 'key' as edge identifier in MultiDiGraph,
        # so we need to handle it specially if present in metadata
        edge_attrs = {
            "type": edge.type,
            "weight": edge.weight,
            "span_id": edge.span_id,
        }
        # Add metadata, but rename 'key' to avoid collision with NetworkX's key parameter
        for k, v in edge.metadata.items():
            if k == "key":
                edge_attrs["_key"] = v  # Store as _key to avoid NetworkX collision
            else:
                edge_attrs[k] = v

        self.G.add_edge(edge.src, edge.dst, **edge_attrs)

        # Update entity index for fast lookups
        if edge.src not in self._entity_index:
            self._entity_index[edge.src] = set()
        if edge.dst not in self._entity_index:
            self._entity_index[edge.dst] = set()

        self._entity_index[edge.src].add(edge.dst)
        self._entity_index[edge.dst].add(edge.src)

        return True
    
    def add_edges(self, edges: List[KGEdge]) -> None:
        """Bulk add edges (more efficient than individual adds)."""
        for edge in edges:
            self.add_edge(edge)
    
    def connect_entity_to_time(
        self,
        entity: str,
        timestamp: TimeInput,
        *,
        edge_type: str = "IN_TIME",
        weight: float = 1.0,
    ) -> str:
        """Attach an entity node to a coarse-grained time thread.

        The Neo4j migrations stored under ``archive/`` group events by
        day-part buckets (e.g. ``2024-01-31-evening``).  When ingesting
        events into the in-memory graph we mirror that behaviour by creating
        (or reusing) a dedicated node for the bucket and linking the entity
        with an ``IN_TIME`` edge.

        Args:
            entity: Name of the entity/event node to attach.
            timestamp: Datetime/ISO string/epoch seconds identifying the event
                time.
            edge_type: Relationship label to use for the connection.
            weight: Optional weight applied to the created edge.

        Returns:
            The identifier of the time thread node that was connected.
        """
        dt = to_utc_datetime(timestamp)
        bucket = time_bucket(dt)
        thread_id = f"time::{bucket}"

        if thread_id not in self.G:
            self.G.add_node(
                thread_id,
                kind="time_thread",
                bucket=bucket,
            )

        if entity not in self.G:
            self.G.add_node(entity)

        self.G.add_edge(
            entity,
            thread_id,
            type=edge_type,
            weight=weight,
            bucket=bucket,
            timestamp=dt.isoformat(),
        )

        self._entity_index.setdefault(entity, set()).add(thread_id)
        self._entity_index.setdefault(thread_id, set()).add(entity)

        return thread_id

    def get_neighbors(
        self,
        entity: str,
        direction: str = "both",
        max_hops: int = 1
    ) -> Set[str]:
        """
        Get neighboring entities.
        
        Args:
            entity: Starting entity
            direction: "out" (successors), "in" (predecessors), or "both"
            max_hops: Maximum number of hops (1 = direct neighbors)
            
        Returns:
            Set of neighbor entity names
        """
        if entity not in self.G:
            return set()
        
        neighbors = set()
        
        if max_hops == 1:
            # Fast path: direct neighbors only
            if direction in ("out", "both"):
                neighbors.update(self.G.successors(entity))
            if direction in ("in", "both"):
                neighbors.update(self.G.predecessors(entity))
        else:
            # Multi-hop traversal (BFS)
            visited = {entity}
            current_level = {entity}
            
            for _ in range(max_hops):
                next_level = set()
                for node in current_level:
                    if direction in ("out", "both"):
                        next_level.update(n for n in self.G.successors(node) if n not in visited)
                    if direction in ("in", "both"):
                        next_level.update(n for n in self.G.predecessors(node) if n not in visited)
                
                neighbors.update(next_level)
                visited.update(next_level)
                current_level = next_level
                
                if not current_level:
                    break
        
        return neighbors
    
    def subgraph_for_entities(
        self,
        entities: List[str],
        expand: bool = True,
        max_hops: int = 1
    ) -> nx.MultiDiGraph:
        """
        Extract subgraph containing entities and their neighborhoods.
        
        This is used to provide context for queries - we find all relevant
        knowledge connected to the entities mentioned in the query.
        
        Args:
            entities: List of entity names to include
            expand: If True, include neighbors (1-hop expansion)
            max_hops: How many hops to expand (if expand=True)
            
        Returns:
            MultiDiGraph containing the subgraph
        """
        nodes = set()
        
        # Add requested entities
        for entity in entities:
            if entity in self.G:
                nodes.add(entity)
                
                # Optionally expand to neighbors
                if expand:
                    neighbors = self.get_neighbors(entity, direction="both", max_hops=max_hops)
                    nodes.update(neighbors)
        
        # Extract subgraph
        if not nodes:
            return nx.MultiDiGraph()
        
        return self.G.subgraph(nodes).copy()
    
    def get_edge_types(self, src: str, dst: str) -> List[str]:
        """
        Get all edge types between two entities.
        
        Returns:
            List of edge types (may have duplicates if multi-edge)
        """
        if not self.G.has_edge(src, dst):
            return []

        # For MultiDiGraph, get_edge_data returns dict of {key: edge_data}
        edge_dict = self.G.get_edge_data(src, dst)
        if edge_dict is None:
            return []
        return [data.get("type", "unknown") for data in edge_dict.values()]
    
    def get_paths(
        self,
        src: str,
        dst: str,
        max_length: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two entities.
        
        Useful for reasoning: "How is X related to Y?"
        
        Args:
            src: Source entity
            dst: Destination entity
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of entity names)
        """
        if src not in self.G or dst not in self.G:
            return []
        
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self.G,
                source=src,
                target=dst,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_related_by_type(
        self,
        entity: str,
        edge_type: str,
        direction: str = "out"
    ) -> List[str]:
        """
        Get entities related by a specific edge type.
        
        Examples:
        - get_related_by_type("Python", "IS_A", "out") → ["programming_language"]
        - get_related_by_type("attention", "USES", "in") → ["transformer", "BERT"]
        
        Args:
            entity: Starting entity
            edge_type: Relationship type to follow
            direction: "out" or "in"
            
        Returns:
            List of related entity names
        """
        if entity not in self.G:
            return []
        
        related = []
        
        if direction == "out":
            for _, dst, data in self.G.out_edges(entity, data=True):
                if data.get("type") == edge_type:
                    related.append(dst)
        elif direction == "in":
            for src, _, data in self.G.in_edges(entity, data=True):
                if data.get("type") == edge_type:
                    related.append(src)
        
        return related
    
    def stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "avg_degree": sum(dict(self.G.degree()).values()) / max(1, self.G.number_of_nodes()),
            "is_connected": nx.is_weakly_connected(self.G) if self.G.number_of_nodes() > 0 else False,
        }

    # ========================================================================
    # Eviction Methods (Phase 1: Bounded Growth - December 2025)
    # ========================================================================

    def evict_to_capacity(self) -> EvictionResult:
        """
        Evict nodes to bring KG back within capacity limits.

        Uses the configured eviction strategy (LRU, LFU, importance-weighted, or scope-aware)
        to select which nodes to remove. Evicts down to target ratio (default: 75% of max).

        Returns:
            EvictionResult with eviction statistics
        """
        start_time = time.time()
        nodes_before = self.G.number_of_nodes()
        edges_before = self.G.number_of_edges()

        # Calculate target (evict to 75% of max by default)
        if self._max_nodes > 0:
            target_nodes = int(self._max_nodes * self._eviction_target_ratio)
            nodes_to_evict = max(0, nodes_before - target_nodes)
        else:
            nodes_to_evict = 0

        if nodes_to_evict == 0:
            return EvictionResult(
                nodes_evicted=0,
                edges_removed=0,
                nodes_before=nodes_before,
                nodes_after=nodes_before,
                edges_before=edges_before,
                edges_after=edges_before,
                strategy_used=self._eviction_strategy,
                eviction_time_ms=0.0,
                evicted_node_ids=[]
            )

        # Cap at batch size
        nodes_to_evict = min(nodes_to_evict, self._eviction_batch_size)

        # Select nodes to evict based on strategy
        if self._eviction_strategy == EvictionStrategy.LRU:
            nodes_to_remove = self._select_lru_nodes(nodes_to_evict)
        elif self._eviction_strategy == EvictionStrategy.LFU:
            nodes_to_remove = self._select_lfu_nodes(nodes_to_evict)
        elif self._eviction_strategy == EvictionStrategy.IMPORTANCE_WEIGHTED:
            nodes_to_remove = self._select_importance_weighted_nodes(nodes_to_evict)
        elif self._eviction_strategy == EvictionStrategy.SCOPE_AWARE:
            nodes_to_remove = self._select_scope_aware_nodes(nodes_to_evict)
        else:
            # Fallback to LRU
            nodes_to_remove = self._select_lru_nodes(nodes_to_evict)

        # Remove selected nodes (NetworkX removes associated edges automatically)
        evicted_ids = []
        for node_id in nodes_to_remove:
            if node_id in self.G:
                self.G.remove_node(node_id)
                evicted_ids.append(node_id)
                # Clean up tracking
                self._node_stats.pop(node_id, None)
                self._entity_index.pop(node_id, None)

        # Update entity index for remaining nodes
        for node_id in list(self._entity_index.keys()):
            if node_id in self._entity_index:
                self._entity_index[node_id] -= set(evicted_ids)

        nodes_after = self.G.number_of_nodes()
        edges_after = self.G.number_of_edges()
        elapsed_ms = (time.time() - start_time) * 1000

        result = EvictionResult(
            nodes_evicted=len(evicted_ids),
            edges_removed=edges_before - edges_after,
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            edges_before=edges_before,
            edges_after=edges_after,
            strategy_used=self._eviction_strategy,
            eviction_time_ms=elapsed_ms,
            evicted_node_ids=evicted_ids
        )

        # Track history
        self._eviction_history.append(result)
        self._total_evictions += result.nodes_evicted

        self._logger.info(
            f"Eviction complete: {result.nodes_evicted} nodes, {result.edges_removed} edges "
            f"removed in {elapsed_ms:.1f}ms ({result.nodes_before} → {result.nodes_after} nodes)"
        )

        return result

    def _select_lru_nodes(self, count: int) -> List[str]:
        """
        Select nodes by Least Recently Used (evict nodes accessed longest ago).

        Args:
            count: Number of nodes to select

        Returns:
            List of node IDs to evict
        """
        now = datetime.now()
        min_age = timedelta(hours=self._eviction_min_age_hours)

        candidates = []
        for node_id in self.G.nodes():
            stats = self._node_stats.get(node_id)
            if stats is None:
                # No stats = very old node, high priority for eviction
                candidates.append((node_id, datetime.min))
                continue

            # Skip permanent nodes
            if self._eviction_preserve_permanent and stats.lifecycle_scope == LifecycleScope.PERMANENT:
                continue

            # Skip nodes younger than min age
            age = now - stats.created_at
            if age < min_age:
                continue

            candidates.append((node_id, stats.last_accessed))

        # Sort by last_accessed (oldest first)
        candidates.sort(key=lambda x: x[1])

        return [node_id for node_id, _ in candidates[:count]]

    def _select_lfu_nodes(self, count: int) -> List[str]:
        """
        Select nodes by Least Frequently Used (evict nodes with fewest accesses).

        Args:
            count: Number of nodes to select

        Returns:
            List of node IDs to evict
        """
        now = datetime.now()
        min_age = timedelta(hours=self._eviction_min_age_hours)

        candidates = []
        for node_id in self.G.nodes():
            stats = self._node_stats.get(node_id)
            if stats is None:
                # No stats = no accesses tracked, high priority for eviction
                candidates.append((node_id, 0))
                continue

            # Skip permanent nodes
            if self._eviction_preserve_permanent and stats.lifecycle_scope == LifecycleScope.PERMANENT:
                continue

            # Skip nodes younger than min age
            age = now - stats.created_at
            if age < min_age:
                continue

            candidates.append((node_id, stats.access_count))

        # Sort by access_count (lowest first)
        candidates.sort(key=lambda x: x[1])

        return [node_id for node_id, _ in candidates[:count]]

    def _select_importance_weighted_nodes(self, count: int) -> List[str]:
        """
        Select nodes by multi-signal importance score.

        Importance = weighted combination of:
        - Recency: Time since last access (lower = more important)
        - Access frequency: Number of accesses (higher = more important)
        - Centrality: Graph degree (higher = more important)
        - Connections: Number of neighbors (higher = more important)
        - Lifecycle: PERMANENT > SEMANTIC > WORKING > TEMPORARY > EPHEMERAL

        Args:
            count: Number of nodes to select

        Returns:
            List of node IDs to evict (lowest importance first)
        """
        now = datetime.now()
        min_age = timedelta(hours=self._eviction_min_age_hours)

        # Calculate importance for each node
        scored_nodes = []
        max_access = 1
        max_degree = 1

        # First pass: find max values for normalization
        for node_id in self.G.nodes():
            stats = self._node_stats.get(node_id)
            if stats:
                max_access = max(max_access, stats.access_count)
            degree = self.G.degree(node_id)
            max_degree = max(max_degree, degree)

        # Lifecycle scores (higher = more protected)
        lifecycle_scores = {
            LifecycleScope.PERMANENT: 1.0,
            LifecycleScope.SEMANTIC: 0.8,
            LifecycleScope.WORKING: 0.5,
            LifecycleScope.TEMPORARY: 0.2,
            LifecycleScope.EPHEMERAL: 0.0,
        }

        # Second pass: calculate importance scores
        for node_id in self.G.nodes():
            stats = self._node_stats.get(node_id)

            # Skip permanent nodes
            if stats and self._eviction_preserve_permanent and stats.lifecycle_scope == LifecycleScope.PERMANENT:
                continue

            # Skip nodes younger than min age
            if stats:
                age = now - stats.created_at
                if age < min_age:
                    continue

            # Calculate component scores
            if stats:
                # Recency: hours since last access (normalized, inverted: more recent = higher)
                hours_since = (now - stats.last_accessed).total_seconds() / 3600
                recency_score = max(0, 1.0 - (hours_since / 168))  # 168 hours = 1 week

                # Access: normalized access count
                access_score = stats.access_count / max_access if max_access > 0 else 0

                # Lifecycle: from lookup
                lifecycle_score = lifecycle_scores.get(stats.lifecycle_scope, 0.2)
            else:
                recency_score = 0.0
                access_score = 0.0
                lifecycle_score = 0.0

            # Centrality: normalized degree
            degree = self.G.degree(node_id)
            centrality_score = degree / max_degree if max_degree > 0 else 0

            # Connection: number of neighbors (use entity index if available)
            neighbors = len(self._entity_index.get(node_id, set()))
            connection_score = min(1.0, neighbors / 10)  # Normalize to 10 neighbors

            # Weighted importance
            importance = (
                self._importance_weights['recency'] * recency_score +
                self._importance_weights['access'] * access_score +
                self._importance_weights['centrality'] * centrality_score +
                self._importance_weights['connection'] * connection_score +
                self._importance_weights['lifecycle'] * lifecycle_score
            )

            scored_nodes.append((node_id, importance))

        # Sort by importance (lowest first - these get evicted)
        scored_nodes.sort(key=lambda x: x[1])

        return [node_id for node_id, _ in scored_nodes[:count]]

    def _select_scope_aware_nodes(self, count: int) -> List[str]:
        """
        Select nodes by lifecycle scope (evict EPHEMERAL → TEMPORARY → WORKING first).

        Within each scope, uses LRU as tiebreaker.

        Args:
            count: Number of nodes to select

        Returns:
            List of node IDs to evict
        """
        now = datetime.now()
        min_age = timedelta(hours=self._eviction_min_age_hours)

        # Group nodes by lifecycle scope
        scope_order = [
            LifecycleScope.EPHEMERAL,
            LifecycleScope.TEMPORARY,
            LifecycleScope.WORKING,
            LifecycleScope.SEMANTIC,
            # PERMANENT never evicted
        ]

        scope_buckets: Dict[LifecycleScope, List[Tuple[str, datetime]]] = {
            scope: [] for scope in scope_order
        }

        for node_id in self.G.nodes():
            stats = self._node_stats.get(node_id)

            if stats is None:
                # No stats = treat as TEMPORARY
                scope_buckets[LifecycleScope.TEMPORARY].append((node_id, datetime.min))
                continue

            # Skip permanent nodes
            if self._eviction_preserve_permanent and stats.lifecycle_scope == LifecycleScope.PERMANENT:
                continue

            # Skip nodes younger than min age
            age = now - stats.created_at
            if age < min_age:
                continue

            if stats.lifecycle_scope in scope_buckets:
                scope_buckets[stats.lifecycle_scope].append((node_id, stats.last_accessed))

        # Collect nodes in scope order, using LRU within each scope
        result = []
        for scope in scope_order:
            bucket = scope_buckets[scope]
            # Sort by last_accessed within scope (oldest first)
            bucket.sort(key=lambda x: x[1])

            for node_id, _ in bucket:
                result.append(node_id)
                if len(result) >= count:
                    return result

        return result

    def set_node_lifecycle(self, node_id: str, scope: LifecycleScope) -> bool:
        """
        Set lifecycle scope for a node.

        Args:
            node_id: Node to update
            scope: New lifecycle scope

        Returns:
            True if node exists and was updated
        """
        if node_id not in self.G:
            return False

        if node_id not in self._node_stats:
            self._node_stats[node_id] = NodeAccessStats(lifecycle_scope=scope)
        else:
            self._node_stats[node_id].lifecycle_scope = scope

        return True

    def pin_node(self, node_id: str) -> bool:
        """
        Pin a node to prevent eviction (sets lifecycle to PERMANENT).

        Args:
            node_id: Node to pin

        Returns:
            True if node exists and was pinned
        """
        return self.set_node_lifecycle(node_id, LifecycleScope.PERMANENT)

    def get_eviction_stats(self) -> Dict:
        """
        Get eviction statistics.

        Returns:
            Dict with eviction history and totals
        """
        return {
            "total_evictions": self._total_evictions,
            "eviction_count": len(self._eviction_history),
            "recent_evictions": [
                {
                    "nodes_evicted": e.nodes_evicted,
                    "edges_removed": e.edges_removed,
                    "strategy": e.strategy_used.value,
                    "time_ms": e.eviction_time_ms,
                }
                for e in self._eviction_history[-10:]  # Last 10
            ],
            "current_usage": {
                "nodes": self.G.number_of_nodes(),
                "edges": self.G.number_of_edges(),
                "max_nodes": self._max_nodes,
                "max_edges": self._max_edges,
                "node_usage_ratio": self.G.number_of_nodes() / self._max_nodes if self._max_nodes > 0 else 0,
                "edge_usage_ratio": self.G.number_of_edges() / self._max_edges if self._max_edges > 0 else 0,
            }
        }

    def record_access(self, node_id: str) -> None:
        """
        Record an access to a node (updates last_accessed and access_count).

        Call this when a node is used in retrieval or reasoning.

        Args:
            node_id: Node that was accessed
        """
        if node_id in self._node_stats:
            self._node_stats[node_id].record_access()
        elif node_id in self.G:
            # Create stats if missing
            self._node_stats[node_id] = NodeAccessStats()
            self._node_stats[node_id].record_access()

    # ========================================================================
    # Temporal Methods - Bi-Temporal Model & Edge Invalidation
    # ========================================================================

    def invalidate_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        timestamp: Optional['datetime'] = None
    ) -> bool:
        """
        Invalidate edge(s) matching criteria (for temporal edge invalidation).

        From Graphiti research: Instead of deleting edges when info changes,
        we mark old edges as invalid and add new ones.

        Example:
            # Original: "Blake uses Python"
            kg.add_edge(KGEdge("Blake", "Python", "USES"))

            # Update: "Blake uses Rust"
            kg.invalidate_edge("Blake", "Python", "USES")  # Mark old edge invalid
            kg.add_edge(KGEdge("Blake", "Rust", "USES"))  # Add new edge

        Args:
            src: Source entity
            dst: Destination entity
            edge_type: Edge type to invalidate
            timestamp: When edge became invalid (default: now)

        Returns:
            True if any edges were invalidated
        """
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now()

        if not self.G.has_edge(src, dst):
            return False

        invalidated = False

        # Find matching edges and add valid_to timestamp
        for u, v, key, data in list(self.G.edges(src, dst, keys=True, data=True)):
            if data.get("type") == edge_type:
                # Only invalidate if not already invalid
                if data.get("valid_to") is None:
                    data["valid_to"] = timestamp.isoformat()
                    invalidated = True

        return invalidated

    def get_valid_edges(
        self,
        src: Optional[str] = None,
        dst: Optional[str] = None,
        timestamp: Optional['datetime'] = None
    ) -> List[KGEdge]:
        """
        Get edges valid at given timestamp.

        Use case: Point-in-time queries ("What did we know on Oct 12?")

        Args:
            src: Filter by source entity (None = all)
            dst: Filter by destination entity (None = all)
            timestamp: Point in time to query (default: now)

        Returns:
            List of KGEdge objects valid at timestamp
        """
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now()

        valid_edges = []

        # Determine which edges to check
        if src and dst:
            edges = self.G.edges(src, dst, data=True)
        elif src:
            edges = self.G.out_edges(src, data=True)
        elif dst:
            edges = self.G.in_edges(dst, data=True)
        else:
            edges = self.G.edges(data=True)

        # Filter to valid edges
        for u, v, data in edges:
            # Parse temporal fields
            valid_from_str = data.get("valid_from")
            valid_to_str = data.get("valid_to")

            # Check if valid at timestamp
            if valid_from_str:
                valid_from = datetime.fromisoformat(valid_from_str)
                if timestamp < valid_from:
                    continue  # Not yet valid

            if valid_to_str:
                valid_to = datetime.fromisoformat(valid_to_str)
                if timestamp >= valid_to:
                    continue  # No longer valid

            # Edge is valid - reconstruct KGEdge
            edge = KGEdge(
                src=u,
                dst=v,
                type=data.get("type", "unknown"),
                weight=data.get("weight", 1.0),
                span_id=data.get("span_id"),
                metadata={k: v for k, v in data.items()
                         if k not in ["type", "weight", "span_id", "event_time", "ingestion_time", "valid_from", "valid_to"]},
                event_time=datetime.fromisoformat(data["event_time"]) if "event_time" in data else None,
                ingestion_time=datetime.fromisoformat(data["ingestion_time"]) if "ingestion_time" in data else None,
                valid_from=datetime.fromisoformat(data["valid_from"]) if "valid_from" in data else None,
                valid_to=datetime.fromisoformat(data["valid_to"]) if "valid_to" in data else None
            )

            valid_edges.append(edge)

        return valid_edges

    def get_edge_history(
        self,
        src: str,
        dst: str,
        edge_type: Optional[str] = None
    ) -> List[KGEdge]:
        """
        Get complete history of edges between entities (including invalidated).

        Use case: Audit trail, understanding how relationship evolved over time.

        Example:
            # Get history of Blake's language preferences
            history = kg.get_edge_history("Blake", "Python", "USES")
            # Returns:
            # [
            #   KGEdge(valid_from=2024-01-01, valid_to=2024-06-01),  # Old: used Python
            #   KGEdge(valid_from=2024-06-01, valid_to=None)         # Current: still uses Python
            # ]

        Args:
            src: Source entity
            dst: Destination entity
            edge_type: Optional filter by edge type

        Returns:
            List of all KGEdge objects (past and present), sorted by valid_from
        """
        from datetime import datetime

        if not self.G.has_edge(src, dst):
            return []

        all_edges = []

        for u, v, data in self.G.edges(src, dst, data=True):
            # Filter by type if specified
            if edge_type and data.get("type") != edge_type:
                continue

            # Reconstruct KGEdge
            edge = KGEdge(
                src=u,
                dst=v,
                type=data.get("type", "unknown"),
                weight=data.get("weight", 1.0),
                span_id=data.get("span_id"),
                metadata={k: v for k, v in data.items()
                         if k not in ["type", "weight", "span_id", "event_time", "ingestion_time", "valid_from", "valid_to"]},
                event_time=datetime.fromisoformat(data["event_time"]) if "event_time" in data else None,
                ingestion_time=datetime.fromisoformat(data["ingestion_time"]) if "ingestion_time" in data else None,
                valid_from=datetime.fromisoformat(data["valid_from"]) if "valid_from" in data else None,
                valid_to=datetime.fromisoformat(data["valid_to"]) if "valid_to" in data else None
            )

            all_edges.append(edge)

        # Sort by valid_from
        all_edges.sort(key=lambda e: e.valid_from if e.valid_from else datetime.min)

        return all_edges

    # ========================================================================
    # Spectral Methods - Diffusion Maps & Clustering
    # ========================================================================

    def compute_diffusion_map(
        self,
        n_dims: int = 32,
        t: float = 1.0,
        cache: bool = True
    ) -> np.ndarray:
        """
        Compute diffusion map for dimensionality reduction.

        Diffusion maps reveal intrinsic geometry via random walk distances.
        This is useful for:
        - Clustering similar entities
        - Dimensionality reduction for large graphs
        - Semantic similarity beyond local structure

        Args:
            n_dims: Target embedding dimension
            t: Diffusion time (larger = more global structure)
            cache: If True, cache result for repeated queries

        Returns:
            Embedding matrix (n_nodes × n_dims)
        """
        # Check cache
        cache_key = f'diffusion_{n_dims}_{t}'
        if cache and hasattr(self, '_diffusion_cache') and cache_key in self._diffusion_cache:
            return self._diffusion_cache[cache_key]

        if self.G.number_of_nodes() <= n_dims:
            # Graph too small for dimensionality reduction
            import warnings
            warnings.warn(f"Graph has {self.G.number_of_nodes()} nodes, less than n_dims={n_dims}. Returning identity.")
            return np.eye(self.G.number_of_nodes())

        try:
            from HoloLoom.warp.spectral_methods import GraphLaplacian, DiffusionMap, LaplacianType

            # Compute diffusion map
            laplacian = GraphLaplacian(self, laplacian_type=LaplacianType.RANDOM_WALK)
            diffusion = DiffusionMap(
                laplacian=laplacian,
                t=t,
                n_components=min(n_dims, self.G.number_of_nodes() - 1)
            )

            embedding = diffusion.compute_embedding()

            # Cache result
            if cache:
                if not hasattr(self, '_diffusion_cache'):
                    self._diffusion_cache = {}
                self._diffusion_cache[cache_key] = embedding

            return embedding

        except ImportError:
            import warnings
            warnings.warn("scipy not available - diffusion maps require scipy. Returning identity.")
            return np.eye(min(n_dims, self.G.number_of_nodes()))

    def get_diffusion_coordinates(self, entity: str, n_dims: int = 32) -> Optional[np.ndarray]:
        """
        Get diffusion map coordinates for a specific entity.

        Args:
            entity: Entity name
            n_dims: Diffusion embedding dimension

        Returns:
            Coordinates in diffusion space (length n_dims), or None if entity not found
        """
        if entity not in self.G:
            return None

        # Compute diffusion map (uses cache if available)
        embedding = self.compute_diffusion_map(n_dims=n_dims)

        # Find entity index
        nodes = list(self.G.nodes())
        if entity not in nodes:
            return None

        idx = nodes.index(entity)
        return embedding[idx]

    def spectral_cluster(
        self,
        n_clusters: int,
        method: str = 'spectral'
    ) -> Dict[str, int]:
        """
        Cluster graph nodes using spectral methods.

        Uses the Fiedler vector (2nd smallest eigenvector) to partition
        the graph into communities.

        Args:
            n_clusters: Number of clusters
            method: Clustering method ('spectral' or 'fiedler')

        Returns:
            Dict mapping entity → cluster_id
        """
        if self.G.number_of_nodes() == 0:
            return {}

        try:
            from HoloLoom.warp.spectral_methods import GraphLaplacian, spectral_clustering, LaplacianType

            # Compute spectral clustering
            laplacian = GraphLaplacian(self, laplacian_type=LaplacianType.NORMALIZED)

            if method == 'fiedler' and n_clusters == 2:
                # Simple bisection using Fiedler vector
                _, eigenvectors = laplacian.compute_spectrum(k=2)
                fiedler_vector = eigenvectors[:, 1]

                # Bisect at median
                median = np.median(fiedler_vector)
                labels = (fiedler_vector >= median).astype(int)

            else:
                # Full spectral clustering
                labels = spectral_clustering(laplacian, n_clusters=n_clusters)

            # Map to entity names
            nodes = list(self.G.nodes())
            return {node: int(label) for node, label in zip(nodes, labels)}

        except ImportError as e:
            import warnings
            warnings.warn(f"Spectral clustering requires scipy and sklearn: {e}")
            return {}

    def clear_spectral_cache(self) -> None:
        """Clear cached diffusion maps and spectral computations."""
        if hasattr(self, '_diffusion_cache'):
            self._diffusion_cache.clear()

    # ========================================================================
    # Persistence
    # ========================================================================
    
    def save(self, path: str) -> None:
        """
        Save graph to disk (JSONL format).
        
        Args:
            path: File path to save to
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with p.open('w', encoding='utf-8') as f:
            for src, dst, key, data in self.G.edges(keys=True, data=True):
                edge = KGEdge(
                    src=src,
                    dst=dst,
                    type=data.get("type", "unknown"),
                    weight=data.get("weight", 1.0),
                    span_id=data.get("span_id"),
                    metadata={k: v for k, v in data.items() if k not in ["type", "weight", "span_id"]}
                )
                f.write(json.dumps(edge.to_dict()) + "\n")
    
    @classmethod
    def load(cls, path: str) -> 'KG':
        """
        Load graph from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded KG instance
        """
        kg = cls()
        p = Path(path)
        
        if not p.exists():
            return kg
        
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    edge = KGEdge.from_dict(data)
                    kg.add_edge(edge)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to load edge: {e}")
        
        return kg
    
    def merge(self, other: 'KG') -> None:
        """
        Merge another KG into this one.

        Useful for combining knowledge from multiple sources.
        """
        for src, dst, key, data in other.G.edges(keys=True, data=True):
            edge = KGEdge(
                src=src,
                dst=dst,
                type=data.get("type", "unknown"),
                weight=data.get("weight", 1.0),
                span_id=data.get("span_id"),
                metadata={k: v for k, v in data.items() if k not in ["type", "weight", "span_id"]}
            )
            self.add_edge(edge)

    # ========================================================================
    # MemoryStore Protocol Implementation
    # ========================================================================

    async def store(self, memory, user_id: str = "default") -> str:
        """
        Store a Memory object as a node in the knowledge graph.

        Creates a memory node with full text and metadata, then connects
        it to entity nodes mentioned in the context.

        Args:
            memory: Memory object from protocol
            user_id: User identifier (for multi-tenant support)

        Returns:
            memory_id: The memory's unique identifier
        """
        from datetime import datetime

        # Add memory node
        self.G.add_node(
            memory.id,
            node_type="memory",
            text=memory.text,
            timestamp=memory.timestamp.isoformat() if isinstance(memory.timestamp, datetime) else memory.timestamp,
            user_id=user_id,
            context=memory.context,
            metadata=memory.metadata
        )

        # Connect to entities
        entities = memory.context.get('entities', [])
        for entity in entities:
            edge = KGEdge(
                src=memory.id,
                dst=entity,
                type="MENTIONS",
                weight=1.0,
                span_id=memory.id
            )
            self.add_edge(edge)

        # Connect to time thread
        if hasattr(memory, 'timestamp') and memory.timestamp:
            self.connect_entity_to_time(
                entity=memory.id,
                timestamp=memory.timestamp,
                edge_type="OCCURRED_AT"
            )

        return memory.id

    async def store_many(self, memories: List, user_id: str = "default") -> List[str]:
        """
        Batch store multiple memories.

        Args:
            memories: List of Memory objects
            user_id: User identifier

        Returns:
            List of memory IDs
        """
        ids = []
        for memory in memories:
            memory_id = await self.store(memory, user_id)
            ids.append(memory_id)
        return ids

    async def recall(
        self,
        query,
        limit: int = 5
    ):
        """
        Retrieve memories matching a query using graph traversal.

        Strategy:
        1. Extract entities from query text
        2. Find memories connected to those entities
        3. Rank by number of entity matches
        4. Return top-k memories

        Args:
            query: MemoryQuery object with text field
            limit: Maximum number of memories to return

        Returns:
            RetrievalResult with memories, scores, and metadata
        """
        from HoloLoom.memory.protocol import Memory, RetrievalResult
        from datetime import datetime

        query_text = query.text if hasattr(query, 'text') else str(query)

        # Extract entities from query (simple heuristic)
        query_entities = extract_entities_simple(query_text)

        # Find all memory nodes
        memory_nodes = [
            node for node, data in self.G.nodes(data=True)
            if data.get('node_type') == 'memory'
        ]

        # Score memories by entity overlap
        scored_memories = []

        for mem_id in memory_nodes:
            mem_data = self.G.nodes[mem_id]

            # Get entities this memory mentions
            mem_entities = set()
            for _, dst in self.G.out_edges(mem_id):
                if dst in self.G and self.G.nodes.get(dst, {}).get('node_type') != 'memory':
                    mem_entities.add(dst)

            # Calculate overlap score
            if query_entities:
                overlap = len(set(query_entities) & mem_entities)
                score = overlap / len(query_entities)
            else:
                # No entities extracted - use recency
                score = 0.5

            # Convert node data to Memory object
            memory = Memory(
                id=mem_id,
                text=mem_data.get('text', ''),
                timestamp=datetime.fromisoformat(mem_data['timestamp']) if 'timestamp' in mem_data else datetime.now(),
                context=mem_data.get('context', {}),
                metadata=mem_data.get('metadata', {})
            )

            scored_memories.append((memory, score))

        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        top_memories = scored_memories[:limit]

        memories = [m for m, _ in top_memories]
        scores = [s for _, s in top_memories]

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used="graph_entity_overlap",
            metadata={'query_entities': query_entities, 'total_memories': len(memory_nodes)}
        )

    def select_threads(
        self,
        temporal_window,
        query
    ):
        """
        Select threads (memory shards) from Yarn Graph based on temporal window and query.

        This method implements the core "Yarn Graph thread selection" step in the
        weaving metaphor. It filters graph nodes by temporal bounds, ranks by
        relevance to the query, and returns them as MemoryShard objects.

        Strategy:
        1. Filter memory nodes by temporal window (if timestamps available)
        2. Expand to neighboring entities (1-hop subgraph for context)
        3. Score by relevance to query (entity overlap + recency)
        4. Return top-k as MemoryShards

        Args:
            temporal_window: TemporalWindow object with time bounds and recency bias
            query: Query object with text field

        Returns:
            List[MemoryShard]: Selected threads from the graph
        """
        from HoloLoom.protocols.types import MemoryShard
        from datetime import datetime

        query_text = query.text if hasattr(query, 'text') else str(query)

        # Extract entities from query for relevance scoring
        query_entities = extract_entities_simple(query_text)

        # Find all memory nodes
        memory_nodes = [
            node for node, data in self.G.nodes(data=True)
            if data.get('node_type') == 'memory'
        ]

        # If no memory nodes, return empty list
        if not memory_nodes:
            return []

        # Score and filter memories
        scored_memories = []

        for mem_id in memory_nodes:
            mem_data = self.G.nodes[mem_id]

            # Parse timestamp if available
            timestamp = None
            if 'timestamp' in mem_data:
                try:
                    timestamp = datetime.fromisoformat(mem_data['timestamp'])
                except (ValueError, TypeError):
                    pass

            # Apply temporal filter
            temporal_score = 1.0
            if timestamp and temporal_window:
                if not temporal_window.contains(timestamp):
                    # Outside temporal window - skip
                    continue
                # Apply recency weighting
                temporal_score = temporal_window.recency_weight(timestamp)

            # Get entities this memory mentions
            mem_entities = set()
            for _, dst in self.G.out_edges(mem_id):
                if dst in self.G and self.G.nodes.get(dst, {}).get('node_type') != 'memory':
                    mem_entities.add(dst)

            # Calculate relevance score (entity overlap)
            if query_entities:
                overlap = len(set(query_entities) & mem_entities)
                relevance_score = overlap / len(query_entities)
            else:
                # No entities extracted - use neutral score
                relevance_score = 0.5

            # Combined score: relevance × temporal
            combined_score = relevance_score * temporal_score

            # Convert to MemoryShard
            shard = MemoryShard(
                id=mem_id,
                text=mem_data.get('text', ''),
                episode=mem_data.get('metadata', {}).get('episode', 'default'),
                entities=list(mem_entities),
                motifs=mem_data.get('metadata', {}).get('motifs', []),
                metadata=mem_data.get('metadata', {})
            )

            scored_memories.append((shard, combined_score))

        # Sort by score (highest first)
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Return all shards (orchestrator will limit via retrieval_k)
        # This ensures thread selection is comprehensive
        threads = [shard for shard, _ in scored_memories]

        return threads

    # ========================================================================
    # Multimodal Photo Support (November 2025)
    # ========================================================================

    def add_photo_node(self, photo_token) -> str:
        """
        Add photo as multimodal node in knowledge graph.

        Creates a photo_token node with visual embeddings and metadata,
        then automatically links it to:
        - Entities extracted from caption (DEPICTS edges)
        - Tags (TAGGED_AS edges)
        - Time thread (OCCURRED_AT edge)

        Args:
            photo_token: PhotoToken object from photo_tokens.py

        Returns:
            Node ID (token_id)

        Example:
            >>> from HoloLoom.memory.photo_tokens import PhotoToken
            >>> kg = KG()
            >>> photo = PhotoToken(token_id="photo_abc", caption="Architecture diagram", ...)
            >>> kg.add_photo_node(photo)
            'photo_abc'
        """
        from datetime import datetime

        # Convert PhotoToken to node attributes
        node_data = photo_token.to_yarn_node()

        # Add photo node
        self.G.add_node(
            photo_token.token_id,
            node_type="photo_token",
            **node_data
        )

        # Create DEPICTS edges to entities (from caption)
        if photo_token.caption:
            caption_entities = extract_entities_simple(photo_token.caption)
            for entity in caption_entities[:5]:  # Limit to top 5
                edge = KGEdge(
                    src=photo_token.token_id,
                    dst=entity,
                    type="DEPICTS",
                    weight=1.0,
                    metadata={'extracted_from': 'caption'}
                )
                self.add_edge(edge)

        # Create DEPICTS edges to explicit entities
        for entity in photo_token.entities[:10]:  # Limit to 10
            edge = KGEdge(
                src=photo_token.token_id,
                dst=entity,
                type="DEPICTS",
                weight=1.0,
                metadata={'extracted_from': 'entities'}
            )
            self.add_edge(edge)

        # Create TAGGED_AS edges
        for tag in photo_token.tags:
            edge = KGEdge(
                src=photo_token.token_id,
                dst=tag,
                type="TAGGED_AS",
                weight=1.0
            )
            self.add_edge(edge)

        # Connect to time thread
        if photo_token.timestamp:
            self.connect_entity_to_time(
                entity=photo_token.token_id,
                timestamp=photo_token.timestamp,
                edge_type="OCCURRED_AT"
            )

        return photo_token.token_id

    def link_photo_to_memory(
        self,
        photo_token_id: str,
        memory_id: str,
        edge_type: str = "ILLUSTRATES"
    ) -> None:
        """
        Link photo to text memory.

        Creates a semantic relationship between a photo and a text memory.
        Common edge types:
        - ILLUSTRATES: Photo explains/depicts the memory
        - REFERENCED_IN: Photo mentioned in the text
        - ACCOMPANIES: Photo was captured during the memory event

        Args:
            photo_token_id: Photo token ID
            memory_id: Memory/shard ID
            edge_type: Relationship type (default: ILLUSTRATES)

        Example:
            >>> kg.link_photo_to_memory("photo_abc", "memory_123", "ILLUSTRATES")
        """
        edge = KGEdge(
            src=photo_token_id,
            dst=memory_id,
            type=edge_type,
            weight=1.0
        )
        self.add_edge(edge)

    def link_similar_photos(
        self,
        photo_token_id_1: str,
        photo_token_id_2: str,
        similarity: float
    ) -> None:
        """
        Link two visually similar photos.

        Creates bidirectional SIMILAR_TO edges between photos based on
        CLIP embedding similarity.

        Args:
            photo_token_id_1: First photo ID
            photo_token_id_2: Second photo ID
            similarity: CLIP similarity score (0-1)

        Example:
            >>> kg.link_similar_photos("photo_abc", "photo_def", 0.85)
        """
        # Bidirectional similarity
        edge1 = KGEdge(
            src=photo_token_id_1,
            dst=photo_token_id_2,
            type="SIMILAR_TO",
            weight=similarity
        )
        edge2 = KGEdge(
            src=photo_token_id_2,
            dst=photo_token_id_1,
            type="SIMILAR_TO",
            weight=similarity
        )
        self.add_edge(edge1)
        self.add_edge(edge2)

    def get_photos_by_entity(
        self,
        entity: str,
        edge_type: str = "DEPICTS"
    ) -> List[str]:
        """
        Get all photos depicting an entity.

        Args:
            entity: Entity name
            edge_type: Edge type to follow (default: DEPICTS)

        Returns:
            List of photo token IDs

        Example:
            >>> kg.get_photos_by_entity("architecture")
            ['photo_abc', 'photo_def']
        """
        if entity not in self.G:
            return []

        photos = []
        for src, _, data in self.G.in_edges(entity, data=True):
            if (data.get("type") == edge_type and
                self.G.nodes.get(src, {}).get('node_type') == 'photo_token'):
                photos.append(src)

        return photos

    def get_photos_by_tag(self, tag: str) -> List[str]:
        """
        Get all photos with a specific tag.

        Args:
            tag: Tag name

        Returns:
            List of photo token IDs

        Example:
            >>> kg.get_photos_by_tag("diagram")
            ['photo_abc', 'photo_xyz']
        """
        return self.get_photos_by_entity(tag, edge_type="TAGGED_AS")

    async def search_multimodal(
        self,
        query: str,
        return_types: List[str] = None,
        k: int = 10,
        photo_memory=None
    ) -> List[Dict]:
        """
        Search across text and photo memories (multimodal retrieval).

        Combines:
        1. Graph-based text search (entity overlap)
        2. CLIP-based photo search (semantic similarity)
        3. Caption-based photo search (text similarity)

        Args:
            query: Text query
            return_types: Types to return ['text', 'photo', 'both'] (default: both)
            k: Total number of results
            photo_memory: Optional PhotoTokenMemory for CLIP search

        Returns:
            List of dicts: {'type': 'text'|'photo', 'id': str, 'score': float, 'data': Dict}

        Example:
            >>> results = await kg.search_multimodal(
            ...     "architecture diagram",
            ...     return_types=['text', 'photo'],
            ...     k=5
            ... )
            >>> for r in results:
            ...     print(f"{r['type']}: {r['id']} (score: {r['score']:.3f})")
        """
        if return_types is None:
            return_types = ['text', 'photo']

        results = []
        query_entities = extract_entities_simple(query)

        # Text search (existing graph-based retrieval)
        if 'text' in return_types or 'both' in return_types:
            memory_nodes = [
                node for node, data in self.G.nodes(data=True)
                if data.get('node_type') == 'memory'
            ]

            for mem_id in memory_nodes:
                mem_data = self.G.nodes[mem_id]

                # Get entities this memory mentions
                mem_entities = set()
                for _, dst in self.G.out_edges(mem_id):
                    if dst in self.G and self.G.nodes.get(dst, {}).get('node_type') != 'memory':
                        mem_entities.add(dst)

                # Calculate relevance score (entity overlap)
                if query_entities:
                    overlap = len(set(query_entities) & mem_entities)
                    score = overlap / len(query_entities)
                else:
                    score = 0.3  # Low baseline for no entities

                if score > 0:
                    results.append({
                        'type': 'text',
                        'id': mem_id,
                        'score': float(score),
                        'data': mem_data
                    })

        # Photo search
        if 'photo' in return_types or 'both' in return_types:
            photo_nodes = [
                (node, data) for node, data in self.G.nodes(data=True)
                if data.get('node_type') == 'photo_token'
            ]

            if photo_memory:
                # CLIP-based search (best quality)
                try:
                    clip_results = await photo_memory.retrieve_by_text(query, k=k//2)

                    for photo_token, clip_score in clip_results:
                        # Check if photo is in graph
                        if photo_token.token_id in self.G:
                            results.append({
                                'type': 'photo',
                                'id': photo_token.token_id,
                                'score': float(clip_score),
                                'data': self.G.nodes[photo_token.token_id]
                            })
                except Exception as e:
                    import warnings
                    warnings.warn(f"CLIP search failed: {e}. Falling back to caption search.")

            # Caption-based fallback (if no CLIP or as supplement)
            if not photo_memory or len(results) < k:
                for node_id, node_data in photo_nodes:
                    caption = node_data.get('caption', '')

                    if not caption:
                        continue

                    # Simple keyword matching
                    caption_lower = caption.lower()
                    query_lower = query.lower()

                    # Calculate overlap score
                    query_words = set(query_lower.split())
                    caption_words = set(caption_lower.split())

                    if query_words:
                        overlap = len(query_words & caption_words)
                        score = overlap / len(query_words)
                    else:
                        score = 0.0

                    # Also check tag overlap
                    tags = node_data.get('tags', [])
                    tag_score = 0.0
                    if tags and query_entities:
                        tag_overlap = len(set(t.lower() for t in tags) & set(e.lower() for e in query_entities))
                        tag_score = tag_overlap / len(query_entities)

                    # Combined score
                    combined = max(score, tag_score)

                    if combined > 0.1:  # Minimum threshold
                        # Check if not already added via CLIP
                        if not any(r['id'] == node_id for r in results):
                            results.append({
                                'type': 'photo',
                                'id': node_id,
                                'score': float(combined * 0.7),  # Scale down vs CLIP
                                'data': node_data
                            })

        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top-k
        return results[:k]


# ============================================================================
# Entity Extraction Helpers
# ============================================================================

def extract_entities_simple(text: str) -> List[str]:
    """
    Simple entity extraction: capitalized words.
    
    This is a heuristic - in production, use spaCy NER or similar.
    
    Args:
        text: Input text
        
    Returns:
        List of potential entity names
    """
    words = text.split()
    entities = []
    
    for word in words:
        # Remove punctuation
        cleaned = word.strip('.,!?;:()[]{}"\'-')
        # Check if starts with capital
        if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
            entities.append(cleaned)
    
    return entities


def build_kg_from_text(
    text: str,
    entities: Optional[List[str]] = None,
    context_entity: str = "query"
) -> KG:
    """
    Build a simple KG from text.
    
    Creates MENTIONS edges from entities to a context node.
    This provides basic graph structure for spectral analysis.
    
    Args:
        text: Input text
        entities: Optional explicit entity list (if None, extracts from text)
        context_entity: Central node name (e.g., "query", "document")
        
    Returns:
        KG with MENTIONS relationships
    """
    kg = KG()
    
    if entities is None:
        entities = extract_entities_simple(text)
    
    # Create edges: entity → MENTIONS → context
    for entity in entities[:10]:  # Limit to avoid huge graphs
        kg.add_edge(KGEdge(
            src=entity,
            dst=context_entity,
            type="MENTIONS",
            weight=1.0
        ))
    
    return kg


# ============================================================================
# Legacy Shards Adapter (Backward Compatibility)
# ============================================================================

class LegacyShardsAdapter:
    """
    Simple adapter for backward compatibility with deprecated shards parameter.

    This class provides a minimal interface matching the old YarnGraph class
    that was embedded in weaving_orchestrator.py. It wraps a list of shards
    and implements select_threads() for the weaving pipeline.

    .. deprecated:: November 2025
        Use KG (Knowledge Graph) directly with yarn_graph parameter instead.
        The shards parameter will be removed in a future version.

    Example:
        >>> adapter = LegacyShardsAdapter(shards)
        >>> threads = adapter.select_threads(temporal_window, query)
    """

    def __init__(self, shards: list):
        """
        Initialize with memory shards.

        Args:
            shards: List of MemoryShard objects
        """
        self.shards = {shard.id: shard for shard in shards}
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"LegacyShardsAdapter initialized with {len(shards)} threads")

    def select_threads(self, temporal_window, query) -> list:
        """
        Select threads based on temporal window.

        This is a simple implementation that returns all shards.
        For more sophisticated filtering, use KG.select_threads().

        Args:
            temporal_window: Time bounds for selection (ignored in legacy mode)
            query: Query for relevance filtering (ignored in legacy mode)

        Returns:
            List of all memory shards
        """
        threads = list(self.shards.values())
        self._logger.debug(f"Selected {len(threads)} threads (legacy mode)")
        return threads


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== Knowledge Graph Demo ===\n")
    
    # Create knowledge graph
    kg = KG()
    
    # Add domain knowledge
    edges = [
        KGEdge("attention", "transformer", "USES", 1.0),
        KGEdge("transformer", "neural_network", "IS_A", 1.0),
        KGEdge("attention", "neural_network", "PART_OF", 0.8),
        KGEdge("BERT", "transformer", "IS_A", 1.0),
        KGEdge("GPT", "transformer", "IS_A", 1.0),
        KGEdge("multi-head attention", "attention", "IS_A", 1.0),
        KGEdge("self-attention", "attention", "IS_A", 1.0),
    ]
    
    kg.add_edges(edges)
    
    # Graph statistics
    print("Graph stats:")
    stats = kg.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Query: Get subgraph for entities
    print("\nSubgraph for ['attention', 'BERT']:")
    subgraph = kg.subgraph_for_entities(["attention", "BERT"], expand=True)
    print(f"  Nodes: {list(subgraph.nodes())}")
    print(f"  Edges: {subgraph.number_of_edges()}")
    
    # Find relationships
    print("\nWhat is BERT?")
    is_a = kg.get_related_by_type("BERT", "IS_A", direction="out")
    print(f"  BERT IS_A {is_a}")
    
    print("\nWhat uses attention?")
    uses = kg.get_related_by_type("attention", "USES", direction="in")
    print(f"  {uses} USES attention")
    
    # Find paths
    print("\nPath from 'BERT' to 'neural_network':")
    paths = kg.get_paths("BERT", "neural_network", max_length=3)
    for path in paths:
        print(f"  {' → '.join(path)}")
    
    # Persistence
    print("\nSaving and loading...")
    kg.save("demo_kg.jsonl")
    kg2 = KG.load("demo_kg.jsonl")
    print(f"  Loaded graph has {kg2.G.number_of_nodes()} nodes")
    
    print("\n✓ Demo complete!")