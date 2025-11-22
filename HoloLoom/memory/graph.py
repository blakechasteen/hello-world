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
from typing import List, Dict, Optional, Protocol, Set
import json
from pathlib import Path

import numpy as np
import networkx as nx

from HoloLoom.utils.time_bucket import TimeInput, time_bucket, to_utc_datetime

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

    def invalidate(self, timestamp: Optional['datetime'] = None):
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
    
    Use Cases:
    - Entity relationship tracking
    - Context expansion (find related entities)
    - Reasoning over structured knowledge
    - Spectral analysis of knowledge structure
    """
    
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self._entity_index: Dict[str, Set[str]] = {}  # Fast neighbor lookup
    
    def add_edge(self, edge: KGEdge) -> None:
        """
        Add an edge to the knowledge graph.
        
        Automatically creates nodes if they don't exist.
        Supports multiple edges between the same entities.
        
        Args:
            edge: KGEdge to add
        """
        # Ensure nodes exist
        if edge.src not in self.G:
            self.G.add_node(edge.src)
        if edge.dst not in self.G:
            self.G.add_node(edge.dst)
        
        # Add edge with all metadata
        self.G.add_edge(
            edge.src,
            edge.dst,
            type=edge.type,
            weight=edge.weight,
            span_id=edge.span_id,
            **edge.metadata
        )
        
        # Update entity index for fast lookups
        if edge.src not in self._entity_index:
            self._entity_index[edge.src] = set()
        if edge.dst not in self._entity_index:
            self._entity_index[edge.dst] = set()
        
        self._entity_index[edge.src].add(edge.dst)
        self._entity_index[edge.dst].add(edge.src)
    
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
        
        return [data.get("type", "unknown") for _, _, data in self.G.edges(src, dst, data=True)]
    
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

    def clear_spectral_cache(self):
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