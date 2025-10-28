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

import networkx as nx

from HoloLoom.utils.time_bucket import TimeInput, time_bucket, to_utc_datetime

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class KGEdge:
    """
    A directed edge in the knowledge graph.
    
    Represents a typed relationship between two entities.
    Examples:
    - ("Python", "programming_language", IS_A)
    - ("attention", "transformer", USES)
    - ("cause", "effect", LEADS_TO)
    """
    src: str                    # Source entity
    dst: str                    # Destination entity
    type: str                   # Relationship type (e.g., IS_A, USES, MENTIONS)
    weight: float = 1.0         # Edge weight/confidence
    span_id: Optional[str] = None  # Optional: link to source span/shard
    metadata: Dict = field(default_factory=dict)  # Additional properties
    
    def to_dict(self) -> Dict:
        """Serialize edge for persistence."""
        return {
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "weight": self.weight,
            "span_id": self.span_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KGEdge':
        """Deserialize edge from storage."""
        return cls(
            src=data["src"],
            dst=data["dst"],
            type=data["type"],
            weight=data.get("weight", 1.0),
            span_id=data.get("span_id"),
            metadata=data.get("metadata", {})
        )


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