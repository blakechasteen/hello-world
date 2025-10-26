"""
HoloLoom Neo4j Knowledge Graph Store
=====================================
Production-grade graph database implementation using Neo4j.

This is a "warp thread" module - independent graph storage with Neo4j backend.

Architecture:
- Implements KGStore protocol (drop-in replacement for KG)
- Neo4j Bolt driver for high-performance graph operations
- Cypher query language for complex graph traversals
- APOC procedures for advanced graph algorithms
- Zero dependencies on other HoloLoom modules (except types)

Philosophy:
The Neo4j KG provides persistent, scalable graph storage with:
- ACID transactions
- Advanced indexing and query optimization
- Native graph algorithms (PageRank, community detection, etc.)
- Multi-user concurrent access
- Distributed deployment capabilities

Weaving Metaphor:
This is the "Yarn Graph" backed by Neo4j - persistent symbolic memory
that can be "tensioned" into Warp Space for continuous computation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any
import json
from pathlib import Path
import warnings

try:
    from neo4j import GraphDatabase, Driver, Transaction
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    warnings.warn(
        "neo4j package not available. Install with: pip install neo4j>=5.14.0"
    )

import networkx as nx

from HoloLoom.Utils.time_bucket import TimeInput, time_bucket, to_utc_datetime

# Import KGEdge from the base graph module
try:
    from holoLoom.memory.graph import KGEdge
except ImportError:
    from HoloLoom.memory.graph import KGEdge


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "hololoom123"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Load config from environment variables."""
        import os
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "hololoom123"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


# ============================================================================
# Neo4j Knowledge Graph Implementation
# ============================================================================

class Neo4jKG:
    """
    Neo4j-backed knowledge graph implementing KGStore protocol.

    Features:
    - Persistent storage with ACID guarantees
    - High-performance Cypher queries
    - Native graph algorithms via APOC
    - Concurrent access support
    - Advanced indexing and constraints

    Node Structure:
    - Labels: Entity, TimeThread
    - Properties: name, kind, metadata fields

    Relationship Structure:
    - Type: edge.type (e.g., IS_A, USES, MENTIONS, IN_TIME)
    - Properties: weight, span_id, metadata fields

    Use Cases:
    - Production knowledge graph storage
    - Multi-user collaborative knowledge base
    - Large-scale graph analysis (millions of nodes)
    - Complex graph queries and pattern matching
    """

    def __init__(self, config: Optional[Neo4jConfig] = None):
        """
        Initialize Neo4j knowledge graph.

        Args:
            config: Neo4j connection configuration
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package is required. Install with: pip install neo4j>=5.14.0"
            )

        self.config = config or Neo4jConfig()
        self.driver: Optional[Driver] = None
        self._connected = False

        # Connect to Neo4j
        self._connect()

        # Create indexes and constraints
        self._setup_schema()

    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
            )

            # Verify connectivity
            self.driver.verify_connectivity()
            self._connected = True

        except (ServiceUnavailable, AuthError) as e:
            raise ConnectionError(
                f"Failed to connect to Neo4j at {self.config.uri}: {e}"
            )

    def _setup_schema(self) -> None:
        """Create indexes and constraints for performance."""
        with self.driver.session(database=self.config.database) as session:
            # Unique constraint on Entity.name
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

            # Index on TimeThread.bucket for temporal queries
            session.run(
                "CREATE INDEX time_bucket_idx IF NOT EXISTS "
                "FOR (t:TimeThread) ON (t.bucket)"
            )

            # Full-text index for entity search (requires APOC or native full-text)
            try:
                session.run(
                    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                    "FOR (e:Entity) ON EACH [e.name]"
                )
            except Exception:
                # Fallback if full-text not available
                pass

    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self._connected = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # ========================================================================
    # KGStore Protocol Implementation
    # ========================================================================

    def add_edge(self, edge: KGEdge) -> None:
        """
        Add an edge to the knowledge graph.

        Creates nodes if they don't exist and establishes relationship.

        Args:
            edge: KGEdge to add
        """
        query = """
        MERGE (src:Entity {name: $src_name})
        MERGE (dst:Entity {name: $dst_name})
        CREATE (src)-[r:RELATES {
            type: $edge_type,
            weight: $weight,
            span_id: $span_id
        }]->(dst)
        SET r += $metadata
        RETURN r
        """

        with self.driver.session(database=self.config.database) as session:
            session.run(
                query,
                src_name=edge.src,
                dst_name=edge.dst,
                edge_type=edge.type,
                weight=edge.weight,
                span_id=edge.span_id,
                metadata=edge.metadata or {}
            )

    def add_edges(self, edges: List[KGEdge]) -> None:
        """
        Bulk add edges (more efficient than individual adds).

        Uses batched transaction for better performance.
        """
        batch_size = 1000

        with self.driver.session(database=self.config.database) as session:
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]

                with session.begin_transaction() as tx:
                    for edge in batch:
                        tx.run(
                            """
                            MERGE (src:Entity {name: $src_name})
                            MERGE (dst:Entity {name: $dst_name})
                            CREATE (src)-[r:RELATES {
                                type: $edge_type,
                                weight: $weight,
                                span_id: $span_id
                            }]->(dst)
                            SET r += $metadata
                            """,
                            src_name=edge.src,
                            dst_name=edge.dst,
                            edge_type=edge.type,
                            weight=edge.weight,
                            span_id=edge.span_id,
                            metadata=edge.metadata or {}
                        )
                    tx.commit()

    def connect_entity_to_time(
        self,
        entity: str,
        timestamp: TimeInput,
        *,
        edge_type: str = "IN_TIME",
        weight: float = 1.0,
    ) -> str:
        """
        Attach an entity node to a coarse-grained time thread.

        Creates or reuses time bucket nodes for temporal organization.

        Args:
            entity: Name of the entity/event node
            timestamp: Datetime/ISO string/epoch seconds
            edge_type: Relationship label for the connection
            weight: Edge weight

        Returns:
            The identifier of the time thread node
        """
        dt = to_utc_datetime(timestamp)
        bucket = time_bucket(dt)
        thread_id = f"time::{bucket}"

        query = """
        MERGE (e:Entity {name: $entity})
        MERGE (t:TimeThread {name: $thread_id, bucket: $bucket})
        CREATE (e)-[r:RELATES {
            type: $edge_type,
            weight: $weight,
            bucket: $bucket,
            timestamp: $timestamp
        }]->(t)
        RETURN t.name AS thread_id
        """

        with self.driver.session(database=self.config.database) as session:
            result = session.run(
                query,
                entity=entity,
                thread_id=thread_id,
                bucket=bucket,
                edge_type=edge_type,
                weight=weight,
                timestamp=dt.isoformat()
            )
            record = result.single()
            return record["thread_id"] if record else thread_id

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
        # Build direction pattern
        if direction == "out":
            pattern = "-[*1..{}]->".format(max_hops)
        elif direction == "in":
            pattern = "<-[*1..{}]-".format(max_hops)
        else:  # both
            pattern = "-[*1..{}]-".format(max_hops)

        query = f"""
        MATCH (e:Entity {{name: $entity}}){pattern}(neighbor)
        WHERE neighbor.name <> $entity
        RETURN DISTINCT neighbor.name AS name
        """

        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, entity=entity)
            return {record["name"] for record in result}

    def subgraph_for_entities(
        self,
        entities: List[str],
        expand: bool = True,
        max_hops: int = 1
    ) -> nx.MultiDiGraph:
        """
        Extract subgraph containing entities and their neighborhoods.

        Returns NetworkX graph for compatibility with existing code.

        Args:
            entities: List of entity names to include
            expand: If True, include neighbors
            max_hops: How many hops to expand

        Returns:
            MultiDiGraph containing the subgraph
        """
        if not entities:
            return nx.MultiDiGraph()

        # Build query based on expansion
        if expand:
            pattern = f"-[r*0..{max_hops}]-"
        else:
            pattern = ""

        query = f"""
        MATCH (e:Entity)
        WHERE e.name IN $entities
        {"OPTIONAL MATCH (e)" + pattern + "(neighbor)" if expand else ""}
        {"WITH e, neighbor, relationships(p) AS rels" if expand else ""}
        {"UNWIND rels AS r" if expand else ""}
        RETURN DISTINCT
            {"startNode(r).name AS src," if expand else "e.name AS src,"}
            {"endNode(r).name AS dst," if expand else "null AS dst,"}
            {"r.type AS type," if expand else "null AS type,"}
            {"r.weight AS weight," if expand else "null AS weight,"}
            {"r.span_id AS span_id," if expand else "null AS span_id,"}
            {"properties(r) AS metadata" if expand else "{} AS metadata"}
        """

        # Simpler query for non-expanded case
        if not expand:
            query = """
            MATCH (e:Entity)
            WHERE e.name IN $entities
            RETURN e.name AS src, null AS dst, null AS type,
                   null AS weight, null AS span_id, {} AS metadata
            """
        else:
            query = """
            MATCH path = (e:Entity)-[r*0..{}]-(neighbor)
            WHERE e.name IN $entities
            UNWIND relationships(path) AS rel
            RETURN DISTINCT
                startNode(rel).name AS src,
                endNode(rel).name AS dst,
                rel.type AS type,
                rel.weight AS weight,
                rel.span_id AS span_id,
                properties(rel) AS metadata
            """.format(max_hops)

        G = nx.MultiDiGraph()

        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, entities=entities)

            for record in result:
                src = record["src"]
                dst = record.get("dst")

                # Add source node
                if src and src not in G:
                    G.add_node(src)

                # Add edge if exists
                if dst and dst != src:
                    if dst not in G:
                        G.add_node(dst)

                    metadata = record.get("metadata", {})
                    # Remove type, weight, span_id from metadata to avoid duplication
                    metadata = {k: v for k, v in metadata.items()
                               if k not in ["type", "weight", "span_id"]}

                    G.add_edge(
                        src,
                        dst,
                        type=record.get("type", "unknown"),
                        weight=record.get("weight", 1.0),
                        span_id=record.get("span_id"),
                        **metadata
                    )

        return G

    def get_edge_types(self, src: str, dst: str) -> List[str]:
        """
        Get all edge types between two entities.

        Returns:
            List of edge types
        """
        query = """
        MATCH (src:Entity {name: $src})-[r:RELATES]->(dst:Entity {name: $dst})
        RETURN r.type AS type
        """

        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, src=src, dst=dst)
            return [record["type"] for record in result]

    def get_paths(
        self,
        src: str,
        dst: str,
        max_length: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two entities.

        Args:
            src: Source entity
            dst: Destination entity
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of entity names)
        """
        query = """
        MATCH path = (src:Entity {name: $src})-[*1..{}]->(dst:Entity {name: $dst})
        RETURN [node IN nodes(path) | node.name] AS path
        LIMIT 100
        """.format(max_length)

        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, src=src, dst=dst)
            return [record["path"] for record in result]

    def get_related_by_type(
        self,
        entity: str,
        edge_type: str,
        direction: str = "out"
    ) -> List[str]:
        """
        Get entities related by a specific edge type.

        Args:
            entity: Starting entity
            edge_type: Relationship type to follow
            direction: "out" or "in"

        Returns:
            List of related entity names
        """
        if direction == "out":
            pattern = "-[r:RELATES]->"
            return_node = "dst"
        else:  # in
            pattern = "<-[r:RELATES]-"
            return_node = "src"

        query = f"""
        MATCH (e:Entity {{name: $entity}}){pattern}(other:Entity)
        WHERE r.type = $edge_type
        RETURN other.name AS name
        """

        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, entity=entity, edge_type=edge_type)
            return [record["name"] for record in result]

    def stats(self) -> Dict:
        """Get graph statistics."""
        query = """
        MATCH (n)
        OPTIONAL MATCH ()-[r]->()
        RETURN
            count(DISTINCT n) AS num_nodes,
            count(r) AS num_edges
        """

        with self.driver.session(database=self.config.database) as session:
            result = session.run(query)
            record = result.single()

            num_nodes = record["num_nodes"] or 0
            num_edges = record["num_edges"] or 0
            avg_degree = (2 * num_edges) / max(1, num_nodes)

            return {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "avg_degree": avg_degree,
                "backend": "neo4j",
                "database": self.config.database,
            }

    # ========================================================================
    # Neo4j-Specific Advanced Features
    # ========================================================================

    def run_cypher(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute raw Cypher query.

        Advanced use cases: custom graph algorithms, complex patterns.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dicts
        """
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def pagerank(self, limit: int = 10) -> List[tuple[str, float]]:
        """
        Compute PageRank centrality using APOC.

        Requires APOC plugin installed in Neo4j.

        Returns:
            List of (entity_name, score) tuples, sorted by score
        """
        query = """
        CALL gds.pageRank.stream({
            nodeProjection: 'Entity',
            relationshipProjection: 'RELATES'
        })
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS entity, score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, limit=limit)
                return [(record["entity"], record["score"]) for record in result]
        except Exception as e:
            warnings.warn(f"PageRank failed (requires GDS plugin): {e}")
            return []

    def clear(self) -> None:
        """
        Clear all nodes and relationships.

        WARNING: This deletes all data in the graph!
        """
        query = """
        MATCH (n)
        DETACH DELETE n
        """

        with self.driver.session(database=self.config.database) as session:
            session.run(query)

    # ========================================================================
    # Persistence & Migration
    # ========================================================================

    def export_to_networkx(self) -> nx.MultiDiGraph:
        """
        Export entire graph to NetworkX format.

        Useful for migration or local analysis.
        """
        query = """
        MATCH (src)-[r:RELATES]->(dst)
        RETURN
            src.name AS src,
            dst.name AS dst,
            r.type AS type,
            r.weight AS weight,
            r.span_id AS span_id,
            properties(r) AS metadata
        """

        G = nx.MultiDiGraph()

        with self.driver.session(database=self.config.database) as session:
            result = session.run(query)

            for record in result:
                src = record["src"]
                dst = record["dst"]

                if src not in G:
                    G.add_node(src)
                if dst not in G:
                    G.add_node(dst)

                metadata = record["metadata"] or {}
                metadata = {k: v for k, v in metadata.items()
                           if k not in ["type", "weight", "span_id"]}

                G.add_edge(
                    src,
                    dst,
                    type=record["type"],
                    weight=record["weight"],
                    span_id=record["span_id"],
                    **metadata
                )

        return G

    def import_from_networkx(self, G: nx.MultiDiGraph) -> None:
        """
        Import NetworkX graph into Neo4j.

        Useful for migration from in-memory graph.
        """
        edges = []

        for src, dst, key, data in G.edges(keys=True, data=True):
            edge = KGEdge(
                src=str(src),
                dst=str(dst),
                type=data.get("type", "unknown"),
                weight=data.get("weight", 1.0),
                span_id=data.get("span_id"),
                metadata={k: v for k, v in data.items()
                         if k not in ["type", "weight", "span_id"]}
            )
            edges.append(edge)

        self.add_edges(edges)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== Neo4j Knowledge Graph Demo ===\n")

    try:
        # Create Neo4j knowledge graph
        config = Neo4jConfig()
        kg = Neo4jKG(config)

        print(f"Connected to Neo4j at {config.uri}")
        print(f"Database: {config.database}\n")

        # Clear existing data for demo
        print("Clearing existing data...")
        kg.clear()

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

        print("Adding edges...")
        kg.add_edges(edges)

        # Graph statistics
        print("\nGraph stats:")
        stats = kg.stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Query: Get neighbors
        print("\nNeighbors of 'attention':")
        neighbors = kg.get_neighbors("attention", direction="both", max_hops=1)
        print(f"  {neighbors}")

        # Query: Get subgraph for entities
        print("\nSubgraph for ['attention', 'BERT']:")
        subgraph = kg.subgraph_for_entities(["attention", "BERT"], expand=True)
        print(f"  Nodes: {list(subgraph.nodes())}")
        print(f"  Edges: {subgraph.number_of_edges()}")

        # Find relationships
        print("\nWhat is BERT?")
        is_a = kg.get_related_by_type("BERT", "IS_A", direction="out")
        print(f"  BERT IS_A {is_a}")

        # Find paths
        print("\nPath from 'BERT' to 'neural_network':")
        paths = kg.get_paths("BERT", "neural_network", max_length=3)
        for path in paths[:3]:  # Show first 3 paths
            print(f"  {' → '.join(path)}")

        # Export to NetworkX
        print("\nExporting to NetworkX...")
        nx_graph = kg.export_to_networkx()
        print(f"  Exported graph has {nx_graph.number_of_nodes()} nodes")

        print("\n✓ Demo complete!")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker-compose up -d")

    finally:
        if 'kg' in locals():
            kg.close()
