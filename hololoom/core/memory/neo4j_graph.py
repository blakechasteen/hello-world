"""
HoloLoom Neo4j Knowledge Graph Store
=====================================
Production-grade graph database implementation with Neo4j connection pooling.

This is a "warp thread" module - independent graph storage with Neo4j backend.

Architecture:
- Implements KGStore protocol (drop-in replacement for KG)
- Neo4j Bolt driver with connection pooling for high-performance operations
- Cypher query language for complex graph traversals
- APOC procedures for advanced graph algorithms
- Zero dependencies on other HoloLoom modules (except types)

Connection Pooling:
- Built-in driver pooling with configurable size
- Connection health checks and automatic retry
- Graceful degradation on pool exhaustion
- Monitoring of pool utilization

Philosophy:
The Neo4j KG provides persistent, scalable graph storage with:
- ACID transactions
- Advanced indexing and query optimization
- Native graph algorithms (PageRank, community detection, etc.)
- Multi-user concurrent access with connection pooling
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
import logging
import os
import time
from threading import Lock

try:
    from neo4j import GraphDatabase, Driver, Transaction
    from neo4j.exceptions import ServiceUnavailable, AuthError, SessionExpired
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    warnings.warn(
        "neo4j package not available. Install with: pip install neo4j>=5.14.0"
    )

logger = logging.getLogger(__name__)

import networkx as nx

from hololoom.utils.time_bucket import TimeInput, time_bucket, to_utc_datetime

# Import KGEdge from the base graph module
try:
    from hololoom.memory.graph import KGEdge
except ImportError:
    from hololoom.core.memory.graph import KGEdge


# ============================================================================
# Environment Variable Bounds Validation
# ============================================================================

def _get_bounded_int_env(
    name: str,
    default: int,
    min_val: int,
    max_val: int
) -> int:
    """
    SECURITY: Parse integer environment variable with bounds validation.

    Prevents resource exhaustion or denial of service via malicious environment
    variable configuration (e.g., pool_size=999999999 causing OOM).

    Args:
        name: Environment variable name
        default: Default value if not set
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Bounded integer value

    Raises:
        ValueError: If value cannot be parsed as integer
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            f"SECURITY: Invalid integer for {name}='{raw}', using default {default}"
        )
        return default

    if value < min_val:
        logger.warning(
            f"SECURITY: {name}={value} below minimum {min_val}, clamping to {min_val}"
        )
        return min_val
    if value > max_val:
        logger.warning(
            f"SECURITY: {name}={value} above maximum {max_val}, clamping to {max_val}"
        )
        return max_val

    return value


def _get_bounded_float_env(
    name: str,
    default: float,
    min_val: float,
    max_val: float
) -> float:
    """
    SECURITY: Parse float environment variable with bounds validation.

    Prevents resource exhaustion or denial of service via malicious environment
    variable configuration (e.g., timeout=0 causing immediate failures or
    timeout=999999 causing hung connections).

    Args:
        name: Environment variable name
        default: Default value if not set
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Bounded float value

    Raises:
        ValueError: If value cannot be parsed as float
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            f"SECURITY: Invalid float for {name}='{raw}', using default {default}"
        )
        return default

    if value < min_val:
        logger.warning(
            f"SECURITY: {name}={value} below minimum {min_val}, clamping to {min_val}"
        )
        return min_val
    if value > max_val:
        logger.warning(
            f"SECURITY: {name}={value} above maximum {max_val}, clamping to {max_val}"
        )
        return max_val

    return value


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Neo4jConfig:
    """
    Configuration for Neo4j connection with production pooling.

    Connection Pool Settings:
        max_connection_pool_size: Maximum connections in pool (default: 50)
        connection_acquisition_timeout: Max time to wait for connection (default: 30s)
        max_transaction_retry_time: Max time for transaction retries (default: 15s)
        connection_timeout: Network timeout for connections (default: 30s)
        max_connection_lifetime: Max lifetime of pooled connection (default: 1h)

    Environment Variables (REQUIRED):
        NEO4J_URI: Database URI (default: bolt://localhost:7687)
        NEO4J_USERNAME: Username (default: neo4j)
        NEO4J_PASSWORD: Password (REQUIRED - no default)
        NEO4J_DATABASE: Database name (default: neo4j)
        NEO4J_POOL_SIZE: Max pool size (default: 50)
        NEO4J_TIMEOUT: Connection timeout in seconds (default: 30)
        NEO4J_RETRY_TIME: Max transaction retry time (default: 15)

    Security Note:
        Password must be provided via environment variable or explicit parameter.
        No hardcoded default is provided for security reasons.
    """
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""  # SECURITY: No default - must be set via env or parameter
    database: str = "neo4j"
    max_connection_lifetime: int = 3600  # 1 hour
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: float = 30.0
    connection_timeout: float = 30.0
    max_transaction_retry_time: float = 15.0
    # Monitoring
    enable_metrics: bool = True
    log_pool_exhaustion: bool = True

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Load config from environment variables with pooling settings.

        SECURITY: All numeric values are bounds-validated to prevent:
        - Resource exhaustion (e.g., pool_size=999999999)
        - Denial of service (e.g., timeout=0 or timeout=infinity)
        - Invalid configurations that could crash the system

        Bounds:
            NEO4J_POOL_SIZE: 1-1000 (default: 50)
            NEO4J_TIMEOUT: 1-300 seconds (default: 30)
            NEO4J_ACQUISITION_TIMEOUT: 1-300 seconds (default: 30)
            NEO4J_RETRY_TIME: 1-120 seconds (default: 15)
            NEO4J_LIFETIME: 60-86400 seconds (default: 3600)

        Raises:
            ValueError: If NEO4J_PASSWORD environment variable is not set.
        """
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError(
                "NEO4J_PASSWORD environment variable is required. "
                "Set it before connecting to Neo4j: export NEO4J_PASSWORD='your-password'"
            )
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=password,
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            # SECURITY: Bounded env var parsing prevents resource exhaustion
            max_connection_pool_size=_get_bounded_int_env(
                "NEO4J_POOL_SIZE", default=50, min_val=1, max_val=1000
            ),
            connection_timeout=_get_bounded_float_env(
                "NEO4J_TIMEOUT", default=30.0, min_val=1.0, max_val=300.0
            ),
            connection_acquisition_timeout=_get_bounded_float_env(
                "NEO4J_ACQUISITION_TIMEOUT", default=30.0, min_val=1.0, max_val=300.0
            ),
            max_transaction_retry_time=_get_bounded_float_env(
                "NEO4J_RETRY_TIME", default=15.0, min_val=1.0, max_val=120.0
            ),
            max_connection_lifetime=_get_bounded_int_env(
                "NEO4J_LIFETIME", default=3600, min_val=60, max_val=86400
            ),
            enable_metrics=os.getenv("NEO4J_ENABLE_METRICS", "true").lower() == "true",
        )


# ============================================================================
# Neo4j Knowledge Graph Implementation
# ============================================================================

class Neo4jKG:
    """
    Neo4j-backed knowledge graph with production connection pooling.

    Features:
    - Connection pooling with configurable size and timeouts
    - Health checks and automatic connection recovery
    - Pool exhaustion detection and graceful degradation
    - Connection metrics and monitoring
    - Persistent storage with ACID guarantees
    - High-performance Cypher queries
    - Native graph algorithms via APOC
    - Concurrent access support
    - Advanced indexing and constraints

    Connection Pool:
    - Driver manages connection pool internally
    - Automatic connection health checks
    - Retry logic with exponential backoff
    - Graceful degradation on pool exhaustion

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

    # Class-level driver singleton with lock for thread safety
    _driver_instance: Optional[Driver] = None
    _driver_lock = Lock()
    _connection_metrics = {
        'total_connections': 0,
        'failed_connections': 0,
        'pool_exhaustion_count': 0,
        'last_health_check': None,
        'health_status': 'unknown'
    }

    def __init__(self, config: Optional[Neo4jConfig] = None):
        """
        Initialize Neo4j knowledge graph with connection pooling.

        Args:
            config: Neo4j connection configuration with pooling settings
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package is required. Install with: pip install neo4j>=5.14.0"
            )

        self.config = config or Neo4jConfig()
        self._connected = False
        self._metrics_enabled = self.config.enable_metrics

        # Initialize or reuse driver singleton
        self._initialize_driver()

        # Create indexes and constraints
        self._setup_schema()

    def _initialize_driver(self) -> None:
        """Initialize driver singleton with connection pooling."""
        with Neo4jKG._driver_lock:
            # Reuse existing driver if available and healthy
            if Neo4jKG._driver_instance:
                try:
                    Neo4jKG._driver_instance.verify_connectivity()
                    self.driver = Neo4jKG._driver_instance
                    self._connected = True
                    logger.info("Reusing existing Neo4j driver connection pool")
                    return
                except Exception as e:
                    logger.warning(f"Existing driver unhealthy, recreating: {e}")
                    try:
                        Neo4jKG._driver_instance.close()
                    except Exception:
                        pass
                    Neo4jKG._driver_instance = None

            # Create new driver with connection pooling
            self._connect()
            Neo4jKG._driver_instance = self.driver

    def _connect(self) -> None:
        """Establish connection to Neo4j with production pooling settings."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to Neo4j at {self.config.uri} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                # Create driver with full pooling configuration
                self.driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.username, self.config.password),
                    max_connection_lifetime=self.config.max_connection_lifetime,
                    max_connection_pool_size=self.config.max_connection_pool_size,
                    connection_acquisition_timeout=self.config.connection_acquisition_timeout,
                    connection_timeout=self.config.connection_timeout,
                    max_transaction_retry_time=self.config.max_transaction_retry_time,
                )

                # Verify connectivity
                self.driver.verify_connectivity()
                self._connected = True

                if self._metrics_enabled:
                    Neo4jKG._connection_metrics['total_connections'] += 1
                    Neo4jKG._connection_metrics['health_status'] = 'healthy'
                    Neo4jKG._connection_metrics['last_health_check'] = time.time()

                logger.info(
                    f"Successfully connected to Neo4j with pool size: "
                    f"{self.config.max_connection_pool_size}"
                )
                return

            except (ServiceUnavailable, AuthError) as e:
                if self._metrics_enabled:
                    Neo4jKG._connection_metrics['failed_connections'] += 1

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    Neo4jKG._connection_metrics['health_status'] = 'unhealthy'
                    raise ConnectionError(
                        f"Failed to connect to Neo4j after {max_retries} attempts: {e}"
                    )

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Neo4j connection.

        Returns:
            Dict containing health status and metrics
        """
        health = {
            'status': 'unknown',
            'connected': False,
            'pool_metrics': {},
            'last_check': None,
            'error': None
        }

        try:
            # Check driver connectivity
            if self.driver:
                self.driver.verify_connectivity()
                health['status'] = 'healthy'
                health['connected'] = True

                # Get pool metrics if available
                if self._metrics_enabled:
                    health['pool_metrics'] = {
                        'total_connections': Neo4jKG._connection_metrics['total_connections'],
                        'failed_connections': Neo4jKG._connection_metrics['failed_connections'],
                        'pool_exhaustion_count': Neo4jKG._connection_metrics['pool_exhaustion_count'],
                    }

            health['last_check'] = time.time()

            if self._metrics_enabled:
                Neo4jKG._connection_metrics['last_health_check'] = health['last_check']
                Neo4jKG._connection_metrics['health_status'] = health['status']

        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")

            if self._metrics_enabled:
                Neo4jKG._connection_metrics['health_status'] = 'unhealthy'

        return health

    def _execute_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic and pool exhaustion detection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            Exception: If all retries fail
        """
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            except SessionExpired as e:
                # Session expired, retry with new session
                logger.warning(f"Session expired on attempt {attempt + 1}, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

            except ServiceUnavailable as e:
                # Possible pool exhaustion or server issues
                if "Unable to acquire connection from the pool" in str(e):
                    if self._metrics_enabled:
                        Neo4jKG._connection_metrics['pool_exhaustion_count'] += 1

                    if self.config.log_pool_exhaustion:
                        logger.warning(
                            f"Connection pool exhausted (attempt {attempt + 1}/{max_retries}). "
                            f"Consider increasing NEO4J_POOL_SIZE (current: {self.config.max_connection_pool_size})"
                        )

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

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
        """
        Close Neo4j connection and clean up pool.

        Note: Only closes the driver if this is the last instance using it.
        The driver singleton is shared across instances for connection pooling.
        """
        with Neo4jKG._driver_lock:
            if self.driver:
                self._connected = False
                # Don't close the driver singleton - it's shared
                # Only close if we need to force cleanup
                logger.info("Neo4j instance closed (driver pool remains active)")

    @classmethod
    def force_close_driver(cls) -> None:
        """
        Force close the driver singleton and all connections.

        Use this for application shutdown or when you need to fully reset connections.
        """
        with cls._driver_lock:
            if cls._driver_instance:
                try:
                    cls._driver_instance.close()
                    logger.info("Neo4j driver singleton closed, pool terminated")
                except Exception as e:
                    logger.error(f"Error closing Neo4j driver: {e}")
                finally:
                    cls._driver_instance = None

    def get_pool_metrics(self) -> Dict[str, Any]:
        """
        Get connection pool metrics for monitoring.

        Returns:
            Dict with pool utilization and performance metrics
        """
        metrics = {
            'pool_size': self.config.max_connection_pool_size,
            'connection_timeout': self.config.connection_timeout,
            'acquisition_timeout': self.config.connection_acquisition_timeout,
            'health_status': Neo4jKG._connection_metrics.get('health_status', 'unknown'),
            'total_connections': Neo4jKG._connection_metrics.get('total_connections', 0),
            'failed_connections': Neo4jKG._connection_metrics.get('failed_connections', 0),
            'pool_exhaustion_count': Neo4jKG._connection_metrics.get('pool_exhaustion_count', 0),
            'last_health_check': Neo4jKG._connection_metrics.get('last_health_check'),
        }

        # Calculate failure rate
        if metrics['total_connections'] > 0:
            metrics['failure_rate'] = metrics['failed_connections'] / metrics['total_connections']
        else:
            metrics['failure_rate'] = 0.0

        # Add warnings if needed
        if metrics['pool_exhaustion_count'] > 0:
            metrics['warning'] = f"Pool exhaustion detected {metrics['pool_exhaustion_count']} times. Consider increasing pool size."

        return metrics

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
        # SECURITY: Validate max_hops to prevent Cypher injection
        # Neo4j doesn't support parameterized hop counts, so we must validate input
        if not isinstance(max_hops, int) or max_hops < 1 or max_hops > 10:
            raise ValueError(f"max_hops must be integer between 1-10, got: {max_hops}")

        # SECURITY: Validate direction is one of allowed values
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be 'out', 'in', or 'both', got: {direction}")

        # Build direction pattern (safe after validation)
        if direction == "out":
            pattern = f"-[*1..{max_hops}]->"
        elif direction == "in":
            pattern = f"<-[*1..{max_hops}]-"
        else:  # both
            pattern = f"-[*1..{max_hops}]-"

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

        # SECURITY: Validate max_hops to prevent Cypher injection
        # Neo4j doesn't support parameterized hop counts, so we must validate input
        if not isinstance(max_hops, int) or max_hops < 0 or max_hops > 10:
            raise ValueError(f"max_hops must be integer between 0-10, got: {max_hops}")

        # Simpler query for non-expanded case
        if not expand:
            query = """
            MATCH (e:Entity)
            WHERE e.name IN $entities
            RETURN e.name AS src, null AS dst, null AS type,
                   null AS weight, null AS span_id, {} AS metadata
            """
        else:
            # SECURITY: max_hops is validated above (integer 0-10), safe to interpolate
            query = f"""
            MATCH path = (e:Entity)-[r*0..{max_hops}]-(neighbor)
            WHERE e.name IN $entities
            UNWIND relationships(path) AS rel
            RETURN DISTINCT
                startNode(rel).name AS src,
                endNode(rel).name AS dst,
                rel.type AS type,
                rel.weight AS weight,
                rel.span_id AS span_id,
                properties(rel) AS metadata
            """

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
            max_length: Maximum path length (1-10, validated)

        Returns:
            List of paths (each path is a list of entity names)
        """
        # SECURITY: Validate max_length to prevent Cypher injection
        # Neo4j doesn't support parameterized path lengths, so we must validate input
        if not isinstance(max_length, int) or max_length < 1 or max_length > 10:
            raise ValueError(f"max_length must be integer between 1-10, got: {max_length}")

        # SECURITY: max_length is validated above (integer 1-10), safe to interpolate
        query = f"""
        MATCH path = (src:Entity {{name: $src}})-[*1..{max_length}]->(dst:Entity {{name: $dst}})
        RETURN [node IN nodes(path) | node.name] AS path
        LIMIT 100
        """

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

    # ========================================================================
    # MemoryStore Protocol Implementation
    # ========================================================================

    async def store(self, memory, user_id: str = "default") -> str:
        """
        Store a Memory object as a node in the Neo4j graph.

        Creates a Memory node with MENTIONS edges to entities.

        Args:
            memory: Memory object to store
            user_id: User identifier for multi-tenant support

        Returns:
            The memory ID
        """
        from datetime import datetime

        # Convert timestamp to ISO format
        timestamp_iso = (
            memory.timestamp.isoformat()
            if isinstance(memory.timestamp, datetime)
            else str(memory.timestamp)
        )

        # Extract entities from context
        entities = memory.context.get('entities', [])

        # Store memory node
        query = """
        CREATE (m:Memory {
            id: $memory_id,
            text: $text,
            timestamp: $timestamp,
            user_id: $user_id
        })
        SET m.context = $context
        SET m.metadata = $metadata
        RETURN m.id AS memory_id
        """

        with self.driver.session(database=self.config.database) as session:
            # Create memory node
            session.run(
                query,
                memory_id=memory.id,
                text=memory.text,
                timestamp=timestamp_iso,
                user_id=user_id,
                context=json.dumps(memory.context),
                metadata=json.dumps(memory.metadata)
            )

            # Create MENTIONS edges to entities
            if entities:
                entity_query = """
                MATCH (m:Memory {id: $memory_id})
                UNWIND $entities AS entity_name
                MERGE (e:Entity {name: entity_name})
                CREATE (m)-[r:MENTIONS {weight: 1.0}]->(e)
                """
                session.run(entity_query, memory_id=memory.id, entities=entities)

            # Connect to time thread if timestamp exists
            if memory.timestamp:
                dt = to_utc_datetime(memory.timestamp)
                bucket = time_bucket(dt)
                thread_id = f"time::{bucket}"

                time_query = """
                MATCH (m:Memory {id: $memory_id})
                MERGE (t:TimeThread {name: $thread_id, bucket: $bucket})
                CREATE (m)-[r:OCCURRED_AT {
                    weight: 1.0,
                    bucket: $bucket,
                    timestamp: $timestamp
                }]->(t)
                """
                session.run(
                    time_query,
                    memory_id=memory.id,
                    thread_id=thread_id,
                    bucket=bucket,
                    timestamp=timestamp_iso
                )

        return memory.id

    async def store_many(self, memories: List, user_id: str = "default") -> List[str]:
        """
        Batch store multiple memories.

        More efficient than individual stores due to batched transactions.

        Args:
            memories: List of Memory objects
            user_id: User identifier

        Returns:
            List of memory IDs
        """
        ids = []
        batch_size = 100

        # Process in batches for better performance
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]

            for memory in batch:
                memory_id = await self.store(memory, user_id)
                ids.append(memory_id)

        return ids

    async def recall(self, query, limit: int = 5):
        """
        Retrieve memories matching a query using Cypher graph traversal.

        Uses entity overlap scoring strategy:
        1. Extract entities from query
        2. Find memories that mention overlapping entities
        3. Score by entity overlap ratio
        4. Return top-k scored memories

        Args:
            query: MemoryQuery or string query
            limit: Maximum number of memories to return

        Returns:
            RetrievalResult with memories, scores, and metadata
        """
        from hololoom.core.memory.protocol import Memory, RetrievalResult
        from datetime import datetime

        # Extract query text
        query_text = query.text if hasattr(query, 'text') else str(query)

        # Simple entity extraction (could be enhanced with NLP)
        def extract_entities_simple(text: str) -> List[str]:
            """Extract potential entities from text using basic heuristics."""
            words = text.lower().split()
            # Filter out common stop words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
            return [w for w in words if len(w) > 2 and w not in stop_words]

        query_entities = extract_entities_simple(query_text)

        # Build Cypher query for entity-based recall
        cypher_query = """
        // Find all Memory nodes
        MATCH (m:Memory)

        // Get entities this memory mentions
        OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)

        // Group by memory and collect entities
        WITH m, collect(e.name) AS memory_entities

        // Calculate overlap with query entities
        WITH m, memory_entities,
             size([entity IN $query_entities WHERE entity IN memory_entities]) AS overlap,
             size($query_entities) AS query_size

        // Calculate score (handle division by zero)
        WITH m, memory_entities,
             CASE
                 WHEN query_size > 0 THEN toFloat(overlap) / query_size
                 ELSE 0.5
             END AS score

        // Sort by score and limit
        ORDER BY score DESC
        LIMIT $limit

        // Return memory data
        RETURN
            m.id AS id,
            m.text AS text,
            m.timestamp AS timestamp,
            m.context AS context,
            m.metadata AS metadata,
            score
        """

        scored_memories = []

        with self.driver.session(database=self.config.database) as session:
            result = session.run(
                cypher_query,
                query_entities=query_entities,
                limit=limit
            )

            for record in result:
                # Parse JSON context and metadata
                try:
                    context = json.loads(record["context"]) if record.get("context") else {}
                except (json.JSONDecodeError, TypeError):
                    context = {}

                try:
                    metadata = json.loads(record["metadata"]) if record.get("metadata") else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Parse timestamp
                timestamp_str = record.get("timestamp")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
                except (ValueError, TypeError):
                    timestamp = datetime.now()

                # Create Memory object
                memory = Memory(
                    id=record["id"],
                    text=record["text"],
                    timestamp=timestamp,
                    context=context,
                    metadata=metadata
                )

                score = record["score"] or 0.0
                scored_memories.append((memory, score))

        # Extract memories and scores
        memories = [m for m, _ in scored_memories]
        scores = [s for _, s in scored_memories]

        # Count total memories for metadata
        count_query = "MATCH (m:Memory) RETURN count(m) AS total"
        with self.driver.session(database=self.config.database) as session:
            count_result = session.run(count_query)
            total_memories = count_result.single()["total"]

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used="neo4j_entity_overlap",
            metadata={
                'query_entities': query_entities,
                'total_memories': total_memories,
                'backend': 'neo4j'
            }
        )


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
