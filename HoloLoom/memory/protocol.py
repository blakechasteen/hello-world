"""
Memory Protocols - Protocol-based interfaces for memory backends.
All backends implement MemoryStore protocol for easy extension.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

# ============================================================================
# Data Types
# ============================================================================


@dataclass
class Memory:
    """Single memory unit. Compatible with MemoryShard from SpinningWheel."""

    id: str
    text: str
    timestamp: datetime
    context: dict[str, Any]
    metadata: dict[str, Any]
    embedding: Any | None = None  # Vector embedding (numpy array or list)

    @classmethod
    def from_shard(cls, shard: Any, timestamp: datetime | None = None) -> "Memory":
        """Create Memory from MemoryShard (SpinningWheel output)."""
        return cls(
            id=shard.id,
            text=shard.text,
            timestamp=timestamp or datetime.now(),
            context={
                "episode": getattr(shard, "episode", None),
                "entities": getattr(shard, "entities", []),
                "motifs": getattr(shard, "motifs", []),
            },
            metadata=getattr(shard, "metadata", None) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "metadata": self.metadata,
        }
        # Convert embedding to list for JSON serialization
        if self.embedding is not None:
            import numpy as np

            if isinstance(self.embedding, np.ndarray):
                result["embedding"] = self.embedding.tolist()
            else:
                result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        data = data.copy()
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        # Convert embedding back to numpy array
        if "embedding" in data and data["embedding"] is not None:
            import numpy as np

            if isinstance(data["embedding"], list):
                data["embedding"] = np.array(data["embedding"])
        return cls(**data)


@dataclass
class MemoryQuery:
    """Memory query."""

    text: str
    user_id: str = "default"
    limit: int = 5
    filters: dict[str, Any] | None = None
    strategy: Optional["Strategy"] = None


@dataclass
class RetrievalResult:
    """Retrieval results with scores and metadata."""

    memories: list[Memory]
    scores: list[float]
    strategy_used: str
    metadata: dict[str, Any]


class Strategy(Enum):
    """Retrieval strategies."""

    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    GRAPH = "graph"
    PATTERN = "pattern"
    FUSED = "fused"
    BALANCED = "balanced"


class QueryMode(Enum):
    """Query complexity modes."""

    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"
    RESEARCH = "research"


# ============================================================================
# Core Protocol
# ============================================================================

try:
    from HoloLoom.protocols import MemoryStore
except ImportError:
    # Fallback if canonical protocol unavailable
    @runtime_checkable
    class MemoryStore(Protocol):
        """Memory storage backend protocol."""

        async def store(self, memory: Memory, user_id: str = "default") -> str: ...
        async def store_many(
            self, memories: list[Memory], user_id: str = "default"
        ) -> list[str]: ...
        async def get_by_id(self, memory_id: str) -> Memory | None: ...
        async def retrieve(
            self, query: MemoryQuery, strategy: Strategy = Strategy.FUSED
        ) -> RetrievalResult: ...
        async def delete(self, memory_id: str) -> bool: ...
        async def health_check(self) -> dict[str, Any]: ...


# ============================================================================
# Helper Functions
# ============================================================================


def shards_to_memories(shards: list[Any], timestamp: datetime | None = None) -> list[Memory]:
    """Convert SpinningWheel shards to Memory objects."""
    return [Memory.from_shard(shard, timestamp) for shard in shards]


async def create_unified_memory(
    user_id: str = "default",
    backend: str | None = None,
    config: dict[str, Any] | None = None,
    **kwargs,
):
    """
    Backwards-compatible async factory with backend detection and config support.

    Many modules import `create_unified_memory` from this protocol module.
    This enhanced factory:
    - Auto-detects available backends (Neo4j, Qdrant, in-memory)
    - Supports explicit backend selection
    - Handles configuration files
    - Provides graceful degradation with logging
    - Constructs UnifiedMemory from HoloLoom.memory.unified

    Args:
        user_id: User identifier passed to UnifiedMemory constructor
        backend: Optional explicit backend ("neo4j", "qdrant", "in-memory", "hybrid")
                 If None, auto-detects best available backend
        config: Optional config dict with keys:
                - neo4j_uri: Neo4j connection URI
                - qdrant_url: Qdrant server URL
                - enable_mem0: Enable mem0 extraction (default: True)
                - enable_hofstadter: Enable Hofstadter patterns (default: True)
        **kwargs: Additional args forwarded to UnifiedMemory

    Returns:
        UnifiedMemory instance configured with best available backend

    Examples:
        # Auto-detect backend
        memory = await create_unified_memory(user_id="blake")

        # Explicit backend
        memory = await create_unified_memory(
            user_id="blake",
            backend="neo4j",
            config={"neo4j_uri": "bolt://localhost:7687"}
        )

        # In-memory (testing/development)
        memory = await create_unified_memory(
            user_id="test",
            backend="in-memory"
        )
    """
    import logging
    import os

    logger = logging.getLogger(__name__)
    config = config or {}

    # Try to import UnifiedMemory
    try:
        from HoloLoom.memory.unified import UnifiedMemory
    except ImportError as e:
        raise ImportError(
            f"UnifiedMemory implementation not available: {e}\n"
            "Ensure HoloLoom.memory.unified exists or pass a memory backend directly."
        )

    # Backend detection and configuration
    enable_neo4j = True
    enable_qdrant = True
    enable_mem0 = config.get("enable_mem0", True)
    enable_hofstadter = config.get("enable_hofstadter", True)

    if backend:
        # Explicit backend selection
        backend_lower = backend.lower()

        if backend_lower == "in-memory":
            # Disable external backends
            enable_neo4j = False
            enable_qdrant = False
            logger.info("Using in-memory backend (no persistence)")

        elif backend_lower == "neo4j":
            enable_qdrant = False
            logger.info("Using Neo4j backend")

        elif backend_lower == "qdrant":
            enable_neo4j = False
            logger.info("Using Qdrant backend")

        elif backend_lower == "hybrid":
            # Use both Neo4j and Qdrant
            logger.info("Using hybrid Neo4j + Qdrant backend")

        else:
            logger.warning(f"Unknown backend '{backend}', falling back to auto-detect")

    else:
        # Auto-detect available backends
        logger.info("Auto-detecting available memory backends...")

        # Check Neo4j availability
        neo4j_uri = config.get("neo4j_uri") or os.getenv("NEO4J_URI")
        if not neo4j_uri:
            enable_neo4j = False
            logger.debug("Neo4j not configured (no URI)")
        else:
            # Try to connect (health check)
            try:
                from neo4j import GraphDatabase

                neo4j_user = config.get("neo4j_user") or os.getenv("NEO4J_USER", "neo4j")
                neo4j_password = config.get("neo4j_password") or os.getenv("NEO4J_PASSWORD")

                driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                # Verify connection with a simple query
                with driver.session() as session:
                    session.run("RETURN 1")
                driver.close()

                enable_neo4j = True
                logger.info(f"✓ Neo4j connection verified at {neo4j_uri}")
            except ImportError:
                enable_neo4j = False
                logger.debug("Neo4j driver not installed (pip install neo4j)")
            except Exception as e:
                enable_neo4j = False
                logger.warning(f"Neo4j unavailable: {e}")

        # Check Qdrant availability
        qdrant_url = config.get("qdrant_url") or os.getenv("QDRANT_URL")
        if not qdrant_url:
            enable_qdrant = False
            logger.debug("Qdrant not configured (no URL)")
        else:
            # Try to connect (health check)
            try:
                from qdrant_client import QdrantClient

                # Parse URL to determine if local or remote
                if qdrant_url.startswith("http"):
                    client = QdrantClient(url=qdrant_url)
                else:
                    # Local path
                    client = QdrantClient(path=qdrant_url)

                # Verify connection by getting collections
                collections = client.get_collections()

                enable_qdrant = True
                logger.info(
                    f"✓ Qdrant connection verified at {qdrant_url} ({len(collections.collections)} collections)"
                )
            except ImportError:
                enable_qdrant = False
                logger.debug("Qdrant client not installed (pip install qdrant-client)")
            except Exception as e:
                enable_qdrant = False
                logger.warning(f"Qdrant unavailable: {e}")

        # Log selected backend
        if enable_neo4j and enable_qdrant:
            logger.info("✓ Using hybrid Neo4j + Qdrant backend")
        elif enable_neo4j:
            logger.info("✓ Using Neo4j backend")
        elif enable_qdrant:
            logger.info("✓ Using Qdrant backend")
        else:
            logger.info("✓ Using in-memory backend (fallback)")

    # Construct UnifiedMemory with detected/configured backends
    try:
        memory = UnifiedMemory(
            user_id=user_id,
            enable_mem0=enable_mem0,
            enable_neo4j=enable_neo4j,
            enable_qdrant=enable_qdrant,
            enable_hofstadter=enable_hofstadter,
            **kwargs,
        )

        logger.info(f"✓ UnifiedMemory initialized for user '{user_id}'")
        return memory

    except Exception as e:
        logger.error(f"Failed to initialize UnifiedMemory: {e}")
        raise RuntimeError(
            f"UnifiedMemory initialization failed: {e}\n"
            "Check backend configuration and dependencies."
        )
