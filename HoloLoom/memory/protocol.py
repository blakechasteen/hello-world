"""
Memory System Protocols - Best Practice Architecture
====================================================
Protocol-based design following HoloLoom standards.

Philosophy:
- Define WHAT, not HOW (protocols define interface)
- Dependency injection (orchestrator doesn't know concrete implementations)
- Graceful degradation (missing backends don't break system)
- Async-first (non-blocking operations)
- Testable (protocols are easily mocked)

Inspiration:
- holoLoom/policy/unified.py → PolicyEngine protocol
- holoLoom/Modules/Features.py → MotifDetector/Embedder protocols
- Model Context Protocol → Resource exposure pattern
"""

from typing import List, Dict, Optional, Protocol, runtime_checkable, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


# ============================================================================
# Core Data Types (Simple and Clean)
# ============================================================================

@dataclass
class Memory:
    """
    A single memory - pure data.

    This is compatible with MemoryShard from SpinningWheel,
    making data piping obvious and clean!
    """
    id: str
    text: str
    timestamp: datetime
    context: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def from_shard(cls, shard: Any, timestamp: Optional[datetime] = None) -> 'Memory':
        """
        Create Memory from MemoryShard (SpinningWheel output).

        **This is the bridge that makes data piping obvious!**

        Spinner → MemoryShard → Memory → MemoryStore

        Args:
            shard: MemoryShard from SpinningWheel
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Memory object ready for storage

        Example:
            # Get shards from spinner
            shards = await spin_text("My document")

            # Convert to memories
            memories = [Memory.from_shard(s) for s in shards]

            # Store
            await memory.store_many(memories)
        """
        if timestamp is None:
            timestamp = datetime.now()

        return cls(
            id=shard.id,
            text=shard.text,
            timestamp=timestamp,
            context={
                'episode': getattr(shard, 'episode', None),
                'entities': getattr(shard, 'entities', []),
                'motifs': getattr(shard, 'motifs', []),
            },
            metadata=getattr(shard, 'metadata', None) or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'text': self.text,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Deserialize from dictionary."""
        data = data.copy()
        # Parse timestamp if string
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class MemoryQuery:
    """A memory query - pure data."""
    text: str
    user_id: str = "default"
    limit: int = 5
    filters: Optional[Dict[str, Any]] = None
    strategy: Optional['Strategy'] = None  # Optional retrieval strategy hint


@dataclass
class RetrievalResult:
    """Results from memory retrieval - pure data."""
    memories: List[Memory]
    scores: List[float]
    strategy_used: str
    metadata: Dict[str, Any]


class Strategy(Enum):
    """Retrieval strategies."""
    TEMPORAL = "temporal"  # Recent memories
    SEMANTIC = "semantic"  # Meaning similarity
    GRAPH = "graph"  # Relationship traversal
    PATTERN = "pattern"  # Mathematical patterns
    FUSED = "fused"  # Weighted fusion
    BALANCED = "balanced"  # Balanced retrieval (default)


# ============================================================================
# Memory Store Protocol - The Core Interface
# ============================================================================

@runtime_checkable
class MemoryStore(Protocol):
    """
    Protocol for memory storage backends.
    
    All memory stores (Mem0, Neo4j, Qdrant, HoloLoom) implement this.
    The orchestrator doesn't know which implementation it's using.
    
    Examples of implementations:
    - Mem0MemoryStore: User-specific, LLM-extracted memories
    - Neo4jMemoryStore: Graph-based, thread-model storage
    - QdrantMemoryStore: Vector-based similarity search
    - HoloLoomMemoryStore: Multi-scale, domain-aware retrieval
    - HybridMemoryStore: Fusion of multiple stores
    """
    
    async def store(self, memory: Memory) -> str:
        """
        Store a memory.

        Args:
            memory: Memory to store

        Returns:
            memory_id: Unique identifier
        """
        ...

    async def store_many(self, memories: List[Memory]) -> List[str]:
        """
        Store multiple memories (batch operation).

        Perfect for SpinningWheel outputs that produce many shards!

        Args:
            memories: List of Memory objects

        Returns:
            List of memory IDs
        """
        ...

    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Unique identifier

        Returns:
            Memory object or None if not found
        """
        ...

    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories matching query.
        
        Args:
            query: What to search for
            strategy: How to search
            
        Returns:
            RetrievalResult with memories and scores
        """
        ...
    
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check backend health.
        
        Returns:
            Dict with status, latency, memory count, etc.
        """
        ...


# ============================================================================
# Memory Navigator Protocol - Spatial Operations
# ============================================================================

@runtime_checkable
class MemoryNavigator(Protocol):
    """
    Protocol for memory navigation.
    
    Enables spatial traversal of memory space without exposing
    the underlying mechanism (Hofstadter, graphs, etc.)
    """
    
    async def navigate_forward(
        self,
        from_memory_id: str,
        steps: int = 5
    ) -> List[Memory]:
        """Navigate forward in time/sequence."""
        ...
    
    async def navigate_backward(
        self,
        from_memory_id: str,
        steps: int = 5
    ) -> List[Memory]:
        """Navigate backward in time/sequence."""
        ...
    
    async def find_neighbors(
        self,
        memory_id: str,
        radius: int = 2
    ) -> List[Memory]:
        """Find nearby memories in graph/vector space."""
        ...


# ============================================================================
# Pattern Detector Protocol - Emergent Structure Discovery
# ============================================================================

@dataclass
class MemoryPattern:
    """Discovered pattern in memories."""
    type: str  # "loop", "cluster", "resonance", "thread"
    memories: List[str]  # Memory IDs
    strength: float  # [0, 1]
    description: str


@runtime_checkable
class PatternDetector(Protocol):
    """
    Protocol for pattern detection.
    
    Implementations can use different algorithms:
    - Strange loop detection (Hofstadter)
    - Spectral clustering
    - Thread analysis (Neo4j)
    - Resonance patterns
    """
    
    async def detect_patterns(
        self,
        min_strength: float = 0.5,
        pattern_types: Optional[List[str]] = None
    ) -> List[MemoryPattern]:
        """
        Detect emergent patterns in memories.
        
        Args:
            min_strength: Minimum pattern strength
            pattern_types: Which types to detect (None = all)
            
        Returns:
            List of discovered patterns
        """
        ...


# ============================================================================
# Unified Memory Interface - Facade Pattern
# ============================================================================

@dataclass
class UnifiedMemoryInterface:
    """
    Unified memory interface using protocol-based dependency injection.

    This is the ONLY class users interact with. It's a facade over:
    - MemoryStore (storage/retrieval)
    - MemoryNavigator (spatial operations)
    - PatternDetector (structure discovery)

    The beauty: Components are INJECTED via protocols.
    The orchestrator doesn't know if it's using Mem0, Neo4j, or Qdrant.

    Usage:
        # Inject implementations (DI pattern)
        store = HybridMemoryStore(...)  # implements MemoryStore protocol
        nav = HofstadterNavigator(...)  # implements MemoryNavigator protocol
        detector = SpectraDetector(...) # implements PatternDetector protocol

        # Create interface
        memory = UnifiedMemoryInterface(
            _store=store,
            navigator=nav,
            detector=detector
        )

        # Use naturally
        memory_id = await memory.store("Hive Jodi prep")
        results = await memory.recall("winter beekeeping")
        forward = await memory.navigate_forward(memory_id)
        patterns = await memory.discover_patterns()
    """

    _store: MemoryStore
    navigator: Optional[MemoryNavigator] = None
    detector: Optional[PatternDetector] = None
    
    # ========================================================================
    # Core Operations - Delegate to Protocols
    # ========================================================================
    
    async def store(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: str = "default"
    ) -> str:
        """Store a memory from text (delegates to MemoryStore protocol)."""
        memory = Memory(
            id="",  # Store will assign
            text=text,
            timestamp=datetime.now(),
            context=context or {},
            metadata={'user_id': user_id}
        )
        return await self._store.store(memory)

    async def store_memory(self, memory: Memory) -> str:
        """
        Store a Memory object directly.

        **This is the bridge for SpinningWheel outputs!**

        Args:
            memory: Memory object (from Memory.from_shard())

        Returns:
            memory_id
        """
        return await self._store.store(memory)

    async def store_many(self, memories: List[Memory]) -> List[str]:
        """
        Store multiple memories (batch operation).

        Perfect for SpinningWheel outputs that produce many shards!

        Args:
            memories: List of Memory objects

        Returns:
            List of memory IDs
        """
        return await self._store.store_many(memories)
    
    async def recall(
        self,
        query: str,
        strategy: Strategy = Strategy.FUSED,
        limit: int = 5,
        user_id: str = "default"
    ) -> RetrievalResult:
        """Recall memories (delegates to MemoryStore protocol)."""
        query_obj = MemoryQuery(
            text=query,
            user_id=user_id,
            limit=limit
        )
        return await self._store.retrieve(query_obj, strategy)
    
    async def navigate_forward(
        self,
        from_memory_id: str,
        steps: int = 5
    ) -> List[Memory]:
        """Navigate forward (delegates to MemoryNavigator protocol)."""
        if self.navigator is None:
            raise RuntimeError("Navigator not configured")
        return await self.navigator.navigate_forward(from_memory_id, steps)
    
    async def navigate_backward(
        self,
        from_memory_id: str,
        steps: int = 5
    ) -> List[Memory]:
        """Navigate backward (delegates to MemoryNavigator protocol)."""
        if self.navigator is None:
            raise RuntimeError("Navigator not configured")
        return await self.navigator.navigate_backward(from_memory_id, steps)
    
    async def discover_patterns(
        self,
        min_strength: float = 0.5
    ) -> List[MemoryPattern]:
        """Discover patterns (delegates to PatternDetector protocol)."""
        if self.detector is None:
            raise RuntimeError("Pattern detector not configured")
        return await self.detector.detect_patterns(min_strength)
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    async def similar_to(self, memory_id: str, limit: int = 5) -> RetrievalResult:
        """Find memories similar to a given one."""
        # Get the memory
        memory = await self._store.get_by_id(memory_id)
        if memory is None:
            return RetrievalResult(
                memories=[],
                scores=[],
                strategy_used="semantic",
                metadata={'error': 'memory_not_found'}
            )

        # Search using its text
        return await self.recall(
            query=memory.text,
            strategy=Strategy.SEMANTIC,
            limit=limit
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        health = {
            'store': await self._store.health_check(),
            'navigator': 'not configured' if self.navigator is None else 'available',
            'detector': 'not configured' if self.detector is None else 'available',
        }
        return health


# ============================================================================
# Factory Function - Creates Interface with Graceful Degradation
# ============================================================================

async def create_unified_memory(
    user_id: str = "default",
    enable_mem0: bool = True,
    enable_neo4j: bool = True,
    enable_qdrant: bool = True,
    enable_patterns: bool = True
) -> UnifiedMemoryInterface:
    """
    Factory function with graceful degradation.
    
    Follows HoloLoom pattern: Try to load optional components,
    fall back gracefully if unavailable.
    
    Args:
        user_id: User identifier
        enable_*: Feature flags
        
    Returns:
        Configured UnifiedMemoryInterface
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Try to create store (required)
    store = None
    backends = []
    
    # Try mem0
    if enable_mem0:
        try:
            try:
                from holoLoom.memory.stores.mem0_store import Mem0MemoryStore
            except ImportError:
                from .stores.mem0_store import Mem0MemoryStore
            backends.append(Mem0MemoryStore(user_id=user_id))
            logger.info("✓ Mem0 store available")
        except ImportError as e:
            logger.warning(f"Mem0 not available: {e}")
    
    # Try neo4j
    if enable_neo4j:
        try:
            try:
                from holoLoom.memory.stores.neo4j_store import Neo4jMemoryStore
            except ImportError:
                from .stores.neo4j_store import Neo4jMemoryStore
            backends.append(Neo4jMemoryStore(password="hololoom123"))
            logger.info("✓ Neo4j store available")
        except ImportError as e:
            logger.warning(f"Neo4j not available: {e}")
    
    # Try qdrant
    if enable_qdrant:
        try:
            try:
                from holoLoom.memory.stores.qdrant_store import QdrantMemoryStore
            except ImportError:
                from .stores.qdrant_store import QdrantMemoryStore
            backends.append(QdrantMemoryStore())
            logger.info("✓ Qdrant store available")
        except ImportError as e:
            logger.warning(f"Qdrant not available: {e}")
    
    # Create hybrid store if multiple backends, otherwise use single
    if len(backends) > 1:
        try:
            from holoLoom.memory.stores.hybrid_store import HybridMemoryStore, BackendConfig
        except ImportError:
            from .stores.hybrid_store import HybridMemoryStore, BackendConfig
        
        # Wrap backends in BackendConfig
        backend_configs = []
        for backend in backends:
            name = backend.__class__.__name__.replace('MemoryStore', '').lower()
            backend_configs.append(BackendConfig(
                store=backend,
                weight=1.0,
                enabled=True,
                name=name
            ))
        
        store = HybridMemoryStore(backends=backend_configs)
        logger.info(f"✓ Hybrid store with {len(backends)} backends")
    elif len(backends) == 1:
        store = backends[0]
        logger.info("✓ Single backend store")
    else:
        # Fall back to in-memory
        try:
            from holoLoom.memory.stores.in_memory_store import InMemoryStore
        except ImportError:
            from .stores.in_memory_store import InMemoryStore
        store = InMemoryStore()
        logger.warning("⚠ Using in-memory store (no persistence)")
    
    # Optional: Navigator
    navigator = None
    try:
        from holoLoom.memory.navigators.hofstadter_nav import HofstadterNavigator
        navigator = HofstadterNavigator(store=store)
        logger.info("✓ Hofstadter navigator available")
    except ImportError:
        logger.warning("⚠ Navigator not available (navigation disabled)")
    
    # Optional: Pattern detector
    detector = None
    if enable_patterns:
        try:
            from holoLoom.memory.detectors.multi_detector import MultiPatternDetector
            detector = MultiPatternDetector(store=store)
            logger.info("✓ Pattern detector available")
        except ImportError:
            logger.warning("⚠ Pattern detector not available")
    
    return UnifiedMemoryInterface(
        _store=store,
        navigator=navigator,
        detector=detector
    )


# ============================================================================
# Utility Functions - Data Piping Helpers
# ============================================================================

def shards_to_memories(shards: List[Any], timestamp: Optional[datetime] = None) -> List[Memory]:
    """
    Convert SpinningWheel shards to Memory objects.

    **This is the data piping bridge!**

    Spinner → Shards → Memories → Store

    Args:
        shards: List of MemoryShard objects from SpinningWheel
        timestamp: Optional timestamp (defaults to now)

    Returns:
        List of Memory objects ready for storage

    Example:
        # Full pipeline: text → spinner → shards → memories → store
        from HoloLoom.spinningWheel.text import spin_text

        # Step 1: Spinner → Shards
        shards = await spin_text("My document")

        # Step 2: Shards → Memories
        memories = shards_to_memories(shards)

        # Step 3: Memories → Store
        await memory.store_many(memories)
    """
    return [Memory.from_shard(shard, timestamp) for shard in shards]


async def pipe_text_to_memory(
    text: str,
    memory: UnifiedMemoryInterface,
    source: str = 'document',
    chunk_by: Optional[str] = None,
    chunk_size: int = 500
) -> List[str]:
    """
    Complete pipeline: text → spinner → memories → store.

    **One-liner to pipe text into memory!**

    Args:
        text: Text content
        memory: UnifiedMemoryInterface instance
        source: Source identifier
        chunk_by: How to chunk ('paragraph', 'sentence', None)
        chunk_size: Chunk size in characters

    Returns:
        List of memory IDs

    Example:
        memory = await create_unified_memory(user_id="blake")

        # Pipe text directly into memory
        ids = await pipe_text_to_memory(
            text="My long document...",
            memory=memory,
            chunk_by='paragraph'
        )

        print(f"Stored {len(ids)} memories")
    """
    from HoloLoom.spinningWheel.text import spin_text

    # Spin text into shards
    shards = await spin_text(
        text=text,
        source=source,
        chunk_by=chunk_by,
        chunk_size=chunk_size
    )

    # Convert to memories
    memories = shards_to_memories(shards)

    # Store
    return await memory.store_many(memories)


# ============================================================================
# Example Usage - Clean and Simple
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("=== Protocol-Based Memory Interface Demo ===\n")
        
        # Factory creates interface with graceful degradation
        memory = await create_unified_memory(user_id="blake")
        
        # Check what's available
        health = await memory.health_check()
        print(f"System health: {health}\n")
        
        # Store memory
        print("Storing memory...")
        mem_id = await memory.store(
            "Hive Jodi has 8 frames of brood",
            context={'place': 'apiary', 'time': 'evening'}
        )
        print(f"  Stored: {mem_id}\n")
        
        # Recall with strategy
        print("Recalling memories...")
        results = await memory.recall(
            "hive inspection",
            strategy=Strategy.FUSED
        )
        print(f"  Found {len(results.memories)} memories")
        print(f"  Strategy: {results.strategy_used}\n")
        
        # Navigate (if available)
        try:
            print("Navigating forward...")
            forward = await memory.navigate_forward(mem_id, steps=3)
            print(f"  Path length: {len(forward)}\n")
        except RuntimeError as e:
            print(f"  {e}\n")
        
        # Discover patterns (if available)
        try:
            print("Discovering patterns...")
            patterns = await memory.discover_patterns(min_strength=0.5)
            print(f"  Found {len(patterns)} patterns\n")
        except RuntimeError as e:
            print(f"  {e}\n")
        
        print("✓ Demo complete!")
    
    asyncio.run(demo())
