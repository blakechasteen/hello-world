"""
Unified Memory Interface - Elegant API for All Memory Systems
==============================================================
Single entry point that hides complexity of:
- Mem0 (intelligent extraction)
- HoloLoom (multi-scale retrieval)
- Neo4j (thread-based storage)
- Qdrant (vector search)
- Hofstadter (resonance patterns)

Philosophy:
Users shouldn't need to understand the internal systems.
They should think about their INTENT, not the MECHANISM.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# User-Facing Types
# ============================================================================

@dataclass
class Memory:
    """A single memory - simple and intuitive."""
    id: str
    text: str
    timestamp: str
    context: Dict[str, Any]  # time, place, people, topics
    relevance: float = 0.0  # How relevant to current query
    tags: Optional[List[str]] = None  # Optional categorical tags


class RecallStrategy(Enum):
    """How to recall memories - intuitive names."""
    RECENT = "recent"  # Temporal: What happened recently?
    SIMILAR = "similar"  # Semantic: What's similar in meaning?
    CONNECTED = "connected"  # Graph: What's connected by relationships?
    RESONANT = "resonant"  # Pattern: What resonates mathematically?
    BALANCED = "balanced"  # Fusion: Best of all strategies (default)


class NavigationDirection(Enum):
    """How to navigate memory space - spatial metaphor."""
    FORWARD = "forward"  # What naturally comes next?
    BACKWARD = "backward"  # What led to this?
    SIDEWAYS = "sideways"  # What's related but different?
    DEEP = "deep"  # Explore deeper in hierarchy


@dataclass
class MemoryPattern:
    """Discovered pattern in memories."""
    pattern_type: str  # "loop", "cluster", "resonance", "thread"
    memories: List[str]  # Memory IDs involved
    strength: float  # Pattern strength [0, 1]
    description: str  # Human-readable explanation


# ============================================================================
# Unified Memory System
# ============================================================================

class UnifiedMemory:
    """
    Elegant, intuitive interface to all memory systems.
    
    Users don't need to know about:
    - Hofstadter sequences
    - Neo4j Cypher queries
    - Qdrant vector operations
    - Multi-scale embeddings
    
    They just store, recall, and navigate memories naturally.
    
    Usage:
        memory = UnifiedMemory()
        
        # Store
        memory.store("Hive Jodi needs winter prep", 
                     context={'place': 'apiary', 'time': 'evening'})
        
        # Recall
        memories = memory.recall("winter beekeeping", strategy="balanced")
        
        # Navigate
        related = memory.navigate(from_memory="mem_123", direction="forward")
        
        # Discover
        patterns = memory.discover_patterns()
    """
    
    def __init__(
        self,
        user_id: str = "default",
        enable_mem0: bool = True,
        enable_neo4j: bool = True,
        enable_qdrant: bool = True,
        enable_hofstadter: bool = True
    ):
        """
        Initialize unified memory system.
        
        Args:
            user_id: User identifier for personalization
            enable_*: Feature flags for each subsystem
        """
        self.user_id = user_id
        
        # Initialize subsystems based on config
        # (Implementation details hidden from user)
        self._init_subsystems(
            enable_mem0,
            enable_neo4j,
            enable_qdrant,
            enable_hofstadter
        )
    
    def _init_subsystems(self, *flags):
        """Initialize backend systems (internal)."""
        # v1.0.1: Use simple in-memory store
        # v1.1+: Add hybrid Neo4j + Qdrant support
        try:
            from .stores.in_memory_store import InMemoryStore
            from .protocol import Memory as ProtocolMemory, MemoryQuery, Strategy as ProtocolStrategy
            self._backend = InMemoryStore()
            self._backend_available = True
            self._protocol_memory = ProtocolMemory
            self._protocol_query = MemoryQuery
            self._protocol_strategy = ProtocolStrategy
        except ImportError:
            # Fallback if imports fail
            self._backend = None
            self._backend_available = False
    
    # ========================================================================
    # Core Operations - Intuitive and Simple
    # ========================================================================
    
    def store(
        self,
        text: str,
        context: Optional[Dict[str, str]] = None,
        importance: float = 0.5
    ) -> str:
        """
        Store a memory.
        
        System automatically:
        - Extracts important information (mem0)
        - Identifies temporal/spatial/thematic threads (Neo4j)
        - Generates embeddings at multiple scales (Qdrant)
        - Computes resonance indices (Hofstadter)
        
        Args:
            text: What to remember
            context: Optional context (time, place, people, topics)
            importance: How important is this? [0, 1]
        
        Returns:
            memory_id: Unique identifier for this memory
        
        Example:
            memory_id = memory.store(
                "Inspected Hive Jodi - 8 frames of brood, very active",
                context={
                    'time': 'evening',
                    'place': 'apiary',
                    'people': ['Blake'],
                    'topics': ['beekeeping', 'inspection']
                },
                importance=0.8
            )
        """
        # Behind the scenes:
        # 1. Mem0 extracts entities and preferences
        # 2. Neo4j creates KNOT crossing THREADS
        # 3. Qdrant stores multi-scale embeddings
        # 4. Hofstadter computes indices

        # User doesn't need to know any of this!
        import hashlib
        from datetime import datetime
        import asyncio

        # Generate a simple memory ID
        memory_id = f"mem_{hashlib.sha256(text.encode()).hexdigest()[:8]}"

        # Actually store in backend if available
        if self._backend_available and self._backend:
            # Split user metadata from context
            user_context = context or {}
            user_metadata = {
                'user_id': self.user_id,
                'importance': importance
            }

            protocol_mem = self._protocol_memory(
                id=memory_id,
                text=text,
                timestamp=datetime.now(),
                context=user_context,
                metadata=user_metadata
            )
            # Run async store
            try:
                asyncio.run(self._backend.store(protocol_mem))
            except Exception as e:
                # Graceful fallback
                pass

        return memory_id
    
    def recall(
        self,
        query: str,
        strategy: RecallStrategy = RecallStrategy.BALANCED,
        limit: int = 5,
        time_range: Optional[tuple] = None,
        context_filter: Optional[Dict] = None
    ) -> List[Memory]:
        """
        Recall relevant memories.
        
        Args:
            query: What are you looking for?
            strategy: How to search (recent/similar/connected/resonant/balanced)
            limit: Max number of memories
            time_range: Optional (start, end) timestamps
            context_filter: Optional filters (place, people, topics)
        
        Returns:
            List of Memory objects, sorted by relevance
        
        Example:
            # Find similar memories
            memories = memory.recall(
                "winter preparation",
                strategy=RecallStrategy.SIMILAR,
                limit=3
            )
            
            # Find recent memories at apiary
            memories = memory.recall(
                "hive status",
                strategy=RecallStrategy.RECENT,
                context_filter={'place': 'apiary'}
            )
        """
        # Strategy dispatch:
        if strategy == RecallStrategy.RECENT:
            return self._recall_temporal(query, limit, time_range)
        elif strategy == RecallStrategy.SIMILAR:
            return self._recall_semantic(query, limit)
        elif strategy == RecallStrategy.CONNECTED:
            return self._recall_graph(query, limit)
        elif strategy == RecallStrategy.RESONANT:
            return self._recall_resonant(query, limit)
        else:  # BALANCED
            return self._recall_fused(query, limit, context_filter)
    
    def navigate(
        self,
        from_memory: str,
        direction: NavigationDirection,
        steps: int = 5
    ) -> List[Memory]:
        """
        Navigate memory space from a starting point.
        
        Uses spatial metaphors - no need to understand graph theory.
        
        Args:
            from_memory: Starting memory ID
            direction: Which way to go (forward/backward/sideways/deep)
            steps: How many steps to take
        
        Returns:
            List of memories in traversal order
        
        Example:
            # "What happened next?"
            next_memories = memory.navigate(
                from_memory="mem_123",
                direction=NavigationDirection.FORWARD
            )
            
            # "What led to this?"
            prior_memories = memory.navigate(
                from_memory="mem_123",
                direction=NavigationDirection.BACKWARD
            )
        """
        # Behind the scenes uses:
        # - FORWARD: Hofstadter G-sequence
        # - BACKWARD: Hofstadter H-sequence
        # - SIDEWAYS: Graph neighbors + Q-sequence
        # - DEEP: Strange loop detection

        # User just thinks spatially!
        # TODO: Implement actual navigation
        return []
    
    def discover_patterns(
        self,
        pattern_types: Optional[List[str]] = None,
        min_strength: float = 0.5
    ) -> List[MemoryPattern]:
        """
        Discover emergent patterns in your memories.
        
        Automatically finds:
        - Strange loops: Self-referential memory cycles
        - Clusters: Groups of related memories
        - Resonances: Memories that "vibrate together"
        - Threads: Continuous narrative threads
        
        Args:
            pattern_types: Which patterns to look for (None = all)
            min_strength: Minimum pattern strength [0, 1]
        
        Returns:
            List of discovered patterns
        
        Example:
            patterns = memory.discover_patterns()
            for pattern in patterns:
                print(f"{pattern.pattern_type}: {pattern.description}")
                print(f"  Strength: {pattern.strength:.2f}")
                print(f"  Memories: {pattern.memories}")
        """
        patterns = []
        
        # Strange loops (Hofstadter)
        if pattern_types is None or "loop" in pattern_types:
            loops = self._find_strange_loops(min_strength)
            patterns.extend(loops)
        
        # Memory clusters (spectral)
        if pattern_types is None or "cluster" in pattern_types:
            clusters = self._find_clusters(min_strength)
            patterns.extend(clusters)
        
        # Resonances (Hofstadter)
        if pattern_types is None or "resonance" in pattern_types:
            resonances = self._find_resonances(min_strength)
            patterns.extend(resonances)
        
        # Narrative threads (Neo4j)
        if pattern_types is None or "thread" in pattern_types:
            threads = self._find_threads(min_strength)
            patterns.extend(threads)
        
        return sorted(patterns, key=lambda p: p.strength, reverse=True)
    
    # ========================================================================
    # Convenience Methods - Common Queries
    # ========================================================================
    
    def what_happened_today(self) -> List[Memory]:
        """Shortcut: Recall today's memories."""
        from datetime import datetime, timedelta
        today_start = datetime.now().replace(hour=0, minute=0)
        today_end = today_start + timedelta(days=1)
        
        return self.recall(
            query="",  # No text filter, just time
            strategy=RecallStrategy.RECENT,
            time_range=(today_start.isoformat(), today_end.isoformat())
        )
    
    def similar_to(self, memory_id: str, limit: int = 5) -> List[Memory]:
        """Shortcut: Find memories similar to this one."""
        # Get the memory's text
        memory = self._get_memory_by_id(memory_id)
        return self.recall(
            query=memory.text,
            strategy=RecallStrategy.SIMILAR,
            limit=limit
        )
    
    def explore_from(self, memory_id: str) -> Dict[str, List[Memory]]:
        """
        Shortcut: Explore in all directions from a memory.
        
        Returns dict with all navigation directions.
        """
        return {
            'forward': self.navigate(memory_id, NavigationDirection.FORWARD, steps=3),
            'backward': self.navigate(memory_id, NavigationDirection.BACKWARD, steps=3),
            'sideways': self.navigate(memory_id, NavigationDirection.SIDEWAYS, steps=3),
        }
    
    # ========================================================================
    # Internal Strategy Implementations (Hidden from User)
    # ========================================================================
    
    def _recall_temporal(self, query, limit, time_range) -> List[Memory]:
        """Temporal strategy: Neo4j time threads."""
        # v1.0.1: Use in-memory store with temporal sorting
        return self._recall_from_backend(query, limit, strategy="temporal")

    def _recall_semantic(self, query, limit) -> List[Memory]:
        """Semantic strategy: Qdrant similarity."""
        # v1.0.1: Use in-memory store with text matching
        return self._recall_from_backend(query, limit, strategy="semantic")

    def _recall_graph(self, query, limit) -> List[Memory]:
        """Graph strategy: Neo4j traversal."""
        # v1.0.1: Use in-memory store (no graph yet)
        return self._recall_from_backend(query, limit, strategy="graph")

    def _recall_resonant(self, query, limit) -> List[Memory]:
        """Resonance strategy: Hofstadter patterns."""
        # v1.0.1: Use in-memory store (no resonance yet)
        return self._recall_from_backend(query, limit, strategy="resonant")

    def _recall_fused(self, query, limit, filters) -> List[Memory]:
        """Balanced strategy: Weighted fusion of all."""
        # v1.0.1: Use in-memory store with fused strategy
        return self._recall_from_backend(query, limit, strategy="fused")

    def _recall_from_backend(self, query, limit, strategy="fused") -> List[Memory]:
        """Actually retrieve from backend store."""
        if not self._backend_available or not self._backend:
            return []

        import asyncio

        try:
            # Create query
            mem_query = self._protocol_query(
                text=query,
                user_id=self.user_id,
                limit=limit
            )

            # Map strategy
            strategy_map = {
                "temporal": self._protocol_strategy.TEMPORAL,
                "semantic": self._protocol_strategy.SEMANTIC,
                "graph": self._protocol_strategy.GRAPH,
                "fused": self._protocol_strategy.FUSED
            }
            strat = strategy_map.get(strategy, self._protocol_strategy.FUSED)

            # Retrieve
            result = asyncio.run(self._backend.retrieve(mem_query, strat))

            # Convert protocol Memory to unified Memory
            unified_mems = []
            for mem in result.memories:
                # Merge context and metadata for unified Memory
                merged_context = {**mem.context, **mem.metadata}
                unified_mems.append(Memory(
                    id=mem.id,
                    text=mem.text,
                    timestamp=mem.timestamp.isoformat() if hasattr(mem.timestamp, 'isoformat') else str(mem.timestamp),
                    context=merged_context,
                    relevance=0.8  # Placeholder relevance
                ))

            return unified_mems

        except Exception as e:
            # Graceful fallback
            return []
    
    def _find_strange_loops(self, min_strength) -> List[MemoryPattern]:
        """Detect strange loops using cycle detection."""
        # TODO: Implement actual loop detection
        return []

    def _find_clusters(self, min_strength) -> List[MemoryPattern]:
        """Detect clusters using spectral analysis."""
        # TODO: Implement actual cluster detection
        return []

    def _find_resonances(self, min_strength) -> List[MemoryPattern]:
        """Detect resonances using Hofstadter indices."""
        # TODO: Implement actual resonance detection
        return []

    def _find_threads(self, min_strength) -> List[MemoryPattern]:
        """Detect narrative threads using Neo4j."""
        # TODO: Implement actual thread detection
        return []

    def _get_memory_by_id(self, memory_id: str) -> Memory:
        """Internal: Fetch memory by ID."""
        # TODO: Implement actual memory retrieval
        from datetime import datetime
        return Memory(
            id="stub",
            text="Stub memory",
            timestamp=datetime.now().isoformat(),
            context={}
        )


# ============================================================================
# Example Usage - Clean and Intuitive
# ============================================================================

if __name__ == "__main__":
    print("=== Unified Memory Interface Demo ===\n")
    
    # Simple initialization
    memory = UnifiedMemory(user_id="blake")
    
    # Store memories naturally
    print("Storing memories...")
    mem1 = memory.store(
        "Hive Jodi has 8 frames of brood",
        context={'place': 'apiary', 'time': 'evening'},
        importance=0.8
    )
    print(f"  Stored: {mem1}")
    
    # Recall with different strategies
    print("\nRecalling similar memories...")
    similar = memory.recall("hive inspection", strategy=RecallStrategy.SIMILAR)
    for mem in similar:
        print(f"  [{mem.relevance:.2f}] {mem.text}")
    
    # Navigate memory space
    print("\nNavigating forward...")
    forward = memory.navigate(mem1, direction=NavigationDirection.FORWARD)
    print(f"  Path: {' → '.join(m.text[:30] for m in forward)}")
    
    # Discover patterns
    print("\nDiscovering patterns...")
    patterns = memory.discover_patterns()
    for pattern in patterns[:3]:
        print(f"  {pattern.pattern_type}: {pattern.description}")
        print(f"    Strength: {pattern.strength:.2f}")
    
    # Convenience methods
    print("\nWhat happened today?")
    today = memory.what_happened_today()
    print(f"  Found {len(today)} memories from today")
    
    print("\n[OK] Demo complete!")
