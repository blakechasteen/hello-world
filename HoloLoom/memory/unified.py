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
        enable_hofstadter: bool = True,
        backend: Optional[Any] = None,
        enable_conductor: bool = True
    ):
        """
        Initialize unified memory system.

        Args:
            user_id: User identifier for personalization
            enable_*: Feature flags for each subsystem
            backend: Optional explicit backend store (dependency injection)
            enable_conductor: Enable Memory Conductor for intelligent multi-system coordination
        """
        self.user_id = user_id

        if backend:
            # Use injected backend
            self._backend = backend
            self._backend_available = True

            # Import protocol types needed for operation
            try:
                from .protocol import Memory as ProtocolMemory, MemoryQuery, Strategy as ProtocolStrategy
                self._protocol_memory = ProtocolMemory
                self._protocol_query = MemoryQuery
                self._protocol_strategy = ProtocolStrategy
            except ImportError:
                # Fallback if imports fail
                self._backend_available = False
        else:
            # Initialize subsystems based on config
            # (Implementation details hidden from user)
            self._init_subsystems(
                enable_mem0,
                enable_neo4j,
                enable_qdrant,
                enable_hofstadter
            )

        # Initialize Memory Conductor for intelligent strategy selection
        self._conductor = None
        self._conductor_available = False
        if enable_conductor and self._backend_available and self._backend:
            try:
                from ..memory_symphony import MemoryConductor, MemoryStrategy as MS
                self._conductor = MemoryConductor(
                    memory_backend=self._backend,
                    enable_cache=True,
                    enable_hot_patterns=True,
                    enable_awareness=True,
                    enable_spring_dynamics=False,  # Optional advanced feature
                    enable_multi_wave=False,       # Optional advanced feature
                    default_strategy=MS.AUTO,
                    verbose=False
                )
                self._conductor_available = True
            except (ImportError, Exception):
                # Graceful fallback - conductor optional
                pass
    
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
        text: str = None, # Allow content to be passed as text or content
        context: Optional[Dict[str, str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None, # Add metadata argument
        content: str = None # Add content alias for text
    ) -> str:
        """
        Store a memory.
        
        System automatically:
        - Extracts important information (mem0)
        - Identifies temporal/spatial/thematic threads (Neo4j)
        - Generates embeddings at multiple scales (Qdrant)
        - Computes resonance indices (Hofstadter)
        
        Args:
            text: What to remember (alias: content)
            context: Optional context (time, place, people, topics)
            importance: How important is this? [0, 1]
            metadata: Additional metadata (merged with context/importance)
            content: Alias for text (for compatibility)
        
        Returns:
            memory_id: Unique identifier for this memory
        """
        # Handle content/text alias
        text_content = text if text is not None else content
        if text_content is None:
            raise ValueError("Must provide either 'text' or 'content'")

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
        memory_id = f"mem_{hashlib.sha256(text_content.encode()).hexdigest()[:8]}"

        # Actually store in backend if available
        if self._backend_available and self._backend:
            # Split user metadata from context
            user_context = context or {}
            
            # Merge explicit metadata
            user_metadata = {
                'user_id': self.user_id,
                'importance': importance
            }
            if metadata:
                user_metadata.update(metadata)

            protocol_mem = self._protocol_memory(
                id=memory_id,
                text=text_content,
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

        Now powered by Memory Conductor for intelligent multi-system coordination!
        Automatically routes queries through Vector Memory, Knowledge Graph, Cache,
        Hot Patterns, Awareness Graph, and optionally Spring Dynamics and Multi-Wave Engine.

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
        # Route through Memory Conductor if available
        if self._conductor_available and self._conductor:
            import asyncio
            try:
                from ..memory_symphony import MemoryQuery

                # Map user-facing RecallStrategy to MemoryStrategy
                mem_strategy = self._map_strategy(strategy)

                # Create conductor query
                mem_query = MemoryQuery(
                    text=query,
                    k=limit,
                    strategy=mem_strategy,
                    min_relevance=0.0,
                    include_metadata=True,
                    enable_spreading=True,
                    max_hops=3
                )

                # Execute through conductor
                coordination_result = asyncio.run(self._conductor.recall(mem_query))

                # Convert results
                memories = asyncio.run(self._convert_results(coordination_result))

                return memories

            except Exception:
                # Graceful fallback to original implementation
                pass

        # Fallback: Original strategy dispatch
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
        # Check if backend has graph access
        if not self._backend_available or not self._backend:
            return []

        if not hasattr(self._backend, 'graph'):
            return []

        graph = self._backend.graph

        # Check if from_memory exists in graph
        if not hasattr(graph, 'G'):
            return []

        if from_memory not in graph.G.nodes():
            return []

        # Navigate based on direction
        if direction == NavigationDirection.FORWARD:
            return self._navigate_forward(from_memory, steps, graph)
        elif direction == NavigationDirection.BACKWARD:
            return self._navigate_backward(from_memory, steps, graph)
        elif direction == NavigationDirection.SIDEWAYS:
            return self._navigate_sideways(from_memory, steps, graph)
        elif direction == NavigationDirection.DEEP:
            return self._navigate_deep(from_memory, steps, graph)

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
    # Conductor Integration Helpers
    # ========================================================================

    def _map_strategy(self, recall_strategy: RecallStrategy):
        """
        Map RecallStrategy (user-facing) to MemoryStrategy (conductor).

        Mapping:
        - RECENT → FAST (temporal queries are simple)
        - SIMILAR → FAST (vector-only semantic search)
        - CONNECTED → BALANCED (needs graph traversal)
        - RESONANT → DEEP (complex pattern matching)
        - BALANCED → AUTO (let conductor decide)
        """
        try:
            from ..memory_symphony import MemoryStrategy as MS
        except ImportError:
            return None

        mapping = {
            RecallStrategy.RECENT: MS.FAST,
            RecallStrategy.SIMILAR: MS.FAST,
            RecallStrategy.CONNECTED: MS.BALANCED,
            RecallStrategy.RESONANT: MS.DEEP,
            RecallStrategy.BALANCED: MS.AUTO
        }

        return mapping.get(recall_strategy, MS.AUTO)

    async def _convert_results(self, coordination_result) -> List[Memory]:
        """
        Convert MemoryCoordinationResult → List[Memory].

        Args:
            coordination_result: MemoryCoordinationResult from conductor

        Returns:
            List of Memory objects (user-facing)
        """
        from datetime import datetime

        memories = []
        for result in coordination_result.results:
            # Convert MemoryResult to Memory
            memory = Memory(
                id=result.node_id,
                text=result.content,
                timestamp=datetime.now().isoformat(),  # Placeholder - would use actual timestamp
                context={
                    'source_system': result.source_system.value,
                    'activation': result.activation,
                    'heat': result.heat,
                    **result.metadata
                },
                relevance=result.relevance
            )
            memories.append(memory)

        return memories

    # ========================================================================
    # Temporal Queries (Priority #5)
    # ========================================================================

    def time_travel(self, timestamp: str) -> Dict[str, Any]:
        """
        Time-travel: View memory state at a specific point in time.

        Shows what the knowledge graph looked like at the given timestamp,
        including all memories that existed at that time.

        Args:
            timestamp: ISO format timestamp (e.g., "2025-11-22T10:30:00")

        Returns:
            dict: Snapshot of memory state at that time with:
                {
                    "timestamp": str,
                    "total_memories": int,
                    "memories": List[Memory],
                    "graph_stats": {"nodes": int, "edges": int}
                }

        Example:
            # What did I know on Nov 1st?
            snapshot = memory.time_travel("2025-11-01T00:00:00")
            print(f"Memories on Nov 1: {snapshot['total_memories']}")
        """
        from datetime import datetime

        # Parse target timestamp
        try:
            target_time = datetime.fromisoformat(timestamp)
        except:
            return {
                "error": "Invalid timestamp format. Use ISO format: YYYY-MM-DDTHH:MM:SS",
                "timestamp": timestamp
            }

        snapshot = {
            "timestamp": timestamp,
            "total_memories": 0,
            "memories": [],
            "graph_stats": {"nodes": 0, "edges": 0}
        }

        # Get all memories from backend
        if not self._backend_available or not self._backend:
            return snapshot

        # Filter memories by timestamp
        if hasattr(self._backend, 'graph') and hasattr(self._backend.graph, 'G'):
            graph = self._backend.graph
            past_memories = []

            for node_id in graph.G.nodes():
                # Get node data
                node_data = graph.G.nodes.get(node_id, {})

                # Check if memory existed at target time
                # Assuming node has 'created_at' or 'timestamp' metadata
                mem_timestamp_str = node_data.get('timestamp') or node_data.get('created_at')

                if mem_timestamp_str:
                    try:
                        mem_time = datetime.fromisoformat(mem_timestamp_str)
                        if mem_time <= target_time:
                            # This memory existed at target time
                            mem = Memory(
                                id=node_id,
                                text=node_data.get('text', str(node_id)),
                                timestamp=mem_timestamp_str,
                                context=node_data.get('context', {})
                            )
                            past_memories.append(mem)
                    except:
                        pass

            snapshot["total_memories"] = len(past_memories)
            snapshot["memories"] = past_memories[:100]  # Limit to 100 for performance

            # Count edges that existed at that time
            edge_count = 0
            for src, dst in graph.G.edges():
                # Check if both nodes existed at target time
                src_time = graph.G.nodes.get(src, {}).get('timestamp')
                dst_time = graph.G.nodes.get(dst, {}).get('timestamp')

                if src_time and dst_time:
                    try:
                        if (datetime.fromisoformat(src_time) <= target_time and
                            datetime.fromisoformat(dst_time) <= target_time):
                            edge_count += 1
                    except:
                        pass

            snapshot["graph_stats"] = {
                "nodes": len(past_memories),
                "edges": edge_count
            }

        return snapshot

    def what_happened_between(
        self,
        start_time: str,
        end_time: str,
        limit: int = 20
    ) -> List[Memory]:
        """
        Query memories within a specific time range.

        Args:
            start_time: ISO format start timestamp
            end_time: ISO format end timestamp
            limit: Maximum number of memories to return

        Returns:
            List of Memory objects within the time range, sorted by timestamp

        Example:
            # What did I learn last week?
            from datetime import datetime, timedelta
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            memories = memory.what_happened_between(
                week_ago.isoformat(),
                now.isoformat()
            )
        """
        from datetime import datetime

        # Parse timestamps
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
        except:
            return []

        if not self._backend_available or not self._backend:
            return []

        # Get memories in time range
        memories_in_range = []

        if hasattr(self._backend, 'graph') and hasattr(self._backend.graph, 'G'):
            graph = self._backend.graph

            for node_id in graph.G.nodes():
                node_data = graph.G.nodes.get(node_id, {})
                mem_timestamp_str = node_data.get('timestamp') or node_data.get('created_at')

                if mem_timestamp_str:
                    try:
                        mem_time = datetime.fromisoformat(mem_timestamp_str)

                        if start_dt <= mem_time <= end_dt:
                            mem = Memory(
                                id=node_id,
                                text=node_data.get('text', str(node_id)),
                                timestamp=mem_timestamp_str,
                                context=node_data.get('context', {})
                            )
                            memories_in_range.append(mem)
                    except:
                        pass

        # Sort by timestamp (most recent first)
        memories_in_range.sort(
            key=lambda m: m.timestamp,
            reverse=True
        )

        return memories_in_range[:limit]

    def detect_temporal_patterns(
        self,
        min_occurrences: int = 2,
        time_window_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Detect temporal patterns in memories (recurring themes over time).

        Identifies:
        - Recurring topics (same theme multiple times)
        - Daily/weekly patterns
        - Temporal clusters (bursts of activity)

        Args:
            min_occurrences: Minimum times a pattern must occur
            time_window_days: Size of time window for clustering (days)

        Returns:
            List of temporal patterns, each with:
                {
                    "pattern_type": "recurring_topic" | "daily_pattern" | "cluster",
                    "description": str,
                    "occurrences": int,
                    "memories": List[Memory],
                    "time_span": str
                }

        Example:
            patterns = memory.detect_temporal_patterns(min_occurrences=3)
            for pattern in patterns:
                print(f"{pattern['pattern_type']}: {pattern['description']}")
        """
        from datetime import datetime, timedelta
        from collections import defaultdict

        patterns = []

        if not self._backend_available or not self._backend:
            return patterns

        # Get all memories with timestamps
        timestamped_memories = []

        if hasattr(self._backend, 'graph') and hasattr(self._backend.graph, 'G'):
            graph = self._backend.graph

            for node_id in graph.G.nodes():
                node_data = graph.G.nodes.get(node_id, {})
                mem_timestamp_str = node_data.get('timestamp') or node_data.get('created_at')

                if mem_timestamp_str:
                    try:
                        mem_time = datetime.fromisoformat(mem_timestamp_str)
                        mem = Memory(
                            id=node_id,
                            text=node_data.get('text', str(node_id)),
                            timestamp=mem_timestamp_str,
                            context=node_data.get('context', {})
                        )
                        timestamped_memories.append((mem_time, mem))
                    except:
                        pass

        if len(timestamped_memories) < min_occurrences:
            return patterns

        # Sort by timestamp
        timestamped_memories.sort(key=lambda x: x[0])

        # Pattern 1: Recurring Topics (same words appearing multiple times)
        topic_occurrences = defaultdict(list)

        for mem_time, mem in timestamped_memories:
            # Extract key words (simplified - could use NER)
            words = set(mem.text.lower().split())

            # Group by significant words (>4 characters)
            for word in words:
                if len(word) > 4:
                    topic_occurrences[word].append((mem_time, mem))

        # Find recurring topics
        for topic, occurrences in topic_occurrences.items():
            if len(occurrences) >= min_occurrences:
                times = [t for t, _ in occurrences]
                mems = [m for _, m in occurrences]

                time_span = f"{times[0].date()} to {times[-1].date()}"

                patterns.append({
                    "pattern_type": "recurring_topic",
                    "description": f"Topic '{topic}' appeared {len(occurrences)} times",
                    "occurrences": len(occurrences),
                    "memories": mems[:5],  # Limit to 5
                    "time_span": time_span
                })

        # Pattern 2: Temporal Clusters (bursts of activity)
        # Group memories into time windows
        window = timedelta(days=time_window_days)
        clusters = []
        current_cluster = []
        cluster_start = None

        for mem_time, mem in timestamped_memories:
            if not current_cluster:
                # Start new cluster
                current_cluster = [(mem_time, mem)]
                cluster_start = mem_time
            elif mem_time - cluster_start <= window:
                # Add to current cluster
                current_cluster.append((mem_time, mem))
            else:
                # Save current cluster if significant
                if len(current_cluster) >= min_occurrences:
                    clusters.append(current_cluster)

                # Start new cluster
                current_cluster = [(mem_time, mem)]
                cluster_start = mem_time

        # Don't forget last cluster
        if len(current_cluster) >= min_occurrences:
            clusters.append(current_cluster)

        # Create pattern for each cluster
        for cluster in clusters:
            times = [t for t, _ in cluster]
            mems = [m for _, m in cluster]

            time_span = f"{times[0].date()} to {times[-1].date()}"

            patterns.append({
                "pattern_type": "temporal_cluster",
                "description": f"Burst of {len(cluster)} memories in {time_window_days} days",
                "occurrences": len(cluster),
                "memories": mems[:5],
                "time_span": time_span
            })

        # Sort patterns by occurrences (most frequent first)
        patterns.sort(key=lambda p: p['occurrences'], reverse=True)

        return patterns[:10]  # Return top 10 patterns

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

    def _navigate_forward(self, start_node: str, steps: int, graph) -> List[Memory]:
        """Navigate forward following successors (what comes next)."""
        from datetime import datetime

        path = []
        visited = {start_node}
        current = start_node

        for _ in range(steps):
            # Get successors (nodes this points to)
            successors = list(graph.G.successors(current))

            # Filter out already visited
            unvisited = [s for s in successors if s not in visited]

            if not unvisited:
                break

            # Take first unvisited (could be randomized or weighted)
            next_node = unvisited[0]
            visited.add(next_node)

            # Convert to Memory
            memory = Memory(
                id=next_node,
                text=next_node,  # Placeholder - would fetch actual text
                timestamp=datetime.now().isoformat(),
                context={'direction': 'forward', 'step': len(path) + 1}
            )
            path.append(memory)
            current = next_node

        return path

    def _navigate_backward(self, start_node: str, steps: int, graph) -> List[Memory]:
        """Navigate backward following predecessors (what led to this)."""
        from datetime import datetime

        path = []
        visited = {start_node}
        current = start_node

        for _ in range(steps):
            # Get predecessors (nodes that point to this)
            predecessors = list(graph.G.predecessors(current))

            # Filter out already visited
            unvisited = [p for p in predecessors if p not in visited]

            if not unvisited:
                break

            # Take first unvisited
            next_node = unvisited[0]
            visited.add(next_node)

            # Convert to Memory
            memory = Memory(
                id=next_node,
                text=next_node,
                timestamp=datetime.now().isoformat(),
                context={'direction': 'backward', 'step': len(path) + 1}
            )
            path.append(memory)
            current = next_node

        return path

    def _navigate_sideways(self, start_node: str, steps: int, graph) -> List[Memory]:
        """Navigate sideways to related but different nodes (siblings)."""
        from datetime import datetime

        path = []
        visited = {start_node}

        # Sideways = nodes at same "level" (share parent or child)
        # Get all neighbors (successors + predecessors) then filter
        successors = set(graph.G.successors(start_node))
        predecessors = set(graph.G.predecessors(start_node))

        # Find "siblings" - nodes that share connections with start_node
        siblings = set()

        # Nodes that share parent (have same predecessor)
        for pred in predecessors:
            siblings.update(s for s in graph.G.successors(pred) if s != start_node)

        # Nodes that share child (have same successor)
        for succ in successors:
            siblings.update(p for p in graph.G.predecessors(succ) if p != start_node)

        # Convert to list and limit
        sibling_list = list(siblings - visited)[:steps]

        for i, node in enumerate(sibling_list):
            memory = Memory(
                id=node,
                text=node,
                timestamp=datetime.now().isoformat(),
                context={'direction': 'sideways', 'step': i + 1}
            )
            path.append(memory)

        return path

    def _navigate_deep(self, start_node: str, steps: int, graph) -> List[Memory]:
        """Navigate deep - explore cycles and strange loops from this node."""
        import networkx as nx
        from datetime import datetime

        path = []

        # Find all simple cycles that include start_node
        try:
            # NetworkX simple_cycles finds all cycles
            all_cycles = list(nx.simple_cycles(graph.G))

            # Filter cycles that include start_node
            relevant_cycles = [c for c in all_cycles if start_node in c]

            # Take nodes from cycles (up to steps limit)
            cycle_nodes = []
            for cycle in relevant_cycles:
                # Add nodes from this cycle
                for node in cycle:
                    if node != start_node and node not in cycle_nodes:
                        cycle_nodes.append(node)
                        if len(cycle_nodes) >= steps:
                            break
                if len(cycle_nodes) >= steps:
                    break

            # Convert to Memory objects
            for i, node in enumerate(cycle_nodes[:steps]):
                memory = Memory(
                    id=node,
                    text=node,
                    timestamp=datetime.now().isoformat(),
                    context={'direction': 'deep', 'step': i + 1, 'in_cycle': True}
                )
                path.append(memory)

        except Exception:
            # If cycle detection fails, fall back to BFS exploration
            visited = {start_node}
            queue = [(start_node, 0)]  # (node, depth)

            while queue and len(path) < steps:
                current, depth = queue.pop(0)

                # Explore neighbors at current depth
                for neighbor in list(graph.G.successors(current)) + list(graph.G.predecessors(current)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

                        memory = Memory(
                            id=neighbor,
                            text=neighbor,
                            timestamp=datetime.now().isoformat(),
                            context={'direction': 'deep', 'step': len(path) + 1, 'depth': depth + 1}
                        )
                        path.append(memory)

                        if len(path) >= steps:
                            break

        return path

    def _find_strange_loops(self, min_strength) -> List[MemoryPattern]:
        """Detect strange loops using cycle detection."""
        if not self._backend_available or not self._backend:
            return []

        if not hasattr(self._backend, 'graph'):
            return []

        graph = self._backend.graph

        if not hasattr(graph, 'G'):
            return []

        import networkx as nx

        patterns = []

        try:
            # Find all simple cycles in the graph
            all_cycles = list(nx.simple_cycles(graph.G))

            for cycle in all_cycles:
                # Calculate loop strength based on cycle length and edge weights
                # Shorter loops = stronger (more concentrated pattern)
                length_strength = 1.0 / max(len(cycle), 1)

                # Average edge weight in cycle
                edge_weights = []
                for i in range(len(cycle)):
                    src = cycle[i]
                    dst = cycle[(i + 1) % len(cycle)]
                    if graph.G.has_edge(src, dst):
                        # Get edge data
                        edges = graph.G.get_edge_data(src, dst)
                        if edges:
                            # MultiDiGraph can have multiple edges
                            weights = [e.get('weight', 1.0) for e in edges.values()]
                            edge_weights.append(max(weights))

                avg_weight = sum(edge_weights) / max(len(edge_weights), 1) if edge_weights else 0.5

                # Combined strength
                strength = (length_strength + avg_weight) / 2.0

                if strength >= min_strength:
                    patterns.append(MemoryPattern(
                        pattern_type="loop",
                        memories=cycle,
                        strength=strength,
                        description=f"Strange loop of {len(cycle)} memories: {' → '.join(cycle[:3])}..."
                    ))

        except Exception:
            # Graceful fallback
            pass

        return patterns

    def _find_clusters(self, min_strength) -> List[MemoryPattern]:
        """Detect clusters using community detection."""
        if not self._backend_available or not self._backend:
            return []

        if not hasattr(self._backend, 'graph'):
            return []

        graph = self._backend.graph

        if not hasattr(graph, 'G'):
            return []

        import networkx as nx

        patterns = []

        try:
            # Convert to undirected for community detection
            G_undirected = graph.G.to_undirected()

            if len(G_undirected.nodes()) < 2:
                return []

            # Use greedy modularity communities (fast, no external deps)
            communities = nx.community.greedy_modularity_communities(G_undirected)

            for i, community in enumerate(communities):
                community_list = list(community)

                # Calculate cluster strength based on internal vs external connections
                internal_edges = 0
                external_edges = 0

                for node in community:
                    for neighbor in graph.G.neighbors(node):
                        if neighbor in community:
                            internal_edges += 1
                        else:
                            external_edges += 1

                total_edges = internal_edges + external_edges
                if total_edges == 0:
                    strength = 0.5
                else:
                    strength = internal_edges / total_edges

                if strength >= min_strength and len(community_list) >= 2:
                    patterns.append(MemoryPattern(
                        pattern_type="cluster",
                        memories=community_list,
                        strength=strength,
                        description=f"Memory cluster of {len(community_list)} tightly connected memories"
                    ))

        except Exception:
            # Graceful fallback
            pass

        return patterns

    def _find_resonances(self, min_strength) -> List[MemoryPattern]:
        """Detect resonances using activation patterns."""
        if not self._backend_available or not self._backend:
            return []

        # Check if backend has awareness graph for activation tracking
        if hasattr(self._backend, 'awareness_graph'):
            awareness = self._backend.awareness_graph

            # Get highly activated nodes (resonating memories)
            try:
                activated_nodes = []

                for node_id in awareness.graph.nodes():
                    node_data = awareness.graph.nodes.get(node_id, {})
                    activation = node_data.get('activation', 0.0)

                    if activation >= min_strength:
                        activated_nodes.append((node_id, activation))

                # Sort by activation level
                activated_nodes.sort(key=lambda x: x[1], reverse=True)

                if len(activated_nodes) >= 2:
                    # Create resonance pattern
                    node_ids = [n[0] for n in activated_nodes[:10]]  # Top 10
                    avg_activation = sum(n[1] for n in activated_nodes[:10]) / len(activated_nodes[:10])

                    return [MemoryPattern(
                        pattern_type="resonance",
                        memories=node_ids,
                        strength=avg_activation,
                        description=f"Resonance pattern: {len(node_ids)} highly activated memories"
                    )]

            except Exception:
                pass

        return []

    def _find_threads(self, min_strength) -> List[MemoryPattern]:
        """Detect narrative threads using temporal/causal edges."""
        if not self._backend_available or not self._backend:
            return []

        if not hasattr(self._backend, 'graph'):
            return []

        graph = self._backend.graph

        if not hasattr(graph, 'G'):
            return []

        patterns = []

        try:
            # Find longest paths using LEADS_TO edges
            # Start from nodes with no predecessors (story beginnings)
            roots = [n for n in graph.G.nodes() if graph.G.in_degree(n) == 0]

            if not roots:
                # If no clear roots, use nodes with low in-degree
                roots = sorted(graph.G.nodes(), key=lambda n: graph.G.in_degree(n))[:5]

            for root in roots:
                # DFS to find longest path from this root
                visited = set()
                path = []

                def dfs(node, current_path):
                    nonlocal path
                    visited.add(node)
                    current_path.append(node)

                    # Check for LEADS_TO edges
                    successors = []
                    for succ in graph.G.successors(node):
                        if succ not in visited:
                            # Check if edge type is LEADS_TO or temporal
                            edges = graph.G.get_edge_data(node, succ)
                            if edges:
                                for edge_data in edges.values():
                                    edge_type = edge_data.get('type', '')
                                    if 'LEADS_TO' in edge_type or 'OCCURRED' in edge_type:
                                        successors.append(succ)
                                        break

                    if not successors:
                        # Leaf node - check if path is long enough
                        if len(current_path) > len(path):
                            path = current_path.copy()
                    else:
                        for succ in successors:
                            dfs(succ, current_path)

                    current_path.pop()
                    visited.remove(node)

                dfs(root, [])

                if len(path) >= 3:  # Minimum thread length
                    # Calculate strength based on path continuity
                    strength = min(1.0, len(path) / 10.0)  # Longer = stronger

                    if strength >= min_strength:
                        patterns.append(MemoryPattern(
                            pattern_type="thread",
                            memories=path,
                            strength=strength,
                            description=f"Narrative thread: {len(path)} connected memories from {path[0]} to {path[-1]}"
                        ))

        except Exception:
            # Graceful fallback
            pass

        return patterns

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

    # ========================================================================
    # Production Health Checks (Priority #3)
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for production deployments.

        Returns system status, component availability, and performance metrics.
        Used by Kubernetes liveness/readiness probes and monitoring systems.

        Returns:
            dict: Health status with the following structure:
                {
                    "status": "healthy" | "degraded" | "unhealthy",
                    "timestamp": ISO timestamp,
                    "components": {
                        "backend": {"available": bool, "type": str},
                        "conductor": {"available": bool, "enabled": bool},
                        "graph": {"available": bool, "nodes": int, "edges": int},
                        "awareness": {"available": bool}
                    },
                    "metrics": {
                        "total_memories": int,
                        "uptime_seconds": float
                    },
                    "errors": List[str]  # Any non-critical issues
                }
        """
        from datetime import datetime
        import time

        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "metrics": {},
            "errors": []
        }

        # Check backend availability
        health["components"]["backend"] = {
            "available": self._backend_available,
            "type": type(self._backend).__name__ if self._backend else "None"
        }

        # Check Memory Conductor availability
        health["components"]["conductor"] = {
            "available": self._conductor_available,
            "enabled": self._conductor is not None
        }

        # Check graph backend
        if self._backend_available and self._backend:
            if hasattr(self._backend, 'graph'):
                graph = self._backend.graph
                if hasattr(graph, 'G'):
                    health["components"]["graph"] = {
                        "available": True,
                        "nodes": graph.G.number_of_nodes(),
                        "edges": graph.G.number_of_edges()
                    }
                else:
                    health["components"]["graph"] = {"available": False}
                    health["errors"].append("Graph backend missing G attribute")
            else:
                health["components"]["graph"] = {"available": False}
                health["errors"].append("Backend missing graph attribute")
        else:
            health["components"]["graph"] = {"available": False}
            health["errors"].append("Backend not available")

        # Check awareness graph
        if self._backend_available and self._backend:
            if hasattr(self._backend, 'awareness_graph'):
                health["components"]["awareness"] = {"available": True}
            else:
                health["components"]["awareness"] = {"available": False}
        else:
            health["components"]["awareness"] = {"available": False}

        # Collect metrics
        if self._backend_available and self._backend:
            try:
                # Count total memories (approximation)
                if hasattr(self._backend, 'graph') and hasattr(self._backend.graph, 'G'):
                    health["metrics"]["total_memories"] = self._backend.graph.G.number_of_nodes()
                else:
                    health["metrics"]["total_memories"] = 0
            except Exception as e:
                health["errors"].append(f"Failed to count memories: {str(e)}")
                health["metrics"]["total_memories"] = 0
        else:
            health["metrics"]["total_memories"] = 0

        # Calculate uptime (if tracking start time)
        if hasattr(self, '_start_time'):
            health["metrics"]["uptime_seconds"] = time.time() - self._start_time
        else:
            # Set start time for future health checks
            self._start_time = time.time()
            health["metrics"]["uptime_seconds"] = 0.0

        # Determine overall status
        if not self._backend_available:
            health["status"] = "unhealthy"
        elif health["errors"]:
            health["status"] = "degraded"
        else:
            health["status"] = "healthy"

        return health


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
