# Mathematical Modules & Advanced Graph Integration

## Executive Summary

This document outlines the design for integrating advanced mathematical modules (including Hofstadter-inspired recursive patterns), Neo4j graph database, and Qdrant vector store into the HoloLoom + mem0 hybrid memory system.

**Key Innovations**:
1. **Mathematical Memory Encoding**: Use Hofstadter-style recursive patterns and number theory for memory addressing
2. **Neo4j Thread Model**: Implement junction-first retrieval with temporal/spatial/thematic threads
3. **Qdrant Vector Store**: Replace or augment current embeddings with production-grade similarity search
4. **Spectral Graph Analysis**: Advanced graph Laplacian features for memory relationship discovery

**Result**: A mathematically-principled memory system that discovers emergent patterns through recursive self-reference and graph resonance.

---

## 1. Hofstadter Module: Recursive Memory Patterns

### 1.1 Hofstadter Sequences for Memory Addressing

**Concept**: Use self-referential sequences (like Hofstadter's G, H, Q sequences) to create memory indices that encode temporal and structural relationships.

```python
"""
Hofstadter sequence-based memory addressing.

Inspired by GEB: memories reference other memories through
self-referential number sequences, creating emergent patterns.
"""

class HofstadterMemoryIndex:
    """
    Memory indexing using Hofstadter sequences.
    
    Sequences encode:
    - G(n): Forward-reference (what comes next)
    - H(n): Backward-reference (what came before)
    - Q(n): Chaotic lookup (associative jumps)
    
    These create a web of self-referential memory indices.
    """
    
    def __init__(self, max_n: int = 10000):
        self.max_n = max_n
        self._g_cache = {0: 0}
        self._h_cache = {0: 0, 1: 1}
        self._q_cache = {1: 1, 2: 1}
    
    def G(self, n: int) -> int:
        """
        Hofstadter G sequence: G(n) = n - G(G(n-1))
        
        Encodes forward temporal jumps.
        """
        if n in self._g_cache:
            return self._g_cache[n]
        
        result = n - self.G(self.G(n - 1))
        self._g_cache[n] = result
        return result
    
    def H(self, n: int) -> int:
        """
        Hofstadter H sequence: H(n) = n - H(H(H(n-1)))
        
        Encodes backward temporal resonance.
        """
        if n in self._h_cache:
            return self._h_cache[n]
        
        result = n - self.H(self.H(self.H(n - 1)))
        self._h_cache[n] = result
        return result
    
    def Q(self, n: int) -> int:
        """
        Hofstadter Q sequence: Q(n) = Q(n - Q(n-1)) + Q(n - Q(n-2))
        
        Encodes chaotic, non-linear memory associations.
        """
        if n in self._q_cache:
            return self._q_cache[n]
        
        result = self.Q(n - self.Q(n - 1)) + self.Q(n - self.Q(n - 2))
        self._q_cache[n] = result
        return result
    
    def index_memory(self, memory_id: int, timestamp: float) -> Dict[str, int]:
        """
        Generate Hofstadter indices for a memory.
        
        Returns:
            - forward: Next memory to explore (G-sequence)
            - backward: Prior related memory (H-sequence)
            - associate: Associative jump (Q-sequence)
        """
        n = memory_id % self.max_n
        
        return {
            'forward': self.G(n),
            'backward': self.H(n),
            'associate': self.Q(n),
            'temporal_phase': int(timestamp) % self.max_n
        }
    
    def find_resonance(
        self,
        memory_ids: List[int],
        depth: int = 3
    ) -> List[Tuple[int, int, float]]:
        """
        Find memories that resonate through Hofstadter indices.
        
        Two memories resonate if their sequences intersect
        within 'depth' iterations.
        
        Returns:
            List of (mem_a, mem_b, resonance_score) tuples
        """
        resonances = []
        
        for i, id_a in enumerate(memory_ids):
            idx_a = self.index_memory(id_a, 0)
            
            for id_b in memory_ids[i+1:]:
                idx_b = self.index_memory(id_b, 0)
                
                # Check if sequences intersect
                score = 0.0
                
                # Forward resonance
                if abs(idx_a['forward'] - idx_b['forward']) < depth:
                    score += 0.4
                
                # Backward resonance
                if abs(idx_a['backward'] - idx_b['backward']) < depth:
                    score += 0.3
                
                # Associative resonance (Q-sequence chaos)
                if abs(idx_a['associate'] - idx_b['associate']) < depth * 2:
                    score += 0.3
                
                if score > 0.5:  # Threshold for resonance
                    resonances.append((id_a, id_b, score))
        
        return sorted(resonances, key=lambda x: x[2], reverse=True)
```

### 1.2 Strange Loop Detection

**Concept**: Detect self-referential patterns in memory retrieval (like Hofstadter's "Strange Loops").

```python
class StrangeLoopDetector:
    """
    Detects self-referential memory patterns.
    
    A "strange loop" occurs when:
    - Memory A references B
    - B references C
    - C references A (or higher-order cycles)
    
    These loops indicate emergent conceptual hierarchies.
    """
    
    def detect_loops(
        self,
        memory_graph: nx.DiGraph,
        min_loop_length: int = 2,
        max_loop_length: int = 10
    ) -> List[List[str]]:
        """
        Find all simple cycles (strange loops) in memory graph.
        
        Returns:
            List of loops, each as [mem_a, mem_b, ..., mem_a]
        """
        loops = []
        
        for cycle in nx.simple_cycles(memory_graph):
            if min_loop_length <= len(cycle) <= max_loop_length:
                loops.append(cycle)
        
        return loops
    
    def loop_strength(
        self,
        loop: List[str],
        memory_graph: nx.DiGraph
    ) -> float:
        """
        Calculate the "strength" of a strange loop.
        
        Considers:
        - Edge weights in the loop
        - Number of alternative paths
        - Temporal coherence
        """
        if len(loop) < 2:
            return 0.0
        
        # Get edge weights
        weights = []
        for i in range(len(loop)):
            src = loop[i]
            dst = loop[(i + 1) % len(loop)]
            
            if memory_graph.has_edge(src, dst):
                edge_data = memory_graph[src][dst]
                weights.append(edge_data.get('weight', 0.5))
        
        # Geometric mean of weights
        if not weights:
            return 0.0
        
        import math
        strength = math.exp(sum(math.log(max(w, 0.01)) for w in weights) / len(weights))
        
        # Bonus for longer loops (more complex hierarchy)
        length_factor = 1.0 + (len(loop) - 2) * 0.1
        
        return min(strength * length_factor, 1.0)
    
    def hierarchical_analysis(
        self,
        loops: List[List[str]],
        memory_graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """
        Analyze the hierarchical structure created by strange loops.
        
        Returns:
            - Tangled hierarchies: Loops that share nodes
            - Meta-loops: Loops that reference other loops
            - Complexity score: Overall system complexity
        """
        # Find tangled hierarchies (loops sharing nodes)
        tangled = []
        for i, loop_a in enumerate(loops):
            for loop_b in loops[i+1:]:
                shared = set(loop_a) & set(loop_b)
                if shared:
                    tangled.append({
                        'loop_a': loop_a,
                        'loop_b': loop_b,
                        'shared_nodes': list(shared),
                        'tangle_strength': len(shared) / min(len(loop_a), len(loop_b))
                    })
        
        # Calculate complexity (entropy of loop distribution)
        loop_lengths = [len(loop) for loop in loops]
        length_dist = np.bincount(loop_lengths)
        probs = length_dist / length_dist.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return {
            'num_loops': len(loops),
            'tangled_hierarchies': tangled,
            'complexity_entropy': float(entropy),
            'avg_loop_length': np.mean(loop_lengths) if loop_lengths else 0.0
        }
```

---

## 2. Neo4j Integration: Junction-First Thread Model

### 2.1 Thread-Based Memory Architecture

**Concept**: Implement the "junction-first" model from your archive - memories are KNOTS that cross THREADS (time, place, actor, theme, glyph).

```python
"""
Neo4j adapter for thread-based memory storage.

Architecture (from 006_junction_first.cypher):
- THREAD nodes: Continuous dimensions (time, place, actor, theme, glyph)
- KNOT nodes: Memory crossings (episodes, events)
- Edges: IN_TIME, AT_PLACE, WITH_ACTOR, ABOUT_THEME, WEARS_GLYPH
"""

from neo4j import GraphDatabase, Driver
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

@dataclass
class Thread:
    """A continuous dimension through memory."""
    id: str
    type: str  # 'time', 'place', 'actor', 'theme', 'glyph'
    key: str   # The actual value (e.g., '2025-10-22-evening', 'apiary', 'Blake')

@dataclass
class Knot:
    """A memory that crosses multiple threads."""
    id: str
    text: str
    salience: float
    timestamp: str
    threads: Dict[str, str]  # thread_type -> thread_key

class Neo4jMemoryStore:
    """
    Neo4j-backed memory storage using thread model.
    
    Advantages over NetworkX:
    - Production-grade graph database
    - Cypher queries for complex traversals
    - Scalable to millions of memories
    - ACID transactions
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create constraints and indices (idempotent)."""
        with self.driver.session(database=self.database) as session:
            # From 006_junction_first.cypher
            session.run("""
                CREATE CONSTRAINT thread_id IF NOT EXISTS
                FOR (t:Thread) REQUIRE t.id IS UNIQUE
            """)
            
            session.run("""
                CREATE INDEX thread_type_key IF NOT EXISTS
                FOR (t:Thread) ON (t.type, t.key)
            """)
            
            session.run("""
                CREATE CONSTRAINT knot_id IF NOT EXISTS
                FOR (k:Knot) REQUIRE k.id IS UNIQUE
            """)
            
            session.run("""
                CREATE INDEX knot_time IF NOT EXISTS
                FOR (k:Knot) ON (k.timestamp)
            """)
            
            session.run("""
                CREATE INDEX knot_salience IF NOT EXISTS
                FOR (k:Knot) ON (k.salience)
            """)
    
    def store_memory_as_knot(
        self,
        memory_id: str,
        text: str,
        threads: Dict[str, str],
        salience: float = 0.5,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Store a memory as a KNOT crossing multiple THREADS.
        
        Args:
            memory_id: Unique memory identifier
            text: Memory content
            threads: Dict mapping thread_type to thread_key
                     e.g., {'time': '2025-10-22-evening', 'place': 'apiary'}
            salience: Importance score [0, 1]
            timestamp: ISO timestamp
            metadata: Additional properties
        """
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        
        with self.driver.session(database=self.database) as session:
            # Create knot
            session.run("""
                MERGE (k:Knot {id: $id})
                SET k.text = $text,
                    k.salience = $salience,
                    k.timestamp = $timestamp,
                    k.metadata = $metadata
            """, id=memory_id, text=text, salience=salience,
                timestamp=timestamp, metadata=metadata or {})
            
            # Create thread edges
            for thread_type, thread_key in threads.items():
                thread_id = f"{thread_type}::{thread_key}"
                
                # Create or merge thread
                session.run("""
                    MERGE (t:Thread {type: $type, key: $key})
                    ON CREATE SET t.id = $id
                """, type=thread_type, key=thread_key, id=thread_id)
                
                # Create edge based on type
                edge_type = self._edge_type_for_thread(thread_type)
                session.run(f"""
                    MATCH (k:Knot {{id: $knot_id}})
                    MATCH (t:Thread {{type: $thread_type, key: $thread_key}})
                    MERGE (k)-[:{edge_type}]->(t)
                """, knot_id=memory_id, thread_type=thread_type, thread_key=thread_key)
    
    def _edge_type_for_thread(self, thread_type: str) -> str:
        """Map thread type to edge type."""
        mapping = {
            'time': 'IN_TIME',
            'place': 'AT_PLACE',
            'actor': 'WITH_ACTOR',
            'theme': 'ABOUT_THEME',
            'glyph': 'WEARS_GLYPH'
        }
        return mapping.get(thread_type, 'CROSSES')
    
    def retrieve_by_thread_intersection(
        self,
        thread_constraints: Dict[str, List[str]],
        limit: int = 10,
        strict: bool = True
    ) -> List[Knot]:
        """
        Retrieve knots that cross specified threads.
        
        Args:
            thread_constraints: Dict of thread_type -> [thread_keys]
                                e.g., {'time': ['2025-10-22-evening'],
                                       'theme': ['beekeeping', 'winter']}
            limit: Max results
            strict: If True, require ALL thread types (AND logic)
                    If False, match ANY thread (OR logic)
        
        Returns:
            List of Knot objects
        """
        with self.driver.session(database=self.database) as session:
            if strict:
                # Strict AND: knot must cross at least one thread from EACH type
                query = """
                    MATCH (k:Knot)-[r]->(t:Thread)
                    WHERE t.type IN $types
                      AND t.key IN $keys_for_type
                    WITH k, collect(DISTINCT t.type) AS crossed_types
                    WHERE size(crossed_types) = $required_count
                    RETURN k.id AS id, k.text AS text, k.salience AS salience,
                           k.timestamp AS timestamp
                    ORDER BY k.salience DESC
                    LIMIT $limit
                """
                
                types = list(thread_constraints.keys())
                keys_for_type = [k for keys in thread_constraints.values() for k in keys]
                
                result = session.run(
                    query,
                    types=types,
                    keys_for_type=keys_for_type,
                    required_count=len(types),
                    limit=limit
                )
            else:
                # OR logic: match any thread
                query = """
                    MATCH (k:Knot)-[r]->(t:Thread)
                    WHERE (t.type, t.key) IN $thread_pairs
                    WITH k, count(DISTINCT t) AS match_count
                    RETURN k.id AS id, k.text AS text, k.salience AS salience,
                           k.timestamp AS timestamp, match_count
                    ORDER BY match_count DESC, k.salience DESC
                    LIMIT $limit
                """
                
                thread_pairs = [
                    (thread_type, key)
                    for thread_type, keys in thread_constraints.items()
                    for key in keys
                ]
                
                result = session.run(query, thread_pairs=thread_pairs, limit=limit)
            
            # Convert to Knot objects
            knots = []
            for record in result:
                knots.append(Knot(
                    id=record['id'],
                    text=record['text'],
                    salience=record['salience'],
                    timestamp=record['timestamp'],
                    threads={}  # Could fetch threads in separate query
                ))
            
            return knots
    
    def find_temporal_resonance(
        self,
        anchor_time: str,
        time_window_days: int = 7,
        min_shared_themes: int = 2
    ) -> List[Tuple[Knot, Knot, float]]:
        """
        Find memories that resonate temporally.
        
        Two memories resonate if:
        - They're within time_window of each other
        - They share multiple themes/actors
        - High combined salience
        
        Returns:
            List of (knot_a, knot_b, resonance_score)
        """
        with self.driver.session(database=self.database) as session:
            query = """
                MATCH (k1:Knot)-[:IN_TIME]->(t1:Thread {type: 'time'})
                WHERE t1.key CONTAINS $anchor_date
                
                MATCH (k2:Knot)-[:IN_TIME]->(t2:Thread {type: 'time'})
                WHERE k2 <> k1
                  AND duration.between(
                        datetime(k1.timestamp),
                        datetime(k2.timestamp)
                      ).days <= $window
                
                // Find shared themes
                MATCH (k1)-[:ABOUT_THEME]->(theme:Thread {type: 'theme'})<-[:ABOUT_THEME]-(k2)
                WITH k1, k2, count(DISTINCT theme) AS shared_themes,
                     (k1.salience + k2.salience) / 2.0 AS avg_salience
                WHERE shared_themes >= $min_themes
                
                RETURN k1.id AS id1, k1.text AS text1,
                       k2.id AS id2, k2.text AS text2,
                       shared_themes, avg_salience,
                       (shared_themes * 0.6 + avg_salience * 0.4) AS resonance
                ORDER BY resonance DESC
                LIMIT 20
            """
            
            anchor_date = anchor_time[:10]  # YYYY-MM-DD
            
            result = session.run(
                query,
                anchor_date=anchor_date,
                window=time_window_days,
                min_themes=min_shared_themes
            )
            
            resonances = []
            for record in result:
                knot1 = Knot(record['id1'], record['text1'], 0.0, '', {})
                knot2 = Knot(record['id2'], record['text2'], 0.0, '', {})
                resonances.append((knot1, knot2, record['resonance']))
            
            return resonances
    
    def close(self):
        """Close driver connection."""
        self.driver.close()
```

### 2.2 Cypher Query Templates

```python
class CypherTemplates:
    """
    Reusable Cypher query templates for common memory operations.
    """
    
    @staticmethod
    def thread_intersection_strict(thread_types: List[str]) -> str:
        """
        Strict AND query: knot must cross all thread types.
        """
        return """
            MATCH (k:Knot)
            WHERE ALL(
                type IN $thread_types
                WHERE EXISTS((k)-[]->(t:Thread {type: type}))
            )
            WITH k
            MATCH (k)-[r]->(t:Thread)
            WHERE t.type IN $thread_types
            RETURN k, collect({type: t.type, key: t.key, edge: type(r)}) AS threads
            ORDER BY k.salience DESC
            LIMIT $limit
        """
    
    @staticmethod
    def find_thread_clusters() -> str:
        """
        Find clusters of threads that frequently co-occur.
        """
        return """
            MATCH (t1:Thread)<-[]-(k:Knot)-[]->(t2:Thread)
            WHERE t1.type < t2.type  // Avoid duplicates
            WITH t1.type + '::' + t1.key AS thread1,
                 t2.type + '::' + t2.key AS thread2,
                 count(k) AS cooccurrence
            WHERE cooccurrence >= $min_cooccur
            RETURN thread1, thread2, cooccurrence
            ORDER BY cooccurrence DESC
        """
    
    @staticmethod
    def temporal_narrative(start_time: str, end_time: str) -> str:
        """
        Extract temporal narrative: all knots between times with themes.
        """
        return """
            MATCH (k:Knot)
            WHERE k.timestamp >= $start AND k.timestamp <= $end
            
            OPTIONAL MATCH (k)-[:ABOUT_THEME]->(theme:Thread {type: 'theme'})
            OPTIONAL MATCH (k)-[:WITH_ACTOR]->(actor:Thread {type: 'actor'})
            
            WITH k, collect(DISTINCT theme.key) AS themes,
                     collect(DISTINCT actor.key) AS actors
            RETURN k.id AS id, k.text AS text, k.timestamp AS timestamp,
                   themes, actors
            ORDER BY k.timestamp ASC
        """
```

---

## 3. Qdrant Integration: Production Vector Store

### 3.1 Qdrant Configuration for Mem0

**Concept**: Replace mem0's default vector store with Qdrant for production-grade similarity search.

```python
"""
Qdrant vector store integration for mem0 + HoloLoom.

Advantages:
- High-performance similarity search
- Supports filtering on payload
- Horizontal scaling
- Multi-tenancy (user isolation)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import numpy as np

class QdrantMemoryStore:
    """
    Qdrant vector store for hybrid memory system.
    
    Collections:
    - holoLoom_shards: Multi-scale embeddings (96d, 192d, 384d)
    - mem0_memories: User-specific memories
    - resonance_patterns: Discovered memory patterns
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        use_cloud: bool = False
    ):
        if use_cloud and api_key:
            self.client = QdrantClient(
                url="https://your-cluster.qdrant.io",
                api_key=api_key
            )
        else:
            self.client = QdrantClient(host=host, port=port)
        
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Create collections if they don't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # HoloLoom shards with multi-scale embeddings
        if "holoLoom_shards" not in collection_names:
            self.client.create_collection(
                collection_name="holoLoom_shards",
                vectors_config={
                    "scale_96": VectorParams(size=96, distance=Distance.COSINE),
                    "scale_192": VectorParams(size=192, distance=Distance.COSINE),
                    "scale_384": VectorParams(size=384, distance=Distance.COSINE),
                }
            )
        
        # Mem0 memories (single embedding for simplicity)
        if "mem0_memories" not in collection_names:
            self.client.create_collection(
                collection_name="mem0_memories",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
    
    def store_holoLoom_shard(
        self,
        shard_id: str,
        embeddings: Dict[int, np.ndarray],  # {96: vec96, 192: vec192, 384: vec384}
        payload: Dict
    ):
        """
        Store HoloLoom shard with multi-scale embeddings.
        
        Args:
            shard_id: Unique shard identifier
            embeddings: Dict of scale -> embedding vector
            payload: Metadata (text, episode, entities, motifs, etc.)
        """
        # Convert numpy arrays to lists
        vectors = {
            f"scale_{size}": emb.tolist()
            for size, emb in embeddings.items()
        }
        
        point = PointStruct(
            id=hash(shard_id),  # Convert string to int ID
            vector=vectors,
            payload=payload
        )
        
        self.client.upsert(
            collection_name="holoLoom_shards",
            points=[point]
        )
    
    def search_multi_scale(
        self,
        query_embeddings: Dict[int, np.ndarray],
        fusion_weights: Dict[int, float] = None,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Multi-scale search with weighted fusion.
        
        Args:
            query_embeddings: Dict of scale -> query vector
            fusion_weights: Dict of scale -> weight (default: equal)
            filters: Qdrant filters on payload
            limit: Max results
        
        Returns:
            List of results with fused scores
        """
        if fusion_weights is None:
            fusion_weights = {96: 0.25, 192: 0.35, 384: 0.40}
        
        # Search each scale
        scale_results = {}
        for scale, query_vec in query_embeddings.items():
            results = self.client.search(
                collection_name="holoLoom_shards",
                query_vector=(f"scale_{scale}", query_vec.tolist()),
                query_filter=filters,
                limit=limit * 2,  # Get more for fusion
                with_payload=True,
                with_vectors=False
            )
            scale_results[scale] = results
        
        # Fuse scores
        fused = self._fuse_scale_results(scale_results, fusion_weights, limit)
        return fused
    
    def _fuse_scale_results(
        self,
        scale_results: Dict[int, List],
        weights: Dict[int, float],
        limit: int
    ) -> List[Dict]:
        """
        Fuse multi-scale results using weighted combination.
        """
        # Collect all unique IDs
        all_ids = set()
        for results in scale_results.values():
            all_ids.update([r.id for r in results])
        
        # Calculate fused scores
        fused_scores = {}
        for point_id in all_ids:
            score = 0.0
            for scale, results in scale_results.items():
                # Find this point in scale results
                for r in results:
                    if r.id == point_id:
                        score += weights.get(scale, 0.0) * r.score
                        break
            fused_scores[point_id] = score
        
        # Sort by fused score
        sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [pid for pid, _ in sorted_ids[:limit]]
        
        # Retrieve full payloads for top results
        points = self.client.retrieve(
            collection_name="holoLoom_shards",
            ids=top_ids,
            with_payload=True
        )
        
        return [
            {
                'id': p.id,
                'score': fused_scores[p.id],
                'payload': p.payload
            }
            for p in points
        ]
    
    def store_mem0_memory(
        self,
        memory_id: str,
        embedding: np.ndarray,
        user_id: str,
        text: str,
        metadata: Dict
    ):
        """Store mem0 memory with user isolation."""
        point = PointStruct(
            id=hash(memory_id),
            vector=embedding.tolist(),
            payload={
                'user_id': user_id,
                'text': text,
                'memory_id': memory_id,
                **metadata
            }
        )
        
        self.client.upsert(
            collection_name="mem0_memories",
            points=[point]
        )
    
    def search_user_memories(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Search memories for specific user."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )
        
        results = self.client.search(
            collection_name="mem0_memories",
            query_vector=query_embedding.tolist(),
            query_filter=user_filter,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                'id': r.id,
                'score': r.score,
                'text': r.payload.get('text'),
                'metadata': r.payload
            }
            for r in results
        ]


# Configure mem0 to use Qdrant
def configure_mem0_with_qdrant(
    host: str = "localhost",
    port: int = 6333
) -> Dict:
    """
    Generate mem0 configuration for Qdrant backend.
    
    Usage:
        config = configure_mem0_with_qdrant()
        memory = Memory.from_config(config)
    """
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": host,
                "port": port,
                "collection_name": "mem0_memories",
                "embedding_model_dims": 384,
                "on_disk": True  # Persist to disk
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1
            }
        }
    }
```

---

## 4. Spectral Graph Features Enhancement

### 4.1 Advanced Graph Laplacian Analysis

```python
"""
Enhanced spectral features using graph Laplacian.

Extracts mathematical properties of memory graph:
- Fiedler value: Graph connectivity
- Spectral gap: Community structure
- Eigenvector centrality: Memory importance
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

class SpectralMemoryAnalyzer:
    """
    Advanced spectral analysis of memory graphs.
    
    Uses graph Laplacian eigenvalues/eigenvectors to:
    - Detect memory communities
    - Find central memories
    - Measure graph coherence
    """
    
    def compute_graph_laplacian(
        self,
        adjacency: np.ndarray,
        normalized: bool = True
    ) -> np.ndarray:
        """
        Compute graph Laplacian.
        
        L = D - A (unnormalized)
        L = I - D^(-1/2) A D^(-1/2) (normalized)
        """
        n = adjacency.shape[0]
        degrees = np.sum(adjacency, axis=1)
        
        if normalized:
            # Normalized Laplacian
            D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-10))
            L = sp.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        else:
            # Unnormalized Laplacian
            D = sp.diags(degrees)
            L = D - adjacency
        
        return L
    
    def spectral_features(
        self,
        adjacency: np.ndarray,
        k: int = 4
    ) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from memory graph.
        
        Args:
            adjacency: Graph adjacency matrix
            k: Number of eigenvalues to compute
        
        Returns:
            - eigenvalues: Smallest k eigenvalues
            - eigenvectors: Corresponding eigenvectors
            - fiedler_value: Second smallest eigenvalue (connectivity)
            - spectral_gap: Difference between λ2 and λ3
        """
        L = self.compute_graph_laplacian(adjacency, normalized=True)
        
        # Compute smallest k eigenvalues/vectors
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'fiedler_value': eigenvalues[1],  # λ2
            'spectral_gap': eigenvalues[2] - eigenvalues[1] if k > 2 else 0.0,
            'coherence': 1.0 / (1.0 + eigenvalues[1])  # Higher = more connected
        }
    
    def detect_communities(
        self,
        adjacency: np.ndarray,
        n_communities: int = 3
    ) -> np.ndarray:
        """
        Spectral clustering to detect memory communities.
        
        Uses Fiedler vector and k-means.
        """
        features = self.spectral_features(adjacency, k=n_communities)
        eigenvectors = features['eigenvectors']
        
        # K-means on eigenvectors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_communities, random_state=42)
        communities = kmeans.fit_predict(eigenvectors)
        
        return communities
    
    def memory_importance(
        self,
        adjacency: np.ndarray
    ) -> np.ndarray:
        """
        Calculate importance of each memory using eigenvector centrality.
        
        Returns:
            Importance scores for each node (memory)
        """
        # Compute leading eigenvector of adjacency matrix
        eigenvalues, eigenvectors = eigsh(adjacency, k=1, which='LM')
        importance = np.abs(eigenvectors[:, 0])
        
        # Normalize
        importance = importance / np.max(importance)
        
        return importance
```

---

## 5. Integration Architecture

### 5.1 Unified Memory System

```
┌──────────────────────────────────────────────────────────┐
│                  User Query                               │
└─────────────────┬────────────────────────────────────────┘
                  │
    ┌─────────────┴────────────────┐
    │                              │
┌───▼────────┐          ┌─────────▼──────┐
│   Mem0     │          │   HoloLoom     │
│            │          │                │
│ • LLM ext  │          │ • Motifs       │
│ • User ctx │          │ • Multi-scale  │
│ • Qdrant   │          │ • Qdrant       │
└───┬────────┘          └─────────┬──────┘
    │                             │
    │ ┌───────────────────────────┴──────────────────────┐
    │ │                                                   │
┌───▼─▼─────────────┐  ┌────────────────┐  ┌────────────▼──────┐
│ Qdrant            │  │ Neo4j          │  │ Hofstadter        │
│ • Multi-scale vec │  │ • Thread model │  │ • Resonance index │
│ • User isolation  │  │ • Junction ret │  │ • Strange loops   │
│ • Fast similarity │  │ • Cypher query │  │ • Recursive addr  │
└──────┬────────────┘  └───────┬────────┘  └────────┬──────────┘
       │                       │                     │
       └───────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Spectral Analysis  │
                    │ • Graph Laplacian  │
                    │ • Community detect │
                    │ • Importance scores│
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Fused Context     │
                    │ • Personal (mem0)  │
                    │ • Domain (HoloLoom)│
                    │ • Temporal (Neo4j) │
                    │ • Resonant (Hofst) │
                    └────────────────────┘
```

### 5.2 Configuration

```python
@dataclass
class MathematicalMemoryConfig:
    """Configuration for enhanced memory system."""
    
    # Hofstadter module
    use_hofstadter: bool = True
    hofstadter_max_n: int = 10000
    resonance_threshold: float = 0.5
    
    # Neo4j
    neo4j_enabled: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # Qdrant
    qdrant_enabled: bool = True
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    
    # Spectral analysis
    spectral_k_eigenvalues: int = 4
    community_detection: bool = True
    n_communities: int = 3
    
    # Fusion weights
    mem0_weight: float = 0.2
    hololoom_weight: float = 0.4
    neo4j_weight: float = 0.2
    hofstadter_weight: float = 0.2
```

---

## 6. Example Usage

```python
async def advanced_memory_example():
    """
    Demonstrate mathematical memory modules.
    """
    # Initialize components
    config = MathematicalMemoryConfig()
    
    # Hofstadter indexing
    hofstadter = HofstadterMemoryIndex(max_n=config.hofstadter_max_n)
    
    # Neo4j storage
    neo4j_store = Neo4jMemoryStore(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password
    )
    
    # Qdrant vectors
    qdrant_store = QdrantMemoryStore(
        host=config.qdrant_host,
        port=config.qdrant_port
    )
    
    # Configure mem0 with Qdrant
    mem0_config = configure_mem0_with_qdrant()
    memory = Memory.from_config(mem0_config)
    
    # Store a memory with ALL systems
    query_text = "Hive Jodi needs winter prep"
    user_id = "blake"
    
    # 1. Mem0 extraction
    mem0_result = memory.add(query_text, user_id=user_id)
    
    # 2. Neo4j thread storage
    neo4j_store.store_memory_as_knot(
        memory_id=f"mem_{hash(query_text)}",
        text=query_text,
        threads={
            'time': '2025-10-22-evening',
            'place': 'apiary',
            'actor': 'Blake',
            'theme': 'beekeeping'
        },
        salience=0.8
    )
    
    # 3. Hofstadter indexing
    memory_id = hash(query_text) % hofstadter.max_n
    indices = hofstadter.index_memory(memory_id, time.time())
    print(f"Hofstadter indices: {indices}")
    
    # 4. Find resonances
    memory_ids = [memory_id, memory_id + 5, memory_id + 10]
    resonances = hofstadter.find_resonance(memory_ids, depth=3)
    print(f"Found {len(resonances)} resonances")
    
    # 5. Retrieve with Neo4j threads
    knots = neo4j_store.retrieve_by_thread_intersection(
        thread_constraints={
            'theme': ['beekeeping'],
            'time': ['2025-10-22-evening']
        },
        strict=True,
        limit=5
    )
    print(f"Retrieved {len(knots)} knots from Neo4j")
    
    # 6. Temporal resonance
    temporal_resonance = neo4j_store.find_temporal_resonance(
        anchor_time='2025-10-22',
        time_window_days=7
    )
    print(f"Found {len(temporal_resonance)} temporal resonances")
    
    # Cleanup
    neo4j_store.close()
```

---

## 7. Next Steps

1. **Implement Hofstadter module** (`HoloLoom/math/hofstadter.py`)
2. **Create Neo4j adapter** (`HoloLoom/memory/neo4j_adapter.py`)
3. **Integrate Qdrant** (`HoloLoom/memory/qdrant_store.py`)
4. **Enhanced spectral features** (`HoloLoom/embedding/spectral_advanced.py`)
5. **Unified memory orchestrator** that coordinates all modules
6. **Benchmark against HoloLoom-only baseline**

## 8. Benefits

| Feature | Baseline | With Math Modules |
|---------|----------|-------------------|
| Memory Addressing | Sequential IDs | Hofstadter resonance |
| Graph Storage | NetworkX (in-memory) | Neo4j (production) |
| Vector Search | numpy dot product | Qdrant (optimized) |
| Pattern Discovery | Manual | Strange loops, spectral |
| Temporal Reasoning | Time buckets | Thread intersections |
| Scaling | Single machine | Distributed (Neo4j + Qdrant) |

---

**Status**: Design Complete  
**Next**: Implementation Phase  
**Target**: Production-grade mathematical memory system with emergent pattern discovery
