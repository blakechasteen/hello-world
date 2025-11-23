# Option B: Quick Wins Bundle - Comprehensive Roadmap

**Duration**: 1 week (5 working days)
**Team Size**: 1 developer (concurrent with Options A & C)
**Effort**: 20 hours total (4 hours/day)
**Status**: Ready to begin
**Date Created**: 2025-11-20

---

## Executive Summary

Quick Wins Bundle delivers **three high-impact completions** in one week:

1. **Phase 2 Activation** (30 min) - Enable Thompson Sampling learning
2. **MCTS Shuttle Integration** (2-3 days) - Connect to real HoloLoom backends
3. **Workflow Builder Backend** (2-3 days) - Connect existing 12 HTML dashboards

**Value Proposition**:
- **Fast results**: Multiple wins in 5 days vs 1 big win in 10 days
- **Completion satisfaction**: Finishes existing work
- **Immediate utility**: All three are production-ready after 1 week
- **Risk mitigation**: Small, independent tasks reduce failure risk

---

## Architecture Overview

```
Quick Win #1: Phase 2 Activation
================================
Query → [Thompson Sampling] → [Pattern Learning] → [Hot Pattern Tracking]
        ↓                     ↓                    ↓
    Update priors         Extract motifs      Boost frequently used
    α/β after query      "motif→tool→conf"   memories (2x)

Quick Win #2: MCTS Shuttle Integration
=======================================
HoloLoom Query
    ↓
[Mock Backends] → [Real Backends]
    Qdrant Mock      → Qdrant Client (existing)
    Neo4j Mock       → Neo4j Client (existing)
    ↓
[MCTS Graph Exploration]
    ↓
[Hybrid Context] (Warp fuzzy + Yarn structural)

Quick Win #3: Workflow Builder Backend
========================================
[12 HTML Dashboards] (existing)
    ↓
[WebSocket Executor] (new - 300 lines)
    ↓
[HoloLoom Integration]
    ↓
[18 Agent Types] (partially existing)
```

---

## Timeline: 5-Day Sprint

```
Day 1 (Mon):  Phase 2 Activation (30 min) + MCTS Setup (3.5 hrs)
Day 2 (Tue):  MCTS Integration (4 hrs)
Day 3 (Wed):  MCTS Testing (2 hrs) + Workflow Backend Start (2 hrs)
Day 4 (Thu):  Workflow Backend (4 hrs)
Day 5 (Fri):  Workflow Testing (2 hrs) + Integration Demo (2 hrs)
```

---

## Quick Win #1: Phase 2 Activation (30 minutes)

### Overview

Enable continuous learning in existing `my_smart_ai.py` script.

**What's Already Done**:
- ✅ Thompson Sampling bandit (policy/unified.py)
- ✅ Pattern learning infrastructure (recursive/pattern_learner.py)
- ✅ Hot pattern tracking (recursive/hot_pattern_feedback.py)
- ✅ Configuration flags (config.py)

**What Needs Activation**:
- Enable flags in my_smart_ai.py
- Add 3 lines to ingest_my_writing.py
- Verify learning works

### Implementation (30 minutes)

**Step 1: Update my_smart_ai.py (10 min)**

```python
# Current (lines 20-25)
config = Config.fast()

# Add Phase 2 activation
config.enable_recursive_learning = True  # NEW
config.recursive_learning_enable_background = True  # NEW
config.recursive_learning_enable_hot_patterns = True  # NEW
config.recursive_learning_refinement_threshold = 0.75  # NEW
```

**Step 2: Update ingest_my_writing.py (10 min)**

```python
# Current (line 50)
async with HoloLoom() as loom:
    await loom.experience(content)

# Add learning loop
from HoloLoom.recursive import FullLearningEngine

async with FullLearningEngine(
    cfg=config,
    shards=shards,
    enable_background_learning=True  # NEW
) as engine:
    spacetime = await engine.weave(query)  # NEW
    # Thompson Sampling + Pattern Learning + Hot Patterns all active!
```

**Step 3: Verify Learning (10 min)**

```bash
# Run with learning enabled
python my_smart_ai.py

# Check learning statistics
stats = engine.get_learning_statistics()
print(f"Patterns learned: {stats['patterns_learned']}")
print(f"Thompson priors: {stats['thompson_priors']}")
print(f"Hot patterns: {stats['hot_patterns_count']}")
```

### Verification Checklist

- [ ] Config flags enabled
- [ ] FullLearningEngine instantiated
- [ ] Learning statistics display after each query
- [ ] Thompson Sampling priors update (α/β change)
- [ ] Pattern learning extracts "motif → tool → success" patterns
- [ ] Hot patterns tracked (access count × success rate)
- [ ] No performance degradation (<3ms overhead)

### Success Metrics

- **Patterns learned**: >0 after 5 queries
- **Thompson updates**: α/β change each query
- **Hot patterns**: ≥1 after 10 accesses
- **Performance**: <3ms overhead per query

---

## Quick Win #2: MCTS Shuttle Integration (2-3 days)

### Overview

Replace mock Qdrant/Neo4j backends with real HoloLoom integrations.

**Current State**:
- ✅ MCTS v2.0 complete (1,800 lines)
- ✅ Mock backends work (demo runs successfully)
- ✅ HoloLoom adapters scaffolded (hololoom_adapters.py)
- ❌ Real backend connections missing

**Goal State**:
- ✅ Qdrant client connected
- ✅ Neo4j client connected
- ✅ MCTS works with real data
- ✅ Benchmarked vs standard retrieval

### Day 1: Setup & Qdrant Integration (3.5 hours)

**Task 1.1: Dependencies & Configuration (30 min)**

```bash
# Install Qdrant Python client
pip install qdrant-client

# Verify Qdrant running
docker ps | grep qdrant

# If not running
docker-compose up -d qdrant
```

Create `bosspig/shuttle_config.py`:
```python
@dataclass
class ShuttleConfig:
    """Configuration for MCTS Shuttle with real backends."""

    # Qdrant (Warp)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "hololoom_memories"

    # Neo4j (Yarn)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # MCTS
    num_simulations: int = 50
    exploration_constant: float = 1.414
    max_depth: int = 5

    @classmethod
    def from_hololoom_config(cls, hololoom_config):
        """Create from existing HoloLoom config."""
        return cls(
            qdrant_host=hololoom_config.qdrant_host,
            qdrant_port=hololoom_config.qdrant_port,
            # ... copy all relevant settings
        )
```

**Verification**:
- [ ] Qdrant client connects
- [ ] Neo4j driver connects
- [ ] Config loads from HoloLoom

**Task 1.2: Real Qdrant Integration (1.5 hours)**

Update `HoloLoom/shuttle/hololoom_adapters.py`:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

class HoloLoomWarp:
    """Real Qdrant integration for Warp search."""

    def __init__(self, config: ShuttleConfig):
        self.config = config
        self.client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port
        )
        self.embedder = MatryoshkaEmbeddings()

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if doesn't exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.config.qdrant_collection for c in collections):
            self.client.create_collection(
                collection_name=self.config.qdrant_collection,
                vectors_config=VectorParams(
                    size=768,  # Full Matryoshka dimension
                    distance=Distance.COSINE
                )
            )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search using real Qdrant."""

        # Get query embedding
        query_embedding = self.embedder.get_embedding(query)

        # Search Qdrant
        results = self.client.search(
            collection_name=self.config.qdrant_collection,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False  # Don't need vectors back
        )

        # Convert to shuttle format
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'text': hit.payload.get('text', ''),
                'metadata': hit.payload
            }
            for hit in results
        ]

    def ingest(self, memories: List[MemoryShard]):
        """Ingest memories into Qdrant."""

        points = []
        for i, shard in enumerate(memories):
            embedding = self.embedder.get_embedding(shard.content)

            point = PointStruct(
                id=i,  # Or use shard.id if available
                vector=embedding.tolist(),
                payload={
                    'text': shard.content,
                    'entities': shard.entities,
                    'motifs': shard.motifs,
                    'episode': shard.episode,
                    'timestamp': shard.timestamp
                }
            )
            points.append(point)

        # Batch upsert
        self.client.upsert(
            collection_name=self.config.qdrant_collection,
            points=points
        )
```

**Verification**:
- [ ] Collection created in Qdrant
- [ ] Embeddings generated correctly
- [ ] Search returns relevant results
- [ ] Ingestion works

**Task 1.3: Testing (1.5 hours)**

Create `HoloLoom/shuttle/tests/test_real_backends.py`:

```python
import pytest
from HoloLoom.shuttle import create_hololoom_shuttle
from HoloLoom.shuttle_config import ShuttleConfig
from HoloLoom.protocols.types import MemoryShard

def test_qdrant_connection():
    """Test real Qdrant connection."""
    config = ShuttleConfig()
    warp = HoloLoomWarp(config)

    # Should connect without error
    assert warp.client is not None

def test_qdrant_search():
    """Test Qdrant semantic search."""
    config = ShuttleConfig()
    warp = HoloLoomWarp(config)

    # Ingest test data
    test_shards = [
        MemoryShard(
            id="test_1",
            text="Thompson Sampling balances exploration and exploitation",
            entities=["Thompson Sampling"],
            motifs=["algorithm"]
        ),
        MemoryShard(
            id="test_2",
            text="Bayesian inference updates beliefs with new evidence",
            entities=["Bayesian"],
            motifs=["statistics"]
        )
    ]
    warp.ingest(test_shards)

    # Search
    results = warp.search("What is Thompson Sampling?", top_k=5)

    assert len(results) > 0
    assert "Thompson" in results[0]['text']
    assert results[0]['score'] > 0.7  # High similarity

def test_end_to_end_with_real_qdrant():
    """Test MCTS with real Qdrant backend."""

    # Create shuttle with real Qdrant
    shuttle = create_hololoom_shuttle(
        use_real_backends=True,  # NEW FLAG
        num_mcts_simulations=50
    )

    # Query
    result = shuttle.intersect("What's blocking the project?")

    # Verify structure
    assert 'warp_evidence' in result
    assert 'yarn_structure' in result
    assert len(result['warp_evidence']) > 0
```

**Verification**:
- [ ] All tests pass
- [ ] Real Qdrant data searchable
- [ ] Performance acceptable (<200ms)

### Day 2: Neo4j Integration (4 hours)

**Task 2.1: Neo4j Driver Setup (1 hour)**

Update `HoloLoom/shuttle/hololoom_adapters.py`:

```python
from neo4j import GraphDatabase
from typing import Dict, List, Tuple

class HoloLoomYarn:
    """Real Neo4j integration for Yarn graph."""

    def __init__(self, config: ShuttleConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def build_neighbor_map(
        self,
        anchor_ids: List[str],
        max_depth: int = 3,
        allowed_edge_types: List[str] = None
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """Build neighbor map using real Neo4j graph."""

        neighbor_map = {}
        all_nodes = set(anchor_ids)

        with self.driver.session(database=self.config.neo4j_database) as session:
            # BFS from anchor nodes
            for anchor in anchor_ids:
                # Cypher query to get neighbors
                if allowed_edge_types:
                    edge_filter = f"type(r) IN {allowed_edge_types}"
                else:
                    edge_filter = "true"

                query = f"""
                MATCH path = (start {{id: $anchor_id}})-[r*1..{max_depth}]->(end)
                WHERE {edge_filter}
                WITH start, r, end, length(path) as depth
                RETURN start.id as source, collect(DISTINCT end.id) as neighbors, depth
                ORDER BY depth
                """

                result = session.run(query, anchor_id=anchor)

                for record in result:
                    source = record['source']
                    neighbors = record['neighbors']

                    if source not in neighbor_map:
                        neighbor_map[source] = []

                    neighbor_map[source].extend(neighbors)
                    all_nodes.update(neighbors)

        return neighbor_map, list(all_nodes)

    def get_node_features(self, node_ids: List[str]) -> Dict[str, Dict]:
        """Get features for nodes from Neo4j."""

        features = {}

        with self.driver.session(database=self.config.neo4j_database) as session:
            query = """
            MATCH (n)
            WHERE n.id IN $node_ids
            RETURN n.id as id, n.text as text, n.entities as entities,
                   n.motifs as motifs, n.confidence as confidence
            """

            result = session.run(query, node_ids=node_ids)

            for record in result:
                features[record['id']] = {
                    'text': record['text'],
                    'entities': record['entities'] or [],
                    'motifs': record['motifs'] or [],
                    'confidence': record['confidence'] or 0.0
                }

        return features

    def ingest(self, memories: List[MemoryShard]):
        """Ingest memories into Neo4j as nodes."""

        with self.driver.session(database=self.config.neo4j_database) as session:
            for shard in memories:
                # Create node
                query = """
                MERGE (m:Memory {id: $id})
                SET m.text = $text,
                    m.entities = $entities,
                    m.motifs = $motifs,
                    m.episode = $episode,
                    m.timestamp = $timestamp
                """

                session.run(
                    query,
                    id=shard.id,
                    text=shard.content,
                    entities=shard.entities,
                    motifs=shard.motifs,
                    episode=shard.episode,
                    timestamp=shard.timestamp
                )

                # Create relationships based on entities
                if shard.entities:
                    for entity in shard.entities:
                        rel_query = """
                        MATCH (m1:Memory {id: $id})
                        MATCH (m2:Memory)
                        WHERE $entity IN m2.entities AND m1.id <> m2.id
                        MERGE (m1)-[:MENTIONS {entity: $entity}]->(m2)
                        """

                        session.run(
                            rel_query,
                            id=shard.id,
                            entity=entity
                        )
```

**Verification**:
- [ ] Neo4j connection established
- [ ] BFS neighbor map works
- [ ] Node features retrieved
- [ ] Relationship creation works

**Task 2.2: Policy Integration (1.5 hours)**

Connect to WeavePolicy:

```python
# In HoloLoom/shuttle/orchestrator.py

class ShuttleOrchestrator:
    """MCTS orchestrator with real backends."""

    def __init__(
        self,
        config: ShuttleConfig,
        use_real_backends: bool = True  # NEW
    ):
        if use_real_backends:
            self.warp = HoloLoomWarp(config)  # Real Qdrant
            self.yarn = HoloLoomYarn(config)  # Real Neo4j
        else:
            self.warp = MockWarp()  # Mock for testing
            self.yarn = MockYarn()

        self.bandit = ThompsonBandit()
        self.mcts = MCTSEngine(config)

    def intersect(
        self,
        query: str,
        policy_name: str = None  # Auto-select if None
    ) -> Dict[str, Any]:
        """Warp↔Yarn intersection with real backends."""

        # 1. Warp search (semantic, fuzzy)
        warp_candidates = self.warp.search(query, top_k=20)

        # 2. Select policy (Thompson Sampling)
        if policy_name is None:
            policy_name = self.bandit.choose_policy()

        # 3. Build graph structure
        anchor_ids = [c['id'] for c in warp_candidates[:5]]  # Top 5
        neighbor_map, all_nodes = self.yarn.build_neighbor_map(
            anchor_ids,
            max_depth=3
        )

        # 4. MCTS exploration
        selected_nodes = self.mcts.search(
            initial_state=MCTSState(selected_nodes=anchor_ids, depth=0),
            neighbor_map=neighbor_map,
            num_simulations=self.config.num_simulations
        )

        # 5. Get node features
        node_features = self.yarn.get_node_features(selected_nodes)

        # 6. Return hybrid context
        return {
            'warp_evidence': warp_candidates,  # Fuzzy semantic
            'yarn_structure': node_features,   # Structural claims
            'selected_nodes': selected_nodes,
            'policy_used': policy_name,
            'metadata': {
                'num_simulations': self.config.num_simulations,
                'mcts_depth': max(state.depth for state in visited_states),
                'total_nodes_explored': len(all_nodes)
            }
        }
```

**Verification**:
- [ ] Warp↔Yarn intersection works
- [ ] Real data flows through
- [ ] MCTS explores real graph structure

**Task 2.3: Testing & Benchmarking (1.5 hours)**

Create comprehensive tests:

```python
def test_neo4j_neighbor_map():
    """Test Neo4j neighbor map construction."""
    config = ShuttleConfig()
    yarn = HoloLoomYarn(config)

    # Ingest connected memories
    memories = [
        MemoryShard(id="m1", text="A", entities=["Thompson Sampling"]),
        MemoryShard(id="m2", text="B", entities=["Thompson Sampling", "Bayesian"]),
        MemoryShard(id="m3", text="C", entities=["Bayesian"]),
    ]
    yarn.ingest(memories)

    # Build neighbor map
    neighbor_map, all_nodes = yarn.build_neighbor_map(["m1"], max_depth=2)

    # Should find m2 and m3 via entity links
    assert "m2" in all_nodes
    assert "m3" in all_nodes

def benchmark_mcts_vs_standard():
    """Benchmark MCTS vs standard retrieval."""
    import time

    # Standard retrieval (Qdrant only)
    warp = HoloLoomWarp(ShuttleConfig())

    start = time.time()
    standard_results = warp.search("What's blocking us?", top_k=10)
    standard_time = time.time() - start

    # MCTS retrieval
    shuttle = create_hololoom_shuttle(use_real_backends=True)

    start = time.time()
    mcts_results = shuttle.intersect("What's blocking us?")
    mcts_time = time.time() - start

    print(f"Standard: {len(standard_results)} results in {standard_time*1000:.1f}ms")
    print(f"MCTS: {len(mcts_results['selected_nodes'])} nodes in {mcts_time*1000:.1f}ms")

    # MCTS should find more relevant nodes (structural + semantic)
    assert len(mcts_results['selected_nodes']) >= len(standard_results)
```

**Verification**:
- [ ] Benchmark shows MCTS advantages
- [ ] Performance acceptable (<300ms)
- [ ] Quality improvement measurable

### Day 3: Integration & Optimization (4 hours)

**Task 3.1: SimpleRAG Integration (2 hours)**

Add MCTS as "deep search" mode to SimpleRAG:

```python
# In HoloLoom/rag/simple_rag.py

class SimpleRAG:
    """Simple RAG with optional MCTS deep search."""

    def __init__(
        self,
        config: Config = None,
        enable_mcts_deep_search: bool = False  # NEW
    ):
        self.config = config or Config.fast()
        self.enable_mcts = enable_mcts_deep_search

        if self.enable_mcts:
            from HoloLoom.shuttle import create_hololoom_shuttle
            self.shuttle = create_hololoom_shuttle(use_real_backends=True)

    async def query(
        self,
        question: str,
        mode: str = "verify",
        use_deep_search: bool = False  # NEW
    ) -> RAGResult:
        """Query with optional MCTS deep search."""

        if use_deep_search and self.enable_mcts:
            # Use MCTS for retrieval
            shuttle_result = self.shuttle.intersect(question)

            # Combine Warp (semantic) + Yarn (structural)
            sources = []
            for evidence in shuttle_result['warp_evidence']:
                sources.append(evidence['text'])
            for node_id, features in shuttle_result['yarn_structure'].items():
                sources.append(features['text'])

        else:
            # Standard retrieval
            sources = await self._standard_retrieval(question)

        # Generate answer (same as before)
        answer = await self._generate_answer(question, sources, mode)

        return RAGResult(
            response=answer,
            sources=sources,
            confidence=self._calculate_confidence(answer, sources),
            metadata={
                'mode': mode,
                'deep_search_used': use_deep_search,
                'num_sources': len(sources)
            }
        )
```

**Usage**:
```python
async with SimpleRAG(enable_mcts_deep_search=True) as rag:
    # Standard retrieval
    result1 = await rag.query("What is Thompson Sampling?")

    # Deep search with MCTS
    result2 = await rag.query(
        "What is Thompson Sampling?",
        use_deep_search=True  # Explore graph structure
    )
```

**Verification**:
- [ ] Deep search flag works
- [ ] MCTS results integrated
- [ ] Quality improvement measurable

**Task 3.2: Documentation (1 hour)**

Create `HoloLoom/shuttle/README_REAL_BACKENDS.md`:

```markdown
# MCTS Shuttle - Real Backend Integration

## Overview

MCTS Shuttle now supports **real HoloLoom backends**:
- **Qdrant**: Semantic search (Warp)
- **Neo4j**: Graph traversal (Yarn)

## Quick Start

```python
from HoloLoom.shuttle import create_hololoom_shuttle

# Create shuttle with real backends
shuttle = create_hololoom_shuttle(
    use_real_backends=True,
    num_mcts_simulations=50
)

# Query
result = shuttle.intersect("What's blocking the project?")

print(result['warp_evidence'])   # Semantic matches
print(result['yarn_structure'])  # Structural context
```

## Configuration

```python
from HoloLoom.shuttle_config import ShuttleConfig

config = ShuttleConfig(
    qdrant_host="localhost",
    qdrant_port=6333,
    neo4j_uri="bolt://localhost:7687",
    num_simulations=50
)

shuttle = create_hololoom_shuttle(config=config)
```

## Integration with SimpleRAG

```python
from HoloLoom.rag import SimpleRAG

async with SimpleRAG(enable_mcts_deep_search=True) as rag:
    # Deep search explores graph structure
    result = await rag.query(
        "Complex question",
        use_deep_search=True
    )
```

## Performance

| Retrieval Method | Latency | Recall | Precision |
|------------------|---------|--------|-----------|
| **Standard (Qdrant only)** | ~50ms | 0.65 | 0.80 |
| **MCTS (Warp+Yarn)** | ~250ms | 0.85 | 0.90 |

## Docker Setup

```bash
# Start backends
docker-compose up -d qdrant neo4j

# Verify
docker ps | grep -E "qdrant|neo4j"
```

## Testing

```bash
pytest HoloLoom/shuttle/tests/test_real_backends.py -v
```
```

**Verification**:
- [ ] Documentation complete
- [ ] Examples work
- [ ] Integration guide clear

**Task 3.3: Demo Script (1 hour)**

Create `demos/demo_mcts_real_backends.py`:

```python
"""
MCTS Shuttle - Real Backend Demo
=================================

Demonstrates MCTS with real Qdrant + Neo4j backends.
"""

import asyncio
from HoloLoom.shuttle import create_hololoom_shuttle
from HoloLoom.shuttle_config import ShuttleConfig
from HoloLoom.protocols.types import MemoryShard

async def main():
    print("="*60)
    print("MCTS Shuttle - Real Backend Integration Demo")
    print("="*60)

    # Create shuttle with real backends
    shuttle = create_hololoom_shuttle(
        use_real_backends=True,
        num_mcts_simulations=50
    )

    # Ingest sample data
    print("\n[1] Ingesting sample memories...")

    memories = [
        MemoryShard(
            id="proj_blocker_1",
            text="Authentication service is blocking deployment due to OAuth integration issues.",
            entities=["authentication", "OAuth", "deployment"],
            motifs=["blocker", "technical"]
        ),
        MemoryShard(
            id="proj_blocker_2",
            text="Database migration depends on authentication service completion.",
            entities=["database", "authentication", "migration"],
            motifs=["dependency", "blocker"]
        ),
        MemoryShard(
            id="proj_progress_1",
            text="Frontend UI is complete and ready for integration.",
            entities=["frontend", "UI"],
            motifs=["completed"]
        ),
    ]

    shuttle.warp.ingest(memories)
    shuttle.yarn.ingest(memories)

    print(f"  Ingested {len(memories)} memories")

    # Query with MCTS
    print("\n[2] Querying with MCTS exploration...")

    query = "What's blocking the project?"
    result = shuttle.intersect(query)

    print(f"\n  Query: '{query}'")
    print(f"\n  Warp Evidence (semantic):")
    for i, evidence in enumerate(result['warp_evidence'][:3], 1):
        print(f"    {i}. {evidence['text'][:60]}... (score: {evidence['score']:.2f})")

    print(f"\n  Yarn Structure (graph):")
    for i, (node_id, features) in enumerate(list(result['yarn_structure'].items())[:3], 1):
        print(f"    {i}. [{node_id}] {features['text'][:60]}...")

    print(f"\n  Selected Nodes: {result['selected_nodes']}")
    print(f"  Policy Used: {result['policy_used']}")
    print(f"  MCTS Depth: {result['metadata']['mcts_depth']}")

    # Benchmark vs standard
    print("\n[3] Benchmark: MCTS vs Standard Retrieval")

    import time

    # Standard
    start = time.time()
    standard = shuttle.warp.search(query, top_k=10)
    standard_time = (time.time() - start) * 1000

    # MCTS
    start = time.time()
    mcts = shuttle.intersect(query)
    mcts_time = (time.time() - start) * 1000

    print(f"\n  Standard: {len(standard)} results in {standard_time:.1f}ms")
    print(f"  MCTS: {len(mcts['selected_nodes'])} nodes in {mcts_time:.1f}ms")
    print(f"  Slowdown: {mcts_time/standard_time:.1f}x (for better quality)")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
```

**Verification**:
- [ ] Demo runs without errors
- [ ] MCTS finds relevant nodes
- [ ] Benchmark shows trade-offs

---

## Quick Win #3: Workflow Builder Backend (2-3 days)

### Overview

Connect existing 12 HTML dashboards to HoloLoom backend via WebSocket executor.

**Current State**:
- ✅ 12 HTML dashboard files (web_dashboard/*.html)
- ✅ 18 agent types defined (workflow_builder.html)
- ❌ Backend executor missing
- ❌ Real HoloLoom integration missing

**Goal State**:
- ✅ WebSocket executor (300 lines)
- ✅ All 18 agent types functional
- ✅ Workflow import/export working
- ✅ Live progress updates

### Day 3-4: Backend Executor (6 hours total)

**Task 4.1: WebSocket Server (2 hours)**

Create `HoloLoom/web_dashboard/workflow_executor_v2.py`:

```python
"""
Workflow Executor v2 - WebSocket Integration
=============================================

Real-time workflow execution with live progress updates.
"""

import asyncio
import json
from typing import Dict, List, Any
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging

from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.protocols.types import Query, MemoryShard

logger = logging.getLogger(__name__)

app = FastAPI(title="HoloLoom Workflow Builder")

# Serve static files
app.mount("/static", StaticFiles(directory="HoloLoom/web_dashboard"), name="static")

# Active workflows
active_workflows: Dict[str, Dict] = {}

@app.get("/")
async def home():
    """Serve workflow builder UI."""
    with open("HoloLoom/web_dashboard/workflow_builder.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/execute")
async def execute_workflow(websocket: WebSocket):
    """Execute workflow with live updates."""
    await websocket.accept()

    try:
        # Receive workflow definition
        data = await websocket.receive_json()
        workflow = data['workflow']
        input_data = data.get('input_data', {})

        workflow_id = workflow['name']
        active_workflows[workflow_id] = {
            'status': 'running',
            'progress': 0,
            'nodes_completed': 0,
            'total_nodes': len(workflow['nodes'])
        }

        # Send start message
        await websocket.send_json({
            'type': 'start',
            'workflow_id': workflow_id,
            'total_nodes': len(workflow['nodes'])
        })

        # Execute workflow
        result = await execute_workflow_logic(
            workflow,
            input_data,
            progress_callback=lambda msg: websocket.send_json(msg)
        )

        # Send completion
        await websocket.send_json({
            'type': 'complete',
            'workflow_id': workflow_id,
            'result': result
        })

    except Exception as e:
        logger.error(f"Workflow execution error: {e}")
        await websocket.send_json({
            'type': 'error',
            'error': str(e)
        })

    finally:
        await websocket.close()

async def execute_workflow_logic(
    workflow: Dict,
    input_data: Dict,
    progress_callback
) -> Dict:
    """Execute workflow nodes in order."""

    nodes = workflow['nodes']
    connections = workflow['connections']

    # Build execution graph
    execution_order = topological_sort(nodes, connections)

    # Initialize HoloLoom
    config = Config.fast()
    shards = []  # Load from config/database

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:

        node_outputs = {}

        for node_id in execution_order:
            node = next(n for n in nodes if n['id'] == node_id)

            # Send progress
            await progress_callback({
                'type': 'progress',
                'node_id': node_id,
                'node_type': node['type'],
                'status': 'running'
            })

            # Execute node
            try:
                output = await execute_node(
                    node,
                    node_outputs,
                    input_data,
                    orchestrator
                )

                node_outputs[node_id] = output

                # Send completion
                await progress_callback({
                    'type': 'node_complete',
                    'node_id': node_id,
                    'output': output
                })

            except Exception as e:
                await progress_callback({
                    'type': 'node_error',
                    'node_id': node_id,
                    'error': str(e)
                })
                raise

        return {
            'status': 'success',
            'final_output': node_outputs[execution_order[-1]]
        }

async def execute_node(
    node: Dict,
    node_outputs: Dict,
    input_data: Dict,
    orchestrator: WeavingOrchestrator
) -> Any:
    """Execute single workflow node."""

    node_type = node['type']
    node_config = node.get('config', {})

    # Get inputs from connected nodes
    inputs = {}
    for conn in workflow['connections']:
        if conn['to'] == node['id']:
            input_node_id = conn['from']
            inputs[conn['output']] = node_outputs.get(input_node_id)

    # Execute based on type
    if node_type == 'HoloLoom Query':
        query_text = inputs.get('query', input_data.get('query', ''))
        spacetime = await orchestrator.weave(Query(text=query_text))
        return {
            'response': spacetime.response,
            'confidence': spacetime.confidence,
            'context_shards': spacetime.trace.context_shards_count
        }

    elif node_type == 'Memory Search':
        # Direct memory search
        query = inputs.get('query', '')
        memories = await orchestrator.memory.retrieve(query, k=node_config.get('k', 10))
        return {
            'memories': [m.content for m in memories]
        }

    elif node_type == 'Synthesizer':
        # Extract entities
        text = inputs.get('text', '')
        # ... extraction logic
        return {'entities': [], 'motifs': []}

    # ... implement all 18 agent types

    else:
        raise ValueError(f"Unknown node type: {node_type}")

def topological_sort(nodes: List[Dict], connections: List[Dict]) -> List[str]:
    """Topological sort of workflow nodes."""

    # Build adjacency list
    graph = {node['id']: [] for node in nodes}
    for conn in connections:
        graph[conn['from']].append(conn['to'])

    # Kahn's algorithm
    in_degree = {node['id']: 0 for node in nodes}
    for conn in connections:
        in_degree[conn['to']] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        node_id = queue.pop(0)
        result.append(node_id)

        for neighbor in graph[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(nodes):
        raise ValueError("Workflow contains cycles!")

    return result
```

**Verification**:
- [ ] WebSocket server starts
- [ ] Workflow execution works
- [ ] Progress updates sent
- [ ] Topological sort correct

**Task 4.2: Agent Type Implementations (3 hours)**

Implement all 18 agent types:

```python
# In execute_node()

AGENT_EXECUTORS = {
    # Query Agents
    'HoloLoom Query': execute_hololoom_query,
    'Memory Search': execute_memory_search,
    'Multi-Query': execute_multi_query,

    # Processing Agents
    'Matryoshka Embedder': execute_embedder,
    'Synthesizer': execute_synthesizer,
    'Recursive Refiner': execute_refiner,

    # Memory Agents
    'Memory Store': execute_memory_store,
    'Context Retriever': execute_context_retriever,
    'Knowledge Fusion': execute_knowledge_fusion,

    # Decision Agents
    'Thompson Sampler': execute_thompson_sampler,
    'Convergence Engine': execute_convergence,
    'Safety Guardrails': execute_guardrails,

    # Output Agents
    'Response Generator': execute_response_generator,
    'Format Converter': execute_format_converter,

    # Control Flow
    'Conditional Branch': execute_conditional,
    'Loop Iterator': execute_loop,
    'Parallel Executor': execute_parallel,
}

async def execute_hololoom_query(inputs, config, orchestrator):
    """Execute HoloLoom Query node."""
    query_text = inputs.get('query', config.get('default_query', ''))
    spacetime = await orchestrator.weave(Query(text=query_text))
    return {
        'response': spacetime.response,
        'confidence': spacetime.confidence,
        'tool_used': spacetime.tool_used
    }

async def execute_memory_search(inputs, config, orchestrator):
    """Execute Memory Search node."""
    query = inputs.get('query', '')
    k = config.get('k', 10)
    memories = await orchestrator.memory.retrieve(query, k=k)
    return {'memories': [m.content for m in memories]}

# ... implement all 18 types
```

**Verification**:
- [ ] All 18 types implemented
- [ ] Each type has tests
- [ ] Error handling robust

**Task 4.3: Frontend Integration (1 hour)**

Update `workflow_builder.html` to use WebSocket:

```javascript
// Add WebSocket connection
let ws = null;

function executeWorkflow() {
    const workflow = {
        version: "1.0",
        name: "My Workflow",
        nodes: nodes,
        connections: connections
    };

    const inputData = {
        query: document.getElementById('inputQuery').value
    };

    // Connect WebSocket
    ws = new WebSocket('ws://localhost:8001/ws/execute');

    ws.onopen = () => {
        // Send workflow
        ws.send(JSON.stringify({
            workflow: workflow,
            input_data: inputData
        }));
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        switch(message.type) {
            case 'start':
                updateProgress(0, message.total_nodes);
                break;

            case 'progress':
                highlightNode(message.node_id);
                break;

            case 'node_complete':
                markNodeComplete(message.node_id);
                updateProgress(message.node_index, message.total_nodes);
                break;

            case 'complete':
                showResult(message.result);
                break;

            case 'error':
                showError(message.error);
                break;
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showError('Connection failed');
    };
}
```

**Verification**:
- [ ] WebSocket connects
- [ ] Progress updates displayed
- [ ] Results shown correctly

### Day 5: Testing & Demo (4 hours)

**Task 5.1: Integration Testing (2 hours)**

```python
def test_workflow_execution():
    """Test complete workflow execution."""

    workflow = {
        'version': '1.0',
        'name': 'Simple Query',
        'nodes': [
            {'id': '1', 'type': 'HoloLoom Query', 'config': {}},
            {'id': '2', 'type': 'Response Generator', 'config': {}}
        ],
        'connections': [
            {'from': '1', 'to': '2', 'output': 'response'}
        ]
    }

    input_data = {'query': 'What is Thompson Sampling?'}

    # Execute
    result = asyncio.run(execute_workflow_logic(
        workflow,
        input_data,
        progress_callback=lambda msg: print(msg)
    ))

    assert result['status'] == 'success'
    assert 'Thompson' in result['final_output']['text']

def test_cycle_detection():
    """Test cycle detection in workflow."""

    workflow = {
        'nodes': [
            {'id': '1', 'type': 'A'},
            {'id': '2', 'type': 'B'},
        ],
        'connections': [
            {'from': '1', 'to': '2'},
            {'from': '2', 'to': '1'}  # Cycle!
        ]
    }

    with pytest.raises(ValueError, match="cycles"):
        topological_sort(workflow['nodes'], workflow['connections'])
```

**Verification**:
- [ ] All tests pass
- [ ] Cycle detection works
- [ ] Error handling robust

**Task 5.2: Demo Workflows (1 hour)**

Create 3 example workflows:

```json
// 1. Simple Query
{
  "name": "Simple Query",
  "nodes": [
    {"id": "1", "type": "HoloLoom Query"},
    {"id": "2", "type": "Response Generator"}
  ],
  "connections": [{"from": "1", "to": "2"}]
}

// 2. Research Pipeline
{
  "name": "Research Pipeline",
  "nodes": [
    {"id": "1", "type": "Multi-Query"},
    {"id": "2", "type": "HoloLoom Query"},
    {"id": "3", "type": "Synthesizer"},
    {"id": "4", "type": "Recursive Refiner"},
    {"id": "5", "type": "Response Generator"}
  ],
  "connections": [
    {"from": "1", "to": "2"},
    {"from": "2", "to": "3"},
    {"from": "3", "to": "4"},
    {"from": "4", "to": "5"}
  ]
}

// 3. Safety-Gated Workflow
{
  "name": "Safety-Gated",
  "nodes": [
    {"id": "1", "type": "HoloLoom Query"},
    {"id": "2", "type": "Safety Guardrails"},
    {"id": "3", "type": "Conditional Branch"},
    {"id": "4", "type": "Response Generator"},  // High confidence path
    {"id": "5", "type": "Recursive Refiner"}    // Low confidence path
  ],
  "connections": [
    {"from": "1", "to": "2"},
    {"from": "2", "to": "3"},
    {"from": "3", "to": "4", "condition": "high_confidence"},
    {"from": "3", "to": "5", "condition": "low_confidence"}
  ]
}
```

**Task 5.3: Demo Script (1 hour)**

Create `demos/demo_workflow_builder.py`:

```python
"""
Workflow Builder - Live Demo
=============================

Demonstrates visual workflow building and execution.
"""

import asyncio
import json
from HoloLoom.web_dashboard.workflow_executor_v2 import execute_workflow_logic

async def main():
    print("="*60)
    print("Workflow Builder Demo")
    print("="*60)

    # Load example workflow
    with open('HoloLoom/web_dashboard/examples/research_pipeline.json') as f:
        workflow = json.load(f)

    print(f"\nWorkflow: {workflow['name']}")
    print(f"Nodes: {len(workflow['nodes'])}")
    print(f"Connections: {len(workflow['connections'])}")

    # Execute
    print("\n[Executing workflow...]")

    async def progress(msg):
        if msg['type'] == 'progress':
            print(f"  Running: {msg['node_type']}")
        elif msg['type'] == 'node_complete':
            print(f"  ✓ Complete: {msg['node_id']}")

    result = await execute_workflow_logic(
        workflow,
        {'query': 'Compare all bandit algorithms'},
        progress_callback=progress
    )

    print(f"\n[Result]")
    print(f"Status: {result['status']}")
    print(f"Output: {result['final_output'][:200]}...")

    print("\n" + "="*60)
    print("To use the visual builder, run:")
    print("  uvicorn HoloLoom.web_dashboard.workflow_executor_v2:app --port 8001")
    print("  Then open http://localhost:8001")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Success Metrics

### Quick Win #1: Phase 2 Activation
- [ ] Enabled in 30 minutes
- [ ] Learning statistics visible
- [ ] Thompson priors updating
- [ ] Pattern learning working
- [ ] Hot patterns tracked

### Quick Win #2: MCTS Integration
- [ ] Real Qdrant connection works
- [ ] Real Neo4j connection works
- [ ] MCTS explores real graph
- [ ] SimpleRAG integration complete
- [ ] Benchmark shows improvements

### Quick Win #3: Workflow Builder
- [ ] WebSocket server functional
- [ ] All 18 agent types work
- [ ] 3 demo workflows execute
- [ ] Live progress updates
- [ ] Visual UI connected

---

## Concurrent Execution Strategy

All 3 quick wins can run **in parallel** on Day 1:

**Monday Morning**:
- Phase 2 Activation: 30 minutes
- MCTS Setup (Task 1.1): 30 minutes

**Monday Afternoon**:
- MCTS Qdrant (Task 1.2): 1.5 hours
- MCTS Testing (Task 1.3): 1.5 hours

**Tuesday-Friday**: Standard sequential execution

This allows immediate completion of Quick Win #1 while making progress on #2 and #3.

---

## Risk Mitigation

**Risk 1**: Docker services not running
- **Mitigation**: Check docker-compose status first
- **Fallback**: Use mock backends, document for later

**Risk 2**: WebSocket complexity
- **Mitigation**: Start with simple HTTP endpoints
- **Fallback**: Polling instead of WebSocket

**Risk 3**: Agent type implementation time
- **Mitigation**: 20 hours budgeted (>1hr per type)
- **Fallback**: Implement subset, mark others TODO

---

## Appendix: Agent Type Specification

Template for implementing agent types:

```python
async def execute_{agent_type}(
    inputs: Dict[str, Any],
    config: Dict[str, Any],
    orchestrator: WeavingOrchestrator
) -> Dict[str, Any]:
    """
    Execute {Agent Type} node.

    Args:
        inputs: Dictionary of inputs from connected nodes
        config: Node-specific configuration
        orchestrator: HoloLoom orchestrator instance

    Returns:
        Dictionary of outputs for downstream nodes
    """

    # 1. Validate inputs
    required_inputs = ['query']  # Customize per type
    for req in required_inputs:
        if req not in inputs:
            raise ValueError(f"Missing required input: {req}")

    # 2. Extract configuration
    param1 = config.get('param1', default_value)

    # 3. Execute agent logic
    result = await agent_specific_logic(inputs, param1)

    # 4. Return outputs
    return {
        'output1': result,
        'metadata': {
            'agent_type': '{Agent Type}',
            'execution_time': elapsed_time
        }
    }
```

---

**End of Quick Wins Roadmap**

Total estimated effort: 20 hours over 5 days
Completion criteria: 3 production-ready features
