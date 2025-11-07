# HoloLoom Memory Systems Summary

Status: Production-Ready
Last Updated: November 3, 2025
Files: 37 Python modules across memory system
Architecture: Protocol-based with auto-fallback

## THE THREE BACKENDS

### 1. INMEMORY (Development)
Implementation: NetworkX in-memory
Speed: <10ms | Persistence: None | Dependencies: Zero
Use Case: Development, testing

### 2. HYBRID (Production - RECOMMENDED)
Backends: Neo4j (graph) + Qdrant (vectors)
Speed: ~50ms | Persistence: Docker-backed | Fallback: Automatic
Use Case: Production deployment

Fallback Chain:
1. Neo4j + Qdrant (production)
2. Neo4j only (degraded)
3. Qdrant only (degraded)
4. NetworkX (emergency fallback - system continues)

### 3. HYPERSPACE (Research)
Algorithm: Gated multipass recursive crawling
Complexity: 4 progressive passes (0.6, 0.75, 0.85, 0.9 thresholds)
Speed: ~150ms total
Features: Matryoshka importance gating

## AUTO-FALLBACK MECHANISM

Key Properties:
1. Never Crashes - Always returns working backend
2. Graceful Degradation - Logs all fallback events
3. Seamless API - Same MemoryStore protocol
4. Transparent - Calling code unchanged

Fallback Messages:
- Warns on backend failures
- Suggests recovery steps
- Shows active mode (production/degraded/fallback)
- Reports per-backend health

## KNOWLEDGE GRAPH (YarnGraph)

Purpose: Structural memory (how concepts relate)
Different from: Vector memory (semantic similarity)

Implementation:
- Default: NetworkX MultiDiGraph (graph.py)
- Production: Neo4j (neo4j_graph.py)
- Features: Typed edges, multi-edges, weights, metadata

Edge Types:
IS_A       - Taxonomy: "Python" IS_A "Language"
USES       - Functional: "Attention" USES "Softmax"
MENTIONS   - Reference: "Paper1" MENTIONS "Concept"
LEADS_TO   - Causal: "Cause" LEADS_TO "Effect"
PART_OF    - Composition: "Neuron" PART_OF "Layer"
IN_TIME    - Temporal: "Event" IN_TIME "2025-11-03"
OCCURRED_AT - Location: "Meeting" OCCURRED_AT "Office"

Key Operations:
- add_edge(edge) - Add relationship
- get_neighbors(entity, direction, max_hops) - Find adjacent
- subgraph_for_entities(entities, expand, max_hops) - Extract context
- get_paths(src, dst, max_length) - Find reasoning paths
- connect_entity_to_time(entity, timestamp) - Temporal threading

## DOCKER INTEGRATION

Services:
neo4j        - Graph database (port 7687)
qdrant       - Vector database (port 6333)
hololoom     - Application (port 8000)
prometheus   - Metrics (port 9090)
grafana      - Dashboards (port 3000)

Environment Variables (auto-detected):
NEO4J_URI: bolt://neo4j:7687
NEO4J_USERNAME: neo4j
NEO4J_PASSWORD: hololoom123
QDRANT_HOST: localhost/qdrant
QDRANT_PORT: 6333

## WEAVING CYCLE INTEGRATION

Step 3 of 9: YARN GRAPH - Thread selection
Memory sources (priority):
1. Unified Backend (persistent, recommended)
2. Yarn Graph (graph-based)
3. Memory Shards (backward compatible)

Thread Selection:
Query -> retrieval_queries -> memory.recall() -> threads
-> context expansion via KG -> features -> DotPlasma

## PERFORMANCE BENCHMARKS

Operation               INMEMORY    HYBRID      HYPERSPACE
store()                <1ms        5ms         10ms
recall(10 items)       10ms        20ms        30ms
recall(50 items)       20ms        50ms        150ms
subgraph_extraction    5ms         15ms        25ms
get_neighbors()        <1ms        2ms         5ms
get_paths()            1ms         5ms         15ms

## KEY FILES

Core:
protocol.py             120 lines   MemoryStore interface
backend_factory.py      231 lines   Creation + fallback logic

Knowledge Graphs:
graph.py                ~400 lines  NetworkX implementation
neo4j_graph.py          ~200 lines  Neo4j backend

Advanced:
hyperspace_backend.py   ~250 lines  Gated multipass crawling
unified.py              ~300 lines  User-facing API
weaving_adapter.py      ~200 lines  Orchestrator bridge

Storage:
cache.py                ~400 lines  BM25 + semantic retrieval
stores/*.py             ~2000 lines 12 store implementations

Total: 37 Python files, ~5,500 lines

## ARCHITECTURE LAYERS

Memory System
  Core Protocols (120 lines)
    MemoryStore - Backend interface
    MemoryNavigator - Graph traversal
    Memory/MemoryQuery/RetrievalResult - Types
  
  Backend Factory (231 lines)
    create_memory_backend() - Intelligent factory
    Auto-fallback chain orchestration
  
  Three Backends
    INMEMORY: NetworkX (dev)
    HYBRID: Neo4j+Qdrant (prod)
    HYPERSPACE: Gated multipass (research)
  
  Knowledge Graph
    NetworkX KG (default, always works)
    Neo4j KG (production, scalable)
  
  Unified Interface
    unified.py - Elegant API
    weaving_adapter.py - Bridge to WeavingOrchestrator
  
  Storage Implementations (12 stores)
    in_memory_store.py - Development
    neo4j_store.py, neo4j_vector_store.py - Neo4j
    qdrant_store.py - Vector DB
    hybrid_store.py, hybrid_neo4j_qdrant.py - Hybrids
    mem0_store.py - Intelligent extraction
    file_store.py, beekeeping_strategy.py - Others

## RELIABILITY FEATURES

Safety Guardrails:
- All operations integrated with alignment framework
- memory.store() - Safety checked
- memory.recall() - Retrieval risk assessed
- memory.delete() - Requires approval

Error Handling:
- Never crashes
- Always falls back gracefully
- Exceptions caught and logged
- System continues working

Monitoring:
- health_check() - Backend status
- Status: healthy | degraded | unhealthy
- Mode: production | fallback
- Per-backend breakdown

## SUMMARY: Why It Works

1. RELIABLE
   - Never crashes, auto-fallback to working backend
   - Even if Docker completely unavailable
   - Even if networks fail
   - Even if services misconfigured

2. FLEXIBLE
   - Three backends for different use cases
   - Development (fast), Production (robust), Research (capable)
   - Easy switching via config

3. TRANSPARENT
   - Clear degradation logging
   - Shows which backends failed
   - Suggests recovery steps
   - API unchanged on fallback

4. INTEGRATED
   - Seamless weaving cycle integration
   - Memory adapter bridge
   - Thread selection from queries

5. PRODUCTION-READY
   - Safety guardrails throughout
   - Health checks available
   - Monitoring and metrics
   - Comprehensive testing

6. SIMPLE
   - 3 backends vs previous 10+
   - Clear mental model
   - Better fallback semantics

Philosophy: "Reliable Systems: Safety First"
Prioritizes continued operation over performance
Fallbacks ensure system never fully breaks
Graceful degradation with clear feedback

## SIMPLIFICATION (October 2025)

Before: 10+ backend enums
NETWORKX, NEO4J, QDRANT, MEM0, HYPERSPACE, etc.

After: 3 clean backends
INMEMORY (NetworkX), HYBRID (Neo4j+Qdrant), HYPERSPACE (gated multipass)

Impact:
- backend_factory.py: 550 -> 231 lines (-58%)
- protocol.py: 787 -> 120 lines (-84%)
- Simpler developer experience
- Better cognitive model
- Clearer migration paths
