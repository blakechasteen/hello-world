# Memory Architecture Spec Analysis

**Date:** 2026-02-09
**Context:** Analysis of the "True Database-Backed Memory Architecture" spec against actual HoloLoom codebase state.

---

## Executive Summary

The spec proposes a principled memory architecture with 7 memory types, multi-backend storage, structured-first queries, token-budgeted retrieval, and federated access. After deep codebase exploration, the finding is:

- **~60-70% of the spec's "MVP" is already built** (Neo4j, Qdrant, hybrid fallback, context packing, consolidation, query cascade)
- **The spec correctly identifies 4-5 real gaps** (Memory Bus gateway, memory typing, contradiction detection, context priming, MCP surface)
- **The spec misses major existing systems** (Federation at 550KB+, Spring Dynamics, Multi-Wave Engine, Thompson Sampling query router)
- **The highest-value work is coordination, not backends** — 11 memory systems exist but lack a unified gateway

---

## What Already Exists (No Work Needed)

| Spec Proposal | Actual Implementation | Location |
|---|---|---|
| Neo4j as authority store | Production-grade with connection pooling, Cypher, health checks | `memory/neo4j_graph.py` (49KB) |
| Qdrant for semantic index | Multi-scale embeddings (96/192/384D), retry logic, gRPC | `memory/stores/qdrant_store.py` (32KB) |
| Hybrid fallback chain | HYBRID→Neo4j+Qdrant→Neo4j→Qdrant→NetworkX | `memory/backend_factory.py` |
| Token-budgeted retrieval | Information-theoretic packing, 7-signal importance, 40-90% savings | `context_packing/packer.py` |
| Structured-first query cascade | BM25→Graph→Semantic with Reciprocal Rank Fusion | `memory/hybrid_retrieval.py` |
| Importance decay | Spring Dynamics (Hooke's Law), Multi-Wave (5 brain modes), Hot Pattern Feedback | `memory/spring_dynamics.py`, `memory/multi_wave_engine.py` |
| Basic consolidation | Background every 60min, episodic→semantic via LLM, dedup at 95% | `memory/consolidation.py` |
| Multi-Loom federation | SWIM gossip, Kademlia DHT, distributed alignment, Byzantine safety | `federation/` (550KB+) |
| Query routing intelligence | 4 patterns + Thompson Sampling for ambiguous queries | `context/router.py` |

## Real Gaps the Spec Correctly Identifies

### Gap 1: No Unified Memory Bus / Gateway
11 memory systems, a Conductor, a Context Packer, a Hybrid Retriever — but no single gateway enforcing budgets, tracking provenance, and deciding promotion on every access.

### Gap 2: No Explicit Memory Typing at Storage Level
Memories stored as `MemoryShard` with metadata. No schema-level distinction between episodic, factual, procedural, plan, config, or artifact. The consolidation system distinguishes episodic vs semantic, and `MemoryScope` has SESSION/AGENT/KNOWLEDGE/ARCHIVE, but the richer 7-type taxonomy isn't codified.

### Gap 3: No Postgres/Relational Layer for Flat Access
SQL exists only for RAG context packing. Facts, plans, configs, and audit logs would benefit from relational storage for fast queries like "all active plans" or "config value for X."

### Gap 4: No Contradiction Detection Within Single-Agent Memory
CRDT-based conflict resolution exists for multi-user collaboration (`collaboration/sync.py`) but not for detecting when a new memory contradicts an existing one during single-agent operation.

### Gap 5: No Context Priming / Auto-Load
No automatic "before you ask, here's what's relevant" injection based on detected entities and active plans. The query classifier and packer exist, but entity-triggered pre-loading doesn't.

### Gap 6: MCP Tool Surface Is Minimal
Existing MCP server exposes primarily `query_sql`. The spec's 3-tool surface (`memory_query`, `memory_store`, `memory_entities`) with structured parameters would be a better interface for model-driven memory access.

## Where the Spec Over-Engineers

1. **Four separate backends** (Neo4j + Postgres + Qdrant + Object Store) is heavy. The existing Hybrid (Neo4j + Qdrant) + NetworkX fallback works. Adding Postgres is justified for flat access; Object Store is premature.

2. **Backend schemas before gateway logic** — The spec spends more time on Cypher patterns and SQL tables than on the Memory Bus enforcement logic, which is the actual gap.

3. **Assumes greenfield** — The "Week 1-2 MVP" items are largely built. An implementation plan should start from current state.

## Adjusted Priority Stack

```
HIGH VALUE / LOW EFFORT (do first):
├── Expand MCP tool surface (memory_query, memory_store, memory_entities)
├── Add MemoryType enum (EPISODIC, ENTITY, FACTUAL, PROCEDURAL, PLAN, CONFIG, ARTIFACT)
├── Wire context packer as mandatory gate on all context injection
├── Add contradiction detection on memory store path
└── Context priming: auto-load entities mentioned in user input

MEDIUM VALUE / MEDIUM EFFORT:
├── Unified Memory Bus wrapping Conductor + Packer + Promotion filter
├── Postgres for flat access (plans, configs, audit log)
├── Entity versioning with HAS_VERSION edges in Neo4j
├── Promotion filter with novelty scoring (should_promote())
└── Document the Federation system in CLAUDE.md (550KB+ undocumented!)

LOWER PRIORITY:
├── Summarization engine (episode clusters → facts)
├── Object Store for large artifacts
├── Memory-about-memory (Meta Loom self-observation)
└── SAE-driven memory inspection (Dark Trace integration)
```

## Open Questions from the Spec (With Current Answers)

| Question | Current State |
|---|---|
| Token budget: fixed % vs adaptive? | Context packer uses adaptive MI-based budgets per complexity level. Adaptive Budget Learner with Thompson Sampling exists. |
| Qdrant vs pgvector? | Qdrant is deployed and integrated. pgvector not present. Qdrant is the better choice for multi-scale Matryoshka embeddings. |
| How does model learn structured queries? | Thompson Sampling query router in `context/router.py` with 7-rule decision tree + bandit for ambiguous cases. Prompt engineering remains the unsolved piece. |
| Memory Bus sync or async? | Existing systems are async (asyncio throughout). Memory Bus should be async. |
| Memory during strange loop detection? | Multi-Wave Engine's REM mode creates creative bridges. No explicit strange-loop-aware memory behavior yet. |

## Key Takeaway

The spec is a good architectural document that correctly names the design thesis and several real gaps. Its main value is as a coordination framework — the Memory Bus / Gateway concept and the explicit memory typing taxonomy. The backend and query infrastructure it proposes is largely built. The work ahead is wiring existing systems behind a disciplined gateway, not building new backends.
