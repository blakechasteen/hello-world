# Memory Bus MVP — Architecture v0.2 (2026-02-09)

## Design Thesis

> The model is a reasoning engine, not a memory warehouse.
> Durable memory is external, queryable, auditable, versioned.

## Architectural Invariants

1. Embeddings are an index, not a store (DEFERRED in MVP).
2. Structured query first; semantic fallback is deferred/stubbed.
3. Every memory injection has a token budget; never exceed it.
4. Plans never live only in context (write-through at all pressure tiers).
5. Context is rebuildable from the database.
6. Scarcity triggers delegation, not compression.
7. Forgetting is first-class (decay/summarize/archive are hooks; minimal scaffolding in MVP).

## Quick Start

### 1. Start Services

```bash
docker compose -f docker-compose.memory.yml up -d
```

This starts:
- **Neo4j** on `bolt://localhost:7687` (authority store for entities/episodes)
- **Postgres** on `localhost:5432` (audit log, configs, plans, entity aliases)

### 2. Install Dependencies

```bash
pip install pydantic asyncpg neo4j pytest pytest-asyncio
```

### 3. Run Demo

```bash
python -m memory_bus "What treatment should I apply to Aurora next?"
```

### 4. Run Unit Tests

```bash
pytest memory_bus/tests/ -v -k "not integration"
```

### 5. Run Integration Tests (requires Docker)

```bash
MEMORY_BUS_INTEGRATION=1 pytest memory_bus/tests/test_integration.py -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MemoryBus (tools.py)                  │
│  MCP Tools: query, store, entities, resolve, explain    │
│  Resources: memory://status, memory://pressure          │
├──────────────┬──────────────┬───────────────────────────┤
│    Router    │  Promotion   │     Pressure Engine       │
│  (router.py) │ (promotion.py)│     (pressure.py)        │
├──────────────┴──────────────┴───────────────────────────┤
│                  Formatter (formatter.py)                │
│          Progressive detail + token budgeting            │
├─────────────────────┬───────────────────────────────────┤
│   Neo4j Adapter     │       Postgres Adapter            │
│  (neo4j_adapter.py) │    (postgres_adapter.py)          │
│  :Entity, :Episode  │  audit, configs, plans, aliases   │
└─────────────────────┴───────────────────────────────────┘
```

## Query Routing Cascade

Queries are routed through a structured-first cascade:

| Priority | Path | Trigger | Description |
|----------|------|---------|-------------|
| 1 | **EXACT** | `entity_ids` provided | Direct fetch by node ID |
| 2 | **STRUCTURED** | `entity_names` provided | Resolve name → ID, then Cypher query |
| 3 | **GRAPH** | Free text, no IDs/names | Recent episodes, graph traversal |
| 4 | **SEMANTIC** | Fallback (STUB) | Returns "not available" + guidance |

Semantic search is intentionally stubbed in MVP. The interface exists for future Qdrant integration.

## Token-Budgeted Formatting

The formatter enforces a hard token budget using progressive detail degradation:

1. Try at requested detail level (FULL → COMPACT → SUMMARY → TITLE)
2. If over budget, degrade to next lower detail level
3. If still over at TITLE, truncate item count
4. **Never exceed the token budget**

Default budget: 15% of context window, modulated by pressure tier.

## Pressure Tiers

| Tier | Utilization | Effects |
|------|-------------|---------|
| **Tier 0** | <50% | Full priming, no restrictions |
| **Tier 1** | 50-75% | Reduced priming, min_importance=0.2 |
| **Tier 2** | 75-90% | Minimal priming, semantic disabled, 2000 token cap |
| **Tier 3** | >90% | Emergency, min_importance=0.6, 500 token cap |

Plan write-through is enforced at **all tiers** (Invariant #4).

## MCP Tools

### `memory_query`
Query memory with structured-first routing.

```python
result = await bus.memory_query(
    text="What treatment should I apply?",
    entity_names=["Aurora"],
    detail_level="compact",
    token_budget=2000,
)
```

### `memory_store`
Store entities, episodes, aliases, plans.

```python
await bus.memory_store(
    entities=[{"name": "Hive Aurora", "type": "object", "importance": 0.8}],
    episodes=[{"summary": "Inspected hive", "entity_ids": ["ent_aurora"]}],
)
```

### `memory_entities`
List or search entities.

```python
entities = await bus.memory_entities(name="Aurora")
```

### `memory_resolve_entity`
Resolve a name to ranked entity candidates.

```python
result = await bus.memory_resolve_entity(name="Aurora")
# result.best_match.entity_id, result.candidates
```

### `memory_explain`
Get provenance chain for a memory item.

```python
explain = await bus.memory_explain(item_id="ep_001")
# explain.derived_from, explain.confidence, explain.audit_trail
```

## MCP Resources

### `memory://status`
```json
{
  "status": "healthy",
  "neo4j_connected": true,
  "postgres_connected": true,
  "pressure_tier": "tier0",
  "context_window": 128000
}
```

### `memory://pressure`
```json
{
  "tier": "tier1",
  "signals": {
    "context_tokens_used": 60000,
    "context_utilization_pct": 46.9,
    "retrieval_result_count": 5,
    "retrieval_latency_ms": 45.2
  },
  "policy": {
    "priming_strategy": "reduced",
    "min_importance": 0.2,
    "semantic_fallback_enabled": true,
    "per_query_token_cap": null,
    "plan_write_through": true
  }
}
```

## Data Models

### Neo4j

- `:Entity` — `{id, type, name, created_at, updated_at, importance, loom_origin}`
- `:Episode` — `{id, type, timestamp, summary, content, significance, loom_origin}`
- `(:Episode)-[:ABOUT]->(:Entity)` — relationship

### Postgres

- `memory_audit` — audit log for every query/store/resolve/explain
- `configs` — key-value config store
- `plans` — plan state tracking (write-through)
- `entity_aliases` — name→entity_id mapping with confidence scores

## Auditability

Every operation produces a `memory_audit` row containing:
- `loop_id` / `loom_id` for traceability
- `action` (query/store/resolve/explain/promote/error)
- `resolution_path` (exact/structured/graph/semantic)
- `tokens_used`
- `pressure_tier`
- `details` (JSON with query text, result IDs, latency, errors)

Resolution path distribution is tracked and can be queried to detect over-reliance on semantic fallback (>20% threshold).

## Deferred Features (Interfaces Only)

- **Qdrant** (vector store) — interface ready, returns "not available"
- **Object store** — not implemented
- **Decay/summarize/archive** — hooks exist in PromotionEngine
- **Embedding-based similarity** — deferred
- **Deferred promotion batch** — queue exists, flush_deferred() implemented

## File Layout

```
memory_bus/
├── __init__.py              # Package metadata
├── __main__.py              # CLI entry point
├── models.py                # Pydantic data models
├── neo4j_adapter.py         # Neo4j entity/episode CRUD
├── postgres_adapter.py      # Postgres audit/configs/plans/aliases
├── router.py                # Query routing cascade
├── formatter.py             # Token-budgeted context formatting
├── pressure.py              # Pressure policy engine
├── tools.py                 # MemoryBus facade (MCP tools + resources)
├── resolve.py               # Entity resolution
├── explain.py               # Provenance/explain
├── promotion.py             # Write-through + deferred promotion
├── token_estimator.py       # Simple heuristic (replaceable)
├── demo.py                  # Demo script
└── tests/
    ├── conftest.py          # Shared fixtures
    ├── test_models.py       # Model unit tests
    ├── test_formatter.py    # Formatter invariant tests
    ├── test_pressure.py     # Pressure tier + policy tests
    ├── test_router.py       # Routing cascade tests
    ├── test_resolve.py      # Entity resolution tests
    └── test_integration.py  # Docker integration tests
```
