#!/usr/bin/env python3
"""
Integration test: LiteMemoryBus ↔ WeavingOrchestrator ↔ Contextual Bandits ↔ LLM

Tests the full cascade:
  1. LiteMemoryBus stores facts, entities, and edges
  2. Queries resolve through 4 levels (exact → structured → graph → semantic)
  3. Orchestrator uses bus as memory backend for weaving
  4. Forgetting (TTL + decay + consolidation) works
  5. Resolution path tracking shows which levels are used

Run: python test_memory_bus_integration.py
"""
import asyncio
import sys
import time
import logging

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.WARNING)

from hololoom.memory.lite_bus import LiteMemoryBus
from hololoom.memory.bus import MemoryQuery, MemoryItem
from hololoom.memory.bus_config import MemoryBusConfig, ResolutionPath
from hololoom.documentation.types import MemoryShard


def banner(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


async def test_cascade():
    """Test the 4-level cascade directly."""
    banner("Test 1: 4-Level Cascade")
    bus = LiteMemoryBus()
    await bus.initialize()

    # Store some entities and facts
    ts_id = await bus.store(MemoryItem(
        content="Thompson Sampling",
        memory_type="entity",
        properties={"type": "algorithm"},
        importance=0.9,
    ))
    await bus.add_alias("ts", ts_id)
    await bus.add_alias("thompson sampling", ts_id)

    eg_id = await bus.store(MemoryItem(
        content="Epsilon Greedy",
        memory_type="entity",
        properties={"type": "algorithm"},
        importance=0.8,
    ))
    await bus.add_alias("epsilon greedy", eg_id)
    await bus.add_alias("e-greedy", eg_id)

    # Store facts connected to entities
    f1_id = await bus.store(MemoryItem(
        content="Thompson Sampling uses Bayesian posteriors to balance exploration and exploitation",
        memory_type="factual",
        entity_ids=[ts_id],
        importance=0.85,
    ))
    f2_id = await bus.store(MemoryItem(
        content="Epsilon greedy explores with probability epsilon and exploits otherwise",
        memory_type="factual",
        entity_ids=[eg_id],
        importance=0.8,
    ))
    f3_id = await bus.store(MemoryItem(
        content="Thompson Sampling outperforms epsilon-greedy when the action space is large",
        memory_type="factual",
        entity_ids=[ts_id, eg_id],
        importance=0.9,
    ))

    # Add edge between the two algorithms
    await bus.store_edge(ts_id, eg_id, "COMPARED_TO")

    # Store episodic memory
    ep_id = await bus.store(MemoryItem(
        content="User asked about bandit algorithms for meal prep rotation on 2025-12-01",
        memory_type="episodic",
        entity_ids=[ts_id, eg_id],
        importance=0.7,
    ))

    print(f"  Stored: 2 entities, 3 facts, 1 episode, 1 edge")
    print(f"  IDs: TS={ts_id}, EG={eg_id}")

    # Level 1: EXACT — lookup by ID
    r1 = await bus.query(MemoryQuery(intent="", entity_ids=[ts_id]))
    assert r1.resolution_path == "exact", f"Expected exact, got {r1.resolution_path}"
    assert len(r1.items) == 1
    print(f"  L1 EXACT:      {len(r1.items)} items | {r1.resolution_path} | '{r1.items[0].get('content', r1.items[0].get('name', ''))[:50]}'")

    # Level 2: STRUCTURED — filter by type
    r2 = await bus.query(MemoryQuery(intent="", memory_type="factual"))
    assert r2.resolution_path == "structured", f"Expected structured, got {r2.resolution_path}"
    assert len(r2.items) == 3
    print(f"  L2 STRUCTURED: {len(r2.items)} items | {r2.resolution_path}")

    # Level 3: GRAPH — keyword search
    r3 = await bus.query(MemoryQuery(intent="Bayesian exploration exploitation"))
    assert r3.resolution_path == "graph", f"Expected graph, got {r3.resolution_path}"
    assert len(r3.items) > 0
    print(f"  L3 GRAPH:      {len(r3.items)} items | {r3.resolution_path} | top: '{r3.items[0].get('content', '')[:60]}'")

    # Level 4: SEMANTIC — should return empty (no semantic_fn configured)
    r4 = await bus.query(MemoryQuery(intent="", semantic_text="What is bandit optimization?"))
    assert r4.resolution_path in ("semantic", "none")
    print(f"  L4 SEMANTIC:   {len(r4.items)} items | {r4.resolution_path} (no semantic_fn = empty)")

    # Entity resolution
    candidates = await bus.resolve_entity("ts")
    assert len(candidates) > 0
    assert candidates[0]["entity_id"] == ts_id
    print(f"  Alias 'ts' → {candidates[0]['entity_id']} (confidence={candidates[0]['confidence']})")

    # Stats
    status = bus.status()
    print(f"\n  Status: {status['item_count']} items, {status['edge_count']} edges, {status['alias_count']} aliases")
    print(f"  Resolution distribution: {status['resolution_distribution']}")

    await bus.close()
    return True


async def test_forgetting():
    """Test TTL, decay, and consolidation."""
    banner("Test 2: Forgetting (Invariant 7)")
    bus = LiteMemoryBus(default_ttl=2.0)  # 2 second TTL
    await bus.initialize()

    # Store items
    for i in range(5):
        await bus.store(MemoryItem(
            content=f"Ephemeral fact {i}",
            memory_type="factual",
            importance=0.3 + i * 0.1,
        ))

    count_before = bus.status()["item_count"]
    print(f"  Before TTL: {count_before} items")

    # Wait for TTL
    await asyncio.sleep(2.5)

    # Query triggers GC
    r = await bus.query(MemoryQuery(intent="anything"))
    count_after = bus.status()["item_count"]
    print(f"  After TTL:  {count_after} items (forgot {count_before - count_after})")
    assert count_after == 0, f"Expected 0 items after TTL, got {count_after}"

    # Test consolidation
    bus2 = LiteMemoryBus()
    await bus2.initialize()
    for i in range(15):
        await bus2.store(MemoryItem(
            content=f"Episode {i}: something happened",
            memory_type="episodic",
            importance=0.3 + (i % 5) * 0.1,
        ))

    before = bus2.status()["item_count"]
    summary_id = await bus2.consolidate(memory_type="episodic", max_items=5)
    after = bus2.status()["item_count"]
    print(f"  Consolidation: {before} → {after} items (merged {before - after} into summary)")
    assert after <= 6  # 5 kept + 1 summary
    assert summary_id is not None

    await bus.close()
    await bus2.close()
    return True


async def test_shard_migration():
    """Test loading existing shards into the bus."""
    banner("Test 3: Shard Migration")

    shards = [
        MemoryShard(id="s1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.", metadata={"topic": "algorithms"}),
        MemoryShard(id="s2", text="It explores proportionally to the probability of being optimal.", metadata={"topic": "algorithms"}),
        MemoryShard(id="s3", text="Applications include A/B testing, recommendation systems, and clinical trials.", metadata={"topic": "applications"}),
        MemoryShard(id="s4", text="The main tradeoff is computational overhead from maintaining distributions.", metadata={"topic": "tradeoffs"}),
    ]

    bus = LiteMemoryBus()
    await bus.initialize()
    loaded = await bus.load_shards(shards)
    print(f"  Loaded {loaded} shards into bus")

    # Query by keyword (Level 3)
    r = await bus.query(MemoryQuery(intent="Bayesian bandit"))
    assert len(r.items) > 0
    print(f"  Query 'Bayesian bandit': {len(r.items)} items via {r.resolution_path}")
    print(f"    Top: '{r.items[0].get('content', '')[:80]}'")

    # Query by type (Level 2)
    r2 = await bus.query(MemoryQuery(intent="", memory_type="factual"))
    assert len(r2.items) == 4
    print(f"  Query type=factual: {len(r2.items)} items via {r2.resolution_path}")

    await bus.close()
    return True


async def test_bus_with_orchestrator():
    """Test LiteMemoryBus as the orchestrator's memory backend."""
    banner("Test 4: Bus ↔ Orchestrator Integration")

    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode
    from hololoom.awareness.llm_integration import OllamaLLM
    from hololoom.config import Config
    from hololoom.documentation.types import Query

    MODEL = "qwen2.5:14b"

    # Create bus and load knowledge
    bus = LiteMemoryBus()
    await bus.initialize()

    # Store structured knowledge (entities + facts + edges)
    ts_id = await bus.store(MemoryItem(
        content="Thompson Sampling", memory_type="entity",
        properties={"type": "algorithm"}, importance=0.9
    ))
    eg_id = await bus.store(MemoryItem(
        content="Epsilon Greedy", memory_type="entity",
        properties={"type": "algorithm"}, importance=0.8
    ))
    await bus.store_edge(ts_id, eg_id, "COMPARED_TO")
    await bus.add_alias("thompson sampling", ts_id)
    await bus.add_alias("epsilon greedy", eg_id)

    facts = [
        ("Thompson Sampling uses Bayesian posteriors to balance exploration and exploitation.", [ts_id]),
        ("It explores proportionally to the probability of being optimal.", [ts_id]),
        ("Applications include A/B testing, recommendation systems, and clinical trials.", [ts_id]),
        ("The main tradeoff is computational overhead from maintaining distributions.", [ts_id]),
        ("Epsilon greedy explores with probability epsilon and exploits otherwise.", [eg_id]),
        ("Thompson Sampling outperforms epsilon-greedy when the action space is large.", [ts_id, eg_id]),
    ]
    for text, entities in facts:
        await bus.store(MemoryItem(
            content=text, memory_type="factual",
            entity_ids=entities, importance=0.85
        ))

    print(f"  Bus loaded: {bus.status()['item_count']} items, {bus.status()['edge_count']} edges")

    # Create orchestrator with bus (no raw shards!)
    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False

    # We need some minimal shards for the yarn graph initialization
    # but the actual retrieval will go through the bus
    dummy_shards = [MemoryShard(id="placeholder", text="placeholder", metadata={})]

    orchestrator = await create_agentic_orchestrator(
        config=config,
        shards=dummy_shards,  # minimal for initialization
        memory_bus=bus,
        enable_verification=True,
        enable_goal_tracking=True
    )

    # Patch LLM
    llm = OllamaLLM(model=MODEL)
    orchestrator.llm = llm
    if hasattr(orchestrator.learning_engine, 'orchestrator'):
        wo = orchestrator.learning_engine.orchestrator
        if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
            wo.tool_executor.llm = llm

    query = Query(text="Compare Thompson Sampling vs epsilon-greedy for optimizing meal prep rotation")

    print(f"  Running query through bus-backed orchestrator...")
    t0 = time.perf_counter()
    result = await orchestrator.reason(query=query, mode=ReasoningMode.DIRECT, max_steps=1)
    elapsed = time.perf_counter() - t0

    resp = result.spacetime.response or ""
    is_real = len(resp) > 50
    print(f"  Time: {elapsed:.1f}s | Response: {len(resp)} chars | {'REAL' if is_real else 'STUB'}")
    print(f"  Preview: {resp[:200]}...")

    # Check bus resolution stats
    status = bus.status()
    print(f"\n  Bus stats after weave:")
    print(f"    Queries: {status['query_count']}")
    print(f"    Resolution: {status['resolution_distribution']}")
    print(f"    Tokens used: {status['total_tokens_used']}")

    await bus.close()
    return is_real


async def test_semantic_level():
    """Test Level 4 semantic search with a pluggable function."""
    banner("Test 5: Level 4 Semantic (Pluggable)")

    # Create a simple embedding-free semantic function
    knowledge = [
        {"id": "k1", "content": "Thompson Sampling uses Beta distributions for binary rewards", "importance": 0.9},
        {"id": "k2", "content": "UCB1 provides deterministic upper confidence bounds", "importance": 0.85},
        {"id": "k3", "content": "Epsilon-greedy is the simplest exploration strategy", "importance": 0.7},
        {"id": "k4", "content": "Contextual bandits use features to make decisions", "importance": 0.8},
    ]

    async def mock_semantic(query_text: str, max_results: int = 5):
        """Simple keyword-based mock for testing without embeddings."""
        keywords = query_text.lower().split()
        scored = []
        for item in knowledge:
            content_lower = item["content"].lower()
            hits = sum(1 for kw in keywords if kw in content_lower)
            if hits > 0:
                scored.append({**item, "_search_score": hits / len(keywords)})
        scored.sort(key=lambda x: x["_search_score"], reverse=True)
        return scored[:max_results]

    bus = LiteMemoryBus(semantic_fn=mock_semantic)
    await bus.initialize()

    # Query that won't match any stored items (bus is empty) but will hit semantic
    r = await bus.query(MemoryQuery(intent="", semantic_text="Thompson Beta distributions"))
    assert r.resolution_path == "semantic"
    assert len(r.items) > 0
    print(f"  Semantic query: {len(r.items)} items via {r.resolution_path}")
    for item in r.items:
        print(f"    [{item.get('_search_score', 0):.2f}] {item['content'][:70]}")

    await bus.close()
    return True


async def main():
    banner("LiteMemoryBus Integration Tests")

    results = {}

    # Run non-LLM tests
    results["cascade"] = await test_cascade()
    results["forgetting"] = await test_forgetting()
    results["shard_migration"] = await test_shard_migration()
    results["semantic"] = await test_semantic_level()

    # Run LLM-dependent test
    try:
        results["orchestrator"] = await test_bus_with_orchestrator()
    except Exception as e:
        print(f"\n  ORCHESTRATOR TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["orchestrator"] = False

    banner("Results")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {passed}/{total} tests passed")
    if passed == total:
        print("  MEMORY BUS INTEGRATION COMPLETE!")
    else:
        print(f"  {total - passed} tests need attention")


if __name__ == "__main__":
    asyncio.run(main())
