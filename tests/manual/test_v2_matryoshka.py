#!/usr/bin/env python3
"""
Integration test: Memory Bus v2 Matryoshka Shell Loop

Tests the full pipeline:
  1. Navigator (PPR traversal over LiteMemoryBus)
  2. Formatter (fixed-budget context packing)
  3. Confidence (structural + narrative estimation)
  4. Orchestrator (PRIME → VERIFY → FLAG shell loop)
  5. Full loop with real LLM (Qwen3 30B)

Run: python test_v2_matryoshka.py
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
from hololoom.memory.v2 import (
    Navigator, PPRConfig, LiteBusGraph,
    Formatter, FormatConfig,
    ConfidenceEstimator, ConfidenceConfig,
    MatryoshkaOrchestrator, ShellType,
)


def banner(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


async def build_test_bus() -> LiteMemoryBus:
    """Build a LiteMemoryBus with rich test data."""
    bus = LiteMemoryBus()
    await bus.initialize()

    # Entities
    ts_id = await bus.store(MemoryItem(
        content="Thompson Sampling",
        memory_type="entity",
        properties={"type": "algorithm"},
        importance=0.9,
    ))
    eg_id = await bus.store(MemoryItem(
        content="Epsilon Greedy",
        memory_type="entity",
        properties={"type": "algorithm"},
        importance=0.8,
    ))
    ucb_id = await bus.store(MemoryItem(
        content="UCB1",
        memory_type="entity",
        properties={"type": "algorithm"},
        importance=0.85,
    ))
    cb_id = await bus.store(MemoryItem(
        content="Contextual Bandits",
        memory_type="entity",
        properties={"type": "framework"},
        importance=0.9,
    ))
    mab_id = await bus.store(MemoryItem(
        content="Multi-Armed Bandit",
        memory_type="entity",
        properties={"type": "concept"},
        importance=0.95,
    ))

    # Aliases
    for alias, eid in [
        ("thompson sampling", ts_id), ("ts", ts_id),
        ("epsilon greedy", eg_id), ("e-greedy", eg_id),
        ("ucb1", ucb_id), ("ucb", ucb_id),
        ("contextual bandits", cb_id), ("contextual", cb_id),
        ("multi-armed bandit", mab_id), ("mab", mab_id), ("bandit", mab_id),
    ]:
        await bus.add_alias(alias, eid)

    # Facts
    facts = [
        ("Thompson Sampling uses Bayesian posteriors to balance exploration and exploitation.", [ts_id], 0.90),
        ("It samples from the posterior distribution of each arm's reward.", [ts_id], 0.85),
        ("Thompson Sampling converges to the optimal arm with probability 1.", [ts_id], 0.88),
        ("Epsilon Greedy explores with probability epsilon and exploits otherwise.", [eg_id], 0.80),
        ("Epsilon Greedy is simple but has linear regret.", [eg_id], 0.82),
        ("UCB1 computes upper confidence bounds using the Hoeffding inequality.", [ucb_id], 0.87),
        ("UCB1 achieves logarithmic regret.", [ucb_id], 0.86),
        ("Thompson Sampling outperforms Epsilon Greedy when the action space is large.", [ts_id, eg_id], 0.92),
        ("Contextual bandits extend MAB by using feature vectors for each decision.", [cb_id, mab_id], 0.90),
        ("LinUCB and Thompson Sampling are popular contextual bandit algorithms.", [cb_id, ts_id], 0.88),
        ("The multi-armed bandit problem balances exploration vs exploitation.", [mab_id], 0.95),
        ("Applications include A/B testing, recommendation, and clinical trials.", [mab_id], 0.85),
    ]
    for text, entities, importance in facts:
        await bus.store(MemoryItem(
            content=text,
            memory_type="factual",
            entity_ids=entities,
            importance=importance,
        ))

    # Edges
    await bus.store_edge(ts_id, eg_id, "COMPARED_TO")
    await bus.store_edge(ts_id, ucb_id, "COMPARED_TO")
    await bus.store_edge(eg_id, ucb_id, "COMPARED_TO")
    await bus.store_edge(ts_id, mab_id, "SOLVES")
    await bus.store_edge(eg_id, mab_id, "SOLVES")
    await bus.store_edge(ucb_id, mab_id, "SOLVES")
    await bus.store_edge(cb_id, mab_id, "EXTENDS")
    await bus.store_edge(ts_id, cb_id, "USED_IN")

    # Episodes
    await bus.store(MemoryItem(
        content="User asked about bandit algorithms for meal prep rotation on 2025-12-01.",
        memory_type="episodic",
        entity_ids=[ts_id, eg_id, mab_id],
        importance=0.7,
    ))
    await bus.store(MemoryItem(
        content="User ran Thompson Sampling for recipe selection, got 85% satisfaction.",
        memory_type="episodic",
        entity_ids=[ts_id],
        importance=0.75,
    ))

    status = bus.status()
    print(f"  Bus: {status['item_count']} items, {status['edge_count']} edges, {status['alias_count']} aliases")
    return bus


async def test_navigator():
    """Test PPR navigator directly."""
    banner("Test 1: Navigator (PPR Traversal)")
    bus = await build_test_bus()
    nav = Navigator.from_lite_bus(bus)

    query = "Compare Thompson Sampling vs epsilon-greedy for meal optimization"
    result = await nav.navigate(query)

    print(f"  Seeds: {len(result.seed_nodes)}")
    for s in result.seed_nodes[:5]:
        print(f"    {s.node_id[:12]} (w={s.weight:.2f}, src={s.source})")

    print(f"  PPR: {result.iterations} iterations, converged={result.converged}")
    print(f"  Results: {len(result.ranked_nodes)} nodes")
    for nid, score in result.ranked_nodes[:8]:
        data = result.node_data.get(nid, {})
        content = data.get("content", "")[:60]
        mtype = data.get("memory_type", "?")
        print(f"    [{mtype:8}] {score:.4f}  {content}")

    assert len(result.ranked_nodes) > 0, "Navigator returned no results"
    assert result.seed_nodes, "No seeds extracted"
    await bus.close()
    return True


async def test_formatter():
    """Test fixed-budget context packing."""
    banner("Test 2: Formatter (Fixed Budget)")
    bus = await build_test_bus()
    nav = Navigator.from_lite_bus(bus)

    query = "How does Thompson Sampling work?"
    nav_result = await nav.navigate(query)

    # Pack with 512-token budget
    fmt = Formatter(config=FormatConfig(token_budget=512))
    block = fmt.pack(nav_result.ranked_nodes, nav_result.node_data)

    print(f"  {block.summary()}")
    print(f"  Overflow: {block.overflow}")
    print(f"  Preview:\n{block.text[:500]}")

    assert not block.is_empty, "Context block is empty"
    assert not block.overflow, "Context block overflows budget"
    assert block.utilization > 0.1, "Utilization suspiciously low"

    # Test with very tight budget (128 tokens)
    fmt_tight = Formatter(config=FormatConfig(token_budget=128))
    block_tight = fmt_tight.pack(nav_result.ranked_nodes, nav_result.node_data)
    print(f"\n  Tight budget: {block_tight.summary()}")
    assert block_tight.items_packed <= block.items_packed, "Tight budget packed more items"

    await bus.close()
    return True


async def test_confidence():
    """Test dual confidence estimation."""
    banner("Test 3: Confidence (Structural + Narrative)")
    bus = await build_test_bus()
    nav = Navigator.from_lite_bus(bus)
    estimator = ConfidenceEstimator(graph=nav.graph)

    # Good query (well-grounded in graph)
    result_good = await nav.navigate("Thompson Sampling Bayesian exploration")
    conf_good = estimator.estimate(result_good)

    print(f"  Good query:")
    print(f"    Structural: {conf_good.structural.score:.3f} (coverage={conf_good.structural.seed_coverage:.2f}, grounding={conf_good.structural.entity_grounding:.2f}, connectivity={conf_good.structural.graph_connectivity:.2f})")
    print(f"    Narrative:  {conf_good.narrative.score:.3f} (coherence={conf_good.narrative.edge_coherence:.2f}, temporal={conf_good.narrative.temporal_order:.2f}, contradictions={conf_good.narrative.contradiction_count})")
    print(f"    Combined:   {conf_good.combined:.3f} → {conf_good.decision}")

    # Empty query (no grounding at all)
    from hololoom.memory.v2.navigator import NavigatorResult
    empty_result = NavigatorResult(ranked_nodes=[], seed_nodes=[], iterations=0, converged=True, node_data={})
    conf_vague = estimator.estimate(empty_result)

    print(f"\n  Vague query:")
    print(f"    Structural: {conf_vague.structural.score:.3f}")
    print(f"    Narrative:  {conf_vague.narrative.score:.3f}")
    print(f"    Combined:   {conf_vague.combined:.3f} → {conf_vague.decision}")

    # Per-node confidence
    print(f"\n  Per-node confidence (top 5):")
    sorted_nodes = sorted(conf_good.per_node.items(), key=lambda x: x[1], reverse=True)
    for nid, score in sorted_nodes[:5]:
        data = result_good.node_data.get(nid, {})
        print(f"    {score:.3f}  {data.get('content', '')[:50]}")

    assert conf_good.combined > conf_vague.combined, "Good query should have higher confidence"
    await bus.close()
    return True


async def test_shell_loop_mock():
    """Test shell loop with mock LLM."""
    banner("Test 4: Shell Loop (Mock LLM)")
    bus = await build_test_bus()
    nav = Navigator.from_lite_bus(bus)

    call_count = 0

    async def mock_llm(prompt: str, max_tokens: int = 1024) -> str:
        nonlocal call_count
        call_count += 1
        return f"[Mock response #{call_count}] Based on the context provided, Thompson Sampling uses Bayesian posteriors for exploration-exploitation balance."

    audit_log = []
    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        audit_fn=lambda entry: audit_log.append(entry),
    )

    result = await orch.run("Compare Thompson Sampling vs epsilon-greedy")

    print(f"  {result.summary()}")
    print(f"  Response: {result.response[:150]}...")
    print(f"  LLM calls: {call_count}")
    print(f"  Audit entries: {len(audit_log)}")

    if audit_log:
        entry = audit_log[0]
        print(f"  Audit: shells={len(entry['shells'])}, conf={entry['final_confidence']:.2f}")

    assert result.response, "No response generated"
    assert result.shell_count >= 1, "No shells executed"
    assert len(audit_log) == 1, "Expected exactly 1 audit entry"

    await bus.close()
    return True


async def test_shell_loop_llm():
    """Test shell loop with real LLM (Qwen3 30B)."""
    banner("Test 5: Shell Loop (Qwen3 30B)")

    import httpx

    MODEL = "qwen3:30b"

    async def ollama_generate(prompt: str, max_tokens: int = 1024) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.7},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    bus = await build_test_bus()
    nav = Navigator.from_lite_bus(bus)

    audit_log = []
    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=ollama_generate,
        audit_fn=lambda entry: audit_log.append(entry),
    )

    query = "Compare Thompson Sampling vs epsilon-greedy for optimizing meal prep rotation. Which would you recommend and why?"

    print(f"  Query: {query}")
    print(f"  Running Matryoshka shell loop...")
    t0 = time.perf_counter()
    result = await orch.run(query)
    elapsed = time.perf_counter() - t0

    print(f"\n  {result.summary()}")
    print(f"\n  Shell trace:")
    for sr in result.shells_executed:
        print(f"    {sr.shell_type.value:8} | conf={sr.confidence.combined:.2f} ({sr.decision:10}) | ctx={sr.context_block.summary()} | {sr.elapsed_seconds:.1f}s")

    print(f"\n  Final response ({len(result.response)} chars):")
    print(f"  {'─'*60}")
    print(f"  {result.response[:800]}")
    if len(result.response) > 800:
        print(f"  ... [{len(result.response) - 800} more chars]")
    print(f"  {'─'*60}")

    is_real = len(result.response) > 100
    print(f"\n  Verdict: {'REAL' if is_real else 'STUB'} | {elapsed:.1f}s total")

    if audit_log:
        entry = audit_log[0]
        print(f"  Audit: {len(entry['shells'])} shells, final_conf={entry['final_confidence']:.2f}")

    await bus.close()
    return is_real


async def main():
    banner("Memory Bus v2 — Matryoshka Integration Tests")

    results = {}

    # Structural tests (no LLM)
    results["navigator"] = await test_navigator()
    results["formatter"] = await test_formatter()
    results["confidence"] = await test_confidence()
    results["shell_loop_mock"] = await test_shell_loop_mock()

    # LLM test
    try:
        results["shell_loop_llm"] = await test_shell_loop_llm()
    except Exception as e:
        print(f"\n  LLM TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["shell_loop_llm"] = False

    banner("Results")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {passed}/{total} tests passed")
    if passed == total:
        print("  MATRYOSHKA V2 INTEGRATION COMPLETE!")
    else:
        print(f"  {total - passed} tests need attention")


if __name__ == "__main__":
    asyncio.run(main())
