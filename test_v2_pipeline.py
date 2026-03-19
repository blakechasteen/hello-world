#!/usr/bin/env python3
"""
Integration test: v2 Phase 2 — Matryoshka Shell Loop in Pipeline

Tests the full Matryoshka path:
  1. Steps 1-6 still run (pattern, features, warp, retrieval)
  2. Steps 7-8 replaced by MatryoshkaOrchestrator shell loop
  3. Shell loop does its own navigate → format → generate → score
  4. Spacetime assembly uses shell loop output
  5. Audit entries emitted from shell loop path

Run: python test_v2_pipeline.py
"""
import asyncio
import sys
import time
import logging

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
for name in ['sentence_transformers', 'httpx', 'httpcore', 'urllib3']:
    logging.getLogger(name).setLevel(logging.WARNING)

from hololoom.memory.lite_bus import LiteMemoryBus
from hololoom.memory.bus import MemoryItem
from hololoom.documentation.types import MemoryShard, Query
from hololoom.config import Config


def banner(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


async def build_rich_bus() -> LiteMemoryBus:
    """Build a bus with enough data to exercise the full pipeline."""
    bus = LiteMemoryBus()
    await bus.initialize()

    ids = {}
    entities = [
        ("Thompson Sampling", "algorithm", 0.95),
        ("Epsilon Greedy", "algorithm", 0.85),
        ("UCB1", "algorithm", 0.88),
        ("Contextual Bandits", "framework", 0.92),
        ("Multi-Armed Bandit", "concept", 0.97),
        ("Bayesian Optimization", "technique", 0.90),
    ]
    for name, etype, imp in entities:
        ids[name] = await bus.store(MemoryItem(
            content=name, memory_type="entity",
            properties={"type": etype}, importance=imp,
        ))
        await bus.add_alias(name.lower(), ids[name])

    await bus.add_alias("ts", ids["Thompson Sampling"])
    await bus.add_alias("mab", ids["Multi-Armed Bandit"])
    await bus.add_alias("ucb", ids["UCB1"])
    await bus.add_alias("bandit", ids["Multi-Armed Bandit"])

    facts = [
        ("Thompson Sampling uses Bayesian posteriors to balance exploration and exploitation.", ["Thompson Sampling"]),
        ("It samples from the posterior distribution of each arm's reward.", ["Thompson Sampling"]),
        ("Thompson Sampling converges to the optimal arm with probability 1.", ["Thompson Sampling"]),
        ("Epsilon Greedy explores with probability epsilon and exploits otherwise.", ["Epsilon Greedy"]),
        ("Epsilon Greedy is simple to implement but has linear regret.", ["Epsilon Greedy"]),
        ("UCB1 computes upper confidence bounds using the Hoeffding inequality.", ["UCB1"]),
        ("UCB1 achieves logarithmic regret without knowing the time horizon.", ["UCB1"]),
        ("Thompson Sampling outperforms Epsilon Greedy when the action space is large.", ["Thompson Sampling", "Epsilon Greedy"]),
        ("Contextual bandits use feature vectors to make per-decision adaptations.", ["Contextual Bandits", "Multi-Armed Bandit"]),
        ("LinUCB and Thompson Sampling are popular contextual bandit algorithms.", ["Contextual Bandits", "Thompson Sampling"]),
        ("The multi-armed bandit problem models the exploration-exploitation tradeoff.", ["Multi-Armed Bandit"]),
        ("Applications include A/B testing, recommendation systems, and clinical trials.", ["Multi-Armed Bandit"]),
        ("Bayesian Optimization uses a surrogate model to guide expensive function evaluation.", ["Bayesian Optimization"]),
        ("Thompson Sampling can be seen as a special case of Bayesian Optimization for bandits.", ["Thompson Sampling", "Bayesian Optimization"]),
    ]
    for text, entity_names in facts:
        await bus.store(MemoryItem(
            content=text, memory_type="factual",
            entity_ids=[ids[n] for n in entity_names],
            importance=0.85,
        ))

    edges = [
        ("Thompson Sampling", "Epsilon Greedy", "COMPARED_TO"),
        ("Thompson Sampling", "UCB1", "COMPARED_TO"),
        ("Epsilon Greedy", "UCB1", "COMPARED_TO"),
        ("Thompson Sampling", "Multi-Armed Bandit", "SOLVES"),
        ("Epsilon Greedy", "Multi-Armed Bandit", "SOLVES"),
        ("UCB1", "Multi-Armed Bandit", "SOLVES"),
        ("Contextual Bandits", "Multi-Armed Bandit", "EXTENDS"),
        ("Thompson Sampling", "Contextual Bandits", "USED_IN"),
        ("Thompson Sampling", "Bayesian Optimization", "RELATED_TO"),
    ]
    for src, dst, rel in edges:
        await bus.store_edge(ids[src], ids[dst], rel)

    await bus.store(MemoryItem(
        content="User asked about bandit algorithms for meal prep rotation.",
        memory_type="episodic",
        entity_ids=[ids["Thompson Sampling"], ids["Multi-Armed Bandit"]],
        importance=0.7,
    ))

    status = bus.status()
    print(f"  Bus: {status['item_count']} items, {status['edge_count']} edges, {status['alias_count']} aliases")
    return bus


async def setup_orchestrator(bus):
    """Create and configure WeavingOrchestrator with Qwen3 30B."""
    from hololoom.weaving_orchestrator import WeavingOrchestrator
    from hololoom.awareness.llm_integration import OllamaLLM

    MODEL = "qwen3:30b"

    config = Config.fast()
    config.enable_safety_guardrails = False

    dummy_shards = [MemoryShard(id="placeholder", text="placeholder", metadata={})]

    wo = WeavingOrchestrator(
        cfg=config,
        shards=dummy_shards,
        memory_bus=bus,
        enable_semantic_cache=False,
    )
    await wo.__aenter__()

    llm = OllamaLLM(model=MODEL)
    wo.llm = llm
    wo.tool_executor.llm = llm
    print(f"  LLM: {MODEL}")
    return wo, MODEL


async def test_matryoshka_pipeline():
    """Test: Full pipeline with Matryoshka shell loop replacing steps 7-8."""
    banner("Test 1: Matryoshka Shell Loop in Pipeline")

    bus = await build_rich_bus()
    audit_before = len(bus._audit)

    wo, model = await setup_orchestrator(bus)

    query = Query(text="Compare Thompson Sampling vs epsilon-greedy for optimizing meal prep rotation. Which is better and why?")
    print(f"  Query: {query.text}")

    t0 = time.perf_counter()
    result = await wo.weave(query)
    elapsed = time.perf_counter() - t0

    resp = result.response or ""
    v2_conf = getattr(wo, '_v2_confidence', None)
    v2_block = getattr(wo, '_v2_context_block', None)

    print(f"\n  Time:     {elapsed:.1f}s")
    print(f"  Response: {len(resp)} chars")
    print(f"  Tool:     {result.tool_used}")

    # Verify Matryoshka path was taken
    assert "matryoshka" in result.tool_used, f"Expected matryoshka path, got tool={result.tool_used}"
    print(f"  Path:     MATRYOSHKA (confirmed)")

    # Check shell type in tool name
    shell_type = result.tool_used.split(":")[-1] if ":" in result.tool_used else "unknown"
    print(f"  Shell:    {shell_type}")

    # v2 confidence from shell loop
    if v2_conf:
        print(f"  v2 Confidence: {v2_conf.combined:.3f} ({v2_conf.decision})")
        print(f"    Structural: {v2_conf.structural.score:.3f}")
        print(f"    Narrative:  {v2_conf.narrative.score:.3f}")

    if v2_block:
        print(f"  Context: {v2_block.summary()}")

    # Spacetime metadata
    print(f"  Spacetime confidence: {result.confidence:.3f}")
    print(f"  Quality score: {result.quality_score:.3f}")

    # Check trace
    trace = result.trace
    assert trace.policy_adapter == "matryoshka", f"Expected matryoshka adapter, got {trace.policy_adapter}"
    print(f"  Trace adapter: {trace.policy_adapter}")
    print(f"  Trace tool: {trace.tool_selected}")
    if 'matryoshka_loop' in trace.stage_durations:
        print(f"  Loop timing: {trace.stage_durations['matryoshka_loop']:.0f}ms")

    # Audit emission
    audit_after = len(bus._audit)
    new_entries = audit_after - audit_before
    print(f"\n  Audit: {audit_before} -> {audit_after} (+{new_entries})")

    v2_entries = [e for e in bus._audit if e.get("type") == "v2_weave_cycle"]
    if v2_entries:
        entry = v2_entries[-1]
        print(f"  Audit entry:")
        print(f"    tool:       {entry['tool_used']}")
        print(f"    confidence: {entry['v2_confidence']:.3f}")
        print(f"    reward:     {entry['reward']:.3f}")

    # Response preview
    print(f"\n  Response preview:")
    print(f"  {'─'*60}")
    print(f"  {resp[:600]}")
    if len(resp) > 600:
        print(f"  ... [{len(resp) - 600} more chars]")
    print(f"  {'─'*60}")

    await wo.__aexit__(None, None, None)
    await bus.close()

    is_real = len(resp) > 100
    return is_real


async def test_matryoshka_shell_progression():
    """Test: Shell progression (PRIME → VERIFY if needed)."""
    banner("Test 2: Shell Progression")

    bus = await build_rich_bus()
    wo, model = await setup_orchestrator(bus)

    # This query should trigger the shell loop with confidence checks
    query = Query(text="How does Thompson Sampling work?")

    t0 = time.perf_counter()
    result = await wo.weave(query)
    elapsed = time.perf_counter() - t0

    resp = result.response or ""
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Response: {len(resp)} chars")
    print(f"  Tool:     {result.tool_used}")

    v2_conf = getattr(wo, '_v2_confidence', None)
    if v2_conf:
        print(f"  Final confidence: {v2_conf.combined:.3f} ({v2_conf.decision})")

    # Check that we got a real response through the matryoshka path
    assert "matryoshka" in result.tool_used
    assert len(resp) > 50, f"Response too short: {len(resp)} chars"

    print(f"\n  Response preview:")
    print(f"  {'─'*60}")
    print(f"  {resp[:400]}")
    print(f"  {'─'*60}")

    await wo.__aexit__(None, None, None)
    await bus.close()
    return True


async def main():
    banner("Phase 2: Matryoshka Pipeline Integration Tests")
    results = {}

    # Test 1: Full matryoshka pipeline
    try:
        results["matryoshka_pipeline"] = await test_matryoshka_pipeline()
    except Exception as e:
        print(f"\n  TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["matryoshka_pipeline"] = False

    # Test 2: Shell progression
    try:
        results["shell_progression"] = await test_matryoshka_shell_progression()
    except Exception as e:
        print(f"\n  TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["shell_progression"] = False

    banner("Results")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {passed}/{total} tests passed")
    if passed == total:
        print("  PHASE 2: MATRYOSHKA SHELL LOOP WIRED INTO PIPELINE!")


if __name__ == "__main__":
    asyncio.run(main())
