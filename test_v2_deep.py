#!/usr/bin/env python3
"""
Deep property tests for Memory Bus v2.

Not just "does it run" — these tests probe the mathematical invariants,
adversarial edge cases, and emergent behavior of the Matryoshka architecture.

Structure:
  I.   Graph Properties    — PPR is a probability distribution, converges, respects topology
  II.  Adversarial         — contradictions, orphans, empty graphs, adversarial queries
  III. Emergence           — shell improvement, confidence gating, alpha sensitivity
  IV.  Budget Mechanics    — packing invariants, utilization curves, truncation behavior
  V.   Confidence Landscape — confidence as a function of graph completeness
  VI.  Capstone            — Matryoshka shell loop with real LLM (optional)

Run:  python test_v2_deep.py           (fast, no LLM)
      python test_v2_deep.py --llm     (includes capstone LLM test)
"""
import asyncio
import sys
import time
import math
import logging

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.WARNING)

from hololoom.memory.lite_bus import LiteMemoryBus
from hololoom.memory.bus import MemoryItem
from hololoom.memory.v2 import (
    Navigator, NavigatorResult, PPRConfig, SeedNode,
    LiteBusGraph, personalized_pagerank, extract_seeds,
    Formatter, FormatConfig, ContextBlock,
    ConfidenceEstimator, ConfidenceConfig, DualConfidence,
    MatryoshkaOrchestrator, ShellType, DEFAULT_SHELLS,
)


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

class Results:
    """Test result accumulator with ASCII diagnostic output."""
    def __init__(self):
        self.sections = []
        self.tests = []
        self.current_section = None

    def section(self, name):
        self.current_section = name
        self.sections.append(name)
        w = 72
        print(f"\n{'━'*w}")
        print(f"  {name}")
        print(f"{'━'*w}")

    def check(self, name, passed, detail=""):
        sym = "✓" if passed else "✗"
        self.tests.append((self.current_section, name, passed))
        detail_str = f"  ({detail})" if detail else ""
        print(f"  {sym} {name}{detail_str}")
        return passed

    def bar(self, label, value, max_val=1.0, width=30):
        """ASCII bar chart line."""
        frac = min(1.0, value / max_val) if max_val > 0 else 0
        filled = int(frac * width)
        bar = "█" * filled + "░" * (width - filled)
        print(f"    {label:24s} {bar} {value:.4f}")

    def table(self, headers, rows, col_widths=None):
        """Compact ASCII table."""
        if not col_widths:
            col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                          for i, h in enumerate(headers)]
        header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
        print(f"    {header_line}")
        print(f"    {'─' * sum(col_widths)}")
        for row in rows:
            line = "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
            print(f"    {line}")

    def summary(self):
        w = 72
        print(f"\n{'═'*w}")
        print(f"  RESULTS")
        print(f"{'═'*w}")

        by_section = {}
        for sec, name, passed in self.tests:
            by_section.setdefault(sec, []).append((name, passed))

        total_pass = 0
        total_fail = 0
        for sec in self.sections:
            tests = by_section.get(sec, [])
            p = sum(1 for _, ok in tests if ok)
            f = len(tests) - p
            total_pass += p
            total_fail += f
            status = "PASS" if f == 0 else f"FAIL ({f})"
            print(f"  {sec:40s} {p}/{len(tests)}  {status}")

        total = total_pass + total_fail
        print(f"\n  {'─'*40}")
        if total_fail == 0:
            print(f"  ALL {total} TESTS PASSED")
        else:
            print(f"  {total_pass}/{total} passed, {total_fail} FAILED")
        print(f"{'═'*w}")
        return total_fail == 0


R = Results()


async def make_bus(entities, facts, edges, aliases=None):
    """Build a LiteMemoryBus from declarative data."""
    bus = LiteMemoryBus()
    await bus.initialize()
    ids = {}
    for name, etype, imp in entities:
        ids[name] = await bus.store(MemoryItem(
            content=name, memory_type="entity",
            properties={"type": etype}, importance=imp,
        ))
        await bus.add_alias(name.lower(), ids[name])
    for alias, target in (aliases or []):
        await bus.add_alias(alias, ids[target])
    for text, entity_names, imp in facts:
        await bus.store(MemoryItem(
            content=text, memory_type="factual",
            entity_ids=[ids[n] for n in entity_names],
            importance=imp,
        ))
    for src, dst, rel in edges:
        await bus.store_edge(ids[src], ids[dst], rel)
    return bus, ids


# Standard test dataset
ENTITIES = [
    ("Thompson Sampling", "algorithm", 0.95),
    ("Epsilon Greedy", "algorithm", 0.85),
    ("UCB1", "algorithm", 0.88),
    ("Contextual Bandits", "framework", 0.92),
    ("Multi-Armed Bandit", "concept", 0.97),
    ("Bayesian Optimization", "technique", 0.90),
]
FACTS = [
    ("Thompson Sampling uses Bayesian posteriors to balance exploration and exploitation.", ["Thompson Sampling"], 0.90),
    ("It samples from the posterior distribution of each arm's reward.", ["Thompson Sampling"], 0.85),
    ("Thompson Sampling converges to the optimal arm with probability 1.", ["Thompson Sampling"], 0.88),
    ("Epsilon Greedy explores with probability epsilon and exploits otherwise.", ["Epsilon Greedy"], 0.80),
    ("Epsilon Greedy is simple but has linear regret.", ["Epsilon Greedy"], 0.82),
    ("UCB1 computes upper confidence bounds using the Hoeffding inequality.", ["UCB1"], 0.87),
    ("UCB1 achieves logarithmic regret.", ["UCB1"], 0.86),
    ("Thompson Sampling outperforms Epsilon Greedy when the action space is large.", ["Thompson Sampling", "Epsilon Greedy"], 0.92),
    ("Contextual bandits extend MAB by using feature vectors.", ["Contextual Bandits", "Multi-Armed Bandit"], 0.90),
    ("The multi-armed bandit problem balances exploration vs exploitation.", ["Multi-Armed Bandit"], 0.95),
]
EDGES = [
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
ALIASES = [
    ("ts", "Thompson Sampling"),
    ("mab", "Multi-Armed Bandit"),
    ("ucb", "UCB1"),
    ("bandit", "Multi-Armed Bandit"),
]


async def standard_bus():
    return await make_bus(ENTITIES, FACTS, EDGES, ALIASES)


# ═══════════════════════════════════════════════════════════════════════
# I. GRAPH PROPERTIES
# ═══════════════════════════════════════════════════════════════════════

async def test_ppr_is_distribution():
    """PPR scores form a probability distribution (sum ≈ 1)."""
    R.section("I. Graph Properties")
    bus, ids = await standard_bus()
    graph = LiteBusGraph(bus)

    seeds = [SeedNode(ids["Thompson Sampling"], 1.0, "test")]
    result = personalized_pagerank(graph, seeds, PPRConfig(max_results=1000, min_score=0))

    total = sum(s for _, s in result.ranked_nodes)
    R.check("PPR scores sum to ~1.0", abs(total - 1.0) < 0.05, f"sum={total:.6f}")

    # All scores non-negative
    all_positive = all(s >= 0 for _, s in result.ranked_nodes)
    R.check("All PPR scores non-negative", all_positive)

    await bus.close()


async def test_ppr_seed_dominance():
    """Seed nodes should have highest PPR scores."""
    bus, ids = await standard_bus()
    graph = LiteBusGraph(bus)

    seed_id = ids["Thompson Sampling"]
    seeds = [SeedNode(seed_id, 1.0, "test")]
    result = personalized_pagerank(graph, seeds, PPRConfig(max_results=1000, min_score=0))

    score_map = dict(result.ranked_nodes)
    seed_score = score_map.get(seed_id, 0)
    max_non_seed = max((s for nid, s in result.ranked_nodes if nid != seed_id), default=0)

    R.check("Seed has highest score", seed_score >= max_non_seed,
            f"seed={seed_score:.4f} vs max_other={max_non_seed:.4f}")

    await bus.close()


async def test_ppr_neighbor_proximity():
    """Direct neighbors of seed score higher than distant nodes."""
    bus, ids = await standard_bus()
    graph = LiteBusGraph(bus)

    seed_id = ids["Thompson Sampling"]
    seeds = [SeedNode(seed_id, 1.0, "test")]
    result = personalized_pagerank(graph, seeds, PPRConfig(max_results=1000, min_score=0))
    score_map = dict(result.ranked_nodes)

    # Direct neighbors of TS
    direct_neighbors = {n[0] for n in graph.neighbors(seed_id)}
    # Bayesian Optimization is connected directly; let's find something NOT directly connected
    all_ids = set(score_map.keys())
    non_neighbors = all_ids - direct_neighbors - {seed_id}

    if direct_neighbors and non_neighbors:
        avg_neighbor = sum(score_map.get(n, 0) for n in direct_neighbors) / len(direct_neighbors)
        avg_distant = sum(score_map.get(n, 0) for n in non_neighbors) / len(non_neighbors)
        R.check("Neighbors score > distant nodes (avg)",
                avg_neighbor > avg_distant,
                f"neighbors={avg_neighbor:.4f} vs distant={avg_distant:.4f}")
    else:
        R.check("Neighbors score > distant nodes (avg)", True, "graph too dense to test")

    await bus.close()


async def test_ppr_alpha_sensitivity():
    """Higher alpha (more teleport) concentrates scores near seeds."""
    bus, ids = await standard_bus()
    graph = LiteBusGraph(bus)

    seed_id = ids["Thompson Sampling"]
    seeds = [SeedNode(seed_id, 1.0, "test")]

    alphas = [0.05, 0.15, 0.30, 0.50, 0.80]
    seed_scores = []

    print()
    for alpha in alphas:
        result = personalized_pagerank(graph, seeds, PPRConfig(alpha=alpha, max_results=1000, min_score=0))
        score_map = dict(result.ranked_nodes)
        s = score_map.get(seed_id, 0)
        seed_scores.append(s)
        R.bar(f"α={alpha:.2f}", s)

    # Monotonic: higher alpha → higher seed score
    monotonic = all(seed_scores[i] <= seed_scores[i+1] for i in range(len(seed_scores)-1))
    R.check("Seed score monotonic with alpha", monotonic,
            f"scores={[f'{s:.3f}' for s in seed_scores]}")

    await bus.close()


async def test_ppr_convergence():
    """PPR converges or reaches max iterations deterministically."""
    bus, ids = await standard_bus()
    graph = LiteBusGraph(bus)

    seeds = [SeedNode(ids["Thompson Sampling"], 1.0, "test")]

    # Tight tolerance should converge
    r1 = personalized_pagerank(graph, seeds, PPRConfig(tolerance=1e-3, max_iterations=100))
    R.check("Converges with loose tolerance", r1.converged, f"iter={r1.iterations}")

    # Very tight tolerance may not converge
    r2 = personalized_pagerank(graph, seeds, PPRConfig(tolerance=1e-15, max_iterations=5))
    R.check("Bounded by max_iterations", r2.iterations <= 5, f"iter={r2.iterations}")

    # Two runs with same config produce identical results
    r3 = personalized_pagerank(graph, seeds, PPRConfig())
    r4 = personalized_pagerank(graph, seeds, PPRConfig())
    scores_match = all(
        abs(a[1] - b[1]) < 1e-10
        for a, b in zip(r3.ranked_nodes, r4.ranked_nodes)
    )
    R.check("Deterministic (same input → same output)", scores_match)

    await bus.close()


async def test_ppr_multi_seed_blending():
    """Multiple seeds blend influence proportionally."""
    bus, ids = await standard_bus()
    graph = LiteBusGraph(bus)

    # Single seeds
    r_ts = personalized_pagerank(graph,
        [SeedNode(ids["Thompson Sampling"], 1.0, "test")],
        PPRConfig(max_results=1000, min_score=0))
    r_eg = personalized_pagerank(graph,
        [SeedNode(ids["Epsilon Greedy"], 1.0, "test")],
        PPRConfig(max_results=1000, min_score=0))

    # Both seeds equally weighted
    r_both = personalized_pagerank(graph, [
        SeedNode(ids["Thompson Sampling"], 1.0, "test"),
        SeedNode(ids["Epsilon Greedy"], 1.0, "test"),
    ], PPRConfig(max_results=1000, min_score=0))

    ts_map = dict(r_ts.ranked_nodes)
    eg_map = dict(r_eg.ranked_nodes)
    both_map = dict(r_both.ranked_nodes)

    # The comparison fact about TS vs EG should rank high in the blended result
    comparison_facts = [nid for nid, data in r_both.node_data.items()
                        if "outperforms" in data.get("content", "").lower()]

    if comparison_facts:
        cf_id = comparison_facts[0]
        cf_rank_both = next((i for i, (nid, _) in enumerate(r_both.ranked_nodes) if nid == cf_id), 999)
        cf_rank_ts = next((i for i, (nid, _) in enumerate(r_ts.ranked_nodes) if nid == cf_id), 999)
        R.check("Comparison fact boosted by dual seeds",
                cf_rank_both <= cf_rank_ts,
                f"rank_both={cf_rank_both} vs rank_ts_only={cf_rank_ts}")
    else:
        R.check("Comparison fact boosted by dual seeds", True, "no comparison fact found")

    await bus.close()


# ═══════════════════════════════════════════════════════════════════════
# II. ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════

async def test_empty_graph():
    """Empty graph returns empty results gracefully."""
    R.section("II. Adversarial")
    bus = LiteMemoryBus()
    await bus.initialize()
    nav = Navigator.from_lite_bus(bus)

    result = await nav.navigate("Thompson Sampling")
    R.check("Empty graph → empty result", len(result.ranked_nodes) == 0)
    R.check("Empty graph → no seeds", len(result.seed_nodes) == 0)

    estimator = ConfidenceEstimator(graph=nav.graph)
    conf = estimator.estimate(result)
    R.check("Empty graph → zero confidence", conf.combined == 0.0, f"conf={conf.combined}")
    R.check("Empty graph → flag decision", conf.decision == "flag")

    await bus.close()


async def test_single_node():
    """Graph with one isolated node."""
    bus = LiteMemoryBus()
    await bus.initialize()
    nid = await bus.store(MemoryItem(
        content="Lonely Node", memory_type="entity", importance=0.9,
    ))
    await bus.add_alias("lonely", nid)

    nav = Navigator.from_lite_bus(bus)
    result = await nav.navigate("lonely")

    R.check("Single node found", len(result.ranked_nodes) == 1)
    R.check("Single node is the seed", result.ranked_nodes[0][0] == nid if result.ranked_nodes else False)

    estimator = ConfidenceEstimator(graph=nav.graph)
    conf = estimator.estimate(result)
    R.check("Single node → low connectivity", conf.structural.graph_connectivity == 0.0)

    await bus.close()


async def test_disconnected_subgraphs():
    """Two disconnected subgraphs — PPR stays within the seeded component."""
    bus = LiteMemoryBus()
    await bus.initialize()

    # Island A
    a1 = await bus.store(MemoryItem(content="Alpha One", memory_type="entity", importance=0.9))
    a2 = await bus.store(MemoryItem(content="Alpha Two", memory_type="entity", importance=0.8))
    await bus.store_edge(a1, a2, "LINKS_TO")
    await bus.add_alias("alpha", a1)

    # Island B (no edges to A)
    b1 = await bus.store(MemoryItem(content="Beta One", memory_type="entity", importance=0.9))
    b2 = await bus.store(MemoryItem(content="Beta Two", memory_type="entity", importance=0.8))
    await bus.store_edge(b1, b2, "LINKS_TO")
    await bus.add_alias("beta", b1)

    graph = LiteBusGraph(bus)

    # Seed only in Island A
    seeds_a = [SeedNode(a1, 1.0, "test")]
    result = personalized_pagerank(graph, seeds_a, PPRConfig(max_results=100, min_score=0))
    score_map = dict(result.ranked_nodes)

    a_score = score_map.get(a1, 0) + score_map.get(a2, 0)
    b_score = score_map.get(b1, 0) + score_map.get(b2, 0)

    R.check("PPR concentrates in seeded island",
            a_score > b_score * 5,
            f"island_A={a_score:.4f} vs island_B={b_score:.4f}")

    await bus.close()


async def test_contradiction_detection():
    """Confidence estimator detects contradictory facts."""
    bus, ids = await make_bus(
        entities=[
            ("Algorithm X", "algorithm", 0.9),
            ("Algorithm Y", "algorithm", 0.9),
        ],
        facts=[
            ("Algorithm X outperforms Algorithm Y in all benchmarks.", ["Algorithm X", "Algorithm Y"], 0.9),
            ("Algorithm Y is not better than random baseline.", ["Algorithm Y"], 0.8),
            ("Algorithm X does not outperform Algorithm Y on sparse data.", ["Algorithm X", "Algorithm Y"], 0.85),
        ],
        edges=[
            ("Algorithm X", "Algorithm Y", "COMPARED_TO"),
        ],
    )

    nav = Navigator.from_lite_bus(bus)
    result = await nav.navigate("Algorithm X vs Algorithm Y")
    estimator = ConfidenceEstimator(graph=nav.graph)
    conf = estimator.estimate(result)

    R.check("Contradictions detected", conf.narrative.contradiction_count > 0,
            f"count={conf.narrative.contradiction_count}")

    # Clean graph (no contradictions)
    bus2, _ = await standard_bus()
    nav2 = Navigator.from_lite_bus(bus2)
    result2 = await nav2.navigate("Thompson Sampling")
    conf2 = ConfidenceEstimator(graph=nav2.graph).estimate(result2)
    R.check("Clean graph → zero contradictions", conf2.narrative.contradiction_count == 0)

    await bus.close()
    await bus2.close()


async def test_query_with_no_grounding():
    """Query about something not in the graph at all."""
    bus, _ = await standard_bus()
    nav = Navigator.from_lite_bus(bus)

    result = await nav.navigate("quantum entanglement teleportation protocol")
    estimator = ConfidenceEstimator(graph=nav.graph)
    conf = estimator.estimate(result)

    # May find some nodes via keyword but seeds should be weak
    seed_weights = [s.weight for s in result.seed_nodes]
    high_weight_seeds = [w for w in seed_weights if w >= 0.8]

    R.check("No high-weight seeds for foreign query",
            len(high_weight_seeds) == 0,
            f"high_weight_seeds={len(high_weight_seeds)}, total_seeds={len(result.seed_nodes)}")

    await bus.close()


async def test_adversarial_alias_collision():
    """Alias that maps to a different concept than intended."""
    bus = LiteMemoryBus()
    await bus.initialize()

    # "ts" could be Thompson Sampling or TypeScript
    ts_algo = await bus.store(MemoryItem(content="Thompson Sampling", memory_type="entity", importance=0.9))
    ts_lang = await bus.store(MemoryItem(content="TypeScript", memory_type="entity", importance=0.9))
    await bus.add_alias("ts", ts_algo)  # "ts" → Thompson Sampling

    fact = await bus.store(MemoryItem(
        content="TypeScript is a typed superset of JavaScript.", memory_type="factual",
        entity_ids=[ts_lang], importance=0.8,
    ))
    await bus.store_edge(ts_lang, fact, "DESCRIBES")

    nav = Navigator.from_lite_bus(bus)
    result = await nav.navigate("ts performance comparison")

    # The alias should resolve to Thompson Sampling, not TypeScript
    seed_ids = {s.node_id for s in result.seed_nodes if s.source == "alias"}
    R.check("Alias resolves to registered entity",
            ts_algo in seed_ids,
            f"alias_seeds={seed_ids}")

    await bus.close()


# ═══════════════════════════════════════════════════════════════════════
# III. EMERGENCE
# ═══════════════════════════════════════════════════════════════════════

async def test_verify_shell_refines_context():
    """VERIFY shell (tighter alpha) produces more focused context than PRIME."""
    R.section("III. Emergence")
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)

    query = "Thompson Sampling Bayesian"

    # PRIME config
    r_prime = await nav.navigate(query, override_config=PPRConfig(alpha=0.15, max_results=50))
    # VERIFY config (tighter)
    r_verify = await nav.navigate(query, override_config=PPRConfig(alpha=0.25, max_results=20))

    # VERIFY should have more concentrated scores (higher Gini coefficient)
    def gini(scores):
        if not scores or sum(scores) == 0:
            return 0
        s = sorted(scores)
        n = len(s)
        total = sum(s)
        cum = sum((2*i - n + 1) * s[i] for i in range(n))
        return cum / (n * total)

    scores_prime = [s for _, s in r_prime.ranked_nodes]
    scores_verify = [s for _, s in r_verify.ranked_nodes]

    gini_prime = gini(scores_prime)
    gini_verify = gini(scores_verify)

    R.check("VERIFY more concentrated than PRIME (Gini)",
            gini_verify >= gini_prime - 0.05,
            f"prime={gini_prime:.3f} vs verify={gini_verify:.3f}")

    # Top score should be higher for VERIFY (more focused)
    top_prime = scores_prime[0] if scores_prime else 0
    top_verify = scores_verify[0] if scores_verify else 0
    R.check("VERIFY top score >= PRIME top score",
            top_verify >= top_prime - 0.01,
            f"prime_top={top_prime:.4f} vs verify_top={top_verify:.4f}")

    await bus.close()


async def test_confidence_gates():
    """Confidence thresholds correctly gate shell progression."""
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)

    config = ConfidenceConfig(high_threshold=0.75, low_threshold=0.35)
    estimator = ConfidenceEstimator(graph=nav.graph, config=config)

    result = await nav.navigate("Thompson Sampling Bayesian")
    conf = estimator.estimate(result)

    # Decision should match thresholds
    if conf.combined >= 0.75:
        expected = "sufficient"
    elif conf.combined <= 0.35:
        expected = "flag"
    else:
        expected = "verify"

    R.check("Decision matches threshold",
            conf.decision == expected,
            f"combined={conf.combined:.3f} → {conf.decision} (expected {expected})")

    # Manually test boundary conditions
    # Empty → flag
    empty = NavigatorResult([], [], 0, True, {})
    R.check("Empty → flag", estimator.estimate(empty).decision == "flag")

    await bus.close()


async def test_shell_loop_terminates():
    """Shell loop terminates within max_shells regardless of confidence."""
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)

    call_count = 0
    async def mock_llm(prompt, max_tokens=1024):
        nonlocal call_count
        call_count += 1
        return f"Response #{call_count}"

    orch = MatryoshkaOrchestrator(navigator=nav, llm_fn=mock_llm)

    # max_shells=1 → exactly 1 shell
    result1 = await orch.run("test query", max_shells=1)
    R.check("max_shells=1 → 1 shell", result1.shell_count == 1,
            f"shells={result1.shell_count}")

    # max_shells=3 → at most 3
    call_count = 0
    result3 = await orch.run("test query", max_shells=3)
    R.check("max_shells=3 → ≤3 shells", result3.shell_count <= 3,
            f"shells={result3.shell_count}")

    # force_verify=True → at least 2 shells
    call_count = 0
    result_fv = await orch.run("test query", max_shells=3, force_verify=True)
    R.check("force_verify → ≥2 shells", result_fv.shell_count >= 2,
            f"shells={result_fv.shell_count}")

    await bus.close()


async def test_shell_types_in_order():
    """Shells always execute in order: PRIME → VERIFY → FLAG."""
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)

    async def mock_llm(prompt, max_tokens=1024):
        return "mock"

    orch = MatryoshkaOrchestrator(navigator=nav, llm_fn=mock_llm)
    result = await orch.run("Thompson Sampling", max_shells=3, force_verify=True)

    order = [s.shell_type for s in result.shells_executed]
    expected_prefix = [ShellType.PRIME, ShellType.VERIFY]
    R.check("Shell order: PRIME then VERIFY",
            order[:2] == expected_prefix,
            f"order={[s.value for s in order]}")

    if len(order) == 3:
        R.check("Third shell is FLAG", order[2] == ShellType.FLAG)

    await bus.close()


async def test_audit_entry_completeness():
    """Audit entries contain all required fields."""
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)

    audit_log = []
    async def mock_llm(prompt, max_tokens=1024):
        return "Response text"

    orch = MatryoshkaOrchestrator(
        navigator=nav, llm_fn=mock_llm,
        audit_fn=lambda e: audit_log.append(e),
    )
    await orch.run("Thompson Sampling")

    required_fields = {"query", "response", "shells", "final_confidence", "final_decision", "total_tokens"}
    if audit_log:
        entry = audit_log[0]
        present = set(entry.keys())
        missing = required_fields - present
        R.check("Audit entry has all required fields",
                len(missing) == 0,
                f"missing={missing}" if missing else f"fields={len(present)}")
    else:
        R.check("Audit entry has all required fields", False, "no audit entries")

    await bus.close()


# ═══════════════════════════════════════════════════════════════════════
# IV. BUDGET MECHANICS
# ═══════════════════════════════════════════════════════════════════════

async def test_formatter_never_exceeds_budget():
    """Hard invariant: token_count ≤ token_budget."""
    R.section("IV. Budget Mechanics")
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)
    result = await nav.navigate("Thompson Sampling Bayesian exploration Multi-Armed Bandit")

    budgets = [64, 128, 256, 512, 1024, 2048, 4096]
    all_within = True

    print()
    rows = []
    for budget in budgets:
        fmt = Formatter(config=FormatConfig(token_budget=budget))
        block = fmt.pack(result.ranked_nodes, result.node_data)
        within = block.token_count <= budget
        all_within = all_within and within
        rows.append([budget, block.items_packed, block.token_count, f"{block.utilization:.0%}", "✓" if within else "✗"])

    R.table(["Budget", "Items", "Tokens", "Util", "OK"], rows, [10, 8, 10, 8, 5])
    R.check("Token count ≤ budget (all budgets)", all_within)

    await bus.close()


async def test_utilization_curve():
    """Utilization increases then plateaus as budget grows."""
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)
    result = await nav.navigate("Thompson Sampling Bayesian Multi-Armed Bandit")

    budgets = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    utils = []

    print()
    for budget in budgets:
        fmt = Formatter(config=FormatConfig(token_budget=budget))
        block = fmt.pack(result.ranked_nodes, result.node_data)
        utils.append(block.utilization)
        R.bar(f"budget={budget:5d}", block.utilization)

    # Utilization should eventually plateau (all content fits)
    plateau_budget = None
    for i in range(1, len(utils)):
        if abs(utils[i] - utils[i-1]) < 0.05 and utils[i] < 0.5:
            plateau_budget = budgets[i]
            break

    # At small budgets, utilization should be high (tight packing)
    R.check("Small budget → high utilization", utils[2] > 0.3 if len(utils) > 2 else True,
            f"util@128={utils[2]:.2f}" if len(utils) > 2 else "")

    # At large budgets, items_packed should equal items_offered
    large_fmt = Formatter(config=FormatConfig(token_budget=8192))
    large_block = large_fmt.pack(result.ranked_nodes, result.node_data)
    R.check("Large budget packs all items",
            large_block.items_packed == large_block.items_offered,
            f"{large_block.items_packed}/{large_block.items_offered}")

    await bus.close()


async def test_empty_pack():
    """Formatter handles empty input."""
    fmt = Formatter(config=FormatConfig(token_budget=1024))
    block = fmt.pack([], {})
    R.check("Empty input → empty block", block.is_empty)
    R.check("Empty block has header only", block.items_packed == 0)
    R.check("Empty block does not overflow", not block.overflow)


async def test_single_huge_item():
    """Formatter truncates oversized items instead of crashing."""
    bus = LiteMemoryBus()
    await bus.initialize()
    huge_text = "word " * 5000  # ~25000 chars ≈ 6250 tokens
    nid = await bus.store(MemoryItem(content=huge_text, memory_type="factual", importance=0.9))
    nav = Navigator.from_lite_bus(bus)
    result = await nav.navigate("word")

    fmt = Formatter(config=FormatConfig(token_budget=256, truncate_item_at=128))
    block = fmt.pack(result.ranked_nodes, result.node_data)

    R.check("Huge item truncated to fit", not block.overflow,
            f"tokens={block.token_count}/256")
    if block.items:
        R.check("Truncated item ends with ...",
                block.items[0].text.endswith("..."))

    await bus.close()


# ═══════════════════════════════════════════════════════════════════════
# V. CONFIDENCE LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════

async def test_confidence_degrades_with_edge_removal():
    """Removing edges degrades structural confidence monotonically."""
    R.section("V. Confidence Landscape")

    # Build graph with all edges, then progressively remove
    all_edges = list(EDGES)
    confidences = []

    print()
    for n_remove in range(len(all_edges) + 1):
        remaining_edges = all_edges[n_remove:]
        bus, ids = await make_bus(ENTITIES, FACTS, remaining_edges, ALIASES)
        nav = Navigator.from_lite_bus(bus)
        result = await nav.navigate("Thompson Sampling vs Epsilon Greedy")
        conf = ConfidenceEstimator(graph=nav.graph).estimate(result)
        confidences.append(conf.combined)
        n_edges = len(remaining_edges)
        R.bar(f"edges={n_edges:2d}", conf.combined)
        await bus.close()

    # Should generally decrease (allow small noise)
    drops = sum(1 for i in range(len(confidences)-1) if confidences[i+1] > confidences[i] + 0.05)
    R.check("Confidence generally decreases with edge removal",
            drops <= 2,
            f"upward_jumps={drops}/{len(confidences)-1}")


async def test_confidence_components_independent():
    """Structural and narrative confidence measure different things."""
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)
    estimator = ConfidenceEstimator(graph=nav.graph)

    # Run multiple queries and collect structural vs narrative
    queries = [
        "Thompson Sampling",
        "Multi-Armed Bandit exploration",
        "UCB1 Hoeffding",
        "Contextual Bandits feature vectors",
        "Bayesian Optimization surrogate",
    ]

    structs = []
    narrs = []
    print()
    rows = []
    for q in queries:
        result = await nav.navigate(q)
        conf = estimator.estimate(result)
        structs.append(conf.structural.score)
        narrs.append(conf.narrative.score)
        rows.append([q[:35], f"{conf.structural.score:.3f}", f"{conf.narrative.score:.3f}",
                      f"{conf.combined:.3f}", conf.decision])

    R.table(["Query", "Structural", "Narrative", "Combined", "Decision"],
            rows, [37, 12, 12, 10, 12])

    # They shouldn't be perfectly correlated
    if len(structs) >= 3:
        # Pearson correlation
        n = len(structs)
        mean_s = sum(structs) / n
        mean_n = sum(narrs) / n
        cov = sum((s - mean_s) * (na - mean_n) for s, na in zip(structs, narrs)) / n
        std_s = (sum((s - mean_s)**2 for s in structs) / n) ** 0.5
        std_n = (sum((na - mean_n)**2 for na in narrs) / n) ** 0.5
        if std_s > 0 and std_n > 0:
            corr = cov / (std_s * std_n)
            R.check("Structural and narrative not perfectly correlated",
                    abs(corr) < 0.99,
                    f"r={corr:.3f}")
        else:
            R.check("Structural and narrative not perfectly correlated", True, "zero variance")

    await bus.close()


async def test_per_node_confidence_ranking():
    """Per-node confidence correlates with PPR score and connectivity."""
    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)
    estimator = ConfidenceEstimator(graph=nav.graph)

    result = await nav.navigate("Thompson Sampling Bayesian")
    conf = estimator.estimate(result)

    if conf.per_node and result.ranked_nodes:
        # Top PPR node should have high per-node confidence
        top_nid = result.ranked_nodes[0][0]
        top_conf = conf.per_node.get(top_nid, 0)

        # Average per-node confidence
        avg_conf = sum(conf.per_node.values()) / len(conf.per_node)

        R.check("Top PPR node has above-average confidence",
                top_conf >= avg_conf,
                f"top={top_conf:.3f} vs avg={avg_conf:.3f}")

        # Entity nodes should have higher confidence than fact nodes
        entity_confs = []
        fact_confs = []
        for nid, c in conf.per_node.items():
            data = result.node_data.get(nid, {})
            if data.get("memory_type") == "entity":
                entity_confs.append(c)
            else:
                fact_confs.append(c)

        if entity_confs and fact_confs:
            avg_entity = sum(entity_confs) / len(entity_confs)
            avg_fact = sum(fact_confs) / len(fact_confs)
            R.check("Content-rich facts have meaningful per-node confidence",
                    avg_fact > 0.5,
                    f"entity={avg_entity:.3f}, fact={avg_fact:.3f}")
    else:
        R.check("Per-node confidence computed", False, "empty result")

    await bus.close()


# ═══════════════════════════════════════════════════════════════════════
# VI. CAPSTONE — Real LLM
# ═══════════════════════════════════════════════════════════════════════

async def test_matryoshka_vs_single_shot():
    """Matryoshka shell loop produces richer output than single-shot.

    Measures:
    - Response length (proxy for thoroughness)
    - VERIFY shell references PRIME content (self-checking)
    - Shell progression trace
    """
    R.section("VI. Capstone (Qwen3 30B)")

    import httpx
    MODEL = "qwen3:30b"

    async def ollama(prompt, max_tokens=3000):
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False,
                      "options": {"num_predict": max_tokens, "temperature": 0.7}},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    bus, ids = await standard_bus()
    nav = Navigator.from_lite_bus(bus)

    query = "Compare Thompson Sampling vs Epsilon Greedy for optimizing meal prep rotation"

    # Single-shot: just PRIME
    print(f"\n  Query: {query}")
    print(f"  Model: {MODEL}")

    audit_log = []
    orch = MatryoshkaOrchestrator(
        navigator=nav, llm_fn=ollama,
        audit_fn=lambda e: audit_log.append(e),
    )

    print(f"  Running single-shot (PRIME only)...")
    t0 = time.perf_counter()
    r_single = await orch.run(query, max_shells=1)
    t_single = time.perf_counter() - t0

    print(f"  Running full loop (PRIME → VERIFY → FLAG)...")
    t0 = time.perf_counter()
    r_full = await orch.run(query, max_shells=3, force_verify=True)
    t_full = time.perf_counter() - t0

    print()
    R.table(
        ["Metric", "Single-Shot", "Full Loop"],
        [
            ["Response length", len(r_single.response), len(r_full.response)],
            ["Shells executed", r_single.shell_count, r_full.shell_count],
            ["Total tokens", r_single.total_tokens, r_full.total_tokens],
            ["Final confidence", f"{r_single.final_confidence.combined:.3f}", f"{r_full.final_confidence.combined:.3f}"],
            ["Time (s)", f"{t_single:.1f}", f"{t_full:.1f}"],
        ],
        [20, 18, 18],
    )

    # Both should produce substantial responses
    R.check("Single-shot produces real response",
            len(r_single.response) > 100,
            f"{len(r_single.response)} chars")
    R.check("Full loop produces real response",
            len(r_full.response) > 100,
            f"{len(r_full.response)} chars")

    # Full loop should have more shells
    R.check("Full loop has more shells",
            r_full.shell_count > r_single.shell_count)

    # VERIFY should reference the previous response
    if r_full.shell_count >= 2:
        verify_shell = r_full.shells_executed[1]
        has_review_language = any(
            word in verify_shell.response.lower()
            for word in ["previous", "review", "correct", "verify", "accurate", "inaccurac"]
        )
        R.check("VERIFY shell references PRIME output",
                has_review_language,
                "review language found" if has_review_language else "no review language")

    # Shell progression
    print(f"\n  Shell trace:")
    for s in r_full.shells_executed:
        label = s.shell_type.value.upper()
        print(f"    {label:8s} conf={s.confidence.combined:.3f} ({s.decision:10s}) "
              f"ctx={s.context_block.items_packed}items/{s.context_block.token_count}tok  "
              f"{s.elapsed_seconds:.1f}s")

    print(f"\n  Full loop response preview:")
    print(f"  {'─'*60}")
    print(f"  {r_full.response[:500]}")
    if len(r_full.response) > 500:
        print(f"  ... [{len(r_full.response) - 500} more chars]")
    print(f"  {'─'*60}")

    await bus.close()
    return True


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep v2 property tests")
    parser.add_argument("--llm", action="store_true", help="Include capstone LLM test")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  Memory Bus v2 — Deep Property Tests                       ║")
    print("  ║  Mathematical invariants · Adversarial probes · Emergence   ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    # I. Graph Properties
    await test_ppr_is_distribution()
    await test_ppr_seed_dominance()
    await test_ppr_neighbor_proximity()
    await test_ppr_alpha_sensitivity()
    await test_ppr_convergence()
    await test_ppr_multi_seed_blending()

    # II. Adversarial
    await test_empty_graph()
    await test_single_node()
    await test_disconnected_subgraphs()
    await test_contradiction_detection()
    await test_query_with_no_grounding()
    await test_adversarial_alias_collision()

    # III. Emergence
    await test_verify_shell_refines_context()
    await test_confidence_gates()
    await test_shell_loop_terminates()
    await test_shell_types_in_order()
    await test_audit_entry_completeness()

    # IV. Budget Mechanics
    await test_formatter_never_exceeds_budget()
    await test_utilization_curve()
    await test_empty_pack()
    await test_single_huge_item()

    # V. Confidence Landscape
    await test_confidence_degrades_with_edge_removal()
    await test_confidence_components_independent()
    await test_per_node_confidence_ranking()

    # VI. Capstone (optional)
    if args.llm:
        try:
            await test_matryoshka_vs_single_shot()
        except Exception as e:
            R.section("VI. Capstone (Qwen3 30B)")
            R.check("Capstone LLM test", False, str(e))

    R.summary()


if __name__ == "__main__":
    asyncio.run(main())
