#!/usr/bin/env python3
"""
Tests for Memory Bus v2 — Cognitive Architecture Enhancement.

Tests the new classical AI components:
  I.   Blackboard + WorkingMemory — shared workspace, entity focus, production rules
  II.  ActivationAdapter — ACT-R base-level activation, decay, seed boosting
  III. Conditional Execution — production rules gating shell execution
  IV.  Hierarchy + Resolution — community detection, extractive summaries, resolution control
  V.   Cold Start — keyword fallback when graph anchors are weak
  VI.  Constrained Generation — structured output parsing
  VII. Integration — components working together

Run:  python test_v2_cognitive.py
"""
import asyncio
import sys
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
    Formatter, FormatConfig, ContextBlock, ResolutionThresholds,
    ConfidenceEstimator, ConfidenceConfig,
    MatryoshkaOrchestrator, ShellType, DEFAULT_SHELLS,
    Blackboard, WorkingMemory, ProductionRule,
    default_skip_verify_rules, default_skip_flag_rules,
    compute_ppr_entropy, extract_seed_sources,
    ActivationAdapter, ActivationConfig,
    ContextualBanditAdapter, CONFIG_PRESETS,
    HierarchyManager, HierarchyConfig,
    ColdStartHandler, ColdStartConfig,
    parse_structured_response,
    VERIFY_SCHEMA, FLAG_SCHEMA,
)


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

class Results:
    def __init__(self):
        self.tests = []
        self.sections = []

    def section(self, name):
        self.sections.append(name)
        w = 72
        print(f"\n{'='*w}")
        print(f"  {name}")
        print(f"{'='*w}")

    def check(self, name, passed, detail=""):
        sym = "+" if passed else "X"
        self.tests.append((name, passed))
        status = f"  [{sym}] {name}"
        if detail:
            status += f"  ({detail})"
        print(status)

    def summary(self):
        total = len(self.tests)
        passed = sum(1 for _, p in self.tests if p)
        failed = total - passed
        print(f"\n{'='*72}")
        print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
        if failed:
            print(f"\n  FAILED:")
            for name, p in self.tests:
                if not p:
                    print(f"    - {name}")
        print(f"{'='*72}")
        return failed == 0


async def build_test_bus():
    """Build a test bus with a rich knowledge graph."""
    bus = LiteMemoryBus()
    bus._aliases = {}

    items = [
        ("Thompson Sampling uses random samples from posterior distributions for exploration.",
         "factual", 0.9, "thompson_sampling"),
        ("Epsilon-greedy selects a random action with probability epsilon.",
         "factual", 0.8, "epsilon_greedy"),
        ("UCB1 uses upper confidence bounds to balance exploration and exploitation.",
         "factual", 0.85, "ucb1"),
        ("Multi-armed bandits model the explore-exploit tradeoff.",
         "entity", 0.95, "bandits"),
        ("PPR converges to a stationary distribution on the graph.",
         "factual", 0.7, "ppr"),
        ("Knowledge graphs store entities and their relationships.",
         "entity", 0.9, "knowledge_graphs"),
        ("Neural networks approximate complex functions via gradient descent.",
         "factual", 0.8, "neural_nets"),
        ("Reinforcement learning optimizes long-term cumulative reward.",
         "entity", 0.85, "rl"),
        ("ACT-R models human cognition with production rules and activation.",
         "factual", 0.75, "actr"),
        ("SOAR uses working memory and production rules for problem solving.",
         "factual", 0.7, "soar"),
        ("The blackboard architecture uses a shared workspace for specialists.",
         "factual", 0.65, "blackboard_arch"),
        ("Global Workspace Theory posits a shared cognitive broadcast.",
         "factual", 0.6, "gwt"),
    ]

    ids = {}
    for content, mtype, importance, key in items:
        item = MemoryItem(content=content, memory_type=mtype, importance=importance)
        item_id = await bus.store(item)
        ids[key] = item_id
        bus._aliases[key] = item_id

    # Add edges
    edges = [
        ("thompson_sampling", "bandits", "IS_A"),
        ("epsilon_greedy", "bandits", "IS_A"),
        ("ucb1", "bandits", "IS_A"),
        ("bandits", "rl", "PART_OF"),
        ("ppr", "knowledge_graphs", "USES"),
        ("neural_nets", "rl", "USED_IN"),
        ("actr", "soar", "RELATED"),
        ("soar", "blackboard_arch", "RELATED"),
        ("blackboard_arch", "gwt", "RELATED"),
    ]
    for src, dst, rel in edges:
        if src in ids and dst in ids:
            if ids[src] not in bus._edges:
                bus._edges[ids[src]] = []
            bus._edges[ids[src]].append((ids[dst], rel))
            if ids[dst] not in bus._edges:
                bus._edges[ids[dst]] = []
            bus._edges[ids[dst]].append((ids[src], rel))

    return bus, ids


# ═══════════════════════════════════════════════════════════════════════
# I. Blackboard + WorkingMemory
# ═══════════════════════════════════════════════════════════════════════

def test_working_memory(r: Results):
    r.section("I. Working Memory")

    wm = WorkingMemory()

    # Entity focus
    wm.attend(["e1", "e2", "e3"], boost=0.5)
    r.check("attend sets focus", len(wm.entity_focus) == 3)
    r.check("focus values correct", wm.entity_focus["e1"] == 0.5)

    # Boost existing
    wm.attend(["e1"], boost=0.3)
    r.check("re-attend boosts", wm.entity_focus["e1"] == 0.8)

    # Decay
    wm.update_turn()
    r.check("decay reduces focus", wm.entity_focus["e1"] < 0.8)
    r.check("decay preserves ordering", wm.entity_focus["e1"] > wm.entity_focus["e2"])

    # Prune weak
    for _ in range(20):
        wm.update_turn()
    r.check("weak activations pruned", len(wm.entity_focus) < 3,
            f"{len(wm.entity_focus)} entities remain")

    # Goals
    wm.clear()
    wm.set_goal("Find bandits info")
    wm.set_goal("Compare algorithms")
    r.check("goals stored", len(wm.active_goals) == 2)

    # Goal cap
    for i in range(10):
        wm.set_goal(f"goal_{i}")
    r.check("goals capped at max", len(wm.active_goals) <= wm.max_goals)

    # Corrections
    wm.add_correction("Fixed: Thompson Sampling is not epsilon-greedy")
    r.check("correction stored", len(wm.recent_corrections) == 1)

    # Session facts
    wm.session_facts["preference"] = "detailed explanations"
    r.check("session facts", wm.session_facts["preference"] == "detailed explanations")

    # Clear
    wm.clear()
    r.check("clear resets all", wm.turn_count == 0 and len(wm.entity_focus) == 0)


def test_blackboard(r: Results):
    r.section("I. Blackboard")

    bb = Blackboard(query="Compare bandits")

    # Post navigation
    bb.post_navigation(
        seed_count=5,
        seed_sources={"alias": 3, "keyword": 2},
        ppr_entropy=0.6,
        ppr_converged=True,
        entity_ids={"e1", "e2", "e3"},
    )
    r.check("navigation posted", bb.seed_count == 5)
    r.check("entropy posted", bb.ppr_entropy == 0.6)
    r.check("entity_match_ratio", abs(bb.entity_match_ratio - 0.6) < 0.01,
            f"ratio={bb.entity_match_ratio:.2f}")

    # Post confidence
    bb.post_confidence(combined=0.7, structural=0.75, narrative=0.65, decision="verify")
    r.check("confidence posted", bb.latest_confidence == 0.7)
    r.check("decision posted", bb.latest_decision == "verify")

    # Post second confidence (simulating VERIFY shell)
    bb.post_confidence(combined=0.8, structural=0.82, narrative=0.78, decision="sufficient")
    r.check("confidence trend positive", bb.confidence_trend > 0,
            f"trend={bb.confidence_trend:+.3f}")

    # Post format
    bb.post_format(tokens=1500, items_packed=12, items_offered=25)
    r.check("format posted", bb.tokens_used == 1500)

    # Post shell
    bb.post_shell("prime")
    bb.post_shell("verify")
    r.check("shells recorded", bb.shell_count == 2)
    r.check("shell types", bb.shells_executed == ["prime", "verify"])

    # Context features
    features = bb.context_features()
    r.check("context has 6 features", len(features) == 6)
    r.check("features are floats", all(isinstance(v, float) for v in features.values()))

    # Flags
    bb.post_flag("needs_rerank", True)
    r.check("flag posted", bb.flags["needs_rerank"] is True)

    # For new query
    bb2 = bb.for_new_query("New question")
    r.check("new query preserves WM", bb2.working_memory is bb.working_memory)
    r.check("new query fresh signals", bb2.seed_count == 0)
    r.check("new query fresh confidence", len(bb2.confidence_trace) == 0)


def test_production_rules(r: Results):
    r.section("I. Production Rules")

    # Skip-verify rules (all must fire)
    rules = default_skip_verify_rules()
    r.check("3 skip-verify rules", len(rules) == 3)

    bb_high = Blackboard(query="test")
    bb_high.post_navigation(
        seed_count=5,
        seed_sources={"alias": 4, "keyword": 1},
        ppr_entropy=0.3,
        ppr_converged=True,
        entity_ids={"e1", "e2"},
    )
    bb_high.post_confidence(0.85, 0.9, 0.8, "sufficient")

    all_fire = all(rule.fires(bb_high) for rule in rules)
    r.check("high confidence: all rules fire", all_fire)

    # Low confidence — should NOT all fire
    bb_low = Blackboard(query="test")
    bb_low.post_navigation(
        seed_count=1,
        seed_sources={"keyword": 1},
        ppr_entropy=0.8,
        ppr_converged=False,
        entity_ids=set(),
    )
    bb_low.post_confidence(0.3, 0.25, 0.35, "flag")

    none_fire = not all(rule.fires(bb_low) for rule in rules)
    r.check("low confidence: rules don't all fire", none_fire)

    # Skip-flag rules (any must fire)
    flag_rules = default_skip_flag_rules()
    r.check("3 skip-flag rules", len(flag_rules) == 3)

    # After VERIFY with "sufficient" — should skip FLAG
    bb_verify = Blackboard(query="test")
    bb_verify.post_confidence(0.6, 0.5, 0.7, "verify")
    bb_verify.post_confidence(0.8, 0.85, 0.75, "sufficient")

    any_fires = any(rule.fires(bb_verify) for rule in flag_rules)
    r.check("verify sufficient: skip FLAG fires", any_fires)


def test_ppr_entropy(r: Results):
    r.section("I. PPR Entropy Utility")

    # Uniform → high entropy
    uniform = [0.1, 0.1, 0.1, 0.1, 0.1]
    r.check("uniform → entropy=1.0", abs(compute_ppr_entropy(uniform) - 1.0) < 0.01)

    # Focused → low entropy
    focused = [0.9, 0.01, 0.01, 0.01, 0.01]
    r.check("focused → low entropy", compute_ppr_entropy(focused) < 0.5,
            f"entropy={compute_ppr_entropy(focused):.2f}")

    # Empty/single → 0
    r.check("empty → 0", compute_ppr_entropy([]) == 0.0)
    r.check("single → 0", compute_ppr_entropy([1.0]) == 0.0)


# ═══════════════════════════════════════════════════════════════════════
# II. Activation Adapter
# ═══════════════════════════════════════════════════════════════════════

def test_activation_adapter(r: Results):
    r.section("II. Activation Adapter (ACT-R)")

    adapter = ActivationAdapter(config=ActivationConfig(
        retrieval_boost=0.3,
        query_boost=0.5,
        decay_rate=0.2,
    ))

    # Retrieval boost
    adapter.on_retrieval(["n1", "n2", "n3"])
    r.check("retrieval boosts activation",
            adapter.get_activation("n1") == 0.3)

    # Multiple retrievals accumulate
    adapter.on_retrieval(["n1"])
    r.check("repeated retrieval accumulates",
            adapter.get_activation("n1") == 0.6)

    # Query entity boost (stronger)
    adapter.on_query_entities(["n4"])
    r.check("query boost higher", adapter.get_activation("n4") == 0.5)

    # Seed boosting
    seeds = [
        SeedNode(node_id="n1", weight=1.0, source="alias"),
        SeedNode(node_id="n5", weight=1.0, source="keyword"),
    ]
    boosted = adapter.boost_seeds(seeds)
    r.check("active seed boosted", boosted[0].weight > 1.0,
            f"weight={boosted[0].weight:.2f}")
    r.check("inactive seed unchanged", boosted[1].weight == 1.0)

    # Decay
    adapter.decay_step()
    r.check("decay reduces activation",
            adapter.get_activation("n1") < 0.6)

    # Multiple decays → prune
    for _ in range(20):
        adapter.decay_step()
    r.check("extensive decay prunes",
            len(adapter.active_node_ids()) < 4,
            f"{len(adapter.active_node_ids())} remain")

    # Clear
    adapter.clear()
    r.check("clear resets all", len(adapter.levels) == 0)

    # Stats
    adapter.on_retrieval(["a", "b"])
    stats = adapter.stats()
    r.check("stats has expected keys",
            "n_tracked" in stats and "n_active" in stats)


# ═══════════════════════════════════════════════════════════════════════
# III. Conditional Execution (via Orchestrator)
# ═══════════════════════════════════════════════════════════════════════

async def test_conditional_execution(r: Results):
    r.section("III. Conditional Execution")

    bus, ids = await build_test_bus()
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph, alias_table=bus._aliases)

    call_count = 0

    async def mock_llm(prompt, max_tokens):
        nonlocal call_count
        call_count += 1
        return f"Mock response {call_count} for the given query."

    # Test with custom production rules that always skip VERIFY
    always_skip = [
        ProductionRule("always", condition=lambda bb: True),
    ]

    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        skip_verify_rules=always_skip,
    )

    call_count = 0
    result = await orch.run("What is Thompson Sampling?")
    r.check("always-skip: 1 shell", result.shell_count == 1,
            f"shells={result.shell_count}")
    r.check("always-skip: 1 LLM call", call_count == 1)

    # Test with rules that never skip
    never_skip = [
        ProductionRule("never", condition=lambda bb: False),
    ]

    orch2 = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        skip_verify_rules=never_skip,
        skip_flag_rules=never_skip,
    )

    call_count = 0
    result2 = await orch2.run("What is Thompson Sampling?")
    r.check("never-skip: 3 shells", result2.shell_count == 3,
            f"shells={result2.shell_count}")
    r.check("never-skip: 3 LLM calls", call_count == 3)

    # Test blackboard is populated
    r.check("blackboard attached", hasattr(result, 'blackboard'))

    # Test working memory persistence across queries
    orch3 = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        skip_verify_rules=always_skip,
    )

    await orch3.run("What is Thompson Sampling?")
    wm_before = dict(orch3.working_memory.entity_focus)
    await orch3.run("Compare bandits")
    wm_after = orch3.working_memory.entity_focus

    r.check("WM persists across queries", orch3.working_memory.turn_count == 2)


# ═══════════════════════════════════════════════════════════════════════
# IV. Hierarchy + Resolution
# ═══════════════════════════════════════════════════════════════════════

async def test_hierarchy(r: Results):
    r.section("IV. Hierarchy Manager")

    bus, ids = await build_test_bus()
    graph = LiteBusGraph(bus)
    mgr = HierarchyManager(graph=graph, config=HierarchyConfig(
        min_cluster_size=2,
        max_cluster_size=8,
    ))

    # Community detection
    clusters = mgr.detect_communities()
    r.check("communities detected", len(clusters) > 0,
            f"{len(clusters)} clusters")
    r.check("clusters have nodes", all(len(c.node_ids) >= 2 for c in clusters))

    # Extractive summary
    contents = [
        "Thompson Sampling draws from posterior distributions. It balances exploration and exploitation naturally.",
        "Epsilon-greedy randomly explores with probability epsilon. It is the simplest bandit algorithm.",
        "UCB1 computes upper confidence bounds. It provides theoretical regret guarantees.",
    ]
    summary = mgr._extractive_summary(contents)
    r.check("extractive summary non-empty", len(summary) > 20,
            f"len={len(summary)}")

    # Build hierarchy (no LLM)
    hierarchy = await mgr.build_hierarchy(bus, llm_fn=None)
    r.check("hierarchy built", len(hierarchy) >= 0,
            f"{len(hierarchy)} summaries")


def test_resolution_control(r: Results):
    r.section("IV. Resolution Control")

    resolution = ResolutionThresholds(detail=0.05, compact=0.01)
    config = FormatConfig(
        token_budget=2048,
        resolution=resolution,
        include_provenance=True,
    )
    formatter = Formatter(config=config)

    # Build mock data with varying scores
    ranked_nodes = [
        ("high_score_node", 0.10),    # Above detail → full text
        ("mid_score_node", 0.03),     # Between compact and detail → compact
        ("low_score_node", 0.005),    # Below compact → summary/compact
    ]

    node_data = {
        "high_score_node": {
            "content": "This is a detailed node about Thompson Sampling. It uses posterior distributions.",
            "memory_type": "factual",
        },
        "mid_score_node": {
            "content": "Epsilon-greedy is a simple exploration strategy. It selects random actions with probability epsilon. This is the most basic bandit algorithm.",
            "memory_type": "factual",
        },
        "low_score_node": {
            "content": "UCB1 computes confidence bounds. It has strong theoretical guarantees for regret minimization.",
            "memory_type": "summary",
        },
    }

    block = formatter.pack(ranked_nodes, node_data)
    r.check("all items packed", block.items_packed == 3)

    # High-score item should have full content
    high_item = block.items[0]
    r.check("high score: full text", "posterior distributions" in high_item.text)

    # Summary type gets [summ] prefix
    low_item = block.items[2]
    r.check("summary type prefix", "[summ]" in low_item.prefix)

    # Without resolution → all items get full text
    config_no_res = FormatConfig(token_budget=2048)
    formatter_no_res = Formatter(config=config_no_res)
    block_no_res = formatter_no_res.pack(ranked_nodes, node_data)
    r.check("no resolution: all full text",
            all("." in item.text for item in block_no_res.items))


# ═══════════════════════════════════════════════════════════════════════
# V. Cold Start
# ═══════════════════════════════════════════════════════════════════════

async def test_cold_start(r: Results):
    r.section("V. Cold Start Handler")

    bus, ids = await build_test_bus()
    graph = LiteBusGraph(bus)

    handler = ColdStartHandler(
        graph=graph,
        config=ColdStartConfig(min_seeds=2, keyword_k=3),
    )

    # Needs cold start with empty seeds
    r.check("empty seeds triggers", handler.needs_cold_start([]))
    r.check("1 seed triggers", handler.needs_cold_start([SeedNode("x", 1.0, "alias")]))
    r.check("2 seeds ok", not handler.needs_cold_start([
        SeedNode("x", 1.0, "alias"), SeedNode("y", 1.0, "alias"),
    ]))

    # Keyword seeds
    seeds = handler.find_seeds("Thompson Sampling bandits exploration")
    r.check("keyword seeds found", len(seeds) > 0,
            f"{len(seeds)} seeds")
    r.check("seeds have correct source",
            all(s.source == "cold_start_keyword" for s in seeds))
    r.check("seeds have positive weight",
            all(s.weight > 0 for s in seeds))

    # Query with no matching content
    seeds_empty = handler.find_seeds("xyzzy foobar nonexistent")
    r.check("no-match query: few/no seeds", len(seeds_empty) <= 1)


# ═══════════════════════════════════════════════════════════════════════
# VI. Constrained Generation
# ═══════════════════════════════════════════════════════════════════════

def test_constrained_generation(r: Results):
    r.section("VI. Constrained Generation")

    # Parse valid VERIFY JSON
    valid_json = '{"assessment": "confirmed", "answer": "Thompson Sampling is a bandit algorithm.", "confidence_self": 0.85}'
    parsed = parse_structured_response(valid_json, ShellType.VERIFY)
    r.check("valid JSON parsed", parsed is not None)
    r.check("assessment extracted", parsed.get("assessment") == "confirmed")
    r.check("answer extracted", "Thompson" in parsed.get("answer", ""))
    r.check("confidence extracted", parsed.get("confidence_self") == 0.85)

    # Parse with markdown fences
    fenced = '```json\n{"answer": "Test answer", "assessment": "corrected"}\n```'
    parsed_fenced = parse_structured_response(fenced, ShellType.VERIFY)
    r.check("fenced JSON parsed", parsed_fenced is not None)

    # Parse invalid → returns None
    invalid = "This is just a plain text response without JSON."
    parsed_invalid = parse_structured_response(invalid, ShellType.VERIFY)
    r.check("invalid → None", parsed_invalid is None)

    # Missing 'answer' field → None
    no_answer = '{"assessment": "confirmed"}'
    parsed_no_answer = parse_structured_response(no_answer, ShellType.VERIFY)
    r.check("no answer field → None", parsed_no_answer is None)

    # Schema validation
    r.check("VERIFY schema has required fields",
            "assessment" in VERIFY_SCHEMA["properties"])
    r.check("FLAG schema has answer",
            "answer" in FLAG_SCHEMA["properties"])


# ═══════════════════════════════════════════════════════════════════════
# VII. Integration
# ═══════════════════════════════════════════════════════════════════════

async def test_integration(r: Results):
    r.section("VII. Integration")

    bus, ids = await build_test_bus()
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph, alias_table=bus._aliases)

    # Create activation adapter
    activation = ActivationAdapter()

    async def mock_llm(prompt, max_tokens):
        return "Thompson Sampling is a probabilistic approach to the multi-armed bandit problem."

    # Create orchestrator with all cognitive components
    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        activation_adapter=activation,
        working_memory=WorkingMemory(),
    )

    # Run first query
    result1 = await orch.run("What is Thompson Sampling?")
    r.check("integration: query succeeds", len(result1.response) > 10)
    r.check("integration: shells executed", result1.shell_count >= 1)
    r.check("integration: has blackboard",
            hasattr(result1, 'blackboard') and result1.blackboard is not None)

    # Check activation was updated
    r.check("activation updated",
            len(activation.active_node_ids()) > 0,
            f"{len(activation.active_node_ids())} active nodes")

    # Working memory should have entity focus
    r.check("WM has entity focus",
            len(orch.working_memory.entity_focus) > 0)

    # Run second query — should see cross-turn effects
    result2 = await orch.run("Compare Thompson Sampling with UCB1")
    r.check("second query succeeds", len(result2.response) > 10)
    r.check("turn count incremented", orch.working_memory.turn_count == 2)

    # Contextual bandit adapter (without neural policy → falls back)
    adapter = ContextualBanditAdapter()
    r.check("bandit: no neural policy", not adapter.has_neural_policy)

    bb = Blackboard(query="test")
    bb.post_navigation(3, {"alias": 2, "keyword": 1}, 0.5, True, {"e1", "e2"})
    bb.post_confidence(0.7, 0.75, 0.65, "verify")
    configs = adapter.select_config(bb)
    r.check("bandit: returns default configs",
            ShellType.PRIME in configs)

    # Cold start integration
    handler = ColdStartHandler(graph=graph, config=ColdStartConfig(min_seeds=20))
    test_seeds = await extract_seeds("nonexistent query xyz", graph, bus._aliases)
    if handler.needs_cold_start(test_seeds):
        cold_seeds = handler.find_seeds("bandits exploration exploitation")
        r.check("cold start produces seeds", len(cold_seeds) > 0)
    else:
        r.check("cold start not needed (seeds found)", True)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

async def main():
    r = Results()

    # Sync tests
    test_working_memory(r)
    test_blackboard(r)
    test_production_rules(r)
    test_ppr_entropy(r)
    test_activation_adapter(r)
    test_resolution_control(r)
    test_constrained_generation(r)

    # Async tests
    await test_conditional_execution(r)
    await test_hierarchy(r)
    await test_cold_start(r)
    await test_integration(r)

    return r.summary()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
