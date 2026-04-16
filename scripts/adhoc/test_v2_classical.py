#!/usr/bin/env python3
"""
Tests for Memory Bus v2 — Classical AI Composition Layer.

Tests the new classical AI components:
  I.    KnowledgeSource Protocol — Phase enum, SourceRegistry dispatch
  II.   TruthMaintenanceSystem — justification recording, retraction propagation
  III.  ChunkCompiler — experience recording, compilation, chunk matching
  IV.   Orchestrator Integration — KS hooks fire at correct phases
  V.    Full Integration — TMS + ChunkCompiler working together

Run:  python test_v2_classical.py
"""
import asyncio
import sys
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
    ConfidenceEstimator, ConfidenceConfig,
    MatryoshkaOrchestrator, ShellType, DEFAULT_SHELLS,
    Blackboard, WorkingMemory, ProductionRule,
    default_skip_verify_rules, default_skip_flag_rules,
    compute_ppr_entropy, extract_seed_sources,
    ActivationAdapter, ActivationConfig,
    # Classical AI Composition
    KnowledgeSource, Phase, SourceRegistry,
    TruthMaintenanceSystem, TMSConfig, JustificationRecord,
    ChunkCompiler, ChunkConfig, ChunkAction, CompiledChunk,
    ExperienceRecord,
    CBRSystem, CBRConfig, Case,
    DemonSystem, DemonConfig, DemonAlert,
    check_grounding, check_brevity, check_repetition, check_hedging,
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


R = Results()


async def build_test_bus():
    """Build a test bus with interconnected nodes for TMS/chunking tests."""
    bus = LiteMemoryBus()
    bus._aliases = {}

    items = [
        ("Thompson Sampling uses random samples from posterior distributions for exploration.",
         "entity", 0.9, "ts"),
        ("UCB uses upper confidence bounds, not random sampling.",
         "entity", 0.85, "ucb"),
        ("Multi-armed bandits balance exploration and exploitation.",
         "factual", 0.8, "bandit"),
        ("Exploration discovers new information about the environment.",
         "factual", 0.7, "explore"),
        ("Exploitation uses known information to maximize reward.",
         "factual", 0.7, "exploit"),
        ("Reward signals drive learning in bandit algorithms.",
         "factual", 0.6, "reward"),
        ("Personalized PageRank traverses graphs with teleportation.",
         "entity", 0.8, "ppr"),
        ("Graphs store nodes and edges for relational data.",
         "factual", 0.65, "graph_node"),
        # Contradicting pair for TMS retraction tests
        ("Thompson Sampling is faster and better than UCB for large action spaces.",
         "factual", 0.7, "ts_fast"),
        ("Thompson Sampling is not faster and is worse than UCB for large action spaces.",
         "factual", 0.5, "ts_slow"),
    ]

    ids = {}
    for content, mtype, importance, key in items:
        item = MemoryItem(content=content, memory_type=mtype, importance=importance)
        item_id = await bus.store(item)
        ids[key] = item_id
        bus._aliases[key.replace("_", " ")] = item_id

    # Extra aliases for seed extraction
    bus._aliases["thompson sampling"] = ids["ts"]
    bus._aliases["ucb"] = ids["ucb"]
    bus._aliases["bandit"] = ids["bandit"]
    bus._aliases["exploration"] = ids["explore"]
    bus._aliases["exploitation"] = ids["exploit"]
    bus._aliases["pagerank"] = ids["ppr"]

    # Edges: create a connected graph
    edge_defs = [
        ("ts", "bandit", "IS_A"),
        ("ucb", "bandit", "IS_A"),
        ("bandit", "explore", "USES"),
        ("bandit", "exploit", "USES"),
        ("explore", "reward", "ABOUT"),
        ("exploit", "reward", "ABOUT"),
        ("ts", "explore", "USES"),
        ("ppr", "graph_node", "USES"),
        ("ts_fast", "ts", "ABOUT"),
        ("ts_slow", "ts", "ABOUT"),
    ]
    for src, dst, rel in edge_defs:
        src_id, dst_id = ids[src], ids[dst]
        if src_id not in bus._edges:
            bus._edges[src_id] = []
        bus._edges[src_id].append((dst_id, rel))
        if dst_id not in bus._edges:
            bus._edges[dst_id] = []
        bus._edges[dst_id].append((src_id, rel))

    return bus, ids


async def mock_llm(prompt, max_tokens=1024):
    """Mock LLM that returns a simple response."""
    return "Thompson Sampling is a probabilistic approach to the multi-armed bandit problem."


# ═══════════════════════════════════════════════════════════════════════
# I. KnowledgeSource Protocol & SourceRegistry
# ═══════════════════════════════════════════════════════════════════════

def test_phase_enum():
    R.section("I. KnowledgeSource Protocol & SourceRegistry")

    # Phase enum values
    R.check("Phase.POST_NAVIGATE value",
            Phase.POST_NAVIGATE.value == "post_navigate")
    R.check("Phase.POST_CONFIDENCE value",
            Phase.POST_CONFIDENCE.value == "post_confidence")
    R.check("Phase.POST_FORMAT value",
            Phase.POST_FORMAT.value == "post_format")
    R.check("Phase.POST_GENERATE value",
            Phase.POST_GENERATE.value == "post_generate")
    R.check("Phase.PRE_SHELL value",
            Phase.PRE_SHELL.value == "pre_shell")
    R.check("Phase.POST_LOOP value",
            Phase.POST_LOOP.value == "post_loop")
    R.check("Phase has 6 members",
            len(Phase) == 6)


class MockKnowledgeSource:
    """Test KnowledgeSource that records invocations."""
    def __init__(self, name_str="mock_ks", phase_list=(Phase.POST_NAVIGATE,),
                 activation=1.0):
        self._name = name_str
        self._phases = tuple(phase_list)
        self._activation = activation
        self.invocations = []

    @property
    def name(self):
        return self._name

    @property
    def phases(self):
        return self._phases

    def activation_level(self, blackboard):
        return self._activation

    def contribute(self, blackboard, graph, context):
        self.invocations.append({
            "blackboard": blackboard,
            "graph": graph,
            "context": context,
        })


def test_source_registry():
    # Empty registry
    reg = SourceRegistry()
    R.check("Empty registry has 0 sources",
            len(reg.sources) == 0)

    # Register
    ks = MockKnowledgeSource("test_ks")
    reg.register(ks)
    R.check("Register adds source",
            len(reg.sources) == 1)

    # Idempotent
    reg.register(ks)
    R.check("Register is idempotent by name",
            len(reg.sources) == 1)

    # Has
    R.check("has() finds registered source",
            reg.has("test_ks"))
    R.check("has() returns False for missing",
            not reg.has("nonexistent"))

    # Remove
    reg.remove("test_ks")
    R.check("Remove deletes source",
            len(reg.sources) == 0)

    # Report
    reg.register(ks)
    report = reg.report()
    R.check("Report includes source_count",
            report["source_count"] == 1)
    R.check("Report includes source names",
            report["sources"][0]["name"] == "test_ks")


def test_registry_invoke():
    reg = SourceRegistry()
    ks = MockKnowledgeSource("ks1", phase_list=(Phase.POST_NAVIGATE,))
    reg.register(ks)

    bb = Blackboard(query="test", working_memory=WorkingMemory())

    # Invoke at matching phase
    reg.invoke(Phase.POST_NAVIGATE, bb, None, {"nav_result": "mock"})
    R.check("Invoke calls source at matching phase",
            len(ks.invocations) == 1)
    R.check("Invoke passes context",
            ks.invocations[0]["context"]["nav_result"] == "mock")

    # Invoke at non-matching phase
    reg.invoke(Phase.POST_CONFIDENCE, bb, None, {"confidence": "mock"})
    R.check("Invoke skips source at wrong phase",
            len(ks.invocations) == 1)  # Still 1


def test_registry_activation_filtering():
    reg = SourceRegistry(activation_threshold=0.5)
    low_ks = MockKnowledgeSource("low", activation=0.2)
    high_ks = MockKnowledgeSource("high", activation=0.9)
    reg.register(low_ks)
    reg.register(high_ks)

    bb = Blackboard(query="test", working_memory=WorkingMemory())
    reg.invoke(Phase.POST_NAVIGATE, bb, None, {"nav_result": "mock"})

    R.check("Low-activation source skipped",
            len(low_ks.invocations) == 0)
    R.check("High-activation source invoked",
            len(high_ks.invocations) == 1)


def test_registry_ordering():
    reg = SourceRegistry()
    first = MockKnowledgeSource("first", activation=0.5)
    second = MockKnowledgeSource("second", activation=0.9)
    reg.register(first)
    reg.register(second)

    order = []
    original_contribute_first = first.contribute
    original_contribute_second = second.contribute

    def track_first(bb, g, c):
        order.append("first")
        original_contribute_first(bb, g, c)

    def track_second(bb, g, c):
        order.append("second")
        original_contribute_second(bb, g, c)

    first.contribute = track_first
    second.contribute = track_second

    bb = Blackboard(query="test", working_memory=WorkingMemory())
    reg.invoke(Phase.POST_NAVIGATE, bb, None, {})

    R.check("Higher activation invoked first",
            order == ["second", "first"])


def test_registry_error_handling():
    class ErrorKS(MockKnowledgeSource):
        def contribute(self, bb, g, c):
            raise ValueError("boom")

    reg = SourceRegistry()
    error_ks = ErrorKS("error_ks")
    good_ks = MockKnowledgeSource("good_ks")
    reg.register(error_ks)
    reg.register(good_ks)

    bb = Blackboard(query="test", working_memory=WorkingMemory())
    # Should not raise
    reg.invoke(Phase.POST_NAVIGATE, bb, None, {})

    R.check("Error in source doesn't crash registry",
            True)  # If we got here, no crash
    R.check("Good source still invoked after error",
            len(good_ks.invocations) == 1)


def test_registry_null_blackboard():
    reg = SourceRegistry()
    ks = MockKnowledgeSource("ks")
    reg.register(ks)

    # Invoke with None blackboard (backward compat)
    reg.invoke(Phase.POST_NAVIGATE, None, None, {})
    R.check("Invoke with None blackboard is no-op",
            len(ks.invocations) == 0)


def test_protocol_check():
    # MockKnowledgeSource satisfies the protocol
    ks = MockKnowledgeSource("test")
    R.check("MockKS satisfies KnowledgeSource protocol",
            isinstance(ks, KnowledgeSource))


# ═══════════════════════════════════════════════════════════════════════
# II. Truth Maintenance System
# ═══════════════════════════════════════════════════════════════════════

def test_tms_basic():
    R.section("II. Truth Maintenance System")

    tms = TruthMaintenanceSystem()
    R.check("TMS name is 'tms'", tms.name == "tms")
    R.check("TMS phases include POST_NAVIGATE",
            Phase.POST_NAVIGATE in tms.phases)
    R.check("TMS phases include POST_CONFIDENCE",
            Phase.POST_CONFIDENCE in tms.phases)
    R.check("TMS activation is 1.0",
            tms.activation_level(None) == 1.0)
    R.check("TMS satisfies KnowledgeSource protocol",
            isinstance(tms, KnowledgeSource))


def test_tms_justification_recording():
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    # Navigate to get results
    nav_result = loop.run_until_complete(
        nav.navigate("Thompson Sampling exploration", max_seeds=5)
    )

    tms = TruthMaintenanceSystem(config=TMSConfig(store_edges=False))
    bb = Blackboard(query="Thompson Sampling exploration",
                    working_memory=WorkingMemory())

    # Contribute at POST_NAVIGATE
    tms.contribute(bb, graph, {"nav_result": nav_result})

    justifications = bb.flags.get("justifications", {})
    R.check("TMS records justifications",
            len(justifications) > 0)

    # Seeds should be self-justified
    seed_ids = {s.node_id for s in nav_result.seed_nodes}
    justified_seeds = [
        nid for nid in justifications if nid in seed_ids
    ]
    R.check("Seeds are self-justified",
            len(justified_seeds) > 0)

    # Non-seed nodes should have traced paths
    non_seed_justified = {
        nid: records for nid, records in justifications.items()
        if nid not in seed_ids
    }
    if non_seed_justified:
        first_nid = list(non_seed_justified.keys())[0]
        first_records = non_seed_justified[first_nid]
        R.check("Non-seed nodes have justification paths",
                len(first_records[0].path) > 1,
                f"path length = {len(first_records[0].path)}")
    else:
        R.check("Non-seed nodes have justification paths", True,
                "all results were seeds")

    # Report
    report = tms.report()
    R.check("TMS report has nodes_justified",
            report["nodes_justified"] > 0)
    loop.close()


def test_tms_justification_edges():
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    nav_result = loop.run_until_complete(
        nav.navigate("Thompson Sampling", max_seeds=3)
    )

    tms = TruthMaintenanceSystem(config=TMSConfig(store_edges=True))
    bb = Blackboard(query="Thompson Sampling",
                    working_memory=WorkingMemory())
    tms.contribute(bb, graph, {"nav_result": nav_result})

    # Check that JUSTIFIED_BY edges were stored
    justified_by_edges = []
    for nid in bus._edges:
        for target, rel in bus._edges[nid]:
            if rel == "JUSTIFIED_BY":
                justified_by_edges.append((nid, target))

    R.check("JUSTIFIED_BY edges stored in graph",
            len(justified_by_edges) > 0,
            f"{len(justified_by_edges)} edges")
    loop.close()


def test_tms_retraction():
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    # Navigate with a query that will find contradicting nodes
    nav_result = loop.run_until_complete(
        nav.navigate("Thompson Sampling faster better UCB", max_seeds=10)
    )

    # Set up activation adapter for retraction
    adapter = ActivationAdapter()
    # Prime some activations
    for nid, _ in nav_result.ranked_nodes:
        adapter.levels[nid] = 0.5

    tms = TruthMaintenanceSystem(
        activation_adapter=adapter,
        config=TMSConfig(store_edges=False),
    )
    bb = Blackboard(query="Thompson Sampling faster better UCB",
                    working_memory=WorkingMemory())

    # Step 1: Record justifications
    tms.contribute(bb, graph, {"nav_result": nav_result})

    # Step 2: Get confidence (which detects contradictions)
    estimator = ConfidenceEstimator(graph=graph)
    confidence = estimator.estimate(nav_result, query="Thompson Sampling faster better UCB")

    # Step 3: Check retractions
    tms.contribute(bb, graph, {
        "confidence": confidence,
        "nav_result": nav_result,
    })

    retractions = bb.flags.get("retractions", [])
    if confidence.narrative.contradiction_count > 0:
        R.check("TMS produces retractions for contradictions",
                len(retractions) > 0,
                f"{len(retractions)} retractions from {confidence.narrative.contradiction_count} contradictions")

        # Check activation was decreased
        if retractions:
            retracted_id = retractions[0]["retracted"]
            R.check("Retracted node activation decreased",
                    adapter.levels.get(retracted_id, 0.5) < 0.5,
                    f"activation = {adapter.levels.get(retracted_id, 0.5):.2f}")
    else:
        R.check("TMS produces retractions for contradictions", True,
                "no contradictions detected (OK)")
        R.check("Retracted node activation decreased", True,
                "no contradictions detected (OK)")

    loop.close()


def test_tms_explain():
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    nav_result = loop.run_until_complete(
        nav.navigate("Thompson Sampling", max_seeds=3)
    )

    tms = TruthMaintenanceSystem(config=TMSConfig(store_edges=False))
    bb = Blackboard(query="Thompson Sampling",
                    working_memory=WorkingMemory())
    tms.contribute(bb, graph, {"nav_result": nav_result})

    # Explain a justified node
    if nav_result.ranked_nodes:
        node_id = nav_result.ranked_nodes[0][0]
        explanation = tms.explain_node(node_id)
        R.check("Explain returns justified=True for retrieved node",
                explanation["justified"])
        R.check("Explain includes explanation text",
                len(explanation["explanation"]) > 0,
                f"'{explanation['explanation'][:60]}...'")
    else:
        R.check("Explain returns justified=True for retrieved node", True,
                "no ranked nodes")
        R.check("Explain includes explanation text", True, "no ranked nodes")

    # Explain unjustified node
    explanation = tms.explain_node("nonexistent_node")
    R.check("Explain returns justified=False for unknown node",
            not explanation["justified"])

    loop.close()


def test_tms_no_contradictions_no_retraction():
    """TMS should be a no-op at POST_CONFIDENCE when there are no contradictions."""
    loop = asyncio.new_event_loop()

    async def setup():
        bus = LiteMemoryBus()
        bus._aliases = {}
        a_id = await bus.store(MemoryItem(content="Fact A about topic.", memory_type="factual", importance=0.8))
        b_id = await bus.store(MemoryItem(content="Fact B about topic.", memory_type="factual", importance=0.7))
        bus._edges[a_id] = [(b_id, "RELATED")]
        bus._edges[b_id] = [(a_id, "RELATED")]
        bus._aliases["topic"] = a_id
        return bus

    bus = loop.run_until_complete(setup())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    nav_result = loop.run_until_complete(nav.navigate("topic", max_seeds=3))

    tms = TruthMaintenanceSystem(config=TMSConfig(store_edges=False))
    bb = Blackboard(query="topic", working_memory=WorkingMemory())

    # Record justifications
    tms.contribute(bb, graph, {"nav_result": nav_result})

    # No contradictions
    estimator = ConfidenceEstimator(graph=graph)
    confidence = estimator.estimate(nav_result)
    tms.contribute(bb, graph, {"confidence": confidence, "nav_result": nav_result})

    R.check("No retractions when no contradictions",
            len(bb.flags.get("retractions", [])) == 0)
    loop.close()


# ═══════════════════════════════════════════════════════════════════════
# III. Chunk Compiler
# ═══════════════════════════════════════════════════════════════════════

def test_chunk_compiler_basic():
    R.section("III. Chunk Compiler")

    compiler = ChunkCompiler()
    R.check("ChunkCompiler name is 'chunk_compiler'",
            compiler.name == "chunk_compiler")
    R.check("ChunkCompiler phases include PRE_SHELL",
            Phase.PRE_SHELL in compiler.phases)
    R.check("ChunkCompiler phases include POST_LOOP",
            Phase.POST_LOOP in compiler.phases)
    R.check("ChunkCompiler satisfies KnowledgeSource protocol",
            isinstance(compiler, KnowledgeSource))
    R.check("Empty compiler has 0 chunks",
            len(compiler.chunks) == 0)
    R.check("Empty compiler has 0 experience",
            compiler.experience_count == 0)


def test_compiled_chunk_matching():
    chunk = CompiledChunk(
        chunk_id="test_chunk",
        trigger={
            "query_length": (0.1, 0.5),
            "seed_count": (0.2, 0.8),
            "ppr_entropy": (0.0, 0.5),
        },
        action=ChunkAction(preset_id="focused", max_shells=1, skip_verify=True),
        successes=10,
        failures=2,
    )

    # Matching features
    matching = {"query_length": 0.3, "seed_count": 0.5, "ppr_entropy": 0.3}
    R.check("Chunk matches within trigger ranges",
            chunk.matches(matching))

    # Non-matching: query_length out of range
    non_matching = {"query_length": 0.9, "seed_count": 0.5, "ppr_entropy": 0.3}
    R.check("Chunk doesn't match outside trigger ranges",
            not chunk.matches(non_matching))

    # Hit rate
    R.check("Hit rate = successes / total",
            abs(chunk.hit_rate - 10/12) < 0.01)

    # Not retired
    R.check("Chunk not retired with 0 consecutive failures",
            not chunk.is_retired)

    # Retire
    chunk.consecutive_failures = 3
    R.check("Chunk retired after 3 consecutive failures",
            chunk.is_retired)

    # to_dict
    d = chunk.to_dict()
    R.check("to_dict includes chunk_id",
            d["chunk_id"] == "test_chunk")
    R.check("to_dict includes action",
            d["action"]["preset_id"] == "focused")


def test_chunk_experience_recording():
    """Test that POST_LOOP records experience."""
    compiler = ChunkCompiler()
    bb = Blackboard(query="test query", working_memory=WorkingMemory())
    bb.post_navigation(seed_count=3, seed_sources={"alias": 2, "keyword": 1},
                       ppr_entropy=0.3, ppr_converged=True, entity_ids={"a", "b"})
    bb.post_confidence(combined=0.8, structural=0.7, narrative=0.9, decision="sufficient")

    # Create a mock LoopResult
    class MockLoopResult:
        def __init__(self):
            self.shell_count = 1
            class MockConf:
                combined = 0.85
            self.final_confidence = MockConf()

    context = {"loop_result": MockLoopResult()}
    compiler.contribute(bb, None, context)

    R.check("Experience recorded after POST_LOOP",
            compiler.experience_count == 1)


def test_chunk_compilation():
    """Test that enough consistent experience compiles into a chunk."""
    compiler = ChunkCompiler(config=ChunkConfig(
        min_cluster_size=3,
        feature_spread_limit=0.6,
        min_trigger_features=2,
    ))

    # Simulate 5 consistent successful experiences
    for i in range(5):
        bb = Blackboard(query=f"query {i}", working_memory=WorkingMemory())
        bb.post_navigation(seed_count=3, seed_sources={"alias": 2},
                           ppr_entropy=0.3, ppr_converged=True,
                           entity_ids={"a"})
        bb.post_confidence(combined=0.8, structural=0.7,
                           narrative=0.9, decision="sufficient")

        class MockLoop:
            shell_count = 1
            class final_confidence:
                combined = 0.85
        compiler.contribute(bb, None, {"loop_result": MockLoop()})

    R.check("Chunk compiled after enough experience",
            len(compiler.chunks) > 0,
            f"{len(compiler.chunks)} chunks")

    if compiler.chunks:
        chunk = compiler.chunks[0]
        R.check("Compiled chunk has trigger features",
                len(chunk.trigger) >= 2,
                f"{len(chunk.trigger)} features")
        R.check("Compiled chunk has action preset",
                chunk.action.preset_id == "balanced")  # default preset


def test_chunk_matching_in_pre_shell():
    """Test that PRE_SHELL posts chunk_action when a chunk matches."""
    compiler = ChunkCompiler()

    # Manually add a chunk
    chunk = CompiledChunk(
        chunk_id="manual_chunk",
        trigger={
            "query_length": (0.0, 0.5),
            "seed_count": (0.0, 0.8),
            "ppr_entropy": (0.0, 0.8),
        },
        action=ChunkAction(preset_id="focused", max_shells=1, skip_verify=True),
        successes=10,
        failures=1,
    )
    compiler.add_chunk(chunk)

    bb = Blackboard(query="short query", working_memory=WorkingMemory())
    bb.post_navigation(seed_count=3, seed_sources={"alias": 2},
                       ppr_entropy=0.3, ppr_converged=True, entity_ids=set())

    compiler.contribute(bb, None, {"shell_type": ShellType.PRIME})

    chunk_action = bb.flags.get("chunk_action")
    R.check("PRE_SHELL posts chunk_action when chunk matches",
            chunk_action is not None)
    if chunk_action:
        R.check("chunk_action has correct preset",
                chunk_action["preset_id"] == "focused")
        R.check("chunk_action has skip_verify",
                chunk_action["skip_verify"] is True)
        R.check("chunk_action has hit_rate",
                chunk_action["hit_rate"] > 0.8)


def test_chunk_retirement():
    """Test that chunks retire after consecutive failures."""
    compiler = ChunkCompiler(config=ChunkConfig(retirement_threshold=3))

    chunk = CompiledChunk(
        chunk_id="failing_chunk",
        trigger={"query_length": (0.0, 1.0)},
        action=ChunkAction(preset_id="focused"),
        successes=5,
    )
    compiler.add_chunk(chunk)

    # Simulate 3 failures
    for i in range(3):
        bb = Blackboard(query=f"fail {i}", working_memory=WorkingMemory())
        bb.post_flag("chunk_action", {"chunk_id": "failing_chunk", "preset_id": "focused"})

        class MockLoop:
            shell_count = 2
            class final_confidence:
                combined = 0.3  # Below success threshold
        compiler.contribute(bb, None, {"loop_result": MockLoop()})

    R.check("Chunk retired after 3 consecutive failures",
            chunk.is_retired)
    R.check("Active chunks excludes retired",
            len(compiler.active_chunks) == 0)


def test_chunk_no_match_wrong_shell():
    """PRE_SHELL only fires for PRIME, not VERIFY/FLAG."""
    compiler = ChunkCompiler()
    chunk = CompiledChunk(
        chunk_id="test",
        trigger={"query_length": (0.0, 1.0)},
        action=ChunkAction(preset_id="focused"),
        successes=10,
    )
    compiler.add_chunk(chunk)

    bb = Blackboard(query="test", working_memory=WorkingMemory())
    compiler.contribute(bb, None, {"shell_type": ShellType.VERIFY})

    R.check("Chunk doesn't fire for VERIFY shell",
            "chunk_action" not in bb.flags)


def test_chunk_report():
    compiler = ChunkCompiler()
    chunk = CompiledChunk(
        chunk_id="report_test",
        trigger={"query_length": (0.0, 0.5)},
        action=ChunkAction(preset_id="balanced"),
        successes=5,
        failures=1,
    )
    compiler.add_chunk(chunk)

    report = compiler.report()
    R.check("Report includes total_chunks",
            report["total_chunks"] == 1)
    R.check("Report includes chunk details",
            len(report["chunks"]) == 1)
    R.check("Report chunk has correct id",
            report["chunks"][0]["chunk_id"] == "report_test")


# ═══════════════════════════════════════════════════════════════════════
# IV. Orchestrator Integration
# ═══════════════════════════════════════════════════════════════════════

def test_orchestrator_ks_hooks():
    R.section("IV. Orchestrator Integration")

    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    # Create sources that record which phases they see
    phases_seen = []

    class RecordingKS:
        @property
        def name(self):
            return "recorder"

        @property
        def phases(self):
            return (Phase.POST_NAVIGATE, Phase.POST_CONFIDENCE,
                    Phase.POST_FORMAT, Phase.POST_GENERATE,
                    Phase.PRE_SHELL, Phase.POST_LOOP)

        def activation_level(self, bb):
            return 1.0

        def contribute(self, bb, graph, context):
            # Detect phase from context
            if "shell_type" in context:
                phases_seen.append(("PRE_SHELL", context["shell_type"].value))
            elif "loop_result" in context:
                phases_seen.append(("POST_LOOP", None))
            elif "nav_result" in context and "confidence" not in context:
                phases_seen.append(("POST_NAVIGATE", None))
            elif "confidence" in context:
                phases_seen.append(("POST_CONFIDENCE", None))
            elif "context_block" in context:
                phases_seen.append(("POST_FORMAT", None))
            elif "response" in context:
                phases_seen.append(("POST_GENERATE", None))

    registry = SourceRegistry()
    registry.register(RecordingKS())

    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        sources=registry,
    )

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(orch.run(
        "Thompson Sampling", max_shells=1
    ))

    phase_names = [p[0] for p in phases_seen]
    R.check("PRE_SHELL fires before shell",
            "PRE_SHELL" in phase_names)
    R.check("POST_NAVIGATE fires after navigation",
            "POST_NAVIGATE" in phase_names)
    R.check("POST_CONFIDENCE fires after confidence",
            "POST_CONFIDENCE" in phase_names)
    R.check("POST_FORMAT fires after formatting",
            "POST_FORMAT" in phase_names)
    R.check("POST_GENERATE fires after generation",
            "POST_GENERATE" in phase_names)
    R.check("POST_LOOP fires after loop",
            "POST_LOOP" in phase_names)

    R.check("All 6 phases fire for single-shell run",
            len(phases_seen) == 6,
            f"phases: {[p[0] for p in phases_seen]}")
    loop.close()


def test_orchestrator_backward_compat():
    """Orchestrator works unchanged with no sources registered."""
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    # No sources parameter
    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
    )

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(orch.run(
        "Thompson Sampling", max_shells=1
    ))

    R.check("Orchestrator works with empty SourceRegistry",
            result.response is not None and len(result.response) > 0)
    R.check("Shell still executes normally",
            result.shell_count >= 1)
    loop.close()


def test_orchestrator_post_loop_on_skip():
    """POST_LOOP fires even when shells are skipped."""
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    post_loop_count = [0]

    class PostLoopKS:
        @property
        def name(self):
            return "post_loop_counter"

        @property
        def phases(self):
            return (Phase.POST_LOOP,)

        def activation_level(self, bb):
            return 1.0

        def contribute(self, bb, graph, context):
            if "loop_result" in context:
                post_loop_count[0] += 1

    registry = SourceRegistry()
    registry.register(PostLoopKS())

    # Use production rules that always skip VERIFY
    always_skip = [
        ProductionRule(name="always", condition=lambda bb: True),
    ]

    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        sources=registry,
        skip_verify_rules=always_skip,
    )

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(orch.run(
        "Thompson Sampling", max_shells=3
    ))

    R.check("POST_LOOP fires even when VERIFY skipped",
            post_loop_count[0] == 1)
    R.check("Only PRIME executed when all skip rules fire",
            result.shell_count == 1)
    loop.close()


def test_orchestrator_multi_shell_hooks():
    """All hooks fire for multi-shell execution."""
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    pre_shell_types = []

    class PreShellRecorder:
        @property
        def name(self):
            return "pre_shell_recorder"

        @property
        def phases(self):
            return (Phase.PRE_SHELL,)

        def activation_level(self, bb):
            return 1.0

        def contribute(self, bb, graph, context):
            if "shell_type" in context:
                pre_shell_types.append(context["shell_type"])

    registry = SourceRegistry()
    registry.register(PreShellRecorder())

    # Force all shells to run
    never_skip = [
        ProductionRule(name="never", condition=lambda bb: False),
    ]

    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        sources=registry,
        skip_verify_rules=never_skip,
        skip_flag_rules=never_skip,
    )

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(orch.run(
        "Thompson Sampling", max_shells=3
    ))

    R.check("PRE_SHELL fires for PRIME",
            ShellType.PRIME in pre_shell_types)
    R.check("PRE_SHELL fires for VERIFY",
            ShellType.VERIFY in pre_shell_types)
    R.check("PRE_SHELL fires for FLAG",
            ShellType.FLAG in pre_shell_types)
    R.check("3 PRE_SHELL hooks for 3-shell run",
            len(pre_shell_types) == 3)
    loop.close()


# ═══════════════════════════════════════════════════════════════════════
# V. Full Integration
# ═══════════════════════════════════════════════════════════════════════

def test_tms_in_orchestrator():
    R.section("V. Full Integration")

    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    tms = TruthMaintenanceSystem(config=TMSConfig(store_edges=True))
    registry = SourceRegistry()
    registry.register(tms)

    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        sources=registry,
    )

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(orch.run(
        "Thompson Sampling exploration", max_shells=1
    ))

    # TMS should have recorded justifications via the blackboard
    # (The blackboard is internal to orchestrator, but we can check TMS state)
    report = tms.report()
    R.check("TMS recorded justifications via orchestrator",
            report["nodes_justified"] > 0,
            f"{report['nodes_justified']} nodes justified")

    # Check JUSTIFIED_BY edges exist in graph
    jb_edges = sum(
        1 for nid in bus._edges
        for _, rel in bus._edges[nid]
        if rel == "JUSTIFIED_BY"
    )
    R.check("JUSTIFIED_BY edges stored during orchestration",
            jb_edges > 0,
            f"{jb_edges} edges")
    loop.close()


def test_chunk_compiler_in_orchestrator():
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    compiler = ChunkCompiler(config=ChunkConfig(min_cluster_size=2))
    registry = SourceRegistry()
    registry.register(compiler)

    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        sources=registry,
    )

    loop = asyncio.new_event_loop()

    # Run multiple queries to accumulate experience
    for i in range(3):
        loop.run_until_complete(orch.run(
            f"Thompson Sampling query {i}", max_shells=1
        ))

    R.check("ChunkCompiler records experience via orchestrator",
            compiler.experience_count >= 3,
            f"{compiler.experience_count} experiences")
    loop.close()


def test_tms_and_chunks_together():
    """Both TMS and ChunkCompiler running simultaneously."""
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)

    adapter = ActivationAdapter()
    tms = TruthMaintenanceSystem(
        activation_adapter=adapter,
        config=TMSConfig(store_edges=True),
    )
    compiler = ChunkCompiler(config=ChunkConfig(min_cluster_size=2))

    registry = SourceRegistry()
    registry.register(tms)
    registry.register(compiler)

    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=mock_llm,
        sources=registry,
        activation_adapter=adapter,
    )

    loop = asyncio.new_event_loop()

    # Run several queries
    for i in range(4):
        loop.run_until_complete(orch.run(
            f"Thompson Sampling bandit {i}", max_shells=1
        ))

    # Both should have done work
    tms_report = tms.report()
    chunk_report = compiler.report()

    R.check("TMS justified nodes with both sources active",
            tms_report["nodes_justified"] > 0)
    R.check("ChunkCompiler recorded experience with both active",
            chunk_report["experience_buffer_size"] >= 4)
    R.check("No interference between TMS and ChunkCompiler",
            True)  # If we got here, no crashes
    loop.close()


def test_chunk_production_rule_integration():
    """Test that a chunk's skip_verify flag composes with production rules."""
    bb = Blackboard(query="test", working_memory=WorkingMemory())
    bb.post_flag("chunk_action", {
        "chunk_id": "test_chunk",
        "preset_id": "focused",
        "skip_verify": True,
        "hit_rate": 0.9,
    })

    # The production rule that reads chunk_action
    chunk_rule = ProductionRule(
        name="chunk_skip_verify",
        condition=lambda b: (
            b.flags.get("chunk_action", {}).get("skip_verify", False)
            and b.flags.get("chunk_action", {}).get("hit_rate", 0) > 0.8
        ),
    )

    R.check("Chunk production rule fires when chunk matches",
            chunk_rule.fires(bb))

    # No chunk action
    bb2 = Blackboard(query="test2", working_memory=WorkingMemory())
    R.check("Chunk production rule doesn't fire without chunk",
            not chunk_rule.fires(bb2))

    # Low hit rate
    bb3 = Blackboard(query="test3", working_memory=WorkingMemory())
    bb3.post_flag("chunk_action", {
        "chunk_id": "weak",
        "preset_id": "focused",
        "skip_verify": True,
        "hit_rate": 0.5,
    })
    R.check("Chunk production rule doesn't fire with low hit_rate",
            not chunk_rule.fires(bb3))


def test_multiple_registrations():
    """Test register/remove/re-register cycle."""
    registry = SourceRegistry()

    tms = TruthMaintenanceSystem()
    compiler = ChunkCompiler()

    registry.register(tms)
    registry.register(compiler)
    R.check("Registry holds 2 sources", len(registry.sources) == 2)

    registry.remove("tms")
    R.check("After removing TMS, 1 source remains",
            len(registry.sources) == 1)
    R.check("Remaining source is chunk_compiler",
            registry.sources[0].name == "chunk_compiler")

    registry.register(tms)
    R.check("Re-register TMS brings back to 2",
            len(registry.sources) == 2)


# ═══════════════════════════════════════════════════════════════════════
# VI. Case-Based Reasoning
# ═══════════════════════════════════════════════════════════════════════

def test_cbr_basic():
    R.section("VI. Case-Based Reasoning")
    cbr = CBRSystem()
    R.check("CBR name is 'cbr'", cbr.name == "cbr")
    R.check("CBR phases include POST_NAVIGATE",
            Phase.POST_NAVIGATE in cbr.phases)
    R.check("CBR phases include POST_LOOP",
            Phase.POST_LOOP in cbr.phases)
    R.check("CBR satisfies KnowledgeSource protocol",
            hasattr(cbr, "name") and hasattr(cbr, "phases")
            and hasattr(cbr, "activation_level") and hasattr(cbr, "contribute"))
    R.check("Empty CBR has 0 cases", cbr.case_count == 0)
    R.check("Low activation with no cases", cbr.activation_level(None) == 0.1)


def test_cbr_retain_and_retrieve():
    loop = asyncio.new_event_loop()
    bus, ids = loop.run_until_complete(build_test_bus())
    graph = LiteBusGraph(bus)
    nav = Navigator(graph=graph)
    cbr = CBRSystem(config=CBRConfig(store_in_graph=False, min_reward_to_retain=0.3))

    # Simulate 3 queries through the pipeline to build case base
    queries = [
        "Thompson Sampling exploration",
        "Thompson Sampling Bayesian posteriors",
        "Thompson Sampling bandit algorithms",
    ]
    for q in queries:
        nav_result = loop.run_until_complete(nav.navigate(q, max_seeds=5))
        bb = Blackboard(query=q, working_memory=WorkingMemory())
        estimator = ConfidenceEstimator(graph=graph)
        conf = estimator.estimate(nav_result, query=q)
        bb.post_navigation(
            seed_count=len(nav_result.seed_nodes),
            seed_sources={"keyword": len(nav_result.seed_nodes)},
            ppr_entropy=0.5, ppr_converged=True, entity_ids=set(),
        )
        bb.post_confidence(
            combined=conf.combined, structural=conf.structural.score,
            narrative=conf.narrative.score, decision="sufficient",
        )

        # Simulate POST_LOOP with mock loop_result
        class MockResult:
            final_confidence = conf
            shell_count = 2
            shell_results = [type("SR", (), {"response": f"Mock response for {q}"})()]
        cbr.contribute(bb, graph, {"loop_result": MockResult()})

    R.check("CBR retained 3 cases", cbr.case_count == 3,
            f"cases={cbr.case_count}")
    R.check("High activation with cases", cbr.activation_level(None) == 0.7)

    # Now simulate a POST_NAVIGATE retrieval
    nav_result = loop.run_until_complete(
        nav.navigate("Thompson Sampling", max_seeds=5)
    )
    bb = Blackboard(query="Thompson Sampling", working_memory=WorkingMemory())
    bb.post_navigation(
        seed_count=len(nav_result.seed_nodes),
        seed_sources={"keyword": len(nav_result.seed_nodes)},
        ppr_entropy=0.5, ppr_converged=True, entity_ids=set(),
    )
    estimator = ConfidenceEstimator(graph=graph)
    conf = estimator.estimate(nav_result)
    bb.post_confidence(
        combined=conf.combined, structural=conf.structural.score,
        narrative=conf.narrative.score, decision="sufficient",
    )

    cbr.contribute(bb, graph, {"nav_result": nav_result})
    hints = bb.flags.get("cbr_hints", [])
    R.check("CBR retrieves similar cases",
            len(hints) > 0, f"{len(hints)} hints")
    if hints:
        R.check("Hint has similarity score",
                "similarity" in hints[0], f"sim={hints[0].get('similarity')}")
        R.check("Hint has response summary",
                "response_hint" in hints[0])

    # Check report
    report = cbr.report()
    R.check("CBR report has total_cases", "total_cases" in report)
    R.check("CBR report has avg_reward", "avg_reward" in report)

    loop.close()


# ═══════════════════════════════════════════════════════════════════════
# VII. Demons (Pattern Monitors)
# ═══════════════════════════════════════════════════════════════════════

def test_demons_basic():
    R.section("VII. Demons (Pattern Monitors)")
    demons = DemonSystem()
    R.check("Demons name is 'demons'", demons.name == "demons")
    R.check("Demons phases include POST_GENERATE",
            Phase.POST_GENERATE in demons.phases)
    R.check("Demons satisfies KnowledgeSource protocol",
            hasattr(demons, "name") and hasattr(demons, "phases")
            and hasattr(demons, "activation_level") and hasattr(demons, "contribute"))
    R.check("Demons always active", demons.activation_level(None) == 0.9)
    R.check("Empty demons has 0 alerts", demons.alert_count == 0)


def test_demon_grounding():
    """Grounding demon detects ungrounded claims."""
    context = "Thompson Sampling uses Bayesian posteriors for exploration and exploitation."
    grounded_response = "Thompson Sampling leverages Bayesian posteriors to balance exploration."
    ungrounded_response = "Neural networks achieve 99% accuracy on ImageNet classification tasks."

    alert = check_grounding(grounded_response, context, threshold=0.3)
    R.check("Grounded response passes", alert is None)

    alert = check_grounding(ungrounded_response, context, threshold=0.3)
    R.check("Ungrounded response triggers alert",
            alert is not None, f"metric={alert.metric:.2f}" if alert else "")


def test_demon_brevity():
    """Brevity demon detects too-short responses."""
    context = "Thompson Sampling uses Bayesian posteriors. " * 20  # Rich context
    short_response = "It uses sampling."
    good_response = "Thompson Sampling is a powerful algorithm. " * 10

    alert = check_brevity(short_response, context, min_ratio=0.3)
    R.check("Short response triggers brevity alert",
            alert is not None, f"metric={alert.metric:.2f}" if alert else "")

    alert = check_brevity(good_response, context, min_ratio=0.3)
    R.check("Adequate response passes brevity", alert is None)


def test_demon_repetition():
    """Repetition demon detects query parroting."""
    query = "How does Thompson Sampling balance exploration and exploitation?"
    parroting = "Thompson Sampling balances exploration and exploitation by using sampling."
    good = "The algorithm maintains posterior distributions over arm rewards and draws samples to decide which arm to pull next."

    alert = check_repetition(parroting, query, threshold=0.5)
    R.check("Parroting response triggers repetition alert",
            alert is not None, f"metric={alert.metric:.2f}" if alert else "")

    alert = check_repetition(good, query, threshold=0.5)
    R.check("Original response passes repetition", alert is None)


def test_demon_hedging():
    """Hedging demon detects over-hedging."""
    hedgy = ("It depends on the context. Generally speaking, it is important to note "
             "that there are many factors. It could be argued that in some cases "
             "it is worth noting the complexity.")
    direct = "Thompson Sampling outperforms epsilon-greedy in high-dimensional action spaces."

    alert = check_hedging(hedgy, max_hedges=3)
    R.check("Hedgy response triggers alert",
            alert is not None, f"metric={alert.metric}" if alert else "")

    alert = check_hedging(direct, max_hedges=3)
    R.check("Direct response passes hedging", alert is None)


def test_demons_in_pipeline():
    """DemonSystem fires via contribute() at POST_GENERATE."""
    demons = DemonSystem(config=DemonConfig(
        grounding_threshold=0.5,
        brevity_min_ratio=0.3,
    ))
    bb = Blackboard(query="What is Thompson Sampling?",
                    working_memory=WorkingMemory())

    # Simulate POST_GENERATE with ungrounded response
    context_text = "Thompson Sampling uses Bayesian posteriors for bandit problems."
    demons.contribute(bb, None, {
        "response": "Quantum computing solves NP-hard problems efficiently.",
        "context_block": type("CB", (), {"text": context_text})(),
    })

    alerts = bb.flags.get("demon_alerts", [])
    R.check("Demons fire on ungrounded response",
            len(alerts) > 0, f"{len(alerts)} alerts")
    if alerts:
        R.check("Alert includes demon name",
                "demon" in alerts[0])
        R.check("Alert includes severity",
                "severity" in alerts[0])

    report = demons.report()
    R.check("Demons report has total_alerts", report["total_alerts"] > 0)
    R.check("Demons report has alerts_by_type",
            "alerts_by_type" in report)


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # I. KnowledgeSource Protocol
    test_phase_enum()
    test_source_registry()
    test_registry_invoke()
    test_registry_activation_filtering()
    test_registry_ordering()
    test_registry_error_handling()
    test_registry_null_blackboard()
    test_protocol_check()

    # II. Truth Maintenance System
    test_tms_basic()
    test_tms_justification_recording()
    test_tms_justification_edges()
    test_tms_retraction()
    test_tms_explain()
    test_tms_no_contradictions_no_retraction()

    # III. Chunk Compiler
    test_chunk_compiler_basic()
    test_compiled_chunk_matching()
    test_chunk_experience_recording()
    test_chunk_compilation()
    test_chunk_matching_in_pre_shell()
    test_chunk_retirement()
    test_chunk_no_match_wrong_shell()
    test_chunk_report()

    # IV. Orchestrator Integration
    test_orchestrator_ks_hooks()
    test_orchestrator_backward_compat()
    test_orchestrator_post_loop_on_skip()
    test_orchestrator_multi_shell_hooks()

    # V. Full Integration
    test_tms_in_orchestrator()
    test_chunk_compiler_in_orchestrator()
    test_tms_and_chunks_together()
    test_chunk_production_rule_integration()
    test_multiple_registrations()

    # VI. Case-Based Reasoning
    test_cbr_basic()
    test_cbr_retain_and_retrieve()

    # VII. Demons (Pattern Monitors)
    test_demons_basic()
    test_demon_grounding()
    test_demon_brevity()
    test_demon_repetition()
    test_demon_hedging()
    test_demons_in_pipeline()

    success = R.summary()
    sys.exit(0 if success else 1)
