import pytest
import time
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from uuid import UUID

import hololoom  # noqa: F401 — installs _CoreRedirectFinder


# ============================================================================
# types.py — Enums
# ============================================================================

class TestComplexityLevel:
    def test_members(self):
        from hololoom.protocols.types import ComplexityLevel
        assert set(ComplexityLevel.__members__) == {"LITE", "FAST", "FULL", "RESEARCH"}

    @pytest.mark.parametrize("name,value", [
        ("LITE", 3), ("FAST", 5), ("FULL", 7), ("RESEARCH", 9),
    ])
    def test_values(self, name, value):
        from hololoom.protocols.types import ComplexityLevel
        assert ComplexityLevel[name].value == value

    def test_ordering_by_value(self):
        from hololoom.protocols.types import ComplexityLevel
        ordered = sorted(ComplexityLevel, key=lambda c: c.value)
        assert [c.name for c in ordered] == ["LITE", "FAST", "FULL", "RESEARCH"]


class TestBanditStrategy:
    def test_members(self):
        from hololoom.protocols.types import BanditStrategy
        assert set(BanditStrategy.__members__) == {
            "EPSILON_GREEDY", "BAYESIAN_BLEND", "PURE_THOMPSON",
        }

    @pytest.mark.parametrize("name,value", [
        ("EPSILON_GREEDY", "epsilon_greedy"),
        ("BAYESIAN_BLEND", "bayesian_blend"),
        ("PURE_THOMPSON", "pure_thompson"),
    ])
    def test_values(self, name, value):
        from hololoom.protocols.types import BanditStrategy
        assert BanditStrategy[name].value == value


# ============================================================================
# types.py — Dataclasses
# ============================================================================

class TestProvenanceTrace:
    def test_instantiation_minimal(self):
        from hololoom.protocols.types import ProvenanceTrace, ComplexityLevel
        t = ProvenanceTrace(
            operation_id="op_1",
            complexity_level=ComplexityLevel.FAST,
            start_time=time.perf_counter(),
        )
        assert t.operation_id == "op_1"
        assert t.complexity_level == ComplexityLevel.FAST
        assert t.modules_invoked == []
        assert t.protocol_calls == []
        assert t.shuttle_events == []
        assert t.performance_metrics == {}
        assert t.synthesis_chain == []
        assert t.temporal_contexts == []

    def test_add_protocol_call(self):
        from hololoom.protocols.types import ProvenanceTrace, ComplexityLevel
        t = ProvenanceTrace("op", ComplexityLevel.LITE, time.perf_counter())
        t.add_protocol_call("memory", "retrieve", 1.5, "10 results")
        assert len(t.protocol_calls) == 1
        call = t.protocol_calls[0]
        assert call["protocol"] == "memory"
        assert call["method"] == "retrieve"
        assert call["duration_ms"] == 1.5
        assert call["result_summary"] == "10 results"
        assert "timestamp" in call

    def test_add_shuttle_event(self):
        from hololoom.protocols.types import ProvenanceTrace, ComplexityLevel
        t = ProvenanceTrace("op", ComplexityLevel.LITE, time.perf_counter())
        t.add_shuttle_event("synthesis", "Combined 3", {"n": 3})
        assert len(t.shuttle_events) == 1
        assert t.shuttle_events[0]["event_type"] == "synthesis"
        assert t.shuttle_events[0]["data"] == {"n": 3}

    def test_add_shuttle_event_default_data(self):
        from hololoom.protocols.types import ProvenanceTrace, ComplexityLevel
        t = ProvenanceTrace("op", ComplexityLevel.LITE, time.perf_counter())
        t.add_shuttle_event("start", "started")
        assert t.shuttle_events[0]["data"] == {}

    def test_add_synthesis_step(self):
        from hololoom.protocols.types import ProvenanceTrace, ComplexityLevel
        t = ProvenanceTrace("op", ComplexityLevel.FULL, time.perf_counter())
        t.add_synthesis_step("merge", ["a", "b"], "ab", 2.0)
        assert len(t.synthesis_chain) == 1
        assert t.synthesis_chain[0]["step"] == "merge"
        assert t.synthesis_chain[0]["inputs"] == ["a", "b"]

    def test_add_temporal_context(self):
        from hololoom.protocols.types import ProvenanceTrace, ComplexityLevel
        t = ProvenanceTrace("op", ComplexityLevel.FULL, time.perf_counter())
        t.add_temporal_context(1.0, 2.0, "session")
        assert len(t.temporal_contexts) == 1
        assert t.temporal_contexts[0]["context_type"] == "session"
        assert t.temporal_contexts[0]["metadata"] == {}

    def test_get_total_duration_ms(self):
        from hololoom.protocols.types import ProvenanceTrace, ComplexityLevel
        start = time.perf_counter()
        t = ProvenanceTrace("op", ComplexityLevel.LITE, start)
        duration = t.get_total_duration_ms()
        assert duration >= 0

    def test_get_protocol_summary(self):
        from hololoom.protocols.types import ProvenanceTrace, ComplexityLevel
        t = ProvenanceTrace("op", ComplexityLevel.FULL, time.perf_counter())
        t.add_protocol_call("memory", "retrieve", 1.0, "ok")
        t.add_protocol_call("memory", "store", 0.5, "ok")
        t.add_protocol_call("policy", "choose", 0.3, "ok")
        summary = t.get_protocol_summary()
        assert summary == {"memory": 2, "policy": 1}


class TestMythRLResult:
    def _make_result(self):
        from hololoom.protocols.types import MythRLResult, ProvenanceTrace, ComplexityLevel
        trace = ProvenanceTrace("op", ComplexityLevel.FAST, time.perf_counter())
        return MythRLResult(
            query="test?",
            output="answer",
            confidence=0.85,
            complexity_level=ComplexityLevel.FAST,
            provenance=trace,
            spacetime_coordinates={"x": 0.5},
        )

    def test_instantiation(self):
        r = self._make_result()
        assert r.query == "test?"
        assert r.output == "answer"
        assert r.confidence == 0.85

    def test_get_performance_summary(self):
        r = self._make_result()
        summary = r.get_performance_summary()
        assert "total_duration_ms" in summary
        assert summary["complexity_level"] == "FAST"
        assert summary["modules_used"] == 0
        assert summary["protocol_calls"] == 0

    def test_get_confidence_breakdown(self):
        r = self._make_result()
        breakdown = r.get_confidence_breakdown()
        assert breakdown["overall"] == 0.85
        assert "complexity_bonus" in breakdown
        assert "protocol_diversity" in breakdown


class TestQuery:
    def test_defaults(self):
        from hololoom.protocols.types import Query
        q = Query(text="hello")
        assert q.text == "hello"
        assert q.metadata == {}

    def test_with_metadata(self):
        from hololoom.protocols.types import Query
        q = Query(text="hello", metadata={"source": "test"})
        assert q.metadata["source"] == "test"


class TestContext:
    def test_defaults(self):
        from hololoom.protocols.types import Context
        c = Context()
        assert c.memories == []
        assert c.metadata == {}


class TestFeatures:
    def test_defaults(self):
        from hololoom.protocols.types import Features
        f = Features()
        assert f.motifs == []
        assert f.embeddings is None
        assert f.spectral is None
        assert f.metadata == {}


class TestMemoryShard:
    def test_required_fields(self):
        from hololoom.protocols.types import MemoryShard
        s = MemoryShard(id="s1", text="content")
        assert s.id == "s1"
        assert s.text == "content"
        assert s.entities == []
        assert s.motifs == []
        assert s.episode is None
        assert s.timestamp is None
        assert s.metadata == {}


class TestPolicyAction:
    def test_instantiation(self):
        from hololoom.protocols.types import PolicyAction
        a = PolicyAction(tool="answer", confidence=0.9)
        assert a.tool == "answer"
        assert a.confidence == 0.9
        assert a.metadata == {}


class TestActionPlan:
    def test_defaults(self):
        from hololoom.protocols.types import ActionPlan
        p = ActionPlan()
        assert p.actions == []
        assert p.tool is None
        assert p.confidence == 0.0
        assert p.adapter == "general"

    def test_post_init_derives_tool_from_actions(self):
        from hololoom.protocols.types import ActionPlan, PolicyAction
        actions = [PolicyAction(tool="search", confidence=0.8)]
        p = ActionPlan(actions=actions)
        assert p.tool == "search"
        assert p.confidence == 0.8

    def test_post_init_creates_action_from_tool(self):
        from hololoom.protocols.types import ActionPlan
        p = ActionPlan(tool="search", confidence=0.7)
        assert len(p.actions) == 1
        assert p.actions[0].tool == "search"


class TestToolCall:
    def test_defaults(self):
        from hololoom.protocols.types import ToolCall
        tc = ToolCall(tool="search")
        assert tc.tool == "search"
        assert tc.args == {}
        assert tc.metadata == {}


class TestToolResult:
    def test_instantiation(self):
        from hololoom.protocols.types import ToolResult
        tr = ToolResult(success=True, output="done")
        assert tr.success is True
        assert tr.output == "done"


class TestResponse:
    def test_instantiation(self):
        from hololoom.protocols.types import Response
        r = Response(text="hi", confidence=0.9)
        assert r.text == "hi"
        assert r.confidence == 0.9


class TestMotif:
    def test_defaults(self):
        from hololoom.protocols.types import Motif
        m = Motif(pattern="test")
        assert m.pattern == "test"
        assert m.span is None
        assert m.score == 0.0


# ============================================================================
# memory_types.py — Enums
# ============================================================================

class TestStrategy:
    def test_members(self):
        from hololoom.protocols.memory_types import Strategy
        assert set(Strategy.__members__) == {
            "TEMPORAL", "SEMANTIC", "GRAPH", "PATTERN", "FUSED", "BALANCED",
        }

    @pytest.mark.parametrize("name,value", [
        ("TEMPORAL", "temporal"), ("SEMANTIC", "semantic"),
        ("GRAPH", "graph"), ("FUSED", "fused"),
    ])
    def test_values(self, name, value):
        from hololoom.protocols.memory_types import Strategy
        assert Strategy[name].value == value


class TestQueryMode:
    def test_members(self):
        from hololoom.protocols.memory_types import QueryMode
        assert set(QueryMode.__members__) == {
            "FAST", "BALANCED", "COMPREHENSIVE", "RESEARCH",
        }


# ============================================================================
# memory_types.py — Dataclasses
# ============================================================================

class TestMemory:
    def _make_memory(self, **overrides):
        from hololoom.protocols.memory_types import Memory
        defaults = dict(
            id="m1",
            text="hello",
            timestamp=datetime(2025, 1, 1),
            context={"episode": "ep1"},
            metadata={"source": "test"},
        )
        defaults.update(overrides)
        return Memory(**defaults)

    def test_instantiation(self):
        m = self._make_memory()
        assert m.id == "m1"
        assert m.text == "hello"
        assert m.embedding is None

    def test_to_dict(self):
        m = self._make_memory()
        d = m.to_dict()
        assert d["id"] == "m1"
        assert d["timestamp"] == "2025-01-01T00:00:00"
        assert "embedding" not in d

    def test_to_dict_with_list_embedding(self):
        m = self._make_memory(embedding=[0.1, 0.2])
        d = m.to_dict()
        assert d["embedding"] == [0.1, 0.2]

    def test_from_dict_roundtrip(self):
        from hololoom.protocols.memory_types import Memory
        m = self._make_memory()
        d = m.to_dict()
        m2 = Memory.from_dict(d)
        assert m2.id == m.id
        assert m2.text == m.text
        assert m2.timestamp == m.timestamp

    def test_from_shard(self):
        from hololoom.protocols.memory_types import Memory
        shard = MagicMock()
        shard.id = "s1"
        shard.text = "shard text"
        shard.episode = "ep"
        shard.entities = ["e1"]
        shard.motifs = ["m1"]
        shard.metadata = {"k": "v"}
        m = Memory.from_shard(shard)
        assert m.id == "s1"
        assert m.text == "shard text"
        assert m.context["episode"] == "ep"
        assert m.context["entities"] == ["e1"]


class TestMemoryQuery:
    def test_defaults(self):
        from hololoom.protocols.memory_types import MemoryQuery
        q = MemoryQuery(text="search")
        assert q.text == "search"
        assert q.user_id == "default"
        assert q.limit == 5
        assert q.filters is None
        assert q.strategy is None

    def test_with_strategy(self):
        from hololoom.protocols.memory_types import MemoryQuery, Strategy
        q = MemoryQuery(text="s", strategy=Strategy.GRAPH)
        assert q.strategy == Strategy.GRAPH


class TestMemoryRetrievalResult:
    def test_instantiation(self):
        from hololoom.protocols.memory_types import MemoryRetrievalResult
        r = MemoryRetrievalResult(
            memories=[],
            scores=[],
            strategy_used="fused",
            metadata={},
        )
        assert r.memories == []
        assert r.strategy_used == "fused"


class TestShardsToMemories:
    def test_converts_list(self):
        from hololoom.protocols.memory_types import shards_to_memories
        shard = MagicMock()
        shard.id = "s1"
        shard.text = "text"
        shard.episode = None
        shard.entities = []
        shard.motifs = []
        shard.metadata = {}
        result = shards_to_memories([shard])
        assert len(result) == 1
        assert result[0].id == "s1"

    def test_empty_list(self):
        from hololoom.protocols.memory_types import shards_to_memories
        assert shards_to_memories([]) == []


# ============================================================================
# memory_protocols.py — Protocol Compliance
# ============================================================================

class TestMemoryStoreProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.memory_protocols import MemoryStore

        class Impl:
            async def store(self, memory, user_id="default"): ...
            async def store_many(self, memories, user_id="default"): ...
            async def get_by_id(self, memory_id): ...
            async def retrieve(self, query, strategy=None): ...
            async def delete(self, memory_id): ...
            async def health_check(self): ...

        assert isinstance(Impl(), MemoryStore)

    def test_non_conforming_rejected(self):
        from hololoom.protocols.memory_protocols import MemoryStore

        class Bad:
            pass

        assert not isinstance(Bad(), MemoryStore)


class TestMemoryNavigatorProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.memory_protocols import MemoryNavigator

        class Impl:
            async def navigate_forward(self, memory_id, relation_type=None): ...
            async def navigate_backward(self, memory_id, relation_type=None): ...
            async def find_path(self, start_id, end_id, max_depth=5): ...
            async def get_neighborhood(self, memory_id, radius=1): ...
            async def compute_similarity(self, id1, id2): ...

        assert isinstance(Impl(), MemoryNavigator)


class TestPatternDetectorProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.memory_protocols import PatternDetector

        class Impl:
            async def discover_patterns(self, memory_ids=None): ...
            async def find_anomalies(self, memory_ids=None, threshold=0.95): ...
            async def extract_themes(self, memory_ids=None, n_themes=5): ...

        assert isinstance(Impl(), PatternDetector)


# ============================================================================
# retrieval.py — Dataclasses and Protocol
# ============================================================================

class TestRetrievalResult:
    def test_instantiation(self):
        from hololoom.protocols.retrieval import RetrievalResult
        r = RetrievalResult(
            shards=[],
            strategy="static",
            query_text="q",
            k_requested=5,
            k_returned=0,
            retrieval_time_ms=1.0,
            avg_confidence=0.0,
            min_confidence=0.0,
            max_confidence=0.0,
            metadata={},
        )
        assert r.strategy == "static"
        assert r.k_requested == 5


class TestSpringActivationMetadata:
    def test_instantiation(self):
        from hololoom.protocols.retrieval import SpringActivationMetadata
        m = SpringActivationMetadata(
            iterations=47,
            converged=True,
            final_energy=0.23,
            seed_nodes=["n1"],
            activated_count=12,
            activation_threshold=0.1,
            node_activations={"n1": 0.9},
            embedding_time_ms=5.0,
            propagation_time_ms=10.0,
            shard_retrieval_time_ms=2.0,
        )
        assert m.iterations == 47
        assert m.converged is True
        assert m.seed_nodes == ["n1"]


class TestRetrievalStrategyProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.retrieval import RetrievalStrategy

        class Impl:
            async def retrieve(self, query, k=5, **kwargs): ...
            async def retrieve_with_metadata(self, query, k=5, **kwargs): ...

        assert isinstance(Impl(), RetrievalStrategy)


# ============================================================================
# safety.py — Enums, Dataclasses, Protocol
# ============================================================================

class TestSafetyRiskLevel:
    def test_members(self):
        from hololoom.protocols.safety import SafetyRiskLevel
        assert set(SafetyRiskLevel.__members__) == {
            "SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL",
        }


class TestSafetyGateDecision:
    def test_instantiation(self):
        from hololoom.protocols.safety import SafetyGateDecision, SafetyRiskLevel
        d = SafetyGateDecision(
            allowed=True,
            risk_level=SafetyRiskLevel.SAFE,
            reason="all good",
        )
        assert d.allowed is True
        assert d.blocked is False
        assert d.requires_approval is False

    def test_blocked_property(self):
        from hololoom.protocols.safety import SafetyGateDecision, SafetyRiskLevel
        d = SafetyGateDecision(
            allowed=False,
            risk_level=SafetyRiskLevel.HIGH,
            reason="dangerous",
        )
        assert d.blocked is True

    def test_defaults(self):
        from hololoom.protocols.safety import SafetyGateDecision, SafetyRiskLevel
        d = SafetyGateDecision(
            allowed=True,
            risk_level=SafetyRiskLevel.SAFE,
            reason="ok",
        )
        assert d.epistemic_confidence is None
        assert d.epistemic_warning is None
        assert d.metadata == {}
        assert isinstance(d.timestamp, datetime)


class TestSafetyGateProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.safety import SafetyGateProtocol

        class Impl:
            async def gate_reasoning(self, query_text, reasoning_mode,
                                     epistemic_confidence=None, context=None): ...

        assert isinstance(Impl(), SafetyGateProtocol)


# ============================================================================
# core_features.py — Protocols
# ============================================================================

class TestEmbedderProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.core_features import Embedder

        class Impl:
            def encode(self, texts): return [[0.1] * 96] * len(texts)
            @property
            def dimension(self): return 96

        impl = Impl()
        assert isinstance(impl, Embedder)
        assert impl.dimension == 96


class TestMotifDetectorProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.core_features import MotifDetector

        class Impl:
            async def detect(self, text): return []

        assert isinstance(Impl(), MotifDetector)


class TestPolicyEngineProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.core_features import PolicyEngine

        class Impl:
            async def choose_action(self, query, features, context): ...
            async def update(self, reward, metadata=None): ...

        assert isinstance(Impl(), PolicyEngine)


class TestRoutingStrategyProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.core_features import RoutingStrategy

        class Impl:
            async def select_mode(self, query, context=None): return "fast"
            async def update(self, mode, latency, quality): ...

        assert isinstance(Impl(), RoutingStrategy)


class TestExecutionEngineProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.core_features import ExecutionEngine

        class Impl:
            async def execute(self, tool, args, context=None): ...

        assert isinstance(Impl(), ExecutionEngine)


class TestToolRegistryProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.core_features import ToolRegistry

        class Impl:
            def register(self, name, tool): ...
            def get(self, name): return None
            def list_tools(self): return []

        assert isinstance(Impl(), ToolRegistry)


# ============================================================================
# shuttle.py — Protocols
# ============================================================================

class TestPatternSelectionProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.shuttle import PatternSelectionProtocol

        class Impl:
            async def select_pattern(self, query, context, complexity): ...
            async def assess_pattern_necessity(self, query): ...
            async def synthesize_patterns(self, primary, secondary): ...

        assert isinstance(Impl(), PatternSelectionProtocol)


class TestFeatureExtractionProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.shuttle import FeatureExtractionProtocol

        class Impl:
            async def extract_features(self, data, scales): ...
            async def extract_motifs(self, data, complexity): ...
            async def assess_extraction_needs(self, data): ...

        assert isinstance(Impl(), FeatureExtractionProtocol)


class TestWarpSpaceProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.shuttle import WarpSpaceProtocol

        class Impl:
            async def create_manifold(self, features, complexity): ...
            async def tension_threads(self, manifold, threads): ...
            async def compute_trajectories(self, manifold, start_points): ...
            async def experimental_operations(self, manifold, experiments): ...

        assert isinstance(Impl(), WarpSpaceProtocol)


class TestDecisionEngineProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.shuttle import DecisionEngineProtocol

        class Impl:
            async def make_decision(self, features, context, options): ...
            async def assess_decision_complexity(self, features): ...
            async def optimize_multi_criteria(self, criteria, constraints): ...

        assert isinstance(Impl(), DecisionEngineProtocol)


class TestShuttleToolExecutor:
    def test_runtime_checkable(self):
        from hololoom.protocols.shuttle import ToolExecutor

        class Impl:
            async def execute_tool(self, tool_name, parameters, context): ...
            async def list_available_tools(self, context): ...
            async def assess_tool_necessity(self, query, context): ...

        assert isinstance(Impl(), ToolExecutor)


# ============================================================================
# conscience.py — Enums
# ============================================================================

class TestStepType:
    def test_members(self):
        from hololoom.protocols.conscience import StepType
        expected = {"QUERY", "VERIFICATION", "RESEARCH", "PLAN_STEP",
                    "TOOL_EXECUTION", "SYNTHESIS"}
        assert set(StepType.__members__) == expected

    @pytest.mark.parametrize("name,value", [
        ("QUERY", 0), ("VERIFICATION", 1), ("RESEARCH", 2),
        ("PLAN_STEP", 3), ("TOOL_EXECUTION", 4), ("SYNTHESIS", 5),
    ])
    def test_values(self, name, value):
        from hololoom.protocols.conscience import StepType
        assert StepType[name].value == value

    def test_is_int_enum(self):
        from hololoom.protocols.conscience import StepType
        assert isinstance(StepType.QUERY, int)


class TestRiskLevel:
    def test_members(self):
        from hololoom.protocols.conscience import RiskLevel
        assert set(RiskLevel.__members__) == {
            "NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL",
        }

    def test_ordering(self):
        from hololoom.protocols.conscience import RiskLevel
        assert RiskLevel.NONE < RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_is_int_enum(self):
        from hololoom.protocols.conscience import RiskLevel
        assert isinstance(RiskLevel.NONE, int)


# ============================================================================
# conscience.py — ConscienceDecision
# ============================================================================

class TestConscienceDecision:
    def _make(self, **overrides):
        from hololoom.protocols.conscience import ConscienceDecision, RiskLevel
        defaults = dict(
            allowed=True,
            risk_level=RiskLevel.NONE,
            reason="ok",
            voice="QUIET",
        )
        defaults.update(overrides)
        return ConscienceDecision(**defaults)

    def test_defaults(self):
        d = self._make()
        assert d.voice_level == 0
        assert d.concerns == []
        assert d.guidance == ""
        assert d.witness_id is None
        assert d.step_index == 0
        assert d.evaluation_time_ms == 0.0
        assert d.metadata == {}

    def test_should_block_when_not_allowed(self):
        from hololoom.protocols.conscience import RiskLevel
        d = self._make(allowed=False, risk_level=RiskLevel.LOW)
        assert d.should_block is True

    def test_should_block_when_high_risk(self):
        from hololoom.protocols.conscience import RiskLevel
        d = self._make(allowed=True, risk_level=RiskLevel.HIGH)
        assert d.should_block is True

    def test_should_not_block_allowed_low_risk(self):
        d = self._make()
        assert d.should_block is False

    def test_requires_human_medium_risk(self):
        from hololoom.protocols.conscience import RiskLevel
        d = self._make(risk_level=RiskLevel.MEDIUM)
        assert d.requires_human is True

    def test_requires_human_false_low_risk(self):
        from hololoom.protocols.conscience import RiskLevel
        d = self._make(risk_level=RiskLevel.LOW)
        assert d.requires_human is False

    def test_to_dict(self):
        d = self._make()
        result = d.to_dict()
        assert result["allowed"] is True
        assert result["risk_level"] == "NONE"
        assert result["risk_level_value"] == 0
        assert result["voice"] == "QUIET"
        assert "timestamp" in result

    def test_to_safety_gate_decision(self):
        d = self._make(guidance="proceed")
        sgd = d.to_safety_gate_decision()
        assert sgd["allowed"] is True
        assert sgd["reason"] == "ok"
        assert sgd["guidance"] == "proceed"


# ============================================================================
# conscience.py — Factory Functions
# ============================================================================

class TestConscienceFactories:
    def test_create_allowed_decision(self):
        from hololoom.protocols.conscience import create_allowed_decision, RiskLevel
        d = create_allowed_decision()
        assert d.allowed is True
        assert d.risk_level == RiskLevel.NONE
        assert d.voice == "QUIET"

    def test_create_blocked_decision(self):
        from hololoom.protocols.conscience import create_blocked_decision, RiskLevel
        d = create_blocked_decision("dangerous op", risk_level=RiskLevel.CRITICAL)
        assert d.allowed is False
        assert d.risk_level == RiskLevel.CRITICAL
        assert d.voice == "SHOUT"
        assert d.voice_level == 3

    def test_create_blocked_decision_medium_risk(self):
        from hololoom.protocols.conscience import create_blocked_decision, RiskLevel
        d = create_blocked_decision("risky", risk_level=RiskLevel.MEDIUM)
        assert d.voice == "VOICE"
        assert d.voice_level == 2

    def test_create_review_decision(self):
        from hololoom.protocols.conscience import create_review_decision, RiskLevel
        d = create_review_decision("needs review")
        assert d.allowed is True
        assert d.risk_level == RiskLevel.MEDIUM
        assert d.requires_human is True
        assert d.voice == "VOICE"


# ============================================================================
# conscience.py — NullConscience
# ============================================================================

class TestNullConscience:
    @pytest.mark.asyncio
    async def test_consider_always_allows(self):
        from hololoom.protocols.conscience import NullConscience
        c = NullConscience()
        result = await c.consider("anything")
        assert result["allowed"] is True
        assert result["voice"] == "QUIET"

    @pytest.mark.asyncio
    async def test_witness_returns_empty_string(self):
        from hololoom.protocols.conscience import NullConscience
        c = NullConscience()
        record_id = await c.witness("act", {}, {})
        assert record_id == ""

    @pytest.mark.asyncio
    async def test_learn_is_noop(self):
        from hololoom.protocols.conscience import NullConscience
        c = NullConscience()
        await c.learn({"feedback": "good"})

    @pytest.mark.asyncio
    async def test_is_safe_always_true(self):
        from hololoom.protocols.conscience import NullConscience
        c = NullConscience()
        assert await c.is_safe("anything") is True

    def test_get_statistics(self):
        from hololoom.protocols.conscience import NullConscience
        c = NullConscience()
        assert c.get_statistics() == {"status": "disabled"}

    def test_satisfies_conscience_protocol(self):
        from hololoom.protocols.conscience import NullConscience, ConscienceProtocol
        assert isinstance(NullConscience(), ConscienceProtocol)


# ============================================================================
# department.py — Enums
# ============================================================================

class TestConfidenceLevel:
    def test_members(self):
        from hololoom.protocols.department import ConfidenceLevel
        assert set(ConfidenceLevel.__members__) == {
            "CRITICAL", "LOW", "MEDIUM", "HIGH", "VERIFIED",
        }


class TestPrivacyLevel:
    def test_members(self):
        from hololoom.protocols.department import PrivacyLevel
        assert set(PrivacyLevel.__members__) == {
            "PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED", "CRITICAL",
        }


class TestVerificationStatus:
    def test_members(self):
        from hololoom.protocols.department import VerificationStatus
        assert set(VerificationStatus.__members__) == {
            "PASSED", "FAILED", "SKIPPED", "IN_PROGRESS",
        }


# ============================================================================
# department.py — ConfidenceMetadata
# ============================================================================

class TestConfidenceMetadata:
    @pytest.mark.parametrize("score,expected_level", [
        (0.1, "CRITICAL"),
        (0.3, "LOW"),
        (0.6, "MEDIUM"),
        (0.8, "HIGH"),
        (0.96, "VERIFIED"),
    ])
    def test_from_score_level_classification(self, score, expected_level):
        from hololoom.protocols.department import ConfidenceMetadata
        cm = ConfidenceMetadata.from_score(score)
        assert cm.level.name == expected_level

    def test_from_score_clamps(self):
        from hololoom.protocols.department import ConfidenceMetadata
        cm = ConfidenceMetadata.from_score(1.5)
        assert cm.score == 1.0
        cm2 = ConfidenceMetadata.from_score(-0.5)
        assert cm2.score == 0.0

    def test_from_score_with_justification(self):
        from hololoom.protocols.department import ConfidenceMetadata
        cm = ConfidenceMetadata.from_score(0.9, justification=["high quality"])
        assert cm.justification == ["high quality"]

    def test_from_score_uncertainty_sources_alias(self):
        from hololoom.protocols.department import ConfidenceMetadata
        cm = ConfidenceMetadata.from_score(0.5, uncertainty_sources=["ambiguity"])
        assert cm.sources == ["ambiguity"]

    def test_defaults(self):
        from hololoom.protocols.department import ConfidenceMetadata, ConfidenceLevel
        cm = ConfidenceMetadata(score=0.5, level=ConfidenceLevel.MEDIUM)
        assert cm.justification == []
        assert cm.sources == []
        assert cm.calibration_count == 0
        assert cm.calibration_history == []


# ============================================================================
# department.py — PrivacyEnvelope
# ============================================================================

class TestPrivacyEnvelope:
    def test_instantiation(self):
        from hololoom.protocols.department import PrivacyEnvelope, PrivacyLevel
        env = PrivacyEnvelope(content="secret", level=PrivacyLevel.RESTRICTED)
        assert env.content == "secret"
        assert env.access_log == []

    def test_log_access(self):
        from hololoom.protocols.department import PrivacyEnvelope, PrivacyLevel
        env = PrivacyEnvelope(content="x", level=PrivacyLevel.CONFIDENTIAL)
        env.log_access("system_a", "processing")
        assert len(env.access_log) == 1
        assert env.access_log[0]["accessor"] == "system_a"
        assert env.access_log[0]["approved"] is True

    def test_log_access_denied(self):
        from hololoom.protocols.department import PrivacyEnvelope, PrivacyLevel
        env = PrivacyEnvelope(content="x", level=PrivacyLevel.CRITICAL)
        env.log_access("attacker", "probe", approved=False)
        assert env.access_log[0]["approved"] is False


# ============================================================================
# department.py — Request / Response
# ============================================================================

class TestDepartmentRequest:
    def test_defaults(self):
        from hololoom.protocols.department import DepartmentRequest
        req = DepartmentRequest()
        assert isinstance(req.task_id, UUID)
        assert req.task_type == ""
        assert req.parameters == {}
        assert req.context == {}
        assert req.constraints == {}
        assert req.priority == 50

    def test_custom_values(self):
        from hololoom.protocols.department import DepartmentRequest
        req = DepartmentRequest(task_type="retrieve", priority=90)
        assert req.task_type == "retrieve"
        assert req.priority == 90


class TestDepartmentResponse:
    def test_instantiation(self):
        from hololoom.protocols.department import (
            DepartmentResponse, ConfidenceMetadata,
        )
        from uuid import uuid4
        tid = uuid4()
        resp = DepartmentResponse(
            task_id=tid,
            result="answer",
            confidence=ConfidenceMetadata.from_score(0.8),
        )
        assert resp.task_id == tid
        assert resp.result == "answer"
        assert resp.latency_ms == 0.0
        assert resp.error is None


# ============================================================================
# department.py — Verification
# ============================================================================

class TestVerificationCheck:
    def test_instantiation(self):
        from hololoom.protocols.department import VerificationCheck, VerificationStatus
        check = VerificationCheck(
            name="domain",
            status=VerificationStatus.PASSED,
            reason="correct",
            score=0.9,
        )
        assert check.name == "domain"
        assert check.score == 0.9
        assert check.details == {}


class TestDSStarCheck:
    def test_instantiation(self):
        from hololoom.protocols.department import DSStarCheck
        check = DSStarCheck(dimension="Domain", passed=True, score=0.85)
        assert check.dimension == "Domain"
        assert check.passed is True
        assert check.details == ""


class TestVerificationResult:
    def test_defaults(self):
        from hololoom.protocols.department import VerificationResult
        vr = VerificationResult()
        assert vr.verified is False
        assert vr.checks == []
        assert vr.overall_score == 0.0

    def test_passed_no_checks(self):
        from hololoom.protocols.department import VerificationResult
        vr = VerificationResult()
        assert vr.passed is True

    def test_passed_all_pass(self):
        from hololoom.protocols.department import (
            VerificationResult, VerificationCheck, VerificationStatus,
        )
        vr = VerificationResult(checks=[
            VerificationCheck("a", VerificationStatus.PASSED, "ok"),
            VerificationCheck("b", VerificationStatus.PASSED, "ok"),
        ])
        assert vr.passed is True

    def test_passed_with_failure(self):
        from hololoom.protocols.department import (
            VerificationResult, VerificationCheck, VerificationStatus,
        )
        vr = VerificationResult(checks=[
            VerificationCheck("a", VerificationStatus.PASSED, "ok"),
            VerificationCheck("b", VerificationStatus.FAILED, "bad"),
        ])
        assert vr.passed is False

    def test_passed_all_skipped(self):
        from hololoom.protocols.department import (
            VerificationResult, VerificationCheck, VerificationStatus,
        )
        vr = VerificationResult(checks=[
            VerificationCheck("a", VerificationStatus.SKIPPED, "n/a"),
        ])
        assert vr.passed is True


# ============================================================================
# department.py — Department Protocol
# ============================================================================

class TestDepartmentProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.department import Department

        class Impl:
            async def execute(self, request): ...
            async def verify(self, response): ...
            async def refine(self, response): ...
            async def update_strategy(self, feedback): ...
            async def get_capabilities(self): ...
            async def get_metrics(self): ...
            async def health_check(self): ...

        assert isinstance(Impl(), Department)

    def test_alias(self):
        from hololoom.protocols.department import Department, DepartmentProtocol
        assert DepartmentProtocol is Department


# ============================================================================
# department.py — Helper Functions
# ============================================================================

class TestDepartmentHelpers:
    def test_create_simple_request(self):
        from hololoom.protocols.department import create_simple_request
        req = create_simple_request("retrieve", {"query": "hello"})
        assert req.task_type == "retrieve"
        assert req.parameters["query"] == "hello"
        assert req.constraints == {}

    def test_create_simple_request_with_constraints(self):
        from hololoom.protocols.department import create_simple_request
        req = create_simple_request("x", {}, {"max_latency_ms": 100})
        assert req.constraints["max_latency_ms"] == 100

    def test_create_simple_response(self):
        from hololoom.protocols.department import create_simple_response
        from uuid import uuid4
        tid = uuid4()
        resp = create_simple_response(tid, "result", 0.9)
        assert resp.task_id == tid
        assert resp.result == "result"
        assert resp.confidence.score == 0.9
        assert resp.confidence.level.name == "HIGH"


# ============================================================================
# department.py — Config and Manifest
# ============================================================================

class TestDepartmentConfig:
    def test_defaults(self):
        from hololoom.protocols.department import DepartmentConfig
        cfg = DepartmentConfig(name="ctx", domain="general")
        assert cfg.version == "1.0.0"
        assert cfg.enable_learning is True
        assert cfg.enable_verification is True
        assert cfg.max_latency_ms == 5000.0
        assert cfg.short_term_capacity == 100
        assert cfg.medium_term_capacity == 500
        assert cfg.long_term_capacity == 2000


class TestDepartmentManifest:
    def test_defaults(self):
        from hololoom.protocols.department import DepartmentManifest, DepartmentConfig
        cfg = DepartmentConfig(name="x", domain="y")
        m = DepartmentManifest(config=cfg)
        assert m.capabilities == []
        assert m.dependencies == []
        assert m.metadata == {}


# ============================================================================
# department.py — Learning Functions
# ============================================================================

class TestComputeLearningRate:
    def test_initial(self):
        from hololoom.protocols.department import compute_learning_rate
        assert compute_learning_rate(0) == pytest.approx(0.1)

    def test_decays(self):
        from hololoom.protocols.department import compute_learning_rate
        lr0 = compute_learning_rate(0)
        lr10 = compute_learning_rate(10)
        assert lr10 < lr0

    def test_floor(self):
        from hololoom.protocols.department import compute_learning_rate
        lr = compute_learning_rate(10000)
        assert lr == pytest.approx(0.001)


class TestShouldUpdateNow:
    def test_never_updated(self):
        from hololoom.protocols.department import should_update_now
        assert should_update_now(None) is True

    def test_recently_updated(self):
        from hololoom.protocols.department import should_update_now
        assert should_update_now(datetime.utcnow(), min_interval_seconds=60) is False

    def test_old_update(self):
        from hololoom.protocols.department import should_update_now
        old = datetime(2020, 1, 1)
        assert should_update_now(old, min_interval_seconds=60) is True


# ============================================================================
# orchestrator.py — WeavingOrchestratorProtocol
# ============================================================================

class TestWeavingOrchestratorProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.orchestrator import WeavingOrchestratorProtocol

        class Impl:
            async def weave(self, query, pattern_override=None,
                            complexity=None, auto_enhance=None): ...

        assert isinstance(Impl(), WeavingOrchestratorProtocol)


# ============================================================================
# jenny.py — Enums
# ============================================================================

class TestCompilationStrategy:
    def test_members(self):
        from hololoom.protocols.jenny import CompilationStrategy
        assert set(CompilationStrategy.__members__) == {
            "MINIMAL", "BALANCED", "COMPREHENSIVE", "AUTO",
        }

    def test_is_str_enum(self):
        from hololoom.protocols.jenny import CompilationStrategy
        assert isinstance(CompilationStrategy.MINIMAL, str)
        assert CompilationStrategy.MINIMAL == "minimal"


class TestRenderTarget:
    def test_members(self):
        from hololoom.protocols.jenny import RenderTarget
        assert set(RenderTarget.__members__) == {
            "HTML", "REACT", "AR", "JSON", "TERMINAL",
        }

    def test_is_str_enum(self):
        from hololoom.protocols.jenny import RenderTarget
        assert isinstance(RenderTarget.HTML, str)


# ============================================================================
# jenny.py — Protocols
# ============================================================================

class TestJennyCompilerProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.jenny import JennyCompilerProtocol

        class Impl:
            async def compile(self, spacetime, strategy=None, context=None): ...
            async def compile_stream(self, spacetime, strategy=None, context=None):
                yield None
            def get_panel_strategy(self, spacetime): return {}
            @property
            def compiler_version(self): return "1.0"

        assert isinstance(Impl(), JennyCompilerProtocol)


class TestJennyRendererProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.jenny import JennyRendererProtocol

        class Impl:
            async def render(self, specs, target=None, options=None): return ""
            async def render_single(self, spec, target=None, options=None): return ""
            async def render_update(self, old, new, target=None): return ""
            def supports_target(self, target): return True
            @property
            def supported_targets(self): return []
            @property
            def name(self): return "test"
            def supports_concurrent(self): return True

        assert isinstance(Impl(), JennyRendererProtocol)


class TestJennyLifecycleProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.jenny import JennyLifecycleProtocol

        class Impl:
            async def spawn(self, spec): ...
            async def pin(self, spec_id): ...
            async def dissolve(self, spec_id, trigger): ...
            async def archive(self, spec_id): ...
            async def get_active_panels(self, session_id=None): ...
            async def get_panel(self, spec_id): ...
            async def cleanup_stale(self, max_age_seconds=300): ...
            async def handle_memory_pressure(self, target_panel_count): ...

        assert isinstance(Impl(), JennyLifecycleProtocol)


class TestSpecLedgerProtocol:
    def test_runtime_checkable(self):
        from hololoom.protocols.jenny import SpecLedgerProtocol

        class Impl:
            async def log_spec(self, spec, spacetime_id, metadata=None): ...
            async def get_spec(self, spec_id): ...
            async def query_by_spacetime(self, spacetime_id): ...
            async def query_by_time_range(self, start, end, limit=100): ...
            async def query_by_panel_type(self, panel_type, limit=100): ...
            async def get_statistics(self, start_time=None, end_time=None): ...
            async def replay_session(self, session_id): ...

        assert isinstance(Impl(), SpecLedgerProtocol)


# ============================================================================
# recursive_reasoning.py — Enums
# ============================================================================

class TestReasoningStrategy:
    def test_members(self):
        from hololoom.protocols.recursive_reasoning import ReasoningStrategy
        expected = {"DIRECT", "VERIFY", "REFINE", "CRITIQUE", "DECOMPOSE",
                    "EXPLORE", "HOFSTADTER", "ADAPTIVE"}
        assert set(ReasoningStrategy.__members__) == expected


class TestStopCondition:
    def test_members(self):
        from hololoom.protocols.recursive_reasoning import StopCondition
        expected = {"MAX_ITERATIONS", "QUALITY_THRESHOLD", "NO_IMPROVEMENT",
                    "CONVERGENCE", "BUDGET_EXCEEDED"}
        assert set(StopCondition.__members__) == expected


# ============================================================================
# recursive_reasoning.py — Dataclasses
# ============================================================================

class TestRecursiveConfig:
    def test_defaults(self):
        from hololoom.protocols.recursive_reasoning import (
            RecursiveConfig, ReasoningStrategy,
        )
        cfg = RecursiveConfig()
        assert cfg.strategy == ReasoningStrategy.REFINE
        assert cfg.max_iterations == 5
        assert cfg.quality_threshold == 0.9
        assert cfg.min_improvement == 0.05
        assert cfg.time_budget_ms is None
        assert cfg.cost_budget is None
        assert cfg.enable_journal is True
        assert cfg.parallel_explore_branches == 3
        assert cfg.hofstadter_depth == 3


class TestReasoningJournal:
    def test_empty(self):
        from hololoom.protocols.recursive_reasoning import ReasoningJournal
        j = ReasoningJournal()
        assert j.traces == []
        assert j.get_latest() is None
        assert j.get_history() == ""
        assert j.get_confidence_trajectory() == []
        assert j.converged() is False

    def test_add_trace(self):
        from hololoom.protocols.recursive_reasoning import (
            ReasoningJournal, ReasoningTrace,
        )
        from hololoom.protocols.types import Features
        j = ReasoningJournal()
        trace = ReasoningTrace(
            iteration=0,
            thought="thinking",
            action="search",
            observation="found it",
            confidence=0.7,
            features_extracted=Features(),
        )
        j.add_trace(trace)
        assert j.get_latest() is trace
        assert j.get_confidence_trajectory() == [0.7]

    def test_converged_when_delta_small(self):
        from hololoom.protocols.recursive_reasoning import (
            ReasoningJournal, ReasoningTrace,
        )
        from hololoom.protocols.types import Features
        j = ReasoningJournal()
        for i, conf in enumerate([0.8, 0.82]):
            j.add_trace(ReasoningTrace(
                iteration=i, thought="t", action="a",
                observation="o", confidence=conf,
                features_extracted=Features(),
            ))
        assert j.converged(min_improvement=0.05) is True

    def test_not_converged_when_improving(self):
        from hololoom.protocols.recursive_reasoning import (
            ReasoningJournal, ReasoningTrace,
        )
        from hololoom.protocols.types import Features
        j = ReasoningJournal()
        for i, conf in enumerate([0.5, 0.8]):
            j.add_trace(ReasoningTrace(
                iteration=i, thought="t", action="a",
                observation="o", confidence=conf,
                features_extracted=Features(),
            ))
        assert j.converged(min_improvement=0.05) is False


class TestReasoningTrace:
    def test_to_text(self):
        from hololoom.protocols.recursive_reasoning import ReasoningTrace
        from hololoom.protocols.types import Features
        trace = ReasoningTrace(
            iteration=1,
            thought="hmm",
            action="search",
            observation="found",
            confidence=0.85,
            features_extracted=Features(),
        )
        text = trace.to_text()
        assert "Iteration 1" in text
        assert "hmm" in text
        assert "0.85" in text


# ============================================================================
# __init__.py — Package-level Imports
# ============================================================================

class TestPackageExports:
    def test_all_core_types_importable(self):
        from hololoom.protocols import (
            ComplexityLevel, BanditStrategy, ProvenanceTrace, MythRLResult,
        )

    def test_all_memory_types_importable(self):
        from hololoom.protocols import (
            Memory, MemoryQuery, MemoryRetrievalResult, Strategy, QueryMode,
            shards_to_memories,
        )

    def test_all_memory_protocols_importable(self):
        from hololoom.protocols import MemoryStore, MemoryNavigator, PatternDetector

    def test_all_core_feature_protocols_importable(self):
        from hololoom.protocols import (
            Embedder, MotifDetector, PolicyEngine,
            RoutingStrategy, ExecutionEngine, ToolRegistry,
        )

    def test_all_shuttle_protocols_importable(self):
        from hololoom.protocols import (
            PatternSelectionProtocol, FeatureExtractionProtocol,
            WarpSpaceProtocol, DecisionEngineProtocol, ToolExecutor,
        )

    def test_all_retrieval_importable(self):
        from hololoom.protocols import (
            RetrievalStrategy, RetrievalResult, SpringActivationMetadata,
        )

    def test_all_jenny_importable(self):
        from hololoom.protocols import (
            CompilationStrategy, RenderTarget,
            JennyCompilerProtocol, JennyRendererProtocol,
            JennyLifecycleProtocol, SpecLedgerProtocol,
        )

    def test_all_conscience_importable(self):
        from hololoom.protocols import (
            StepType, RiskLevel, ConscienceDecision,
            ConscienceProtocol, ExtendedConscienceProtocol,
            NullConscience,
            create_allowed_decision, create_blocked_decision, create_review_decision,
        )

    def test_all_department_importable(self):
        from hololoom.protocols import (
            ConfidenceLevel, ConfidenceMetadata,
            PrivacyLevel, PrivacyEnvelope,
            DepartmentRequest, DepartmentResponse,
            VerificationStatus, VerificationCheck, DSStarCheck, VerificationResult,
            Department, DepartmentProtocol,
            create_simple_request, create_simple_response,
            DepartmentConfig, DepartmentManifest,
            compute_learning_rate, should_update_now,
        )

    def test_compatibility_alias(self):
        from hololoom.protocols import ToolExecutionProtocol, ToolExecutor
        assert ToolExecutionProtocol is ToolExecutor


# ============================================================================
# core.py — Deprecated Shim
# ============================================================================

class TestCoreDeprecation:
    def test_imports_with_deprecation_warning(self):
        import importlib
        with pytest.warns(DeprecationWarning, match="deprecated"):
            if "hololoom.core.protocols.core" in __import__("sys").modules:
                del __import__("sys").modules["hololoom.core.protocols.core"]
            importlib.import_module("hololoom.core.protocols.core")
