#!/usr/bin/env python3
"""
Unit tests for core orchestrator internals.

Covers:
- jenny/panel_detection: classify_query_type, panel candidates, heuristic detection
- protocols/stage: StageExecutorProtocol, BaseStageExecutor, GateExecutor
- protocols/defaults: DefaultConvergence, DefaultToolExecutor, DefaultFeatureExtractor,
                      DefaultSpacetimeAssembler
- core/complexity_detection: assess_complexity_level, create_provenance_trace
- core/stat_mech_integration: shards_to_microstates, macrostates_to_shards
- core/background_tasks: spawn_background_task, start_background_consolidation
- core/metrics_collection: get_reflection_metrics, get_recursive_learning_stats, get_metrics
- retrieval/multipass_retrieval: multipass_memory_crawl, query_memory_backend
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from hololoom.protocols.types import (
    ComplexityLevel,
    MemoryShard,
    ProvenanceTrace,
    Query,
)
from hololoom.core.orchestrator.context import WeavingContext, create_weaving_context


# ============================================================================
# Helpers / Fixtures
# ============================================================================

def _make_query(text: str = "What is Thompson Sampling?") -> Query:
    return Query(text=text)


def _make_shard(id: str = "s1", text: str = "hello", **meta) -> MemoryShard:
    return MemoryShard(id=id, text=text, metadata=meta)


def _make_spacetime(
    response: str = "answer",
    confidence: float = 0.8,
    query_text: str = "",
    trace: Any = None,
):
    """Build a lightweight Spacetime-like namespace for panel detection tests."""
    if trace is None:
        trace = SimpleNamespace(
            threads_activated=[],
            stage_durations={},
            duration_ms=0,
            to_dict=lambda: {},
        )
    return SimpleNamespace(
        response=response,
        confidence=confidence,
        query_text=query_text,
        trace=trace,
        metadata={"session_id": "test", "spacetime_id": "st_1"},
        tool_used="answer",
        sources_used=[],
    )


def _make_orchestrator_stub(**overrides):
    """Minimal orchestrator-shaped object for functions that take orchestrator as arg."""
    defaults = {
        "enable_complexity_auto_detect": True,
        "default_pattern": None,
        "_complexity_thresholds": {
            "lite_max_words": 2,
            "fast_max_words": 20,
            "full_max_words": 50,
            "greeting_patterns": ["hi", "hello", "hey", "thanks", "thank you", "bye"],
            "simple_commands": ["show", "list", "get", "find", "open"],
            "question_words": ["what", "why", "how", "when", "where", "who", "which"],
            "knowledge_verbs": ["explain", "describe", "tell", "define", "clarify"],
            "analysis_verbs": ["analyze", "compare", "evaluate", "assess", "investigate"],
            "research_keywords": [
                "research", "deep", "comprehensive", "detailed", "thorough",
                "in-depth", "extensive",
            ],
        },
        "_crawl_config": {
            ComplexityLevel.LITE: {
                "passes": 1,
                "thresholds": [0.7],
                "limits": [5],
                "graph_expansion": False,
            },
            ComplexityLevel.FAST: {
                "passes": 2,
                "thresholds": [0.6, 0.75],
                "limits": [8, 12],
                "graph_expansion": True,
            },
            ComplexityLevel.FULL: {
                "passes": 3,
                "thresholds": [0.6, 0.75, 0.85],
                "limits": [12, 20, 15],
                "graph_expansion": True,
            },
            ComplexityLevel.RESEARCH: {
                "passes": 4,
                "thresholds": [0.5, 0.65, 0.8, 0.9],
                "limits": [20, 30, 25, 15],
                "graph_expansion": True,
            },
        },
        "memory": None,
        "enable_reflection": False,
        "enable_recursive_learning": False,
        "_recursive_components": None,
        "enable_production_hardening": False,
        "monitor": None,
        "enable_statistical_mechanics": False,
        "stat_mech_engine": None,
        "_consolidation_task": None,
        "_background_tasks": [],
        "_closed": False,
        "consolidation_interval": 60,
        "yarn_graph": SimpleNamespace(shards={}),
        "shards": [],
        "cfg": SimpleNamespace(user_id="test"),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ============================================================================
# Jenny Panel Detection Tests
# ============================================================================

class TestClassifyQueryType:
    def test_procedural(self):
        from hololoom.core.orchestrator.jenny.panel_detection import classify_query_type

        st = _make_spacetime(query_text="how to implement a cache")
        assert classify_query_type(st) == "procedural"

    def test_analytical(self):
        from hololoom.core.orchestrator.jenny.panel_detection import classify_query_type

        st = _make_spacetime(query_text="why is this approach better")
        assert classify_query_type(st) == "analytical"

    def test_exploratory(self):
        from hololoom.core.orchestrator.jenny.panel_detection import classify_query_type

        st = _make_spacetime(query_text="explore the landscape of LLMs")
        assert classify_query_type(st) == "exploratory"

    def test_factual_default(self):
        from hololoom.core.orchestrator.jenny.panel_detection import classify_query_type

        st = _make_spacetime(query_text="what is the capital of France")
        assert classify_query_type(st) == "factual"

    def test_empty_query(self):
        from hololoom.core.orchestrator.jenny.panel_detection import classify_query_type

        st = _make_spacetime(query_text="")
        assert classify_query_type(st) == "factual"

    def test_no_query_text_attr(self):
        from hololoom.core.orchestrator.jenny.panel_detection import classify_query_type

        st = SimpleNamespace()  # no query_text
        assert classify_query_type(st) == "factual"


class TestGetPanelTypeCandidates:
    def test_always_includes_text(self):
        from hololoom.core.orchestrator.jenny.panel_detection import get_panel_type_candidates
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        st = _make_spacetime(response="short", confidence=0.95)
        candidates = get_panel_type_candidates(st)
        assert PanelTypeJenny.TEXT in candidates

    def test_code_panel_on_code_blocks(self):
        from hololoom.core.orchestrator.jenny.panel_detection import get_panel_type_candidates
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        st = _make_spacetime(response="here is code:\n```python\nprint('hi')\n```", confidence=0.9)
        candidates = get_panel_type_candidates(st)
        assert PanelTypeJenny.CODE in candidates

    def test_confidence_panel_on_low_confidence(self):
        from hololoom.core.orchestrator.jenny.panel_detection import get_panel_type_candidates
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        st = _make_spacetime(confidence=0.5)
        candidates = get_panel_type_candidates(st)
        assert PanelTypeJenny.CONFIDENCE in candidates

    def test_graph_panel_on_multiple_threads(self):
        from hololoom.core.orchestrator.jenny.panel_detection import get_panel_type_candidates
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        trace = SimpleNamespace(
            threads_activated=["t1", "t2"],
            stage_durations={},
            duration_ms=0,
        )
        st = _make_spacetime(confidence=0.9, trace=trace)
        candidates = get_panel_type_candidates(st)
        assert PanelTypeJenny.GRAPH in candidates

    def test_timeline_panel_on_stage_durations(self):
        from hololoom.core.orchestrator.jenny.panel_detection import get_panel_type_candidates
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        trace = SimpleNamespace(
            threads_activated=[],
            stage_durations={"s1": 10.0, "s2": 20.0},
            duration_ms=30,
        )
        st = _make_spacetime(confidence=0.9, trace=trace)
        candidates = get_panel_type_candidates(st)
        assert PanelTypeJenny.TIMELINE in candidates

    def test_metric_panel_on_high_duration(self):
        from hololoom.core.orchestrator.jenny.panel_detection import get_panel_type_candidates
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        trace = SimpleNamespace(
            threads_activated=[],
            stage_durations={},
            duration_ms=100,
        )
        st = _make_spacetime(confidence=0.9, trace=trace)
        candidates = get_panel_type_candidates(st)
        assert PanelTypeJenny.METRIC in candidates

    def test_reasoning_panel_on_long_confident_response(self):
        from hololoom.core.orchestrator.jenny.panel_detection import get_panel_type_candidates
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        st = _make_spacetime(response="x" * 600, confidence=0.85)
        candidates = get_panel_type_candidates(st)
        assert PanelTypeJenny.REASONING in candidates


class TestDetectPanelTypeHeuristic:
    def test_code_detection(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_panel_type_heuristic
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        st = _make_spacetime(response="```python\nprint('hi')```")
        assert detect_panel_type_heuristic(st) == PanelTypeJenny.CODE

    def test_low_confidence(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_panel_type_heuristic
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        st = _make_spacetime(response="not sure", confidence=0.3)
        assert detect_panel_type_heuristic(st) == PanelTypeJenny.CONFIDENCE

    def test_graph_panel(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_panel_type_heuristic
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        trace = SimpleNamespace(
            threads_activated=["a", "b", "c"],  # >2 threshold
            stage_durations={},
            duration_ms=0,
        )
        st = _make_spacetime(confidence=0.9, trace=trace)
        assert detect_panel_type_heuristic(st) == PanelTypeJenny.GRAPH

    def test_timeline_panel(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_panel_type_heuristic
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        trace = SimpleNamespace(
            threads_activated=[],
            stage_durations={"s1": 1, "s2": 2, "s3": 3, "s4": 4},  # >3 threshold
            duration_ms=10,
        )
        st = _make_spacetime(confidence=0.9, trace=trace)
        assert detect_panel_type_heuristic(st) == PanelTypeJenny.TIMELINE

    def test_metric_panel(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_panel_type_heuristic
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        trace = SimpleNamespace(
            threads_activated=[],
            stage_durations={},
            duration_ms=200,  # >100 threshold
        )
        st = _make_spacetime(confidence=0.9, trace=trace)
        assert detect_panel_type_heuristic(st) == PanelTypeJenny.METRIC

    def test_default_text(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_panel_type_heuristic
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        st = _make_spacetime(response="plain answer", confidence=0.9)
        assert detect_panel_type_heuristic(st) == PanelTypeJenny.TEXT


class TestDetectJennyPanelType:
    def test_heuristic_fallback_without_mrf(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_jenny_panel_type
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        st = _make_spacetime(response="```code```", confidence=0.9)
        result = detect_jenny_panel_type(st, jenny_mrf_compiler=None, jenny_learner=None)
        assert result == PanelTypeJenny.CODE

    def test_mrf_selection_used_when_available(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_jenny_panel_type
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        mock_compiler = MagicMock()
        mock_learner = MagicMock()
        mock_learner.select.return_value = PanelTypeJenny.REASONING

        st = _make_spacetime(response="long" * 200, confidence=0.85, query_text="how to do X")
        result = detect_jenny_panel_type(st, jenny_mrf_compiler=mock_compiler, jenny_learner=mock_learner)
        assert result == PanelTypeJenny.REASONING
        mock_learner.select.assert_called_once()

    def test_mrf_failure_falls_back_to_heuristic(self):
        from hololoom.core.orchestrator.jenny.panel_detection import detect_jenny_panel_type
        from hololoom.visualization.jenny_spec import PanelTypeJenny

        mock_compiler = MagicMock()
        mock_learner = MagicMock()
        mock_learner.select.side_effect = RuntimeError("boom")

        st = _make_spacetime(response="plain text", confidence=0.9)
        result = detect_jenny_panel_type(st, jenny_mrf_compiler=mock_compiler, jenny_learner=mock_learner)
        assert result == PanelTypeJenny.TEXT


class TestBuildJennyPanelContext:
    def test_builds_context_dict(self):
        from hololoom.core.orchestrator.jenny.panel_detection import build_jenny_panel_context

        query = _make_query("test query")
        trace = SimpleNamespace(to_dict=lambda: {"stages": 3})
        st = _make_spacetime(response="answer", confidence=0.7, trace=trace)
        st.metadata = {"session_id": "s1", "spacetime_id": "st_1"}
        st.tool_used = "research"

        pattern = SimpleNamespace(name="FAST")
        complexity = SimpleNamespace(name="FAST")

        ctx = build_jenny_panel_context(query, st, pattern, complexity)
        assert ctx["session_id"] == "s1"
        assert ctx["pattern"] == "FAST"
        assert ctx["complexity"] == "FAST"
        assert ctx["response"] == "answer"
        assert ctx["confidence"] == 0.7
        assert ctx["tool_used"] == "research"


# ============================================================================
# Stage Protocol Tests
# ============================================================================

class TestStageExecutorProtocol:
    def test_protocol_compliance_with_matching_class(self):
        from hololoom.core.orchestrator.protocols.stage import StageExecutorProtocol

        class MyExecutor:
            stage_id = 1
            stage_name = "Test"

            async def execute(self, ctx):
                return ctx

        assert isinstance(MyExecutor(), StageExecutorProtocol)

    def test_protocol_non_compliance(self):
        from hololoom.core.orchestrator.protocols.stage import StageExecutorProtocol

        class Incomplete:
            pass

        assert not isinstance(Incomplete(), StageExecutorProtocol)


class TestBaseStageExecutor:
    def test_timing_helpers(self):
        from hololoom.core.orchestrator.protocols.stage import BaseStageExecutor

        class Concrete(BaseStageExecutor):
            stage_id = 5
            stage_name = "Warp"

            async def execute(self, ctx):
                return ctx

        executor = Concrete()
        start = executor._start_timing()
        assert isinstance(start, float)

        ctx = create_weaving_context(_make_query())
        duration = executor._record_timing(ctx, "warp_space", start)
        assert duration >= 0
        assert "warp_space" in ctx.stage_timings

    def test_logging_helpers(self):
        from hololoom.core.orchestrator.protocols.stage import BaseStageExecutor

        class Concrete(BaseStageExecutor):
            stage_id = 2
            stage_name = "Chrono"

            async def execute(self, ctx):
                return ctx

        executor = Concrete()
        # Should not raise
        executor._log_stage_start()
        executor._log_stage_complete(42.5)
        executor._log_stage_error("something went wrong")
        executor._log_stage_warning("heads up")

    def test_context_helpers(self):
        from hololoom.core.orchestrator.protocols.stage import BaseStageExecutor

        class Concrete(BaseStageExecutor):
            stage_id = 3
            stage_name = "Yarn"

            async def execute(self, ctx):
                return ctx

        executor = Concrete()
        ctx = create_weaving_context(_make_query())
        executor._add_error(ctx, "error msg")
        executor._add_warning(ctx, "warn msg")
        assert "error msg" in ctx.errors
        assert "warn msg" in ctx.warnings

    def test_emit_stage_event_callback(self):
        from hololoom.core.orchestrator.protocols.stage import BaseStageExecutor

        events = []

        class Concrete(BaseStageExecutor):
            stage_id = 4
            stage_name = "Resonance"

            async def execute(self, ctx):
                return ctx

        executor = Concrete(emit_stage_event=lambda sid, sname, dur: events.append((sid, sname, dur)))
        ctx = create_weaving_context(_make_query())
        start = executor._start_timing()
        executor._record_timing(ctx, "resonance", start)
        assert len(events) == 1
        assert events[0][0] == 4
        assert events[0][1] == "Resonance"


class TestGateExecutor:
    @pytest.mark.asyncio
    async def test_gate_pass_through(self):
        from hololoom.core.orchestrator.protocols.stage import GateExecutor

        class PassGate(GateExecutor):
            stage_id = -1
            stage_name = "Pass Gate"
            timing_key = "pass_gate"

            async def evaluate(self, ctx):
                return None  # pass through

        gate = PassGate()
        ctx = create_weaving_context(_make_query())
        result_ctx = await gate.execute(ctx)
        assert not hasattr(result_ctx, "gate_result") or result_ctx.gate_result is None
        assert "pass_gate" in result_ctx.stage_timings

    @pytest.mark.asyncio
    async def test_gate_short_circuit(self):
        from hololoom.core.orchestrator.protocols.stage import GateExecutor

        class CacheGate(GateExecutor):
            stage_id = -1
            stage_name = "Cache"
            timing_key = "cache_check"

            async def evaluate(self, ctx):
                return {"cached": True, "response": "from cache"}

        gate = CacheGate()
        ctx = create_weaving_context(_make_query())
        result_ctx = await gate.execute(ctx)
        assert result_ctx.gate_result == {"cached": True, "response": "from cache"}


# ============================================================================
# Defaults Tests
# ============================================================================

class TestDefaultConvergence:
    def test_argmax_strategy(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultConvergence

        conv = DefaultConvergence(n_tools=4)
        result = conv.collapse([0.1, 0.8, 0.05, 0.05], strategy="argmax")
        assert result.tool_index == 1
        assert result.strategy == "argmax"

    def test_bayesian_blend_strategy(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultConvergence

        conv = DefaultConvergence(n_tools=3)
        result = conv.collapse([0.7, 0.2, 0.1], strategy="bayesian_blend")
        assert 0 <= result.tool_index < 3
        assert result.strategy == "bayesian_blend"

    def test_epsilon_greedy_strategy(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultConvergence

        conv = DefaultConvergence(n_tools=3, epsilon=0.0)  # no exploration
        result = conv.collapse([0.1, 0.9, 0.0], strategy="epsilon_greedy")
        assert result.tool_index == 1

    def test_pure_thompson_strategy(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultConvergence

        conv = DefaultConvergence(n_tools=3)
        result = conv.collapse([0.3, 0.3, 0.4], strategy="pure_thompson")
        assert 0 <= result.tool_index < 3

    def test_update_priors_success(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultConvergence

        conv = DefaultConvergence(n_tools=3)
        conv.update_priors(0, success=True, confidence=0.9)
        assert conv._alpha[0] == pytest.approx(1.9)
        assert conv._beta[0] == pytest.approx(1.0)

    def test_update_priors_failure(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultConvergence

        conv = DefaultConvergence(n_tools=3)
        conv.update_priors(1, success=False, confidence=0.3)
        assert conv._alpha[1] == pytest.approx(1.0)
        assert conv._beta[1] == pytest.approx(1.7)

    def test_update_priors_out_of_range(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultConvergence

        conv = DefaultConvergence(n_tools=2)
        conv.update_priors(5, success=True, confidence=1.0)  # out of range, no crash
        assert conv._alpha == [1.0, 1.0]

    def test_exploration_rate(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultConvergence

        conv = DefaultConvergence(epsilon=0.15)
        assert conv.get_exploration_rate() == pytest.approx(0.15)


class TestDefaultToolExecutor:
    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultToolExecutor

        executor = DefaultToolExecutor()
        query = _make_query("test question")
        result = await executor.execute("answer", query, {})
        assert "answer" in result["response"].lower() or "test question" in result["response"].lower()
        assert result["metadata"]["fallback"] is True

    @pytest.mark.asyncio
    async def test_available_tools(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultToolExecutor

        executor = DefaultToolExecutor(tools=["answer", "research"])
        assert executor.get_available_tools() == ["answer", "research"]

    @pytest.mark.asyncio
    async def test_validate_tool(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultToolExecutor

        executor = DefaultToolExecutor(tools=["answer", "research"])
        assert await executor.validate_tool("answer", {}) is True
        assert await executor.validate_tool("unknown", {}) is False


class TestDefaultFeatureExtractor:
    @pytest.mark.asyncio
    async def test_extract_basic_features(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultFeatureExtractor

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros(384)

        cfg = SimpleNamespace(embedding_dim=384)
        extractor = DefaultFeatureExtractor(embedder=mock_embedder, cfg=cfg)

        query = _make_query("What is Thompson Sampling?")
        threads = [_make_shard("t1", "thread text")]

        features = await extractor.extract(query, threads)
        assert "embeddings" in features
        assert "motifs" in features
        assert "spectral" in features
        assert "interrogative" in features["motifs"]

    @pytest.mark.asyncio
    async def test_extract_motifs_comparative(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultFeatureExtractor

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros(384)
        cfg = SimpleNamespace(embedding_dim=384)
        extractor = DefaultFeatureExtractor(embedder=mock_embedder, cfg=cfg)

        query = _make_query("compare Python vs Rust")
        features = await extractor.extract(query, [])
        assert "comparative" in features["motifs"]

    def test_get_feature_dim(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultFeatureExtractor

        mock_embedder = MagicMock()
        cfg = SimpleNamespace(embedding_dim=768)
        extractor = DefaultFeatureExtractor(embedder=mock_embedder, cfg=cfg)
        assert extractor.get_feature_dim() == 768

    def test_get_feature_dim_default(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultFeatureExtractor

        mock_embedder = MagicMock()
        cfg = SimpleNamespace()  # no embedding_dim
        extractor = DefaultFeatureExtractor(embedder=mock_embedder, cfg=cfg)
        assert extractor.get_feature_dim() == 384


class TestDefaultSpacetimeAssembler:
    def test_assemble_basic(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultSpacetimeAssembler

        cfg = SimpleNamespace()
        assembler = DefaultSpacetimeAssembler(cfg=cfg)

        ctx = create_weaving_context(_make_query("test"))
        ctx.tool_result = {"response": "the answer"}
        ctx.pattern_spec = SimpleNamespace(name="FAST")
        ctx.threads = [_make_shard("t1", "t")]

        spacetime = assembler.assemble(ctx)
        assert spacetime.response == "the answer"
        assert spacetime.confidence > 0
        assert spacetime.metadata["pattern_used"] == "FAST"

    def test_confidence_calculation_with_threads(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultSpacetimeAssembler

        cfg = SimpleNamespace()
        assembler = DefaultSpacetimeAssembler(cfg=cfg)

        ctx = create_weaving_context(_make_query("test"))
        ctx.threads = [_make_shard(f"t{i}", f"text{i}") for i in range(5)]
        ctx.collapse_result = SimpleNamespace(confidence=0.8, tool_index=0)

        confidence = assembler.calculate_confidence(ctx)
        # 0.8 base + min(0.1, 5 * 0.02) = 0.8 + 0.1 = 0.9
        assert confidence == pytest.approx(0.9)

    def test_confidence_calculation_with_errors(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultSpacetimeAssembler

        cfg = SimpleNamespace()
        assembler = DefaultSpacetimeAssembler(cfg=cfg)

        ctx = create_weaving_context(_make_query("test"))
        ctx.errors = ["err1", "err2"]

        confidence = assembler.calculate_confidence(ctx)
        # 0.5 base + 0 threads - min(0.2, 2*0.05) = 0.5 - 0.1 = 0.4
        assert confidence == pytest.approx(0.4)

    def test_create_trace(self):
        from hololoom.core.orchestrator.protocols.defaults import DefaultSpacetimeAssembler

        cfg = SimpleNamespace()
        assembler = DefaultSpacetimeAssembler(cfg=cfg)

        ctx = create_weaving_context(_make_query("test"))
        ctx.stage_timings = {"pattern": 10.0, "chrono": 5.0}

        trace = assembler.create_trace(ctx)
        assert trace.duration_ms == pytest.approx(15.0)
        assert trace.stage_durations == {"pattern": 10.0, "chrono": 5.0}


# ============================================================================
# Complexity Detection Tests
# ============================================================================

class TestAssessComplexityLevel:
    def test_lite_for_greeting(self):
        from hololoom.core.orchestrator.core.complexity_detection import assess_complexity_level

        orch = _make_orchestrator_stub()
        level = assess_complexity_level(orch, _make_query("hi"))
        assert level == ComplexityLevel.LITE

    def test_fast_for_question(self):
        from hololoom.core.orchestrator.core.complexity_detection import assess_complexity_level

        orch = _make_orchestrator_stub()
        level = assess_complexity_level(orch, _make_query("what is Python?"))
        assert level == ComplexityLevel.FAST

    def test_fast_for_knowledge_verb(self):
        from hololoom.core.orchestrator.core.complexity_detection import assess_complexity_level

        orch = _make_orchestrator_stub()
        level = assess_complexity_level(orch, _make_query("explain recursion"))
        assert level == ComplexityLevel.FAST

    def test_research_for_analysis_verb(self):
        from hololoom.core.orchestrator.core.complexity_detection import assess_complexity_level

        orch = _make_orchestrator_stub()
        level = assess_complexity_level(orch, _make_query("analyze the market trends"))
        assert level == ComplexityLevel.RESEARCH

    def test_research_for_keyword(self):
        from hololoom.core.orchestrator.core.complexity_detection import assess_complexity_level

        orch = _make_orchestrator_stub()
        level = assess_complexity_level(orch, _make_query("give me a comprehensive overview"))
        assert level == ComplexityLevel.RESEARCH

    def test_full_for_medium_query(self):
        from hololoom.core.orchestrator.core.complexity_detection import assess_complexity_level

        orch = _make_orchestrator_stub()
        words = " ".join(["word"] * 25)
        level = assess_complexity_level(orch, _make_query(words))
        # 25 words >= fast_max_words (20) and no analysis/research keywords
        assert level == ComplexityLevel.FULL

    def test_auto_detect_disabled_uses_pattern(self):
        from hololoom.core.orchestrator.core.complexity_detection import assess_complexity_level
        from hololoom.core.loom.command import PatternCard

        orch = _make_orchestrator_stub(
            enable_complexity_auto_detect=False,
            default_pattern=PatternCard.BARE,
        )
        level = assess_complexity_level(orch, _make_query("analyze deeply"))
        assert level == ComplexityLevel.LITE  # BARE maps to LITE

    def test_trace_records_event(self):
        from hololoom.core.orchestrator.core.complexity_detection import assess_complexity_level

        orch = _make_orchestrator_stub()
        trace = ProvenanceTrace(
            operation_id="test_op",
            complexity_level=ComplexityLevel.FAST,
            start_time=time.perf_counter(),
        )
        assess_complexity_level(orch, _make_query("what is X?"), trace=trace)
        assert len(trace.shuttle_events) == 1
        assert trace.shuttle_events[0]["event_type"] == "complexity_assessment"


class TestCreateProvenanceTrace:
    def test_creates_trace(self):
        from hololoom.core.orchestrator.core.complexity_detection import create_provenance_trace

        orch = _make_orchestrator_stub()
        trace = create_provenance_trace(orch, _make_query("test"), ComplexityLevel.FAST)
        assert trace.operation_id.startswith("weave_")
        assert trace.complexity_level == ComplexityLevel.FAST
        assert len(trace.shuttle_events) == 1


# ============================================================================
# Statistical Mechanics Integration Tests
# ============================================================================

class TestShardsToMicrostates:
    def test_converts_shards_with_embedding(self):
        from hololoom.core.orchestrator.core.stat_mech_integration import (
            shards_to_microstates,
            STATISTICAL_MECHANICS_AVAILABLE,
        )

        if not STATISTICAL_MECHANICS_AVAILABLE:
            pytest.skip("statistical_mechanics module not available")

        orch = _make_orchestrator_stub()
        shard = _make_shard("s1", "hello", source="test")
        shard.embedding = [0.1, 0.2, 0.3]

        microstates = shards_to_microstates(orch, [shard])
        assert len(microstates) == 1
        assert microstates[0].id == "s1"

    def test_skips_shards_without_embedding(self):
        from hololoom.core.orchestrator.core.stat_mech_integration import (
            shards_to_microstates,
            STATISTICAL_MECHANICS_AVAILABLE,
        )

        if not STATISTICAL_MECHANICS_AVAILABLE:
            pytest.skip("statistical_mechanics module not available")

        orch = _make_orchestrator_stub()
        shard = _make_shard("s1", "no embedding")

        microstates = shards_to_microstates(orch, [shard])
        assert len(microstates) == 0

    def test_returns_empty_when_stat_mech_unavailable(self):
        from hololoom.core.orchestrator.core import stat_mech_integration as smi

        original = smi.STATISTICAL_MECHANICS_AVAILABLE
        try:
            smi.STATISTICAL_MECHANICS_AVAILABLE = False
            orch = _make_orchestrator_stub()
            shard = _make_shard("s1", "hello")
            shard.embedding = [0.1, 0.2]
            result = smi.shards_to_microstates(orch, [shard])
            assert result == []
        finally:
            smi.STATISTICAL_MECHANICS_AVAILABLE = original


class TestMacrostatesToShards:
    def test_converts_macrostates(self):
        from hololoom.core.orchestrator.core.stat_mech_integration import (
            macrostates_to_shards,
            STATISTICAL_MECHANICS_AVAILABLE,
        )

        if not STATISTICAL_MECHANICS_AVAILABLE:
            pytest.skip("statistical_mechanics module not available")

        micro = SimpleNamespace(
            id="m1",
            state_vector=np.array([0.1, 0.2]),
            metadata={"content_preview": "test content"},
        )
        macro = SimpleNamespace(
            label="cluster_0",
            microstates=[micro],
            average_energy=1.0,
            entropy=0.5,
            free_energy=0.5,
            order_parameter=0.8,
        )

        orch = _make_orchestrator_stub()
        shards = macrostates_to_shards(orch, [macro])
        assert len(shards) == 1
        assert shards[0].id == "cluster_0_consolidated"
        assert shards[0].metadata["cluster_size"] == 1

    def test_skips_empty_macrostates(self):
        from hololoom.core.orchestrator.core.stat_mech_integration import (
            macrostates_to_shards,
            STATISTICAL_MECHANICS_AVAILABLE,
        )

        if not STATISTICAL_MECHANICS_AVAILABLE:
            pytest.skip("statistical_mechanics module not available")

        macro = SimpleNamespace(
            label="empty_cluster",
            microstates=[],
        )

        orch = _make_orchestrator_stub()
        shards = macrostates_to_shards(orch, [macro])
        assert len(shards) == 0

    def test_truncates_to_five_microstates_in_preview(self):
        from hololoom.core.orchestrator.core.stat_mech_integration import (
            macrostates_to_shards,
            STATISTICAL_MECHANICS_AVAILABLE,
        )

        if not STATISTICAL_MECHANICS_AVAILABLE:
            pytest.skip("statistical_mechanics module not available")

        micros = [
            SimpleNamespace(
                id=f"m{i}",
                state_vector=np.array([float(i)]),
                metadata={"content_preview": f"content {i}"},
            )
            for i in range(8)
        ]
        macro = SimpleNamespace(
            label="big_cluster",
            microstates=micros,
            average_energy=1.0,
            entropy=0.5,
            free_energy=0.5,
            order_parameter=0.8,
        )

        orch = _make_orchestrator_stub()
        shards = macrostates_to_shards(orch, [macro])
        assert "and 3 more" in shards[0].text


# ============================================================================
# Background Tasks Tests
# ============================================================================

class TestSpawnBackgroundTask:
    @pytest.mark.asyncio
    async def test_spawn_and_track(self):
        from hololoom.core.orchestrator.core.background_tasks import spawn_background_task

        orch = _make_orchestrator_stub(_background_tasks=[])

        async def dummy():
            return 42

        task = spawn_background_task(orch, dummy())
        assert task in orch._background_tasks
        result = await task
        assert result == 42

    @pytest.mark.asyncio
    async def test_cleanup_on_completion(self):
        from hololoom.core.orchestrator.core.background_tasks import spawn_background_task

        orch = _make_orchestrator_stub(_background_tasks=[])

        async def dummy():
            return 1

        task = spawn_background_task(orch, dummy())
        await task
        # Give the event loop a tick for the callback to fire
        await asyncio.sleep(0)
        assert task not in orch._background_tasks


class TestStartBackgroundConsolidation:
    def test_warns_when_stat_mech_disabled(self):
        from hololoom.core.orchestrator.core.background_tasks import start_background_consolidation

        orch = _make_orchestrator_stub(
            enable_statistical_mechanics=False,
            stat_mech_engine=None,
        )
        start_background_consolidation(orch)
        assert orch._consolidation_task is None

    def test_warns_when_already_running(self):
        from hololoom.core.orchestrator.core.background_tasks import start_background_consolidation

        orch = _make_orchestrator_stub(
            enable_statistical_mechanics=True,
            stat_mech_engine=MagicMock(),
            _consolidation_task=MagicMock(),  # already running
        )
        start_background_consolidation(orch)
        # Should not create a new task


# ============================================================================
# Metrics Collection Tests
# ============================================================================

class TestGetReflectionMetrics:
    def test_returns_none_when_disabled(self):
        from hololoom.core.orchestrator.core.metrics_collection import get_reflection_metrics

        orch = _make_orchestrator_stub(enable_reflection=False)
        assert get_reflection_metrics(orch) is None

    def test_returns_metrics_when_enabled(self):
        from hololoom.core.orchestrator.core.metrics_collection import get_reflection_metrics

        mock_buffer = MagicMock()
        mock_buffer.get_metrics.return_value = SimpleNamespace(
            total_cycles=10,
            tool_success_rates={"answer": 0.8},
            pattern_success_rates={"FAST": 0.9},
        )
        mock_buffer.get_success_rate.return_value = 0.85
        mock_buffer.get_tool_recommendations.return_value = ["use research"]

        orch = _make_orchestrator_stub(
            enable_reflection=True,
            reflection_buffer=mock_buffer,
        )
        metrics = get_reflection_metrics(orch)
        assert metrics["total_cycles"] == 10
        assert metrics["success_rate"] == 0.85


class TestGetRecursiveLearningStats:
    def test_returns_none_when_disabled(self):
        from hololoom.core.orchestrator.core.metrics_collection import get_recursive_learning_stats

        orch = _make_orchestrator_stub(enable_recursive_learning=False)
        assert get_recursive_learning_stats(orch) is None

    def test_returns_stats_when_enabled(self):
        from hololoom.core.orchestrator.core.metrics_collection import get_recursive_learning_stats

        mock_thompson = MagicMock()
        mock_thompson.tool_priors = {"answer": {"alpha": 2, "beta": 1}}
        mock_thompson.get_expected_reward.return_value = 0.7
        mock_thompson.get_uncertainty.return_value = 0.3

        mock_policy = MagicMock()
        mock_policy.adapter_weights = {"general": 1.0}
        mock_policy.adapter_total = {"general": 50}
        mock_policy.get_weight.return_value = 1.0
        mock_policy.get_success_rate.return_value = 0.8

        components = {
            "metrics": SimpleNamespace(queries_processed=100, avg_confidence=0.75),
            "thompson_priors": mock_thompson,
            "policy_weights": mock_policy,
        }

        orch = _make_orchestrator_stub(
            enable_recursive_learning=True,
            _recursive_components=components,
        )
        stats = get_recursive_learning_stats(orch)
        assert stats["queries_processed"] == 100
        assert "answer" in stats["thompson_priors"]


class TestGetMetrics:
    def test_returns_none_when_disabled(self):
        from hololoom.core.orchestrator.core.metrics_collection import get_metrics

        orch = _make_orchestrator_stub(enable_production_hardening=False)
        assert get_metrics(orch) is None

    def test_returns_metrics_when_enabled(self):
        from hololoom.core.orchestrator.core.metrics_collection import get_metrics

        mock_monitor = MagicMock()
        mock_monitor.performance.get_metrics.return_value = {"qps": 10}
        mock_monitor.resources.get_metrics.return_value = {"cpu": 0.5}
        mock_monitor.learning.get_metrics.return_value = {"reward": 0.8}

        orch = _make_orchestrator_stub(
            enable_production_hardening=True,
            monitor=mock_monitor,
        )
        metrics = get_metrics(orch)
        assert metrics["performance"]["qps"] == 10
        assert "timestamp" in metrics

    def test_handles_monitor_error(self):
        from hololoom.core.orchestrator.core.metrics_collection import get_metrics

        mock_monitor = MagicMock()
        mock_monitor.performance.get_metrics.side_effect = RuntimeError("boom")

        orch = _make_orchestrator_stub(
            enable_production_hardening=True,
            monitor=mock_monitor,
        )
        metrics = get_metrics(orch)
        assert "error" in metrics


# ============================================================================
# Multipass Retrieval Tests
# ============================================================================

class TestMultipassMemoryCrawl:
    @pytest.mark.asyncio
    async def test_returns_empty_without_memory(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import multipass_memory_crawl

        orch = _make_orchestrator_stub(memory=None)
        results = await multipass_memory_crawl(orch, _make_query(), ComplexityLevel.LITE)
        assert results == []

    @pytest.mark.asyncio
    async def test_single_pass_lite(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import multipass_memory_crawl

        mem1 = SimpleNamespace(id="m1", text="Memory 1", context={}, metadata={})
        mock_memory = AsyncMock()
        mock_memory.recall.return_value = SimpleNamespace(
            memories=[mem1],
            scores=[0.9],
        )

        orch = _make_orchestrator_stub(memory=mock_memory)
        results = await multipass_memory_crawl(orch, _make_query(), ComplexityLevel.LITE)
        assert len(results) == 1
        assert results[0].id == "m1"

    @pytest.mark.asyncio
    async def test_multipass_deduplication(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import multipass_memory_crawl

        mem1 = SimpleNamespace(id="m1", text="Memory 1", context={}, metadata={})
        mem2 = SimpleNamespace(id="m2", text="Memory 2", context={}, metadata={})

        mock_memory = AsyncMock()
        # Same item returned in both passes — should deduplicate
        mock_memory.recall.side_effect = [
            SimpleNamespace(memories=[mem1, mem2], scores=[0.9, 0.7]),
            SimpleNamespace(memories=[mem1], scores=[0.8]),  # duplicate
        ]

        orch = _make_orchestrator_stub(memory=mock_memory)
        results = await multipass_memory_crawl(orch, _make_query(), ComplexityLevel.FAST)
        # Should have 2 unique items, not 3
        ids = [r.id for r in results]
        assert len(ids) == 2
        assert "m1" in ids
        assert "m2" in ids

    @pytest.mark.asyncio
    async def test_trace_records_events(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import multipass_memory_crawl

        mem1 = SimpleNamespace(id="m1", text="Mem", context={}, metadata={})
        mock_memory = AsyncMock()
        mock_memory.recall.return_value = SimpleNamespace(memories=[mem1], scores=[0.8])

        orch = _make_orchestrator_stub(memory=mock_memory)
        trace = ProvenanceTrace(
            operation_id="test",
            complexity_level=ComplexityLevel.LITE,
            start_time=time.perf_counter(),
        )
        await multipass_memory_crawl(orch, _make_query(), ComplexityLevel.LITE, trace=trace)
        event_types = [e["event_type"] for e in trace.shuttle_events]
        assert "crawl_start" in event_types
        assert "crawl_complete" in event_types

    @pytest.mark.asyncio
    async def test_handles_recall_failure(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import multipass_memory_crawl

        mock_memory = AsyncMock()
        mock_memory.recall.side_effect = RuntimeError("connection lost")

        orch = _make_orchestrator_stub(memory=mock_memory)
        # Should not raise, just return empty
        results = await multipass_memory_crawl(orch, _make_query(), ComplexityLevel.LITE)
        assert results == []

    @pytest.mark.asyncio
    async def test_graph_expansion(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import multipass_memory_crawl

        mem1 = SimpleNamespace(id="m1", text="Mem 1", context={}, metadata={})
        related = SimpleNamespace(id="r1", text="Related", context={}, metadata={})

        mock_memory = AsyncMock()
        mock_memory.recall.side_effect = [
            SimpleNamespace(memories=[mem1], scores=[0.9]),
            SimpleNamespace(memories=[], scores=[]),
        ]
        mock_memory.get_related = AsyncMock(return_value=[related])

        orch = _make_orchestrator_stub(memory=mock_memory)
        results = await multipass_memory_crawl(orch, _make_query(), ComplexityLevel.FAST)
        ids = [r.id for r in results]
        assert "m1" in ids
        assert "r1" in ids


class TestQueryMemoryBackend:
    @pytest.mark.asyncio
    async def test_returns_empty_without_memory(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import query_memory_backend

        orch = _make_orchestrator_stub(memory=None)
        result = await query_memory_backend(orch, "test")
        assert result == []

    @pytest.mark.asyncio
    async def test_converts_to_shards(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import query_memory_backend

        mem = SimpleNamespace(
            id="m1",
            text="Memory text",
            context={"episode": "ep1", "entities": ["e1"]},
            metadata={"motifs": ["question"]},
        )
        mock_memory = AsyncMock()
        mock_memory.recall.return_value = SimpleNamespace(memories=[mem])

        orch = _make_orchestrator_stub(memory=mock_memory)
        shards = await query_memory_backend(orch, "test", limit=5)
        assert len(shards) == 1
        assert shards[0].id == "m1"
        assert shards[0].episode == "ep1"

    @pytest.mark.asyncio
    async def test_handles_error(self):
        from hololoom.core.orchestrator.retrieval.multipass_retrieval import query_memory_backend

        mock_memory = AsyncMock()
        mock_memory.recall.side_effect = RuntimeError("fail")

        orch = _make_orchestrator_stub(memory=mock_memory)
        result = await query_memory_backend(orch, "test")
        assert result == []
