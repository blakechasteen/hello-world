"""
Unit tests for hololoom/core/loom/ — targeting 60%+ coverage.

Covers:
- DreamingMixin: insight logging, sharing, receiving, integration
- BaseLoom: Thompson Sampling, metrics, verification, refinement
- DreamOrchestrator: bleed rates, dream cycles, distribution
- WeaveHouse: collective blind spots, admits_ignorance, falsifiable_by,
              fabric creation, tension exploration, loom management
- PatternCard / card_loader: from_dict, to_dict, merge_configs, tools config
- DreamCycleStats / DreamOrchestratorMetrics: serialization, cumulative tracking

Date: 2026-03-18
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from hololoom.loom.protocol import (
    DreamInsight,
    InsightType,
    MathInsight,
    PatternInsight,
    CorrectionInsight,
    DiscoveryInsight,
    RECALL,
    REASON,
    REACH,
    REFLECT,
    REFUSE,
    COLLECTIVE,
)
from hololoom.loom.base_loom import DreamingMixin, BaseLoom
from hololoom.loom.dreaming import (
    DreamOrchestrator,
    DreamCycleStats,
    DreamOrchestratorMetrics,
    create_dream_orchestrator,
)
from hololoom.loom.weave_house import (
    WeaveHouse,
    WeaveResult,
    ExplorationResult,
    create_weave_house,
)
from hololoom.loom.card_loader import (
    PatternCard,
    MathCapabilities,
    MemoryConfig,
    ToolsConfig,
    PerformanceProfile,
)
from hololoom.loom.consensus import LoomConsensus, Tension
from hololoom.fabric.fabric import Fabric, Resolution
from hololoom.fabric.spacetime import WeavingTrace
from hololoom.protocols.department import (
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_trace() -> WeavingTrace:
    now = datetime.now()
    return WeavingTrace(start_time=now, end_time=now, duration_ms=10.0)


def _make_fabric(
    perspective: str = "test",
    response: str = "Test response",
    confidence: float = 0.8,
    blind_spots: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Fabric:
    return Fabric(
        query_text="test query",
        response=response,
        tool_used="answer",
        confidence=confidence,
        trace=_make_trace(),
        perspective=perspective,
        blind_spots=blind_spots or [],
        epistemic_confidence=confidence * 0.9,
        falsifiable_by=["contradicting evidence"],
        admits_ignorance=["unknown domains"],
        tensions_with={},
        agreement_with={},
        metadata=metadata or {},
    )


def _make_request(query: str = "test query") -> DepartmentRequest:
    req = DepartmentRequest()
    req.query = query
    return req


def _make_response(
    result: str = "answer",
    score: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None,
) -> DepartmentResponse:
    return DepartmentResponse(
        task_id=uuid4(),
        result=result,
        confidence=ConfidenceMetadata.from_score(score),
        metadata=metadata or {},
    )


class StubLoom(BaseLoom):
    """Minimal concrete BaseLoom for testing."""

    def __init__(self, name: str = "stub", blind: Optional[List[str]] = None, **kwargs):
        self._perspective = name
        self._blind = blind or ["unknown things"]
        super().__init__(**kwargs)

    @property
    def perspective(self) -> str:
        return self._perspective

    @property
    def blind_spots(self) -> List[str]:
        return self._blind

    def admits_ignorance(self, query: str) -> List[str]:
        return [f"I don't know about {query}"]

    def falsifiable_by(self, claim: str) -> List[str]:
        return [f"Evidence against {claim}"]

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        return _make_response(
            result=f"[{self._perspective}] answer",
            score=0.85,
            metadata={"fabric": _make_fabric(perspective=self._perspective)},
        )


# ============================================================================
# DreamingMixin Tests
# ============================================================================


class TestDreamingMixin:
    """Tests for DreamingMixin insight lifecycle."""

    def test_init_dreaming_state(self):
        loom = StubLoom()
        assert loom._dream_buffer == []
        assert loom._received_insights == []
        assert loom._insights_shared == 0

    def test_log_insight_adds_to_buffer(self):
        loom = StubLoom()
        insight = MathInsight(source_loom="stub", content="pi=3.14", confidence=0.9)
        loom.log_insight(insight)
        assert len(loom._dream_buffer) == 1
        assert loom._dream_buffer[0].insight_type == InsightType.MATH

    def test_log_math_insight_convenience(self):
        loom = StubLoom()
        loom.log_math_insight(content="eigenvalue", confidence=0.85)
        assert len(loom._dream_buffer) == 1
        assert loom._dream_buffer[0].insight_type == InsightType.MATH
        assert loom._dream_buffer[0].confidence == 0.85

    def test_log_pattern_insight_convenience(self):
        loom = StubLoom()
        loom.log_pattern_insight(content="tool_selection", confidence=0.7)
        assert len(loom._dream_buffer) == 1
        assert loom._dream_buffer[0].insight_type == InsightType.PATTERN

    def test_log_correction_insight_convenience(self):
        loom = StubLoom()
        loom.log_correction_insight(content="wrong claim", confidence=0.9)
        assert len(loom._dream_buffer) == 1
        assert loom._dream_buffer[0].insight_type == InsightType.CORRECTION

    def test_log_discovery_insight_convenience(self):
        loom = StubLoom()
        loom.log_discovery_insight(content="new finding", confidence=0.85)
        assert len(loom._dream_buffer) == 1
        assert loom._dream_buffer[0].insight_type == InsightType.DISCOVERY

    @pytest.mark.asyncio
    async def test_dream_share_returns_shareable_and_clears_buffer(self):
        loom = StubLoom()
        loom.log_math_insight("a", 0.8)
        loom.log_correction_insight("b", 0.9)

        # Add a non-shareable insight
        non_share = DreamInsight(
            source_loom="stub",
            insight_type=InsightType.PATTERN,
            content="c",
            confidence=0.5,
            shareable=False,
        )
        loom._dream_buffer.append(non_share)

        shared = await loom.dream_share()
        # Only shareable ones returned
        assert len(shared) == 2
        assert loom._dream_buffer == []
        assert loom._insights_shared == 2

    @pytest.mark.asyncio
    async def test_dream_receive_correction_always_integrates(self):
        loom = StubLoom()
        correction = CorrectionInsight(
            source_loom="other", content="fix this", confidence=0.95
        )
        await loom.dream_receive(correction)
        assert loom._insights_received == 1
        assert loom._insights_integrated == 1
        assert hasattr(loom, "_corrections")
        assert len(loom._corrections) == 1

    @pytest.mark.asyncio
    async def test_dream_receive_math_integrates_probabilistically(self):
        loom = StubLoom()
        loom._math_integration_rate = 1.0  # force integration
        math_insight = MathInsight(
            source_loom="other", content="tensor field", confidence=0.8
        )
        await loom.dream_receive(math_insight)
        assert loom._insights_integrated == 1
        assert hasattr(loom, "_warp_priors")

    @pytest.mark.asyncio
    async def test_dream_receive_math_skips_when_rate_zero(self):
        loom = StubLoom()
        loom._math_integration_rate = 0.0  # never integrate
        math_insight = MathInsight(
            source_loom="other", content="tensor field", confidence=0.8
        )
        await loom.dream_receive(math_insight)
        assert loom._insights_integrated == 0

    @pytest.mark.asyncio
    async def test_dream_receive_discovery_high_confidence_integrates(self):
        loom = StubLoom()
        loom._discovery_confidence_threshold = 0.7
        disc = DiscoveryInsight(
            source_loom="other", content="novel connection", confidence=0.9
        )
        await loom.dream_receive(disc)
        assert loom._insights_integrated == 1
        assert hasattr(loom, "_discoveries")

    @pytest.mark.asyncio
    async def test_dream_receive_discovery_low_confidence_skips(self):
        loom = StubLoom()
        loom._discovery_confidence_threshold = 0.9
        disc = DiscoveryInsight(
            source_loom="other", content="maybe", confidence=0.5
        )
        await loom.dream_receive(disc)
        assert loom._insights_integrated == 0

    @pytest.mark.asyncio
    async def test_pattern_integration_with_forced_rate(self):
        loom = StubLoom()
        loom._pattern_integration_rate = 1.0  # always integrate
        pat = PatternInsight(
            source_loom="other", content="useful pattern", confidence=0.7
        )
        await loom.dream_receive(pat)
        assert loom._insights_integrated == 1
        assert hasattr(loom, "_learned_patterns")

    def test_get_dreaming_stats(self):
        loom = StubLoom()
        loom.log_math_insight("x", 0.8)
        stats = loom.get_dreaming_stats()
        assert stats["buffer_size"] == 1
        assert stats["insights_shared"] == 0
        assert stats["insights_received"] == 0


# ============================================================================
# BaseLoom Tests
# ============================================================================


class TestBaseLoom:
    """Tests for BaseLoom: Thompson Sampling, metrics, verification."""

    def test_init_defaults(self):
        loom = StubLoom()
        assert loom._enable_learning is True
        assert loom._enable_verification is True
        assert loom._tasks_executed == 0
        assert loom._tool_priors == {}

    def test_get_expected_reward_uniform_prior(self):
        loom = StubLoom()
        assert loom.get_expected_reward("unknown_tool") == 0.5

    def test_get_expected_reward_after_update(self):
        loom = StubLoom()
        loom._tool_priors["search"] = (5.0, 1.0)
        reward = loom.get_expected_reward("search")
        assert reward == pytest.approx(5.0 / 6.0, abs=0.01)

    def test_sample_tool_returns_from_candidates(self):
        loom = StubLoom()
        # With uniform priors, any tool could win
        result = loom.sample_tool(["a", "b", "c"])
        assert result in ["a", "b", "c"]

    def test_sample_tool_biased_priors(self):
        loom = StubLoom()
        loom._tool_priors["strong"] = (100.0, 1.0)
        loom._tool_priors["weak"] = (1.0, 100.0)
        # Strong tool should almost always win
        wins = sum(1 for _ in range(50) if loom.sample_tool(["strong", "weak"]) == "strong")
        assert wins >= 45

    @pytest.mark.asyncio
    async def test_update_strategy_success(self):
        loom = StubLoom()
        await loom.update_strategy({
            "outcome": "success",
            "confidence": 0.9,
            "tool_used": "search",
        })
        alpha, beta = loom._tool_priors["search"]
        assert alpha == 1.0 + 0.9  # initial 1.0 + confidence
        assert beta == 1.0  # unchanged

    @pytest.mark.asyncio
    async def test_update_strategy_failure(self):
        loom = StubLoom()
        await loom.update_strategy({
            "outcome": "failure",
            "confidence": 0.3,
            "tool_used": "search",
        })
        alpha, beta = loom._tool_priors["search"]
        assert alpha == 1.0  # unchanged
        assert beta == pytest.approx(1.0 + 0.7, abs=0.01)  # 1 - 0.3

    @pytest.mark.asyncio
    async def test_update_strategy_learning_disabled(self):
        loom = StubLoom(enable_learning=False)
        await loom.update_strategy({"outcome": "success", "confidence": 0.9})
        assert loom._tool_priors == {}

    @pytest.mark.asyncio
    async def test_verify_passes_with_good_response(self):
        loom = StubLoom()
        response = _make_response(result="good answer", score=0.8)
        result = await loom.verify(response)
        assert result.verified is True
        assert result.overall_score > 0.5
        assert loom._verification_passes == 1

    @pytest.mark.asyncio
    async def test_verify_fails_with_empty_result(self):
        loom = StubLoom()
        response = _make_response(result="", score=0.8)
        result = await loom.verify(response)
        assert result.verified is False
        assert loom._verification_fails == 1

    @pytest.mark.asyncio
    async def test_verify_fails_with_low_confidence(self):
        loom = StubLoom()
        response = _make_response(result="ok", score=0.3)
        result = await loom.verify(response)
        assert result.verified is False

    @pytest.mark.asyncio
    async def test_refine_returns_original_when_confident(self):
        loom = StubLoom()
        response = _make_response(result="great", score=0.9)
        refined = await loom.refine(response)
        assert refined.result == "great"

    @pytest.mark.asyncio
    async def test_refine_returns_original_when_low_confidence(self):
        # Default implementation returns as-is regardless
        loom = StubLoom()
        response = _make_response(result="ok", score=0.3)
        refined = await loom.refine(response)
        assert refined.result == "ok"

    @pytest.mark.asyncio
    async def test_get_capabilities_with_default_name(self):
        loom = StubLoom()
        caps = await loom.get_capabilities()
        assert caps["perspective"] == "stub"
        # When no BaseLoom name provided, falls back to perspective
        assert caps["name"] == "stub"
        assert "weave" in caps["supported_tasks"]

    @pytest.mark.asyncio
    async def test_get_metrics_initial(self):
        loom = StubLoom()
        metrics = await loom.get_metrics()
        assert metrics["tasks_executed"] == 0
        assert metrics["avg_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_after_execution(self):
        loom = StubLoom()
        loom._update_metrics(confidence=0.8, latency_ms=100.0)
        loom._update_metrics(confidence=0.6, latency_ms=200.0)
        metrics = await loom.get_metrics()
        assert metrics["tasks_executed"] == 2
        assert metrics["avg_confidence"] == pytest.approx(0.7)
        assert metrics["avg_latency_ms"] == pytest.approx(150.0)

    @pytest.mark.asyncio
    async def test_health_check(self):
        loom = StubLoom()
        assert await loom.health_check() is True

    @pytest.mark.asyncio
    async def test_execute_returns_response(self):
        loom = StubLoom()
        req = _make_request("What is TS?")
        resp = await loom.execute(req)
        assert "[stub] answer" in resp.result


# ============================================================================
# DreamCycleStats & DreamOrchestratorMetrics Tests
# ============================================================================


class TestDreamingDataclasses:
    """Tests for dreaming statistics dataclasses."""

    def test_dream_cycle_stats_to_dict(self):
        stats = DreamCycleStats(
            cycle_number=1,
            timestamp=datetime(2025, 1, 1),
            total_insights_collected=5,
            math_shared=2,
            patterns_shared=1,
            corrections_shared=1,
            discoveries_shared=1,
        )
        d = stats.to_dict()
        assert d["cycle_number"] == 1
        assert d["total_insights_collected"] == 5
        assert d["math_shared"] == 2

    def test_metrics_add_cycle(self):
        metrics = DreamOrchestratorMetrics()
        stats = DreamCycleStats(
            cycle_number=1,
            timestamp=datetime.now(),
            total_insights_collected=10,
            math_shared=3,
            patterns_shared=2,
            corrections_shared=4,
            discoveries_shared=1,
            duration_ms=50.0,
        )
        metrics.add_cycle(stats)
        assert metrics.total_cycles == 1
        assert metrics.total_insights_processed == 10
        assert metrics.total_insights_shared == 10
        assert metrics.math_insights_shared == 3
        assert metrics.avg_cycle_duration_ms == pytest.approx(50.0)

    def test_metrics_add_multiple_cycles(self):
        metrics = DreamOrchestratorMetrics()
        for i in range(12):
            stats = DreamCycleStats(
                cycle_number=i + 1,
                timestamp=datetime.now(),
                duration_ms=10.0,
            )
            metrics.add_cycle(stats)
        assert metrics.total_cycles == 12
        # Only last 10 cycles kept
        assert len(metrics.recent_cycles) == 10

    def test_metrics_to_dict(self):
        metrics = DreamOrchestratorMetrics()
        d = metrics.to_dict()
        assert d["total_cycles"] == 0
        assert "recent_cycles" in d


# ============================================================================
# DreamOrchestrator Tests
# ============================================================================


class TestDreamOrchestrator:
    """Tests for collective dreaming orchestration."""

    def _make_looms(self, n=3):
        return [StubLoom(name=f"loom_{i}") for i in range(n)]

    def test_init(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms)
        assert len(orch.looms) == 3
        assert orch.math_bleed_rate == 0.3
        assert orch.pattern_bleed_rate == 0.2
        assert orch._running is False

    @pytest.mark.asyncio
    async def test_dream_once_empty_buffers(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms)
        stats = await orch.dream_once()
        assert stats.cycle_number == 1
        assert stats.total_insights_collected == 0

    @pytest.mark.asyncio
    async def test_dream_once_with_corrections(self):
        looms = self._make_looms()
        # Loom 0 logs a correction
        looms[0].log_correction_insight("error found", 0.95)
        orch = DreamOrchestrator(looms=looms)
        stats = await orch.dream_once()
        assert stats.total_insights_collected == 1
        assert stats.corrections_shared == 1
        # Correction should be distributed to other looms (not source)
        assert stats.successful_integrations >= 1

    @pytest.mark.asyncio
    async def test_dream_once_with_math_bleed(self):
        looms = self._make_looms()
        # Log several math insights
        for i in range(10):
            looms[0].log_math_insight(f"math_{i}", 0.8)

        orch = DreamOrchestrator(looms=looms, math_bleed_rate=0.5)
        stats = await orch.dream_once()
        assert stats.total_insights_collected == 10
        # At least some math should be shared (bleed 50%)
        assert stats.math_shared >= 1

    @pytest.mark.asyncio
    async def test_dream_once_discovery_confidence_filter(self):
        looms = self._make_looms()
        # High-confidence discovery -> shared
        looms[0].log_discovery_insight("good find", 0.95)
        # Low-confidence discovery -> NOT shared by orchestrator
        looms[1].log_discovery_insight("uncertain", 0.5)

        orch = DreamOrchestrator(
            looms=looms, discovery_confidence_threshold=0.8
        )
        stats = await orch.dream_once()
        # Low-confidence discovery (0.5) is not shareable at the insight level
        # (DiscoveryInsight sets shareable=False if confidence < 0.8)
        # So it won't even be collected
        assert stats.discoveries_shared <= 1

    @pytest.mark.asyncio
    async def test_dream_once_max_insights_cap(self):
        looms = self._make_looms()
        # Log many corrections to exceed cap
        for i in range(200):
            looms[0].log_correction_insight(f"fix_{i}", 0.9)

        orch = DreamOrchestrator(looms=looms, max_insights_per_cycle=10)
        stats = await orch.dream_once()
        assert stats.insights_distributed <= 10

    @pytest.mark.asyncio
    async def test_dream_once_callback(self):
        looms = self._make_looms()
        callback_calls = []
        orch = DreamOrchestrator(
            looms=looms,
            on_cycle_complete=lambda s: callback_calls.append(s),
        )
        await orch.dream_once()
        assert len(callback_calls) == 1

    @pytest.mark.asyncio
    async def test_dream_once_callback_error_doesnt_crash(self):
        looms = self._make_looms()

        def bad_callback(s):
            raise ValueError("boom")

        orch = DreamOrchestrator(looms=looms, on_cycle_complete=bad_callback)
        # Should not raise
        stats = await orch.dream_once()
        assert stats.cycle_number == 1

    def test_get_metrics(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms)
        m = orch.get_metrics()
        assert m["running"] is False
        assert m["cycle_count"] == 0
        assert len(m["looms"]) == 3

    def test_reset_metrics(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms)
        orch._cycle_count = 5
        orch.reset_metrics()
        assert orch._cycle_count == 0

    @pytest.mark.asyncio
    async def test_sample_insights_empty(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms)
        result = orch._sample_insights([], 0.5)
        assert result == []

    @pytest.mark.asyncio
    async def test_sample_insights_zero_confidence(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms)
        insights = [
            MathInsight(source_loom="x", content="a", confidence=0.0),
            MathInsight(source_loom="x", content="b", confidence=0.0),
        ]
        result = orch._sample_insights(insights, 1.0)
        # With zero confidence, should use uniform sampling
        assert len(result) >= 1

    def test_create_dream_orchestrator_factory(self):
        looms = self._make_looms()
        orch = create_dream_orchestrator(
            looms=looms,
            math_bleed_rate=0.4,
            pattern_bleed_rate=0.1,
        )
        assert isinstance(orch, DreamOrchestrator)
        assert orch.math_bleed_rate == 0.4

    @pytest.mark.asyncio
    async def test_start_stop_dreaming(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms, consolidation_interval=100)
        await orch.start_dreaming()
        assert orch._running is True
        await orch.stop_dreaming()
        assert orch._running is False

    @pytest.mark.asyncio
    async def test_start_dreaming_idempotent(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms, consolidation_interval=100)
        await orch.start_dreaming()
        task1 = orch._task
        await orch.start_dreaming()  # Should not create new task
        assert orch._task is task1
        await orch.stop_dreaming()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        looms = self._make_looms()
        orch = DreamOrchestrator(looms=looms)
        # Should not raise
        await orch.stop_dreaming()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        looms = self._make_looms()
        async with DreamOrchestrator(
            looms=looms, consolidation_interval=100
        ) as orch:
            assert orch._running is True
        assert orch._running is False

    @pytest.mark.asyncio
    async def test_distribute_excludes_source_loom(self):
        looms = self._make_looms(2)
        orch = DreamOrchestrator(looms=looms)
        insight = CorrectionInsight(
            source_loom="loom_0", content="fix", confidence=0.9
        )
        count = await orch._distribute_insights([insight])
        # Should deliver to loom_1 only (not loom_0)
        assert count == 1


# ============================================================================
# WeaveHouse Tests
# ============================================================================


class TestWeaveHouse:
    """Tests for WeaveHouse collective orchestration."""

    def _make_house(self, n_looms=3, **kwargs):
        looms = [StubLoom(name=f"loom_{i}", blind=["shared_blind"]) for i in range(n_looms)]
        return WeaveHouse(looms=looms, **kwargs)

    def test_perspective_is_collective(self):
        house = self._make_house()
        assert house.perspective == COLLECTIVE

    def test_collective_blind_spots_intersection(self):
        loom_a = StubLoom(name="a", blind=["blind_1", "blind_2"])
        loom_b = StubLoom(name="b", blind=["blind_2", "blind_3"])
        house = WeaveHouse(looms=[loom_a, loom_b])
        assert "blind_2" in house.blind_spots
        assert "blind_1" not in house.blind_spots

    def test_collective_blind_spots_empty_looms(self):
        house = WeaveHouse(looms=[])
        assert house.blind_spots == []

    def test_admits_ignorance_aggregates(self):
        house = self._make_house(2)
        ignorance = house.admits_ignorance("test query")
        # Each loom contributes, plus collective note for <3 looms
        assert len(ignorance) >= 2
        assert any("[collective]" in i for i in ignorance)

    def test_admits_ignorance_no_collective_note_with_3_plus(self):
        house = self._make_house(3)
        ignorance = house.admits_ignorance("test")
        assert not any("[collective]" in i for i in ignorance)

    def test_falsifiable_by_aggregates(self):
        house = self._make_house(2)
        falsifiers = house.falsifiable_by("some claim")
        assert len(falsifiers) >= 2
        assert all("[" in f for f in falsifiers)  # Has loom prefix

    def test_get_loom(self):
        house = self._make_house(3)
        found = house.get_loom("loom_1")
        assert found is not None
        assert found.perspective == "loom_1"

    def test_get_loom_not_found(self):
        house = self._make_house()
        assert house.get_loom("nonexistent") is None

    def test_add_loom(self):
        house = self._make_house(2)
        new_loom = StubLoom(name="new", blind=["new_blind"])
        house.add_loom(new_loom)
        assert len(house.looms) == 3
        assert house.get_loom("new") is not None

    def test_remove_loom(self):
        house = self._make_house(3)
        removed = house.remove_loom("loom_1")
        assert removed is True
        assert len(house.looms) == 2
        assert house.get_loom("loom_1") is None

    def test_remove_loom_not_found(self):
        house = self._make_house()
        assert house.remove_loom("nonexistent") is False

    @pytest.mark.asyncio
    async def test_execute_parallel_hits_fabric_bug(self):
        """execute() hits _create_collective_fabric which constructs Fabric
        without required Spacetime args (query_text, tool_used, trace).
        Documents latent bug in weave_house.py."""
        house = self._make_house(3)
        req = _make_request("What is TS?")
        with pytest.raises(TypeError):
            await house.execute(req)

    @pytest.mark.asyncio
    async def test_execute_sequential_hits_fabric_bug(self):
        """Same latent bug path as parallel execution."""
        house = self._make_house(2, enable_parallel=False)
        req = _make_request("test")
        with pytest.raises(TypeError):
            await house.execute(req)

    @pytest.mark.asyncio
    async def test_execute_all_looms_fail(self):
        """When all looms raise, WeaveHouse tries to return error response.
        Due to a latent bug in _create_error_response (invalid ConfidenceMetadata kwargs),
        this currently raises TypeError."""

        class FailLoom(StubLoom):
            async def execute(self, request):
                raise RuntimeError("loom failed")

        looms = [FailLoom(name=f"fail_{i}") for i in range(2)]
        house = WeaveHouse(looms=looms)
        req = _make_request("test")
        with pytest.raises(TypeError):
            await house.execute(req)

    def test_generate_tension_query(self):
        house = self._make_house()
        tension = Tension(
            loom_a="recall",
            loom_b="reason",
            claim_a="Memory says X",
            claim_b="Logic says Y",
            severity=0.7,
        )
        query = house._generate_tension_query(tension)
        assert "Memory says X" in query
        assert "Logic says Y" in query
        assert "disagreement" in query.lower() or "Resolve" in query

    def test_get_disagreeing_looms(self):
        looms = [StubLoom(name="recall"), StubLoom(name="reason"), StubLoom(name="reach")]
        house = WeaveHouse(looms=looms)
        tension = Tension(
            loom_a="recall", loom_b="reason",
            claim_a="X", claim_b="Y", severity=0.5,
        )
        disagreeing = house._get_disagreeing_looms(tension, [])
        assert len(disagreeing) == 2
        perspectives = {l.perspective for l in disagreeing}
        assert perspectives == {"recall", "reason"}

    def test_create_fabric_from_response_missing_args(self):
        """_create_fabric_from_response constructs Fabric without query_text/tool_used/trace.
        Documents the same latent Fabric constructor bug."""
        house = self._make_house()
        resp = _make_response(result="test result", score=0.7)
        with pytest.raises(TypeError):
            house._create_fabric_from_response(resp, "test_perspective")

    def test_incorporate_resolution_missing_fabric_args(self):
        """_incorporate_resolution constructs Fabric without query_text/tool_used/trace.
        This documents a latent bug in weave_house.py (missing required Spacetime args)."""
        house = self._make_house()
        synthesis = _make_fabric(
            perspective=COLLECTIVE,
            response="Original",
            confidence=0.7,
            metadata={"tensions": []},
        )
        resolution = Resolution(
            resolved=True,
            consensus_claim="Both X and Y are true",
            confidence=0.85,
            supporting_looms=["recall", "reason"],
        )
        with pytest.raises(TypeError):
            house._incorporate_resolution(synthesis, resolution)

    def test_mark_irreducible_missing_fabric_args(self):
        """_mark_irreducible constructs Fabric without query_text/tool_used/trace.
        This documents a latent bug in weave_house.py (missing required Spacetime args)."""
        house = self._make_house()
        synthesis = _make_fabric(
            perspective=COLLECTIVE,
            confidence=0.7,
            metadata={},
        )
        tension = Tension(
            loom_a="recall", loom_b="reason",
            claim_a="X", claim_b="Y", severity=0.5,
        )
        with pytest.raises(TypeError):
            house._mark_irreducible(synthesis, tension)

    def test_create_error_response_missing_confidence_level(self):
        """_create_error_response constructs ConfidenceMetadata with invalid kwargs.
        This documents a latent bug in weave_house.py."""
        house = self._make_house()
        with pytest.raises(TypeError):
            house._create_error_response("all failed", ["a", "b"], datetime.now())

    @pytest.mark.asyncio
    async def test_dream_share_includes_exploration_insights(self):
        house = self._make_house()
        # Simulate resolved exploration
        tension = Tension(
            loom_a="a", loom_b="b",
            claim_a="X", claim_b="Y", severity=0.5,
        )
        resolution = Resolution(
            resolved=True,
            consensus_claim="Resolved!",
            confidence=0.9,
            supporting_looms=["a", "b"],
        )
        house._exploration_history.append(
            ExplorationResult(
                tension=tension,
                resolution=resolution,
                exploration_fabrics=[],
                depth_reached=1,
                resolved=True,
            )
        )
        insights = await house.dream_share()
        # Should include exploration-derived insights
        # Note: WeaveHouse uses string "discovery" for insight_type, not InsightType enum
        assert any(
            isinstance(i.content, dict) and i.content.get("type") == "tension_resolution"
            for i in insights
        )

    @pytest.mark.asyncio
    async def test_dream_share_unresolved_exploration(self):
        house = self._make_house()
        tension = Tension(
            loom_a="a", loom_b="b",
            claim_a="X", claim_b="Y", severity=0.5,
        )
        resolution = Resolution(
            resolved=False, consensus_claim=None, confidence=0.3,
        )
        house._exploration_history.append(
            ExplorationResult(
                tension=tension,
                resolution=resolution,
                exploration_fabrics=[],
                depth_reached=1,
                resolved=False,
            )
        )
        insights = await house.dream_share()
        assert any(
            isinstance(i.content, dict) and i.content.get("type") == "irreducible_tension"
            for i in insights
        )

    @pytest.mark.asyncio
    async def test_dream_receive_forwards_to_member_looms(self):
        looms = [StubLoom(name="a"), StubLoom(name="b")]
        house = WeaveHouse(looms=looms)
        insight = CorrectionInsight(
            source_loom="external", content="fix", confidence=0.9
        )
        await house.dream_receive(insight)
        # Both looms should have received the insight
        assert looms[0]._insights_received >= 1
        assert looms[1]._insights_received >= 1


# ============================================================================
# PatternCard / card_loader Tests
# ============================================================================


class TestCardLoader:
    """Tests for PatternCard configuration system."""

    def test_from_dict_minimal(self):
        card = PatternCard.from_dict({"name": "test"})
        assert card.name == "test"
        assert card.version == "1.0"
        assert card.extends is None

    def test_from_dict_full(self):
        data = {
            "name": "research",
            "display_name": "Research Mode",
            "description": "Deep exploration",
            "version": "2.0",
            "math": {"semantic_calculus": {"enabled": True}},
            "memory": {"backend": "neo4j"},
            "tools": {"enabled": ["search", "compute"], "disabled": ["web"]},
            "performance": {"target_latency_ms": 500},
            "extensions": {"custom": True},
            "extends": "base",
        }
        card = PatternCard.from_dict(data)
        assert card.display_name == "Research Mode"
        assert card.memory_config.backend == "neo4j"
        assert card.tools_config.is_tool_enabled("search") is True
        assert card.tools_config.is_tool_enabled("web") is False
        assert card.tools_config.is_tool_enabled("unknown") is False
        assert card.performance_profile.target_latency_ms == 500
        assert card.extends == "base"

    def test_to_dict_roundtrip(self):
        original = PatternCard(
            name="test",
            display_name="Test Card",
            description="For testing",
            version="1.5",
        )
        d = original.to_dict()
        reconstructed = PatternCard.from_dict(d)
        assert reconstructed.name == "test"
        assert reconstructed.display_name == "Test Card"
        assert reconstructed.version == "1.5"

    def test_merge_configs_simple(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        merged = PatternCard._merge_configs(base, override)
        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_merge_configs_nested(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        merged = PatternCard._merge_configs(base, override)
        assert merged["x"] == {"a": 1, "b": 3, "c": 4}

    def test_merge_configs_override_non_dict_with_dict(self):
        base = {"x": "string"}
        override = {"x": {"nested": True}}
        merged = PatternCard._merge_configs(base, override)
        assert merged["x"] == {"nested": True}

    def test_math_capabilities_is_enabled(self):
        caps = MathCapabilities(
            semantic_calculus={"enabled": True},
            spectral_embedding={"enabled": False},
        )
        assert caps.is_enabled("semantic_calculus") is True
        assert caps.is_enabled("spectral_embedding") is False
        assert caps.is_enabled("nonexistent") is False

    def test_tools_config_get_tool_config(self):
        tools = ToolsConfig(
            enabled=["search"],
            configs={"search": {"max_results": 10}},
        )
        assert tools.get_tool_config("search") == {"max_results": 10}
        assert tools.get_tool_config("disabled_tool") is None

    def test_tools_config_disabled_overrides_enabled(self):
        tools = ToolsConfig(
            enabled=["search"],
            disabled=["search"],
        )
        assert tools.is_tool_enabled("search") is False

    def test_performance_profile_defaults(self):
        perf = PerformanceProfile()
        assert perf.target_latency_ms == 200
        assert perf.max_latency_ms == 500
        assert perf.timeout_ms == 2000

    def test_save_and_load(self, tmp_path):
        card = PatternCard(
            name="test_save",
            display_name="Save Test",
            description="Testing save/load",
        )
        card.save(cards_dir=tmp_path)
        loaded = PatternCard.load("test_save", cards_dir=tmp_path)
        assert loaded.name == "test_save"
        assert loaded.display_name == "Save Test"

    def test_load_with_overrides(self, tmp_path):
        card = PatternCard(name="base_card", display_name="Base")
        card.save(cards_dir=tmp_path)
        loaded = PatternCard.load(
            "base_card",
            cards_dir=tmp_path,
            overrides={"display_name": "Overridden"},
        )
        assert loaded.display_name == "Overridden"

    def test_load_with_inheritance(self, tmp_path):
        parent = PatternCard(
            name="parent",
            display_name="Parent",
            description="Parent card",
        )
        parent.save(cards_dir=tmp_path)

        # Create child card YAML manually
        import yaml

        child_data = {
            "name": "child",
            "display_name": "Child",
            "extends": "parent",
        }
        child_path = tmp_path / "child.yaml"
        with open(child_path, "w") as f:
            yaml.dump(child_data, f)

        loaded = PatternCard.load("child", cards_dir=tmp_path)
        assert loaded.display_name == "Child"
        # Should inherit parent's description
        assert loaded.description == "Parent card"

    def test_load_missing_card_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            PatternCard.load("nonexistent", cards_dir=tmp_path)

    def test_repr(self):
        card = PatternCard(name="test", display_name="Test", version="1.0")
        r = repr(card)
        assert "test" in r
        assert "Test" in r


# ============================================================================
# Protocol / Insight Types Tests
# ============================================================================


class TestInsightTypes:
    """Tests for insight type specializations."""

    def test_math_insight_type(self):
        m = MathInsight(source_loom="x", content="data", confidence=0.8)
        assert m.insight_type == InsightType.MATH
        assert m.shareable is True

    def test_pattern_insight_type(self):
        p = PatternInsight(source_loom="x", content="data", confidence=0.7)
        assert p.insight_type == InsightType.PATTERN
        assert p.shareable is True

    def test_correction_always_shareable(self):
        c = CorrectionInsight(source_loom="x", content="fix", confidence=0.9)
        assert c.insight_type == InsightType.CORRECTION
        assert c.shareable is True

    def test_discovery_shareable_when_confident(self):
        d = DiscoveryInsight(source_loom="x", content="find", confidence=0.9)
        assert d.shareable is True

    def test_discovery_not_shareable_when_low_confidence(self):
        d = DiscoveryInsight(source_loom="x", content="maybe", confidence=0.5)
        assert d.shareable is False

    def test_dream_insight_to_dict(self):
        insight = MathInsight(source_loom="recall", content="pi", confidence=0.8)
        d = insight.to_dict()
        assert d["source_loom"] == "recall"
        assert d["insight_type"] == "math"
        assert d["confidence"] == 0.8

    def test_dream_insight_from_dict(self):
        data = {
            "source_loom": "reason",
            "insight_type": "correction",
            "content": "fix",
            "confidence": 0.9,
            "shareable": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
        insight = DreamInsight.from_dict(data)
        assert insight.source_loom == "reason"
        assert insight.insight_type == InsightType.CORRECTION

    def test_dream_insight_from_dict_no_timestamp(self):
        data = {
            "source_loom": "x",
            "insight_type": "math",
            "content": "data",
            "confidence": 0.5,
        }
        insight = DreamInsight.from_dict(data)
        assert insight.timestamp is not None


# ============================================================================
# Core Looms Tests (RecallLoom, ReasonLoom, ReachLoom, ReflectLoom, RefuseLoom)
# ============================================================================


class TestRecallLoom:
    """Tests for RecallLoom - The Librarian."""

    def test_init_defaults(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        assert loom.perspective == "recall"
        assert loom.retrieval_k == 20
        assert loom.temperature == 0.2
        assert loom._memory is None

    def test_blind_spots(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        spots = loom.blind_spots
        assert len(spots) > 0
        assert any("novel" in s for s in spots)

    def test_admits_ignorance_future(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        ignorance = loom.admits_ignorance("What will happen tomorrow?")
        assert any("predict" in i or "future" in i for i in ignorance)

    def test_admits_ignorance_creative(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        ignorance = loom.admits_ignorance("Imagine a new approach")
        assert any("novel" in i or "generate" in i.lower() for i in ignorance)

    def test_admits_ignorance_default(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        ignorance = loom.admits_ignorance("What is Thompson Sampling?")
        assert len(ignorance) >= 1

    def test_falsifiable_by(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        falsifiers = loom.falsifiable_by("claim")
        assert len(falsifiers) >= 2

    def test_compile_memory_response_empty(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        result = loom._compile_memory_response([])
        assert "No relevant" in result

    def test_compile_memory_response_with_threads(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        threads = [
            {"content": "First memory", "relevance": 0.9},
            {"content": "Second memory", "relevance": 0.7},
        ]
        result = loom._compile_memory_response(threads)
        assert "First memory" in result
        assert "Second memory" in result

    def test_compile_memory_response_many_threads(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        threads = [{"content": f"mem_{i}", "relevance": 0.5} for i in range(10)]
        result = loom._compile_memory_response(threads)
        assert "5 more" in result

    def test_identify_memory_gaps_no_threads(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        gaps = loom._identify_memory_gaps("test query", [])
        assert any("No memories" in g for g in gaps)

    def test_identify_memory_gaps_low_relevance(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        threads = [{"relevance": 0.2}, {"relevance": 0.3}, {"relevance": 0.1}]
        gaps = loom._identify_memory_gaps("test", threads)
        assert any("low relevance" in g for g in gaps)

    def test_identify_memory_gaps_limited_count(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        threads = [{"relevance": 0.9}]
        gaps = loom._identify_memory_gaps("test", threads)
        assert any("Limited" in g for g in gaps)

    def test_compute_confidence_empty(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        assert loom._compute_confidence([]) == 0.2

    def test_compute_confidence_with_threads(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        threads = [{"relevance": 0.9} for _ in range(20)]
        conf = loom._compute_confidence(threads)
        assert conf > 0.5
        assert conf <= 1.0

    def test_compute_epistemic_no_gaps(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        epistemic = loom._compute_epistemic_confidence([], [{"relevance": 0.8}])
        assert epistemic >= 0.6

    def test_compute_epistemic_with_gaps(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        gaps = ["gap1", "gap2", "gap3"]
        epistemic = loom._compute_epistemic_confidence(gaps, [])
        assert epistemic < 0.8

    def test_compute_epistemic_high_variance(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        threads = [{"relevance": 0.1}, {"relevance": 0.9}]
        epistemic = loom._compute_epistemic_confidence([], threads)
        # High variance should reduce epistemic confidence
        assert epistemic <= 0.8

    @pytest.mark.asyncio
    async def test_retrieve_memories_no_backend(self):
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        result = await loom._retrieve_memories("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_execute_hits_confidence_metadata_bug(self):
        """execute() constructs ConfidenceMetadata with invalid kwargs.
        Documents latent bug shared by all core looms."""
        from hololoom.loom.core_looms.recall_loom import RecallLoom
        loom = RecallLoom()
        req = _make_request("What is TS?")
        with pytest.raises(TypeError):
            await loom.execute(req)


class TestReasonLoom:
    """Tests for ReasonLoom - The Logician."""

    def test_init_defaults(self):
        from hololoom.loom.core_looms.reason_loom import ReasonLoom
        loom = ReasonLoom()
        assert loom.perspective == "reason"
        assert loom.max_inference_depth == 5

    def test_blind_spots(self):
        from hololoom.loom.core_looms.reason_loom import ReasonLoom
        loom = ReasonLoom()
        assert len(loom.blind_spots) > 0
        assert any("intuitive" in s for s in loom.blind_spots)

    def test_admits_ignorance_feelings(self):
        from hololoom.loom.core_looms.reason_loom import ReasonLoom
        loom = ReasonLoom()
        ignorance = loom.admits_ignorance("How does this feel?")
        assert any("feel" in i for i in ignorance)

    def test_admits_ignorance_uncertain(self):
        from hololoom.loom.core_looms.reason_loom import ReasonLoom
        loom = ReasonLoom()
        ignorance = loom.admits_ignorance("This is probably true")
        assert any("probability" in i.lower() or "uncertainty" in i.lower() for i in ignorance)

    def test_admits_ignorance_default(self):
        from hololoom.loom.core_looms.reason_loom import ReasonLoom
        loom = ReasonLoom()
        ignorance = loom.admits_ignorance("What is 2+2?")
        assert len(ignorance) >= 1

    def test_falsifiable_by(self):
        from hololoom.loom.core_looms.reason_loom import ReasonLoom
        loom = ReasonLoom()
        assert len(loom.falsifiable_by("claim")) >= 2


class TestReachLoom:
    """Tests for ReachLoom - The Artist."""

    def test_init_defaults(self):
        from hololoom.loom.core_looms.reach_loom import ReachLoom
        loom = ReachLoom()
        assert loom.perspective == "reach"
        assert loom.temperature == 0.9

    def test_blind_spots(self):
        from hololoom.loom.core_looms.reach_loom import ReachLoom
        loom = ReachLoom()
        assert len(loom.blind_spots) > 0

    def test_admits_ignorance(self):
        from hololoom.loom.core_looms.reach_loom import ReachLoom
        loom = ReachLoom()
        ignorance = loom.admits_ignorance("test query")
        assert len(ignorance) >= 1

    def test_falsifiable_by(self):
        from hololoom.loom.core_looms.reach_loom import ReachLoom
        loom = ReachLoom()
        assert len(loom.falsifiable_by("claim")) >= 1


class TestReflectLoom:
    """Tests for ReflectLoom - The Auditor."""

    def test_init_defaults(self):
        from hololoom.loom.core_looms.reflect_loom import ReflectLoom
        loom = ReflectLoom()
        assert loom.perspective == "reflect"

    def test_blind_spots(self):
        from hololoom.loom.core_looms.reflect_loom import ReflectLoom
        loom = ReflectLoom()
        assert len(loom.blind_spots) > 0

    def test_admits_ignorance(self):
        from hololoom.loom.core_looms.reflect_loom import ReflectLoom
        loom = ReflectLoom()
        ignorance = loom.admits_ignorance("test query")
        assert len(ignorance) >= 1

    def test_falsifiable_by(self):
        from hololoom.loom.core_looms.reflect_loom import ReflectLoom
        loom = ReflectLoom()
        assert len(loom.falsifiable_by("claim")) >= 1


class TestRefuseLoom:
    """Tests for RefuseLoom - The Skeptic."""

    def test_init_defaults(self):
        from hololoom.loom.core_looms.refuse_loom import RefuseLoom
        loom = RefuseLoom()
        assert loom.perspective == "refuse"

    def test_blind_spots(self):
        from hololoom.loom.core_looms.refuse_loom import RefuseLoom
        loom = RefuseLoom()
        assert len(loom.blind_spots) > 0

    def test_admits_ignorance(self):
        from hololoom.loom.core_looms.refuse_loom import RefuseLoom
        loom = RefuseLoom()
        ignorance = loom.admits_ignorance("test query")
        assert len(ignorance) >= 1

    def test_falsifiable_by(self):
        from hololoom.loom.core_looms.refuse_loom import RefuseLoom
        loom = RefuseLoom()
        assert len(loom.falsifiable_by("claim")) >= 1


class TestCreateAllLooms:
    """Tests for the create_all_looms factory."""

    def test_creates_five_looms(self):
        from hololoom.loom.core_looms import create_all_looms
        looms = create_all_looms()
        assert len(looms) == 5
        perspectives = {l.perspective for l in looms}
        assert perspectives == {"recall", "reason", "reach", "reflect", "refuse"}
