#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Weaving Pipeline Steps 4-6
==========================================

Covers:
- _step4_feature_extraction
- _step5_warp_tensioning
- _step6_memory_retrieval
- execute_steps_4_6_parallel
- execute_step5_5_warp_compute
- execute_step6_5_beta_wave_packing
- select_pattern_embedder

All heavy ML models are mocked. Tests stay <5s each.

Patching strategy for locally-imported classes (MatryoshkaEmbeddings,
ZeroCopyMatryoshkaEmbeddings, BetaWaveContextPacker, TokenBudget):
These are imported inside functions, so we patch their actual module
paths in sys.modules rather than patching the steps_4_6 module namespace.
"""

from __future__ import annotations

import sys
import pytest
import asyncio
import numpy as np
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, call
from typing import Any, List, Tuple

import hololoom  # noqa: F401 — installs _CoreRedirectFinder

from hololoom.protocols.types import Query
from hololoom.core.orchestrator.context import WeavingContext, create_weaving_context
from hololoom.core.orchestrator.stages.steps_4_6 import (
    _step4_feature_extraction,
    _step5_warp_tensioning,
    _step6_memory_retrieval,
    execute_steps_4_6_parallel,
    execute_step5_5_warp_compute,
    execute_step6_5_beta_wave_packing,
    select_pattern_embedder,
)


# =============================================================================
# Helpers for patching lazily-imported symbols
# =============================================================================

@contextmanager
def _inject_module(full_name: str, fake_module: Any):
    """Temporarily inject fake_module at sys.modules[full_name]."""
    old = sys.modules.get(full_name)
    sys.modules[full_name] = fake_module
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop(full_name, None)
        else:
            sys.modules[full_name] = old


def _make_emit_events() -> tuple[list, Any]:
    """Return (event_list, callable) that captures (stage_id, name, duration) tuples."""
    events: list = []

    def emit(stage_id, name, duration):
        events.append((stage_id, name, duration))

    return events, emit


# =============================================================================
# Shared helpers / fixtures
# =============================================================================

def _make_ctx(text: str = "What is Thompson Sampling?") -> WeavingContext:
    """Create a minimal WeavingContext suitable for steps 4-6."""
    ctx = create_weaving_context(Query(text=text))

    ps = Mock()
    ps.name = "FAST"
    ps.scales = [96, 192, 384]
    ps.motif_mode = "fast"
    ps.enable_spectral = False
    ps.enable_semantic_flow = False
    ps.retrieval_k = 5
    ps.retrieval_mode = "fast"
    ctx.pattern_spec = ps

    ctx.thread_ids = ["t1", "t2"]
    ctx.thread_texts = ["text one", "text two"]
    return ctx


def _make_resonance_shed(thread_count: int = 2) -> Mock:
    """Create a mock ResonanceShed."""
    shed = AsyncMock()
    shed.weave = AsyncMock(return_value={
        "threads": [{"name": f"thread_{i}"} for i in range(thread_count)],
        "psi": [0.1, 0.2, 0.3],
    })
    return shed


def _make_warp_space() -> Mock:
    """Create a mock WarpSpace."""
    ws = AsyncMock()
    ws.tension = AsyncMock(return_value=None)
    ws.compute = Mock(return_value={
        "attention_entropy": 0.42,
        "context_vector": np.zeros(96, dtype=np.float32),
        "metadata": {"spectral_computed": False},
    })
    return ws


def _make_cfg(
    *,
    enable_linguistic_gate: bool = False,
    linguistic_mode: str = "standard",
    use_compositional_cache: bool = False,
    enable_zero_copy_embeddings: bool = False,
    base_model_name: str = "test-model",
    enable_beta_wave_packing: bool = False,
    packing_token_budget: int = 4096,
    packing_query_reserve: int = 200,
    packing_response_reserve: int = 800,
    packing_activation_threshold: float = 0.3,
    packing_compression_threshold: float = 0.5,
    zero_copy_cache_path: str = "/tmp/zc",
    zero_copy_cache_size: int = 1000,
) -> Mock:
    cfg = Mock()
    cfg.enable_linguistic_gate = enable_linguistic_gate
    cfg.linguistic_mode = linguistic_mode
    cfg.use_compositional_cache = use_compositional_cache
    cfg.enable_zero_copy_embeddings = enable_zero_copy_embeddings
    cfg.base_model_name = base_model_name
    cfg.enable_beta_wave_packing = enable_beta_wave_packing
    cfg.packing_token_budget = packing_token_budget
    cfg.packing_query_reserve = packing_query_reserve
    cfg.packing_response_reserve = packing_response_reserve
    cfg.packing_activation_threshold = packing_activation_threshold
    cfg.packing_compression_threshold = packing_compression_threshold
    cfg.zero_copy_cache_path = zero_copy_cache_path
    cfg.zero_copy_cache_size = zero_copy_cache_size
    return cfg


# =============================================================================
# select_pattern_embedder
# =============================================================================

class TestSelectPatternEmbedder:
    """Tests for the embedder selection logic."""

    def test_returns_linguistic_gate_when_enabled(self):
        """When linguistic gate is configured, it is returned directly."""
        gate = Mock(name="LinguisticGate")
        cfg = _make_cfg(enable_linguistic_gate=True)
        pattern_spec = Mock(scales=[96, 192, 384])

        result = select_pattern_embedder(cfg, pattern_spec, linguistic_gate=gate)

        assert result is gate

    def test_ignores_linguistic_gate_when_flag_off(self):
        """linguistic_gate object is ignored when cfg.enable_linguistic_gate is False."""
        gate = Mock(name="LinguisticGate")
        cfg = _make_cfg(enable_linguistic_gate=False, enable_zero_copy_embeddings=False)
        pattern_spec = Mock(scales=[96, 192, 384])

        # Inject a fake spectral module so the real model doesn't load.
        fake_embedder = Mock(name="StandardEmbedder")
        fake_spectral = Mock()
        fake_spectral.MatryoshkaEmbeddings = Mock(return_value=fake_embedder)

        with _inject_module("hololoom.embedding.spectral", fake_spectral), \
             _inject_module("hololoom.core.embedding.spectral", fake_spectral):
            result = select_pattern_embedder(cfg, pattern_spec, linguistic_gate=gate)

        assert result is not gate

    def test_uses_zero_copy_when_enabled(self):
        """Returns ZeroCopyMatryoshkaEmbeddings when cfg flag is set."""
        cfg = _make_cfg(enable_zero_copy_embeddings=True)
        pattern_spec = Mock(scales=[96, 192, 384])

        fake_zc_instance = Mock(name="ZeroCopyEmbedder")
        fake_zc_cls = Mock(return_value=fake_zc_instance)
        fake_zc_module = Mock()
        fake_zc_module.ZeroCopyMatryoshkaEmbeddings = fake_zc_cls

        with _inject_module("hololoom.embedding.zero_copy", fake_zc_module), \
             _inject_module("hololoom.core.embedding.zero_copy", fake_zc_module):
            result = select_pattern_embedder(cfg, pattern_spec)

        assert result is fake_zc_instance
        fake_zc_cls.assert_called_once_with(
            sizes=pattern_spec.scales,
            base_model_name=cfg.base_model_name,
            store_path=cfg.zero_copy_cache_path,
            max_cache_size=cfg.zero_copy_cache_size,
        )

    def test_falls_back_to_standard_embedder(self):
        """When no special flags, returns a MatryoshkaEmbeddings instance."""
        cfg = _make_cfg(enable_zero_copy_embeddings=False, enable_linguistic_gate=False)
        pattern_spec = Mock(scales=[96, 192, 384])

        fake_std_instance = Mock(name="StandardEmbedder")
        fake_std_cls = Mock(return_value=fake_std_instance)
        fake_spectral = Mock()
        fake_spectral.MatryoshkaEmbeddings = fake_std_cls

        with _inject_module("hololoom.embedding.spectral", fake_spectral), \
             _inject_module("hololoom.core.embedding.spectral", fake_spectral):
            result = select_pattern_embedder(cfg, pattern_spec)

        assert result is fake_std_instance
        fake_std_cls.assert_called_once_with(
            sizes=pattern_spec.scales,
            base_model_name=cfg.base_model_name,
        )

    def test_linguistic_gate_takes_precedence_over_zero_copy(self):
        """LinguisticGate beats ZeroCopy when both flags are set."""
        gate = Mock(name="LinguisticGate")
        cfg = _make_cfg(enable_linguistic_gate=True, enable_zero_copy_embeddings=True)
        pattern_spec = Mock(scales=[96, 192, 384])

        result = select_pattern_embedder(cfg, pattern_spec, linguistic_gate=gate)

        assert result is gate

    def test_no_gate_returns_non_none(self):
        """Without gate, standard path still returns something (not None)."""
        cfg = _make_cfg(enable_linguistic_gate=False, enable_zero_copy_embeddings=False)
        pattern_spec = Mock(scales=[96, 192, 384])

        # Don't inject a fake — let real embedder load (lightweight constructor)
        result = select_pattern_embedder(cfg, pattern_spec)

        assert result is not None


# =============================================================================
# _step4_feature_extraction
# =============================================================================

class TestStep4FeatureExtraction:
    """Tests for the resonance-shed feature-extraction step."""

    @pytest.mark.asyncio
    async def test_returns_dot_plasma_and_duration(self):
        ctx = _make_ctx()
        shed = _make_resonance_shed(thread_count=3)

        dot_plasma, duration = await _step4_feature_extraction(ctx, shed)

        assert isinstance(dot_plasma, dict)
        assert "threads" in dot_plasma
        assert len(dot_plasma["threads"]) == 3
        assert duration >= 0

    @pytest.mark.asyncio
    async def test_calls_shed_with_current_query_text(self):
        ctx = _make_ctx("Explain Thompson Sampling")
        shed = _make_resonance_shed()

        await _step4_feature_extraction(ctx, shed)

        shed.weave.assert_awaited_once_with(
            text="Explain Thompson Sampling",
            context_graph=None,
        )

    @pytest.mark.asyncio
    async def test_emit_stage_event_called_twice(self):
        """emit_stage_event should be called at start (no duration) and end (with duration)."""
        ctx = _make_ctx()
        shed = _make_resonance_shed()
        events, emit = _make_emit_events()

        await _step4_feature_extraction(ctx, shed, emit_stage_event=emit)

        assert len(events) == 2
        assert events[0] == (4, "Resonance Shed", None)
        assert events[1][0] == 4
        assert events[1][2] is not None

    @pytest.mark.asyncio
    async def test_emit_stage_event_optional(self):
        """No emit callback — should not raise."""
        ctx = _make_ctx()
        shed = _make_resonance_shed()

        dot_plasma, duration = await _step4_feature_extraction(ctx, shed)

        assert dot_plasma is not None

    @pytest.mark.asyncio
    async def test_empty_threads_dot_plasma(self):
        """Handles a DotPlasma with no threads key gracefully."""
        ctx = _make_ctx()
        shed = AsyncMock()
        shed.weave = AsyncMock(return_value={"psi": []})

        dot_plasma, duration = await _step4_feature_extraction(ctx, shed)

        assert dot_plasma is not None
        assert len(dot_plasma.get("threads", [])) == 0

    @pytest.mark.asyncio
    async def test_duration_is_float(self):
        ctx = _make_ctx()
        shed = _make_resonance_shed()

        _, duration = await _step4_feature_extraction(ctx, shed)

        assert isinstance(duration, float)


# =============================================================================
# _step5_warp_tensioning
# =============================================================================

class TestStep5WarpTensioning:
    """Tests for the warp-space tensioning step."""

    @pytest.mark.asyncio
    async def test_returns_warp_operations_and_duration(self):
        ctx = _make_ctx()
        ws = _make_warp_space()

        warp_ops, duration = await _step5_warp_tensioning(ctx, ws)

        assert isinstance(warp_ops, list)
        assert len(warp_ops) == 1
        assert duration >= 0

    @pytest.mark.asyncio
    async def test_warp_op_contains_thread_count(self):
        ctx = _make_ctx()
        ctx.thread_ids = ["a", "b", "c"]
        ws = _make_warp_space()

        warp_ops, _ = await _step5_warp_tensioning(ctx, ws)

        timestamp, op_name, count = warp_ops[0]
        assert op_name == "tension"
        assert count == 3

    @pytest.mark.asyncio
    async def test_calls_tension_with_thread_texts_and_ids(self):
        ctx = _make_ctx()
        ctx.thread_texts = ["alpha", "beta"]
        ctx.thread_ids = ["id1", "id2"]
        ws = _make_warp_space()

        await _step5_warp_tensioning(ctx, ws)

        ws.tension.assert_awaited_once_with(["alpha", "beta"], thread_ids=["id1", "id2"])

    @pytest.mark.asyncio
    async def test_emit_stage_event_called(self):
        ctx = _make_ctx()
        ws = _make_warp_space()
        events, emit = _make_emit_events()

        await _step5_warp_tensioning(ctx, ws, emit_stage_event=emit)

        assert len(events) == 2
        assert events[0] == (5, "Warp Space", None)
        assert events[1][0] == 5
        assert events[1][2] is not None

    @pytest.mark.asyncio
    async def test_warp_op_timestamp_is_iso_string(self):
        ctx = _make_ctx()
        ws = _make_warp_space()

        warp_ops, _ = await _step5_warp_tensioning(ctx, ws)

        ts = warp_ops[0][0]
        datetime.fromisoformat(ts)

    @pytest.mark.asyncio
    async def test_no_emit_does_not_crash(self):
        ctx = _make_ctx()
        ws = _make_warp_space()

        ops, duration = await _step5_warp_tensioning(ctx, ws)

        assert ops is not None


# =============================================================================
# _step6_memory_retrieval
# =============================================================================

class TestStep6MemoryRetrieval:
    """Tests for the memory retrieval step."""

    def _make_shard(self, text: str = "shard text") -> Mock:
        s = Mock()
        s.text = text
        return s

    @pytest.mark.asyncio
    async def test_no_memory_no_retriever_returns_empty(self):
        ctx = _make_ctx()

        shards, shard_texts, hits, duration = await _step6_memory_retrieval(
            ctx, memory=None, retriever=None,
            complexity=None, provenance=None,
        )

        assert shards == []
        assert shard_texts == []
        assert hits == []
        assert duration >= 0

    @pytest.mark.asyncio
    async def test_multipass_crawl_path(self):
        ctx = _make_ctx()
        shard1 = self._make_shard("Crawled result")
        memory = Mock(name="DynMemory")

        async def fake_crawl(mem, query, complexity, provenance):
            return [shard1]

        shards, shard_texts, hits, duration = await _step6_memory_retrieval(
            ctx,
            memory=memory,
            retriever=None,
            complexity=Mock(),
            provenance=Mock(),
            multipass_memory_crawl=fake_crawl,
        )

        assert shards == [shard1]
        assert shard_texts == ["Crawled result"]
        assert hits == [(shard1, 1.0)]

    @pytest.mark.asyncio
    async def test_legacy_retriever_path(self):
        ctx = _make_ctx()
        shard1 = self._make_shard("Legacy shard")

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=[(shard1, 0.9)])

        shards, shard_texts, hits, duration = await _step6_memory_retrieval(
            ctx,
            memory=None,
            retriever=retriever,
            complexity=None,
            provenance=None,
        )

        assert shards == [shard1]
        assert shard_texts == ["Legacy shard"]
        assert hits == [(shard1, 0.9)]

    @pytest.mark.asyncio
    async def test_legacy_retriever_uses_pattern_spec_k(self):
        ctx = _make_ctx()
        ctx.pattern_spec.retrieval_k = 7
        ctx.pattern_spec.retrieval_mode = "hybrid"

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=[])

        await _step6_memory_retrieval(
            ctx, memory=None, retriever=retriever,
            complexity=None, provenance=None,
        )

        retriever.search.assert_awaited_once_with(
            query=ctx.current_query_text,
            k=7,
            fast=False,  # "hybrid" → fast=False
        )

    @pytest.mark.asyncio
    async def test_emit_stage_event_called(self):
        ctx = _make_ctx()
        events, emit = _make_emit_events()

        await _step6_memory_retrieval(
            ctx, memory=None, retriever=None,
            complexity=None, provenance=None,
            emit_stage_event=emit,
        )

        assert len(events) == 2
        assert events[0] == (6, "Memory Retrieval", None)
        assert events[1][0] == 6

    @pytest.mark.asyncio
    async def test_multipass_crawl_preferred_over_retriever(self):
        ctx = _make_ctx()
        shard1 = self._make_shard("Crawl wins")
        memory = Mock()

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=[])

        async def fake_crawl(mem, query, complexity, provenance):
            return [shard1]

        shards, _, _, _ = await _step6_memory_retrieval(
            ctx,
            memory=memory,
            retriever=retriever,
            complexity=None,
            provenance=None,
            multipass_memory_crawl=fake_crawl,
        )

        assert shards == [shard1]
        retriever.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fast_retrieval_mode_sets_fast_true(self):
        ctx = _make_ctx()
        ctx.pattern_spec.retrieval_mode = "fast"

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=[])

        await _step6_memory_retrieval(
            ctx, memory=None, retriever=retriever,
            complexity=None, provenance=None,
        )

        _, kwargs = retriever.search.call_args
        assert kwargs.get("fast") is True

    @pytest.mark.asyncio
    async def test_default_k_when_no_pattern_spec(self):
        """k defaults to 10 when pattern_spec is None."""
        ctx = _make_ctx()
        ctx.pattern_spec = None

        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=[])

        await _step6_memory_retrieval(
            ctx, memory=None, retriever=retriever,
            complexity=None, provenance=None,
        )

        _, kwargs = retriever.search.call_args
        assert kwargs["k"] == 10


# =============================================================================
# execute_step5_5_warp_compute
# =============================================================================

class TestStep5_5WarpCompute:
    """Tests for the post-parallel warp compute step."""

    def _ctx_with_warp(self, psi=None) -> WeavingContext:
        ctx = _make_ctx()
        ws = _make_warp_space()
        ctx.warp_space = ws
        ctx.dot_plasma = {"threads": [], "psi": psi if psi is not None else [0.1, 0.2, 0.3]}
        return ctx

    @pytest.mark.asyncio
    async def test_skips_when_warp_space_is_none(self):
        ctx = _make_ctx()
        ctx.warp_space = None

        result = await execute_step5_5_warp_compute(ctx)

        assert result.warp_compute_results is None
        assert "warp_compute" not in result.stage_timings

    @pytest.mark.asyncio
    async def test_populates_warp_compute_results(self):
        ctx = self._ctx_with_warp()

        result = await execute_step5_5_warp_compute(ctx)

        assert result.warp_compute_results is not None
        assert "attention_entropy" in result.warp_compute_results
        assert "warp_compute" in result.stage_timings

    @pytest.mark.asyncio
    async def test_calls_compute_with_numpy_embedding(self):
        ctx = self._ctx_with_warp(psi=[1.0, 2.0, 3.0])
        ws = ctx.warp_space

        await execute_step5_5_warp_compute(ctx)

        ws.compute.assert_called_once()
        emb = ws.compute.call_args[1]["query_embedding"]
        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32

    @pytest.mark.asyncio
    async def test_handles_dict_psi(self):
        """When psi is a dict keyed by scale, selects the largest key."""
        ctx = _make_ctx()
        ws = _make_warp_space()
        ctx.warp_space = ws
        ctx.dot_plasma = {
            "threads": [],
            "psi": {96: [0.1, 0.2], 384: [0.3, 0.4]},
        }

        await execute_step5_5_warp_compute(ctx)

        emb = ws.compute.call_args[1]["query_embedding"]
        np.testing.assert_array_almost_equal(emb, np.array([0.3, 0.4], dtype=np.float32))

    @pytest.mark.asyncio
    async def test_compute_failure_sets_none_and_adds_warning(self):
        ctx = _make_ctx()
        ws = Mock()
        ws.compute = Mock(side_effect=RuntimeError("GPU OOM"))
        ctx.warp_space = ws
        ctx.dot_plasma = {"psi": []}

        result = await execute_step5_5_warp_compute(ctx)

        assert result.warp_compute_results is None
        assert result.has_warnings
        assert result.stage_timings.get("warp_compute") == 0.0

    @pytest.mark.asyncio
    async def test_adds_warp_operation_on_success(self):
        ctx = self._ctx_with_warp()
        initial_ops = len(ctx.warp_operations)

        result = await execute_step5_5_warp_compute(ctx)

        assert len(result.warp_operations) == initial_ops + 1
        _, op_name, _ = result.warp_operations[-1]
        assert op_name == "compute"

    @pytest.mark.asyncio
    async def test_none_dot_plasma_treated_as_empty(self):
        """Should not crash when dot_plasma is None."""
        ctx = _make_ctx()
        ctx.warp_space = _make_warp_space()
        ctx.dot_plasma = None

        result = await execute_step5_5_warp_compute(ctx)

        assert result is not None

    @pytest.mark.asyncio
    async def test_timing_recorded_on_success(self):
        ctx = self._ctx_with_warp()

        result = await execute_step5_5_warp_compute(ctx)

        assert result.stage_timings.get("warp_compute", -1) >= 0

    @pytest.mark.asyncio
    async def test_attention_entropy_in_results(self):
        ctx = self._ctx_with_warp()

        result = await execute_step5_5_warp_compute(ctx)

        assert result.warp_compute_results["attention_entropy"] == pytest.approx(0.42)


# =============================================================================
# execute_step6_5_beta_wave_packing
# =============================================================================

class TestStep6_5BetaWavePacking:
    """Tests for the optional beta-wave context packing step."""

    @pytest.mark.asyncio
    async def test_disabled_by_config(self):
        ctx = _make_ctx()
        cfg = _make_cfg(enable_beta_wave_packing=False)
        memory = Mock()

        result = await execute_step6_5_beta_wave_packing(ctx, cfg, memory=memory)

        assert result.packed_context is None

    @pytest.mark.asyncio
    async def test_disabled_when_memory_none(self):
        ctx = _make_ctx()
        cfg = _make_cfg(enable_beta_wave_packing=True)

        result = await execute_step6_5_beta_wave_packing(ctx, cfg, memory=None)

        assert result.packed_context is None

    @pytest.mark.asyncio
    async def test_disabled_when_memory_lacks_spring_engine(self):
        ctx = _make_ctx()
        cfg = _make_cfg(enable_beta_wave_packing=True)
        memory = Mock(spec=[])  # no spring_engine attribute

        result = await execute_step6_5_beta_wave_packing(ctx, cfg, memory=memory)

        assert result.packed_context is None

    @pytest.mark.asyncio
    async def test_context_returned_when_disabled(self):
        """Returns the same context object even when disabled."""
        ctx = _make_ctx()
        cfg = _make_cfg(enable_beta_wave_packing=False)

        result = await execute_step6_5_beta_wave_packing(ctx, cfg)

        assert result is ctx

    @pytest.mark.asyncio
    async def test_success_populates_packed_context(self):
        ctx = _make_ctx()
        ctx.dot_plasma = {"psi": [0.1, 0.2, 0.3]}
        ctx.shards = []
        cfg = _make_cfg(enable_beta_wave_packing=True)

        memory = Mock()
        memory.spring_engine = Mock(name="SpringEngine")

        packed_result = Mock(
            elements_included=3,
            elements_compressed=1,
            elements_excluded=0,
            total_tokens=200,
            avg_activation=0.7,
        )

        mock_packer = AsyncMock()
        mock_packer.pack_context = AsyncMock(return_value=packed_result)

        mock_token_budget = Mock()
        mock_token_budget.available_for_context = 3096

        mock_packer_cls = Mock(return_value=mock_packer)
        mock_token_budget_cls = Mock(return_value=mock_token_budget)

        fake_packer_module = Mock()
        fake_packer_module.BetaWaveContextPacker = mock_packer_cls
        fake_packer_module.TokenBudget = mock_token_budget_cls

        # The source imports from hololoom.awareness.beta_wave_packer
        with _inject_module("hololoom.awareness.beta_wave_packer", fake_packer_module), \
             _inject_module("hololoom.core.memory.awareness.beta_wave_packer", fake_packer_module):
            result = await execute_step6_5_beta_wave_packing(ctx, cfg, memory=memory)

        assert result.packed_context is packed_result
        mock_packer.pack_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failure_falls_back_gracefully(self):
        """When packer raises, packed_context is None and warning added."""
        ctx = _make_ctx()
        ctx.dot_plasma = {"psi": [0.1]}
        ctx.shards = []
        cfg = _make_cfg(enable_beta_wave_packing=True)

        memory = Mock()
        memory.spring_engine = Mock()

        # Simulate ImportError for beta_wave_packer (module not installed)
        with _inject_module("hololoom.awareness.beta_wave_packer", None):
            # None module causes AttributeError when we try to get .BetaWaveContextPacker
            result = await execute_step6_5_beta_wave_packing(ctx, cfg, memory=memory)

        assert result.packed_context is None
        assert result.has_warnings


# =============================================================================
# execute_steps_4_6_parallel (integration of the three sub-steps)
# =============================================================================

class TestExecuteSteps4To6Parallel:
    """Integration tests for the parallel orchestration function."""

    def _full_ctx(self) -> WeavingContext:
        ctx = _make_ctx()
        ctx.thread_ids = ["t1", "t2"]
        ctx.thread_texts = ["Alpha text", "Beta text"]
        return ctx

    @pytest.mark.asyncio
    async def test_populates_all_ctx_fields(self):
        ctx = self._full_ctx()
        cfg = _make_cfg()
        embedder = Mock(name="Embedder")
        shed = _make_resonance_shed()
        ws = _make_warp_space()

        with patch(
            "hololoom.core.orchestrator.stages.steps_4_6.select_pattern_embedder",
            return_value=embedder,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_resonance_shed",
            return_value=shed,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_warp_space",
            return_value=ws,
        ):
            result = await execute_steps_4_6_parallel(ctx, cfg, embedder=embedder)

        assert result.dot_plasma is not None
        assert result.warp_operations is not None
        assert result.shards is not None
        assert result.shard_texts is not None
        assert result.hits is not None

    @pytest.mark.asyncio
    async def test_timings_recorded(self):
        ctx = self._full_ctx()
        cfg = _make_cfg()
        embedder = Mock(name="Embedder")
        shed = _make_resonance_shed()
        ws = _make_warp_space()

        with patch(
            "hololoom.core.orchestrator.stages.steps_4_6.select_pattern_embedder",
            return_value=embedder,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_resonance_shed",
            return_value=shed,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_warp_space",
            return_value=ws,
        ):
            result = await execute_steps_4_6_parallel(ctx, cfg, embedder=embedder)

        assert "feature_extraction" in result.stage_timings
        assert "warp_tensioning" in result.stage_timings
        assert "retrieval" in result.stage_timings
        assert "parallel_execution_wall_time" in result.stage_timings
        assert "parallel_speedup" in result.stage_timings

    @pytest.mark.asyncio
    async def test_emit_stage_event_propagated(self):
        ctx = self._full_ctx()
        cfg = _make_cfg()
        embedder = Mock(name="Embedder")
        shed = _make_resonance_shed()
        ws = _make_warp_space()
        events, emit = _make_emit_events()

        with patch(
            "hololoom.core.orchestrator.stages.steps_4_6.select_pattern_embedder",
            return_value=embedder,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_resonance_shed",
            return_value=shed,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_warp_space",
            return_value=ws,
        ):
            await execute_steps_4_6_parallel(
                ctx, cfg, embedder=embedder, emit_stage_event=emit
            )

        stage_ids = {e[0] for e in events}
        assert 4 in stage_ids
        assert 5 in stage_ids
        assert 6 in stage_ids

    @pytest.mark.asyncio
    async def test_resonance_shed_and_warp_space_stored_on_ctx(self):
        ctx = self._full_ctx()
        cfg = _make_cfg()
        embedder = Mock(name="Embedder")
        shed = _make_resonance_shed()
        ws = _make_warp_space()

        with patch(
            "hololoom.core.orchestrator.stages.steps_4_6.select_pattern_embedder",
            return_value=embedder,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_resonance_shed",
            return_value=shed,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_warp_space",
            return_value=ws,
        ):
            result = await execute_steps_4_6_parallel(ctx, cfg, embedder=embedder)

        assert result.resonance_shed is shed
        assert result.warp_space is ws
        assert result.pattern_embedder is embedder

    @pytest.mark.asyncio
    async def test_with_retriever_populates_shards(self):
        ctx = self._full_ctx()
        cfg = _make_cfg()
        embedder = Mock(name="Embedder")
        shed = _make_resonance_shed()
        ws = _make_warp_space()

        shard = Mock()
        shard.text = "retrieved shard"
        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=[(shard, 0.8)])

        with patch(
            "hololoom.core.orchestrator.stages.steps_4_6.select_pattern_embedder",
            return_value=embedder,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_resonance_shed",
            return_value=shed,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_warp_space",
            return_value=ws,
        ):
            result = await execute_steps_4_6_parallel(
                ctx, cfg, embedder=embedder, retriever=retriever
            )

        assert len(result.shards) == 1
        assert result.shard_texts == ["retrieved shard"]

    @pytest.mark.asyncio
    async def test_parallel_speedup_recorded(self):
        """parallel_speedup key is always recorded (value may be >=0 due to mocking)."""
        ctx = self._full_ctx()
        cfg = _make_cfg()
        embedder = Mock(name="Embedder")
        shed = _make_resonance_shed()
        ws = _make_warp_space()

        with patch(
            "hololoom.core.orchestrator.stages.steps_4_6.select_pattern_embedder",
            return_value=embedder,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_resonance_shed",
            return_value=shed,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_warp_space",
            return_value=ws,
        ):
            result = await execute_steps_4_6_parallel(ctx, cfg, embedder=embedder)

        assert "parallel_speedup" in result.stage_timings
        # Speedup is computed as sequential/parallel; both >=0 when mocked
        assert isinstance(result.stage_timings["parallel_speedup"], float)

    @pytest.mark.asyncio
    async def test_multipass_crawl_used_when_provided(self):
        ctx = self._full_ctx()
        cfg = _make_cfg()
        embedder = Mock(name="Embedder")
        shed = _make_resonance_shed()
        ws = _make_warp_space()

        crawled_shard = Mock()
        crawled_shard.text = "crawled"

        async def fake_crawl(mem, query, complexity, provenance):
            return [crawled_shard]

        memory = Mock()

        with patch(
            "hololoom.core.orchestrator.stages.steps_4_6.select_pattern_embedder",
            return_value=embedder,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_resonance_shed",
            return_value=shed,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_warp_space",
            return_value=ws,
        ):
            result = await execute_steps_4_6_parallel(
                ctx, cfg, embedder=embedder,
                memory=memory,
                multipass_memory_crawl=fake_crawl,
            )

        assert result.shard_texts == ["crawled"]


# =============================================================================
# Additional edge-case tests
# =============================================================================

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_step4_custom_logger_used(self):
        import logging
        ctx = _make_ctx()
        shed = _make_resonance_shed()
        custom_log = Mock(spec=logging.Logger)

        await _step4_feature_extraction(ctx, shed, log=custom_log)

        custom_log.info.assert_called()

    @pytest.mark.asyncio
    async def test_step5_custom_logger_used(self):
        import logging
        ctx = _make_ctx()
        ws = _make_warp_space()
        custom_log = Mock(spec=logging.Logger)

        await _step5_warp_tensioning(ctx, ws, log=custom_log)

        custom_log.info.assert_called()

    @pytest.mark.asyncio
    async def test_step6_custom_logger_on_no_source(self):
        import logging
        ctx = _make_ctx()
        custom_log = Mock(spec=logging.Logger)

        await _step6_memory_retrieval(
            ctx, memory=None, retriever=None,
            complexity=None, provenance=None,
            log=custom_log,
        )

        custom_log.warning.assert_called()

    @pytest.mark.asyncio
    async def test_step5_5_returns_ctx_always(self):
        ctx = _make_ctx()
        ctx.warp_space = None

        result = await execute_step5_5_warp_compute(ctx)

        assert result is ctx

    @pytest.mark.asyncio
    async def test_step6_multiple_shards_from_crawl(self):
        ctx = _make_ctx()
        shards = [Mock(text=f"text{i}") for i in range(5)]
        memory = Mock()

        async def crawl(mem, query, c, p):
            return shards

        result_shards, texts, hits, _ = await _step6_memory_retrieval(
            ctx, memory=memory, retriever=None,
            complexity=None, provenance=None,
            multipass_memory_crawl=crawl,
        )

        assert len(result_shards) == 5
        assert len(texts) == 5
        assert all(score == 1.0 for _, score in hits)

    @pytest.mark.asyncio
    async def test_parallel_execution_sets_dot_plasma_threads(self):
        """dot_plasma threads from resonance shed are stored correctly."""
        ctx = _make_ctx()
        ctx.thread_ids = ["x"]
        ctx.thread_texts = ["x text"]
        cfg = _make_cfg()
        embedder = Mock(name="Embedder")
        shed = _make_resonance_shed(thread_count=4)
        ws = _make_warp_space()

        with patch(
            "hololoom.core.orchestrator.stages.steps_4_6.select_pattern_embedder",
            return_value=embedder,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_resonance_shed",
            return_value=shed,
        ), patch(
            "hololoom.core.orchestrator.stages.steps_4_6.create_warp_space",
            return_value=ws,
        ):
            result = await execute_steps_4_6_parallel(ctx, cfg, embedder=embedder)

        assert len(result.dot_plasma.get("threads", [])) == 4
