#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Weaving Pipeline Stages — Coverage Gaps
==================================================

Targets uncovered lines in:
- steps_4_6.py: create_resonance_shed, create_warp_space
- steps_7_9.py: minimax gate, _categorize_tool_to_category, semantic cache
  exception, provenance attachment, safety-blocked spacetime, gradient router
- gates/conscience_gate.py: ImportError fallback in audit trail logging

Author: Claude Code
Date: 2026-03-18
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from hololoom.orchestrator.context import WeavingContext, create_weaving_context
from hololoom.protocols.types import Query


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_query():
    return Query(text="What is Thompson Sampling?")


@pytest.fixture
def ctx(sample_query):
    """Fresh WeavingContext with pattern_spec and threads set up."""
    c = create_weaving_context(sample_query)
    c.pattern_spec = Mock()
    c.pattern_spec.scales = [96, 192, 384]
    c.pattern_spec.motif_mode = "fast"
    c.pattern_spec.enable_spectral = False
    c.pattern_spec.enable_semantic_flow = False
    c.pattern_spec.retrieval_k = 10
    c.pattern_spec.retrieval_mode = "fast"
    c.pattern_spec.name = "test_card"
    c.pattern_spec.card = Mock(value="FAST")
    c.pattern_spec.pipeline_timeout = 30.0
    c.thread_ids = ["t1", "t2"]
    c.thread_texts = ["thread one text", "thread two text"]
    return c


# ============================================================================
# Test: create_resonance_shed and create_warp_space (steps_4_6.py)
# Note: These import hololoom.resonance.motif_detector, spectral_fusion, etc.
# which don't exist as submodules. We test them via sys.modules injection.
# ============================================================================

class TestCreateResonanceShed:

    def _setup_mock_modules(self):
        """Inject mock modules for the lazy imports in create_resonance_shed."""
        import sys

        mock_motif_mod = MagicMock()
        mock_motif_mod.create_motif_detector = Mock(return_value=Mock())

        mock_sf_mod = MagicMock()
        mock_sf_mod.SpectralFusion = Mock(return_value=Mock())

        mock_shed_mod = MagicMock()
        mock_shed_instance = Mock()
        mock_shed_mod.ResonanceShed = Mock(return_value=mock_shed_instance)

        modules = {
            'hololoom.resonance.motif_detector': mock_motif_mod,
            'hololoom.resonance.spectral_fusion': mock_sf_mod,
            'hololoom.resonance.shed': mock_shed_mod,
        }
        return modules, mock_shed_instance, mock_motif_mod, mock_sf_mod, mock_shed_mod

    def test_basic_creation(self, ctx):
        """Create a ResonanceShed with mocked components."""
        import sys
        from hololoom.orchestrator.stages.steps_4_6 import create_resonance_shed

        modules, mock_shed_instance, mock_motif_mod, _, mock_shed_mod = self._setup_mock_modules()
        saved = {k: sys.modules.get(k) for k in modules}
        try:
            sys.modules.update(modules)
            result = create_resonance_shed(ctx, Mock(), Mock())
            assert result is mock_shed_instance
            mock_motif_mod.create_motif_detector.assert_called_once_with(mode="fast")
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_with_spectral_enabled(self, ctx):
        """Spectral fusion created when enable_spectral is True."""
        import sys
        from hololoom.orchestrator.stages.steps_4_6 import create_resonance_shed

        ctx.pattern_spec.enable_spectral = True
        modules, _, _, mock_sf_mod, mock_shed_mod = self._setup_mock_modules()
        saved = {k: sys.modules.get(k) for k in modules}
        try:
            sys.modules.update(modules)
            result = create_resonance_shed(ctx, Mock(), Mock())
            mock_sf_mod.SpectralFusion.assert_called_once()
            call_kwargs = mock_shed_mod.ResonanceShed.call_args[1]
            assert call_kwargs['spectral_fusion'] is not None
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_with_semantic_flow_enabled(self, ctx):
        """Semantic analyzer created when enable_semantic_flow is True."""
        import sys
        from hololoom.orchestrator.stages.steps_4_6 import create_resonance_shed

        ctx.pattern_spec.enable_semantic_flow = True

        modules, _, _, _, mock_shed_mod = self._setup_mock_modules()

        mock_sem_analyzer = MagicMock()
        mock_sem_analyzer.create_semantic_analyzer = Mock(return_value=Mock())
        mock_sem_config = MagicMock()
        mock_sem_config.SemanticCalculusConfig.from_pattern_spec.return_value = Mock(
            dimensions=3, enable_cache=True, compute_ethics=False,
        )
        modules['hololoom.semantic_calculus.analyzer'] = mock_sem_analyzer
        modules['hololoom.semantic_calculus.config'] = mock_sem_config

        saved = {k: sys.modules.get(k) for k in modules}
        try:
            sys.modules.update(modules)
            mock_embedder = Mock()
            mock_embedder.encode = Mock(return_value=[0.0] * 96)
            result = create_resonance_shed(ctx, Mock(), mock_embedder)
            mock_sem_analyzer.create_semantic_analyzer.assert_called_once()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_semantic_flow_failure_graceful(self, ctx):
        """Semantic analyzer failure is caught, shed still created."""
        import sys
        from hololoom.orchestrator.stages.steps_4_6 import create_resonance_shed

        ctx.pattern_spec.enable_semantic_flow = True
        modules, mock_shed_instance, _, _, _ = self._setup_mock_modules()
        # Don't provide semantic_calculus modules — they will ImportError

        saved = {k: sys.modules.get(k) for k in modules}
        try:
            sys.modules.update(modules)
            result = create_resonance_shed(ctx, Mock(), Mock())
            assert result is mock_shed_instance
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class TestCreateWarpSpace:

    def _setup_mock_modules(self):
        """Inject mock modules for the lazy imports in create_warp_space."""
        mock_sf_mod = MagicMock()
        mock_sf_mod.SpectralFusion = Mock(return_value=Mock())

        mock_ws_mod = MagicMock()
        mock_ws_instance = Mock()
        mock_ws_mod.WarpSpace = Mock(return_value=mock_ws_instance)

        modules = {
            'hololoom.resonance.spectral_fusion': mock_sf_mod,
            'hololoom.warp.space': mock_ws_mod,
        }
        return modules, mock_ws_instance, mock_sf_mod, mock_ws_mod

    def test_basic_creation(self, ctx):
        """Create WarpSpace with mocked components."""
        import sys
        from hololoom.orchestrator.stages.steps_4_6 import create_warp_space

        modules, mock_ws_instance, _, mock_ws_mod = self._setup_mock_modules()
        saved = {k: sys.modules.get(k) for k in modules}
        try:
            sys.modules.update(modules)
            result = create_warp_space(ctx, Mock())
            assert result is mock_ws_instance
            call_kwargs = mock_ws_mod.WarpSpace.call_args[1]
            assert call_kwargs['scales'] == [96, 192, 384]
            assert call_kwargs['spectral_fusion'] is None
            assert call_kwargs['guardrails'] is None
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_with_spectral_enabled(self, ctx):
        """SpectralFusion created when enabled."""
        import sys
        from hololoom.orchestrator.stages.steps_4_6 import create_warp_space

        ctx.pattern_spec.enable_spectral = True
        modules, _, mock_sf_mod, mock_ws_mod = self._setup_mock_modules()
        saved = {k: sys.modules.get(k) for k in modules}
        try:
            sys.modules.update(modules)
            result = create_warp_space(ctx, Mock())
            mock_sf_mod.SpectralFusion.assert_called_once()
            call_kwargs = mock_ws_mod.WarpSpace.call_args[1]
            assert call_kwargs['spectral_fusion'] is not None
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_with_guardrails(self, ctx):
        """Guardrails passed through to WarpSpace."""
        import sys
        from hololoom.orchestrator.stages.steps_4_6 import create_warp_space

        modules, _, _, mock_ws_mod = self._setup_mock_modules()
        mock_guardrails = Mock()
        saved = {k: sys.modules.get(k) for k in modules}
        try:
            sys.modules.update(modules)
            result = create_warp_space(ctx, Mock(), guardrails=mock_guardrails)
            call_kwargs = mock_ws_mod.WarpSpace.call_args[1]
            assert call_kwargs['guardrails'] is mock_guardrails
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


# ============================================================================
# Test: execute_step7_convergence (steps_7_9.py)
# ============================================================================

class TestStep7Convergence:

    @pytest.mark.asyncio
    async def test_basic_convergence(self, ctx):
        """Step 7 collapses to a tool via convergence engine."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step7_convergence

        policy = AsyncMock()
        policy.decide.return_value = Mock(
            tool="answer",
            confidence=0.9,
            tool_probs={"answer": 0.7, "research": 0.2, "remember": 0.05, "clarify": 0.05},
            adapter="default",
        )

        tool_executor = Mock()
        tool_executor.tools = ['answer', 'research', 'remember', 'clarify']

        cfg = Mock()
        cfg.bandit_strategy = 'epsilon_greedy'
        cfg.epsilon = 0.1

        mock_collapse = Mock(tool="answer", confidence=0.9, bandit_stats={})
        mock_ce_instance = Mock()
        mock_ce_instance.collapse.return_value = mock_collapse

        with patch("hololoom.convergence.engine.ConvergenceEngine", return_value=mock_ce_instance), \
             patch("hololoom.convergence.engine.CollapseStrategy") as mock_cs:

            mock_cs.EPSILON_GREEDY = "eg"
            mock_cs.BAYESIAN_BLEND = "bb"
            mock_cs.PURE_THOMPSON = "pt"

            result = await execute_step7_convergence(ctx, cfg, policy, tool_executor)

        assert result.collapse_result.tool == "answer"
        assert result.action_plan is not None

    @pytest.mark.asyncio
    async def test_policy_timeout_fallback(self, ctx):
        """Step 7 falls back to safe default on policy timeout."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step7_convergence

        async def slow_decide(**kwargs):
            import asyncio
            await asyncio.sleep(1.0)

        policy = AsyncMock()
        policy.decide = slow_decide

        tool_executor = Mock()
        tool_executor.tools = ['answer', 'research']

        cfg = Mock()
        cfg.bandit_strategy = 'epsilon_greedy'
        cfg.epsilon = 0.1

        mock_collapse = Mock(tool="answer", confidence=0.5, bandit_stats={})
        mock_ce_instance = Mock()
        mock_ce_instance.collapse.return_value = mock_collapse

        with patch("hololoom.convergence.engine.ConvergenceEngine", return_value=mock_ce_instance), \
             patch("hololoom.convergence.engine.CollapseStrategy") as mock_cs:

            mock_cs.EPSILON_GREEDY = "eg"
            mock_cs.BAYESIAN_BLEND = "bb"
            mock_cs.PURE_THOMPSON = "pt"

            result = await execute_step7_convergence(ctx, cfg, policy, tool_executor)

        # Should have used fallback ActionPlan with timeout metadata
        assert result.action_plan.metadata.get("timeout") is True

    @pytest.mark.asyncio
    async def test_with_gradient_router(self, ctx):
        """Step 7 blends gradient flow when router provided."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step7_convergence

        policy = AsyncMock()
        policy.decide.return_value = Mock(
            tool="answer", confidence=0.8,
            tool_probs={"answer": 0.6, "research": 0.4},
            adapter="default",
        )

        tool_executor = Mock()
        tool_executor.tools = ['answer', 'research']

        gradient_router = AsyncMock()
        gradient_router.select_tool.return_value = Mock(target="research", loss=0.1)

        cfg = Mock()
        cfg.bandit_strategy = 'epsilon_greedy'
        cfg.epsilon = 0.1

        mock_collapse = Mock(tool="research", confidence=0.85, bandit_stats={})
        mock_ce_instance = Mock()
        mock_ce_instance.collapse.return_value = mock_collapse

        with patch("hololoom.convergence.engine.ConvergenceEngine", return_value=mock_ce_instance), \
             patch("hololoom.convergence.engine.CollapseStrategy") as mock_cs:

            mock_cs.EPSILON_GREEDY = "eg"
            mock_cs.BAYESIAN_BLEND = "bb"
            mock_cs.PURE_THOMPSON = "pt"

            result = await execute_step7_convergence(
                ctx, cfg, policy, tool_executor, gradient_router=gradient_router,
            )

        assert result.gradient_decision is not None
        assert result.gradient_decision.target == "research"
        # Neural probs should have been blended
        assert result.neural_probs is not None

    @pytest.mark.asyncio
    async def test_gradient_router_failure_graceful(self, ctx):
        """Step 7 continues when gradient router throws."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step7_convergence

        policy = AsyncMock()
        policy.decide.return_value = Mock(
            tool="answer", confidence=0.8,
            tool_probs={"answer": 0.9, "research": 0.1},
            adapter="default",
        )

        tool_executor = Mock()
        tool_executor.tools = ['answer', 'research']

        gradient_router = AsyncMock()
        gradient_router.select_tool.side_effect = RuntimeError("gradient exploded")

        cfg = Mock()
        cfg.bandit_strategy = 'epsilon_greedy'
        cfg.epsilon = 0.1

        mock_collapse = Mock(tool="answer", confidence=0.9, bandit_stats={})
        mock_ce_instance = Mock()
        mock_ce_instance.collapse.return_value = mock_collapse

        with patch("hololoom.convergence.engine.ConvergenceEngine", return_value=mock_ce_instance), \
             patch("hololoom.convergence.engine.CollapseStrategy") as mock_cs:

            mock_cs.EPSILON_GREEDY = "eg"
            mock_cs.BAYESIAN_BLEND = "bb"
            mock_cs.PURE_THOMPSON = "pt"

            result = await execute_step7_convergence(
                ctx, cfg, policy, tool_executor, gradient_router=gradient_router,
            )

        # Should still complete despite gradient router failure
        assert result.collapse_result.tool == "answer"

    @pytest.mark.asyncio
    async def test_emit_stage_events(self, ctx):
        """Step 7 emits start and completion events."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step7_convergence

        policy = AsyncMock()
        policy.decide.return_value = Mock(
            tool="answer", confidence=0.9,
            tool_probs={"answer": 1.0},
            adapter="default",
        )
        tool_executor = Mock()
        tool_executor.tools = ['answer']
        cfg = Mock()
        cfg.bandit_strategy = 'epsilon_greedy'
        cfg.epsilon = 0.1

        mock_collapse = Mock(tool="answer", confidence=0.9, bandit_stats={})
        mock_ce_instance = Mock()
        mock_ce_instance.collapse.return_value = mock_collapse

        events = []
        def emit(step, name, dur):
            events.append((step, name, dur))

        with patch("hololoom.convergence.engine.ConvergenceEngine", return_value=mock_ce_instance), \
             patch("hololoom.convergence.engine.CollapseStrategy") as mock_cs:

            mock_cs.EPSILON_GREEDY = "eg"
            mock_cs.BAYESIAN_BLEND = "bb"
            mock_cs.PURE_THOMPSON = "pt"

            await execute_step7_convergence(
                ctx, cfg, policy, tool_executor, emit_stage_event=emit,
            )

        assert events[0] == (7, "Convergence Engine", None)
        assert events[1][0] == 7


# ============================================================================
# Test: execute_step8_tool_execution — minimax gate (lines 217-249)
# ============================================================================

class TestStep8MinimaxGate:

    @pytest.mark.asyncio
    async def test_minimax_gate_verified(self, ctx):
        """Step 8 runs minimax gate when provided with collapse_result."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step8_tool_execution

        ctx.collapse_result = Mock(tool="answer", confidence=0.9)
        ctx.neural_probs = np.array([0.7, 0.2, 0.05, 0.05])

        tool_executor = AsyncMock()
        tool_executor.tools = ['answer', 'research', 'remember', 'clarify']
        tool_executor.execute.return_value = {"result": "test answer"}

        minimax_gate = Mock()
        minimax_gate.verify.return_value = Mock(verified=True, minimax_tool=0, regret=0.01)

        result = await execute_step8_tool_execution(
            ctx, tool_executor, minimax_gate=minimax_gate,
        )

        assert result.minimax_result.verified is True
        assert result.tool_result == {"result": "test answer"}

    @pytest.mark.asyncio
    async def test_minimax_gate_divergence(self, ctx):
        """Step 8 logs divergence when minimax disagrees with TS."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step8_tool_execution

        ctx.collapse_result = Mock(tool="answer", confidence=0.6)
        ctx.neural_probs = np.array([0.4, 0.35, 0.15, 0.1])

        tool_executor = AsyncMock()
        tool_executor.tools = ['answer', 'research', 'remember', 'clarify']
        tool_executor.execute.return_value = {"result": "diverged"}

        minimax_gate = Mock()
        minimax_gate.verify.return_value = Mock(verified=False, minimax_tool=1, regret=0.15)

        result = await execute_step8_tool_execution(
            ctx, tool_executor, minimax_gate=minimax_gate,
        )

        assert result.minimax_result.verified is False
        assert result.minimax_result.regret == 0.15

    @pytest.mark.asyncio
    async def test_minimax_gate_exception_non_blocking(self, ctx):
        """Step 8 continues when minimax gate throws."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step8_tool_execution

        ctx.collapse_result = Mock(tool="answer", confidence=0.9)
        ctx.neural_probs = np.array([0.9, 0.1])

        tool_executor = AsyncMock()
        tool_executor.tools = ['answer', 'research']
        tool_executor.execute.return_value = {"result": "ok"}

        minimax_gate = Mock()
        minimax_gate.verify.side_effect = RuntimeError("gate exploded")

        result = await execute_step8_tool_execution(
            ctx, tool_executor, minimax_gate=minimax_gate,
        )

        assert result.tool_result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_minimax_gate_without_tools_attr(self, ctx):
        """Step 8 minimax uses default tool list when executor has no .tools."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step8_tool_execution

        ctx.collapse_result = Mock(tool="answer", confidence=0.8)
        ctx.neural_probs = np.array([0.6, 0.2, 0.1, 0.1])

        tool_executor = AsyncMock(spec=[])
        tool_executor.execute = AsyncMock(return_value={"result": "no tools attr"})

        minimax_gate = Mock()
        minimax_gate.verify.return_value = Mock(verified=True, minimax_tool=0, regret=0.0)

        result = await execute_step8_tool_execution(
            ctx, tool_executor, minimax_gate=minimax_gate,
        )

        assert minimax_gate.verify.called


# ============================================================================
# Test: _categorize_tool_to_category (lines 323-345)
# ============================================================================

class TestCategorizeToolToCategory:

    def test_safe_tools(self):
        """Safe tools (answer, clarify, remember) return QUERY."""
        from hololoom.orchestrator.stages.steps_7_9 import _categorize_tool_to_category
        from hololoom.alignment.safety_guardrails import ActionCategory

        assert _categorize_tool_to_category("answer") == ActionCategory.QUERY
        assert _categorize_tool_to_category("clarify") == ActionCategory.QUERY
        assert _categorize_tool_to_category("remember") == ActionCategory.QUERY

    def test_moderate_tools(self):
        """Moderate tools (research, search, retrieve) return RETRIEVAL."""
        from hololoom.orchestrator.stages.steps_7_9 import _categorize_tool_to_category
        from hololoom.alignment.safety_guardrails import ActionCategory

        assert _categorize_tool_to_category("research") == ActionCategory.RETRIEVAL
        assert _categorize_tool_to_category("search") == ActionCategory.RETRIEVAL
        assert _categorize_tool_to_category("retrieve") == ActionCategory.RETRIEVAL

    def test_risky_tools(self):
        """Risky tools (execute, modify, delete, create) return MODIFICATION."""
        from hololoom.orchestrator.stages.steps_7_9 import _categorize_tool_to_category
        from hololoom.alignment.safety_guardrails import ActionCategory

        assert _categorize_tool_to_category("execute") == ActionCategory.MODIFICATION
        assert _categorize_tool_to_category("modify") == ActionCategory.MODIFICATION
        assert _categorize_tool_to_category("delete") == ActionCategory.MODIFICATION
        assert _categorize_tool_to_category("create") == ActionCategory.MODIFICATION

    def test_unknown_tool_defaults_to_query(self):
        """Unknown tools default to QUERY (safe category)."""
        from hololoom.orchestrator.stages.steps_7_9 import _categorize_tool_to_category
        from hololoom.alignment.safety_guardrails import ActionCategory

        assert _categorize_tool_to_category("unknown_tool") == ActionCategory.QUERY
        assert _categorize_tool_to_category("custom_action") == ActionCategory.QUERY


# ============================================================================
# Test: execute_step9_spacetime_fabric — coverage gaps
# ============================================================================

class TestStep9SpacetimeFabric:

    def _make_ctx_for_step9(self, ctx):
        """Prepare a ctx with the minimum fields for step 9."""
        ctx.features = Mock(motifs=[], metrics={'spectral': None})
        ctx.action_plan = Mock(adapter="default")
        ctx.collapse_result = Mock(tool="answer", confidence=0.85, bandit_stats={})
        ctx.tool_result = {"result": "The answer is 42", "artifacts": []}
        ctx.safety_blocked = False
        ctx.safety_decision = None
        ctx.warp_space = None
        ctx.provenance = None
        ctx.complexity = None
        return ctx

    @pytest.mark.asyncio
    async def test_basic_spacetime_assembly(self, ctx):
        """Step 9 assembles Spacetime from tool_result."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        cfg = Mock()

        result = await execute_step9_spacetime_fabric(ctx, cfg)

        assert result.spacetime is not None
        assert result.spacetime.response == "The answer is 42"
        assert result.spacetime.confidence == 0.85
        assert result.spacetime.tool_used == "answer"

    @pytest.mark.asyncio
    async def test_safety_blocked_spacetime(self, ctx):
        """Step 9 creates blocked Spacetime when safety_blocked=True."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        ctx.safety_blocked = True
        ctx.safety_decision = Mock(
            reason="Risky operation",
            risk_level=Mock(value="HIGH"),
            safety_score=0.2,
        )
        cfg = Mock()

        result = await execute_step9_spacetime_fabric(ctx, cfg)

        assert result.spacetime is not None
        assert result.spacetime.confidence == 0.0
        assert "blocked" in result.spacetime.response.lower()
        assert result.spacetime.metadata.get('safety_blocked') is True

    @pytest.mark.asyncio
    async def test_semantic_cache_stats(self, ctx):
        """Step 9 includes semantic cache stats in metadata."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        cfg = Mock()

        semantic_cache = Mock()
        semantic_cache.get_stats.return_value = {
            'cache_hit_rate': 0.75,
            'hot_hits': 10,
            'warm_hits': 5,
            'cold_misses': 3,
            'estimated_speedup': 2.1,
        }

        result = await execute_step9_spacetime_fabric(ctx, cfg, semantic_cache=semantic_cache)

        cache_meta = result.spacetime.metadata.get('semantic_cache', {})
        assert cache_meta['enabled'] is True
        assert cache_meta['hit_rate'] == 0.75

    @pytest.mark.asyncio
    async def test_semantic_cache_exception(self, ctx):
        """Step 9 handles semantic cache exception gracefully (lines 448-449)."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        cfg = Mock()

        semantic_cache = Mock()
        semantic_cache.get_stats.side_effect = RuntimeError("cache broken")

        result = await execute_step9_spacetime_fabric(ctx, cfg, semantic_cache=semantic_cache)

        cache_meta = result.spacetime.metadata.get('semantic_cache', {})
        assert cache_meta['enabled'] is False

    @pytest.mark.asyncio
    async def test_awareness_context_injection(self, ctx):
        """Step 9 injects awareness context into metadata."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        cfg = Mock()

        awareness = {
            'activation_level': 0.75,
            'coherence': 0.88,
            'active_nodes': 12,
            'shift_detected': True,
            'semantic_position': [0.1, 0.2],
            'perception_time_ms': 5.3,
        }

        result = await execute_step9_spacetime_fabric(
            ctx, cfg, awareness_context=awareness,
        )

        aw_meta = result.spacetime.metadata.get('awareness', {})
        assert aw_meta['activation_level'] == 0.75
        assert aw_meta['shift_detected'] is True

    @pytest.mark.asyncio
    async def test_provenance_attachment(self, ctx):
        """Step 9 attaches provenance with stage timings (lines 508-524)."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        ctx.complexity = Mock(value="FAST")
        ctx.provenance = Mock()
        ctx.provenance.add_protocol_call = Mock()
        ctx.provenance.add_shuttle_event = Mock()
        ctx.stage_timings = {"convergence": 5.0, "tool_execution": 10.0}
        cfg = Mock()

        result = await execute_step9_spacetime_fabric(ctx, cfg)

        assert result.spacetime.complexity is not None
        assert result.spacetime.provenance is not None
        # Should have called add_protocol_call for each stage timing
        assert ctx.provenance.add_protocol_call.call_count >= 2
        ctx.provenance.add_shuttle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_warp_detension(self, ctx):
        """Step 9 collapses warp space (detension)."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        ctx.warp_space = Mock()
        ctx.warp_space.collapse.return_value = ["update1", "update2"]
        cfg = Mock()

        result = await execute_step9_spacetime_fabric(ctx, cfg)

        ctx.warp_space.collapse.assert_called_once()
        assert any(op[1] == "detension" for op in result.warp_operations)

    @pytest.mark.asyncio
    async def test_dashboard_constructor(self, ctx):
        """Step 9 generates dashboard when constructor provided."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        cfg = Mock()

        dashboard = Mock()
        dashboard.panels = [Mock(), Mock()]
        dashboard.layout = Mock(value="grid")

        constructor = Mock()
        constructor.construct.return_value = dashboard

        result = await execute_step9_spacetime_fabric(
            ctx, cfg, dashboard_constructor=constructor,
        )

        constructor.construct.assert_called_once()
        assert result.spacetime.metadata.get('dashboard') is dashboard

    @pytest.mark.asyncio
    async def test_dashboard_constructor_failure(self, ctx):
        """Step 9 handles dashboard failure gracefully."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        cfg = Mock()

        constructor = Mock()
        constructor.construct.side_effect = RuntimeError("dashboard crash")

        result = await execute_step9_spacetime_fabric(
            ctx, cfg, dashboard_constructor=constructor,
        )

        assert result.spacetime is not None
        assert 'dashboard' not in result.spacetime.metadata

    @pytest.mark.asyncio
    async def test_emit_stage_events(self, ctx):
        """Step 9 emits start, end, and complete events."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        cfg = Mock()

        events = []
        def emit(step, name, dur):
            events.append((step, name, dur))

        result = await execute_step9_spacetime_fabric(ctx, cfg, emit_stage_event=emit)

        assert events[0] == (9, "Spacetime Fabric", None)
        assert events[1][0] == 9
        assert events[2] == (0, "Complete", None)

    @pytest.mark.asyncio
    async def test_no_tool_result(self, ctx):
        """Step 9 handles None tool_result."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = self._make_ctx_for_step9(ctx)
        ctx.tool_result = None
        cfg = Mock()

        result = await execute_step9_spacetime_fabric(ctx, cfg)

        assert result.spacetime.response == 'No response'


# ============================================================================
# Test: Step 8 safety gating branches
# ============================================================================

class TestStep8SafetyGating:

    @pytest.mark.asyncio
    async def test_safety_blocked(self, ctx):
        """Step 8 blocks execution when guardrails reject."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step8_tool_execution
        from hololoom.alignment.safety_guardrails import ActionRequest, ActionCategory
        from hololoom.alignment.audit_trail import DecisionType, OutcomeType

        ctx.collapse_result = Mock(tool="execute", confidence=0.7)
        ctx.complexity = Mock(value="FUSED")

        tool_executor = AsyncMock()

        guardrails = Mock()
        safety_decision = Mock(
            allowed=False,
            reason="Too risky",
            risk_level=Mock(value="HIGH"),
            safety_score=0.1,
        )
        guardrails.evaluate.return_value = safety_decision

        result = await execute_step8_tool_execution(ctx, tool_executor, guardrails=guardrails)

        assert result.safety_blocked is True
        tool_executor.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_safety_approved_with_audit(self, ctx):
        """Step 8 logs to audit trail when safety approves."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step8_tool_execution

        ctx.collapse_result = Mock(tool="answer", confidence=0.9)
        ctx.complexity = Mock(value="FAST")

        tool_executor = AsyncMock()
        tool_executor.execute.return_value = {"result": "approved answer"}

        guardrails = Mock()
        safety_decision = Mock(
            allowed=True,
            reason="Safe",
            risk_level=Mock(value="LOW"),
            safety_score=0.95,
        )
        guardrails.evaluate.return_value = safety_decision

        audit_trail = Mock()

        result = await execute_step8_tool_execution(
            ctx, tool_executor, guardrails=guardrails, audit_trail=audit_trail,
        )

        assert result.safety_blocked is False
        assert result.tool_result == {"result": "approved answer"}
        audit_trail.log_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_safety_service_failure_degrades(self, ctx):
        """Step 8 continues with tool execution when safety service fails."""
        from hololoom.orchestrator.stages.steps_7_9 import execute_step8_tool_execution

        ctx.collapse_result = Mock(tool="answer", confidence=0.9)

        tool_executor = AsyncMock()
        tool_executor.execute.return_value = {"result": "degraded but working"}

        guardrails = Mock()
        guardrails.evaluate.side_effect = ConnectionError("safety service down")

        result = await execute_step8_tool_execution(ctx, tool_executor, guardrails=guardrails)

        assert result.tool_result == {"result": "degraded but working"}
        assert any("Safety gating error" in w for w in result.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
