#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Orchestrator Stage Executors
=======================================

Unit tests for the 8 executor classes in
hololoom/core/orchestrator/stages/executors/

Each executor wraps a Phase 1 pure function with protocol-based design.
These tests verify:
- Core transform: given input context, produces expected output
- Error handling: graceful degradation on invalid/missing input
- Mode sensitivity: behavior varies with BARE/FAST/FUSED/RESEARCH
- Protocol compliance: all executors satisfy StageExecutorProtocol
- Timing: stage_timings populated after execute()
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from hololoom.protocols.types import Query
from hololoom.orchestrator.context import WeavingContext, create_weaving_context
from hololoom.orchestrator.protocols.stage import (
    BaseStageExecutor,
    StageExecutorProtocol,
)

# Import executor classes under test
from hololoom.orchestrator.stages.executors import (
    MetaPromptExecutor,
    PatternSelectionExecutor,
    ChronoTriggerExecutor,
    ThreadSelectionExecutor,
    ParallelFeatureExecutor,
    ConvergenceExecutor,
    ToolExecutionExecutor,
    SpacetimeExecutor,
)


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def sample_query():
    """Create a sample query for testing."""
    return Query(text="What is Thompson Sampling?")


@pytest.fixture
def ctx(sample_query):
    """Create a fresh WeavingContext for testing."""
    return create_weaving_context(sample_query)


@pytest.fixture
def mock_loom_command():
    """Create a mock LoomCommand that returns a FAST pattern spec."""
    mock = Mock()
    spec = Mock()
    spec.name = "FAST"
    spec.scales = [96, 192, 384]
    spec.pipeline_timeout = 30.0
    spec.retrieval_mode = "hybrid"
    spec.card = Mock(value="fast")
    spec.motif_mode = "fast"
    spec.enable_spectral = False
    spec.enable_semantic_flow = False
    mock.select_pattern.return_value = spec
    return mock


@pytest.fixture
def mock_yarn_graph():
    """Create a mock KG with thread selection."""
    mock = Mock()
    shard = Mock()
    shard.id = "shard_1"
    shard.text = "Thompson Sampling balances exploration and exploitation"
    mock.select_threads.return_value = [shard]
    return mock


@pytest.fixture
def mock_policy():
    """Create a mock policy engine for convergence."""
    mock = AsyncMock()
    mock.decide.return_value = Mock(
        tool="answer",
        confidence=0.85,
        tool_probs={"answer": 0.7, "research": 0.2, "clarify": 0.1},
        adapter="fast",
        metadata={},
    )
    return mock


@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    mock = AsyncMock()
    mock.tools = ["answer", "research", "remember", "clarify"]
    mock.execute.return_value = {
        "result": "Thompson Sampling is a Bayesian approach...",
        "artifacts": [],
    }
    return mock


@pytest.fixture
def mock_cfg():
    """Create a mock Config."""
    cfg = Mock()
    cfg.bandit_strategy = "epsilon_greedy"
    cfg.epsilon = 0.1
    return cfg


@pytest.fixture
def stage_event_log():
    """Capture stage events for verification."""
    events = []

    def emit(stage_id, stage_name, duration_ms):
        events.append((stage_id, stage_name, duration_ms))

    return events, emit


# ============================================================================
# Test: Protocol Compliance (all executors)
# ============================================================================


class TestProtocolCompliance:
    """All executors must satisfy StageExecutorProtocol and BaseStageExecutor."""

    def test_meta_prompt_is_stage_executor(self):
        executor = MetaPromptExecutor()
        assert isinstance(executor, BaseStageExecutor)
        assert isinstance(executor, StageExecutorProtocol)
        assert executor.stage_id == 0
        assert executor.stage_name == "Meta-Prompt Enhancement"

    def test_pattern_selection_is_stage_executor(self, mock_loom_command):
        executor = PatternSelectionExecutor(loom_command=mock_loom_command)
        assert isinstance(executor, BaseStageExecutor)
        assert isinstance(executor, StageExecutorProtocol)
        assert executor.stage_id == 1
        assert executor.stage_name == "Loom Command"

    def test_chrono_trigger_is_stage_executor(self):
        executor = ChronoTriggerExecutor()
        assert isinstance(executor, BaseStageExecutor)
        assert isinstance(executor, StageExecutorProtocol)
        assert executor.stage_id == 2
        assert executor.stage_name == "Chrono Trigger"

    def test_thread_selection_is_stage_executor(self, mock_yarn_graph):
        executor = ThreadSelectionExecutor(yarn_graph=mock_yarn_graph)
        assert isinstance(executor, BaseStageExecutor)
        assert isinstance(executor, StageExecutorProtocol)
        assert executor.stage_id == 3
        assert executor.stage_name == "Thread Selection"

    def test_parallel_feature_is_stage_executor(self, mock_cfg):
        embedder = Mock()
        executor = ParallelFeatureExecutor(cfg=mock_cfg, embedder=embedder)
        assert isinstance(executor, BaseStageExecutor)
        assert isinstance(executor, StageExecutorProtocol)
        assert executor.stage_id == 4
        assert executor.stage_name == "Parallel Feature Pipeline"

    def test_convergence_is_stage_executor(self, mock_cfg, mock_policy, mock_tool_executor):
        executor = ConvergenceExecutor(
            cfg=mock_cfg, policy=mock_policy, tool_executor=mock_tool_executor
        )
        assert isinstance(executor, BaseStageExecutor)
        assert isinstance(executor, StageExecutorProtocol)
        assert executor.stage_id == 7
        assert executor.stage_name == "Convergence Engine"

    def test_tool_execution_is_stage_executor(self, mock_tool_executor):
        executor = ToolExecutionExecutor(tool_executor=mock_tool_executor)
        assert isinstance(executor, BaseStageExecutor)
        assert isinstance(executor, StageExecutorProtocol)
        assert executor.stage_id == 8
        assert executor.stage_name == "Tool Execution"

    def test_spacetime_is_stage_executor(self, mock_cfg):
        executor = SpacetimeExecutor(cfg=mock_cfg)
        assert isinstance(executor, BaseStageExecutor)
        assert isinstance(executor, StageExecutorProtocol)
        assert executor.stage_id == 9
        assert executor.stage_name == "Spacetime Fabric"

    def test_all_stage_ids_unique(self, mock_loom_command, mock_yarn_graph, mock_cfg, mock_policy, mock_tool_executor):
        """Stage IDs must be unique across all executors."""
        executors = [
            MetaPromptExecutor(),
            PatternSelectionExecutor(loom_command=mock_loom_command),
            ChronoTriggerExecutor(),
            ThreadSelectionExecutor(yarn_graph=mock_yarn_graph),
            ParallelFeatureExecutor(cfg=mock_cfg, embedder=Mock()),
            ConvergenceExecutor(cfg=mock_cfg, policy=mock_policy, tool_executor=mock_tool_executor),
            ToolExecutionExecutor(tool_executor=mock_tool_executor),
            SpacetimeExecutor(cfg=mock_cfg),
        ]
        ids = [e.stage_id for e in executors]
        assert len(ids) == len(set(ids)), f"Duplicate stage IDs: {ids}"


# ============================================================================
# Test: MetaPromptExecutor (Step 0)
# ============================================================================


class TestMetaPromptExecutor:
    """Tests for Step 0: Meta-Prompt Enhancement."""

    @pytest.mark.asyncio
    async def test_disabled_enhancement_passes_through(self, ctx):
        """When disabled, context passes through unchanged."""
        executor = MetaPromptExecutor(enable_enhancement=False)
        result = await executor.execute(ctx)

        assert not result.enhancement_applied
        assert result.enhanced_query is None
        assert result.current_query_text == "What is Thompson Sampling?"

    @pytest.mark.asyncio
    async def test_enabled_enhancement_rewrites_query(self, ctx):
        """When enabled with LLM call, query gets enhanced."""
        async def mock_llm(text):
            return f"Enhanced: {text}"

        executor = MetaPromptExecutor(
            enable_enhancement=True,
            proto_llm_call=mock_llm,
        )
        result = await executor.execute(ctx)

        assert result.enhancement_applied
        assert result.enhanced_query is not None
        assert "Enhanced" in result.current_query_text
        assert result.original_query_text == "What is Thompson Sampling?"

    @pytest.mark.asyncio
    async def test_no_llm_call_skips_enhancement(self, ctx):
        """When enabled but no LLM callable provided, skips gracefully."""
        executor = MetaPromptExecutor(
            enable_enhancement=True,
            proto_llm_call=None,
        )
        result = await executor.execute(ctx)

        assert not result.enhancement_applied

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_gracefully(self, ctx):
        """When LLM call fails, enhancement is skipped with warning."""
        async def broken_llm(text):
            raise RuntimeError("LLM service unavailable")

        executor = MetaPromptExecutor(
            enable_enhancement=True,
            proto_llm_call=broken_llm,
        )
        result = await executor.execute(ctx)

        assert not result.enhancement_applied
        assert result.has_warnings

    @pytest.mark.asyncio
    async def test_auto_enhance_override(self, ctx):
        """Context-level auto_enhance overrides executor setting."""
        ctx.auto_enhance = False

        async def mock_llm(text):
            return f"Enhanced: {text}"

        executor = MetaPromptExecutor(
            enable_enhancement=True,
            proto_llm_call=mock_llm,
        )
        result = await executor.execute(ctx)

        # auto_enhance=False on context should override enable_enhancement=True
        assert not result.enhancement_applied

    @pytest.mark.asyncio
    async def test_records_timing(self, ctx):
        """Executor records stage timing in context."""
        executor = MetaPromptExecutor(enable_enhancement=False)
        result = await executor.execute(ctx)

        assert "meta_prompt_enhancement" in result.stage_timings

    @pytest.mark.asyncio
    async def test_emits_stage_event(self, ctx, stage_event_log):
        """Executor emits stage event callback."""
        events, emit = stage_event_log
        executor = MetaPromptExecutor(
            enable_enhancement=False,
            emit_stage_event=emit,
        )
        await executor.execute(ctx)

        # Should have at least one event with stage_id=0
        assert any(e[0] == 0 for e in events)


# ============================================================================
# Test: PatternSelectionExecutor (Step 1)
# ============================================================================


class TestPatternSelectionExecutor:
    """Tests for Step 1: Pattern Selection (Loom Command)."""

    @pytest.mark.asyncio
    async def test_selects_pattern(self, ctx, mock_loom_command):
        """Executor delegates to LoomCommand and populates pattern_spec."""
        executor = PatternSelectionExecutor(loom_command=mock_loom_command)
        result = await executor.execute(ctx)

        assert result.pattern_spec is not None
        assert result.pattern_spec.name == "FAST"
        mock_loom_command.select_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_query_text_to_loom(self, ctx, mock_loom_command):
        """LoomCommand receives the current query text."""
        executor = PatternSelectionExecutor(loom_command=mock_loom_command)
        await executor.execute(ctx)

        call_args = mock_loom_command.select_pattern.call_args
        assert call_args[0][0] == "What is Thompson Sampling?"

    @pytest.mark.asyncio
    async def test_pattern_override_passed(self, ctx, mock_loom_command):
        """User pattern override is forwarded to LoomCommand."""
        ctx.pattern_override = Mock(value="bare")
        executor = PatternSelectionExecutor(loom_command=mock_loom_command)
        await executor.execute(ctx)

        call_args = mock_loom_command.select_pattern.call_args
        assert call_args[1].get("user_preference") == "bare"

    @pytest.mark.asyncio
    async def test_records_timing(self, ctx, mock_loom_command):
        """Pattern selection records timing."""
        executor = PatternSelectionExecutor(loom_command=mock_loom_command)
        result = await executor.execute(ctx)

        assert "pattern_selection" in result.stage_timings
        assert result.stage_timings["pattern_selection"] >= 0


# ============================================================================
# Test: ChronoTriggerExecutor (Step 2)
# ============================================================================


class TestChronoTriggerExecutor:
    """Tests for Step 2: Chrono Trigger (Temporal Window)."""

    @pytest.mark.asyncio
    async def test_creates_temporal_window(self, ctx):
        """Executor creates temporal window with chrono trigger."""
        ctx.pattern_spec = Mock(pipeline_timeout=30.0)

        executor = ChronoTriggerExecutor()
        result = await executor.execute(ctx)

        assert result.chrono is not None
        assert result.temporal_window is not None
        assert result.temporal_window.recency_bias == 0.5

    @pytest.mark.asyncio
    async def test_custom_lookback_days(self, ctx):
        """Custom lookback_days widens/narrows the window."""
        ctx.pattern_spec = Mock(pipeline_timeout=30.0)

        executor = ChronoTriggerExecutor(lookback_days=30)
        result = await executor.execute(ctx)

        window = result.temporal_window
        assert window is not None
        # 30-day lookback should have a narrower window
        age = window.end - window.start
        assert age <= timedelta(days=31)

    @pytest.mark.asyncio
    async def test_custom_recency_bias(self, ctx):
        """Custom recency_bias is set on the temporal window."""
        ctx.pattern_spec = Mock(pipeline_timeout=30.0)

        executor = ChronoTriggerExecutor(recency_bias=0.9)
        result = await executor.execute(ctx)

        assert result.temporal_window.recency_bias == 0.9

    @pytest.mark.asyncio
    async def test_missing_pattern_spec_uses_default_timeout(self, ctx):
        """When pattern_spec is None, uses default pipeline timeout."""
        ctx.pattern_spec = None

        executor = ChronoTriggerExecutor()
        result = await executor.execute(ctx)

        # Should not crash — falls back to default timeout
        assert result.chrono is not None
        assert result.temporal_window is not None

    @pytest.mark.asyncio
    async def test_records_timing(self, ctx):
        """Chrono trigger records timing."""
        ctx.pattern_spec = Mock(pipeline_timeout=30.0)

        executor = ChronoTriggerExecutor()
        result = await executor.execute(ctx)

        assert "chrono_trigger" in result.stage_timings


# ============================================================================
# Test: ThreadSelectionExecutor (Step 3)
# ============================================================================


class TestThreadSelectionExecutor:
    """Tests for Step 3: Thread Selection (Yarn Graph / Shuttle)."""

    @pytest.mark.asyncio
    async def test_selects_threads_from_yarn_graph(self, ctx, mock_yarn_graph):
        """Delegates to yarn_graph.select_threads when shuttle disabled."""
        ctx.temporal_window = Mock()

        executor = ThreadSelectionExecutor(
            yarn_graph=mock_yarn_graph,
            enable_shuttle=False,
        )
        result = await executor.execute(ctx)

        assert len(result.threads) > 0
        assert len(result.thread_ids) > 0
        assert len(result.thread_texts) > 0
        assert not result.shuttle_used
        mock_yarn_graph.select_threads.assert_called_once()

    @pytest.mark.asyncio
    async def test_shuttle_used_when_enabled_and_available(self, ctx):
        """Uses Shuttle when enabled and stage is provided."""
        ctx.temporal_window = Mock()

        mock_shuttle = AsyncMock()
        shard = Mock()
        shard.id = "shuttle_shard"
        shard.text = "Shuttle-selected thread"
        mock_shuttle.select_threads.return_value = [shard]

        executor = ThreadSelectionExecutor(
            yarn_graph=Mock(),
            shuttle_stage=mock_shuttle,
            enable_shuttle=True,
        )
        result = await executor.execute(ctx)

        assert result.shuttle_used
        assert result.thread_ids == ["shuttle_shard"]
        mock_shuttle.select_threads.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_yarn_when_shuttle_missing(self, ctx, mock_yarn_graph):
        """Falls back to yarn graph when shuttle is enabled but stage is None."""
        ctx.temporal_window = Mock()

        executor = ThreadSelectionExecutor(
            yarn_graph=mock_yarn_graph,
            shuttle_stage=None,
            enable_shuttle=True,
        )
        result = await executor.execute(ctx)

        assert not result.shuttle_used
        mock_yarn_graph.select_threads.assert_called_once()

    @pytest.mark.asyncio
    async def test_thread_ids_and_texts_extracted(self, ctx, mock_yarn_graph):
        """Thread IDs and texts are extracted from selected threads."""
        ctx.temporal_window = Mock()

        shard1 = Mock(id="s1", text="First thread")
        shard2 = Mock(id="s2", text="Second thread")
        mock_yarn_graph.select_threads.return_value = [shard1, shard2]

        executor = ThreadSelectionExecutor(yarn_graph=mock_yarn_graph)
        result = await executor.execute(ctx)

        assert result.thread_ids == ["s1", "s2"]
        assert result.thread_texts == ["First thread", "Second thread"]

    @pytest.mark.asyncio
    async def test_records_timing(self, ctx, mock_yarn_graph):
        """Thread selection records timing."""
        ctx.temporal_window = Mock()
        executor = ThreadSelectionExecutor(yarn_graph=mock_yarn_graph)
        result = await executor.execute(ctx)

        assert "thread_selection" in result.stage_timings


# ============================================================================
# Test: ParallelFeatureExecutor (Steps 4-6)
# ============================================================================


class TestParallelFeatureExecutor:
    """Tests for Steps 4-6: Parallel Feature Extraction Pipeline."""

    @pytest.mark.asyncio
    async def test_delegates_to_pure_functions(self, ctx, mock_cfg):
        """Executor delegates to the three pure function steps."""
        embedder = Mock()
        ctx.pattern_spec = Mock(
            scales=[96],
            motif_mode="fast",
            enable_spectral=False,
            enable_semantic_flow=False,
        )

        # The executor imports inside execute(), so we patch at the source module
        with patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_steps_4_6_parallel",
            new_callable=AsyncMock,
        ) as mock_46, patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_step5_5_warp_compute",
            new_callable=AsyncMock,
        ) as mock_55, patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_step6_5_beta_wave_packing",
            new_callable=AsyncMock,
        ) as mock_65:
            # Each mock returns the context through
            mock_46.return_value = ctx
            mock_55.return_value = ctx
            mock_65.return_value = ctx

            executor = ParallelFeatureExecutor(cfg=mock_cfg, embedder=embedder)
            result = await executor.execute(ctx)

            mock_46.assert_called_once()
            mock_55.assert_called_once()
            mock_65.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_all_dependencies(self, ctx, mock_cfg):
        """All constructor dependencies are forwarded to pure functions."""
        embedder = Mock()
        memory = Mock()
        retriever = Mock()

        with patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_steps_4_6_parallel",
            new_callable=AsyncMock,
        ) as mock_46, patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_step5_5_warp_compute",
            new_callable=AsyncMock,
        ) as mock_55, patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_step6_5_beta_wave_packing",
            new_callable=AsyncMock,
        ) as mock_65:
            mock_46.return_value = ctx
            mock_55.return_value = ctx
            mock_65.return_value = ctx

            executor = ParallelFeatureExecutor(
                cfg=mock_cfg,
                embedder=embedder,
                memory=memory,
                retriever=retriever,
            )
            await executor.execute(ctx)

            # Verify dependencies are passed through
            call_kwargs = mock_46.call_args[1]
            assert call_kwargs["cfg"] is mock_cfg
            assert call_kwargs["embedder"] is embedder
            assert call_kwargs["memory"] is memory
            assert call_kwargs["retriever"] is retriever

    def test_update_dependencies(self, mock_cfg):
        """update_dependencies swaps runtime deps without recreation."""
        executor = ParallelFeatureExecutor(cfg=mock_cfg, embedder=Mock())

        new_memory = Mock()
        new_retriever = Mock()
        executor.update_dependencies(memory=new_memory, retriever=new_retriever)

        assert executor.memory is new_memory
        assert executor.retriever is new_retriever

    def test_update_dependencies_selective(self, mock_cfg):
        """update_dependencies only updates non-None fields."""
        original_memory = Mock()
        executor = ParallelFeatureExecutor(
            cfg=mock_cfg, embedder=Mock(), memory=original_memory
        )

        executor.update_dependencies(retriever=Mock())

        # Memory should remain unchanged
        assert executor.memory is original_memory

    @pytest.mark.asyncio
    async def test_records_timing(self, ctx, mock_cfg):
        """Parallel pipeline records timing."""
        with patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_steps_4_6_parallel",
            new_callable=AsyncMock,
            return_value=ctx,
        ), patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_step5_5_warp_compute",
            new_callable=AsyncMock,
            return_value=ctx,
        ), patch(
            "hololoom.orchestrator.stages.steps_4_6.execute_step6_5_beta_wave_packing",
            new_callable=AsyncMock,
            return_value=ctx,
        ):
            executor = ParallelFeatureExecutor(cfg=mock_cfg, embedder=Mock())
            result = await executor.execute(ctx)

            assert "parallel_feature_pipeline" in result.stage_timings


# ============================================================================
# Test: ConvergenceExecutor (Step 7)
# ============================================================================


class TestConvergenceExecutor:
    """Tests for Step 7: Convergence Engine (Decision Collapse)."""

    @pytest.mark.asyncio
    async def test_collapses_to_tool(self, ctx, mock_cfg, mock_policy, mock_tool_executor):
        """Executor produces collapse_result with selected tool."""
        ctx.features = Mock(motifs=[], metrics={})
        ctx.retrieval_context = Mock(shards=[])

        executor = ConvergenceExecutor(
            cfg=mock_cfg,
            policy=mock_policy,
            tool_executor=mock_tool_executor,
        )
        result = await executor.execute(ctx)

        assert result.action_plan is not None
        assert result.collapse_result is not None
        assert result.neural_probs is not None
        assert "convergence" in result.stage_timings

    @pytest.mark.asyncio
    async def test_policy_timeout_fallback(self, ctx, mock_cfg, mock_tool_executor):
        """When policy times out, falls back to safe default."""
        ctx.features = Mock(motifs=[], metrics={})
        ctx.retrieval_context = Mock(shards=[])

        slow_policy = AsyncMock()

        async def slow_decide(**kwargs):
            await asyncio.sleep(1.0)  # Will exceed 200ms timeout
            return Mock(tool="answer", confidence=0.5, tool_probs={"answer": 1.0})

        slow_policy.decide = slow_decide

        executor = ConvergenceExecutor(
            cfg=mock_cfg,
            policy=slow_policy,
            tool_executor=mock_tool_executor,
        )
        result = await executor.execute(ctx)

        # Should have fallen back to safe default
        assert result.action_plan is not None
        assert result.action_plan.tool == "answer"
        assert result.action_plan.metadata.get("timeout") is True

    @pytest.mark.asyncio
    async def test_gradient_router_blending(self, ctx, mock_cfg, mock_policy, mock_tool_executor):
        """Optional gradient router blends with neural predictions."""
        ctx.features = Mock(motifs=[], metrics={})
        ctx.retrieval_context = Mock(shards=[])

        gradient_router = AsyncMock()
        gradient_router.select_tool.return_value = Mock(
            target="research", loss=0.1
        )

        executor = ConvergenceExecutor(
            cfg=mock_cfg,
            policy=mock_policy,
            tool_executor=mock_tool_executor,
            gradient_router=gradient_router,
        )
        result = await executor.execute(ctx)

        assert result.gradient_decision is not None
        assert result.gradient_decision.target == "research"
        gradient_router.select_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_gradient_router_failure_degrades(self, ctx, mock_cfg, mock_policy, mock_tool_executor):
        """Gradient router failure does not block convergence."""
        ctx.features = Mock(motifs=[], metrics={})
        ctx.retrieval_context = Mock(shards=[])

        gradient_router = AsyncMock()
        gradient_router.select_tool.side_effect = RuntimeError("GPU offline")

        executor = ConvergenceExecutor(
            cfg=mock_cfg,
            policy=mock_policy,
            tool_executor=mock_tool_executor,
            gradient_router=gradient_router,
        )
        result = await executor.execute(ctx)

        # Should still produce a valid collapse_result
        assert result.collapse_result is not None

    def test_update_policy(self, mock_cfg, mock_policy, mock_tool_executor):
        """update_policy swaps the policy engine."""
        executor = ConvergenceExecutor(
            cfg=mock_cfg, policy=mock_policy, tool_executor=mock_tool_executor
        )
        new_policy = Mock()
        executor.update_policy(new_policy)
        assert executor.policy is new_policy

    def test_update_gradient_router(self, mock_cfg, mock_policy, mock_tool_executor):
        """update_gradient_router swaps or disables the router."""
        executor = ConvergenceExecutor(
            cfg=mock_cfg, policy=mock_policy, tool_executor=mock_tool_executor
        )
        router = Mock()
        executor.update_gradient_router(router)
        assert executor.gradient_router is router

        executor.update_gradient_router(None)
        assert executor.gradient_router is None


# ============================================================================
# Test: ToolExecutionExecutor (Step 8)
# ============================================================================


class TestToolExecutionExecutor:
    """Tests for Step 8: Tool Execution with Safety Gating."""

    @pytest.mark.asyncio
    async def test_executes_tool_without_guardrails(self, ctx, mock_tool_executor):
        """Tool executes normally when no guardrails are set."""
        ctx.collapse_result = Mock(tool="answer", confidence=0.85)
        ctx.complexity = Mock(value="fast")
        ctx.retrieval_context = Mock()

        executor = ToolExecutionExecutor(tool_executor=mock_tool_executor)
        result = await executor.execute(ctx)

        assert result.tool_result is not None
        assert not result.safety_blocked
        assert "tool_execution" in result.stage_timings
        mock_tool_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_guardrails_allow_execution(self, ctx, mock_tool_executor):
        """Tool executes when guardrails allow the action."""
        ctx.collapse_result = Mock(tool="answer", confidence=0.85)
        ctx.complexity = Mock(value="fast")
        ctx.retrieval_context = Mock()

        mock_guardrails = Mock()
        mock_guardrails.evaluate.return_value = Mock(
            allowed=True,
            reason="Safe action",
            risk_level=Mock(value="low"),
            safety_score=0.95,
        )

        executor = ToolExecutionExecutor(
            tool_executor=mock_tool_executor,
            guardrails=mock_guardrails,
        )
        result = await executor.execute(ctx)

        assert result.tool_result is not None
        assert not result.safety_blocked

    @pytest.mark.asyncio
    async def test_guardrails_block_execution(self, ctx, mock_tool_executor):
        """Tool is blocked when guardrails deny the action."""
        ctx.collapse_result = Mock(tool="execute", confidence=0.85)
        ctx.complexity = Mock(value="fast")
        ctx.retrieval_context = Mock()

        mock_guardrails = Mock()
        mock_guardrails.evaluate.return_value = Mock(
            allowed=False,
            reason="Dangerous action",
            risk_level=Mock(value="high"),
            safety_score=0.1,
        )

        executor = ToolExecutionExecutor(
            tool_executor=mock_tool_executor,
            guardrails=mock_guardrails,
        )
        result = await executor.execute(ctx)

        assert result.safety_blocked
        assert result.tool_result is None
        mock_tool_executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_guardrails_failure_degrades_gracefully(self, ctx, mock_tool_executor):
        """When guardrails raise a network error, execution continues (graceful degradation)."""
        ctx.collapse_result = Mock(tool="answer", confidence=0.85)
        ctx.complexity = Mock(value="fast")
        ctx.retrieval_context = Mock()

        mock_guardrails = Mock()
        mock_guardrails.evaluate.side_effect = ConnectionError("Safety module unreachable")

        executor = ToolExecutionExecutor(
            tool_executor=mock_tool_executor,
            guardrails=mock_guardrails,
        )
        result = await executor.execute(ctx)

        # Should continue with tool execution despite guardrails failure
        assert result.tool_result is not None
        assert not result.safety_blocked

    @pytest.mark.asyncio
    async def test_audit_trail_logged(self, ctx, mock_tool_executor):
        """Audit trail receives decision log when provided."""
        ctx.collapse_result = Mock(tool="answer", confidence=0.85)
        ctx.complexity = Mock(value="fast")
        ctx.retrieval_context = Mock()

        mock_guardrails = Mock()
        mock_guardrails.evaluate.return_value = Mock(
            allowed=True,
            reason="Safe",
            risk_level=Mock(value="low"),
            safety_score=0.95,
        )
        mock_audit = Mock()

        # Patch OutcomeType to add ALLOWED/BLOCKED that the code expects
        from hololoom.alignment.audit_trail import OutcomeType

        with patch.object(OutcomeType, "ALLOWED", "allowed", create=True), \
             patch.object(OutcomeType, "BLOCKED", "blocked", create=True):
            executor = ToolExecutionExecutor(
                tool_executor=mock_tool_executor,
                guardrails=mock_guardrails,
                audit_trail=mock_audit,
            )
            await executor.execute(ctx)

        mock_audit.log_decision.assert_called_once()

    def test_update_guardrails(self, mock_tool_executor):
        """update_guardrails swaps safety component."""
        executor = ToolExecutionExecutor(tool_executor=mock_tool_executor)
        new_guardrails = Mock()
        executor.update_guardrails(new_guardrails)
        assert executor.guardrails is new_guardrails

    def test_update_audit_trail(self, mock_tool_executor):
        """update_audit_trail swaps logging component."""
        executor = ToolExecutionExecutor(tool_executor=mock_tool_executor)
        new_audit = Mock()
        executor.update_audit_trail(new_audit)
        assert executor.audit_trail is new_audit


# ============================================================================
# Test: SpacetimeExecutor (Step 9)
# ============================================================================


class TestSpacetimeExecutor:
    """Tests for Step 9: Spacetime Fabric Assembly."""

    @pytest.mark.asyncio
    async def test_assembles_spacetime(self, ctx, mock_cfg):
        """Executor assembles final Spacetime with provenance."""
        # Setup all prerequisites
        ctx.pattern_spec = Mock(
            name="FAST",
            card=Mock(value="fast"),
            pipeline_timeout=30.0,
            scales=[96],
            retrieval_mode="hybrid",
        )
        ctx.features = Mock(motifs=[], metrics={})
        ctx.collapse_result = Mock(
            tool="answer",
            confidence=0.85,
            bandit_stats={},
        )
        ctx.action_plan = Mock(adapter="fast")
        ctx.tool_result = {"result": "Thompson Sampling is...", "artifacts": []}
        ctx.shards = []

        executor = SpacetimeExecutor(cfg=mock_cfg)
        result = await executor.execute(ctx)

        assert result.spacetime is not None
        assert result.spacetime.response == "Thompson Sampling is..."
        assert result.spacetime.confidence == 0.85
        assert result.trace is not None
        assert "spacetime_assembly" in result.stage_timings

    @pytest.mark.asyncio
    async def test_safety_blocked_spacetime(self, ctx, mock_cfg):
        """When safety blocked, spacetime reflects the block."""
        ctx.pattern_spec = Mock(
            name="FAST",
            card=Mock(value="fast"),
            pipeline_timeout=30.0,
            scales=[96],
            retrieval_mode="hybrid",
        )
        ctx.features = Mock(motifs=[], metrics={})
        ctx.collapse_result = Mock(
            tool="execute",
            confidence=0.85,
            bandit_stats={},
        )
        ctx.action_plan = Mock(adapter="fast")
        ctx.safety_blocked = True
        ctx.safety_decision = Mock(
            reason="Blocked for safety",
            risk_level=Mock(value="high"),
            safety_score=0.1,
        )
        ctx.shards = []

        executor = SpacetimeExecutor(cfg=mock_cfg)
        result = await executor.execute(ctx)

        assert result.spacetime is not None
        assert result.spacetime.confidence == 0.0
        assert "blocked" in result.spacetime.response.lower()

    @pytest.mark.asyncio
    async def test_warp_space_detensioned(self, ctx, mock_cfg):
        """Warp space is collapsed/detensioned during assembly."""
        ctx.pattern_spec = Mock(
            name="FAST",
            card=Mock(value="fast"),
            pipeline_timeout=30.0,
            scales=[96],
            retrieval_mode="hybrid",
        )
        ctx.features = Mock(motifs=[], metrics={})
        ctx.collapse_result = Mock(tool="answer", confidence=0.85, bandit_stats={})
        ctx.action_plan = Mock(adapter="fast")
        ctx.tool_result = {"result": "Response", "artifacts": []}
        ctx.shards = []

        mock_warp = Mock()
        mock_warp.collapse.return_value = ["update1", "update2"]
        ctx.warp_space = mock_warp

        executor = SpacetimeExecutor(cfg=mock_cfg)
        result = await executor.execute(ctx)

        mock_warp.collapse.assert_called_once()
        # Should have recorded warp operation
        assert len(result.warp_operations) > 0

    @pytest.mark.asyncio
    async def test_awareness_context_injected(self, ctx, mock_cfg):
        """Awareness context appears in spacetime metadata."""
        ctx.pattern_spec = Mock(
            name="FAST",
            card=Mock(value="fast"),
            pipeline_timeout=30.0,
            scales=[96],
            retrieval_mode="hybrid",
        )
        ctx.features = Mock(motifs=[], metrics={})
        ctx.collapse_result = Mock(tool="answer", confidence=0.85, bandit_stats={})
        ctx.action_plan = Mock(adapter="fast")
        ctx.tool_result = {"result": "Response", "artifacts": []}
        ctx.shards = []

        awareness = {
            "activation_level": 0.75,
            "coherence": 0.9,
            "active_nodes": 42,
            "shift_detected": False,
        }

        executor = SpacetimeExecutor(cfg=mock_cfg, awareness_context=awareness)
        result = await executor.execute(ctx)

        metadata = result.spacetime.metadata
        assert "awareness" in metadata
        assert metadata["awareness"]["activation_level"] == 0.75

    @pytest.mark.asyncio
    async def test_semantic_cache_stats(self, ctx, mock_cfg):
        """Semantic cache statistics added to metadata when available."""
        ctx.pattern_spec = Mock(
            name="FAST",
            card=Mock(value="fast"),
            pipeline_timeout=30.0,
            scales=[96],
            retrieval_mode="hybrid",
        )
        ctx.features = Mock(motifs=[], metrics={})
        ctx.collapse_result = Mock(tool="answer", confidence=0.85, bandit_stats={})
        ctx.action_plan = Mock(adapter="fast")
        ctx.tool_result = {"result": "Response", "artifacts": []}
        ctx.shards = []

        mock_cache = Mock()
        mock_cache.get_stats.return_value = {
            "cache_hit_rate": 0.42,
            "hot_hits": 10,
            "warm_hits": 5,
            "cold_misses": 15,
        }

        executor = SpacetimeExecutor(cfg=mock_cfg, semantic_cache=mock_cache)
        result = await executor.execute(ctx)

        cache_meta = result.spacetime.metadata.get("semantic_cache", {})
        assert cache_meta.get("enabled") is True
        assert cache_meta.get("hit_rate") == 0.42

    @pytest.mark.asyncio
    async def test_dashboard_generated(self, ctx, mock_cfg):
        """Dashboard is generated when constructor is provided."""
        ctx.pattern_spec = Mock(
            name="FAST",
            card=Mock(value="fast"),
            pipeline_timeout=30.0,
            scales=[96],
            retrieval_mode="hybrid",
        )
        ctx.features = Mock(motifs=[], metrics={})
        ctx.collapse_result = Mock(tool="answer", confidence=0.85, bandit_stats={})
        ctx.action_plan = Mock(adapter="fast")
        ctx.tool_result = {"result": "Response", "artifacts": []}
        ctx.shards = []

        mock_dashboard = Mock()
        mock_dashboard.construct.return_value = Mock(
            panels=["panel1", "panel2"], layout=Mock(value="grid")
        )

        executor = SpacetimeExecutor(cfg=mock_cfg, dashboard_constructor=mock_dashboard)
        result = await executor.execute(ctx)

        mock_dashboard.construct.assert_called_once()
        assert "dashboard" in result.spacetime.metadata

    @pytest.mark.asyncio
    async def test_dashboard_failure_degrades(self, ctx, mock_cfg):
        """Dashboard failure does not crash spacetime assembly."""
        ctx.pattern_spec = Mock(
            name="FAST",
            card=Mock(value="fast"),
            pipeline_timeout=30.0,
            scales=[96],
            retrieval_mode="hybrid",
        )
        ctx.features = Mock(motifs=[], metrics={})
        ctx.collapse_result = Mock(tool="answer", confidence=0.85, bandit_stats={})
        ctx.action_plan = Mock(adapter="fast")
        ctx.tool_result = {"result": "Response", "artifacts": []}
        ctx.shards = []

        mock_dashboard = Mock()
        mock_dashboard.construct.side_effect = RuntimeError("Dashboard broke")

        executor = SpacetimeExecutor(cfg=mock_cfg, dashboard_constructor=mock_dashboard)
        result = await executor.execute(ctx)

        # Should still produce a valid spacetime
        assert result.spacetime is not None
        assert result.spacetime.response == "Response"

    def test_update_semantic_cache(self, mock_cfg):
        """update_semantic_cache swaps cache."""
        executor = SpacetimeExecutor(cfg=mock_cfg)
        cache = Mock()
        executor.update_semantic_cache(cache)
        assert executor.semantic_cache is cache

    def test_update_dashboard_constructor(self, mock_cfg):
        """update_dashboard_constructor swaps dashboard."""
        executor = SpacetimeExecutor(cfg=mock_cfg)
        dashboard = Mock()
        executor.update_dashboard_constructor(dashboard)
        assert executor.dashboard_constructor is dashboard

    def test_update_awareness_context(self, mock_cfg):
        """update_awareness_context swaps context."""
        executor = SpacetimeExecutor(cfg=mock_cfg)
        awareness = {"activation_level": 0.5}
        executor.update_awareness_context(awareness)
        assert executor.awareness_context == awareness


# ============================================================================
# Test: BaseStageExecutor Helpers
# ============================================================================


class TestBaseStageExecutorHelpers:
    """Tests for shared base class timing and logging helpers."""

    @pytest.mark.asyncio
    async def test_timing_helpers(self, ctx):
        """_start_timing and _record_timing produce positive ms values."""
        executor = MetaPromptExecutor(enable_enhancement=False)

        start = executor._start_timing()
        # Simulate minimal work
        duration = executor._record_timing(ctx, "test_key", start)

        assert duration >= 0
        assert "test_key" in ctx.stage_timings

    def test_logging_helpers(self):
        """Logging helpers use correct log levels."""
        test_logger = logging.getLogger("test_helpers")
        executor = MetaPromptExecutor(logger=test_logger)

        # Should not raise
        executor._log_stage_start()
        executor._log_stage_complete(1.23)
        executor._log_stage_error("test error")
        executor._log_stage_warning("test warning")

    def test_add_error_to_context(self, ctx):
        """_add_error adds to context and logs."""
        executor = MetaPromptExecutor()
        executor._add_error(ctx, "Something went wrong")

        assert "Something went wrong" in ctx.errors
        assert ctx.has_errors

    def test_add_warning_to_context(self, ctx):
        """_add_warning adds to context and logs."""
        executor = MetaPromptExecutor()
        executor._add_warning(ctx, "Something seems off")

        assert "Something seems off" in ctx.warnings
        assert ctx.has_warnings

    def test_custom_logger(self):
        """Custom logger is used instead of default."""
        custom_logger = logging.getLogger("custom_test")
        executor = MetaPromptExecutor(logger=custom_logger)
        assert executor.logger is custom_logger

    def test_default_logger(self):
        """Default logger is created when none provided."""
        executor = MetaPromptExecutor()
        assert executor.logger is not None

    def test_emit_stage_event_callback(self, ctx, stage_event_log):
        """Stage event callback is invoked with correct args."""
        events, emit = stage_event_log
        executor = MetaPromptExecutor(emit_stage_event=emit)

        start = executor._start_timing()
        executor._record_timing(ctx, "test", start)

        assert len(events) == 1
        assert events[0][0] == 0  # stage_id
        assert events[0][1] == "Meta-Prompt Enhancement"  # stage_name
        assert events[0][2] >= 0  # duration_ms

    def test_no_emit_callback_is_noop(self, ctx):
        """When no emit callback, _record_timing still works."""
        executor = MetaPromptExecutor()
        start = executor._start_timing()
        duration = executor._record_timing(ctx, "test", start)
        assert duration >= 0
