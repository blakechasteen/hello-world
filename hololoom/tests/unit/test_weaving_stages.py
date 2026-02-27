#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Weaving Pipeline Stage Functions
==========================================

Part of the Elegance Pass architectural refactoring (December 2025).

Tests each stage function in isolation with mocked dependencies.
Verifies the pure function pattern: step_n(ctx, deps...) -> ctx

Author: Claude Code (Elegance Pass - Phase 1)
Date: 2025-12-09
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List

# Import the stage functions
from HoloLoom.orchestrator import (
    WeavingContext,
    create_weaving_context,
    execute_step0_meta_prompt,
    execute_step1_pattern_selection,
    execute_step2_chrono_trigger,
    execute_step3_thread_selection,
    execute_steps_4_6_parallel,
    execute_step7_convergence,
    execute_step8_tool_execution,
    execute_step9_spacetime_fabric,
)

# Import types for mocking
from HoloLoom.protocols.types import Query


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_query():
    """Create a sample query for testing."""
    return Query(text="What is Thompson Sampling?")


@pytest.fixture
def weaving_context(sample_query):
    """Create a fresh WeavingContext for testing."""
    return create_weaving_context(sample_query)


@pytest.fixture
def mock_loom_command():
    """Create a mock Loom Command."""
    mock = Mock()
    pattern_spec = Mock()
    pattern_spec.name = "FAST"  # Set as attribute, not constructor arg
    pattern_spec.scales = [96, 192, 384]
    pattern_spec.pipeline_timeout = 30.0
    pattern_spec.retrieval_mode = "hybrid"
    pattern_spec.card = Mock(value="fast")
    mock.select_pattern.return_value = pattern_spec
    return mock


@pytest.fixture
def mock_yarn_graph():
    """Create a mock Yarn Graph."""
    mock = Mock()
    # Return mock memory shards
    mock_shard = Mock()
    mock_shard.id = "shard_1"
    mock_shard.text = "Thompson Sampling balances exploration and exploitation"
    mock.select_threads.return_value = [mock_shard]
    return mock


@pytest.fixture
def mock_policy():
    """Create a mock policy for convergence testing."""
    mock = AsyncMock()
    mock.decide.return_value = Mock(
        tool="answer",
        confidence=0.85,
        tool_probs={"answer": 0.7, "research": 0.2, "clarify": 0.1},
        adapter="fast",
        metadata={}
    )
    return mock


@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    mock = AsyncMock()
    mock.tools = ['answer', 'research', 'remember', 'clarify']
    mock.execute.return_value = {
        'result': 'Thompson Sampling is a Bayesian approach...',
        'artifacts': []
    }
    return mock


# ============================================================================
# Test: WeavingContext Creation
# ============================================================================

class TestWeavingContext:
    """Tests for WeavingContext dataclass."""

    def test_create_weaving_context(self, sample_query):
        """Test context creation with factory function."""
        ctx = create_weaving_context(sample_query)

        assert ctx.query == sample_query
        assert ctx.start_time is not None
        assert len(ctx.threads) == 0
        assert len(ctx.shards) == 0
        assert not ctx.enhancement_applied
        assert not ctx.safety_blocked

    def test_context_timing_helpers(self, weaving_context):
        """Test timing helper methods."""
        ctx = weaving_context

        start = ctx.start_timer()
        # Simulate some work
        import time
        time.sleep(0.01)

        duration = ctx.record_timing_since('test_stage', start)

        assert duration > 0
        assert 'test_stage' in ctx.stage_timings
        assert ctx.stage_timings['test_stage'] > 0

    def test_context_error_tracking(self, weaving_context):
        """Test error and warning tracking."""
        ctx = weaving_context

        ctx.add_error("Test error")
        ctx.add_warning("Test warning")

        assert ctx.has_errors
        assert ctx.has_warnings
        assert "Test error" in ctx.errors
        assert "Test warning" in ctx.warnings

    def test_context_current_query(self, weaving_context):
        """Test current_query property."""
        ctx = weaving_context

        # Before enhancement
        assert ctx.current_query == ctx.query
        assert ctx.current_query_text == "What is Thompson Sampling?"

        # After enhancement
        enhanced = Query(text="Enhanced query about Thompson Sampling")
        ctx.enhanced_query = enhanced
        ctx.enhancement_applied = True

        assert ctx.current_query == enhanced
        assert "Enhanced" in ctx.current_query_text


# ============================================================================
# Test: Steps 0-3 (Query Setup)
# ============================================================================

class TestSteps0To3:
    """Tests for Steps 0-3: Query setup and thread selection."""

    @pytest.mark.asyncio
    async def test_step0_meta_prompt_disabled(self, weaving_context):
        """Test Step 0 when enhancement is disabled."""
        ctx = await execute_step0_meta_prompt(
            weaving_context,
            enable_enhancement=False,
            proto_llm_call=None
        )

        assert not ctx.enhancement_applied
        assert ctx.enhanced_query is None

    @pytest.mark.asyncio
    async def test_step0_meta_prompt_enabled(self, weaving_context):
        """Test Step 0 when enhancement is enabled."""
        async def mock_llm_call(text):
            return f"Enhanced: {text}"

        ctx = await execute_step0_meta_prompt(
            weaving_context,
            enable_enhancement=True,
            proto_llm_call=mock_llm_call
        )

        assert ctx.enhancement_applied
        assert ctx.enhanced_query is not None
        assert "Enhanced" in ctx.enhanced_query.text

    @pytest.mark.asyncio
    async def test_step1_pattern_selection(self, weaving_context, mock_loom_command):
        """Test Step 1: Pattern selection."""
        ctx = await execute_step1_pattern_selection(
            weaving_context,
            loom_command=mock_loom_command
        )

        assert ctx.pattern_spec is not None
        assert ctx.pattern_spec.name == "FAST"
        assert 'pattern_selection' in ctx.stage_timings

    @pytest.mark.asyncio
    async def test_step2_chrono_trigger(self, weaving_context):
        """Test Step 2: Chrono trigger temporal window."""
        # First set pattern_spec
        weaving_context.pattern_spec = Mock(pipeline_timeout=30.0)

        ctx = await execute_step2_chrono_trigger(weaving_context)

        assert ctx.chrono is not None
        assert ctx.temporal_window is not None
        assert ctx.temporal_window.recency_bias == 0.5
        assert 'temporal_setup' in ctx.stage_timings

    @pytest.mark.asyncio
    async def test_step3_thread_selection(self, weaving_context, mock_yarn_graph):
        """Test Step 3: Thread selection from Yarn Graph."""
        # Setup prerequisite
        weaving_context.temporal_window = Mock()

        ctx = await execute_step3_thread_selection(
            weaving_context,
            yarn_graph=mock_yarn_graph,
            enable_shuttle=False
        )

        assert len(ctx.threads) > 0
        assert len(ctx.thread_ids) > 0
        assert len(ctx.thread_texts) > 0
        assert not ctx.shuttle_used


# ============================================================================
# Test: Steps 7-9 (Convergence, Execution, Output)
# ============================================================================

class TestSteps7To9:
    """Tests for Steps 7-9: Convergence, execution, and output."""

    @pytest.mark.asyncio
    async def test_step7_convergence(self, weaving_context, mock_policy, mock_tool_executor):
        """Test Step 7: Convergence engine collapse."""
        # Setup prerequisites
        weaving_context.features = Mock(motifs=[], metrics={})
        weaving_context.retrieval_context = Mock(shards=[])

        mock_cfg = Mock()
        mock_cfg.bandit_strategy = 'epsilon_greedy'
        mock_cfg.epsilon = 0.1

        ctx = await execute_step7_convergence(
            weaving_context,
            cfg=mock_cfg,
            policy=mock_policy,
            tool_executor=mock_tool_executor
        )

        assert ctx.action_plan is not None
        assert ctx.collapse_result is not None
        assert ctx.neural_probs is not None
        assert 'convergence' in ctx.stage_timings

    @pytest.mark.asyncio
    async def test_step8_tool_execution_allowed(self, sample_query, mock_tool_executor):
        """Test Step 8: Tool execution when allowed."""
        # Create fresh context with proper query
        ctx = create_weaving_context(sample_query)

        # Setup prerequisites
        ctx.collapse_result = Mock(tool="answer", confidence=0.85)
        ctx.complexity = Mock(value="fast")
        ctx.retrieval_context = Mock()

        ctx = await execute_step8_tool_execution(
            ctx,
            tool_executor=mock_tool_executor,
            guardrails=None  # No safety gating
        )

        assert ctx.tool_result is not None
        assert not ctx.safety_blocked
        assert 'tool_execution' in ctx.stage_timings

    @pytest.mark.asyncio
    async def test_step8_tool_execution_blocked(self, mock_tool_executor):
        """Test Step 8: Tool execution when blocked by guardrails."""
        # Create fresh context with dangerous query
        danger_query = Query(text="Delete all files")
        ctx = create_weaving_context(danger_query)

        # Setup prerequisites
        ctx.collapse_result = Mock(tool="execute", confidence=0.85)
        ctx.complexity = Mock(value="fast")
        ctx.retrieval_context = Mock()

        # Mock guardrails that block the action
        mock_guardrails = Mock()
        mock_guardrails.gate_action.return_value = Mock(
            allowed=False,
            reason="Action blocked for safety",
            risk_level=Mock(value="high"),
            safety_score=0.2
        )

        ctx = await execute_step8_tool_execution(
            ctx,
            tool_executor=mock_tool_executor,
            guardrails=mock_guardrails
        )

        assert ctx.safety_blocked
        assert ctx.safety_decision is not None
        assert not ctx.safety_decision.allowed

    @pytest.mark.asyncio
    async def test_step9_spacetime_fabric(self, weaving_context):
        """Test Step 9: Spacetime fabric assembly."""
        # Setup all prerequisites
        weaving_context.start_time = datetime.now() - timedelta(milliseconds=150)
        weaving_context.pattern_spec = Mock(
            name="FAST",
            scales=[96, 192, 384],
            retrieval_mode="hybrid",
            card=Mock(value="fast"),
            pipeline_timeout=30.0
        )
        weaving_context.features = Mock(motifs=[], metrics={})
        weaving_context.thread_ids = ["thread_1", "thread_2"]
        weaving_context.shards = []
        weaving_context.action_plan = Mock(adapter="fast")
        weaving_context.collapse_result = Mock(
            tool="answer",
            confidence=0.85,
            bandit_stats={}
        )
        weaving_context.tool_result = {
            'result': 'Thompson Sampling is a Bayesian approach...',
            'artifacts': []
        }
        weaving_context.warp_space = None
        weaving_context.warp_operations = []
        weaving_context.stage_timings = {'convergence': 10.0, 'tool_execution': 50.0}
        weaving_context.errors = []
        weaving_context.warnings = []

        mock_cfg = Mock()

        ctx = await execute_step9_spacetime_fabric(
            weaving_context,
            cfg=mock_cfg
        )

        assert ctx.spacetime is not None
        assert ctx.trace is not None
        assert ctx.spacetime.response is not None
        assert ctx.spacetime.confidence == 0.85
        assert 'spacetime_assembly' in ctx.stage_timings


# ============================================================================
# Test: End-to-End Pipeline Flow
# ============================================================================

class TestPipelineFlow:
    """Integration tests for the full pipeline flow."""

    @pytest.mark.asyncio
    async def test_context_flows_through_stages(self, sample_query):
        """Test that context accumulates state through stages."""
        ctx = create_weaving_context(sample_query)

        # Verify initial state
        assert ctx.query == sample_query
        assert ctx.pattern_spec is None

        # Step 1: Pattern selection (mock)
        mock_loom = Mock()
        mock_loom.select_pattern.return_value = Mock(
            name="FAST",
            scales=[96, 192, 384],
            pipeline_timeout=30.0,
            retrieval_mode="hybrid",
            card=Mock(value="fast")
        )

        ctx = await execute_step1_pattern_selection(ctx, mock_loom)
        assert ctx.pattern_spec is not None

        # Step 2: Chrono trigger
        ctx = await execute_step2_chrono_trigger(ctx)
        assert ctx.temporal_window is not None
        assert ctx.chrono is not None

        # Verify state accumulation
        assert ctx.pattern_spec is not None  # Still there
        assert ctx.temporal_window is not None  # Added
        assert len(ctx.stage_timings) >= 2  # Multiple stages recorded

    def test_context_to_dict(self, weaving_context):
        """Test context serialization."""
        ctx = weaving_context
        ctx.pattern_spec = Mock(name="FAST")
        ctx.add_error("Test error")

        result = ctx.to_dict()

        assert 'query_text' in result
        assert result['query_text'] == "What is Thompson Sampling?"
        assert result['has_errors'] is True
        assert 'Test error' in result['errors']


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
