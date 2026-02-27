#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Executor Pipeline (Phase 2)
==========================================

Tests for the ExecutorPipeline class and factory functions
created as part of the Elegance Pass refactoring.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass

from HoloLoom.orchestrator import (
    ExecutorPipeline,
    WeavingContext,
    create_weaving_context,
    StageExecutorProtocol,
    BaseStageExecutor,
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
# Mock Executor for Testing
# ============================================================================

class MockExecutor(BaseStageExecutor):
    """Mock executor for testing pipeline behavior."""

    def __init__(self, stage_id: int, stage_name: str = "Mock Stage"):
        super().__init__()
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.execute_count = 0

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        self.execute_count += 1
        ctx.record_timing(f'mock_stage_{self.stage_id}', 1.0)
        return ctx


# ============================================================================
# Test ExecutorPipeline
# ============================================================================

class TestExecutorPipeline:
    """Tests for ExecutorPipeline class."""

    @pytest.mark.asyncio
    async def test_pipeline_executes_all_stages(self):
        """Pipeline should execute all stages in order."""
        # Create mock executors
        executors = [
            MockExecutor(stage_id=0, stage_name="Stage 0"),
            MockExecutor(stage_id=1, stage_name="Stage 1"),
            MockExecutor(stage_id=2, stage_name="Stage 2"),
        ]

        # Create pipeline
        pipeline = ExecutorPipeline(executors)

        # Create context
        mock_query = Mock()
        mock_query.text = "Test query"
        ctx = create_weaving_context(mock_query)

        # Execute pipeline
        result = await pipeline.execute(ctx)

        # Verify all executors were called
        for executor in executors:
            assert executor.execute_count == 1

        # Verify timings were recorded
        assert 'mock_stage_0' in result.stage_timings
        assert 'mock_stage_1' in result.stage_timings
        assert 'mock_stage_2' in result.stage_timings

    def test_pipeline_length(self):
        """Pipeline should report correct length."""
        executors = [MockExecutor(i) for i in range(5)]
        pipeline = ExecutorPipeline(executors)
        assert len(pipeline) == 5

    def test_pipeline_iteration(self):
        """Pipeline should support iteration."""
        executors = [MockExecutor(i) for i in range(3)]
        pipeline = ExecutorPipeline(executors)

        collected = list(pipeline)
        assert len(collected) == 3
        assert all(isinstance(e, MockExecutor) for e in collected)

    def test_pipeline_indexing(self):
        """Pipeline should support indexing."""
        executors = [MockExecutor(i) for i in range(4)]
        pipeline = ExecutorPipeline(executors)

        assert pipeline[0].stage_id == 0
        assert pipeline[2].stage_id == 2
        assert pipeline[-1].stage_id == 3

    def test_pipeline_stage_ids(self):
        """Pipeline should return stage IDs in order."""
        executors = [MockExecutor(i) for i in [0, 1, 4, 7, 9]]
        pipeline = ExecutorPipeline(executors)

        assert pipeline.stage_ids() == [0, 1, 4, 7, 9]

    def test_pipeline_stage_names(self):
        """Pipeline should return stage names in order."""
        executors = [
            MockExecutor(0, "First"),
            MockExecutor(1, "Second"),
            MockExecutor(2, "Third"),
        ]
        pipeline = ExecutorPipeline(executors)

        assert pipeline.stage_names() == ["First", "Second", "Third"]


# ============================================================================
# Test Executor Classes
# ============================================================================

class TestExecutorClasses:
    """Tests for individual executor classes."""

    def test_meta_prompt_executor_attributes(self):
        """MetaPromptExecutor should have correct stage attributes."""
        executor = MetaPromptExecutor()
        assert executor.stage_id == 0
        assert executor.stage_name == "Meta-Prompt Enhancement"

    def test_pattern_selection_executor_attributes(self):
        """PatternSelectionExecutor should have correct stage attributes."""
        mock_loom = Mock()
        executor = PatternSelectionExecutor(loom_command=mock_loom)
        assert executor.stage_id == 1
        assert executor.stage_name == "Loom Command"

    def test_chrono_trigger_executor_attributes(self):
        """ChronoTriggerExecutor should have correct stage attributes."""
        executor = ChronoTriggerExecutor()
        assert executor.stage_id == 2
        assert executor.stage_name == "Chrono Trigger"

    def test_thread_selection_executor_attributes(self):
        """ThreadSelectionExecutor should have correct stage attributes."""
        mock_kg = Mock()
        executor = ThreadSelectionExecutor(yarn_graph=mock_kg)
        assert executor.stage_id == 3
        assert executor.stage_name == "Thread Selection"

    def test_parallel_feature_executor_attributes(self):
        """ParallelFeatureExecutor should have correct stage attributes."""
        mock_cfg = Mock()
        mock_embedder = Mock()
        executor = ParallelFeatureExecutor(cfg=mock_cfg, embedder=mock_embedder)
        assert executor.stage_id == 4
        assert executor.stage_name == "Parallel Feature Pipeline"

    def test_convergence_executor_attributes(self):
        """ConvergenceExecutor should have correct stage attributes."""
        mock_cfg = Mock()
        mock_policy = Mock()
        mock_tool_executor = Mock()
        executor = ConvergenceExecutor(
            cfg=mock_cfg,
            policy=mock_policy,
            tool_executor=mock_tool_executor
        )
        assert executor.stage_id == 7
        assert executor.stage_name == "Convergence Engine"

    def test_tool_execution_executor_attributes(self):
        """ToolExecutionExecutor should have correct stage attributes."""
        mock_tool_executor = Mock()
        executor = ToolExecutionExecutor(tool_executor=mock_tool_executor)
        assert executor.stage_id == 8
        assert executor.stage_name == "Tool Execution"

    def test_spacetime_executor_attributes(self):
        """SpacetimeExecutor should have correct stage attributes."""
        mock_cfg = Mock()
        executor = SpacetimeExecutor(cfg=mock_cfg)
        assert executor.stage_id == 9
        assert executor.stage_name == "Spacetime Fabric"


# ============================================================================
# Test Protocol Compliance
# ============================================================================

class TestProtocolCompliance:
    """Tests for StageExecutorProtocol compliance."""

    def test_executors_implement_protocol(self):
        """All executors should implement StageExecutorProtocol."""
        mock_cfg = Mock()
        mock_loom = Mock()
        mock_kg = Mock()
        mock_embedder = Mock()
        mock_policy = Mock()
        mock_tool_executor = Mock()

        executors = [
            MetaPromptExecutor(),
            PatternSelectionExecutor(loom_command=mock_loom),
            ChronoTriggerExecutor(),
            ThreadSelectionExecutor(yarn_graph=mock_kg),
            ParallelFeatureExecutor(cfg=mock_cfg, embedder=mock_embedder),
            ConvergenceExecutor(cfg=mock_cfg, policy=mock_policy, tool_executor=mock_tool_executor),
            ToolExecutionExecutor(tool_executor=mock_tool_executor),
            SpacetimeExecutor(cfg=mock_cfg),
        ]

        for executor in executors:
            # Check protocol attributes
            assert hasattr(executor, 'stage_id')
            assert hasattr(executor, 'stage_name')
            assert hasattr(executor, 'execute')
            assert isinstance(executor.stage_id, int)
            assert isinstance(executor.stage_name, str)

    def test_base_executor_timing_helpers(self):
        """BaseStageExecutor should provide timing helper methods."""
        executor = MockExecutor(0)

        # Test timing methods exist
        assert hasattr(executor, '_start_timing')
        assert hasattr(executor, '_record_timing')
        assert hasattr(executor, '_log_stage_start')
        assert hasattr(executor, '_log_stage_complete')

        # Test _start_timing returns a float
        start = executor._start_timing()
        assert isinstance(start, float)


# ============================================================================
# Test Pipeline with Real Executors (Mocked Dependencies)
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for pipeline with executor classes."""

    def test_create_pipeline_with_all_executors(self):
        """Should create a pipeline with all 8 executor types."""
        mock_cfg = Mock()
        mock_loom = Mock()
        mock_kg = Mock()
        mock_embedder = Mock()
        mock_policy = Mock()
        mock_tool_executor = Mock()

        executors = [
            MetaPromptExecutor(),
            PatternSelectionExecutor(loom_command=mock_loom),
            ChronoTriggerExecutor(),
            ThreadSelectionExecutor(yarn_graph=mock_kg),
            ParallelFeatureExecutor(cfg=mock_cfg, embedder=mock_embedder),
            ConvergenceExecutor(cfg=mock_cfg, policy=mock_policy, tool_executor=mock_tool_executor),
            ToolExecutionExecutor(tool_executor=mock_tool_executor),
            SpacetimeExecutor(cfg=mock_cfg),
        ]

        pipeline = ExecutorPipeline(executors)

        assert len(pipeline) == 8
        assert pipeline.stage_ids() == [0, 1, 2, 3, 4, 7, 8, 9]

    def test_pipeline_executor_order(self):
        """Pipeline should preserve executor order."""
        mock_cfg = Mock()
        mock_loom = Mock()
        mock_kg = Mock()
        mock_embedder = Mock()
        mock_policy = Mock()
        mock_tool_executor = Mock()

        executors = [
            SpacetimeExecutor(cfg=mock_cfg),  # Out of order
            MetaPromptExecutor(),
            ConvergenceExecutor(cfg=mock_cfg, policy=mock_policy, tool_executor=mock_tool_executor),
        ]

        pipeline = ExecutorPipeline(executors)

        # Order should be preserved as given (9, 0, 7)
        assert pipeline.stage_ids() == [9, 0, 7]


# ============================================================================
# Test Empty Pipeline
# ============================================================================

class TestEmptyPipeline:
    """Tests for edge cases with empty pipeline."""

    def test_empty_pipeline_length(self):
        """Empty pipeline should have length 0."""
        pipeline = ExecutorPipeline([])
        assert len(pipeline) == 0

    def test_empty_pipeline_stage_ids(self):
        """Empty pipeline should return empty stage IDs list."""
        pipeline = ExecutorPipeline([])
        assert pipeline.stage_ids() == []

    @pytest.mark.asyncio
    async def test_empty_pipeline_execute(self):
        """Empty pipeline should pass through context unchanged."""
        pipeline = ExecutorPipeline([])
        mock_query = Mock()
        mock_query.text = "Test"
        ctx = create_weaving_context(mock_query)

        result = await pipeline.execute(ctx)
        assert result is ctx  # Same context object
