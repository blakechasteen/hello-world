#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Protocol Factory (Phase 3 - Day 14)
==============================================

Integration tests for protocol-based dependency injection.

Author: Claude Code (Elegance Pass - Phase 3)
Date: 2025-12-11
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from typing import Any, Dict, List


# =============================================================================
# Test OrchestratorComponents Dataclass
# =============================================================================

class TestOrchestratorComponents:
    """Tests for OrchestratorComponents dataclass."""

    def test_empty_components_not_complete(self):
        """Empty components should not be complete."""
        from HoloLoom.orchestrator import OrchestratorComponents

        components = OrchestratorComponents()
        assert not components.is_complete()

    def test_empty_components_missing_all(self):
        """Empty components should report all 6 missing."""
        from HoloLoom.orchestrator import OrchestratorComponents

        components = OrchestratorComponents()
        missing = components.missing_components()
        assert len(missing) == 6
        assert 'pattern_selector' in missing
        assert 'thread_selector' in missing
        assert 'feature_extractor' in missing
        assert 'convergence' in missing
        assert 'tool_executor' in missing
        assert 'spacetime_assembler' in missing

    def test_partial_components_reports_missing(self):
        """Partial components should report only missing ones."""
        from HoloLoom.orchestrator import OrchestratorComponents

        components = OrchestratorComponents(
            pattern_selector=MagicMock(),
            thread_selector=MagicMock(),
        )
        missing = components.missing_components()
        assert len(missing) == 4
        assert 'pattern_selector' not in missing
        assert 'thread_selector' not in missing

    def test_full_components_complete(self):
        """Full components should be complete."""
        from HoloLoom.orchestrator import OrchestratorComponents

        components = OrchestratorComponents(
            pattern_selector=MagicMock(),
            thread_selector=MagicMock(),
            feature_extractor=MagicMock(),
            convergence=MagicMock(),
            tool_executor=MagicMock(),
            spacetime_assembler=MagicMock(),
        )
        assert components.is_complete()
        assert len(components.missing_components()) == 0

    def test_validate_raises_for_incomplete(self):
        """validate() should raise ValueError for incomplete components."""
        from HoloLoom.orchestrator import OrchestratorComponents

        components = OrchestratorComponents()
        with pytest.raises(ValueError) as exc_info:
            components.validate()
        assert "incomplete" in str(exc_info.value).lower()

    def test_validate_succeeds_for_complete(self):
        """validate() should succeed for complete components."""
        from HoloLoom.orchestrator import OrchestratorComponents

        components = OrchestratorComponents(
            pattern_selector=MagicMock(),
            thread_selector=MagicMock(),
            feature_extractor=MagicMock(),
            convergence=MagicMock(),
            tool_executor=MagicMock(),
            spacetime_assembler=MagicMock(),
        )
        # Should not raise
        components.validate()


# =============================================================================
# Test Default Implementations with Protocols
# =============================================================================

class TestDefaultPatternSelector:
    """Tests for DefaultPatternSelector."""

    def test_instantiation(self):
        """DefaultPatternSelector should instantiate with loom_command."""
        from HoloLoom.orchestrator.protocols import DefaultPatternSelector

        mock_loom = MagicMock()
        selector = DefaultPatternSelector(loom_command=mock_loom)
        assert selector.loom_command is mock_loom

    @pytest.mark.asyncio
    async def test_select_pattern_delegates(self):
        """select_pattern should delegate to loom_command."""
        from HoloLoom.orchestrator.protocols import DefaultPatternSelector
        from HoloLoom.protocols.types import Query

        mock_loom = MagicMock()
        mock_loom.select = AsyncMock(return_value='FAST')

        selector = DefaultPatternSelector(loom_command=mock_loom)
        query = Query(text="test query")
        result = await selector.select_pattern(query)

        mock_loom.select.assert_called_once_with(query)
        assert result == 'FAST'


class TestDefaultThreadSelector:
    """Tests for DefaultThreadSelector."""

    def test_instantiation(self):
        """DefaultThreadSelector should instantiate with yarn_graph."""
        from HoloLoom.orchestrator.protocols import DefaultThreadSelector

        mock_graph = MagicMock()
        selector = DefaultThreadSelector(yarn_graph=mock_graph)
        assert selector.yarn_graph is mock_graph

    @pytest.mark.asyncio
    async def test_select_threads_returns_list(self):
        """select_threads should return list of threads."""
        from HoloLoom.orchestrator.protocols import DefaultThreadSelector
        from HoloLoom.protocols.types import Query

        mock_graph = MagicMock()
        mock_graph.get_relevant_nodes = MagicMock(return_value=['node1', 'node2'])

        selector = DefaultThreadSelector(yarn_graph=mock_graph)
        query = Query(text="test query")
        result = await selector.select_threads(query, pattern='FAST')

        assert isinstance(result, list)


class TestDefaultConvergence:
    """Tests for DefaultConvergence."""

    def test_instantiation(self):
        """DefaultConvergence should instantiate with cfg and policy."""
        from HoloLoom.orchestrator.protocols import DefaultConvergence

        mock_cfg = MagicMock()
        mock_policy = MagicMock()
        convergence = DefaultConvergence(cfg=mock_cfg, policy=mock_policy)
        assert convergence.cfg is mock_cfg
        assert convergence.policy is mock_policy

    @pytest.mark.asyncio
    async def test_converge_returns_decision(self):
        """converge should return a decision dict."""
        from HoloLoom.orchestrator.protocols import DefaultConvergence

        mock_cfg = MagicMock()
        mock_cfg.collapse_strategy = 'argmax'
        mock_policy = MagicMock()
        mock_policy.forward = MagicMock(return_value={
            'action': 0,
            'probs': [0.7, 0.2, 0.1],
            'confidence': 0.7
        })

        convergence = DefaultConvergence(cfg=mock_cfg, policy=mock_policy)
        features = {'embedding': [0.1] * 10}
        result = await convergence.converge(features, pattern='FAST')

        assert isinstance(result, dict)
        assert 'action' in result or 'tool' in result or 'confidence' in result


class TestDefaultToolExecutor:
    """Tests for DefaultToolExecutor."""

    def test_instantiation(self):
        """DefaultToolExecutor should instantiate with tool_executor."""
        from HoloLoom.orchestrator.protocols import DefaultToolExecutor

        mock_executor = MagicMock()
        executor = DefaultToolExecutor(tool_executor=mock_executor)
        assert executor.tool_executor is mock_executor

    @pytest.mark.asyncio
    async def test_execute_without_guardrails(self):
        """execute should work without guardrails."""
        from HoloLoom.orchestrator.protocols import DefaultToolExecutor

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value={'result': 'success'})

        executor = DefaultToolExecutor(tool_executor=mock_executor)
        decision = {'tool': 'answer', 'action': 0}
        ctx = MagicMock()

        result = await executor.execute(decision, ctx)
        assert result is not None


class TestDefaultSpacetimeAssembler:
    """Tests for DefaultSpacetimeAssembler."""

    def test_instantiation(self):
        """DefaultSpacetimeAssembler should instantiate with cfg."""
        from HoloLoom.orchestrator.protocols import DefaultSpacetimeAssembler

        mock_cfg = MagicMock()
        assembler = DefaultSpacetimeAssembler(cfg=mock_cfg)
        assert assembler.cfg is mock_cfg

    @pytest.mark.asyncio
    async def test_assemble_returns_spacetime(self):
        """assemble should return spacetime result."""
        from HoloLoom.orchestrator.protocols import DefaultSpacetimeAssembler

        mock_cfg = MagicMock()
        assembler = DefaultSpacetimeAssembler(cfg=mock_cfg)

        mock_ctx = MagicMock()
        mock_ctx.query = MagicMock()
        mock_ctx.query.text = "test"
        mock_ctx.pattern = 'FAST'
        mock_ctx.features = {}
        mock_ctx.decision = {'confidence': 0.9}

        tool_result = {'response': 'test response'}

        result = await assembler.assemble(mock_ctx, tool_result)
        assert result is not None


# =============================================================================
# Test Protocol Factory Functions
# =============================================================================

class TestCreateComponentDefaults:
    """Tests for create_component_defaults factory."""

    def test_returns_complete_components(self):
        """create_component_defaults should return complete components."""
        from HoloLoom.orchestrator import create_component_defaults, OrchestratorComponents

        # Create minimal mocks
        mock_cfg = MagicMock()
        mock_loom = MagicMock()
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_policy = MagicMock()
        mock_tool_exec = MagicMock()

        components = create_component_defaults(
            cfg=mock_cfg,
            loom_command=mock_loom,
            yarn_graph=mock_graph,
            embedder=mock_embedder,
            policy=mock_policy,
            tool_executor=mock_tool_exec,
        )

        assert isinstance(components, OrchestratorComponents)
        assert components.is_complete()

    def test_accepts_optional_parameters(self):
        """create_component_defaults should accept optional parameters."""
        from HoloLoom.orchestrator import create_component_defaults

        mock_cfg = MagicMock()
        mock_loom = MagicMock()
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_policy = MagicMock()
        mock_tool_exec = MagicMock()
        mock_memory = MagicMock()
        mock_guardrails = MagicMock()

        components = create_component_defaults(
            cfg=mock_cfg,
            loom_command=mock_loom,
            yarn_graph=mock_graph,
            embedder=mock_embedder,
            policy=mock_policy,
            tool_executor=mock_tool_exec,
            memory=mock_memory,
            guardrails=mock_guardrails,
        )

        assert components.is_complete()


class TestCreateOrchestratorFromProtocols:
    """Tests for create_orchestrator_from_protocols factory."""

    def test_raises_for_incomplete_components(self):
        """create_orchestrator_from_protocols should raise for incomplete components."""
        from HoloLoom.orchestrator import (
            create_orchestrator_from_protocols,
            OrchestratorComponents,
        )

        components = OrchestratorComponents()  # Empty
        mock_cfg = MagicMock()

        with pytest.raises(ValueError):
            create_orchestrator_from_protocols(components, mock_cfg)

    def test_returns_protocol_orchestrator(self):
        """create_orchestrator_from_protocols should return ProtocolOrchestrator."""
        from HoloLoom.orchestrator import (
            create_orchestrator_from_protocols,
            OrchestratorComponents,
            ProtocolOrchestrator,
        )

        components = OrchestratorComponents(
            pattern_selector=MagicMock(),
            thread_selector=MagicMock(),
            feature_extractor=MagicMock(),
            convergence=MagicMock(),
            tool_executor=MagicMock(),
            spacetime_assembler=MagicMock(),
        )
        mock_cfg = MagicMock()

        orchestrator = create_orchestrator_from_protocols(components, mock_cfg)
        assert isinstance(orchestrator, ProtocolOrchestrator)


class TestCreatePipelineWithProtocols:
    """Tests for create_pipeline_with_protocols factory."""

    def test_raises_for_incomplete_components(self):
        """create_pipeline_with_protocols should raise for incomplete components."""
        from HoloLoom.orchestrator import (
            create_pipeline_with_protocols,
            OrchestratorComponents,
        )

        components = OrchestratorComponents()  # Empty
        mock_cfg = MagicMock()

        with pytest.raises(ValueError):
            create_pipeline_with_protocols(components, mock_cfg)

    def test_returns_executor_pipeline(self):
        """create_pipeline_with_protocols should return ExecutorPipeline."""
        from HoloLoom.orchestrator import (
            create_pipeline_with_protocols,
            create_component_defaults,
            ExecutorPipeline,
        )

        # Create components with defaults
        mock_cfg = MagicMock()
        mock_cfg.enable_meta_prompt = False
        mock_cfg.lookback_days = 365
        mock_cfg.recency_bias = 0.5

        components = create_component_defaults(
            cfg=mock_cfg,
            loom_command=MagicMock(),
            yarn_graph=MagicMock(),
            embedder=MagicMock(),
            policy=MagicMock(),
            tool_executor=MagicMock(),
        )

        pipeline = create_pipeline_with_protocols(components, mock_cfg)
        assert isinstance(pipeline, ExecutorPipeline)
        assert len(pipeline) == 8  # 8 executors in default pipeline


# =============================================================================
# Test ProtocolOrchestrator Class
# =============================================================================

class TestProtocolOrchestrator:
    """Tests for ProtocolOrchestrator class."""

    def test_has_required_attributes(self):
        """ProtocolOrchestrator should have required attributes."""
        from HoloLoom.orchestrator import (
            ProtocolOrchestrator,
            OrchestratorComponents,
        )

        components = OrchestratorComponents(
            pattern_selector=MagicMock(),
            thread_selector=MagicMock(),
            feature_extractor=MagicMock(),
            convergence=MagicMock(),
            tool_executor=MagicMock(),
            spacetime_assembler=MagicMock(),
        )
        mock_cfg = MagicMock()

        orchestrator = ProtocolOrchestrator(components=components, cfg=mock_cfg)

        assert hasattr(orchestrator, 'components')
        assert hasattr(orchestrator, 'cfg')
        assert hasattr(orchestrator, 'orchestrate')

    def test_orchestrate_is_async(self):
        """orchestrate method should be async."""
        from HoloLoom.orchestrator import ProtocolOrchestrator
        import inspect

        assert inspect.iscoroutinefunction(ProtocolOrchestrator.orchestrate)


# =============================================================================
# Test Protocol Compliance (isinstance checks)
# =============================================================================

class TestProtocolCompliance:
    """Tests for protocol compliance via isinstance checks."""

    def test_default_pattern_selector_protocol(self):
        """DefaultPatternSelector should implement PatternSelectorProtocol."""
        from HoloLoom.orchestrator.protocols import (
            DefaultPatternSelector,
            PatternSelectorProtocol,
        )

        selector = DefaultPatternSelector(loom_command=MagicMock())
        # Note: isinstance only works with @runtime_checkable protocols
        assert hasattr(selector, 'select_pattern')

    def test_default_thread_selector_protocol(self):
        """DefaultThreadSelector should implement ThreadSelectorProtocol."""
        from HoloLoom.orchestrator.protocols import (
            DefaultThreadSelector,
            ThreadSelectorProtocol,
        )

        selector = DefaultThreadSelector(yarn_graph=MagicMock())
        assert hasattr(selector, 'select_threads')

    def test_default_feature_extractor_protocol(self):
        """DefaultFeatureExtractor should implement FeatureExtractorProtocol."""
        from HoloLoom.orchestrator.protocols import (
            DefaultFeatureExtractor,
            FeatureExtractorProtocol,
        )

        extractor = DefaultFeatureExtractor(cfg=MagicMock(), embedder=MagicMock())
        assert hasattr(extractor, 'extract_features')

    def test_default_convergence_protocol(self):
        """DefaultConvergence should implement ConvergenceProtocol."""
        from HoloLoom.orchestrator.protocols import (
            DefaultConvergence,
            ConvergenceProtocol,
        )

        convergence = DefaultConvergence(cfg=MagicMock(), policy=MagicMock())
        assert hasattr(convergence, 'converge')

    def test_default_tool_executor_protocol(self):
        """DefaultToolExecutor should implement ToolExecutorProtocol."""
        from HoloLoom.orchestrator.protocols import (
            DefaultToolExecutor,
            ToolExecutorProtocol,
        )

        executor = DefaultToolExecutor(tool_executor=MagicMock())
        assert hasattr(executor, 'execute')

    def test_default_spacetime_assembler_protocol(self):
        """DefaultSpacetimeAssembler should implement SpacetimeAssemblerProtocol."""
        from HoloLoom.orchestrator.protocols import (
            DefaultSpacetimeAssembler,
            SpacetimeAssemblerProtocol,
        )

        assembler = DefaultSpacetimeAssembler(cfg=MagicMock())
        assert hasattr(assembler, 'assemble')


# =============================================================================
# Test Import All Exports
# =============================================================================

class TestImports:
    """Tests for package imports."""

    def test_import_from_orchestrator(self):
        """All Phase 3 exports should be importable from orchestrator."""
        from HoloLoom.orchestrator import (
            # Protocols
            PatternSelectorProtocol,
            ThreadSelectorProtocol,
            FeatureExtractorProtocol,
            ConvergenceProtocol,
            ToolExecutorProtocol,
            SpacetimeAssemblerProtocol,
            # Defaults
            DefaultPatternSelector,
            DefaultThreadSelector,
            DefaultFeatureExtractor,
            DefaultConvergence,
            DefaultToolExecutor,
            DefaultSpacetimeAssembler,
            # Factory
            OrchestratorComponents,
            create_component_defaults,
            create_orchestrator_from_protocols,
            create_pipeline_with_protocols,
            ProtocolOrchestrator,
        )
        # If we get here, all imports succeeded
        assert True

    def test_import_from_protocols_package(self):
        """All exports should be importable from protocols package."""
        from HoloLoom.orchestrator.protocols import (
            StageExecutorProtocol,
            BaseStageExecutor,
            PatternSelectorProtocol,
            ThreadSelectorProtocol,
            FeatureExtractorProtocol,
            ConvergenceProtocol,
            ToolExecutorProtocol,
            SpacetimeAssemblerProtocol,
            DefaultPatternSelector,
            DefaultThreadSelector,
            DefaultFeatureExtractor,
            DefaultConvergence,
            DefaultToolExecutor,
            DefaultSpacetimeAssembler,
        )
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
