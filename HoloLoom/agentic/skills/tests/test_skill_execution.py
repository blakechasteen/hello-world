#!/usr/bin/env python3
"""
Test Skill Execution
====================
Tests for SkillExecutor with mocked RecursiveWeavingOrchestrator.

Test Coverage:
- Skill execution pipeline
- Parameter validation during execution
- Reasoning strategy selection
- Result structure validation
- Error handling
- Mock orchestrator integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from HoloLoom.agentic.skills import (
    SkillExecutor,
    SkillRegistry,
    SkillExecutionResult,
    execute_skill
)
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query


# Module-level fixtures available to all test classes
@pytest.fixture
def mock_orchestrator():
    """Create mock RecursiveWeavingOrchestrator."""
    orchestrator = AsyncMock()

    # Mock weave_with_strategy to return a spacetime-like object
    mock_spacetime = MagicMock()
    mock_spacetime.response = "Mocked response from orchestrator"
    mock_spacetime.confidence = 0.85
    mock_spacetime.iterations = 2
    mock_spacetime.strategy_used = MagicMock()
    mock_spacetime.strategy_used.value = "refine"
    mock_spacetime.metadata = {
        "execution_time_ms": 150.5,
        "tool_used": "answer"
    }

    orchestrator.weave_with_strategy = AsyncMock(return_value=mock_spacetime)

    return orchestrator


@pytest.fixture
def mock_config():
    """Create mock Config."""
    config = MagicMock(spec=Config)
    config.bare = MagicMock(return_value=config)
    return config


@pytest.fixture
def mock_registry():
    """Create mock SkillRegistry with test skills."""
    registry = MagicMock(spec=SkillRegistry)

    # Create mock skill template
    mock_skill = MagicMock()
    mock_skill.name = "code-reviewer"
    mock_skill.metadata = MagicMock()
    mock_skill.metadata.category = "development"
    mock_skill.reasoning = MagicMock()
    mock_skill.reasoning.default_strategy = "refine"
    mock_skill.reasoning.max_iterations = 3
    mock_skill.reasoning.quality_threshold = 0.85
    mock_skill.user_prompt_template = "Review this code: {code}"
    mock_skill.parameters = [
        MagicMock(name="code", type="string", required=True),
        MagicMock(name="language", type="string", required=True)
    ]

    registry.get_skill = MagicMock(return_value=mock_skill)
    return registry


class TestSkillExecutor:
    """Test SkillExecutor class."""

    @pytest.fixture
    def executor(self, mock_orchestrator, mock_config, mock_registry):
        """Create SkillExecutor with mocked dependencies."""
        with patch('HoloLoom.agentic.skill_agents.RecursiveWeavingOrchestrator', return_value=mock_orchestrator):
            executor = SkillExecutor(registry=mock_registry, config=mock_config, shards=[])
            executor.orchestrator = mock_orchestrator
            return executor

    @pytest.mark.asyncio
    async def test_executor_initializes(self, mock_config, mock_registry):
        """Test that executor can be created."""
        executor = SkillExecutor(registry=mock_registry, config=mock_config, shards=[])
        assert executor is not None
        assert executor.config == mock_config
        assert executor.registry == mock_registry

    @pytest.mark.asyncio
    async def test_execute_skill_basic(self, executor):
        """Test basic skill execution."""
        result = await executor.execute(
            skill_name="code_reviewer",
            parameters={"code": "print('hello')", "language": "python"}
        )

        assert isinstance(result, SkillExecutionResult)
        assert result.skill_name == "code_reviewer"
        assert result.success is True
        assert result.output == "Mocked response from orchestrator"
        assert result.confidence == 0.85
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_execute_skill_validates_parameters(self, executor):
        """Test that execution validates required parameters."""
        # code_reviewer requires 'code' parameter
        with pytest.raises(ValueError, match="Missing required parameter"):
            await executor.execute(
                skill_name="code_reviewer",
                parameters={"language": "python"}  # Missing 'code'
            )

    @pytest.mark.asyncio
    async def test_execute_skill_formats_prompt(self, executor, mock_orchestrator):
        """Test that execution formats prompt with parameters."""
        await executor.execute(
            skill_name="code_reviewer",
            parameters={"code": "def foo(): pass", "language": "python"}
        )

        # Check that orchestrator was called
        assert mock_orchestrator.weave_with_strategy.called

        # Check the query text contains the code
        call_args = mock_orchestrator.weave_with_strategy.call_args
        query = call_args[1]['query']  # kwargs
        assert "def foo(): pass" in query.text or "def foo(): pass" in str(query)

    @pytest.mark.asyncio
    async def test_execute_skill_uses_reasoning_strategy(self, executor, mock_orchestrator):
        """Test that execution uses template reasoning strategy."""
        await executor.execute(
            skill_name="code_reviewer",
            parameters={"code": "print('test')", "language": "python"}
        )

        # Check that strategy was passed
        call_args = mock_orchestrator.weave_with_strategy.call_args
        assert 'strategy' in call_args[1]  # Check kwargs

    @pytest.mark.asyncio
    async def test_execute_skill_handles_nonexistent_skill(self, executor):
        """Test execution with nonexistent skill."""
        result = await executor.execute(
            skill_name="nonexistent_skill_12345",
            parameters={}
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_skill_handles_orchestrator_error(self, executor, mock_orchestrator):
        """Test execution when orchestrator raises exception."""
        # Make orchestrator raise an error
        mock_orchestrator.weave_with_strategy = AsyncMock(
            side_effect=Exception("Orchestrator error")
        )

        result = await executor.execute(
            skill_name="code_reviewer",
            parameters={"code": "test", "language": "python"}
        )

        assert result.success is False
        assert "orchestrator error" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_skill_returns_metadata(self, executor):
        """Test that execution result includes metadata."""
        result = await executor.execute(
            skill_name="code_reviewer",
            parameters={"code": "print('hello')", "language": "python"}
        )

        assert result.execution_time_ms > 0
        assert result.strategy_used == "refine"
        assert result.metadata is not None


class TestExecuteSkillConvenience:
    """Test execute_skill() convenience function."""

    @pytest.fixture
    def mock_executor(self):
        """Create mock SkillExecutor."""
        executor = AsyncMock(spec=SkillExecutor)

        mock_result = SkillExecutionResult(
            skill_name="test_skill",
            success=True,
            output="Test output",
            confidence=0.9,
            iterations=1,
            strategy_used="direct",
            execution_time_ms=100.0,
            metadata={}
        )

        executor.execute = AsyncMock(return_value=mock_result)
        return executor

    @pytest.mark.asyncio
    async def test_execute_skill_convenience_function(self):
        """Test execute_skill() convenience function."""
        with patch('HoloLoom.agentic.skill_agents.SkillExecutor') as MockExecutor:
            mock_instance = AsyncMock()
            mock_result = SkillExecutionResult(
                skill_name="code_reviewer",
                success=True,
                output="Review complete",
                confidence=0.88,
                iterations=2,
                strategy_used="refine",
                execution_time_ms=150.0,
                metadata={}
            )
            mock_instance.execute = AsyncMock(return_value=mock_result)
            MockExecutor.return_value = mock_instance

            result = await execute_skill(
                skill_name="code_reviewer",
                parameters={"code": "test", "language": "python"},
                config=Config.bare()
            )

            assert isinstance(result, SkillExecutionResult)
            assert result.skill_name == "code_reviewer"
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_skill_passes_config(self):
        """Test that execute_skill() passes config to executor."""
        with patch('HoloLoom.agentic.skill_agents.SkillExecutor') as MockExecutor:
            mock_instance = AsyncMock()
            mock_result = SkillExecutionResult(
                skill_name="test",
                success=True,
                output="",
                confidence=0.9,
                iterations=1,
                strategy_used="direct",
                execution_time_ms=100.0,
                metadata={}
            )
            mock_instance.execute = AsyncMock(return_value=mock_result)
            MockExecutor.return_value = mock_instance

            config = Config.fast()
            await execute_skill(
                skill_name="test_skill",
                parameters={},
                config=config
            )

            # Check that SkillExecutor was initialized with the config
            MockExecutor.assert_called_once()
            call_kwargs = MockExecutor.call_args[1]
            assert call_kwargs['config'] == config


class TestResultStructure:
    """Test SkillExecutionResult structure."""

    def test_result_success_case(self):
        """Test result structure for successful execution."""
        result = SkillExecutionResult(
            skill_name="test_skill",
            success=True,
            output="Test output",
            confidence=0.92,
            iterations=3,
            strategy_used="refine",
            execution_time_ms=200.5,
            metadata={"tool": "answer"}
        )

        assert result.skill_name == "test_skill"
        assert result.success is True
        assert result.output == "Test output"
        assert result.confidence == 0.92
        assert result.iterations == 3
        assert result.strategy_used == "refine"
        assert result.execution_time_ms == 200.5
        assert result.error is None
        assert result.metadata["tool"] == "answer"

    def test_result_failure_case(self):
        """Test result structure for failed execution."""
        result = SkillExecutionResult(
            skill_name="test_skill",
            success=False,
            output="",
            confidence=0.0,
            iterations=0,
            strategy_used="",
            execution_time_ms=50.0,
            metadata={},
            error="Execution failed: test error"
        )

        assert result.success is False
        assert result.error == "Execution failed: test error"
        assert result.output == ""
        assert result.confidence == 0.0


class TestErrorHandling:
    """Test error handling during execution."""

    @pytest.fixture
    def executor_with_failing_orchestrator(self, mock_registry):
        """Create executor with orchestrator that raises errors."""
        mock_config = MagicMock(spec=Config)
        executor = SkillExecutor(registry=mock_registry, config=mock_config, shards=[])

        # Mock orchestrator that raises various errors
        mock_orchestrator = AsyncMock()
        executor.orchestrator = mock_orchestrator

        return executor, mock_orchestrator

    @pytest.mark.asyncio
    async def test_handles_value_error(self, executor_with_failing_orchestrator):
        """Test handling of ValueError from orchestrator."""
        executor, mock_orch = executor_with_failing_orchestrator
        mock_orch.weave_with_strategy = AsyncMock(
            side_effect=ValueError("Invalid input")
        )

        result = await executor.execute(
            skill_name="code_reviewer",
            parameters={"code": "test", "language": "python"}
        )

        assert result.success is False
        assert "invalid input" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self, executor_with_failing_orchestrator):
        """Test handling of RuntimeError from orchestrator."""
        executor, mock_orch = executor_with_failing_orchestrator
        mock_orch.weave_with_strategy = AsyncMock(
            side_effect=RuntimeError("Runtime failure")
        )

        result = await executor.execute(
            skill_name="code_reviewer",
            parameters={"code": "test", "language": "python"}
        )

        assert result.success is False
        assert "runtime failure" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handles_generic_exception(self, executor_with_failing_orchestrator):
        """Test handling of generic Exception from orchestrator."""
        executor, mock_orch = executor_with_failing_orchestrator
        mock_orch.weave_with_strategy = AsyncMock(
            side_effect=Exception("Unknown error")
        )

        result = await executor.execute(
            skill_name="code_reviewer",
            parameters={"code": "test", "language": "python"}
        )

        assert result.success is False
        assert "unknown error" in result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
