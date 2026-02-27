#!/usr/bin/env python3
"""
End-to-End Integration Tests
============================
Full integration tests for Skills System with real orchestrator.

Test Coverage:
- Complete execution pipeline
- Real RecursiveWeavingOrchestrator integration
- Multiple skills execution
- Performance characteristics
- Cross-skill integration
"""

import pytest
import asyncio
from pathlib import Path

from hololoom.agentic.skills import (
    execute_skill,
    list_available_skills,
    get_registry,
    SkillExecutionResult
)
from hololoom.config import Config
from hololoom.protocols.types import MemoryShard


class TestEndToEndExecution:
    """Test end-to-end skill execution with real orchestrator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        # Use BARE mode for fastest execution in tests
        return Config.bare()

    @pytest.fixture
    def test_shards(self):
        """Create test memory shards."""
        return [
            MemoryShard(
                content="Thompson Sampling is a Bayesian approach to exploration-exploitation tradeoffs.",
                timestamp=1698595200.0
            ),
            MemoryShard(
                content="Python uses indentation for code blocks. Functions are defined with 'def'.",
                timestamp=1698595201.0
            ),
            MemoryShard(
                content="Code review should check for correctness, efficiency, readability, and maintainability.",
                timestamp=1698595202.0
            )
        ]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execute_code_reviewer_skill(self, config, test_shards):
        """Test executing code_reviewer skill end-to-end."""
        sample_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total
"""

        result = await execute_skill(
            skill_name="code_reviewer",
            parameters={
                "code": sample_code,
                "language": "python"
            },
            config=config
        )

        # Verify result structure
        assert isinstance(result, SkillExecutionResult)
        assert result.skill_name == "code_reviewer"

        # Verify execution succeeded
        assert result.success is True, f"Execution failed: {result.error}"

        # Verify output contains a review
        assert result.output, "Output should not be empty"
        assert len(result.output) > 10, "Output should be substantial"

        # Verify metadata
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        assert result.iterations >= 1
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execute_code_explainer_skill(self, config, test_shards):
        """Test executing code_explainer skill end-to-end."""
        sample_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

        result = await execute_skill(
            skill_name="code_explainer",
            parameters={
                "code": sample_code,
                "language": "python"
            },
            config=config
        )

        # Verify execution
        assert result.success is True
        assert result.output
        assert "factorial" in result.output.lower() or "recursion" in result.output.lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_list_all_skills_integration(self):
        """Test listing all skills returns complete structure."""
        skills = await list_available_skills()

        # Verify structure
        assert isinstance(skills, dict)
        assert len(skills) > 0

        # Verify we have expected categories
        assert "development" in skills
        assert "architecture" in skills or "education" in skills

        # Verify each category has skills
        for category, skill_list in skills.items():
            assert isinstance(skill_list, list)
            assert len(skill_list) > 0
            for skill_name in skill_list:
                assert isinstance(skill_name, str)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execute_multiple_skills_sequentially(self, config):
        """Test executing multiple different skills in sequence."""
        # Execute code_reviewer
        result1 = await execute_skill(
            skill_name="code_reviewer",
            parameters={"code": "print('hello')", "language": "python"},
            config=config
        )

        # Execute code_explainer
        result2 = await execute_skill(
            skill_name="code_explainer",
            parameters={"code": "x = [1, 2, 3]", "language": "python"},
            config=config
        )

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

        # Both should have different outputs
        assert result1.output != result2.output


class TestRegistryIntegration:
    """Test SkillRegistry integration."""

    @pytest.mark.asyncio
    async def test_registry_singleton(self):
        """Test that get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    @pytest.mark.asyncio
    async def test_registry_loads_all_13_skills(self):
        """Test that registry loads all 13 known skills."""
        registry = get_registry()
        skills = registry.list_skills()

        total_skills = sum(len(skill_list) for skill_list in skills.values())

        assert total_skills >= 13, f"Expected at least 13 skills, got {total_skills}"

    @pytest.mark.asyncio
    async def test_registry_skills_are_accessible(self):
        """Test that all listed skills can be retrieved."""
        registry = get_registry()
        skills = registry.list_skills()

        for category, skill_list in skills.items():
            for skill_name in skill_list:
                template = registry.get_skill(skill_name)
                assert template is not None, f"Could not retrieve skill: {skill_name}"
                assert template.name == skill_name


class TestPerformanceCharacteristics:
    """Test performance characteristics of skill execution."""

    @pytest.fixture
    def config(self):
        """Use BARE config for performance testing."""
        return Config.bare()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_execution_completes_in_reasonable_time(self, config):
        """Test that skill execution completes in reasonable time."""
        import time

        start = time.time()

        result = await execute_skill(
            skill_name="code_reviewer",
            parameters={"code": "def foo(): pass", "language": "python"},
            config=config
        )

        duration = time.time() - start

        # BARE mode should complete within 5 seconds
        assert duration < 5.0, f"Execution took too long: {duration:.2f}s"

        # Result should report similar time
        assert result.execution_time_ms < 5000

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_skill_loading_is_fast(self):
        """Test that skill loading from YAML is fast."""
        import time

        start = time.time()
        registry = get_registry()
        skills = registry.list_skills()
        duration = time.time() - start

        # Loading all YAMLs should be <100ms
        assert duration < 0.1, f"Skill loading took too long: {duration:.3f}s"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_skill_execution(self, config):
        """Test executing multiple skills concurrently."""
        # Create multiple skill execution tasks
        tasks = [
            execute_skill(
                skill_name="code_reviewer",
                parameters={"code": f"# Test {i}", "language": "python"},
                config=config
            )
            for i in range(3)
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 3


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.bare()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_handles_invalid_skill_gracefully(self, config):
        """Test that invalid skill names are handled gracefully."""
        result = await execute_skill(
            skill_name="completely_nonexistent_skill",
            parameters={},
            config=config
        )

        # Should return failure result, not raise exception
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_handles_missing_parameters_gracefully(self, config):
        """Test that missing required parameters are handled gracefully."""
        result = await execute_skill(
            skill_name="code_reviewer",
            parameters={},  # Missing required 'code' parameter
            config=config
        )

        # Should return failure result, not raise exception
        assert result.success is False
        assert "missing" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_handles_invalid_parameter_types(self, config):
        """Test that invalid parameter types are handled."""
        result = await execute_skill(
            skill_name="code_reviewer",
            parameters={
                "code": 12345,  # Should be string
                "language": "python"
            },
            config=config
        )

        # Should either succeed (coerce to string) or fail gracefully
        assert isinstance(result, SkillExecutionResult)


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config.fast()  # Use FAST for more realistic results

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_review_actual_python_code(self, config):
        """Test reviewing actual Python code."""
        code = """
import asyncio
from typing import List, Dict, Any

async def process_items(items: List[str]) -> Dict[str, Any]:
    '''Process a list of items asynchronously.'''
    results = {}
    for item in items:
        result = await process_single_item(item)
        results[item] = result
    return results

async def process_single_item(item: str) -> Any:
    '''Process a single item.'''
    await asyncio.sleep(0.1)
    return item.upper()
"""

        result = await execute_skill(
            skill_name="code_reviewer",
            parameters={"code": code, "language": "python"},
            config=config
        )

        # Should succeed
        assert result.success is True
        assert result.output

        # Output should be substantial (real review)
        assert len(result.output) > 50

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_explain_complex_algorithm(self, config):
        """Test explaining a complex algorithm."""
        code = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""

        result = await execute_skill(
            skill_name="code_explainer",
            parameters={"code": code, "language": "python"},
            config=config
        )

        # Should succeed and explain the algorithm
        assert result.success is True
        assert "sort" in result.output.lower() or "pivot" in result.output.lower()


# Mark slow tests
pytest.mark.slow = pytest.mark.slow


if __name__ == "__main__":
    # Run with: pytest test_integration.py -v -m "not slow"
    # Run slow tests: pytest test_integration.py -v -m slow
    pytest.main([__file__, "-v", "-m", "not slow"])
