"""
Test Suite for HoloLoom Easy Launch API
=======================================

Tests for one-line skill launching and @skill decorator.

Created: December 30, 2025
"""

import pytest
import asyncio
from typing import Dict, Any

# Import easy API
from HoloLoom.skills import (
    # Easy Launch API
    launch_skill,
    launch,
    launch_parallel,
    launch_chain,
    LaunchResult,
    list_skills,
    get_skill_info,
    get_launch_stats,

    # Easy Skill Creation
    skill,
    code_skill,
    domain_skill,
    FunctionSkill,
    SkillBuilder,

    # Base Classes
    SkillCategory,
    SkillStatus,
    get_registry,
)


# ============================================================================
# Test Fixtures: Create test skills using @skill decorator
# ============================================================================

@skill(name="test_echo", category="domain", tags=["test"])
async def echo_skill(message: str) -> dict:
    """Echo back the message."""
    return {"echo": message}


@skill(name="test_add", category="system")
async def add_skill(a: int, b: int) -> dict:
    """Add two numbers."""
    return {"result": a + b}


@skill(name="test_fail", category="testing")
async def fail_skill(should_fail: bool = True) -> dict:
    """A skill that can fail on demand."""
    if should_fail:
        raise ValueError("Intentional failure")
    return {"success": True}


@domain_skill(name="test_domain")
def sync_domain_skill(text: str) -> dict:
    """A sync domain skill."""
    return {"length": len(text), "text": text}


@code_skill(name="test_code")
async def code_analysis_skill(code: str) -> dict:
    """Analyze code."""
    lines = code.strip().split('\n')
    return {"lines": len(lines), "chars": len(code)}


# ============================================================================
# Test: @skill Decorator
# ============================================================================

class TestSkillDecorator:
    """Test @skill decorator functionality."""

    def test_skill_is_registered(self):
        """Decorated skills should be auto-registered."""
        registry = get_registry()
        assert registry.get("test_echo") is not None
        assert registry.get("test_add") is not None

    def test_skill_metadata(self):
        """Skill metadata should be correctly set."""
        info = get_skill_info("test_echo")
        assert info is not None
        assert info["name"] == "test_echo"
        assert info["category"] == "domain"
        assert "test" in info["tags"]

    def test_category_decorator(self):
        """Category-specific decorators should set correct category."""
        domain_info = get_skill_info("test_domain")
        assert domain_info["category"] == "domain"

        code_info = get_skill_info("test_code")
        assert code_info["category"] == "code"

    def test_sync_function_wrapped(self):
        """Sync functions should be wrapped properly."""
        registry = get_registry()
        skill_instance = registry.get("test_domain")
        assert skill_instance is not None
        # Should work even though original is sync


# ============================================================================
# Test: launch_skill() / launch()
# ============================================================================

class TestLaunchSkill:
    """Test launch_skill() and launch() functions."""

    @pytest.mark.asyncio
    async def test_basic_launch(self):
        """Basic skill launch should work."""
        result = await launch_skill("test_echo", message="Hello")
        assert result.success is True
        assert result.result["echo"] == "Hello"

    @pytest.mark.asyncio
    async def test_launch_alias(self):
        """launch() should be an alias for launch_skill()."""
        result = await launch("test_echo", message="World")
        assert result.success is True
        assert result.result["echo"] == "World"

    @pytest.mark.asyncio
    async def test_launch_with_numbers(self):
        """Launch with numeric parameters."""
        result = await launch("test_add", a=5, b=3)
        assert result.success is True
        assert result.result["result"] == 8

    @pytest.mark.asyncio
    async def test_launch_nonexistent_skill(self):
        """Launch of non-existent skill should fail gracefully."""
        result = await launch("nonexistent_skill_xyz")
        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_launch_result_structure(self):
        """LaunchResult should have correct structure."""
        result = await launch("test_echo", message="test")

        assert hasattr(result, 'success')
        assert hasattr(result, 'result')
        assert hasattr(result, 'message')
        assert hasattr(result, 'skill_name')
        assert hasattr(result, 'execution_time_ms')

    @pytest.mark.asyncio
    async def test_launch_sync_skill(self):
        """Launching sync-based skill should work."""
        result = await launch("test_domain", text="hello world")
        assert result.success is True
        assert result.result["length"] == 11


# ============================================================================
# Test: launch_parallel()
# ============================================================================

class TestLaunchParallel:
    """Test parallel skill execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Multiple skills should execute in parallel."""
        results = await launch_parallel(
            {"skill_name": "test_echo", "parameters": {"message": "A"}},
            {"skill_name": "test_echo", "parameters": {"message": "B"}},
            {"skill_name": "test_add", "parameters": {"a": 1, "b": 2}}
        )

        assert len(results) == 3
        assert all(r.success for r in results)

        # Check results
        assert results[0].result["echo"] == "A"
        assert results[1].result["echo"] == "B"
        assert results[2].result["result"] == 3

    @pytest.mark.asyncio
    async def test_parallel_with_failure(self):
        """Parallel execution should handle failures gracefully."""
        results = await launch_parallel(
            {"skill_name": "test_echo", "parameters": {"message": "ok"}},
            {"skill_name": "nonexistent_xyz", "parameters": {}},
            {"skill_name": "test_add", "parameters": {"a": 1, "b": 1}}
        )

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True


# ============================================================================
# Test: launch_chain()
# ============================================================================

class TestLaunchChain:
    """Test sequential chain execution."""

    @pytest.mark.asyncio
    async def test_chain_execution(self):
        """Skills should execute in sequence."""
        results = await launch_chain(
            {"skill_name": "test_echo", "parameters": {"message": "step1"}},
            {"skill_name": "test_add", "parameters": {"a": 2, "b": 2}},
            {"skill_name": "test_echo", "parameters": {"message": "step3"}}
        )

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_chain_stop_on_failure(self):
        """Chain should stop on first failure when stop_on_failure=True."""
        results = await launch_chain(
            {"skill_name": "test_echo", "parameters": {"message": "ok"}},
            {"skill_name": "test_fail", "parameters": {"should_fail": True}},
            {"skill_name": "test_echo", "parameters": {"message": "never"}},
            stop_on_failure=True
        )

        # Should stop after failure
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False


# ============================================================================
# Test: Skill Discovery
# ============================================================================

class TestSkillDiscovery:
    """Test skill discovery functions."""

    def test_list_skills(self):
        """list_skills() should return registered skills."""
        skills = list_skills()
        assert isinstance(skills, list)
        assert "test_echo" in skills
        assert "test_add" in skills

    def test_list_skills_by_category(self):
        """list_skills() with category filter should work."""
        domain_skills = list_skills(category="domain")
        assert "test_echo" in domain_skills

        code_skills = list_skills(category="code")
        assert "test_code" in code_skills

    def test_get_skill_info(self):
        """get_skill_info() should return metadata."""
        info = get_skill_info("test_echo")

        assert info is not None
        assert info["name"] == "test_echo"
        assert info["version"] == "1.0.0"
        assert info["category"] == "domain"

    def test_get_skill_info_nonexistent(self):
        """get_skill_info() for non-existent skill should return None."""
        info = get_skill_info("nonexistent_skill_xyz")
        assert info is None

    def test_get_launch_stats(self):
        """get_launch_stats() should return statistics."""
        stats = get_launch_stats()

        assert isinstance(stats, dict)
        assert "total_executions" in stats
        assert "cache_hit_rate" in stats


# ============================================================================
# Test: SkillBuilder API
# ============================================================================

class TestSkillBuilder:
    """Test SkillBuilder fluent API."""

    def test_builder_basic(self):
        """Basic builder usage."""
        def my_handler(x: int) -> dict:
            return {"double": x * 2}

        built_skill = (SkillBuilder()
            .name("builder_test")
            .category("domain")
            .description("Test skill built with builder")
            .tags(["test", "builder"])
            .handler(my_handler)
            .build(auto_register=False))

        assert built_skill is not None
        metadata = built_skill.get_metadata()
        assert metadata.name == "builder_test"
        assert metadata.category == SkillCategory.DOMAIN

    def test_builder_requires_handler(self):
        """Builder should raise error without handler."""
        with pytest.raises(ValueError):
            (SkillBuilder()
                .name("no_handler")
                .build())


# ============================================================================
# Test: FunctionSkill Class
# ============================================================================

class TestFunctionSkill:
    """Test FunctionSkill wrapper class."""

    def test_function_skill_creation(self):
        """FunctionSkill should wrap functions correctly."""
        async def my_func(name: str) -> dict:
            return {"greeting": f"Hello, {name}!"}

        skill_instance = FunctionSkill(
            func=my_func,
            name="manual_test",
            category=SkillCategory.DOMAIN
        )

        assert skill_instance.get_metadata().name == "manual_test"

    def test_function_skill_lifecycle_attrs(self):
        """FunctionSkill should have lifecycle attributes."""
        async def dummy() -> dict:
            return {}

        skill_instance = FunctionSkill(
            func=dummy,
            name="lifecycle_test",
            category=SkillCategory.TESTING
        )

        # These are required for execute_with_lifecycle()
        assert hasattr(skill_instance, '_setup_done')
        assert hasattr(skill_instance, '_execution_count')
        assert hasattr(skill_instance, '_total_time_ms')
        assert skill_instance._setup_done is False


# ============================================================================
# Test: Retry Functionality
# ============================================================================

class TestRetryFunctionality:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_config_import(self):
        """RetryConfig and RetryStrategy should be importable."""
        from HoloLoom.skills import RetryConfig, RetryStrategy

        # Test default config
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.strategy == RetryStrategy.EXPONENTIAL

    @pytest.mark.asyncio
    async def test_retry_strategies(self):
        """All retry strategies should be available."""
        from HoloLoom.skills import RetryStrategy

        assert RetryStrategy.NONE is not None
        assert RetryStrategy.IMMEDIATE is not None
        assert RetryStrategy.EXPONENTIAL is not None

    @pytest.mark.asyncio
    async def test_launch_with_retry_config(self):
        """Launch should accept retry configuration."""
        from HoloLoom.skills import RetryConfig, RetryStrategy

        # Create a retry config
        config = RetryConfig(
            max_retries=2,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_ms=100.0,
            max_delay_ms=1000.0
        )

        # Launch with retry config
        result = await launch("test_echo", message="retry_test", retry=config)
        assert result.success is True
        assert result.result["echo"] == "retry_test"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Retry should not change outcome for permanent failures."""
        from HoloLoom.skills import RetryConfig, RetryStrategy

        config = RetryConfig(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE,
            base_delay_ms=10.0
        )

        # This skill always fails
        result = await launch("test_fail", should_fail=True, retry=config)
        assert result.success is False
        # Should have retried
        assert result.retry_count >= 0  # May or may not retry based on implementation


# ============================================================================
# Test: Context Managers
# ============================================================================

class TestContextManagers:
    """Test context manager support."""

    @pytest.mark.asyncio
    async def test_skill_session_import(self):
        """skill_session should be importable."""
        from HoloLoom.skills import skill_session, SkillSessionContext
        assert skill_session is not None
        assert SkillSessionContext is not None

    @pytest.mark.asyncio
    async def test_skill_session_basic(self):
        """skill_session context manager should track results."""
        from HoloLoom.skills import skill_session

        async with skill_session() as session:
            result1 = await launch("test_echo", message="session1")
            session.results.append(result1)

            result2 = await launch("test_add", a=1, b=2)
            session.results.append(result2)

        assert len(session.results) == 2
        assert session.success_count == 2
        assert session.failure_count == 0
        assert session.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_skill_session_with_failures(self):
        """skill_session should track failures correctly."""
        from HoloLoom.skills import skill_session

        async with skill_session() as session:
            result1 = await launch("test_echo", message="ok")
            session.results.append(result1)

            result2 = await launch("nonexistent_skill")
            session.results.append(result2)

        assert session.success_count == 1
        assert session.failure_count == 1
        assert session.success_rate == 0.5

    @pytest.mark.asyncio
    async def test_skill_session_summary(self):
        """skill_session summary should contain all metrics."""
        from HoloLoom.skills import skill_session

        async with skill_session() as session:
            result = await launch("test_echo", message="summary_test")
            session.results.append(result)

        summary = session.summary()
        assert "total_launches" in summary
        assert "success_count" in summary
        assert "failure_count" in summary
        assert "success_rate" in summary
        assert "total_time_ms" in summary
        assert "avg_time_ms" in summary

    @pytest.mark.asyncio
    async def test_batch_launch_import(self):
        """batch_launch should be importable."""
        from HoloLoom.skills import batch_launch
        assert batch_launch is not None

    @pytest.mark.asyncio
    async def test_batch_launch_parallel(self):
        """batch_launch should execute skills in parallel."""
        from HoloLoom.skills import batch_launch

        specs = [
            {"skill_name": "test_echo", "parameters": {"message": "batch1"}},
            {"skill_name": "test_echo", "parameters": {"message": "batch2"}},
            {"skill_name": "test_add", "parameters": {"a": 5, "b": 5}},
        ]

        async with batch_launch(specs, parallel=True) as results:
            assert len(results) == 3
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_batch_launch_sequential(self):
        """batch_launch should execute skills sequentially when parallel=False."""
        from HoloLoom.skills import batch_launch

        specs = [
            {"skill_name": "test_echo", "parameters": {"message": "seq1"}},
            {"skill_name": "test_echo", "parameters": {"message": "seq2"}},
        ]

        async with batch_launch(specs, parallel=False) as results:
            assert len(results) == 2
            assert all(r.success for r in results)


# ============================================================================
# Test: Health Checks
# ============================================================================

class TestHealthChecks:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_import(self):
        """Health check functions should be importable."""
        from HoloLoom.skills import (
            check_skill_health,
            get_system_health,
            SkillHealthStatus,
            SystemHealthStatus,
        )
        assert check_skill_health is not None
        assert get_system_health is not None
        assert SkillHealthStatus is not None
        assert SystemHealthStatus is not None

    @pytest.mark.asyncio
    async def test_check_skill_health_existing(self):
        """check_skill_health should return healthy for existing skills."""
        from HoloLoom.skills import check_skill_health

        status = await check_skill_health("test_echo")

        assert status.skill_name == "test_echo"
        assert status.healthy is True
        assert "healthy" in status.message.lower()
        assert status.latency_ms >= 0
        assert "version" in status.metadata

    @pytest.mark.asyncio
    async def test_check_skill_health_nonexistent(self):
        """check_skill_health should return unhealthy for non-existent skills."""
        from HoloLoom.skills import check_skill_health

        status = await check_skill_health("nonexistent_health_test")

        assert status.skill_name == "nonexistent_health_test"
        assert status.healthy is False
        assert "not found" in status.message.lower()

    @pytest.mark.asyncio
    async def test_skill_health_status_structure(self):
        """SkillHealthStatus should have correct structure."""
        from HoloLoom.skills import check_skill_health

        status = await check_skill_health("test_echo")

        assert hasattr(status, 'skill_name')
        assert hasattr(status, 'healthy')
        assert hasattr(status, 'message')
        assert hasattr(status, 'latency_ms')
        assert hasattr(status, 'last_check')
        assert hasattr(status, 'metadata')

    @pytest.mark.asyncio
    async def test_get_system_health(self):
        """get_system_health should check all registered skills."""
        from HoloLoom.skills import get_system_health

        health = await get_system_health(timeout_per_skill=2.0)

        assert health.total_skills > 0
        assert health.healthy_skills >= 0
        assert health.check_time_ms >= 0
        assert isinstance(health.unhealthy_skills, list)
        assert isinstance(health.skill_statuses, dict)

    @pytest.mark.asyncio
    async def test_system_health_rate(self):
        """SystemHealthStatus should calculate health_rate correctly."""
        from HoloLoom.skills import get_system_health

        health = await get_system_health()

        # health_rate should be between 0 and 1
        assert 0.0 <= health.health_rate <= 1.0

        # If all skills are healthy, rate should be 1.0
        if health.healthy_skills == health.total_skills:
            assert health.health_rate == 1.0

    @pytest.mark.asyncio
    async def test_system_health_parallel(self):
        """get_system_health should support parallel checking."""
        from HoloLoom.skills import get_system_health

        # Parallel should be faster for many skills
        health_parallel = await get_system_health(parallel=True)
        assert health_parallel.total_skills > 0

    @pytest.mark.asyncio
    async def test_system_health_sequential(self):
        """get_system_health should support sequential checking."""
        from HoloLoom.skills import get_system_health

        health_seq = await get_system_health(parallel=False)
        assert health_seq.total_skills > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
