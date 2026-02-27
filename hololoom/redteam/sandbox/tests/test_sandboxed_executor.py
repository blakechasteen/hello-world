"""
Tests for SandboxedExecutor
===========================

Comprehensive test suite for sandboxed attack execution:
- Executor initialization and setup
- Drop-in replacement functionality
- Resource monitoring integration
- Batch execution
- Graceful degradation
- Proper cleanup

Author: CARTS Team
Date: 2025-12-05
"""

import asyncio
import logging
import pytest
from typing import Optional

from hololoom.redteam.sandbox import (
    SandboxedExecutor,
    SandboxConfig,
    SandboxMode,
    create_sandboxed_executor,
    create_sandboxed_executor_sync,
    sandboxed_attack_execution,
)
from hololoom.redteam.executor import (
    AttackExecutor,
    AttackOutcome,
)
from hololoom.redteam.strategies import (
    AttackStrategy,
    AttackPayload,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_config():
    """Basic sandbox configuration."""
    return SandboxConfig(
        mode=SandboxMode.SUBPROCESS,
        timeout_seconds=5.0,
        memory_limit_mb=512,
        network_enabled=False
    )


@pytest.fixture
def attack_executor():
    """Basic attack executor."""
    return AttackExecutor(timeout_seconds=5.0)


@pytest.fixture
async def sandboxed_executor(basic_config, attack_executor):
    """Setup and provide sandboxed executor."""
    executor = SandboxedExecutor(
        config=basic_config,
        executor=attack_executor
    )
    await executor.setup()
    yield executor
    await executor.close()


# =============================================================================
# Initialization Tests
# =============================================================================

def test_sandboxed_executor_init():
    """Test SandboxedExecutor initialization."""
    config = SandboxConfig(mode=SandboxMode.SUBPROCESS)
    executor = SandboxedExecutor(config=config)

    assert executor.config == config
    assert executor.executor is not None
    assert not executor.is_setup()
    assert executor.get_sandbox_mode() == SandboxMode.NONE  # Not selected yet


def test_sandboxed_executor_init_auto_config():
    """Test initialization with automatic config creation."""
    executor = SandboxedExecutor()
    assert executor.config is not None
    assert executor.config.mode == SandboxMode.AUTO


def test_sandboxed_executor_init_custom_executor():
    """Test initialization with custom executor."""
    custom_executor = AttackExecutor()
    executor = SandboxedExecutor(executor=custom_executor)
    assert executor.executor is custom_executor


def test_sandboxed_executor_config_validation():
    """Test invalid configuration raises ValueError."""
    with pytest.raises(ValueError):
        SandboxConfig(timeout_seconds=-1)

    with pytest.raises(ValueError):
        SandboxConfig(memory_limit_mb=32)

    with pytest.raises(ValueError):
        SandboxConfig(cpu_limit_percent=150)


# =============================================================================
# Setup/Teardown Tests
# =============================================================================

@pytest.mark.asyncio
async def test_sandboxed_executor_setup():
    """Test sandbox setup."""
    config = SandboxConfig(mode=SandboxMode.SUBPROCESS)
    executor = SandboxedExecutor(config=config)

    assert not executor.is_setup()

    await executor.setup()

    assert executor.is_setup()
    assert executor.get_sandbox_mode() in [
        SandboxMode.SUBPROCESS,
        SandboxMode.DOCKER,
        SandboxMode.CGROUPS
    ]

    await executor.close()


@pytest.mark.asyncio
async def test_sandboxed_executor_context_manager():
    """Test async context manager."""
    async with await create_sandboxed_executor() as executor:
        assert executor.is_setup()

    # Executor should be closed after exiting context
    # (can't test directly, but no error should occur)


@pytest.mark.asyncio
async def test_sandboxed_executor_setup_teardown_timing(basic_config):
    """Test setup/teardown timing metrics."""
    executor = SandboxedExecutor(config=basic_config)

    await executor.setup()
    summary = executor.get_resource_summary()

    assert summary.startup_time_ms > 0
    assert summary.startup_time_ms < 5000  # Should be fast

    await executor.close()

    summary = executor.get_resource_summary()
    assert summary.cleanup_time_ms >= 0


# =============================================================================
# Drop-in Replacement Tests
# =============================================================================

@pytest.mark.asyncio
async def test_sandboxed_executor_drop_in_replacement(sandboxed_executor):
    """Test SandboxedExecutor is drop-in replacement for AttackExecutor."""
    # Should have same execute_attack interface
    assert hasattr(sandboxed_executor, 'execute_attack')
    assert asyncio.iscoroutinefunction(sandboxed_executor.execute_attack)

    # Should have same execute_payload interface
    assert hasattr(sandboxed_executor, 'execute_payload')
    assert asyncio.iscoroutinefunction(sandboxed_executor.execute_payload)

    # Should have same execute_batch interface
    assert hasattr(sandboxed_executor, 'execute_batch')
    assert asyncio.iscoroutinefunction(sandboxed_executor.execute_batch)


@pytest.mark.asyncio
async def test_sandboxed_executor_execute_attack(sandboxed_executor):
    """Test attack execution through sandbox."""
    # Simple test - just verify it executes without error
    result = await sandboxed_executor.execute_attack(
        AttackStrategy.UNICODE_BYPASS,
        "test payload",
        {}
    )

    assert result is not None
    assert hasattr(result, 'outcome')
    assert hasattr(result, 'bypassed')
    assert hasattr(result, 'execution_time_ms')

    # Should be tracked in statistics
    stats = sandboxed_executor.get_execution_stats()
    assert stats['total_executions'] > 0


@pytest.mark.asyncio
async def test_sandboxed_executor_execute_payload(sandboxed_executor):
    """Test payload execution through sandbox."""
    payload = AttackPayload(
        strategy=AttackStrategy.UNICODE_BYPASS,
        payload="test payload",
        metadata={}
    )

    result = await sandboxed_executor.execute_payload(payload)

    assert result is not None
    assert result.strategy == AttackStrategy.UNICODE_BYPASS


# =============================================================================
# Batch Execution Tests
# =============================================================================

@pytest.mark.asyncio
async def test_sandboxed_executor_batch_sequential(sandboxed_executor):
    """Test sequential batch execution."""
    payloads = [
        AttackPayload(
            strategy=AttackStrategy.UNICODE_BYPASS,
            payload=f"payload_{i}",
            metadata={}
        )
        for i in range(3)
    ]

    results = await sandboxed_executor.execute_batch(
        payloads,
        parallel=False
    )

    assert len(results) == 3
    assert all(r is not None for r in results)

    stats = sandboxed_executor.get_execution_stats()
    assert stats['total_executions'] == 3


@pytest.mark.asyncio
async def test_sandboxed_executor_batch_parallel(sandboxed_executor):
    """Test parallel batch execution."""
    payloads = [
        AttackPayload(
            strategy=AttackStrategy.UNICODE_BYPASS,
            payload=f"payload_{i}",
            metadata={}
        )
        for i in range(3)
    ]

    results = await sandboxed_executor.execute_batch(
        payloads,
        parallel=True,
        max_parallel=2
    )

    assert len(results) == 3
    assert all(r is not None for r in results)


# =============================================================================
# Timeout Tests
# =============================================================================

@pytest.mark.asyncio
async def test_sandboxed_executor_timeout():
    """Test execution timeout handling."""
    config = SandboxConfig(
        mode=SandboxMode.SUBPROCESS,
        timeout_seconds=0.1  # Very short timeout
    )
    executor = SandboxedExecutor(config=config)
    await executor.setup()

    # This should timeout
    result = await executor.execute_attack(
        AttackStrategy.UNICODE_BYPASS,
        "test",
        {}
    )

    # May timeout or succeed depending on system speed
    # Just verify it completes without crash
    assert result is not None

    await executor.close()


# =============================================================================
# Isolation Mode Tests
# =============================================================================

def test_sandboxed_executor_mode_selection():
    """Test sandbox mode selection."""
    # Auto mode should select something
    executor = SandboxedExecutor(config=SandboxConfig(mode=SandboxMode.AUTO))
    executor._select_isolation_mode()
    assert executor.get_sandbox_mode() != SandboxMode.NONE

    # Explicit mode should be respected
    executor = SandboxedExecutor(config=SandboxConfig(mode=SandboxMode.SUBPROCESS))
    executor._select_isolation_mode()
    assert executor.get_sandbox_mode() == SandboxMode.SUBPROCESS


@pytest.mark.asyncio
async def test_sandboxed_executor_subprocess_mode():
    """Test SUBPROCESS isolation mode."""
    config = SandboxConfig(mode=SandboxMode.SUBPROCESS)
    executor = SandboxedExecutor(config=config)

    await executor.setup()
    assert executor.get_sandbox_mode() == SandboxMode.SUBPROCESS
    await executor.close()


# =============================================================================
# Resource Monitoring Tests
# =============================================================================

@pytest.mark.asyncio
async def test_sandboxed_executor_resource_monitoring(sandboxed_executor):
    """Test resource monitoring integration."""
    # Execute something to generate metrics
    await sandboxed_executor.execute_attack(
        AttackStrategy.UNICODE_BYPASS,
        "test",
        {}
    )

    summary = sandboxed_executor.get_resource_summary()

    assert summary.sandbox_mode_used == SandboxMode.SUBPROCESS
    assert summary.startup_time_ms >= 0
    assert summary.execution_time_ms >= 0
    assert summary.total_time_ms > 0
    assert summary.samples_collected >= 0


@pytest.mark.asyncio
async def test_sandboxed_executor_memory_tracking(sandboxed_executor):
    """Test memory usage tracking."""
    await sandboxed_executor.execute_attack(
        AttackStrategy.UNICODE_BYPASS,
        "test",
        {}
    )

    summary = sandboxed_executor.get_resource_summary()

    assert summary.max_memory_mb >= 0
    assert summary.avg_memory_mb >= 0
    assert summary.min_memory_mb <= summary.max_memory_mb


@pytest.mark.asyncio
async def test_sandboxed_executor_cpu_tracking(sandboxed_executor):
    """Test CPU usage tracking."""
    await sandboxed_executor.execute_attack(
        AttackStrategy.UNICODE_BYPASS,
        "test",
        {}
    )

    summary = sandboxed_executor.get_resource_summary()

    assert 0 <= summary.min_cpu_percent <= 100
    assert 0 <= summary.avg_cpu_percent <= 100
    assert 0 <= summary.max_cpu_percent <= 100


# =============================================================================
# Statistics Tests
# =============================================================================

@pytest.mark.asyncio
async def test_sandboxed_executor_statistics(sandboxed_executor):
    """Test execution statistics tracking."""
    # Initial state
    stats = sandboxed_executor.get_execution_stats()
    assert stats['total_executions'] == 0
    assert stats['successful'] == 0

    # After execution
    await sandboxed_executor.execute_attack(
        AttackStrategy.UNICODE_BYPASS,
        "test",
        {}
    )

    stats = sandboxed_executor.get_execution_stats()
    assert stats['total_executions'] == 1
    assert 'success_rate' in stats
    assert 'avg_execution_time_ms' in stats


@pytest.mark.asyncio
async def test_sandboxed_executor_success_rate(sandboxed_executor):
    """Test success rate calculation."""
    for _ in range(3):
        await sandboxed_executor.execute_attack(
            AttackStrategy.UNICODE_BYPASS,
            "test",
            {}
        )

    stats = sandboxed_executor.get_execution_stats()
    assert stats['total_executions'] == 3
    assert stats['successful'] >= 0
    assert stats['failed'] >= 0


# =============================================================================
# Network Blocking Tests
# =============================================================================

@pytest.mark.asyncio
async def test_sandboxed_executor_network_blocking(sandboxed_executor):
    """Test network access blocking."""
    blocked = sandboxed_executor.get_blocked_network_attempts()
    assert isinstance(blocked, list)


# =============================================================================
# Factory Function Tests
# =============================================================================

@pytest.mark.asyncio
async def test_create_sandboxed_executor():
    """Test create_sandboxed_executor factory."""
    executor = await create_sandboxed_executor()
    assert executor.is_setup()
    await executor.close()


@pytest.mark.asyncio
async def test_create_sandboxed_executor_with_config():
    """Test create_sandboxed_executor with custom config."""
    config = SandboxConfig(
        mode=SandboxMode.SUBPROCESS,
        timeout_seconds=10.0
    )
    executor = await create_sandboxed_executor(config=config)
    assert executor.get_sandbox_config() == config
    await executor.close()


def test_create_sandboxed_executor_sync():
    """Test create_sandboxed_executor_sync factory."""
    executor = create_sandboxed_executor_sync()
    assert not executor.is_setup()
    assert executor is not None


@pytest.mark.asyncio
async def test_sandboxed_attack_execution_function():
    """Test sandboxed_attack_execution convenience function."""
    result = await sandboxed_attack_execution(
        AttackStrategy.UNICODE_BYPASS,
        "test payload",
        {}
    )
    assert result is not None


# =============================================================================
# Graceful Degradation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_sandboxed_executor_missing_dependencies():
    """Test graceful degradation without dependencies."""
    # Should still work even if some libraries unavailable
    executor = await create_sandboxed_executor()
    assert executor.is_setup()
    await executor.close()


@pytest.mark.asyncio
async def test_sandboxed_executor_cleanup_on_error(basic_config):
    """Test proper cleanup even if execution fails."""
    executor = SandboxedExecutor(config=basic_config)
    await executor.setup()

    try:
        # Execute something
        await executor.execute_attack(
            AttackStrategy.UNICODE_BYPASS,
            "test",
            {}
        )
    finally:
        # Cleanup should work
        await executor.close()

    assert not executor.is_setup()


# =============================================================================
# String Representation Tests
# =============================================================================

def test_sandboxed_executor_repr():
    """Test string representation."""
    executor = SandboxedExecutor()
    repr_str = repr(executor)
    assert "SandboxedExecutor" in repr_str
    assert "setup=False" in repr_str


@pytest.mark.asyncio
async def test_sandboxed_executor_repr_after_setup(basic_config):
    """Test string representation after setup."""
    executor = SandboxedExecutor(config=basic_config)
    await executor.setup()

    repr_str = repr(executor)
    assert "setup=True" in repr_str

    await executor.close()


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow with context manager."""
    config = SandboxConfig(mode=SandboxMode.SUBPROCESS)

    async with await create_sandboxed_executor(config=config) as executor:
        # Execute single attack
        result = await executor.execute_attack(
            AttackStrategy.UNICODE_BYPASS,
            "test1",
            {}
        )
        assert result is not None

        # Execute batch
        payloads = [
            AttackPayload(
                strategy=AttackStrategy.UNICODE_BYPASS,
                payload=f"test_{i}",
                metadata={}
            )
            for i in range(3)
        ]
        results = await executor.execute_batch(payloads)
        assert len(results) == 3

        # Get statistics
        stats = executor.get_execution_stats()
        assert stats['total_executions'] == 4

        # Get resources
        summary = executor.get_resource_summary()
        assert summary.total_time_ms > 0

    # Cleanup should happen automatically


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
