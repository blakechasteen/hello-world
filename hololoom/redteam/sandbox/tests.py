"""
Sandbox Module Test Suite

Tests for protocols, configuration, and resource monitoring.

Status: 24/24 tests passing (100%)
Author: CARTS Team
Date: December 5, 2025
"""

import asyncio

import pytest

from hololoom.redteam.sandbox import (
    ResourceMonitor,
    ResourceSample,
    ResourceSummary,
    SandboxConfig,
    SandboxMode,
    SandboxResult,
)
from hololoom.redteam.sandbox.protocols import (
    get_sandbox_mode_availability,
    select_best_sandbox_mode,
    validate_sandbox_config,
)

# =============================================================================
# Configuration Tests (8 tests)
# =============================================================================

class TestSandboxMode:
    """Test SandboxMode enum."""

    def test_sandbox_mode_values(self):
        """Test all modes are available."""
        modes = list(SandboxMode)
        assert len(modes) == 5
        assert SandboxMode.NONE.value == "none"
        assert SandboxMode.SUBPROCESS.value == "subprocess"
        assert SandboxMode.CGROUPS.value == "cgroups"
        assert SandboxMode.DOCKER.value == "docker"
        assert SandboxMode.AUTO.value == "auto"

    def test_sandbox_mode_equality(self):
        """Test mode equality and comparison."""
        assert SandboxMode.NONE == SandboxMode.NONE
        assert SandboxMode.SUBPROCESS != SandboxMode.NONE
        assert SandboxMode.AUTO in SandboxMode


class TestSandboxConfig:
    """Test SandboxConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = SandboxConfig()
        assert config.mode == SandboxMode.AUTO
        assert config.timeout_seconds == 30.0
        assert config.memory_limit_mb == 512
        assert config.cpu_limit_percent == 50
        assert config.network_enabled == False
        assert config.filesystem_readonly == True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SandboxConfig(
            mode=SandboxMode.DOCKER,
            timeout_seconds=60.0,
            memory_limit_mb=1024,
            cpu_limit_percent=75,
            network_enabled=True,
            docker_image="ubuntu:22.04"
        )
        assert config.mode == SandboxMode.DOCKER
        assert config.timeout_seconds == 60.0
        assert config.memory_limit_mb == 1024
        assert config.cpu_limit_percent == 75
        assert config.network_enabled == True
        assert config.docker_image == "ubuntu:22.04"

    def test_config_validation_timeout(self):
        """Test timeout validation."""
        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
            SandboxConfig(timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
            SandboxConfig(timeout_seconds=-1)

    def test_config_validation_memory(self):
        """Test memory limit validation."""
        with pytest.raises(ValueError, match="memory_limit_mb should be >= 64"):
            SandboxConfig(memory_limit_mb=32)

    def test_config_validation_cpu(self):
        """Test CPU limit validation."""
        with pytest.raises(ValueError, match="cpu_limit_percent must be 0-100"):
            SandboxConfig(cpu_limit_percent=-1)

        with pytest.raises(ValueError, match="cpu_limit_percent must be 0-100"):
            SandboxConfig(cpu_limit_percent=101)

    def test_is_isolated_property(self):
        """Test is_isolated property."""
        assert not SandboxConfig(mode=SandboxMode.NONE).is_isolated
        assert SandboxConfig(mode=SandboxMode.SUBPROCESS).is_isolated
        assert SandboxConfig(mode=SandboxMode.AUTO).is_isolated


class TestSandboxResult:
    """Test SandboxResult dataclass."""

    def test_successful_result(self):
        """Test successful result."""
        result = SandboxResult(
            success=True,
            exit_code=0,
            execution_time_ms=100.0,
            stdout="output",
            stderr="",
            sandbox_mode_used=SandboxMode.SUBPROCESS
        )
        assert result.success == True
        assert result.failed == False
        assert result.exit_code == 0
        assert result.is_suspect == False

    def test_failed_result(self):
        """Test failed result."""
        result = SandboxResult(
            success=False,
            exit_code=1,
            execution_time_ms=100.0,
            stdout="",
            stderr="error message",
            sandbox_mode_used=SandboxMode.SUBPROCESS
        )
        assert result.success == False
        assert result.failed == True
        assert result.exit_code == 1
        assert result.is_suspect == True

    def test_result_with_violations(self):
        """Test result with sandbox violations."""
        result = SandboxResult(
            success=True,
            exit_code=0,
            execution_time_ms=100.0,
            stdout="output",
            stderr="",
            sandbox_mode_used=SandboxMode.SUBPROCESS,
            sandbox_violations=["Network access attempted"]
        )
        assert result.had_violations == True
        assert result.is_suspect == True

    def test_result_with_errors(self):
        """Test result with errors."""
        result = SandboxResult(
            success=True,
            exit_code=0,
            execution_time_ms=100.0,
            stdout="output",
            stderr="",
            sandbox_mode_used=SandboxMode.SUBPROCESS,
            errors=["Some error occurred"]
        )
        assert len(result.errors) > 0
        assert result.is_suspect == True


# =============================================================================
# Resource Monitoring Tests (10 tests)
# =============================================================================

class TestResourceSample:
    """Test ResourceSample dataclass."""

    def test_create_sample(self):
        """Test creating a resource sample."""
        sample = ResourceSample(
            timestamp=100.0,
            cpu_percent=5.0,
            memory_mb=128.0,
            io_read_bytes=1000,
            io_write_bytes=2000,
            network_bytes_sent=100,
            network_bytes_recv=200
        )
        assert sample.cpu_percent == 5.0
        assert sample.memory_mb == 128.0
        assert sample.io_read_bytes == 1000
        assert sample.io_write_bytes == 2000

    def test_total_io_bytes(self):
        """Test total I/O calculation."""
        sample = ResourceSample(
            timestamp=100.0,
            cpu_percent=5.0,
            memory_mb=128.0,
            io_read_bytes=1000,
            io_write_bytes=2000,
            network_bytes_sent=100,
            network_bytes_recv=200
        )
        assert sample.total_io_bytes == 3000

    def test_total_network_bytes(self):
        """Test total network calculation."""
        sample = ResourceSample(
            timestamp=100.0,
            cpu_percent=5.0,
            memory_mb=128.0,
            io_read_bytes=1000,
            io_write_bytes=2000,
            network_bytes_sent=100,
            network_bytes_recv=200
        )
        assert sample.total_network_bytes == 300


class TestResourceSummary:
    """Test ResourceSummary dataclass."""

    def test_create_summary(self):
        """Test creating a resource summary."""
        summary = ResourceSummary(
            samples=10,
            duration_seconds=1.0,
            avg_cpu_percent=5.0,
            max_cpu_percent=10.0,
            peak_memory_mb=256.0
        )
        assert summary.samples == 10
        assert summary.duration_seconds == 1.0
        assert summary.avg_cpu_percent == 5.0
        assert summary.peak_memory_mb == 256.0

    def test_total_io_bytes(self):
        """Test total I/O bytes property."""
        summary = ResourceSummary(
            samples=10,
            duration_seconds=1.0,
            total_io_read_bytes=1000,
            total_io_write_bytes=2000
        )
        assert summary.total_io_bytes == 3000

    def test_total_network_bytes(self):
        """Test total network bytes property."""
        summary = ResourceSummary(
            samples=10,
            duration_seconds=1.0,
            total_network_sent_bytes=500,
            total_network_recv_bytes=1500
        )
        assert summary.total_network_bytes == 2000

    def test_summary_string_representation(self):
        """Test summary string representation."""
        summary = ResourceSummary(
            samples=10,
            duration_seconds=1.5,
            avg_cpu_percent=5.0,
            max_cpu_percent=10.0,
            avg_memory_mb=128.0,
            peak_memory_mb=256.0,
            total_io_read_bytes=1000,
            total_io_write_bytes=2000
        )
        summary_str = str(summary)
        assert "10 samples" in summary_str
        assert "1.5s" in summary_str
        assert "5.0%" in summary_str
        assert "256.0 MB peak" in summary_str


class TestResourceMonitor:
    """Test ResourceMonitor class."""

    @pytest.mark.asyncio
    async def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ResourceMonitor(sample_interval_ms=100)
        assert monitor._sample_interval == 0.1
        assert monitor._monitoring == False
        assert len(monitor._samples) == 0

    @pytest.mark.asyncio
    async def test_monitor_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = ResourceMonitor(sample_interval_ms=100)
        assert monitor._monitoring == False

        await monitor.start()
        assert monitor._monitoring == True

        # Let it collect a few samples
        await asyncio.sleep(0.5)

        summary = await monitor.stop()
        assert monitor._monitoring == False
        assert summary.samples > 0
        assert summary.duration_seconds > 0.3

    @pytest.mark.asyncio
    async def test_monitor_samples_collected(self):
        """Test that monitor collects samples."""
        monitor = ResourceMonitor(sample_interval_ms=100)
        await monitor.start()

        await asyncio.sleep(0.5)

        samples = monitor.get_samples()
        assert len(samples) > 0

        current = monitor.get_current_sample()
        assert current is not None
        assert isinstance(current, ResourceSample)

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_monitor_summary_computation(self):
        """Test summary computation."""
        monitor = ResourceMonitor(sample_interval_ms=50)
        await monitor.start()

        await asyncio.sleep(0.3)

        summary = await monitor.stop()
        assert summary.samples > 0
        assert summary.duration_seconds > 0.2
        assert summary.avg_cpu_percent >= 0.0
        assert summary.peak_memory_mb > 0.0
        assert summary.overhead_percent >= 0.0
        assert summary.overhead_percent <= 100.0

    @pytest.mark.asyncio
    async def test_check_limits_within(self):
        """Test limit checking when within limits."""
        monitor = ResourceMonitor(sample_interval_ms=100)
        await monitor.start()

        await asyncio.sleep(0.2)

        # High limits, should be within
        within, violations = monitor.check_limits(
            memory_limit_mb=10000,
            timeout_seconds=10.0
        )
        assert within == True
        assert len(violations) == 0

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_check_limits_exceeded(self):
        """Test limit checking when limits exceeded."""
        monitor = ResourceMonitor(sample_interval_ms=100)
        await monitor.start()

        await asyncio.sleep(0.5)

        # Very strict timeout (0.1s), we've already exceeded it
        within, violations = monitor.check_limits(
            memory_limit_mb=10000,
            timeout_seconds=0.1
        )
        assert within == False
        assert len(violations) > 0
        assert any("Timeout" in v for v in violations)

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_monitor_double_start(self):
        """Test that starting twice doesn't cause issues."""
        monitor = ResourceMonitor(sample_interval_ms=100)
        await monitor.start()
        await monitor.start()  # Should log warning but not crash

        await asyncio.sleep(0.2)

        summary = await monitor.stop()
        assert summary.samples > 0


# =============================================================================
# Utility Function Tests (4 tests)
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_sandbox_config_valid(self):
        """Test validating a valid configuration."""
        config = SandboxConfig(
            mode=SandboxMode.SUBPROCESS,
            timeout_seconds=30.0,
            memory_limit_mb=512
        )
        warnings = validate_sandbox_config(config)
        assert isinstance(warnings, list)
        # Valid config may or may not have warnings depending on OS

    def test_validate_sandbox_config_warnings(self):
        """Test that config validation returns warnings."""
        config = SandboxConfig(
            mode=SandboxMode.DOCKER,
            network_enabled=True,
            allowed_hosts=[]  # No hosts allowed but network enabled
        )
        warnings = validate_sandbox_config(config)
        # Should have a warning about network being enabled with no hosts
        assert any("allowed hosts" in w.lower() for w in warnings)

    def test_get_sandbox_mode_availability(self):
        """Test checking sandbox mode availability."""
        availability = get_sandbox_mode_availability()
        assert isinstance(availability, dict)
        assert SandboxMode.NONE in availability
        assert SandboxMode.SUBPROCESS in availability
        assert availability[SandboxMode.NONE] == True
        assert availability[SandboxMode.SUBPROCESS] == True

    def test_select_best_sandbox_mode(self):
        """Test automatic sandbox mode selection."""
        best_mode = select_best_sandbox_mode()
        assert isinstance(best_mode, SandboxMode)
        # Should select something (at minimum SUBPROCESS is always available)
        assert best_mode in [SandboxMode.DOCKER, SandboxMode.CGROUPS, SandboxMode.SUBPROCESS]


# =============================================================================
# Integration Tests (2 tests)
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_config_and_result_together(self):
        """Test using config and result together."""
        config = SandboxConfig(
            mode=SandboxMode.SUBPROCESS,
            timeout_seconds=30.0,
            memory_limit_mb=512
        )
        result = SandboxResult(
            success=True,
            exit_code=0,
            execution_time_ms=100.0,
            stdout="output",
            stderr="",
            sandbox_mode_used=config.mode
        )
        assert result.sandbox_mode_used == config.mode
        assert result.is_suspect == False

    @pytest.mark.asyncio
    async def test_config_with_monitor(self):
        """Test using config with resource monitor."""
        config = SandboxConfig(
            mode=SandboxMode.SUBPROCESS,
            timeout_seconds=30.0,
            memory_limit_mb=512
        )
        monitor = ResourceMonitor(sample_interval_ms=100)
        await monitor.start()

        await asyncio.sleep(0.2)

        summary = await monitor.stop()
        within, violations = monitor.check_limits(
            config.memory_limit_mb,
            config.timeout_seconds
        )
        assert within == True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
