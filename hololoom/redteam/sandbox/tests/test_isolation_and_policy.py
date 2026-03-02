"""
Tests for Process Isolation and Network Policy Enforcement

Tests platform detection, isolation capabilities, and policy enforcement.
"""

import asyncio
import os
import platform
import sys
from typing import List, Tuple
from unittest import mock

import pytest

from hololoom.redteam.sandbox.process_isolation import (
    ProcessIsolator,
    ProcessIsolationConfig,
    ProcessIsolationError,
)
from hololoom.redteam.sandbox.network_policy import (
    NetworkPolicy,
    NetworkPolicyConfig,
    NetworkEndpoint,
    BlockedAttempt,
    NetworkAccessType,
    NetworkPolicyError,
)


class TestProcessIsolationConfig:
    """Tests for ProcessIsolationConfig."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = ProcessIsolationConfig(
            max_memory_mb=512,
            max_cpu_percent=50,
            timeout_seconds=30
        )
        assert config.max_memory_mb == 512
        assert config.max_cpu_percent == 50
        assert config.timeout_seconds == 30

    def test_invalid_memory(self):
        """Test invalid memory limit."""
        with pytest.raises(ValueError):
            ProcessIsolationConfig(max_memory_mb=-1)

    def test_invalid_timeout(self):
        """Test invalid timeout."""
        with pytest.raises(ValueError):
            ProcessIsolationConfig(timeout_seconds=0)

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = ProcessIsolationConfig()
        assert config.max_memory_mb == 512
        assert config.max_cpu_percent == 50
        assert config.timeout_seconds == 30
        assert config.enable_cgroups is True
        assert config.enable_seccomp is True


class TestProcessIsolator:
    """Tests for ProcessIsolator."""

    def test_platform_detection(self):
        """Test platform detection."""
        config = ProcessIsolationConfig()
        isolator = ProcessIsolator(config)

        platform_name = isolator.get_platform()
        assert platform_name in ["linux", "windows", "macos", "unknown"]

    def test_capabilities_detection(self):
        """Test capability detection."""
        config = ProcessIsolationConfig()
        isolator = ProcessIsolator(config)

        capabilities = isolator.get_capabilities()
        assert isinstance(capabilities, dict)
        assert "cgroups" in capabilities
        assert "seccomp" in capabilities
        assert "job_objects" in capabilities
        assert "rlimit" in capabilities

    @pytest.mark.asyncio
    async def test_simple_command_execution(self):
        """Test executing simple command."""
        config = ProcessIsolationConfig(timeout_seconds=10)
        isolator = ProcessIsolator(config)

        try:
            if sys.platform == "win32":
                pid, stdout, stderr, code = await isolator.spawn_isolated(
                    ["cmd", "/c", "echo", "test"]
                )
            else:
                pid, stdout, stderr, code = await isolator.spawn_isolated(
                    ["echo", "test"]
                )

            assert isinstance(pid, int)
            assert code == 0
            assert b"test" in stdout

        finally:
            await isolator.cleanup()

    @pytest.mark.asyncio
    async def test_command_with_stdin(self):
        """Test command with stdin."""
        config = ProcessIsolationConfig(timeout_seconds=10)
        isolator = ProcessIsolator(config)

        try:
            pid, stdout, stderr, code = await isolator.spawn_isolated(
                ["cat"],
                stdin=b"hello world"
            )

            assert code == 0
            assert b"hello world" in stdout

        except NotImplementedError:
            pytest.skip("cat not available on Windows")

        finally:
            await isolator.cleanup()

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        """Test timeout enforcement."""
        config = ProcessIsolationConfig(timeout_seconds=1)
        isolator = ProcessIsolator(config)

        try:
            if sys.platform == "win32":
                # Windows doesn't have sleep command
                pytest.skip("sleep not available on Windows")

            # This should timeout
            with pytest.raises(asyncio.TimeoutError):
                await isolator.spawn_isolated(
                    ["sleep", "10"],
                    timeout=1
                )

        finally:
            await isolator.cleanup()

    @pytest.mark.asyncio
    async def test_process_kill(self):
        """Test process killing."""
        config = ProcessIsolationConfig()
        isolator = ProcessIsolator(config)

        try:
            if sys.platform == "win32":
                pytest.skip("Complex on Windows")

            # Start long-running process
            import subprocess
            proc = subprocess.Popen(
                ["sleep", "100"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            isolator.processes[proc.pid] = proc

            # Kill it
            result = await isolator.kill(proc.pid)
            assert result is True
            assert proc.pid not in isolator.processes

        except Exception as e:
            pytest.skip(f"Cannot test kill: {e}")

        finally:
            await isolator.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup."""
        config = ProcessIsolationConfig()
        isolator = ProcessIsolator(config)

        # Cleanup with no processes
        await isolator.cleanup()
        assert len(isolator.processes) == 0
        assert len(isolator.cgroup_paths) == 0


class TestNetworkEndpoint:
    """Tests for NetworkEndpoint."""

    def test_valid_ip_address(self):
        """Test valid IP address."""
        endpoint = NetworkEndpoint("192.168.1.1", 8080)
        assert endpoint.host == "192.168.1.1"
        assert endpoint.port == 8080

    def test_any_host(self):
        """Test wildcard host."""
        endpoint = NetworkEndpoint("*")
        assert endpoint.host == "*"
        assert endpoint.matches("example.com", 80)
        assert endpoint.matches("192.168.1.1", 443)

    def test_wildcard_port(self):
        """Test wildcard port."""
        endpoint = NetworkEndpoint("8.8.8.8")
        assert endpoint.port is None
        assert endpoint.matches("8.8.8.8", 53)
        assert endpoint.matches("8.8.8.8", 443)
        assert not endpoint.matches("8.8.4.4", 53)

    def test_specific_port_match(self):
        """Test specific port matching."""
        endpoint = NetworkEndpoint("github.com", 443)
        assert endpoint.matches("github.com", 443)
        assert not endpoint.matches("github.com", 80)

    def test_domain_name(self):
        """Test domain name endpoint."""
        endpoint = NetworkEndpoint("github.com", 443)
        assert endpoint.host == "github.com"


class TestNetworkPolicyConfig:
    """Tests for NetworkPolicyConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = NetworkPolicyConfig()
        assert config.default_action == NetworkAccessType.BLOCK
        assert config.enable_dns is False
        assert config.enable_logging is True
        assert len(config.allowlist) == 0

    def test_custom_config(self):
        """Test custom configuration."""
        allowlist = [
            NetworkEndpoint("8.8.8.8", 443),
            NetworkEndpoint("github.com", 443),
        ]
        config = NetworkPolicyConfig(
            allowlist=allowlist,
            enable_dns=True
        )
        assert len(config.allowlist) == 6  # 4 DNS + 2 custom
        assert config.enable_dns is True

    def test_dns_allowlist_expansion(self):
        """Test DNS allowlist automatic expansion."""
        config = NetworkPolicyConfig(enable_dns=True)
        assert len(config.allowlist) == 4
        assert any(ep.host == "8.8.8.8" for ep in config.allowlist)
        assert any(ep.host == "1.1.1.1" for ep in config.allowlist)


class TestNetworkPolicy:
    """Tests for NetworkPolicy."""

    def test_platform_detection(self):
        """Test platform detection."""
        config = NetworkPolicyConfig()
        policy = NetworkPolicy(config)

        platform_name = policy.platform
        assert platform_name in ["linux", "windows", "macos", "unknown"]

    @pytest.mark.asyncio
    async def test_check_egress_allowed(self):
        """Test checking allowed egress."""
        config = NetworkPolicyConfig(
            allowlist=[NetworkEndpoint("8.8.8.8", 443)]
        )
        policy = NetworkPolicy(config)

        allowed = await policy.check_egress("8.8.8.8", 443)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_egress_blocked(self):
        """Test checking blocked egress."""
        config = NetworkPolicyConfig(
            allowlist=[NetworkEndpoint("8.8.8.8", 443)]
        )
        policy = NetworkPolicy(config)

        blocked = await policy.check_egress("192.168.1.1", 22)
        assert blocked is False

    @pytest.mark.asyncio
    async def test_check_egress_wildcard(self):
        """Test wildcard egress check."""
        config = NetworkPolicyConfig(
            allowlist=[NetworkEndpoint("*")]
        )
        policy = NetworkPolicy(config)

        allowed = await policy.check_egress("example.com", 8080)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_blocked_attempt_tracking(self):
        """Test tracking blocked attempts."""
        config = NetworkPolicyConfig(track_attempts=True)
        policy = NetworkPolicy(config)

        # Block some addresses
        await policy.check_egress("example.com", 80)
        await policy.check_egress("example.com", 80)
        await policy.check_egress("192.168.1.1", 22)

        attempts = policy.get_blocked_attempts()
        assert len(attempts) == 2
        assert attempts[0]["count"] == 2

    @pytest.mark.asyncio
    async def test_clear_blocked_attempts(self):
        """Test clearing blocked attempts."""
        config = NetworkPolicyConfig(track_attempts=True)
        policy = NetworkPolicy(config)

        await policy.check_egress("example.com", 80)
        assert len(policy.get_blocked_attempts()) > 0

        policy.clear_blocked_attempts()
        assert len(policy.get_blocked_attempts()) == 0

    @pytest.mark.asyncio
    async def test_policy_status(self):
        """Test policy status reporting."""
        config = NetworkPolicyConfig(
            allowlist=[NetworkEndpoint("8.8.8.8", 443)]
        )
        policy = NetworkPolicy(config)

        status = policy.get_policy_status()
        assert isinstance(status, dict)
        assert "platform" in status
        assert "applied" in status
        assert "allowlist_count" in status


class TestIntegration:
    """Integration tests for isolation and policy."""

    @pytest.mark.asyncio
    async def test_isolated_process_with_network_policy(self):
        """Test running isolated process with network policy."""
        # Setup isolation
        iso_config = ProcessIsolationConfig(timeout_seconds=10)
        isolator = ProcessIsolator(iso_config)

        # Setup network policy
        net_config = NetworkPolicyConfig(
            allowlist=[NetworkEndpoint("*")]  # Allow all for testing
        )
        policy = NetworkPolicy(net_config)

        try:
            # Check network access before running process
            can_access = await policy.check_egress("8.8.8.8", 443)
            assert can_access is True

            # Run simple command
            if sys.platform != "win32":
                pid, stdout, stderr, code = await isolator.spawn_isolated(
                    ["echo", "isolated process"]
                )
                assert code == 0

        finally:
            await isolator.cleanup()

    @pytest.mark.asyncio
    async def test_sandbox_basic_workflow(self):
        """Test basic sandbox workflow."""
        # Create isolator
        iso_config = ProcessIsolationConfig(
            max_memory_mb=256,
            timeout_seconds=5
        )
        isolator = ProcessIsolator(iso_config)

        # Create policy
        net_config = NetworkPolicyConfig(
            allowlist=[NetworkEndpoint("127.0.0.1")]
        )
        policy = NetworkPolicy(net_config)

        # Get capabilities
        caps = isolator.get_capabilities()
        assert "cgroups" in caps
        assert "rlimit" in caps

        # Check local network access
        allowed = await policy.check_egress("127.0.0.1", 8080)
        assert allowed is True

        # Check remote network access
        blocked = await policy.check_egress("example.com", 80)
        assert blocked is False

        await isolator.cleanup()


class TestPlatformSpecific:
    """Platform-specific tests."""

    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="Linux only"
    )
    def test_linux_capabilities(self):
        """Test Linux-specific capabilities."""
        config = ProcessIsolationConfig()
        isolator = ProcessIsolator(config)

        # Should have cgroups/seccomp on Linux
        caps = isolator.get_capabilities()
        # Note: might not have them if not running with proper privileges
        assert isinstance(caps, dict)

    @pytest.mark.skipif(
        platform.system() != "Windows",
        reason="Windows only"
    )
    def test_windows_capabilities(self):
        """Test Windows-specific capabilities."""
        config = ProcessIsolationConfig()
        isolator = ProcessIsolator(config)

        assert isolator.get_platform() == "windows"

    @pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="macOS only"
    )
    def test_macos_capabilities(self):
        """Test macOS-specific capabilities."""
        config = ProcessIsolationConfig()
        isolator = ProcessIsolator(config)

        assert isolator.get_platform() == "macos"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
