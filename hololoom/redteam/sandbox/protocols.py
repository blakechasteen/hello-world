"""
Sandbox Protocol Definitions for CARTS Phase 2
===============================================

Defines sandboxing protocols and configuration structures for safe,
isolated attack execution.

Protocols:
- ProcessIsolationProtocol: Spawn/manage isolated processes
- NetworkPolicyProtocol: Control network access
- FilesystemSandboxProtocol: Filesystem isolation (overlay, readonly)

Author: CARTS Team
Date: 2025-12-05
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Awaitable
from typing_extensions import Protocol, runtime_checkable
import logging

logger = logging.getLogger("hololoom.redteam.sandbox.protocols")


# =============================================================================
# Enums
# =============================================================================

class SandboxMode(Enum):
    """Sandbox isolation modes with increasing security/overhead tradeoff."""

    NONE = "none"              # No isolation (testing/development only)
    SUBPROCESS = "subprocess"   # Basic subprocess with resource limits
    CGROUPS = "cgroups"        # Linux cgroups + seccomp filtering
    DOCKER = "docker"          # Full Docker container isolation
    AUTO = "auto"              # Auto-detect best available


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    # Sandbox mode
    mode: SandboxMode = SandboxMode.AUTO

    # Execution limits
    timeout_seconds: float = 30.0
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50

    # Network policy
    network_enabled: bool = False  # Block egress by default
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)

    # Filesystem policy
    filesystem_readonly: bool = True
    allowed_read_paths: List[str] = field(default_factory=list)
    allowed_write_paths: List[str] = field(default_factory=list)

    # Overlay filesystem (Linux only)
    overlay_root: Optional[str] = None
    overlay_lower: Optional[str] = None
    overlay_upper: Optional[str] = None

    # Docker-specific
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"  # 'none', 'host', or network name

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.memory_limit_mb < 64:
            raise ValueError(f"memory_limit_mb should be >= 64 MB, got {self.memory_limit_mb}")
        if not (0 <= self.cpu_limit_percent <= 100):
            raise ValueError(f"cpu_limit_percent must be 0-100, got {self.cpu_limit_percent}")

    @property
    def is_isolated(self) -> bool:
        """Check if configuration has any isolation enabled."""
        return self.mode != SandboxMode.NONE


@dataclass
class SandboxResult:
    """Result of sandbox execution."""

    # Execution status
    success: bool
    exit_code: int
    execution_time_ms: float

    # Output streams
    stdout: str
    stderr: str

    # Resource usage
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    # Sandbox details
    sandbox_mode_used: SandboxMode = SandboxMode.NONE
    sandbox_violations: List[str] = field(default_factory=list)

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Check if execution failed."""
        return not self.success

    @property
    def had_violations(self) -> bool:
        """Check if sandbox had any violations."""
        return len(self.sandbox_violations) > 0

    @property
    def is_suspect(self) -> bool:
        """Check if result is suspect (errors, violations, or non-zero exit)."""
        return (
            self.failed or
            self.exit_code != 0 or
            self.had_violations or
            len(self.errors) > 0
        )


# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class ProcessIsolationProtocol(Protocol):
    """
    Protocol for spawning and managing isolated processes.

    Implementations:
    - SubprocessIsolator: Basic subprocess with resource limits
    - CGroupsIsolator: Linux cgroups + seccomp filtering
    - DockerIsolator: Docker container execution
    """

    async def spawn(
        self,
        command: List[str],
        env: Dict[str, str],
        cwd: Optional[str] = None
    ) -> SandboxResult:
        """
        Spawn an isolated process.

        Args:
            command: Command and arguments to execute
            env: Environment variables for process
            cwd: Working directory

        Returns:
            SandboxResult with execution outcome
        """
        ...

    async def kill(self) -> None:
        """Kill the running process (if any)."""
        ...

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage metrics."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources (temp files, containers, etc)."""
        ...


@runtime_checkable
class NetworkPolicyProtocol(Protocol):
    """
    Protocol for controlling network access from sandbox.

    Implementations:
    - NetfilterPolicy: Linux netfilter/iptables
    - EbpfPolicy: Linux eBPF network filters
    - DockerNetworkPolicy: Docker network isolation
    """

    def block_egress(self) -> None:
        """Block all outgoing network traffic."""
        ...

    def allow_host(self, host: str, port: int) -> None:
        """Allow traffic to specific host:port."""
        ...

    def allow_dns(self) -> None:
        """Allow DNS resolution (53/UDP)."""
        ...

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get list of network policy violations."""
        ...

    async def cleanup(self) -> None:
        """Clean up network policies."""
        ...


@runtime_checkable
class FilesystemSandboxProtocol(Protocol):
    """
    Protocol for filesystem isolation.

    Implementations:
    - OverlayfsIsolator: Linux overlayfs mount
    - BindMountIsolator: Linux bind mount with options
    - DockerFilesystemIsolator: Docker volume/mount isolation
    """

    def mount_readonly(self, path: str) -> None:
        """Mount path as read-only."""
        ...

    def mount_overlay(self, lower: str, upper: str, work: str) -> str:
        """
        Mount overlay filesystem.

        Returns: Path to mounted overlay root
        """
        ...

    def allow_read(self, path: str) -> None:
        """Allow read access to path."""
        ...

    def allow_write(self, path: str) -> None:
        """Allow write access to path."""
        ...

    async def cleanup(self) -> None:
        """Unmount overlays and clean up."""
        ...


# =============================================================================
# Utility Functions
# =============================================================================

def validate_sandbox_config(config: SandboxConfig) -> List[str]:
    """
    Validate sandbox configuration.

    Returns: List of warnings (empty if valid)
    """
    warnings = []

    # Mode-specific validation
    if config.mode == SandboxMode.CGROUPS and config.overlay_root:
        # cgroups + overlay might have permission issues
        warnings.append(
            "CGROUPS mode with overlay filesystem may require elevated privileges"
        )

    if config.mode == SandboxMode.DOCKER and not config.docker_image:
        warnings.append("DOCKER mode requires docker_image to be specified")

    # Network validation
    if config.network_enabled and not config.allowed_hosts:
        warnings.append(
            "Network enabled but no allowed hosts specified - all traffic allowed!"
        )

    # Filesystem validation
    if config.filesystem_readonly and config.allowed_write_paths:
        warnings.append(
            "Filesystem is readonly but write paths are allowed - behavior may be unexpected"
        )

    return warnings


def get_sandbox_mode_availability() -> Dict[SandboxMode, bool]:
    """
    Check availability of different sandbox modes on this system.

    Returns: Dict mapping SandboxMode to whether it's available
    """
    availability = {
        SandboxMode.NONE: True,  # Always available
        SandboxMode.SUBPROCESS: True,  # Always available
        SandboxMode.CGROUPS: False,  # Will be detected
        SandboxMode.DOCKER: False,  # Will be detected
    }

    # Check for cgroups (Linux only)
    import os
    if os.path.exists("/proc/self/cgroup"):
        availability[SandboxMode.CGROUPS] = True

    # Check for Docker
    try:
        import subprocess
        subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=2
        )
        availability[SandboxMode.DOCKER] = True
    except Exception:
        pass

    return availability


def select_best_sandbox_mode() -> SandboxMode:
    """
    Auto-select the best available sandbox mode.

    Preference: DOCKER > CGROUPS > SUBPROCESS > NONE
    """
    availability = get_sandbox_mode_availability()

    if availability[SandboxMode.DOCKER]:
        return SandboxMode.DOCKER
    elif availability[SandboxMode.CGROUPS]:
        return SandboxMode.CGROUPS
    else:
        return SandboxMode.SUBPROCESS
