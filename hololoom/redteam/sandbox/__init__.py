"""
Sandbox Isolation for CARTS Red Team Execution
===============================================

Phase 2 Foundation: Sandbox isolation for safe attack execution.

Implements multiple isolation strategies:
- NONE: No isolation (testing/development only)
- SUBPROCESS: Basic subprocess isolation with resource limits
- CGROUPS: Linux cgroups + seccomp filtering (Linux only)
- DOCKER: Full Docker container execution (requires Docker)
- AUTO: Auto-detect best available (recommended)

Key Components:
- SandboxMode, SandboxConfig: Configuration
- ProcessIsolationProtocol: Isolated process execution
- NetworkPolicyProtocol: Network access control
- FilesystemSandboxProtocol: Filesystem isolation
- ResourceMonitor: <5% overhead resource tracking

Usage:
    from HoloLoom.redteam.sandbox import (
        SandboxMode, SandboxConfig, SandboxManager
    )

    config = SandboxConfig(
        mode=SandboxMode.AUTO,
        timeout_seconds=30.0,
        memory_limit_mb=512,
        network_enabled=False
    )

    manager = SandboxManager(config)
    result = await manager.execute(
        ["python", "-c", "print('safe execution')"],
        {}
    )
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")

Philosophy:
    "Isolate attacks, understand constraints, learn safely."

    CARTS Phase 2 uses sandboxing to safely execute attack
    payloads, monitor resource usage, prevent system damage,
    and extract failure patterns for learning.

Author: CARTS Team
Date: 2025-12-05
"""

# =============================================================================
# Protocols
# =============================================================================

from .protocols import (
    SandboxMode,
    SandboxConfig,
    SandboxResult,
    ProcessIsolationProtocol,
    NetworkPolicyProtocol,
    FilesystemSandboxProtocol,
)

# =============================================================================
# Monitoring
# =============================================================================

from .monitor import (
    ResourceSample,
    ResourceSummary,
    ResourceMonitor,
)

# =============================================================================
# Sandboxed Executor (Main Integration)
# =============================================================================

from .sandboxed_executor import (
    SandboxedExecutor,
    create_sandboxed_executor,
    create_sandboxed_executor_sync,
    sandboxed_attack_execution,
)

# =============================================================================
# Filesystem Isolation (NEW - November 2025)
# =============================================================================

from . import filesystem as _fs_module

FilesystemSandbox = _fs_module.FilesystemSandbox
FilesystemBackend = _fs_module.SandboxBackend
FilesystemSandboxConfig = _fs_module.SandboxConfig
OverlayMount = _fs_module.OverlayMount
FilesystemSandboxResult = _fs_module.SandboxResult
create_filesystem_sandbox = _fs_module.create_filesystem_sandbox
mount_isolated_environment = _fs_module.mount_isolated_environment

# =============================================================================
# Container Execution (NEW - November 2025)
# =============================================================================

from . import container as _container_module

ContainerExecutor = _container_module.ContainerExecutor
ContainerBackend = _container_module.ContainerBackend
ContainerSandboxConfig = _container_module.SandboxConfig
ResourceLimits = _container_module.ResourceLimits
ContainerSandboxResult = _container_module.SandboxResult
create_container_executor = _container_module.create_container_executor
execute_in_container = _container_module.execute_in_container

# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Protocols & Config
    'SandboxMode',
    'SandboxConfig',
    'SandboxResult',
    'ProcessIsolationProtocol',
    'NetworkPolicyProtocol',
    'FilesystemSandboxProtocol',

    # Monitoring
    'ResourceSample',
    'ResourceSummary',
    'ResourceMonitor',

    # Sandboxed Executor (Phase 2 - November 2025)
    'SandboxedExecutor',
    'create_sandboxed_executor',
    'create_sandboxed_executor_sync',
    'sandboxed_attack_execution',

    # Filesystem Isolation (NEW - November 2025)
    'FilesystemSandbox',
    'FilesystemBackend',
    'FilesystemSandboxConfig',
    'OverlayMount',
    'FilesystemSandboxResult',
    'create_filesystem_sandbox',
    'mount_isolated_environment',

    # Container Execution (NEW - November 2025)
    'ContainerExecutor',
    'ContainerBackend',
    'ContainerSandboxConfig',
    'ResourceLimits',
    'ContainerSandboxResult',
    'create_container_executor',
    'execute_in_container',
]

__version__ = '0.2.0'  # Phase 2 with SandboxedExecutor Integration
__author__ = 'CARTS Team'
