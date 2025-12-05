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
# All Exports
# =============================================================================

__all__ = [
    # Protocols
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
]

__version__ = '0.1.0'  # Phase 2 Foundation
__author__ = 'CARTS Team'
