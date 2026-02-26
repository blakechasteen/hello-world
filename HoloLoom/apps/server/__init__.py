"""
HoloLoom HTTP Server
====================

FastAPI server exposing agentic intelligence to external clients
(VS Code extension, web apps, Agent Manager UI, etc).

Components:
- agentic_api: Main API server for agentic reasoning
- agent_manager_hub: Central orchestrator for multi-threaded agent management
- agent_manager_api: REST API for thread/swarm management
- agent_git_store: Git-based temporal rollback (dulwich)
- agent_manager_integration: Connects hub events to WebSocket broadcasts

Status: Production Ready (December 2025)
"""

from .agentic_api import app

__all__ = ["app"]

# Lazy imports for Agent Manager components (graceful degradation)
try:
    from .agent_manager_hub import (
        AgentManagerHub,
        AgentThread,
        TaskNode,
        ThreadStatus,
        ReasoningMode,
        TokenCostEstimate,
    )
    __all__.extend([
        "AgentManagerHub",
        "AgentThread",
        "TaskNode",
        "ThreadStatus",
        "ReasoningMode",
        "TokenCostEstimate",
    ])
except ImportError:
    pass

try:
    from .agent_manager_api import app as agent_manager_app, get_hub
    __all__.extend(["agent_manager_app", "get_hub"])
except ImportError:
    pass

try:
    from .agent_git_store import (
        AgentGitStore,
        StepCommit,
        InMemoryGitStore,
    )
    __all__.extend([
        "AgentGitStore",
        "StepCommit",
        "InMemoryGitStore",
    ])
except ImportError:
    pass

try:
    from .agent_manager_integration import (
        AgentManagerIntegration,
        create_integration,
        shutdown_integration,
        get_integration,
        create_integrated_app,
    )
    __all__.extend([
        "AgentManagerIntegration",
        "create_integration",
        "shutdown_integration",
        "get_integration",
        "create_integrated_app",
    ])
except ImportError:
    pass
