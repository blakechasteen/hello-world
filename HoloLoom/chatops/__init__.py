"""
HoloLoom ChatOps
================
Matrix.org chatbot integration with HoloLoom neural decision-making.

Organization:
    core/       - Core bot components (Matrix client, bridge, memory, skills)
    handlers/   - Advanced features (multimodal, threads, proactive)
    examples/   - Example implementations
    docs/       - Documentation
    deploy/     - Deployment scripts and tests

Quick Start:
    >>> from HoloLoom.chatops import ChatOpsRunner
    >>> import asyncio
    >>>
    >>> config = {
    ...     "matrix": {
    ...         "homeserver_url": "https://matrix.org",
    ...         "user_id": "@bot:matrix.org",
    ...         "access_token": "YOUR_TOKEN",
    ...         "rooms": ["#test:matrix.org"]
    ...     }
    ... }
    >>>
    >>> runner = ChatOpsRunner(config)
    >>> asyncio.run(runner.run())
"""

# Core components - always available
from HoloLoom.chatops.core import (
    MatrixBot,
    MatrixBotConfig,
    ChatOpsOrchestrator,
    ConversationContext,
    ConversationMemory,
    EntityType,
    RelationType,
    ChatOpsSkills,
    SkillResult,
    ChatOpsSkill,
)

# Main runner
try:
    from HoloLoom.chatops.run_chatops import ChatOpsRunner
    _RUNNER_AVAILABLE = True
except ImportError:
    _RUNNER_AVAILABLE = False

# Optional handlers - import if available
try:
    from HoloLoom.chatops.handlers import (
        MultimodalHandler,
        ThreadHandler,
        ProactiveAgent,
    )
    _HANDLERS_AVAILABLE = True
except ImportError:
    _HANDLERS_AVAILABLE = False

__all__ = [
    # Core
    "MatrixBot",
    "MatrixBotConfig",
    "ChatOpsOrchestrator",
    "ConversationContext",
    "ConversationMemory",
    "EntityType",
    "RelationType",
    "ChatOpsSkills",
    "SkillResult",
    "ChatOpsSkill",
]

if _RUNNER_AVAILABLE:
    __all__.append("ChatOpsRunner")

if _HANDLERS_AVAILABLE:
    __all__.extend(["MultimodalHandler", "ThreadHandler", "ProactiveAgent"])

__version__ = "0.2.0"  # Bumped for reorganization
