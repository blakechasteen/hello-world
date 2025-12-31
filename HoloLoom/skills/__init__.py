"""
HoloLoom Skills Package
=======================

**One-Line Skill Launching for HoloLoom**

Quick Start:
    from HoloLoom.skills import launch_skill, skill

    # Launch any skill with one line
    result = await launch_skill("code_reviewer", code="def foo(): pass")

    # Create skills with a decorator
    @skill(name="my_analyzer", category="domain")
    async def analyze(text: str) -> dict:
        return {"score": 0.95}

Skill Categories:
- code: Code analysis and testing
- communication: Communication platform integrations
- creative: Visual, audio, and document generation
- data: Database operations
- domain: Domain-specific data processing
- infrastructure: System operations and utilities
- system: System-level utilities
- testing: Testing and validation
- web: Web scraping and API interactions

Skills are automatically registered when their modules are imported.
"""

# ============================================================================
# Easy Launch API
# ============================================================================

from HoloLoom.skills.launcher import (
    # Primary API
    launch_skill,
    launch,
    launch_parallel,
    launch_chain,
    launch_sync,

    # Results
    LaunchResult,

    # Retry Configuration
    RetryConfig,
    RetryStrategy,

    # Context Managers
    skill_session,
    batch_launch,
    SkillSessionContext,

    # Health Checks
    check_skill_health,
    get_system_health,
    SkillHealthStatus,
    SystemHealthStatus,

    # Error Handling
    SkillError,
    SkillErrorCode,

    # Discovery
    list_skills,
    get_skill_info,
    get_launch_stats,
)

# ============================================================================
# Easy Skill Creation
# ============================================================================

from HoloLoom.skills.decorators import (
    # Main decorator
    skill,

    # Category-specific decorators
    code_skill,
    domain_skill,
    testing_skill,
    web_skill,
    system_skill,

    # Classes
    FunctionSkill,
    SkillBuilder,
)

# ============================================================================
# Base Classes (for advanced usage)
# ============================================================================

from HoloLoom.skills.base import (
    BaseSkill,
    SkillInput,
    SkillOutput,
    SkillStatus,
    SkillMetadata,
    SkillCategory,
    get_registry,
    register_skill,
)

from HoloLoom.skills.executor import (
    SkillExecutor,
    ExecutionResult,
    BatchResult,
)

# ============================================================================
# Import Skill Categories (auto-registration)
# ============================================================================

try:
    from . import code
except ImportError:
    pass

try:
    from . import communication
except ImportError:
    pass

try:
    from . import creative
except ImportError:
    pass

try:
    from . import data
except ImportError:
    pass

try:
    from . import domain
except ImportError:
    pass

try:
    from . import infrastructure
except ImportError:
    pass

try:
    from . import system
except ImportError:
    pass

try:
    from . import testing
except ImportError:
    pass

try:
    from . import web
except ImportError:
    pass

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Easy Launch API
    "launch_skill",
    "launch",
    "launch_parallel",
    "launch_chain",
    "launch_sync",
    "LaunchResult",
    "list_skills",
    "get_skill_info",
    "get_launch_stats",

    # Retry Configuration
    "RetryConfig",
    "RetryStrategy",

    # Context Managers
    "skill_session",
    "batch_launch",
    "SkillSessionContext",

    # Health Checks
    "check_skill_health",
    "get_system_health",
    "SkillHealthStatus",
    "SystemHealthStatus",

    # Error Handling
    "SkillError",
    "SkillErrorCode",

    # Easy Skill Creation
    "skill",
    "code_skill",
    "domain_skill",
    "testing_skill",
    "web_skill",
    "system_skill",
    "FunctionSkill",
    "SkillBuilder",

    # Base Classes
    "BaseSkill",
    "SkillInput",
    "SkillOutput",
    "SkillStatus",
    "SkillMetadata",
    "SkillCategory",
    "get_registry",
    "register_skill",
    "SkillExecutor",
    "ExecutionResult",
    "BatchResult",
]
