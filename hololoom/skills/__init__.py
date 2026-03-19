"""
HoloLoom Skills Package
=======================

**One-Line Skill Launching for HoloLoom**

Quick Start:
    from hololoom.skills import launch_skill, skill

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

# ============================================================================
# Base Classes (for advanced usage)
# ============================================================================
from hololoom.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillInput,
    SkillMetadata,
    SkillOutput,
    SkillStatus,
    get_registry,
    register_skill,
)

# ============================================================================
# Easy Skill Creation
# ============================================================================
from hololoom.skills.decorators import (
    # Classes
    FunctionSkill,
    SkillBuilder,
    # Category-specific decorators
    code_skill,
    domain_skill,
    # Main decorator
    skill,
    system_skill,
    testing_skill,
    web_skill,
)
from hololoom.skills.executor import (
    BatchResult,
    ExecutionResult,
    SkillExecutor,
)
from hololoom.skills.launcher import (
    # Results
    LaunchResult,
    # Retry Configuration
    RetryConfig,
    RetryStrategy,
    # Error Handling
    SkillError,
    SkillErrorCode,
    SkillHealthStatus,
    SkillSessionContext,
    SystemHealthStatus,
    batch_launch,
    # Health Checks
    check_skill_health,
    get_launch_stats,
    get_skill_info,
    get_system_health,
    launch,
    launch_chain,
    launch_parallel,
    # Primary API
    launch_skill,
    launch_sync,
    # Discovery
    list_skills,
    # Context Managers
    skill_session,
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
