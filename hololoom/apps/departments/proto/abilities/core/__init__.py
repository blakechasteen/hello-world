"""Proto Core Abilities - Tier 1 Skill Wrappers + Tier 2 Plugins.

HoloLoom's 13 professional skills exposed through the Proto ability system.

Available Skills (13 total):

Base Skills (8):
    - review: Code review with best practices
    - explain: Explain code and concepts
    - test: Generate unit tests
    - document: Generate documentation
    - debug: Debug issues and errors
    - refactor: Refactor code for quality
    - architect: System architecture design
    - sql: SQL database design

Extended Skills (5):
    - performance: SQL query optimization
    - security: Security auditing (OWASP)
    - api: API design (RESTful + OpenAPI)
    - name: Code naming consultant
    - migrate: Database migration planning

Plugin Abilities (Tier 2):
    - git_operations: Safe read-only Git repository operations
    - code_execution: Safe Python code execution with sandboxing
    - test_runner: Run tests with pytest/unittest
    - security_scan: Static security analysis

Status: Production ready (2025-12-04)
"""

from .skill_wrapper import (
    SkillWrapperAbility,
    SKILL_MAP,
    create_skill_abilities,
    get_available_skills,
    get_skill_description,
    get_skill_tags,
)
from .git_operations import GitOperationsAbility
from .code_execution import (
    CodeExecutionAbility,
    CodeExecutionConfig,
    create_code_execution_ability,
)
from .test_runner import TestRunnerAbility
from .security_scan import SecurityScanAbility

__all__ = [
    # Tier 1: Skill wrappers
    "SkillWrapperAbility",
    "SKILL_MAP",
    "create_skill_abilities",
    "get_available_skills",
    "get_skill_description",
    "get_skill_tags",
    # Tier 2: Plugin abilities
    "GitOperationsAbility",
    "CodeExecutionAbility",
    "CodeExecutionConfig",
    "create_code_execution_ability",
    "TestRunnerAbility",
    "SecurityScanAbility",
]
