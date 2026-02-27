"""
HoloLoom Skills System
======================
Production-ready skill execution using YAML templates.

Integrated from Promptly (November 2025) - 13 skill templates for common AI tasks.

This module provides:
- 13 pre-built skill templates (YAML-based)
- Skill registry for loading and managing templates
- Skill executor using HoloLoom's recursive weaving
- Integration with agentic reasoning modes (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)

Quick Start:
-----------
```python
from hololoom.agentic.skills import execute_skill, list_available_skills
from hololoom.config import Config

# List available skills
skills = await list_available_skills()
print(skills)  # {'development': ['code_reviewer', 'bug_detective', ...], ...}

# Execute a skill
result = await execute_skill(
    skill_name="code_reviewer",
    parameters={"code": code, "language": "python"},
    config=Config.fast()
)

print(result.output)
print(f"Confidence: {result.confidence:.2f}")
print(f"Iterations: {result.iterations}")
```

Available Skills (13):
----------------------
Development:
- code_reviewer - Review code for best practices, bugs, security
- bug_detective - Deep analysis of bugs with root cause investigation
- test_generator - Generate comprehensive unit tests
- code_explainer - Explain code functionality and patterns
- refactoring_expert - Refactor code for maintainability
- naming_consultant - Suggest better names for variables/functions

Architecture & Design:
- api_designer - Design RESTful APIs with OpenAPI specs
- architecture_advisor - Design scalable system architectures
- migration_planner - Plan migrations between systems

Performance & Security:
- performance_profiler - Analyze and optimize performance
- security_auditor - Audit code for security vulnerabilities
- sql_optimizer - Optimize SQL queries

Documentation:
- documentation_writer - Generate comprehensive documentation

Architecture:
------------
Skills are YAML templates loaded by SkillRegistry and executed via SkillExecutor
using HoloLoom's RecursiveWeavingOrchestrator for quality-driven refinement.

Flow:
1. Load YAML template → SkillTemplate
2. Validate parameters
3. Build prompt from template
4. Execute with recursive reasoning (refine/verify/critique strategies)
5. Return SkillExecutionResult with provenance

Integration Points:
------------------
- RecursiveWeavingOrchestrator - Multi-pass reasoning
- ReasoningStrategy - REFINE, VERIFY, CRITIQUE, etc.
- Quality thresholds - Auto-refinement when confidence < threshold
- Complete provenance - Full reasoning trace in metadata

Status: Production (v1.0.0)
Created: November 2025
"""

# Re-export from skill_agents.py (working implementation in parent directory)
from ..skill_agents import (
    # Core execution
    execute_skill,
    list_available_skills,
    get_registry,

    # Classes
    SkillRegistry,
    SkillExecutor,
    SkillTemplate,
    SkillExecutionResult,
    SkillMetadata,
    SkillReasoningConfig,
    SkillParameter,
)

__all__ = [
    # Convenience functions (recommended API)
    'execute_skill',
    'list_available_skills',
    'get_registry',

    # Classes (for advanced use)
    'SkillRegistry',
    'SkillExecutor',
    'SkillTemplate',
    'SkillExecutionResult',
    'SkillMetadata',
    'SkillReasoningConfig',
    'SkillParameter',
]

__version__ = "1.0.0"
__author__ = "HoloLoom (Integrated from Promptly)"
__status__ = "Production"
