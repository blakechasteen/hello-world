"""
Skills Package - Unified Skills Meta-System
============================================

**Created**: November 22, 2025
**Purpose**: Unified interface for HoloLoom skills ecosystem

This package unifies three skill systems:
1. **Meta-Skills** (skills/meta/) - Self-improving skills about skills
2. **Domain Skills** (skills/domain/) - Task-focused domain expertise
3. **HoloLoom Agentic Skills** (HoloLoom/agentic/) - Production YAML skills

Integration with Zero-G:
- Skills are "apps" in the App Orbit Layer
- Event bus for cross-skill communication
- Shared WarpSpace and YarnGraph semantic layer

## Quick Start

```python
from skills import SkillRegistry, load_all_skills, execute_skill

# Load all skills (meta + domain + agentic)
registry = SkillRegistry()
await registry.load_all_skills()

# List available skills
skills = registry.list_skills(category="meta")

# Execute a skill
result = await execute_skill(
    "continuous_learning_capture",
    parameters={"mode": "pattern_detection"}
)
```

## Architecture

```
Skills Package
├── Meta-Skills (skills/meta/)
│   ├── continuous_learning_capture - Learn from interactions
│   ├── skill_gap_analyzer - Identify missing capabilities
│   ├── skill_tester - Validate functionality
│   ├── skill_security_analyzer - Security checks
│   └── token_budget_adviser - Token optimization
│
├── Domain Skills (skills/domain/)
│   ├── programming/ - Language-specific skills
│   ├── frameworks/ - Framework expertise
│   └── infrastructure/ - DevOps and infrastructure
│
└── Integration
    ├── HoloLoom Agentic Skills (13 YAML skills)
    ├── Zero-G App Protocol (docking interface)
    └── Event Bus (cross-skill communication)
```
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

__version__ = '1.0.0'
__author__ = 'HoloLoom Team'

logger = logging.getLogger(__name__)

# Package root directory
SKILLS_ROOT = Path(__file__).parent

# Skill type directories
META_SKILLS_DIR = SKILLS_ROOT / "meta"
DOMAIN_SKILLS_DIR = SKILLS_ROOT / "domain"
TEMPLATES_DIR = SKILLS_ROOT / "templates"

# Check for HoloLoom integration
try:
    from hololoom.agentic.skill_agents import (
        SkillRegistry as HoloLoomSkillRegistry,
        execute_skill as hololoom_execute_skill,
        SkillTemplate,
        SkillMetadata
    )
    HOLOLOOM_AVAILABLE = True
except ImportError:
    logger.warning("HoloLoom skill system not available")
    HOLOLOOM_AVAILABLE = False
    HoloLoomSkillRegistry = None
    hololoom_execute_skill = None


# ============================================================================
# Skill Types
# ============================================================================

@dataclass
class MetaSkill:
    """Meta-skill definition (markdown-based)."""
    name: str
    path: Path
    category: str = "meta"
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class DomainSkill:
    """Domain skill definition (markdown or YAML)."""
    name: str
    path: Path
    category: str
    domain: str
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# ============================================================================
# Unified Skill Registry
# ============================================================================

class SkillRegistry:
    """
    Unified registry for all skill types.

    Integrates:
    - Meta-skills (self-improvement)
    - Domain skills (task-focused)
    - HoloLoom agentic skills (YAML-based)
    """

    def __init__(self):
        self.meta_skills: Dict[str, MetaSkill] = {}
        self.domain_skills: Dict[str, DomainSkill] = {}
        self.hololoom_registry = None

        if HOLOLOOM_AVAILABLE:
            self.hololoom_registry = HoloLoomSkillRegistry()

    async def load_all_skills(self):
        """Load all skills from all sources."""
        await self.load_meta_skills()
        await self.load_domain_skills()

        if HOLOLOOM_AVAILABLE and self.hololoom_registry:
            await self.hololoom_registry.load_all_skills()

    async def load_meta_skills(self):
        """Load meta-skills from skills/meta/ directory."""
        if not META_SKILLS_DIR.exists():
            logger.warning(f"Meta-skills directory not found: {META_SKILLS_DIR}")
            return

        for skill_dir in META_SKILLS_DIR.iterdir():
            if not skill_dir.is_dir():
                continue

            # Look for skill.markdown file
            skill_file = skill_dir / "skill.markdown"
            if not skill_file.exists():
                # Try skill.md
                skill_file = skill_dir / "skill.md"
                if not skill_file.exists():
                    logger.warning(f"No skill definition found in {skill_dir}")
                    continue

            # Create meta-skill object
            meta_skill = MetaSkill(
                name=skill_dir.name,
                path=skill_file,
                category="meta"
            )

            self.meta_skills[skill_dir.name] = meta_skill
            logger.info(f"Loaded meta-skill: {skill_dir.name}")

    async def load_domain_skills(self):
        """Load domain skills from skills/domain/ directory."""
        if not DOMAIN_SKILLS_DIR.exists():
            logger.warning(f"Domain skills directory not found: {DOMAIN_SKILLS_DIR}")
            return

        # Domain skills are organized by domain (programming, frameworks, etc.)
        for domain_dir in DOMAIN_SKILLS_DIR.iterdir():
            if not domain_dir.is_dir():
                continue

            domain = domain_dir.name

            for skill_dir in domain_dir.iterdir():
                if not skill_dir.is_dir():
                    continue

                # Look for skill definition file
                skill_file = None
                for ext in ["skill.markdown", "skill.md", "skill.yaml", "skill.yml"]:
                    candidate = skill_dir / ext
                    if candidate.exists():
                        skill_file = candidate
                        break

                if not skill_file:
                    logger.warning(f"No skill definition found in {skill_dir}")
                    continue

                # Create domain skill object
                domain_skill = DomainSkill(
                    name=skill_dir.name,
                    path=skill_file,
                    category="domain",
                    domain=domain
                )

                self.domain_skills[f"{domain}/{skill_dir.name}"] = domain_skill
                logger.info(f"Loaded domain skill: {domain}/{skill_dir.name}")

    def list_skills(
        self,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List all skills matching filters.

        Args:
            category: Filter by category ("meta", "domain", "agentic")
            domain: Filter by domain (for domain skills)
            tags: Filter by tags

        Returns:
            List of skill summaries
        """
        results = []

        # Meta-skills
        if category is None or category == "meta":
            for name, skill in self.meta_skills.items():
                results.append({
                    "name": name,
                    "category": "meta",
                    "version": skill.version,
                    "description": skill.description,
                    "tags": skill.tags,
                    "source": "skills/meta"
                })

        # Domain skills
        if category is None or category == "domain":
            for name, skill in self.domain_skills.items():
                if domain and skill.domain != domain:
                    continue

                results.append({
                    "name": name,
                    "category": "domain",
                    "domain": skill.domain,
                    "version": skill.version,
                    "description": skill.description,
                    "tags": skill.tags,
                    "source": "skills/domain"
                })

        # HoloLoom agentic skills
        if (category is None or category == "agentic") and self.hololoom_registry:
            for name, skill in self.hololoom_registry.skills.items():
                results.append({
                    "name": name,
                    "category": "agentic",
                    "version": skill.version,
                    "description": skill.description,
                    "tags": skill.metadata.tags if hasattr(skill.metadata, 'tags') else [],
                    "source": "HoloLoom/agentic"
                })

        return results

    def get_skill(self, name: str) -> Optional[Any]:
        """Get a skill by name."""
        # Try meta-skills
        if name in self.meta_skills:
            return self.meta_skills[name]

        # Try domain skills
        if name in self.domain_skills:
            return self.domain_skills[name]

        # Try HoloLoom agentic skills
        if self.hololoom_registry and name in self.hololoom_registry.skills:
            return self.hololoom_registry.skills[name]

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_skills": len(self.meta_skills) + len(self.domain_skills) +
                          (len(self.hololoom_registry.skills) if self.hololoom_registry else 0),
            "meta_skills": len(self.meta_skills),
            "domain_skills": len(self.domain_skills),
            "agentic_skills": len(self.hololoom_registry.skills) if self.hololoom_registry else 0,
            "hololoom_available": HOLOLOOM_AVAILABLE
        }


# ============================================================================
# Convenience Functions
# ============================================================================

async def load_all_skills() -> SkillRegistry:
    """
    Load all skills and return registry.

    Returns:
        SkillRegistry with all skills loaded
    """
    registry = SkillRegistry()
    await registry.load_all_skills()
    return registry


async def execute_skill(
    skill_name: str,
    parameters: Dict[str, Any],
    config: Optional[Any] = None
) -> Any:
    """
    Execute a skill by name.

    Args:
        skill_name: Name of the skill to execute
        parameters: Skill parameters
        config: Optional HoloLoom config

    Returns:
        Skill execution result
    """
    # For now, delegate to HoloLoom if available
    if HOLOLOOM_AVAILABLE and hololoom_execute_skill:
        return await hololoom_execute_skill(skill_name, parameters, config)
    else:
        raise NotImplementedError(
            f"Skill execution not available. "
            f"Install HoloLoom for skill execution support."
        )


# ============================================================================
# Package Exports
# ============================================================================

__all__ = [
    # Core classes
    'SkillRegistry',
    'MetaSkill',
    'DomainSkill',

    # Convenience functions
    'load_all_skills',
    'execute_skill',

    # Constants
    'SKILLS_ROOT',
    'META_SKILLS_DIR',
    'DOMAIN_SKILLS_DIR',
    'HOLOLOOM_AVAILABLE',
]
