"""
HoloLoom Skill Loader
=====================
Load and manage skill templates for agentic reasoning.

Features:
- Load skills from templates
- Create custom skills
- Skill validation and metadata management
- Integration with agentic reasoning modes
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from HoloLoom.agentic.skills.templates import get_all_templates, get_template


logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class SkillFile:
    """
    A file within a skill template.

    Attributes:
        filename: Name of the file (e.g., 'review_checklist.md')
        filetype: File extension (md, py, yaml, etc.)
        content: File content as string
    """
    filename: str
    filetype: str
    content: str

    @property
    def is_markdown(self) -> bool:
        """Check if file is markdown."""
        return self.filetype.lower() in ['md', 'markdown']

    @property
    def is_python(self) -> bool:
        """Check if file is Python code."""
        return self.filetype.lower() == 'py'

    @property
    def is_yaml(self) -> bool:
        """Check if file is YAML config."""
        return self.filetype.lower() in ['yaml', 'yml']


@dataclass
class SkillMetadata:
    """
    Metadata for a skill.

    Attributes:
        runtime: Runtime environment (claude, ollama, local)
        tags: List of tags for categorization
        model: LLM model to use (if runtime is LLM)
        requires: List of required Python packages
        version: Skill version
        author: Skill author
        created_at: Creation timestamp
    """
    runtime: str = "claude"
    tags: List[str] = field(default_factory=list)
    model: Optional[str] = None
    requires: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "runtime": self.runtime,
            "tags": self.tags,
            "model": self.model,
            "requires": self.requires,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at
        }


@dataclass
class Skill:
    """
    A complete skill with metadata and files.

    Attributes:
        name: Skill name (e.g., 'code_reviewer')
        description: Skill description
        metadata: Skill metadata
        files: Dict of filename -> SkillFile
    """
    name: str
    description: str
    metadata: SkillMetadata
    files: Dict[str, SkillFile] = field(default_factory=dict)

    def get_file(self, filename: str) -> Optional[SkillFile]:
        """Get a specific file from the skill."""
        return self.files.get(filename)

    def get_markdown_files(self) -> List[SkillFile]:
        """Get all markdown documentation files."""
        return [f for f in self.files.values() if f.is_markdown]

    def get_python_files(self) -> List[SkillFile]:
        """Get all Python code files."""
        return [f for f in self.files.values() if f.is_python]

    def get_config_files(self) -> List[SkillFile]:
        """Get all YAML config files."""
        return [f for f in self.files.values() if f.is_yaml]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata.to_dict(),
            "files": {
                filename: {
                    "filename": skill_file.filename,
                    "filetype": skill_file.filetype,
                    "content": skill_file.content
                }
                for filename, skill_file in self.files.items()
            }
        }


# ============================================================================
# Skill Loader
# ============================================================================

class SkillLoader:
    """
    Load and manage skills for HoloLoom agentic reasoning.

    Usage:
        loader = SkillLoader()
        skill = loader.load('code_reviewer')
        custom_skill = loader.create_custom_skill(name, description, files)
    """

    def __init__(self):
        """Initialize skill loader."""
        self._templates = get_all_templates()
        logger.info(f"SkillLoader initialized with {len(self._templates)} templates")

    def load(self, skill_name: str) -> Skill:
        """
        Load a skill from templates.

        Args:
            skill_name: Name of the skill template

        Returns:
            Skill instance

        Raises:
            ValueError: If skill not found
        """
        template = self._templates.get(skill_name)
        if not template:
            available = list(self._templates.keys())
            raise ValueError(
                f"Skill '{skill_name}' not found. "
                f"Available skills: {', '.join(available)}"
            )

        return self._load_from_template(skill_name, template)

    def _load_from_template(self, skill_name: str, template: Dict[str, Any]) -> Skill:
        """Load skill from template dict."""
        # Extract metadata
        template_metadata = template.get("metadata", {})
        metadata = SkillMetadata(
            runtime=template_metadata.get("runtime", "claude"),
            tags=template_metadata.get("tags", []),
            model=template_metadata.get("model"),
            requires=template_metadata.get("requires", [])
        )

        # Extract files
        files = {}
        for filename, content in template.get("files", {}).items():
            # Determine file type from extension
            filetype = Path(filename).suffix.lstrip('.')
            if not filetype:
                filetype = "txt"

            files[filename] = SkillFile(
                filename=filename,
                filetype=filetype,
                content=content
            )

        # Create skill
        skill = Skill(
            name=skill_name,
            description=template.get("description", ""),
            metadata=metadata,
            files=files
        )

        logger.debug(f"Loaded skill '{skill_name}' with {len(files)} files")
        return skill

    def create_custom_skill(
        self,
        name: str,
        description: str,
        files: Dict[str, str],
        metadata: Optional[SkillMetadata] = None
    ) -> Skill:
        """
        Create a custom skill.

        Args:
            name: Skill name
            description: Skill description
            files: Dict of filename -> content
            metadata: Optional metadata (uses defaults if not provided)

        Returns:
            Skill instance
        """
        if metadata is None:
            metadata = SkillMetadata()

        # Create skill files
        skill_files = {}
        for filename, content in files.items():
            filetype = Path(filename).suffix.lstrip('.')
            if not filetype:
                filetype = "txt"

            skill_files[filename] = SkillFile(
                filename=filename,
                filetype=filetype,
                content=content
            )

        skill = Skill(
            name=name,
            description=description,
            metadata=metadata,
            files=skill_files
        )

        logger.info(f"Created custom skill '{name}' with {len(files)} files")
        return skill

    def list_available(self) -> List[str]:
        """
        List all available skill templates.

        Returns:
            List of skill names
        """
        return list(self._templates.keys())

    def get_skill_info(self, skill_name: str) -> Dict[str, Any]:
        """
        Get information about a skill without loading it.

        Args:
            skill_name: Name of the skill

        Returns:
            Dict with name, description, tags

        Raises:
            ValueError: If skill not found
        """
        template = self._templates.get(skill_name)
        if not template:
            raise ValueError(f"Skill '{skill_name}' not found")

        return {
            "name": skill_name,
            "description": template.get("description", ""),
            "tags": template.get("metadata", {}).get("tags", []),
            "runtime": template.get("metadata", {}).get("runtime", "unknown"),
            "file_count": len(template.get("files", {}))
        }

    def search_by_tag(self, tag: str) -> List[str]:
        """
        Search for skills by tag.

        Args:
            tag: Tag to search for (e.g., 'code', 'security')

        Returns:
            List of skill names matching the tag
        """
        matching = []
        for skill_name, template in self._templates.items():
            tags = template.get("metadata", {}).get("tags", [])
            if tag.lower() in [t.lower() for t in tags]:
                matching.append(skill_name)

        return matching

    def validate_skill(self, skill: Skill) -> List[str]:
        """
        Validate a skill and return any issues.

        Args:
            skill: Skill to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required fields
        if not skill.name:
            errors.append("Skill name is required")

        if not skill.description:
            errors.append("Skill description is required")

        if not skill.files:
            errors.append("Skill must have at least one file")

        # Check metadata
        if skill.metadata.runtime not in ['claude', 'ollama', 'local', 'python']:
            errors.append(f"Invalid runtime: {skill.metadata.runtime}")

        # Check for required Python packages (basic validation)
        for package in skill.metadata.requires:
            if not isinstance(package, str):
                errors.append(f"Invalid package requirement: {package}")

        return errors


# ============================================================================
# Convenience Functions
# ============================================================================

def load_skill(skill_name: str) -> Skill:
    """
    Quick load a skill.

    Args:
        skill_name: Name of the skill template

    Returns:
        Skill instance
    """
    loader = SkillLoader()
    return loader.load(skill_name)


def list_skills() -> List[str]:
    """
    Quick list all available skills.

    Returns:
        List of skill names
    """
    loader = SkillLoader()
    return loader.list_available()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("HoloLoom Skill Loader\n")

    # List available skills
    loader = SkillLoader()
    skills = loader.list_available()
    print(f"Available skills ({len(skills)}):")
    for skill_name in sorted(skills):
        info = loader.get_skill_info(skill_name)
        print(f"  - {skill_name}: {info['description']}")
        print(f"    Tags: {', '.join(info['tags'])}")

    # Load a skill
    print("\nLoading 'code_reviewer' skill...")
    skill = loader.load('code_reviewer')
    print(f"  Name: {skill.name}")
    print(f"  Description: {skill.description}")
    print(f"  Files: {len(skill.files)}")
    print(f"  Runtime: {skill.metadata.runtime}")

    # Search by tag
    print("\nSkills tagged with 'security':")
    security_skills = loader.search_by_tag('security')
    for name in security_skills:
        print(f"  - {name}")