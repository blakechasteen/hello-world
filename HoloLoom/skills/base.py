"""
Skill System Framework
======================
Base classes and protocols for HoloLoom's extensible skill system.

Philosophy:
- Skills are WORKFLOWS, not atomic operations
- Each skill wraps complex multi-step processes
- Consistent interface across all skills
- Concurrent execution when possible
- Rich metadata and provenance tracking

Created: November 24, 2025
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Skill categories for organization."""
    DOMAIN = "domain"           # Domain-specific (MRF, prompt tools)
    INFRASTRUCTURE = "infrastructure"  # DevOps, deployment
    TESTING = "testing"         # Test execution, validation
    CODE = "code"              # Code intelligence, analysis
    COMMUNICATION = "communication"  # Slack, email, etc.
    DATA = "data"              # Data processing, ML
    WEB = "web"                # Browser automation, APIs
    CREATIVE = "creative"      # Image, video, audio generation
    SYSTEM = "system"          # Monitoring, profiling


class SkillStatus(Enum):
    """Skill execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SkillInput:
    """Standardized skill input."""
    operation: str              # What to do
    parameters: Dict[str, Any]  # Operation-specific params
    context: Optional[Dict[str, Any]] = None  # Additional context
    timeout_seconds: Optional[float] = None   # Execution timeout
    metadata: Optional[Dict[str, Any]] = None  # User metadata


@dataclass
class SkillOutput:
    """Standardized skill output with rich metadata."""
    status: SkillStatus
    result: Any                 # Operation result
    message: str               # Human-readable summary

    # Execution metadata
    execution_time_ms: float
    timestamp: float = field(default_factory=time.time)

    # Detailed information
    details: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Provenance
    skill_name: Optional[str] = None
    skill_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'result': self.result,
            'message': self.message,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp,
            'details': self.details,
            'warnings': self.warnings,
            'errors': self.errors,
            'skill_name': self.skill_name,
            'skill_version': self.skill_version
        }


@dataclass
class SkillMetadata:
    """Skill metadata from skill.markdown."""
    name: str
    version: str
    author: str
    category: SkillCategory
    tags: List[str]
    description_short: str
    description_long: str

    # Capabilities
    requires_filesystem_read: bool = False
    requires_filesystem_write: bool = False
    requires_code_execution: bool = False
    requires_network: bool = False
    requires_mcp: bool = False
    requires_user_input: bool = False

    # Dependencies
    external_dependencies: List[str] = field(default_factory=list)
    skill_dependencies: List[str] = field(default_factory=list)

    # Performance
    expected_latency_ms: Optional[str] = None  # "50-500ms"
    token_usage: Optional[str] = None          # "600-2500 tokens"


class BaseSkill(ABC):
    """
    Abstract base class for all skills.

    All skills must implement:
    - execute() - Main entry point
    - validate_input() - Input validation
    - get_metadata() - Return skill metadata

    Optional overrides:
    - setup() - One-time initialization
    - teardown() - Cleanup
    - estimate_cost() - Cost estimation
    """

    def __init__(self):
        """Initialize skill."""
        self._metadata = self.get_metadata()
        self._setup_done = False
        self._execution_count = 0
        self._total_time_ms = 0.0

    @abstractmethod
    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        """
        Execute the skill with given input.

        Args:
            skill_input: Standardized skill input

        Returns:
            SkillOutput with results and metadata
        """
        pass

    @abstractmethod
    def validate_input(self, skill_input: SkillInput) -> bool:
        """
        Validate input before execution.

        Args:
            skill_input: Input to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        pass

    @abstractmethod
    def get_metadata(self) -> SkillMetadata:
        """
        Return skill metadata.

        Returns:
            SkillMetadata instance
        """
        pass

    async def setup(self):
        """
        One-time setup (optional override).

        Called before first execution. Use for:
        - Loading models/data
        - Establishing connections
        - Initializing resources
        """
        self._setup_done = True

    async def teardown(self):
        """
        Cleanup (optional override).

        Called when skill is no longer needed. Use for:
        - Closing connections
        - Releasing resources
        - Saving state
        """
        pass

    def estimate_cost(self, skill_input: SkillInput) -> Dict[str, Any]:
        """
        Estimate execution cost (optional override).

        Returns:
            Dict with cost estimates (time, tokens, money, etc.)
        """
        return {
            "estimated_time_ms": self._metadata.expected_latency_ms,
            "estimated_tokens": self._metadata.token_usage
        }

    async def execute_with_lifecycle(self, skill_input: SkillInput) -> SkillOutput:
        """
        Execute skill with full lifecycle management.

        Handles:
        - Setup (if needed)
        - Input validation
        - Timeout enforcement
        - Error handling
        - Metrics tracking
        - Teardown (on error)

        Args:
            skill_input: Skill input

        Returns:
            SkillOutput
        """
        # Setup if needed
        if not self._setup_done:
            await self.setup()

        # Validate input
        try:
            self.validate_input(skill_input)
        except ValueError as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Input validation failed: {e}",
                execution_time_ms=0.0,
                errors=[str(e)],
                skill_name=self._metadata.name,
                skill_version=self._metadata.version
            )

        # Execute with timeout and error handling
        start_time = time.time()

        try:
            output = await self.execute(skill_input)

            # Update metrics
            self._execution_count += 1
            self._total_time_ms += output.execution_time_ms

            # Add metadata
            output.skill_name = self._metadata.name
            output.skill_version = self._metadata.version

            return output

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Skill {self._metadata.name} failed: {e}")

            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Skill execution failed: {e}",
                execution_time_ms=execution_time,
                errors=[str(e)],
                skill_name=self._metadata.name,
                skill_version=self._metadata.version
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get skill execution statistics.

        Returns:
            Dict with execution metrics
        """
        avg_time = self._total_time_ms / self._execution_count if self._execution_count > 0 else 0.0

        return {
            "skill_name": self._metadata.name,
            "execution_count": self._execution_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": avg_time
        }


class SkillRegistry:
    """
    Central registry for all skills.

    Features:
    - Skill registration and lookup
    - Category-based filtering
    - Dependency resolution
    - Concurrent execution support
    """

    def __init__(self):
        """Initialize registry."""
        self._skills: Dict[str, BaseSkill] = {}
        self._categories: Dict[SkillCategory, List[str]] = {}

    def register(self, skill: BaseSkill):
        """
        Register a skill.

        Args:
            skill: Skill instance to register
        """
        metadata = skill.get_metadata()
        self._skills[metadata.name] = skill

        # Add to category index
        if metadata.category not in self._categories:
            self._categories[metadata.category] = []
        self._categories[metadata.category].append(metadata.name)

        logger.info(f"Registered skill: {metadata.name} v{metadata.version}")

    def get(self, name: str) -> Optional[BaseSkill]:
        """
        Get skill by name.

        Args:
            name: Skill name

        Returns:
            Skill instance or None
        """
        return self._skills.get(name)

    def list_by_category(self, category: SkillCategory) -> List[str]:
        """
        List skills in category.

        Args:
            category: Category to filter by

        Returns:
            List of skill names
        """
        return self._categories.get(category, [])

    def list_all(self) -> List[str]:
        """
        List all registered skills.

        Returns:
            List of skill names
        """
        return list(self._skills.keys())

    def get_metadata_all(self) -> Dict[str, SkillMetadata]:
        """
        Get metadata for all skills.

        Returns:
            Dict of skill_name -> SkillMetadata
        """
        return {
            name: skill.get_metadata()
            for name, skill in self._skills.items()
        }


# Global registry instance
_global_registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    """Get global skill registry."""
    return _global_registry


def register_skill(skill: BaseSkill):
    """Register skill in global registry."""
    _global_registry.register(skill)


__all__ = [
    "BaseSkill",
    "SkillInput",
    "SkillOutput",
    "SkillMetadata",
    "SkillCategory",
    "SkillStatus",
    "SkillRegistry",
    "get_registry",
    "register_skill"
]
