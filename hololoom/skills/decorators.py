"""
Skill Decorators for HoloLoom
=============================

Simple decorators to create skills from functions.

**Created**: December 30, 2025
**Purpose**: Make skill creation as easy as decorating a function

Usage:
    from hololoom.skills import skill

    @skill(name="code_analyzer", category="code")
    async def analyze_code(code: str, language: str = "python") -> dict:
        # Your analysis logic
        issues = check_for_issues(code)
        return {"issues": issues, "score": 0.95}

    # That's it! The skill is automatically registered and can be launched:
    result = await launch_skill("code_analyzer", code="def foo(): pass")
"""

import asyncio
import inspect
import logging
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_type_hints,
)
from dataclasses import dataclass, field
from functools import wraps

from hololoom.skills.base import (
    BaseSkill,
    SkillInput,
    SkillOutput,
    SkillStatus,
    SkillMetadata,
    SkillCategory,
    register_skill,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Function-Based Skill Wrapper
# ============================================================================

class FunctionSkill(BaseSkill):
    """
    Wraps a function as a skill.

    Created by the @skill decorator.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        category: SkillCategory,
        version: str = "1.0.0",
        author: str = "hololoom",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        requires_network: bool = False,
        requires_filesystem_read: bool = False,
        requires_filesystem_write: bool = False,
        requires_code_execution: bool = False,
        expected_latency_ms: str = "50-500ms",
    ):
        """
        Initialize function-based skill.

        Args:
            func: The function to wrap
            name: Skill name
            category: Skill category
            version: Skill version
            author: Skill author
            description: Short description (uses docstring if not provided)
            tags: Skill tags
            requires_*: Capability flags
            expected_latency_ms: Expected execution time (string like "50-500ms")
        """
        self._func = func
        self._name = name
        self._is_async = asyncio.iscoroutinefunction(func)

        # Extract description from docstring if not provided
        if description is None:
            description = func.__doc__ or f"Skill: {name}"
            # Use first line of docstring
            description = description.strip().split('\n')[0]

        # Extract parameter info from function signature
        self._param_info = self._extract_params(func)

        # Build metadata
        self._metadata = SkillMetadata(
            name=name,
            version=version,
            author=author,
            category=category,
            description_short=description,
            description_long=func.__doc__ or description,
            tags=tags or [],
            requires_network=requires_network,
            requires_filesystem_read=requires_filesystem_read,
            requires_filesystem_write=requires_filesystem_write,
            requires_code_execution=requires_code_execution,
            expected_latency_ms=expected_latency_ms,
        )

        # Initialize lifecycle attributes (from BaseSkill)
        self._setup_done = False
        self._execution_count = 0
        self._total_time_ms = 0.0

    def _extract_params(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter info from function signature."""
        sig = inspect.signature(func)
        params = {}

        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        for param_name, param in sig.parameters.items():
            params[param_name] = {
                'name': param_name,
                'type': hints.get(param_name, Any),
                'default': param.default if param.default is not inspect.Parameter.empty else None,
                'required': param.default is inspect.Parameter.empty,
            }

        return params

    def get_metadata(self) -> SkillMetadata:
        """Return skill metadata."""
        return self._metadata

    def validate_input(self, skill_input: SkillInput) -> bool:
        """Validate input parameters."""
        # Check required parameters
        for param_name, param_info in self._param_info.items():
            if param_info['required'] and param_name not in skill_input.parameters:
                return False
        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        """Execute the wrapped function."""
        start_time = time.time()

        try:
            # Extract parameters
            params = skill_input.parameters.copy()

            # Call the function
            if self._is_async:
                result = await self._func(**params)
            else:
                # Run sync function in executor to not block
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self._func(**params))

            execution_time = (time.time() - start_time) * 1000

            # Wrap result in SkillOutput
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Skill '{self._name}' executed successfully",
                execution_time_ms=execution_time,
                skill_name=self._name,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Skill '{self._name}' failed: {e}")

            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Skill '{self._name}' failed: {e}",
                execution_time_ms=execution_time,
                errors=[str(e)],
                skill_name=self._name,
            )


# ============================================================================
# The @skill Decorator
# ============================================================================

def skill(
    name: Optional[str] = None,
    category: Union[str, SkillCategory] = "domain",
    version: str = "1.0.0",
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    requires_network: bool = False,
    requires_filesystem_read: bool = False,
    requires_filesystem_write: bool = False,
    requires_code_execution: bool = False,
    expected_latency_ms: float = 100.0,
    auto_register: bool = True,
) -> Callable:
    """
    Decorator to create a skill from a function.

    This is the easiest way to create a HoloLoom skill. Just decorate
    a function and it becomes a launchable skill.

    Args:
        name: Skill name (defaults to function name)
        category: Skill category (domain, code, testing, etc.)
        version: Skill version
        description: Short description (defaults to docstring)
        tags: Tags for discovery
        requires_network: Whether skill needs network access
        requires_filesystem_read: Whether skill reads files
        requires_filesystem_write: Whether skill writes files
        requires_code_execution: Whether skill runs code
        expected_latency_ms: Expected execution time
        auto_register: Automatically register with skill registry

    Returns:
        Decorated function that's also a registered skill

    Examples:
        # Basic usage
        @skill(name="hello")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        # Async with options
        @skill(name="analyzer", category="code", tags=["python", "analysis"])
        async def analyze(code: str, language: str = "python") -> dict:
            '''Analyze code for issues.'''
            issues = find_issues(code)
            return {"issues": issues, "score": 0.95}

        # Then launch with:
        result = await launch_skill("analyzer", code="def foo(): pass")
    """
    # Handle being called with or without arguments
    def decorator(func: Callable) -> Callable:
        # Use function name if skill name not provided
        skill_name = name or func.__name__

        # Convert category string to enum
        if isinstance(category, str):
            try:
                skill_category = SkillCategory(category)
            except ValueError:
                skill_category = SkillCategory.DOMAIN
        else:
            skill_category = category

        # Create the skill wrapper
        skill_instance = FunctionSkill(
            func=func,
            name=skill_name,
            category=skill_category,
            version=version,
            description=description,
            tags=tags,
            requires_network=requires_network,
            requires_filesystem_read=requires_filesystem_read,
            requires_filesystem_write=requires_filesystem_write,
            requires_code_execution=requires_code_execution,
            expected_latency_ms=expected_latency_ms,
        )

        # Register with global registry
        if auto_register:
            register_skill(skill_instance)
            logger.info(f"Registered skill: {skill_name}")

        # Create wrapper that preserves original function behavior
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Attach skill instance for direct access
        wrapper._skill = skill_instance
        wrapper._skill_name = skill_name

        return wrapper

    return decorator


# ============================================================================
# Convenience Decorators for Common Categories
# ============================================================================

def code_skill(
    name: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Create a code analysis/testing skill.

    Example:
        @code_skill(name="linter")
        async def lint(code: str) -> dict:
            return {"issues": []}
    """
    return skill(name=name, category=SkillCategory.CODE, **kwargs)


def domain_skill(
    name: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Create a domain-specific skill.

    Example:
        @domain_skill(name="sentiment")
        async def analyze_sentiment(text: str) -> dict:
            return {"sentiment": "positive", "score": 0.8}
    """
    return skill(name=name, category=SkillCategory.DOMAIN, **kwargs)


def testing_skill(
    name: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Create a testing/validation skill.

    Example:
        @testing_skill(name="unit_tests")
        async def run_tests(path: str) -> dict:
            return {"passed": 10, "failed": 0}
    """
    return skill(name=name, category=SkillCategory.TESTING, **kwargs)


def web_skill(
    name: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Create a web interaction skill.

    Example:
        @web_skill(name="scraper", requires_network=True)
        async def scrape(url: str) -> dict:
            return {"content": "..."}
    """
    kwargs.setdefault('requires_network', True)
    return skill(name=name, category=SkillCategory.WEB, **kwargs)


def system_skill(
    name: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Create a system utility skill.

    Example:
        @system_skill(name="disk_usage")
        async def check_disk() -> dict:
            return {"used_percent": 45.2}
    """
    return skill(name=name, category=SkillCategory.SYSTEM, **kwargs)


# ============================================================================
# Skill Builder (Alternative API)
# ============================================================================

@dataclass
class SkillBuilder:
    """
    Fluent builder for creating skills.

    Alternative to decorator when more control is needed.

    Example:
        skill = (SkillBuilder()
            .name("my_skill")
            .category("domain")
            .handler(my_function)
            .tags(["tag1", "tag2"])
            .build())
    """
    _name: Optional[str] = None
    _category: SkillCategory = SkillCategory.DOMAIN
    _version: str = "1.0.0"
    _description: Optional[str] = None
    _tags: List[str] = field(default_factory=list)
    _handler: Optional[Callable] = None
    _requires_network: bool = False
    _requires_filesystem_read: bool = False
    _requires_filesystem_write: bool = False
    _requires_code_execution: bool = False
    _expected_latency_ms: float = 100.0

    def name(self, name: str) -> "SkillBuilder":
        """Set skill name."""
        self._name = name
        return self

    def category(self, category: Union[str, SkillCategory]) -> "SkillBuilder":
        """Set skill category."""
        if isinstance(category, str):
            self._category = SkillCategory(category)
        else:
            self._category = category
        return self

    def version(self, version: str) -> "SkillBuilder":
        """Set skill version."""
        self._version = version
        return self

    def description(self, description: str) -> "SkillBuilder":
        """Set skill description."""
        self._description = description
        return self

    def tags(self, tags: List[str]) -> "SkillBuilder":
        """Set skill tags."""
        self._tags = tags
        return self

    def handler(self, func: Callable) -> "SkillBuilder":
        """Set the handler function."""
        self._handler = func
        return self

    def requires_network(self, value: bool = True) -> "SkillBuilder":
        """Mark as requiring network access."""
        self._requires_network = value
        return self

    def requires_filesystem(self, read: bool = False, write: bool = False) -> "SkillBuilder":
        """Mark filesystem requirements."""
        self._requires_filesystem_read = read
        self._requires_filesystem_write = write
        return self

    def requires_code_execution(self, value: bool = True) -> "SkillBuilder":
        """Mark as requiring code execution."""
        self._requires_code_execution = value
        return self

    def expected_latency(self, ms: float) -> "SkillBuilder":
        """Set expected latency in milliseconds."""
        self._expected_latency_ms = ms
        return self

    def build(self, auto_register: bool = True) -> FunctionSkill:
        """
        Build and optionally register the skill.

        Returns:
            The created FunctionSkill instance
        """
        if self._handler is None:
            raise ValueError("Handler function must be set")

        if self._name is None:
            self._name = self._handler.__name__

        skill_instance = FunctionSkill(
            func=self._handler,
            name=self._name,
            category=self._category,
            version=self._version,
            description=self._description,
            tags=self._tags,
            requires_network=self._requires_network,
            requires_filesystem_read=self._requires_filesystem_read,
            requires_filesystem_write=self._requires_filesystem_write,
            requires_code_execution=self._requires_code_execution,
            expected_latency_ms=self._expected_latency_ms,
        )

        if auto_register:
            register_skill(skill_instance)

        return skill_instance


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main decorator
    "skill",

    # Category-specific decorators
    "code_skill",
    "domain_skill",
    "testing_skill",
    "web_skill",
    "system_skill",

    # Classes
    "FunctionSkill",
    "SkillBuilder",
]
