"""Proto Ability Protocol.

Three-tier extensibility protocol for Proto abilities.

Tier 1: Skill Mapping
    Thin wrappers around HoloLoom's 13 built-in skills.
    - Direct skill invocation with unified interface
    - Minimal overhead
    - Full trust (core abilities)

Tier 2: Plugin Protocol
    Typed interface with manifest and permissions.
    - Structured execution model with lifecycle
    - Permission checking and trust levels
    - Can be loaded from external packages
    - User-defined or verified marketplace

Tier 3: Full Sandbox (Future - Phase 4)
    Container/process isolation for untrusted code.
    - Docker container execution
    - Complete resource isolation
    - Network, file system restrictions

All abilities implement the same Protocol for consistency.

Status: Production ready (2025-12-02)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class AbilityTrustLevel(Enum):
    """Trust levels for abilities.

    Determines what access/isolation level an ability gets.
    """
    CORE = "core"                  # Built-in, full access, no isolation
    VERIFIED = "verified"           # Marketplace verified, process isolation
    COMMUNITY = "community"         # Community contributed, container isolation
    LOCAL = "local"                # User-defined, process isolation
    UNTRUSTED = "untrusted"         # Unknown, maximum isolation


class AbilityTier(Enum):
    """Ability implementation tier."""
    SKILL_MAPPING = 1   # Tier 1: HoloLoom skill wrapper
    PLUGIN = 2          # Tier 2: Protocol with manifest
    SANDBOX = 3         # Tier 3: Full sandboxed execution


@dataclass
class AbilityManifest:
    """Manifest describing an ability's capabilities and requirements.

    Provides metadata about what an ability can do, what it needs,
    and what trust/permissions it requires.

    Attributes:
        name: Unique ability identifier (e.g., "code_analysis")
        version: Semantic version string (e.g., "1.0.0")
        description: Human-readable description
        author: Author name or organization

        tier: Implementation tier (SKILL_MAPPING, PLUGIN, SANDBOX)
        trust_level: Required trust level (CORE, VERIFIED, etc.)

        permissions: List of required permissions (read_file, write_file, execute, network, etc.)
        requires_confirmation: Require user confirmation before running

        requires: List of dependencies (Python packages, system tools)

        tags: Categorical tags for discovery
        homepage: URL to ability documentation/homepage
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""

    # Tier and trust
    tier: AbilityTier = AbilityTier.PLUGIN
    trust_level: AbilityTrustLevel = AbilityTrustLevel.LOCAL

    # Permissions
    permissions: list[str] = field(default_factory=list)
    requires_confirmation: bool = False

    # Dependencies
    requires: list[str] = field(default_factory=list)

    # Metadata
    tags: list[str] = field(default_factory=list)
    homepage: str | None = None

    def requires_permission(self, permission: str) -> bool:
        """Check if ability requires a specific permission."""
        return permission in self.permissions

    def has_tag(self, tag: str) -> bool:
        """Check if ability has a tag."""
        return tag in self.tags


@dataclass
class AbilityContext:
    """Context passed to ability execution.

    Contains execution context, permissions, and metadata.

    Attributes:
        session_id: Session identifier for tracking
        working_directory: Current working directory for file operations
        user_confirmed: Whether user has confirmed execution of risky abilities
        trust_level: Current trust level for this execution
        timeout_seconds: Maximum execution time
        metadata: Additional execution metadata
    """
    session_id: str = ""
    working_directory: str = ""
    user_confirmed: bool = False
    trust_level: AbilityTrustLevel = AbilityTrustLevel.LOCAL
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    """Result of preflight check.

    Indicates whether an ability can execute in the current context.

    Attributes:
        can_execute: Whether ability can execute
        reason: Reason if cannot execute
        warnings: List of warnings (e.g., permission requirements)
        estimated_duration_ms: Estimated execution time
    """
    can_execute: bool = True
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    estimated_duration_ms: float = 0.0


@dataclass
class AbilityResult:
    """Result from ability execution.

    Contains output, status, and execution metadata.

    Attributes:
        success: Whether execution succeeded
        output: The result data (can be any type)
        error: Error message if failed
        confidence: Confidence score 0.0-1.0 (credibility of result)
        duration_ms: Actual execution time
        metadata: Additional result metadata
    """
    success: bool = True
    output: Any = None
    error: str | None = None
    confidence: float = 0.8
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of output verification.

    Indicates whether the ability output is valid/trustworthy.

    Attributes:
        verified: Whether output passed verification
        issues: List of detected issues
        suggestions: List of suggestions for improvement
    """
    verified: bool = True
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@runtime_checkable
class Ability(Protocol):
    """Protocol for sideloadable abilities.

    All abilities (Tier 1-3) must implement this interface for consistency.

    Lifecycle:
    1. preflight(context) - Check if can execute
    2. execute(params, context) - Run the ability
    3. verify(result) - Validate the output

    Example:
        ability = MyAbility()
        context = AbilityContext(session_id="123", trust_level=AbilityTrustLevel.LOCAL)

        # Check if can execute
        preflight = await ability.preflight(context)
        if not preflight.can_execute:
            print(f"Cannot execute: {preflight.reason}")
            return

        # Execute
        result = await ability.execute({"param": "value"}, context)
        if not result.success:
            print(f"Error: {result.error}")
            return

        # Verify output
        verification = await ability.verify(result)
        if not verification.verified:
            print(f"Issues: {verification.issues}")
    """

    @property
    def manifest(self) -> AbilityManifest:
        """Return ability manifest with metadata and capabilities."""
        ...

    async def preflight(self, context: AbilityContext) -> PreflightResult:
        """Check if ability can execute in current context.

        Should check:
        - Required permissions are available
        - Dependencies are installed
        - Trust level is sufficient
        - User confirmation obtained if needed

        Args:
            context: Execution context with permissions and metadata

        Returns:
            PreflightResult indicating if execution is possible
        """
        ...

    async def execute(
        self,
        params: dict[str, Any],
        context: AbilityContext
    ) -> AbilityResult:
        """Execute the ability with given parameters.

        Args:
            params: Ability-specific parameters
            context: Execution context with permissions

        Returns:
            AbilityResult with output and status
        """
        ...

    async def verify(self, result: AbilityResult) -> VerificationResult:
        """Verify the execution result.

        Should check:
        - Output format is correct
        - No hallucinations or invalid data
        - Confidence score is justified

        Args:
            result: Result from execute()

        Returns:
            VerificationResult with verification status and issues
        """
        ...


class BaseAbility(ABC):
    """Base implementation with common functionality.

    Extend this for Tier 2 plugin abilities.

    Provides:
    - Manifest property
    - Default preflight (always can execute)
    - Default verification (pass if success)
    - Common logging

    Example:
        class MyAbility(BaseAbility):
            def __init__(self):
                super().__init__(AbilityManifest(
                    name="my_ability",
                    description="Does something useful",
                    permissions=["read_file"],
                    tags=["utility"]
                ))

            async def execute(self, params, context):
                # Implementation
                return AbilityResult(success=True, output=...)
    """

    def __init__(self, manifest: AbilityManifest):
        """Initialize ability with manifest.

        Args:
            manifest: AbilityManifest with metadata
        """
        self._manifest = manifest

    @property
    def manifest(self) -> AbilityManifest:
        """Return ability manifest."""
        return self._manifest

    @property
    def name(self) -> str:
        """Return ability name."""
        return self._manifest.name

    @property
    def version(self) -> str:
        """Return ability version."""
        return self._manifest.version

    async def preflight(self, context: AbilityContext) -> PreflightResult:
        """Default preflight: always can execute.

        Override for custom permission/dependency checking.
        """
        return PreflightResult(can_execute=True)

    @abstractmethod
    async def execute(
        self,
        params: dict[str, Any],
        context: AbilityContext
    ) -> AbilityResult:
        """Execute the ability - must be implemented by subclasses.

        Args:
            params: Ability-specific parameters
            context: Execution context

        Returns:
            AbilityResult with output and status
        """
        ...

    async def verify(self, result: AbilityResult) -> VerificationResult:
        """Default verification: pass if success.

        Override for custom output validation.
        """
        return VerificationResult(verified=result.success)
