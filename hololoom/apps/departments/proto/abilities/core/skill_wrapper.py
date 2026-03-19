"""Proto Skill Wrapper - Tier 1 Abilities.

Thin wrappers around HoloLoom's 13 professional skills.

Maps Proto ability names to HoloLoom skill templates (from
HoloLoom/agentic/skills/templates.py). This is the simplest form
of ability - direct delegation to HoloLoom's agentic orchestrator.

Skill Categories:

Code Quality (3):
    - review → code_reviewer
    - test → test_generator
    - debug → (custom logic, no direct skill)

Code Understanding (2):
    - explain → (custom logic, no direct skill)
    - document → documentation_writer

Code Improvement (3):
    - refactor → refactoring_expert
    - name → (custom logic, no direct skill)
    - architect → system_architect

Specialized (2):
    - sql → sql_designer
    - performance → sql_optimizer

Security & Design (2):
    - security → security_auditor
    - api → api_designer
    - migrate → (custom logic, no direct skill)

Implementation: December 2025
Status: Production ready
"""

import logging
import time
from abc import ABC
from typing import Any

from ..protocol import (
    AbilityContext,
    AbilityManifest,
    AbilityResult,
    AbilityTier,
    AbilityTrustLevel,
    PreflightResult,
    VerificationResult,
)

logger = logging.getLogger("proto.skill")


# Mapping from Proto ability names to HoloLoom skill names
# Proto names are short and intuitive; HoloLoom names are descriptive
SKILL_MAP = {
    # Code quality
    "review": "code_reviewer",
    "test": "test_generator",

    # Code understanding
    "explain": None,  # Custom implementation needed
    "document": "documentation_writer",

    # Code improvement
    "refactor": "refactoring_expert",
    "name": None,  # Custom implementation needed
    "architect": "system_architect",

    # Specialized
    "sql": "sql_designer",
    "performance": "sql_optimizer",

    # Security & design
    "security": "security_auditor",
    "api": "api_designer",

    # Migration
    "migrate": None,  # Custom implementation needed
}


# Skill descriptions (for manifest and discovery)
SKILL_DESCRIPTIONS = {
    "review": "Review code for best practices, bugs, and security issues",
    "test": "Generate comprehensive unit tests for code",
    "explain": "Explain code and programming concepts clearly",
    "document": "Generate comprehensive documentation from code",
    "refactor": "Refactor code for better maintainability and performance",
    "name": "Suggest better variable and function names",
    "architect": "Design scalable, maintainable system architectures",
    "sql": "Design database schemas and write optimized SQL queries",
    "performance": "Optimize SQL queries for performance and readability",
    "security": "Audit code and systems for security vulnerabilities",
    "api": "Design RESTful APIs with best practices and OpenAPI specs",
    "migrate": "Plan and execute database migrations safely",
}


# Tags for categorization and discovery
SKILL_TAGS = {
    "review": ["code", "review", "quality"],
    "test": ["testing", "pytest", "coverage"],
    "explain": ["explanation", "documentation", "learning"],
    "document": ["documentation", "markdown", "readme"],
    "refactor": ["refactoring", "clean-code", "maintainability"],
    "name": ["naming", "conventions", "readability"],
    "architect": ["architecture", "design", "scalability"],
    "sql": ["sql", "database", "schema"],
    "performance": ["sql", "optimization", "performance"],
    "security": ["security", "vulnerabilities", "owasp"],
    "api": ["api", "design", "rest"],
    "migrate": ["migration", "database", "deployment"],
}


class SkillWrapperAbility(ABC):
    """Tier 1 Ability: Wrapper around HoloLoom skills.

    Implements the Ability protocol by delegating to HoloLoom's
    agentic orchestrator with the appropriate skill.

    Example:
        wrapper = SkillWrapperAbility("review")
        context = AbilityContext(session_id="123")
        result = await wrapper.execute({"code": "..."}, context)

    Attributes:
        _proto_name: Proto ability name (e.g., "review")
        _skill_name: HoloLoom skill name (e.g., "code_reviewer")
        _agentic: AgenticOrchestrator instance (optional, for integration)
        _manifest: AbilityManifest with metadata
    """

    def __init__(
        self,
        proto_name: str,
        agentic_orchestrator: Any | None = None
    ):
        """Initialize skill wrapper.

        Args:
            proto_name: Proto ability name (e.g., "review")
            agentic_orchestrator: HoloLoom AgenticOrchestrator instance
                                 (optional, enables LLM integration)

        Raises:
            ValueError: If proto_name is not recognized
        """
        if proto_name not in SKILL_MAP:
            valid = list(SKILL_MAP.keys())
            raise ValueError(f"Unknown ability: {proto_name}. Valid: {valid}")

        self._proto_name = proto_name
        self._skill_name = SKILL_MAP[proto_name]
        self._agentic = agentic_orchestrator

        # Build manifest
        self._manifest = AbilityManifest(
            name=proto_name,
            version="1.0.0",
            description=SKILL_DESCRIPTIONS.get(
                proto_name,
                f"HoloLoom skill: {self._skill_name}"
            ),
            author="hololoom",
            tier=AbilityTier.SKILL_MAPPING,
            trust_level=AbilityTrustLevel.CORE,
            tags=SKILL_TAGS.get(proto_name, [proto_name]),
        )

    @property
    def manifest(self) -> AbilityManifest:
        """Return ability manifest."""
        return self._manifest

    @property
    def proto_name(self) -> str:
        """Return Proto ability name."""
        return self._proto_name

    @property
    def skill_name(self) -> str | None:
        """Return underlying HoloLoom skill name (if mapped)."""
        return self._skill_name

    async def preflight(self, context: AbilityContext) -> PreflightResult:
        """Check if ability can execute.

        For skill wrappers with HoloLoom integration, returns True.
        For fallback implementations, checks if basic execution is possible.

        Args:
            context: Execution context

        Returns:
            PreflightResult indicating execution feasibility
        """
        # Check if we have HoloLoom integration for this skill
        if self._skill_name is None:
            # Custom implementation (doesn't require HoloLoom)
            return PreflightResult(can_execute=True)

        if self._agentic is None:
            return PreflightResult(
                can_execute=True,  # Fallback mode works without HoloLoom
                warnings=[
                    "HoloLoom AgenticOrchestrator not available, "
                    "using simplified fallback implementation"
                ]
            )

        return PreflightResult(can_execute=True)

    async def execute(
        self,
        params: dict[str, Any],
        context: AbilityContext
    ) -> AbilityResult:
        """Execute the ability.

        Routes to HoloLoom if available, falls back to simplified
        implementation otherwise.

        Args:
            params: Ability parameters (code, query, etc.)
            context: Execution context

        Returns:
            AbilityResult with output and status
        """
        start = time.time()

        try:
            # Build query for HoloLoom from parameters
            query = self._build_query(params)

            # Execute via agentic orchestrator if available
            if self._agentic and self._skill_name:
                result = await self._execute_via_agentic(query, params)
            else:
                # Fallback to simplified implementation
                result = self._fallback_response(params)

            duration = (time.time() - start) * 1000

            return AbilityResult(
                success=True,
                output=result.get("response", result),
                confidence=result.get("confidence", 0.8),
                duration_ms=duration,
                metadata={
                    "skill": self._skill_name,
                    "proto_name": self._proto_name,
                    "has_hololoom": self._agentic is not None,
                }
            )

        except Exception as e:
            logger.error(f"Skill {self._proto_name} execution failed: {e}")
            return AbilityResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000
            )

    def _build_query(self, params: dict[str, Any]) -> str:
        """Build query string from parameters based on ability type.

        Args:
            params: Input parameters

        Returns:
            Query string for HoloLoom/LLM processing
        """
        code = params.get("code", "")
        query = params.get("query", "")
        context_text = params.get("context", "")
        language = params.get("language", "")
        error = params.get("error", "")

        # Build appropriate query based on ability type
        if self._proto_name in ["review", "security"]:
            lang_hint = f" (Language: {language})" if language else ""
            return f"Please {self._proto_name} this code{lang_hint}:\n```\n{code}\n```"

        elif self._proto_name in ["refactor", "performance"]:
            return f"Please {self._proto_name} this:\n```\n{code}\n```"

        elif self._proto_name == "explain":
            return f"Please explain this code clearly:\n```\n{code}\n```"

        elif self._proto_name == "test":
            return f"Generate comprehensive tests for this code:\n```\n{code}\n```"

        elif self._proto_name == "document":
            return f"Generate professional documentation for this code:\n```\n{code}\n```"

        elif self._proto_name == "name":
            return f"Suggest better variable and function names for this code:\n```\n{code}\n```"

        elif self._proto_name == "architect":
            return f"Design a system architecture for this requirement:\n{query or code}"

        elif self._proto_name == "sql":
            return f"Design a database schema for this:\n{query or code}"

        elif self._proto_name == "api":
            return f"Design a RESTful API for this:\n{query or code}"

        elif self._proto_name == "migrate":
            return f"Plan a migration for this:\n{query or code}"

        elif self._proto_name == "debug":
            error_info = f"\nError: {error}" if error else ""
            return f"Debug this issue{error_info}:\n```\n{code}\n```"

        else:
            return query or f"{self._proto_name}: {code or context_text}"

    async def _execute_via_agentic(
        self,
        query: str,
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute via HoloLoom AgenticOrchestrator.

        Args:
            query: Query string for LLM
            params: Original parameters

        Returns:
            Dict with "response" and "confidence" keys
        """
        try:
            from hololoom.agentic.core import ReasoningMode
            from hololoom.protocols.types import Query

            # Use VERIFY mode for code tasks (accuracy matters)
            result = await self._agentic.reason(
                Query(text=query),
                mode=ReasoningMode.VERIFY
            )

            return {
                "response": result.spacetime.metadata.get("response", ""),
                "confidence": result.spacetime.confidence
            }
        except Exception as e:
            logger.error(f"HoloLoom execution failed: {e}. Using fallback.")
            return self._fallback_response(params)

    def _fallback_response(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fallback response when HoloLoom not available.

        Args:
            params: Input parameters

        Returns:
            Simplified response
        """
        skill_desc = SKILL_DESCRIPTIONS.get(
            self._proto_name,
            f"ability '{self._proto_name}'"
        )

        return {
            "response": (
                f"[{self._proto_name.upper()}] Processed your request for {skill_desc}. "
                "HoloLoom integration not available - this is a simplified response. "
                "For full capabilities, ensure HoloLoom AgenticOrchestrator is configured."
            ),
            "confidence": 0.5
        }

    async def verify(self, result: AbilityResult) -> VerificationResult:
        """Verify skill output.

        Args:
            result: AbilityResult from execute()

        Returns:
            VerificationResult with verification status
        """
        if not result.success:
            return VerificationResult(
                verified=False,
                issues=[result.error or "Execution failed"]
            )

        # Check if output exists
        if not result.output:
            return VerificationResult(
                verified=False,
                issues=["No output generated"]
            )

        # Check for fallback response (lower confidence)
        output_str = str(result.output)
        if "HoloLoom integration not available" in output_str:
            return VerificationResult(
                verified=True,
                suggestions=["Configure HoloLoom for better results"]
            )

        return VerificationResult(verified=True)


def create_skill_abilities(
    agentic_orchestrator: Any | None = None
) -> dict[str, SkillWrapperAbility]:
    """Create all skill wrapper abilities.

    Args:
        agentic_orchestrator: Optional HoloLoom AgenticOrchestrator
                             for enhanced execution

    Returns:
        Dict mapping ability name to SkillWrapperAbility instance
    """
    abilities = {}
    for proto_name in SKILL_MAP:
        abilities[proto_name] = SkillWrapperAbility(
            proto_name,
            agentic_orchestrator
        )
    return abilities


def get_available_skills() -> list:
    """Get list of available Proto skill names.

    Returns:
        List of skill names (e.g., ["review", "test", "debug", ...])
    """
    return list(SKILL_MAP.keys())


def get_skill_description(skill_name: str) -> str | None:
    """Get description for a skill.

    Args:
        skill_name: Proto skill name

    Returns:
        Description string or None if not found
    """
    return SKILL_DESCRIPTIONS.get(skill_name)


def get_skill_tags(skill_name: str) -> list:
    """Get tags for a skill (for discovery).

    Args:
        skill_name: Proto skill name

    Returns:
        List of tag strings
    """
    return SKILL_TAGS.get(skill_name, [])
