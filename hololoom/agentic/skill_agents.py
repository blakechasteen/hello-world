#!/usr/bin/env python3
"""
Skill Agent System for HoloLoom
================================

Loads YAML-based skill templates and executes them using HoloLoom's
recursive weaving architecture.

**UPDATED (Phase 1.3 - Nov 2025):** Enhanced with UnifiedMRF integration.

Integrates Promptly's 13 professional skills with HoloLoom's:
- Recursive reasoning strategies
- Quality-driven refinement
- Complete provenance tracking
- Analytics integration
- 7-component metaprompt framework
- Model-specific optimizations

Created: 2025-11-16
Integration: Promptly Skills → HoloLoom Agents
Enhanced: 2025-11-22 (UnifiedMRF integration)

Usage:
    from hololoom.agentic.skill_agents import SkillRegistry, execute_skill

    # Load all skills
    registry = SkillRegistry()
    await registry.load_all_skills()

    # Execute a skill with metaprompt enhancement
    result = await execute_skill(
        skill_name="code-reviewer",
        parameters={"code": code, "language": "python"},
        config=Config.fast(),
        model_provider="claude"  # Optional: claude, gemini, gpt, ollama
    )

    print(result.output)
    print(result.confidence)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hololoom.config import Config
from hololoom.prompting.unified_mrf import MetapromptConfig, ModelProvider, UnifiedMRF
from hololoom.protocols.recursive_reasoning import ReasoningStrategy
from hololoom.protocols.types import MemoryShard, Query
from hololoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SkillMetadata:
    """Metadata for a skill template."""
    name: str
    version: str
    description: str
    category: str
    tags: list[str]
    author: str
    created: str
    updated: str | None = None  # NEW: Track enhancement dates


@dataclass
class SkillMetaprompt:
    """
    7-component metaprompt framework for skills.

    NEW in v1.1.0 (Phase 1.3): Enhanced prompting using UnifiedMRF.
    """
    role: str
    objective: dict[str, Any]  # {primary: str, secondary: List[str]}
    process: list[Any]  # Can be strings or dicts
    format: str
    constraints: list[str]
    uncertainty: str
    validation: list[str]


@dataclass
class SkillModelConfig:
    """Model-specific configuration for skills."""
    preferred_provider: str = "claude"
    fallback_providers: list[str] = field(default_factory=lambda: ["gpt", "ollama"])
    claude_hints: dict[str, Any] = field(default_factory=dict)
    gemini_hints: dict[str, Any] = field(default_factory=dict)
    gpt_hints: dict[str, Any] = field(default_factory=dict)
    ollama_hints: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillReasoningConfig:
    """Reasoning configuration for a skill."""
    default_strategy: str
    max_iterations: int
    quality_threshold: float
    refinement_passes: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SkillParameter:
    """Parameter definition for a skill."""
    name: str
    type: str
    required: bool
    default: Any = None
    description: str = ""


@dataclass
class SkillTemplate:
    """
    Complete skill template loaded from YAML.

    **UPDATED (Phase 1.3):** Added metaprompt and model_config fields.
    """
    name: str
    version: str
    description: str
    metadata: SkillMetadata
    reasoning: SkillReasoningConfig
    system_prompt: str
    user_prompt_template: str
    parameters: list[SkillParameter]
    output: dict[str, Any]
    quality_checks: list[dict[str, str]] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    # NEW: Metaprompt framework (Phase 1.3)
    metaprompt: SkillMetaprompt | None = None
    model_config: SkillModelConfig | None = None


@dataclass
class SkillExecutionResult:
    """Result from executing a skill."""
    skill_name: str
    output: str
    confidence: float
    iterations: int
    strategy_used: str
    execution_time_ms: float
    metadata: dict[str, Any]
    success: bool
    error: str | None = None


# ============================================================================
# Skill Registry
# ============================================================================

class SkillRegistry:
    """
    Central registry for all skill templates.

    Loads YAML files from hololoom/agentic/skills/ and provides
    access to skill definitions.
    """

    def __init__(self, skills_dir: Path | None = None):
        """
        Initialize skill registry.

        Args:
            skills_dir: Directory containing skill YAML files
                       (default: HoloLoom/agentic/skills/)
        """
        if skills_dir is None:
            # Default to skills/ directory relative to this file
            skills_dir = Path(__file__).parent / "skills"

        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, SkillTemplate] = {}
        self.logger = logging.getLogger("skill_registry")

    async def load_all_skills(self):
        """Load all skill templates from YAML files."""
        if not self.skills_dir.exists():
            self.logger.warning(f"Skills directory not found: {self.skills_dir}")
            return

        yaml_files = list(self.skills_dir.glob("*.yaml"))
        self.logger.info(f"Loading {len(yaml_files)} skill templates...")

        for yaml_file in yaml_files:
            try:
                skill = self._load_skill_yaml(yaml_file)
                self.skills[skill.name] = skill
                self.logger.info(f"  ✓ Loaded: {skill.name} (v{skill.version})")
            except Exception as e:
                self.logger.error(f"  ✗ Failed to load {yaml_file.name}: {e}")

        self.logger.info(f"Loaded {len(self.skills)} skills successfully")

    def _load_skill_yaml(self, yaml_path: Path) -> SkillTemplate:
        """
        Load a single skill template from YAML.

        **UPDATED (Phase 1.3):** Parses metaprompt and model_config sections.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Parse metadata
        metadata_dict = data.get("metadata", {})
        metadata = SkillMetadata(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            category=metadata_dict.get("category", "general"),
            tags=metadata_dict.get("tags", []),
            author=metadata_dict.get("author", "Unknown"),
            created=metadata_dict.get("created", "Unknown"),
            updated=metadata_dict.get("updated")  # NEW
        )

        # Parse reasoning config
        reasoning_dict = data.get("reasoning", {})
        reasoning = SkillReasoningConfig(
            default_strategy=reasoning_dict.get("default_strategy", "refine"),
            max_iterations=reasoning_dict.get("max_iterations", 3),
            quality_threshold=reasoning_dict.get("quality_threshold", 0.85),
            refinement_passes=reasoning_dict.get("refinement_passes", [])
        )

        # Parse parameters
        parameters = []
        for param_dict in data.get("parameters", []):
            param = SkillParameter(
                name=param_dict["name"],
                type=param_dict["type"],
                required=param_dict.get("required", False),
                default=param_dict.get("default"),
                description=param_dict.get("description", "")
            )
            parameters.append(param)

        # NEW: Parse metaprompt (if present)
        metaprompt = None
        if "metaprompt" in data:
            mp_dict = data["metaprompt"]
            metaprompt = SkillMetaprompt(
                role=mp_dict.get("role", ""),
                objective=mp_dict.get("objective", {"primary": "", "secondary": []}),
                process=mp_dict.get("process", []),
                format=mp_dict.get("format", ""),
                constraints=mp_dict.get("constraints", []),
                uncertainty=mp_dict.get("uncertainty", ""),
                validation=mp_dict.get("validation", [])
            )

        # NEW: Parse model_config (if present)
        model_config = None
        if "model_config" in data:
            mc_dict = data["model_config"]
            model_config = SkillModelConfig(
                preferred_provider=mc_dict.get("preferred_provider", "claude"),
                fallback_providers=mc_dict.get("fallback_providers", ["gpt", "ollama"]),
                claude_hints=mc_dict.get("claude_hints", {}),
                gemini_hints=mc_dict.get("gemini_hints", {}),
                gpt_hints=mc_dict.get("gpt_hints", {}),
                ollama_hints=mc_dict.get("ollama_hints", {})
            )

        # Create skill template
        skill = SkillTemplate(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            metadata=metadata,
            reasoning=reasoning,
            system_prompt=data.get("system_prompt", ""),
            user_prompt_template=data.get("user_prompt_template", ""),
            parameters=parameters,
            output=data.get("output", {}),
            quality_checks=data.get("quality_checks", []),
            examples=data.get("examples", []),
            metaprompt=metaprompt,  # NEW
            model_config=model_config  # NEW
        )

        return skill

    def get_skill(self, skill_name: str) -> SkillTemplate | None:
        """Get a skill template by name."""
        return self.skills.get(skill_name)

    def list_skills(self) -> list[str]:
        """List all available skill names."""
        return list(self.skills.keys())

    def list_skills_by_category(self) -> dict[str, list[str]]:
        """List skills grouped by category."""
        by_category = {}
        for skill in self.skills.values():
            category = skill.metadata.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(skill.name)
        return by_category


# ============================================================================
# Skill Execution
# ============================================================================

class SkillExecutor:
    """
    Executes skills using HoloLoom's recursive weaving architecture.

    **UPDATED (Phase 1.3):** Integrated with UnifiedMRF for enhanced prompting.

    Integrates skill templates with RecursiveWeavingOrchestrator for:
    - Quality-driven refinement
    - Strategy-based recursive reasoning
    - Complete provenance tracking
    - Analytics integration
    - 7-component metaprompt framework
    - Model-specific optimizations
    """

    def __init__(
        self,
        registry: SkillRegistry,
        config: Config,
        shards: list[MemoryShard] | None = None,
        enable_analytics: bool = True,
        model_provider: str | None = None
    ):
        """
        Initialize skill executor.

        Args:
            registry: Skill registry with loaded templates
            config: HoloLoom configuration
            shards: Optional memory shards for context
            enable_analytics: Enable analytics tracking
            model_provider: Optional model provider (claude, gemini, gpt, ollama)
        """
        self.registry = registry
        self.config = config
        self.shards = shards or []
        self.enable_analytics = enable_analytics
        self.model_provider = model_provider
        self.logger = logging.getLogger("skill_executor")

        # UnifiedMRF for enhanced prompting
        self.mrf = UnifiedMRF()

    async def execute(
        self,
        skill_name: str,
        parameters: dict[str, Any],
        override_strategy: str | None = None,
        override_iterations: int | None = None
    ) -> SkillExecutionResult:
        """
        Execute a skill with given parameters.

        Args:
            skill_name: Name of skill to execute
            parameters: Input parameters for the skill
            override_strategy: Override default reasoning strategy
            override_iterations: Override max iterations

        Returns:
            SkillExecutionResult with output and metadata
        """
        import time

        # Get skill template
        skill = self.registry.get_skill(skill_name)
        if not skill:
            return SkillExecutionResult(
                skill_name=skill_name,
                output="",
                confidence=0.0,
                iterations=0,
                strategy_used="none",
                execution_time_ms=0.0,
                metadata={},
                success=False,
                error=f"Skill not found: {skill_name}"
            )

        start_time = time.time()

        try:
            # Validate parameters
            self._validate_parameters(skill, parameters)

            # Fill in defaults for missing parameters
            full_params = self._fill_defaults(skill, parameters)

            # Build prompt from template
            query_text = self._build_prompt(skill, full_params)

            # Determine reasoning strategy
            strategy_name = override_strategy or skill.reasoning.default_strategy
            strategy = self._parse_strategy(strategy_name)

            # Determine max iterations
            max_iterations = override_iterations or skill.reasoning.max_iterations

            # Create orchestrator
            async with RecursiveWeavingOrchestrator(
                cfg=self.config,
                shards=self.shards,
                enable_recursive=True,
                enable_analytics=self.enable_analytics,
                quality_threshold=skill.reasoning.quality_threshold,
                max_iterations=max_iterations,
                default_strategy=strategy
            ) as orchestrator:

                # Execute with recursive reasoning
                query = Query(text=query_text)
                spacetime = await orchestrator.weave_with_strategy(
                    query=query,
                    strategy=strategy,
                    max_iterations=max_iterations,
                    quality_threshold=skill.reasoning.quality_threshold
                )

                # Calculate execution time
                execution_time = (time.time() - start_time) * 1000

                # Build result
                result = SkillExecutionResult(
                    skill_name=skill_name,
                    output=spacetime.response,
                    confidence=spacetime.confidence,
                    iterations=spacetime.iterations,
                    strategy_used=strategy.value if spacetime.strategy_used else strategy_name,
                    execution_time_ms=execution_time,
                    metadata={
                        "skill_version": skill.version,
                        "parameters": full_params,
                        "reasoning_journal": spacetime.reasoning_journal.get_history() if spacetime.reasoning_journal else None,
                        "refinement_passes": skill.reasoning.refinement_passes
                    },
                    success=True
                )

                self.logger.info(
                    f"Skill '{skill_name}' executed: "
                    f"confidence={result.confidence:.2f}, "
                    f"iterations={result.iterations}, "
                    f"time={result.execution_time_ms:.1f}ms"
                )

                return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Skill execution failed: {e}")

            return SkillExecutionResult(
                skill_name=skill_name,
                output="",
                confidence=0.0,
                iterations=0,
                strategy_used="none",
                execution_time_ms=execution_time,
                metadata={"parameters": parameters},
                success=False,
                error=str(e)
            )

    def _validate_parameters(self, skill: SkillTemplate, parameters: dict[str, Any]):
        """Validate that required parameters are provided."""
        for param in skill.parameters:
            if param.required and param.name not in parameters:
                raise ValueError(f"Missing required parameter: {param.name}")

    def _fill_defaults(self, skill: SkillTemplate, parameters: dict[str, Any]) -> dict[str, Any]:
        """Fill in default values for missing parameters."""
        full_params = dict(parameters)

        for param in skill.parameters:
            if param.name not in full_params and param.default is not None:
                full_params[param.name] = param.default

        return full_params

    def _build_prompt(self, skill: SkillTemplate, parameters: dict[str, Any]) -> str:
        """
        Build complete prompt from template and parameters.

        **UPDATED (Phase 1.3):** Uses UnifiedMRF when metaprompt is present.
        """
        # If skill has metaprompt, use UnifiedMRF for enhanced prompting
        if skill.metaprompt is not None:
            # Convert SkillMetaprompt to MetapromptConfig
            metaprompt_config = MetapromptConfig(
                role=skill.metaprompt.role,
                objective=skill.metaprompt.objective,
                process=skill.metaprompt.process,
                format_spec=skill.metaprompt.format,
                constraints=skill.metaprompt.constraints,
                uncertainty=skill.metaprompt.uncertainty,
                validation=skill.metaprompt.validation
            )

            # Format user prompt with parameters
            user_prompt = skill.user_prompt_template.format(**parameters) if skill.user_prompt_template else ""

            # Use UnifiedMRF to build enhanced prompt
            # Determine model provider from skill config or fallback
            provider = None
            if skill.model_config and self.model_provider:
                # Map string to ModelProvider enum
                provider_map = {
                    "claude": ModelProvider.CLAUDE,
                    "gemini": ModelProvider.GEMINI,
                    "gpt": ModelProvider.GPT,
                    "ollama": ModelProvider.OLLAMA
                }
                provider = provider_map.get(self.model_provider.lower())

            enhanced_prompt = self.mrf.enhance_prompt(
                metaprompt=metaprompt_config,
                query=user_prompt,
                model=provider
            )

            return enhanced_prompt

        # Fallback: Traditional system_prompt + user_prompt_template
        else:
            prompt_parts = []

            if skill.system_prompt:
                prompt_parts.append(skill.system_prompt.strip())

            if skill.user_prompt_template:
                # Format user prompt with parameters
                user_prompt = skill.user_prompt_template.format(**parameters)
                prompt_parts.append(user_prompt.strip())

            return "\n\n".join(prompt_parts)

    def _parse_strategy(self, strategy_name: str) -> ReasoningStrategy:
        """Parse strategy name to ReasoningStrategy enum."""
        strategy_map = {
            "direct": ReasoningStrategy.DIRECT,
            "verify": ReasoningStrategy.VERIFY,
            "refine": ReasoningStrategy.REFINE,
            "critique": ReasoningStrategy.CRITIQUE,
            "decompose": ReasoningStrategy.DECOMPOSE,
            "explore": ReasoningStrategy.EXPLORE,
            "hofstadter": ReasoningStrategy.HOFSTADTER,
            "adaptive": ReasoningStrategy.ADAPTIVE
        }

        return strategy_map.get(strategy_name.lower(), ReasoningStrategy.REFINE)


# ============================================================================
# Convenience Functions
# ============================================================================

# Global registry instance
_global_registry: SkillRegistry | None = None


async def get_registry() -> SkillRegistry:
    """Get or create global skill registry."""
    global _global_registry

    if _global_registry is None:
        _global_registry = SkillRegistry()
        await _global_registry.load_all_skills()

    return _global_registry


async def execute_skill(
    skill_name: str,
    parameters: dict[str, Any],
    config: Config | None = None,
    shards: list[MemoryShard] | None = None,
    override_strategy: str | None = None,
    override_iterations: int | None = None,
    enable_analytics: bool = True,
    model_provider: str | None = None  # NEW (Phase 1.3)
) -> SkillExecutionResult:
    """
    Convenient function to execute a skill.

    **UPDATED (Phase 1.3):** Added model_provider parameter for UnifiedMRF integration.

    Args:
        skill_name: Name of skill to execute
        parameters: Input parameters
        config: HoloLoom config (default: Config.fast())
        shards: Optional memory context
        override_strategy: Override reasoning strategy
        override_iterations: Override max iterations
        enable_analytics: Enable analytics tracking
        model_provider: Model provider for enhanced prompting ("claude", "gemini", "gpt", "ollama")

    Returns:
        SkillExecutionResult

    Example:
        # Basic usage
        result = await execute_skill(
            "code-reviewer",
            {"code": code, "language": "python"},
            config=Config.fused()
        )

        # With model provider (enables UnifiedMRF enhancements)
        result = await execute_skill(
            "code-reviewer",
            {"code": code, "language": "python"},
            config=Config.fused(),
            model_provider="claude"
        )
    """
    registry = await get_registry()
    executor = SkillExecutor(
        registry=registry,
        config=config or Config.fast(),
        shards=shards,
        enable_analytics=enable_analytics,
        model_provider=model_provider  # NEW
    )

    return await executor.execute(
        skill_name=skill_name,
        parameters=parameters,
        override_strategy=override_strategy,
        override_iterations=override_iterations
    )


async def list_available_skills() -> dict[str, list[str]]:
    """
    List all available skills grouped by category.

    Returns:
        Dictionary mapping category → list of skill names

    Example:
        skills = await list_available_skills()
        print(skills["development"])  # ['code-reviewer', 'bug-detective', ...]
    """
    registry = await get_registry()
    return registry.list_skills_by_category()
