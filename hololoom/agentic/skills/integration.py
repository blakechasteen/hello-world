"""
Skills Integration with Agentic Reasoning
==========================================
Integrates HoloLoom skills system with agentic reasoning modes.

Features:
- Execute skills in DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE modes
- Skill-based query routing
- Multi-skill workflows
- Skill result verification

Usage:
    from hololoom.agentic.skills import SkillLoader, execute_skill
    from hololoom.agentic import AgenticOrchestrator

    skill = SkillLoader().load('code_reviewer')
    result = await execute_skill(skill, context, orchestrator)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hololoom.agentic.skills.skill_loader import Skill, SkillLoader
from hololoom.fabric.spacetime import Spacetime
from hololoom.protocols.types import Query

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class SkillContext:
    """
    Context for skill execution.

    Attributes:
        query: Original query
        parameters: Skill-specific parameters
        mode: Reasoning mode to use
        metadata: Additional context metadata
    """
    query: Query
    parameters: dict[str, Any] = field(default_factory=dict)
    mode: str = "direct"  # direct, verify, research, plan_execute
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """
    Result of skill execution.

    Attributes:
        skill_name: Name of executed skill
        success: Whether execution succeeded
        output: Skill output
        spacetime: Underlying spacetime result (if orchestrator used)
        confidence: Confidence in result (0.0-1.0)
        duration_ms: Execution duration
        errors: List of errors encountered
        metadata: Additional result metadata
    """
    skill_name: str
    success: bool
    output: Any
    spacetime: Spacetime | None = None
    confidence: float = 0.0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Skill Execution
# ============================================================================

class SkillExecutor:
    """
    Execute skills within agentic reasoning context.

    Integrates skills with WeavingOrchestrator for enhanced reasoning.
    """

    def __init__(
        self,
        orchestrator: Any | None = None,
        skill_loader: SkillLoader | None = None
    ):
        """
        Initialize skill executor.

        Args:
            orchestrator: WeavingOrchestrator or AgenticOrchestrator instance
            skill_loader: SkillLoader instance (creates default if not provided)
        """
        self.orchestrator = orchestrator
        self.skill_loader = skill_loader or SkillLoader()
        self.logger = logging.getLogger(__name__)

    async def execute(
        self,
        skill: Skill,
        context: SkillContext
    ) -> SkillResult:
        """
        Execute a skill with given context.

        Args:
            skill: Skill to execute
            context: Execution context

        Returns:
            SkillResult with output and metadata
        """
        start_time = datetime.now()

        try:
            # Build enhanced query with skill context
            enhanced_query = self._build_skill_query(skill, context)

            # Execute through orchestrator if available
            if self.orchestrator:
                spacetime = await self._execute_with_orchestrator(
                    enhanced_query,
                    context
                )

                result = SkillResult(
                    skill_name=skill.name,
                    success=True,
                    output=spacetime.response,
                    spacetime=spacetime,
                    confidence=spacetime.confidence,
                    duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    metadata={
                        "skill_description": skill.description,
                        "skill_tags": skill.metadata.tags,
                        "mode": context.mode
                    }
                )
            else:
                # Fallback: execute skill without orchestrator
                output = self._execute_standalone(skill, context)

                result = SkillResult(
                    skill_name=skill.name,
                    success=True,
                    output=output,
                    confidence=0.5,  # Default confidence without verification
                    duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    metadata={"mode": "standalone"}
                )

            self.logger.info(
                f"Executed skill '{skill.name}' in {result.duration_ms:.1f}ms "
                f"(confidence: {result.confidence:.2f})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Skill execution failed for '{skill.name}': {e}")

            return SkillResult(
                skill_name=skill.name,
                success=False,
                output=None,
                errors=[str(e)],
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _build_skill_query(
        self,
        skill: Skill,
        context: SkillContext
    ) -> Query:
        """Build enhanced query with skill context."""
        # Get skill documentation
        docs = []
        for doc_file in skill.get_markdown_files():
            docs.append(f"# {doc_file.filename}\n{doc_file.content}")

        skill_context = "\n\n".join(docs) if docs else skill.description

        # Build enhanced query text
        query_text = f"""Skill: {skill.name}

Skill Context:
{skill_context}

User Query: {context.query.text}

Parameters: {context.parameters if context.parameters else 'None'}

Please execute this skill and provide a structured response.
"""

        return Query(text=query_text)

    async def _execute_with_orchestrator(
        self,
        query: Query,
        context: SkillContext
    ) -> Spacetime:
        """Execute query through orchestrator."""
        # Check if orchestrator supports reasoning modes
        if hasattr(self.orchestrator, 'reason'):
            # AgenticOrchestrator with reasoning modes
            from hololoom.agentic.core import ReasoningMode

            mode_map = {
                "direct": ReasoningMode.DIRECT,
                "verify": ReasoningMode.VERIFY,
                "research": ReasoningMode.RESEARCH,
                "plan_execute": ReasoningMode.PLAN_EXECUTE
            }

            mode = mode_map.get(context.mode, ReasoningMode.DIRECT)
            result = await self.orchestrator.reason(query, mode=mode)
            return result.spacetime

        else:
            # WeavingOrchestrator - basic weaving
            return await self.orchestrator.weave(query)

    def _execute_standalone(
        self,
        skill: Skill,
        context: SkillContext
    ) -> str:
        """
        Execute skill without orchestrator (fallback).

        Returns basic skill template information.
        """
        output = [
            f"Skill: {skill.name}",
            f"Description: {skill.description}",
            "",
            f"Query: {context.query.text}",
            "",
            "Available Resources:"
        ]

        # List available files
        for filename in skill.files.keys():
            output.append(f"  - {filename}")

        output.append("")
        output.append("Note: Full skill execution requires WeavingOrchestrator integration.")

        return "\n".join(output)


# ============================================================================
# Multi-Skill Workflows
# ============================================================================

class SkillWorkflow:
    """
    Execute multi-skill workflows.

    Chains multiple skills together for complex tasks.
    """

    def __init__(
        self,
        executor: SkillExecutor,
        skills: list[Skill]
    ):
        """
        Initialize workflow.

        Args:
            executor: SkillExecutor instance
            skills: List of skills to execute in sequence
        """
        self.executor = executor
        self.skills = skills
        self.logger = logging.getLogger(__name__)

    async def execute(
        self,
        initial_context: SkillContext,
        pass_results: bool = True
    ) -> list[SkillResult]:
        """
        Execute skill workflow.

        Args:
            initial_context: Initial context for first skill
            pass_results: Whether to pass results between skills

        Returns:
            List of SkillResults (one per skill)
        """
        results = []
        context = initial_context

        for i, skill in enumerate(self.skills):
            self.logger.info(f"Executing workflow step {i+1}/{len(self.skills)}: {skill.name}")

            # Execute skill
            result = await self.executor.execute(skill, context)
            results.append(result)

            # Pass result to next skill if requested
            if pass_results and i < len(self.skills) - 1:
                # Update context with previous result
                context = SkillContext(
                    query=Query(text=f"Previous result: {result.output}\n\n{context.query.text}"),
                    parameters=context.parameters,
                    mode=context.mode,
                    metadata={
                        **context.metadata,
                        f"step_{i}_result": result.output
                    }
                )

        self.logger.info(f"Workflow completed: {len(results)} skills executed")
        return results


# ============================================================================
# Convenience Functions
# ============================================================================

async def execute_skill(
    skill_name: str,
    query: Query,
    orchestrator: Any | None = None,
    mode: str = "direct",
    parameters: dict[str, Any] | None = None
) -> SkillResult:
    """
    Quick skill execution.

    Args:
        skill_name: Name of skill template
        query: Query to process
        orchestrator: Optional WeavingOrchestrator
        mode: Reasoning mode (direct, verify, research, plan_execute)
        parameters: Optional skill parameters

    Returns:
        SkillResult
    """
    loader = SkillLoader()
    skill = loader.load(skill_name)

    executor = SkillExecutor(orchestrator=orchestrator)

    context = SkillContext(
        query=query,
        parameters=parameters or {},
        mode=mode
    )

    return await executor.execute(skill, context)


async def execute_workflow(
    skill_names: list[str],
    query: Query,
    orchestrator: Any | None = None,
    mode: str = "direct"
) -> list[SkillResult]:
    """
    Quick multi-skill workflow execution.

    Args:
        skill_names: List of skill names to execute in sequence
        query: Initial query
        orchestrator: Optional WeavingOrchestrator
        mode: Reasoning mode for all skills

    Returns:
        List of SkillResults
    """
    loader = SkillLoader()
    skills = [loader.load(name) for name in skill_names]

    executor = SkillExecutor(orchestrator=orchestrator)
    workflow = SkillWorkflow(executor, skills)

    context = SkillContext(query=query, mode=mode)

    return await workflow.execute(context)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Skills Integration with Agentic Reasoning\n")

    print("Example - Single Skill Execution:")
    print("""
from hololoom.agentic.skills import execute_skill
from hololoom.protocols.types import Query

# Execute code review skill
result = await execute_skill(
    skill_name='code_reviewer',
    query=Query(text='Review this Python function: def add(a, b): return a + b'),
    mode='verify'
)

print(f"Result: {result.output}")
print(f"Confidence: {result.confidence}")
""")

    print("\nExample - Multi-Skill Workflow:")
    print("""
from hololoom.agentic.skills import execute_workflow

# Execute code review → test generation workflow
results = await execute_workflow(
    skill_names=['code_reviewer', 'test_generator'],
    query=Query(text='Review and test this code: ...'),
    mode='verify'
)

for i, result in enumerate(results):
    print(f"Step {i+1}: {result.skill_name} - {result.success}")
""")
