# HoloLoom/ralph/templates.py
"""
Ralph Loop Engine - Loop Templates.

Pre-defined loop patterns for common use cases.
Templates define the structure and behavior of iteration loops.

"The technique is deterministically bad in an undeterministic world."
- Geoff Huntley

Created: 2026-01-28
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import re


class LoopTemplate(Enum):
    """Built-in loop templates."""

    # Iteratively refine until quality threshold met
    ITERATIVE_REFINEMENT = "iterative_refinement"

    # Research mode: explore topic from multiple angles
    RESEARCH = "research"

    # Plan then execute: decompose into subtasks, execute each
    PLAN_EXECUTE = "plan_execute"

    # Verify claims and fix errors
    VERIFY_FIX = "verify_fix"

    # Build feature incrementally
    INCREMENTAL_BUILD = "incremental_build"

    # Debug and fix: iteratively identify and fix issues
    DEBUG_FIX = "debug_fix"

    # Custom template (user-defined)
    CUSTOM = "custom"


@dataclass
class TemplateSpec:
    """
    Specification for a loop template.

    Defines the behavior, completion criteria, and prompts for a loop pattern.
    """

    name: str
    description: str

    # Loop behavior
    max_iterations: int = 50
    min_confidence: float = 0.7
    quality_threshold: float = 0.8

    # Completion criteria
    completion_check: Optional[str] = None  # Python expression or keyword
    early_stop_on_high_confidence: bool = True
    stop_on_no_progress: bool = True
    no_progress_iterations: int = 3

    # Prompts/instructions
    system_prompt: str = ""
    iteration_prompt_template: str = ""
    handoff_prompt_template: str = ""

    # Task decomposition
    decompose_tasks: bool = False
    max_subtasks: int = 10

    # Memory integration
    use_memory: bool = True
    consolidate_per_iterations: int = 5
    preserve_patterns: bool = True

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_iteration_prompt(self, context: Dict[str, Any]) -> str:
        """Generate iteration prompt from template."""
        if not self.iteration_prompt_template:
            return ""

        # Simple string formatting
        prompt = self.iteration_prompt_template
        for key, value in context.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def get_handoff_prompt(self, context: Dict[str, Any]) -> str:
        """Generate handoff prompt from template."""
        if not self.handoff_prompt_template:
            return ""

        prompt = self.handoff_prompt_template
        for key, value in context.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def check_completion(self, state: Dict[str, Any]) -> bool:
        """Check if loop should complete based on criteria."""
        # High confidence check
        if self.early_stop_on_high_confidence:
            confidence = state.get("confidence", 0)
            if confidence >= self.quality_threshold:
                return True

        # No progress check
        if self.stop_on_no_progress:
            iterations = state.get("iterations", [])
            if len(iterations) >= self.no_progress_iterations:
                recent = iterations[-self.no_progress_iterations:]
                if all(i.get("status") == "no_change" for i in recent):
                    return True

        # Custom completion check
        if self.completion_check:
            try:
                # Safety: only allow specific keywords
                if self.completion_check in ("task_complete", "all_tests_pass", "quality_met"):
                    return state.get(self.completion_check, False)
            except Exception:
                pass

        return False


# Built-in template definitions
BUILTIN_TEMPLATES: Dict[str, TemplateSpec] = {
    "iterative_refinement": TemplateSpec(
        name="iterative_refinement",
        description="Iteratively refine output until quality threshold is met",
        max_iterations=20,
        min_confidence=0.6,
        quality_threshold=0.85,
        early_stop_on_high_confidence=True,
        system_prompt="""You are working on a task iteratively. Each iteration should:
1. Review previous progress
2. Identify what needs improvement
3. Make targeted improvements
4. Assess quality and confidence

Focus on incremental progress. Small, focused changes are better than large rewrites.""",
        iteration_prompt_template="""
# Iteration {iteration} of {max_iterations}
Task: {task}

## Previous Progress
{previous_summary}

## Current State
Confidence: {confidence:.1%}
Quality: {quality:.1%}

## Instructions
1. Review what was done
2. Identify the most impactful improvement
3. Make that improvement
4. Report progress and new confidence level

Focus on the highest-value change for this iteration.
""",
        handoff_prompt_template="""
# CONTEXT HANDOFF - Iterative Refinement

## Task
{task}

## Progress Summary
- Iterations completed: {iteration}
- Current confidence: {confidence:.1%}
- Current quality: {quality:.1%}

## Recent Changes
{recent_changes}

## Next Steps
{next_steps}

## Key Files
{files_modified}

Continue from where the previous context left off. The task is not complete until quality reaches {quality_threshold:.0%}.
""",
        tags=["refinement", "quality", "iterative"],
    ),

    "research": TemplateSpec(
        name="research",
        description="Explore a topic from multiple angles with synthesis",
        max_iterations=10,
        min_confidence=0.5,
        quality_threshold=0.7,
        decompose_tasks=True,
        max_subtasks=5,
        system_prompt="""You are conducting research on a topic. Each iteration should:
1. Choose an unexplored angle or question
2. Investigate that angle thoroughly
3. Synthesize findings with previous knowledge
4. Identify remaining gaps

Build comprehensive understanding through systematic exploration.""",
        iteration_prompt_template="""
# Research Iteration {iteration}
Topic: {task}

## Knowledge So Far
{accumulated_knowledge}

## Explored Angles
{explored_angles}

## Instructions
1. Identify an unexplored angle or question
2. Research that angle
3. Add findings to accumulated knowledge
4. Note connections to previous findings
5. List remaining questions

Aim for breadth first, then depth on important areas.
""",
        handoff_prompt_template="""
# CONTEXT HANDOFF - Research

## Topic
{task}

## Accumulated Knowledge
{accumulated_knowledge}

## Explored Angles
{explored_angles}

## Remaining Questions
{remaining_questions}

## Key Sources
{sources}

Continue research from this knowledge base. Focus on remaining questions.
""",
        tags=["research", "exploration", "synthesis"],
    ),

    "plan_execute": TemplateSpec(
        name="plan_execute",
        description="Decompose into subtasks, execute each in sequence",
        max_iterations=30,
        decompose_tasks=True,
        max_subtasks=10,
        stop_on_no_progress=True,
        no_progress_iterations=2,
        system_prompt="""You are executing a plan. The task has been decomposed into subtasks.
Each iteration should:
1. Identify the current subtask
2. Execute that subtask completely
3. Mark it complete when done
4. Move to the next subtask

Stay focused on one subtask at a time. Don't jump ahead or revisit completed tasks.""",
        iteration_prompt_template="""
# Plan Execution - Iteration {iteration}
Overall Task: {task}

## Subtask Progress
{subtask_status}

## Current Subtask
{current_subtask}

## Instructions
Execute the current subtask. When complete:
1. Mark it done
2. Note any artifacts created
3. Identify blockers for remaining tasks

Do not start the next subtask - just complete this one.
""",
        handoff_prompt_template="""
# CONTEXT HANDOFF - Plan Execution

## Overall Task
{task}

## Plan Status
{subtask_status}

## Current Subtask
{current_subtask}

## Completed Subtasks
{completed_subtasks}

## Blockers
{blockers}

## Files Created
{files_created}

Continue executing the plan from the current subtask.
""",
        tags=["planning", "execution", "decomposition"],
    ),

    "verify_fix": TemplateSpec(
        name="verify_fix",
        description="Verify claims or code, fix issues found",
        max_iterations=15,
        min_confidence=0.8,
        quality_threshold=0.9,
        system_prompt="""You are verifying and fixing issues. Each iteration should:
1. Identify something to verify
2. Check if it's correct
3. If incorrect, fix it
4. Re-verify the fix

Be thorough in verification. False negatives (missing issues) are worse than false positives.""",
        iteration_prompt_template="""
# Verify/Fix Iteration {iteration}
Task: {task}

## Verified So Far
{verified_items}

## Issues Found & Fixed
{issues_fixed}

## Remaining to Verify
{remaining_items}

## Instructions
1. Pick the next item to verify
2. Check it thoroughly
3. If wrong, fix it
4. Verify the fix worked
5. Report status

Prioritize high-risk items first.
""",
        handoff_prompt_template="""
# CONTEXT HANDOFF - Verify/Fix

## Task
{task}

## Verification Status
- Verified: {verified_count}
- Issues found: {issues_count}
- Fixed: {fixed_count}

## Remaining Items
{remaining_items}

## Known Issues
{known_issues}

Continue verification from remaining items.
""",
        tags=["verification", "debugging", "quality"],
    ),

    "incremental_build": TemplateSpec(
        name="incremental_build",
        description="Build a feature incrementally, testing as you go",
        max_iterations=50,
        decompose_tasks=True,
        consolidate_per_iterations=3,
        system_prompt="""You are building incrementally. Each iteration should:
1. Implement a small, testable piece
2. Verify it works
3. Commit the change
4. Plan the next piece

Small, working increments are better than large, untested changes.""",
        iteration_prompt_template="""
# Build Iteration {iteration}
Feature: {task}

## Built So Far
{built_components}

## Current Component
{current_component}

## Tests Passing
{test_status}

## Instructions
1. Implement the next small piece
2. Add/update tests for it
3. Verify tests pass
4. Commit with descriptive message
5. Identify next piece

Keep changes small and testable.
""",
        handoff_prompt_template="""
# CONTEXT HANDOFF - Incremental Build

## Feature
{task}

## Components Built
{built_components}

## Current Component
{current_component}

## Test Status
{test_status}

## Files Modified
{files_modified}

## Git Commits
{git_commits}

Continue building from the current component.
""",
        tags=["building", "incremental", "testing"],
    ),

    "debug_fix": TemplateSpec(
        name="debug_fix",
        description="Debug an issue through systematic investigation",
        max_iterations=20,
        system_prompt="""You are debugging an issue. Each iteration should:
1. Form a hypothesis about the cause
2. Design an experiment to test it
3. Run the experiment
4. Analyze results
5. Either fix (if found) or form new hypothesis

Be systematic. Document hypotheses tested to avoid repetition.""",
        iteration_prompt_template="""
# Debug Iteration {iteration}
Issue: {task}

## Symptoms
{symptoms}

## Hypotheses Tested
{tested_hypotheses}

## Current Hypothesis
{current_hypothesis}

## Instructions
1. Design experiment to test current hypothesis
2. Run experiment
3. Analyze results
4. If cause found: fix it
5. If not: form new hypothesis

Document your reasoning clearly.
""",
        handoff_prompt_template="""
# CONTEXT HANDOFF - Debug/Fix

## Issue
{task}

## Symptoms
{symptoms}

## Hypotheses Tested
{tested_hypotheses}

## Current Hypothesis
{current_hypothesis}

## Root Cause (if found)
{root_cause}

## Fix Applied (if any)
{fix_applied}

Continue debugging from current hypothesis.
""",
        tags=["debugging", "investigation", "systematic"],
    ),
}


class TemplateRegistry:
    """
    Registry for loop templates.

    Allows registering custom templates and retrieving by name.
    """

    def __init__(self):
        self._templates: Dict[str, TemplateSpec] = BUILTIN_TEMPLATES.copy()

    def register(self, template: TemplateSpec):
        """Register a custom template."""
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[TemplateSpec]:
        """Get template by name."""
        return self._templates.get(name)

    def list_templates(self) -> List[Dict[str, str]]:
        """List all available templates."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._templates.values()
        ]

    def get_by_tag(self, tag: str) -> List[TemplateSpec]:
        """Get templates with a specific tag."""
        return [t for t in self._templates.values() if tag in t.tags]


# Global registry
_registry = TemplateRegistry()


def get_template(name: str) -> Optional[TemplateSpec]:
    """Get a template by name from the global registry."""
    # Check if it's a LoopTemplate enum
    if isinstance(name, LoopTemplate):
        name = name.value
    return _registry.get(name)


def register_template(template: TemplateSpec):
    """Register a custom template in the global registry."""
    _registry.register(template)


def list_templates() -> List[Dict[str, str]]:
    """List all available templates."""
    return _registry.list_templates()
