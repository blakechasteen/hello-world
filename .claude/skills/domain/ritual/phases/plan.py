"""
Ritual PLAN Phase
=================

Created: December 30, 2025
Purpose: Captures requirements and creates implementation plan

The PLAN phase:
1. Ingests requirements into HoloLoom memory
2. Uses MRF to refine the implementation prompt (optional)
3. Finds similar implementations via loom recall
4. Creates structured implementation plan

Decision Point:
- "Approve plan" - Ready to implement
- "Refine plan" - Need more detail/clarity
- "Research more" - Need additional context
- "Skip to IMPLEMENT" - Already know what to do
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base import (
    AbstractRitualPhase,
    AgentCapability,
    PhaseResult,
    PhaseExecutionContext,
)
from ..ritual_events import RitualPhase, PHASE_DECISION_OPTIONS

logger = logging.getLogger(__name__)


# ============================================================================
# Plan-Specific Data Structures
# ============================================================================

@dataclass
class ImplementationStep:
    """Single step in the implementation plan."""
    step_number: int
    description: str
    estimated_complexity: str = "medium"  # low, medium, high
    dependencies: List[str] = field(default_factory=list)
    files_affected: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SimilarPattern:
    """A similar implementation found in the codebase/memory."""
    memory_id: str
    description: str
    relevance_score: float
    file_path: Optional[str] = None
    code_snippet: Optional[str] = None
    key_learnings: List[str] = field(default_factory=list)


@dataclass
class ImplementationPlan:
    """
    Complete implementation plan output from PLAN phase.

    Contains refined requirements, similar patterns found,
    step-by-step implementation steps, and key decisions.
    """
    # Feature identification
    feature_name: str
    feature_description: str

    # Refined requirements (MRF-enhanced if available)
    original_requirements: str
    refined_requirements: str
    requirements_memory_id: Optional[str] = None

    # Similar patterns found
    similar_patterns: List[SimilarPattern] = field(default_factory=list)

    # Implementation steps
    steps: List[ImplementationStep] = field(default_factory=list)

    # Key decisions to make
    key_decisions: List[str] = field(default_factory=list)

    # Architecture considerations
    architecture_notes: List[str] = field(default_factory=list)

    # Risk assessment
    risks: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)

    # Metadata
    plan_created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence_score: float = 0.0

    def to_markdown(self) -> str:
        """Convert plan to readable markdown format."""
        lines = [
            f"# Implementation Plan: {self.feature_name}",
            "",
            f"**Created**: {self.plan_created_at}",
            f"**Confidence**: {self.confidence_score:.0%}",
            "",
            "## Description",
            self.feature_description,
            "",
            "## Requirements",
            "",
            "### Original",
            self.original_requirements,
            "",
            "### Refined",
            self.refined_requirements,
            "",
        ]

        if self.similar_patterns:
            lines.extend([
                "## Similar Patterns Found",
                "",
            ])
            for pattern in self.similar_patterns:
                lines.append(f"- **{pattern.description}** (relevance: {pattern.relevance_score:.0%})")
                if pattern.key_learnings:
                    for learning in pattern.key_learnings:
                        lines.append(f"  - {learning}")
            lines.append("")

        if self.steps:
            lines.extend([
                "## Implementation Steps",
                "",
            ])
            for step in self.steps:
                lines.append(f"### Step {step.step_number}: {step.description}")
                lines.append(f"- **Complexity**: {step.estimated_complexity}")
                if step.dependencies:
                    lines.append(f"- **Dependencies**: {', '.join(step.dependencies)}")
                if step.files_affected:
                    lines.append(f"- **Files**: {', '.join(step.files_affected)}")
                if step.acceptance_criteria:
                    lines.append("- **Acceptance Criteria**:")
                    for criteria in step.acceptance_criteria:
                        lines.append(f"  - {criteria}")
                if step.notes:
                    lines.append(f"- **Notes**: {step.notes}")
                lines.append("")

        if self.key_decisions:
            lines.extend([
                "## Key Decisions Needed",
                "",
            ])
            for i, decision in enumerate(self.key_decisions, 1):
                lines.append(f"{i}. {decision}")
            lines.append("")

        if self.architecture_notes:
            lines.extend([
                "## Architecture Notes",
                "",
            ])
            for note in self.architecture_notes:
                lines.append(f"- {note}")
            lines.append("")

        if self.risks:
            lines.extend([
                "## Risks & Mitigations",
                "",
            ])
            for i, (risk, mitigation) in enumerate(zip(self.risks, self.mitigations), 1):
                lines.append(f"{i}. **Risk**: {risk}")
                lines.append(f"   **Mitigation**: {mitigation}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "feature_description": self.feature_description,
            "original_requirements": self.original_requirements,
            "refined_requirements": self.refined_requirements,
            "requirements_memory_id": self.requirements_memory_id,
            "similar_patterns": [
                {
                    "memory_id": p.memory_id,
                    "description": p.description,
                    "relevance_score": p.relevance_score,
                    "file_path": p.file_path,
                    "key_learnings": p.key_learnings,
                }
                for p in self.similar_patterns
            ],
            "steps": [
                {
                    "step_number": s.step_number,
                    "description": s.description,
                    "estimated_complexity": s.estimated_complexity,
                    "dependencies": s.dependencies,
                    "files_affected": s.files_affected,
                    "acceptance_criteria": s.acceptance_criteria,
                    "notes": s.notes,
                }
                for s in self.steps
            ],
            "key_decisions": self.key_decisions,
            "architecture_notes": self.architecture_notes,
            "risks": self.risks,
            "mitigations": self.mitigations,
            "plan_created_at": self.plan_created_at,
            "confidence_score": self.confidence_score,
        }


# ============================================================================
# PLAN Phase Implementation
# ============================================================================

class PlanPhase(AbstractRitualPhase):
    """
    PLAN Phase: Capture requirements and create implementation plan.

    This phase:
    1. Ingests requirements into HoloLoom memory
    2. Searches for similar implementations/patterns
    3. Creates structured implementation plan
    4. Presents decision point for approval

    Capabilities:
    - PLANNING: Core capability for this phase
    - MEMORY_RETRIEVAL: Find similar patterns
    - MEMORY_STORAGE: Store requirements and plan
    """

    def __init__(self):
        super().__init__(
            phase=RitualPhase.PLAN,
            name="ritual-plan",
            version="1.0.0",
            description="Capture requirements and create implementation plan",
            capabilities=[
                AgentCapability.PLANNING,
                AgentCapability.MEMORY_RETRIEVAL,
                AgentCapability.MEMORY_STORAGE,
            ],
        )

        # Default decision options for PLAN phase
        self._decision_options = [
            "Approve plan",
            "Refine plan",
            "Research more",
            "Skip to IMPLEMENT",
        ]

    # ========================================================================
    # Preflight Implementation
    # ========================================================================

    async def _phase_specific_preflight(
        self,
        context: PhaseExecutionContext
    ) -> Dict[str, Any]:
        """
        PLAN phase preflight checks.

        Checks:
        - Feature name is specified
        - Requirements are provided (or can be gathered)
        - Previous AWAKEN phase completed (if in full ritual)

        Memory Context:
        - Recalls any existing plans for this feature
        - Finds related architectural decisions
        """
        checks = []
        failures = []
        warnings = []
        memory_context = {}

        # Check 1: Feature name specified
        feature_name = context.feature_name or context.ritual_context.feature_name
        if feature_name:
            checks.append("feature_name_specified")
            memory_context["feature_name"] = feature_name
        else:
            warnings.append("No feature name specified - will use generic planning")
            memory_context["feature_name"] = "unnamed_feature"

        # Check 2: Previous AWAKEN phase (if in full ritual)
        phase_results = context.ritual_context.phase_results
        if "awaken" in phase_results:
            checks.append("awaken_completed")
            awaken_result = phase_results["awaken"]
            memory_context["awaken_context"] = awaken_result.get("output", {})
        elif context.handoff_context:
            checks.append("handoff_context_available")
            memory_context["handoff_context"] = context.handoff_context
        else:
            warnings.append("No AWAKEN context - starting fresh")

        # Check 3: Look for existing plans in memory
        if feature_name:
            try:
                existing_plans = await self.recall_memories(
                    query=f"implementation plan for {feature_name}",
                    context=context,
                    strategy="semantic",
                    limit=3
                )
                if existing_plans:
                    memory_context["existing_plans"] = existing_plans
                    checks.append("checked_existing_plans")
                    warnings.append(f"Found {len(existing_plans)} existing plan(s) for similar features")
            except Exception as e:
                logger.warning(f"Could not check existing plans: {e}")

        # Check 4: Find related architecture decisions
        try:
            arch_decisions = await self.recall_memories(
                query="architecture decisions patterns conventions",
                context=context,
                strategy="semantic",
                limit=5
            )
            if arch_decisions:
                memory_context["architecture_context"] = arch_decisions
                checks.append("architecture_context_retrieved")
        except Exception as e:
            logger.warning(f"Could not retrieve architecture context: {e}")

        return {
            "checks": checks,
            "failures": failures,
            "warnings": warnings,
            "memory_context": memory_context,
        }

    def _estimate_duration(self, context: PhaseExecutionContext) -> float:
        """Estimate PLAN phase duration based on complexity."""
        # Base duration: 10 seconds
        base_ms = 10000.0

        # Add time if requirements are complex
        requirements = context.ritual_context.metadata.get("requirements", "")
        if len(requirements) > 1000:
            base_ms += 5000.0  # Complex requirements

        # Add time if searching for patterns
        if context.preflight_result and context.preflight_result.memory_context:
            base_ms += 3000.0  # Memory search

        return base_ms

    # ========================================================================
    # Execution Implementation
    # ========================================================================

    async def _phase_specific_execute(
        self,
        parameters: Dict[str, Any],
        context: PhaseExecutionContext
    ) -> PhaseResult:
        """
        Execute PLAN phase.

        Steps:
        1. Parse/validate requirements
        2. Ingest requirements into memory
        3. Search for similar patterns
        4. Generate implementation plan
        5. Present decision point
        """
        memories_stored = []
        memories_retrieved = []
        errors = []
        warnings = []

        # Extract parameters
        feature_name = parameters.get("_feature_name", "unnamed_feature")
        requirements = parameters.get("requirements", "")
        handoff_context = parameters.get("_handoff_context", "")
        memory_context = parameters.get("_memory_context", {})

        # If no requirements provided, check handoff context
        if not requirements and handoff_context:
            requirements = f"Based on context: {handoff_context}"

        # Step 1: Refine requirements (simplified MRF-like refinement)
        refined_requirements = await self._refine_requirements(
            original=requirements,
            feature_name=feature_name,
            context=context
        )

        # Step 2: Store requirements in memory
        try:
            req_memory_id = await self.store_memory(
                content=f"Requirements for {feature_name}: {refined_requirements}",
                context=context,
                tags=["requirements", "plan", feature_name],
                importance=0.8
            )
            if req_memory_id:
                memories_stored.append(req_memory_id)
        except Exception as e:
            warnings.append(f"Could not store requirements: {e}")
            req_memory_id = None

        # Step 3: Search for similar patterns
        similar_patterns = await self._find_similar_patterns(
            feature_name=feature_name,
            requirements=refined_requirements,
            context=context
        )
        for pattern in similar_patterns:
            memories_retrieved.append(pattern.memory_id)

        # Step 4: Generate implementation steps
        steps = await self._generate_implementation_steps(
            feature_name=feature_name,
            requirements=refined_requirements,
            similar_patterns=similar_patterns,
            context=context
        )

        # Step 5: Identify key decisions and risks
        key_decisions = self._identify_key_decisions(
            requirements=refined_requirements,
            similar_patterns=similar_patterns
        )

        risks, mitigations = self._assess_risks(
            requirements=refined_requirements,
            steps=steps
        )

        # Step 6: Calculate confidence
        confidence = self._calculate_plan_confidence(
            requirements=refined_requirements,
            similar_patterns=similar_patterns,
            steps=steps
        )

        # Create the implementation plan
        plan = ImplementationPlan(
            feature_name=feature_name,
            feature_description=self._extract_description(requirements),
            original_requirements=requirements,
            refined_requirements=refined_requirements,
            requirements_memory_id=req_memory_id,
            similar_patterns=similar_patterns,
            steps=steps,
            key_decisions=key_decisions,
            risks=risks,
            mitigations=mitigations,
            confidence_score=confidence,
        )

        # Step 7: Store the plan in memory
        try:
            plan_memory_id = await self.store_memory(
                content=plan.to_markdown(),
                context=context,
                tags=["implementation_plan", feature_name],
                importance=0.9
            )
            if plan_memory_id:
                memories_stored.append(plan_memory_id)
        except Exception as e:
            warnings.append(f"Could not store plan: {e}")

        # Step 8: Determine suggested decision
        suggested_decision, decision_reasoning = self._suggest_decision(
            plan=plan,
            context=context
        )

        return PhaseResult(
            success=True,
            output={
                "plan": plan.to_dict(),
                "plan_markdown": plan.to_markdown(),
                "feature_name": feature_name,
                "step_count": len(steps),
                "patterns_found": len(similar_patterns),
            },
            confidence=confidence,
            memories_stored=memories_stored,
            memories_retrieved=memories_retrieved,
            decision_needed=True,
            decision_options=self._decision_options,
            suggested_decision=suggested_decision,
            decision_reasoning=decision_reasoning,
            errors=errors,
            warnings=warnings,
            metadata={
                "plan_id": plan_memory_id if 'plan_memory_id' in locals() else None,
                "requirements_id": req_memory_id,
            }
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _refine_requirements(
        self,
        original: str,
        feature_name: str,
        context: PhaseExecutionContext
    ) -> str:
        """
        Refine requirements using MRF-like enhancement.

        In production, this would integrate with the MRF system.
        For now, applies structured formatting.
        """
        if not original:
            return f"Implement {feature_name}"

        # Simple refinement: structure and clarify
        refined_lines = [
            f"Feature: {feature_name}",
            "",
            "Objectives:",
        ]

        # Parse objectives from requirements
        sentences = original.replace("\n", ". ").split(". ")
        for i, sentence in enumerate(sentences[:5], 1):  # Max 5 objectives
            sentence = sentence.strip()
            if sentence:
                refined_lines.append(f"  {i}. {sentence}")

        refined_lines.extend([
            "",
            "Success Criteria:",
            "  - Feature works as described",
            "  - Tests pass",
            "  - Code follows project conventions",
        ])

        return "\n".join(refined_lines)

    async def _find_similar_patterns(
        self,
        feature_name: str,
        requirements: str,
        context: PhaseExecutionContext
    ) -> List[SimilarPattern]:
        """Find similar implementations in memory."""
        patterns = []

        # Search 1: By feature name
        try:
            name_results = await self.recall_memories(
                query=f"implementation {feature_name}",
                context=context,
                strategy="semantic",
                limit=3
            )
            for result in name_results:
                patterns.append(SimilarPattern(
                    memory_id=result.get("id", "unknown"),
                    description=result.get("content", "")[:100],
                    relevance_score=result.get("score", 0.5),
                    key_learnings=result.get("learnings", []),
                ))
        except Exception as e:
            logger.warning(f"Pattern search by name failed: {e}")

        # Search 2: By requirements content
        try:
            req_results = await self.recall_memories(
                query=requirements[:200],  # First 200 chars
                context=context,
                strategy="fused",
                limit=3
            )
            for result in req_results:
                # Avoid duplicates
                if not any(p.memory_id == result.get("id") for p in patterns):
                    patterns.append(SimilarPattern(
                        memory_id=result.get("id", "unknown"),
                        description=result.get("content", "")[:100],
                        relevance_score=result.get("score", 0.5) * 0.9,  # Slight discount
                        key_learnings=result.get("learnings", []),
                    ))
        except Exception as e:
            logger.warning(f"Pattern search by requirements failed: {e}")

        # Sort by relevance
        patterns.sort(key=lambda p: p.relevance_score, reverse=True)
        return patterns[:5]  # Return top 5

    async def _generate_implementation_steps(
        self,
        feature_name: str,
        requirements: str,
        similar_patterns: List[SimilarPattern],
        context: PhaseExecutionContext
    ) -> List[ImplementationStep]:
        """Generate implementation steps from requirements."""
        steps = []

        # Step 1: Setup/preparation
        steps.append(ImplementationStep(
            step_number=1,
            description=f"Set up foundation for {feature_name}",
            estimated_complexity="low",
            acceptance_criteria=[
                "Project structure ready",
                "Dependencies identified",
            ],
        ))

        # Step 2-N: Parse from requirements
        objectives = self._extract_objectives(requirements)
        for i, objective in enumerate(objectives, 2):
            complexity = "medium"
            if any(word in objective.lower() for word in ["complex", "advanced", "integrate"]):
                complexity = "high"
            elif any(word in objective.lower() for word in ["simple", "basic", "add"]):
                complexity = "low"

            steps.append(ImplementationStep(
                step_number=i,
                description=objective,
                estimated_complexity=complexity,
                dependencies=[f"Step {i-1}"] if i > 2 else [],
                acceptance_criteria=[f"{objective} is complete"],
            ))

        # Final step: Testing & verification
        steps.append(ImplementationStep(
            step_number=len(steps) + 1,
            description="Test and verify implementation",
            estimated_complexity="medium",
            dependencies=[f"Step {len(steps)}"],
            acceptance_criteria=[
                "All tests pass",
                "Feature works as expected",
                "Code reviewed",
            ],
        ))

        return steps

    def _extract_objectives(self, requirements: str) -> List[str]:
        """Extract individual objectives from requirements."""
        objectives = []

        # Look for numbered items
        lines = requirements.split("\n")
        for line in lines:
            line = line.strip()
            # Match patterns like "1. ...", "- ...", "• ..."
            if line and (
                (len(line) > 2 and line[0].isdigit() and line[1] in ".)") or
                line.startswith("- ") or
                line.startswith("• ")
            ):
                # Remove prefix
                if line[0].isdigit():
                    objective = line[2:].strip()
                else:
                    objective = line[2:].strip()
                if objective:
                    objectives.append(objective)

        # If no structured items found, split by sentences
        if not objectives:
            sentences = requirements.replace("\n", " ").split(". ")
            objectives = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10][:5]

        return objectives[:7]  # Max 7 objectives

    def _extract_description(self, requirements: str) -> str:
        """Extract a brief description from requirements."""
        # Take first sentence or first 100 chars
        first_line = requirements.split("\n")[0].strip()
        if len(first_line) > 100:
            return first_line[:97] + "..."
        return first_line or "No description provided"

    def _identify_key_decisions(
        self,
        requirements: str,
        similar_patterns: List[SimilarPattern]
    ) -> List[str]:
        """Identify key decisions that need to be made."""
        decisions = []

        # Check for ambiguous terms
        ambiguous_terms = ["or", "either", "option", "choice", "decide", "should"]
        for term in ambiguous_terms:
            if term in requirements.lower():
                decisions.append(f"Clarify decision point related to '{term}' in requirements")
                break

        # Check for architecture choices
        arch_terms = ["database", "api", "framework", "library", "pattern"]
        for term in arch_terms:
            if term in requirements.lower():
                decisions.append(f"Select appropriate {term} approach")

        # Add common decisions if few found
        if len(decisions) < 2:
            decisions.append("Confirm implementation approach aligns with project conventions")

        return decisions[:5]  # Max 5 decisions

    def _assess_risks(
        self,
        requirements: str,
        steps: List[ImplementationStep]
    ) -> tuple:
        """Assess risks and generate mitigations."""
        risks = []
        mitigations = []

        # Risk 1: Complex steps
        high_complexity_count = sum(1 for s in steps if s.estimated_complexity == "high")
        if high_complexity_count > 0:
            risks.append(f"{high_complexity_count} high-complexity step(s) may cause delays")
            mitigations.append("Break down complex steps further before implementation")

        # Risk 2: Many steps
        if len(steps) > 5:
            risks.append("Multiple steps increase coordination complexity")
            mitigations.append("Consider batching related steps together")

        # Risk 3: Unclear requirements
        if len(requirements) < 50:
            risks.append("Requirements may be underspecified")
            mitigations.append("Gather more details before starting implementation")

        # Default risk if none found
        if not risks:
            risks.append("Standard implementation risks apply")
            mitigations.append("Follow best practices and test thoroughly")

        return risks, mitigations

    def _calculate_plan_confidence(
        self,
        requirements: str,
        similar_patterns: List[SimilarPattern],
        steps: List[ImplementationStep]
    ) -> float:
        """Calculate confidence score for the plan."""
        confidence = 0.5  # Base confidence

        # Boost for clear requirements
        if len(requirements) > 100:
            confidence += 0.1
        if len(requirements) > 300:
            confidence += 0.1

        # Boost for similar patterns found
        if similar_patterns:
            avg_relevance = sum(p.relevance_score for p in similar_patterns) / len(similar_patterns)
            confidence += avg_relevance * 0.2

        # Slight boost for reasonable step count
        if 3 <= len(steps) <= 7:
            confidence += 0.1

        return min(confidence, 0.95)  # Cap at 95%

    def _suggest_decision(
        self,
        plan: ImplementationPlan,
        context: PhaseExecutionContext
    ) -> tuple:
        """Suggest a decision based on plan quality."""
        confidence = plan.confidence_score

        if confidence >= 0.7:
            return (
                "Approve plan",
                f"Plan has {confidence:.0%} confidence with {len(plan.steps)} clear steps"
            )
        elif confidence >= 0.5:
            if plan.key_decisions:
                return (
                    "Refine plan",
                    f"Plan needs refinement - {len(plan.key_decisions)} decision(s) pending"
                )
            else:
                return (
                    "Approve plan",
                    f"Plan is ready at {confidence:.0%} confidence"
                )
        else:
            return (
                "Research more",
                f"Low confidence ({confidence:.0%}) - need more context"
            )

    # ========================================================================
    # Decision Options Override
    # ========================================================================

    def get_decision_options(self) -> List[str]:
        """Get available decision options for PLAN phase."""
        return self._decision_options


# ============================================================================
# Factory Function
# ============================================================================

def create_plan_phase() -> PlanPhase:
    """Factory function to create a PLAN phase instance."""
    return PlanPhase()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'ImplementationStep',
    'SimilarPattern',
    'ImplementationPlan',
    'PlanPhase',
    'create_plan_phase',
]