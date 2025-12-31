"""
Ritual REFLECT Phase Skill
==========================

Created: December 30, 2025
Purpose: End-of-session knowledge consolidation and preparation for next session

The REFLECT phase:
- Generates session summary from ritual context
- Extracts key learnings and patterns
- Identifies next steps and open questions
- Stores consolidated knowledge with high importance
- Prepares context for future AWAKEN phases

Follows "software for agents" paradigm - can be invoked by:
- Humans via /ritual-reflect command
- Agents via capability registry (KNOWLEDGE_CONSOLIDATION)
- Orchestrator as final ritual phase
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Import base class and types
from .base import (
    AbstractRitualPhase,
    PhaseResult,
    PhaseExecutionContext,
    AgentCapability,
)

# Import ritual types
from ..ritual_events import RitualPhase, RitualContext

logger = logging.getLogger(__name__)


# ============================================================================
# Session Summary Data Structure
# ============================================================================

@dataclass
class SessionSummary:
    """
    Consolidated session summary for REFLECT phase.

    Captures:
    - What was accomplished during the session
    - Key learnings and insights
    - Patterns discovered
    - Open questions and blockers
    - Recommended next steps
    - Decision trail from ritual phases

    Used for:
    - End-of-session review
    - High-importance memory storage
    - Future AWAKEN phase context restoration
    """

    # Session metadata
    session_id: str = ""
    feature_name: str = ""
    started_at: str = ""
    ended_at: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_minutes: float = 0.0

    # Accomplishments
    accomplishments: List[str] = field(default_factory=list)
    phases_completed: List[str] = field(default_factory=list)
    code_changes: List[Dict[str, Any]] = field(default_factory=list)

    # Learnings and insights
    key_learnings: List[str] = field(default_factory=list)
    patterns_discovered: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)

    # Open items
    open_questions: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    technical_debt: List[str] = field(default_factory=list)

    # Next steps
    next_steps: List[str] = field(default_factory=list)
    priority_for_next_session: str = ""

    # Decision trail
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics
    confidence: float = 0.0
    completeness: float = 0.0

    # Raw context for detailed storage
    raw_context: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> str:
        """
        Generate human-readable summary for display.

        Returns:
            Formatted markdown summary
        """
        lines = [
            f"## Session Summary: {self.feature_name or 'Coding Session'}",
            "",
            f"**Session ID**: {self.session_id}",
            f"**Duration**: {self.duration_minutes:.1f} minutes",
            f"**Phases Completed**: {', '.join(self.phases_completed) if self.phases_completed else 'None'}",
            "",
        ]

        # Accomplishments
        if self.accomplishments:
            lines.append("### Accomplishments")
            for acc in self.accomplishments:
                lines.append(f"- {acc}")
            lines.append("")

        # Key learnings
        if self.key_learnings:
            lines.append("### Key Learnings")
            for learning in self.key_learnings:
                lines.append(f"- {learning}")
            lines.append("")

        # Patterns discovered
        if self.patterns_discovered:
            lines.append("### Patterns Discovered")
            for pattern in self.patterns_discovered:
                lines.append(f"- {pattern}")
            lines.append("")

        # Insights
        if self.insights:
            lines.append("### Insights")
            for insight in self.insights:
                lines.append(f"- {insight}")
            lines.append("")

        # Open questions
        if self.open_questions:
            lines.append("### Open Questions")
            for question in self.open_questions:
                lines.append(f"- {question}")
            lines.append("")

        # Blockers
        if self.blockers:
            lines.append("### Blockers")
            for blocker in self.blockers:
                lines.append(f"- ⚠️ {blocker}")
            lines.append("")

        # Technical debt
        if self.technical_debt:
            lines.append("### Technical Debt Identified")
            for debt in self.technical_debt:
                lines.append(f"- {debt}")
            lines.append("")

        # Next steps
        if self.next_steps:
            lines.append("### Recommended Next Steps")
            for i, step in enumerate(self.next_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        # Priority
        if self.priority_for_next_session:
            lines.append("### Priority for Next Session")
            lines.append(f"**{self.priority_for_next_session}**")
            lines.append("")

        # Decisions made
        if self.decisions_made:
            lines.append("### Decisions Made")
            for decision in self.decisions_made:
                phase = decision.get("phase", "unknown")
                choice = decision.get("decision", "unknown")
                lines.append(f"- **{phase}**: {choice}")
            lines.append("")

        # Quality indicators
        lines.append("---")
        lines.append(f"*Confidence: {self.confidence:.0%} | Completeness: {self.completeness:.0%}*")

        return "\n".join(lines)

    def to_storage_content(self) -> str:
        """
        Generate content optimized for memory storage.

        Returns:
            Structured content for HoloLoom memory with high recall potential
        """
        sections = []

        # Header with searchable metadata
        sections.append(f"SESSION SUMMARY: {self.feature_name}")
        sections.append(f"Date: {self.ended_at[:10]}")
        sections.append(f"Duration: {self.duration_minutes:.0f} minutes")
        sections.append("")

        # Accomplishments (high importance)
        if self.accomplishments:
            sections.append("ACCOMPLISHED:")
            sections.extend([f"- {a}" for a in self.accomplishments])
            sections.append("")

        # Learnings (high importance for future recall)
        if self.key_learnings:
            sections.append("LEARNED:")
            sections.extend([f"- {l}" for l in self.key_learnings])
            sections.append("")

        # Patterns (valuable for pattern matching retrieval)
        if self.patterns_discovered:
            sections.append("PATTERNS:")
            sections.extend([f"- {p}" for p in self.patterns_discovered])
            sections.append("")

        # Next steps (important for next session AWAKEN)
        if self.next_steps:
            sections.append("NEXT STEPS:")
            sections.extend([f"{i}. {s}" for i, s in enumerate(self.next_steps, 1)])
            sections.append("")

        # Open questions (for continuity)
        if self.open_questions:
            sections.append("OPEN QUESTIONS:")
            sections.extend([f"- {q}" for q in self.open_questions])
            sections.append("")

        # Priority marker
        if self.priority_for_next_session:
            sections.append(f"PRIORITY: {self.priority_for_next_session}")

        return "\n".join(sections)

    def get_tags(self) -> List[str]:
        """
        Generate tags for memory storage.

        Returns:
            List of tags for searchability
        """
        tags = [
            "session_summary",
            "reflect",
            "ritual",
        ]

        if self.feature_name:
            # Add feature name as tag (normalized)
            normalized = self.feature_name.lower().replace(" ", "_")
            tags.append(f"feature:{normalized}")

        if self.phases_completed:
            tags.append(f"phases:{len(self.phases_completed)}")

        if self.blockers:
            tags.append("has_blockers")

        if self.confidence >= 0.8:
            tags.append("high_confidence")
        elif self.confidence < 0.5:
            tags.append("low_confidence")

        return tags


# ============================================================================
# REFLECT Phase Implementation
# ============================================================================

class ReflectPhase(AbstractRitualPhase):
    """
    REFLECT phase skill for end-of-session knowledge consolidation.

    Implements:
    - Session summary generation from ritual context
    - Key learnings extraction
    - Pattern discovery
    - High-importance memory storage
    - Next session preparation

    Decision Point Options:
    - "Confirm summary" (default) - Accept and store summary
    - "Add more notes" - Extend summary with additional context
    - "Quick close" - Minimal storage, fast exit

    Agent Capabilities:
    - KNOWLEDGE_CONSOLIDATION - Primary capability
    - MEMORY_STORAGE - Stores session summary
    """

    def __init__(
        self,
        storage_importance: float = 0.8,
        min_accomplishments: int = 1,
        extract_patterns: bool = True,
        extract_learnings: bool = True,
        include_decision_trail: bool = True,
    ):
        """
        Initialize REFLECT phase.

        Args:
            storage_importance: Importance score for stored summary (0.0-1.0)
            min_accomplishments: Minimum accomplishments for meaningful session
            extract_patterns: Whether to extract patterns from context
            extract_learnings: Whether to extract learnings from context
            include_decision_trail: Whether to include decision history
        """
        super().__init__(
            phase=RitualPhase.REFLECT,
            name="ritual-reflect",
            version="1.0.0",
            description="End-of-session knowledge consolidation and preparation for next session",
            capabilities=[
                AgentCapability.KNOWLEDGE_CONSOLIDATION,
                AgentCapability.MEMORY_STORAGE,
            ],
        )

        self._storage_importance = storage_importance
        self._min_accomplishments = min_accomplishments
        self._extract_patterns = extract_patterns
        self._extract_learnings = extract_learnings
        self._include_decision_trail = include_decision_trail

    # ========================================================================
    # Preflight Implementation
    # ========================================================================

    async def _phase_specific_preflight(
        self,
        context: PhaseExecutionContext
    ) -> Dict[str, Any]:
        """
        REFLECT-specific preflight checks.

        Validates:
        - Ritual context has meaningful content
        - Session has accomplishments to summarize
        - Memory system ready for storage

        Returns:
            Dict with checks, failures, warnings, memory_context
        """
        checks = []
        failures = []
        warnings = []
        memory_context = {}

        # Check 1: Ritual context exists
        if context.ritual_context:
            checks.append("ritual_context_present")

            # Extract summary data from context
            memory_context["session_id"] = context.ritual_context.ritual_id
            memory_context["feature_name"] = context.ritual_context.feature_name
            memory_context["started_at"] = context.ritual_context.started_at
            memory_context["decisions_made"] = context.ritual_context.decisions_made
            memory_context["phase_results"] = context.ritual_context.phase_results
        else:
            warnings.append("No ritual context - will generate minimal summary")

        # Check 2: Session has content
        if context.ritual_context:
            phase_count = len(context.ritual_context.phase_results)
            if phase_count > 0:
                checks.append("phases_completed")
                memory_context["phases_completed"] = list(
                    context.ritual_context.phase_results.keys()
                )
            else:
                warnings.append("No phases completed in this ritual")

        # Check 3: Handoff context from previous phase
        if context.handoff_context:
            checks.append("handoff_context_available")
            memory_context["handoff_context"] = context.handoff_context
            memory_context["handoff_tokens"] = context.handoff_tokens
        else:
            warnings.append("No handoff context from previous phase")

        # Check 4: Feature name available
        feature_name = context.feature_name or (
            context.ritual_context.feature_name if context.ritual_context else ""
        )
        if feature_name:
            checks.append("feature_name_available")
            memory_context["feature_name"] = feature_name
        else:
            warnings.append("No feature name - summary will be generic")

        return {
            "checks": checks,
            "failures": failures,
            "warnings": warnings,
            "memory_context": memory_context,
        }

    def _estimate_duration(self, context: PhaseExecutionContext) -> float:
        """Estimate REFLECT phase duration."""
        # Base time for summary generation
        base_ms = 3000.0

        # Add time for memory storage
        base_ms += 2000.0

        # Add time based on ritual complexity
        if context.ritual_context:
            phase_count = len(context.ritual_context.phase_results)
            base_ms += phase_count * 500.0

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
        Execute REFLECT phase.

        1. Extract accomplishments from ritual phases
        2. Extract learnings and patterns
        3. Identify open questions and next steps
        4. Generate session summary
        5. Present for confirmation (decision point)

        Returns:
            PhaseResult with session summary and decision options
        """
        logger.info("Executing REFLECT phase")

        # Get preflight memory context
        memory_context = parameters.get("_memory_context", {})

        # Initialize summary
        summary = SessionSummary(
            session_id=memory_context.get("session_id", "unknown"),
            feature_name=memory_context.get("feature_name", ""),
            started_at=memory_context.get("started_at", ""),
        )

        # Calculate duration
        if summary.started_at:
            try:
                start = datetime.fromisoformat(summary.started_at)
                now = datetime.now()
                summary.duration_minutes = (now - start).total_seconds() / 60.0
            except (ValueError, TypeError):
                summary.duration_minutes = 0.0

        # Extract phases completed
        summary.phases_completed = memory_context.get("phases_completed", [])

        # Extract accomplishments from phase results
        if context.ritual_context:
            summary.accomplishments = self._extract_accomplishments(
                context.ritual_context
            )

        # Extract learnings if enabled
        if self._extract_learnings:
            summary.key_learnings = self._extract_key_learnings(
                context.ritual_context,
                parameters
            )

        # Extract patterns if enabled
        if self._extract_patterns:
            summary.patterns_discovered = self._extract_patterns_discovered(
                context.ritual_context,
                parameters
            )

        # Extract open questions and blockers
        summary.open_questions = self._extract_open_questions(
            context.ritual_context,
            parameters
        )
        summary.blockers = self._extract_blockers(
            context.ritual_context,
            parameters
        )

        # Generate next steps
        summary.next_steps = self._generate_next_steps(
            summary,
            context.ritual_context,
            parameters
        )

        # Determine priority for next session
        summary.priority_for_next_session = self._determine_priority(summary)

        # Include decision trail if enabled
        if self._include_decision_trail:
            summary.decisions_made = memory_context.get("decisions_made", [])

        # Calculate quality metrics
        summary.confidence = self._calculate_confidence(summary)
        summary.completeness = self._calculate_completeness(summary)

        # Store raw context for detailed analysis
        summary.raw_context = {
            "parameters": parameters,
            "handoff_context": parameters.get("_handoff_context", ""),
            "handoff_tokens": parameters.get("_handoff_tokens", 0),
        }

        # Determine suggested decision based on summary quality
        if summary.confidence >= 0.7 and len(summary.accomplishments) >= self._min_accomplishments:
            suggested = "Confirm summary"
            reasoning = f"Session summary is complete with {len(summary.accomplishments)} accomplishments and {summary.confidence:.0%} confidence."
        elif summary.confidence < 0.5:
            suggested = "Add more notes"
            reasoning = f"Summary confidence is low ({summary.confidence:.0%}). Consider adding more context."
        else:
            suggested = "Confirm summary"
            reasoning = "Summary captures session context adequately."

        return PhaseResult(
            success=True,
            output=summary,
            confidence=summary.confidence,
            memories_stored=[],  # Will be populated after decision
            memories_retrieved=[],
            decision_needed=True,
            decision_options=self.get_decision_options(),
            suggested_decision=suggested,
            decision_reasoning=reasoning,
            metadata={
                "accomplishments_count": len(summary.accomplishments),
                "learnings_count": len(summary.key_learnings),
                "patterns_count": len(summary.patterns_discovered),
                "next_steps_count": len(summary.next_steps),
                "open_questions_count": len(summary.open_questions),
                "blockers_count": len(summary.blockers),
                "duration_minutes": summary.duration_minutes,
            }
        )

    # ========================================================================
    # Decision Options
    # ========================================================================

    def get_decision_options(self) -> List[str]:
        """
        Get REFLECT-specific decision options.

        Returns:
            List of available decisions
        """
        return [
            "Confirm summary",
            "Add more notes",
            "Quick close",
        ]

    # ========================================================================
    # Summary Storage
    # ========================================================================

    async def store_summary(
        self,
        summary: SessionSummary,
        context: PhaseExecutionContext,
        decision: str = "Confirm summary"
    ) -> Optional[str]:
        """
        Store session summary to memory.

        Called after decision confirmation.

        Args:
            summary: The session summary to store
            context: Execution context with memory access
            decision: The user's decision

        Returns:
            Memory ID if stored, None otherwise
        """
        if decision == "Quick close":
            # Minimal storage for quick close
            content = f"Quick session close: {summary.feature_name or 'coding session'}"
            importance = 0.3
            tags = ["session_close", "quick"]
        else:
            # Full storage
            content = summary.to_storage_content()
            importance = self._storage_importance
            tags = summary.get_tags()

        # Store to memory
        memory_id = await self.store_memory(
            content=content,
            context=context,
            tags=tags,
            importance=importance
        )

        if memory_id:
            logger.info(f"Stored session summary with ID: {memory_id}")
        else:
            logger.warning("Failed to store session summary")

        return memory_id

    # ========================================================================
    # Extraction Helpers
    # ========================================================================

    def _extract_accomplishments(
        self,
        ritual_context: Optional[RitualContext]
    ) -> List[str]:
        """
        Extract accomplishments from ritual phase results.

        Returns:
            List of accomplishment strings
        """
        accomplishments = []

        if not ritual_context:
            return accomplishments

        # Extract from phase results
        for phase_name, result in ritual_context.phase_results.items():
            if isinstance(result, dict):
                # Phase completed successfully
                if result.get("success", False):
                    phase_output = result.get("output", {})

                    if phase_name == "awaken":
                        accomplishments.append("Restored session context from memory")
                    elif phase_name == "plan":
                        plan = phase_output.get("implementation_plan", [])
                        if plan:
                            accomplishments.append(f"Created implementation plan with {len(plan)} steps")
                    elif phase_name == "review":
                        quality = phase_output.get("quality_score", 0)
                        accomplishments.append(f"Completed code review (quality: {quality:.0%})")

        # Check for feature name as accomplishment
        if ritual_context.feature_name:
            accomplishments.append(f"Worked on: {ritual_context.feature_name}")

        # Check memories stored
        if ritual_context.memories_stored:
            accomplishments.append(f"Stored {len(ritual_context.memories_stored)} memories")

        return accomplishments

    def _extract_key_learnings(
        self,
        ritual_context: Optional[RitualContext],
        parameters: Dict[str, Any]
    ) -> List[str]:
        """
        Extract key learnings from session.

        Returns:
            List of learning strings
        """
        learnings = []

        # Extract from handoff context (contains accumulated insights)
        handoff = parameters.get("_handoff_context", "")
        if handoff and "learn" in handoff.lower():
            # Simple extraction - in production would use NLP
            lines = handoff.split("\n")
            for line in lines:
                if "learn" in line.lower() or "discover" in line.lower():
                    learnings.append(line.strip())

        # Extract from phase results
        if ritual_context:
            for phase_name, result in ritual_context.phase_results.items():
                if isinstance(result, dict):
                    output = result.get("output", {})
                    if isinstance(output, dict):
                        phase_learnings = output.get("learnings", [])
                        learnings.extend(phase_learnings)

        # Deduplicate
        return list(dict.fromkeys(learnings))[:10]  # Limit to 10

    def _extract_patterns_discovered(
        self,
        ritual_context: Optional[RitualContext],
        parameters: Dict[str, Any]
    ) -> List[str]:
        """
        Extract patterns discovered during session.

        Returns:
            List of pattern strings
        """
        patterns = []

        # Extract from memory context
        memory_context = parameters.get("_memory_context", {})
        phase_results = memory_context.get("phase_results", {})

        # Look for patterns in phase outputs
        for phase_name, result in phase_results.items():
            if isinstance(result, dict):
                output = result.get("output", {})
                if isinstance(output, dict):
                    phase_patterns = output.get("patterns", [])
                    patterns.extend(phase_patterns)

        # Extract from handoff context
        handoff = parameters.get("_handoff_context", "")
        if "pattern" in handoff.lower():
            # Simple extraction
            lines = handoff.split("\n")
            for line in lines:
                if "pattern" in line.lower():
                    patterns.append(line.strip())

        # Deduplicate
        return list(dict.fromkeys(patterns))[:5]  # Limit to 5

    def _extract_open_questions(
        self,
        ritual_context: Optional[RitualContext],
        parameters: Dict[str, Any]
    ) -> List[str]:
        """
        Extract open questions from session.

        Returns:
            List of question strings
        """
        questions = []

        # Extract from handoff context
        handoff = parameters.get("_handoff_context", "")
        if handoff:
            lines = handoff.split("\n")
            for line in lines:
                line = line.strip()
                if line.endswith("?"):
                    questions.append(line)

        # Extract from phase results
        if ritual_context:
            for phase_name, result in ritual_context.phase_results.items():
                if isinstance(result, dict):
                    output = result.get("output", {})
                    if isinstance(output, dict):
                        phase_questions = output.get("open_questions", [])
                        questions.extend(phase_questions)

        # Deduplicate
        return list(dict.fromkeys(questions))[:5]

    def _extract_blockers(
        self,
        ritual_context: Optional[RitualContext],
        parameters: Dict[str, Any]
    ) -> List[str]:
        """
        Extract blockers encountered during session.

        Returns:
            List of blocker strings
        """
        blockers = []

        # Check phase results for failures
        if ritual_context:
            for phase_name, result in ritual_context.phase_results.items():
                if isinstance(result, dict):
                    if not result.get("success", True):
                        errors = result.get("errors", [])
                        blockers.extend(errors)

                    # Check for explicit blockers
                    output = result.get("output", {})
                    if isinstance(output, dict):
                        phase_blockers = output.get("blockers", [])
                        blockers.extend(phase_blockers)

        # Deduplicate
        return list(dict.fromkeys(blockers))

    def _generate_next_steps(
        self,
        summary: SessionSummary,
        ritual_context: Optional[RitualContext],
        parameters: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommended next steps.

        Returns:
            List of next step strings
        """
        next_steps = []

        # Address blockers first
        for blocker in summary.blockers[:2]:
            next_steps.append(f"Resolve blocker: {blocker}")

        # Answer open questions
        for question in summary.open_questions[:2]:
            next_steps.append(f"Investigate: {question}")

        # Extract from phase results
        if ritual_context:
            for phase_name, result in ritual_context.phase_results.items():
                if isinstance(result, dict):
                    output = result.get("output", {})
                    if isinstance(output, dict):
                        phase_next = output.get("next_steps", [])
                        next_steps.extend(phase_next)

        # Default if empty
        if not next_steps and summary.feature_name:
            next_steps.append(f"Continue work on {summary.feature_name}")

        # Deduplicate and limit
        return list(dict.fromkeys(next_steps))[:5]

    def _determine_priority(self, summary: SessionSummary) -> str:
        """
        Determine priority for next session.

        Returns:
            Priority string
        """
        # Blockers are highest priority
        if summary.blockers:
            return f"BLOCKER: {summary.blockers[0]}"

        # Open questions next
        if summary.open_questions:
            return f"INVESTIGATE: {summary.open_questions[0]}"

        # Next steps
        if summary.next_steps:
            return summary.next_steps[0]

        # Feature continuation
        if summary.feature_name:
            return f"Continue: {summary.feature_name}"

        return "Review previous session context"

    def _calculate_confidence(self, summary: SessionSummary) -> float:
        """
        Calculate confidence in summary quality.

        Returns:
            Confidence score 0.0-1.0
        """
        score = 0.0

        # Has accomplishments
        if summary.accomplishments:
            score += 0.3

        # Has learnings
        if summary.key_learnings:
            score += 0.2

        # Has next steps
        if summary.next_steps:
            score += 0.2

        # Has feature name
        if summary.feature_name:
            score += 0.1

        # Has phases completed
        if summary.phases_completed:
            score += 0.1

        # Has duration
        if summary.duration_minutes > 0:
            score += 0.1

        return min(score, 1.0)

    def _calculate_completeness(self, summary: SessionSummary) -> float:
        """
        Calculate completeness of summary.

        Returns:
            Completeness score 0.0-1.0
        """
        fields_present = 0
        total_fields = 7

        if summary.accomplishments:
            fields_present += 1
        if summary.key_learnings:
            fields_present += 1
        if summary.patterns_discovered:
            fields_present += 1
        if summary.next_steps:
            fields_present += 1
        if summary.feature_name:
            fields_present += 1
        if summary.phases_completed:
            fields_present += 1
        if summary.duration_minutes > 0:
            fields_present += 1

        return fields_present / total_fields


# ============================================================================
# Factory Function
# ============================================================================

def create_reflect_phase(
    storage_importance: float = 0.8,
    min_accomplishments: int = 1,
    extract_patterns: bool = True,
    extract_learnings: bool = True,
    include_decision_trail: bool = True,
) -> ReflectPhase:
    """
    Factory function to create REFLECT phase skill.

    Args:
        storage_importance: Importance score for stored summary (0.0-1.0)
        min_accomplishments: Minimum accomplishments for meaningful session
        extract_patterns: Whether to extract patterns from context
        extract_learnings: Whether to extract learnings from context
        include_decision_trail: Whether to include decision history

    Returns:
        Configured ReflectPhase instance

    Example:
        >>> reflect = create_reflect_phase(storage_importance=0.9)
        >>> result = await reflect.execute(params, context)
        >>> if result.decision_needed:
        ...     # Present options to user/agent
        ...     decision = await get_decision(result.decision_options)
        ...     await reflect.store_summary(result.output, context, decision)
    """
    return ReflectPhase(
        storage_importance=storage_importance,
        min_accomplishments=min_accomplishments,
        extract_patterns=extract_patterns,
        extract_learnings=extract_learnings,
        include_decision_trail=include_decision_trail,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'SessionSummary',
    'ReflectPhase',
    'create_reflect_phase',
]