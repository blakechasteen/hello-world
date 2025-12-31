"""
AWAKEN Phase Skill
==================

Created: December 30, 2025
Purpose: Context restoration at session start

The AWAKEN phase is the first step in the coding ritual. It:
- Restores context from previous sessions
- Queries HoloLoom memory for recent work (temporal recall)
- Finds feature-specific knowledge (semantic recall)
- Identifies unfinished tasks or open questions
- Presents decision point for workflow continuation

Implements "software for agents" paradigm - can be invoked by:
- Humans via /ritual-awaken command
- Agents via capability registry (CONTEXT_RESTORATION)
- Orchestrator as part of full ritual workflow
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from .base import (
    AbstractRitualPhase,
    AgentCapability,
    PhasePreflightResult,
    PhaseResult,
    PhaseExecutionContext,
)
from ..ritual_events import RitualPhase

logger = logging.getLogger(__name__)


# ============================================================================
# AWAKEN-Specific Data Structures
# ============================================================================

@dataclass
class RestoredContext:
    """Context restored during AWAKEN phase."""

    # Recent activity summary
    recent_work: List[Dict[str, Any]] = field(default_factory=list)
    recent_work_summary: str = ""

    # Feature-specific knowledge
    feature_memories: List[Dict[str, Any]] = field(default_factory=list)
    feature_knowledge_summary: str = ""

    # Open items
    unfinished_tasks: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)

    # Recommendations
    recommended_focus: str = ""
    confidence: float = 0.0

    # Metadata
    memories_retrieved: int = 0
    temporal_window_hours: int = 48

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["## Context Restored", ""]

        if self.recent_work_summary:
            lines.append("### Recent Activity")
            lines.append(self.recent_work_summary)
            lines.append("")

        if self.feature_knowledge_summary:
            lines.append("### Relevant Knowledge")
            lines.append(self.feature_knowledge_summary)
            lines.append("")

        if self.unfinished_tasks:
            lines.append("### Open Items")
            for task in self.unfinished_tasks[:5]:
                lines.append(f"- [ ] {task}")
            lines.append("")

        if self.open_questions:
            lines.append("### Open Questions")
            for q in self.open_questions[:3]:
                lines.append(f"- {q}")
            lines.append("")

        if self.recommended_focus:
            lines.append("### Recommended Focus")
            lines.append(f"**{self.recommended_focus}**")
            lines.append("")

        lines.append(f"*Retrieved {self.memories_retrieved} memories from last {self.temporal_window_hours} hours*")

        return "\n".join(lines)


# ============================================================================
# AWAKEN Phase Implementation
# ============================================================================

class AwakenPhase(AbstractRitualPhase):
    """
    AWAKEN Phase: Context restoration at session start.

    Capabilities:
    - CONTEXT_RESTORATION: Primary capability
    - MEMORY_RETRIEVAL: Uses HoloLoom memory tools

    Workflow:
    1. Preflight: Pre-fetch recent memories for faster execution
    2. Execute: Restore context via temporal + semantic recall
    3. Decision: "Proceed to PLAN" / "Explore more" / "Skip to IMPLEMENT"

    Can be invoked:
    - Standalone: /ritual-awaken
    - Via orchestrator: /ritual start [feature]
    - By agents: registry.find_by_capability(CONTEXT_RESTORATION)
    """

    def __init__(
        self,
        temporal_window_hours: int = 48,
        max_recent_memories: int = 10,
        max_feature_memories: int = 15,
        recall_strategy: str = "fused",
    ):
        """
        Initialize AWAKEN phase.

        Args:
            temporal_window_hours: How far back to look for recent work
            max_recent_memories: Limit for recent activity recall
            max_feature_memories: Limit for feature-specific recall
            recall_strategy: HoloLoom recall strategy (temporal, semantic, fused)
        """
        super().__init__(
            phase=RitualPhase.AWAKEN,
            name="ritual-awaken",
            version="1.0.0",
            description="Restore session context from memory",
            capabilities=[
                AgentCapability.CONTEXT_RESTORATION,
                AgentCapability.MEMORY_RETRIEVAL,
            ],
        )

        self.temporal_window_hours = temporal_window_hours
        self.max_recent_memories = max_recent_memories
        self.max_feature_memories = max_feature_memories
        self.recall_strategy = recall_strategy

    # ========================================================================
    # Preflight: Pre-fetch recent memories
    # ========================================================================

    async def _phase_specific_preflight(
        self,
        context: PhaseExecutionContext
    ) -> Dict[str, Any]:
        """
        AWAKEN-specific preflight checks.

        Pre-fetches recent memories to enable faster execution.
        Validates that memory system can support context restoration.
        """
        checks = []
        failures = []
        warnings = []
        memory_context = {}

        # Check 1: Feature name available (optional but helpful)
        feature_name = context.feature_name or context.ritual_context.feature_name
        if feature_name:
            checks.append("feature_name_available")
            memory_context["feature_name"] = feature_name
        else:
            warnings.append("No feature name - will restore general context only")

        # Check 2: Try to pre-fetch recent memories
        try:
            recent_memories = await self._prefetch_recent_work(context)
            memory_context["recent_work_prefetch"] = recent_memories
            memory_context["prefetch_count"] = len(recent_memories)

            if recent_memories:
                checks.append("recent_memories_available")
            else:
                warnings.append("No recent memories found - starting fresh session")

        except Exception as e:
            warnings.append(f"Pre-fetch failed (will retry in execute): {str(e)[:50]}")
            memory_context["prefetch_error"] = str(e)

        # Check 3: Estimate complexity based on feature name
        if feature_name:
            estimated_queries = 2 + len(feature_name.split())  # Base + per-word
            memory_context["estimated_queries"] = min(estimated_queries, 5)
        else:
            memory_context["estimated_queries"] = 2

        return {
            "checks": checks,
            "failures": failures,
            "warnings": warnings,
            "memory_context": memory_context,
        }

    async def _prefetch_recent_work(
        self,
        context: PhaseExecutionContext
    ) -> List[Dict[str, Any]]:
        """
        Pre-fetch recent work memories during preflight.

        Uses temporal strategy for fast recent activity lookup.
        """
        # Calculate temporal window
        cutoff_time = datetime.now() - timedelta(hours=self.temporal_window_hours)

        # This would integrate with actual MCP tool call
        # For now, return structure showing expected format
        query = f"recent work since {cutoff_time.isoformat()}"

        memories = await self.recall_memories(
            query=query,
            context=context,
            strategy="temporal",
            limit=self.max_recent_memories
        )

        return memories

    def _estimate_duration(self, context: PhaseExecutionContext) -> float:
        """Estimate AWAKEN phase duration."""
        # Base: 2 queries × ~500ms each
        base_ms = 1000.0

        # Add for feature-specific queries
        feature_name = context.feature_name or context.ritual_context.feature_name
        if feature_name:
            base_ms += 500.0  # Additional semantic query

        # Check if preflight already fetched recent memories
        if context.preflight_result:
            prefetch_count = context.preflight_result.memory_context.get("prefetch_count", 0)
            if prefetch_count > 0:
                base_ms -= 300.0  # Faster with prefetch

        return max(base_ms, 500.0)

    # ========================================================================
    # Execute: Restore context
    # ========================================================================

    async def _phase_specific_execute(
        self,
        parameters: Dict[str, Any],
        context: PhaseExecutionContext
    ) -> PhaseResult:
        """
        Execute AWAKEN phase: restore session context.

        Steps:
        1. Recall recent work (temporal)
        2. Recall feature-specific knowledge (semantic)
        3. Identify open items and questions
        4. Generate recommendations
        5. Return with decision point
        """
        memories_retrieved = []
        errors = []
        warnings = []

        # Get feature name
        feature_name = (
            parameters.get("_feature_name") or
            parameters.get("feature_name") or
            context.feature_name or
            context.ritual_context.feature_name or
            ""
        )

        # Initialize restored context
        restored = RestoredContext(
            temporal_window_hours=self.temporal_window_hours
        )

        # Step 1: Get recent work (may use preflight cache)
        recent_memories = await self._get_recent_work(context, parameters)
        restored.recent_work = recent_memories
        restored.recent_work_summary = self._summarize_recent_work(recent_memories)
        memories_retrieved.extend([m.get("id", "") for m in recent_memories])

        # Step 2: Get feature-specific knowledge (if feature provided)
        if feature_name:
            feature_memories = await self._get_feature_knowledge(
                feature_name, context
            )
            restored.feature_memories = feature_memories
            restored.feature_knowledge_summary = self._summarize_feature_knowledge(
                feature_memories, feature_name
            )
            memories_retrieved.extend([m.get("id", "") for m in feature_memories])

        # Step 3: Extract open items
        restored.unfinished_tasks = self._extract_open_items(
            recent_memories, feature_name
        )
        restored.open_questions = self._extract_open_questions(
            recent_memories + restored.feature_memories
        )

        # Step 4: Generate recommendation
        restored.recommended_focus = self._generate_recommendation(
            restored, feature_name
        )
        restored.memories_retrieved = len(memories_retrieved)

        # Step 5: Calculate confidence
        restored.confidence = self._calculate_confidence(restored)

        # Determine decision options and suggestion
        decision_options = self.get_decision_options()
        suggested_decision, decision_reasoning = self._suggest_decision(restored)

        return PhaseResult(
            success=True,
            output=restored,
            confidence=restored.confidence,
            memories_stored=[],  # AWAKEN doesn't store, only retrieves
            memories_retrieved=memories_retrieved,
            decision_needed=True,
            decision_options=decision_options,
            suggested_decision=suggested_decision,
            decision_reasoning=decision_reasoning,
            metadata={
                "feature_name": feature_name,
                "recent_count": len(restored.recent_work),
                "feature_count": len(restored.feature_memories),
                "open_items_count": len(restored.unfinished_tasks),
                "summary": restored.to_summary(),
            }
        )

    async def _get_recent_work(
        self,
        context: PhaseExecutionContext,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get recent work memories.

        Uses preflight cache if available, otherwise fetches fresh.
        """
        # Check preflight cache
        memory_context = parameters.get("_memory_context", {})
        prefetch = memory_context.get("recent_work_prefetch", [])

        if prefetch:
            logger.info(f"Using {len(prefetch)} pre-fetched recent memories")
            return prefetch

        # Fetch fresh
        logger.info("Fetching recent work memories (no prefetch available)")
        return await self._prefetch_recent_work(context)

    async def _get_feature_knowledge(
        self,
        feature_name: str,
        context: PhaseExecutionContext
    ) -> List[Dict[str, Any]]:
        """
        Get feature-specific knowledge via semantic recall.
        """
        query = f"knowledge about {feature_name}"

        memories = await self.recall_memories(
            query=query,
            context=context,
            strategy="semantic",
            limit=self.max_feature_memories
        )

        return memories

    def _summarize_recent_work(
        self,
        memories: List[Dict[str, Any]]
    ) -> str:
        """Generate summary of recent work."""
        if not memories:
            return "No recent activity found."

        # Group by topic/tag if available
        topics = {}
        for mem in memories:
            topic = mem.get("topic", mem.get("tags", ["general"])[0] if mem.get("tags") else "general")
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(mem.get("content", mem.get("text", ""))[:100])

        lines = []
        for topic, items in list(topics.items())[:5]:
            lines.append(f"**{topic}**: {len(items)} items")
            if items:
                lines.append(f"  - {items[0]}...")

        return "\n".join(lines) if lines else f"Found {len(memories)} recent activities"

    def _summarize_feature_knowledge(
        self,
        memories: List[Dict[str, Any]],
        feature_name: str
    ) -> str:
        """Generate summary of feature-specific knowledge."""
        if not memories:
            return f"No prior knowledge about '{feature_name}' found."

        # Extract key points
        key_points = []
        for mem in memories[:5]:
            content = mem.get("content", mem.get("text", ""))
            if content:
                key_points.append(f"- {content[:150]}...")

        if key_points:
            return f"Found {len(memories)} memories about '{feature_name}':\n" + "\n".join(key_points)
        return f"Found {len(memories)} memories about '{feature_name}'"

    def _extract_open_items(
        self,
        memories: List[Dict[str, Any]],
        feature_name: str
    ) -> List[str]:
        """Extract unfinished tasks from memories."""
        open_items = []

        # Look for TODO patterns, incomplete status, etc.
        todo_keywords = ["todo", "TODO", "fixme", "FIXME", "unfinished", "incomplete", "WIP"]

        for mem in memories:
            content = mem.get("content", mem.get("text", "")).lower()
            for keyword in todo_keywords:
                if keyword.lower() in content:
                    # Extract the task
                    task = mem.get("content", mem.get("text", ""))[:100]
                    if task and task not in open_items:
                        open_items.append(task)
                        break

        return open_items[:10]  # Limit to 10 items

    def _extract_open_questions(
        self,
        memories: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract open questions from memories."""
        questions = []

        for mem in memories:
            content = mem.get("content", mem.get("text", ""))
            # Look for question patterns
            if "?" in content:
                # Find the question
                sentences = content.split(".")
                for sentence in sentences:
                    if "?" in sentence:
                        q = sentence.strip()
                        if q and len(q) > 10 and q not in questions:
                            questions.append(q[:150])
                            break

        return questions[:5]  # Limit to 5 questions

    def _generate_recommendation(
        self,
        restored: RestoredContext,
        feature_name: str
    ) -> str:
        """Generate recommended focus based on restored context."""
        if restored.unfinished_tasks:
            return f"Continue with: {restored.unfinished_tasks[0][:80]}"

        if feature_name and restored.feature_memories:
            return f"Build on existing knowledge of '{feature_name}'"

        if feature_name:
            return f"Start fresh work on '{feature_name}'"

        if restored.recent_work:
            return "Review recent activity and decide on focus"

        return "Begin new work session"

    def _calculate_confidence(self, restored: RestoredContext) -> float:
        """
        Calculate confidence in context restoration.

        Higher confidence when:
        - More memories retrieved
        - Both recent and feature-specific found
        - Clear open items identified
        """
        confidence = 0.3  # Base confidence

        # Recent work available
        if restored.recent_work:
            confidence += 0.2
            if len(restored.recent_work) >= 5:
                confidence += 0.1

        # Feature knowledge available
        if restored.feature_memories:
            confidence += 0.2
            if len(restored.feature_memories) >= 5:
                confidence += 0.1

        # Clear open items
        if restored.unfinished_tasks:
            confidence += 0.1

        return min(confidence, 1.0)

    def _suggest_decision(
        self,
        restored: RestoredContext
    ) -> tuple[str, str]:
        """
        Suggest next decision based on restored context.

        Returns (suggested_decision, reasoning).
        """
        if restored.confidence >= 0.7:
            return (
                "Proceed to PLAN",
                f"Good context restored ({restored.memories_retrieved} memories, "
                f"{len(restored.unfinished_tasks)} open items). Ready to plan."
            )

        if restored.confidence >= 0.4:
            return (
                "Proceed to PLAN",
                "Moderate context available. Can proceed with planning."
            )

        if restored.recent_work or restored.feature_memories:
            return (
                "Explore more context",
                "Limited context found. Consider exploring more before planning."
            )

        return (
            "Skip to IMPLEMENT",
            "No prior context found. Starting fresh - can skip to implementation."
        )

    # ========================================================================
    # Decision Options
    # ========================================================================

    def get_decision_options(self) -> List[str]:
        """Get available decision options for AWAKEN phase."""
        return [
            "Proceed to PLAN",
            "Explore more context",
            "Skip to IMPLEMENT",
        ]


# ============================================================================
# Factory Function
# ============================================================================

def create_awaken_phase(
    temporal_window_hours: int = 48,
    max_recent_memories: int = 10,
    max_feature_memories: int = 15,
    recall_strategy: str = "fused",
) -> AwakenPhase:
    """
    Factory function to create AWAKEN phase skill.

    Args:
        temporal_window_hours: How far back to look for recent work
        max_recent_memories: Limit for recent activity recall
        max_feature_memories: Limit for feature-specific recall
        recall_strategy: HoloLoom recall strategy

    Returns:
        Configured AwakenPhase instance
    """
    return AwakenPhase(
        temporal_window_hours=temporal_window_hours,
        max_recent_memories=max_recent_memories,
        max_feature_memories=max_feature_memories,
        recall_strategy=recall_strategy,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'RestoredContext',
    'AwakenPhase',
    'create_awaken_phase',
]