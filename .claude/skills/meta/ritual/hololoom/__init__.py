"""
HoloLoom Integration for Coding Ritual
======================================

Unified integration with HoloLoom's advanced capabilities:

1. **Memory System** - Store sessions in Yarn Graph, recall past sessions
2. **RAG** - Search past sessions for relevant decisions
3. **Alignment Framework** - Integrate with safety guardrails
4. **Thompson Sampling** - Learn optimal ritual patterns

Usage:
    from ritual.hololoom import HoloLoomIntegration

    # Create integration (auto-initializes available components)
    integration = HoloLoomIntegration()

    # Check availability
    print(integration.status())

    # Use during ritual phases
    recommendations = await integration.get_open_recommendations(task)
    await integration.store_session(session)
    safety_result = await integration.run_safety_checks(code)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Import adapters
from .memory_adapter import (
    RitualMemoryAdapter,
    store_ritual_session,
    find_similar_ritual_sessions,
    get_lessons_for_ritual_task,
    HOLOLOOM_AVAILABLE,
)
from .rag_adapter import (
    RitualRAGAdapter,
    RitualSearchResult,
    search_ritual_sessions,
    ask_ritual_history,
    get_task_recommendations,
    HOLOLOOM_RAG_AVAILABLE,
)
from .alignment_adapter import (
    RitualAlignmentAdapter,
    SafetyCheckResult,
    RiskLevel,
    run_ritual_safety_checks,
    gate_ritual_action,
    get_safety_check_definitions,
    HOLOLOOM_ALIGNMENT_AVAILABLE,
)
from .learning_adapter import (
    RitualLearningAdapter,
    RitualLearningState,
    BetaPrior,
    recommend_ritual_complexity,
    update_ritual_learning,
    get_ritual_recommendations,
    HOLOLOOM_LEARNING_AVAILABLE,
)


class HoloLoomIntegration:
    """
    Unified HoloLoom integration for Coding Ritual.

    Provides a single interface to all HoloLoom capabilities:
    - Memory: Store and retrieve sessions
    - RAG: Semantic search of past sessions
    - Alignment: Safety checks and guardrails
    - Learning: Thompson Sampling recommendations

    All components gracefully degrade if HoloLoom is not available.
    """

    def __init__(
        self,
        memory_adapter: Optional[RitualMemoryAdapter] = None,
        rag_adapter: Optional[RitualRAGAdapter] = None,
        alignment_adapter: Optional[RitualAlignmentAdapter] = None,
        learning_adapter: Optional[RitualLearningAdapter] = None,
        auto_initialize: bool = True,
    ):
        """
        Initialize HoloLoom integration.

        Args:
            memory_adapter: Custom memory adapter (optional)
            rag_adapter: Custom RAG adapter (optional)
            alignment_adapter: Custom alignment adapter (optional)
            learning_adapter: Custom learning adapter (optional)
            auto_initialize: Create default adapters if not provided
        """
        self.memory = memory_adapter
        self.rag = rag_adapter
        self.alignment = alignment_adapter
        self.learning = learning_adapter

        if auto_initialize:
            self._auto_initialize()

    def _auto_initialize(self) -> None:
        """Auto-initialize adapters."""
        if self.memory is None:
            self.memory = RitualMemoryAdapter()

        if self.rag is None:
            self.rag = RitualRAGAdapter(memory_adapter=self.memory)

        if self.alignment is None:
            self.alignment = RitualAlignmentAdapter()

        if self.learning is None:
            self.learning = RitualLearningAdapter()

    def status(self) -> Dict[str, Any]:
        """
        Get integration status.

        Returns:
            Status of all integration components
        """
        return {
            "memory": {
                "available": HOLOLOOM_AVAILABLE,
                "initialized": self.memory is not None and self.memory.is_available,
            },
            "rag": {
                "available": HOLOLOOM_RAG_AVAILABLE,
                "initialized": self.rag is not None and self.rag.is_available,
            },
            "alignment": {
                "available": HOLOLOOM_ALIGNMENT_AVAILABLE,
                "initialized": self.alignment is not None and self.alignment.is_available,
            },
            "learning": {
                "available": True,  # Learning always available (no HoloLoom dep)
                "initialized": self.learning is not None,
                "sessions_tracked": (
                    self.learning.state.total_sessions
                    if self.learning else 0
                ),
            },
        }

    # ========== Memory Integration ==========

    async def store_session(
        self,
        session_id: str,
        task: str,
        complexity: str,
        success_criteria: List[str],
        started_at: datetime,
        completed_at: Optional[datetime] = None,
        phases_completed: Optional[List[str]] = None,
        lessons_learned: Optional[List[str]] = None,
        builds: Optional[List[Dict[str, Any]]] = None,
        checks: Optional[List[Dict[str, Any]]] = None,
        notes: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a completed session in memory."""
        if not self.memory:
            return {"status": "unavailable", "message": "Memory not initialized"}

        # Also update learning
        if self.learning:
            success = completed_at is not None
            self.learning.update_from_session(
                task=task,
                complexity=complexity,
                success=success,
                phases_completed=phases_completed or [],
                builds=builds,
                checks=checks,
            )

        return await self.memory.store_session(
            session_id=session_id,
            task=task,
            complexity=complexity,
            success_criteria=success_criteria,
            started_at=started_at,
            completed_at=completed_at,
            phases_completed=phases_completed,
            lessons_learned=lessons_learned,
            builds=builds,
            checks=checks,
            notes=notes,
            correlation_id=correlation_id,
        )

    async def find_similar_sessions(
        self,
        task: str,
        complexity: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar past sessions."""
        if not self.memory:
            return []
        return await self.memory.find_similar_sessions(task, complexity, limit)

    async def get_lessons(
        self,
        task: str,
        limit: int = 10,
    ) -> List[str]:
        """Get lessons learned for a task."""
        if not self.memory:
            return []
        return await self.memory.get_lessons_for_task(task, limit)

    # ========== RAG Integration ==========

    async def search_sessions(
        self,
        query: str,
        limit: int = 5,
    ) -> List[RitualSearchResult]:
        """Search past sessions semantically."""
        if not self.rag:
            return []
        return await self.rag.search_sessions(query, limit)

    async def ask_history(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """Ask a question about past sessions."""
        if not self.rag:
            return {"answer": "RAG not available", "sources": []}
        return await self.rag.ask_about_sessions(question)

    async def research_topic(
        self,
        topic: str,
    ) -> Dict[str, Any]:
        """Research a topic across past sessions."""
        if not self.rag:
            return {"topic": topic, "sources": []}
        return await self.rag.research_topic(topic)

    # ========== Alignment Integration ==========

    async def run_safety_checks(
        self,
        code_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run all safety checks."""
        if not self.alignment:
            return {"status": "unavailable", "passed": 0, "failed": 0}
        return await self.alignment.run_all_safety_checks(code_content, context)

    async def run_check(
        self,
        check_id: str,
        code_content: Optional[str] = None,
    ) -> SafetyCheckResult:
        """Run a specific safety check."""
        if not self.alignment:
            return SafetyCheckResult(
                check_id=check_id,
                passed=True,
                risk_level=RiskLevel.LOW,
                message="Alignment not available",
                recommendations=[],
            )
        return await self.alignment.run_safety_check(check_id, code_content)

    async def gate_action(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Gate an action through safety guardrails."""
        if not self.alignment:
            return {"allowed": True, "reason": "Alignment not available"}
        return await self.alignment.gate_action(action, context)

    # ========== Learning Integration ==========

    def recommend_complexity(
        self,
        task: str,
    ) -> Tuple[str, float]:
        """Get complexity recommendation for a task."""
        if not self.learning:
            return ("fast", 0.5)
        return self.learning.recommend_complexity(task)

    def recommend_phases(
        self,
        complexity: str,
    ) -> List[Tuple[str, float]]:
        """Get phase recommendations for a complexity level."""
        if not self.learning:
            return [("open", 0.5), ("design", 0.5), ("build", 0.5)]
        return self.learning.recommend_phases(complexity)

    def get_recommendations(
        self,
        task: str,
    ) -> Dict[str, Any]:
        """Get comprehensive recommendations for a task."""
        if not self.learning:
            return {
                "task_type": "other",
                "recommended_complexity": "fast",
                "complexity_confidence": 0.5,
            }
        return self.learning.get_recommendations(task)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.learning:
            return {"total_sessions": 0}
        return self.learning.get_statistics()

    # ========== Combined Operations ==========

    async def get_open_recommendations(
        self,
        task: str,
    ) -> Dict[str, Any]:
        """
        Get all recommendations for starting a new session.

        Combines:
        - Complexity recommendations from learning
        - Similar past sessions from memory
        - Relevant lessons from RAG
        - Safety considerations
        """
        recommendations = {
            "task": task,
            "timestamp": datetime.now().isoformat(),
        }

        # Learning recommendations
        if self.learning:
            learning_rec = self.learning.get_recommendations(task)
            recommendations["learning"] = learning_rec
            recommendations["recommended_complexity"] = learning_rec.get(
                "recommended_complexity", "fast"
            )
        else:
            recommendations["recommended_complexity"] = "fast"

        # Similar sessions
        if self.memory:
            similar = await self.memory.find_similar_sessions(task, limit=3)
            recommendations["similar_sessions"] = similar
        else:
            recommendations["similar_sessions"] = []

        # Lessons from past
        if self.memory:
            lessons = await self.memory.get_lessons_for_task(task, limit=5)
            recommendations["relevant_lessons"] = lessons
        else:
            recommendations["relevant_lessons"] = []

        # Warnings from RAG
        if self.rag:
            task_rec = await self.rag.get_task_recommendations(task)
            recommendations["warnings"] = task_rec.get("warnings", [])
        else:
            recommendations["warnings"] = []

        return recommendations

    async def get_close_summary(
        self,
        session_id: str,
        task: str,
        complexity: str,
        phases_completed: List[str],
        lessons_learned: List[str],
        success: bool,
    ) -> Dict[str, Any]:
        """
        Get summary when closing a session.

        Includes:
        - Session storage confirmation
        - Learning updates
        - Comparison with similar sessions
        """
        summary = {
            "session_id": session_id,
            "task": task,
            "success": success,
        }

        # Update learning
        if self.learning:
            learning_update = self.learning.update_from_session(
                task=task,
                complexity=complexity,
                success=success,
                phases_completed=phases_completed,
            )
            summary["learning_updated"] = True
            summary["priors_updated"] = len(learning_update.get("priors_updated", []))

        # Find how this session compares
        if self.memory:
            similar = await self.memory.find_similar_sessions(task, limit=5)
            if similar:
                avg_phases = sum(
                    len(s.get("phases_completed", [])) for s in similar
                ) / len(similar)
                summary["comparison"] = {
                    "similar_sessions": len(similar),
                    "avg_phases_in_similar": avg_phases,
                    "your_phases": len(phases_completed),
                }

        return summary


# Export key classes and functions
__all__ = [
    # Main integration
    "HoloLoomIntegration",
    # Memory
    "RitualMemoryAdapter",
    "store_ritual_session",
    "find_similar_ritual_sessions",
    "get_lessons_for_ritual_task",
    # RAG
    "RitualRAGAdapter",
    "RitualSearchResult",
    "search_ritual_sessions",
    "ask_ritual_history",
    "get_task_recommendations",
    # Alignment
    "RitualAlignmentAdapter",
    "SafetyCheckResult",
    "RiskLevel",
    "run_ritual_safety_checks",
    "gate_ritual_action",
    "get_safety_check_definitions",
    # Learning
    "RitualLearningAdapter",
    "RitualLearningState",
    "BetaPrior",
    "recommend_ritual_complexity",
    "update_ritual_learning",
    "get_ritual_recommendations",
    # Availability flags
    "HOLOLOOM_AVAILABLE",
    "HOLOLOOM_RAG_AVAILABLE",
    "HOLOLOOM_ALIGNMENT_AVAILABLE",
    "HOLOLOOM_LEARNING_AVAILABLE",
]
