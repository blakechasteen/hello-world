"""
HoloLoom Memory Integration for Coding Ritual
==============================================

Stores ritual sessions in HoloLoom's memory system:
- Yarn Graph (Knowledge Graph) for session relationships
- Vector Memory for semantic session search
- Awareness Graph for activation tracking

This enables:
- Recall of past sessions for similar tasks
- Learning from historical session outcomes
- Building institutional knowledge over time
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

# Try to import HoloLoom - graceful degradation if not available
HOLOLOOM_AVAILABLE = False
try:
    # Add HoloLoom to path if needed
    hololoom_path = Path(__file__).parent.parent.parent.parent.parent.parent
    if str(hololoom_path) not in sys.path:
        sys.path.insert(0, str(hololoom_path))

    from hololoom.memory.graph import KG, KGEdge
    from hololoom.memory.unified import UnifiedMemory
    from hololoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError:
    pass


class RitualMemoryAdapter:
    """
    Adapter for storing ritual sessions in HoloLoom's memory system.

    Stores:
    - Session metadata as graph nodes
    - Session relationships (task similarity, learned lessons)
    - Embeddings for semantic search

    Enables:
    - Find similar past sessions
    - Recall lessons learned
    - Track session patterns over time
    """

    def __init__(
        self,
        unified_memory: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
        auto_initialize: bool = True,
    ):
        """
        Initialize memory adapter.

        Args:
            unified_memory: HoloLoom UnifiedMemory instance (optional)
            knowledge_graph: HoloLoom KG instance (optional)
            auto_initialize: Create default instances if not provided
        """
        self.unified_memory = unified_memory
        self.knowledge_graph = knowledge_graph
        self._initialized = False

        if auto_initialize and HOLOLOOM_AVAILABLE:
            self._auto_initialize()

    def _auto_initialize(self) -> None:
        """Auto-initialize HoloLoom components if available."""
        if not HOLOLOOM_AVAILABLE:
            return

        try:
            if self.knowledge_graph is None:
                self.knowledge_graph = KG()
            self._initialized = True
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """Check if HoloLoom memory is available."""
        return HOLOLOOM_AVAILABLE and self._initialized

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
        """
        Store a ritual session in HoloLoom memory.

        Args:
            session_id: Unique session identifier
            task: Task description
            complexity: Complexity level (lite/fast/full/research)
            success_criteria: List of success criteria
            started_at: Session start time
            completed_at: Session completion time (optional)
            phases_completed: List of completed phases
            lessons_learned: List of lessons from session
            builds: Build wave records
            checks: Safety check records
            notes: Additional notes
            correlation_id: Workflow correlation ID

        Returns:
            Storage result with node IDs and status
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "message": "HoloLoom memory not available",
                "session_id": session_id,
            }

        try:
            # Create session node in knowledge graph
            session_node_id = f"ritual_session:{session_id}"

            # Build session metadata
            metadata = {
                "type": "ritual_session",
                "session_id": session_id,
                "task": task,
                "complexity": complexity,
                "success_criteria": success_criteria,
                "started_at": started_at.isoformat() if started_at else None,
                "completed_at": completed_at.isoformat() if completed_at else None,
                "phases_completed": phases_completed or [],
                "lessons_learned": lessons_learned or [],
                "builds": builds or [],
                "checks": checks or [],
                "notes": notes,
                "correlation_id": correlation_id,
                "stored_at": datetime.now().isoformat(),
            }

            # Store in knowledge graph
            edges = []

            # Create session → task relationship
            task_node_id = f"ritual_task:{_sanitize_node_id(task[:50])}"
            edges.append(KGEdge(
                source=session_node_id,
                target=task_node_id,
                relation="HAS_TASK",
                weight=1.0,
            ))

            # Create session → complexity relationship
            complexity_node_id = f"ritual_complexity:{complexity}"
            edges.append(KGEdge(
                source=session_node_id,
                target=complexity_node_id,
                relation="HAS_COMPLEXITY",
                weight=1.0,
            ))

            # Create lesson edges
            for i, lesson in enumerate(lessons_learned or []):
                lesson_node_id = f"ritual_lesson:{session_id}:{i}"
                edges.append(KGEdge(
                    source=session_node_id,
                    target=lesson_node_id,
                    relation="LEARNED",
                    weight=1.0,
                ))

            # Add edges to knowledge graph
            if self.knowledge_graph:
                self.knowledge_graph.add_edges(edges)

                # Store metadata on session node
                self.knowledge_graph.G.nodes[session_node_id].update(metadata)

            return {
                "status": "success",
                "session_id": session_id,
                "node_id": session_node_id,
                "edges_created": len(edges),
                "metadata": metadata,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "session_id": session_id,
            }

    async def find_similar_sessions(
        self,
        task: str,
        complexity: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find past sessions similar to a given task.

        Args:
            task: Task description to match
            complexity: Optional complexity filter
            limit: Maximum number of sessions to return

        Returns:
            List of similar sessions with metadata
        """
        if not self.is_available:
            return []

        try:
            similar_sessions = []

            # Search knowledge graph for session nodes
            if self.knowledge_graph:
                for node_id in self.knowledge_graph.G.nodes():
                    if not node_id.startswith("ritual_session:"):
                        continue

                    node_data = self.knowledge_graph.G.nodes[node_id]

                    # Filter by complexity if specified
                    if complexity and node_data.get("complexity") != complexity:
                        continue

                    # Simple text matching for now
                    # In production, use semantic similarity
                    node_task = node_data.get("task", "").lower()
                    query_task = task.lower()

                    # Calculate simple similarity
                    words_node = set(node_task.split())
                    words_query = set(query_task.split())
                    overlap = len(words_node & words_query)
                    union = len(words_node | words_query)
                    similarity = overlap / union if union > 0 else 0.0

                    if similarity > 0.1:  # Minimum threshold
                        similar_sessions.append({
                            "session_id": node_data.get("session_id"),
                            "task": node_data.get("task"),
                            "complexity": node_data.get("complexity"),
                            "lessons_learned": node_data.get("lessons_learned", []),
                            "similarity": similarity,
                            "completed_at": node_data.get("completed_at"),
                        })

            # Sort by similarity and return top results
            similar_sessions.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_sessions[:limit]

        except Exception:
            return []

    async def get_lessons_for_task(
        self,
        task: str,
        limit: int = 10,
    ) -> List[str]:
        """
        Get lessons learned from similar past sessions.

        Args:
            task: Task description
            limit: Maximum lessons to return

        Returns:
            List of relevant lessons learned
        """
        similar = await self.find_similar_sessions(task, limit=5)

        lessons = []
        for session in similar:
            for lesson in session.get("lessons_learned", []):
                if lesson not in lessons:
                    lessons.append(lesson)
                    if len(lessons) >= limit:
                        break
            if len(lessons) >= limit:
                break

        return lessons

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata or None if not found
        """
        if not self.is_available:
            return None

        try:
            node_id = f"ritual_session:{session_id}"
            if self.knowledge_graph and node_id in self.knowledge_graph.G.nodes:
                return dict(self.knowledge_graph.G.nodes[node_id])
            return None
        except Exception:
            return None

    async def get_session_history(
        self,
        limit: int = 20,
        complexity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent session history.

        Args:
            limit: Maximum sessions to return
            complexity: Optional complexity filter

        Returns:
            List of recent sessions
        """
        if not self.is_available:
            return []

        try:
            sessions = []

            if self.knowledge_graph:
                for node_id in self.knowledge_graph.G.nodes():
                    if not node_id.startswith("ritual_session:"):
                        continue

                    node_data = self.knowledge_graph.G.nodes[node_id]

                    if complexity and node_data.get("complexity") != complexity:
                        continue

                    sessions.append({
                        "session_id": node_data.get("session_id"),
                        "task": node_data.get("task"),
                        "complexity": node_data.get("complexity"),
                        "started_at": node_data.get("started_at"),
                        "completed_at": node_data.get("completed_at"),
                        "phases_completed": node_data.get("phases_completed", []),
                    })

            # Sort by start time (most recent first)
            sessions.sort(
                key=lambda x: x.get("started_at", ""),
                reverse=True
            )

            return sessions[:limit]

        except Exception:
            return []


def _sanitize_node_id(text: str) -> str:
    """Sanitize text for use as node ID."""
    # Replace spaces and special chars with underscores
    sanitized = "".join(
        c if c.isalnum() else "_"
        for c in text.lower()
    )
    # Remove consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


# Convenience functions
async def store_ritual_session(
    session_id: str,
    task: str,
    complexity: str,
    **kwargs,
) -> Dict[str, Any]:
    """Store a ritual session (convenience function)."""
    adapter = RitualMemoryAdapter()
    return await adapter.store_session(
        session_id=session_id,
        task=task,
        complexity=complexity,
        **kwargs,
    )


async def find_similar_ritual_sessions(
    task: str,
    complexity: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Find similar sessions (convenience function)."""
    adapter = RitualMemoryAdapter()
    return await adapter.find_similar_sessions(task, complexity, limit)


async def get_lessons_for_ritual_task(
    task: str,
    limit: int = 10,
) -> List[str]:
    """Get lessons for a task (convenience function)."""
    adapter = RitualMemoryAdapter()
    return await adapter.get_lessons_for_task(task, limit)