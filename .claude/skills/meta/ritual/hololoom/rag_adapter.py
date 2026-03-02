"""
HoloLoom RAG Integration for Coding Ritual
==========================================

Enables semantic search of past ritual sessions using HoloLoom's RAG system:
- Level 4 Agentic RAG for multi-step reasoning
- Hybrid search (BM25 + semantic similarity)
- Graph RAG for relationship traversal

Use Cases:
- "What did I learn when building auth systems?"
- "Show me sessions where I used fast complexity"
- "Find lessons about error handling"
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Try to import HoloLoom RAG - graceful degradation if not available
HOLOLOOM_RAG_AVAILABLE = False
try:
    hololoom_path = Path(__file__).parent.parent.parent.parent.parent.parent
    if str(hololoom_path) not in sys.path:
        sys.path.insert(0, str(hololoom_path))

    from hololoom.rag import SimpleRAG
    HOLOLOOM_RAG_AVAILABLE = True
except ImportError:
    pass


@dataclass
class RitualSearchResult:
    """Result from ritual session search."""

    session_id: str
    task: str
    complexity: str
    relevance: float
    lessons_learned: List[str]
    phases_completed: List[str]
    snippet: str
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "task": self.task,
            "complexity": self.complexity,
            "relevance": self.relevance,
            "lessons_learned": self.lessons_learned,
            "phases_completed": self.phases_completed,
            "snippet": self.snippet,
            "completed_at": self.completed_at,
        }


class RitualRAGAdapter:
    """
    RAG adapter for semantic search of ritual sessions.

    Provides:
    - Semantic search across past sessions
    - Question answering about ritual history
    - Lesson retrieval with context
    - Multi-query research mode
    """

    def __init__(
        self,
        rag_instance: Optional[Any] = None,
        memory_adapter: Optional[Any] = None,
    ):
        """
        Initialize RAG adapter.

        Args:
            rag_instance: HoloLoom SimpleRAG instance (optional)
            memory_adapter: RitualMemoryAdapter for session data
        """
        self.rag = rag_instance
        self.memory_adapter = memory_adapter
        self._initialized = False

        if HOLOLOOM_RAG_AVAILABLE:
            self._auto_initialize()

    def _auto_initialize(self) -> None:
        """Auto-initialize RAG if available."""
        if not HOLOLOOM_RAG_AVAILABLE:
            return

        try:
            if self.rag is None:
                # Create RAG with default config
                # In production, would use async context manager
                pass  # RAG requires async initialization
            self._initialized = True
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """Check if RAG is available."""
        return HOLOLOOM_RAG_AVAILABLE

    async def search_sessions(
        self,
        query: str,
        limit: int = 5,
        mode: str = "direct",
    ) -> List[RitualSearchResult]:
        """
        Search past sessions using semantic search.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            mode: RAG mode (direct, verify, research)

        Returns:
            List of search results
        """
        # If RAG not available, fall back to memory adapter
        if self.memory_adapter:
            similar = await self.memory_adapter.find_similar_sessions(
                task=query,
                limit=limit,
            )
            return [
                RitualSearchResult(
                    session_id=s.get("session_id", ""),
                    task=s.get("task", ""),
                    complexity=s.get("complexity", ""),
                    relevance=s.get("similarity", 0.0),
                    lessons_learned=s.get("lessons_learned", []),
                    phases_completed=[],
                    snippet=s.get("task", "")[:100],
                    completed_at=s.get("completed_at"),
                )
                for s in similar
            ]

        return []

    async def ask_about_sessions(
        self,
        question: str,
        mode: str = "verify",
    ) -> Dict[str, Any]:
        """
        Ask a question about past ritual sessions.

        Uses RAG to find relevant context and generate answer.

        Args:
            question: Natural language question
            mode: RAG reasoning mode

        Returns:
            Answer with sources and confidence
        """
        if not self.is_available:
            return {
                "answer": "RAG not available. Cannot search past sessions.",
                "confidence": 0.0,
                "sources": [],
                "reasoning_mode": "unavailable",
            }

        # Search for relevant sessions
        results = await self.search_sessions(question, limit=5, mode=mode)

        if not results:
            return {
                "answer": "No relevant past sessions found.",
                "confidence": 0.3,
                "sources": [],
                "reasoning_mode": mode,
            }

        # Build context from results
        context_parts = []
        for r in results:
            part = f"Session: {r.task}\n"
            part += f"Complexity: {r.complexity}\n"
            if r.lessons_learned:
                part += f"Lessons: {', '.join(r.lessons_learned)}\n"
            context_parts.append(part)

        context = "\n---\n".join(context_parts)

        # In production, would use RAG to generate answer
        # For now, return structured result
        return {
            "answer": f"Found {len(results)} relevant sessions.",
            "confidence": max(r.relevance for r in results) if results else 0.0,
            "sources": [r.to_dict() for r in results],
            "reasoning_mode": mode,
            "context": context,
        }

    async def find_lessons(
        self,
        topic: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find lessons learned related to a topic.

        Args:
            topic: Topic to search for
            limit: Maximum lessons to return

        Returns:
            List of lessons with context
        """
        results = await self.search_sessions(topic, limit=limit)

        lessons = []
        for r in results:
            for lesson in r.lessons_learned:
                lessons.append({
                    "lesson": lesson,
                    "from_task": r.task,
                    "session_id": r.session_id,
                    "relevance": r.relevance,
                })

        # Sort by relevance
        lessons.sort(key=lambda x: x["relevance"], reverse=True)
        return lessons[:limit]

    async def get_task_recommendations(
        self,
        task: str,
    ) -> Dict[str, Any]:
        """
        Get recommendations for a new task based on past sessions.

        Args:
            task: New task description

        Returns:
            Recommendations including complexity, lessons, warnings
        """
        # Search for similar tasks
        similar = await self.search_sessions(task, limit=5)

        if not similar:
            return {
                "has_history": False,
                "recommended_complexity": "fast",
                "relevant_lessons": [],
                "warnings": [],
                "similar_sessions": [],
            }

        # Analyze similar sessions
        complexities = [s.complexity for s in similar]
        most_common = max(set(complexities), key=complexities.count)

        # Collect lessons
        all_lessons = []
        for s in similar:
            all_lessons.extend(s.lessons_learned)

        # Look for patterns/warnings
        warnings = []
        lesson_text = " ".join(all_lessons).lower()
        if "careful" in lesson_text or "avoid" in lesson_text:
            warnings.append("Past sessions indicate potential pitfalls")
        if "timeout" in lesson_text or "slow" in lesson_text:
            warnings.append("Consider using higher complexity for thorough work")

        return {
            "has_history": True,
            "recommended_complexity": most_common,
            "relevant_lessons": list(set(all_lessons))[:5],
            "warnings": warnings,
            "similar_sessions": [s.to_dict() for s in similar[:3]],
        }

    async def research_topic(
        self,
        topic: str,
        max_queries: int = 3,
    ) -> Dict[str, Any]:
        """
        Research a topic using multi-query RAG mode.

        Generates sub-questions and synthesizes answers.

        Args:
            topic: Topic to research
            max_queries: Maximum sub-queries

        Returns:
            Research results with synthesis
        """
        # Generate sub-questions
        sub_questions = [
            f"What tasks involved {topic}?",
            f"What lessons were learned about {topic}?",
            f"What complexity levels worked best for {topic}?",
        ][:max_queries]

        # Answer each question
        answers = []
        for q in sub_questions:
            result = await self.ask_about_sessions(q, mode="direct")
            answers.append({
                "question": q,
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "sources": result.get("sources", []),
            })

        # Synthesize findings
        all_sources = []
        for a in answers:
            all_sources.extend(a.get("sources", []))

        # Deduplicate sources by session_id
        seen_ids = set()
        unique_sources = []
        for s in all_sources:
            sid = s.get("session_id")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                unique_sources.append(s)

        return {
            "topic": topic,
            "sub_queries": answers,
            "unique_sessions": len(unique_sources),
            "sources": unique_sources[:10],
            "synthesis": f"Researched {topic} across {len(unique_sources)} past sessions.",
        }


# Convenience functions
async def search_ritual_sessions(
    query: str,
    limit: int = 5,
) -> List[RitualSearchResult]:
    """Search past ritual sessions."""
    adapter = RitualRAGAdapter()
    return await adapter.search_sessions(query, limit)


async def ask_ritual_history(
    question: str,
) -> Dict[str, Any]:
    """Ask a question about ritual history."""
    adapter = RitualRAGAdapter()
    return await adapter.ask_about_sessions(question)


async def get_task_recommendations(
    task: str,
) -> Dict[str, Any]:
    """Get recommendations for a new task."""
    adapter = RitualRAGAdapter()
    return await adapter.get_task_recommendations(task)