"""
RAG-Powered Tutoring Engine

Context-aware question answering using HoloLoom RAG system.

**Key Features**:
- Grade-appropriate explanations
- Builds on student's prior knowledge
- Multi-step reasoning (DIRECT/VERIFY/RESEARCH)
- Source attribution (curriculum objectives)
- Multimodal support (text, images, videos)

**Implementation Date**: November 15, 2025
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import asyncio

try:
    from HoloLoom.rag import SimpleRAG
    from HoloLoom.config import Config
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("⚠️  HoloLoom RAG not available - tutoring will use fallback mode")

from .curriculum_graph import EdWINKnowledgeGraph
from .student_model import EdWINStudentModel


@dataclass
class TutoringResponse:
    """
    Response to student question

    Contains answer, metadata, and recommendations.
    """
    question: str
    answer: str
    confidence: float  # 0.0-1.0
    sources: List[str] = field(default_factory=list)  # Objective IDs
    reasoning_mode: str = "direct"  # direct, verify, research

    # Metadata
    response_time_ms: float = 0.0
    reading_level: Optional[float] = None  # Flesch-Kincaid
    safety_score: float = 1.0  # 0.0-1.0

    # Related content
    related_objectives: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

    # Multimodal
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.utcnow)


class EdWINTutor:
    """
    RAG-Powered AI Tutor

    Uses HoloLoom RAG to provide context-aware explanations tailored to:
    - Student's grade level
    - Mastered concepts (prior knowledge)
    - Learning style preferences
    - Knowledge gaps

    Example:
        >>> tutor = EdWINTutor(curriculum_kg)
        >>>
        >>> result = await tutor.answer_question(
        ...     "How do I solve 2x + 5 = 13?",
        ...     student_model=student,
        ...     mode="verify"
        ... )
        >>>
        >>> print(result.answer)
        >>> print(f"Confidence: {result.confidence:.1%}")
        >>> print(f"Sources: {result.sources}")
    """

    def __init__(
        self,
        curriculum_kg: EdWINKnowledgeGraph,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-3-5-sonnet-20241022",
        enable_caching: bool = True
    ):
        """
        Initialize tutoring engine

        Args:
            curriculum_kg: Curriculum knowledge graph
            llm_provider: LLM provider (anthropic, openai, ollama)
            llm_model: Model name
            enable_caching: Enable query caching (100x speedup)
        """
        self.curriculum_kg = curriculum_kg
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.enable_caching = enable_caching

        # Initialize RAG if available
        self.rag = None
        if HAS_RAG:
            config = Config.fused()  # Full HoloLoom features
            self.rag = SimpleRAG(
                config=config,
                llm_provider=llm_provider,
                llm_model=llm_model,
                enable_caching=enable_caching
            )

        self._content_ingested = False

    async def initialize(self, ingest_curriculum: bool = True):
        """
        Initialize tutor (ingest curriculum content)

        Args:
            ingest_curriculum: If True, ingest all curriculum into RAG
        """
        if not HAS_RAG:
            print("⚠️  RAG not available - using fallback mode")
            return

        if ingest_curriculum and not self._content_ingested:
            print("📚 Ingesting curriculum content into RAG...")

            # Ingest all objectives
            for obj in self.curriculum_kg.curriculum.objectives.values():
                content = self._format_objective_for_ingestion(obj)
                await self.rag.ingest(content)

            self._content_ingested = True
            print(f"✅ Ingested {len(self.curriculum_kg.curriculum.objectives)} objectives")

    async def answer_question(
        self,
        question: str,
        student_model: EdWINStudentModel,
        mode: str = "verify"
    ) -> TutoringResponse:
        """
        Answer student question with curriculum context

        Args:
            question: Student's question
            student_model: Student profile (grade, mastery, learning style)
            mode: Reasoning mode (direct, verify, research, plan_execute)

        Returns:
            TutoringResponse with answer and metadata
        """
        start_time = datetime.utcnow()

        # Build context
        context = self._build_context(question, student_model)

        # Query RAG
        if HAS_RAG and self.rag:
            result = await self.rag.query(
                f"{context}\n\nStudent Question: {question}",
                mode=mode
            )

            answer = result.response
            confidence = result.confidence
            sources = result.sources[:5]  # Top 5 sources
        else:
            # Fallback mode (no RAG)
            answer = await self._fallback_answer(question, student_model)
            confidence = 0.5
            sources = []

        # Calculate response time
        response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Get related objectives and next steps
        related = self._get_related_objectives(question, student_model)
        next_steps = self._get_next_steps(student_model)

        return TutoringResponse(
            question=question,
            answer=answer,
            confidence=confidence,
            sources=sources,
            reasoning_mode=mode,
            response_time_ms=response_time_ms,
            related_objectives=related,
            next_steps=next_steps
        )

    def _build_context(
        self,
        question: str,
        student: EdWINStudentModel
    ) -> str:
        """
        Build curriculum + student context for RAG

        Creates context string with:
        - Student grade level
        - Mastered concepts
        - Knowledge gaps (for target question)
        - Learning style preferences
        """
        # Get relevant objectives
        relevant_objs = self.curriculum_kg.retrieve_relevant(
            question,
            k=5,
            grade_filter=student.grade
        )

        # Get mastered concepts
        mastered = student.get_mastered_concepts()

        # Identify knowledge gaps
        gaps = []
        if relevant_objs:
            target_id = relevant_objs[0].id
            gaps = self.curriculum_kg.get_knowledge_gaps(target_id, mastered)

        # Format context
        context = f"""Student Profile:
- Grade: {student.grade}
- Mastered Concepts: {len(mastered)} objectives
- Current Focus: {', '.join(obj.title for obj in relevant_objs[:3])}

Relevant Curriculum:
{self._format_objectives_for_context(relevant_objs)}

Knowledge Gaps (missing prerequisites):
{', '.join(gaps[:5]) if gaps else 'None'}

Instructions:
- Explain at grade {student.grade} level
- Build on mastered concepts
- Address knowledge gaps if relevant
- Use clear, age-appropriate language
- Provide step-by-step explanations
"""

        # Add learning style preferences if available
        if hasattr(student, 'preferred_learning_style'):
            context += f"\n- Learning Style: {student.preferred_learning_style}\n"

        return context

    def _format_objective_for_ingestion(self, obj) -> str:
        """Format objective for RAG ingestion"""
        return f"""Learning Objective: {obj.title}
Grade: {obj.grade_level}
Subject: {obj.subject.value}
Bloom Level: {obj.bloom_level.name}

Description: {obj.description}

Keywords: {', '.join(obj.keywords)}
Standards: {', '.join(obj.standards)}
"""

    def _format_objectives_for_context(self, objectives: List) -> str:
        """Format objectives for context string"""
        formatted = []
        for obj in objectives:
            formatted.append(
                f"- {obj.title} (Grade {obj.grade_level}, {obj.bloom_level.name}): "
                f"{obj.description}"
            )
        return '\n'.join(formatted)

    async def _fallback_answer(
        self,
        question: str,
        student: EdWINStudentModel
    ) -> str:
        """
        Fallback answer when RAG is not available

        Provides basic guidance based on curriculum matching.
        """
        # Find relevant objectives
        relevant = self.curriculum_kg.retrieve_relevant(
            question,
            k=3,
            grade_filter=student.grade
        )

        if not relevant:
            return (
                f"I found this question relates to your grade {student.grade} curriculum. "
                "To get a detailed explanation, please ensure HoloLoom RAG is properly configured."
            )

        # Simple response with curriculum references
        answer = f"Based on your grade {student.grade} curriculum, this question relates to:\n\n"

        for i, obj in enumerate(relevant, 1):
            answer += f"{i}. **{obj.title}** - {obj.description}\n"

        answer += "\nFor a detailed step-by-step explanation, please configure the RAG system."

        return answer

    def _get_related_objectives(
        self,
        question: str,
        student: EdWINStudentModel
    ) -> List[str]:
        """Get related objective IDs"""
        relevant = self.curriculum_kg.retrieve_relevant(
            question,
            k=5,
            grade_filter=student.grade
        )
        return [obj.id for obj in relevant]

    def _get_next_steps(self, student: EdWINStudentModel) -> List[str]:
        """Get recommended next objectives"""
        next_objs = self.curriculum_kg.get_next_objectives(
            mastered_ids=list(student.get_mastered_concepts()),
            grade_filter=student.grade
        )
        return [obj.id for obj in next_objs[:3]]


# Helper functions

async def create_tutor(
    curriculum_kg: Optional[EdWINKnowledgeGraph] = None,
    llm_provider: str = "anthropic"
) -> EdWINTutor:
    """
    Create and initialize tutoring engine

    Args:
        curriculum_kg: Optional curriculum graph (creates new if None)
        llm_provider: LLM provider (anthropic, openai, ollama)

    Returns:
        Initialized EdWINTutor
    """
    if curriculum_kg is None:
        from .curriculum_graph import create_curriculum_graph
        curriculum_kg = await create_curriculum_graph()

    tutor = EdWINTutor(
        curriculum_kg=curriculum_kg,
        llm_provider=llm_provider
    )

    await tutor.initialize()

    return tutor
