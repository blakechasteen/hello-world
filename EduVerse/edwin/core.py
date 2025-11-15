"""
EdWIN Core - Main Integration

Ties together all EdWIN components into a unified AI tutor system.

**Implementation Date**: November 15, 2025
"""

from typing import Optional
from datetime import datetime

from .curriculum_graph import EdWINKnowledgeGraph, create_curriculum_graph
from .tutoring_engine import EdWINTutor, TutoringResponse
from .student_model import EdWINStudentModel, create_student
from .adaptive_difficulty import AdaptiveDifficultyEngine
from .safety_layer import EdWINSafetyLayer


class EdWIN:
    """
    EdWIN - Educational Weaving Intelligence Network

    Complete AI tutoring system with:
    - Curriculum knowledge graph (220+ objectives)
    - RAG-powered tutoring
    - Student progress tracking
    - Adaptive difficulty (Thompson Sampling)
    - K-12 safety guardrails

    Example:
        >>> edwin = EdWIN(
        ...     student_id="student_001",
        ...     student_name="Jane Doe",
        ...     grade=8
        ... )
        >>>
        >>> # Initialize (one-time setup)
        >>> await edwin.initialize()
        >>>
        >>> # Ask a question
        >>> result = await edwin.teach("What is a linear equation?")
        >>> print(result.answer)
        >>>
        >>> # Get next recommended lesson
        >>> suggestion = await edwin.suggest_next_lesson()
        >>> print(suggestion['title'])
    """

    def __init__(
        self,
        student_id: str,
        student_name: str,
        grade: int,
        llm_provider: str = "anthropic",
        enable_safety: bool = True
    ):
        """
        Initialize EdWIN

        Args:
            student_id: Unique student identifier
            student_name: Student name
            grade: Grade level (4-12)
            llm_provider: LLM provider (anthropic, openai, ollama)
            enable_safety: Enable K-12 safety guardrails
        """
        self.student_id = student_id
        self.student_name = student_name
        self.grade = grade
        self.llm_provider = llm_provider
        self.enable_safety = enable_safety

        # Components (initialized in initialize())
        self.curriculum_kg: Optional[EdWINKnowledgeGraph] = None
        self.tutor: Optional[EdWINTutor] = None
        self.student: Optional[EdWINStudentModel] = None
        self.adaptive_engine: Optional[AdaptiveDifficultyEngine] = None
        self.safety: Optional[EdWINSafetyLayer] = None

        self._initialized = False

    async def initialize(self):
        """
        Initialize EdWIN (one-time setup)

        Loads curriculum, creates student profile, initializes all components.
        """
        if self._initialized:
            print("✅ EdWIN already initialized")
            return

        print(f"🎓 Initializing EdWIN for {self.student_name} (Grade {self.grade})...\n")

        # 1. Create curriculum knowledge graph
        print("📚 Loading curriculum knowledge graph...")
        self.curriculum_kg = await create_curriculum_graph()

        # 2. Create student model
        print(f"👤 Creating student profile...")
        self.student = create_student(
            student_id=self.student_id,
            name=self.student_name,
            grade=self.grade
        )

        # 3. Create tutoring engine
        print(f"🤖 Initializing tutoring engine ({self.llm_provider})...")
        from .tutoring_engine import create_tutor
        self.tutor = await create_tutor(
            curriculum_kg=self.curriculum_kg,
            llm_provider=self.llm_provider
        )

        # 4. Create adaptive difficulty engine
        print("🎯 Initializing adaptive difficulty engine...")
        self.adaptive_engine = AdaptiveDifficultyEngine(
            student_model=self.student,
            curriculum_kg=self.curriculum_kg
        )

        # 5. Create safety layer (if enabled)
        if self.enable_safety:
            print("🛡️  Initializing K-12 safety guardrails...")
            self.safety = EdWINSafetyLayer(enable_human_in_loop=False)

        self._initialized = True
        print(f"\n✅ EdWIN initialized successfully!")
        print(f"   Student: {self.student_name}")
        print(f"   Grade: {self.grade}")
        print(f"   Curriculum: {len(self.curriculum_kg.curriculum.objectives)} objectives")
        print(f"   Safety: {'Enabled' if self.enable_safety else 'Disabled'}\n")

    async def teach(self, student_question: str) -> str:
        """
        Main teaching loop - answer student question

        Steps:
        1. Answer question with RAG
        2. Safety check (if enabled)
        3. Record interaction
        4. Update adaptive engine
        5. Return answer

        Args:
            student_question: Student's question

        Returns:
            Answer string (or error message if blocked)
        """
        if not self._initialized:
            await self.initialize()

        # 1. Answer question with RAG
        result = await self.tutor.answer_question(
            question=student_question,
            student_model=self.student,
            mode="verify"  # Always verify for accuracy
        )

        # 2. Safety check
        if self.enable_safety and self.safety:
            validation = await self.safety.validate_response(
                query=student_question,
                response=result.answer,
                grade_level=self.grade
            )

            if not validation.allowed:
                return f"🛡️  I can't answer that because: {validation.reason}"

        # 3. Record interaction
        # Infer objective from sources
        if result.sources:
            objective_id = result.sources[0]  # Primary source
            success = result.confidence > 0.7
            self.student.update_mastery(
                objective_id=objective_id,
                success=success,
                confidence=result.confidence
            )

            # 4. Update adaptive engine
            self.adaptive_engine.update_after_interaction(
                objective_id=objective_id,
                success=success,
                engagement=result.confidence  # Use confidence as engagement proxy
            )

        return result.answer

    async def suggest_next_lesson(self) -> dict:
        """
        Suggest optimal next learning objective

        Uses adaptive difficulty engine to select best challenge.

        Returns:
            Dict with lesson details (title, description, difficulty, etc.)
        """
        if not self._initialized:
            await self.initialize()

        # Get unlocked objectives
        available = self.curriculum_kg.get_next_objectives(
            mastered_ids=list(self.student.mastered_objectives),
            grade_filter=self.grade
        )

        if not available:
            return {
                "title": "Great work!",
                "description": "You've mastered all available objectives for your grade level!",
                "difficulty": 0.0
            }

        # Use adaptive engine to select best
        next_obj = self.adaptive_engine.select_next_objective(available)

        difficulty = self.adaptive_engine.estimate_difficulty(next_obj.id)
        expected_reward = self.adaptive_engine.expected_reward(next_obj.id)

        return {
            "id": next_obj.id,
            "title": next_obj.title,
            "description": next_obj.description,
            "estimated_time": next_obj.estimated_hours,
            "grade_level": next_obj.grade_level,
            "bloom_level": next_obj.bloom_level.name,
            "difficulty": difficulty,
            "expected_success": expected_reward,
            "keywords": next_obj.keywords
        }

    def get_progress_summary(self) -> dict:
        """
        Get student progress summary

        Returns:
            Dict with statistics (mastered, in_progress, recommendations, etc.)
        """
        if not self._initialized or not self.student:
            return {"error": "EdWIN not initialized"}

        return self.student.get_progress_summary(self.curriculum_kg)

    def save_progress(self, filepath: str):
        """
        Save student progress to file

        Args:
            filepath: Path to save student data
        """
        if not self.student:
            raise ValueError("EdWIN not initialized")

        self.student.save(filepath)

    @classmethod
    async def load_from_file(
        cls,
        filepath: str,
        llm_provider: str = "anthropic"
    ) -> "EdWIN":
        """
        Load student from saved file

        Args:
            filepath: Path to student data
            llm_provider: LLM provider

        Returns:
            EdWIN instance with loaded student
        """
        # Load student model
        student = EdWINStudentModel.load(filepath)

        # Create EdWIN instance
        edwin = cls(
            student_id=student.student_id,
            student_name=student.name,
            grade=student.grade,
            llm_provider=llm_provider
        )

        # Initialize
        await edwin.initialize()

        # Replace student model with loaded one
        edwin.student = student

        return edwin


# Helper function

async def create_edwin(
    student_id: str,
    student_name: str,
    grade: int,
    llm_provider: str = "anthropic"
) -> EdWIN:
    """
    Create and initialize EdWIN

    Args:
        student_id: Unique identifier
        student_name: Student name
        grade: Grade level (4-12)
        llm_provider: LLM provider

    Returns:
        Initialized EdWIN instance
    """
    edwin = EdWIN(
        student_id=student_id,
        student_name=student_name,
        grade=grade,
        llm_provider=llm_provider
    )

    await edwin.initialize()

    return edwin
