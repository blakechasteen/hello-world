"""
AI Tutor - HoloLoom-powered intelligent tutoring system.

The AI tutor uses HoloLoom itself to answer questions about HoloLoom,
provide hints, and offer personalized learning recommendations.
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json

# HoloLoom imports (graceful degradation if not available)
try:
    from HoloLoom import HoloLoom
    from HoloLoom.config import Config
    from HoloLoom.documentation.types import Query, MemoryShard
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    print("⚠️  HoloLoom not available - AI tutor will use fallback mode")

from .lesson import LessonManager
from .progress import ProgressTracker


@dataclass
class TutorResponse:
    """Response from the AI tutor."""
    answer: str
    confidence: float
    hint: bool = False
    follow_up_questions: List[str] = None
    recommended_lessons: List[str] = None
    source_lessons: List[str] = None

    def __post_init__(self):
        if self.follow_up_questions is None:
            self.follow_up_questions = []
        if self.recommended_lessons is None:
            self.recommended_lessons = []
        if self.source_lessons is None:
            self.source_lessons = []


class AITutor:
    """
    AI-powered tutoring system using HoloLoom.

    Features:
    - Answers questions about HoloLoom concepts
    - Provides hints without spoiling answers
    - Recommends next lessons based on progress
    - Adapts to student performance
    - Uses HoloLoom's own memory system
    """

    def __init__(
        self,
        content_dir: Path,
        progress_tracker: Optional[ProgressTracker] = None,
        config_mode: str = "fast"
    ):
        self.content_dir = Path(content_dir)
        self.lesson_manager = LessonManager(content_dir)
        self.progress_tracker = progress_tracker or ProgressTracker()
        self.config_mode = config_mode

        # Initialize HoloLoom if available
        self.hololoom: Optional[HoloLoom] = None
        self.hololoom_ready = False

        # Knowledge base path
        self.kb_path = self.content_dir / "ai_tutor_kb.json"

        # Load or create knowledge base
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load pre-built knowledge base about HoloLoom."""
        if self.kb_path.exists():
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        else:
            # Create default knowledge base
            self.knowledge_base = self._create_default_kb()
            self._save_knowledge_base()

    def _save_knowledge_base(self):
        """Save knowledge base to disk."""
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, indent=2, fp=f)

    def _create_default_kb(self) -> Dict[str, Any]:
        """Create default knowledge base with HoloLoom concepts."""
        return {
            "concepts": {
                "memory_shard": {
                    "definition": "A single piece of knowledge or fact stored in HoloLoom's memory system",
                    "examples": ["Paris is the capital of France", "Thompson Sampling balances exploration"],
                    "related_lessons": ["beginner_02", "beginner_03"]
                },
                "knowledge_graph": {
                    "definition": "A network of connected memories with typed relationships",
                    "examples": ["IS_A, USES, MENTIONS relationships"],
                    "related_lessons": ["beginner_05"]
                },
                "thompson_sampling": {
                    "definition": "Bayesian algorithm for exploration-exploitation tradeoff",
                    "examples": ["Beta distribution sampling", "Multi-armed bandits"],
                    "related_lessons": ["beginner_06"]
                },
                "config_modes": {
                    "definition": "Three processing modes: BARE (fast), FAST (balanced), FUSED (quality)",
                    "examples": ["BARE: ~50ms", "FAST: ~150ms", "FUSED: ~300ms"],
                    "related_lessons": ["beginner_04"]
                },
                "weaving_orchestrator": {
                    "definition": "HoloLoom's main processing pipeline that weaves queries into responses",
                    "examples": ["9-step weaving cycle", "Pattern cards", "Spacetime fabric"],
                    "related_lessons": ["beginner_02", "beginner_07"]
                }
            },
            "common_questions": {
                "what_is_hololoom": {
                    "answer": "HoloLoom is an AI system with persistent memory that learns from every interaction. It combines knowledge graphs, embeddings, and Thompson Sampling for intelligent decision-making.",
                    "related_lessons": ["beginner_01"]
                },
                "how_to_start": {
                    "answer": "Start by creating memory shards and using the WeavingOrchestrator to process queries. See Lesson 2 for your first query example.",
                    "related_lessons": ["beginner_02"]
                },
                "why_async": {
                    "answer": "HoloLoom uses async/await for efficient I/O operations like database queries and API calls. This allows multiple operations to run concurrently.",
                    "related_lessons": ["beginner_08"]
                },
                "import_errors": {
                    "answer": "Import errors are usually fixed by setting PYTHONPATH=. before running your code. See the troubleshooting guide in Lesson 8.",
                    "related_lessons": ["beginner_08"]
                }
            }
        }

    async def initialize_hololoom(self):
        """Initialize HoloLoom instance with lesson content."""
        if not HOLOLOOM_AVAILABLE:
            print("⚠️  HoloLoom not available - using fallback knowledge base")
            return

        try:
            # Create HoloLoom instance
            config = Config.fast() if self.config_mode == "fast" else Config.fused()
            self.hololoom = HoloLoom(config=config)

            # Load all lesson content into HoloLoom's memory
            await self._ingest_lesson_content()

            self.hololoom_ready = True
            print("✅ AI Tutor initialized with HoloLoom")

        except Exception as e:
            print(f"⚠️  Failed to initialize HoloLoom: {e}")
            print("Using fallback knowledge base instead")

    async def _ingest_lesson_content(self):
        """Ingest all lesson content into HoloLoom's memory."""
        if not self.hololoom:
            return

        async with self.hololoom as loom:
            # Load all lessons
            for difficulty in ['beginner', 'intermediate', 'advanced']:
                lessons = self.lesson_manager.get_lessons_by_difficulty(difficulty)

                for lesson in lessons:
                    # Extract key concepts from lesson content
                    content = lesson.get('content', '')

                    # Create memory shard for each major section
                    sections = content.split('\n## ')
                    for section in sections:
                        if len(section.strip()) > 50:  # Skip very short sections
                            await loom.experience(
                                f"Lesson {lesson['id']}: {section[:500]}",
                                source=f"lesson_{lesson['id']}"
                            )

                    # Add quiz questions as memories
                    for quiz in lesson.get('quiz_questions', []):
                        question_text = quiz['question']
                        correct_idx = quiz['correct']
                        correct_answer = quiz['options'][correct_idx]

                        await loom.experience(
                            f"Q: {question_text} A: {correct_answer}",
                            source=f"quiz_{lesson['id']}"
                        )

    async def ask(
        self,
        question: str,
        current_lesson: Optional[str] = None,
        hint_mode: bool = False
    ) -> TutorResponse:
        """
        Ask the AI tutor a question.

        Args:
            question: Student's question
            current_lesson: Current lesson ID for context
            hint_mode: If True, provide hints instead of full answers

        Returns:
            TutorResponse with answer and metadata
        """
        # Check if question is in common questions
        for key, qa in self.knowledge_base['common_questions'].items():
            if key.replace('_', ' ') in question.lower():
                return TutorResponse(
                    answer=qa['answer'],
                    confidence=0.95,
                    hint=False,
                    source_lessons=qa['related_lessons']
                )

        # Check if question is about a known concept
        for concept, info in self.knowledge_base['concepts'].items():
            if concept.replace('_', ' ') in question.lower():
                if hint_mode:
                    answer = f"💡 Hint: Think about {info['definition'].lower()}. Check the examples in the relevant lessons."
                else:
                    answer = info['definition']

                return TutorResponse(
                    answer=answer,
                    confidence=0.9,
                    hint=hint_mode,
                    source_lessons=info['related_lessons']
                )

        # Use HoloLoom if available
        if self.hololoom_ready and self.hololoom:
            try:
                async with self.hololoom as loom:
                    memories = await loom.recall(question, limit=5)

                    if memories:
                        # Combine top memories into answer
                        answer_parts = []
                        sources = set()

                        for mem in memories[:3]:
                            answer_parts.append(mem.text)
                            sources.add(mem.source)

                        answer = "\n\n".join(answer_parts)

                        if hint_mode:
                            answer = f"💡 Hint: {answer[:200]}... (See the full concept in the lesson)"

                        # Extract lesson IDs from sources
                        source_lessons = [s.replace('lesson_', '').replace('quiz_', '')
                                        for s in sources if 'lesson' in s or 'quiz' in s]

                        return TutorResponse(
                            answer=answer,
                            confidence=0.85,
                            hint=hint_mode,
                            source_lessons=list(set(source_lessons))
                        )
            except Exception as e:
                print(f"⚠️  HoloLoom query failed: {e}")

        # Fallback response
        return TutorResponse(
            answer="I'm not sure about that. Try rephrasing your question or check the lesson content.",
            confidence=0.3,
            hint=False,
            follow_up_questions=[
                "What specific concept are you asking about?",
                "Which lesson are you currently working on?"
            ]
        )

    def get_personalized_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get personalized lesson recommendations based on progress.

        Returns:
            List of recommended lessons with reasons
        """
        progress = self.progress_tracker.get_progress()
        completed_lessons = set(progress.get('completed_lessons', []))
        current_level = progress.get('level', 1)

        recommendations = []

        # Get all available lessons
        all_lessons = []
        for difficulty in ['beginner', 'intermediate', 'advanced']:
            all_lessons.extend(self.lesson_manager.get_lessons_by_difficulty(difficulty))

        # Find next uncompleted lessons
        for lesson in all_lessons:
            lesson_id = lesson['id']

            if lesson_id not in completed_lessons:
                # Check if student is ready for this lesson
                required_level = lesson.get('required_level', 1)

                if current_level >= required_level:
                    recommendations.append({
                        'lesson_id': lesson_id,
                        'title': lesson['title'],
                        'reason': 'Next in sequence',
                        'difficulty': lesson.get('difficulty', 'beginner'),
                        'xp_reward': lesson.get('xp_reward', 50)
                    })

                    if len(recommendations) >= 3:
                        break

        return recommendations

    def get_hint_for_quiz(self, lesson_id: str, question_idx: int) -> str:
        """
        Get a hint for a specific quiz question.

        Args:
            lesson_id: Lesson identifier
            question_idx: Index of the quiz question

        Returns:
            Hint text
        """
        lesson = self.lesson_manager.get_lesson(lesson_id)
        if not lesson:
            return "Lesson not found."

        quiz_questions = lesson.get('quiz_questions', [])
        if question_idx >= len(quiz_questions):
            return "Question not found."

        question = quiz_questions[question_idx]

        # Generate hint based on question
        hints = {
            "what makes a good memory shard": "Think about focus - should one shard contain multiple unrelated facts?",
            "when to use bare mode": "Consider the tradeoff between speed and quality",
            "thompson sampling": "Think about the beta distribution and balancing exploration vs exploitation",
            "knowledge graph": "Think about how entities relate to each other",
            "async": "Think about why HoloLoom uses async/await for I/O operations"
        }

        question_text = question['question'].lower()

        for key, hint in hints.items():
            if key in question_text:
                return f"💡 Hint: {hint}"

        # Generic hint
        return f"💡 Hint: Review the lesson content for '{lesson['title']}' - the answer is explained there."

    async def close(self):
        """Clean up resources."""
        if self.hololoom:
            await self.hololoom.__aexit__(None, None, None)


# Convenience function
async def create_ai_tutor(content_dir: Path, **kwargs) -> AITutor:
    """Create and initialize an AI tutor instance."""
    tutor = AITutor(content_dir, **kwargs)
    await tutor.initialize_hololoom()
    return tutor
