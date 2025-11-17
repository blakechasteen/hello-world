"""
EdWIN Tutor - Core Lesson Engine

Defines lesson structure and management for all learning interfaces.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json
from pathlib import Path


class LessonType(Enum):
    """Types of lessons EdWIN can teach."""
    CONCEPT = "concept"  # Theory/explanation
    CODE_ALONG = "code_along"  # Follow along coding
    CHALLENGE = "challenge"  # Solve it yourself
    QUIZ = "quiz"  # Multiple choice/true-false
    PROJECT = "project"  # Build something real


class Difficulty(Enum):
    """Lesson difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class Hint:
    """Progressive hint system - reveals more over time."""
    level: int  # 1 = subtle, 2 = direct, 3 = almost the answer
    text: str
    code_snippet: Optional[str] = None


@dataclass
class Challenge:
    """A coding or conceptual challenge for the learner."""
    description: str
    starter_code: Optional[str] = None
    solution: Optional[str] = None  # For validation
    validation_fn: Optional[Callable] = None  # Custom validation
    hints: List[Hint] = field(default_factory=list)
    points: int = 10


@dataclass
class Lesson:
    """A single learning unit in EdWIN."""
    id: str
    title: str
    description: str
    lesson_type: LessonType
    difficulty: Difficulty

    # Content
    content: str  # Markdown-formatted explanation
    code_examples: List[str] = field(default_factory=list)

    # Interactivity
    challenges: List[Challenge] = field(default_factory=list)
    quiz_questions: List[Dict[str, Any]] = field(default_factory=list)

    # Learning path
    prerequisites: List[str] = field(default_factory=list)  # Lesson IDs
    next_lessons: List[str] = field(default_factory=list)

    # Gamification
    xp_reward: int = 50
    badges: List[str] = field(default_factory=list)  # Badge IDs earned

    # Meta
    estimated_time: int = 10  # minutes
    tags: List[str] = field(default_factory=list)


class LessonManager:
    """Manages loading, validating, and serving lessons."""

    def __init__(self, content_dir: Path):
        self.content_dir = Path(content_dir)
        self.lessons: Dict[str, Lesson] = {}
        self.load_lessons()

    def load_lessons(self):
        """Load all lessons from content directory."""
        for json_file in self.content_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    lesson = self._deserialize_lesson(data)
                    self.lessons[lesson.id] = lesson
            except Exception as e:
                print(f"⚠️  Failed to load {json_file}: {e}")

    def _deserialize_lesson(self, data: Dict[str, Any]) -> Lesson:
        """Convert JSON to Lesson object."""
        # Convert enums
        data['lesson_type'] = LessonType(data['lesson_type'])
        data['difficulty'] = Difficulty(data['difficulty'])

        # Convert challenges
        if 'challenges' in data:
            challenges = []
            for ch in data['challenges']:
                hints = [Hint(**h) for h in ch.get('hints', [])]
                ch['hints'] = hints
                challenges.append(Challenge(**ch))
            data['challenges'] = challenges

        return Lesson(**data)

    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """Retrieve a lesson by ID."""
        return self.lessons.get(lesson_id)

    def get_lessons_by_difficulty(self, difficulty: Difficulty) -> List[Lesson]:
        """Get all lessons of a certain difficulty."""
        return [l for l in self.lessons.values() if l.difficulty == difficulty]

    def get_next_lessons(self, current_lesson_id: str, completed: List[str]) -> List[Lesson]:
        """Get recommended next lessons based on progress."""
        current = self.get_lesson(current_lesson_id)
        if not current:
            return []

        # Get explicitly defined next lessons
        next_lessons = [self.get_lesson(lid) for lid in current.next_lessons]
        next_lessons = [l for l in next_lessons if l is not None]

        # Filter out lessons with incomplete prerequisites
        available = []
        for lesson in next_lessons:
            prereqs_met = all(p in completed for p in lesson.prerequisites)
            if prereqs_met:
                available.append(lesson)

        return available

    def validate_challenge(self, lesson_id: str, challenge_idx: int,
                          user_code: str) -> Dict[str, Any]:
        """Validate a user's challenge submission."""
        lesson = self.get_lesson(lesson_id)
        if not lesson or challenge_idx >= len(lesson.challenges):
            return {"success": False, "error": "Challenge not found"}

        challenge = lesson.challenges[challenge_idx]

        # Use custom validation function if provided
        if challenge.validation_fn:
            try:
                result = challenge.validation_fn(user_code)
                return {
                    "success": result,
                    "message": "Correct!" if result else "Not quite right. Try again!",
                    "points": challenge.points if result else 0
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Simple string comparison fallback
        if challenge.solution:
            # Normalize whitespace
            user_normalized = ' '.join(user_code.split())
            solution_normalized = ' '.join(challenge.solution.split())
            success = user_normalized == solution_normalized

            return {
                "success": success,
                "message": "Perfect!" if success else "Close, but not exactly right.",
                "points": challenge.points if success else 0
            }

        # No validation method - assume correct
        return {"success": True, "message": "Submitted!", "points": challenge.points}
