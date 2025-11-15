"""
Student Progress Tracking

Enhanced student model extending EduVerse PlayerModel with EdWIN-specific features.

**Key Features**:
- Mastery tracking (220+ objectives)
- Personal knowledge graph
- Learning style adaptation (VARK model)
- Reflection history
- Knowledge gap detection

**Implementation Date**: November 15, 2025
"""

from typing import Set, List, Dict, Optional
from datetime import datetime
from pathlib import Path

from EduVerse.education.player_model import PlayerModel, Skill, SkillLevel
from HoloLoom.memory.graph import KG

try:
    from HoloLoom.reflection.buffer import ReflectionBuffer
    HAS_REFLECTION = True
except ImportError:
    HAS_REFLECTION = False
    print("⚠️  Reflection buffer not available")


class EdWINStudentModel:
    """
    Enhanced Student Model for EdWIN

    Extends PlayerModel with:
    - Mastered objectives tracking
    - Personal knowledge graph
    - Learning style preferences
    - Reflection history
    - Knowledge gap analysis

    Example:
        >>> student = EdWINStudentModel(
        ...     student_id="student_001",
        ...     name="Jane Doe",
        ...     grade=8
        ... )
        >>>
        >>> # Update mastery
        >>> leveled_up = student.update_mastery(
        ...     objective_id="math.algebra.8.linear_equations",
        ...     success=True,
        ...     confidence=0.85
        ... )
        >>>
        >>> # Get recommendations
        >>> next_objs = student.get_recommended_next(curriculum_kg)
    """

    def __init__(
        self,
        student_id: str,
        name: str,
        grade: int,
        data_dir: Optional[Path] = None
    ):
        """
        Initialize student model

        Args:
            student_id: Unique student identifier
            name: Student name
            grade: Grade level (4-12)
            data_dir: Optional data directory for persistence
        """
        # Core PlayerModel
        self.player_model = PlayerModel(
            student_id=student_id,
            name=name,
            grade=grade
        )

        # EdWIN extensions
        self.student_id = student_id
        self.name = name
        self.grade = grade

        # Mastery tracking
        self.mastered_objectives: Set[str] = set()
        self.in_progress_objectives: Set[str] = set()
        self.mastery_scores: Dict[str, float] = {}  # objective_id → mastery (0-1)

        # Personal knowledge graph (student's view of curriculum)
        self.personal_kg = KG()

        # Reflection buffer (learning history)
        self.reflection_buffer = None
        if HAS_REFLECTION:
            data_dir = data_dir or Path("./student_data")
            data_dir.mkdir(parents=True, exist_ok=True)

            self.reflection_buffer = ReflectionBuffer(
                capacity=1000,
                persist_path=str(data_dir / f"{student_id}_reflections")
            )

        # Learning preferences (VARK model)
        from EduVerse.education.player_model import LearningStyle
        self.learning_style_scores: Dict[LearningStyle, float] = {
            LearningStyle.VISUAL: 0.5,
            LearningStyle.AUDITORY: 0.5,
            LearningStyle.READING_WRITING: 0.5,
            LearningStyle.KINESTHETIC: 0.5,
            LearningStyle.SOCIAL: 0.5,
            LearningStyle.SOLITARY: 0.5,
        }

        # Difficulty preference (0.0 = easy, 1.0 = hard)
        self.difficulty_preference = 0.5

        # Metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def update_mastery(
        self,
        objective_id: str,
        success: bool,
        confidence: float
    ) -> bool:
        """
        Update mastery for an objective

        Args:
            objective_id: Learning objective ID
            success: Whether attempt was successful
            confidence: Confidence score (0.0-1.0)

        Returns:
            True if newly mastered (crossed threshold), False otherwise
        """
        # Get current mastery score
        current_mastery = self.mastery_scores.get(objective_id, 0.0)

        # Calculate new mastery (exponential moving average)
        alpha = 0.3  # Learning rate
        new_score = alpha * (confidence if success else 0.0) + (1 - alpha) * current_mastery

        # Update mastery score
        self.mastery_scores[objective_id] = new_score

        # Update mastered set (threshold: 0.8)
        was_mastered = objective_id in self.mastered_objectives
        is_now_mastered = new_score >= 0.8

        if is_now_mastered and not was_mastered:
            self.mastered_objectives.add(objective_id)
            self.in_progress_objectives.discard(objective_id)
            self.updated_at = datetime.utcnow()
            return True  # Newly mastered!

        elif not is_now_mastered and was_mastered:
            # Dropped below threshold (forgotten?)
            self.mastered_objectives.discard(objective_id)
            self.in_progress_objectives.add(objective_id)

        elif not is_now_mastered:
            # In progress
            self.in_progress_objectives.add(objective_id)

        self.updated_at = datetime.utcnow()
        return False

    def get_mastered_concepts(self) -> Set[str]:
        """Get set of mastered objective IDs"""
        return self.mastered_objectives.copy()

    def get_mastery_score(self, objective_id: str) -> float:
        """
        Get mastery score for an objective

        Returns:
            Float 0.0-1.0 (0.8+ = mastered)
        """
        return self.mastery_scores.get(objective_id, 0.0)

    def get_recommended_next(
        self,
        curriculum_kg,
        count: int = 5
    ) -> List:
        """
        Get recommended next objectives

        Uses curriculum KG to find objectives where:
        - All prerequisites are mastered
        - Grade appropriate (±1 grade)
        - Not yet mastered

        Args:
            curriculum_kg: Curriculum knowledge graph
            count: Number of recommendations

        Returns:
            List of recommended learning objectives
        """
        next_objs = curriculum_kg.get_next_objectives(
            mastered_ids=list(self.mastered_objectives),
            grade_filter=self.grade
        )

        # Filter out already mastered
        next_objs = [
            obj for obj in next_objs
            if obj.id not in self.mastered_objectives
        ]

        return next_objs[:count]

    def get_knowledge_gaps(
        self,
        target_objective_id: str,
        curriculum_kg
    ) -> List[str]:
        """
        Identify missing prerequisites for a target objective

        Args:
            target_objective_id: Target objective ID
            curriculum_kg: Curriculum knowledge graph

        Returns:
            List of missing prerequisite IDs
        """
        return curriculum_kg.get_knowledge_gaps(
            target_objective_id,
            self.mastered_objectives
        )

    def get_frontier(self) -> List[str]:
        """
        Get learning frontier (edge of knowledge)

        Returns objectives at the boundary of mastered knowledge -
        concepts that are unlocked but not yet mastered.

        Returns:
            List of frontier objective IDs
        """
        return list(self.in_progress_objectives)

    def get_progress_summary(self, curriculum_kg) -> Dict:
        """
        Get progress summary

        Returns:
            Dict with statistics (mastered, in_progress, total, etc.)
        """
        total_objectives = len(curriculum_kg.curriculum.objectives)
        mastered_count = len(self.mastered_objectives)
        in_progress_count = len(self.in_progress_objectives)

        # Get next recommended
        recommended = self.get_recommended_next(curriculum_kg, count=3)

        return {
            "student_id": self.student_id,
            "name": self.name,
            "grade": self.grade,
            "total_count": total_objectives,
            "mastered_count": mastered_count,
            "in_progress_count": in_progress_count,
            "mastery_percentage": (mastered_count / total_objectives) * 100,
            "recommended": recommended,
            "total_xp": self.player_model.stats.total_xp,
            "level": self.player_model.stats.level,
            "streak_days": self.player_model.stats.streak_days
        }

    @property
    def preferred_learning_style(self) -> str:
        """Get preferred learning style (highest score)"""
        max_style = max(
            self.learning_style_scores.items(),
            key=lambda x: x[1]
        )
        return max_style[0].value

    def save(self, filepath: str):
        """
        Save student model to file

        Args:
            filepath: Path to save (JSON format)
        """
        import json

        data = {
            "student_id": self.student_id,
            "name": self.name,
            "grade": self.grade,
            "mastered_objectives": list(self.mastered_objectives),
            "in_progress_objectives": list(self.in_progress_objectives),
            "mastery_scores": self.mastery_scores,
            "difficulty_preference": self.difficulty_preference,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved student model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EdWINStudentModel":
        """
        Load student model from file

        Args:
            filepath: Path to load from

        Returns:
            Loaded EdWINStudentModel
        """
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        student = cls(
            student_id=data["student_id"],
            name=data["name"],
            grade=data["grade"]
        )

        student.mastered_objectives = set(data["mastered_objectives"])
        student.in_progress_objectives = set(data["in_progress_objectives"])
        student.mastery_scores = data["mastery_scores"]
        student.difficulty_preference = data["difficulty_preference"]

        print(f"📂 Loaded student model from {filepath}")

        return student


# Helper functions

def create_student(
    student_id: str,
    name: str,
    grade: int
) -> EdWINStudentModel:
    """
    Create new student model

    Args:
        student_id: Unique identifier
        name: Student name
        grade: Grade level (4-12)

    Returns:
        New EdWINStudentModel
    """
    return EdWINStudentModel(
        student_id=student_id,
        name=name,
        grade=grade
    )
