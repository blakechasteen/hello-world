"""
Player Model System - Student Progression Tracking

**Purpose**: Track student skills, knowledge, learning style, and pace
**Integration**: Uses HoloLoom Knowledge Graph for persistent storage
**Date**: November 13, 2025

This module implements comprehensive player modeling for adaptive learning:
- Skill tracking (mastery levels, XP, skill trees)
- Knowledge graph (concepts mastered, prerequisites)
- Learning style (visual, auditory, kinesthetic, social)
- Adaptive difficulty (Thompson Sampling for optimal challenge)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

# HoloLoom integration
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.policy.thompson_sampling import TSBandit


class LearningStyle(Enum):
    """Learning style preferences (VARK model)"""
    VISUAL = "visual"  # Prefer images, diagrams, videos
    AUDITORY = "auditory"  # Prefer lectures, discussions, audio
    READING_WRITING = "reading_writing"  # Prefer text, notes, writing
    KINESTHETIC = "kinesthetic"  # Prefer hands-on, experiments, movement
    SOCIAL = "social"  # Prefer collaboration, group work
    SOLITARY = "solitary"  # Prefer individual work, reflection


class SkillLevel(Enum):
    """Bloom's taxonomy levels"""
    NOVICE = 0  # No knowledge
    REMEMBER = 1  # Can recall facts
    UNDERSTAND = 2  # Can explain concepts
    APPLY = 3  # Can use in new contexts
    ANALYZE = 4  # Can break down into parts
    EVALUATE = 5  # Can make judgments
    CREATE = 6  # Can build new solutions


@dataclass
class Skill:
    """A specific skill with mastery level and XP"""
    id: str
    name: str
    subject: str  # math, science, ela, social_studies, ai_readiness
    description: str
    level: SkillLevel = SkillLevel.NOVICE
    xp: float = 0.0  # Experience points (0-100 per level)
    xp_to_next_level: float = 100.0
    prerequisites: List[str] = field(default_factory=list)  # Skill IDs
    last_practiced: Optional[datetime] = None
    practice_count: int = 0
    success_rate: float = 0.0  # 0.0 - 1.0


@dataclass
class Concept:
    """A knowledge concept (node in knowledge graph)"""
    id: str
    name: str
    subject: str
    grade_level: int  # 4-12
    description: str
    mastered: bool = False
    mastery_score: float = 0.0  # 0.0 - 1.0
    last_assessed: Optional[datetime] = None
    attempts: int = 0
    successes: int = 0


@dataclass
class PlayerStats:
    """Overall player statistics"""
    total_xp: float = 0.0
    level: int = 1  # Overall player level (1-100)
    quests_completed: int = 0
    quests_attempted: int = 0
    minigames_completed: int = 0
    total_play_time_minutes: int = 0
    streak_days: int = 0  # Consecutive days played
    last_login: Optional[datetime] = None
    achievements: List[str] = field(default_factory=list)


class PlayerModel:
    """
    Complete player model for a student

    Tracks:
    - Skills (Bloom's taxonomy levels + XP)
    - Knowledge graph (concepts mastered)
    - Learning style preferences
    - Adaptive difficulty (Thompson Sampling)
    - Performance metrics

    Integration with HoloLoom:
    - Knowledge graph (Neo4j) stores long-term memory
    - Thompson Sampling for optimal quest selection
    - Reflection Buffer for continuous improvement
    """

    def __init__(
        self,
        student_id: str,
        name: str,
        grade: int,
        knowledge_graph: Optional[KG] = None
    ):
        self.student_id = student_id
        self.name = name
        self.grade = grade

        # Skills and knowledge
        self.skills: Dict[str, Skill] = {}
        self.concepts: Dict[str, Concept] = {}

        # Learning profile
        self.learning_styles: Dict[LearningStyle, float] = {
            style: 0.5 for style in LearningStyle
        }  # Preferences (0.0 - 1.0)

        # Performance tracking
        self.stats = PlayerStats()

        # Adaptive systems
        self.difficulty_preference: float = 0.5  # 0.0 (easy) - 1.0 (hard)
        self.quest_bandit = TSBandit(n_arms=10)  # For quest selection

        # Knowledge graph integration
        self.kg = knowledge_graph or KG()

        # Metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    # ========== Skill Management ==========

    def add_skill(self, skill: Skill) -> None:
        """Add a new skill to track"""
        self.skills[skill.id] = skill
        self.updated_at = datetime.utcnow()

    def gain_xp(self, skill_id: str, xp_amount: float, success: bool = True) -> bool:
        """
        Award XP for a skill

        Returns:
            True if leveled up, False otherwise
        """
        if skill_id not in self.skills:
            return False

        skill = self.skills[skill_id]
        skill.xp += xp_amount
        skill.practice_count += 1
        skill.last_practiced = datetime.utcnow()

        # Update success rate (exponential moving average)
        alpha = 0.2
        skill.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * skill.success_rate

        # Check for level up
        leveled_up = False
        while skill.xp >= skill.xp_to_next_level:
            skill.xp -= skill.xp_to_next_level
            current_level_value = skill.level.value
            if current_level_value < len(SkillLevel) - 1:
                skill.level = SkillLevel(current_level_value + 1)
                leveled_up = True
                # Increase XP requirement for next level
                skill.xp_to_next_level *= 1.2

        # Update total XP
        self.stats.total_xp += xp_amount
        self.updated_at = datetime.utcnow()

        return leveled_up

    def get_skill_mastery(self, skill_id: str) -> float:
        """Get skill mastery level (0.0 - 1.0)"""
        if skill_id not in self.skills:
            return 0.0

        skill = self.skills[skill_id]
        # Combine level and XP progress
        level_progress = skill.level.value / len(SkillLevel)
        xp_progress = (skill.xp / skill.xp_to_next_level) * (1.0 / len(SkillLevel))
        return level_progress + xp_progress

    def get_unlocked_skills(self) -> List[Skill]:
        """Get skills where all prerequisites are mastered"""
        unlocked = []
        for skill in self.skills.values():
            if all(
                self.get_skill_mastery(prereq_id) >= 0.7
                for prereq_id in skill.prerequisites
            ):
                unlocked.append(skill)
        return unlocked

    # ========== Concept Mastery ==========

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to track"""
        self.concepts[concept.id] = concept

        # TODO: Add to knowledge graph once we implement proper KG integration
        # For now, concepts are tracked in memory only

        self.updated_at = datetime.utcnow()

    def update_concept_mastery(
        self,
        concept_id: str,
        score: float,
        success: bool = True
    ) -> None:
        """Update mastery for a concept based on assessment"""
        if concept_id not in self.concepts:
            return

        concept = self.concepts[concept_id]
        concept.attempts += 1
        if success:
            concept.successes += 1

        # Update mastery score (exponential moving average)
        alpha = 0.3
        concept.mastery_score = alpha * score + (1 - alpha) * concept.mastery_score

        # Mark as mastered if score consistently high
        if concept.mastery_score >= 0.8 and concept.attempts >= 3:
            concept.mastered = True
            # TODO: Update KG when we implement KG integration

        concept.last_assessed = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def get_mastered_concepts(self) -> List[Concept]:
        """Get all mastered concepts"""
        return [c for c in self.concepts.values() if c.mastered]

    def get_concepts_by_subject(self, subject: str) -> List[Concept]:
        """Get concepts for a specific subject"""
        return [c for c in self.concepts.values() if c.subject == subject]

    def get_next_concepts_to_learn(self, subject: str, limit: int = 5) -> List[Concept]:
        """
        Get recommended concepts to learn next

        Uses knowledge graph to find concepts where:
        - Prerequisites are mastered
        - Not yet mastered
        - Appropriate for grade level
        """
        candidates = []
        for concept in self.concepts.values():
            if (
                concept.subject == subject
                and not concept.mastered
                and concept.grade_level <= self.grade + 1  # Allow slightly ahead
            ):
                candidates.append(concept)

        # Sort by number of mastered prerequisites (descending)
        # TODO: Use Neo4j query for this (more efficient)
        candidates.sort(key=lambda c: c.mastery_score, reverse=True)

        return candidates[:limit]

    # ========== Learning Style ==========

    def update_learning_style(self, style: LearningStyle, weight: float) -> None:
        """Update preference for a learning style (exponential moving average)"""
        alpha = 0.2
        self.learning_styles[style] = alpha * weight + (1 - alpha) * self.learning_styles[style]
        self.updated_at = datetime.utcnow()

    def get_preferred_learning_styles(self, top_n: int = 2) -> List[LearningStyle]:
        """Get top N preferred learning styles"""
        sorted_styles = sorted(
            self.learning_styles.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [style for style, _ in sorted_styles[:top_n]]

    # ========== Adaptive Difficulty ==========

    def select_quest_difficulty(self, available_difficulties: List[float]) -> float:
        """
        Select optimal quest difficulty using Thompson Sampling

        Args:
            available_difficulties: List of difficulty levels (0.0 - 1.0)

        Returns:
            Selected difficulty level
        """
        # Map difficulties to bandit arms
        arm_index = self.quest_bandit.choose()
        difficulty_index = arm_index % len(available_difficulties)
        return available_difficulties[difficulty_index]

    def update_quest_outcome(self, difficulty: float, success: bool, score: float) -> None:
        """Update bandit after quest completion"""
        # Map difficulty to arm index
        arm_index = int(difficulty * 10)  # 0.0-1.0 → 0-10
        reward = score if success else 0.0
        self.quest_bandit.update(arm_index, reward)
        self.updated_at = datetime.utcnow()

    # ========== Statistics ==========

    def record_quest_completion(self, success: bool, duration_minutes: int) -> None:
        """Record quest completion statistics"""
        self.stats.quests_attempted += 1
        if success:
            self.stats.quests_completed += 1
        self.stats.total_play_time_minutes += duration_minutes
        self.updated_at = datetime.utcnow()

    def record_minigame_completion(self) -> None:
        """Record minigame completion"""
        self.stats.minigames_completed += 1
        self.updated_at = datetime.utcnow()

    def update_streak(self) -> None:
        """Update login streak (call on each login)"""
        now = datetime.utcnow()
        if self.stats.last_login:
            time_since_last = (now - self.stats.last_login).total_seconds()
            hours_since_last = time_since_last / 3600

            if hours_since_last < 48:  # Within 48 hours = streak continues
                self.stats.streak_days += 1
            else:  # Streak broken
                self.stats.streak_days = 1
        else:
            self.stats.streak_days = 1

        self.stats.last_login = now
        self.updated_at = datetime.utcnow()

    def add_achievement(self, achievement_id: str) -> None:
        """Award an achievement"""
        if achievement_id not in self.stats.achievements:
            self.stats.achievements.append(achievement_id)
            self.updated_at = datetime.utcnow()

    # ========== Serialization ==========

    def to_dict(self) -> Dict:
        """Serialize to dictionary (for API responses, storage)"""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "grade": self.grade,
            "skills": {
                skill_id: {
                    "name": skill.name,
                    "subject": skill.subject,
                    "level": skill.level.name,
                    "xp": skill.xp,
                    "mastery": self.get_skill_mastery(skill_id)
                }
                for skill_id, skill in self.skills.items()
            },
            "concepts": {
                concept_id: {
                    "name": concept.name,
                    "subject": concept.subject,
                    "mastered": concept.mastered,
                    "mastery_score": concept.mastery_score
                }
                for concept_id, concept in self.concepts.items()
            },
            "learning_styles": {
                style.value: weight
                for style, weight in self.learning_styles.items()
            },
            "stats": {
                "total_xp": self.stats.total_xp,
                "level": self.stats.level,
                "quests_completed": self.stats.quests_completed,
                "quests_attempted": self.stats.quests_attempted,
                "minigames_completed": self.stats.minigames_completed,
                "total_play_time_minutes": self.stats.total_play_time_minutes,
                "streak_days": self.stats.streak_days,
                "achievements": self.stats.achievements
            },
            "difficulty_preference": self.difficulty_preference,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlayerModel":
        """Deserialize from dictionary"""
        player = cls(
            student_id=data["student_id"],
            name=data["name"],
            grade=data["grade"]
        )

        # Restore skills
        for skill_id, skill_data in data.get("skills", {}).items():
            skill = Skill(
                id=skill_id,
                name=skill_data["name"],
                subject=skill_data["subject"],
                description="",
                level=SkillLevel[skill_data["level"]],
                xp=skill_data["xp"]
            )
            player.skills[skill_id] = skill

        # Restore concepts
        for concept_id, concept_data in data.get("concepts", {}).items():
            concept = Concept(
                id=concept_id,
                name=concept_data["name"],
                subject=concept_data["subject"],
                grade_level=player.grade,
                description="",
                mastered=concept_data["mastered"],
                mastery_score=concept_data["mastery_score"]
            )
            player.concepts[concept_id] = concept

        # Restore learning styles
        for style_str, weight in data.get("learning_styles", {}).items():
            style = LearningStyle(style_str)
            player.learning_styles[style] = weight

        # Restore stats
        stats_data = data.get("stats", {})
        player.stats.total_xp = stats_data.get("total_xp", 0.0)
        player.stats.level = stats_data.get("level", 1)
        player.stats.quests_completed = stats_data.get("quests_completed", 0)
        player.stats.quests_attempted = stats_data.get("quests_attempted", 0)
        player.stats.minigames_completed = stats_data.get("minigames_completed", 0)
        player.stats.total_play_time_minutes = stats_data.get("total_play_time_minutes", 0)
        player.stats.streak_days = stats_data.get("streak_days", 0)
        player.stats.achievements = stats_data.get("achievements", [])

        player.difficulty_preference = data.get("difficulty_preference", 0.5)

        return player

    def __repr__(self) -> str:
        mastered_count = len(self.get_mastered_concepts())
        total_concepts = len(self.concepts)
        return (
            f"PlayerModel(student={self.name}, grade={self.grade}, "
            f"xp={self.stats.total_xp:.0f}, concepts_mastered={mastered_count}/{total_concepts})"
        )


# ========== Helper Functions ==========

def create_sample_player(student_id: str = "student_123") -> PlayerModel:
    """Create a sample player for testing"""
    player = PlayerModel(
        student_id=student_id,
        name="Alice Johnson",
        grade=8
    )

    # Add sample math skills
    algebra_basics = Skill(
        id="math_algebra_basics",
        name="Algebra Basics",
        subject="math",
        description="Solve linear equations, graph functions",
        level=SkillLevel.UNDERSTAND,
        xp=45.0,
        prerequisites=[]
    )
    player.add_skill(algebra_basics)

    # Add sample concepts
    linear_equations = Concept(
        id="math_linear_equations",
        name="Linear Equations",
        subject="math",
        grade_level=8,
        description="Solve equations like 2x + 5 = 13",
        mastered=True,
        mastery_score=0.92,
        attempts=10,
        successes=9
    )
    player.add_concept(linear_equations)

    quadratic_equations = Concept(
        id="math_quadratic_equations",
        name="Quadratic Equations",
        subject="math",
        grade_level=9,
        description="Solve equations like x^2 + 3x - 4 = 0",
        mastered=False,
        mastery_score=0.35,
        attempts=3,
        successes=1
    )
    player.add_concept(quadratic_equations)

    # Set learning style preferences
    player.update_learning_style(LearningStyle.VISUAL, 0.8)
    player.update_learning_style(LearningStyle.KINESTHETIC, 0.6)

    return player


if __name__ == "__main__":
    # Demo usage
    print("=== EduVerse Player Model Demo ===\n")

    player = create_sample_player()
    print(f"Created: {player}\n")

    # Gain XP
    print("Gaining XP for algebra basics...")
    leveled_up = player.gain_xp("math_algebra_basics", 60, success=True)
    if leveled_up:
        print(f"✨ Level up! Now at: {player.skills['math_algebra_basics'].level.name}")

    # Get mastery
    mastery = player.get_skill_mastery("math_algebra_basics")
    print(f"Mastery: {mastery:.2%}\n")

    # Get next concepts
    print("Next concepts to learn (math):")
    next_concepts = player.get_next_concepts_to_learn("math", limit=3)
    for concept in next_concepts:
        print(f"  - {concept.name} (mastery: {concept.mastery_score:.2%})")

    # Preferred learning styles
    print(f"\nPreferred learning styles: {[s.value for s in player.get_preferred_learning_styles()]}")

    # Serialize
    print("\n=== Serialization ===")
    player_dict = player.to_dict()
    print(f"Serialized to dict with keys: {list(player_dict.keys())}")

    # Deserialize
    player_restored = PlayerModel.from_dict(player_dict)
    print(f"Restored: {player_restored}")
    print(f"\nMastery preserved: {player_restored.get_skill_mastery('math_algebra_basics'):.2%}")
