"""
Quest Engine - Dynamic Quest Generation with Adaptive Difficulty

**Purpose**: Generate personalized learning quests based on player state
**Date**: November 13, 2025
**Integration**: Player Model + Curriculum + Thompson Sampling

This module implements the core quest system:
- 5 quest types (Tutorial, Practice, Challenge, Assessment, Exploration)
- Dynamic generation from templates
- Adaptive difficulty via Thompson Sampling
- Learning objective mapping
- Progress tracking and rewards
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4
import random

# EduVerse imports
from EduVerse.education.player_model import PlayerModel, SkillLevel
from EduVerse.education.curriculum import CurriculumFramework, LearningObjective, Subject


class QuestType(Enum):
    """Types of quests"""
    TUTORIAL = "tutorial"        # Introduce new concepts (easy, guided)
    PRACTICE = "practice"        # Reinforce learned concepts (medium)
    CHALLENGE = "challenge"      # Test mastery (hard, creative)
    ASSESSMENT = "assessment"    # Formal evaluation (medium-hard)
    EXPLORATION = "exploration"  # Free discovery (variable difficulty)


class QuestState(Enum):
    """Quest completion states"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class DifficultyLevel(Enum):
    """Quest difficulty levels"""
    VERY_EASY = 0.2
    EASY = 0.4
    MEDIUM = 0.6
    HARD = 0.8
    VERY_HARD = 1.0


@dataclass
class QuestReward:
    """Rewards for completing a quest"""
    xp: float = 100.0
    skill_xp: Dict[str, float] = field(default_factory=dict)  # skill_id → XP
    achievements: List[str] = field(default_factory=list)
    unlocks: List[str] = field(default_factory=list)  # Unlocked content IDs


@dataclass
class Quest:
    """
    A single quest (learning activity)

    Example:
        Quest(
            id="quest_algebra_linear_001",
            title="The Equation Adventure",
            description="Help the wizard solve linear equations to unlock the magic portal",
            quest_type=QuestType.PRACTICE,
            difficulty=0.6,
            learning_objectives=["math.algebra.8.linear_equations"],
            minigame_id="quiz_algebra_linear",
            estimated_minutes=15,
            rewards=QuestReward(xp=150, skill_xp={"math_algebra_basics": 50})
        )
    """
    id: str
    title: str
    description: str
    quest_type: QuestType
    difficulty: float  # 0.0 - 1.0
    learning_objectives: List[str]  # Objective IDs from curriculum
    minigame_id: str  # Which minigame to play
    minigame_config: Dict[str, Any] = field(default_factory=dict)
    estimated_minutes: int = 10
    rewards: QuestReward = field(default_factory=QuestReward)

    # State
    state: QuestState = QuestState.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    score: float = 0.0  # Last attempt score (0.0 - 1.0)
    best_score: float = 0.0
    time_spent_seconds: int = 0

    # Metadata
    subject: Optional[Subject] = None
    grade_level: Optional[int] = None
    prerequisite_quests: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def start(self) -> None:
        """Start the quest"""
        self.state = QuestState.IN_PROGRESS
        self.started_at = datetime.utcnow()
        self.attempts += 1

    def complete(self, score: float, time_seconds: int) -> QuestReward:
        """
        Complete the quest

        Args:
            score: Final score (0.0 - 1.0)
            time_seconds: Time spent

        Returns:
            Quest rewards
        """
        self.state = QuestState.COMPLETED
        self.completed_at = datetime.utcnow()
        self.score = score
        self.best_score = max(self.best_score, score)
        self.time_spent_seconds += time_seconds

        # Scale rewards by score
        scaled_rewards = QuestReward(
            xp=self.rewards.xp * score,
            skill_xp={k: v * score for k, v in self.rewards.skill_xp.items()},
            achievements=self.rewards.achievements if score >= 0.8 else [],
            unlocks=self.rewards.unlocks if score >= 0.7 else []
        )

        return scaled_rewards

    def fail(self) -> None:
        """Fail the quest (score below passing threshold)"""
        self.state = QuestState.FAILED

    def abandon(self) -> None:
        """Abandon the quest (quit mid-way)"""
        self.state = QuestState.ABANDONED

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "quest_type": self.quest_type.value,
            "difficulty": self.difficulty,
            "learning_objectives": self.learning_objectives,
            "minigame_id": self.minigame_id,
            "minigame_config": self.minigame_config,
            "estimated_minutes": self.estimated_minutes,
            "rewards": {
                "xp": self.rewards.xp,
                "skill_xp": self.rewards.skill_xp,
                "achievements": self.rewards.achievements,
                "unlocks": self.rewards.unlocks
            },
            "state": self.state.value,
            "score": self.score,
            "best_score": self.best_score,
            "attempts": self.attempts,
            "subject": self.subject.value if self.subject else None,
            "grade_level": self.grade_level
        }


@dataclass
class QuestTemplate:
    """Template for generating quests"""
    id: str
    title_template: str  # e.g., "Solve {topic} Problems"
    description_template: str
    quest_type: QuestType
    subject: Subject
    difficulty_range: tuple = (0.3, 0.7)  # (min, max)
    minigame_type: str = "quiz"
    base_xp: float = 100.0
    estimated_minutes: int = 10


class QuestEngine:
    """
    Quest generation and management engine

    Responsibilities:
    - Generate quests from templates + player state
    - Adaptive difficulty via Thompson Sampling
    - Track quest progress
    - Award rewards
    """

    def __init__(
        self,
        curriculum: CurriculumFramework,
        player_model: PlayerModel
    ):
        self.curriculum = curriculum
        self.player = player_model

        # Quest storage
        self.active_quests: Dict[str, Quest] = {}
        self.completed_quests: Dict[str, Quest] = {}
        self.available_quests: List[Quest] = []

        # Templates
        self.templates: Dict[str, QuestTemplate] = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize quest templates"""

        # Math templates
        self.templates["math_tutorial"] = QuestTemplate(
            id="math_tutorial",
            title_template="Learn: {topic}",
            description_template="Introduction to {topic}. Complete guided examples.",
            quest_type=QuestType.TUTORIAL,
            subject=Subject.MATH,
            difficulty_range=(0.2, 0.4),
            minigame_type="tutorial_quiz",
            base_xp=75.0,
            estimated_minutes=8
        )

        self.templates["math_practice"] = QuestTemplate(
            id="math_practice",
            title_template="{topic} Practice",
            description_template="Practice {topic} with {count} problems.",
            quest_type=QuestType.PRACTICE,
            subject=Subject.MATH,
            difficulty_range=(0.4, 0.7),
            minigame_type="practice_quiz",
            base_xp=100.0,
            estimated_minutes=12
        )

        self.templates["math_challenge"] = QuestTemplate(
            id="math_challenge",
            title_template="Master {topic}",
            description_template="Advanced {topic} challenge. Show mastery!",
            quest_type=QuestType.CHALLENGE,
            subject=Subject.MATH,
            difficulty_range=(0.7, 0.95),
            minigame_type="challenge_quiz",
            base_xp=200.0,
            estimated_minutes=20
        )

        # Science templates
        self.templates["science_tutorial"] = QuestTemplate(
            id="science_tutorial",
            title_template="Discover: {topic}",
            description_template="Explore {topic} through interactive experiments.",
            quest_type=QuestType.TUTORIAL,
            subject=Subject.SCIENCE,
            difficulty_range=(0.2, 0.4),
            minigame_type="simulation",
            base_xp=80.0,
            estimated_minutes=10
        )

        self.templates["science_practice"] = QuestTemplate(
            id="science_practice",
            title_template="{topic} Lab",
            description_template="Conduct experiments to understand {topic}.",
            quest_type=QuestType.PRACTICE,
            subject=Subject.SCIENCE,
            difficulty_range=(0.4, 0.7),
            minigame_type="lab_simulation",
            base_xp=120.0,
            estimated_minutes=15
        )

        # AI templates
        self.templates["ai_tutorial"] = QuestTemplate(
            id="ai_tutorial",
            title_template="AI Basics: {topic}",
            description_template="Learn how {topic} works in AI systems.",
            quest_type=QuestType.TUTORIAL,
            subject=Subject.AI_READINESS,
            difficulty_range=(0.3, 0.5),
            minigame_type="interactive_demo",
            base_xp=90.0,
            estimated_minutes=12
        )

        self.templates["ai_practice"] = QuestTemplate(
            id="ai_practice",
            title_template="Build: {topic}",
            description_template="Create your own {topic} project.",
            quest_type=QuestType.PRACTICE,
            subject=Subject.AI_READINESS,
            difficulty_range=(0.5, 0.8),
            minigame_type="code_challenge",
            base_xp=150.0,
            estimated_minutes=18
        )

    def generate_quest(
        self,
        objective: LearningObjective,
        difficulty: Optional[float] = None
    ) -> Quest:
        """
        Generate a quest for a learning objective

        Args:
            objective: Learning objective to target
            difficulty: Override difficulty (uses adaptive if None)

        Returns:
            Generated quest
        """
        # Select appropriate template
        template_key = f"{objective.subject.value}_practice"  # Default to practice
        if objective.bloom_level.value <= 2:  # Remember/Understand
            template_key = f"{objective.subject.value}_tutorial"
        elif objective.bloom_level.value >= 5:  # Evaluate/Create
            template_key = f"{objective.subject.value}_challenge"

        template = self.templates.get(template_key)
        if not template:
            # Fallback to generic practice
            template = list(self.templates.values())[0]

        # Determine difficulty
        if difficulty is None:
            # Use Thompson Sampling for adaptive difficulty
            available_difficulties = [
                DifficultyLevel.EASY.value,
                DifficultyLevel.MEDIUM.value,
                DifficultyLevel.HARD.value
            ]
            difficulty = self.player.select_quest_difficulty(available_difficulties)

        # Clamp to template range
        min_diff, max_diff = template.difficulty_range
        difficulty = max(min_diff, min(max_diff, difficulty))

        # Generate quest
        quest = Quest(
            id=f"quest_{objective.id}_{uuid4().hex[:8]}",
            title=template.title_template.format(topic=objective.title),
            description=template.description_template.format(
                topic=objective.title,
                count=random.randint(5, 10)
            ),
            quest_type=template.quest_type,
            difficulty=difficulty,
            learning_objectives=[objective.id],
            minigame_id=f"{template.minigame_type}_{objective.subject.value}",
            minigame_config={
                "objective_id": objective.id,
                "difficulty": difficulty,
                "bloom_level": objective.bloom_level.value
            },
            estimated_minutes=template.estimated_minutes,
            rewards=QuestReward(
                xp=template.base_xp * (1 + difficulty * 0.5),  # Scale by difficulty
                skill_xp={f"{objective.subject.value}_basics": 50 * difficulty}
            ),
            subject=objective.subject,
            grade_level=objective.grade_level
        )

        return quest

    def generate_recommended_quests(
        self,
        count: int = 5
    ) -> List[Quest]:
        """
        Generate recommended quests for the player

        Uses:
        - Player's current skill level
        - Mastered objectives
        - Learning path recommendations
        - Adaptive difficulty

        Returns:
            List of recommended quests
        """
        # Get mastered objectives from player
        mastered_concepts = self.player.get_mastered_concepts()
        mastered_ids = [c.id for c in mastered_concepts]

        # Get next objectives from curriculum
        next_objectives = self.curriculum.get_next_objectives(mastered_ids)

        # Filter by player's grade level (allow ±1 grade)
        grade_filtered = [
            obj for obj in next_objectives
            if abs(obj.grade_level - self.player.grade) <= 1
        ]

        # Prioritize by:
        # 1. Player's preferred learning styles
        # 2. Objectives with more mastered prerequisites (closer to ready)
        # 3. Player's subject preferences (from stats)

        def objective_score(obj: LearningObjective) -> float:
            score = 0.0

            # More mastered prerequisites = higher priority
            prereqs_mastered = sum(1 for p in obj.prerequisites if p in mastered_ids)
            total_prereqs = len(obj.prerequisites)
            if total_prereqs > 0:
                score += (prereqs_mastered / total_prereqs) * 10

            # Match grade level (prefer at-level)
            grade_diff = abs(obj.grade_level - self.player.grade)
            score += (2 - grade_diff) * 5

            # Randomness for variety
            score += random.uniform(0, 2)

            return score

        # Sort and take top N
        sorted_objectives = sorted(grade_filtered, key=objective_score, reverse=True)
        top_objectives = sorted_objectives[:count]

        # Generate quests
        recommended_quests = []
        for obj in top_objectives:
            quest = self.generate_quest(obj)
            recommended_quests.append(quest)

        return recommended_quests

    def start_quest(self, quest_id: str) -> bool:
        """
        Start a quest

        Returns:
            True if started, False if not available
        """
        # Find quest
        quest = None
        for q in self.available_quests:
            if q.id == quest_id:
                quest = q
                break

        if not quest:
            return False

        # Start quest
        quest.start()
        self.active_quests[quest_id] = quest
        self.available_quests.remove(quest)

        return True

    def complete_quest(
        self,
        quest_id: str,
        score: float,
        time_seconds: int
    ) -> Optional[QuestReward]:
        """
        Complete a quest and award rewards

        Args:
            quest_id: Quest ID
            score: Final score (0.0 - 1.0)
            time_seconds: Time spent

        Returns:
            Rewards earned (None if quest not found)
        """
        quest = self.active_quests.get(quest_id)
        if not quest:
            return None

        # Complete quest
        rewards = quest.complete(score, time_seconds)

        # Move to completed
        self.completed_quests[quest_id] = quest
        del self.active_quests[quest_id]

        # Update player model
        self._apply_rewards(rewards, quest)

        # Update Thompson Sampling bandit
        success = score >= 0.7  # Passing threshold
        self.player.update_quest_outcome(quest.difficulty, success, score)

        # Record quest completion stats
        self.player.record_quest_completion(success, time_seconds // 60)

        # Update learning objectives mastery
        for obj_id in quest.learning_objectives:
            if obj_id in [c.id for c in self.player.concepts.values()]:
                self.player.update_concept_mastery(obj_id, score, success)

        return rewards

    def _apply_rewards(self, rewards: QuestReward, quest: Quest) -> None:
        """Apply quest rewards to player"""
        # Award XP to skills
        for skill_id, xp in rewards.skill_xp.items():
            if skill_id in self.player.skills:
                success = quest.score >= 0.7
                self.player.gain_xp(skill_id, xp, success)

        # Award achievements
        for achievement_id in rewards.achievements:
            self.player.add_achievement(achievement_id)

    def get_active_quests(self) -> List[Quest]:
        """Get all active quests"""
        return list(self.active_quests.values())

    def get_completed_quests(self) -> List[Quest]:
        """Get all completed quests"""
        return list(self.completed_quests.values())

    def get_quest_progress(self, quest_id: str) -> Optional[Dict]:
        """Get progress for a specific quest"""
        quest = self.active_quests.get(quest_id) or self.completed_quests.get(quest_id)
        if not quest:
            return None

        return {
            "id": quest.id,
            "title": quest.title,
            "state": quest.state.value,
            "score": quest.score,
            "best_score": quest.best_score,
            "attempts": quest.attempts,
            "time_spent_minutes": quest.time_spent_seconds // 60,
            "estimated_minutes": quest.estimated_minutes
        }

    def get_stats(self) -> Dict:
        """Get quest engine statistics"""
        return {
            "active_quests": len(self.active_quests),
            "completed_quests": len(self.completed_quests),
            "available_quests": len(self.available_quests),
            "completion_rate": (
                len(self.completed_quests) / max(1, len(self.completed_quests) + len(self.active_quests))
            ),
            "average_score": (
                sum(q.score for q in self.completed_quests.values()) / max(1, len(self.completed_quests))
            )
        }


# ========== Helper Functions ==========

def create_sample_quest_engine() -> QuestEngine:
    """Create sample quest engine for testing"""
    from EduVerse.education.player_model import create_sample_player
    from EduVerse.education.curriculum import create_sample_curriculum

    player = create_sample_player()
    curriculum = create_sample_curriculum()

    engine = QuestEngine(curriculum, player)

    # Generate recommended quests
    engine.available_quests = engine.generate_recommended_quests(count=5)

    return engine


if __name__ == "__main__":
    print("=== EduVerse Quest Engine Demo ===\n")

    # Create quest engine
    engine = create_sample_quest_engine()

    # Show recommended quests
    print(f"Recommended Quests ({len(engine.available_quests)}):\n")
    for i, quest in enumerate(engine.available_quests, 1):
        print(f"{i}. {quest.title}")
        print(f"   Type: {quest.quest_type.value}, Difficulty: {quest.difficulty:.2f}")
        print(f"   Subject: {quest.subject.value if quest.subject else 'N/A'}, Grade: {quest.grade_level}")
        print(f"   Estimated: {quest.estimated_minutes} min, XP: {quest.rewards.xp:.0f}")
        print(f"   Objectives: {', '.join(quest.learning_objectives)}")
        print()

    # Start a quest
    if engine.available_quests:
        first_quest = engine.available_quests[0]
        print(f"Starting quest: {first_quest.title}...")
        engine.start_quest(first_quest.id)
        print(f"Quest state: {first_quest.state.value}\n")

        # Simulate completion
        print("Completing quest with score 0.85...")
        rewards = engine.complete_quest(first_quest.id, score=0.85, time_seconds=12 * 60)

        if rewards:
            print(f"Rewards earned:")
            print(f"  XP: {rewards.xp:.0f}")
            print(f"  Skill XP: {rewards.skill_xp}")
            print(f"  Achievements: {rewards.achievements}")

    # Stats
    print(f"\n=== Quest Engine Stats ===")
    stats = engine.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\nQuest engine ready!")
