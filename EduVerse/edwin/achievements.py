"""
EdWIN Achievement and Badge System

100+ achievements across categories to maximize student engagement.

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Optional, Callable
from enum import Enum


class AchievementCategory(Enum):
    """Achievement categories"""
    MASTERY = "mastery"  # Mastering objectives
    ENGAGEMENT = "engagement"  # Daily activity, streaks
    SOCIAL = "social"  # Helping peers, collaboration
    CHALLENGE = "challenge"  # Speed, perfect scores, marathons
    EXPLORATION = "exploration"  # Trying new subjects
    SPECIAL = "special"  # Unique achievements


class AchievementRarity(Enum):
    """Achievement rarity levels"""
    COMMON = "common"  # 60% of achievements
    UNCOMMON = "uncommon"  # 25%
    RARE = "rare"  # 10%
    EPIC = "epic"  # 4%
    LEGENDARY = "legendary"  # 1%


@dataclass
class Achievement:
    """
    An unlockable achievement/badge

    Attributes:
        id: Unique achievement ID
        name: Display name
        description: What the achievement is for
        category: Achievement category
        rarity: How rare/difficult
        xp_reward: XP awarded when unlocked
        icon: Icon/emoji
        requirement: Description of requirement
        check_function: Function to check if unlocked (takes StudentProgress)
    """
    id: str
    name: str
    description: str
    category: AchievementCategory
    rarity: AchievementRarity
    xp_reward: int
    icon: str
    requirement: str
    hidden: bool = False  # Hidden until unlocked
    check_function: Optional[Callable] = None


# ========== Achievement Definitions ==========

def create_all_achievements() -> Dict[str, Achievement]:
    """Create all 100+ achievements"""
    achievements = {}

    # ========== MASTERY ACHIEVEMENTS (40) ==========

    # First steps (5)
    achievements["first_steps"] = Achievement(
        id="first_steps",
        name="First Steps",
        description="Master your first objective",
        category=AchievementCategory.MASTERY,
        rarity=AchievementRarity.COMMON,
        xp_reward=25,
        icon="🎯",
        requirement="Master 1 objective"
    )

    achievements["getting_started"] = Achievement(
        id="getting_started",
        name="Getting Started",
        description="Master 5 objectives",
        category=AchievementCategory.MASTERY,
        rarity=AchievementRarity.COMMON,
        xp_reward=50,
        icon="📚",
        requirement="Master 5 objectives"
    )

    achievements["on_a_roll"] = Achievement(
        id="on_a_roll",
        name="On a Roll",
        description="Master 10 objectives",
        category=AchievementCategory.MASTERY,
        rarity=AchievementRarity.COMMON,
        xp_reward=100,
        icon="🔥",
        requirement="Master 10 objectives"
    )

    achievements["dedicated_learner"] = Achievement(
        id="dedicated_learner",
        name="Dedicated Learner",
        description="Master 25 objectives",
        category=AchievementCategory.MASTERY,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=250,
        icon="⭐",
        requirement="Master 25 objectives"
    )

    achievements["knowledge_collector"] = Achievement(
        id="knowledge_collector",
        name="Knowledge Collector",
        description="Master 50 objectives",
        category=AchievementCategory.MASTERY,
        rarity=AchievementRarity.RARE,
        xp_reward=500,
        icon="💎",
        requirement="Master 50 objectives"
    )

    # Subject expertise (15)
    subjects = ["math", "science", "ela", "social_studies", "ai_readiness"]
    subject_icons = {"math": "🔢", "science": "🔬", "ela": "📖", "social_studies": "🌍", "ai_readiness": "🤖"}
    subject_names = {
        "math": "Math Whiz",
        "science": "Science Guru",
        "ela": "Reading Champion",
        "social_studies": "History Buff",
        "ai_readiness": "AI Pioneer"
    }

    for subject in subjects:
        # 5 objectives in subject
        achievements[f"{subject}_beginner"] = Achievement(
            id=f"{subject}_beginner",
            name=f"{subject_names[subject]} I",
            description=f"Master 5 {subject.replace('_', ' ')} objectives",
            category=AchievementCategory.MASTERY,
            rarity=AchievementRarity.COMMON,
            xp_reward=75,
            icon=subject_icons[subject],
            requirement=f"Master 5 {subject} objectives"
        )

        # 15 objectives in subject
        achievements[f"{subject}_intermediate"] = Achievement(
            id=f"{subject}_intermediate",
            name=f"{subject_names[subject]} II",
            description=f"Master 15 {subject.replace('_', ' ')} objectives",
            category=AchievementCategory.MASTERY,
            rarity=AchievementRarity.UNCOMMON,
            xp_reward=150,
            icon=subject_icons[subject],
            requirement=f"Master 15 {subject} objectives"
        )

        # Complete all in subject
        achievements[f"{subject}_master"] = Achievement(
            id=f"{subject}_master",
            name=f"{subject_names[subject]} Master",
            description=f"Master all {subject.replace('_', ' ')} objectives",
            category=AchievementCategory.MASTERY,
            rarity=AchievementRarity.EPIC,
            xp_reward=1000,
            icon="🏆",
            requirement=f"Master all {subject} objectives"
        )

    # Grade level completion (5)
    for grade in [4, 5, 6, 7, 8]:
        achievements[f"grade_{grade}_complete"] = Achievement(
            id=f"grade_{grade}_complete",
            name=f"Grade {grade} Graduate",
            description=f"Complete all Grade {grade} objectives",
            category=AchievementCategory.MASTERY,
            rarity=AchievementRarity.RARE,
            xp_reward=750,
            icon="🎓",
            requirement=f"Master all Grade {grade} objectives"
        )

    # Bloom levels (5)
    bloom_levels = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
    for i, level in enumerate(bloom_levels):
        achievements[f"bloom_{level}"] = Achievement(
            id=f"bloom_{level}",
            name=f"{level.capitalize()} Master",
            description=f"Master 10 objectives at {level.capitalize()} level",
            category=AchievementCategory.MASTERY,
            rarity=AchievementRarity.UNCOMMON,
            xp_reward=100 + i * 50,
            icon="🧠",
            requirement=f"Master 10 objectives at {level} Bloom level"
        )

    # ========== ENGAGEMENT ACHIEVEMENTS (30) ==========

    # Streaks (10)
    streak_milestones = [3, 7, 14, 30, 60, 100, 180, 365, 500, 1000]
    streak_names = [
        "Getting Consistent", "Daily Learner", "Two Weeks Strong", "Monthly Dedication",
        "Two Month Marathon", "Unstoppable", "Half Year Hero", "Year-Long Legend",
        "Incredible Consistency", "Ultimate Dedication"
    ]
    streak_rarities = [
        AchievementRarity.COMMON, AchievementRarity.COMMON, AchievementRarity.UNCOMMON,
        AchievementRarity.RARE, AchievementRarity.RARE, AchievementRarity.EPIC,
        AchievementRarity.EPIC, AchievementRarity.LEGENDARY, AchievementRarity.LEGENDARY,
        AchievementRarity.LEGENDARY
    ]

    for days, name, rarity in zip(streak_milestones, streak_names, streak_rarities):
        achievements[f"streak_{days}"] = Achievement(
            id=f"streak_{days}",
            name=name,
            description=f"Maintain a {days}-day streak",
            category=AchievementCategory.ENGAGEMENT,
            rarity=rarity,
            xp_reward=days * 5,
            icon="🔥",
            requirement=f"{days}-day streak"
        )

    # Questions (5)
    question_milestones = [10, 50, 100, 500, 1000]
    for count in question_milestones:
        rarity = AchievementRarity.COMMON if count <= 50 else (
            AchievementRarity.UNCOMMON if count <= 100 else (
                AchievementRarity.RARE if count <= 500 else AchievementRarity.EPIC
            )
        )
        achievements[f"questions_{count}"] = Achievement(
            id=f"questions_{count}",
            name=f"Curious Mind ({count})",
            description=f"Ask {count} questions",
            category=AchievementCategory.ENGAGEMENT,
            rarity=rarity,
            xp_reward=count // 2,
            icon="❓",
            requirement=f"Ask {count} questions"
        )

    # Practice (5)
    practice_milestones = [100, 500, 1000, 5000, 10000]
    for count in practice_milestones:
        rarity = AchievementRarity.COMMON if count <= 500 else (
            AchievementRarity.UNCOMMON if count <= 1000 else (
                AchievementRarity.RARE if count <= 5000 else AchievementRarity.EPIC
            )
        )
        achievements[f"practice_{count}"] = Achievement(
            id=f"practice_{count}",
            name=f"Practice Master ({count})",
            description=f"Complete {count} practice problems",
            category=AchievementCategory.ENGAGEMENT,
            rarity=rarity,
            xp_reward=count // 5,
            icon="✏️",
            requirement=f"Complete {count} practice problems"
        )

    # Time-based (5)
    achievements["early_bird"] = Achievement(
        id="early_bird",
        name="Early Bird",
        description="Study before 8am 10 times",
        category=AchievementCategory.ENGAGEMENT,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=100,
        icon="🌅",
        requirement="Study before 8am (10 times)"
    )

    achievements["night_owl"] = Achievement(
        id="night_owl",
        name="Night Owl",
        description="Study after 8pm 10 times",
        category=AchievementCategory.ENGAGEMENT,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=100,
        icon="🦉",
        requirement="Study after 8pm (10 times)"
    )

    achievements["weekend_warrior"] = Achievement(
        id="weekend_warrior",
        name="Weekend Warrior",
        description="Study on 10 weekends",
        category=AchievementCategory.ENGAGEMENT,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=150,
        icon="🏋️",
        requirement="Study on 10 weekends"
    )

    achievements["all_nighter"] = Achievement(
        id="all_nighter",
        name="Marathon Session",
        description="Study for 4+ hours in one session",
        category=AchievementCategory.ENGAGEMENT,
        rarity=AchievementRarity.RARE,
        xp_reward=200,
        icon="⏰",
        requirement="4-hour study session"
    )

    achievements["consistent_weekly"] = Achievement(
        id="consistent_weekly",
        name="Weekly Consistency",
        description="Study every day for 4 weeks straight",
        category=AchievementCategory.ENGAGEMENT,
        rarity=AchievementRarity.RARE,
        xp_reward=300,
        icon="📅",
        requirement="Study daily for 4 weeks"
    )

    # ========== SOCIAL ACHIEVEMENTS (15) ==========

    achievements["team_player"] = Achievement(
        id="team_player",
        name="Team Player",
        description="Join your first study group",
        category=AchievementCategory.SOCIAL,
        rarity=AchievementRarity.COMMON,
        xp_reward=50,
        icon="👥",
        requirement="Join a study group"
    )

    achievements["peer_tutor"] = Achievement(
        id="peer_tutor",
        name="Peer Tutor",
        description="Help 10 students",
        category=AchievementCategory.SOCIAL,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=250,
        icon="🤝",
        requirement="Help 10 students"
    )

    achievements["collaboration_king"] = Achievement(
        id="collaboration_king",
        name="Collaboration Champion",
        description="Complete 5 group challenges",
        category=AchievementCategory.SOCIAL,
        rarity=AchievementRarity.RARE,
        xp_reward=200,
        icon="👑",
        requirement="Complete 5 group challenges"
    )

    achievements["helpful_friend"] = Achievement(
        id="helpful_friend",
        name="Helpful Friend",
        description="Give 50 helpful answers to peers",
        category=AchievementCategory.SOCIAL,
        rarity=AchievementRarity.RARE,
        xp_reward=500,
        icon="💚",
        requirement="50 helpful answers"
    )

    # Helping in subjects (5)
    for subject in subjects:
        achievements[f"tutor_{subject}"] = Achievement(
            id=f"tutor_{subject}",
            name=f"{subject.replace('_', ' ').title()} Tutor",
            description=f"Help 5 students with {subject.replace('_', ' ')}",
            category=AchievementCategory.SOCIAL,
            rarity=AchievementRarity.UNCOMMON,
            xp_reward=150,
            icon="🎓",
            requirement=f"Help 5 students with {subject}"
        )

    # Study groups (5)
    group_milestones = [1, 5, 10, 25, 50]
    for count in group_milestones:
        achievements[f"group_sessions_{count}"] = Achievement(
            id=f"group_sessions_{count}",
            name=f"Group Learner ({count})",
            description=f"Complete {count} group study sessions",
            category=AchievementCategory.SOCIAL,
            rarity=AchievementRarity.COMMON if count <= 5 else AchievementRarity.UNCOMMON,
            xp_reward=count * 20,
            icon="📚",
            requirement=f"Complete {count} group sessions"
        )

    # ========== CHALLENGE ACHIEVEMENTS (20) ==========

    achievements["speed_demon"] = Achievement(
        id="speed_demon",
        name="Speed Demon",
        description="Answer 5 questions correctly in under 5 minutes",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=100,
        icon="⚡",
        requirement="5 questions in 5 minutes"
    )

    achievements["perfect_score_5"] = Achievement(
        id="perfect_score_5",
        name="Perfectionist",
        description="Get perfect scores on 5 objectives",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=200,
        icon="💯",
        requirement="5 perfect scores"
    )

    achievements["perfect_score_10"] = Achievement(
        id="perfect_score_10",
        name="Flawless",
        description="Get perfect scores on 10 objectives",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.RARE,
        xp_reward=500,
        icon="🌟",
        requirement="10 perfect scores"
    )

    achievements["no_mistakes"] = Achievement(
        id="no_mistakes",
        name="Error-Free",
        description="Complete 5 objectives without any mistakes",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.RARE,
        xp_reward=300,
        icon="✨",
        requirement="5 objectives with 0 mistakes"
    )

    achievements["quick_study"] = Achievement(
        id="quick_study",
        name="Quick Study",
        description="Master an objective in under 30 minutes",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=150,
        icon="🚀",
        requirement="Master objective in <30 minutes"
    )

    # Speed challenges (5)
    speed_subjects = ["math", "science", "ela", "social_studies", "ai_readiness"]
    for subject in speed_subjects:
        achievements[f"speed_{subject}"] = Achievement(
            id=f"speed_{subject}",
            name=f"{subject.replace('_', ' ').title()} Speedster",
            description=f"Complete 10 {subject.replace('_', ' ')} objectives quickly",
            category=AchievementCategory.CHALLENGE,
            rarity=AchievementRarity.RARE,
            xp_reward=250,
            icon="💨",
            requirement=f"Fast completion of 10 {subject} objectives"
        )

    # Difficult objectives (5)
    achievements["hard_mode"] = Achievement(
        id="hard_mode",
        name="Challenge Accepted",
        description="Complete 5 above-grade-level objectives",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.RARE,
        xp_reward=400,
        icon="🎯",
        requirement="5 above-grade objectives"
    )

    achievements["underdog"] = Achievement(
        id="underdog",
        name="Underdog Victory",
        description="Beat a challenge 2 grades above your level",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.EPIC,
        xp_reward=500,
        icon="🏆",
        requirement="Complete objective 2+ grades above"
    )

    achievements["comeback"] = Achievement(
        id="comeback",
        name="Comeback Kid",
        description="Improve from failing (< 60%) to passing (> 80%)",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=200,
        icon="📈",
        requirement="Improve from <60% to >80%"
    )

    achievements["never_give_up"] = Achievement(
        id="never_give_up",
        name="Never Give Up",
        description="Retry a failed objective 5 times until mastering it",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.RARE,
        xp_reward=300,
        icon="💪",
        requirement="Master after 5+ failed attempts"
    )

    achievements["triple_perfect"] = Achievement(
        id="triple_perfect",
        name="Triple Threat",
        description="Get perfect scores on 3 objectives in one day",
        category=AchievementCategory.CHALLENGE,
        rarity=AchievementRarity.RARE,
        xp_reward=350,
        icon="🔱",
        requirement="3 perfect scores in 1 day"
    )

    # ========== EXPLORATION ACHIEVEMENTS (15) ==========

    achievements["renaissance"] = Achievement(
        id="renaissance",
        name="Renaissance Student",
        description="Try all 5 subjects",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.COMMON,
        xp_reward=100,
        icon="🎨",
        requirement="Try all subjects"
    )

    achievements["curious_explorer"] = Achievement(
        id="curious_explorer",
        name="Curious Explorer",
        description="Ask questions in 5+ different subjects",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=150,
        icon="🔍",
        requirement="Questions in 5+ subjects"
    )

    achievements["adventurer"] = Achievement(
        id="adventurer",
        name="Adventurer",
        description="Complete objectives from 3+ different grade levels",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=200,
        icon="🗺️",
        requirement="Objectives from 3+ grades"
    )

    achievements["diverse_learner"] = Achievement(
        id="diverse_learner",
        name="Diverse Learner",
        description="Try 20+ different objectives",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.COMMON,
        xp_reward=100,
        icon="🌈",
        requirement="Try 20+ objectives"
    )

    # Subject exploration (5)
    for subject in subjects:
        achievements[f"explore_{subject}"] = Achievement(
            id=f"explore_{subject}",
            name=f"{subject.replace('_', ' ').title()} Explorer",
            description=f"Try 10 different {subject.replace('_', ' ')} objectives",
            category=AchievementCategory.EXPLORATION,
            rarity=AchievementRarity.COMMON,
            xp_reward=75,
            icon="🧭",
            requirement=f"Try 10 {subject} objectives"
        )

    # Cross-grade (5)
    achievements["grade_jumper"] = Achievement(
        id="grade_jumper",
        name="Grade Jumper",
        description="Complete an objective 1 grade above",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.COMMON,
        xp_reward=50,
        icon="⬆️",
        requirement="Complete +1 grade objective"
    )

    achievements["advanced_student"] = Achievement(
        id="advanced_student",
        name="Advanced Student",
        description="Complete 5 objectives above your grade",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.UNCOMMON,
        xp_reward=150,
        icon="🎓",
        requirement="5 above-grade objectives"
    )

    achievements["time_traveler"] = Achievement(
        id="time_traveler",
        name="Time Traveler",
        description="Complete objectives from 5+ different grades",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.RARE,
        xp_reward=250,
        icon="⏳",
        requirement="Objectives from 5+ grades"
    )

    achievements["subject_hopper"] = Achievement(
        id="subject_hopper",
        name="Subject Hopper",
        description="Switch between 3+ subjects in one study session",
        category=AchievementCategory.EXPLORATION,
        rarity=AchievementRarity.COMMON,
        xp_reward=50,
        icon="🦘",
        requirement="3+ subjects in 1 session"
    )

    # ========== SPECIAL ACHIEVEMENTS (10) ==========

    achievements["birthday_scholar"] = Achievement(
        id="birthday_scholar",
        name="Birthday Scholar",
        description="Study on your birthday",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.RARE,
        xp_reward=200,
        icon="🎂",
        requirement="Study on birthday",
        hidden=True
    )

    achievements["holiday_hero"] = Achievement(
        id="holiday_hero",
        name="Holiday Hero",
        description="Study on a major holiday",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.RARE,
        xp_reward=150,
        icon="🎉",
        requirement="Study on holiday",
        hidden=True
    )

    achievements["midnight_scholar"] = Achievement(
        id="midnight_scholar",
        name="Midnight Scholar",
        description="Study at exactly midnight",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.EPIC,
        xp_reward=300,
        icon="🌙",
        requirement="Study at midnight",
        hidden=True
    )

    achievements["perfect_week"] = Achievement(
        id="perfect_week",
        name="Perfect Week",
        description="Master at least 1 objective every day for a week",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.RARE,
        xp_reward=500,
        icon="📆",
        requirement="Master 1+ objective daily for 7 days"
    )

    achievements["legend"] = Achievement(
        id="legend",
        name="Legend",
        description="Unlock all other achievements",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.LEGENDARY,
        xp_reward=5000,
        icon="👑",
        requirement="Unlock all achievements",
        hidden=True
    )

    achievements["first_login"] = Achievement(
        id="first_login",
        name="Welcome to EdWIN!",
        description="Complete your first login",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.COMMON,
        xp_reward=10,
        icon="👋",
        requirement="First login"
    )

    achievements["profile_complete"] = Achievement(
        id="profile_complete",
        name="Profile Complete",
        description="Fill out your complete profile",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.COMMON,
        xp_reward=50,
        icon="📝",
        requirement="Complete profile"
    )

    achievements["level_50"] = Achievement(
        id="level_50",
        name="Halfway There",
        description="Reach level 50",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.EPIC,
        xp_reward=1000,
        icon="5️⃣0️⃣",
        requirement="Reach level 50"
    )

    achievements["level_100"] = Achievement(
        id="level_100",
        name="Maximum Power",
        description="Reach level 100 (max level)",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.LEGENDARY,
        xp_reward=10000,
        icon="💯",
        requirement="Reach level 100"
    )

    achievements["completionist"] = Achievement(
        id="completionist",
        name="Completionist",
        description="Master all 220 objectives",
        category=AchievementCategory.SPECIAL,
        rarity=AchievementRarity.LEGENDARY,
        xp_reward=15000,
        icon="🌟",
        requirement="Master all objectives",
        hidden=True
    )

    return achievements


# ========== Achievement Manager ==========

@dataclass
class StudentAchievements:
    """Track student achievement progress"""
    student_id: str
    unlocked: Set[str] = field(default_factory=set)
    unlock_times: Dict[str, datetime] = field(default_factory=dict)
    progress: Dict[str, float] = field(default_factory=dict)  # For progressive achievements


class AchievementManager:
    """
    Manages student achievements

    Features:
    - Check achievement unlock conditions
    - Track progress toward achievements
    - Award XP for unlocked achievements
    - Get recommendations for next achievements

    Example:
        >>> manager = AchievementManager()
        >>> student_achievements = StudentAchievements(student_id="student_001")
        >>>
        >>> # Check for unlocks
        >>> unlocked = manager.check_achievement(
        ...     achievement_id="first_steps",
        ...     student_achievements=student_achievements,
        ...     student_stats={"objectives_mastered": 1}
        ... )
        >>> if unlocked:
        ...     print(f"🎉 Achievement unlocked: {unlocked['name']}")
    """

    def __init__(self):
        self.achievements = create_all_achievements()

    def get_achievement(self, achievement_id: str) -> Optional[Achievement]:
        """Get achievement by ID"""
        return self.achievements.get(achievement_id)

    def get_all_achievements(self) -> Dict[str, Achievement]:
        """Get all achievements"""
        return self.achievements

    def get_achievements_by_category(self, category: AchievementCategory) -> List[Achievement]:
        """Get achievements by category"""
        return [
            ach for ach in self.achievements.values()
            if ach.category == category
        ]

    def get_achievements_by_rarity(self, rarity: AchievementRarity) -> List[Achievement]:
        """Get achievements by rarity"""
        return [
            ach for ach in self.achievements.values()
            if ach.rarity == rarity
        ]

    def check_achievement(
        self,
        achievement_id: str,
        student_achievements: StudentAchievements,
        student_stats: Dict
    ) -> Optional[Dict]:
        """
        Check if achievement should be unlocked

        Args:
            achievement_id: Achievement to check
            student_achievements: Student's achievement progress
            student_stats: Student statistics (objectives_mastered, etc.)

        Returns:
            Dict with achievement details if unlocked, None otherwise
        """
        # Already unlocked?
        if achievement_id in student_achievements.unlocked:
            return None

        achievement = self.achievements.get(achievement_id)
        if not achievement:
            return None

        # Check unlock condition (simplified - would use check_function in production)
        should_unlock = self._check_unlock_condition(
            achievement, student_stats
        )

        if should_unlock:
            # Unlock!
            student_achievements.unlocked.add(achievement_id)
            student_achievements.unlock_times[achievement_id] = datetime.utcnow()

            return {
                "id": achievement.id,
                "name": achievement.name,
                "description": achievement.description,
                "category": achievement.category.value,
                "rarity": achievement.rarity.value,
                "xp_reward": achievement.xp_reward,
                "icon": achievement.icon,
                "unlocked_at": student_achievements.unlock_times[achievement_id].isoformat()
            }

        return None

    def _check_unlock_condition(self, achievement: Achievement, stats: Dict) -> bool:
        """Check if unlock condition is met (simplified)"""
        # This would be more sophisticated in production
        # For now, basic checks based on achievement ID

        if achievement.id == "first_steps":
            return stats.get("objectives_mastered", 0) >= 1

        if achievement.id == "getting_started":
            return stats.get("objectives_mastered", 0) >= 5

        if achievement.id == "on_a_roll":
            return stats.get("objectives_mastered", 0) >= 10

        if achievement.id.startswith("streak_"):
            days = int(achievement.id.split("_")[1])
            return stats.get("current_streak", 0) >= days

        if achievement.id.startswith("questions_"):
            count = int(achievement.id.split("_")[1])
            return stats.get("questions_asked", 0) >= count

        # Default: not unlocked
        return False

    def get_progress(
        self,
        student_achievements: StudentAchievements
    ) -> Dict:
        """Get achievement progress statistics"""
        total = len(self.achievements)
        unlocked = len(student_achievements.unlocked)

        # Count by category
        by_category = {}
        for category in AchievementCategory:
            category_achs = self.get_achievements_by_category(category)
            unlocked_in_category = sum(
                1 for ach in category_achs
                if ach.id in student_achievements.unlocked
            )
            by_category[category.value] = {
                "total": len(category_achs),
                "unlocked": unlocked_in_category,
                "percent": (unlocked_in_category / len(category_achs) * 100) if category_achs else 0
            }

        # Count by rarity
        by_rarity = {}
        for rarity in AchievementRarity:
            rarity_achs = self.get_achievements_by_rarity(rarity)
            unlocked_in_rarity = sum(
                1 for ach in rarity_achs
                if ach.id in student_achievements.unlocked
            )
            by_rarity[rarity.value] = {
                "total": len(rarity_achs),
                "unlocked": unlocked_in_rarity,
                "percent": (unlocked_in_rarity / len(rarity_achs) * 100) if rarity_achs else 0
            }

        return {
            "student_id": student_achievements.student_id,
            "total_achievements": total,
            "unlocked_count": unlocked,
            "completion_percent": (unlocked / total * 100),
            "by_category": by_category,
            "by_rarity": by_rarity,
            "recent_unlocks": self.get_recent_unlocks(student_achievements, limit=5)
        }

    def get_recent_unlocks(
        self,
        student_achievements: StudentAchievements,
        limit: int = 10
    ) -> List[Dict]:
        """Get recently unlocked achievements"""
        sorted_unlocks = sorted(
            student_achievements.unlock_times.items(),
            key=lambda x: x[1],
            reverse=True
        )

        recent = []
        for ach_id, unlock_time in sorted_unlocks[:limit]:
            achievement = self.achievements.get(ach_id)
            if achievement:
                recent.append({
                    "id": ach_id,
                    "name": achievement.name,
                    "icon": achievement.icon,
                    "rarity": achievement.rarity.value,
                    "unlocked_at": unlock_time.isoformat()
                })

        return recent

    def get_next_recommendations(
        self,
        student_achievements: StudentAchievements,
        limit: int = 5
    ) -> List[Dict]:
        """Get recommended next achievements to pursue"""
        # Find achievements close to unlocking
        # (This would be more sophisticated in production)

        recommendations = []
        for ach_id, achievement in self.achievements.items():
            if ach_id not in student_achievements.unlocked and not achievement.hidden:
                recommendations.append({
                    "id": ach_id,
                    "name": achievement.name,
                    "description": achievement.description,
                    "icon": achievement.icon,
                    "xp_reward": achievement.xp_reward,
                    "requirement": achievement.requirement,
                    "rarity": achievement.rarity.value
                })

        # Sort by rarity (easier first) and XP reward
        recommendations.sort(
            key=lambda x: (
                list(AchievementRarity).index(AchievementRarity(x['rarity'])),
                -x['xp_reward']
            )
        )

        return recommendations[:limit]


# ========== Utility Functions ==========

def create_achievement_manager() -> AchievementManager:
    """Create new achievement manager"""
    return AchievementManager()


def get_achievement_count() -> Dict[str, int]:
    """Get count of achievements by category"""
    achievements = create_all_achievements()

    counts = {
        "total": len(achievements),
        "by_category": {},
        "by_rarity": {}
    }

    for ach in achievements.values():
        # By category
        cat = ach.category.value
        counts["by_category"][cat] = counts["by_category"].get(cat, 0) + 1

        # By rarity
        rar = ach.rarity.value
        counts["by_rarity"][rar] = counts["by_rarity"].get(rar, 0) + 1

    return counts


if __name__ == "__main__":
    # Demo
    print("=== EdWIN Achievement System Demo ===\n")

    print("Achievement Count:")
    counts = get_achievement_count()
    print(f"Total: {counts['total']}")
    print("\nBy Category:")
    for cat, count in counts['by_category'].items():
        print(f"  {cat}: {count}")
    print("\nBy Rarity:")
    for rar, count in counts['by_rarity'].items():
        print(f"  {rar}: {count}")

    # Create manager
    manager = AchievementManager()

    # Simulate student
    student_ach = StudentAchievements(student_id="student_001")

    # Check achievement
    print("\n=== Checking Achievements ===")
    stats = {"objectives_mastered": 1, "current_streak": 0}

    unlocked = manager.check_achievement("first_steps", student_ach, stats)
    if unlocked:
        print(f"🎉 {unlocked['icon']} {unlocked['name']}")
        print(f"   {unlocked['description']}")
        print(f"   +{unlocked['xp_reward']} XP")

    # Get progress
    print("\n=== Achievement Progress ===")
    progress = manager.get_progress(student_ach)
    print(f"Unlocked: {progress['unlocked_count']}/{progress['total_achievements']} ({progress['completion_percent']:.1f}%)")

    # Get recommendations
    print("\n=== Recommended Next Achievements ===")
    recommendations = manager.get_next_recommendations(student_ach, limit=5)
    for rec in recommendations:
        print(f"{rec['icon']} {rec['name']} ({rec['rarity']})")
        print(f"   {rec['requirement']} → +{rec['xp_reward']} XP")
