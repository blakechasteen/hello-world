#!/usr/bin/env python3
"""
Gamification Service - Points, Achievements, and Progression

Makes kitchen management engaging through:
- Points system with variable rewards
- Achievement badges and milestones
- Level progression (skill-based advancement)
- Leaderboards (household and global)
- Streak tracking (consistency rewards)
- Weekly challenges
- Recipe unlocking system
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum


class AchievementType(Enum):
    """Types of achievements"""
    TASK_COMPLETION = "task_completion"
    SKILL_MASTERY = "skill_mastery"
    RECIPE_MASTERY = "recipe_mastery"
    STREAK = "streak"
    COLLABORATION = "collaboration"
    WASTE_PREVENTION = "waste_prevention"
    MILESTONE = "milestone"


class BadgeRarity(Enum):
    """Badge rarity levels"""
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    EPIC = 4
    LEGENDARY = 5


@dataclass
class Achievement:
    """
    Achievement/badge definition.

    Example achievements:
    - "First Steps": Complete your first task (10 points)
    - "Master Chef": Master 10 recipes (500 points)
    - "Week Warrior": Cook every day for a week (200 points)
    - "Zero Waste Hero": Prevent 50kg of food waste (1000 points)
    """
    achievement_id: str
    name: str
    description: str
    achievement_type: AchievementType
    points: int
    rarity: BadgeRarity
    icon: str  # Emoji or icon name
    requirement: Dict  # {"count": 10, "entity": "recipes"}
    repeatable: bool = False

    def check_completion(self, user_stats: Dict) -> bool:
        """Check if achievement requirements are met."""
        entity = self.requirement.get("entity")
        required_count = self.requirement.get("count", 1)

        if not entity:
            return False

        actual_count = user_stats.get(entity, 0)
        return actual_count >= required_count

    def to_dict(self) -> dict:
        return {
            "achievement_id": self.achievement_id,
            "name": self.name,
            "description": self.description,
            "achievement_type": self.achievement_type.value,
            "points": self.points,
            "rarity": self.rarity.value,
            "icon": self.icon,
            "requirement": self.requirement,
            "repeatable": self.repeatable
        }


@dataclass
class UserAchievement:
    """User's earned achievement"""
    achievement_id: str
    earned_at: datetime
    progress: int = 100  # 0-100

    def to_dict(self) -> dict:
        return {
            "achievement_id": self.achievement_id,
            "earned_at": self.earned_at.isoformat(),
            "progress": self.progress
        }


@dataclass
class Challenge:
    """
    Weekly or limited-time challenge.

    Example challenges:
    - "Plant-Based Week": Cook 5 vegetarian meals (300 points)
    - "Speed Demon": Complete 10 tasks in under estimated time (200 points)
    - "Team Player": Help 3 different household members (150 points)
    """
    challenge_id: str
    name: str
    description: str
    points: int
    start_date: datetime
    end_date: datetime
    requirement: Dict
    participants: Set[str] = field(default_factory=set)  # user_ids
    completed_by: Set[str] = field(default_factory=set)  # user_ids who completed

    def is_active(self) -> bool:
        """Check if challenge is currently active."""
        now = datetime.now()
        return self.start_date <= now <= self.end_date

    def check_completion(self, user_id: str, user_stats: Dict) -> bool:
        """Check if user has completed challenge."""
        if user_id in self.completed_by:
            return True

        entity = self.requirement.get("entity")
        required_count = self.requirement.get("count", 1)

        if not entity:
            return False

        actual_count = user_stats.get(entity, 0)
        return actual_count >= required_count

    def to_dict(self) -> dict:
        return {
            "challenge_id": self.challenge_id,
            "name": self.name,
            "description": self.description,
            "points": self.points,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "requirement": self.requirement,
            "participants": list(self.participants),
            "completed_by": list(self.completed_by),
            "is_active": self.is_active()
        }


@dataclass
class Streak:
    """User cooking streak"""
    user_id: str
    current_streak: int = 0
    longest_streak: int = 0
    last_activity: Optional[datetime] = None

    def update(self, activity_date: datetime):
        """Update streak based on activity."""
        if not self.last_activity:
            self.current_streak = 1
            self.longest_streak = 1
            self.last_activity = activity_date
            return

        days_since = (activity_date - self.last_activity).days

        if days_since == 0:
            # Same day activity - no change
            pass
        elif days_since == 1:
            # Consecutive day - increment streak
            self.current_streak += 1
            self.longest_streak = max(self.longest_streak, self.current_streak)
        else:
            # Streak broken - reset
            self.current_streak = 1

        self.last_activity = activity_date

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "current_streak": self.current_streak,
            "longest_streak": self.longest_streak,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


class GamificationEngine:
    """
    Core gamification engine.

    Manages:
    - Point awards and level progression
    - Achievement tracking and unlocking
    - Leaderboards
    - Streak tracking
    - Challenges
    """

    def __init__(self):
        # Achievement catalog
        self.achievements: Dict[str, Achievement] = {}
        self._initialize_achievements()

        # User achievements
        self.user_achievements: Dict[str, List[UserAchievement]] = {}  # user_id -> achievements

        # Challenges
        self.challenges: Dict[str, Challenge] = {}

        # Streaks
        self.streaks: Dict[str, Streak] = {}  # user_id -> streak

        # Points configuration
        self.points_config = {
            "task_completion_base": 10,
            "task_priority_multiplier": {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0},
            "recipe_mastery": 50,
            "level_up": 100,
            "achievement": lambda a: a.points,
            "streak_bonus": 5,  # Per day in streak
            "challenge_completion": lambda c: c.points
        }

    def _initialize_achievements(self):
        """Initialize standard achievement catalog."""

        # Task completion achievements
        self.add_achievement(Achievement(
            achievement_id="first_steps",
            name="First Steps",
            description="Complete your first cooking task",
            achievement_type=AchievementType.TASK_COMPLETION,
            points=10,
            rarity=BadgeRarity.COMMON,
            icon="🎯",
            requirement={"entity": "tasks_completed", "count": 1}
        ))

        self.add_achievement(Achievement(
            achievement_id="task_master",
            name="Task Master",
            description="Complete 100 cooking tasks",
            achievement_type=AchievementType.TASK_COMPLETION,
            points=500,
            rarity=BadgeRarity.RARE,
            icon="⭐",
            requirement={"entity": "tasks_completed", "count": 100}
        ))

        # Recipe mastery
        self.add_achievement(Achievement(
            achievement_id="master_chef",
            name="Master Chef",
            description="Master 10 different recipes",
            achievement_type=AchievementType.RECIPE_MASTERY,
            points=500,
            rarity=BadgeRarity.EPIC,
            icon="👨‍🍳",
            requirement={"entity": "recipes_mastered", "count": 10}
        ))

        # Streak achievements
        self.add_achievement(Achievement(
            achievement_id="week_warrior",
            name="Week Warrior",
            description="Cook every day for a week",
            achievement_type=AchievementType.STREAK,
            points=200,
            rarity=BadgeRarity.UNCOMMON,
            icon="🔥",
            requirement={"entity": "current_streak", "count": 7}
        ))

        self.add_achievement(Achievement(
            achievement_id="month_champion",
            name="Month Champion",
            description="Cook every day for 30 days",
            achievement_type=AchievementType.STREAK,
            points=1000,
            rarity=BadgeRarity.LEGENDARY,
            icon="🏆",
            requirement={"entity": "current_streak", "count": 30}
        ))

        # Waste prevention
        self.add_achievement(Achievement(
            achievement_id="zero_waste_hero",
            name="Zero Waste Hero",
            description="Prevent 50kg of food waste",
            achievement_type=AchievementType.WASTE_PREVENTION,
            points=1000,
            rarity=BadgeRarity.EPIC,
            icon="♻️",
            requirement={"entity": "waste_prevented_kg", "count": 50}
        ))

        # Collaboration
        self.add_achievement(Achievement(
            achievement_id="team_player",
            name="Team Player",
            description="Help 20 different household members",
            achievement_type=AchievementType.COLLABORATION,
            points=300,
            rarity=BadgeRarity.RARE,
            icon="🤝",
            requirement={"entity": "users_helped", "count": 20}
        ))

    def add_achievement(self, achievement: Achievement):
        """Add achievement to catalog."""
        self.achievements[achievement.achievement_id] = achievement

    def award_points(
        self,
        user_id: str,
        user,  # UserProfile
        reason: str,
        custom_points: Optional[int] = None
    ) -> int:
        """
        Award points to user for an action.

        Returns points awarded.
        """
        # Calculate points based on reason
        points = 0

        if custom_points:
            points = custom_points
        elif "task_completion" in reason:
            points = self.points_config["task_completion_base"]
        elif "recipe_mastery" in reason:
            points = self.points_config["recipe_mastery"]
        elif "level_up" in reason:
            points = self.points_config["level_up"]

        # Add streak bonus
        if user_id in self.streaks:
            streak_days = self.streaks[user_id].current_streak
            points += streak_days * self.points_config["streak_bonus"]

        # Award points
        user.add_points(points, reason)

        return points

    def check_achievements(
        self,
        user_id: str,
        user_stats: Dict
    ) -> List[Achievement]:
        """
        Check for newly earned achievements.

        Returns list of newly earned achievements.
        """
        newly_earned = []

        # Get user's current achievements
        current = set(
            ua.achievement_id
            for ua in self.user_achievements.get(user_id, [])
        )

        # Check all achievements
        for achievement_id, achievement in self.achievements.items():
            if achievement_id in current and not achievement.repeatable:
                continue  # Already earned and not repeatable

            if achievement.check_completion(user_stats):
                # Earned!
                if user_id not in self.user_achievements:
                    self.user_achievements[user_id] = []

                user_achievement = UserAchievement(
                    achievement_id=achievement_id,
                    earned_at=datetime.now(),
                    progress=100
                )
                self.user_achievements[user_id].append(user_achievement)
                newly_earned.append(achievement)

        return newly_earned

    def update_streak(self, user_id: str):
        """Update user's cooking streak."""
        if user_id not in self.streaks:
            self.streaks[user_id] = Streak(user_id=user_id)

        self.streaks[user_id].update(datetime.now())

        # Check for streak achievements
        current_streak = self.streaks[user_id].current_streak
        user_stats = {"current_streak": current_streak}

        newly_earned = self.check_achievements(user_id, user_stats)
        return newly_earned

    def add_challenge(self, challenge: Challenge):
        """Add a challenge."""
        self.challenges[challenge.challenge_id] = challenge

    def join_challenge(self, challenge_id: str, user_id: str) -> bool:
        """User joins a challenge."""
        if challenge_id not in self.challenges:
            return False

        challenge = self.challenges[challenge_id]
        if not challenge.is_active():
            return False

        challenge.participants.add(user_id)
        return True

    def check_challenge_completion(
        self,
        user_id: str,
        user_stats: Dict
    ) -> List[Challenge]:
        """Check for completed challenges."""
        completed = []

        for challenge in self.challenges.values():
            if not challenge.is_active():
                continue

            if user_id not in challenge.participants:
                continue

            if challenge.check_completion(user_id, user_stats):
                challenge.completed_by.add(user_id)
                completed.append(challenge)

        return completed

    def get_leaderboard(
        self,
        household_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get leaderboard.

        Args:
            household_id: Filter by household (None = global)
            limit: Number of users to return
        """
        # This would integrate with UserProfile data
        # For now, return placeholder structure
        leaderboard = []

        # Would query user database here
        # Sorting by total_points descending

        return leaderboard[:limit]

    def get_user_summary(self, user_id: str, user) -> Dict:
        """Get gamification summary for user."""
        achievements = self.user_achievements.get(user_id, [])
        streak = self.streaks.get(user_id, Streak(user_id=user_id))

        return {
            "user_id": user_id,
            "level": user.level,
            "total_points": user.total_points,
            "achievements_earned": len(achievements),
            "achievements_total": len(self.achievements),
            "current_streak": streak.current_streak,
            "longest_streak": streak.longest_streak,
            "recipes_mastered": len(user.recipes_mastered),
            "tasks_completed": user.tasks_completed,
            "recent_achievements": [
                {
                    "achievement_id": ua.achievement_id,
                    "name": self.achievements[ua.achievement_id].name,
                    "icon": self.achievements[ua.achievement_id].icon,
                    "earned_at": ua.earned_at.isoformat()
                }
                for ua in sorted(achievements, key=lambda x: x.earned_at, reverse=True)[:5]
            ]
        }

    def to_dict(self) -> dict:
        """Serialize gamification state."""
        return {
            "achievements": {aid: a.to_dict() for aid, a in self.achievements.items()},
            "user_achievements": {
                uid: [ua.to_dict() for ua in uas]
                for uid, uas in self.user_achievements.items()
            },
            "challenges": {cid: c.to_dict() for cid, c in self.challenges.items()},
            "streaks": {uid: s.to_dict() for uid, s in self.streaks.items()}
        }


# Helper functions

def create_weekly_challenge(
    challenge_id: str,
    name: str,
    description: str,
    points: int,
    requirement: Dict
) -> Challenge:
    """Create a challenge that runs for one week starting today."""
    start = datetime.now()
    end = start + timedelta(days=7)

    return Challenge(
        challenge_id=challenge_id,
        name=name,
        description=description,
        points=points,
        start_date=start,
        end_date=end,
        requirement=requirement
    )


def calculate_level_from_points(points: int) -> int:
    """Calculate level based on total points (1000 points per level)."""
    return 1 + (points // 1000)


def points_to_next_level(current_points: int) -> int:
    """Calculate points needed to reach next level."""
    current_level = calculate_level_from_points(current_points)
    next_level_points = current_level * 1000
    return next_level_points - current_points
