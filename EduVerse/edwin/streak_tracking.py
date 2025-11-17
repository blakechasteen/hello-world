"""
EdWIN Streak Tracking System

Motivational streak system with rewards and recovery options.

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
from enum import Enum


class StreakType(Enum):
    """Types of streaks to track"""
    DAILY_LOGIN = "daily_login"  # Login once per day
    DAILY_MASTERY = "daily_mastery"  # Master 1+ objective daily
    QUESTION_STREAK = "question_streak"  # Ask 3+ questions daily
    PERFECT_SCORE = "perfect_score"  # Perfect scores only


@dataclass
class StreakMilestone:
    """Milestone reward for streak length"""
    days: int
    xp_multiplier: float
    reward_description: str
    badge_id: Optional[str] = None


# Streak milestones and rewards
STREAK_MILESTONES = [
    StreakMilestone(7, 1.1, "1.1x XP multiplier", "streak_7"),
    StreakMilestone(14, 1.15, "1.15x XP multiplier"),
    StreakMilestone(30, 1.5, "1.5x XP multiplier + streak freeze", "streak_30"),
    StreakMilestone(60, 1.75, "1.75x XP multiplier"),
    StreakMilestone(100, 2.0, "2x XP multiplier + special badge", "streak_100"),
    StreakMilestone(180, 2.5, "2.5x XP multiplier"),
    StreakMilestone(365, 3.0, "3x XP multiplier + legendary status", "streak_365"),
]


def get_streak_multiplier(days: int) -> float:
    """Get XP multiplier based on streak length"""
    multiplier = 1.0
    for milestone in STREAK_MILESTONES:
        if days >= milestone.days:
            multiplier = milestone.xp_multiplier
    return multiplier


@dataclass
class StreakDay:
    """Record of a single day in a streak"""
    date: date
    completed: bool
    objective_count: int = 0
    question_count: int = 0
    perfect_scores: int = 0
    metadata: Dict = field(default_factory=dict)


class Streak:
    """
    A single streak tracker

    Features:
    - Daily tracking
    - Automatic expiration (48-hour grace period)
    - Streak freezes (preserve streak)
    - Milestone rewards

    Example:
        >>> streak = Streak(
        ...     student_id="student_001",
        ...     streak_type=StreakType.DAILY_LOGIN
        ... )
        >>>
        >>> # Check in today
        >>> result = streak.check_in()
        >>> print(f"Current streak: {result['current_days']} days")
        >>>
        >>> # Use streak freeze
        >>> if streak.freezes_available > 0:
        ...     streak.use_freeze()
    """

    def __init__(
        self,
        student_id: str,
        streak_type: StreakType,
        starting_streak: int = 0
    ):
        self.student_id = student_id
        self.streak_type = streak_type
        self.current_days = starting_streak
        self.best_days = starting_streak
        self.total_days = starting_streak

        # Tracking
        self.last_check_in: Optional[datetime] = None
        self.streak_start: Optional[datetime] = None
        self.days_history: List[StreakDay] = []

        # Streak protection
        self.freezes_available = 2  # 2 free freezes per month
        self.freezes_used = 0
        self.freeze_active: Optional[date] = None

        # Milestones
        self.milestones_reached: List[int] = []

        # Metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def check_in(
        self,
        objectives_mastered: int = 0,
        questions_asked: int = 0,
        perfect_scores: int = 0
    ) -> Dict:
        """
        Check in for today

        Args:
            objectives_mastered: Objectives mastered today
            questions_asked: Questions asked today
            perfect_scores: Perfect scores today

        Returns:
            Dict with streak status
        """
        now = datetime.utcnow()
        today = now.date()

        # Check if already checked in today
        if self.last_check_in and self.last_check_in.date() == today:
            return {
                "already_checked_in": True,
                "current_days": self.current_days,
                "message": "Already checked in today!"
            }

        # Check if streak is expired
        if self._is_expired(today):
            # Streak broken!
            old_streak = self.current_days
            self.current_days = 1  # Start fresh
            self.streak_start = now

            return {
                "streak_broken": True,
                "old_streak": old_streak,
                "current_days": 1,
                "message": f"Streak broken! Previous best: {old_streak} days. Starting fresh!"
            }

        # Valid check-in
        if self.last_check_in is None:
            # First check-in
            self.current_days = 1
            self.streak_start = now
        else:
            # Continue streak
            self.current_days += 1

        self.total_days += 1
        self.last_check_in = now

        # Record day
        self.days_history.append(
            StreakDay(
                date=today,
                completed=True,
                objective_count=objectives_mastered,
                question_count=questions_asked,
                perfect_scores=perfect_scores
            )
        )

        # Update best
        if self.current_days > self.best_days:
            self.best_days = self.current_days

        # Check milestones
        new_milestones = self._check_milestones()

        # Refill freezes monthly
        self._refill_freezes()

        self.updated_at = now

        return {
            "success": True,
            "current_days": self.current_days,
            "best_days": self.best_days,
            "xp_multiplier": get_streak_multiplier(self.current_days),
            "new_milestones": new_milestones,
            "message": f"Streak extended to {self.current_days} days!"
        }

    def use_freeze(self) -> bool:
        """
        Use a streak freeze to protect today's streak

        Returns:
            True if freeze used successfully, False if none available
        """
        if self.freezes_available <= 0:
            return False

        today = date.today()
        self.freeze_active = today
        self.freezes_available -= 1
        self.freezes_used += 1
        self.updated_at = datetime.utcnow()

        return True

    def _is_expired(self, today: date) -> bool:
        """Check if streak has expired"""
        if self.last_check_in is None:
            return False

        # Check freeze
        if self.freeze_active == today:
            return False  # Protected by freeze

        # Grace period: 48 hours
        hours_since = (datetime.now() - self.last_check_in).total_seconds() / 3600

        return hours_since > 48

    def _check_milestones(self) -> List[Dict]:
        """Check for newly reached milestones"""
        new_milestones = []

        for milestone in STREAK_MILESTONES:
            if (self.current_days >= milestone.days and
                    milestone.days not in self.milestones_reached):

                self.milestones_reached.append(milestone.days)

                new_milestones.append({
                    "days": milestone.days,
                    "xp_multiplier": milestone.xp_multiplier,
                    "reward": milestone.reward_description,
                    "badge_id": milestone.badge_id
                })

        return new_milestones

    def _refill_freezes(self):
        """Refill freezes monthly (max 2)"""
        now = datetime.utcnow()

        # Check if month changed
        if (self.last_check_in and
            (now.month != self.last_check_in.month or
             now.year != self.last_check_in.year)):

            # Refill to 2
            self.freezes_available = 2

    def get_status(self) -> Dict:
        """Get current streak status"""
        today = date.today()
        is_expired = self._is_expired(today)
        needs_check_in = (
            self.last_check_in is None or
            self.last_check_in.date() < today
        )

        return {
            "student_id": self.student_id,
            "streak_type": self.streak_type.value,
            "current_days": self.current_days,
            "best_days": self.best_days,
            "total_days": self.total_days,
            "is_expired": is_expired,
            "needs_check_in": needs_check_in,
            "last_check_in": self.last_check_in.isoformat() if self.last_check_in else None,
            "streak_start": self.streak_start.isoformat() if self.streak_start else None,
            "xp_multiplier": get_streak_multiplier(self.current_days),
            "freezes_available": self.freezes_available,
            "freezes_used": self.freezes_used,
            "milestones_reached": self.milestones_reached,
            "next_milestone": self._get_next_milestone()
        }

    def _get_next_milestone(self) -> Optional[Dict]:
        """Get next milestone to reach"""
        for milestone in STREAK_MILESTONES:
            if milestone.days not in self.milestones_reached:
                days_to_go = milestone.days - self.current_days
                return {
                    "days": milestone.days,
                    "days_to_go": days_to_go,
                    "reward": milestone.reward_description
                }
        return None

    def get_calendar(self, days: int = 30) -> List[StreakDay]:
        """Get streak calendar (last N days)"""
        return self.days_history[-days:]


class StreakManager:
    """
    Manages multiple streak types for a student

    Features:
    - Multiple concurrent streaks
    - Automatic expiration checking
    - Streak recovery with freezes
    - Milestone rewards

    Example:
        >>> manager = StreakManager(student_id="student_001")
        >>>
        >>> # Daily login
        >>> result = manager.check_in_login()
        >>>
        >>> # Daily mastery (master 1+ objectives)
        >>> result = manager.check_in_mastery(objectives_mastered=2)
        >>>
        >>> # Get all streaks
        >>> summary = manager.get_summary()
    """

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.streaks: Dict[StreakType, Streak] = {}

        # Initialize all streak types
        for streak_type in StreakType:
            self.streaks[streak_type] = Streak(
                student_id=student_id,
                streak_type=streak_type
            )

    def check_in_login(self) -> Dict:
        """Check in for daily login streak"""
        return self.streaks[StreakType.DAILY_LOGIN].check_in()

    def check_in_mastery(self, objectives_mastered: int) -> Dict:
        """Check in for daily mastery streak (requires 1+ objectives)"""
        if objectives_mastered >= 1:
            return self.streaks[StreakType.DAILY_MASTERY].check_in(
                objectives_mastered=objectives_mastered
            )
        return {
            "success": False,
            "message": "Need to master at least 1 objective for mastery streak"
        }

    def check_in_questions(self, questions_asked: int) -> Dict:
        """Check in for question streak (requires 3+ questions)"""
        if questions_asked >= 3:
            return self.streaks[StreakType.QUESTION_STREAK].check_in(
                questions_asked=questions_asked
            )
        return {
            "success": False,
            "message": "Need to ask at least 3 questions for question streak"
        }

    def check_in_perfect(self, perfect_scores: int) -> Dict:
        """Check in for perfect score streak"""
        if perfect_scores >= 1:
            return self.streaks[StreakType.PERFECT_SCORE].check_in(
                perfect_scores=perfect_scores
            )
        return {
            "success": False,
            "message": "Need at least 1 perfect score for perfect streak"
        }

    def use_freeze(self, streak_type: StreakType) -> bool:
        """Use freeze on a specific streak"""
        return self.streaks[streak_type].use_freeze()

    def get_summary(self) -> Dict:
        """Get summary of all streaks"""
        return {
            "student_id": self.student_id,
            "streaks": {
                streak_type.value: streak.get_status()
                for streak_type, streak in self.streaks.items()
            },
            "total_xp_multiplier": max(
                streak.get_status()["xp_multiplier"]
                for streak in self.streaks.values()
            )
        }

    def get_longest_streak(self) -> Dict:
        """Get the longest current streak"""
        longest = max(
            self.streaks.values(),
            key=lambda s: s.current_days
        )

        return {
            "streak_type": longest.streak_type.value,
            "days": longest.current_days,
            "xp_multiplier": get_streak_multiplier(longest.current_days)
        }


# ========== Helper Functions ==========

def create_streak_manager(student_id: str) -> StreakManager:
    """Create new streak manager for student"""
    return StreakManager(student_id=student_id)


if __name__ == "__main__":
    # Demo
    print("=== EdWIN Streak Tracking Demo ===\n")

    manager = StreakManager(student_id="student_001")

    # Day 1: Login
    print("Day 1: Login")
    result = manager.check_in_login()
    print(f"   {result['message']}")
    print(f"   Current: {result['current_days']} days\n")

    # Day 1: Master an objective
    print("Day 1: Master 2 objectives")
    result = manager.check_in_mastery(objectives_mastered=2)
    print(f"   {result['message']}")
    print()

    # Simulate 7-day streak
    print("Simulating 7-day streak...")
    login_streak = manager.streaks[StreakType.DAILY_LOGIN]
    for day in range(2, 8):
        # Simulate time passing
        login_streak.last_check_in = datetime.utcnow() - timedelta(days=8-day)
        result = login_streak.check_in()

    print(f"   Current streak: {result['current_days']} days")
    print(f"   XP multiplier: {result['xp_multiplier']}x")

    if result['new_milestones']:
        for milestone in result['new_milestones']:
            print(f"   🎉 Milestone: {milestone['days']} days - {milestone['reward']}")

    # Get summary
    print("\n=== Streak Summary ===")
    summary = manager.get_summary()
    for streak_type, status in summary['streaks'].items():
        if status['current_days'] > 0:
            print(f"{streak_type}: {status['current_days']} days (×{status['xp_multiplier']})")
            if status['needs_check_in']:
                print("   ⚠️  Needs check-in today!")

    # Longest streak
    print("\n=== Longest Streak ===")
    longest = manager.get_longest_streak()
    print(f"{longest['streak_type']}: {longest['days']} days")
    print(f"XP Multiplier: ×{longest['xp_multiplier']}")

    # Freezes
    print("\n=== Streak Freezes ===")
    login_status = manager.streaks[StreakType.DAILY_LOGIN].get_status()
    print(f"Available: {login_status['freezes_available']}")
    print(f"Used: {login_status['freezes_used']}")
