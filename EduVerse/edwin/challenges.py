"""
EdWIN Challenge System

Daily/weekly challenges and special events for engagement.

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Callable
from enum import Enum
import random


class ChallengeType(Enum):
    """Types of challenges"""
    DAILY = "daily"  # Resets daily at midnight
    WEEKLY = "weekly"  # Resets Monday
    SPECIAL_EVENT = "special_event"  # Limited time events
    PERSONAL = "personal"  # Personalized challenges


class ChallengeCategory(Enum):
    """Challenge categories"""
    MASTERY = "mastery"  # Master objectives
    ENGAGEMENT = "engagement"  # Daily activity
    SOCIAL = "social"  # Help others
    SPEED = "speed"  # Timed challenges
    EXPLORATION = "exploration"  # Try new subjects


@dataclass
class ChallengeRequirement:
    """A single requirement for a challenge"""
    description: str
    target_value: float
    current_value: float = 0.0
    unit: str = ""  # "objectives", "questions", "hours", etc.

    @property
    def completed(self) -> bool:
        return self.current_value >= self.target_value

    @property
    def progress_percent(self) -> float:
        if self.target_value == 0:
            return 100.0
        return min(100.0, (self.current_value / self.target_value) * 100)


@dataclass
class Challenge:
    """
    A single challenge

    Attributes:
        id: Unique challenge ID
        name: Display name
        description: Challenge description
        challenge_type: Daily, weekly, or special
        category: Challenge category
        requirements: List of requirements
        xp_reward: XP awarded on completion
        bonus_rewards: Additional rewards (badges, items, etc.)
        expires_at: When challenge expires
        completed: Whether completed
        completed_at: When completed
    """
    id: str
    name: str
    description: str
    challenge_type: ChallengeType
    category: ChallengeCategory
    requirements: List[ChallengeRequirement]
    xp_reward: int
    bonus_rewards: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    completed: bool = False
    completed_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def all_requirements_met(self) -> bool:
        """Check if all requirements are completed"""
        return all(req.completed for req in self.requirements)

    @property
    def overall_progress(self) -> float:
        """Get overall progress percentage"""
        if not self.requirements:
            return 0.0
        return sum(req.progress_percent for req in self.requirements) / len(self.requirements)

    def update_progress(self, requirement_index: int, value: float):
        """Update progress on a specific requirement"""
        if 0 <= requirement_index < len(self.requirements):
            self.requirements[requirement_index].current_value = value

            # Check if challenge completed
            if self.all_requirements_met and not self.completed:
                self.completed = True
                self.completed_at = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if challenge has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


# ========== Challenge Templates ==========

def create_daily_challenges() -> List[Challenge]:
    """Create daily challenge pool"""
    tomorrow = datetime.utcnow() + timedelta(days=1)
    tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

    challenges = []

    # Master objectives
    challenges.append(Challenge(
        id="daily_master_3",
        name="Daily Mastery",
        description="Master 3 objectives today",
        challenge_type=ChallengeType.DAILY,
        category=ChallengeCategory.MASTERY,
        requirements=[
            ChallengeRequirement("Master objectives", 3, 0, "objectives")
        ],
        xp_reward=50,
        expires_at=tomorrow
    ))

    # Answer questions
    challenges.append(Challenge(
        id="daily_questions_10",
        name="Curious Mind",
        description="Ask 10 questions today",
        challenge_type=ChallengeType.DAILY,
        category=ChallengeCategory.ENGAGEMENT,
        requirements=[
            ChallengeRequirement("Ask questions", 10, 0, "questions")
        ],
        xp_reward=30,
        expires_at=tomorrow
    ))

    # Maintain streak
    challenges.append(Challenge(
        id="daily_streak",
        name="Keep the Streak",
        description="Maintain your learning streak",
        challenge_type=ChallengeType.DAILY,
        category=ChallengeCategory.ENGAGEMENT,
        requirements=[
            ChallengeRequirement("Check in today", 1, 0, "check-ins")
        ],
        xp_reward=20,
        expires_at=tomorrow
    ))

    # Try new subject
    challenges.append(Challenge(
        id="daily_new_subject",
        name="Branch Out",
        description="Try a new subject today",
        challenge_type=ChallengeType.DAILY,
        category=ChallengeCategory.EXPLORATION,
        requirements=[
            ChallengeRequirement("Try new subject", 1, 0, "subjects")
        ],
        xp_reward=40,
        expires_at=tomorrow
    ))

    # Help a peer
    challenges.append(Challenge(
        id="daily_help_peer",
        name="Be Helpful",
        description="Help a classmate today",
        challenge_type=ChallengeType.DAILY,
        category=ChallengeCategory.SOCIAL,
        requirements=[
            ChallengeRequirement("Help peers", 1, 0, "peers")
        ],
        xp_reward=25,
        expires_at=tomorrow
    ))

    # Perfect score
    challenges.append(Challenge(
        id="daily_perfect",
        name="Perfectionist",
        description="Get a perfect score on any objective",
        challenge_type=ChallengeType.DAILY,
        category=ChallengeCategory.MASTERY,
        requirements=[
            ChallengeRequirement("Perfect scores", 1, 0, "scores")
        ],
        xp_reward=60,
        expires_at=tomorrow
    ))

    # Speed challenge
    challenges.append(Challenge(
        id="daily_speed",
        name="Speed Learner",
        description="Complete an objective in under 30 minutes",
        challenge_type=ChallengeType.DAILY,
        category=ChallengeCategory.SPEED,
        requirements=[
            ChallengeRequirement("Fast completions", 1, 0, "objectives")
        ],
        xp_reward=35,
        expires_at=tomorrow
    ))

    # Study time
    challenges.append(Challenge(
        id="daily_study_time",
        name="Dedicated Hour",
        description="Study for at least 1 hour today",
        challenge_type=ChallengeType.DAILY,
        category=ChallengeCategory.ENGAGEMENT,
        requirements=[
            ChallengeRequirement("Study time", 60, 0, "minutes")
        ],
        xp_reward=45,
        expires_at=tomorrow
    ))

    return challenges


def create_weekly_challenges() -> List[Challenge]:
    """Create weekly challenge pool"""
    # Next Monday
    today = datetime.utcnow()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = today + timedelta(days=days_until_monday)
    next_monday = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)

    challenges = []

    # Master many objectives
    challenges.append(Challenge(
        id="weekly_master_15",
        name="Weekly Mastery Goal",
        description="Master 15 objectives this week",
        challenge_type=ChallengeType.WEEKLY,
        category=ChallengeCategory.MASTERY,
        requirements=[
            ChallengeRequirement("Master objectives", 15, 0, "objectives")
        ],
        xp_reward=200,
        expires_at=next_monday
    ))

    # Study hours
    challenges.append(Challenge(
        id="weekly_study_5h",
        name="Study Marathon",
        description="Study for 5 hours this week",
        challenge_type=ChallengeType.WEEKLY,
        category=ChallengeCategory.ENGAGEMENT,
        requirements=[
            ChallengeRequirement("Study time", 300, 0, "minutes")
        ],
        xp_reward=150,
        expires_at=next_monday
    ))

    # Perfect scores
    challenges.append(Challenge(
        id="weekly_perfect_5",
        name="Perfectionist Week",
        description="Get perfect scores on 5 objectives",
        challenge_type=ChallengeType.WEEKLY,
        category=ChallengeCategory.MASTERY,
        requirements=[
            ChallengeRequirement("Perfect scores", 5, 0, "scores")
        ],
        xp_reward=300,
        bonus_rewards=["perfect_week_badge"],
        expires_at=next_monday
    ))

    # Complete all dailies
    challenges.append(Challenge(
        id="weekly_all_dailies",
        name="Daily Champion",
        description="Complete all daily challenges every day this week",
        challenge_type=ChallengeType.WEEKLY,
        category=ChallengeCategory.ENGAGEMENT,
        requirements=[
            ChallengeRequirement("Daily challenges completed", 7, 0, "days")
        ],
        xp_reward=500,
        bonus_rewards=["daily_champion_badge"],
        expires_at=next_monday
    ))

    # Social challenge
    challenges.append(Challenge(
        id="weekly_help_10",
        name="Community Helper",
        description="Help 10 different students this week",
        challenge_type=ChallengeType.WEEKLY,
        category=ChallengeCategory.SOCIAL,
        requirements=[
            ChallengeRequirement("Students helped", 10, 0, "students")
        ],
        xp_reward=250,
        expires_at=next_monday
    ))

    # Subject diversity
    challenges.append(Challenge(
        id="weekly_all_subjects",
        name="Renaissance Week",
        description="Master at least 1 objective in each subject",
        challenge_type=ChallengeType.WEEKLY,
        category=ChallengeCategory.EXPLORATION,
        requirements=[
            ChallengeRequirement("Math objectives", 1, 0, "objectives"),
            ChallengeRequirement("Science objectives", 1, 0, "objectives"),
            ChallengeRequirement("ELA objectives", 1, 0, "objectives"),
            ChallengeRequirement("Social Studies objectives", 1, 0, "objectives"),
            ChallengeRequirement("AI Readiness objectives", 1, 0, "objectives"),
        ],
        xp_reward=350,
        bonus_rewards=["renaissance_badge"],
        expires_at=next_monday
    ))

    return challenges


def create_special_event_challenges() -> List[Challenge]:
    """Create special event challenges (seasonal, etc.)"""
    challenges = []

    # Math Week
    challenges.append(Challenge(
        id="event_math_week",
        name="Math Week Challenge",
        description="Master 10 math objectives during Math Week",
        challenge_type=ChallengeType.SPECIAL_EVENT,
        category=ChallengeCategory.MASTERY,
        requirements=[
            ChallengeRequirement("Math objectives", 10, 0, "objectives")
        ],
        xp_reward=500,
        bonus_rewards=["math_week_badge", "2x_xp_boost_math"],
        expires_at=datetime.utcnow() + timedelta(days=7),
        metadata={"event_name": "Math Week"}
    ))

    # Reading Marathon
    challenges.append(Challenge(
        id="event_reading_24h",
        name="24-Hour Reading Marathon",
        description="Complete 5 reading objectives in 24 hours",
        challenge_type=ChallengeType.SPECIAL_EVENT,
        category=ChallengeCategory.MASTERY,
        requirements=[
            ChallengeRequirement("Reading objectives", 5, 0, "objectives")
        ],
        xp_reward=400,
        bonus_rewards=["reading_marathon_badge"],
        expires_at=datetime.utcnow() + timedelta(hours=24),
        metadata={"event_name": "Reading Marathon"}
    ))

    # Science Fair
    challenges.append(Challenge(
        id="event_science_fair",
        name="Science Fair Challenge",
        description="Master all science objectives for your grade",
        challenge_type=ChallengeType.SPECIAL_EVENT,
        category=ChallengeCategory.MASTERY,
        requirements=[
            ChallengeRequirement("Science objectives", 20, 0, "objectives")
        ],
        xp_reward=750,
        bonus_rewards=["science_fair_winner", "exclusive_avatar"],
        expires_at=datetime.utcnow() + timedelta(days=14),
        metadata={"event_name": "Science Fair"}
    ))

    # Speed Challenge Weekend
    challenges.append(Challenge(
        id="event_speed_weekend",
        name="Speed Challenge Weekend",
        description="Complete 10 objectives in under 20 minutes each",
        challenge_type=ChallengeType.SPECIAL_EVENT,
        category=ChallengeCategory.SPEED,
        requirements=[
            ChallengeRequirement("Speed completions", 10, 0, "objectives")
        ],
        xp_reward=600,
        bonus_rewards=["speed_demon_badge"],
        expires_at=datetime.utcnow() + timedelta(days=3),
        metadata={"event_name": "Speed Weekend"}
    ))

    return challenges


class ChallengeManager:
    """
    Manages challenges for students

    Features:
    - Daily/weekly challenge rotation
    - Progress tracking
    - Automatic expiration
    - Special events

    Example:
        >>> manager = ChallengeManager(student_id="student_001")
        >>>
        >>> # Get active challenges
        >>> active = manager.get_active_challenges()
        >>>
        >>> # Update progress
        >>> manager.update_progress(
        ...     challenge_id="daily_master_3",
        ...     requirement_index=0,
        ...     value=2
        ... )
        >>>
        >>> # Check for completions
        >>> completed = manager.get_completed_challenges()
    """

    def __init__(self, student_id: str, grade: int = 8):
        self.student_id = student_id
        self.grade = grade
        self.challenges: Dict[str, Challenge] = {}

        # Load initial challenges
        self._refresh_daily_challenges()
        self._refresh_weekly_challenges()

        # Metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def _refresh_daily_challenges(self):
        """Refresh daily challenges (select 3 random)"""
        # Remove expired daily challenges
        self.challenges = {
            cid: c for cid, c in self.challenges.items()
            if c.challenge_type != ChallengeType.DAILY or not c.is_expired()
        }

        # Check if we need new dailies
        daily_count = sum(
            1 for c in self.challenges.values()
            if c.challenge_type == ChallengeType.DAILY
        )

        if daily_count < 3:
            # Select 3 random daily challenges
            all_dailies = create_daily_challenges()
            selected = random.sample(all_dailies, min(3, len(all_dailies)))

            for challenge in selected:
                self.challenges[challenge.id] = challenge

    def _refresh_weekly_challenges(self):
        """Refresh weekly challenges (select 2 random)"""
        # Remove expired weekly challenges
        self.challenges = {
            cid: c for cid, c in self.challenges.items()
            if c.challenge_type != ChallengeType.WEEKLY or not c.is_expired()
        }

        # Check if we need new weeklies
        weekly_count = sum(
            1 for c in self.challenges.values()
            if c.challenge_type == ChallengeType.WEEKLY
        )

        if weekly_count < 2:
            # Select 2 random weekly challenges
            all_weeklies = create_weekly_challenges()
            selected = random.sample(all_weeklies, min(2, len(all_weeklies)))

            for challenge in selected:
                self.challenges[challenge.id] = challenge

    def add_special_event(self, challenge: Challenge):
        """Add a special event challenge"""
        self.challenges[challenge.id] = challenge
        self.updated_at = datetime.utcnow()

    def get_active_challenges(self) -> List[Challenge]:
        """Get all active (non-expired, non-completed) challenges"""
        # Refresh if needed
        self._refresh_daily_challenges()
        self._refresh_weekly_challenges()

        active = []
        for challenge in self.challenges.values():
            if not challenge.completed and not challenge.is_expired():
                active.append(challenge)

        return active

    def get_completed_challenges(self) -> List[Challenge]:
        """Get completed challenges"""
        return [c for c in self.challenges.values() if c.completed]

    def update_progress(
        self,
        challenge_id: str,
        requirement_index: int,
        value: float
    ) -> Optional[Dict]:
        """
        Update progress on a challenge

        Returns:
            Dict with completion info if challenge completed, None otherwise
        """
        if challenge_id not in self.challenges:
            return None

        challenge = self.challenges[challenge_id]

        # Already completed?
        if challenge.completed:
            return None

        # Update progress
        challenge.update_progress(requirement_index, value)

        # Check if newly completed
        if challenge.completed:
            self.updated_at = datetime.utcnow()

            return {
                "challenge_id": challenge.id,
                "challenge_name": challenge.name,
                "xp_reward": challenge.xp_reward,
                "bonus_rewards": challenge.bonus_rewards,
                "completed_at": challenge.completed_at.isoformat()
            }

        return None

    def get_progress_summary(self) -> Dict:
        """Get summary of challenge progress"""
        active = self.get_active_challenges()
        completed = self.get_completed_challenges()

        # Count by type
        daily_active = [c for c in active if c.challenge_type == ChallengeType.DAILY]
        weekly_active = [c for c in active if c.challenge_type == ChallengeType.WEEKLY]
        event_active = [c for c in active if c.challenge_type == ChallengeType.SPECIAL_EVENT]

        daily_completed = [c for c in completed if c.challenge_type == ChallengeType.DAILY]
        weekly_completed = [c for c in completed if c.challenge_type == ChallengeType.WEEKLY]

        return {
            "student_id": self.student_id,
            "active_challenges": len(active),
            "completed_challenges": len(completed),
            "daily": {
                "active": len(daily_active),
                "completed_today": len([
                    c for c in daily_completed
                    if c.completed_at and c.completed_at.date() == date.today()
                ])
            },
            "weekly": {
                "active": len(weekly_active),
                "completed_this_week": len(weekly_completed)
            },
            "events": {
                "active": len(event_active)
            },
            "total_xp_earned": sum(c.xp_reward for c in completed)
        }


# ========== Helper Functions ==========

def create_challenge_manager(student_id: str, grade: int = 8) -> ChallengeManager:
    """Create new challenge manager for student"""
    return ChallengeManager(student_id=student_id, grade=grade)


if __name__ == "__main__":
    # Demo
    print("=== EdWIN Challenge System Demo ===\n")

    manager = ChallengeManager(student_id="student_001", grade=8)

    # Get active challenges
    print("=== Active Challenges ===")
    active = manager.get_active_challenges()
    for i, challenge in enumerate(active, 1):
        print(f"\n{i}. {challenge.name} ({challenge.challenge_type.value})")
        print(f"   {challenge.description}")
        print(f"   Reward: {challenge.xp_reward} XP")
        for j, req in enumerate(challenge.requirements):
            print(f"   - {req.description}: {req.current_value}/{req.target_value} {req.unit} ({req.progress_percent:.0f}%)")

    # Simulate progress
    print("\n\n=== Simulating Progress ===")
    daily_challenge = active[0]
    print(f"Updating '{daily_challenge.name}'...")

    # Update progress
    result = manager.update_progress(
        challenge_id=daily_challenge.id,
        requirement_index=0,
        value=3  # Complete the requirement
    )

    if result:
        print(f"🎉 Challenge completed!")
        print(f"   Reward: +{result['xp_reward']} XP")
        if result['bonus_rewards']:
            print(f"   Bonus: {', '.join(result['bonus_rewards'])}")

    # Get summary
    print("\n=== Challenge Summary ===")
    summary = manager.get_progress_summary()
    print(f"Active: {summary['active_challenges']}")
    print(f"Completed: {summary['completed_challenges']}")
    print(f"Total XP earned: {summary['total_xp_earned']}")
    print(f"\nDaily: {summary['daily']['completed_today']}/3 completed today")
    print(f"Weekly: {summary['weekly']['active']} active")
