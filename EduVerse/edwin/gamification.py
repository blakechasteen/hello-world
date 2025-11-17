"""
EdWIN Gamification Engine

Core gamification system with XP, levels, progression, and multipliers.

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import math


class XPSource(Enum):
    """Sources of XP awards"""
    QUESTION_ASKED = "question_asked"
    OBJECTIVE_MASTERED = "objective_mastered"
    PERFECT_SCORE = "perfect_score"
    HELP_PEER = "help_peer"
    DAILY_LOGIN = "daily_login"
    STREAK_BONUS = "streak_bonus"
    CHALLENGE_COMPLETED = "challenge_completed"
    SPEED_BONUS = "speed_bonus"
    QUIZ_COMPLETED = "quiz_completed"
    READING_COMPLETED = "reading_completed"
    VIDEO_WATCHED = "video_watched"
    PRACTICE_PROBLEM = "practice_problem"
    SUBJECT_MILESTONE = "subject_milestone"
    GRADE_MILESTONE = "grade_milestone"


# XP Awards Configuration
XP_AWARDS = {
    XPSource.QUESTION_ASKED: 10,
    XPSource.OBJECTIVE_MASTERED: 50,
    XPSource.PERFECT_SCORE: 100,
    XPSource.HELP_PEER: 25,
    XPSource.DAILY_LOGIN: 5,
    XPSource.CHALLENGE_COMPLETED: 75,
    XPSource.SPEED_BONUS: 20,
    XPSource.QUIZ_COMPLETED: 30,
    XPSource.READING_COMPLETED: 15,
    XPSource.VIDEO_WATCHED: 10,
    XPSource.PRACTICE_PROBLEM: 8,
    XPSource.SUBJECT_MILESTONE: 200,
    XPSource.GRADE_MILESTONE: 500,
}


def streak_bonus_xp(days: int) -> int:
    """Calculate streak bonus XP based on days"""
    if days < 7:
        return 0
    elif days < 30:
        return days * 5  # 35-145 XP
    elif days < 100:
        return days * 10  # 300-990 XP
    else:
        return days * 15  # 1500+ XP


def get_xp_multiplier(streak_days: int) -> float:
    """Get XP multiplier based on streak length"""
    if streak_days >= 100:
        return 2.0  # 2x for 100+ day streak
    elif streak_days >= 30:
        return 1.5  # 1.5x for 30+ day streak
    elif streak_days >= 7:
        return 1.1  # 1.1x for 7+ day streak
    else:
        return 1.0  # No multiplier


@dataclass
class XPTransaction:
    """Record of an XP award"""
    timestamp: datetime
    source: XPSource
    amount: int
    multiplier: float
    total: int
    student_id: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class LevelInfo:
    """Information about a level"""
    level: int
    xp_required: int
    xp_total_to_level: int  # Cumulative XP needed to reach this level
    title: str
    reward_description: str


class LevelSystem:
    """
    Level progression system (1-100)

    XP required increases exponentially with level.
    Formula: XP = 100 * level^1.5

    Example progression:
    - Level 1→2: 100 XP
    - Level 2→3: 283 XP
    - Level 5→6: 1,118 XP
    - Level 10→11: 3,162 XP
    - Level 50→51: 35,355 XP
    - Level 99→100: 98,995 XP
    """

    MAX_LEVEL = 100

    # Level titles (every 10 levels)
    LEVEL_TITLES = {
        1: "Novice Scholar",
        10: "Dedicated Learner",
        20: "Knowledge Seeker",
        30: "Skilled Student",
        40: "Advanced Scholar",
        50: "Expert Learner",
        60: "Master Student",
        70: "Academic Champion",
        80: "Learning Legend",
        90: "Grand Scholar",
        100: "EdWIN Master"
    }

    @staticmethod
    def xp_for_level(level: int) -> int:
        """Calculate XP required to advance from level to level+1"""
        if level >= LevelSystem.MAX_LEVEL:
            return 0
        return int(100 * math.pow(level, 1.5))

    @staticmethod
    def total_xp_for_level(level: int) -> int:
        """Calculate total cumulative XP needed to reach a level"""
        total = 0
        for lvl in range(1, level):
            total += LevelSystem.xp_for_level(lvl)
        return total

    @staticmethod
    def level_from_xp(total_xp: int) -> int:
        """Calculate level from total XP"""
        level = 1
        xp_accumulated = 0

        while level < LevelSystem.MAX_LEVEL:
            xp_needed = LevelSystem.xp_for_level(level)
            if xp_accumulated + xp_needed > total_xp:
                break
            xp_accumulated += xp_needed
            level += 1

        return level

    @staticmethod
    def progress_to_next_level(total_xp: int) -> tuple[int, int, float]:
        """
        Get progress to next level

        Returns:
            (current_xp, xp_needed, progress_fraction)
        """
        current_level = LevelSystem.level_from_xp(total_xp)

        if current_level >= LevelSystem.MAX_LEVEL:
            return (0, 0, 1.0)

        xp_for_current = LevelSystem.total_xp_for_level(current_level)
        xp_for_next = LevelSystem.total_xp_for_level(current_level + 1)

        current_progress = total_xp - xp_for_current
        xp_needed = xp_for_next - xp_for_current

        progress = current_progress / xp_needed if xp_needed > 0 else 1.0

        return (current_progress, xp_needed, progress)

    @staticmethod
    def get_level_title(level: int) -> str:
        """Get title for a level"""
        # Find the highest title <= level
        for title_level in sorted(LevelSystem.LEVEL_TITLES.keys(), reverse=True):
            if level >= title_level:
                return LevelSystem.LEVEL_TITLES[title_level]
        return "Beginner"

    @staticmethod
    def get_level_info(level: int) -> LevelInfo:
        """Get complete information about a level"""
        xp_required = LevelSystem.xp_for_level(level)
        xp_total = LevelSystem.total_xp_for_level(level)
        title = LevelSystem.get_level_title(level)

        # Reward descriptions (examples)
        rewards = {
            10: "Unlock custom avatar",
            20: "Unlock dashboard themes",
            30: "Unlock advanced statistics",
            40: "Unlock peer tutoring features",
            50: "Unlock expert mode",
            60: "Unlock challenge creator",
            70: "Unlock all subjects",
            80: "Unlock prestige mode",
            90: "Unlock legendary status",
            100: "Unlock EdWIN Master badge"
        }

        reward = rewards.get(level, "Keep learning!")

        return LevelInfo(
            level=level,
            xp_required=xp_required,
            xp_total_to_level=xp_total,
            title=title,
            reward_description=reward
        )


@dataclass
class SubjectXP:
    """Subject-specific XP tracking"""
    subject: str
    total_xp: int = 0
    level: int = 1
    objectives_mastered: int = 0


class GamificationEngine:
    """
    Core gamification engine for EdWIN

    Features:
    - XP awards from multiple sources
    - Level progression (1-100)
    - Streak-based XP multipliers
    - Subject-specific XP tracks
    - XP transaction history

    Example:
        >>> engine = GamificationEngine(student_id="student_001")
        >>>
        >>> # Award XP for mastering objective
        >>> result = engine.award_xp(
        ...     source=XPSource.OBJECTIVE_MASTERED,
        ...     metadata={"objective_id": "math.algebra.8.linear_eq"}
        ... )
        >>> print(f"Awarded {result['xp_awarded']} XP")
        >>> print(f"Level: {result['new_level']}")
        >>>
        >>> # Check progress
        >>> progress = engine.get_progress()
        >>> print(f"Level {progress['level']} - {progress['progress_percent']:.1f}% to next")
    """

    def __init__(
        self,
        student_id: str,
        starting_xp: int = 0,
        starting_streak: int = 0
    ):
        self.student_id = student_id
        self.total_xp = starting_xp
        self.current_streak = starting_streak

        # Subject-specific tracking
        self.subject_xp: Dict[str, SubjectXP] = {}

        # Transaction history
        self.transactions: List[XPTransaction] = []

        # Metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def award_xp(
        self,
        source: XPSource,
        subject: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Award XP to student

        Args:
            source: Source of XP
            subject: Optional subject for subject-specific XP
            metadata: Optional metadata (objective_id, etc.)

        Returns:
            Dict with award details (xp_awarded, new_total, new_level, leveled_up)
        """
        # Get base XP
        if source == XPSource.STREAK_BONUS:
            base_xp = streak_bonus_xp(self.current_streak)
        else:
            base_xp = XP_AWARDS.get(source, 0)

        # Apply streak multiplier
        multiplier = get_xp_multiplier(self.current_streak)
        total_xp = int(base_xp * multiplier)

        # Record old level
        old_level = LevelSystem.level_from_xp(self.total_xp)

        # Award XP
        self.total_xp += total_xp

        # Calculate new level
        new_level = LevelSystem.level_from_xp(self.total_xp)
        leveled_up = new_level > old_level

        # Update subject XP if applicable
        if subject:
            if subject not in self.subject_xp:
                self.subject_xp[subject] = SubjectXP(subject=subject)

            self.subject_xp[subject].total_xp += total_xp
            self.subject_xp[subject].level = LevelSystem.level_from_xp(
                self.subject_xp[subject].total_xp
            )

            # Track objectives mastered
            if source == XPSource.OBJECTIVE_MASTERED:
                self.subject_xp[subject].objectives_mastered += 1

        # Record transaction
        transaction = XPTransaction(
            timestamp=datetime.utcnow(),
            source=source,
            amount=base_xp,
            multiplier=multiplier,
            total=total_xp,
            student_id=self.student_id,
            metadata=metadata or {}
        )
        self.transactions.append(transaction)

        self.updated_at = datetime.utcnow()

        return {
            "xp_awarded": total_xp,
            "base_xp": base_xp,
            "multiplier": multiplier,
            "new_total": self.total_xp,
            "old_level": old_level,
            "new_level": new_level,
            "leveled_up": leveled_up,
            "level_title": LevelSystem.get_level_title(new_level) if leveled_up else None
        }

    def update_streak(self, new_streak: int):
        """Update current streak (affects multiplier)"""
        self.current_streak = new_streak
        self.updated_at = datetime.utcnow()

    def get_progress(self) -> Dict:
        """Get current progress information"""
        level = LevelSystem.level_from_xp(self.total_xp)
        current_xp, xp_needed, progress_fraction = LevelSystem.progress_to_next_level(self.total_xp)

        return {
            "student_id": self.student_id,
            "total_xp": self.total_xp,
            "level": level,
            "level_title": LevelSystem.get_level_title(level),
            "current_xp": current_xp,
            "xp_needed": xp_needed,
            "progress_percent": progress_fraction * 100,
            "current_streak": self.current_streak,
            "xp_multiplier": get_xp_multiplier(self.current_streak),
            "next_level_title": LevelSystem.get_level_title(level + 1) if level < 100 else "Max Level"
        }

    def get_subject_progress(self, subject: str) -> Optional[Dict]:
        """Get progress for a specific subject"""
        if subject not in self.subject_xp:
            return None

        subj_xp = self.subject_xp[subject]
        current_xp, xp_needed, progress = LevelSystem.progress_to_next_level(subj_xp.total_xp)

        return {
            "subject": subject,
            "total_xp": subj_xp.total_xp,
            "level": subj_xp.level,
            "level_title": LevelSystem.get_level_title(subj_xp.level),
            "current_xp": current_xp,
            "xp_needed": xp_needed,
            "progress_percent": progress * 100,
            "objectives_mastered": subj_xp.objectives_mastered
        }

    def get_all_subjects(self) -> List[Dict]:
        """Get progress for all subjects"""
        return [
            self.get_subject_progress(subject)
            for subject in self.subject_xp.keys()
        ]

    def get_recent_transactions(self, limit: int = 10) -> List[XPTransaction]:
        """Get recent XP transactions"""
        return self.transactions[-limit:]

    def get_xp_by_source(self) -> Dict[str, int]:
        """Get XP breakdown by source"""
        breakdown = {}
        for transaction in self.transactions:
            source_name = transaction.source.value
            breakdown[source_name] = breakdown.get(source_name, 0) + transaction.total
        return breakdown

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "student_id": self.student_id,
            "total_xp": self.total_xp,
            "current_streak": self.current_streak,
            "subject_xp": {
                subject: {
                    "total_xp": xp.total_xp,
                    "level": xp.level,
                    "objectives_mastered": xp.objectives_mastered
                }
                for subject, xp in self.subject_xp.items()
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GamificationEngine":
        """Deserialize from dictionary"""
        engine = cls(
            student_id=data["student_id"],
            starting_xp=data["total_xp"],
            starting_streak=data["current_streak"]
        )

        # Restore subject XP
        for subject, xp_data in data.get("subject_xp", {}).items():
            engine.subject_xp[subject] = SubjectXP(
                subject=subject,
                total_xp=xp_data["total_xp"],
                level=xp_data["level"],
                objectives_mastered=xp_data["objectives_mastered"]
            )

        return engine


# Helper functions

def create_gamification_engine(student_id: str) -> GamificationEngine:
    """Create new gamification engine for student"""
    return GamificationEngine(student_id=student_id)


if __name__ == "__main__":
    # Demo
    print("=== EdWIN Gamification Engine Demo ===\n")

    engine = GamificationEngine(student_id="student_001")

    # Award various XP
    print("1. Daily login:")
    result = engine.award_xp(XPSource.DAILY_LOGIN)
    print(f"   Awarded {result['xp_awarded']} XP\n")

    print("2. Master an objective:")
    result = engine.award_xp(
        XPSource.OBJECTIVE_MASTERED,
        subject="math",
        metadata={"objective_id": "math.algebra.8.linear_eq"}
    )
    print(f"   Awarded {result['xp_awarded']} XP")
    if result['leveled_up']:
        print(f"   🎉 Level up! Now level {result['new_level']} ({result['level_title']})")
    print()

    print("3. Perfect score:")
    result = engine.award_xp(XPSource.PERFECT_SCORE, subject="math")
    print(f"   Awarded {result['xp_awarded']} XP\n")

    # Simulate streak
    print("4. Adding 7-day streak (1.1x multiplier):")
    engine.update_streak(7)
    result = engine.award_xp(XPSource.CHALLENGE_COMPLETED)
    print(f"   Base: {result['base_xp']} XP × {result['multiplier']} = {result['xp_awarded']} XP\n")

    # Get progress
    print("=== Current Progress ===")
    progress = engine.get_progress()
    print(f"Level: {progress['level']} ({progress['level_title']})")
    print(f"Total XP: {progress['total_xp']}")
    print(f"Progress to next: {progress['current_xp']}/{progress['xp_needed']} ({progress['progress_percent']:.1f}%)")
    print(f"Streak: {progress['current_streak']} days (×{progress['xp_multiplier']} XP)")

    # Subject progress
    print("\n=== Subject Progress ===")
    for subject_prog in engine.get_all_subjects():
        print(f"{subject_prog['subject'].upper()}: Level {subject_prog['level']} ({subject_prog['total_xp']} XP)")
        print(f"  {subject_prog['objectives_mastered']} objectives mastered")

    # XP breakdown
    print("\n=== XP by Source ===")
    breakdown = engine.get_xp_by_source()
    for source, xp in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {xp} XP")
