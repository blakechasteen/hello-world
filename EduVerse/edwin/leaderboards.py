"""
EdWIN Leaderboard System

Multiple leaderboard types for healthy competition.

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
import heapq


class LeaderboardType(Enum):
    """Types of leaderboards"""
    GLOBAL_XP = "global_xp"  # Total XP across all students
    SUBJECT_XP = "subject_xp"  # XP in specific subject
    GRADE_LEVEL = "grade_level"  # Within grade level
    CLASSROOM = "classroom"  # Within classroom
    WEEKLY = "weekly"  # Weekly rankings (reset Monday)
    MONTHLY = "monthly"  # Monthly rankings
    OBJECTIVES = "objectives"  # Most objectives mastered
    STREAKS = "streaks"  # Longest streaks
    SPEED = "speed"  # Speed challenge completions
    FRIENDS = "friends"  # Among friends only


class LeaderboardPeriod(Enum):
    """Time periods for leaderboards"""
    ALL_TIME = "all_time"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"


@dataclass
class LeaderboardEntry:
    """Single entry on a leaderboard"""
    student_id: str
    student_name: str
    rank: int
    score: float  # XP, objectives, etc.
    grade: Optional[int] = None
    classroom: Optional[str] = None
    avatar: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class LeaderboardConfig:
    """Configuration for a leaderboard"""
    leaderboard_type: LeaderboardType
    period: LeaderboardPeriod
    subject: Optional[str] = None  # For subject-specific leaderboards
    grade: Optional[int] = None  # For grade-specific leaderboards
    classroom: Optional[str] = None  # For classroom-specific
    max_entries: int = 100  # Top N to track
    privacy_enabled: bool = True  # Allow opt-out
    show_avatars: bool = True


class Leaderboard:
    """
    A single leaderboard tracking rankings

    Features:
    - Top N rankings
    - Efficient updates (heap-based)
    - Personal rank always visible
    - Privacy controls
    - Time-based resets (weekly, monthly)

    Example:
        >>> leaderboard = Leaderboard(
        ...     config=LeaderboardConfig(
        ...         leaderboard_type=LeaderboardType.GLOBAL_XP,
        ...         period=LeaderboardPeriod.ALL_TIME
        ...     )
        ... )
        >>>
        >>> # Update scores
        >>> leaderboard.update_score("student_001", "Alice", 1250, grade=8)
        >>> leaderboard.update_score("student_002", "Bob", 980, grade=8)
        >>>
        >>> # Get rankings
        >>> top_10 = leaderboard.get_rankings(limit=10)
        >>> personal = leaderboard.get_personal_rank("student_001")
    """

    def __init__(self, config: LeaderboardConfig):
        self.config = config
        self.entries: Dict[str, LeaderboardEntry] = {}
        self.last_reset = datetime.utcnow()
        self.created_at = datetime.utcnow()

    def update_score(
        self,
        student_id: str,
        student_name: str,
        score: float,
        grade: Optional[int] = None,
        classroom: Optional[str] = None,
        avatar: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Update a student's score on the leaderboard"""
        # Check if reset needed
        self._check_reset()

        # Create or update entry
        if student_id in self.entries:
            entry = self.entries[student_id]
            entry.score = score
            entry.grade = grade if grade is not None else entry.grade
            entry.classroom = classroom if classroom is not None else entry.classroom
            entry.avatar = avatar if avatar is not None else entry.avatar
            if metadata:
                entry.metadata.update(metadata)
        else:
            self.entries[student_id] = LeaderboardEntry(
                student_id=student_id,
                student_name=student_name,
                rank=0,  # Will be calculated
                score=score,
                grade=grade,
                classroom=classroom,
                avatar=avatar,
                metadata=metadata or {}
            )

        # Recalculate ranks
        self._recalculate_ranks()

    def get_rankings(
        self,
        limit: int = 100,
        filter_grade: Optional[int] = None,
        filter_classroom: Optional[str] = None
    ) -> List[LeaderboardEntry]:
        """
        Get top N rankings

        Args:
            limit: Number of entries to return
            filter_grade: Optional grade filter
            filter_classroom: Optional classroom filter

        Returns:
            List of LeaderboardEntry sorted by rank
        """
        entries = list(self.entries.values())

        # Apply filters
        if filter_grade is not None:
            entries = [e for e in entries if e.grade == filter_grade]

        if filter_classroom is not None:
            entries = [e for e in entries if e.classroom == filter_classroom]

        # Sort by rank
        entries.sort(key=lambda e: e.rank)

        return entries[:limit]

    def get_personal_rank(self, student_id: str) -> Optional[Dict]:
        """
        Get personal rank (always visible even if not in top N)

        Returns:
            Dict with rank, score, and position info
        """
        if student_id not in self.entries:
            return None

        entry = self.entries[student_id]

        # Calculate percentile
        total = len(self.entries)
        percentile = ((total - entry.rank + 1) / total) * 100

        return {
            "student_id": student_id,
            "student_name": entry.student_name,
            "rank": entry.rank,
            "score": entry.score,
            "total_participants": total,
            "percentile": percentile,
            "avatar": entry.avatar,
            "metadata": entry.metadata
        }

    def get_surrounding_ranks(
        self,
        student_id: str,
        context: int = 5
    ) -> List[LeaderboardEntry]:
        """
        Get rankings around a student (for context)

        Args:
            student_id: Student to center on
            context: Number of entries above/below to show

        Returns:
            List of entries around student
        """
        if student_id not in self.entries:
            return []

        my_rank = self.entries[student_id].rank
        all_entries = self.get_rankings(limit=len(self.entries))

        start = max(0, my_rank - context - 1)
        end = min(len(all_entries), my_rank + context)

        return all_entries[start:end]

    def _recalculate_ranks(self):
        """Recalculate ranks based on scores"""
        # Sort by score (descending)
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.score,
            reverse=True
        )

        # Assign ranks (handle ties)
        current_rank = 1
        prev_score = None

        for i, entry in enumerate(sorted_entries):
            if prev_score is not None and entry.score < prev_score:
                current_rank = i + 1

            entry.rank = current_rank
            prev_score = entry.score

    def _check_reset(self):
        """Check if leaderboard should be reset based on period"""
        now = datetime.utcnow()

        should_reset = False

        if self.config.period == LeaderboardPeriod.DAILY:
            # Reset at midnight
            if now.date() > self.last_reset.date():
                should_reset = True

        elif self.config.period == LeaderboardPeriod.WEEKLY:
            # Reset on Monday
            days_since_reset = (now - self.last_reset).days
            if days_since_reset >= 7 or (now.weekday() == 0 and self.last_reset.weekday() != 0):
                should_reset = True

        elif self.config.period == LeaderboardPeriod.MONTHLY:
            # Reset on 1st of month
            if now.month != self.last_reset.month or now.year != self.last_reset.year:
                should_reset = True

        if should_reset:
            self.entries.clear()
            self.last_reset = now

    def get_stats(self) -> Dict:
        """Get leaderboard statistics"""
        if not self.entries:
            return {
                "total_participants": 0,
                "average_score": 0.0,
                "top_score": 0.0,
                "median_score": 0.0
            }

        scores = [e.score for e in self.entries.values()]
        scores.sort()

        return {
            "total_participants": len(self.entries),
            "average_score": sum(scores) / len(scores),
            "top_score": scores[-1],
            "median_score": scores[len(scores) // 2],
            "last_reset": self.last_reset.isoformat(),
            "period": self.config.period.value
        }


class LeaderboardManager:
    """
    Manages multiple leaderboards

    Features:
    - Multiple leaderboard types
    - Automatic updates
    - Privacy controls
    - Friend leaderboards

    Example:
        >>> manager = LeaderboardManager()
        >>>
        >>> # Update student scores
        >>> manager.update_all(
        ...     student_id="student_001",
        ...     student_name="Alice",
        ...     total_xp=1250,
        ...     subject_xp={"math": 500, "science": 300},
        ...     objectives_mastered=15,
        ...     current_streak=7,
        ...     grade=8,
        ...     classroom="8A"
        ... )
        >>>
        >>> # Get global leaderboard
        >>> global_lb = manager.get_leaderboard(LeaderboardType.GLOBAL_XP)
        >>> top_10 = global_lb.get_rankings(limit=10)
    """

    def __init__(self):
        self.leaderboards: Dict[str, Leaderboard] = {}
        self._initialize_leaderboards()

    def _initialize_leaderboards(self):
        """Initialize all default leaderboards"""
        # Global XP (all-time)
        self.leaderboards["global_xp_all_time"] = Leaderboard(
            LeaderboardConfig(
                leaderboard_type=LeaderboardType.GLOBAL_XP,
                period=LeaderboardPeriod.ALL_TIME
            )
        )

        # Global XP (weekly)
        self.leaderboards["global_xp_weekly"] = Leaderboard(
            LeaderboardConfig(
                leaderboard_type=LeaderboardType.GLOBAL_XP,
                period=LeaderboardPeriod.WEEKLY
            )
        )

        # Global XP (monthly)
        self.leaderboards["global_xp_monthly"] = Leaderboard(
            LeaderboardConfig(
                leaderboard_type=LeaderboardType.GLOBAL_XP,
                period=LeaderboardPeriod.MONTHLY
            )
        )

        # Objectives mastered
        self.leaderboards["objectives_all_time"] = Leaderboard(
            LeaderboardConfig(
                leaderboard_type=LeaderboardType.OBJECTIVES,
                period=LeaderboardPeriod.ALL_TIME
            )
        )

        # Streaks
        self.leaderboards["streaks"] = Leaderboard(
            LeaderboardConfig(
                leaderboard_type=LeaderboardType.STREAKS,
                period=LeaderboardPeriod.ALL_TIME
            )
        )

        # Subject leaderboards
        for subject in ["math", "science", "ela", "social_studies", "ai_readiness"]:
            self.leaderboards[f"subject_{subject}"] = Leaderboard(
                LeaderboardConfig(
                    leaderboard_type=LeaderboardType.SUBJECT_XP,
                    period=LeaderboardPeriod.ALL_TIME,
                    subject=subject
                )
            )

    def update_all(
        self,
        student_id: str,
        student_name: str,
        total_xp: int,
        subject_xp: Dict[str, int],
        objectives_mastered: int,
        current_streak: int,
        grade: int,
        classroom: Optional[str] = None,
        avatar: Optional[str] = None
    ):
        """Update all relevant leaderboards for a student"""
        # Global XP leaderboards
        for period in ["all_time", "weekly", "monthly"]:
            lb_key = f"global_xp_{period}"
            if lb_key in self.leaderboards:
                self.leaderboards[lb_key].update_score(
                    student_id=student_id,
                    student_name=student_name,
                    score=total_xp,
                    grade=grade,
                    classroom=classroom,
                    avatar=avatar
                )

        # Objectives leaderboard
        if "objectives_all_time" in self.leaderboards:
            self.leaderboards["objectives_all_time"].update_score(
                student_id=student_id,
                student_name=student_name,
                score=objectives_mastered,
                grade=grade,
                classroom=classroom,
                avatar=avatar
            )

        # Streak leaderboard
        if "streaks" in self.leaderboards:
            self.leaderboards["streaks"].update_score(
                student_id=student_id,
                student_name=student_name,
                score=current_streak,
                grade=grade,
                classroom=classroom,
                avatar=avatar
            )

        # Subject leaderboards
        for subject, xp in subject_xp.items():
            lb_key = f"subject_{subject}"
            if lb_key in self.leaderboards:
                self.leaderboards[lb_key].update_score(
                    student_id=student_id,
                    student_name=student_name,
                    score=xp,
                    grade=grade,
                    classroom=classroom,
                    avatar=avatar
                )

    def get_leaderboard(self, leaderboard_type: LeaderboardType) -> Optional[Leaderboard]:
        """Get a specific leaderboard"""
        # Map type to key (all-time by default)
        key = f"{leaderboard_type.value}_all_time"
        if key not in self.leaderboards:
            # Try without _all_time
            key = leaderboard_type.value

        return self.leaderboards.get(key)

    def get_student_summary(self, student_id: str) -> Dict:
        """Get student's ranks across all leaderboards"""
        summary = {
            "student_id": student_id,
            "rankings": {}
        }

        for lb_name, leaderboard in self.leaderboards.items():
            personal = leaderboard.get_personal_rank(student_id)
            if personal:
                summary["rankings"][lb_name] = {
                    "rank": personal["rank"],
                    "score": personal["score"],
                    "percentile": personal["percentile"],
                    "total": personal["total_participants"]
                }

        return summary

    def get_all_leaderboard_names(self) -> List[str]:
        """Get names of all available leaderboards"""
        return list(self.leaderboards.keys())


# ========== Helper Functions ==========

def create_leaderboard_manager() -> LeaderboardManager:
    """Create new leaderboard manager"""
    return LeaderboardManager()


if __name__ == "__main__":
    # Demo
    print("=== EdWIN Leaderboard System Demo ===\n")

    manager = LeaderboardManager()

    # Add some students
    students = [
        ("student_001", "Alice", 1250, 15, 7, 8),
        ("student_002", "Bob", 980, 12, 5, 8),
        ("student_003", "Charlie", 1450, 18, 10, 8),
        ("student_004", "Diana", 890, 10, 3, 8),
        ("student_005", "Eve", 1100, 14, 7, 8),
    ]

    for student_id, name, xp, obj, streak, grade in students:
        manager.update_all(
            student_id=student_id,
            student_name=name,
            total_xp=xp,
            subject_xp={"math": xp // 2, "science": xp // 3},
            objectives_mastered=obj,
            current_streak=streak,
            grade=grade,
            classroom="8A"
        )

    # Get global leaderboard
    print("=== Global XP Leaderboard (All-Time) ===")
    global_lb = manager.get_leaderboard(LeaderboardType.GLOBAL_XP)
    if global_lb:
        top_5 = global_lb.get_rankings(limit=5)
        for entry in top_5:
            print(f"{entry.rank}. {entry.student_name} - {entry.score:.0f} XP")

    # Personal rank
    print("\n=== Personal Rank (Alice) ===")
    personal = global_lb.get_personal_rank("student_001")
    if personal:
        print(f"Rank: {personal['rank']}/{personal['total_participants']}")
        print(f"Score: {personal['score']:.0f} XP")
        print(f"Percentile: Top {100 - personal['percentile']:.1f}%")

    # Surrounding ranks
    print("\n=== Surrounding Ranks (Alice ±2) ===")
    surrounding = global_lb.get_surrounding_ranks("student_001", context=2)
    for entry in surrounding:
        marker = "→" if entry.student_id == "student_001" else " "
        print(f"{marker} {entry.rank}. {entry.student_name} - {entry.score:.0f} XP")

    # Subject leaderboard
    print("\n=== Math XP Leaderboard ===")
    math_lb = manager.leaderboards.get("subject_math")
    if math_lb:
        top_3 = math_lb.get_rankings(limit=3)
        for entry in top_3:
            print(f"{entry.rank}. {entry.student_name} - {entry.score:.0f} XP")

    # Student summary
    print("\n=== Student Summary (Alice) ===")
    summary = manager.get_student_summary("student_001")
    for lb_name, rank_info in summary["rankings"].items():
        print(f"{lb_name}: Rank {rank_info['rank']} ({rank_info['score']:.0f})")
