"""
EdWIN Tutor - Progress Tracking System

Tracks learner progress across all interfaces (terminal, web, notebooks).
Saves to a local JSON file that all UIs can read/write.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Any
from pathlib import Path
from datetime import datetime
import json


@dataclass
class Badge:
    """Achievement badge."""
    id: str
    name: str
    description: str
    icon: str  # Emoji or unicode
    earned_at: Optional[datetime] = None


@dataclass
class LearnerProgress:
    """Tracks a learner's journey through EdWIN."""
    # Identity
    learner_name: str = "Learner"

    # Progress
    completed_lessons: Set[str] = field(default_factory=set)
    current_lesson: Optional[str] = None
    lesson_scores: Dict[str, int] = field(default_factory=dict)  # lesson_id -> score

    # Gamification
    total_xp: int = 0
    level: int = 1
    badges: List[str] = field(default_factory=list)  # Badge IDs

    # Statistics
    total_time_minutes: int = 0
    lessons_started: int = 0
    challenges_completed: int = 0
    hints_used: int = 0

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)


class ProgressTracker:
    """Manages learner progress across all EdWIN interfaces."""

    def __init__(self, save_path: Path = Path(".edwin_progress.json")):
        self.save_path = save_path
        self.progress = self.load_progress()

        # XP thresholds for leveling up
        self.level_thresholds = [
            0, 100, 250, 500, 1000, 2000, 3500,
            5000, 7500, 10000  # Level 10 = master
        ]

        # Available badges
        self.available_badges = self._define_badges()

    def _define_badges(self) -> Dict[str, Badge]:
        """Define all available achievement badges."""
        return {
            "first_lesson": Badge(
                "first_lesson", "🎓 First Steps",
                "Completed your first lesson", "🎓"
            ),
            "fast_learner": Badge(
                "fast_learner", "⚡ Fast Learner",
                "Completed 5 lessons in one session", "⚡"
            ),
            "problem_solver": Badge(
                "problem_solver", "🧩 Problem Solver",
                "Completed 10 challenges", "🧩"
            ),
            "no_hints": Badge(
                "no_hints", "🎯 Sharp Mind",
                "Completed 3 challenges without hints", "🎯"
            ),
            "persistent": Badge(
                "persistent", "💪 Persistent",
                "Attempted a challenge 5+ times", "💪"
            ),
            "knowledge_seeker": Badge(
                "knowledge_seeker", "📚 Knowledge Seeker",
                "Completed all beginner lessons", "📚"
            ),
            "intermediate_master": Badge(
                "intermediate_master", "🔧 Intermediate Master",
                "Completed all intermediate lessons", "🔧"
            ),
            "advanced_scholar": Badge(
                "advanced_scholar", "🔬 Advanced Scholar",
                "Completed all advanced lessons", "🔬"
            ),
            "hololoom_master": Badge(
                "hololoom_master", "🏆 HoloLoom Master",
                "Completed all lessons and challenges", "🏆"
            ),
        }

    def load_progress(self) -> LearnerProgress:
        """Load progress from disk or create new."""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)

                # Convert sets and datetimes
                data['completed_lessons'] = set(data.get('completed_lessons', []))
                if 'started_at' in data:
                    data['started_at'] = datetime.fromisoformat(data['started_at'])
                if 'last_active' in data:
                    data['last_active'] = datetime.fromisoformat(data['last_active'])

                return LearnerProgress(**data)
            except Exception as e:
                print(f"⚠️  Failed to load progress: {e}")
                return LearnerProgress()
        else:
            return LearnerProgress()

    def save_progress(self):
        """Save progress to disk."""
        try:
            data = asdict(self.progress)

            # Convert sets to lists for JSON
            data['completed_lessons'] = list(data['completed_lessons'])

            # Convert datetimes to ISO format
            data['started_at'] = data['started_at'].isoformat()
            data['last_active'] = data['last_active'].isoformat()

            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save progress: {e}")

    def mark_lesson_complete(self, lesson_id: str, xp_earned: int, score: int = 100):
        """Mark a lesson as complete and award XP."""
        if lesson_id not in self.progress.completed_lessons:
            self.progress.completed_lessons.add(lesson_id)
            self.progress.total_xp += xp_earned
            self.progress.lesson_scores[lesson_id] = score

            # Check for level up
            old_level = self.progress.level
            self.progress.level = self._calculate_level(self.progress.total_xp)

            # Update stats
            self.progress.last_active = datetime.now()

            # Check for badges
            self._check_badges()

            self.save_progress()

            return {
                "leveled_up": self.progress.level > old_level,
                "new_level": self.progress.level,
                "total_xp": self.progress.total_xp
            }

    def mark_challenge_complete(self, challenge_id: str, points: int):
        """Mark a challenge as complete."""
        self.progress.challenges_completed += 1
        self.progress.total_xp += points
        self.progress.last_active = datetime.now()

        # Check for badges
        self._check_badges()

        self.save_progress()

    def use_hint(self):
        """Record that a hint was used."""
        self.progress.hints_used += 1
        self.save_progress()

    def start_lesson(self, lesson_id: str):
        """Record starting a new lesson."""
        self.progress.current_lesson = lesson_id
        self.progress.lessons_started += 1
        self.progress.last_active = datetime.now()
        self.save_progress()

    def _calculate_level(self, xp: int) -> int:
        """Calculate level based on XP."""
        for level, threshold in enumerate(self.level_thresholds, start=1):
            if xp < threshold:
                return level - 1
        return len(self.level_thresholds)  # Max level

    def _check_badges(self):
        """Check if any new badges should be awarded."""
        new_badges = []

        # First lesson badge
        if len(self.progress.completed_lessons) >= 1 and \
           "first_lesson" not in self.progress.badges:
            self.progress.badges.append("first_lesson")
            new_badges.append(self.available_badges["first_lesson"])

        # Problem solver badge
        if self.progress.challenges_completed >= 10 and \
           "problem_solver" not in self.progress.badges:
            self.progress.badges.append("problem_solver")
            new_badges.append(self.available_badges["problem_solver"])

        # Add more badge checks here...

        return new_badges

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive progress statistics."""
        return {
            "level": self.progress.level,
            "xp": self.progress.total_xp,
            "xp_to_next_level": self._xp_to_next_level(),
            "lessons_completed": len(self.progress.completed_lessons),
            "challenges_completed": self.progress.challenges_completed,
            "badges_earned": len(self.progress.badges),
            "hints_used": self.progress.hints_used,
            "time_spent": self.progress.total_time_minutes,
            "completion_rate": self._calculate_completion_rate()
        }

    def _xp_to_next_level(self) -> int:
        """Calculate XP needed for next level."""
        if self.progress.level >= len(self.level_thresholds):
            return 0  # Max level
        next_threshold = self.level_thresholds[self.progress.level]
        return next_threshold - self.progress.total_xp

    def _calculate_completion_rate(self) -> float:
        """Calculate overall completion percentage."""
        # This would need total lesson count from LessonManager
        # For now, estimate based on 30 total lessons
        return (len(self.progress.completed_lessons) / 30) * 100
