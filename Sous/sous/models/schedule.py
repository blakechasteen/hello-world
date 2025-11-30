"""Schedule models"""

from dataclasses import dataclass, field
from datetime import date, time, datetime, timedelta
from typing import List, Optional
from enum import Enum

from .recipe import SkillLevel


@dataclass
class TimeSlot:
    """
    Precise time window for task scheduling

    Attributes:
        start_time: When task begins (HH:MM format)
        end_time: When task ends (HH:MM format)
        is_flexible: Whether time can be adjusted
    """
    start_time: time
    end_time: time
    is_flexible: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "TimeSlot":
        """Create TimeSlot from JSON dict"""
        return cls(
            start_time=time.fromisoformat(data["start_time"]),
            end_time=time.fromisoformat(data["end_time"]),
            is_flexible=data.get("is_flexible", True)
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "is_flexible": self.is_flexible
        }

    def duration_minutes(self) -> int:
        """Calculate duration in minutes (handles midnight crossing)"""
        start_dt = datetime.combine(date.today(), self.start_time)
        end_dt = datetime.combine(date.today(), self.end_time)

        # Handle midnight crossing (e.g., 23:00 to 01:00)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)

        return int((end_dt - start_dt).total_seconds() / 60)

    def overlaps(self, other: "TimeSlot") -> bool:
        """Check if this time slot overlaps with another (handles midnight crossing)"""
        # Convert to datetime for proper comparison
        today = date.today()
        self_start = datetime.combine(today, self.start_time)
        self_end = datetime.combine(today, self.end_time)
        other_start = datetime.combine(today, other.start_time)
        other_end = datetime.combine(today, other.end_time)

        # Handle midnight crossing
        if self_end < self_start:
            self_end += timedelta(days=1)
        if other_end < other_start:
            other_end += timedelta(days=1)

        # Standard overlap check: NOT (A ends before B starts OR A starts after B ends)
        return not (self_end <= other_start or self_start >= other_end)

    def __repr__(self) -> str:
        return f"{self.start_time.strftime('%H:%M')}-{self.end_time.strftime('%H:%M')}"


@dataclass
class Task:
    """
    Scheduled task for a specific recipe step

    Attributes:
        id: Unique task identifier
        recipe_id: Which recipe this belongs to
        step_id: Which step in the recipe
        start_time: Approximate start time (optional for MVP)
        duration_minutes: How long this will take
        appliance_id: Which appliance to use (if any)
        is_optional: Whether task is optional (bonus task)
        status: Current status (planned, completed, skipped)
        time_slot: Precise time window (Phase 2)
        skill_level: Required skill level (Phase 5)
        scaled_duration: Duration after scaling (Phase 3)
    """

    id: str
    recipe_id: str
    step_id: str
    duration_minutes: int
    start_time: Optional[str] = None
    appliance_id: Optional[str] = None
    is_optional: bool = False
    status: str = "planned"  # planned, completed, skipped
    time_slot: Optional[TimeSlot] = None
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    scaled_duration: Optional[int] = None  # Scaled duration for batch cooking

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create Task from JSON dict"""
        time_slot = None
        if "time_slot" in data:
            time_slot = TimeSlot.from_dict(data["time_slot"])

        skill_level = SkillLevel.INTERMEDIATE
        if "skill_level" in data:
            skill_level = SkillLevel(data["skill_level"])

        return cls(
            id=data["id"],
            recipe_id=data["recipe_id"],
            step_id=data["step_id"],
            duration_minutes=data["duration_minutes"],
            start_time=data.get("start_time"),
            appliance_id=data.get("appliance_id"),
            is_optional=data.get("is_optional", False),
            status=data.get("status", "planned"),
            time_slot=time_slot,
            skill_level=skill_level,
            scaled_duration=data.get("scaled_duration"),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        result = {
            "id": self.id,
            "recipe_id": self.recipe_id,
            "step_id": self.step_id,
            "duration_minutes": self.duration_minutes,
            "status": self.status,
        }
        if self.start_time:
            result["start_time"] = self.start_time
        if self.appliance_id:
            result["appliance_id"] = self.appliance_id
        if self.is_optional:
            result["is_optional"] = self.is_optional
        if self.time_slot:
            result["time_slot"] = self.time_slot.to_dict()
        if self.skill_level != SkillLevel.INTERMEDIATE:
            result["skill_level"] = self.skill_level.value
        if self.scaled_duration:
            result["scaled_duration"] = self.scaled_duration
        return result

    def get_effective_duration(self) -> int:
        """Get duration considering scaling"""
        return self.scaled_duration if self.scaled_duration else self.duration_minutes

    def set_time_window(self, start: time, duration_minutes: Optional[int] = None) -> None:
        """Set precise time window for this task"""
        duration = duration_minutes or self.get_effective_duration()
        start_dt = datetime.combine(date.today(), start)
        end_dt = start_dt + timedelta(minutes=duration)
        self.time_slot = TimeSlot(start_time=start, end_time=end_dt.time())
        self.start_time = start.isoformat()

    def conflicts_with(self, other: "Task") -> bool:
        """Check if this task conflicts with another (same appliance, overlapping time)"""
        if self.appliance_id != other.appliance_id:
            return False
        if not self.time_slot or not other.time_slot:
            return False  # Can't determine without time slots
        return self.time_slot.overlaps(other.time_slot)

    def mark_completed(self):
        """Mark task as completed"""
        self.status = "completed"

    def mark_skipped(self):
        """Mark task as skipped"""
        self.status = "skipped"

    def is_completed(self) -> bool:
        """Check if task is completed"""
        return self.status == "completed"

    def is_skipped(self) -> bool:
        """Check if task is skipped"""
        return self.status == "skipped"

    def is_planned(self) -> bool:
        """Check if task is still planned"""
        return self.status == "planned"


@dataclass
class DailyPlan:
    """
    Collection of tasks for a single day

    Attributes:
        date: Date for this plan
        tasks: Ordered list of tasks for the day
    """

    date: date
    tasks: List[Task] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "DailyPlan":
        """Create DailyPlan from JSON dict"""
        plan_date = date.fromisoformat(data["date"])
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]

        return cls(date=plan_date, tasks=tasks)

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "date": self.date.isoformat(),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def add_task(self, task: Task):
        """Add task to this day's plan"""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def total_duration_minutes(self) -> int:
        """Calculate total duration for all tasks"""
        return sum(task.duration_minutes for task in self.tasks)

    def optional_tasks(self) -> List[Task]:
        """Get list of optional tasks"""
        return [t for t in self.tasks if t.is_optional]

    def required_tasks(self) -> List[Task]:
        """Get list of required tasks"""
        return [t for t in self.tasks if not t.is_optional]

    def tasks_by_appliance(self, appliance_id: str) -> List[Task]:
        """Get tasks using given appliance"""
        return [t for t in self.tasks if t.appliance_id == appliance_id]

    def appliances_used(self) -> List[str]:
        """Get list of appliance IDs used this day"""
        appliances = set()
        for task in self.tasks:
            if task.appliance_id:
                appliances.add(task.appliance_id)
        return sorted(list(appliances))

    def has_appliance_conflict(self) -> bool:
        """
        Check if multiple tasks use same appliance at same time

        For MVP: Just checks if >1 task uses same appliance
        Phase 2: Check actual time windows
        """
        appliances_used = self.appliances_used()
        for appliance_id in appliances_used:
            tasks = self.tasks_by_appliance(appliance_id)
            if len(tasks) > 1:
                # Check if any have time slots that overlap
                for i, task1 in enumerate(tasks):
                    for task2 in tasks[i+1:]:
                        if task1.conflicts_with(task2):
                            return True
                # If no time slots, just check count
                if not any(t.time_slot for t in tasks):
                    return True
        return False

    def assign_time_windows(self, start_time: time = time(8, 0), buffer_minutes: int = 15) -> None:
        """
        Assign precise time windows to all tasks (Phase 2)

        Args:
            start_time: When to start the day's tasks (default 8:00 AM)
            buffer_minutes: Buffer between tasks (default 15 min)
        """
        current_time = datetime.combine(date.today(), start_time)

        # Group tasks by appliance to schedule serially
        appliance_schedule: dict = {}  # appliance_id -> next available time

        for task in self.tasks:
            appliance = task.appliance_id or "general"

            # Get next available time for this appliance
            if appliance in appliance_schedule:
                task_start = max(current_time, appliance_schedule[appliance])
            else:
                task_start = current_time

            # Assign time window
            task.set_time_window(task_start.time())

            # Update appliance schedule
            task_end = task_start + timedelta(minutes=task.get_effective_duration() + buffer_minutes)
            appliance_schedule[appliance] = task_end

            # Also advance general time (no parallel tasks for simplicity)
            current_time = max(current_time, task_end)

    def get_time_conflicts(self) -> List[tuple]:
        """
        Get list of conflicting task pairs (Phase 2)

        Returns:
            List of (task1, task2) tuples that conflict
        """
        conflicts = []
        for i, task1 in enumerate(self.tasks):
            for task2 in self.tasks[i+1:]:
                if task1.conflicts_with(task2):
                    conflicts.append((task1, task2))
        return conflicts

    def get_timeline(self) -> List[dict]:
        """
        Get timeline view of the day's tasks (Phase 2)

        Returns:
            List of {task, start_time, end_time} dicts sorted by start time
        """
        timeline = []
        for task in self.tasks:
            if task.time_slot:
                timeline.append({
                    "task": task,
                    "start_time": task.time_slot.start_time,
                    "end_time": task.time_slot.end_time,
                    "appliance": task.appliance_id,
                    "duration": task.get_effective_duration()
                })
        return sorted(timeline, key=lambda x: x["start_time"])

    def __repr__(self) -> str:
        return f"DailyPlan({self.date}, {len(self.tasks)} tasks, {self.total_duration_minutes()} min)"
