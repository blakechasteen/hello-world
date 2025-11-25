"""Schedule models"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


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
    """

    id: str
    recipe_id: str
    step_id: str
    duration_minutes: int
    start_time: Optional[str] = None
    appliance_id: Optional[str] = None
    is_optional: bool = False
    status: str = "planned"  # planned, completed, skipped

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create Task from JSON dict"""
        return cls(
            id=data["id"],
            recipe_id=data["recipe_id"],
            step_id=data["step_id"],
            duration_minutes=data["duration_minutes"],
            start_time=data.get("start_time"),
            appliance_id=data.get("appliance_id"),
            is_optional=data.get("is_optional", False),
            status=data.get("status", "planned"),
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
        return result

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
        Future: Check actual time windows
        """
        appliances_used = self.appliances_used()
        for appliance_id in appliances_used:
            if len(self.tasks_by_appliance(appliance_id)) > 1:
                return True
        return False

    def __repr__(self) -> str:
        return f"DailyPlan({self.date}, {len(self.tasks)} tasks, {self.total_duration_minutes()} min)"
