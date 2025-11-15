"""
Elle Core - Task Models

Defines tasks and projects Elle helps track.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority level for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    """
    A single task - something to be done.

    Tasks are created from user intents, reminders, or project breakdowns.
    """
    id: str
    description: str  # What needs to be done
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    location: Optional[str] = None  # Where to do this (e.g., "shed", "garden")
    required_objects: List[str] = field(default_factory=list)  # Object IDs needed
    estimated_duration: Optional[timedelta] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_overdue(self) -> bool:
        """Check if task is past due date."""
        if self.due_date and self.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return datetime.now() > self.due_date
        return False

    def mark_completed(self):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()

    def __str__(self) -> str:
        status_icon = {
            TaskStatus.PENDING: "○",
            TaskStatus.IN_PROGRESS: "◐",
            TaskStatus.COMPLETED: "●",
            TaskStatus.DEFERRED: "⊙",
            TaskStatus.CANCELLED: "✕"
        }
        icon = status_icon.get(self.status, "?")
        return f"{icon} {self.description}"


@dataclass
class Project:
    """
    A collection of related tasks forming a larger goal.

    Projects help Elle understand context and suggest next steps.
    """
    id: str
    name: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    tasks: List[str] = field(default_factory=list)  # Task IDs
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        task_count = len(self.tasks)
        return f"{self.name} ({task_count} tasks)"
