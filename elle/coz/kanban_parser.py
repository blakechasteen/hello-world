"""
Kanban Parser for Elle Core

Reads coz/kanban.csv and converts to Elle Core task format.
Provides daily task suggestions based on priorities and due dates.

Created: 2025-11-15
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from enum import Enum


class TaskPriority(Enum):
    """Task priority levels"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class TaskStatus(Enum):
    """Task status"""
    PLANNED = "Planned"
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    BLOCKED = "Blocked"


@dataclass
class KanbanTask:
    """Task from Kanban board"""
    task: str
    category: str
    priority: TaskPriority
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PLANNED
    notes: str = ""

    def is_due_today(self, today: Optional[datetime] = None) -> bool:
        """Check if task is due today"""
        if self.due_date is None:
            return False
        if today is None:
            today = datetime.now()
        return self.due_date.date() == today.date()

    def is_overdue(self, today: Optional[datetime] = None) -> bool:
        """Check if task is overdue"""
        if self.due_date is None:
            return False
        if today is None:
            today = datetime.now()
        return self.due_date.date() < today.date() and self.status != TaskStatus.COMPLETED

    def days_until_due(self, today: Optional[datetime] = None) -> Optional[int]:
        """Days until task is due"""
        if self.due_date is None:
            return None
        if today is None:
            today = datetime.now()
        delta = (self.due_date.date() - today.date()).days
        return delta

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "task": self.task,
            "category": self.category,
            "priority": self.priority.name,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "status": self.status.value,
            "notes": self.notes,
        }


class KanbanParser:
    """Parser for Kanban CSV file"""

    def __init__(self, kanban_path: str = "coz/kanban.csv"):
        self.kanban_path = Path(kanban_path)
        self.tasks: List[KanbanTask] = []
        self.categories: Dict[str, List[KanbanTask]] = {}

    def parse(self) -> List[KanbanTask]:
        """Parse kanban.csv file"""
        if not self.kanban_path.exists():
            raise FileNotFoundError(f"Kanban file not found: {self.kanban_path}")

        self.tasks = []
        self.categories = {}

        with open(self.kanban_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows
                if not row.get('Task'):
                    continue

                task = self._parse_row(row)
                self.tasks.append(task)

                # Organize by category
                if task.category not in self.categories:
                    self.categories[task.category] = []
                self.categories[task.category].append(task)

        return self.tasks

    def _parse_row(self, row: Dict) -> KanbanTask:
        """Parse a single kanban row"""
        task_name = row.get('Task', '').strip()
        category = row.get('Category', '').strip()
        priority_str = row.get('Priority', 'Medium').strip()
        start_date_str = row.get('Start Date', '').strip()
        due_date_str = row.get('Due Date', '').strip()
        status_str = row.get('Status', 'Planned').strip()
        notes = row.get('Notes', '').strip()

        # Parse priority
        priority_map = {'High': TaskPriority.HIGH, 'Medium': TaskPriority.MEDIUM, 'Low': TaskPriority.LOW}
        priority = priority_map.get(priority_str, TaskPriority.MEDIUM)

        # Parse dates
        start_date = self._parse_date(start_date_str)
        due_date = self._parse_date(due_date_str)

        # Parse status
        status_map = {
            'Planned': TaskStatus.PLANNED,
            'Pending': TaskStatus.PENDING,
            'In Progress': TaskStatus.IN_PROGRESS,
            'Completed': TaskStatus.COMPLETED,
            'Blocked': TaskStatus.BLOCKED,
        }
        status = status_map.get(status_str, TaskStatus.PLANNED)

        return KanbanTask(
            task=task_name,
            category=category,
            priority=priority,
            start_date=start_date,
            due_date=due_date,
            status=status,
            notes=notes,
        )

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string"""
        if not date_str:
            return None

        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def get_daily_tasks(self, today: Optional[datetime] = None) -> List[KanbanTask]:
        """Get tasks due today, sorted by priority"""
        if today is None:
            today = datetime.now()

        daily_tasks = [
            t for t in self.tasks
            if t.is_due_today(today) and t.status != TaskStatus.COMPLETED
        ]

        # Sort by priority (HIGH first)
        daily_tasks.sort(key=lambda t: t.priority.value)
        return daily_tasks

    def get_overdue_tasks(self, today: Optional[datetime] = None) -> List[KanbanTask]:
        """Get overdue tasks"""
        if today is None:
            today = datetime.now()

        overdue = [t for t in self.tasks if t.is_overdue(today)]
        overdue.sort(key=lambda t: t.due_date)
        return overdue

    def get_upcoming_tasks(self, days_ahead: int = 7, today: Optional[datetime] = None) -> List[KanbanTask]:
        """Get tasks due in the next N days"""
        if today is None:
            today = datetime.now()

        upcoming = []
        for task in self.tasks:
            if task.status == TaskStatus.COMPLETED:
                continue

            days = task.days_until_due(today)
            if days is not None and 0 <= days <= days_ahead:
                upcoming.append(task)

        # Sort by due date
        upcoming.sort(key=lambda t: t.due_date)
        return upcoming

    def get_by_category(self, category: str) -> List[KanbanTask]:
        """Get all tasks in a category"""
        return self.categories.get(category, [])

    def get_by_priority(self, priority: TaskPriority) -> List[KanbanTask]:
        """Get all tasks with given priority"""
        return [t for t in self.tasks if t.priority == priority]

    def get_active_tasks(self) -> List[KanbanTask]:
        """Get in-progress tasks"""
        return [t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS]

    def suggest_daily_plan(self, today: Optional[datetime] = None) -> Dict:
        """Suggest daily plan based on priorities and due dates"""
        if today is None:
            today = datetime.now()

        daily_tasks = self.get_daily_tasks(today)
        active_tasks = self.get_active_tasks()
        overdue = self.get_overdue_tasks(today)
        upcoming = self.get_upcoming_tasks(3, today)

        return {
            "date": today.date().isoformat(),
            "due_today": daily_tasks,
            "in_progress": active_tasks,
            "overdue": overdue,
            "upcoming_3_days": upcoming,
            "suggested_order": daily_tasks + active_tasks + upcoming[:3],
        }

    def get_summary(self) -> Dict:
        """Get kanban summary statistics"""
        total = len(self.tasks)
        completed = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        in_progress = len([t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS])
        high_priority = len([t for t in self.tasks if t.priority == TaskPriority.HIGH])

        return {
            "total_tasks": total,
            "completed": completed,
            "completion_rate": completed / total if total > 0 else 0,
            "in_progress": in_progress,
            "high_priority": high_priority,
            "categories": list(self.categories.keys()),
            "by_category": {
                cat: len(tasks) for cat, tasks in self.categories.items()
            },
        }
