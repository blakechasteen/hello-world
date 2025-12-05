#!/usr/bin/env python3
"""
Task Models for Multi-User Kitchen Coordination

Supports:
- Task dependency graphs (DAG) for complex recipes
- Resource scheduling (oven, stovetop conflicts)
- Skill-based task assignment
- Priority and urgency management
- Parallel task execution when possible
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
from sous.models.user import ResourceType


class TaskState(Enum):
    """Task execution states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"  # Waiting for dependencies
    FAILED = "failed"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """
    Individual cooking task with dependencies and resource requirements.

    Supports:
    - Task dependencies (DAG structure)
    - Resource booking (oven, stovetop, counter)
    - Skill requirements
    - Time estimation
    - Progress tracking
    """
    task_id: str
    household_id: str
    recipe_id: str
    description: str

    # Assignment
    assigned_to: Optional[str] = None  # user_id
    state: TaskState = TaskState.PENDING

    # Dependencies (for DAG)
    dependencies: List[str] = field(default_factory=list)  # task_ids that must complete first
    blocking: List[str] = field(default_factory=list)  # task_ids that depend on this

    # Resources
    resources_needed: List[ResourceType] = field(default_factory=list)
    resource_bookings: Dict[ResourceType, Dict[str, datetime]] = field(default_factory=dict)

    # Skills
    required_skill: Optional[str] = None  # Cooking technique required
    min_skill_level: float = 1.0  # Minimum proficiency (1.0-4.0)

    # Timing
    estimated_duration_min: int = 10
    actual_duration_min: Optional[int] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None

    # Priority
    priority: TaskPriority = TaskPriority.MEDIUM
    deadline: Optional[datetime] = None

    # Progress
    progress_percent: int = 0
    notes: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def can_start(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)

    def is_blocked(self, completed_tasks: Set[str]) -> bool:
        """Check if task is blocked by incomplete dependencies."""
        return not self.can_start(completed_tasks)

    def assign(self, user_id: str):
        """Assign task to a user."""
        self.assigned_to = user_id
        self.state = TaskState.PENDING
        self.updated_at = datetime.now()

    def start(self):
        """Start task execution."""
        if self.state != TaskState.PENDING:
            return False

        self.state = TaskState.IN_PROGRESS
        self.actual_start = datetime.now()
        self.updated_at = datetime.now()
        return True

    def complete(self, notes: str = ""):
        """Mark task as completed."""
        if self.state != TaskState.IN_PROGRESS:
            return False

        self.state = TaskState.COMPLETED
        self.actual_end = datetime.now()
        self.progress_percent = 100
        self.updated_at = datetime.now()

        if self.actual_start and self.actual_end:
            delta = self.actual_end - self.actual_start
            self.actual_duration_min = int(delta.total_seconds() / 60)

        if notes:
            self.notes = notes

        return True

    def fail(self, reason: str = ""):
        """Mark task as failed."""
        self.state = TaskState.FAILED
        self.updated_at = datetime.now()
        if reason:
            self.notes = f"Failed: {reason}"

    def update_progress(self, percent: int):
        """Update task progress."""
        self.progress_percent = max(0, min(100, percent))
        self.updated_at = datetime.now()

    def schedule(self, start: datetime):
        """Schedule task for specific time."""
        self.scheduled_start = start
        self.scheduled_end = start + timedelta(minutes=self.estimated_duration_min)
        self.updated_at = datetime.now()

    def add_dependency(self, task_id: str):
        """Add a task that must complete before this one."""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
            self.updated_at = datetime.now()

    def remove_dependency(self, task_id: str):
        """Remove a dependency."""
        if task_id in self.dependencies:
            self.dependencies.remove(task_id)
            self.updated_at = datetime.now()

    def book_resource(self, resource: ResourceType, start: datetime, end: datetime):
        """Book a kitchen resource for this task."""
        self.resource_bookings[resource] = {"start": start, "end": end}

    def release_resources(self):
        """Release all booked resources."""
        self.resource_bookings.clear()

    def get_urgency_score(self) -> float:
        """
        Calculate urgency score (0.0-10.0) for task prioritization.

        Factors:
        - Priority level (1-4)
        - Deadline proximity
        - Number of blocking tasks
        - Estimated duration
        """
        score = self.priority.value * 2.0  # 2-8 base

        # Deadline urgency
        if self.deadline:
            time_until = (self.deadline - datetime.now()).total_seconds() / 3600  # hours
            if time_until < 1:
                score += 2.0  # Very urgent
            elif time_until < 4:
                score += 1.0  # Moderately urgent

        # Blocking other tasks
        score += min(2.0, len(self.blocking) * 0.5)  # Up to +2.0

        # Short tasks get slight boost (do them quick)
        if self.estimated_duration_min < 5:
            score += 0.5

        return min(10.0, score)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "household_id": self.household_id,
            "recipe_id": self.recipe_id,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "state": self.state.value,
            "dependencies": self.dependencies,
            "blocking": self.blocking,
            "resources_needed": [r.value for r in self.resources_needed],
            "required_skill": self.required_skill,
            "min_skill_level": self.min_skill_level,
            "estimated_duration_min": self.estimated_duration_min,
            "actual_duration_min": self.actual_duration_min,
            "scheduled_start": self.scheduled_start.isoformat() if self.scheduled_start else None,
            "scheduled_end": self.scheduled_end.isoformat() if self.scheduled_end else None,
            "actual_start": self.actual_start.isoformat() if self.actual_start else None,
            "actual_end": self.actual_end.isoformat() if self.actual_end else None,
            "priority": self.priority.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "progress_percent": self.progress_percent,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """Deserialize from dictionary."""
        task = cls(
            task_id=data["task_id"],
            household_id=data["household_id"],
            recipe_id=data["recipe_id"],
            description=data["description"],
            assigned_to=data.get("assigned_to"),
            state=TaskState(data.get("state", "pending")),
            dependencies=data.get("dependencies", []),
            blocking=data.get("blocking", []),
            resources_needed=[ResourceType(r) for r in data.get("resources_needed", [])],
            required_skill=data.get("required_skill"),
            min_skill_level=data.get("min_skill_level", 1.0),
            estimated_duration_min=data.get("estimated_duration_min", 10),
            actual_duration_min=data.get("actual_duration_min"),
            priority=TaskPriority(data.get("priority", 2)),
            progress_percent=data.get("progress_percent", 0),
            notes=data.get("notes", "")
        )

        # Parse timestamps
        if "scheduled_start" in data and data["scheduled_start"]:
            task.scheduled_start = datetime.fromisoformat(data["scheduled_start"])
        if "scheduled_end" in data and data["scheduled_end"]:
            task.scheduled_end = datetime.fromisoformat(data["scheduled_end"])
        if "actual_start" in data and data["actual_start"]:
            task.actual_start = datetime.fromisoformat(data["actual_start"])
        if "actual_end" in data and data["actual_end"]:
            task.actual_end = datetime.fromisoformat(data["actual_end"])
        if "deadline" in data and data["deadline"]:
            task.deadline = datetime.fromisoformat(data["deadline"])
        if "created_at" in data:
            task.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            task.updated_at = datetime.fromisoformat(data["updated_at"])

        return task


@dataclass
class TaskDependencyGraph:
    """
    Directed Acyclic Graph (DAG) for task dependencies.

    Manages:
    - Dependency validation (no cycles)
    - Topological sorting (execution order)
    - Parallel task identification
    - Critical path analysis
    """
    tasks: Dict[str, Task] = field(default_factory=dict)

    def add_task(self, task: Task):
        """Add a task to the graph."""
        self.tasks[task.task_id] = task

    def add_dependency(self, dependent_id: str, dependency_id: str):
        """Add dependency: dependent_task depends on dependency_task."""
        if dependent_id not in self.tasks or dependency_id not in self.tasks:
            return False

        # Check for cycles
        if self._would_create_cycle(dependent_id, dependency_id):
            return False

        self.tasks[dependent_id].add_dependency(dependency_id)
        self.tasks[dependency_id].blocking.append(dependent_id)
        return True

    def _would_create_cycle(self, start: str, end: str) -> bool:
        """Check if adding edge start→end would create a cycle."""
        # DFS from 'end' - if we reach 'start', there's a cycle
        visited = set()
        stack = [end]

        while stack:
            node_id = stack.pop()
            if node_id == start:
                return True

            if node_id in visited:
                continue
            visited.add(node_id)

            # Follow dependencies
            if node_id in self.tasks:
                for dep in self.tasks[node_id].dependencies:
                    stack.append(dep)

        return False

    def get_ready_tasks(self, completed: Set[str]) -> List[Task]:
        """Get all tasks that can start now (dependencies satisfied)."""
        ready = []
        for task in self.tasks.values():
            if task.state == TaskState.PENDING and task.can_start(completed):
                ready.append(task)
        return sorted(ready, key=lambda t: t.get_urgency_score(), reverse=True)

    def get_parallel_tasks(self, completed: Set[str]) -> List[List[Task]]:
        """
        Get tasks grouped by parallelization level.

        Returns list of task groups that can execute in parallel.
        Each group must complete before the next group starts.
        """
        levels = []
        remaining = {tid: t for tid, t in self.tasks.items() if t.state != TaskState.COMPLETED}
        completed_set = completed.copy()

        while remaining:
            # Find tasks with no unsatisfied dependencies
            current_level = [
                t for t in remaining.values()
                if all(dep in completed_set for dep in t.dependencies)
            ]

            if not current_level:
                break  # No more tasks can be scheduled (shouldn't happen in valid DAG)

            levels.append(current_level)

            # Mark as completed for next iteration
            for task in current_level:
                completed_set.add(task.task_id)
                del remaining[task.task_id]

        return levels

    def get_topological_order(self) -> List[Task]:
        """Get tasks in topological order (valid execution sequence)."""
        visited = set()
        order = []

        def dfs(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)

            # Visit dependencies first
            if task_id in self.tasks:
                for dep in self.tasks[task_id].dependencies:
                    dfs(dep)

            order.append(self.tasks[task_id])

        # Visit all tasks
        for task_id in self.tasks:
            dfs(task_id)

        return order

    def get_critical_path(self) -> List[Task]:
        """
        Find critical path (longest path through DAG).

        This is the sequence of tasks that determines minimum completion time.
        """
        # Calculate earliest start times
        earliest_start = {}
        topo_order = self.get_topological_order()

        for task in topo_order:
            max_dep_end = 0
            for dep_id in task.dependencies:
                dep_task = self.tasks[dep_id]
                dep_end = earliest_start[dep_id] + dep_task.estimated_duration_min
                max_dep_end = max(max_dep_end, dep_end)
            earliest_start[task.task_id] = max_dep_end

        # Find task with latest end time
        max_end_time = 0
        last_task_id = None
        for task_id, start_time in earliest_start.items():
            end_time = start_time + self.tasks[task_id].estimated_duration_min
            if end_time > max_end_time:
                max_end_time = end_time
                last_task_id = task_id

        # Backtrack to find critical path
        critical_path = []
        if last_task_id:
            self._find_critical_path_recursive(last_task_id, earliest_start, critical_path)

        return list(reversed(critical_path))

    def _find_critical_path_recursive(
        self,
        task_id: str,
        earliest_start: Dict[str, int],
        path: List[Task]
    ):
        """Recursive helper for critical path finding."""
        task = self.tasks[task_id]
        path.append(task)

        if not task.dependencies:
            return

        # Find dependency on critical path (earliest start matches requirement)
        task_start = earliest_start[task_id]
        for dep_id in task.dependencies:
            dep_task = self.tasks[dep_id]
            dep_end = earliest_start[dep_id] + dep_task.estimated_duration_min
            if dep_end == task_start:
                self._find_critical_path_recursive(dep_id, earliest_start, path)
                break

    def validate(self) -> Dict[str, List[str]]:
        """Validate DAG structure and return any errors."""
        errors = {}

        # Check for cycles
        for task_id in self.tasks:
            for dep_id in self.tasks[task_id].dependencies:
                if self._would_create_cycle(task_id, dep_id):
                    if "cycles" not in errors:
                        errors["cycles"] = []
                    errors["cycles"].append(f"{task_id} -> {dep_id}")

        # Check for missing dependencies
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    if "missing" not in errors:
                        errors["missing"] = []
                    errors["missing"].append(f"{task_id} depends on missing {dep_id}")

        return errors
