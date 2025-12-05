#!/usr/bin/env python3
"""
Task Manager Service - Multi-User Kitchen Coordination

Orchestrates:
- Intelligent task assignment based on skills
- Resource conflict resolution
- Dependency-aware scheduling
- Load balancing across household members
- Real-time task tracking and updates
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from sous.models.user import UserProfile, Household, ResourceType
from sous.models.task import Task, TaskState, TaskDependencyGraph, TaskPriority


@dataclass
class TaskAssignment:
    """Result of task assignment algorithm"""
    task_id: str
    assigned_to: str  # user_id
    scheduled_start: datetime
    scheduled_end: datetime
    confidence: float  # 0.0-1.0 confidence in assignment quality
    reasoning: str


@dataclass
class ScheduleConflict:
    """Detected scheduling conflict"""
    task_id: str
    conflict_type: str  # "resource", "dependency", "time"
    description: str
    affected_tasks: List[str]


class TaskManager:
    """
    Intelligent task manager for multi-user kitchen coordination.

    Capabilities:
    - Skill-based task assignment
    - Resource-aware scheduling
    - DAG dependency resolution
    - Load balancing across users
    - Conflict detection and resolution
    """

    def __init__(
        self,
        household: Household,
        users: Dict[str, UserProfile]
    ):
        self.household = household
        self.users = users
        self.graph = TaskDependencyGraph()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()

    def add_task(self, task: Task):
        """Add a task to the manager."""
        self.graph.add_task(task)
        self.active_tasks[task.task_id] = task

    def add_tasks(self, tasks: List[Task]):
        """Add multiple tasks."""
        for task in tasks:
            self.add_task(task)

    def add_dependency(self, dependent_id: str, dependency_id: str) -> bool:
        """Add task dependency (dependent waits for dependency to complete)."""
        return self.graph.add_dependency(dependent_id, dependency_id)

    async def assign_tasks_intelligently(
        self,
        use_ai: bool = False
    ) -> List[TaskAssignment]:
        """
        Intelligently assign all unassigned tasks to household members.

        Algorithm:
        1. Topological sort of tasks (respects dependencies)
        2. For each task, score all eligible users
        3. Assign to highest-scoring user
        4. Schedule considering resource conflicts
        5. Balance workload across household

        Returns list of assignments with confidence scores.
        """
        assignments = []
        errors = self.graph.validate()
        if errors:
            print(f"⚠️  DAG validation errors: {errors}")
            return assignments

        # Get tasks in dependency order
        ordered_tasks = self.graph.get_topological_order()

        # Track user workload for balancing
        user_workload = {uid: 0 for uid in self.users.keys()}

        for task in ordered_tasks:
            if task.state != TaskState.PENDING or task.assigned_to:
                continue  # Skip already assigned or non-pending tasks

            # Score all users for this task
            scores = self._score_users_for_task(task, user_workload)

            if not scores:
                continue  # No eligible users

            # Assign to highest-scoring user
            best_user_id, confidence = max(scores.items(), key=lambda x: x[1])

            # Schedule the task
            schedule = self._schedule_task(task, best_user_id)

            if schedule:
                start, end = schedule
                task.assign(best_user_id)
                task.schedule(start)

                # Update workload tracking
                user_workload[best_user_id] += task.estimated_duration_min

                assignment = TaskAssignment(
                    task_id=task.task_id,
                    assigned_to=best_user_id,
                    scheduled_start=start,
                    scheduled_end=end,
                    confidence=confidence,
                    reasoning=self._explain_assignment(task, best_user_id, confidence)
                )
                assignments.append(assignment)

        return assignments

    def _score_users_for_task(
        self,
        task: Task,
        current_workload: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Score all users for task suitability (0.0-1.0).

        Factors:
        - Skill match (weight: 0.4)
        - Availability (weight: 0.3)
        - Current workload balance (weight: 0.2)
        - Previous performance (weight: 0.1)
        """
        scores = {}

        for user_id, user in self.users.items():
            # Check if user is in this household
            if user.household_id != self.household.household_id:
                continue

            score = 0.0

            # 1. Skill match (0.4 weight)
            if task.required_skill:
                user_skill = user.get_skill_level(task.required_skill)
                if user_skill >= task.min_skill_level:
                    # Higher skill = higher score, but cap at 1.0
                    skill_score = min(1.0, user_skill / 4.0)  # 4.0 = EXPERT
                    score += skill_score * 0.4
                else:
                    # User lacks required skill
                    continue
            else:
                # No skill requirement - everyone gets baseline
                score += 0.3

            # 2. Availability (0.3 weight)
            if task.scheduled_start:
                day = task.scheduled_start.weekday()
                hour = task.scheduled_start.hour
                if user.is_available_at(day, hour):
                    score += 0.3
                else:
                    score += 0.05  # Penalize but don't exclude
            else:
                # Not scheduled yet - assume available
                score += 0.2

            # 3. Workload balance (0.2 weight)
            max_workload = max(current_workload.values()) if current_workload else 1
            if max_workload > 0:
                user_load = current_workload.get(user_id, 0)
                balance_score = 1.0 - (user_load / max_workload)
                score += balance_score * 0.2

            # 4. Previous performance (0.1 weight)
            completion_rate = 1.0  # Would track actual completion rates
            score += completion_rate * 0.1

            scores[user_id] = min(1.0, score)

        return scores

    def _schedule_task(
        self,
        task: Task,
        user_id: str
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Find earliest available time slot for task considering:
        - User availability
        - Resource availability
        - Dependency completion
        """
        # Start from now or after dependencies complete
        earliest_start = datetime.now()

        # Check dependencies
        for dep_id in task.dependencies:
            dep_task = self.active_tasks.get(dep_id)
            if dep_task and dep_task.scheduled_end:
                earliest_start = max(earliest_start, dep_task.scheduled_end)

        # Find next available slot for user and resources
        duration = timedelta(minutes=task.estimated_duration_min)
        search_end = earliest_start + timedelta(hours=24)  # Search up to 24 hours ahead

        current = earliest_start
        while current < search_end:
            end = current + duration

            # Check if user is available
            user = self.users[user_id]
            day = current.weekday()
            hour = current.hour
            if not user.is_available_at(day, hour):
                current += timedelta(hours=1)
                continue

            # Check if resources are available
            resources_available = True
            for resource_type in task.resources_needed:
                if resource_type in self.household.resources:
                    if not self.household.resources[resource_type].is_available(current, end):
                        resources_available = False
                        break

            if resources_available:
                # Book resources
                for resource_type in task.resources_needed:
                    if resource_type in self.household.resources:
                        self.household.book_resource(
                            resource_type,
                            user_id,
                            task.task_id,
                            current,
                            end
                        )
                return (current, end)

            current += timedelta(minutes=15)  # Try next 15-min slot

        return None  # No available slot found

    def _explain_assignment(
        self,
        task: Task,
        user_id: str,
        confidence: float
    ) -> str:
        """Generate human-readable explanation for assignment."""
        user = self.users[user_id]
        reasons = []

        if task.required_skill:
            user_skill = user.get_skill_level(task.required_skill)
            reasons.append(f"{task.required_skill} skill: {user_skill:.1f}/4.0")

        if confidence > 0.8:
            reasons.append("excellent match")
        elif confidence > 0.6:
            reasons.append("good match")
        else:
            reasons.append("acceptable match")

        return f"Assigned to {user.name} ({', '.join(reasons)})"

    def detect_conflicts(self) -> List[ScheduleConflict]:
        """Detect scheduling conflicts in current task assignments."""
        conflicts = []

        # Check for resource conflicts
        resource_bookings = {}
        for task_id, task in self.active_tasks.items():
            if task.state not in [TaskState.PENDING, TaskState.IN_PROGRESS]:
                continue

            for resource_type in task.resources_needed:
                if resource_type not in resource_bookings:
                    resource_bookings[resource_type] = []

                if task.scheduled_start and task.scheduled_end:
                    resource_bookings[resource_type].append({
                        "task_id": task_id,
                        "start": task.scheduled_start,
                        "end": task.scheduled_end
                    })

        for resource_type, bookings in resource_bookings.items():
            # Check for overlapping bookings
            bookings.sort(key=lambda b: b["start"])
            for i in range(len(bookings) - 1):
                b1 = bookings[i]
                b2 = bookings[i + 1]
                if b1["end"] > b2["start"]:
                    conflicts.append(ScheduleConflict(
                        task_id=b1["task_id"],
                        conflict_type="resource",
                        description=f"{resource_type.value} double-booked",
                        affected_tasks=[b1["task_id"], b2["task_id"]]
                    ))

        # Check for dependency violations
        for task_id, task in self.active_tasks.items():
            if not task.scheduled_start:
                continue

            for dep_id in task.dependencies:
                dep_task = self.active_tasks.get(dep_id)
                if dep_task and dep_task.scheduled_end:
                    if task.scheduled_start < dep_task.scheduled_end:
                        conflicts.append(ScheduleConflict(
                            task_id=task_id,
                            conflict_type="dependency",
                            description=f"Starts before dependency {dep_id} completes",
                            affected_tasks=[task_id, dep_id]
                        ))

        return conflicts

    async def start_task(self, task_id: str) -> bool:
        """Start a task (mark as in progress)."""
        if task_id not in self.active_tasks:
            return False

        task = self.active_tasks[task_id]

        # Check dependencies
        if task.is_blocked(self.completed_tasks):
            return False

        return task.start()

    async def complete_task(self, task_id: str, notes: str = "") -> bool:
        """Complete a task."""
        if task_id not in self.active_tasks:
            return False

        task = self.active_tasks[task_id]
        if task.complete(notes):
            self.completed_tasks.add(task_id)

            # Update household stats
            if task.recipe_id:
                self.household.record_meal()

            # Award points to user
            if task.assigned_to and task.assigned_to in self.users:
                user = self.users[task.assigned_to]
                points = 10 + task.priority.value * 5  # 15-30 points based on priority
                user.complete_task(points)

                # Update skill if task required specific technique
                if task.required_skill:
                    skill_gain = 0.05  # Small skill improvement per task
                    user.update_skill(task.required_skill, skill_gain)

            # Release resources
            for resource_type in task.resources_needed:
                self.household.release_resource(resource_type, task_id)

            return True

        return False

    async def reassign_task(
        self,
        task_id: str,
        new_user_id: str,
        reason: str = ""
    ) -> bool:
        """Reassign a task to a different user."""
        if task_id not in self.active_tasks:
            return False

        task = self.active_tasks[task_id]

        # Release current resources
        for resource_type in task.resources_needed:
            self.household.release_resource(resource_type, task_id)

        # Assign to new user
        task.assign(new_user_id)

        # Reschedule
        schedule = self._schedule_task(task, new_user_id)
        if schedule:
            start, end = schedule
            task.schedule(start)
            return True

        return False

    def get_user_tasks(self, user_id: str) -> List[Task]:
        """Get all tasks assigned to a specific user."""
        return [
            task for task in self.active_tasks.values()
            if task.assigned_to == user_id
        ]

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that can start now (dependencies satisfied)."""
        return self.graph.get_ready_tasks(self.completed_tasks)

    def get_parallel_tasks(self) -> List[List[Task]]:
        """Get tasks grouped by parallelization level."""
        return self.graph.get_parallel_tasks(self.completed_tasks)

    def get_critical_path(self) -> List[Task]:
        """Get critical path through task DAG."""
        return self.graph.get_critical_path()

    def estimate_completion_time(self) -> int:
        """Estimate total completion time in minutes (based on critical path)."""
        critical_path = self.get_critical_path()
        return sum(task.estimated_duration_min for task in critical_path)

    def get_dashboard_summary(self) -> Dict:
        """Get summary statistics for dashboard display."""
        pending = sum(1 for t in self.active_tasks.values() if t.state == TaskState.PENDING)
        in_progress = sum(1 for t in self.active_tasks.values() if t.state == TaskState.IN_PROGRESS)
        completed = len(self.completed_tasks)
        blocked = sum(1 for t in self.active_tasks.values() if t.is_blocked(self.completed_tasks))

        total_time = self.estimate_completion_time()
        elapsed_time = sum(
            t.actual_duration_min or 0
            for t in self.active_tasks.values()
            if t.state == TaskState.COMPLETED
        )

        return {
            "total_tasks": len(self.active_tasks),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "blocked": blocked,
            "estimated_completion_min": total_time,
            "elapsed_min": elapsed_time,
            "progress_percent": int((completed / len(self.active_tasks)) * 100) if self.active_tasks else 0,
            "conflicts": len(self.detect_conflicts())
        }

    def get_user_leaderboard(self) -> List[Dict]:
        """Get user leaderboard sorted by points."""
        leaderboard = []
        for user_id, user in self.users.items():
            if user.household_id != self.household.household_id:
                continue

            tasks_assigned = len([t for t in self.active_tasks.values() if t.assigned_to == user_id])
            tasks_completed_now = len([
                t for t in self.active_tasks.values()
                if t.assigned_to == user_id and t.state == TaskState.COMPLETED
            ])

            leaderboard.append({
                "user_id": user_id,
                "name": user.name,
                "points": user.total_points,
                "level": user.level,
                "tasks_assigned": tasks_assigned,
                "tasks_completed": tasks_completed_now,
                "completion_rate": tasks_completed_now / tasks_assigned if tasks_assigned > 0 else 0.0
            })

        return sorted(leaderboard, key=lambda x: x["points"], reverse=True)

    def to_dict(self) -> dict:
        """Serialize task manager state."""
        return {
            "household_id": self.household.household_id,
            "active_tasks": [t.to_dict() for t in self.active_tasks.values()],
            "completed_tasks": list(self.completed_tasks),
            "summary": self.get_dashboard_summary()
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        household: Household,
        users: Dict[str, UserProfile]
    ) -> 'TaskManager':
        """Deserialize task manager state."""
        manager = cls(household, users)

        # Restore active tasks
        for task_data in data.get("active_tasks", []):
            task = Task.from_dict(task_data)
            manager.add_task(task)

            # Restore dependencies
            for dep_id in task.dependencies:
                manager.add_dependency(task.task_id, dep_id)

        # Restore completed tasks
        manager.completed_tasks = set(data.get("completed_tasks", []))

        return manager
