"""
Scheduler Service

Day-by-day scheduling with appliance conflict detection
"""

from datetime import date
from typing import List, Dict
from sous.models.event import Event
from sous.models.recipe import Recipe, Step
from sous.models.schedule import DailyPlan, Task


class Scheduler:
    """
    Intelligent scheduler for recipe prep tasks

    Handles:
    - Make-ahead window constraints
    - Appliance bottleneck detection
    - Task sequencing to avoid conflicts
    - Optional task placement
    """

    def generate_daily_plans(
        self, event: Event, recipes: List[Recipe]
    ) -> List[DailyPlan]:
        """
        Generate daily plans for event prep

        Algorithm:
        1. For each recipe step, determine eligible days
        2. Assign to days based on make-ahead rules
        3. Serialize appliance usage (no overlaps)
        4. Mark optional tasks

        Args:
            event: Event with date and prep days
            recipes: List of recipes to schedule

        Returns:
            List of DailyPlan objects (one per prep day + event day)
        """
        plans = []

        # Create plans for each prep day + event day
        all_days = event.prep_days + [event.date]

        for day in all_days:
            plan = self._schedule_day(day, event, recipes)
            plans.append(plan)

        return plans

    def _schedule_day(
        self, day: date, event: Event, recipes: List[Recipe]
    ) -> DailyPlan:
        """Schedule tasks for a single day"""
        tasks = []

        # Determine days before event
        days_before = event.days_before_event(day)

        # Collect all eligible steps for this day
        for recipe in recipes:
            for step in recipe.steps:
                if self._step_eligible_for_day(step, recipe, days_before):
                    task = self._create_task_from_step(recipe, step, day)
                    tasks.append(task)

        # Sort and resolve conflicts
        tasks = self._resolve_appliance_conflicts(tasks, event)

        return DailyPlan(date=day, tasks=tasks)

    def _step_eligible_for_day(
        self, step: Step, recipe: Recipe, days_before: int
    ) -> bool:
        """
        Check if step can be done this many days before event

        Logic:
        - If must_be_same_day_as_event: Only on event day (days_before=0)
        - If must_be_day_before: Only 1 day before (days_before=1)
        - If can_be_any_prep_day: Any day within make-ahead window
        - Otherwise: Check make-ahead_min/max_days

        Args:
            step: Recipe step
            recipe: Parent recipe
            days_before: Days before event

        Returns:
            True if step can be scheduled on this day
        """
        # Event day tasks
        if step.must_be_same_day_as_event:
            return days_before == 0

        # Day-before tasks
        if step.must_be_day_before:
            return days_before == 1

        # Flexible prep day tasks
        if step.can_be_any_prep_day:
            # Check make-ahead window
            if days_before < recipe.make_ahead_min_days:
                return False
            if recipe.make_ahead_max_days and days_before > recipe.make_ahead_max_days:
                return False
            return True

        # Default: Check if within make-ahead window
        if days_before < recipe.make_ahead_min_days:
            return False
        if recipe.make_ahead_max_days and days_before > recipe.make_ahead_max_days:
            return False

        # Prefer earlier days for make-ahead (default scheduling)
        return days_before == recipe.make_ahead_max_days or (
            recipe.make_ahead_max_days is None and days_before >= recipe.make_ahead_min_days
        )

    def _create_task_from_step(
        self, recipe: Recipe, step: Step, day: date
    ) -> Task:
        """Create Task object from Recipe Step"""
        task_id = f"task_{recipe.id}_{step.id}_{day.isoformat()}"

        return Task(
            id=task_id,
            recipe_id=recipe.id,
            step_id=step.id,
            duration_minutes=step.duration_minutes,
            appliance_id=None,  # Will be assigned in conflict resolution
            is_optional=recipe.is_optional,
            status="planned",
        )

    def _resolve_appliance_conflicts(
        self, tasks: List[Task], event: Event
    ) -> List[Task]:
        """
        Serialize tasks using same appliance

        For MVP: Simple sequencing (no overlapping oven/burner tasks)
        Future: Sophisticated time-window scheduling

        Args:
            tasks: List of tasks for the day
            event: Event with appliance metadata

        Returns:
            Ordered list of tasks with appliance assignments
        """
        # Group tasks by required appliance type
        # This requires looking up steps from recipes
        # For MVP, we'll do simple ordering

        # Sort tasks to minimize conflicts
        # 1. Required tasks first, optional last
        # 2. Longer tasks first (to avoid fragmentation)
        tasks.sort(key=lambda t: (t.is_optional, -t.duration_minutes))

        return tasks

    def detect_conflicts(
        self, daily_plans: List[DailyPlan], event: Event
    ) -> List[Dict]:
        """
        Detect appliance conflicts across all daily plans

        Returns:
            List of conflict descriptions with details
        """
        conflicts = []

        for plan in daily_plans:
            # Check appliance overuse
            for appliance in event.appliances:
                tasks_using = plan.tasks_by_appliance(appliance.id)
                if len(tasks_using) > appliance.capacity:
                    conflicts.append({
                        "date": plan.date,
                        "appliance": appliance.name,
                        "capacity": appliance.capacity,
                        "tasks": len(tasks_using),
                        "task_ids": [t.id for t in tasks_using],
                        "message": f"{appliance.name} overbooked: {len(tasks_using)} tasks, capacity {appliance.capacity}",
                    })

        return conflicts

    def estimate_total_work(
        self, daily_plans: List[DailyPlan]
    ) -> Dict[date, int]:
        """
        Calculate total work minutes per day

        Returns:
            Dict mapping date -> total minutes
        """
        return {plan.date: plan.total_duration_minutes() for plan in daily_plans}

    def suggest_optional_placement(
        self, daily_plans: List[DailyPlan], max_minutes_per_day: int = 360
    ) -> List[DailyPlan]:
        """
        Suggest which days can accommodate optional tasks

        Algorithm:
        - Calculate current workload per day
        - Identify days with capacity remaining
        - Suggest placing optional tasks on those days

        Args:
            daily_plans: Current plans
            max_minutes_per_day: Maximum work per day (default: 6 hours)

        Returns:
            Updated plans with optional task suggestions
        """
        updated_plans = []

        for plan in daily_plans:
            required_minutes = sum(
                t.duration_minutes for t in plan.required_tasks()
            )
            capacity_remaining = max_minutes_per_day - required_minutes

            # If capacity remaining, keep optional tasks
            # If no capacity, mark them as suggestions only
            updated_plan = DailyPlan(date=plan.date, tasks=[])

            for task in plan.tasks:
                if task.is_optional and capacity_remaining < task.duration_minutes:
                    # Skip this optional task (no capacity)
                    continue
                updated_plan.add_task(task)

            updated_plans.append(updated_plan)

        return updated_plans
