"""
Curriculum Learning — Progressive Difficulty, Self-Paced, Automatic
====================================================================

Training strategies that present tasks in a meaningful order,
progressing from easy to hard. Curriculum learning can significantly
accelerate training and improve final performance.

Core Concepts:
- Curriculum Scheduler: Select tasks based on agent competence
- Self-Paced Learning: Weight samples by loss — easy first, hard later
- Automatic Curriculum: Generate task sequences from difficulty functions

Mathematical Foundation:
Self-paced: min_{w,θ} Σ wᵢ L(xᵢ, θ) - λ Σ wᵢ  (easy-first bias)
  Solution: wᵢ = 1 if L(xᵢ, θ) < λ, else 0
Competence: C(t) = min(1, √(t/T)) (growing competence over time)
Difficulty: d(task) ∈ [0, 1], present task when C(t) ≥ d(task)

Applications to Warp Space:
- Progressive complexity in multi-step weaving tasks
- Self-paced memory consolidation (learn easy patterns first)
- Automatic difficulty sequencing for Thompson Sampling exploration
- Curriculum over scoring term complexity during calibration

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Task:
    """A curriculum task.

    Attributes:
        id: Unique task identifier.
        difficulty: Difficulty level in [0, 1].
        data: Task-specific data.
        metadata: Additional information.
    """
    id: str
    difficulty: float
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumState:
    """State of the curriculum scheduler.

    Attributes:
        current_difficulty: Current difficulty threshold.
        tasks_completed: Number of tasks completed.
        performance_history: Recent performance scores.
        competence: Estimated agent competence in [0, 1].
    """
    current_difficulty: float
    tasks_completed: int
    performance_history: list[float]
    competence: float


# ============================================================================
# CURRICULUM SCHEDULER
# ============================================================================

class CurriculumScheduler:
    """Progressive difficulty curriculum scheduling.

    Selects tasks from a pool based on the agent's current competence.
    Competence grows as the agent demonstrates mastery of current-level
    tasks, unlocking harder ones.

    Strategies:
    - linear: Difficulty threshold increases linearly with time
    - sqrt: Competence grows as √(t/T) (slower start, faster ramp)
    - adaptive: Difficulty adjusts based on actual performance

    Example:
        >>> scheduler = CurriculumScheduler()
        >>> tasks = [Task(f"t{i}", difficulty=i/20) for i in range(20)]
        >>> for step in range(100):
        ...     task = scheduler.select_task(performance=0.8, task_pool=tasks)
        ...     # ... train on task ...
        ...     scheduler.update(performance=0.85)
    """

    def __init__(
        self,
        strategy: str = 'adaptive',
        initial_difficulty: float = 0.1,
        difficulty_increment: float = 0.05,
        performance_threshold: float = 0.7,
        window_size: int = 10,
    ):
        """Initialize curriculum scheduler.

        Args:
            strategy: 'linear', 'sqrt', or 'adaptive'.
            initial_difficulty: Starting difficulty threshold.
            difficulty_increment: How much to increase difficulty per success.
            performance_threshold: Minimum performance to advance difficulty.
            window_size: Window for performance averaging.
        """
        self.strategy = strategy
        self.difficulty_increment = difficulty_increment
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        self._state = CurriculumState(
            current_difficulty=initial_difficulty,
            tasks_completed=0,
            performance_history=[],
            competence=initial_difficulty,
        )

    def select_task(
        self,
        agent_performance: float,
        task_pool: list[Task],
    ) -> Task:
        """Select next task based on agent performance and curriculum state.

        Args:
            agent_performance: Current agent performance score [0, 1].
            task_pool: Available tasks.

        Returns:
            Selected task.
        """
        if not task_pool:
            raise ValueError("Empty task pool")

        # Update performance history
        self._state.performance_history.append(agent_performance)
        if len(self._state.performance_history) > self.window_size:
            self._state.performance_history = self._state.performance_history[-self.window_size:]

        # Update competence based on strategy
        self._update_competence()

        # Filter tasks within current competence
        eligible = [t for t in task_pool if t.difficulty <= self._state.competence]

        if not eligible:
            # If no eligible tasks, pick easiest
            eligible = [min(task_pool, key=lambda t: t.difficulty)]

        # Select from eligible tasks — prefer tasks near competence boundary
        # (zone of proximal development)
        weights = np.array([
            np.exp(-2.0 * (self._state.competence - t.difficulty) ** 2)
            for t in eligible
        ])
        weights /= np.sum(weights) + 1e-10

        idx = np.random.choice(len(eligible), p=weights)
        return eligible[idx]

    def _update_competence(self) -> None:
        """Update competence estimate based on strategy."""
        t = self._state.tasks_completed

        if self.strategy == 'linear':
            self._state.competence = min(1.0, self._state.current_difficulty + t * 0.001)

        elif self.strategy == 'sqrt':
            T = 1000  # Expected total tasks
            self._state.competence = min(1.0, np.sqrt(t / T))

        elif self.strategy == 'adaptive':
            if len(self._state.performance_history) >= self.window_size:
                avg_perf = np.mean(self._state.performance_history[-self.window_size:])
                if avg_perf >= self.performance_threshold:
                    self._state.current_difficulty = min(
                        1.0,
                        self._state.current_difficulty + self.difficulty_increment
                    )
                elif avg_perf < self.performance_threshold * 0.5:
                    self._state.current_difficulty = max(
                        0.0,
                        self._state.current_difficulty - self.difficulty_increment * 0.5
                    )
            self._state.competence = self._state.current_difficulty

    def update(self, performance: float) -> None:
        """Update state after task completion.

        Args:
            performance: Performance on completed task [0, 1].
        """
        self._state.tasks_completed += 1
        self._state.performance_history.append(performance)
        if len(self._state.performance_history) > self.window_size * 2:
            self._state.performance_history = self._state.performance_history[-self.window_size:]
        self._update_competence()

    @property
    def state(self) -> CurriculumState:
        """Current curriculum state."""
        return self._state


# ============================================================================
# SELF-PACED LEARNING
# ============================================================================

class SelfPacedLearning:
    """Self-paced learning via loss-based sample weighting.

    Formulates curriculum as an optimization problem:
    min_{w, θ} Σ wᵢ L(xᵢ, θ) + f(w, λ)

    where f is a self-paced regularizer:
    - Hard: f(w, λ) = -λ Σ wᵢ  →  wᵢ ∈ {0, 1}
    - Soft: f(w, λ) = λ Σ (wᵢ log wᵢ - wᵢ)  →  wᵢ ∈ [0, 1]
    - Linear: f(w, λ) = -λ Σ wᵢ + λ/2 Σ wᵢ²

    As λ increases, more samples are included (curriculum progresses).

    Example:
        >>> spl = SelfPacedLearning()
        >>> losses = compute_losses(model, data)
        >>> weights = spl.weight_samples(losses, lambda_param=0.5)
        >>> # Use weights for weighted training
    """

    def weight_samples(
        self,
        losses: np.ndarray,
        lambda_param: float,
        scheme: str = 'hard',
    ) -> np.ndarray:
        """Compute sample weights from losses.

        Args:
            losses: Per-sample losses (n,).
            lambda_param: Pace parameter (higher = include more samples).
            scheme: Weighting scheme ('hard', 'soft', 'linear', 'mixture').

        Returns:
            Sample weights (n,) in [0, 1].
        """
        losses = np.asarray(losses, dtype=float)

        if scheme == 'hard':
            # Binary: include if loss < λ
            weights = (losses < lambda_param).astype(float)

        elif scheme == 'soft':
            # Soft exponential weighting
            weights = np.exp(-losses / (lambda_param + 1e-8))
            weights = np.clip(weights, 0, 1)

        elif scheme == 'linear':
            # Linear interpolation
            weights = np.clip(1.0 - losses / (lambda_param + 1e-8), 0, 1)

        elif scheme == 'mixture':
            # Mixture of soft and hard
            hard = (losses < lambda_param).astype(float)
            soft = np.exp(-losses / (lambda_param + 1e-8))
            weights = 0.5 * hard + 0.5 * soft

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        return weights

    def adaptive_lambda(
        self,
        losses: np.ndarray,
        target_fraction: float = 0.5,
        scheme: str = 'hard',
    ) -> float:
        """Find λ that includes approximately target_fraction of samples.

        Args:
            losses: Per-sample losses.
            target_fraction: Desired fraction of active samples.
            scheme: Weighting scheme.

        Returns:
            Optimal λ value.
        """
        losses = np.asarray(losses)

        if scheme == 'hard':
            # Set λ to the quantile matching target_fraction
            return float(np.percentile(losses, target_fraction * 100))

        # Binary search for soft/linear schemes
        lo, hi = 0.01, np.max(losses) * 2
        for _ in range(30):
            mid = (lo + hi) / 2
            weights = self.weight_samples(losses, mid, scheme)
            frac = np.mean(weights > 0.5)
            if frac < target_fraction:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2

    def curriculum_schedule(
        self,
        n_epochs: int,
        initial_lambda: float = 0.1,
        final_lambda: float = 10.0,
        schedule: str = 'linear',
    ) -> np.ndarray:
        """Generate λ schedule across training epochs.

        Args:
            n_epochs: Total training epochs.
            initial_lambda: Starting λ (strict, few samples).
            final_lambda: Final λ (permissive, all samples).
            schedule: 'linear', 'exponential', 'cosine'.

        Returns:
            Array of λ values (n_epochs,).
        """
        t = np.linspace(0, 1, n_epochs)

        if schedule == 'linear':
            lambdas = initial_lambda + (final_lambda - initial_lambda) * t

        elif schedule == 'exponential':
            log_init = np.log(initial_lambda + 1e-8)
            log_final = np.log(final_lambda)
            lambdas = np.exp(log_init + (log_final - log_init) * t)

        elif schedule == 'cosine':
            # Cosine annealing from initial to final
            lambdas = final_lambda - (final_lambda - initial_lambda) * 0.5 * (1 + np.cos(np.pi * t))

        else:
            lambdas = np.full(n_epochs, final_lambda)

        return lambdas


# ============================================================================
# AUTOMATIC CURRICULUM
# ============================================================================

class AutomaticCurriculum:
    """Automatic curriculum generation from task specifications.

    Generates task sequences by combining difficulty estimation with
    procedural task generation. Can create infinitely many tasks
    at any difficulty level.

    Example:
        >>> ac = AutomaticCurriculum()
        >>> def difficulty(params): return np.linalg.norm(params)
        >>> def generate(diff): return {'size': int(diff * 100)}
        >>> tasks = ac.generate(
        ...     task_space_dim=3,
        ...     difficulty_fn=difficulty,
        ...     agent_skill=0.5,
        ...     n_tasks=20,
        ... )
    """

    def generate(
        self,
        task_space_dim: int,
        difficulty_fn: Callable[[np.ndarray], float],
        agent_skill: float,
        n_tasks: int = 10,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
        exploration_noise: float = 0.1,
    ) -> list[Task]:
        """Generate a sequence of tasks at appropriate difficulty.

        Samples tasks from the task space, filters by difficulty
        relative to agent skill, and sorts into curriculum order.

        Args:
            task_space_dim: Dimension of task parameter space.
            difficulty_fn: Maps task parameters to difficulty [0, 1].
            agent_skill: Current agent skill level [0, 1].
            n_tasks: Number of tasks to generate.
            difficulty_range: Range of difficulties to sample from.
            exploration_noise: Noise for difficulty targeting.

        Returns:
            List of tasks in curriculum order (easy to hard).
        """
        # Target difficulty zone: [skill - margin, skill + margin]
        margin = 0.15
        target_low = max(difficulty_range[0], agent_skill - margin)
        target_high = min(difficulty_range[1], agent_skill + margin)

        tasks = []
        attempts = 0
        max_attempts = n_tasks * 20

        while len(tasks) < n_tasks and attempts < max_attempts:
            attempts += 1

            # Sample task parameters
            params = np.random.randn(task_space_dim) * 0.5
            difficulty = difficulty_fn(params)

            # Accept if difficulty is in target range (with noise)
            noisy_low = target_low - exploration_noise
            noisy_high = target_high + exploration_noise

            if noisy_low <= difficulty <= noisy_high:
                tasks.append(Task(
                    id=f"auto_{len(tasks)}",
                    difficulty=float(np.clip(difficulty, 0, 1)),
                    data=params,
                    metadata={'source': 'automatic', 'skill_target': agent_skill},
                ))

        # If not enough tasks in target range, expand search
        if len(tasks) < n_tasks:
            while len(tasks) < n_tasks:
                params = np.random.randn(task_space_dim)
                difficulty = difficulty_fn(params)
                tasks.append(Task(
                    id=f"auto_{len(tasks)}",
                    difficulty=float(np.clip(difficulty, 0, 1)),
                    data=params,
                ))

        # Sort by difficulty (curriculum order)
        tasks.sort(key=lambda t: t.difficulty)
        return tasks[:n_tasks]

    def domain_randomization(
        self,
        base_params: np.ndarray,
        n_variants: int = 10,
        randomization_scale: float = 0.1,
        difficulty_fn: Callable | None = None,
    ) -> list[Task]:
        """Generate task variants via domain randomization.

        Perturbs base task parameters to create diverse training
        experiences at similar difficulty levels.

        Args:
            base_params: Base task parameters.
            n_variants: Number of variants.
            randomization_scale: Scale of random perturbations.
            difficulty_fn: Optional difficulty function.

        Returns:
            List of randomized task variants.
        """
        tasks = []
        for i in range(n_variants):
            noise = np.random.randn(*base_params.shape) * randomization_scale
            params = base_params + noise
            diff = difficulty_fn(params) if difficulty_fn else 0.5

            tasks.append(Task(
                id=f"dr_{i}",
                difficulty=float(np.clip(diff, 0, 1)),
                data=params,
                metadata={'source': 'domain_randomization', 'base': base_params.tolist()},
            ))

        return tasks

    def skill_tree(
        self,
        skills: dict[str, float],
        prerequisites: dict[str, list[str]],
        task_fn: Callable[[str], Task],
    ) -> list[Task]:
        """Generate curriculum from a skill dependency tree.

        Produces a topologically sorted sequence of tasks respecting
        skill prerequisites.

        Args:
            skills: Map from skill name to difficulty.
            prerequisites: Map from skill to required prerequisite skills.
            task_fn: Function to create a Task from a skill name.

        Returns:
            Topologically sorted task list.
        """
        # Topological sort via Kahn's algorithm
        in_degree = dict.fromkeys(skills, 0)
        for s, prereqs in prerequisites.items():
            if s in in_degree:
                in_degree[s] = len(prereqs)

        queue = [s for s, d in in_degree.items() if d == 0]
        sorted_skills = []

        # Build reverse adjacency
        children: dict[str, list[str]] = {s: [] for s in skills}
        for s, prereqs in prerequisites.items():
            for p in prereqs:
                if p in children:
                    children[p].append(s)

        while queue:
            # Pick easiest available skill
            queue.sort(key=lambda s: skills.get(s, 0))
            current = queue.pop(0)
            sorted_skills.append(current)

            for child in children.get(current, []):
                if child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] <= 0:
                        queue.append(child)

        return [task_fn(s) for s in sorted_skills]
