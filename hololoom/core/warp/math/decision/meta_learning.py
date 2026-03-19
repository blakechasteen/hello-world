"""
Meta-Learning — Learning to Learn Across Tasks
===============================================

Implements gradient-based meta-learning algorithms that find initializations
from which new tasks can be learned in very few gradient steps.

    1. MAML (Model-Agnostic Meta-Learning): Inner loop adapts to individual
       tasks, outer loop optimizes the initialization across all tasks.

    2. Reptile: Simplified MAML that avoids second derivatives by interpolating
       between the meta-parameters and task-adapted parameters.

    3. TaskDistribution: Synthetic task generators for testing and benchmarking.

Mathematical Foundation:
    MAML inner loop:    theta'_i = theta - alpha * grad_theta L(theta, task_i)
    MAML outer loop:    theta -= beta * sum_i grad_theta L(theta'_i, task_i)
    Reptile:            theta += epsilon * (1/n) * sum_i (theta'_i - theta)

    The key insight: the meta-parameters theta are NOT optimized for any single
    task. They are optimized to be a good STARTING POINT for fast adaptation.

Pure numpy, no framework dependencies. Uses finite differences for gradients.

Author: Claude Code (with HoloLoom architecture by Blake)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Task:
    """A single learning task with train/test split."""
    train_x: np.ndarray                        # Training inputs (n_train, input_dim)
    train_y: np.ndarray                        # Training targets (n_train, output_dim) or (n_train,)
    test_x: np.ndarray                         # Test inputs (n_test, input_dim)
    test_y: np.ndarray                         # Test targets (n_test, output_dim) or (n_test,)


@dataclass
class MAMLState:
    """Internal state of MAML optimizer."""
    theta: np.ndarray                          # Meta-parameters
    inner_lr: float                            # Inner loop learning rate (alpha)
    outer_lr: float                            # Outer loop learning rate (beta)


@dataclass
class MetaLearnResult:
    """Result of a meta-learning run."""
    meta_params: np.ndarray                    # Learned meta-parameters
    task_performances: list[float]             # Per-task test losses after adaptation
    adaptation_speed: float                    # Average improvement per inner step
    n_tasks: int                               # Number of tasks trained on


# ============================================================================
# MAML (Model-Agnostic Meta-Learning)
# ============================================================================

class MAML:
    """
    Model-Agnostic Meta-Learning via finite-difference gradients.

    MAML finds initialization parameters theta* such that a few gradient
    steps on any new task from the task distribution yield good performance.

    Since we operate on raw numpy (no autograd), gradients are computed
    via finite differences. This is O(dim) per gradient but keeps the
    implementation framework-free.

    The loss_fn and grad_fn are provided by the caller, making this
    truly model-agnostic — it works with any differentiable model
    parameterized by a flat parameter vector.
    """

    def __init__(
        self,
        param_dim: int,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        initial_theta: np.ndarray | None = None,
    ):
        """
        Initialize MAML.

        Args:
            param_dim: Dimensionality of parameter vector
            inner_lr: Learning rate for task-specific adaptation
            outer_lr: Learning rate for meta-update
            initial_theta: Starting meta-parameters (small random if None)
        """
        self.param_dim = param_dim

        if initial_theta is not None:
            theta = np.asarray(initial_theta, dtype=np.float64).copy()
        else:
            theta = np.random.randn(param_dim) * 0.01

        self.state = MAMLState(
            theta=theta,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
        )

    def inner_loop(
        self,
        theta: np.ndarray,
        task: Task,
        loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        n_steps: int = 5,
    ) -> np.ndarray:
        """
        Adapt parameters to a single task via gradient descent.

        This is the "fast adaptation" inner loop. Starting from theta,
        take n_steps of gradient descent on the task's training data.

        Args:
            theta: Starting parameters
            task: Task to adapt to
            loss_fn: Loss function (params, x, y) -> scalar
            grad_fn: Gradient function (params, x, y) -> gradient vector
            n_steps: Number of inner gradient steps

        Returns:
            Adapted parameters theta' for this task
        """
        theta_prime = theta.copy()

        for _ in range(n_steps):
            grad = grad_fn(theta_prime, task.train_x, task.train_y)
            theta_prime -= self.state.inner_lr * grad

        return theta_prime

    def outer_loop(
        self,
        theta: np.ndarray,
        tasks: list[Task],
        loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        n_inner_steps: int = 5,
    ) -> np.ndarray:
        """
        Compute the meta-gradient across tasks.

        For each task:
        1. Run inner loop to get theta'_i
        2. Evaluate test loss at theta'_i
        3. Compute gradient of test loss w.r.t. ORIGINAL theta

        The meta-gradient uses finite differences to approximate the
        second-order gradient (gradient through the inner loop).

        Args:
            theta: Current meta-parameters
            tasks: List of tasks
            loss_fn: Loss function
            grad_fn: Gradient function
            n_inner_steps: Inner loop steps per task

        Returns:
            Meta-gradient (same shape as theta)
        """
        meta_grad = np.zeros_like(theta)
        eps = 1e-5  # Finite difference step

        for task in tasks:
            # Compute meta-gradient via finite differences
            # d/d(theta) L_test(theta'(theta))
            for d in range(len(theta)):
                theta_plus = theta.copy()
                theta_plus[d] += eps

                theta_minus = theta.copy()
                theta_minus[d] -= eps

                # Inner loop with perturbed theta
                adapted_plus = self.inner_loop(
                    theta_plus, task, loss_fn, grad_fn, n_inner_steps
                )
                adapted_minus = self.inner_loop(
                    theta_minus, task, loss_fn, grad_fn, n_inner_steps
                )

                # Test loss after adaptation
                loss_plus = loss_fn(adapted_plus, task.test_x, task.test_y)
                loss_minus = loss_fn(adapted_minus, task.test_x, task.test_y)

                meta_grad[d] += (loss_plus - loss_minus) / (2 * eps)

        meta_grad /= len(tasks)
        return meta_grad

    def meta_train(
        self,
        tasks: list[Task],
        loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        n_epochs: int = 100,
        n_inner_steps: int = 5,
        tasks_per_batch: int | None = None,
    ) -> MetaLearnResult:
        """
        Full MAML meta-training loop.

        Args:
            tasks: List of training tasks
            loss_fn: Loss function (params, x, y) -> scalar
            grad_fn: Gradient function (params, x, y) -> gradient vector
            n_epochs: Number of meta-training epochs
            n_inner_steps: Inner loop steps per task
            tasks_per_batch: Tasks per meta-batch (all if None)

        Returns:
            MetaLearnResult with learned meta-parameters
        """
        batch_size = tasks_per_batch if tasks_per_batch is not None else len(tasks)

        for epoch in range(n_epochs):
            # Sample task batch
            indices = np.random.choice(len(tasks), size=min(batch_size, len(tasks)), replace=False)
            batch = [tasks[i] for i in indices]

            # Meta-gradient
            meta_grad = self.outer_loop(
                self.state.theta, batch, loss_fn, grad_fn, n_inner_steps
            )

            # Meta-update
            self.state.theta -= self.state.outer_lr * meta_grad

        # Evaluate final performance
        performances = []
        improvements = []
        for task in tasks:
            loss_before = loss_fn(self.state.theta, task.test_x, task.test_y)
            adapted = self.inner_loop(
                self.state.theta, task, loss_fn, grad_fn, n_inner_steps
            )
            loss_after = loss_fn(adapted, task.test_x, task.test_y)
            performances.append(loss_after)
            improvements.append(loss_before - loss_after)

        return MetaLearnResult(
            meta_params=self.state.theta.copy(),
            task_performances=performances,
            adaptation_speed=float(np.mean(improvements)) if improvements else 0.0,
            n_tasks=len(tasks),
        )

    def adapt(
        self,
        theta: np.ndarray,
        new_task: Task,
        loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        n_steps: int = 5,
    ) -> np.ndarray:
        """
        Fast adaptation to a new task from meta-learned initialization.

        This is the deployment-time operation: given meta-trained theta,
        adapt to a new task in just n_steps of gradient descent.

        Args:
            theta: Meta-learned parameters (or self.state.theta)
            new_task: New task to adapt to
            loss_fn: Loss function
            grad_fn: Gradient function
            n_steps: Adaptation steps

        Returns:
            Adapted parameters for the new task
        """
        return self.inner_loop(theta, new_task, loss_fn, grad_fn, n_steps)


# ============================================================================
# Reptile
# ============================================================================

class Reptile:
    """
    Reptile meta-learning: a simpler alternative to MAML.

    Instead of computing second-order gradients through the inner loop,
    Reptile simply trains on each task for several steps, then moves
    the meta-parameters toward the task-adapted parameters:

        theta += epsilon * (theta'_i - theta)

    This is mathematically related to MAML but avoids the expensive
    meta-gradient computation. Empirically competitive on many benchmarks.

    The intuition: if a point theta is close to the optimal solution
    for many tasks, it's a good initialization. Reptile directly
    optimizes for this by moving toward task optima.
    """

    def __init__(
        self,
        param_dim: int,
        initial_theta: np.ndarray | None = None,
    ):
        """
        Args:
            param_dim: Dimensionality of parameter vector
            initial_theta: Starting point (small random if None)
        """
        self.param_dim = param_dim
        if initial_theta is not None:
            self.theta = np.asarray(initial_theta, dtype=np.float64).copy()
        else:
            self.theta = np.random.randn(param_dim) * 0.01

    def meta_train(
        self,
        tasks: list[Task],
        loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        n_epochs: int = 100,
        inner_steps: int = 10,
        inner_lr: float = 0.01,
        epsilon: float = 0.1,
    ) -> MetaLearnResult:
        """
        Reptile meta-training.

        For each epoch:
        1. Sample a task
        2. Train on it for inner_steps
        3. Move meta-params toward the trained params

        Args:
            tasks: Training tasks
            loss_fn: Loss function (params, x, y) -> scalar
            grad_fn: Gradient function (params, x, y) -> gradient vector
            n_epochs: Number of meta-training epochs
            inner_steps: SGD steps per task
            inner_lr: Inner loop learning rate
            epsilon: Meta step size (interpolation rate)

        Returns:
            MetaLearnResult with learned meta-parameters
        """
        for epoch in range(n_epochs):
            # Sample a random task
            task = tasks[np.random.randint(len(tasks))]

            # Train on this task
            theta_task = self.theta.copy()
            for _ in range(inner_steps):
                grad = grad_fn(theta_task, task.train_x, task.train_y)
                theta_task -= inner_lr * grad

            # Reptile update: move toward task solution
            self.theta += epsilon * (theta_task - self.theta)

        # Evaluate
        performances = []
        improvements = []
        for task in tasks:
            loss_before = loss_fn(self.theta, task.test_x, task.test_y)
            theta_adapted = self.theta.copy()
            for _ in range(inner_steps):
                grad = grad_fn(theta_adapted, task.train_x, task.train_y)
                theta_adapted -= inner_lr * grad
            loss_after = loss_fn(theta_adapted, task.test_x, task.test_y)
            performances.append(loss_after)
            improvements.append(loss_before - loss_after)

        return MetaLearnResult(
            meta_params=self.theta.copy(),
            task_performances=performances,
            adaptation_speed=float(np.mean(improvements)) if improvements else 0.0,
            n_tasks=len(tasks),
        )


# ============================================================================
# Task Distribution (Synthetic Task Generators)
# ============================================================================

class TaskDistribution:
    """
    Generate synthetic task distributions for meta-learning benchmarks.

    Provides standard task families:
    - Sine regression: learn sine waves with varying amplitude/phase
    - Classification: learn separating hyperplanes with varying orientations
    """

    @staticmethod
    def sample_regression_tasks(
        n_tasks: int,
        n_points: int = 20,
        noise: float = 0.1,
        input_range: tuple[float, float] = (-5.0, 5.0),
    ) -> list[Task]:
        """
        Sample sine regression tasks with varying amplitude and phase.

        Each task is: y = A * sin(x + phi) + noise
        where A ~ U[0.1, 5.0] and phi ~ U[0, pi].

        This is the standard MAML benchmark from Finn et al. 2017.

        Args:
            n_tasks: Number of tasks to generate
            n_points: Points per task (split 50/50 train/test)
            noise: Gaussian noise standard deviation
            input_range: Input domain (min, max)

        Returns:
            List of regression Tasks
        """
        tasks = []
        n_train = n_points // 2
        n_test = n_points - n_train

        for _ in range(n_tasks):
            amplitude = np.random.uniform(0.1, 5.0)
            phase = np.random.uniform(0, np.pi)

            # Training data
            train_x = np.random.uniform(input_range[0], input_range[1], (n_train, 1))
            train_y = amplitude * np.sin(train_x + phase) + noise * np.random.randn(n_train, 1)

            # Test data
            test_x = np.random.uniform(input_range[0], input_range[1], (n_test, 1))
            test_y = amplitude * np.sin(test_x + phase) + noise * np.random.randn(n_test, 1)

            tasks.append(Task(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
            ))

        return tasks

    @staticmethod
    def sample_classification_tasks(
        n_tasks: int,
        n_points: int = 40,
        n_classes: int = 2,
        dim: int = 2,
    ) -> list[Task]:
        """
        Sample linear classification tasks with random separating hyperplanes.

        Each task has a random weight vector w. Points are classified by
        sign(w @ x). This tests whether meta-learning can find an
        initialization that quickly adapts to new decision boundaries.

        Args:
            n_tasks: Number of tasks to generate
            n_points: Points per task (split 50/50)
            n_classes: Number of classes (only 2 supported currently)
            dim: Input dimensionality

        Returns:
            List of classification Tasks
        """
        tasks = []
        n_train = n_points // 2
        n_test = n_points - n_train

        for _ in range(n_tasks):
            # Random separating hyperplane
            w = np.random.randn(dim)
            w /= np.linalg.norm(w)

            # Training data
            train_x = np.random.randn(n_train, dim)
            train_y = (train_x @ w > 0).astype(np.float64).reshape(-1, 1)

            # Test data
            test_x = np.random.randn(n_test, dim)
            test_y = (test_x @ w > 0).astype(np.float64).reshape(-1, 1)

            tasks.append(Task(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
            ))

        return tasks

    @staticmethod
    def task_similarity(t1: Task, t2: Task) -> float:
        """
        Measure similarity between two tasks via feature space overlap.

        Computes the cosine similarity between the mean feature vectors
        of each task's training data, giving a rough measure of how
        related the tasks are in input space.

        Args:
            t1: First task
            t2: Second task

        Returns:
            Similarity in [-1, 1]. 1 = identical feature distribution.
        """
        mean_1 = t1.train_x.mean(axis=0)
        mean_2 = t2.train_x.mean(axis=0)

        norm_1 = np.linalg.norm(mean_1)
        norm_2 = np.linalg.norm(mean_2)

        if norm_1 < 1e-12 or norm_2 < 1e-12:
            return 0.0

        return float(np.dot(mean_1, mean_2) / (norm_1 * norm_2))


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Task",
    "MAMLState",
    "MetaLearnResult",
    "MAML",
    "Reptile",
    "TaskDistribution",
]
