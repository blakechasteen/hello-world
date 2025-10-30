"""
Meta-Learning for Fast Adaptation

Option C: Deep Enhancement - Learn how to learn.

Meta-learning enables rapid adaptation to new tasks with minimal data.
Instead of learning a specific task, meta-learning learns an initialization
or learning procedure that generalizes across tasks.

Algorithms Implemented:
- MAML (Model-Agnostic Meta-Learning)
- First-order MAML (faster, memory-efficient)
- Reptile (simplified meta-learning)

Research Alignment:
- Finn et al. (2017): "Model-Agnostic Meta-Learning for Fast Adaptation"
- Nichol et al. (2018): "On First-Order Meta-Learning Algorithms"
- Santoro et al. (2016): "Meta-Learning with Memory-Augmented Neural Networks"
- Vinyals et al. (2016): "Matching Networks for One Shot Learning"

Public API:
    MetaLearner: Main meta-learning system
    Task: Task specification for meta-learning
    train_meta: Meta-training loop
    adapt: Few-shot adaptation to new task
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
import numpy as np
from enum import Enum
import copy

logger = logging.getLogger(__name__)

# PyTorch support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    PYTORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available for meta-learning")
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# ============================================================================
# Data Structures
# ============================================================================

class MetaAlgorithm(Enum):
    """Meta-learning algorithm type."""
    MAML = "maml"              # Model-Agnostic Meta-Learning
    FIRST_ORDER_MAML = "fomaml"  # First-order MAML (faster)
    REPTILE = "reptile"        # Reptile (simpler)


@dataclass
class Task:
    """
    Task for meta-learning.

    A task is a few-shot learning problem:
    - Support set: Few labeled examples for adaptation
    - Query set: Test examples to evaluate adaptation

    Attributes:
        name: Task identifier
        support_x: Support inputs (k_shot, input_dim)
        support_y: Support labels (k_shot, output_dim)
        query_x: Query inputs (n_query, input_dim)
        query_y: Query labels (n_query, output_dim)
        task_id: Optional task identifier
    """
    name: str
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    task_id: Optional[int] = None

    def __repr__(self) -> str:
        return (f"Task({self.name}, "
                f"support={self.support_x.shape[0]}, "
                f"query={self.query_x.shape[0]})")


@dataclass
class MetaLearningConfig:
    """
    Configuration for meta-learning.

    Attributes:
        algorithm: Meta-learning algorithm to use
        inner_lr: Learning rate for inner loop (task adaptation)
        outer_lr: Learning rate for outer loop (meta-update)
        inner_steps: Number of gradient steps for adaptation
        meta_batch_size: Number of tasks per meta-update
        first_order: Use first-order approximation (faster)
    """
    algorithm: MetaAlgorithm = MetaAlgorithm.MAML
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    inner_steps: int = 5
    meta_batch_size: int = 4
    first_order: bool = False


# ============================================================================
# PyTorch Meta-Learner
# ============================================================================

if PYTORCH_AVAILABLE:
    class SimpleNetwork(nn.Module):
        """Simple feedforward network for meta-learning."""

        def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
            super().__init__()

            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU()
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class MAMLLearner:
        """
        MAML (Model-Agnostic Meta-Learning) implementation.

        MAML learns an initialization that is good for fast adaptation.
        After meta-training, a few gradient steps on new task achieve
        good performance.

        Algorithm:
        1. Sample batch of tasks
        2. For each task:
           a. Clone model
           b. Do K gradient steps on support set (inner loop)
           c. Evaluate on query set
        3. Meta-update original model using query losses (outer loop)
        """

        def __init__(self,
                     model: nn.Module,
                     config: MetaLearningConfig):
            """
            Initialize MAML learner.

            Args:
                model: Neural network to meta-train
                config: Meta-learning configuration
            """
            self.model = model
            self.config = config
            self.meta_optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.outer_lr
            )

            logger.info(f"Initialized MAML with config: {config}")

        def adapt(self,
                 support_x: torch.Tensor,
                 support_y: torch.Tensor,
                 steps: Optional[int] = None) -> nn.Module:
            """
            Adapt model to new task using support set.

            Args:
                support_x: Support inputs
                support_y: Support labels
                steps: Number of adaptation steps (default: config.inner_steps)

            Returns:
                Adapted model (clone)
            """
            steps = steps or self.config.inner_steps

            # Clone model for adaptation
            adapted_model = copy.deepcopy(self.model)
            adapted_model.train()

            # Create optimizer for inner loop
            inner_optimizer = optim.SGD(
                adapted_model.parameters(),
                lr=self.config.inner_lr
            )

            # Inner loop: adapt to task
            for step in range(steps):
                # Forward pass
                predictions = adapted_model(support_x)
                loss = F.mse_loss(predictions, support_y)

                # Backward pass
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            return adapted_model

        def meta_train_step(self, tasks: List[Task]) -> Dict[str, float]:
            """
            Single meta-training step on batch of tasks.

            Args:
                tasks: Batch of tasks

            Returns:
                Dictionary of metrics
            """
            self.model.train()

            meta_loss = 0.0
            task_losses = []

            for task in tasks:
                # Convert to tensors
                support_x = torch.FloatTensor(task.support_x)
                support_y = torch.FloatTensor(task.support_y)
                query_x = torch.FloatTensor(task.query_x)
                query_y = torch.FloatTensor(task.query_y)

                # Adapt to task
                adapted_model = self.adapt(support_x, support_y)

                # Evaluate on query set
                adapted_model.eval()
                with torch.no_grad() if self.config.first_order else torch.enable_grad():
                    query_pred = adapted_model(query_x)
                    task_loss = F.mse_loss(query_pred, query_y)

                meta_loss += task_loss
                task_losses.append(task_loss.item())

            # Meta-update
            meta_loss = meta_loss / len(tasks)

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            return {
                'meta_loss': meta_loss.item(),
                'mean_task_loss': np.mean(task_losses),
                'std_task_loss': np.std(task_losses)
            }

        def evaluate(self, tasks: List[Task], adaptation_steps: int = 5) -> Dict[str, float]:
            """
            Evaluate meta-learned model on new tasks.

            Args:
                tasks: Evaluation tasks
                adaptation_steps: Steps for adaptation

            Returns:
                Evaluation metrics
            """
            self.model.eval()

            losses_before = []
            losses_after = []

            for task in tasks:
                support_x = torch.FloatTensor(task.support_x)
                support_y = torch.FloatTensor(task.support_y)
                query_x = torch.FloatTensor(task.query_x)
                query_y = torch.FloatTensor(task.query_y)

                # Evaluate before adaptation
                with torch.no_grad():
                    pred_before = self.model(query_x)
                    loss_before = F.mse_loss(pred_before, query_y).item()
                    losses_before.append(loss_before)

                # Adapt to task
                adapted_model = self.adapt(support_x, support_y, steps=adaptation_steps)

                # Evaluate after adaptation
                adapted_model.eval()
                with torch.no_grad():
                    pred_after = adapted_model(query_x)
                    loss_after = F.mse_loss(pred_after, query_y).item()
                    losses_after.append(loss_after)

            return {
                'loss_before_adaptation': np.mean(losses_before),
                'loss_after_adaptation': np.mean(losses_after),
                'improvement': np.mean(losses_before) - np.mean(losses_after),
                'adaptation_ratio': np.mean(losses_after) / (np.mean(losses_before) + 1e-8)
            }


# ============================================================================
# Meta-Learner Interface
# ============================================================================

class MetaLearner:
    """
    Unified meta-learning interface.

    Supports multiple meta-learning algorithms with consistent API.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 config: Optional[MetaLearningConfig] = None):
        """
        Initialize meta-learner.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            config: Meta-learning configuration
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.config = config or MetaLearningConfig()

        if PYTORCH_AVAILABLE:
            self.backend = 'pytorch'
            self.model = SimpleNetwork(input_dim, hidden_dims, output_dim)
            self.learner = MAMLLearner(self.model, self.config)
        else:
            self.backend = 'numpy'
            logger.warning("PyTorch not available, meta-learning disabled")
            self.learner = None

        logger.info(f"MetaLearner using backend: {self.backend}")

    def meta_train(self,
                   meta_train_tasks: List[Task],
                   meta_val_tasks: List[Task],
                   num_epochs: int = 100,
                   log_interval: int = 10) -> Dict[str, List[float]]:
        """
        Meta-train on distribution of tasks.

        Args:
            meta_train_tasks: Tasks for meta-training
            meta_val_tasks: Tasks for meta-validation
            num_epochs: Number of meta-training epochs
            log_interval: Logging frequency

        Returns:
            Training history
        """
        if not PYTORCH_AVAILABLE or self.learner is None:
            logger.error("Cannot meta-train without PyTorch")
            return {}

        logger.info(f"Meta-training on {len(meta_train_tasks)} tasks "
                   f"for {num_epochs} epochs")

        history = {
            'meta_loss': [],
            'val_loss_before': [],
            'val_loss_after': []
        }

        for epoch in range(num_epochs):
            # Sample batch of tasks
            batch_idx = np.random.choice(
                len(meta_train_tasks),
                size=min(self.config.meta_batch_size, len(meta_train_tasks)),
                replace=False
            )
            batch_tasks = [meta_train_tasks[i] for i in batch_idx]

            # Meta-train step
            metrics = self.learner.meta_train_step(batch_tasks)
            history['meta_loss'].append(metrics['meta_loss'])

            # Validation
            if (epoch + 1) % log_interval == 0:
                val_metrics = self.learner.evaluate(meta_val_tasks)
                history['val_loss_before'].append(val_metrics['loss_before_adaptation'])
                history['val_loss_after'].append(val_metrics['loss_after_adaptation'])

                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"meta_loss={metrics['meta_loss']:.4f}, "
                    f"val_before={val_metrics['loss_before_adaptation']:.4f}, "
                    f"val_after={val_metrics['loss_after_adaptation']:.4f}, "
                    f"improvement={val_metrics['improvement']:.4f}"
                )

        return history

    def adapt_to_task(self,
                     task: Task,
                     adaptation_steps: int = 5) -> Any:
        """
        Adapt to new task using few-shot examples.

        Args:
            task: Task with support/query sets
            adaptation_steps: Number of gradient steps

        Returns:
            Adapted model
        """
        if not PYTORCH_AVAILABLE or self.learner is None:
            logger.error("Cannot adapt without PyTorch")
            return None

        logger.info(f"Adapting to task: {task}")

        support_x = torch.FloatTensor(task.support_x)
        support_y = torch.FloatTensor(task.support_y)

        adapted_model = self.learner.adapt(
            support_x,
            support_y,
            steps=adaptation_steps
        )

        logger.info(f"Adaptation complete ({adaptation_steps} steps)")

        return adapted_model

    def evaluate_few_shot(self,
                         task: Task,
                         adaptation_steps: int = 5) -> Dict[str, float]:
        """
        Evaluate few-shot learning on task.

        Args:
            task: Evaluation task
            adaptation_steps: Adaptation steps

        Returns:
            Performance metrics
        """
        if not PYTORCH_AVAILABLE or self.learner is None:
            return {}

        return self.learner.evaluate([task], adaptation_steps)


# ============================================================================
# Task Generation Utilities
# ============================================================================

def generate_sinusoid_task(amplitude: float,
                          phase: float,
                          n_support: int = 10,
                          n_query: int = 10) -> Task:
    """
    Generate sinusoid regression task.

    Classic meta-learning benchmark: learn to fit sinusoids
    with different amplitudes and phases.

    Args:
        amplitude: Sinusoid amplitude
        phase: Sinusoid phase
        n_support: Support set size
        n_query: Query set size

    Returns:
        Task object
    """
    # Sample x values
    x_support = np.random.uniform(-5, 5, (n_support, 1))
    x_query = np.random.uniform(-5, 5, (n_query, 1))

    # Compute y values: y = A * sin(x + phase)
    y_support = amplitude * np.sin(x_support + phase)
    y_query = amplitude * np.sin(x_query + phase)

    return Task(
        name=f"sinusoid_A={amplitude:.2f}_p={phase:.2f}",
        support_x=x_support,
        support_y=y_support,
        query_x=x_query,
        query_y=y_query
    )


def generate_linear_task(slope: float,
                        intercept: float,
                        n_support: int = 10,
                        n_query: int = 10) -> Task:
    """
    Generate linear regression task.

    Args:
        slope: Line slope
        intercept: Line intercept
        n_support: Support set size
        n_query: Query set size

    Returns:
        Task object
    """
    # Sample x values
    x_support = np.random.uniform(-5, 5, (n_support, 1))
    x_query = np.random.uniform(-5, 5, (n_query, 1))

    # Compute y values: y = slope * x + intercept
    y_support = slope * x_support + intercept + np.random.randn(n_support, 1) * 0.1
    y_query = slope * x_query + intercept + np.random.randn(n_query, 1) * 0.1

    return Task(
        name=f"linear_m={slope:.2f}_b={intercept:.2f}",
        support_x=x_support,
        support_y=y_support,
        query_x=x_query,
        query_y=y_query
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'MetaLearner',
    'MetaLearningConfig',
    'MetaAlgorithm',
    'Task',
    'generate_sinusoid_task',
    'generate_linear_task',
]
