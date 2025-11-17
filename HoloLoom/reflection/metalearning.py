"""
Meta-Learning Module
====================
Enable HoloLoom to adapt quickly to new tasks and domains through meta-learning.

Capabilities:
- Few-shot adaptation to new domains
- Task-specific policy fine-tuning
- Transfer learning across tool categories
- Automatic hyperparameter optimization

Philosophy:
True intelligence is not just learning - it's learning how to learn.
Meta-learning enables the system to rapidly adapt to new situations
by leveraging knowledge from past adaptations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import asyncio

logger = logging.getLogger(__name__)

# Try to import PyTorch for neural meta-learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, meta-learning will use simplified algorithms")
    TORCH_AVAILABLE = False


from . import DecisionTrajectory, ReflectionInsights, PolicyUpdate


@dataclass
class TaskContext:
    """Context for a specific task or domain."""

    task_id: str
    task_type: str  # "math", "text_processing", "data_analysis", etc.
    description: str

    # Task-specific metadata
    required_tools: List[str] = field(default_factory=list)
    typical_features: Dict[str, float] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Adaptation:
    """Record of a task adaptation."""

    task_context: TaskContext
    num_episodes: int
    initial_performance: float
    final_performance: float
    improvement: float

    # Learned parameters
    adapted_parameters: Dict[str, Any]

    # Metadata
    adaptation_time: float  # seconds


class MetaLearner:
    """
    Meta-learning for rapid task adaptation.

    Implements:
    - MAML-like (Model-Agnostic Meta-Learning) adaptation
    - Reptile-style meta-optimization
    - Task embedding learning
    - Few-shot policy adaptation
    """

    def __init__(
        self,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        adaptation_steps: int = 5,
        meta_batch_size: int = 4
    ):
        self.inner_lr = inner_lr  # Task-specific learning rate
        self.outer_lr = outer_lr  # Meta-learning rate
        self.adaptation_steps = adaptation_steps
        self.meta_batch_size = meta_batch_size

        # Storage
        self.task_history: Dict[str, List[Adaptation]] = {}
        self.task_embeddings: Dict[str, np.ndarray] = {}

    async def adapt_to_task(
        self,
        task_context: TaskContext,
        support_trajectories: List[DecisionTrajectory],
        current_policy_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt policy to a new task using few-shot learning.

        Args:
            task_context: Description of the task
            support_trajectories: Few examples for the task
            current_policy_params: Current policy parameters

        Returns:
            Adapted policy parameters
        """

        logger.info(
            f"Adapting to task '{task_context.task_id}' "
            f"with {len(support_trajectories)} support examples"
        )

        # Compute task embedding
        task_embedding = self._compute_task_embedding(
            task_context,
            support_trajectories
        )

        self.task_embeddings[task_context.task_id] = task_embedding

        if TORCH_AVAILABLE:
            adapted_params = await self._adapt_with_gradient_descent(
                task_context,
                support_trajectories,
                current_policy_params,
                task_embedding
            )
        else:
            adapted_params = await self._adapt_with_heuristics(
                task_context,
                support_trajectories,
                current_policy_params,
                task_embedding
            )

        logger.info(f"Task adaptation complete for '{task_context.task_id}'")

        return adapted_params

    def _compute_task_embedding(
        self,
        task_context: TaskContext,
        trajectories: List[DecisionTrajectory]
    ) -> np.ndarray:
        """
        Compute embedding vector for a task.

        Embedding captures:
        - Required tools
        - Feature distribution
        - Success patterns
        """

        embedding_dim = 128
        embedding = np.zeros(embedding_dim)

        # Tool distribution (first 32 dims)
        tool_names = ["calculator", "text_processor", "web_search", "data_analyzer"]
        for i, tool in enumerate(tool_names[:32]):
            if tool in task_context.required_tools:
                embedding[i] = 1.0

        # Feature statistics (next 32 dims)
        if trajectories:
            # Average quality
            avg_quality = np.mean([t.quality_score for t in trajectories])
            embedding[32] = avg_quality

            # Success rate
            success_rate = np.mean([float(t.outcome.success) for t in trajectories])
            embedding[33] = success_rate

            # Average execution time (normalized)
            avg_time = np.mean([t.outcome.execution_time for t in trajectories])
            embedding[34] = min(1.0, avg_time / 10.0)

        # Task type encoding (one-hot in dims 64-96)
        task_types = ["math", "text", "data", "web", "code", "analysis"]
        for i, tt in enumerate(task_types):
            if task_context.task_type == tt:
                embedding[64 + i] = 1.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def _adapt_with_gradient_descent(
        self,
        task_context: TaskContext,
        support_trajectories: List[DecisionTrajectory],
        current_params: Dict[str, Any],
        task_embedding: np.ndarray
    ) -> Dict[str, Any]:
        """
        Adapt using gradient-based meta-learning (MAML-style).
        """

        # Simplified implementation - in production, this would:
        # 1. Clone current policy
        # 2. Perform inner loop adaptation on support set
        # 3. Compute adapted parameters
        # 4. Update meta-parameters

        adapted_params = current_params.copy()

        # Simulate adaptation
        # In production: actual gradient descent on support trajectories
        adaptation_strength = len(support_trajectories) / 20.0  # More data = stronger adaptation

        # Adjust tool preferences based on task
        if "tool_preferences" in adapted_params:
            for tool in task_context.required_tools:
                if tool in adapted_params["tool_preferences"]:
                    # Boost preferred tools
                    adapted_params["tool_preferences"][tool] *= (1.0 + adaptation_strength * 0.2)

        # Adjust exploration based on task familiarity
        if task_context.task_id in self.task_history:
            # Familiar task - reduce exploration
            if "epsilon" in adapted_params:
                adapted_params["epsilon"] *= 0.8
        else:
            # New task - increase exploration
            if "epsilon" in adapted_params:
                adapted_params["epsilon"] = min(0.3, adapted_params["epsilon"] * 1.2)

        return adapted_params

    async def _adapt_with_heuristics(
        self,
        task_context: TaskContext,
        support_trajectories: List[DecisionTrajectory],
        current_params: Dict[str, Any],
        task_embedding: np.ndarray
    ) -> Dict[str, Any]:
        """
        Adapt using heuristic rules (fallback when PyTorch not available).
        """

        adapted_params = current_params.copy()

        # Analyze support trajectories
        if support_trajectories:
            # Find best-performing tools
            tool_performance = {}
            for traj in support_trajectories:
                if hasattr(traj.action_plan, 'selected_tool'):
                    tool = traj.action_plan.selected_tool
                    if tool not in tool_performance:
                        tool_performance[tool] = []
                    tool_performance[tool].append(traj.quality_score)

            # Adjust preferences for well-performing tools
            if "tool_preferences" in adapted_params:
                for tool, scores in tool_performance.items():
                    avg_score = np.mean(scores)
                    if avg_score > 0.7:
                        # Boost good tools
                        if tool in adapted_params["tool_preferences"]:
                            adapted_params["tool_preferences"][tool] *= 1.3
                    elif avg_score < 0.3:
                        # Reduce bad tools
                        if tool in adapted_params["tool_preferences"]:
                            adapted_params["tool_preferences"][tool] *= 0.7

        return adapted_params

    async def meta_update(
        self,
        task_adaptations: List[Adaptation]
    ) -> Dict[str, Any]:
        """
        Meta-update using multiple task adaptations.

        This is the outer loop of meta-learning that updates
        the initial parameters to make future adaptations faster.
        """

        if len(task_adaptations) < self.meta_batch_size:
            logger.warning(
                f"Not enough adaptations for meta-update "
                f"({len(task_adaptations)} < {self.meta_batch_size})"
            )
            return {}

        logger.info(f"Performing meta-update with {len(task_adaptations)} task adaptations")

        # Compute average adaptation direction
        meta_gradient = {}

        for adaptation in task_adaptations:
            # Weight by improvement
            weight = max(0.0, adaptation.improvement)

            for param_name, param_value in adaptation.adapted_parameters.items():
                if param_name not in meta_gradient:
                    meta_gradient[param_name] = 0.0

                # Simple weighted average for meta-update
                if isinstance(param_value, (int, float)):
                    meta_gradient[param_name] += weight * param_value

        # Normalize
        total_weight = sum(max(0.0, a.improvement) for a in task_adaptations)
        if total_weight > 0:
            meta_gradient = {
                k: v / total_weight
                for k, v in meta_gradient.items()
            }

        logger.info("Meta-update complete")

        return meta_gradient

    def find_similar_tasks(
        self,
        task_context: TaskContext,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar tasks based on task embeddings.

        Useful for transfer learning.
        """

        if task_context.task_id not in self.task_embeddings:
            return []

        query_embedding = self.task_embeddings[task_context.task_id]

        similarities = []
        for task_id, embedding in self.task_embeddings.items():
            if task_id != task_context.task_id:
                # Cosine similarity
                similarity = np.dot(query_embedding, embedding)
                similarities.append((task_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class HyperparameterOptimizer:
    """
    Automatic hyperparameter optimization using Bayesian optimization.
    """

    def __init__(
        self,
        search_space: Dict[str, Tuple[float, float]]
    ):
        """
        Args:
            search_space: Dict mapping parameter names to (min, max) ranges
        """
        self.search_space = search_space
        self.trials: List[Dict[str, Any]] = []

    async def suggest_hyperparameters(self) -> Dict[str, float]:
        """
        Suggest hyperparameters to try next.

        Uses:
        - Random search if < 5 trials
        - Bayesian optimization if >= 5 trials
        """

        if len(self.trials) < 5:
            # Random search
            return self._random_sample()
        else:
            # Bayesian optimization (simplified)
            return self._bayesian_sample()

    def _random_sample(self) -> Dict[str, float]:
        """Sample hyperparameters randomly."""

        sample = {}
        for param, (min_val, max_val) in self.search_space.items():
            sample[param] = np.random.uniform(min_val, max_val)

        return sample

    def _bayesian_sample(self) -> Dict[str, float]:
        """
        Sample using Bayesian optimization.

        Simplified implementation - in production, use GPyOpt or similar.
        """

        # Find best previous trial
        best_trial = max(self.trials, key=lambda t: t.get("performance", 0.0))
        best_params = best_trial["hyperparameters"]

        # Sample around best with some exploration
        sample = {}
        for param, (min_val, max_val) in self.search_space.items():
            # Gaussian around best value
            best_val = best_params.get(param, (min_val + max_val) / 2)
            std = (max_val - min_val) / 10.0

            sampled = np.random.normal(best_val, std)
            sample[param] = np.clip(sampled, min_val, max_val)

        return sample

    def record_trial(
        self,
        hyperparameters: Dict[str, float],
        performance: float
    ):
        """Record trial result."""

        self.trials.append({
            "hyperparameters": hyperparameters,
            "performance": performance,
            "trial_number": len(self.trials)
        })

    def get_best_hyperparameters(self) -> Dict[str, float]:
        """Get best hyperparameters found so far."""

        if not self.trials:
            return {}

        best_trial = max(self.trials, key=lambda t: t["performance"])
        return best_trial["hyperparameters"]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Meta-Learning Module Demo")
        print("="*80 + "\n")

        # Create meta-learner
        meta_learner = MetaLearner(
            inner_lr=0.01,
            outer_lr=0.001,
            adaptation_steps=5
        )

        # Define tasks
        math_task = TaskContext(
            task_id="math_problems",
            task_type="math",
            description="Solve mathematical problems",
            required_tools=["calculator"],
            success_criteria={"accuracy": 0.95}
        )

        text_task = TaskContext(
            task_id="text_analysis",
            task_type="text",
            description="Analyze and process text",
            required_tools=["text_processor"],
            success_criteria={"quality": 0.8}
        )

        # Simulate adaptation to math task
        print("📚 Adapting to math task...\n")

        support_trajectories = []
        for i in range(10):
            from . import DecisionTrajectory, Outcome, Feedback

            traj = DecisionTrajectory(
                trajectory_id=f"math_{i}",
                query={"text": f"Math problem {i}"},
                features={"motifs": []},
                context={"shards": []},
                action_plan={"selected_tool": "calculator"},
                outcome=Outcome(
                    success=True,
                    execution_time=np.random.uniform(0.5, 2.0),
                    user_satisfaction=0.9
                ),
                feedback=Feedback(rating=4.5, helpful=True)
            )
            support_trajectories.append(traj)

        current_params = {
            "tool_preferences": {
                "calculator": 1.0,
                "text_processor": 1.0,
                "web_search": 1.0
            },
            "epsilon": 0.1
        }

        adapted_params = await meta_learner.adapt_to_task(
            math_task,
            support_trajectories,
            current_params
        )

        print("✅ Adaptation complete!")
        print(f"\nAdapted parameters:")
        for k, v in adapted_params.items():
            print(f"  {k}: {v}")

        # Find similar tasks
        print("\n\n🔍 Finding similar tasks...")

        # First adapt to text task
        await meta_learner.adapt_to_task(
            text_task,
            support_trajectories[:5],  # Reuse trajectories for demo
            current_params
        )

        similar = meta_learner.find_similar_tasks(math_task, top_k=3)
        print(f"\nTasks similar to '{math_task.task_id}':")
        for task_id, similarity in similar:
            print(f"  • {task_id}: similarity = {similarity:.3f}")

        # Hyperparameter optimization
        print("\n\n⚙️  Hyperparameter Optimization Demo\n")

        optimizer = HyperparameterOptimizer({
            "learning_rate": (0.0001, 0.01),
            "epsilon": (0.05, 0.3),
            "temperature": (0.5, 2.0)
        })

        for i in range(10):
            # Suggest hyperparameters
            suggested = await optimizer.suggest_hyperparameters()

            # Simulate performance (normally would train and evaluate)
            performance = np.random.random()

            # Record trial
            optimizer.record_trial(suggested, performance)

            print(f"Trial {i+1}: performance = {performance:.3f}")

        best_params = optimizer.get_best_hyperparameters()
        print(f"\n✅ Best hyperparameters found:")
        for k, v in best_params.items():
            print(f"  {k}: {v:.4f}")

        print("\n" + "="*80)
        print("✅ Meta-Learning Demo Complete!")
        print("="*80)

    asyncio.run(demo())
