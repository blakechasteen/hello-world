"""
Reflection Learning Integration
================================
Integrates reflection learning components with the HoloLoom orchestrator.

This module bridges:
- ReflectionEngine (analyzes past decisions)
- MetaLearner (rapid task adaptation)
- ExperienceReplayBuffer (prioritized memory)
- Orchestrator (main query processing)

Philosophy:
Learning happens at the intersection of action and reflection.
This module creates a continuous improvement loop where every
decision becomes an opportunity for growth.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from . import (
    ReflectionEngine,
    DecisionTrajectory,
    ReflectionInsights,
    PolicyUpdate,
    Outcome,
    Feedback
)
from .metalearning import MetaLearner, TaskContext, Adaptation
from .experience_replay import ExperienceReplayBuffer


@dataclass
class LearningConfig:
    """Configuration for reflection learning."""

    # Reflection settings
    reflection_interval: int = 50  # Analyze every N decisions
    min_trajectories_for_reflection: int = 10

    # Meta-learning settings
    enable_meta_learning: bool = True
    adaptation_steps: int = 5
    meta_batch_size: int = 4

    # Experience replay settings
    replay_buffer_size: int = 10000
    replay_batch_size: int = 32
    prioritized_sampling: bool = True

    # Policy update settings
    auto_update_policy: bool = True
    min_improvement_threshold: float = 0.1  # Only apply updates with >10% improvement

    # Logging
    verbose: bool = True


class ReflectionLearningSystem:
    """
    Unified reflection learning system.

    Coordinates reflection engine, meta-learner, and experience replay
    to enable continuous improvement.
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None
    ):
        self.config = config or LearningConfig()

        # Core components
        self.reflection_engine = ReflectionEngine()
        self.meta_learner = MetaLearner(
            adaptation_steps=self.config.adaptation_steps,
            meta_batch_size=self.config.meta_batch_size
        )
        self.replay_buffer = ExperienceReplayBuffer(
            max_size=self.config.replay_buffer_size
        )

        # State
        self.decision_count = 0
        self.total_improvements = 0
        self.last_reflection_time = time.time()

        # Policy parameters (managed externally)
        self.current_policy_params: Dict[str, Any] = {}

        # Active task contexts
        self.active_tasks: Dict[str, TaskContext] = {}

    async def record_decision(
        self,
        query: Dict[str, Any],
        features: Dict[str, Any],
        context: Dict[str, Any],
        action_plan: Dict[str, Any],
        outcome: Outcome,
        feedback: Optional[Feedback] = None
    ) -> DecisionTrajectory:
        """
        Record a decision for learning.

        Args:
            query: The input query
            features: Extracted features
            context: Retrieved context
            action_plan: Chosen action plan
            outcome: Execution outcome
            feedback: Optional user feedback

        Returns:
            The recorded trajectory
        """

        # Create trajectory
        trajectory = DecisionTrajectory(
            trajectory_id=f"traj_{self.decision_count}",
            query=query,
            features=features,
            context=context,
            action_plan=action_plan,
            outcome=outcome,
            feedback=feedback
        )

        # Record in reflection engine
        self.reflection_engine.record_trajectory(trajectory)

        # Add to replay buffer
        self.replay_buffer.add(trajectory)

        self.decision_count += 1

        # Periodic reflection
        if self.decision_count % self.config.reflection_interval == 0:
            await self._trigger_reflection()

        if self.config.verbose:
            success_mark = "✅" if outcome.success else "❌"
            logger.info(
                f"{success_mark} Decision {self.decision_count} recorded "
                f"(quality={trajectory.quality_score:.2f})"
            )

        return trajectory

    async def adapt_to_task(
        self,
        task_context: TaskContext,
        support_examples: Optional[List[DecisionTrajectory]] = None
    ) -> Dict[str, Any]:
        """
        Adapt policy to a new task.

        Args:
            task_context: Description of the task
            support_examples: Optional few-shot examples for the task

        Returns:
            Adapted policy parameters
        """

        if not self.config.enable_meta_learning:
            logger.warning("Meta-learning disabled")
            return self.current_policy_params

        # Get support examples from buffer if not provided
        if support_examples is None:
            # Find similar past experiences
            all_trajectories = self.reflection_engine.get_recent_trajectories(100)

            # Simple filtering - in production, use similarity search
            support_examples = all_trajectories[:5]

        if len(support_examples) == 0:
            logger.warning(f"No support examples for task '{task_context.task_id}'")
            return self.current_policy_params

        # Adapt
        adapted_params = await self.meta_learner.adapt_to_task(
            task_context,
            support_examples,
            self.current_policy_params
        )

        # Store task context
        self.active_tasks[task_context.task_id] = task_context

        logger.info(f"Adapted to task '{task_context.task_id}' with {len(support_examples)} examples")

        return adapted_params

    async def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get current learning insights and statistics.

        Returns:
            Dict with reflection insights, buffer stats, and metrics
        """

        # Run reflection
        insights = await self.reflection_engine.analyze_decisions(recent_only=True)

        # Get buffer stats
        buffer_stats = self.replay_buffer.get_stats()

        # Get tool stats
        tool_stats = self.reflection_engine.get_tool_stats()

        # Compile metrics
        metrics = {
            "total_decisions": self.decision_count,
            "total_improvements": self.total_improvements,
            "reflection_insights": {
                "num_trajectories": insights.num_trajectories,
                "avg_quality": insights.avg_quality,
                "success_rate": insights.success_rate,
                "num_success_patterns": len(insights.success_patterns),
                "num_failure_patterns": len(insights.failure_patterns),
                "recommendations": [r.description for r in insights.recommendations]
            },
            "replay_buffer": {
                "total_entries": buffer_stats.total_entries,
                "success_rate": buffer_stats.success_rate,
                "avg_priority": buffer_stats.avg_priority,
                "novel_count": buffer_stats.novel_count,
                "unique_tools": buffer_stats.unique_tools_used,
                "query_diversity": buffer_stats.query_diversity
            },
            "tool_performance": {
                tool: {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "success_rate": stats["success_rate"],
                    "avg_quality": stats["avg_quality"]
                }
                for tool, stats in tool_stats.items()
            }
        }

        return metrics

    async def sample_for_training(
        self,
        batch_size: Optional[int] = None,
        strategy: str = "prioritized"
    ) -> List[DecisionTrajectory]:
        """
        Sample trajectories for training.

        Args:
            batch_size: Number of trajectories (default: config.replay_batch_size)
            strategy: Sampling strategy (prioritized, uniform, diverse, etc.)

        Returns:
            List of sampled trajectories
        """

        batch_size = batch_size or self.config.replay_batch_size

        if self.config.prioritized_sampling:
            strategy = "prioritized"

        trajectories, priorities, weights = self.replay_buffer.sample(
            batch_size,
            strategy=strategy
        )

        return trajectories

    async def sample_contrastive_pairs(
        self,
        n_pairs: int = 10
    ) -> List[tuple]:
        """
        Sample contrastive pairs for learning.

        Returns:
            List of (success, failure) trajectory pairs
        """

        return self.replay_buffer.sample_contrastive_pairs(n_pairs)

    def update_policy_params(self, new_params: Dict[str, Any]):
        """
        Update the current policy parameters.

        Call this after training/updating the policy.
        """

        self.current_policy_params = new_params.copy()
        logger.info("Policy parameters updated")

    async def _trigger_reflection(self):
        """
        Trigger periodic reflection and learning.

        This is called automatically after every N decisions.
        """

        if len(self.reflection_engine.trajectories) < self.config.min_trajectories_for_reflection:
            return

        logger.info("🔍 Triggering reflection analysis...")

        # Analyze decisions
        insights = await self.reflection_engine.analyze_decisions(recent_only=True)

        # Generate improvements
        improvements = await self.reflection_engine.generate_improvements(insights)

        # Log insights
        if self.config.verbose:
            logger.info(f"Reflection Analysis:")
            logger.info(f"  Trajectories: {insights.num_trajectories}")
            logger.info(f"  Avg Quality: {insights.avg_quality:.2f}")
            logger.info(f"  Success Rate: {insights.success_rate:.1%}")
            logger.info(f"  Success Patterns: {len(insights.success_patterns)}")
            logger.info(f"  Failure Patterns: {len(insights.failure_patterns)}")
            logger.info(f"  Improvements: {len(improvements)}")

        # Apply improvements if configured
        if self.config.auto_update_policy and improvements:
            await self._apply_improvements(improvements)

        self.last_reflection_time = time.time()

    async def _apply_improvements(self, improvements: List[PolicyUpdate]):
        """
        Apply policy improvements.

        Note: This is a simplified version. In production, you would:
        1. Validate improvements
        2. A/B test changes
        3. Gradually roll out updates
        4. Monitor for regressions
        """

        significant_improvements = [
            imp for imp in improvements
            if imp.expected_improvement >= self.config.min_improvement_threshold
        ]

        if not significant_improvements:
            logger.info("No significant improvements to apply")
            return

        logger.info(f"Applying {len(significant_improvements)} policy improvements...")

        for improvement in significant_improvements:
            # Apply the update
            if improvement.parameter_name in self.current_policy_params:
                old_value = self.current_policy_params[improvement.parameter_name]
                self.current_policy_params[improvement.parameter_name] = improvement.new_value

                logger.info(
                    f"  Updated {improvement.parameter_name}: "
                    f"{old_value} → {improvement.new_value} "
                    f"(expected improvement: {improvement.expected_improvement:.1%})"
                )

                self.total_improvements += 1

    async def export_learnings(self, filepath: str):
        """
        Export learned knowledge for analysis or transfer.

        Args:
            filepath: Path to export to (JSON format)
        """

        import json

        # Collect all learning data
        data = {
            "config": {
                "reflection_interval": self.config.reflection_interval,
                "replay_buffer_size": self.config.replay_buffer_size,
            },
            "statistics": {
                "decision_count": self.decision_count,
                "total_improvements": self.total_improvements,
            },
            "tool_stats": self.reflection_engine.get_tool_stats(),
            "insights": await self.get_learning_insights(),
            "task_embeddings": {
                task_id: embedding.tolist()
                for task_id, embedding in self.meta_learner.task_embeddings.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported learnings to {filepath}")


# ============================================================================
# Orchestrator Integration Helper
# ============================================================================

class ReflectiveOrchestrator:
    """
    Wrapper that adds reflection learning to any orchestrator.

    Usage:
        original_orchestrator = Orchestrator(...)
        reflective_orchestrator = ReflectiveOrchestrator(original_orchestrator)

        # Process queries as normal
        result = await reflective_orchestrator.process(query)

        # Learning happens automatically in the background
    """

    def __init__(
        self,
        base_orchestrator: Any,
        learning_config: Optional[LearningConfig] = None
    ):
        self.base = base_orchestrator
        self.learning_system = ReflectionLearningSystem(learning_config)

        # Initialize policy params from orchestrator
        if hasattr(base_orchestrator, 'policy') and hasattr(base_orchestrator.policy, 'get_params'):
            self.learning_system.current_policy_params = base_orchestrator.policy.get_params()

    async def process(
        self,
        query: Dict[str, Any],
        collect_feedback: bool = False
    ) -> Dict[str, Any]:
        """
        Process query with automatic reflection learning.

        Args:
            query: Input query
            collect_feedback: Whether to collect user feedback after execution

        Returns:
            Result from base orchestrator
        """

        # Extract features (if available)
        features = {}
        if hasattr(self.base, 'extract_features'):
            features = await self.base.extract_features(query)

        # Retrieve context (if available)
        context = {}
        if hasattr(self.base, 'retrieve_context'):
            context = await self.base.retrieve_context(query, features)

        # Process with base orchestrator
        start_time = time.time()
        try:
            result = await self.base.process(query)
            execution_time = time.time() - start_time

            outcome = Outcome(
                success=True,
                execution_time=execution_time,
                user_satisfaction=0.8  # Default - can be updated with feedback
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query processing failed: {e}")

            outcome = Outcome(
                success=False,
                execution_time=execution_time,
                user_satisfaction=0.0
            )

            result = {"error": str(e)}

        # Extract action plan from result
        action_plan = result.get("action_plan", {})

        # Collect feedback if requested
        feedback = None
        if collect_feedback:
            feedback = await self._collect_feedback(query, result)

        # Record decision for learning
        await self.learning_system.record_decision(
            query=query,
            features=features,
            context=context,
            action_plan=action_plan,
            outcome=outcome,
            feedback=feedback
        )

        return result

    async def _collect_feedback(
        self,
        query: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Feedback:
        """
        Collect user feedback.

        In production, this would:
        - Show result to user
        - Ask for rating
        - Collect additional feedback
        """

        # Simplified - in production, implement actual feedback collection
        return Feedback(
            rating=4.0,
            helpful=True,
            comment="Auto-generated feedback"
        )

    async def get_insights(self) -> Dict[str, Any]:
        """Get current learning insights."""
        return await self.learning_system.get_learning_insights()

    async def adapt_to_task(self, task_context: TaskContext) -> Dict[str, Any]:
        """Adapt to a new task."""
        return await self.learning_system.adapt_to_task(task_context)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("Reflection Learning Integration Demo")
        print("="*80 + "\n")

        # Create learning system
        config = LearningConfig(
            reflection_interval=10,
            verbose=True
        )

        learning_system = ReflectionLearningSystem(config)

        # Simulate decision processing
        print("🔄 Processing decisions...\n")

        import random

        tools = ["web_search", "calculator", "text_analyzer", "data_processor"]

        for i in range(25):
            # Create query
            query = {"text": f"Query about topic {i % 5}"}

            # Simulate processing
            selected_tool = random.choice(tools)
            is_success = random.random() > 0.25  # 75% success rate

            outcome = Outcome(
                success=is_success,
                execution_time=random.uniform(0.5, 3.0),
                user_satisfaction=random.random() if is_success else random.random() * 0.5
            )

            feedback = None
            if random.random() > 0.7:  # 30% of queries get feedback
                feedback = Feedback(
                    rating=random.randint(4, 5) if is_success else random.randint(1, 3),
                    helpful=is_success
                )

            # Record decision
            await learning_system.record_decision(
                query=query,
                features={"motifs": []},
                context={"shards": []},
                action_plan={"selected_tool": selected_tool},
                outcome=outcome,
                feedback=feedback
            )

        # Get insights
        print("\n\n📊 Learning Insights:\n")

        insights = await learning_system.get_learning_insights()

        print("Reflection Insights:")
        print(f"  Total Decisions: {insights['total_decisions']}")
        print(f"  Avg Quality: {insights['reflection_insights']['avg_quality']:.2f}")
        print(f"  Success Rate: {insights['reflection_insights']['success_rate']:.1%}")
        print(f"  Success Patterns: {insights['reflection_insights']['num_success_patterns']}")
        print(f"  Failure Patterns: {insights['reflection_insights']['num_failure_patterns']}")

        print("\nReplay Buffer:")
        print(f"  Total Entries: {insights['replay_buffer']['total_entries']}")
        print(f"  Success Rate: {insights['replay_buffer']['success_rate']:.1%}")
        print(f"  Unique Tools: {insights['replay_buffer']['unique_tools']}")
        print(f"  Query Diversity: {insights['replay_buffer']['query_diversity']:.1%}")

        print("\nTool Performance:")
        for tool, stats in insights['tool_performance'].items():
            print(f"  {tool}:")
            print(f"    Attempts: {stats['attempts']}")
            print(f"    Success Rate: {stats['success_rate']:.1%}")
            print(f"    Avg Quality: {stats['avg_quality']:.2f}")

        if insights['reflection_insights']['recommendations']:
            print("\nRecommendations:")
            for rec in insights['reflection_insights']['recommendations'][:5]:
                print(f"  • {rec}")

        # Sample for training
        print("\n\n🎓 Sampling for Training...\n")

        trajectories = await learning_system.sample_for_training(batch_size=8)
        print(f"Sampled {len(trajectories)} trajectories for training")

        # Contrastive pairs
        pairs = await learning_system.sample_contrastive_pairs(n_pairs=3)
        print(f"Sampled {len(pairs)} contrastive pairs for learning")

        # Task adaptation
        print("\n\n🎯 Task Adaptation Demo...\n")

        task_context = TaskContext(
            task_id="web_search_optimization",
            task_type="web",
            description="Optimize web search queries",
            required_tools=["web_search"]
        )

        adapted_params = await learning_system.adapt_to_task(task_context)
        print(f"Adapted to task: {task_context.task_id}")

        # Export learnings
        print("\n\n💾 Exporting Learnings...\n")

        await learning_system.export_learnings("/tmp/hololoom_learnings.json")
        print("Learnings exported to /tmp/hololoom_learnings.json")

        print("\n" + "="*80)
        print("✅ Reflection Learning Integration Demo Complete!")
        print("="*80)

    asyncio.run(demo())
