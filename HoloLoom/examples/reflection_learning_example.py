"""
Reflection Learning Example
============================
Complete example demonstrating HoloLoom's reflection learning capabilities.

This example shows:
1. Basic reflection learning workflow
2. Task adaptation with meta-learning
3. Experience replay for training
4. Continuous improvement loop
5. Integration with orchestrator

Run with:
    PYTHONPATH=. python HoloLoom/examples/reflection_learning_example.py
"""

import asyncio
import random
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from HoloLoom.reflection import (
    ReflectionEngine,
    DecisionTrajectory,
    Outcome,
    Feedback
)
from HoloLoom.reflection.metalearning import MetaLearner, TaskContext
from HoloLoom.reflection.experience_replay import ExperienceReplayBuffer
from HoloLoom.reflection.integration import (
    ReflectionLearningSystem,
    LearningConfig
)


# ============================================================================
# Example 1: Basic Reflection Learning
# ============================================================================

async def example_basic_reflection():
    """
    Basic example: Record decisions and analyze patterns.
    """

    print("\n" + "="*80)
    print("Example 1: Basic Reflection Learning")
    print("="*80 + "\n")

    # Create reflection engine
    engine = ReflectionEngine()

    # Simulate a series of decisions
    tools = ["web_search", "calculator", "text_processor", "code_analyzer"]

    print("Recording 30 decision trajectories...\n")

    for i in range(30):
        # Simulate decision
        tool = random.choice(tools)
        is_success = random.random() > 0.3  # 70% success rate

        # Create trajectory
        trajectory = DecisionTrajectory(
            trajectory_id=f"decision_{i}",
            query={"text": f"User query {i}", "topic": f"topic_{i % 5}"},
            features={"motifs": [f"motif_{i % 3}"]},
            context={"shards": []},
            action_plan={"selected_tool": tool, "reasoning": "Based on query analysis"},
            outcome=Outcome(
                success=is_success,
                execution_time=random.uniform(0.5, 3.0),
                user_satisfaction=random.random() if is_success else random.random() * 0.5
            ),
            feedback=Feedback(
                rating=random.randint(4, 5) if is_success else random.randint(1, 3),
                helpful=is_success,
                comment="Good result" if is_success else "Could be better"
            ) if random.random() > 0.5 else None
        )

        engine.record_trajectory(trajectory)

    # Analyze decisions
    print("Analyzing decision patterns...\n")
    insights = await engine.analyze_decisions()

    print(f"📊 Analysis Results:")
    print(f"  Trajectories analyzed: {insights.num_trajectories}")
    print(f"  Average quality: {insights.avg_quality:.2f}")
    print(f"  Success rate: {insights.success_rate:.1%}")
    print(f"  Success patterns found: {len(insights.success_patterns)}")
    print(f"  Failure patterns found: {len(insights.failure_patterns)}")

    # Show patterns
    if insights.success_patterns:
        print(f"\n✅ Success Patterns:")
        for pattern in insights.success_patterns[:3]:
            print(f"  • {pattern.description} (confidence: {pattern.confidence:.2f})")

    if insights.failure_patterns:
        print(f"\n❌ Failure Patterns:")
        for pattern in insights.failure_patterns[:3]:
            print(f"  • {pattern.description} (confidence: {pattern.confidence:.2f})")

    # Show recommendations
    if insights.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in insights.recommendations[:5]:
            print(f"  • {rec.description} (priority: {rec.priority})")

    # Generate improvements
    print(f"\n🔧 Generating policy improvements...\n")
    improvements = await engine.generate_improvements(insights)

    if improvements:
        print(f"Generated {len(improvements)} policy updates:")
        for imp in improvements[:3]:
            print(f"  • {imp.parameter_name}: {imp.old_value} → {imp.new_value}")
            print(f"    Expected improvement: {imp.expected_improvement:.1%}")
            print(f"    Rationale: {imp.rationale}")

    # Tool statistics
    print(f"\n🛠️  Tool Performance:")
    tool_stats = engine.get_tool_stats()
    for tool, stats in sorted(tool_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True):
        print(f"  {tool}:")
        print(f"    Attempts: {stats['attempts']}")
        print(f"    Success rate: {stats['success_rate']:.1%}")
        print(f"    Avg quality: {stats['avg_quality']:.2f}")


# ============================================================================
# Example 2: Meta-Learning for Task Adaptation
# ============================================================================

async def example_meta_learning():
    """
    Meta-learning example: Adapt quickly to new tasks.
    """

    print("\n" + "="*80)
    print("Example 2: Meta-Learning Task Adaptation")
    print("="*80 + "\n")

    # Create meta-learner
    meta_learner = MetaLearner(
        inner_lr=0.01,
        outer_lr=0.001,
        adaptation_steps=5
    )

    # Define several tasks
    tasks = [
        TaskContext(
            task_id="math_problems",
            task_type="math",
            description="Solve mathematical equations",
            required_tools=["calculator"],
            success_criteria={"accuracy": 0.95}
        ),
        TaskContext(
            task_id="web_research",
            task_type="web",
            description="Search and summarize web content",
            required_tools=["web_search", "text_processor"],
            success_criteria={"relevance": 0.8}
        ),
        TaskContext(
            task_id="code_analysis",
            task_type="code",
            description="Analyze code quality and bugs",
            required_tools=["code_analyzer"],
            success_criteria={"completeness": 0.9}
        )
    ]

    # Simulate having some experience with each task
    print("Building task experience...\n")

    current_policy_params = {
        "tool_preferences": {
            "calculator": 1.0,
            "web_search": 1.0,
            "text_processor": 1.0,
            "code_analyzer": 1.0
        },
        "epsilon": 0.1,
        "temperature": 1.0
    }

    for task in tasks:
        # Create support trajectories for this task
        support_trajectories = []

        for i in range(10):
            tool = random.choice(task.required_tools)
            is_success = random.random() > 0.2  # 80% success

            trajectory = DecisionTrajectory(
                trajectory_id=f"{task.task_id}_{i}",
                query={"text": f"{task.description} example {i}"},
                features={"motifs": []},
                context={"shards": []},
                action_plan={"selected_tool": tool},
                outcome=Outcome(
                    success=is_success,
                    execution_time=random.uniform(0.5, 2.0),
                    user_satisfaction=random.random()
                ),
                feedback=None
            )

            support_trajectories.append(trajectory)

        # Adapt to task
        print(f"📚 Adapting to task: {task.task_id}")
        adapted_params = await meta_learner.adapt_to_task(
            task,
            support_trajectories,
            current_policy_params
        )

        print(f"  Required tools: {task.required_tools}")
        print(f"  Support examples: {len(support_trajectories)}")

        # Show adaptation
        if "tool_preferences" in adapted_params:
            print(f"  Adapted tool preferences:")
            for tool in task.required_tools:
                if tool in adapted_params["tool_preferences"]:
                    old_pref = current_policy_params["tool_preferences"].get(tool, 1.0)
                    new_pref = adapted_params["tool_preferences"][tool]
                    change = ((new_pref - old_pref) / old_pref) * 100
                    print(f"    {tool}: {old_pref:.2f} → {new_pref:.2f} ({change:+.1f}%)")

        print()

    # Find similar tasks
    print("\n🔍 Finding similar tasks...\n")

    similar_to_math = meta_learner.find_similar_tasks(tasks[0], top_k=2)
    print(f"Tasks similar to '{tasks[0].task_id}':")
    for task_id, similarity in similar_to_math:
        print(f"  • {task_id}: {similarity:.3f}")


# ============================================================================
# Example 3: Experience Replay for Training
# ============================================================================

async def example_experience_replay():
    """
    Experience replay example: Prioritized sampling for training.
    """

    print("\n" + "="*80)
    print("Example 3: Experience Replay")
    print("="*80 + "\n")

    # Create replay buffer
    buffer = ExperienceReplayBuffer(
        max_size=100,
        alpha=0.6,  # Prioritization strength
        beta=0.4   # Importance sampling
    )

    # Add diverse experiences
    print("Adding 50 experiences to buffer...\n")

    tools = ["tool_a", "tool_b", "tool_c", "tool_d"]

    for i in range(50):
        is_success = random.random() > 0.3
        tool = random.choice(tools)

        trajectory = DecisionTrajectory(
            trajectory_id=f"exp_{i}",
            query={"text": f"Query {i}", "category": f"cat_{i % 3}"},
            features={"motifs": []},
            context={"shards": []},
            action_plan={"selected_tool": tool},
            outcome=Outcome(
                success=is_success,
                execution_time=random.uniform(0.5, 3.0),
                user_satisfaction=random.random() if is_success else random.random() * 0.5
            ),
            feedback=Feedback(
                rating=random.randint(4, 5) if is_success else random.randint(1, 3),
                helpful=is_success
            ) if random.random() > 0.6 else None
        )

        buffer.add(trajectory)

    # Get buffer statistics
    stats = buffer.get_stats()
    print(f"📊 Buffer Statistics:")
    print(f"  Total entries: {stats.total_entries}")
    print(f"  Success rate: {stats.success_rate:.1%}")
    print(f"  Average priority: {stats.avg_priority:.3f}")
    print(f"  Novel experiences: {stats.novel_count}")
    print(f"  Unique tools: {stats.unique_tools_used}")
    print(f"  Query diversity: {stats.query_diversity:.1%}")

    # Prioritized sampling
    print(f"\n🎲 Prioritized Sampling (batch_size=10):\n")

    trajectories, priorities, weights = buffer.sample(
        batch_size=10,
        strategy="prioritized"
    )

    for i, (traj, priority, weight) in enumerate(zip(trajectories, priorities, weights)):
        status = "✅" if traj.outcome.success else "❌"
        print(f"  {i+1}. {traj.trajectory_id} {status}")
        print(f"     Priority: {priority:.3f}, Weight: {weight:.3f}")

    # Contrastive pairs
    print(f"\n🔄 Contrastive Pairs (for learning):\n")

    pairs = buffer.sample_contrastive_pairs(n_pairs=5)

    for i, (success, failure) in enumerate(pairs):
        print(f"  Pair {i+1}:")
        print(f"    ✅ Success: {success.trajectory_id} (quality={success.quality_score:.2f})")
        print(f"    ❌ Failure: {failure.trajectory_id} (quality={failure.quality_score:.2f})")

    # High-value experiences
    print(f"\n⭐ Top 5 High-Value Experiences:\n")

    high_value = buffer.get_high_value_experiences(top_k=5)

    for i, traj in enumerate(high_value):
        status = "✅" if traj.outcome.success else "❌"
        print(f"  {i+1}. {traj.trajectory_id} {status} (quality={traj.quality_score:.2f})")


# ============================================================================
# Example 4: Integrated Learning System
# ============================================================================

async def example_integrated_system():
    """
    Complete example: Full reflection learning system.
    """

    print("\n" + "="*80)
    print("Example 4: Integrated Learning System")
    print("="*80 + "\n")

    # Configure learning system
    config = LearningConfig(
        reflection_interval=15,  # Reflect every 15 decisions
        min_trajectories_for_reflection=10,
        enable_meta_learning=True,
        auto_update_policy=True,
        replay_buffer_size=1000,
        verbose=True
    )

    # Create learning system
    learning_system = ReflectionLearningSystem(config)

    # Simulate decision-making over time
    print("🔄 Simulating 40 decisions with continuous learning...\n")

    tools = ["web_search", "calculator", "text_analyzer", "code_debugger"]

    for i in range(40):
        # Create query
        topic = f"topic_{i % 5}"
        query = {"text": f"Question about {topic}", "complexity": random.choice(["simple", "moderate", "complex"])}

        # Simulate tool selection and execution
        selected_tool = random.choice(tools)

        # Success rate improves over time (simulating learning)
        success_prob = 0.6 + (i / 40) * 0.3  # 60% → 90%
        is_success = random.random() < success_prob

        outcome = Outcome(
            success=is_success,
            execution_time=random.uniform(0.5, 3.0),
            user_satisfaction=random.random() if is_success else random.random() * 0.5
        )

        # Occasional feedback
        feedback = None
        if random.random() > 0.7:
            feedback = Feedback(
                rating=random.randint(4, 5) if is_success else random.randint(1, 3),
                helpful=is_success,
                comment=f"Feedback for decision {i}"
            )

        # Record decision (triggers automatic reflection periodically)
        await learning_system.record_decision(
            query=query,
            features={"motifs": [], "complexity": query["complexity"]},
            context={"shards": []},
            action_plan={"selected_tool": selected_tool, "confidence": random.random()},
            outcome=outcome,
            feedback=feedback
        )

    # Get comprehensive insights
    print("\n\n📊 Comprehensive Learning Insights:\n")

    insights = await learning_system.get_learning_insights()

    print("Overall Statistics:")
    print(f"  Total decisions: {insights['total_decisions']}")
    print(f"  Total improvements applied: {insights['total_improvements']}")

    print("\nReflection Analysis:")
    print(f"  Trajectories: {insights['reflection_insights']['num_trajectories']}")
    print(f"  Avg quality: {insights['reflection_insights']['avg_quality']:.2f}")
    print(f"  Success rate: {insights['reflection_insights']['success_rate']:.1%}")
    print(f"  Success patterns: {insights['reflection_insights']['num_success_patterns']}")
    print(f"  Failure patterns: {insights['reflection_insights']['num_failure_patterns']}")

    print("\nMemory Buffer:")
    print(f"  Entries: {insights['replay_buffer']['total_entries']}")
    print(f"  Success rate: {insights['replay_buffer']['success_rate']:.1%}")
    print(f"  Novel experiences: {insights['replay_buffer']['novel_count']}")

    print("\nTool Performance:")
    for tool, stats in sorted(
        insights['tool_performance'].items(),
        key=lambda x: x[1]['success_rate'],
        reverse=True
    ):
        print(f"  {tool}:")
        print(f"    Success rate: {stats['success_rate']:.1%} ({stats['successes']}/{stats['attempts']})")
        print(f"    Avg quality: {stats['avg_quality']:.2f}")

    if insights['reflection_insights']['recommendations']:
        print("\nTop Recommendations:")
        for rec in insights['reflection_insights']['recommendations'][:3]:
            print(f"  • {rec}")

    # Sample for training
    print("\n\n🎓 Sampling experiences for training...\n")

    training_batch = await learning_system.sample_for_training(batch_size=10)
    print(f"Sampled {len(training_batch)} trajectories")

    success_count = sum(1 for t in training_batch if t.outcome.success)
    print(f"  Successes: {success_count}")
    print(f"  Failures: {len(training_batch) - success_count}")
    print(f"  Avg quality: {sum(t.quality_score for t in training_batch) / len(training_batch):.2f}")

    # Contrastive learning
    print("\n🔄 Sampling contrastive pairs...\n")

    pairs = await learning_system.sample_contrastive_pairs(n_pairs=5)
    print(f"Sampled {len(pairs)} contrastive pairs for learning")

    # Task adaptation
    print("\n\n🎯 Task Adaptation Example...\n")

    new_task = TaskContext(
        task_id="advanced_analysis",
        task_type="analysis",
        description="Perform advanced data analysis",
        required_tools=["text_analyzer", "calculator"],
        success_criteria={"accuracy": 0.9}
    )

    print(f"Adapting to new task: {new_task.task_id}")
    adapted_params = await learning_system.adapt_to_task(new_task)
    print(f"  Task type: {new_task.task_type}")
    print(f"  Required tools: {new_task.required_tools}")
    print(f"  Adaptation complete!")

    # Export learnings
    print("\n\n💾 Exporting learnings...\n")

    export_path = "/tmp/hololoom_reflection_learnings.json"
    await learning_system.export_learnings(export_path)
    print(f"Learnings exported to {export_path}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""

    print("\n" + "="*80)
    print("HoloLoom Reflection Learning Examples")
    print("="*80)

    # Run examples
    await example_basic_reflection()
    await example_meta_learning()
    await example_experience_replay()
    await example_integrated_system()

    print("\n" + "="*80)
    print("✅ All Examples Complete!")
    print("="*80 + "\n")

    print("Key Takeaways:")
    print("  • Reflection learning analyzes past decisions to find patterns")
    print("  • Meta-learning enables rapid adaptation to new tasks")
    print("  • Experience replay prioritizes valuable learning experiences")
    print("  • Integrated system provides continuous improvement")
    print("\nNext Steps:")
    print("  • Integrate with your HoloLoom orchestrator")
    print("  • Configure learning parameters for your use case")
    print("  • Monitor learning metrics over time")
    print("  • Export and analyze learnings periodically")


if __name__ == "__main__":
    asyncio.run(main())
