# Reflection Learning System

**Phase 4A: Intelligence Amplification**

The reflection learning system enables HoloLoom to learn from its own decisions and continuously improve performance over time. This document describes the architecture, components, and usage of the reflection learning system.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Integration Guide](#integration-guide)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)

---

## Overview

### What is Reflection Learning?

Reflection learning is a meta-cognitive process where the system:

1. **Records** every decision and its outcome
2. **Analyzes** patterns in successes and failures
3. **Learns** what works and what doesn't
4. **Adapts** policies to improve future decisions
5. **Generalizes** knowledge to new tasks

### Key Capabilities

- **Pattern Detection**: Automatically identifies success and failure patterns
- **Meta-Learning**: Rapid adaptation to new tasks with few examples
- **Experience Replay**: Prioritized sampling of valuable learning experiences
- **Continuous Improvement**: Automatic policy updates based on reflection
- **Provenance Tracking**: Complete lineage of every decision

### Benefits

- 📈 **Improves over time** - Performance increases with experience
- 🚀 **Faster adaptation** - Quickly adjusts to new task types
- 🎯 **Better decisions** - Learns from past mistakes
- 💡 **Actionable insights** - Provides clear recommendations
- 🔄 **Self-optimizing** - Reduces manual tuning

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Reflection Learning System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Reflection  │    │     Meta     │    │  Experience  │      │
│  │   Engine     │◄──►│   Learner    │◄──►│    Replay    │      │
│  │              │    │              │    │    Buffer    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                    │              │
│         └───────────────────┴────────────────────┘              │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │   Integration   │                          │
│                    │     Layer       │                          │
│                    └────────┬────────┘                          │
└─────────────────────────────┼──────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Orchestrator    │
                    │  (Query Processing)│
                    └───────────────────┘
```

### Data Flow

```
Query → Process → Outcome → Record → Analyze → Improve → Update Policy
  ↑                                                           │
  └───────────────────── Feedback Loop ──────────────────────┘
```

---

## Core Components

### 1. Reflection Engine

**Purpose**: Analyzes decision trajectories to extract insights and patterns.

**Location**: `HoloLoom/reflection/__init__.py`

**Key Classes**:

- `ReflectionEngine` - Main analysis engine
- `DecisionTrajectory` - Complete record of a decision cycle
- `ReflectionInsights` - Analysis results
- `PolicyUpdate` - Suggested improvements

**Capabilities**:

- Pattern detection (success/failure patterns)
- Feature importance analysis
- Quality scoring
- Tool performance tracking
- Recommendation generation

**Example**:

```python
from HoloLoom.reflection import ReflectionEngine, DecisionTrajectory, Outcome, Feedback

# Create engine
engine = ReflectionEngine()

# Record decision
trajectory = DecisionTrajectory(
    trajectory_id="decision_001",
    query={"text": "Calculate 2 + 2"},
    features={"motifs": ["arithmetic"]},
    context={"shards": []},
    action_plan={"selected_tool": "calculator"},
    outcome=Outcome(success=True, execution_time=0.5, user_satisfaction=0.9),
    feedback=Feedback(rating=5, helpful=True)
)

engine.record_trajectory(trajectory)

# Analyze after collecting many trajectories
insights = await engine.analyze_decisions()

print(f"Success rate: {insights.success_rate:.1%}")
print(f"Average quality: {insights.avg_quality:.2f}")
print(f"Recommendations: {len(insights.recommendations)}")
```

### 2. Meta-Learner

**Purpose**: Enables rapid adaptation to new tasks through meta-learning.

**Location**: `HoloLoom/reflection/metalearning.py`

**Key Classes**:

- `MetaLearner` - MAML-style meta-learning
- `TaskContext` - Task description and requirements
- `Adaptation` - Record of task adaptation
- `HyperparameterOptimizer` - Bayesian hyperparameter optimization

**Capabilities**:

- Few-shot task adaptation
- Task embedding learning
- Transfer learning across tasks
- Hyperparameter optimization

**Example**:

```python
from HoloLoom.reflection.metalearning import MetaLearner, TaskContext

# Create meta-learner
meta_learner = MetaLearner(inner_lr=0.01, outer_lr=0.001)

# Define new task
task = TaskContext(
    task_id="sentiment_analysis",
    task_type="text",
    description="Analyze sentiment of text",
    required_tools=["text_processor"],
    success_criteria={"accuracy": 0.85}
)

# Adapt with few examples
support_examples = [...]  # List of DecisionTrajectory
current_params = {"tool_preferences": {...}, "epsilon": 0.1}

adapted_params = await meta_learner.adapt_to_task(
    task,
    support_examples,
    current_params
)

print(f"Adapted parameters: {adapted_params}")
```

### 3. Experience Replay Buffer

**Purpose**: Stores and samples decision trajectories for efficient learning.

**Location**: `HoloLoom/reflection/experience_replay.py`

**Key Classes**:

- `ExperienceReplayBuffer` - Prioritized replay buffer
- `ExperienceEntry` - Single experience with metadata
- `ReplayStats` - Buffer statistics

**Capabilities**:

- Prioritized experience sampling
- Contrastive learning (success vs failure pairs)
- Diversity-based sampling
- Memory consolidation

**Example**:

```python
from HoloLoom.reflection.experience_replay import ExperienceReplayBuffer

# Create buffer
buffer = ExperienceReplayBuffer(
    max_size=10000,
    alpha=0.6,  # Prioritization strength
    beta=0.4    # Importance sampling
)

# Add experiences
for trajectory in trajectories:
    buffer.add(trajectory)

# Sample for training
trajectories, priorities, weights = buffer.sample(
    batch_size=32,
    strategy="prioritized"
)

# Get contrastive pairs
pairs = buffer.sample_contrastive_pairs(n_pairs=10)

# Statistics
stats = buffer.get_stats()
print(f"Success rate: {stats.success_rate:.1%}")
print(f"Query diversity: {stats.query_diversity:.1%}")
```

### 4. Integration Layer

**Purpose**: Connects reflection learning with the orchestrator.

**Location**: `HoloLoom/reflection/integration.py`

**Key Classes**:

- `ReflectionLearningSystem` - Unified learning system
- `LearningConfig` - Configuration
- `ReflectiveOrchestrator` - Orchestrator wrapper

**Capabilities**:

- Automatic decision recording
- Periodic reflection analysis
- Policy updates
- Task adaptation
- Learning export

**Example**:

```python
from HoloLoom.reflection.integration import ReflectionLearningSystem, LearningConfig

# Configure
config = LearningConfig(
    reflection_interval=50,  # Reflect every 50 decisions
    auto_update_policy=True,
    enable_meta_learning=True
)

# Create system
learning_system = ReflectionLearningSystem(config)

# Record decision
await learning_system.record_decision(
    query=query,
    features=features,
    context=context,
    action_plan=action_plan,
    outcome=outcome,
    feedback=feedback
)

# Get insights
insights = await learning_system.get_learning_insights()

# Sample for training
training_batch = await learning_system.sample_for_training(batch_size=32)
```

---

## Integration Guide

### Option 1: Wrap Existing Orchestrator

The easiest way to add reflection learning to your existing orchestrator:

```python
from HoloLoom.orchestrator import Orchestrator
from HoloLoom.reflection.integration import ReflectiveOrchestrator, LearningConfig

# Create base orchestrator
base_orchestrator = Orchestrator(config)

# Wrap with reflection learning
config = LearningConfig(
    reflection_interval=25,
    auto_update_policy=True,
    verbose=True
)

reflective_orchestrator = ReflectiveOrchestrator(
    base_orchestrator,
    config
)

# Use as normal - learning happens automatically
result = await reflective_orchestrator.process(query)

# Get learning insights
insights = await reflective_orchestrator.get_insights()
```

### Option 2: Manual Integration

For more control, integrate components directly:

```python
from HoloLoom.reflection import ReflectionEngine, Outcome, Feedback
from HoloLoom.reflection.integration import ReflectionLearningSystem, LearningConfig

# Create learning system
learning_system = ReflectionLearningSystem(LearningConfig())

# In your orchestrator's process method:
async def process(self, query):
    # 1. Extract features
    features = await self.extract_features(query)

    # 2. Retrieve context
    context = await self.retrieve_context(query, features)

    # 3. Make decision
    action_plan = await self.policy.decide(features, context)

    # 4. Execute
    start = time.time()
    try:
        result = await self.execute(action_plan)
        outcome = Outcome(
            success=True,
            execution_time=time.time() - start,
            user_satisfaction=0.8  # Can be updated with feedback
        )
    except Exception as e:
        outcome = Outcome(
            success=False,
            execution_time=time.time() - start,
            user_satisfaction=0.0
        )

    # 5. Record for learning
    await learning_system.record_decision(
        query=query,
        features=features,
        context=context,
        action_plan=action_plan,
        outcome=outcome,
        feedback=None  # Collect separately if needed
    )

    return result
```

### Option 3: Standalone Analysis

Use reflection learning for offline analysis:

```python
from HoloLoom.reflection import ReflectionEngine, DecisionTrajectory

# Collect trajectories during operation
trajectories = []

# ... run system ...

# Offline analysis
engine = ReflectionEngine()
for traj in trajectories:
    engine.record_trajectory(traj)

insights = await engine.analyze_decisions()
improvements = await engine.generate_improvements(insights)

# Review and apply improvements manually
for improvement in improvements:
    print(f"Suggestion: {improvement.rationale}")
    print(f"  {improvement.parameter_name}: {improvement.old_value} → {improvement.new_value}")
    print(f"  Expected improvement: {improvement.expected_improvement:.1%}")
```

---

## Usage Examples

### Example 1: Basic Pattern Analysis

```python
from HoloLoom.reflection import ReflectionEngine, DecisionTrajectory, Outcome

engine = ReflectionEngine()

# Record many decisions
for i in range(100):
    trajectory = DecisionTrajectory(
        trajectory_id=f"decision_{i}",
        query={"text": f"Query {i}"},
        features={"motifs": []},
        context={"shards": []},
        action_plan={"selected_tool": "web_search"},
        outcome=Outcome(success=True, execution_time=1.5, user_satisfaction=0.8)
    )
    engine.record_trajectory(trajectory)

# Analyze
insights = await engine.analyze_decisions()

# Print patterns
for pattern in insights.success_patterns:
    print(f"Success pattern: {pattern.description}")

for pattern in insights.failure_patterns:
    print(f"Failure pattern: {pattern.description}")
```

### Example 2: Task Adaptation

```python
from HoloLoom.reflection.integration import ReflectionLearningSystem
from HoloLoom.reflection.metalearning import TaskContext

system = ReflectionLearningSystem()

# Define new task
task = TaskContext(
    task_id="code_review",
    task_type="code",
    description="Review code for bugs and style issues",
    required_tools=["code_analyzer", "text_processor"]
)

# Adapt (uses past similar experiences)
adapted_params = await system.adapt_to_task(task)

# Use adapted parameters
policy.update_params(adapted_params)
```

### Example 3: Continuous Improvement Loop

```python
from HoloLoom.reflection.integration import ReflectionLearningSystem, LearningConfig

# Configure for automatic improvement
config = LearningConfig(
    reflection_interval=20,  # Analyze every 20 decisions
    auto_update_policy=True,  # Automatically apply improvements
    min_improvement_threshold=0.1  # Only apply if >10% improvement expected
)

system = ReflectionLearningSystem(config)

# Run for extended period
for i in range(1000):
    # Process query
    result = await process_query(query)

    # Record (triggers automatic reflection every 20 decisions)
    await system.record_decision(
        query=query,
        features=features,
        context=context,
        action_plan=action_plan,
        outcome=outcome
    )

# Export learnings
await system.export_learnings("learnings.json")
```

### Example 4: Contrastive Learning

```python
from HoloLoom.reflection.experience_replay import ExperienceReplayBuffer

buffer = ExperienceReplayBuffer(max_size=10000)

# Add experiences
for trajectory in trajectories:
    buffer.add(trajectory)

# Sample contrastive pairs (success vs failure)
pairs = buffer.sample_contrastive_pairs(n_pairs=20)

for success, failure in pairs:
    print(f"Success: {success.action_plan}")
    print(f"Failure: {failure.action_plan}")
    print("Learn: What made the success work?")
```

---

## Configuration

### LearningConfig Parameters

```python
@dataclass
class LearningConfig:
    # Reflection settings
    reflection_interval: int = 50  # Analyze every N decisions
    min_trajectories_for_reflection: int = 10  # Minimum data needed

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
    min_improvement_threshold: float = 0.1  # 10% minimum improvement

    # Logging
    verbose: bool = True
```

### Recommended Configurations

**Development/Testing**:
```python
config = LearningConfig(
    reflection_interval=10,  # Frequent reflection
    replay_buffer_size=1000,
    auto_update_policy=False,  # Manual review
    verbose=True
)
```

**Production - Conservative**:
```python
config = LearningConfig(
    reflection_interval=100,
    replay_buffer_size=50000,
    auto_update_policy=True,
    min_improvement_threshold=0.15,  # 15% threshold
    verbose=False
)
```

**Production - Aggressive**:
```python
config = LearningConfig(
    reflection_interval=25,
    replay_buffer_size=100000,
    auto_update_policy=True,
    min_improvement_threshold=0.05,  # 5% threshold
    verbose=False
)
```

---

## Best Practices

### 1. Data Quality

**Good**:
- Collect user feedback when possible
- Record complete context for each decision
- Maintain diverse experiences in buffer

**Bad**:
- Recording incomplete trajectories
- Missing outcome information
- No user feedback signals

### 2. Reflection Frequency

**Good**:
- Reflect after sufficient new data (50-100 decisions)
- Balance between learning speed and computational cost
- More frequent for development, less for production

**Bad**:
- Reflecting after every decision (too expensive)
- Never reflecting (no learning)

### 3. Policy Updates

**Good**:
- A/B test policy changes before full rollout
- Monitor performance after updates
- Keep old policy as fallback
- Apply threshold for minimum improvement

**Bad**:
- Blindly applying all suggested updates
- No monitoring after updates
- Overfitting to recent data

### 4. Experience Replay

**Good**:
- Use prioritized sampling for training
- Balance success/failure examples
- Consolidate memory periodically
- Sample diverse experiences

**Bad**:
- Only training on successes
- Uniform sampling (ignores priorities)
- Letting buffer grow unbounded

### 5. Meta-Learning

**Good**:
- Define clear task contexts
- Collect quality support examples
- Monitor adaptation performance
- Transfer learning from similar tasks

**Bad**:
- Vague task definitions
- Too few support examples (<5)
- No similarity measure between tasks

---

## Performance Considerations

### Memory Usage

**Reflection Engine**:
- Stores all trajectories in memory
- Memory: ~1KB per trajectory
- 10,000 trajectories ≈ 10MB

**Experience Replay Buffer**:
- Configurable max size
- Automatic consolidation when full
- Memory: ~1-2KB per entry
- 50,000 entries ≈ 50-100MB

**Optimization**:
```python
# Limit trajectory storage
engine = ReflectionEngine()
engine.max_trajectories = 5000  # Keep only recent 5000

# Aggressive buffer consolidation
buffer = ExperienceReplayBuffer(
    max_size=10000,
    consolidation_interval=500  # Consolidate more often
)
```

### Computational Cost

**Reflection Analysis**:
- Pattern detection: O(n²) where n = trajectories
- Feature importance: O(n × f) where f = features
- Recommendation generation: O(n)
- **Total**: ~1-5 seconds for 1000 trajectories

**Meta-Learning Adaptation**:
- Task embedding: O(n × d) where d = embedding dim
- Adaptation: O(k × m) where k = steps, m = parameters
- **Total**: ~0.1-0.5 seconds per task

**Experience Replay Sampling**:
- Prioritized sampling: O(n log n) for heap
- Contrastive pairs: O(n²) in worst case
- **Total**: ~0.01-0.1 seconds for batch

**Optimization**:
```python
# Analyze recent data only
insights = await engine.analyze_decisions(
    recent_only=True,  # Only last 1000
    max_trajectories=1000
)

# Batch policy updates
if len(accumulated_updates) > 10:
    await apply_batch_updates(accumulated_updates)
```

### Scaling

**Single Machine**:
- Up to 100K trajectories
- Up to 1M replay buffer entries
- Reflection every 50-100 decisions

**Distributed**:
```python
# Use shared memory for trajectories
from multiprocessing import Manager

manager = Manager()
shared_trajectories = manager.list()

# Use Redis for distributed buffer
from redis import Redis

class DistributedReplayBuffer:
    def __init__(self, redis_url):
        self.redis = Redis.from_url(redis_url)

    def add(self, trajectory):
        # Store in Redis with priority
        self.redis.zadd("buffer", {
            trajectory.trajectory_id: priority
        })
```

---

## Troubleshooting

### Issue: No patterns detected

**Symptoms**: Empty success_patterns and failure_patterns

**Solutions**:
- Ensure sufficient data (>20 trajectories)
- Check that outcomes vary (not all success or all failure)
- Verify feature extraction is working
- Lower pattern confidence threshold

### Issue: Policy not improving

**Symptoms**: Improvements generated but no performance gain

**Solutions**:
- Check if auto_update_policy is enabled
- Verify policy actually uses updated parameters
- Increase reflection frequency
- Lower min_improvement_threshold
- Review recommendations manually

### Issue: High memory usage

**Symptoms**: Memory grows over time

**Solutions**:
- Set max_trajectories limit on ReflectionEngine
- Reduce replay_buffer_size
- Increase consolidation_interval
- Clear old trajectories periodically

### Issue: Slow reflection analysis

**Symptoms**: Long wait times during reflection

**Solutions**:
- Use recent_only=True for analysis
- Reduce max_trajectories
- Increase reflection_interval
- Run reflection in background thread

---

## Future Enhancements

**Planned for Phase 4B-C**:

1. **Multi-Agent Reflection**
   - Agents learn from each other's experiences
   - Shared experience replay buffer
   - Federated meta-learning

2. **Causal Inference**
   - Identify causal factors for success/failure
   - Counterfactual reasoning
   - Intervention analysis

3. **Active Learning**
   - System requests examples for uncertain cases
   - Targeted data collection
   - Query-by-committee

4. **Curriculum Learning**
   - Gradually increase task difficulty
   - Automatic task ordering
   - Mastery-based progression

---

## References

### Academic Papers

- **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (2017)
- **Reptile**: Nichol et al., "On First-Order Meta-Learning Algorithms" (2018)
- **PER**: Schaul et al., "Prioritized Experience Replay" (2016)
- **Contrastive Learning**: Chen et al., "A Simple Framework for Contrastive Learning" (2020)

### HoloLoom Documentation

- [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) - Full Phase 4-7 vision
- [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md) - How to extend HoloLoom
- [PHASE3_COMPLETE.md](PHASE3_COMPLETE.md) - Phase 3 implementation summary

---

## Support

For questions or issues:

1. Check the [examples](../examples/reflection_learning_example.py)
2. Review this documentation
3. Open an issue on GitHub
4. Contact the HoloLoom team

**Last Updated**: 2025-11-17
**Version**: Phase 4A - Initial Release
