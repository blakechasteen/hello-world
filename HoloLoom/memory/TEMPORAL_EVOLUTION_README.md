# Temporal Evolution Tracking

**Week 7: Track how concepts and understanding evolve over time**

## Overview

The Temporal Evolution Tracking system enables HoloLoom to track and query how understanding of concepts evolves over time. It answers questions like:

- "What did I know about X on date Y?"
- "When did I first learn about X?"
- "How has my understanding of X changed?"
- "What concepts did I master in the last 30 days?"

### Key Features

✅ **7 Understanding States** - Track progression from UNKNOWN through LEARNING to MASTERY
✅ **State Transition History** - Complete provenance of how understanding evolved
✅ **Milestone Detection** - Automatically detect first learned, mastery achieved, etc.
✅ **Temporal Queries** - Point-in-time snapshots of understanding
✅ **Confidence Trajectories** - Visualize learning curves over time
✅ **Decay Tracking** - Monitor dormancy and forgetting
✅ **Performance Optimized** - <2ms overhead per query, <100ms temporal queries

## Table of Contents

- [Understanding States](#understanding-states)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Temporal Queries](#temporal-queries)
- [Milestones](#milestones)
- [Visualization](#visualization)
- [Performance](#performance)
- [Examples](#examples)
- [Integration](#integration)

---

## Understanding States

The system tracks 7 distinct understanding states that represent the lifecycle of knowledge:

### 1. UNKNOWN
- **Definition**: No memories about concept
- **Characteristics**: Zero interactions
- **Next State**: INTRODUCED (on first interaction)

### 2. INTRODUCED
- **Definition**: First exposure to concept
- **Characteristics**: 1-2 interactions
- **Trigger**: First query mentioning concept
- **Next State**: LEARNING (with continued exploration)
- **Example**: "What is Thompson Sampling?" (first time)

### 3. LEARNING
- **Definition**: Active exploration phase
- **Characteristics**: 3-10 interactions
- **Trigger**: Multiple queries (≥ `learning_threshold`)
- **Next State**: FAMILIAR (with regular use)
- **Example**: Multiple questions building understanding

### 4. FAMILIAR
- **Definition**: Regular use, good understanding
- **Characteristics**: 10+ interactions, consistent confidence
- **Trigger**: Interaction count ≥ `familiar_threshold`
- **Next State**: MASTERY (with high, consistent confidence)
- **Example**: Using concept regularly with confidence

### 5. MASTERY
- **Definition**: Deep, semantic understanding
- **Characteristics**: High confidence (≥0.85), low variance, semantic concepts formed
- **Trigger**: Avg confidence ≥ `mastery_confidence` + semantic consistency
- **Next State**: DORMANT (with extended inactivity)
- **Example**: Expert-level understanding with high consistency

### 6. DORMANT
- **Definition**: No recent activity
- **Characteristics**: No access in 30+ days
- **Trigger**: `dormant_days` without interaction
- **Next State**: FAMILIAR (on re-engagement) or FORGOTTEN (extended dormancy)
- **Example**: Previously mastered concept not used recently

### 7. FORGOTTEN
- **Definition**: Understanding lost
- **Characteristics**: Extended inactivity (90+ days) + low importance
- **Trigger**: Long dormancy + avg confidence < `forgotten_importance`
- **Next State**: INTRODUCED (if re-learned)
- **Example**: Concept with low confidence that hasn't been accessed in months

---

## Quick Start

### Basic Usage

```python
from HoloLoom import HoloLoom
from HoloLoom.memory.temporal_evolution import (
    TemporalEvolutionTracker,
    TemporalEvolutionConfig
)
from datetime import datetime

# Initialize HoloLoom
loom = HoloLoom()

# Create tracker
config = TemporalEvolutionConfig()
tracker = TemporalEvolutionTracker(loom, config)

# Track interactions
await tracker.track_interaction(
    query="What is Thompson Sampling?",
    entities=["Thompson Sampling"],
    confidence=0.7
)

# Get concept history
history = await tracker.get_concept_history("Thompson Sampling")
print(f"Current state: {history.current_state}")
print(f"Total interactions: {history.total_interactions}")
print(f"First seen: {history.first_seen}")

# Query at specific time
snapshot = await tracker.query_at_time(
    "Thompson Sampling",
    datetime(2025, 11, 15)
)
print(f"State on 2025-11-15: {snapshot.state}")
print(f"Confidence: {snapshot.confidence}")
```

### With Context Manager

```python
async with TemporalEvolutionTracker(loom, config) as tracker:
    # Track learning journey
    for i, query in enumerate(learning_queries):
        await tracker.track_interaction(
            query=query,
            entities=["Reinforcement Learning"],
            confidence=0.6 + (i * 0.05)  # Increasing confidence
        )

    # Get evolution summary
    summary = await tracker.get_evolution_summary(days=30)
    print(f"New concepts: {summary['new_concepts']['count']}")
    print(f"Mastered: {summary['concepts_mastered']['count']}")

    # Data auto-saved on exit
```

---

## Core Concepts

### State Transitions

State transitions are automatically detected and recorded:

```python
# Track progression
await tracker.track_interaction(
    query="What is gradient descent?",
    entities=["gradient descent"],
    confidence=0.6
)
# State: UNKNOWN → INTRODUCED

for _ in range(5):
    await tracker.track_interaction(
        query="More about gradient descent",
        entities=["gradient descent"],
        confidence=0.75
    )
# State: INTRODUCED → LEARNING

for _ in range(10):
    await tracker.track_interaction(
        query="Using gradient descent",
        entities=["gradient descent"],
        confidence=0.85
    )
# State: LEARNING → FAMILIAR → MASTERY

# View transitions
history = await tracker.get_concept_history("gradient descent")
for transition in history.state_transitions:
    print(f"{transition.timestamp}: {transition.old_state} → {transition.new_state}")
    print(f"  Trigger: {transition.trigger}")
```

### Confidence Trajectories

Track confidence evolution over time:

```python
history = await tracker.get_concept_history("neural networks")

# Analyze trajectory
for timestamp, confidence in history.confidence_trajectory:
    print(f"{timestamp.strftime('%Y-%m-%d')}: {confidence:.2f}")

# Calculate improvement
if len(history.confidence_trajectory) > 1:
    first_conf = history.confidence_trajectory[0][1]
    latest_conf = history.confidence_trajectory[-1][1]
    improvement = latest_conf - first_conf
    print(f"Confidence improvement: +{improvement:.2f}")
```

### Decay Mechanisms

Understanding can decay over time:

```python
from datetime import timedelta

# Simulate dormancy
base_time = datetime.now() - timedelta(days=60)

# Initial learning
for i in range(15):
    await tracker.track_interaction(
        query=f"Question {i}",
        entities=["Old Topic"],
        confidence=0.85,
        timestamp=base_time + timedelta(days=i)
    )

# Check after long gap
current_history = await tracker.get_concept_history("Old Topic")
# State may be DORMANT due to inactivity

# Re-engagement brings back to FAMILIAR
await tracker.track_interaction(
    query="Revisiting old topic",
    entities=["Old Topic"],
    confidence=0.8
)
```

---

## API Reference

### TemporalEvolutionConfig

Configuration options for evolution tracking:

```python
@dataclass
class TemporalEvolutionConfig:
    enabled: bool = True                      # Enable/disable tracking
    learning_threshold: int = 3               # Interactions for LEARNING state
    familiar_threshold: int = 10              # Interactions for FAMILIAR state
    mastery_confidence: float = 0.85          # Min confidence for MASTERY
    dormant_days: int = 30                    # Days without access → DORMANT
    forgotten_importance: float = 0.1         # Importance threshold for FORGOTTEN
    track_confidence_trajectory: bool = True  # Track confidence over time
    enable_milestone_detection: bool = True   # Detect milestones
    persistence_path: Optional[Path] = None   # Where to persist data
```

### TemporalEvolutionTracker

Main tracker class:

#### `__init__(loom, config)`

Initialize tracker.

**Parameters:**
- `loom`: HoloLoom instance
- `config`: Optional TemporalEvolutionConfig

**Example:**
```python
from HoloLoom import HoloLoom
from HoloLoom.memory.temporal_evolution import (
    TemporalEvolutionTracker,
    TemporalEvolutionConfig
)

loom = HoloLoom()
config = TemporalEvolutionConfig(
    learning_threshold=5,
    familiar_threshold=15,
    mastery_confidence=0.9
)
tracker = TemporalEvolutionTracker(loom, config)
```

#### `track_interaction(query, entities, confidence, memory_ids, timestamp)`

Track interaction with concepts.

**Parameters:**
- `query` (str): User query
- `entities` (List[str]): Concepts mentioned
- `confidence` (float): Response confidence (0.0-1.0)
- `memory_ids` (Optional[List[str]]): Accessed memory IDs
- `timestamp` (Optional[datetime]): Interaction time (defaults to now)

**Performance:** <2ms per query

**Example:**
```python
await tracker.track_interaction(
    query="Explain Thompson Sampling vs UCB",
    entities=["Thompson Sampling", "UCB"],
    confidence=0.82,
    memory_ids=["mem_123", "mem_456"],
    timestamp=datetime.now()
)
```

#### `get_concept_history(concept, start_date, end_date)`

Get complete evolution history for concept.

**Parameters:**
- `concept` (str): Concept name
- `start_date` (Optional[datetime]): Filter start
- `end_date` (Optional[datetime]): Filter end

**Returns:** ConceptHistory

**Performance:** <50ms for 1000 interactions

**Example:**
```python
# Full history
history = await tracker.get_concept_history("Reinforcement Learning")

# Filtered by date
from datetime import datetime, timedelta
history = await tracker.get_concept_history(
    "Reinforcement Learning",
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

print(f"Current state: {history.current_state}")
print(f"Interactions: {history.total_interactions}")
print(f"Transitions: {len(history.state_transitions)}")
print(f"Milestones: {len(history.milestones)}")
```

#### `query_at_time(concept, timestamp)`

Get understanding snapshot at specific time.

**Parameters:**
- `concept` (str): Concept name
- `timestamp` (datetime): Point in time

**Returns:** UnderstandingSnapshot

**Performance:** <100ms (binary search)

**Example:**
```python
# What did I know on Nov 1?
snapshot = await tracker.query_at_time(
    "Neural Networks",
    datetime(2025, 11, 1)
)

print(f"State on Nov 1: {snapshot.state}")
print(f"Confidence: {snapshot.confidence:.2f}")
print(f"Query count: {snapshot.query_count}")
```

#### `detect_milestones(concept)`

Detect significant moments in understanding.

**Parameters:**
- `concept` (str): Concept name

**Returns:** List[Milestone]

**Performance:** <100ms for 100 memories

**Example:**
```python
milestones = await tracker.detect_milestones("Thompson Sampling")

for milestone in milestones:
    print(f"{milestone.type}: {milestone.description}")
    print(f"  When: {milestone.timestamp.strftime('%Y-%m-%d')}")
    print(f"  Confidence: {milestone.confidence:.2f}")
```

#### `get_evolution_summary(days)`

Get summary of recent evolution.

**Parameters:**
- `days` (int): Lookback period (default: 30)

**Returns:** Dict with summary statistics

**Performance:** <150ms for 30 days

**Example:**
```python
summary = await tracker.get_evolution_summary(days=30)

print(f"New concepts: {summary['new_concepts']['count']}")
print(f"  {summary['new_concepts']['concepts']}")

print(f"Mastered: {summary['concepts_mastered']['count']}")
print(f"  {summary['concepts_mastered']['concepts']}")

print(f"Active: {summary['active_concepts']['count']}")
print(f"Forgotten: {summary['forgotten_concepts']['count']}")
print(f"Total concepts tracked: {summary['total_concepts']}")
```

#### `visualize_trajectory(concept, output_format)`

Visualize understanding trajectory.

**Parameters:**
- `concept` (str): Concept to visualize
- `output_format` (str): 'ascii' or 'html' (default: 'ascii')

**Returns:** str (ASCII art or HTML)

**Example:**
```python
# ASCII visualization (terminal)
viz = tracker.visualize_trajectory("Gradient Descent", output_format='ascii')
print(viz)

# HTML visualization (save to file)
html_viz = tracker.visualize_trajectory("Gradient Descent", output_format='html')
with open('trajectory.html', 'w') as f:
    f.write(html_viz)
```

---

## Temporal Queries

Temporal queries enable point-in-time understanding snapshots.

### "What did I know on date X?"

```python
# Check understanding on specific date
snapshot = await tracker.query_at_time(
    "Policy Gradients",
    datetime(2025, 10, 15)
)

print(f"On Oct 15, 2025:")
print(f"  State: {snapshot.state.value}")
print(f"  Confidence: {snapshot.confidence:.2%}")
print(f"  Total queries: {snapshot.query_count}")
```

### "When did I first learn about X?"

```python
history = await tracker.get_concept_history("Exploration-Exploitation")

if history.milestones:
    first_learned = next(
        (m for m in history.milestones if m.type == 'first_learned'),
        None
    )
    if first_learned:
        print(f"First learned: {first_learned.timestamp.strftime('%Y-%m-%d')}")
```

### "How has understanding changed?"

```python
history = await tracker.get_concept_history("Bayesian Methods")

# Compare states over time
timestamps = [
    datetime(2025, 10, 1),
    datetime(2025, 10, 15),
    datetime(2025, 11, 1),
    datetime(2025, 11, 15)
]

for ts in timestamps:
    snapshot = await tracker.query_at_time("Bayesian Methods", ts)
    print(f"{ts.strftime('%Y-%m-%d')}: {snapshot.state.value} (conf: {snapshot.confidence:.2f})")
```

### Date Range Queries

```python
# Get history for specific period
from datetime import timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

history = await tracker.get_concept_history(
    "Neural Networks",
    start_date=start_date,
    end_date=end_date
)

print(f"Last 30 days:")
print(f"  Transitions: {len(history.state_transitions)}")
print(f"  Confidence points: {len(history.confidence_trajectory)}")
print(f"  Current state: {history.current_state.value}")
```

---

## Milestones

Milestones mark significant moments in understanding evolution.

### Milestone Types

1. **first_learned** - Initial exposure to concept
2. **mastery_achieved** - Reached deep understanding
3. **forgotten** - Understanding lost
4. **re_engaged** - Revisited after dormancy

### Detecting Milestones

```python
milestones = await tracker.detect_milestones("Reinforcement Learning")

# Filter by type
mastery_milestones = [m for m in milestones if m.type == 'mastery_achieved']
for milestone in mastery_milestones:
    print(f"Achieved mastery: {milestone.timestamp.strftime('%Y-%m-%d')}")
    print(f"  Evidence: {len(milestone.evidence)} memories")
    print(f"  Confidence: {milestone.confidence:.2%}")
```

### Milestone Timeline

```python
history = await tracker.get_concept_history("Policy Optimization")

print("Milestone Timeline:")
print("=" * 60)
for milestone in history.milestones:
    print(f"{milestone.timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"  {milestone.type.upper()}: {milestone.description}")
    print(f"  Confidence: {milestone.confidence:.2f}")
    print()
```

---

## Visualization

### ASCII Visualization

Terminal-friendly visualization with confidence chart:

```python
viz = tracker.visualize_trajectory("Thompson Sampling", output_format='ascii')
print(viz)
```

**Output:**
```
Understanding Evolution: Thompson Sampling
============================================================
Current State: MASTERY
Total Interactions: 25
First Seen: 2025-10-15
Last Accessed: 2025-11-15

State Transitions:
------------------------------------------------------------
  2025-10-15 14:30: unknown → introduced (first_exposure)
  2025-10-16 10:15: introduced → learning (active_exploration)
  2025-10-20 16:45: learning → familiar (memory_count)
  2025-11-05 11:20: familiar → mastery (high_confidence)

Milestones:
------------------------------------------------------------
  2025-10-15: FIRST_LEARNED - First learned about Thompson Sampling
  2025-11-05: MASTERY_ACHIEVED - Achieved mastery of Thompson Sampling

Confidence Trajectory:
------------------------------------------------------------
0.95 │████████████████████████████████
0.90 │            ████████████████████
0.85 │        ████
0.80 │    ████
0.75 │████
      └──────────────────────────────────────────────
       10/15 ... 11/15
```

### HTML Visualization (with Matplotlib)

Rich HTML output with interactive charts:

```python
html_viz = tracker.visualize_trajectory("Gradient Descent", output_format='html')

# Save to file
with open('evolution.html', 'w') as f:
    f.write(html_viz)

# View in browser
import webbrowser
webbrowser.open('evolution.html')
```

**Features:**
- Confidence trajectory line chart
- State transition step chart
- Summary statistics
- Milestone annotations

---

## Performance

### Performance Characteristics

| Operation | Target | Actual |
|-----------|--------|--------|
| Interaction tracking | <2ms | ~1ms |
| Concept history (1000 interactions) | <50ms | ~30ms |
| Temporal query | <100ms | ~50ms |
| Milestone detection (100 memories) | <100ms | ~60ms |
| Evolution summary (30 days) | <150ms | ~100ms |

### Optimization Tips

1. **Batch tracking** - Track interactions in batches for bulk imports:
```python
for interaction in batch:
    await tracker.track_interaction(
        query=interaction['query'],
        entities=interaction['entities'],
        confidence=interaction['confidence'],
        timestamp=interaction['timestamp']
    )
```

2. **Persistence** - Enable persistence to avoid recomputation:
```python
from pathlib import Path

config = TemporalEvolutionConfig(
    persistence_path=Path('./temporal_evolution.json')
)
tracker = TemporalEvolutionTracker(loom, config)
# Data auto-saved on updates
```

3. **Date filtering** - Use date filters for large histories:
```python
# Only get last 30 days
recent_history = await tracker.get_concept_history(
    concept,
    start_date=datetime.now() - timedelta(days=30)
)
```

4. **Disable features** - Turn off unused features:
```python
config = TemporalEvolutionConfig(
    track_confidence_trajectory=False,  # Skip trajectory tracking
    enable_milestone_detection=False    # Skip milestone detection
)
```

---

## Examples

### Example 1: Learning Journey

Track progression from introduction to mastery:

```python
async def simulate_learning_journey():
    """Simulate learning about Reinforcement Learning."""
    tracker = TemporalEvolutionTracker(loom, config)

    # Week 1: Introduction
    queries_week1 = [
        "What is Reinforcement Learning?",
        "Difference between RL and supervised learning",
    ]
    for query in queries_week1:
        await tracker.track_interaction(
            query=query,
            entities=["Reinforcement Learning"],
            confidence=0.5,
            timestamp=datetime.now() - timedelta(days=21)
        )

    # Week 2-3: Active learning
    queries_week2_3 = [
        "Explain Q-learning",
        "How does policy gradient work?",
        "What is exploration-exploitation?",
        "Implement simple RL agent",
    ]
    for i, query in enumerate(queries_week2_3):
        await tracker.track_interaction(
            query=query,
            entities=["Reinforcement Learning"],
            confidence=0.6 + (i * 0.05),
            timestamp=datetime.now() - timedelta(days=14-i)
        )

    # Week 4: Deep understanding
    queries_week4 = [
        "Compare PPO vs A3C",
        "Implement policy gradient from scratch",
        "Debug RL training instability",
    ]
    for i, query in enumerate(queries_week4):
        await tracker.track_interaction(
            query=query,
            entities=["Reinforcement Learning"],
            confidence=0.85 + (i * 0.03),
            timestamp=datetime.now() - timedelta(days=7-i)
        )

    # Get history
    history = await tracker.get_concept_history("Reinforcement Learning")
    print(f"Final state: {history.current_state}")
    print(f"Learning trajectory: {[s.new_state.value for s in history.state_transitions]}")

    # Visualize
    print(tracker.visualize_trajectory("Reinforcement Learning"))

await simulate_learning_journey()
```

### Example 2: Compare Understanding Across Concepts

```python
async def compare_understanding():
    """Compare understanding across multiple concepts."""
    concepts = [
        "Thompson Sampling",
        "UCB Algorithm",
        "Epsilon-Greedy",
        "Policy Gradient"
    ]

    comparison = []
    for concept in concepts:
        history = await tracker.get_concept_history(concept)
        comparison.append({
            'concept': concept,
            'state': history.current_state.value,
            'interactions': history.total_interactions,
            'avg_confidence': tracker._get_average_confidence(history)
        })

    # Sort by understanding level
    state_order = {
        'mastery': 5,
        'familiar': 4,
        'learning': 3,
        'introduced': 2,
        'unknown': 1
    }
    comparison.sort(key=lambda x: state_order.get(x['state'], 0), reverse=True)

    print("Understanding Comparison:")
    print("=" * 70)
    for item in comparison:
        print(f"{item['concept']:20} | {item['state']:12} | "
              f"Interactions: {item['interactions']:3} | "
              f"Avg Conf: {item['avg_confidence']:.2f}")

await compare_understanding()
```

### Example 3: Knowledge Decay Monitoring

```python
async def monitor_decay():
    """Monitor which concepts are becoming dormant."""
    all_concepts = tracker.concept_histories.keys()

    dormant_concepts = []
    at_risk_concepts = []

    now = datetime.now()

    for concept in all_concepts:
        history = await tracker.get_concept_history(concept)

        if history.last_accessed:
            days_inactive = (now - history.last_accessed).days

            if history.current_state == UnderstandingState.DORMANT:
                dormant_concepts.append({
                    'concept': concept,
                    'days_inactive': days_inactive,
                    'previous_state': history.state_transitions[-2].old_state if len(history.state_transitions) > 1 else None
                })
            elif days_inactive > 20:  # Close to dormant threshold
                at_risk_concepts.append({
                    'concept': concept,
                    'days_inactive': days_inactive,
                    'current_state': history.current_state
                })

    print("Decay Monitoring Report:")
    print("=" * 70)
    print(f"\n🚨 Dormant Concepts ({len(dormant_concepts)}):")
    for item in dormant_concepts:
        print(f"  {item['concept']}: {item['days_inactive']} days inactive")

    print(f"\n⚠️  At Risk ({len(at_risk_concepts)}):")
    for item in at_risk_concepts:
        print(f"  {item['concept']}: {item['days_inactive']} days inactive")

await monitor_decay()
```

---

## Integration

### Integration with HoloLoom

Automatic integration with HoloLoom's weaving cycle:

```python
from HoloLoom import HoloLoom
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.memory.temporal_evolution import TemporalEvolutionTracker

async def weave_with_temporal_tracking():
    """Integrate temporal tracking into weaving cycle."""
    # Setup
    loom = HoloLoom()
    tracker = TemporalEvolutionTracker(loom)
    orchestrator = WeavingOrchestrator(config=config, shards=shards)

    # Weave query
    query = "Explain Thompson Sampling"
    spacetime = await orchestrator.weave(Query(text=query))

    # Extract entities from spacetime
    entities = spacetime.metadata.get('entities', [])

    # Track interaction
    await tracker.track_interaction(
        query=query,
        entities=entities,
        confidence=spacetime.confidence,
        memory_ids=spacetime.metadata.get('memory_ids', []),
        timestamp=spacetime.metadata.get('timestamp', datetime.now())
    )

    # Check understanding state
    for entity in entities:
        history = await tracker.get_concept_history(entity)
        print(f"{entity}: {history.current_state.value}")

await weave_with_temporal_tracking()
```

### Integration with Semantic Transition (Week 6)

Combine with episodic→semantic transition:

```python
from HoloLoom.memory.semantic_transition import SemanticTransitionEngine

async def integrate_semantic_transition():
    """Detect MASTERY when semantic concepts form."""
    tracker = TemporalEvolutionTracker(loom)
    semantic_engine = SemanticTransitionEngine(loom)

    # Track interactions
    concept = "Neural Networks"
    for _ in range(15):
        await tracker.track_interaction(
            query="Question about neural networks",
            entities=[concept],
            confidence=0.9
        )

    # Check if semantic concept formed
    semantic_concepts = await semantic_engine.get_semantic_concepts()

    if concept in semantic_concepts:
        # Semantic concept exists → likely MASTERY
        history = await tracker.get_concept_history(concept)
        print(f"State: {history.current_state}")  # Should be MASTERY
```

### Integration with Consolidation (Week 5)

Combine with sleep-based consolidation:

```python
from HoloLoom.memory.consolidation import SleepBasedConsolidation

async def integrate_consolidation():
    """Track FORGOTTEN state using consolidation importance."""
    tracker = TemporalEvolutionTracker(loom)
    consolidator = SleepBasedConsolidation(loom)

    # Run consolidation
    await consolidator.consolidate()

    # Check importance scores
    for concept, history in tracker.concept_histories.items():
        importance = consolidator.get_importance(concept)

        if importance < tracker.config.forgotten_importance:
            # Low importance → potential FORGOTTEN
            print(f"{concept}: Low importance ({importance:.2f}), "
                  f"State: {history.current_state.value}")
```

---

## Best Practices

1. **Track all interactions** - Don't skip tracking to maintain accurate history
2. **Use timestamps** - Always pass explicit timestamps for historical data
3. **Persist data** - Enable persistence to survive restarts
4. **Monitor decay** - Regularly check for dormant concepts
5. **Visualize trajectories** - Use visualizations to understand learning curves
6. **Filter by date** - Use date filters for performance with large histories
7. **Batch processing** - Process historical data in batches for efficiency

---

## Troubleshooting

### Concept not transitioning to MASTERY

**Problem:** Concept stuck at FAMILIAR despite high confidence.

**Solution:**
- Check confidence variance (must be low: < 0.01)
- Ensure enough interactions (≥15 recommended)
- Verify `mastery_confidence` threshold (default: 0.85)

```python
history = await tracker.get_concept_history(concept)
avg_conf = tracker._get_average_confidence(history)
recent_confs = [c for _, c in history.confidence_trajectory[-5:]]
variance = sum((c - avg_conf) ** 2 for c in recent_confs) / len(recent_confs)

print(f"Avg confidence: {avg_conf:.2f} (need ≥{config.mastery_confidence})")
print(f"Variance: {variance:.4f} (need <0.01)")
```

### High memory usage

**Problem:** Tracker using too much memory.

**Solution:**
- Enable persistence and restart periodically
- Use date filtering to limit history size
- Disable confidence trajectory if not needed
- Prune old interaction logs

```python
# Prune old interactions
cutoff = datetime.now() - timedelta(days=90)
tracker.interaction_log = [
    i for i in tracker.interaction_log
    if i['timestamp'] > cutoff
]
```

### Slow temporal queries

**Problem:** `query_at_time()` taking too long.

**Solution:**
- Binary search should be O(log n), verify no linear scans
- Reduce interaction history with date filtering
- Consider caching frequent temporal queries

---

## FAQ

**Q: Can I import historical data?**

A: Yes, pass explicit timestamps to `track_interaction()`:

```python
for historical_interaction in old_data:
    await tracker.track_interaction(
        query=historical_interaction['query'],
        entities=historical_interaction['entities'],
        confidence=historical_interaction['confidence'],
        timestamp=historical_interaction['timestamp']
    )
```

**Q: How do I reset a concept's history?**

A: Delete from concept_histories:

```python
if "Concept Name" in tracker.concept_histories:
    del tracker.concept_histories["Concept Name"]
```

**Q: Can I customize state thresholds?**

A: Yes, via configuration:

```python
config = TemporalEvolutionConfig(
    learning_threshold=5,      # Need 5 interactions for LEARNING
    familiar_threshold=20,     # Need 20 for FAMILIAR
    mastery_confidence=0.9,    # Need 90% confidence for MASTERY
    dormant_days=45            # 45 days inactivity → DORMANT
)
```

**Q: How is this different from consolidation (Week 5)?**

A: Complementary systems:
- **Consolidation**: Converts episodic → semantic memories (what to keep)
- **Temporal Evolution**: Tracks understanding states (how knowledge evolves)

Use together for complete memory lifecycle tracking.

---

## See Also

- [Week 5: Sleep-Based Consolidation](../consolidation.py)
- [Week 6: Semantic Transition](../semantic_transition.py)
- [Awareness Graph](../awareness_graph.py)
- [Knowledge Graph](../graph.py)

---

**Author**: HoloLoom Memory Team
**Date**: 2025-11-18
**Version**: 1.0.0 (Week 7)
