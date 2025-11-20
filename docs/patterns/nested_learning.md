# Pattern: Nested Learning (Continuum Memory)

**Status**: ✅ Implemented (January 2025)
**Location**: `src/hololoom/patterns/nested_learning/`
**Inspiration**: Google Research Nested Learning / Continuum Memory

---

## Intent

Introduce multi-timescale learning across Shuttle components, inspired by
Google's Nested Learning / Continuum Memory ideas.

Different modules (bandit, graph heuristics, policy meta-strategy) update
at different frequencies, enabling continual learning without modifying
core logic.

**Key Innovation**: Core stays clean and predictable. Learning happens in an
observation layer that doesn't affect single-weave semantics.

---

## Applicable To

- **Shuttle decision layer** - Policies, bandits, Thompson Sampling
- **Graph expansion heuristics** - Yarn retrieval strategies
- **Long-horizon behavior** - Meta-policies, high-level strategy

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    WeaveRunner                             │
│                 (Orchestrates core + patterns)             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────┐      ┌──────────────────────────────────┐  │
│  │ Shuttle  │──→   │      NestedLearningPattern       │  │
│  │ (Core)   │      │                                  │  │
│  └──────────┘      │  ┌────────────────────────────┐  │  │
│       │            │  │  Bandit Module (FAST)      │  │  │
│       │            │  │  - Update every episode    │  │  │
│       ↓            │  │  - Learn from reward       │  │  │
│  WeaveResult       │  └────────────────────────────┘  │  │
│       │            │                                  │  │
│       │            │  ┌────────────────────────────┐  │  │
│       └───────────→│  │  Graph Module (MEDIUM)     │  │  │
│                    │  │  - Update every 100 eps    │  │  │
│                    │  │  - Refine heuristics       │  │  │
│                    │  └────────────────────────────┘  │  │
│                    │                                  │  │
│                    │  ┌────────────────────────────┐  │  │
│                    │  │  Policy Module (SLOW)      │  │  │
│                    │  │  - Update every 1000 eps   │  │  │
│                    │  │  - Meta-strategy           │  │  │
│                    │  └────────────────────────────┘  │  │
│                    └──────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

**Flow**:
1. Shuttle performs single weave → WeaveResult
2. WeaveRunner dispatches result to pattern
3. Pattern extracts reward from metadata
4. Pattern updates modules at their respective frequencies

**Frequencies** (configurable):
- Bandit: Every 1 episode (fast horizon)
- Graph: Every 100 episodes (medium horizon)
- Policy: Every 1000 episodes (slow horizon)

---

## Code Location

### Core Types
- `src/hololoom/core/types.py` - WeaveResult shared type

### Pattern Interface
- `src/hololoom/patterns/base.py` - LearningPattern protocol
- `src/hololoom/patterns/nested_learning/pattern.py` - Main implementation
- `src/hololoom/patterns/nested_learning/stubs.py` - No-op scaffolding

### Runner
- `src/hololoom/runners/weave_runner.py` - WeaveRunner orchestrator

---

## How It Hooks In

### Consumes

**WeaveResult** (from `Shuttle.intersect`):
```python
@dataclass
class WeaveResult:
    query: str
    context_blocks: Dict[str, Any]
    metadata: Dict[str, Any]
```

**Uses `result.metadata['reward']`** as scalar learning signal.

### Updates

Three modules at different frequencies:

1. **`bandit_module.learn_from_reward(reward)`**
   - Called every episode
   - Updates Thompson Sampling priors or other bandit state
   - Fast adaptation to recent performance

2. **`graph_module.refine_heuristics()`**
   - Called every 100 episodes
   - Adjusts graph expansion priorities
   - Medium-term structural learning

3. **`policy_module.update_meta_strategy()`**
   - Called every 1000 episodes
   - Updates high-level policy selection
   - Slow long-term adaptation

---

## Usage

### Basic Example

```python
from hololoom.runners import WeaveRunner
from hololoom.patterns.nested_learning import (
    NestedLearningPattern,
    NoOpBanditModule,
    NoOpGraphModule,
    NoOpPolicyModule
)

# Assume shuttle already constructed
shuttle = Shuttle(warp=warp, yarn=yarn)

# Create pattern with no-op stubs
pattern = NestedLearningPattern(
    bandit_module=NoOpBanditModule(),
    graph_module=NoOpGraphModule(),
    policy_module=NoOpPolicyModule()
)

# Create runner
runner = WeaveRunner(shuttle=shuttle, patterns=[pattern])

# Run queries (patterns learn automatically)
warp_results = warp.search("Why is my build failing?", top_k=5)
result = runner.run_query("Why is my build failing?", warp_results)

# Pattern automatically updates bandit/graph/policy at their frequencies
```

### With Real Modules

```python
# Replace no-op stubs with real implementations

class RealBanditModule:
    def __init__(self, bandit):
        self.bandit = bandit  # Your PolicyBandit or Thompson Sampling

    def learn_from_reward(self, reward: float):
        # Update bandit priors based on reward
        self.bandit.update(reward=reward)

class RealGraphModule:
    def __init__(self, yarn):
        self.yarn = yarn  # Your Yarn graph

    def refine_heuristics(self):
        # Adjust graph scoring functions
        self.yarn.refine_expansion_strategy()

class RealPolicyModule:
    def __init__(self, policy_selector):
        self.policy_selector = policy_selector

    def update_meta_strategy(self):
        # Adjust policy selection priors
        self.policy_selector.update_priors()

# Create pattern with real modules
pattern = NestedLearningPattern(
    bandit_module=RealBanditModule(my_bandit),
    graph_module=RealGraphModule(my_yarn),
    policy_module=RealPolicyModule(my_policy_selector)
)

runner = WeaveRunner(shuttle=shuttle, patterns=[pattern])
```

### Context Manager

```python
# Use context manager for lifecycle management
with WeaveRunner(shuttle=shuttle, patterns=[pattern]) as runner:
    for query in queries:
        warp_results = warp.search(query, top_k=5)
        result = runner.run_query(query, warp_results)

# Pattern lifecycle hooks called automatically:
# - on_startup() at entry
# - on_shutdown() at exit
```

---

## Config Knobs

### Update Frequencies

```python
pattern = NestedLearningPattern(
    bandit_module=my_bandit,
    graph_module=my_graph,
    policy_module=my_policy,
    freq={
        'bandit': 1,     # Update every episode
        'graph': 50,     # Update every 50 episodes (faster than default)
        'policy': 500,   # Update every 500 episodes (faster than default)
    }
)
```

### Reward Extraction Logic

By default, extracts `result.metadata['reward']` with fallbacks:
1. `metadata['reward']`
2. `metadata['confidence']`
3. `0.5` (neutral)

To customize, subclass `NestedLearningPattern` and override `_extract_reward()`:

```python
class CustomPattern(NestedLearningPattern):
    def _extract_reward(self, result: WeaveResult) -> float:
        # Custom reward logic
        if result.metadata.get('error'):
            return 0.0
        elif result.metadata.get('high_confidence'):
            return 1.0
        else:
            return result.get_confidence()
```

---

## Monitoring

### Get Pattern Statistics

```python
# After running some queries
stats = pattern.get_stats()

print(f"Total episodes: {stats['total_episodes']}")
print(f"Bandit updates: {stats['bandit_updates']}")
print(f"Graph updates: {stats['graph_updates']}")
print(f"Policy updates: {stats['policy_updates']}")
print(f"Last reward: {stats['last_reward']:.3f}")
```

### Get Runner Statistics

```python
stats = runner.get_stats()

print(f"Query count: {stats['query_count']}")
print(f"Pattern count: {stats['pattern_count']}")
```

### Logging

Pattern emits logs at appropriate levels:
- `DEBUG`: Per-episode updates (bandit)
- `INFO`: Medium/slow updates (graph, policy)
- `WARNING`: Unusual rewards or failures
- `ERROR`: Pattern exceptions (doesn't crash runner)

Configure logging:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('hololoom.patterns.nested_learning')
```

---

## Gotchas

### 1. Memory Growth

**Issue**: Unbounded counters/logs can grow forever.

**Solution**: Pattern already resets counters on each update. For stats, call `pattern.reset_stats()` periodically if running long sessions.

### 2. Start with No-Ops

**Issue**: Trying to implement all three modules at once is complex.

**Solution**: Start with no-op stubs. Replace one module at a time:
1. First: Implement `BanditModule` (fast feedback)
2. Second: Implement `GraphModule` (medium-term)
3. Last: Implement `PolicyModule` (long-term)

### 3. Pattern Exceptions

**Issue**: Buggy pattern crashes the whole system.

**Solution**: WeaveRunner catches all pattern exceptions and logs them. Core weaving continues even if patterns fail.

### 4. Not Required for Correctness

**Important**: Patterns should enhance behavior, not be required. Core must work without any patterns attached.

```python
# This should work fine (no patterns)
shuttle = Shuttle(warp=warp, yarn=yarn)
result = shuttle.intersect(query, warp_results)

# This is enhanced (with patterns)
runner = WeaveRunner(shuttle=shuttle, patterns=[pattern])
result = runner.run_query(query, warp_results)
```

---

## Integration with HoloLoom

### Existing Systems

Nested Learning integrates with:

| HoloLoom System | Integration Point |
|-----------------|-------------------|
| **Thompson Sampling** | `bandit_module` wraps PolicyBandit |
| **Yarn Graph** | `graph_module` wraps graph heuristics |
| **Policy Engine** | `policy_module` wraps policy selection |
| **Recursive Learning** | Can compose with other patterns |
| **Alignment Framework** | Reward can come from safety scores |

### Multiple Patterns

WeaveRunner supports multiple patterns:

```python
runner = WeaveRunner(
    shuttle=shuttle,
    patterns=[
        nested_learning_pattern,
        logging_pattern,
        alignment_pattern,
    ]
)
```

All patterns receive each WeaveResult independently.

---

## Testing

### Unit Tests

```bash
pytest src/hololoom/patterns/nested_learning/tests/ -v
```

### Integration Test Example

```python
def test_nested_learning_updates():
    # Create pattern with spy modules
    bandit_calls = []
    graph_calls = []
    policy_calls = []

    class SpyBandit:
        def learn_from_reward(self, reward):
            bandit_calls.append(reward)

    class SpyGraph:
        def refine_heuristics(self):
            graph_calls.append("refined")

    class SpyPolicy:
        def update_meta_strategy(self):
            policy_calls.append("updated")

    pattern = NestedLearningPattern(
        bandit_module=SpyBandit(),
        graph_module=SpyGraph(),
        policy_module=SpyPolicy(),
        freq={'bandit': 1, 'graph': 3, 'policy': 5}
    )

    # Run 5 episodes
    for i in range(5):
        result = WeaveResult(
            query=f"query {i}",
            context_blocks={},
            metadata={'reward': 0.8}
        )
        pattern.on_episode_end(result)

    # Check updates
    assert len(bandit_calls) == 5      # Every episode
    assert len(graph_calls) == 1       # Episode 3
    assert len(policy_calls) == 1      # Episode 5
```

---

## Future Enhancements

### Phase 2 (Next 3 Months)

- **Adaptive frequencies**: Adjust update rates based on performance
- **Trace storage**: Store (state, action, reward) for offline RL
- **Multi-agent patterns**: Coordinate learning across multiple runners
- **Checkpointing**: Save/load pattern state

### Phase 3 (6 Months)

- **Meta-learning patterns**: Learn which patterns work best
- **Hierarchical learning**: Nested patterns at multiple scales
- **Transfer learning**: Share learned behavior across domains

---

## References

- **Google Research**: Nested Learning, Continuum Memory
- **HoloLoom Core**: `src/hololoom/core/weaving_orchestrator.py`
- **Thompson Sampling**: `src/hololoom/weave/policy/thompson_sampling.py`
- **Recursive Learning**: See `RECURSIVE_LEARNING_COMPLETE.md`

---

## License

MIT License - Same as HoloLoom

---

## Credits

- **Pattern Design**: User + Claude Code
- **Implementation**: Claude Code (January 2025)
- **Inspiration**: Google Research Nested Learning / Continuum Memory
