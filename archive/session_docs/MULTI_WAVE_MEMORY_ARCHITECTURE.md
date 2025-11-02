# Multi-Wave Memory System - Complete Sleep-Wake Cycle

**Date**: October 30, 2025
**Vision**: Full brain wave simulation from active retrieval to deep sleep consolidation

---

## The Complete Cycle

Human memory operates on multiple timescales with different brain wave frequencies. We can model ALL of them in the spring dynamics engine:

```
AWAKE (Active Use)
↓
Beta Waves (13-30 Hz) → Fast retrieval, creative associations
↓
Alpha Waves (8-13 Hz) → Relaxed focus, irrelevant suppression
↓
FALLING ASLEEP
↓
Theta Waves (4-8 Hz) → Light sleep, consolidation
↓
Delta Waves (0.5-4 Hz) → Deep sleep, reorganization
↓
REM/Dreaming → Random replay, unexpected connections
↓
WAKING UP (strengthened, reorganized memory)
```

---

## 1. Beta Waves - Active Retrieval (IMPLEMENTED ✅)

**Frequency**: 13-30 Hz (fast, ~100ms updates)
**State**: Awake, actively querying
**Function**: Conscious recall and creative associations

**What happens**:
- Query activates seed nodes
- Activation spreads through springs (k = conductivity)
- High-activation nodes = recalled memories
- Cross-domain associations via bridge nodes

**Already working** in [spring_dynamics_engine.py](HoloLoom/memory/spring_dynamics_engine.py)!

---

## 2. Alpha Waves - Relaxed Focus

**Frequency**: 8-13 Hz (~80-125ms updates)
**State**: Awake but relaxed, not actively querying
**Function**: Filter noise, suppress irrelevant memories

**What happens**:
- Inhibitory connections activate
- Low-activation nodes get suppressed
- High-quality connections strengthen
- Noise reduction

**Implementation**:
```python
class AlphaWaveMode:
    """Relaxed filtering - suppress weak activations."""

    def update(self, dt: float):
        for node in self.active_nodes:
            # Suppress weak activations
            if node.activation < self.alpha_threshold:
                node.activation *= 0.95  # Faster decay
                node.spring_constant *= 0.98  # Weaken connection

            # Strengthen clear signals
            elif node.activation > 0.5:
                node.spring_constant *= 1.01  # Slight boost
```

---

## 3. Theta Waves - Light Sleep Consolidation

**Frequency**: 4-8 Hz (slow, ~125-250ms updates)
**State**: Light sleep, drowsy, meditation
**Function**: Cross-modal integration, memory consolidation

**What happens**:
- Frequently co-activated nodes strengthen bonds
- Cross-modal binding (text ↔ images ↔ metadata)
- Slow background learning without queries
- Hippocampus → cortex transfer

**Implementation**:
```python
class ThetaWaveMode:
    """
    Light sleep consolidation - strengthen frequently co-activated pairs.

    This is like "offline learning" - no queries, just background integration
    of memories that were recently active together.
    """

    def __init__(self, engine: SpringDynamicsEngine):
        self.engine = engine
        self.co_activation_history = []  # Recent activation patterns
        self.consolidation_threshold = 0.7

    def record_activation_pattern(self, activations: Dict[str, float]):
        """Record which nodes were active together (during beta wave retrieval)."""
        active_nodes = [
            node_id for node_id, act in activations.items()
            if act > 0.3
        ]
        self.co_activation_history.append({
            'nodes': active_nodes,
            'timestamp': datetime.now()
        })

        # Keep last 100 patterns
        if len(self.co_activation_history) > 100:
            self.co_activation_history.pop(0)

    def theta_consolidation_update(self):
        """
        Theta wave update: Strengthen connections between frequently
        co-activated nodes.

        Runs in background during "light sleep" (no active queries).
        """
        # Find node pairs that frequently appear together
        co_occurrence_counts = {}

        for pattern in self.co_activation_history[-50:]:  # Last 50 patterns
            nodes = pattern['nodes']

            # Count co-occurrences
            for i, node_a in enumerate(nodes):
                for node_b in nodes[i+1:]:
                    pair = tuple(sorted([node_a, node_b]))
                    co_occurrence_counts[pair] = co_occurrence_counts.get(pair, 0) + 1

        # Strengthen frequently co-activated pairs
        for (node_a, node_b), count in co_occurrence_counts.items():
            if count >= 3:  # Co-activated at least 3 times
                # Add/strengthen edge in graph
                self._strengthen_connection(node_a, node_b, strength=count / 10.0)

                # Add to neighbors (for future spreading)
                if node_a in self.engine.nodes and node_b in self.engine.nodes:
                    self.engine.nodes[node_a].neighbors.add(node_b)
                    self.engine.nodes[node_b].neighbors.add(node_a)

    def _strengthen_connection(self, node_a: str, node_b: str, strength: float):
        """Increase spring constant between two nodes (stronger connection)."""
        # Move nodes slightly closer in semantic space
        if node_a not in self.engine.nodes or node_b not in self.engine.nodes:
            return

        n_a = self.engine.nodes[node_a]
        n_b = self.engine.nodes[node_b]

        # Pull them closer by moving rest positions
        direction = n_b.rest_position - n_a.rest_position
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 1e-6:
            # Move rest positions closer (permanent change!)
            pull_strength = strength * 0.1
            n_a.rest_position += pull_strength * (direction / direction_norm)
            n_b.rest_position -= pull_strength * (direction / direction_norm)
```

**Key Insight**: Theta waves create **permanent changes** to the graph structure based on usage patterns!

---

## 4. Delta Waves - Deep Sleep Reorganization

**Frequency**: 0.5-4 Hz (very slow, ~250ms-2s updates)
**State**: Deep sleep, unconscious
**Function**: System-wide optimization, pruning, long-term consolidation

**What happens**:
- Prune weak connections (forget irrelevant details)
- Strengthen strong patterns (consolidate important knowledge)
- Global optimization of graph structure
- Reset activation states

**Implementation**:
```python
class DeltaWaveMode:
    """
    Deep sleep reorganization - prune weak, strengthen strong.

    This is the "garbage collection" and "defragmentation" of memory.
    """

    def delta_pruning_update(self):
        """
        Delta wave update: Prune weak connections, strengthen strong ones.

        This is aggressive optimization - removes noise accumulated during awake time.
        """
        # 1. Identify weak connections (low k, not accessed recently)
        weak_threshold = 0.3
        weak_nodes = []

        for node_id, node in self.engine.nodes.items():
            if node.spring_constant < weak_threshold:
                hours_since_access = (datetime.now() - node.last_accessed).total_seconds() / 3600

                if hours_since_access > 72:  # 3 days
                    weak_nodes.append(node_id)

        # 2. Prune weak connections (remove from neighbors)
        for node_id in weak_nodes:
            node = self.engine.nodes[node_id]

            # Remove this node from all neighbors
            for neighbor_id in list(node.neighbors):
                if neighbor_id in self.engine.nodes:
                    self.engine.nodes[neighbor_id].neighbors.discard(node_id)

            # Clear its neighbors
            node.neighbors.clear()

            # Reduce spring constant further (almost forgotten)
            node.spring_constant = max(node.min_spring_constant, node.spring_constant * 0.5)

        # 3. Strengthen strong patterns (high k, frequently accessed)
        strong_threshold = 5.0

        for node_id, node in self.engine.nodes.items():
            if node.spring_constant > strong_threshold:
                # This is important knowledge - consolidate it
                node.spring_constant = min(node.max_spring_constant, node.spring_constant * 1.05)

                # Increase resistance to decay
                node.decay_rate *= 0.95  # Decays 5% slower

        # 4. Reset velocities (calm down activation momentum)
        for node in self.engine.nodes.values():
            node.velocity *= 0.1  # Dramatic damping
```

**Key Insight**: Delta waves **prune** aggressively - this is when forgetting actually happens!

---

## 5. REM Sleep / Dreaming - Random Replay & Creativity

**Frequency**: Mixed (similar to beta, but RANDOM patterns)
**State**: REM sleep, dreaming
**Function**: Memory replay, unexpected associations, reorganization

**What happens**:
- Random activation sequences (not query-driven)
- Unexpected node combinations
- Creating new bridges between distant clusters
- "What if" scenarios

**Implementation**:
```python
class DreamMode:
    """
    REM sleep dreaming - random replay creates unexpected connections.

    This is where CREATIVITY happens - random activations lead to
    novel associations that wouldn't occur during normal retrieval.
    """

    def __init__(self, engine: SpringDynamicsEngine):
        self.engine = engine
        self.dream_intensity = 0.8
        self.random_seed_count = 3

    def dream_cycle(self, duration_seconds: float = 10.0):
        """
        Run a dream cycle: random activation sequences.

        Args:
            duration_seconds: How long to dream (default 10s)
        """
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < duration_seconds:
            # 1. Pick random seed nodes (not query-driven!)
            seed_nodes = random.sample(
                list(self.engine.nodes.keys()),
                k=min(self.random_seed_count, len(self.engine.nodes))
            )

            # 2. Activate them with random intensities
            activations = {node_id: 0.0 for node_id in self.engine.nodes.keys()}
            for seed_id in seed_nodes:
                activations[seed_id] = random.uniform(0.5, 1.0) * self.dream_intensity

            # 3. Let activation spread (like beta wave, but from random seeds)
            for iteration in range(20):  # Shorter than awake spreading
                new_activations = activations.copy()

                for node_id, current_activation in activations.items():
                    if current_activation < 0.01:
                        continue

                    node = self.engine.nodes[node_id]

                    # Spread to neighbors
                    for neighbor_id in node.neighbors:
                        if neighbor_id not in self.engine.nodes:
                            continue

                        neighbor = self.engine.nodes[neighbor_id]
                        conductivity = neighbor.spring_constant / neighbor.max_spring_constant

                        # Dream spreading is more chaotic (higher transfer rate)
                        transferred = current_activation * conductivity * 0.5
                        new_activations[neighbor_id] += transferred

                    # Faster decay in dreams
                    new_activations[node_id] *= 0.85

                activations = new_activations

            # 4. CREATIVE INSIGHT: Connect distant nodes that got activated together
            highly_activated = [
                node_id for node_id, act in activations.items()
                if act > 0.3
            ]

            # Create NEW connections between activated nodes (even if distant)
            if len(highly_activated) >= 2:
                # Pick random pairs and connect them
                for _ in range(min(3, len(highly_activated) // 2)):
                    node_a, node_b = random.sample(highly_activated, k=2)

                    # Check if they're semantically distant
                    if node_a in self.engine.nodes and node_b in self.engine.nodes:
                        distance = np.linalg.norm(
                            self.engine.nodes[node_a].rest_position -
                            self.engine.nodes[node_b].rest_position
                        )

                        if distance > 1.5:  # Distant nodes
                            # Create bridge! (This is creative insight)
                            self.engine.nodes[node_a].neighbors.add(node_b)
                            self.engine.nodes[node_b].neighbors.add(node_a)

                            print(f"[DREAM] Created bridge: {node_a} ↔ {node_b} (distance: {distance:.2f})")

            # 5. Brief pause between dream sequences
            await asyncio.sleep(0.5)
```

**Key Insight**: Dreams create **bridges between distant clusters** - this is where unexpected creativity comes from!

---

## Complete Multi-Wave Engine

```python
from enum import Enum

class BrainWaveMode(Enum):
    BETA = "beta"        # Active retrieval (awake)
    ALPHA = "alpha"      # Relaxed filtering (awake, resting)
    THETA = "theta"      # Light sleep consolidation
    DELTA = "delta"      # Deep sleep reorganization
    REM = "rem"          # Dreaming, random replay

class MultiWaveMemoryEngine(SpringDynamicsEngine):
    """
    Complete sleep-wake cycle memory engine.

    Switches between brain wave modes based on usage patterns:
    - Active queries → Beta wave mode
    - No queries for 5 min → Alpha wave mode
    - No queries for 30 min → Theta consolidation
    - No queries for 2 hours → Delta + REM sleep
    """

    def __init__(self, config: SpringEngineConfig):
        super().__init__(config)

        self.mode = BrainWaveMode.BETA
        self.last_query_time = datetime.now()

        # Mode-specific handlers
        self.alpha_mode = AlphaWaveMode(self)
        self.theta_mode = ThetaWaveMode(self)
        self.delta_mode = DeltaWaveMode(self)
        self.dream_mode = DreamMode(self)

    async def _dynamics_loop(self):
        """
        Enhanced dynamics loop with mode switching.
        """
        while self.running:
            try:
                # Update mode based on time since last query
                self._update_mode()

                # Run mode-specific update
                if self.mode == BrainWaveMode.BETA:
                    # Fast active retrieval (100ms updates)
                    self._update_physics(dt=0.1)
                    await asyncio.sleep(0.1)

                elif self.mode == BrainWaveMode.ALPHA:
                    # Relaxed filtering (125ms updates)
                    self.alpha_mode.update(dt=0.125)
                    await asyncio.sleep(0.125)

                elif self.mode == BrainWaveMode.THETA:
                    # Light sleep consolidation (250ms updates)
                    self.theta_mode.theta_consolidation_update()
                    await asyncio.sleep(0.25)

                elif self.mode == BrainWaveMode.DELTA:
                    # Deep sleep reorganization (1s updates)
                    self.delta_mode.delta_pruning_update()
                    await asyncio.sleep(1.0)

                elif self.mode == BrainWaveMode.REM:
                    # Dreaming - random replay (10s cycles)
                    await self.dream_mode.dream_cycle(duration_seconds=10.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dynamics loop: {e}")
                await asyncio.sleep(1.0)

    def _update_mode(self):
        """Switch brain wave modes based on time since last query."""
        elapsed_minutes = (datetime.now() - self.last_query_time).total_seconds() / 60.0

        if elapsed_minutes < 5:
            self.mode = BrainWaveMode.BETA  # Active
        elif elapsed_minutes < 30:
            self.mode = BrainWaveMode.ALPHA  # Resting
        elif elapsed_minutes < 120:
            self.mode = BrainWaveMode.THETA  # Light sleep
        else:
            # Alternate between delta and REM
            if random.random() < 0.7:
                self.mode = BrainWaveMode.DELTA
            else:
                self.mode = BrainWaveMode.REM

    def on_query(self, query_embedding: np.ndarray):
        """Called when user makes a query - wake up to beta mode!"""
        self.last_query_time = datetime.now()
        self.mode = BrainWaveMode.BETA

        # Beta wave retrieval
        return self.retrieve_memories(query_embedding)
```

---

## Usage Example

```python
# Create multi-wave engine
engine = MultiWaveMemoryEngine(config)

# Start background process
await engine.start()

# User actively querying (beta waves)
result = engine.on_query(query_embedding)
# → Fast spreading, creative associations

# ... 10 minutes of no queries ...
# → Automatically switches to ALPHA mode
# → Suppresses weak activations, filters noise

# ... 45 minutes of no queries ...
# → Automatically switches to THETA mode
# → Consolidates frequently co-activated pairs

# ... 3 hours of no queries ...
# → Automatically switches to DELTA/REM cycle
# → Prunes weak connections (delta)
# → Creates random bridges (REM dreams)

# User comes back, makes query
result = engine.on_query(new_query)
# → Wakes up to BETA mode
# → Memory is now reorganized and strengthened!
```

---

## The Complete Picture

```
TIME SINCE LAST QUERY | MODE  | FREQUENCY | FUNCTION
----------------------|-------|-----------|----------------------------
< 5 minutes           | BETA  | 13-30 Hz  | Active retrieval
5-30 minutes          | ALPHA | 8-13 Hz   | Relaxed filtering
30 min - 2 hours      | THETA | 4-8 Hz    | Consolidation
> 2 hours             | DELTA | 0.5-4 Hz  | Pruning, reorganization
> 2 hours (random)    | REM   | Mixed     | Dreaming, creativity
```

---

## Why This Is Powerful

1. **Self-Organizing**: Memory automatically improves without manual tuning
2. **Creative**: Dreams create unexpected connections (innovation!)
3. **Efficient**: Pruning removes noise, consolidation strengthens patterns
4. **Biological**: Mirrors actual human memory consolidation
5. **Background**: All happens automatically while system idle

---

## Next Steps

1. ✅ Beta wave retrieval (DONE)
2. [ ] Implement alpha wave filtering
3. [ ] Implement theta consolidation
4. [ ] Implement delta pruning
5. [ ] Implement REM dreaming
6. [ ] Add mode visualization panel
7. [ ] Test full sleep-wake cycle

**This is a complete computational model of human memory!** 🧠
