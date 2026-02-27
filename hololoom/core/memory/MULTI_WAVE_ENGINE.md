# Multi-Wave Memory Engine - Brain Wave Inspired Consolidation

**Status**: Production Ready (October 2025)
**Location**: `hololoom/memory/multi_wave_engine.py` (623 lines)
**Modes**: 5 brain wave states (Beta, Alpha, Theta, Delta, REM)

Brain wave-inspired memory consolidation system that automatically switches modes based on system activity.

---

## Overview

The Multi-Wave Memory Engine models the human brain's sleep-wake cycle for memory management:

- **BETA (13-30 Hz)**: Active retrieval during queries
- **ALPHA (8-13 Hz)**: Relaxed filtering when idle 5-30 minutes
- **THETA (4-8 Hz)**: Light sleep consolidation (30 min - 2 hours idle)
- **DELTA (0.5-4 Hz)**: Deep sleep pruning (>2 hours idle)
- **REM**: Dreaming - random replay creates novel connections

This enables automatic memory optimization without explicit garbage collection or maintenance commands.

---

## Quick Start

```python
from hololoom.memory.multi_wave_engine import MultiWaveMemoryEngine, BrainWaveMode
import numpy as np

# Create engine
engine = MultiWaveMemoryEngine()

# Start background dynamics loop
await engine.start()

# Query (wakes up to BETA mode)
query_embedding = np.random.randn(384)
result = engine.on_query(query_embedding)

print(f"Mode: {engine.mode.value}")  # "beta"
print(f"Found: {len(result.recalled_nodes)} memories")

# After 30 minutes idle, automatically switches to THETA (consolidation)
# After 2 hours idle, switches to DELTA/REM (deep sleep)

# Stop engine
await engine.stop()
```

---

## Brain Wave Modes

### BETA (Active Retrieval)

**When**: <5 minutes since last query (or during ingestion)
**What**: Fast 100ms update cycles, full spring dynamics propagation
**Purpose**: Responsive retrieval for active use

```python
# Automatic during queries
result = engine.on_query(query_embedding)
# Engine stays in BETA for 5 minutes after
```

### ALPHA (Relaxed Filtering)

**When**: 5-30 minutes idle
**What**: 125ms cycles, suppress weak activations
**Purpose**: Reduce noise, strengthen clear signals

```python
# Automatically entered after 5 min idle
# Weak signals (velocity < 0.2) decay faster
# Strong signals (spring_constant > 3.0) get slight boost
```

### THETA (Light Sleep Consolidation)

**When**: 30 minutes - 2 hours idle
**What**: 250ms cycles, strengthen co-activated pairs
**Purpose**: Create permanent new connections from usage patterns

```python
# Monitors which nodes were active together
# If pairs co-occur 3+ times, creates permanent connection
# Pulls rest positions closer in embedding space
```

**Key Algorithm**: Co-activation tracking
```
For each activation pattern in history:
    Count node pair co-occurrences
    If pair appeared together 3+ times:
        Add bidirectional neighbor links
        Pull rest positions closer (permanent change!)
```

### DELTA (Deep Sleep Pruning)

**When**: >2 hours idle (70% of deep sleep time)
**What**: 1 second cycles, aggressive optimization
**Purpose**: Remove weak connections, strengthen important ones

```python
# Pruning criteria:
# - spring_constant < 0.3 (weak)
# - last_accessed > 72 hours ago
# - Result: neighbors cleared, spring constant halved

# Strengthening criteria:
# - spring_constant > 5.0 (important)
# - Result: spring constant +5%, decay rate -5%
```

### REM (Dreaming)

**When**: >2 hours idle (30% of deep sleep time)
**What**: 10 second dream cycles
**Purpose**: Create novel connections through random replay

```python
# Dream cycle:
# 1. Pick 3 random seed nodes
# 2. Activate with random intensities
# 3. Let activation spread (chaotic, fast decay)
# 4. Find highly activated distant pairs
# 5. Create "bridge" connections (creative insight!)

# Bridge creation:
# If two nodes activated together but semantically distant (>1.5 embedding distance):
#     Create bidirectional neighbor link
# This is where CREATIVE INSIGHTS happen!
```

---

## Mode Switching Logic

```python
elapsed_minutes = time_since_last_query()

if elapsed_minutes < 5:
    mode = BETA      # Active
elif elapsed_minutes < 30:
    mode = ALPHA     # Resting
elif elapsed_minutes < 120:
    mode = THETA     # Light sleep
else:
    # Deep sleep: 70% DELTA, 30% REM
    mode = DELTA if random() < 0.7 else REM
```

**Exception**: During active ingestion, always stay in BETA mode.

---

## Key Classes

### BrainWaveMode

```python
class BrainWaveMode(Enum):
    BETA = "beta"        # 13-30 Hz - Active retrieval
    ALPHA = "alpha"      # 8-13 Hz - Relaxed filtering
    THETA = "theta"      # 4-8 Hz - Light sleep consolidation
    DELTA = "delta"      # 0.5-4 Hz - Deep sleep pruning
    REM = "rem"          # Mixed - Dreaming
```

### ThetaWaveConsolidator

Consolidation engine for light sleep:

```python
class ThetaWaveConsolidator:
    def record_activation_pattern(activations, source, threshold=0.3)
    def theta_consolidation_update() -> int  # Returns connections created
```

### DeltaWavePruner

Pruning engine for deep sleep:

```python
class DeltaWavePruner:
    weak_threshold = 0.3       # Below this = weak
    strong_threshold = 5.0     # Above this = important
    prune_after_hours = 72     # 3 days without access

    def delta_pruning_update() -> Tuple[int, int]  # (pruned, strengthened)
```

### REMDreamer

Creative connection generator:

```python
class REMDreamer:
    dream_intensity = 0.8
    random_seed_count = 3
    bridge_distance_threshold = 1.5

    async def dream_cycle(duration_seconds=10.0) -> int  # Bridges created
```

### MultiWaveMemoryEngine

Main engine combining all modes:

```python
class MultiWaveMemoryEngine(SpringDynamicsEngine):
    def on_query(query_embedding) -> BetaWaveRecallResult
    async def ingest_stream(shard_stream, embedding_func)
    def get_statistics() -> Dict
```

---

## Streaming Ingestion

Ingest data from SpinningWheel sources while maintaining BETA mode:

```python
from hololoom.spinningWheel import YouTubeSpinner

# Create async shard stream
async def shard_generator():
    spinner = YouTubeSpinner()
    shards = await spinner.spin({'url': 'VIDEO_ID'})
    for shard in shards:
        yield shard

# Embedding function
def embed(text: str) -> np.ndarray:
    return embedder.encode(text)

# Ingest (stays in BETA during ingestion)
await engine.ingest_stream(shard_generator(), embed)
```

During ingestion:
- New nodes are connected to top-3 most similar existing nodes (cosine similarity > 0.7)
- Activation patterns are recorded for theta consolidation
- Mode stays in BETA until ingestion completes

---

## Statistics

```python
stats = engine.get_statistics()

print(stats)
# {
#     'mode': 'theta',
#     'minutes_since_last_query': 45.2,
#     'total_ingested': 1523,
#     'ingestion_active': False,
#     'consolidation_history_size': 87,
#     'total_updates': 12456,
#     'total_nodes': 5432,
#     'active_nodes': 23,
#     ...
# }
```

---

## Mode Timing Summary

| Mode | Trigger | Update Cycle | Purpose |
|------|---------|--------------|---------|
| **BETA** | Query or ingestion | 100ms | Active retrieval |
| **ALPHA** | 5-30 min idle | 125ms | Noise suppression |
| **THETA** | 30 min - 2 hr idle | 250ms | Co-activation consolidation |
| **DELTA** | >2 hr idle (70%) | 1000ms | Weak connection pruning |
| **REM** | >2 hr idle (30%) | 10s cycles | Creative bridging |

---

## Integration with HoloLoom

Multi-Wave Engine is used internally by:
- **UnifiedMemory**: Background consolidation
- **MemorySymphony**: Conductor coordination
- **HoloLoom API**: experience()/recall()/reflect() operations

You typically don't interact with it directly:

```python
from hololoom import hololoom

async with HoloLoom() as loom:
    # Multi-wave engine runs in background
    await loom.experience("Thompson Sampling balances exploration...")

    # After 30 minutes idle, theta consolidation happens automatically
    # After 2 hours idle, delta pruning and REM dreaming occur

    # Query wakes it back to BETA
    memories = await loom.recall("exploration")
```

---

## When Each Mode Matters

### BETA - For Active Sessions
- Real-time query handling
- Streaming data ingestion
- Maximum responsiveness needed

### ALPHA - For Short Breaks
- User taking a coffee break
- System waiting for next query
- Cleans up noise from recent activity

### THETA - For Consolidation
- End of work session
- Lunch break
- Converts usage patterns to permanent knowledge

### DELTA - For Deep Maintenance
- Overnight
- Weekend
- Removes forgotten connections, strengthens important ones

### REM - For Creativity
- Unexpected connections
- Cross-domain insights
- Bridges between distant concepts

---

## Example: Full Cycle

```python
# Morning: User starts working (BETA)
09:00 - Query: "Thompson Sampling"
09:05 - Query: "Exploration vs Exploitation"
09:10 - Query: "Multi-armed bandits"
# → Activation patterns recorded for consolidation

# Coffee break (ALPHA → THETA)
09:15 - Mode switches to ALPHA (5 min idle)
09:45 - Mode switches to THETA (30 min idle)
# → Theta consolidation strengthens Thompson-Exploration-Bandits triangle

# Lunch (continues THETA)
12:00 - Still in THETA
# → More consolidation of morning patterns

# Afternoon session (BETA)
13:00 - Query: "UCB algorithm"
# → Wakes to BETA, new patterns recorded

# Overnight (DELTA + REM)
18:00 - User leaves
19:00 - THETA mode
20:00 - DELTA mode (pruning weak connections)
22:00 - REM mode (dreaming)
# → Creates bridge between "Thompson Sampling" and "UCB" (activated together, distant embeddings)
# → This is a creative insight the user didn't explicitly make!

# Next morning (BETA)
09:00 - Query: "Compare bandit algorithms"
# → Bridge connection helps retrieve both Thompson and UCB together
```

---

## See Also

- [spring_dynamics.py](spring_dynamics.py) - Core physics engine
- [awareness_graph.py](awareness_graph.py) - Activation tracking
- [consolidation.py](consolidation.py) - Memory consolidation system
