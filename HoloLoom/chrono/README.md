# Chrono Trigger - Temporal Control System

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/chrono/`
**Code**: 548 lines across 2 files

---

## Overview

The **Chrono Trigger** is HoloLoom's temporal orchestrator that manages all time-dependent aspects of the weaving process. It controls when threads activate, how long operations run, the cadence of background processes, and the evolution of the system over time.

**Philosophy**: Time is the fourth dimension of weaving. The Chrono Trigger ensures threads activate at the right moment, computations complete on time, and the knowledge graph evolves through temporal decay and learning.

---

## Responsibilities

### 1. Temporal Control
**When to activate threads from Yarn Graph**
- Creates temporal windows defining activation boundaries
- Filters threads by recency, relevance, and decay
- Ensures fresh knowledge takes priority

### 2. Execution Limits
**How long operations can run**
- Enforces timeouts per pattern card (5s/30s/120s)
- Prevents runaway computations
- Graceful timeout handling

### 3. Rhythm (Heartbeat)
**Cadence of background processes**
- Reflection buffer flushes
- Pattern mining cycles
- Metrics export
- Cache cleanup

### 4. Halt Conditions
**When to stop**
- Confidence thresholds met
- Maximum iterations reached
- Timeout exceeded
- User interruption

### 5. Thread Decay
**Aging of knowledge**
- Exponential decay over time
- Priority to recent memories
- Prevents stale information dominance

### 6. System Evolution
**Learning and adaptation**
- Triggers learning cycles
- Manages model checkpoints
- Coordinates reflection periods

---

## Architecture

### File Structure

```
HoloLoom/chrono/
├── __init__.py       # 11 lines - Public exports
└── trigger.py        # 537 lines - ChronoTrigger implementation
```

### Core Classes

#### `ChronoTrigger`

The main temporal control system.

```python
from HoloLoom.chrono import ChronoTrigger
from datetime import timedelta

# Create trigger with default settings
trigger = ChronoTrigger(
    heartbeat_interval=60.0,      # 1 minute
    default_timeout=30.0,          # 30 seconds
    enable_decay=True,             # Thread aging
    decay_half_life=timedelta(days=7)  # 7-day half-life
)

# Fire trigger to create temporal window
window = trigger.fire(
    pattern="fast",
    query_time=datetime.now()
)

# Use window to filter threads
active_threads = trigger.filter_threads_by_window(
    all_threads,
    window
)
```

#### `TemporalWindow`

Defines time boundaries for thread activation.

```python
@dataclass
class TemporalWindow:
    start: datetime              # Window start time
    end: datetime                # Window end time
    recency_weight: float        # 0.0-1.0 (prefer recent?)
    decay_rate: float            # Exponential decay rate
    max_age: timedelta           # Maximum thread age
    query_timestamp: datetime    # When query was made
```

#### `ExecutionLimits`

Timeout and iteration constraints.

```python
@dataclass
class ExecutionLimits:
    timeout_seconds: float       # Max execution time
    max_iterations: int          # Max processing cycles
    confidence_threshold: float  # Stop if confidence exceeds
    enable_interrupts: bool      # Allow user cancellation
```

---

## Usage Examples

### Example 1: Basic Temporal Window

```python
from HoloLoom.chrono import ChronoTrigger
from datetime import datetime, timedelta

trigger = ChronoTrigger()

# Fire to create window
window = trigger.fire(
    pattern="fast",
    query_time=datetime.now()
)

print(f"Window: {window.start} → {window.end}")
print(f"Recency weight: {window.recency_weight}")
print(f"Max age: {window.max_age}")

# Window is typically:
# - 7 days lookback for FAST pattern
# - Exponential decay (half-life: 7 days)
# - Recency weight: 0.7 (prefer recent threads)
```

### Example 2: Thread Filtering by Recency

```python
from HoloLoom.memory.protocol import Memory

# Threads from Yarn Graph
threads = [
    Memory(content="Info from 1 day ago", timestamp=now - timedelta(days=1)),
    Memory(content="Info from 7 days ago", timestamp=now - timedelta(days=7)),
    Memory(content="Info from 30 days ago", timestamp=now - timedelta(days=30)),
]

# Create temporal window
window = trigger.fire(pattern="fast")

# Filter threads
active = trigger.filter_threads_by_window(threads, window)

# Result: Recent threads have higher activation weight
# - 1 day: weight ≈ 1.0 (fully active)
# - 7 days: weight ≈ 0.5 (half-life)
# - 30 days: weight ≈ 0.06 (mostly decayed)
```

### Example 3: Execution Limits

```python
# Configure strict timeouts for BARE pattern
limits = trigger.create_execution_limits(
    pattern="bare",
    timeout_override=5.0,  # 5 second limit
    max_iterations=3       # Maximum 3 cycles
)

print(f"Timeout: {limits.timeout_seconds}s")
print(f"Max iterations: {limits.max_iterations}")

# Use in processing loop
start_time = time.time()
iteration = 0

while iteration < limits.max_iterations:
    # Check timeout
    if time.time() - start_time > limits.timeout_seconds:
        raise TimeoutError("Execution limit exceeded")

    # Process...
    iteration += 1
```

### Example 4: Heartbeat for Background Tasks

```python
import asyncio

async def background_tasks():
    trigger = ChronoTrigger(heartbeat_interval=60.0)  # 1 minute

    while True:
        # Wait for heartbeat
        await trigger.wait_for_heartbeat()

        # Run periodic tasks
        await flush_reflection_buffer()
        await export_metrics()
        await cleanup_cache()

        print(f"Heartbeat: {datetime.now()}")

# Start background loop
asyncio.create_task(background_tasks())
```

### Example 5: Decay Function

```python
from datetime import datetime, timedelta

trigger = ChronoTrigger(
    enable_decay=True,
    decay_half_life=timedelta(days=7)
)

# Calculate decay weight for a thread
thread_age = timedelta(days=14)  # 2 weeks old
decay_weight = trigger.compute_decay_weight(thread_age)

print(f"Age: {thread_age.days} days")
print(f"Decay weight: {decay_weight:.3f}")
# Output: 0.25 (two half-lives = 0.5^2)

# Decay function: weight = 0.5 ^ (age / half_life)
```

---

## Integration with Orchestrator

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.chrono import ChronoTrigger

# Create chrono trigger
trigger = ChronoTrigger(
    default_timeout=30.0,
    enable_decay=True
)

# Orchestrator uses trigger for timing
orchestrator = WeavingOrchestrator(
    chrono_trigger=trigger,
    # ... other config
)

# During weaving:
# 1. Trigger fires, creates temporal window
# 2. Threads filtered by window (recent threads prioritized)
# 3. Execution limits enforced (timeout, max iterations)
# 4. Heartbeat manages background tasks
# 5. Decay weights applied to thread activation
```

---

## Temporal Window Creation

### Pattern-Specific Windows

```python
# BARE pattern: Fast, limited lookback
window = trigger.fire(pattern="bare")
- Lookback: 1 day
- Recency weight: 0.9 (highly prefer recent)
- Decay rate: fast (half-life: 1 day)

# FAST pattern: Balanced
window = trigger.fire(pattern="fast")
- Lookback: 7 days
- Recency weight: 0.7 (prefer recent)
- Decay rate: moderate (half-life: 7 days)

# FUSED pattern: Deep history
window = trigger.fire(pattern="fused")
- Lookback: 30 days
- Recency weight: 0.5 (balanced)
- Decay rate: slow (half-life: 30 days)
```

---

## API Reference

### Core Methods

#### `ChronoTrigger.__init__()`
```python
def __init__(
    self,
    heartbeat_interval: float = 60.0,
    default_timeout: float = 30.0,
    enable_decay: bool = True,
    decay_half_life: timedelta = timedelta(days=7)
)
```

#### `ChronoTrigger.fire()`
```python
def fire(
    self,
    pattern: str,  # "bare" / "fast" / "fused"
    query_time: Optional[datetime] = None
) -> TemporalWindow
```

#### `ChronoTrigger.filter_threads_by_window()`
```python
def filter_threads_by_window(
    self,
    threads: List[Memory],
    window: TemporalWindow
) -> List[Tuple[Memory, float]]  # (thread, activation_weight)
```

#### `ChronoTrigger.compute_decay_weight()`
```python
def compute_decay_weight(
    self,
    age: timedelta
) -> float  # 0.0-1.0
```

#### `ChronoTrigger.wait_for_heartbeat()`
```python
async def wait_for_heartbeat(self) -> None
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Fire trigger** | <0.1ms | Create temporal window |
| **Filter threads** | <1ms per 100 threads | Decay computation |
| **Check timeout** | <0.01ms | Simple time comparison |
| **Heartbeat wait** | Async | No blocking |

**Memory**: Negligible (<1KB per window)

---

## Dependencies

**Internal**:
```python
from HoloLoom.memory.protocol import Memory
from HoloLoom.performance.prometheus_metrics import metrics  # Optional
```

**External**:
```python
import asyncio
import math
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
```

---

## Summary

The Chrono Trigger provides:

✅ **Temporal windows** for thread activation
✅ **Execution limits** (timeouts, max iterations)
✅ **Thread decay** (exponential aging)
✅ **Heartbeat rhythm** for background tasks
✅ **Halt conditions** (confidence thresholds)
✅ **Pattern-specific timing** (BARE/FAST/FUSED)
✅ **Sub-millisecond overhead** (<1ms for filtering)

Time is the fourth dimension—the Chrono Trigger orchestrates when everything happens.
