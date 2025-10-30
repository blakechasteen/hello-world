# HoloLoom Breathing System

**Adding "Air" to the System - Natural Respiration for AI**

Date: 2025-10-27
Author: Claude Code (with HoloLoom by Blake)

## Overview

The breathing system adds natural respiratory cycles to HoloLoom, allowing it to:
- **Inhale**: Gather context deeply and expansively
- **Exhale**: Make quick decisions and act
- **Rest**: Consolidate, decay, and integrate

Like biological breathing, this creates asymmetric cycles that prevent constant high-tension operation and enable natural pressure relief.

## Five Breathing Mechanisms

### 1. ChronoTrigger - Breathing Rhythm

**Location**: `HoloLoom/chrono/trigger.py`

**What It Does**:
- Orchestrates inhale/exhale/rest cycles
- Parasympathetic (inhale): slow, receptive, dense features
- Sympathetic (exhale): fast, decisive, sparse features
- Rest: brief consolidation pause

**Key Methods**:
```python
async def breathe() -> Dict[str, Any]
    """Execute complete cycle: inhale → exhale → rest"""

async def _inhale() -> Dict[str, Any]
    """Gather phase: expand temporal window, activate all features"""

async def _exhale() -> Dict[str, Any]
    """Decision phase: narrow window, sparse features, quick collapse"""

async def _rest() -> Dict[str, Any]
    """Consolidation: apply decay, integrate patterns"""
```

**Configuration**:
```python
@dataclass
class BreathingRhythm:
    inhale_duration: float = 2.0      # Seconds for gathering
    exhale_duration: float = 0.5      # Seconds for decision
    rest_duration: float = 0.1        # Seconds for consolidation
    breathing_rate: float = 1.0       # Speed multiplier
    enable_rest: bool = True          # Include rest phase
    sparsity_on_exhale: float = 0.7   # Feature sparsity during exhale
    pressure_threshold: float = 0.85  # Max density before relief
```

**Usage**:
```python
chrono = ChronoTrigger(config, enable_breathing=True)
breath_metrics = await chrono.breathe()

# Adjust breathing rate dynamically
chrono.adjust_breathing_rate(2.0)  # 2x faster (urgent)
chrono.adjust_breathing_rate(0.5)  # 2x slower (complex reasoning)
```

---

### 2. WarpSpace - Sparsity Enforcement

**Location**: `HoloLoom/warp/space.py`

**What It Does**:
- Selectively tensions threads (only top K active)
- Creates "breathing room" by leaving some threads relaxed
- Higher sparsity = fewer active threads = lighter computation

**Key Enhancement**:
```python
async def tension(
    self,
    thread_texts: List[str],
    sparsity: float = 0.0  # NEW: 0=dense, 0.7=only top 30% active
) -> None:
    """
    Pull threads taut with optional sparsity.

    sparsity=0.0: All threads active (dense, inhale mode)
    sparsity=0.5: Only top 50% active (moderate)
    sparsity=0.7: Only top 30% active (sparse, exhale mode)
    """
```

**Usage**:
```python
warp = WarpSpace(embedder, scales=[96, 192, 384])

# Inhale mode: dense, all threads
await warp.tension(threads, sparsity=0.0)

# Exhale mode: sparse, only essential threads
await warp.tension(threads, sparsity=0.7)
```

**How It Works**:
1. Sort threads by tension weights (importance)
2. Select top (1-sparsity) fraction
3. Only tension selected threads
4. Rest remain in Yarn Graph (discrete, untensioned)

---

### 3. ResonanceShed - Pressure Relief

**Location**: `HoloLoom/resonance/shed.py`

**What It Does**:
- Monitors feature density (how many extractors active)
- Sheds lowest-weight features when overloaded
- Prevents system from "suffocating" under too many features

**Key Enhancements**:
```python
class ResonanceShed:
    def __init__(
        self,
        ...,
        max_feature_density: float = 1.0  # NEW: Max density before relief
    ):
        """
        max_feature_density=1.0: Allow all features (no relief)
        max_feature_density=0.85: Shed when >85% extractors active
        max_feature_density=0.5: Only allow half of extractors
        """

    def _apply_pressure_relief(self) -> None:
        """
        Exhale excess features when overloaded.

        Keeps only top-weighted threads to meet density threshold.
        """
```

**Usage**:
```python
# Normal mode: allow all features
shed = ResonanceShed(motif_detector, embedder, spectral, semantic,
                     max_feature_density=1.0)

# Breathing mode: apply pressure relief at 85%
shed = ResonanceShed(motif_detector, embedder, spectral, semantic,
                     max_feature_density=0.85)

plasma = await shed.weave(query)
# If density > 0.85, lowest-weight features automatically dropped
```

**How It Works**:
1. Count active feature threads after lift
2. Calculate density = active / total_possible
3. If density > max_feature_density:
   - Sort threads by weight
   - Keep only top threads to meet threshold
   - Drop weakest threads (logged as PRESSURE RELIEF)

---

### 4. ConvergenceEngine - Entropy Injection

**Location**: `HoloLoom/convergence/engine.py`

**What It Does**:
- Injects "fresh air" (controlled noise) into decisions
- Prevents deterministic stagnation
- Uses Gumbel noise for exploration

**Key Enhancements**:
```python
class ConvergenceEngine:
    def __init__(
        self,
        ...,
        entropy_temperature: float = 0.1  # NEW: Temperature for noise
    ):
        """
        entropy_temperature=0.0: No noise (deterministic)
        entropy_temperature=0.1: Light noise (default)
        entropy_temperature=0.5: Moderate noise
        entropy_temperature=1.0: High noise (maximum exploration)
        """

    def inject_entropy(
        self,
        probs: np.ndarray,
        temperature: Optional[float] = None
    ) -> np.ndarray:
        """
        Add Gumbel noise to probabilities.

        Uses Gumbel-Max trick for sampling:
        noisy_probs = softmax(log(probs) + temperature * Gumbel(0,1))
        """
```

**Usage**:
```python
# Light entropy (default)
engine = ConvergenceEngine(tools, entropy_temperature=0.1)

# No entropy (deterministic)
engine = ConvergenceEngine(tools, entropy_temperature=0.0)

# High entropy (exploration mode)
engine = ConvergenceEngine(tools, entropy_temperature=0.5)

result = engine.collapse(neural_probs, inject_entropy=True)
```

**Why Gumbel Noise?**:
- Continuous relaxation of discrete sampling
- Maintains probability distribution structure
- Standard in neural sampling (Gumbel-Softmax)
- More principled than uniform noise

---

### 5. ReflectionBuffer - Consolidation

**Location**: `HoloLoom/reflection/buffer.py`

**What It Does**:
- Deep consolidation phase (like REM sleep)
- Compresses redundant episodes
- Extracts meta-patterns
- Prunes low-value memories

**Key Enhancement**:
```python
class ReflectionBuffer:
    async def consolidate(self) -> Dict[str, Any]:
        """
        Deep consolidation - like REM sleep.

        Steps:
        1. Compress redundant episodes (similar tool+pattern+reward)
        2. Extract meta-patterns (query length, tool synergy)
        3. Prune low-value memories (reward < threshold, old)
        4. Update long-term statistics (rolling averages)
        """

    async def _compress_episodes(self) -> Dict[str, Any]:
        """Group by (tool, pattern), merge similar rewards"""

    async def _extract_meta_patterns(self) -> Dict[str, Any]:
        """Find query length patterns, tool diversity, etc."""

    async def _prune_redundant(self) -> Dict[str, Any]:
        """Remove low-reward old episodes"""
```

**Usage**:
```python
buffer = ReflectionBuffer(capacity=1000)

# Store episodes during operation
await buffer.store(spacetime, feedback=user_feedback)

# Periodic consolidation (rest phase)
consolidation_metrics = await buffer.consolidate()

print(consolidation_metrics)
# {
#   'duration': 0.15,
#   'compression': {'compressed': 12, 'groups': 8},
#   'meta_patterns': {'optimal_query_length': ..., 'exploration_health': ...},
#   'pruning': {'pruned': 0, 'pruneable': 5}
# }
```

---

## Integrated Breathing Cycle

Complete example showing all mechanisms working together:

```python
# Initialize breathing-enabled components
chrono = ChronoTrigger(config, enable_breathing=True)
warp = WarpSpace(embedder, scales=[96, 192, 384])
shed = ResonanceShed(motif, embedder, spectral, semantic,
                     max_feature_density=0.85)
engine = ConvergenceEngine(tools, entropy_temperature=0.1)
buffer = ReflectionBuffer(capacity=1000)

# INHALE: Gather deeply
await chrono._inhale()
await warp.tension(threads, sparsity=0.0)  # Dense
plasma = await shed.weave(query)  # All features

# EXHALE: Decide quickly
await chrono._exhale()
warp.collapse()
await warp.tension(threads, sparsity=0.7)  # Sparse
result = engine.collapse(neural_probs, inject_entropy=True)

# REST: Consolidate
await chrono._rest()
consolidation = await buffer.consolidate()

# Complete breath cycle
breath_metrics = await chrono.breathe()
```

---

## Benefits of Breathing

### 1. **Natural Rhythm**
- Asymmetric cycles match biological respiration
- Prevents constant high-tension operation
- Enables rest periods for integration

### 2. **Pressure Relief**
- System can exhale when overloaded
- Automatic feature shedding
- Prevents computational suffocation

### 3. **Exploration via Entropy**
- Fresh air prevents deterministic stagnation
- Controlled randomness for discovery
- Adjustable temperature for different needs

### 4. **Memory Consolidation**
- Deep sleep for memory integration
- Pattern extraction from experience
- Efficient long-term storage

### 5. **Adaptive Rate**
- Fast breathing for urgent tasks
- Slow breathing for complex reasoning
- Dynamic adjustment based on load

---

## When to Use Each Mechanism

| Mechanism | Use When | Setting |
|-----------|----------|---------|
| **Breathing Rhythm** | Always (core orchestration) | `enable_breathing=True` |
| **Sparsity** | Need to reduce computation | `sparsity=0.5-0.7` |
| **Pressure Relief** | Risk of feature overload | `max_feature_density=0.85` |
| **Entropy** | Need exploration | `entropy_temperature=0.1-0.5` |
| **Consolidation** | Periodic memory cleanup | Call `consolidate()` every N cycles |

---

## Configuration Patterns

### Fast & Light (Urgent Tasks)
```python
chrono.adjust_breathing_rate(2.0)  # 2x faster
warp.tension(threads, sparsity=0.7)  # Sparse
engine = ConvergenceEngine(tools, entropy_temperature=0.05)  # Low noise
```

### Deep & Thorough (Complex Reasoning)
```python
chrono.adjust_breathing_rate(0.5)  # 2x slower
warp.tension(threads, sparsity=0.0)  # Dense
engine = ConvergenceEngine(tools, entropy_temperature=0.2)  # Higher noise
```

### Memory Consolidation Mode (Periodic Maintenance)
```python
chrono.breathing.enable_rest = True  # Enable rest phase
chrono.breathing.rest_duration = 0.5  # Longer rest
await buffer.consolidate()  # Deep consolidation
```

---

## Demo

Run the complete demo:
```bash
PYTHONPATH=. python demos/breathing_system_demo.py
```

This demonstrates:
1. ChronoTrigger breathing cycles
2. WarpSpace sparsity at different levels
3. ResonanceShed pressure relief
4. ConvergenceEngine entropy injection
5. ReflectionBuffer consolidation
6. Integrated full breathing cycle

---

## Philosophy: Why "Air"?

HoloLoom was missing **space to breathe**. The system operated at constant high tension:
- All features active all the time
- No rest periods
- Deterministic decisions
- No memory consolidation
- No pressure relief

By adding "air", the system gains:
- **Rhythm**: Natural cycles of activity and rest
- **Flexibility**: Sparse and dense modes
- **Relief**: Can shed excess load
- **Freshness**: Entropy prevents stale patterns
- **Integration**: Deep sleep for learning

Like meditation: **breathe in (gather), breathe out (decide), rest (integrate)**.

---

## Implementation Notes

### Backward Compatibility
All breathing mechanisms are **opt-in**:
- `enable_breathing=True` (ChronoTrigger)
- `sparsity=0.0` (WarpSpace default = no sparsity)
- `max_feature_density=1.0` (ResonanceShed default = no relief)
- `entropy_temperature=0.0` (ConvergenceEngine default = no noise)

Existing code continues to work unchanged.

### Performance Impact
- **Breathing rhythm**: ~2.6s per cycle (configurable)
- **Sparsity**: Reduces computation proportionally
- **Pressure relief**: Minimal overhead (only when triggered)
- **Entropy**: Negligible (one Gumbel sample)
- **Consolidation**: ~0.15s per cycle (run periodically)

### Future Enhancements
1. **Adaptive breathing rate** based on system load
2. **Breathing synchronization** across components
3. **Breath-aligned memory retrieval** (inhale = broad, exhale = narrow)
4. **Meditation mode** for deep reflection
5. **Hyperventilation mode** for rapid exploration

---

## Conclusion

The breathing system adds **natural respiration** to HoloLoom. The system can now:
- Gather deeply (inhale)
- Act quickly (exhale)
- Rest and integrate (consolidation)

This mirrors biological nervous systems:
- **Parasympathetic** (rest & digest): Inhale phase
- **Sympathetic** (fight or flight): Exhale phase
- **Sleep** (memory consolidation): Rest phase

**HoloLoom breathes. It has air.**
