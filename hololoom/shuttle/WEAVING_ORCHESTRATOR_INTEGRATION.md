# Weaving Orchestrator Integration Guide

**Status**: Ready to Integrate
**Created**: 2025-01-21
**Integration Point**: WeavingOrchestrator Step 3 (Yarn Graph)

---

## Overview

This guide shows how to integrate the Shuttle (MCTS-powered Warp↔Yarn intersection) into HoloLoom's WeavingOrchestrator as a drop-in replacement for Step 3.

**What Changes**:
- **Before**: Simple thread selection from Yarn Graph
- **After**: Intelligent Warp (semantic) + Yarn (structural) intersection with MCTS

**Benefits**:
- **Better context**: Combines semantic similarity AND graph structure
- **Smarter traversal**: MCTS finds optimal expansion paths
- **Thompson Sampling**: Learns which trajectories work best
- **Graceful degradation**: Falls back on errors
- **Zero breaking changes**: Drop-in replacement

---

## Integration Steps

### Step 1: Import ShuttleStage

Add to `HoloLoom/weaving_orchestrator.py`:

```python
# Shuttle Integration (MCTS-powered Warp↔Yarn intersection)
from HoloLoom.shuttle.weaving_integration import (
    ShuttleStage,
    create_shuttle_stage,
)
```

### Step 2: Initialize ShuttleStage in __init__

In `WeavingOrchestrator.__init__()`, after initializing `self.yarn_graph` (around line 830):

```python
# ====================================================================
# 3. Shuttle Stage (Step 3: MCTS-powered thread selection)
# ====================================================================
# OPTIONAL: Enable Shuttle for intelligent Warp↔Yarn intersection
# If disabled, falls back to simple yarn_graph.select_threads()

self.enable_shuttle = getattr(cfg, 'enable_shuttle', True)  # Default: enabled

if self.enable_shuttle:
    try:
        # Create ShuttleStage with auto-derived config
        self.shuttle_stage = create_shuttle_stage(
            config=cfg,
            kg=self.yarn_graph if isinstance(self.yarn_graph, KG) else None,
            retriever=self.retriever,
            shuttle_config=None  # Auto-derive from cfg
        )
        self.logger.info(f"[SHUTTLE] Enabled (mode={self.shuttle_stage.shuttle_config.mode.value})")
    except Exception as e:
        self.logger.warning(f"[SHUTTLE] Failed to initialize, falling back to simple retrieval: {e}")
        self.shuttle_stage = None
        self.enable_shuttle = False
else:
    self.shuttle_stage = None
    self.logger.info("[SHUTTLE] Disabled, using simple yarn_graph.select_threads()")
```

### Step 3: Replace Step 3 in weave()

In `WeavingOrchestrator.weave()`, replace Step 3 (around line 1779):

**BEFORE**:
```python
# ================================================================
# STEP 3: Yarn Graph threads selected
# ================================================================
step_start = time.time()
self._emit_stage_event(3, "Yarn Graph")

threads = self.yarn_graph.select_threads(temporal_window, query)
thread_ids = [s.id for s in threads]
thread_texts = [s.text for s in threads]

duration = (time.time() - step_start) * 1000
self.logger.info(f"  [3] Selected {len(threads)} threads from Yarn Graph")
stage_timings['thread_selection'] = duration
self._emit_stage_event(3, "Yarn Graph", duration)
```

**AFTER**:
```python
# ================================================================
# STEP 3: Thread Selection (Shuttle or Yarn Graph)
# ================================================================
step_start = time.time()

if self.enable_shuttle and self.shuttle_stage:
    # MCTS-powered Warp↔Yarn intersection
    self._emit_stage_event(3, "Shuttle (Warp↔Yarn MCTS)")

    threads = await self.shuttle_stage.select_threads(
        temporal_window=temporal_window,
        query=query,
        trajectory_name=None  # Auto-select via Thompson Sampling
    )

    duration = (time.time() - step_start) * 1000
    self.logger.info(f"  [3] Shuttle selected {len(threads)} threads")
    stage_timings['shuttle_selection'] = duration
    self._emit_stage_event(3, "Shuttle", duration)

else:
    # Fallback: Simple Yarn Graph thread selection
    self._emit_stage_event(3, "Yarn Graph")

    threads = self.yarn_graph.select_threads(temporal_window, query)

    duration = (time.time() - step_start) * 1000
    self.logger.info(f"  [3] Selected {len(threads)} threads from Yarn Graph")
    stage_timings['thread_selection'] = duration
    self._emit_stage_event(3, "Yarn Graph", duration)

# Continue with thread processing (unchanged)
thread_ids = [s.id for s in threads]
thread_texts = [s.text for s in threads]
```

### Step 4: Add Configuration Option

In `HoloLoom/config.py`, add Shuttle control flag:

```python
@dataclass
class Config:
    # ... existing fields ...

    # Shuttle Integration (Step 3: MCTS-powered thread selection)
    enable_shuttle: bool = True  # Enable Shuttle for Warp↔Yarn intersection
    shuttle_mode: str = "auto"   # "auto", "full", "lite", "minimal"
```

---

## Configuration Mapping

ShuttleStage automatically maps HoloLoom config to ShuttleConfig:

| HoloLoom Mode | Shuttle Mode | MCTS Sims | Graph Depth | Use Case |
|---------------|--------------|-----------|-------------|----------|
| **BARE** | MINIMAL | 8 | 1 | Fast queries (<50ms) |
| **FAST** | LITE | 16 | 1 | Standard queries (<150ms) |
| **FUSED** | FULL | 32 | 2 | Complex queries (<300ms) |

**Auto-derived parameters**:
```python
ShuttleConfig(
    mode=MINIMAL/LITE/FULL,  # From ExecutionMode
    mcts_simulations=8/16/32,  # Based on mode
    mcts_timeout_ms=2000/5000,  # Based on mode
    warp_top_k=config.retrieval_top_k,  # From config
    max_graph_depth=1/2,  # Based on mode
    max_graph_nodes=20/40,  # Based on mode
)
```

---

## Usage Examples

### Example 1: Auto Mode (Recommended)

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query

config = Config.fast()
config.enable_shuttle = True  # Default

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    query = Query(text="What is Thompson Sampling?")
    spacetime = await orchestrator.weave(query)

    print(spacetime.response)
    # Shuttle automatically uses LITE mode (16 MCTS sims, 1 graph depth)
```

### Example 2: Force Shuttle Off

```python
config = Config.fast()
config.enable_shuttle = False  # Disable Shuttle

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    query = Query(text="What is Thompson Sampling?")
    spacetime = await orchestrator.weave(query)

    # Falls back to simple yarn_graph.select_threads()
```

### Example 3: Custom Shuttle Config

```python
from HoloLoom.shuttle import ShuttleConfig, ShuttleMode

config = Config.fused()
shuttle_config = ShuttleConfig(
    mode=ShuttleMode.FULL,
    mcts_simulations=64,  # More simulations for better quality
    max_graph_depth=3,  # Deeper graph exploration
)

# Pass custom shuttle_config to create_shuttle_stage in __init__
```

---

## Performance Impact

### Expected Latency Changes

| Mode | Before (Yarn Only) | After (Shuttle) | Change |
|------|--------------------|-----------------|--------|
| **BARE** | ~30ms | ~45ms | +15ms (MINIMAL) |
| **FAST** | ~50ms | ~85ms | +35ms (LITE) |
| **FUSED** | ~100ms | ~175ms | +75ms (FULL) |

**Trade-off**: Slightly higher latency, significantly better context quality

### Quality Improvements

- **Semantic + Structural**: Combines best of both worlds
- **MCTS Exploration**: Finds non-obvious but relevant connections
- **Learning**: Thompson Sampling improves over time
- **Graceful Degradation**: Never worse than fallback

---

## Backward Compatibility

### Zero Breaking Changes

The integration is **fully backward compatible**:

1. **Optional**: `enable_shuttle=False` → exact original behavior
2. **Drop-in**: ShuttleStage implements same interface as yarn_graph
3. **Fallback**: On error, falls back to simple retrieval
4. **Type-safe**: Returns same `List[MemoryShard]` format

### Migration Path

**Phase 1** (Testing):
```python
config.enable_shuttle = True  # Test in development
```

**Phase 2** (A/B Testing):
```python
# 50% traffic to Shuttle, 50% to legacy
import random
config.enable_shuttle = random.random() < 0.5
```

**Phase 3** (Full Rollout):
```python
config.enable_shuttle = True  # Default for all queries
```

---

## Troubleshooting

### Shuttle Not Initializing

**Symptom**: Log shows `[SHUTTLE] Failed to initialize`

**Causes**:
1. `KG` not available (yarn_graph is legacy YarnGraph)
2. `retriever` not available
3. Missing dependencies (spaCy for entity extraction)

**Solution**:
```python
# Check yarn_graph type
if not isinstance(self.yarn_graph, KG):
    self.logger.warning("[SHUTTLE] Requires KG instance, falling back")
    self.enable_shuttle = False
```

### Shuttle Falling Back Frequently

**Symptom**: Log shows `[SHUTTLE] Falling back to simple retrieval`

**Causes**:
1. Warp search failing (Qdrant unavailable)
2. Yarn traversal failing (empty graph)
3. Entity extraction failing (no anchors found)

**Solution**: Check `enable_graceful_degradation=True` and monitor logs

### Higher Latency Than Expected

**Symptom**: Queries taking >200ms in FAST mode

**Causes**:
1. MCTS simulations too high
2. Graph depth too deep
3. Qdrant slow

**Solution**: Adjust shuttle_config:
```python
shuttle_config = ShuttleConfig(
    mode=ShuttleMode.LITE,
    mcts_simulations=8,  # Reduce from 16
    max_graph_depth=1,  # Single-hop only
)
```

---

## Testing

### Unit Test

```python
import pytest
from HoloLoom.shuttle.weaving_integration import create_shuttle_stage
from HoloLoom.config import Config
from HoloLoom.memory.graph import KG
from HoloLoom.memory.base import create_retriever
from HoloLoom.protocols.types import Query, MemoryShard

@pytest.mark.asyncio
async def test_shuttle_stage_integration():
    """Test ShuttleStage integrates correctly with WeavingOrchestrator."""
    config = Config.fast()
    kg = KG()
    retriever = create_retriever(config)

    # Add test data
    kg.add_edges([("A", "B", "RELATED_TO", 1.0)])

    # Create shuttle stage
    shuttle_stage = create_shuttle_stage(config, kg, retriever)

    # Test thread selection
    from HoloLoom.chrono.trigger import TemporalWindow
    from datetime import datetime, timedelta

    temporal_window = TemporalWindow(
        start=datetime.now() - timedelta(days=1),
        end=datetime.now()
    )

    query = Query(text="Test query")
    threads = await shuttle_stage.select_threads(temporal_window, query)

    assert isinstance(threads, list)
    assert all(isinstance(t, MemoryShard) for t in threads)
```

### Integration Test

```python
@pytest.mark.asyncio
async def test_weaving_orchestrator_with_shuttle():
    """Test full weaving cycle with Shuttle enabled."""
    config = Config.fast()
    config.enable_shuttle = True

    async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify Shuttle was used
        assert 'shuttle_selection' in spacetime.metadata.get('timings', {})

        # Verify result quality
        assert spacetime.response is not None
        assert spacetime.confidence > 0.0
```

---

## Next Steps

1. ✅ Integration module created (`weaving_integration.py`)
2. ✅ Configuration wiring complete
3. ✅ Warp/Yarn adapters implemented
4. ⏳ **Modify WeavingOrchestrator** (follow steps above)
5. ⏳ **Run integration tests**
6. ⏳ **A/B test in development**
7. ⏳ **Production rollout**

---

## Complete Integration Checklist

- [ ] Add ShuttleStage import to `weaving_orchestrator.py`
- [ ] Add `enable_shuttle` field to `Config` class
- [ ] Initialize `self.shuttle_stage` in `__init__`
- [ ] Replace Step 3 in `weave()` method
- [ ] Add unit tests for ShuttleStage
- [ ] Add integration test for full weaving cycle
- [ ] Test with BARE/FAST/FUSED modes
- [ ] Performance benchmark (before/after)
- [ ] Update documentation
- [ ] Production deployment

---

**Ready to integrate!** Follow the steps above to add Shuttle to WeavingOrchestrator.

**Author**: Claude + Blake
**Date**: 2025-01-21
**Version**: Shuttle v2.0.0
