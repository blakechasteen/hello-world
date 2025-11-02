# Phase 5 - Next Debugging Steps

## Current Status

### ✅ Fixed: Compositional Cache Creation
- **Problem:** Cache wasn't created with `linguistic_mode="disabled"`
- **Solution:** UG chunker now creates automatically when cache enabled
- **File:** `HoloLoom/embedding/linguistic_matryoshka_gate.py:129-160`
- **Result:** Cache creation works! ✅

### ⚠️ Remaining Issue: Dimension Mismatch in Policy

**Error:** `mat1 and mat2 shapes cannot be multiplied (1x192 and 96x8)`

**Analysis:**
- mat1 shape: `(1, 192)` - embeddings from linguistic gate
- mat2 shape: `(96, 8)` - weight matrix in policy
- **Problem:** Policy expects 96d input but receives 192d

**Why this happens:**
1. FAST mode uses scales `[96, 192]`
2. Pattern selection chooses `BARE` for simple queries (scale=96)
3. But linguistic gate returns embeddings at 192d (FAST mode mem_dim)
4. Policy has projection matrices for 96d
5. Matrix multiplication fails!

## Root Cause Analysis

The issue is a **pattern-scale mismatch**:

```python
# In weaving_orchestrator.py
pattern_spec = self.loom_command.select_pattern(query)
# Returns: PatternSpec(mode=BARE, scales=[96])

# But policy uses:
mem_dim = self.cfg.mem_dim  # 192 for FAST mode

# So we get:
embeddings = linguistic_gate.encode_scales(texts, size=192)  # 192d
policy_weights = proj[96]  # Expects 96d!
# ❌ MISMATCH!
```

## Solution Options

### Option 1: Use Pattern Scales (Recommended)

Ensure embeddings match the selected pattern's scales:

```python
# In weaving_orchestrator.py around line 968
pattern_spec = self.loom_command.select_pattern(query)

# Use pattern's scale (not config mem_dim)
if self.linguistic_gate and self.cfg.enable_linguistic_gate:
    pattern_embedder = self.linguistic_gate
    # Configure for pattern's scales
    pattern_embedder.config.scales = pattern_spec.scales
```

**File to modify:** `HoloLoom/weaving_orchestrator.py:963-981`

### Option 2: Force Consistent Scales

Disable pattern auto-selection when Phase 5 enabled:

```python
# In config
if enable_linguistic_gate:
    # Force FAST pattern (consistent scales)
    loom_command_auto_select = False
    loom_command_default = "fast"
```

### Option 3: Dynamic Policy Projection

Make policy adapt to embedding dimensions:

```python
# In policy/unified.py
def decide(self, features, context):
    # Detect embedding dimension
    emb_dim = context.embeddings.shape[1]

    # Use matching projection
    if emb_dim in self.proj:
        proj_matrix = self.proj[emb_dim]
    else:
        # Slice or pad
        proj_matrix = self.proj[self.mem_dim][:emb_dim, :]
```

## Recommended Fix

**Use Option 1** - it's cleanest and respects pattern selection.

### Step 1: Read weaving_orchestrator.py

```bash
# Around line 963-981 where linguistic gate is used
```

### Step 2: Ensure dimension consistency

```python
# OLD:
if self.linguistic_gate and self.cfg.enable_linguistic_gate:
    pattern_embedder = self.linguistic_gate

# NEW:
if self.linguistic_gate and self.cfg.enable_linguistic_gate:
    pattern_embedder = self.linguistic_gate
    # Ensure gate uses pattern scales, not config scales
    pattern_embedder.config.scales = pattern_spec.scales
```

### Step 3: Test

```bash
python tests/test_phase5_integration.py
```

## Debug Commands

### Check what's happening:

```python
# In weaving_orchestrator.py, add logging:
logger.info(f"Pattern: {pattern_spec.mode}, scales: {pattern_spec.scales}")
logger.info(f"Config mem_dim: {self.cfg.mem_dim}")
logger.info(f"Linguistic gate scales: {self.linguistic_gate.config.scales}")
```

### Verify dimensions:

```python
# After encode_scales:
logger.info(f"Embeddings shape: {embeddings.shape}")
logger.info(f"Policy expects: {self.policy.mem_dim}")
```

## Timeline

1. **Now:** Understand the pattern-scale mismatch
2. **Next:** Implement Option 1 (use pattern scales)
3. **Then:** Run tests
4. **Finally:** Measure speedups!

## Key Insight

Phase 5 integration is **architecturally complete**. The remaining issue is a **configuration alignment** problem:
- Pattern selection chooses scales dynamically
- Linguistic gate uses static config scales
- **Fix:** Make gate respect pattern scales

This is a **small fix** (~5 lines of code), not a fundamental architectural issue! 🎉

## Files to Check

1. `HoloLoom/weaving_orchestrator.py:963-981` - Where linguistic gate is used
2. `HoloLoom/loom/command.py` - Pattern selection logic
3. `HoloLoom/policy/unified.py:638` - Where dimension mismatch occurs

## Success Criteria

After fix:
- ✅ Pattern scales match embedding dimensions
- ✅ Policy receives correctly-sized embeddings
- ✅ All 4 integration tests pass
- ✅ Speedups measured (10-300×)

**We're very close!** 🚀
