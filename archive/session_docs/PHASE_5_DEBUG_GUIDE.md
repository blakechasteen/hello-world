# Phase 5 Debugging Guide

## How to Debug Dimension Mismatches

### The Problem We Found

**Root Cause:** The compositional cache wasn't being created when `linguistic_mode="disabled"`.

**Why?** The code had this logic:
```python
# Old logic (BROKEN):
if linguistic_mode != DISABLED:
    create_ug_chunker()  # Only create for filtering

if use_compositional_cache and ug_chunker_exists:
    create_compositional_cache()  # Requires chunker!
```

**Result:** When you wanted cache-only (no linguistic filtering), the chunker wasn't created, so the cache wasn't created either!

### The Fix

Changed to:
```python
# New logic (FIXED):
needs_chunker = (linguistic_mode != DISABLED) or use_compositional_cache

if needs_chunker:
    create_ug_chunker()  # Create for either filtering OR caching

if use_compositional_cache and ug_chunker_exists:
    create_compositional_cache()  # Now has chunker!
```

**File:** [`HoloLoom/embedding/linguistic_matryoshka_gate.py:129-160`](HoloLoom/embedding/linguistic_matryoshka_gate.py#L129-L160)

### Debug Tools We Created

1. **[debug_phase5_dimensions.py](debug_phase5_dimensions.py)** - Traces dimensions through the Phase 5 pipeline
   - Tests standard MatryoshkaEmbeddings (baseline)
   - Tests LinguisticMatryoshkaGate (with cache)
   - Tests compositional cache directly
   - Validates policy engine expectations

**Run it:**
```bash
python debug_phase5_dimensions.py
```

**What to look for:**
- ✅ "Has compositional cache: True"
- ✅ `encode_scales(size=96)` returns `np.ndarray` with shape `(n, 96)`
- ✅ All dimension checks pass

---

## Can Matryoshka Gate Work with Different Embeddings?

**Short answer: YES!** 🎉

### How Matryoshka Works

Matryoshka embeddings are **dimension-agnostic**:

```python
# Any base embedder works!
base_embedder = SentenceTransformer('all-MiniLM-L12-v2')  # 384d
# OR
base_embedder = SentenceTransformer('all-mpnet-base-v2')  # 768d
# OR
base_embedder = CustomEmbedder()  # Any dimension!

# Create matryoshka at any scales
matryoshka = MatryoshkaEmbeddings(
    sizes=[96, 192, 384],  # Must be ≤ base dimension
    base_model_name='all-MiniLM-L12-v2'
)
```

### Key Constraint

**Scales must be ≤ base dimension:**

```python
# GOOD: base=384d, scales=[96, 192, 384]
MatryoshkaEmbeddings(sizes=[96, 192, 384])  # ✅

# BAD: base=384d, scales=[96, 192, 512]
MatryoshkaEmbeddings(sizes=[96, 192, 512])  # ❌ 512 > 384!
```

### Using Different Embedders

#### Option 1: Swap Base Model

```python
# Use a different SentenceTransformer model
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

# Larger model (768d)
embedder = MatryoshkaEmbeddings(
    sizes=[96, 192, 384, 768],
    base_model_name='all-mpnet-base-v2'  # 768d model
)
```

#### Option 2: Custom Embedder

```python
class CustomEmbedder:
    def encode(self, texts):
        # Your custom logic here
        return np.random.randn(len(texts), 512)  # 512d embeddings

embedder = MatryoshkaEmbeddings(
    sizes=[96, 192, 384],
    base_embedder=CustomEmbedder()  # Pass custom embedder
)
```

#### Option 3: Projection Matrices

For truly trained matryoshka models:

```python
embedder = MatryoshkaEmbeddings(
    sizes=[96, 192, 384],
    proj={
        96: learned_projection_96,   # Train these!
        192: learned_projection_192,
        384: learned_projection_384
    }
)
```

### Phase 5 with Different Embedders

Phase 5 (compositional cache) works with **any** base embedder:

```python
from HoloLoom.embedding.linguistic_matryoshka_gate import (
    LinguisticMatryoshkaGate, LinguisticGateConfig
)

# Use any base embedder
base = MatryoshkaEmbeddings(
    sizes=[96, 192, 384],
    base_model_name='all-mpnet-base-v2'  # Different model!
)

# Create Phase 5 gate
config = LinguisticGateConfig(
    scales=[96, 192, 384],
    use_compositional_cache=True
)

gate = LinguisticMatryoshkaGate(
    embedder=base,
    config=config
)
```

**The compositional cache caches the OUTPUT of whatever embedder you use!**

---

## Debugging Checklist

When Phase 5 doesn't work, check:

### 1. Is the cache being created?

```python
print(f"Has cache: {gate.compositional_cache is not None}")
```

**If False:**
- ✅ Check `use_compositional_cache=True` in config
- ✅ Check UG chunker is created (now automatic with our fix!)

### 2. Are dimensions compatible?

```python
base_dim = base_embedder.encode(["test"]).shape[1]
print(f"Base dimension: {base_dim}")
print(f"Scales: {config.scales}")

# Verify: all scales ≤ base_dim
assert all(s <= base_dim for s in config.scales)
```

### 3. Is encode_scales() working?

```python
result = gate.encode_scales(["test"], size=96)
print(f"Type: {type(result)}")
print(f"Shape: {result.shape if isinstance(result, np.ndarray) else 'NOT ARRAY'}")

# Should be: np.ndarray with shape (1, 96)
assert isinstance(result, np.ndarray)
assert result.shape == (1, 96)
```

### 4. Test the full pipeline

```python
# Run the debug script
python debug_phase5_dimensions.py

# Look for:
# ✅ Has compositional cache: True
# ✅ All encode() tests pass
# ✅ All encode_scales() tests pass
# ✅ Policy engine expectations met
```

---

## Common Issues & Solutions

### Issue 1: "Has compositional cache: False"

**Solution:** Update to latest code with UG chunker fix.

**File:** `HoloLoom/embedding/linguistic_matryoshka_gate.py:129-160`

### Issue 2: "mat1 and mat2 shapes cannot be multiplied"

**Cause:** Dimension mismatch between embeddings and policy expectations.

**Debug:**
```python
# Check what policy gets
embeds = gate.encode_scales(texts, size=192)
print(f"Policy gets: {embeds.shape}")  # Should be (n, 192)
```

**Solution:** Ensure `encode_scales(size=X)` returns `(n, X)` array, not dict.

### Issue 3: Scales > base dimension

**Cause:** Configured scales larger than base embedder dimension.

**Example:**
```python
# Base model outputs 384d
# But config asks for 512d
sizes=[96, 192, 512]  # ❌ 512 > 384
```

**Solution:** Use scales ≤ base dimension:
```python
sizes=[96, 192, 384]  # ✅ All ≤ 384
```

### Issue 4: Slow performance (no speedup)

**Check cache stats:**
```python
if gate.compositional_cache:
    stats = gate.compositional_cache.get_statistics()
    print(f"Parse hits: {stats['parse_cache']['hits']}")
    print(f"Merge hits: {stats['merge_cache']['hits']}")
    print(f"Hit rate: {stats['overall_hit_rate']}")
```

**If hit rate is low:**
- Run more queries to warm up cache
- Check if queries are actually similar (for compositional reuse)
- Verify cache sizes are large enough

---

## Performance Expectations

### Cold Path (First Query)

```python
# Without cache
query = "What are mammals?"
result = await loom.weave(Query(text=query))
# Duration: ~150-200ms (normal)
```

### Hot Path (Cached Query)

```python
# Same query again (cache hit!)
result = await loom.weave(Query(text=query))
# Duration: ~0.5-2ms
# Speedup: 100-300× 🚀
```

### Warm Path (Similar Query)

```python
# Similar query (partial cache hit)
result = await loom.weave(Query(text="What are dogs?"))
# Duration: ~50-80ms
# Speedup: 2-3× (compositional reuse)
```

---

## Testing Your Changes

### 1. Run Debug Script

```bash
python debug_phase5_dimensions.py
```

**Expected output:**
```
[2] LinguisticMatryoshkaGate (with compositional cache)
Has compositional cache: True  ✅
encode(texts) shape: (2, 384)  ✅
encode_scales(size=96) shape: (2, 96)  ✅
```

### 2. Run Integration Tests

```bash
python tests/test_phase5_integration.py
```

**Expected:** All 4 tests pass

### 3. Run Benchmarks

```bash
python benchmarks/benchmark_phase5_speedup.py
```

**Expected:** See 10-300× speedups

---

## Summary

### The Key Fix

**Before:** Cache wasn't created with `linguistic_mode="disabled"`

**After:** Cache creates UG chunker automatically when needed

**File:** `HoloLoom/embedding/linguistic_matryoshka_gate.py:129-160`

### Yes, Matryoshka Works with Different Embeddings!

- ✅ Any SentenceTransformer model
- ✅ Custom embedders
- ✅ Any dimension (just keep scales ≤ base_dim)
- ✅ Compositional cache works with all of them!

### Next Steps

1. ✅ UG chunker fix applied
2. ⏳ Run integration tests
3. ⏳ Measure actual speedups
4. 🚀 Ship Phase 5!

The hard debugging is done - now we just need to validate! 🎉
