# 🧪 Verification Plan: Semantic Calculus

**Status: Built, needs testing**

Before adding new features (like voice mode), verify what we've built works correctly.

## Phase 1: Basic Functionality ✅ / ❌

### Test 1: True Matryoshka Nesting
```bash
python matryoshka_streaming.py
```

**Verify:**
- [ ] Word-level uses 96D embeddings
- [ ] Phrase-level uses 192D embeddings
- [ ] Sentence-level uses 384D embeddings
- [ ] Paragraph-level uses 384D embeddings
- [ ] Word-level is ~4x faster than paragraph-level
- [ ] Performance report shows speedup
- [ ] All scales produce valid semantic projections

**Expected output:**
```
Word (96D → 16D):   0.5ms avg
Phrase (192D → 64D): 1.0ms avg
Sentence (384D → 128D): 1.8ms avg
Paragraph (384D → 244D): 2.1ms avg

Speedup: Word-level is 4.1x faster than paragraph-level
```

---

### Test 2: Recursive Composition
```bash
python recursive_matryoshka.py
```

**Verify:**
- [ ] Word-level shows no inheritance (`inherited_from: []`)
- [ ] Phrase-level shows word inheritance (`inherited_from: ['word']`)
- [ ] Sentence-level shows phrase+word inheritance
- [ ] Paragraph-level shows all three levels of inheritance
- [ ] Fusion strategies work (try WEIGHTED_SUM, CONCATENATE, RESIDUAL)
- [ ] Inherited semantics actually influence higher-level projections

**Expected output:**
```
word         (16D) ← inherits from:
phrase       (64D) ← inherits from: word
sentence    (128D) ← inherits from: phrase, word
paragraph   (244D) ← inherits from: sentence, phrase, word
```

---

### Test 3: Multi-Projection Spaces
```bash
python multi_projection.py
```

**Verify:**
- [ ] All projection spaces initialize correctly
- [ ] Semantic projection shows narrative dimensions
- [ ] Emotion projection shows affective dimensions
- [ ] Archetype projection shows Jungian archetypes
- [ ] Cross-projection agreement metric is reasonable (0-1)
- [ ] Dominant projection identified correctly
- [ ] All projections update in real-time

**Expected output:**
```
semantic (mag: 12.34) | Semantic: Heroism (0.82), Courage (0.71)
emotion  (mag:  8.21) | Emotion: Fear (0.45), Excitement (0.38)
archetype (mag: 10.15) | Archetype: Hero-Archetype (0.88)
```

---

## Phase 2: Integration Testing

### Test 4: Real Text Analysis

Create test script:
```python
# test_real_text.py
import asyncio
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

async def test_hemingway_style():
    """Test on clean Hemingway-style text."""
    text = """
    The old man sat in the boat. The sun was hot. He waited for the fish.
    The line pulled. He held tight. The fish was big. He pulled harder.
    """

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    calculator = MatryoshkaSemanticCalculus(embedder, snapshot_interval=0.5)

    snapshots = []
    calculator.on_snapshot(lambda s: snapshots.append(s))

    async def word_stream():
        for word in text.split():
            yield word
            await asyncio.sleep(0.05)

    async for snapshot in calculator.stream_analyze(word_stream()):
        pass

    # Verify Hemingway characteristics
    avg_momentum = sum(s.narrative_momentum for s in snapshots) / len(snapshots)
    avg_complexity = sum(s.complexity_index for s in snapshots) / len(snapshots)

    print(f"Hemingway style:")
    print(f"  Momentum: {avg_momentum:.3f} (expect > 0.7)")
    print(f"  Complexity: {avg_complexity:.3f} (expect < 0.4)")

    assert avg_momentum > 0.6, "Hemingway should have high momentum"
    assert avg_complexity < 0.5, "Hemingway should have low complexity"
    print("✅ Hemingway style verified!")

async def test_complex_style():
    """Test on complex literary text."""
    text = """
    The memory of it came to him in fragments, disjointed and shimmering like
    heat on summer pavement, the way she had looked that day or perhaps another
    day altogether, time having collapsed into itself the way it does when you're
    old and the past is more vivid than the present.
    """

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    calculator = MatryoshkaSemanticCalculus(embedder, snapshot_interval=0.5)

    snapshots = []
    calculator.on_snapshot(lambda s: snapshots.append(s))

    async def word_stream():
        for word in text.split():
            yield word
            await asyncio.sleep(0.05)

    async for snapshot in calculator.stream_analyze(word_stream()):
        pass

    # Verify complex characteristics
    avg_momentum = sum(s.narrative_momentum for s in snapshots) / len(snapshots)
    avg_complexity = sum(s.complexity_index for s in snapshots) / len(snapshots)

    print(f"Complex style:")
    print(f"  Momentum: {avg_momentum:.3f} (expect < 0.5)")
    print(f"  Complexity: {avg_complexity:.3f} (expect > 0.6)")

    assert avg_momentum < 0.6, "Complex prose should have lower momentum"
    assert avg_complexity > 0.5, "Complex prose should have higher complexity"
    print("✅ Complex style verified!")

if __name__ == "__main__":
    asyncio.run(test_hemingway_style())
    asyncio.run(test_complex_style())
```

**Run:**
```bash
python test_real_text.py
```

**Verify:**
- [ ] Hemingway-style text shows high momentum (>0.7)
- [ ] Hemingway-style text shows low complexity (<0.4)
- [ ] Complex literary text shows low momentum (<0.5)
- [ ] Complex literary text shows high complexity (>0.6)
- [ ] Tests pass consistently across multiple runs

---

### Test 5: Performance Benchmarks

```python
# test_performance.py
import asyncio
import time
import numpy as np

async def benchmark_speeds():
    from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    text = " ".join(["word"] * 200)  # 200 words

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    calculator = MatryoshkaSemanticCalculus(embedder, snapshot_interval=0.5)

    async def word_stream():
        for word in text.split():
            yield word

    start = time.time()
    async for snapshot in calculator.stream_analyze(word_stream()):
        pass
    duration = time.time() - start

    words_per_second = 200 / duration

    print(f"Performance:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Speed: {words_per_second:.1f} words/second")
    print(f"  Per-word: {duration*1000/200:.2f}ms")

    # Print per-scale timing
    calculator.print_performance_report()

    # Verify reasonable performance
    assert words_per_second > 50, "Should process at least 50 words/second"
    print("✅ Performance acceptable!")

if __name__ == "__main__":
    asyncio.run(benchmark_speeds())
```

**Run:**
```bash
python test_performance.py
```

**Verify:**
- [ ] Processes at least 50 words/second
- [ ] Word-level is 3-5x faster than paragraph-level
- [ ] No memory leaks over 1000+ words
- [ ] CPU usage is reasonable (<80%)

---

## Phase 3: Edge Cases

### Test 6: Edge Cases

```python
# test_edge_cases.py

async def test_empty_text():
    """Test with empty input."""
    # Should not crash
    pass

async def test_single_word():
    """Test with just one word."""
    # Should produce valid snapshot
    pass

async def test_very_long_text():
    """Test with 10,000+ words."""
    # Should not run out of memory
    pass

async def test_special_characters():
    """Test with emojis, unicode, etc."""
    # Should handle gracefully
    pass

async def test_rapid_streaming():
    """Test with very high word rate."""
    # Should keep up or buffer appropriately
    pass
```

**Verify:**
- [ ] Empty text doesn't crash
- [ ] Single word produces valid snapshot
- [ ] Long text (10k words) completes without OOM
- [ ] Special characters handled correctly
- [ ] Rapid streaming doesn't drop snapshots

---

## Phase 4: Quality Validation

### Test 7: Semantic Quality

**Manual verification with known texts:**

1. **Hero's Journey Text**
   - Should show high Heroism, Transformation, Courage
   - Should detect narrative arc (momentum changes)
   - Should identify key moments (acceleration spikes)

2. **Emotional Text**
   - Grief text should show high Grief, low Hope
   - Joy text should show high Joy, high Warmth
   - Fear text should show high Fear, high Tension

3. **Technical Text**
   - Should show high Complexity, low Emotion
   - Should show steady momentum (not dramatic)
   - Should be consistent (low variance)

**Create validation suite:**
```python
# test_semantic_quality.py
# Load ground-truth labeled texts
# Verify semantic projections match expected dimensions
```

---

## Phase 5: Documentation Verification

### Test 8: Documentation Accuracy

- [ ] All code examples in README.md run without errors
- [ ] All demos produce expected output
- [ ] API matches documentation
- [ ] Performance claims are accurate (benchmarked)
- [ ] File structure matches docs

---

## Known Issues to Check

### Potential Problems:

1. **Embedding model availability**
   - Does it gracefully fall back if sentence-transformers not installed?
   - Does it handle model download failures?

2. **Memory growth**
   - Do deques properly limit size?
   - Are old snapshots garbage collected?

3. **Numerical stability**
   - Do normalized vectors handle zero-magnitude cases?
   - Are divisions by zero protected?

4. **Threading/async**
   - Are async operations properly awaited?
   - Do background tasks clean up?

---

## Test Execution Log

Create a log as you test:

```markdown
## Test Results

### 2025-01-XX - Initial Testing

**Test 1: Matryoshka Nesting**
- Status: ✅ PASS
- Word-level: 0.48ms (4.2x faster than para)
- Notes: Performance as expected

**Test 2: Recursive Composition**
- Status: ✅ PASS
- Inheritance working correctly
- Notes: WEIGHTED_SUM fusion works best

**Test 3: Multi-Projection**
- Status: ⚠️ PARTIAL
- All projections work individually
- Notes: Agreement metric needs calibration

... (continue for all tests)
```

---

## Future Feature Ideas (After Verification)

**Voice Mode / Prosody Generation:**
- Semantic velocity → speaking pace
- Dimension spikes → pauses, sighs, tone changes
- Momentum → overall cadence
- Complexity → vocal texture (smooth vs jagged)

**Save this for later once core system is verified!**

---

## Priority Order

1. **Phase 1** (Basic Functionality) - DO THIS FIRST
2. **Phase 2** (Integration) - Verify it works on real text
3. **Phase 4** (Quality) - Ensure semantic projections make sense
4. **Phase 3** (Edge Cases) - Harden the system
5. **Phase 5** (Documentation) - Polish

---

**Remember: Verify, verify, verify before adding features!** 🧪
