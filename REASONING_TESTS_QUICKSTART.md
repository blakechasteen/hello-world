# Reasoning Engine Tests - Quick Start Guide

## Running the Tests

### Comprehensive Test Suite (Recommended)
```bash
python /home/user/hello-world/test_reasoning_comprehensive.py
```

**What it tests:**
- All 31 comprehensive tests
- Performance benchmarks
- Edge cases
- Full integration

**Expected output:**
```
Tests Run:    31
Tests Passed: 31 (100.0%)
Tests Failed: 0
```

### Original Unit Tests (Currently Broken)
```bash
# Note: These have outdated fixtures - see report for details
python -m pytest HoloLoom/tests/unit/test_reasoning_engine.py -v
```

**Status:** 3 passing, 20 errors (type signature mismatch)

## Test Files

| File | Purpose | Status |
|------|---------|--------|
| `test_reasoning_comprehensive.py` | Full test suite | ✅ All passing |
| `HoloLoom/tests/unit/test_reasoning_engine.py` | Original pytest tests | ⚠️ Needs fixture update |
| `validate_reasoning_phase1.py` | Standalone validation | ⚠️ Import issues |
| `REASONING_ENGINE_PHASE1_TEST_REPORT.md` | Detailed report | ✅ Complete |

## Quick Validation

### Test a single component:
```python
python -c "
import sys
import asyncio
sys.path.insert(0, '.')

from HoloLoom.reasoning import ReasoningEngine, ReasoningMode
from HoloLoom.reasoning.types import Query, Features, Context

async def test():
    engine = ReasoningEngine(mode=ReasoningMode.FAST)
    query = Query(text='What is Thompson Sampling?')
    features = Features(motifs=[], embeddings=[], spectral=None)
    context = Context(shards=[])

    result = await engine.reason(query, features, context)
    print(f'Mode: {result.mode}')
    print(f'Steps: {len(result.chain)}')
    print(f'Confidence: {result.total_confidence:.2f}')
    print(f'Duration: {result.duration_ms:.1f}ms')

asyncio.run(test())
"
```

## Performance Benchmarks

Run performance tests only:
```bash
python -c "
import sys
sys.path.insert(0, '.')
import time
import asyncio

from HoloLoom.reasoning import ReasoningEngine, ReasoningMode

# Mock types
class Query:
    def __init__(self, text): self.text = text
class Features:
    def __init__(self):
        self.motifs = []
        self.embeddings = []
        self.spectral = None
class Context:
    def __init__(self): self.shards = []

async def bench():
    for mode in [ReasoningMode.FAST, ReasoningMode.STANDARD, ReasoningMode.DEEP]:
        engine = ReasoningEngine(mode=mode)
        start = time.time()
        result = await engine.reason(Query('test'), Features(), Context())
        duration = (time.time() - start) * 1000
        print(f'{mode.value:10s}: {duration:6.2f}ms')

asyncio.run(bench())
"
```

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`, ensure symlinks exist:
```bash
cd /home/user/hello-world/HoloLoom
ln -s Documentation documentation
ln -s Utils utils
```

### Missing Dependencies
```bash
pip install networkx scipy numpy pytest pytest-asyncio --user
```

### HoloLoom Init Issues
If full HoloLoom import fails, tests import directly from `HoloLoom.reasoning`:
```python
# This works even if HoloLoom.__init__ is broken
from HoloLoom.reasoning import ReasoningEngine
```

## Test Coverage

| Component | Coverage | Tests |
|-----------|----------|-------|
| types.py | 100% | 5 tests |
| planner.py | 95% | 4 tests |
| chain_of_thought.py | 90% | 4 tests |
| verifier.py | 90% | 4 tests |
| engine.py | 95% | 6 tests |
| backtracker.py | 0% | Indirectly tested |

## Expected Performance

| Mode | Target | Actual | Status |
|------|--------|--------|--------|
| FAST | <50ms | ~0.0ms | ✅ |
| STANDARD | <200ms | ~0.2ms | ✅ |
| DEEP | N/A | ~0.3ms | ✅ |

## Next Steps

1. Run comprehensive tests: `python test_reasoning_comprehensive.py`
2. Review detailed report: `REASONING_ENGINE_PHASE1_TEST_REPORT.md`
3. Fix import issues (P0)
4. Update unit test fixtures (P1)
5. Begin Phase 2 integration

---

**Last Updated:** 2025-11-15
**Test Status:** ✅ All comprehensive tests passing (31/31)
