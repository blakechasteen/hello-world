# HoloLoom Test Suite

**Purpose**: Comprehensive test coverage for all HoloLoom systems
**Status**: 387+ tests, ~40% coverage, all passing
**Organization**: 3-tier structure (unit, integration, e2e)

## Overview

The test suite validates all HoloLoom components through **387+ tests** organized into three tiers:

- **Unit Tests** (6 files, 244+ tests): Fast isolated component testing (<500ms)
- **Integration Tests**: Multi-component testing (<2s)
- **E2E Tests** (9 files, 143 tests): Full pipeline testing (<30s)

## Test Organization

```
hololoom/tests/
├── conftest.py         # Central fixtures, mocks, performance budgets
├── unit/               # Fast isolated tests (<500ms)
│   ├── test_config.py
│   ├── test_weaving_shuttle.py
│   ├── test_unified_policy.py
│   ├── test_embedding_spectral.py
│   ├── test_memory_graph.py
│   └── test_memory_cache.py
├── integration/        # Multi-component tests (<2s)
│   ├── test_backends.py
│   └── [other integration tests]
└── e2e/                # Full pipeline tests (<30s)
    ├── test_error_handling.py
    ├── test_concurrent_queries.py
    ├── test_performance_profile.py
    ├── test_reflection_loop.py
    ├── test_memory_growth.py
    ├── test_persistence.py
    ├── test_edge_cases.py
    ├── test_cache_effectiveness.py
    └── test_integration_scenarios.py
```

## Running Tests

### All Tests
```bash
pytest hololoom/tests/ -v
```

### By Tier
```bash
# Unit tests (fast, <5s total)
pytest hololoom/tests/unit/ -v

# Integration tests (medium, <30s total)
pytest hololoom/tests/integration/ -v

# E2E tests (slow, <5min total)
pytest hololoom/tests/e2e/ -v
```

### Specific Test File
```bash
pytest hololoom/tests/e2e/test_concurrent_queries.py -v
```

### Specific Test
```bash
pytest hololoom/tests/e2e/test_concurrent_queries.py::TestParallelQueryExecution::test_concurrent_weave_calls -v
```

### With Performance Budget Warnings
```bash
pytest hololoom/tests/ -v -W default::UserWarning
```

## Test Statistics

| Tier | Files | Tests | Lines | Budget |
|------|-------|-------|-------|--------|
| **Unit** | 6 | 244+ | ~2,000 | <500ms per test |
| **Integration** | Many | Varies | Varies | <2s per test |
| **E2E** | 9 | 143 | ~3,000 | <30s per test |
| **Total** | 15+ | 387+ | ~5,000+ | Enforced |

## Unit Tests (6 files, 244+ tests)

### test_config.py (20/20 passing)
**Coverage**: Configuration system (BARE/FAST/FUSED modes)
```bash
pytest hololoom/tests/unit/test_config.py -v
```

### test_weaving_shuttle.py (14/14 passing)
**Coverage**: Backward compatibility shim
```bash
pytest hololoom/tests/unit/test_weaving_shuttle.py -v
```

### test_unified_policy.py (60+ assertions)
**Coverage**: Neural decision-making, Thompson Sampling
```bash
pytest hololoom/tests/unit/test_unified_policy.py -v
```

### test_embedding_spectral.py (32/32 passing, 60+ assertions)
**Coverage**: Matryoshka embeddings, spectral features
```bash
pytest hololoom/tests/unit/test_embedding_spectral.py -v
```

### test_memory_graph.py (80+ assertions)
**Coverage**: NetworkX knowledge graph operations
```bash
pytest hololoom/tests/unit/test_memory_graph.py -v
```

### test_memory_cache.py (70+ assertions)
**Coverage**: BM25 and vector retrieval with caching
```bash
pytest hololoom/tests/unit/test_memory_cache.py -v
```

## E2E Tests (9 files, 143 tests)

### test_error_handling.py (20 tests)
**Coverage**: Graceful degradation, network failures, invalid inputs
```bash
pytest hololoom/tests/e2e/test_error_handling.py -v
```

### test_concurrent_queries.py (20 tests)
**Coverage**: 100 concurrent queries, race conditions, deadlock prevention
```bash
pytest hololoom/tests/e2e/test_concurrent_queries.py -v
```

### test_performance_profile.py (15 tests)
**Coverage**: Latency, memory profiling, throughput, scaling
```bash
pytest hololoom/tests/e2e/test_performance_profile.py -v
```

### test_reflection_loop.py (20 tests)
**Coverage**: Thompson Sampling, pattern extraction, learning
```bash
pytest hololoom/tests/e2e/test_reflection_loop.py -v
```

### test_memory_growth.py (10 tests)
**Coverage**: Leak detection (500 queries), long sessions
```bash
pytest hololoom/tests/e2e/test_memory_growth.py -v
```

### test_persistence.py (10 tests)
**Coverage**: Checkpoint save/load, state recovery
```bash
pytest hololoom/tests/e2e/test_persistence.py -v
```

### test_edge_cases.py (17 tests)
**Coverage**: Unicode, 50K char inputs, emoji, pathological patterns
```bash
pytest hololoom/tests/e2e/test_edge_cases.py -v
```

### test_cache_effectiveness.py (15 tests)
**Coverage**: Hit rates, speedup validation, semantic caching
```bash
pytest hololoom/tests/e2e/test_cache_effectiveness.py -v
```

### test_integration_scenarios.py (12 tests)
**Coverage**: Complete workflows, multi-turn conversations
```bash
pytest hololoom/tests/e2e/test_integration_scenarios.py -v
```

## Test Infrastructure

### conftest.py (300+ lines)
Central test configuration providing:

#### Reproducible Random Seeds
```python
@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
```

#### Mock Fixtures
```python
@pytest.fixture
def mock_neo4j():
    """Mock Neo4j database driver."""
    ...

@pytest.fixture
def mock_qdrant():
    """Mock Qdrant vector store."""
    ...
```

#### Performance Budgets
```python
@pytest.fixture(autouse=True)
def performance_budget(request):
    """Enforce performance budgets based on test tier."""
    # Unit: <500ms
    # Integration: <2s
    # E2E: <30s
```

#### pytest-asyncio Configuration
```python
pytest_plugins = ('pytest_asyncio',)

def pytest_configure(config):
    config.option.asyncio_mode = "auto"
```

## Testing Philosophy

All tests validate **"Reliable Systems: Safety First"**:

### 1. Graceful Degradation ✅
- 20 tests for missing dependencies, LLM failures
- All fallback mechanisms tested

### 2. Thread Safety ✅
- 20 tests for race conditions, concurrent access
- asyncio.Lock protection validated (100 concurrent queries)

### 3. Timeout Protection ✅
- Policy decisions timeout at 2.0s
- No infinite hangs in 387+ tests

### 4. Complete Provenance ✅
- All tests verify Spacetime trace exists
- Full audit trail for debugging

### 5. Performance Budgets ✅
- Enforced via conftest.py
- Warnings emitted for budget violations

## Coverage Highlights

| Area | Tests | Coverage |
|------|-------|----------|
| **Concurrency** | 20 | 10-100 parallel queries |
| **Performance** | 15 | Latency, memory, throughput |
| **Error Handling** | 20 | All fallback mechanisms |
| **Learning** | 20 | Thompson Sampling, patterns |
| **Memory** | 10 | Leak detection (500 queries) |
| **Persistence** | 10 | State recovery |
| **Edge Cases** | 17 | Unicode, 50K chars, emoji |
| **Cache** | 15 | Hit rates, speedup |
| **Integration** | 12 | Complete workflows |

## Key Test Patterns

### Async Testing
```python
@pytest.mark.asyncio
async def test_async_operation():
    async with WeavingOrchestrator(...) as orchestrator:
        result = await orchestrator.weave(query)
        assert result is not None
```

### Performance Budget
```python
def test_fast_operation():
    # Automatically enforced by conftest.py
    # Will warn if >500ms (unit), >2s (integration), >30s (E2E)
    result = fast_operation()
```

### Mock External Dependencies
```python
def test_with_mocked_neo4j(mock_neo4j):
    # Neo4j automatically mocked
    memory = create_memory_backend(config)
    # Works without actual Neo4j
```

## Dependencies

```bash
pip install pytest pytest-asyncio pytest-timeout
```

## Future Enhancements

- [ ] Increase coverage to 50%
- [ ] Add property-based testing (hypothesis)
- [ ] Add mutation testing
- [ ] Performance regression detection
- [ ] Test coverage reporting (codecov)

## Related Documentation

- [MOONSHOT_COMPLETION_SUMMARY.md](../../MOONSHOT_COMPLETION_SUMMARY.md)
- [E2E_MOONSHOT_COMPLETE.md](../../E2E_MOONSHOT_COMPLETE.md)
- [EXCEPTION_HANDLING_GUIDE.md](../../EXCEPTION_HANDLING_GUIDE.md)

---

**Status**: 387+ tests, all passing ✅
**Coverage**: ~40%
**Last Updated**: November 2, 2025
