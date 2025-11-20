# RAG Department Unit Tests - Quick Reference

## 📋 Test File Location
`HoloLoom/departments/tests/test_rag_department.py`

## ⏱️ Quick Stats
- **Total Tests**: 43
- **Test Classes**: 9
- **Lines of Code**: ~770
- **Coverage**: All 7 protocol methods + initialization + helpers

## 🚀 Running Tests

### Run All Tests
```bash
python -m pytest HoloLoom/departments/tests/test_rag_department.py -v
```

### Run Specific Protocol Method
```bash
# Execute method (9 tests)
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestExecuteMethod -v

# Verify method (6 tests)
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestVerifyMethod -v

# Refine method (4 tests)
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestRefineMethod -v

# Update strategy method (3 tests)
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestUpdateStrategyMethod -v

# Get capabilities method (4 tests)
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestGetCapabilitiesMethod -v

# Get metrics method (4 tests)
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestGetMetricsMethod -v

# Health check method (3 tests)
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestHealthCheckMethod -v

# Helper methods (7 tests)
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestHelperMethods -v
```

### Run Single Test
```bash
python -m pytest HoloLoom/departments/tests/test_rag_department.py::TestExecuteMethod::test_execute_basic_query -v
```

### List All Tests
```bash
python -m pytest HoloLoom/departments/tests/test_rag_department.py --collect-only
```

## 📊 Test Breakdown by Protocol Method

### 1️⃣ execute() - 9 Tests
**Purpose**: Query execution with RAG backend

| Test | Purpose |
|------|---------|
| `test_execute_basic_query` | Basic query execution |
| `test_execute_no_rag_raises` | Error when RAG unavailable |
| `test_execute_empty_query_raises` | Validation: empty queries |
| `test_execute_missing_query_raises` | Validation: missing query field |
| `test_execute_query_too_long` | Validation: query length limit |
| `test_execute_tracks_confidence` | Confidence history tracking |
| `test_execute_tracks_query_type` | Query type classification |
| `test_execute_error_handling` | Graceful error handling |
| `test_execute_modes` | Multiple reasoning modes |

**Coverage**:
- ✅ Query validation (non-empty, max 1000 tokens)
- ✅ Reasoning modes (direct, verify, research, plan_execute)
- ✅ Confidence tracking
- ✅ Query type classification (5 types)
- ✅ Error handling

### 2️⃣ verify() - 6 Tests
**Purpose**: DS-STAR verification framework

| Test | Purpose |
|------|---------|
| `test_verify_returns_result` | Proper VerificationResult structure |
| `test_verify_ds_star_dimensions` | All 5 DS-STAR dimensions present |
| `test_verify_overall_score` | Score aggregation |
| `test_verify_empty_sources` | Empty source handling |
| `test_verify_empty_answer` | Empty answer handling |
| `test_verify_recommendations` | Recommendation generation |

**Coverage**:
- ✅ Domain relevance checking
- ✅ Sensibility (coherence) checking
- ✅ Temporal freshness checking
- ✅ Argument support checking
- ✅ Reference traceability checking

### 3️⃣ refine() - 4 Tests
**Purpose**: Low-confidence response improvement

| Test | Purpose |
|------|---------|
| `test_refine_high_confidence` | Skip for high confidence |
| `test_refine_low_confidence` | Improve low confidence |
| `test_refine_tracking` | Statistics tracking |
| `test_refine_no_query_graceful` | Graceful fallback |

**Coverage**:
- ✅ Confidence threshold detection (0.75)
- ✅ Source expansion
- ✅ Reranking enablement
- ✅ Success rate tracking

### 4️⃣ update_strategy() - 3 Tests
**Purpose**: Learning from feedback

| Test | Purpose |
|------|---------|
| `test_update_strategy_helpful` | Helpful feedback learning |
| `test_update_strategy_unhelpful` | Unhelpful feedback learning |
| `test_update_strategy_complete` | Complete feedback integration |

**Coverage**:
- ✅ Helpful/unhelpful signals
- ✅ Confidence calibration
- ✅ Query pattern tracking

### 5️⃣ get_capabilities() - 4 Tests
**Purpose**: Capability advertisement

| Test | Purpose |
|------|---------|
| `test_get_capabilities` | Full capability structure |
| `test_get_capabilities_tasks` | Supported tasks |
| `test_get_capabilities_modes` | Reasoning modes |
| `test_get_capabilities_constraints` | Resource limits |

**Coverage**:
- ✅ Task types (question_answering, document_search, batch_processing)
- ✅ Max tokens (1000)
- ✅ Reasoning modes (4 types)
- ✅ Constraints (max_sources: 20, max_batch_size: 100)

### 6️⃣ get_metrics() - 4 Tests
**Purpose**: Performance metrics collection

| Test | Purpose |
|------|---------|
| `test_get_metrics_structure` | Complete metrics structure |
| `test_get_metrics_no_queries` | Empty metrics state |
| `test_get_metrics_after_queries` | Metrics after execution |
| `test_get_metrics_refinement_stats` | Refinement statistics |

**Coverage**:
- ✅ Confidence statistics (mean, stdev, min, max, count)
- ✅ Query patterns (5 types)
- ✅ Refinement stats (total, successful, rates)

### 7️⃣ health_check() - 3 Tests
**Purpose**: System health verification

| Test | Purpose |
|------|---------|
| `test_health_check_success` | All systems operational |
| `test_health_check_no_rag` | RAG initialization check |
| `test_health_check_no_memory` | Memory system check |

**Coverage**:
- ✅ Component dependencies
- ✅ Graceful degradation

### Plus: Initialization (3) + Helpers (7)
**Initialization Tests**:
- Default parameter initialization
- Custom parameter initialization
- Verification threshold setup

**Helper Method Tests**:
- Query type classification (5 types)
- Domain relevance checking
- Sensibility checking
- Temporal freshness checking

## 🎯 Test Organization

### By Category
```
TestInitialization        → 3 tests (setup)
TestExecuteMethod         → 9 tests (core functionality)
TestVerifyMethod          → 6 tests (quality checking)
TestRefineMethod          → 4 tests (improvement)
TestUpdateStrategyMethod  → 3 tests (learning)
TestGetCapabilitiesMethod → 4 tests (advertisement)
TestGetMetricsMethod      → 4 tests (monitoring)
TestHealthCheckMethod     → 3 tests (health)
TestHelperMethods         → 7 tests (utilities)
```

### By Protocol Method
```
Protocol Methods (7):
  execute()          → 9 tests
  verify()           → 6 tests
  refine()           → 4 tests
  update_strategy()  → 3 tests
  get_capabilities() → 4 tests
  get_metrics()      → 4 tests
  health_check()     → 3 tests

Supporting Tests (4):
  Initialization     → 3 tests
  Helpers            → 7 tests
```

## 🔍 Error Cases Covered

| Error Type | Test | Method |
|-----------|------|--------|
| Not initialized | `test_execute_no_rag_raises` | execute() |
| Empty query | `test_execute_empty_query_raises` | execute() |
| Missing field | `test_execute_missing_query_raises` | execute() |
| Query too long | `test_execute_query_too_long` | execute() |
| RAG exception | `test_execute_error_handling` | execute() |
| No RAG | `test_health_check_no_rag` | health_check() |
| No memory | `test_health_check_no_memory` | health_check() |
| Missing query | `test_refine_no_query_graceful` | refine() |

## 📈 Metrics Tracked

### Confidence Statistics
```python
{
    "mean": float,
    "stdev": float,
    "min": float,
    "max": float,
    "count": int
}
```

### Query Patterns
```python
{
    "factual": int,       # What is...?
    "procedural": int,    # How to...?
    "analytical": int,    # Why...?
    "comparative": int,   # Compare...?
    "other": int
}
```

### Refinement Statistics
```python
{
    "total_refinements": int,
    "successful_refinements": int,
    "avg_improvement": float,
    "refinement_rate": float,
    "success_rate": float
}
```

## 🛠️ Fixture Reference

### Available Fixtures
```python
@pytest.fixture
def config()
    # Config.fast() instance

@pytest.fixture
def rag_department()
    # RAGDepartment with mocked RAG

@pytest.fixture
def sample_request()
    # DepartmentRequest about Thompson Sampling

@pytest.fixture
def sample_response(sample_request)
    # DepartmentResponse with 0.85 confidence
```

## 📚 Test Pattern Example

```python
@pytest.mark.asyncio
async def test_execute_basic_query(self, rag_department, sample_request):
    """Test basic query execution."""
    # Setup
    rag_department.rag.query = AsyncMock(return_value=RAGResult(
        response="Thompson Sampling is...",
        sources=["source1", "source2"],
        confidence=0.88,
        reasoning_mode="verify",
        metadata={"cache_hit": False}
    ))

    # Execute
    response = await rag_department.execute(sample_request)

    # Assert
    assert response.confidence.score == 0.88
    rag_department.rag.query.assert_called_once()
```

## 🔐 Key Assertions

### Confidence Levels
```python
score < 0.2        → ConfidenceLevel.CRITICAL
0.2 ≤ score < 0.5  → ConfidenceLevel.LOW
0.5 ≤ score < 0.75 → ConfidenceLevel.MEDIUM
0.75 ≤ score < 0.95 → ConfidenceLevel.HIGH
score ≥ 0.95       → ConfidenceLevel.VERIFIED
```

### Verification Thresholds
```python
domain_min_relevance: 0.6
sensibility_min_score: 0.7
temporal_max_age_days: 365
argument_min_support: 0.8
reference_min_traceable: 0.9
```

### Refinement Threshold
```python
if confidence < 0.75:
    # Trigger refinement
```

## 💡 Common Usage Patterns

### Test Async Method
```python
@pytest.mark.asyncio
async def test_something(self, rag_department):
    result = await rag_department.execute(request)
    assert result is not None
```

### Mock RAG Query
```python
rag_department.rag.query = AsyncMock(return_value=RAGResult(
    response="Answer",
    sources=["source1"],
    confidence=0.80,
    reasoning_mode="verify",
    metadata={}
))
```

### Check Mock Called
```python
rag_department.rag.query.assert_called_once()
call_kwargs = rag_department.rag.query.call_args[1]
```

## 📝 Documentation Files

| File | Purpose |
|------|---------|
| `test_rag_department.py` | Complete test suite (770 lines) |
| `__init__.py` | Package marker for tests |
| `RAG_DEPARTMENT_TEST_SUMMARY.md` | Comprehensive summary |
| `RAG_DEPARTMENT_TESTS_QUICKREF.md` | This file |

## ✅ Checklist for Users

- [ ] Review test file: `HoloLoom/departments/tests/test_rag_department.py`
- [ ] Run: `pytest HoloLoom/departments/tests/test_rag_department.py -v`
- [ ] Check: 43 tests collected and passed
- [ ] Read: `RAG_DEPARTMENT_TEST_SUMMARY.md` for details
- [ ] Use: This quick reference for running specific tests

## 🔗 Related Files

- **Implementation**: `HoloLoom/departments/rag_department.py`
- **Protocol**: `HoloLoom/departments/protocol.py`
- **Base Class**: `HoloLoom/departments/base.py`

---

**Created**: November 2025
**Status**: ✅ Complete and Production-Ready
**Total Coverage**: 43 comprehensive tests for RAG Department