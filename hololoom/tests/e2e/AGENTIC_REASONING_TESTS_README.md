# Agentic Reasoning E2E Tests

**Status**: ✅ All 14 tests passing (100%)
**Location**: `HoloLoom/tests/e2e/test_agentic_reasoning_modes.py`
**Runtime**: ~3.7 minutes for full suite
**Created**: 2025-12-01

## Overview

Comprehensive end-to-end tests for HoloLoom's agentic reasoning system covering all 4 reasoning modes, epistemic confidence tracking, safety integration, error handling, and performance budgets.

## Test Coverage

### 1. DIRECT Mode (2 tests)

**Single-Pass Answering (~150ms target)**

- ✅ `test_direct_mode_single_pass` - Verifies single query/response cycle
  - Checks response generation via LLM
  - Validates 1 total query
  - Confirms single `direct_answer` step
  - Performance budget: <1000ms (relaxed for test env)

- ✅ `test_direct_mode_epistemic_tracking` - Tests epistemic confidence extraction
  - Validates awareness layer integration
  - Checks epistemic confidence in [0.0, 1.0] range
  - Verifies metadata presence in results

### 2. VERIFY Mode (2 tests)

**Answer + Verification Loop (~600ms target)**

- ✅ `test_verify_mode_with_verification_loop` - Tests verification cycle
  - Validates initial answer + verification queries
  - Checks VerificationResult structure
  - Confirms ≥2 total queries (initial + verification)
  - Verifies step types: `initial_answer`, `verification`

- ✅ `test_verify_mode_contradiction_detection` - Tests contradiction detection
  - Mocks contradictory verification responses
  - Validates contradiction detection via "however"/"but" keywords
  - Confirms failed verification when contradictions found
  - Tests refinement suggestion generation

### 3. RESEARCH Mode (2 tests)

**Multi-Query Exploration (3-5 steps)**

- ✅ `test_research_mode_multi_query_exploration` - Tests research workflow
  - Validates multiple research queries generated
  - Checks ≥3 total queries (research + synthesis)
  - Confirms step types: `research_query`, `synthesis`
  - Tests evidence gathering and final synthesis

- ✅ `test_research_mode_early_stopping_low_epistemic` - Tests early stopping
  - Mocks decreasing epistemic confidence (0.7 → 0.25 → 0.2 → 0.15)
  - Validates early stopping when avg recent epistemic < 0.3
  - Confirms fewer steps than max_steps
  - Tests epistemic threshold detection logic

### 4. PLAN_EXECUTE Mode (2 tests)

**Goal Decomposition into Sub-Tasks**

- ✅ `test_plan_execute_mode_goal_decomposition` - Tests planning
  - Validates goal decomposition into sub-goals
  - Checks sub-goal execution
  - Confirms step types: `sub_goal`, `synthesis`
  - Tests intent tracking with sub_goals list

- ✅ `test_plan_execute_mode_sub_goal_completion_tracking` - Tests completion
  - Mocks varying confidence across sub-goals
  - Validates completion tracking (completed flag)
  - Checks synthesis metadata includes `sub_goals_completed` count
  - Tests confidence threshold enforcement

### 5. Epistemic Confidence (1 test)

**Aggregation Across Multiple Steps**

- ✅ `test_epistemic_confidence_aggregation` - Tests weighted averaging
  - Validates aggregation formula: weighted average with recent steps weighted higher
  - Tests with increasing confidence sequence (0.5 → 0.6 → 0.7 → 0.8)
  - Confirms aggregated value in expected range
  - Verifies weight formula: `(step_idx + 1) / total_steps`

### 6. Safety Integration (1 test)

**Alignment Framework Integration**

- ✅ `test_safety_integration_blocks_high_risk` - Tests safety gating
  - Validates safety adapter integration
  - Mocks high-risk decision (blocked, requires_approval)
  - Tests exception handling when queries blocked
  - Confirms safety-first behavior

### 7. Error Handling (2 tests)

**Graceful Degradation**

- ✅ `test_error_handling_llm_timeout` - Tests LLM failure handling
  - Mocks LLM timeout exception
  - Validates graceful fallback (system doesn't crash)
  - Confirms result still returned
  - Tests error resilience

- ✅ `test_error_handling_memory_failure` - Tests memory backend failure
  - Mocks memory backend exception
  - Validates proper exception propagation
  - Tests error handling path

### 8. Performance Budgets (2 tests)

**Latency Targets**

- ✅ `test_performance_budget_direct_mode` - DIRECT mode <500ms
  - Mocks 10ms weave operation
  - Validates total duration <500ms
  - Tests performance tracking

- ✅ `test_performance_budget_verify_mode` - VERIFY mode <1000ms
  - Mocks 50ms per query (realistic)
  - Validates ≥2 queries executed
  - Tests multi-query performance

## Test Architecture

### Mock Components

**MockLLM**
- Simulates LLM API calls without network overhead
- Returns pattern-matched responses based on prompt keywords
- Tracks call count for verification
- Supports verify, research, and answer patterns

**MockAwarenessLayer**
- Provides configurable coherence values
- Simulates awareness graph perception
- Returns mock awareness metrics

**create_mock_spacetime()**
- Creates valid Spacetime objects with all required fields
- Supports optional epistemic confidence metadata
- Generates proper WeavingTrace objects
- Includes awareness metadata when epistemic provided

### Test Patterns

All tests follow consistent pattern:

1. **Setup**: Create FullLearningEngine with test config
2. **Mock**: Replace `learning_engine.weave()` with async mock
3. **Execute**: Call `orchestrator.reason()` with appropriate mode
4. **Assert**: Validate result structure, steps, and behavior
5. **Teardown**: Cleanup via context manager exit

## Running the Tests

**Full suite**:
```bash
python -m pytest HoloLoom/tests/e2e/test_agentic_reasoning_modes.py -v
# 14 passed in ~3.7 minutes
```

**Single mode**:
```bash
# DIRECT mode tests only
python -m pytest HoloLoom/tests/e2e/test_agentic_reasoning_modes.py -k "direct" -v

# VERIFY mode tests only
python -m pytest HoloLoom/tests/e2e/test_agentic_reasoning_modes.py -k "verify" -v

# RESEARCH mode tests only
python -m pytest HoloLoom/tests/e2e/test_agentic_reasoning_modes.py -k "research" -v

# PLAN_EXECUTE mode tests only
python -m pytest HoloLoom/tests/e2e/test_agentic_reasoning_modes.py -k "plan_execute" -v
```

**With detailed output**:
```bash
python -m pytest HoloLoom/tests/e2e/test_agentic_reasoning_modes.py -xvs
```

## Performance Characteristics

| Test | Duration (approx) | Notes |
|------|------------------|-------|
| DIRECT mode tests | ~16s each | Full FullLearningEngine initialization |
| VERIFY mode tests | ~16s each | Multiple query cycle |
| RESEARCH mode tests | ~18s each | Multi-query exploration |
| PLAN_EXECUTE tests | ~17s each | Goal decomposition overhead |
| Epistemic tests | ~16s each | Aggregation logic |
| Safety tests | ~15s each | Safety adapter creation |
| Error tests | ~15s each | Exception handling |
| Performance tests | ~16s each | Mock timing validation |

**Total suite time**: ~224s (3.7 minutes)

Most time spent in FullLearningEngine initialization (embedding models, reflection buffers, etc). Tests themselves are fast (<100ms) once setup complete.

## Test Maintenance

### Adding New Tests

Template for new agentic test:

```python
@pytest.mark.asyncio
async def test_your_new_feature(mock_config, mock_memory_shards):
    """Test description."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=False
        )

        # Mock weave
        async def mock_weave(query, **kwargs):
            return await create_mock_spacetime(
                confidence=0.85,
                epistemic_confidence=0.75,
                response="Your mock response"
            )

        orchestrator.learning_engine.weave = mock_weave

        # Execute and assert
        query = Query(text="Your test query")
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        assert result is not None
        # Your assertions here

    finally:
        await learning_engine.__aexit__(None, None, None)
```

### Common Issues

**Issue**: `AttributeError: 'NoneType' object has no attribute 'trace'`
**Fix**: Ensure mock weave returns proper Spacetime with WeavingTrace

**Issue**: `TypeError: MemoryShard.__init__() got unexpected keyword argument 'content'`
**Fix**: Use `text` not `content` for MemoryShard field

**Issue**: Test hangs during initialization
**Fix**: Ensure `enable_background_learning=False` to avoid background threads

**Issue**: Safety adapter import errors
**Fix**: Use `pytest.skip()` if safety module unavailable

## Future Enhancements

Potential test additions:

1. **Concurrent reasoning** - Multiple agents reasoning simultaneously
2. **Memory persistence** - Test reasoning with persistent backends (Neo4j/Qdrant)
3. **LLM provider switching** - Test with different LLM backends (Ollama, OpenAI, Anthropic)
4. **Complex verification** - Multi-round verification loops
5. **Adaptive research** - Test research query adaptation based on findings
6. **Cross-mode workflows** - Combine modes (e.g., RESEARCH → VERIFY → PLAN_EXECUTE)
7. **Monitoring integration** - Test with AgentMonitor tracking
8. **Long-running tasks** - Test max_steps limits and timeouts
9. **Quality degradation** - Test behavior when confidence consistently low
10. **A/B testing** - Compare reasoning strategies

## Integration Points

Tests validate integration with:

- ✅ **FullLearningEngine** - Recursive learning wrapper
- ✅ **LLM integration** - Mock LLM for answer generation
- ✅ **Awareness layer** - Epistemic confidence tracking
- ✅ **Safety adapter** - Alignment framework gating
- ✅ **AuditTrail** - Decision logging
- ✅ **Spacetime** - Response + trace structure
- ✅ **Query protocols** - Query type handling

## References

- **Implementation**: `HoloLoom/agentic/core.py` (1172 lines)
- **Alignment**: `HoloLoom/alignment/safety_guardrails.py`
- **Consciousness**: `HoloLoom/awareness/`
- **Recursive Learning**: `HoloLoom/recursive/`
- **Documentation**: `CLAUDE.md` (Agentic Reasoning System section)
