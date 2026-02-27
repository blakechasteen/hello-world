# HoloLoom Integration Test Metaplan
**Created**: 2025-11-22
**Status**: Phase 1 - Integration Surface Mapping Complete

## Integration Architecture Map

```
┌─────────────────────────────────────────────────────────────┐
│                     VS Code Extension                        │
│                  (TypeScript - squad/)                       │
│                                                               │
│  HoloLoomBridge.ts (109 lines)                              │
│  ├─ healthCheck() → GET /health                             │
│  ├─ query() → POST /query                                   │
│  ├─ chat() → POST /chat                                     │
│  └─ getStats() → GET /stats                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Server                             │
│              (Python - hololoom/server/)                     │
│                                                               │
│  agentic_api.py (1,825 lines)                               │
│  ├─ GET /health                                             │
│  ├─ POST /query (QueryRequest → AgenticResponse)           │
│  ├─ POST /chat                                              │
│  ├─ GET /stats                                              │
│  ├─ GET /audit-trail                                        │
│  └─ Rate limiting, CORS, error handling                     │
└─────────────────────────────────────────────────────────────┘
                            ↓ Python imports
┌─────────────────────────────────────────────────────────────┐
│                 Agentic Orchestrator                         │
│           (Python - hololoom/agentic/)                       │
│                                                               │
│  create_agentic_orchestrator() → AgenticOrchestrator        │
│  ├─ reason() - Multi-query reasoning                        │
│  ├─ ReasoningMode (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)    │
│  └─ Returns AgenticResult                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ Uses
┌─────────────────────────────────────────────────────────────┐
│              Weaving Orchestrator Core                       │
│        (Python - hololoom/weaving_orchestrator.py)          │
│                                                               │
│  WeavingOrchestrator (3,476 lines)                          │
│  ├─ weave() - 9-step cycle                                  │
│  ├─ Config (BARE/FAST/FUSED modes)                          │
│  └─ Returns Spacetime with trace                            │
└─────────────────────────────────────────────────────────────┘
                            ↓ Uses
┌─────────────────────────────────────────────────────────────┐
│                   Memory Backends                            │
│            (Python - hololoom/memory/)                       │
│                                                               │
│  backend_factory.create_memory_backend()                    │
│  ├─ INMEMORY (NetworkX - always available)                  │
│  ├─ HYBRID (Neo4j + Qdrant - Docker)                        │
│  └─ HYPERSPACE (Advanced - research)                        │
│                                                               │
│  Docker Services:                                            │
│  ├─ Neo4j :7474 (HTTP) :7687 (Bolt)                        │
│  └─ Qdrant :6333 (HTTP) :6334 (gRPC)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓ Fallback
┌─────────────────────────────────────────────────────────────┐
│              Alignment & Safety                              │
│          (Python - hololoom/alignment/)                      │
│                                                               │
│  SafetyGuardrails - Risk-based gating                       │
│  AuditTrail - Provenance logging                            │
│  DeceptionDetection - Goal transparency                     │
│  InstrumentalConvergence - Power-seeking detection          │
└─────────────────────────────────────────────────────────────┘
```

## Discovered Integration Points (Phase 1 Complete)

### Layer 1: TypeScript ↔ FastAPI
**Location**: `squad/src/HoloLoomBridge.ts` ↔ `hololoom/server/agentic_api.py`
**Status**: ❌ No tests exist
**Interface Contract**:
- TypeScript `QueryRequest` → Python `QueryRequest` (Pydantic)
- Python `AgenticResponse` → TypeScript `AgenticResult`
- Match validation: field names, types, required/optional

**Critical Issues**:
1. No schema validation tests
2. No round-trip serialization tests
3. No error handling verification
4. Server has 17 API files but unclear which is canonical

### Layer 2: FastAPI ↔ Agentic Orchestrator
**Location**: `hololoom/server/agentic_api.py` ↔ `hololoom/agentic/core.py`
**Status**: ⚠️ Code exists, untested
**Interface Contract**:
- `QueryRequest` → `Query` (protocols.types)
- `AgenticResult` → `AgenticResponse`
- Reasoning mode mapping (string → enum)

**Critical Issues**:
1. Rate limiter exists but untested
2. AuditTrail imported but not used in /query
3. SafetyGuardrails not integrated
4. Error handling untested

### Layer 3: Agentic ↔ Weaving Orchestrator
**Location**: `hololoom/agentic/core.py` ↔ `hololoom/weaving_orchestrator.py`
**Status**: ✅ Partial tests exist (`test_full_pipeline.py`)
**Interface Contract**:
- `Query` → `Spacetime` via `weave()`
- Multi-query reasoning loops
- Reflection and refinement

**Existing Coverage**:
- `test_full_pipeline.py` - 4 tests (BARE/FAST/FUSED modes)
- `test_orchestrator_9_step_cycle.py` - Full weaving cycle
- `test_reflection_loop.py` - Reflection buffer

### Layer 4: Orchestrator ↔ Memory Backend
**Location**: `hololoom/weaving_orchestrator.py` ↔ `hololoom/memory/backend_factory.py`
**Status**: ⚠️ Code exists, auto-fallback untested
**Interface Contract**:
- `create_memory_backend(config)` → MemoryBackend
- HYBRID → INMEMORY fallback (when Docker unavailable)
- Graph operations, vector search, hybrid retrieval

**Critical Issues**:
1. Auto-fallback logic exists but no test
2. Docker startup time unknown (~30s estimated)
3. Neo4j default password `hololoom123` in docker-compose.yml
4. No health check integration

### Layer 5: Alignment Framework Integration
**Location**: `hololoom/alignment/` ↔ All layers
**Status**: ✅ Tests exist, ❌ Not integrated in server
**Interface Contract**:
- `SafetyGuardrails.gate_action()` → GateResult
- `AuditTrail.log_decision()` → None
- Risk scoring, deception detection

**Critical Issues**:
1. Server imports alignment but doesn't use it
2. No integration in `/query` endpoint
3. No audit trail logging in production code

## Test Coverage Gap Analysis

### Existing E2E Tests (18 files)
✅ Have:
- Orchestrator internal cycles (9-step weaving)
- Memory system benchmarks
- Reflection loops
- Multi-department workflows
- Performance profiling
- Error handling (orchestrator level)

❌ Missing:
- FastAPI server endpoint tests
- TypeScript bridge tests
- Full stack VS Code → Server → Orchestrator → Memory
- Docker fallback validation
- Alignment integration
- Rate limiting behavior
- CORS validation

### Test Pyramid Current State

```
    /\        E2E (18 tests)           ← Orchestrator internals only
   /  \
  /────\      Integration (~90 tests)  ← Component pairs only
 /      \
/────────\    Unit (~60 tests)         ← Isolated components
```

### Test Pyramid Target State

```
    /\        Full Stack E2E (5 tests)     ← VS Code → Response
   /  \
  /────\      API Integration (15 tests)   ← Server ↔ Orchestrator
 /──────\
/────────\    Component Integration (20)   ← Memory, Alignment
```

## Metaprompt Refinement Strategy

### Pass 1: VERIFY (Validate Existing)
**Goal**: Ensure existing tests are accurate and comprehensive
**Actions**:
1. Run all 18 E2E tests, verify 100% pass
2. Check test assertions are meaningful (not just "is not None")
3. Validate test data quality (realistic queries)
4. Ensure cleanup happens (async context managers)

### Pass 2: EXPAND (Add Missing Tests)
**Goal**: Fill critical gaps in integration coverage
**Actions**:
1. Create `test_server_integration.py` (FastAPI endpoints)
2. Create `test_typescript_schema_validation.py` (contracts)
3. Create `test_memory_backend_fallback.py` (Docker failure)
4. Create `test_alignment_integration.py` (safety gating)
5. Create `test_full_stack_e2e.py` (VS Code → Response)

### Pass 3: REFINE (Improve Quality)
**Goal**: Make tests production-grade
**Actions**:
1. Add performance assertions (latency < X ms)
2. Add resource cleanup verification
3. Add error scenario coverage (network failure, rate limit, etc.)
4. Add concurrency tests (multiple simultaneous requests)

### Pass 4: AUTOMATE (CI/CD Integration)
**Goal**: Run tests automatically on every commit
**Actions**:
1. Create `.github/workflows/integration-tests.yml`
2. Docker compose up/down in CI
3. Parallel test execution
4. Failure notifications

## Implementation Plan

### Phase 2: Core E2E Tests (Validate Existing)
**Files to Test**:
- `test_full_pipeline.py` ✅ Already exists
- `test_orchestrator_9_step_cycle.py` ✅ Already exists
- Run and document results

### Phase 3: API Server Integration
**New File**: `hololoom/tests/integration/test_server_integration.py`
**Test Count**: 15 tests
**Coverage**:
1. Health endpoint (GET /health)
2. Query endpoint - DIRECT mode
3. Query endpoint - VERIFY mode
4. Query endpoint - RESEARCH mode
5. Query endpoint - PLAN_EXECUTE mode
6. Chat endpoint
7. Stats endpoint
8. Audit trail endpoint
9. Rate limiting (exceed limit)
10. Rate limiting (within limit)
11. Request validation (text too large)
12. Request validation (invalid max_steps)
13. CORS headers
14. Error responses (500, 400, 429)
15. Concurrent requests

### Phase 4: VS Code Bridge
**New File**: `squad/tests/test_hololoom_bridge.test.ts`
**Test Count**: 8 tests
**Coverage**:
1. Health check success
2. Health check failure (server down)
3. Query request/response schema
4. Chat request/response schema
5. Stats request/response schema
6. Error handling (timeout)
7. Error handling (network failure)
8. Request serialization/deserialization

### Phase 5: Memory Backend Integration
**New File**: `hololoom/tests/integration/test_memory_backend_fallback.py`
**Test Count**: 12 tests
**Coverage**:
1. INMEMORY backend always works
2. HYBRID backend with Docker available
3. HYBRID → INMEMORY fallback (Neo4j down)
4. HYBRID → INMEMORY fallback (Qdrant down)
5. HYBRID → INMEMORY fallback (both down)
6. Graph operations (add/retrieve nodes)
7. Vector operations (similarity search)
8. Hybrid retrieval (graph + vector)
9. Persistence verification (Neo4j)
10. Query cache effectiveness
11. Health check integration
12. Docker startup time measurement

### Phase 6: Full Stack E2E
**New File**: `hololoom/tests/e2e/test_full_stack_vscode.py`
**Test Count**: 5 tests
**Coverage**:
1. Simple query: TypeScript → Server → Orchestrator → Response
2. Code context: Pass selection, get explanation
3. Verification mode: Multi-query reasoning
4. Research mode: 5-step exploration
5. Error propagation: Orchestrator error → TypeScript

### Phase 7: Integration CI/CD
**New File**: `.github/workflows/integration-tests.yml`
**Components**:
1. Docker compose up (Neo4j + Qdrant)
2. Wait for services healthy
3. Run pytest integration tests
4. Run pytest e2e tests
5. Docker compose down
6. Upload test results
7. Notify on failure

## Success Metrics

### Coverage Targets
- [ ] 100% of FastAPI endpoints have tests
- [ ] 100% of TypeScript bridge methods have tests
- [ ] 100% of memory backend modes tested
- [ ] 100% of alignment integration verified
- [ ] 5 full stack E2E scenarios

### Performance Targets
- [ ] /health endpoint < 5ms
- [ ] /query endpoint < 200ms (DIRECT mode)
- [ ] /query endpoint < 700ms (VERIFY mode)
- [ ] /query endpoint < 1000ms (RESEARCH mode)
- [ ] Rate limiting overhead < 1ms

### Quality Targets
- [ ] Zero flaky tests (3 runs, 100% pass)
- [ ] All tests use proper cleanup (async context managers)
- [ ] All tests have meaningful assertions (not just "not None")
- [ ] All tests document what they're verifying

## Risk Assessment

### High Risk (Must Fix)
1. **Server untested** - Could fail during investor demo
2. **Docker fallback untested** - Demo crash if services don't start
3. **Alignment not integrated** - Can't demonstrate safety claims

### Medium Risk (Should Fix)
1. **TypeScript schema drift** - Python changes break extension
2. **Performance unverified** - Claims might be wrong
3. **Error handling gaps** - Poor user experience

### Low Risk (Nice to Have)
1. **Concurrent request handling** - Likely works, but untested
2. **Resource cleanup** - Probably fine, but worth verifying

## Next Steps

1. ✅ **Phase 1 Complete**: Integration surface mapped (this document)
2. 🔄 **Phase 2 Starting**: Run existing E2E tests, document results
3. ⏳ **Phase 3**: Create `test_server_integration.py` (15 tests, ~4 hours)
4. ⏳ **Phase 4**: Create TypeScript bridge tests (~2 hours)
5. ⏳ **Phase 5**: Create memory backend tests (~3 hours)
6. ⏳ **Phase 6**: Create full stack E2E (~3 hours)
7. ⏳ **Phase 7**: Set up CI/CD pipeline (~2 hours)

**Total Estimated Effort**: 16 hours (2 days)
**Critical Path for Tomorrow's Demo**: Phases 3, 5, 6 (10 hours)

---

**Last Updated**: 2025-11-22
**Next Review**: After Phase 2 completion
