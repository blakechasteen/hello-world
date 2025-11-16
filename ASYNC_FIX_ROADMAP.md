# Async Fix Roadmap

**Created**: 2025-11-16
**Total Issues**: 18 (5 Critical, 8 Warning, 5 Info)
**Total Effort**: ~28-42 hours
**Timeline**: 3 sprints (2 weeks each)

---

## Executive Summary

This roadmap organizes the 18 async/await issues identified in the comprehensive code review into 3 actionable sprints. Each sprint focuses on a specific category of fixes, allowing for systematic improvement of async code quality across the HoloLoom codebase.

**Sprint 1 (Week 1-2)**: Critical Fixes - Resource leaks and task cleanup
**Sprint 2 (Week 3-4)**: Warning Fixes - Race conditions and error handling
**Sprint 3 (Week 5-6)**: Info Fixes - Best practices and optimizations

---

## Sprint 1: Critical Fixes (Week 1-2)

**Goal**: Eliminate resource leaks and orphaned background tasks
**Effort**: 10-14 hours
**Priority**: MUST FIX

### Issue #1: Background Tasks Not Tracked (CRITICAL)
**Effort**: 6-8 hours
**Files**: 4 modules

**Tasks**:
1. `HoloLoom/security/alerting/core.py` (2h)
   - Add `_background_tasks: set[asyncio.Task]` to AlertingEngine
   - Track alert dispatch tasks
   - Cancel all tasks in `__aexit__`

2. `HoloLoom/server/unified_server.py` (2h)
   - Add task tracking to UnifiedServer
   - Track background loops (metrics, health checks)
   - Proper shutdown sequence

3. `HoloLoom/spinningWheel/chat_history.py` (1h)
   - Track background update tasks
   - Cancel on cleanup

4. `HoloLoom/chatops/conversational.py` (1h)
   - Track async processing tasks
   - Cleanup in context manager exit

**Pattern to apply**:
```python
class Component:
    def __init__(self):
        self._background_tasks: set[asyncio.Task] = set()

    def _spawn_background_task(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait with timeout
        if self._background_tasks:
            await asyncio.wait(self._background_tasks, timeout=5.0)
```

**Acceptance Criteria**:
- [ ] All background tasks tracked in set
- [ ] Tasks cancelled on cleanup
- [ ] No orphaned tasks after exit
- [ ] Tests verify cleanup

---

### Issue #6: Race Condition in Cleanup Callback (CRITICAL)
**Effort**: 2 hours
**File**: `HoloLoom/weaving_orchestrator.py:332`

**Problem**:
```python
# Current (unsafe)
task.add_done_callback(self._background_tasks.discard)
# If task completes during __aexit__, discard may fail
```

**Fix**:
```python
def _task_done_callback(self, task: asyncio.Task):
    """Safe cleanup callback that handles concurrent modifications."""
    try:
        self._background_tasks.discard(task)
    except (KeyError, RuntimeError):
        # Task already removed or set being modified
        pass

# Usage
task.add_done_callback(self._task_done_callback)
```

**Acceptance Criteria**:
- [ ] No race condition during concurrent task completion
- [ ] Exception handled gracefully
- [ ] Tests verify concurrent cleanup

---

### Issue #8: Visual Q&A Engine Not Closed (CRITICAL)
**Effort**: 2-4 hours
**File**: `HoloLoom/rag/visual_qa.py`

**Tasks**:
1. Add explicit `close()` method to VisualQAEngine
2. Release GPU memory (clear CLIP model cache)
3. Integrate with MultimodalRAG `__aexit__`

**Implementation**:
```python
class VisualQAEngine:
    async def close(self):
        """Release GPU memory and cleanup resources."""
        if hasattr(self, 'clip_model'):
            # Clear CLIP model from GPU
            del self.clip_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if hasattr(self, 'ocr_engine'):
            # Cleanup OCR engine
            await self.ocr_engine.close()

class MultimodalRAG:
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Existing cleanup
        await super().__aexit__(exc_type, exc_val, exc_tb)

        # NEW: Close visual Q&A engine
        if hasattr(self, 'visual_qa_engine'):
            await self.visual_qa_engine.close()
```

**Acceptance Criteria**:
- [ ] GPU memory released on cleanup
- [ ] No memory leaks in repeated usage
- [ ] Tests verify memory cleanup

---

## Sprint 2: Warning Fixes (Week 3-4)

**Goal**: Eliminate race conditions and improve error handling
**Effort**: 12-18 hours
**Priority**: SHOULD FIX

### Issue #2: AgenticOrchestrator No Close Method (WARNING)
**Effort**: 2 hours
**File**: `HoloLoom/agentic/core.py`

**Implementation**:
```python
class AgenticOrchestrator:
    async def close(self):
        """Cleanup orchestrator resources."""
        # Close base orchestrator
        if hasattr(self, 'orchestrator'):
            await self.orchestrator.close()

        # Close any additional resources
        if hasattr(self, '_background_tasks'):
            for task in self._background_tasks:
                task.cancel()
            await asyncio.wait(self._background_tasks, timeout=5.0)
```

**Acceptance Criteria**:
- [ ] All resources cleaned up
- [ ] Context manager works correctly
- [ ] Tests verify cleanup

---

### Issue #3: SQL Parallel Queries Missing Await (WARNING)
**Effort**: 1 hour
**File**: `HoloLoom/rag/sql_integration.py:450`

**Review and verify**:
```python
# Line 450: Verify this is correct
results = await asyncio.gather(*[
    self._execute_sql_query(query, params)
    for query in queries
])
```

**Tasks**:
1. Manual code review of parallel query execution
2. Add tests for concurrent SQL queries
3. Verify proper await on all async operations

**Acceptance Criteria**:
- [ ] All SQL queries properly awaited
- [ ] Concurrent execution works correctly
- [ ] Tests verify parallel execution

---

### Issue #4: Missing Error Logging with exc_info (WARNING)
**Effort**: 3 hours
**Files**: Multiple modules

**Pattern to apply**:
```python
# Before
except Exception as e:
    logger.error(f"Operation failed: {e}")

# After
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
```

**Files to update**:
- `HoloLoom/weaving_orchestrator.py` (4 locations)
- `HoloLoom/rag/simple_rag.py` (3 locations)
- `HoloLoom/memory/backend_factory.py` (2 locations)
- `HoloLoom/security/soar/executor.py` (5 locations)

**Acceptance Criteria**:
- [ ] All exception logs include stack traces
- [ ] Easier debugging of production issues

---

### Issue #5: No Timeout on External LLM Calls (WARNING)
**Effort**: 2 hours
**Files**: `HoloLoom/weaving_orchestrator_llm.py`, `HoloLoom/rag/simple_rag.py`

**Implementation**:
```python
# Add timeout wrapper
async def _call_llm_with_timeout(self, prompt: str, timeout: float = 30.0):
    try:
        return await asyncio.wait_for(
            self._call_llm(prompt),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out after {timeout}s")
        raise
```

**Acceptance Criteria**:
- [ ] All external LLM calls have timeouts
- [ ] Configurable timeout values
- [ ] Proper error messages on timeout

---

### Issue #7: asyncio.wait vs asyncio.gather (WARNING)
**Effort**: 2 hours
**Files**: `HoloLoom/policy/unified.py`, `HoloLoom/recursive/loop_engine.py`

**Refactor**:
```python
# Before (less clear error handling)
done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

# After (clearer error propagation)
results = await asyncio.gather(*tasks, return_exceptions=True)
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"Task {i} failed: {result}")
```

**Acceptance Criteria**:
- [ ] Clearer error handling
- [ ] Easier to identify which task failed
- [ ] Tests verify exception handling

---

### Issue #9: Shared State Without Locks (WARNING)
**Effort**: 2 hours
**File**: `HoloLoom/reflection/buffer.py`

**Implementation**:
```python
class ReflectionBuffer:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._buffer = []

    async def store(self, spacetime, feedback):
        async with self._lock:
            self._buffer.append((spacetime, feedback))
            if len(self._buffer) > self.capacity:
                self._buffer.pop(0)
```

**Acceptance Criteria**:
- [ ] No race conditions on buffer modifications
- [ ] Tests verify concurrent access safety

---

## Sprint 3: Info Fixes (Week 5-6)

**Goal**: Apply best practices and optimize
**Effort**: 6-10 hours
**Priority**: NICE TO FIX

### Issue #10: Use TaskGroup for Better Cleanup (INFO)
**Effort**: 2 hours
**File**: `HoloLoom/weaving_orchestrator.py`

**Refactor to Python 3.11+ TaskGroup**:
```python
# Modern pattern (Python 3.11+)
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(feature_extraction())
    task2 = tg.create_task(memory_retrieval())
    # Automatic cancellation if any task fails
```

**Acceptance Criteria**:
- [ ] Cleaner task management
- [ ] Automatic cancellation on error
- [ ] Requires Python 3.11+

---

### Issue #11: Health Checks for Background Loops (INFO)
**Effort**: 2 hours
**Files**: Background task modules

**Implementation**:
```python
class BackgroundTaskManager:
    def __init__(self):
        self._last_heartbeat = {}

    async def _background_loop(self, name: str):
        while True:
            self._last_heartbeat[name] = time.time()
            await self._do_work()
            await asyncio.sleep(interval)

    def is_healthy(self, name: str, max_age: float = 60.0) -> bool:
        last = self._last_heartbeat.get(name, 0)
        return (time.time() - last) < max_age
```

**Acceptance Criteria**:
- [ ] Detect stuck background tasks
- [ ] Expose health check endpoint
- [ ] Alert on unhealthy tasks

---

### Issue #12: Add Timeouts to File I/O (INFO)
**Effort**: 1 hour
**Files**: `HoloLoom/spinningWheel/`, `HoloLoom/memory/cache.py`

**Pattern**:
```python
# Add timeout to async file operations
async with asyncio.timeout(10.0):
    async with aiofiles.open(path, 'r') as f:
        content = await f.read()
```

**Acceptance Criteria**:
- [ ] File operations don't hang indefinitely
- [ ] Configurable timeouts

---

### Issue #13: Make Cleanup Idempotent (INFO)
**Effort**: 1 hour
**Files**: All async context managers

**Pattern**:
```python
async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self._closed:
        return  # Already closed

    self._closed = True
    # ... cleanup logic
```

**Acceptance Criteria**:
- [ ] Safe to call cleanup multiple times
- [ ] No errors on double-cleanup

---

## Testing Strategy

### Sprint 1 Testing
- **Background task tracking**: Verify all tasks cancelled
- **Race condition fix**: Concurrent cleanup tests
- **Visual Q&A cleanup**: Memory leak tests

### Sprint 2 Testing
- **Error handling**: Exception propagation tests
- **Timeouts**: LLM timeout tests
- **Race conditions**: Concurrent access tests

### Sprint 3 Testing
- **Health checks**: Stuck task detection
- **Idempotency**: Double-cleanup safety

---

## Metrics

Track these metrics before/after each sprint:

1. **Resource Leaks**: Memory usage over time
2. **Orphaned Tasks**: Count of tasks not cleaned up
3. **Async Exceptions**: Rate of unhandled async exceptions
4. **Cleanup Time**: Time taken for `__aexit__`
5. **Test Coverage**: Async code test coverage %

**Baseline (Before)**:
- Resource leaks: 5 known issues
- Orphaned tasks: ~10-15 per session
- Async exceptions: ~5% of operations
- Cleanup time: ~1-5 seconds
- Test coverage: ~60%

**Target (After Sprint 3)**:
- Resource leaks: 0
- Orphaned tasks: 0
- Async exceptions: <1%
- Cleanup time: <1 second
- Test coverage: >80%

---

## Sprint Assignments

### Sprint 1 (Critical) - Recommended Team
- **Senior Engineer**: Issue #6 (race condition - tricky)
- **Mid-level Engineer**: Issue #1 (background tasks - repetitive)
- **Junior Engineer**: Issue #8 (close method - learning opportunity)

### Sprint 2 (Warning) - Recommended Team
- **Senior Engineer**: Issues #5, #7 (timeouts, gather vs wait)
- **Mid-level Engineer**: Issues #2, #4 (orchestrator cleanup, logging)
- **Junior Engineer**: Issues #3, #9 (SQL review, locks)

### Sprint 3 (Info) - Recommended Team
- **Any Engineer**: All issues (best practices, optimization)
- **Can be done in parallel with other work**

---

## Risk Assessment

### High Risk
- **Issue #6 (Race condition)**: Could cause crashes if not fixed carefully
  - Mitigation: Thorough testing, code review

- **Issue #8 (GPU memory)**: Could cause OOM in production
  - Mitigation: Monitor GPU memory usage, gradual rollout

### Medium Risk
- **Issue #1 (Background tasks)**: Could leave orphaned tasks
  - Mitigation: Add tests, monitor task count

### Low Risk
- **Sprint 3 issues**: Best practices, low chance of regression
  - Mitigation: Standard testing

---

## Rollout Plan

### Phase 1: Sprint 1 Critical Fixes
1. Deploy to staging
2. Monitor for 48 hours
3. Run full test suite
4. Deploy to production (canary 10%)
5. Full production rollout

### Phase 2: Sprint 2 Warning Fixes
1. Deploy to staging
2. Monitor for 24 hours
3. Production rollout (gradual 25% → 50% → 100%)

### Phase 3: Sprint 3 Info Fixes
1. Deploy to staging
2. Production rollout (immediate)

---

## Success Criteria

**Sprint 1 Success**:
- ✅ Zero resource leaks detected in 7-day monitoring
- ✅ All background tasks properly tracked and cancelled
- ✅ GPU memory released correctly

**Sprint 2 Success**:
- ✅ All race conditions eliminated
- ✅ LLM timeouts prevent hanging
- ✅ Stack traces available for all errors

**Sprint 3 Success**:
- ✅ Health checks detect stuck tasks
- ✅ Cleanup is idempotent
- ✅ Code follows async best practices

**Overall Success**:
- ✅ All 18 issues resolved
- ✅ Test coverage >80%
- ✅ Zero async-related production incidents for 30 days

---

## Maintenance

After completing all sprints:

1. **Update ASYNC_BEST_PRACTICES.md** with lessons learned
2. **Add async linting** to CI/CD (detect missing await, etc.)
3. **Monthly async code review** to prevent regression
4. **Onboarding checklist** for new engineers

---

## Effort Summary

| Sprint | Issues | Effort (hours) | Priority |
|--------|--------|----------------|----------|
| Sprint 1 | 3 Critical | 10-14 | MUST FIX |
| Sprint 2 | 6 Warning | 12-18 | SHOULD FIX |
| Sprint 3 | 5 Info | 6-10 | NICE TO FIX |
| **Total** | **18** | **28-42** | **3 sprints** |

---

## Next Steps

1. **Review this roadmap** with the team
2. **Assign sprints** to engineers
3. **Create tracking issues** in GitHub (18 issues)
4. **Start Sprint 1** immediately (critical fixes)
5. **Schedule retrospectives** after each sprint

---

**Document Status**: ✅ Ready for Team Review
**Last Updated**: 2025-11-16
**Owner**: HoloLoom Core Team
