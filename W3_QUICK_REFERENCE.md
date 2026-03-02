# W3: Deprecated Code - Quick Reference

**Last Updated**: December 31, 2025
**Status**: RESEARCH COMPLETE - Ready for action

---

## The 3 Biggest Problems

### 1️⃣ Four Orchestrator Implementations (CONSOLIDATE TO 1)

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `weaving_orchestrator.py` | 3,476 | ✅ Canonical | Use this |
| `weaving_orchestrator_refactored.py` | ~500 | ⚠️ Unknown | Archive? |
| `shuttle/orchestrator_v2.py` | ~200 | ⚠️ Future-dated | Archive? |
| `weaving_shuttle.py` | 48 | 🔴 Shim | Keep for v1.x, archive v2.0 |

**Fix**: Use `WeavingOrchestrator` everywhere, archive alternatives

---

### 2️⃣ Duplicate Quality Trajectory Files (CONSOLIDATE TO 1)

| File | Size | Status |
|------|------|--------|
| `redteam/refinement/quality_trajectory.py` | 30 KB | Original |
| `redteam/refinement/quality_trajectory_updated.py` | 30 KB | Duplicate? |
| `redteam/refinement/quality_trajectory_extensions.py` | 13 KB | Extensions |

**Fix**: Determine which is used, archive the other

---

### 3️⃣ 29 Adapter Files (REDUCE TO <10)

**Examples**:
- `memory/weaving_adapter.py` - WeavingShuttle ↔ Memory bridge
- `agentic/safety_adapter.py` - Safety integration
- `agentic/conscience_adapter.py` - Conscience integration
- Plus 26 more...

**Fix**: Refactor for better protocol composition

---

## The 11 Files Using Deprecated WeavingShuttle

Replace `from hololoom.weaving_shuttle import WeavingShuttle` with:
```python
from hololoom.weaving_orchestrator import WeavingOrchestrator as WeavingShuttle
# Or better:
from hololoom.weaving_orchestrator import WeavingOrchestrator
```

**Files to update**:
1. `hololoom/chatops/handlers/hololoom_handlers.py`
2. `hololoom/tests/integration/test_enhanced_shuttle.py`
3. `hololoom/tests/unit/test_orchestrator_refactoring.py`
4. `hololoom/tests/unit/test_weaving_shuttle.py`
5. `hololoom/tests/integration/test_shuttle_integration.py`
6. `hololoom/tests/integration/test_complexity_tuning.py`
7. `hololoom/tests/e2e/test_fused_mode_e2e.py`
8. `hololoom/tests/e2e/test_full_pipeline.py`
9. `hololoom/tests/e2e/test_fast_mode_e2e.py`
10. `hololoom/tests/e2e/test_bare_mode_e2e.py`
11. `hololoom/performance/benchmark.py`

---

## Memory Backend Status

| Backend | Purpose | Status | Action |
|---------|---------|--------|--------|
| INMEMORY | NetworkX in-memory | ✅ Keep | Dev/testing |
| HYBRID | Neo4j + Qdrant + fallback | ✅ Keep | Production |
| HYPERSPACE | Research-grade | ⚠️ Review | Keep or archive? |
| Legacy adapters | Backward compat | ⚠️ Deprecate | Remove in v2.0 |

**Current**: `LegacyShardsAdapter` in `memory/graph.py`

---

## Test Duplication

| Component | Test Count | Solution |
|-----------|-----------|----------|
| Orchestrator | 6 files | Consolidate to 3 (unit/integration/e2e) |
| Shuttle | 4 files | Consolidate to 3 |
| **Total** | **10+** | **Reduce to ~6** |

---

## Files to Archive (Do First)

```
Move these to .archive/:

1. hololoom/weaving_orchestrator_refactored.py
2. hololoom/shuttle/orchestrator_v2.py
3. hololoom/redteam/refinement/quality_trajectory_updated.py
4. hololoom/redteam/refinement/quality_trajectory_extensions.py
5. hololoom/prompting/testing/golden_dataset.py
6. hololoom/prompting/testing/golden_chains.py
```

---

## Priority Action Items

### This Week (4 hours)
- [ ] Identify canonical quality_trajectory.py
- [ ] List imports of quality_trajectory_updated.py
- [ ] Create plan for orchestrator consolidation

### This Sprint (1 week)
- [ ] Update 11 WeavingShuttle imports to WeavingOrchestrator
- [ ] Move duplicate files to .archive/
- [ ] Delete non-canonical versions

### Next Sprint (2 weeks)
- [ ] Consolidate orchestrator tests (6 → 2)
- [ ] Consolidate shuttle tests (4 → 1)
- [ ] Archive if refactored orchestrator unused

### Next Quarter (1 month)
- [ ] Reduce adapters from 29 to <10
- [ ] Complete memory legacy migration
- [ ] Document architecture decisions

---

## Quick Check: Is File Deprecated?

### Look for these markers:

**✅ Clearly Deprecated**:
- `@deprecated` decorator
- "DEPRECATED" in docstring
- "compatibility shim" in description
- Has `warnings.warn(DeprecationWarning)`

**⚠️ Unclear Status**:
- Multiple implementations of same feature
- File named `*_refactored.py`, `*_v2.py`, `*_updated.py`
- Has `Legacy` class name
- File date doesn't match current project date
- Has `migrate.py` or migration documentation

**✅ Not Deprecated**:
- Core production code
- Recent creation date
- Active imports
- Test coverage

---

## Where to Go for Details

| Question | Location |
|----------|----------|
| Complete analysis | `W3_DEPRECATED_CODE_ANALYSIS.md` |
| Summary findings | `W3_FINDINGS_SUMMARY.txt` |
| This quick ref | `W3_QUICK_REFERENCE.md` |

---

## Metrics at a Glance

```
Total Python Files:        2,026
Test Files:                454
Deprecated Patterns:       19 files
Adapter/Bridge Files:      29 (🚩 too many)
Duplicate Implementations: 4
Test File Duplicates:      10+
Archive Readiness:         70% ✅

Critical Issues:           3 (orchestrator, trajectory, adapters)
High Priority Issues:      3 (memory, test consolidation)
Medium Priority:           4 (refactoring, documentation)
```

---

## What Was NOT Modified

⚠️ **THIS IS RESEARCH ONLY**

No code changes were made. This analysis:
- ✅ Identified deprecated patterns
- ✅ Found dead code
- ✅ Categorized issues by severity
- ✅ Provided recommendations
- ❌ Did NOT modify any files
- ❌ Did NOT delete anything
- ❌ Did NOT move files

---

## Next Steps

1. **Read full report**: `W3_DEPRECATED_CODE_ANALYSIS.md`
2. **Review findings**: `W3_FINDINGS_SUMMARY.txt`
3. **Create tracking issues** for each P0 item
4. **Assign cleanup work** to sprint planning
5. **Execute consolidation** following priority order

---

**Report Date**: 2025-12-31
**Status**: ✅ Complete (RESEARCH ONLY)
**Confidence**: HIGH
