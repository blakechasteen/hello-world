# Protocol Consolidation Plan - Phase 0, Task 7

**Date**: 2025-01-12
**Objective**: Create single source of truth for all protocol definitions in `HoloLoom/protocols/`

---

## Executive Summary

This consolidation effort identified **7 scattered protocol locations** across the codebase, with **18 distinct protocols** currently defined. The consolidation will centralize all protocol definitions into `HoloLoom/protocols/` to establish a single source of truth.

**Status**: ✅ ANALYSIS COMPLETE - Ready for implementation

---

## 1. Current State Audit

### 1.1 Canonical Protocol Package (`HoloLoom/protocols/`)

**Files**:
- `__init__.py` - Main exports (170 lines)
- `types.py` - Core types (ComplexityLevel, ProvenanceTrace, MythRLResult)
- `core.py` - Memory protocols (MemoryStore, MemoryNavigator, PatternDetector)
- `core_features.py` - Feature protocols (Embedder, MotifDetector, PolicyEngine, RoutingStrategy, ExecutionEngine, ToolRegistry)
- `shuttle.py` - mythRL Shuttle protocols (PatternSelectionProtocol, FeatureExtractionProtocol, WarpSpaceProtocol, DecisionEngineProtocol, ToolExecutor)
- `retrieval.py` - Retrieval protocols (RetrievalStrategy, RetrievalResult, SpringActivationMetadata)

**Total Protocols Defined**: 15 protocols

**Key Issue**: `MemoryStore` protocol is commented out in `__init__.py` due to circular import with `HoloLoom.memory.protocol`

---

### 1.2 Scattered Protocol Locations

| Location | Protocols Defined | Status | Notes |
|----------|------------------|--------|-------|
| `HoloLoom/memory/protocol.py` | MemoryStore (fallback), Memory, MemoryQuery, RetrievalResult | **DUPLICATE** | Has fallback protocol definition lines 112-125 |
| `HoloLoom/search/protocol.py` | SearchProvider, ContentScraper | **VALID** | Domain-specific, should stay |
| `HoloLoom/spinningWheel/protocol.py` | SpinnerProtocol, BaseSpinner | **VALID** | Comprehensive spinner framework |
| `HoloLoom/spinningWheel/ocr_protocol.py` | OCRProtocol, BaseOCRBackend | **VALID** | OCR abstraction layer |
| `HoloLoom/writing/core/protocol.py` | WriterProtocol, ComposerProtocol, RefinerProtocol, ModeWriterProtocol | **VALID** | Writing system protocols |
| `HoloLoom/modules/Features.py` | MotifDetector (deprecated), Embedder (deprecated) | **DEPRECATED** | Lines 38-78, already has deprecation warnings |
| `HoloLoom/ts_core/base.py` | ThompsonSampler | **VALID** | Thompson Sampling protocol |

---

## 2. Protocol Inventory

### 2.1 Already in Canonical Location (`HoloLoom/protocols/`)

✅ **Core Types** (3):
- `ComplexityLevel` (Enum)
- `ProvenanceTrace` (dataclass)
- `MythRLResult` (dataclass)

✅ **Memory Protocols** (2):
- `MemoryNavigator` (Protocol)
- `PatternDetector` (Protocol)

✅ **Core Feature Protocols** (6):
- `Embedder` (Protocol)
- `MotifDetector` (Protocol)
- `PolicyEngine` (Protocol)
- `RoutingStrategy` (Protocol)
- `ExecutionEngine` (Protocol)
- `ToolRegistry` (Protocol)

✅ **Shuttle Protocols** (5):
- `PatternSelectionProtocol` (Protocol)
- `FeatureExtractionProtocol` (Protocol)
- `WarpSpaceProtocol` (Protocol)
- `DecisionEngineProtocol` (Protocol)
- `ToolExecutor` (Protocol)

✅ **Retrieval Protocols** (3):
- `RetrievalStrategy` (Protocol)
- `RetrievalResult` (dataclass)
- `SpringActivationMetadata` (dataclass)

**Subtotal**: 15 protocols/types

---

### 2.2 Scattered Protocols (Need Action)

#### 🔴 **Needs Consolidation** - `HoloLoom/memory/protocol.py`

**Problem**: Duplicate/fallback `MemoryStore` protocol definition (lines 112-125)

**Current State**:
```python
try:
    from HoloLoom.protocols import MemoryStore
except ImportError:
    # Fallback if canonical protocol unavailable
    @runtime_checkable
    class MemoryStore(Protocol):
        ...
```

**Action Required**:
1. Move canonical `MemoryStore` from `core.py` to `memory_protocols.py` (new file)
2. Remove fallback definition
3. Update `protocols/__init__.py` to export from new location
4. Fix circular import using `TYPE_CHECKING`

**Impact**: Medium - Used by memory backends

---

#### 🟢 **Keep Domain-Specific** - `HoloLoom/search/protocol.py`

**Protocols**:
- `SearchProvider` (Protocol)
- `ContentScraper` (Protocol)

**Rationale**: Search protocols are domain-specific and should remain in search module

**Action Required**: ✅ None - Already properly organized

**Note**: Could optionally create `HoloLoom/protocols/search.py` and re-export for consistency

---

#### 🟢 **Keep Domain-Specific** - `HoloLoom/spinningWheel/protocol.py`

**Protocols**:
- `SpinnerProtocol` (Protocol)
- `BaseSpinner` (ABC)

**Plus 5 supporting types**: SpinnerStatus, SpinnerCapabilities, SpinResult, ImportanceSignals, SpinnerCheckpoint

**Rationale**: Comprehensive spinner framework (827 lines) - too domain-specific for canonical protocols

**Action Required**: ✅ None - Well organized

**Note**: Could create `HoloLoom/protocols/spinning_wheel.py` as alias for consistency

---

#### 🟢 **Keep Domain-Specific** - `HoloLoom/spinningWheel/ocr_protocol.py`

**Protocols**:
- `OCRProtocol` (Protocol)
- `BaseOCRBackend` (ABC)
- `OCRBackendChain` (utility class)

**Plus 4 supporting types**: OCRQuality, OCROutputFormat, OCRBoundingBox, OCRResult

**Rationale**: OCR abstraction (587 lines) - domain-specific to document processing

**Action Required**: ✅ None - Well organized

---

#### 🟢 **Keep Domain-Specific** - `HoloLoom/writing/core/protocol.py`

**Protocols**:
- `WriterProtocol` (Protocol)
- `ComposerProtocol` (Protocol)
- `RefinerProtocol` (Protocol)
- `ModeWriterProtocol` (Protocol)

**Plus 8 supporting types**: WritingMode, RefinementStrategy, StyleGuide, OutputFormat, WritingContext, RefinementPass, WritingResult, QUALITY_DIMENSIONS

**Rationale**: Writing system protocols (319 lines) - domain-specific to content generation

**Action Required**: ✅ None - Properly organized

---

#### 🟢 **Keep Domain-Specific** - `HoloLoom/ts_core/base.py`

**Protocols**:
- `ThompsonSampler` (Protocol)

**Plus 2 supporting types**: ThompsonSamplerConfig, Observation

**Rationale**: Thompson Sampling protocol - domain-specific to bandit algorithms

**Action Required**: ✅ None - Properly placed

---

#### 🟡 **Remove Deprecated** - `HoloLoom/modules/Features.py`

**Protocols** (DEPRECATED):
- `_DeprecatedMotifDetector` (lines 38-47)
- `_DeprecatedEmbedder` (lines 51-63)

**Current State**: Already has deprecation warnings (lines 71-78)

**Action Required**:
1. ✅ Keep deprecation warnings
2. ✅ Ensure all imports use canonical `HoloLoom.protocols`
3. Consider removing in next major version

**Impact**: Low - Already deprecated

---

## 3. Circular Import Analysis

### 3.1 Current Circular Import

**Problem**: `HoloLoom.protocols.core` → `HoloLoom.memory.protocol` → `HoloLoom.protocols`

**Root Cause**: `MemoryStore` protocol needs to import `Memory`, `MemoryQuery`, `RetrievalResult` from `memory.protocol`

**Current Workaround**: `MemoryStore` is commented out in `protocols/__init__.py`

**Solution Strategy**:

#### Option A: Move Data Types to Protocols Package
```
HoloLoom/protocols/
├── memory_types.py      # NEW: Memory, MemoryQuery, RetrievalResult
├── memory_protocols.py  # NEW: MemoryStore, MemoryNavigator, PatternDetector
└── __init__.py          # Export all
```

**Pros**:
- Clean separation of types and protocols
- No circular imports
- Single source of truth

**Cons**:
- Breaking change (imports need updating)

#### Option B: Use TYPE_CHECKING Pattern
```python
# In HoloLoom/protocols/core.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.memory.protocol import Memory, MemoryQuery, RetrievalResult
else:
    Memory = Any
    MemoryQuery = Any
    RetrievalResult = Any
```

**Pros**:
- Minimal changes
- Backward compatible

**Cons**:
- Runtime type checking unavailable
- More complex

#### ✅ **Recommended**: Option A

---

### 3.2 Potential Circular Imports (Safe)

These are already handled properly with fallback patterns:

1. **`HoloLoom.documentation.types` ↔ `HoloLoom.protocols`**
   - Status: ✅ Safe - uses try/except fallback pattern
   - Location: Multiple files

2. **`HoloLoom.protocols.retrieval` → `HoloLoom.documentation.types`**
   - Status: ✅ Safe - one-way import with fallback

---

## 4. Consolidation Actions

### 4.1 High Priority - Fix MemoryStore Circular Import

**Task**: Create `memory_types.py` and `memory_protocols.py`

**Files to Create**:

1. **`HoloLoom/protocols/memory_types.py`** (NEW)
   - Move from `HoloLoom/memory/protocol.py`:
     - `Memory` (dataclass)
     - `MemoryQuery` (dataclass)
     - `RetrievalResult` (dataclass) - NOTE: Also in `protocols/retrieval.py`! Need to dedupe
     - `Strategy` (Enum)
     - `QueryMode` (Enum)

2. **`HoloLoom/protocols/memory_protocols.py`** (NEW)
   - Move from `HoloLoom/protocols/core.py`:
     - `MemoryStore` (Protocol)
     - `MemoryNavigator` (Protocol)
     - `PatternDetector` (Protocol)

**Files to Update**:

3. **`HoloLoom/protocols/__init__.py`**
   - Remove comment on line 65-66
   - Import from new locations:
     ```python
     from .memory_types import Memory, MemoryQuery, Strategy, QueryMode
     from .memory_protocols import MemoryStore, MemoryNavigator, PatternDetector
     ```
   - Update `__all__` exports

4. **`HoloLoom/memory/protocol.py`**
   - Remove duplicate type definitions (lines 15-97)
   - Remove fallback protocol (lines 112-125)
   - Import from canonical location:
     ```python
     from HoloLoom.protocols import MemoryStore, Memory, MemoryQuery, RetrievalResult
     ```

5. **`HoloLoom/protocols/core.py`**
   - Remove memory protocols (moved to `memory_protocols.py`)
   - Keep or remove file depending on whether other protocols remain

**Impact**:
- ✅ Fixes circular import
- ✅ Enables `MemoryStore` in canonical exports
- ⚠️ Breaking change for direct imports from `memory.protocol`

---

### 4.2 Medium Priority - Deduplicate RetrievalResult

**Problem**: `RetrievalResult` is defined in **two locations**:
- `HoloLoom/protocols/retrieval.py` (canonical)
- `HoloLoom/memory/protocol.py` (duplicate)

**Differences**:
```python
# protocols/retrieval.py (canonical)
@dataclass
class RetrievalResult:
    shards: List[MemoryShard]
    strategy: str
    query_text: str
    k_requested: int
    k_returned: int
    retrieval_time_ms: float
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    metadata: dict

# memory/protocol.py (duplicate)
@dataclass
class RetrievalResult:
    memories: List[Memory]
    scores: List[float]
    strategy_used: str
    metadata: Dict[str, Any]
```

**Decision**: These are **different types** serving different purposes:
- `retrieval.RetrievalResult` - For retrieval strategy pattern (shards)
- `memory.RetrievalResult` - For memory backend pattern (memories)

**Action Required**:
1. Rename to avoid confusion:
   - `retrieval.RetrievalResult` → `StrategyRetrievalResult`
   - `memory.RetrievalResult` → `MemoryRetrievalResult`
2. Move both to `HoloLoom/protocols/memory_types.py`
3. Update all references

**Impact**: Medium - Affects retrieval and memory modules

---

### 4.3 Low Priority - Organize Domain-Specific Protocols

**Option**: Create protocol "catalog" in `protocols/__init__.py`

```python
# ============================================================================
# Domain-Specific Protocol References
# ============================================================================
#
# These protocols are intentionally kept in their domain modules:
#
# Search Protocols:
#   - from HoloLoom.search.protocol import SearchProvider, ContentScraper
#
# Spinner Protocols:
#   - from HoloLoom.spinningWheel.protocol import SpinnerProtocol
#
# OCR Protocols:
#   - from HoloLoom.spinningWheel.ocr_protocol import OCRProtocol
#
# Writing Protocols:
#   - from HoloLoom.writing.core.protocol import WriterProtocol, ComposerProtocol
#
# Thompson Sampling Protocols:
#   - from HoloLoom.ts_core.base import ThompsonSampler
#
```

**Benefit**: Single "phonebook" for all protocols without moving domain logic

---

### 4.4 Immediate - Remove Deprecated Fallbacks

**Files to Update**:

1. **`HoloLoom/modules/Features.py`**
   - ✅ Keep deprecation warnings (already present)
   - ✅ Verify all imports use canonical protocols
   - Document removal plan for next major version

---

## 5. Implementation Plan

### Phase 1: Fix MemoryStore Circular Import (Breaking Change)

**Priority**: 🔴 HIGH
**Estimated Time**: 2-3 hours
**Breaking**: ⚠️ YES

**Steps**:
1. Create `HoloLoom/protocols/memory_types.py`
2. Create `HoloLoom/protocols/memory_protocols.py`
3. Update `HoloLoom/protocols/__init__.py`
4. Update `HoloLoom/memory/protocol.py`
5. Remove `HoloLoom/protocols/core.py` (if empty)
6. Update all imports across codebase
7. Run tests

**Files to Update** (estimated 15-20 files):
- `HoloLoom/memory/*.py` (backends, graph, cache, unified)
- `HoloLoom/weaving_orchestrator.py`
- `HoloLoom/hololoom.py`
- `HoloLoom/server/agentic_api.py`
- Tests that import memory protocols

---

### Phase 2: Deduplicate RetrievalResult (Breaking Change)

**Priority**: 🟡 MEDIUM
**Estimated Time**: 1-2 hours
**Breaking**: ⚠️ YES

**Steps**:
1. Rename `retrieval.RetrievalResult` → `StrategyRetrievalResult`
2. Rename `memory.RetrievalResult` → `MemoryRetrievalResult`
3. Move both to `memory_types.py`
4. Update all references
5. Run tests

**Files to Update** (estimated 8-10 files):
- `HoloLoom/memory/retrieval_strategies.py`
- `HoloLoom/memory/backends/*.py`
- Tests using retrieval results

---

### Phase 3: Documentation and Cleanup (Non-Breaking)

**Priority**: 🟢 LOW
**Estimated Time**: 1 hour
**Breaking**: ✅ NO

**Steps**:
1. Add protocol catalog to `protocols/__init__.py`
2. Update `CLAUDE.md` with new protocol organization
3. Create protocol migration guide for external users
4. Document deprecation timeline for `Features.py` protocols

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Test**: `HoloLoom/protocols/__init__.py` imports work

```python
def test_protocol_imports():
    """Verify all canonical protocols are importable."""
    from HoloLoom.protocols import (
        # Core Types
        ComplexityLevel,
        ProvenanceTrace,
        MythRLResult,

        # Memory Types
        Memory,
        MemoryQuery,
        Strategy,
        QueryMode,
        MemoryRetrievalResult,
        StrategyRetrievalResult,

        # Memory Protocols
        MemoryStore,
        MemoryNavigator,
        PatternDetector,

        # Core Feature Protocols
        Embedder,
        MotifDetector,
        PolicyEngine,

        # Shuttle Protocols
        PatternSelectionProtocol,
        DecisionEngineProtocol,

        # Retrieval Protocols
        RetrievalStrategy,
    )
    assert True  # If we get here, all imports worked
```

---

### 6.2 Integration Tests

**Test**: No circular imports

```bash
python -c "from HoloLoom.protocols import MemoryStore; print('✓ No circular import')"
```

**Test**: Memory backends still work

```python
async def test_memory_backend_after_consolidation():
    from HoloLoom.memory.backend_factory import create_memory_backend
    from HoloLoom.config import Config

    config = Config.fast()
    backend = await create_memory_backend(config)

    # Should work with consolidated protocols
    assert hasattr(backend, 'store')
    assert hasattr(backend, 'recall')
```

---

### 6.3 Backward Compatibility Tests

**Test**: Deprecated imports still work (with warning)

```python
def test_deprecated_features_protocols():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from HoloLoom.modules.Features import MotifDetector, Embedder

        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
```

---

## 7. Migration Guide for External Users

### For Library Users

**Before** (Old imports - will break):
```python
from HoloLoom.memory.protocol import MemoryStore, Memory
```

**After** (New canonical imports):
```python
from HoloLoom.protocols import MemoryStore, Memory
```

**Compatibility Period**: 2 minor versions (with deprecation warnings)

---

### For Internal Developers

**Pattern**: Always import from `HoloLoom.protocols` first

```python
# ✅ CORRECT
from HoloLoom.protocols import MemoryStore, PolicyEngine, Embedder

# ❌ WRONG
from HoloLoom.memory.protocol import MemoryStore
from HoloLoom.policy.unified import PolicyEngine
from HoloLoom.modules.Features import Embedder
```

---

## 8. Risk Assessment

### High Risk

**Risk**: Breaking existing code that imports from `memory.protocol`
**Mitigation**:
- Deprecation warnings in current version
- Backward-compatible imports for 2 versions
- Clear migration guide

**Risk**: Circular import regression
**Mitigation**:
- Comprehensive testing of import order
- CI/CD checks for circular imports

---

### Medium Risk

**Risk**: RetrievalResult renaming confusion
**Mitigation**:
- Clear naming convention
- Documentation of differences
- Type aliases for transition period

---

### Low Risk

**Risk**: Test failures after consolidation
**Mitigation**:
- Run full test suite before/after
- Fix imports incrementally

---

## 9. Success Criteria

✅ **Complete** when:

1. All protocols importable from `HoloLoom.protocols`
2. No circular import errors
3. No duplicate protocol definitions (except documented deprecations)
4. `MemoryStore` exported in `protocols/__init__.py`
5. All tests passing
6. Migration guide published

---

## 10. Timeline

**Total Estimated Time**: 4-6 hours

| Phase | Time | Breaking | Priority |
|-------|------|----------|----------|
| Phase 1: Fix MemoryStore | 2-3 hours | ⚠️ YES | 🔴 HIGH |
| Phase 2: Dedupe RetrievalResult | 1-2 hours | ⚠️ YES | 🟡 MEDIUM |
| Phase 3: Documentation | 1 hour | ✅ NO | 🟢 LOW |

---

## 11. Recommendations

### Immediate Actions

1. ✅ **Implement Phase 1** - Highest impact, fixes main architectural issue
2. ✅ **Add deprecation warnings** - Prepare users for changes
3. ✅ **Update CI/CD** - Add circular import detection

### Future Improvements

1. **Protocol Versioning** - Consider adding `__version__` to protocols
2. **Protocol Registry** - Runtime registration system for dynamic protocol discovery
3. **Protocol Documentation Generator** - Auto-generate protocol docs from docstrings

---

## 12. Appendix: File Locations Reference

### Canonical Protocol Package
```
HoloLoom/protocols/
├── __init__.py                # Main exports
├── types.py                   # ComplexityLevel, ProvenanceTrace, MythRLResult
├── core.py                    # Memory protocols (TO BE SPLIT)
├── core_features.py           # Feature protocols
├── shuttle.py                 # Shuttle protocols
├── retrieval.py               # Retrieval protocols
├── memory_types.py            # NEW: Memory types
└── memory_protocols.py        # NEW: Memory protocols
```

### Domain-Specific Protocols (Keep as-is)
```
HoloLoom/search/protocol.py                # SearchProvider, ContentScraper
HoloLoom/spinningWheel/protocol.py         # SpinnerProtocol
HoloLoom/spinningWheel/ocr_protocol.py     # OCRProtocol
HoloLoom/writing/core/protocol.py          # WriterProtocol, ComposerProtocol
HoloLoom/ts_core/base.py                   # ThompsonSampler
```

### Deprecated (Remove in v2.0)
```
HoloLoom/modules/Features.py               # _DeprecatedMotifDetector, _DeprecatedEmbedder
```

---

## 13. References

- Original Task: Phase 0, Task 7 - Protocol Consolidation
- Related Docs:
  - `docs/guides/PROTOCOL_MIGRATION_GUIDE.md`
  - `docs/completion-logs/TASK1_1_PROTOCOL_STANDARDIZATION_COMPLETE.md`
- Previous Work:
  - Task 1.1: Protocol Standardization (Oct 2025)
  - Memory simplification (Oct 2025)

---

**END OF CONSOLIDATION PLAN**
