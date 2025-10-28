# Protocol Standardization Plan (Task 1.1)

## Current State Analysis

### Existing Protocols in `HoloLoom/protocols/__init__.py`
✅ Already standardized:
- **Embedder** - Vector generation (synchronous encode method)
- **MotifDetector** - Pattern detection in text
- **PolicyEngine** - Decision making with Thompson Sampling
- **MemoryStore** - Storage and retrieval
- **MemoryNavigator** - Graph traversal
- **PatternDetector** - Pattern discovery in memory
- **RoutingStrategy** - Mode selection
- **ExecutionEngine** - Tool execution

### New Protocols in `dev/protocol_modules_mythrl.py`
🆕 Need to migrate:
- **PatternSelectionProtocol** - Processing pattern selection (grows with complexity)
- **DecisionEngineProtocol** - Multi-criteria decision making
- **FeatureExtractionProtocol** - Multi-scale feature extraction (Matryoshka)
- **WarpSpaceProtocol** - Mathematical manifold operations (NON-NEGOTIABLE)
- **ToolExecutionProtocol** - Tool execution with necessity assessment
- **MemoryBackendProtocol** - Enhanced memory with multipass support

### Supporting Types in `dev/protocol_modules_mythrl.py`
- **ComplexityLevel** - LITE(3), FAST(5), FULL(7), RESEARCH(9)
- **ProvenceTrace** - Full computational provenance
- **MythRLResult** - Result with provenance
- **MythRLShuttle** - Creative orchestrator (separate migration in Task 1.2)

## Protocol Overlap Analysis

### 1. Feature Extraction Domain
**Current:** `Embedder` (synchronous vector generation)
**New:** `FeatureExtractionProtocol` (multi-scale extraction with motifs)

**Resolution:** Keep both, clarify roles
- `Embedder` = Low-level vector generation (used by FeatureExtraction)
- `FeatureExtractionProtocol` = High-level feature orchestration

### 2. Memory Domain
**Current:** `MemoryStore` (basic CRUD)
**New:** `MemoryBackendProtocol` (multipass crawling)

**Resolution:** Enhance existing
- Add `retrieve_with_threshold()` to `MemoryStore`
- Add `get_related()` to `MemoryNavigator` (already exists!)
- Add `get_context_subgraph()` to `MemoryNavigator`

### 3. Decision Domain
**Current:** `PolicyEngine` (neural + Thompson Sampling)
**New:** `DecisionEngineProtocol` (multi-criteria optimization)

**Resolution:** Keep both, clarify roles
- `PolicyEngine` = Reactive decision-making with learning
- `DecisionEngineProtocol` = Strategic multi-criteria optimization

### 4. Pattern Domain
**Current:** `PatternDetector` (memory access patterns)
**New:** `PatternSelectionProtocol` (processing patterns)

**Resolution:** Keep both, rename for clarity
- `PatternDetector` → `MemoryPatternDetector`
- `PatternSelectionProtocol` = Processing pattern selection

### 5. Tool Execution Domain
**Current:** `ToolExecutor` (execute tool calls)
**New:** `ToolExecutionProtocol` (with necessity assessment)

**Resolution:** Enhance existing
- Add `assess_tool_necessity()` method to `ToolExecutor`

## Migration Strategy

### Phase 1: Extract Core Types (30 min)
1. Create `HoloLoom/protocols/types.py`
2. Move `ComplexityLevel`, `ProvenceTrace`, `MythRLResult` from dev/
3. Update imports in dev/protocol_modules_mythrl.py

### Phase 2: Migrate New Protocols (1 hour)
1. Add to `HoloLoom/protocols/__init__.py`:
   - `PatternSelectionProtocol`
   - `FeatureExtractionProtocol`
   - `WarpSpaceProtocol` (NON-NEGOTIABLE)
   - `DecisionEngineProtocol`

2. Enhance existing protocols:
   - `MemoryStore`: Add `retrieve_with_threshold()`
   - `MemoryNavigator`: Add `get_context_subgraph()`
   - `ToolExecutor`: Add `assess_tool_necessity()`

3. Rename for clarity:
   - `PatternDetector` → `MemoryPatternDetector`

### Phase 3: Update Imports (30 min)
1. Update `dev/protocol_modules_mythrl.py` to import from HoloLoom/protocols
2. Update `dev/narrative_depth_protocol.py` to import from HoloLoom/protocols
3. Search and replace throughout codebase

### Phase 4: Backward Compatibility (30 min)
1. Add deprecated aliases in old locations
2. Add deprecation warnings
3. Update documentation

### Phase 5: Testing (30 min)
1. Run existing tests
2. Verify protocol compliance
3. Test multipass memory crawling

## File Structure After Migration

```
HoloLoom/
├── protocols/
│   ├── __init__.py              # All protocol definitions
│   ├── types.py                 # ComplexityLevel, ProvenceTrace, MythRLResult
│   └── README.md                # Protocol usage guide
├── protocols.py                 # Backward compatibility (deprecated)
└── ...

dev/
├── protocol_modules_mythrl.py   # Imports from HoloLoom/protocols, focuses on MythRLShuttle
├── narrative_depth_protocol.py  # Imports from HoloLoom/protocols
└── ...
```

## Implementation Checklist

### Phase 1: Core Types
- [ ] Create `HoloLoom/protocols/types.py`
- [ ] Move `ComplexityLevel` enum
- [ ] Move `ProvenceTrace` dataclass
- [ ] Move `MythRLResult` dataclass
- [ ] Update imports in dev/

### Phase 2: Protocol Migration
- [ ] Add `PatternSelectionProtocol` to protocols/__init__.py
- [ ] Add `FeatureExtractionProtocol` to protocols/__init__.py
- [ ] Add `WarpSpaceProtocol` to protocols/__init__.py
- [ ] Add `DecisionEngineProtocol` to protocols/__init__.py
- [ ] Enhance `MemoryStore` with threshold retrieval
- [ ] Enhance `MemoryNavigator` with subgraph method
- [ ] Enhance `ToolExecutor` with necessity assessment
- [ ] Rename `PatternDetector` → `MemoryPatternDetector`

### Phase 3: Import Updates
- [ ] Update dev/protocol_modules_mythrl.py imports
- [ ] Update dev/narrative_depth_protocol.py imports
- [ ] Search and replace in HoloLoom/ modules
- [ ] Update test files

### Phase 4: Compatibility
- [ ] Add deprecation warnings to old imports
- [ ] Create alias in protocols.py
- [ ] Update CLAUDE.md documentation

### Phase 5: Validation
- [ ] Run `test_unified_policy.py`
- [ ] Run protocol compliance tests
- [ ] Test multipass memory demo
- [ ] Verify no import errors

## Success Criteria

✅ All protocols in single source of truth: `HoloLoom/protocols/__init__.py`
✅ No duplicate protocol definitions across codebase
✅ Clear separation of concerns (low-level vs high-level)
✅ Backward compatibility maintained with deprecation warnings
✅ All tests passing
✅ Documentation updated

## Timeline

**Total estimated time: 3 hours**
- Phase 1: 30 minutes
- Phase 2: 1 hour
- Phase 3: 30 minutes
- Phase 4: 30 minutes
- Phase 5: 30 minutes

**Completion target: End of Day 2 (Phase 1)**
