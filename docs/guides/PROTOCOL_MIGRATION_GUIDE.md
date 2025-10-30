# Protocol Migration Guide
## Migrating to Canonical Protocols (HoloLoom/protocols/__init__.py)

**Date**: 2025-10-27  
**Status**: In Progress

---

## Overview

Phase 2 consolidation created canonical protocol definitions in `HoloLoom/protocols/__init__.py`. This guide documents the migration process to update all modules to use these canonical protocols.

## Canonical Protocol Location

```python
from HoloLoom.protocols import (
    # Core Protocols
    Embedder,
    MotifDetector,
    PolicyEngine,
    
    # Memory Protocols
    MemoryStore,
    MemoryNavigator,
    PatternDetector,
    
    # Routing Protocols
    RoutingStrategy,
    ExecutionEngine,
    
    # Tool Protocols
    ToolExecutor,
    ToolRegistry
)
```

##Files Requiring Migration

### 1. **HoloLoom/policy/unified.py** (PolicyEngine)
**Current**: Defines `PolicyEngine` protocol locally
**Action**: Import from `HoloLoom.protocols`, add deprecation warning to local definition
**Priority**: HIGH (Core policy component)

```python
# BEFORE
from typing import Protocol
class PolicyEngine(Protocol):
    def choose_action(...):  ...

# AFTER
from HoloLoom.protocols import PolicyEngine
# Keep local definition with deprecation warning for compatibility
import warnings
class _DeprecatedPolicyEngine(Protocol):  # Rename with underscore
    def choose_action(...): ...
warnings.warn("Importing PolicyEngine from policy.unified is deprecated. Use HoloLoom.protocols", DeprecationWarning)
```

### 2. **HoloLoom/embedding/spectral.py** (Embedder)
**Current**: Defines `Embedder` protocol locally
**Action**: Import from `HoloLoom.protocols`
**Priority**: HIGH (Used in all embedding operations)

### 3. **HoloLoom/memory/protocol.py** (MemoryStore, MemoryNavigator, PatternDetector)
**Current**: Defines 3 memory protocols locally
**Action**: Import all 3 from `HoloLoom.protocols`
**Priority**: HIGH (Core memory backend interface)

### 4. **HoloLoom/Modules/Features.py** (MotifDetector, Embedder)
**Current**: Defines `MotifDetector` and duplicate `Embedder`
**Action**: Import from `HoloLoom.protocols`, remove duplicates
**Priority**: MEDIUM (Feature extraction)

### 5. **HoloLoom/memory/routing/protocol.py** (RoutingStrategy)
**Current**: Defines `RoutingStrategy`, `LearnableStrategy`, `ExperimentalStrategy`
**Action**: Import `RoutingStrategy` from canonical, keep experimental ones local
**Priority**: MEDIUM (Routing intelligence)

### 6. **HoloLoom/memory/routing/execution_patterns.py** (ExecutionEngine)
**Current**: Defines `ExecutionEngine` protocol locally
**Action**: Import from `HoloLoom.protocols`
**Priority**: MEDIUM (Tool execution)

### 7. **HoloLoom/memory/graph.py** (KGStore)
**Current**: Defines `KGStore` protocol locally
**Action**: Keep local (domain-specific protocol, not in canonical set)
**Priority**: LOW (Specialized protocol)

### 8. **HoloLoom/memory/cache.py** (Retriever)
**Current**: Defines `Retriever` protocol locally
**Action**: Keep local (internal caching protocol, not in canonical set)
**Priority**: LOW (Internal use only)

---

## Migration Strategy

### Phase 1: High Priority Files (DONE FIRST)
1. policy/unified.py - PolicyEngine
2. embedding/spectral.py - Embedder
3. memory/protocol.py - MemoryStore, MemoryNavigator, PatternDetector

### Phase 2: Medium Priority Files
4. Modules/Features.py - MotifDetector, Embedder (duplicate)
5. memory/routing/protocol.py - RoutingStrategy
6. memory/routing/execution_patterns.py - ExecutionEngine

### Phase 3: Validation
- Run test suite to verify no circular imports
- Check IDE autocomplete still works
- Verify type checking passes
- Test all demos

---

## Migration Pattern

### Step 1: Add canonical import at top of file
```python
# Add to imports section
from HoloLoom.protocols import ProtocolName
```

### Step 2: Rename local protocol definition (if keeping for compatibility)
```python
# Rename local definition
class _DeprecatedProtocolName(Protocol):  # Add underscore prefix
    """
    DEPRECATED: Import from HoloLoom.protocols instead.
    This local definition will be removed in a future version.
    """
    ...

# Add deprecation warning
import warnings
warnings.warn(
    "Importing ProtocolName from this module is deprecated. "
    "Use 'from HoloLoom.protocols import ProtocolName' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Create alias for backward compatibility
ProtocolName = _DeprecatedProtocolName
```

### Step 3: Update all internal usages
```python
# BEFORE
from HoloLoom.policy.unified import PolicyEngine

# AFTER
from HoloLoom.protocols import PolicyEngine
```

### Step 4: Test
```powershell
$env:PYTHONPATH = "."; python -c "from HoloLoom.protocols import PolicyEngine; print('✓')"
$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py
```

---

## Benefits of Migration

1. **Single Source of Truth**: All protocols defined in one canonical location
2. **Better IDE Support**: Autocomplete and type hints work consistently
3. **Reduced Duplication**: No more duplicate protocol definitions
4. **Easier Maintenance**: Update protocols in one place
5. **Clear Architecture**: Protocol contracts separate from implementations
6. **Type Safety**: Consistent protocol definitions across modules

---

## Backward Compatibility

- Old imports will continue to work with deprecation warnings
- No breaking changes in Phase 2
- Deprecation warnings guide developers to canonical imports
- Local aliases maintain compatibility during transition

---

## Testing Checklist

After migration, verify:
- [ ] All tests pass (`pytest tests/`)
- [ ] No circular import errors
- [ ] Type checking passes (`mypy HoloLoom/`)
- [ ] Demos run successfully
- [ ] IDE autocomplete works for protocols
- [ ] Deprecation warnings appear for old imports
- [ ] New code uses canonical imports

---

## Next Steps

1. Migrate high-priority files (policy, embedding, memory/protocol)
2. Run test suite
3. Migrate medium-priority files
4. Update documentation
5. Add type checking to CI/CD pipeline

---

**Status**: Ready for implementation  
**Author**: mythRL Team (Blake + Claude)  
**Last Updated**: 2025-10-27
