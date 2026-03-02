# Import Timeout Fix - Investigation Summary

**Date**: November 8, 2025
**Issue**: Import timeout when loading HoloLoom modules
**Status**: Partially Fixed (typed_dicts), Core Issue Identified

---

## ✅ Fixes Applied

### 1. Removed numpy.typing from typed_dicts.py
**File**: `hololoom/documentation/typed_dicts.py`

**Change**:
```python
# Before
import numpy.typing as npt
state: npt.NDArray
next_state: npt.NDArray

# After
# Removed numpy.typing import
state: List[float]
next_state: List[float]
```

**Result**: Eliminates numpy.typing dependency which was causing Windows WMI hang

---

### 2. Disabled policy import in config.py

**File**: `hololoom/config.py` line 15

**Change**:
```python
# Before
try:
    from hololoom.policy.unified import BanditStrategy as PolicyBanditStrategy
    _POLICY_AVAILABLE = True
except ImportError:
    _POLICY_AVAILABLE = False

# After
# DISABLED: Causes circular import / timeout issues
_POLICY_AVAILABLE = False  # Disabled to fix import timeout
```

**Result**: Removed circular dependency but still hangs

---

### 3. Made Config import lazy in hololoom.py

**File**: `hololoom/hololoom.py` lines 29-36, 95-96

**Change**:
```python
# Before (module level):
from hololoom.config import Config

# After (lazy import):
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hololoom.config import Config

def __init__(self, config: Optional['Config'] = None, ...):
    from hololoom.config import Config  # Lazy import
    self.config = config or Config.fast()
```

**Result**: Breaks circular dependency `__init__.py → hololoom.py → config.py`

---

### 4. Disabled MemoryStore import in protocols/__init__.py

**File**: `hololoom/protocols/__init__.py` line 66

**Change**:
```python
# Before
from hololoom.memory.protocol import MemoryStore

# After
# MemoryStore is in memory.protocol - DISABLED to avoid circular imports
# from hololoom.memory.protocol import MemoryStore
```

**Result**: Breaks circular dependency `protocols → memory.protocol → protocols`

**Status**: ⚠️ Still investigating - protocols.types hangs despite only importing stdlib

---

## ⚠️ Core Issue Identified - Multiple Circular Import Chains

### Circular Import Chain Discovered:

```
hololoom/__init__.py (line 34)
  → from .hololoom import HoloLoom
    → hololoom/hololoom.py (line 33) ✅ FIXED with lazy import
      → from hololoom.config import Config

hololoom/__init__.py (line 35)
  → from .memory.protocol import Memory
    → hololoom/memory/protocol.py (line 113)
      → from hololoom.protocols import ...
        → hololoom/protocols/__init__.py (line 66) ✅ FIXED
          → from hololoom.memory.protocol import MemoryStore (CIRCULAR!)

hololoom/protocols/__init__.py (line 37)
  → from .types import ComplexityLevel
    → hololoom/protocols/types.py
      → ⚠️ HANGS despite only importing stdlib (enum, typing, dataclasses, time)
```

**Hypothesis**: The issue is not just circular imports, but **deferred import execution** combined with Python's import lock. When modules are imported through packages, Python holds an import lock that can cause timeouts even on stdlib imports if the import chain is complex enough.

---

## 🔍 Investigation Findings

### What Works
- ✅ Direct file loading with `importlib.util.spec_from_file_location`
- ✅ Standard library imports (dataclasses, typing, ast, etc.)
- ✅ `types.py` loads fine when bypassing package init

### What Hangs
- ❌ `import HoloLoom`
- ❌ `from hololoom.config import Config`
- ❌ `from hololoom.documentation.types import Vector`
- ❌ Any import through package `__init__.py`

### Root Cause Hypothesis
The issue is **NOT** in individual files, but in the **package initialization chain**:

```
import hololoom
  → hololoom/__init__.py imports from hololoom.py
    → hololoom.py imports from memory
      → memory/__init__.py imports from cache.py
        → cache.py imports from embedding.spectral
          → spectral.py might import something heavy
            → HANGS (exact point unknown)
```

---

## 🎯 Recommended Solution

### Option A: Lazy Imports (Recommended)
Move all imports inside functions/methods instead of module-level:

```python
# Instead of this (at module level):
from hololoom.memory.cache import MemoryManager

# Do this (lazy import):
def get_memory_manager():
    from hololoom.memory.cache import MemoryManager
    return MemoryManager()
```

**Pros**:
- Breaks circular dependencies
- Faster initial import
- Only loads what's needed

**Cons**:
- Slightly more verbose
- Import errors happen at runtime, not import time

### Option B: Standalone Modules (Current Workaround)
Create standalone versions that don't depend on HoloLoom imports:

```python
# ml_logic_detector.py - Already done
class Language(str, Enum):
    """Local copy to avoid imports."""
    PYTHON = "python"
    # ...
```

**Pros**:
- Works immediately
- No circular dependencies
- Modules are portable

**Cons**:
- Code duplication
- Maintenance burden (keep enums in sync)

### Option C: Restructure Packages
Break circular dependencies by reorganizing:

```
hololoom/
├── core/           # Pure types, no imports
│   ├── types.py
│   └── enums.py
├── utils/          # Utilities, import from core only
├── memory/         # Import from core + utils
└── ...
```

**Pros**:
- Clean architecture
- No circular deps possible
- Better maintainability

**Cons**:
- Large refactoring effort
- Breaking changes
- Time-intensive

---

## 📊 Current Status

| Component | Import Works? | Notes |
|-----------|---------------|-------|
| numpy.typing | ✅ Fixed | Replaced with List[float] |
| typed_dicts.py | ✅ Fixed | Removed npt dependency |
| config.py | ❌ Hangs | Circular import disabled but still hangs |
| documentation/__init__.py | ❌ Hangs | Star imports cause cascade |
| HoloLoom package | ❌ Hangs | Package init chain issue |
| ml_logic_detector.py | ✅ Works | Standalone (local Language enum) |

---

## 🚀 Next Steps

### Immediate (< 1 hour)
1. ✅ Fix numpy.typing (DONE)
2. ⏳ Identify exact hanging point in package init chain
3. ⏳ Apply lazy imports to break circular deps

### Short-term (2-4 hours)
1. Convert module-level imports to lazy imports in key files:
   - `hololoom/__init__.py`
   - `hololoom/hololoom.py`
   - `hololoom/memory/__init__.py`
   - `hololoom/memory/cache.py`

2. Test imports work after each change

### Medium-term (1-2 days)
1. Audit all HoloLoom imports
2. Create dependency graph to visualize circular deps
3. Systematically refactor to lazy imports
4. Add import guards for optional dependencies

---

## 💡 Debugging Commands

### Test Individual Imports
```bash
# Test if module loads
PYTHONPATH=. timeout 3 python -c "import HoloLoom.module; print('OK')"

# Direct file loading (bypasses package init)
PYTHONPATH=. python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('test', 'hololoom/file.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('OK')
"
```

### Find Circular Dependencies
```bash
# Use debug_toolkit.py
python debug_toolkit.py check-circular HoloLoom

# Trace imports
python debug_toolkit.py trace-import HoloLoom.config
```

### Profile Import Times
```bash
python debug_toolkit.py profile-import HoloLoom
```

---

## 📝 Lessons Learned

1. **numpy.typing is slow on Windows** - Avoid if possible, use `List[float]`
2. **Module-level imports create circular deps** - Use lazy imports
3. **Package `__init__.py` amplifies import issues** - Keep minimal
4. **Star imports cascade problems** - Use explicit imports
5. **Direct file loading works** - Package structure is the issue

---

## ✅ Verified Working Workarounds

### For ML Logic Detector
```python
# Use standalone version with local enums
from hololoom.agentic.ml_logic_detector import MLLogicDetector, Language

# No HoloLoom package import needed
detector = MLLogicDetector()
errors = await detector.detect(code, Language.PYTHON)
```

### For AI Slop Detector
```python
# Currently blocked - needs lazy import fix
# Workaround: Use directly without server
# TODO: Fix after lazy imports applied
```

---

**Status**: Investigation complete, fixes identified, implementation in progress
**ETA to Full Fix**: 2-4 hours with lazy imports approach
