# HoloLoom Framework + Apps Dependency Graph

## Current State (Before Separation)

```
mythRL/
└── HoloLoom/
    ├── [FRAMEWORK CORE]
    │   ├── weaving_shuttle.py
    │   ├── weaving_orchestrator.py
    │   ├── loom/
    │   ├── chrono/
    │   ├── resonance/
    │   ├── warp/
    │   ├── convergence/
    │   ├── fabric/
    │   ├── reflection/
    │   ├── embedding/
    │   ├── motif/
    │   ├── memory/
    │   ├── policy/
    │   └── config.py
    │
    └── [NARRATIVE APP - MIXED IN]
        ├── narrative_intelligence.py    ⚠️ Should be separate
        ├── narrative_cache.py            ⚠️ Should be separate
        ├── narrative_loop_engine.py      ⚠️ Should be separate
        ├── matryoshka_depth.py           ⚠️ Should be separate
        ├── streaming_depth.py            ⚠️ Should be separate
        └── cross_domain_adapter.py       ⚠️ Should be separate
```

## After Separation (Clean Architecture)

```
mythRL/
├── hololoom/                           # FRAMEWORK PACKAGE
│   ├── __init__.py                     # Public API
│   ├── config.py
│   ├── weaving_shuttle.py
│   ├── weaving_orchestrator.py
│   ├── loom/
│   ├── chrono/
│   ├── resonance/
│   ├── warp/
│   ├── convergence/
│   ├── fabric/
│   ├── reflection/
│   ├── embedding/
│   ├── motif/
│   ├── memory/
│   └── policy/
│
└── hololoom_narrative/                 # APP PACKAGE
    ├── __init__.py                     # App API
    ├── intelligence.py
    ├── cache.py
    ├── loop_engine.py
    ├── matryoshka_depth.py
    ├── streaming_depth.py
    ├── cross_domain_adapter.py
    ├── demos/
    └── tests/
```

## Dependency Flow

### Framework Independence
```
┌─────────────────────────────────────┐
│   HoloLoom Framework (hololoom)     │
│                                     │
│  ✅ Zero external app dependencies  │
│  ✅ Self-contained                  │
│  ✅ Provides public API             │
└─────────────────────────────────────┘
```

### App Dependencies
```
┌──────────────────────────────────────────┐
│  Narrative App (hololoom_narrative)      │
│                                          │
│  📦 depends on: hololoom>=0.1.0          │
│  ✅ Uses only public framework APIs      │
│  ✅ Zero framework internals access      │
└──────────────────────────────────────────┘
            │
            │ imports from
            ▼
┌──────────────────────────────────────────┐
│  HoloLoom Framework (hololoom)           │
│                                          │
│  Public exports:                         │
│  - WeavingShuttle                        │
│  - Config                                │
│  - Spacetime                             │
│  - Memory APIs                           │
└──────────────────────────────────────────┘
```

## Internal Module Dependencies

### Framework Core (No Changes)
```
config.py (leaf - no deps)
    ↓
Documentation/types.py
    ↓
embedding/spectral.py
    ↓
motif/base.py, memory/base.py, policy/unified.py
    ↓
loom/command.py, chrono/trigger.py, resonance/shed.py
    ↓
warp/space.py, convergence/engine.py
    ↓
fabric/spacetime.py, reflection/buffer.py
    ↓
weaving_shuttle.py, weaving_orchestrator.py
```

### Narrative App (Internal Chain)
```
intelligence.py (leaf - no HoloLoom deps!)
    ↓
matryoshka_depth.py (uses intelligence.py)
    ↓
streaming_depth.py (uses matryoshka_depth.py)
    ↓
cross_domain_adapter.py (uses matryoshka + streaming)
    ↓
loop_engine.py (uses cross_domain)

cache.py (independent - stdlib only)
```

### Integration Point (unified_api.py)
```
┌────────────────────────────────────┐
│  unified_api.py                    │
│                                    │
│  try:                              │
│      from hololoom_narrative ...   │
│  except ImportError:               │
│      # Optional feature            │
└────────────────────────────────────┘
```

## Import Path Changes

### Before (Confusing - App Mixed with Framework)
```python
# Framework imports
from HoloLoom import WeavingShuttle, Config

# App imports (LOOKS like framework!)
from HoloLoom.narrative_intelligence import NarrativeIntelligence
from HoloLoom.matryoshka_depth import MatryoshkaDepth

# User confusion: "Is narrative part of HoloLoom or not?"
```

### After (Clear Separation)
```python
# Framework imports
from hololoom import WeavingShuttle, Config

# App imports (clearly separate package)
from hololoom_narrative import NarrativeIntelligence
from hololoom_narrative.matryoshka_depth import MatryoshkaDepth

# User clarity: "Narrative is an app built ON hololoom"
```

## Package Relationships

```
┌────────────────────────────────────────────────────┐
│                  Python Ecosystem                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │  pip install hololoom                    │     │
│  │  (Framework only - no apps)              │     │
│  └──────────────────────────────────────────┘     │
│                      ▲                             │
│                      │                             │
│                      │ depends on                  │
│                      │                             │
│  ┌──────────────────────────────────────────┐     │
│  │  pip install hololoom-narrative          │     │
│  │  (App + framework dependency)            │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │  pip install hololoom-code (future)      │     │
│  │  (Another app on same framework)         │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Framework API Surface

### What Framework Exports (Public API)
```python
# hololoom/__init__.py
from hololoom.config import Config, ExecutionMode, PatternCard
from hololoom.weaving_shuttle import WeavingShuttle
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.Documentation.types import (
    Query, MemoryShard, Context, Features, Spacetime
)
from hololoom.fabric.spacetime import Spacetime, WeavingTrace
from hololoom.memory import create_memory_backend
from hololoom.embedding.spectral import MatryoshkaEmbeddings
from hololoom.policy.unified import create_policy
# ... etc
```

### What Apps Import (Use Public API)
```python
# hololoom_narrative/analyzer.py
from hololoom import (
    WeavingShuttle,      # ✅ Public API
    Config,              # ✅ Public API
    Spacetime,           # ✅ Public API
)

# NOT allowed:
from hololoom.internal.secret import SecretClass  # ❌ No internals
from hololoom.weaving_shuttle import _private     # ❌ No private methods
```

## Circular Dependency Check

### Framework → Apps (MUST be Zero)
```bash
# Framework should never import apps
grep -r "from hololoom_narrative" hololoom/
# Expected: No results ✅
```

### Apps → Framework (Expected)
```bash
# Apps can import framework
grep -r "from hololoom import" hololoom_narrative/
# Expected: Some imports ✅
```

### Apps → Apps (Should be Zero for now)
```bash
# Apps should not depend on each other
grep -r "from hololoom_code" hololoom_narrative/
grep -r "from hololoom_narrative" hololoom_code/
# Expected: No results ✅
```

## Testing Independence

### Framework Tests (No App Dependencies)
```bash
# Framework tests work without apps
cd hololoom/
pytest tests/
# Should pass without hololoom_narrative installed ✅
```

### App Tests (Framework as Fixture)
```bash
# App tests declare framework dependency
cd hololoom_narrative/
pytest tests/
# Requires: pip install hololoom ✅
```

## File Migration Map

```
Source (HoloLoom/)                  →  Destination

narrative_intelligence.py            →  hololoom_narrative/intelligence.py
narrative_cache.py                   →  hololoom_narrative/cache.py
narrative_loop_engine.py             →  hololoom_narrative/loop_engine.py
matryoshka_depth.py                  →  hololoom_narrative/matryoshka_depth.py
streaming_depth.py                   →  hololoom_narrative/streaming_depth.py
cross_domain_adapter.py              →  hololoom_narrative/cross_domain_adapter.py

demos/narrative_depth_dashboard.py   →  hololoom_narrative/demos/depth_dashboard.py
demos/narrative_depth_production.py  →  hololoom_narrative/demos/production_demo.py
tests/test_full_odyssey_depth.py     →  hololoom_narrative/tests/test_odyssey_depth.py
```

## Import Updates Required

### In Narrative Modules (6 files)
```bash
# Find and replace in all moved files
find hololoom_narrative -name "*.py" -type f

# Replace patterns:
from HoloLoom.narrative_intelligence  → from hololoom_narrative.intelligence
from HoloLoom.narrative_cache         → from hololoom_narrative.cache
from HoloLoom.narrative_loop_engine   → from hololoom_narrative.loop_engine
from HoloLoom.matryoshka_depth        → from hololoom_narrative.matryoshka_depth
from HoloLoom.streaming_depth         → from hololoom_narrative.streaming_depth
from HoloLoom.cross_domain_adapter    → from hololoom_narrative.cross_domain_adapter
```

### In Framework Files (1 file)
```python
# unified_api.py (line 62)
OLD: from HoloLoom.narrative_cache import CachedMatryoshkaDepth
NEW: from hololoom_narrative.cache import CachedMatryoshkaDepth
```

### In Dashboard/External (if any)
```bash
# Check for external imports
grep -r "from HoloLoom.narrative" dashboard/ promptly-vscode/ demos/
# Update each occurrence
```

## Validation Checklist

### Framework Independence ✅
- [ ] Framework imports don't mention narrative
- [ ] Framework tests pass without narrative installed
- [ ] Framework README doesn't require narrative

### App Cleanliness ✅
- [ ] Narrative imports only from `hololoom` (public API)
- [ ] Narrative declares `hololoom>=0.1.0` dependency
- [ ] Narrative tests work with framework installed

### No Circular Dependencies ✅
- [ ] Framework → Narrative: 0 imports
- [ ] Narrative → Framework: Only public API
- [ ] Narrative → Narrative: Internal only

### Documentation ✅
- [ ] Framework README mentions apps as separate packages
- [ ] Narrative README shows framework dependency
- [ ] APP_DEVELOPMENT_GUIDE.md exists with template

## Success Criteria

**The separation is successful when:**

1. ✅ Can `pip install hololoom` without narrative code
2. ✅ Can `pip install hololoom-narrative` (auto-installs hololoom)
3. ✅ Framework tests pass with 0 narrative imports
4. ✅ Narrative works using only public framework APIs
5. ✅ Clear docs showing framework vs. app boundaries
6. ✅ Template exists for building future apps

**The framework is validated when:**

1. ✅ Second app can be built following same pattern
2. ✅ No need to modify framework internals for apps
3. ✅ Public API is sufficient for domain logic

---

**Status:** Ready for execution. All dependencies mapped, zero circular deps found.
