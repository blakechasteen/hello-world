# Migration Guide

## v1.0 to v2.0 (Restructuring)

The v2.0 release reorganized the codebase for clarity. All changes are backward-compatible via deprecation shims.

### Package Rename

The package directory was renamed from `HoloLoom/` to `hololoom/` (PEP 8 lowercase).

```python
# Old (still works via shim, emits DeprecationWarning)
from HoloLoom import HoloLoom

# New
from hololoom import HoloLoom
```

### Core Modules Moved to `hololoom/core/`

13 core modules now live under `hololoom/core/`. A `sys.meta_path` finder transparently redirects old paths:

```python
# These are equivalent (redirect is automatic):
from hololoom.memory.graph import KG
from hololoom.core.memory.graph import KG  # canonical
```

The 13 redirected modules: `protocols`, `memory`, `embedding`, `policy`, `convergence`, `orchestrator`, `warp`, `fabric`, `chrono`, `resonance`, `loom`, `recursive`, `reflection`.

### App Layer Extracted to `apps/`

Application-layer packages were moved out of the core:

| Old Path | New Path |
|----------|----------|
| `hololoom/web_dashboard/` | `apps/workflow_builder/` |
| `hololoom/server/` | `apps/server/` |
| `hololoom/chatops/` | `apps/chatops/` |
| `hololoom/departments/` | `apps/departments/` |

Deprecation shims at old paths re-export from new locations.

### Department Protocol Extracted

`DepartmentProtocol` and `DepartmentRegistry` moved from `departments/protocol.py` to `protocols/department.py`:

```python
# Old (still works via shim)
from hololoom.departments.protocol import DepartmentProtocol

# New
from hololoom.core.protocols.department import DepartmentProtocol
```

### Micro-Module Consolidation

Small modules were merged into their conceptual homes:

| Old Module | Merged Into |
|-----------|-------------|
| `math/` (hofstadter.py) | `warp/` |
| `neural/` | `policy/` |
| `clustering/` | `memory/` |
| `nested/` | `orchestrator/` |

### Optional Dependencies

Install extras via `pyproject.toml`:

```bash
pip install hololoom[nlp,server,production]  # selective
pip install hololoom[all]                     # everything
```

Available extras: `nlp`, `voice`, `vision`, `server`, `ml`, `rl`, `production`, `viz`, `dev`, `all`.

### Migration Steps

1. Update imports: `HoloLoom` (uppercase) to `hololoom` (lowercase)
2. Update any direct references to moved modules (or rely on shims)
3. Update `pip install` commands to use extras syntax
4. Run tests to verify: `pytest hololoom/tests/ -v`

All shims emit `DeprecationWarning` — run with `python -W all` to find remaining old-style imports.

## v0.x to v1.0

v1.0 introduced the unified API:

```python
# Old (multiple imports)
from hololoom.weaving_orchestrator import WeavingOrchestrator
orchestrator = WeavingOrchestrator(config=Config.fast())
spacetime = await orchestrator.weave("query")

# New (two imports)
from hololoom import HoloLoom, Memory
loom = HoloLoom()
memory = await loom.experience("content")
memories = await loom.recall("query")
```

The `WeavingOrchestrator` still exists for advanced use cases. `HoloLoom` wraps it with a simpler interface.
