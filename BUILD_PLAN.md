# BUILD_PLAN.md — Restructuring HoloLoom

**Date**: 2026-02-26
**Branch**: `claude/codebase-review-klbya`
**Depends on**: [MODULE_TAXONOMY.md](MODULE_TAXONOMY.md) for the target architecture

---

## What We Learned (Investigation Results)

Before writing code, we traced every reverse import — places where
HoloLoom core imports FROM directories we want to move. Here's what we found:

| Directory to move | Reverse imports from core | Severity |
|---|---|---|
| `web_dashboard/` | 1 (spinner — **now fixed**, lazy import) | ✅ Clear |
| `chatops/` | 1 (`server/agent_manager_integration.py`) | ⚠️ Coupled to server |
| `server/` | 1 (`studio_server.py` at repo root) | ⚠️ Coupled to chatops |
| `departments/` | **9+** (`loom/protocol.py`, `loom/base_loom.py`, all 5 Looms, `chaining/`, `workflows/`) | 🛑 Blocked |

**Key finding**: `departments/protocol.py` defines `Department`, `DepartmentProtocol`,
and `DepartmentRegistry` that the entire Loom subsystem inherits from. You cannot
move `departments/` to `apps/` without first extracting that protocol to a shared
location.

---

## PR Sequence (dependency-ordered)

Each PR is independently mergeable and leaves the repo working.

### PR 1: Extract department protocol to `protocols/`

**Why first**: Unblocks PR 4. The `DepartmentProtocol` is imported by 9+ core
modules. It belongs in `protocols/` (Layer 0), not in `departments/` (Layer 3).

**Work**:
1. Move protocol classes from `departments/protocol.py` → `protocols/department.py`
2. Re-export from `departments/protocol.py` with deprecation warning
3. Update the 9+ core imports to point to `protocols/department`
4. Tests still pass

**Size**: ~50 lines moved, ~15 import lines changed. Small, safe.

**Risk**: Low. Re-export shim means nothing breaks externally.

---

### PR 2: Move `web_dashboard/` → `apps/workflow_builder/`

**Why second**: Cleanest move. Only coupling was the spinner import (already fixed).
The dashboard is a true end-user app with its own HTML/JS/CSS frontend.

**Work**:
1. `git mv HoloLoom/web_dashboard/ apps/workflow_builder/`
2. Rewrite 44 self-imports: `from HoloLoom.web_dashboard.X` → `from workflow_builder.X`
3. Update ~80 imports FROM HoloLoom core to use absolute paths (they already are)
4. Add `apps/workflow_builder/__init__.py` if needed
5. Add re-export shim at old path: `HoloLoom/web_dashboard/__init__.py` that
   warns and re-imports from new location
6. Tests still pass

**Size**: ~44 sed replacements + git mv. Medium, mechanical.

**Risk**: Medium. The 44 self-imports are mechanical but need testing.

---

### PR 3: Move `server/` + `chatops/` → `apps/`

**Why together**: `server/agent_manager_integration.py` imports from `chatops/`.
They're coupled — move them as a unit.

**Work**:
1. `git mv HoloLoom/server/ apps/server/`
2. `git mv HoloLoom/chatops/ apps/chatops/`
3. Update self-imports in both packages
4. Update cross-imports between them
5. Fix `studio_server.py` at repo root (imports from server/)
6. Re-export shims at old paths

**Size**: Medium-large. Two packages, cross-coupled.

**Risk**: Medium. The cross-coupling is known and contained.

---

### PR 4: Move `departments/` → `apps/departments/`

**Why last**: Depends on PR 1 (protocol extraction). After PR 1, departments
only contains the *implementation* (registry, concrete departments) — no longer
the protocol that core depends on.

**Work**:
1. Verify PR 1 landed (protocol is in `protocols/`)
2. `git mv HoloLoom/departments/ apps/departments/`
3. Update self-imports
4. Update any remaining references
5. Re-export shim at old path

**Size**: Medium. Fewer self-imports than web_dashboard.

**Risk**: Low (after PR 1). The hard dependency is already extracted.

---

## Completed Waves

| Wave | Work | Status |
|---|---|---|
| **Wave 1**: Extract apps | PRs 1-4: protocols, web_dashboard, server+chatops, departments → `apps/` | Done |
| **Wave 2**: Consolidate micro-modules | `math/`→`warp/`, `neural/`→`policy/`, `clustering/`→`memory/`, `nested/`→`orchestrator/` | Done |
| **Wave 3**: Create `core/` | 13 core modules under `hololoom/core/`, `sys.meta_path` finder for transparent redirects | Done |
| **Wave 4**: `pyproject.toml` extras | `pip install hololoom[voice,vision,server,ml,rl,production]` | Done |
| **Wave 5**: Lowercase rename | `HoloLoom/` → `hololoom/`, 3,178 files updated | Done |

---

## All Done

- [x] Codebase review (`CODEBASE_REVIEW.md`)
- [x] Module taxonomy (`MODULE_TAXONOMY.md`)
- [x] Reverse import analysis (all 4 targets investigated)
- [x] Lazy import fix for spinner → web_dashboard coupling
- [x] PR 1: Extract department protocol
- [x] PR 2: Move web_dashboard
- [x] PR 3: Move server + chatops
- [x] PR 4: Move departments
- [x] Wave 2: Consolidate micro-modules (PRs 5-7, 9; skipped ml/ — not micro)
- [x] Wave 3: Create core/ directory (13 modules, meta_path finder)
- [x] Wave 4: pyproject.toml extras (voice, vision, server, ml, rl, production)
- [x] Wave 5: Lowercase rename (hololoom/, PEP 8)
