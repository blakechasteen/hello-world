# HoloLoom Codebase Review — April 2026

**Date**: 2026-04-02
**Branch**: `claude/review-codebase-7KPbi`
**Scope**: Full repository health check

---

## Executive Summary

HoloLoom is a ~1.47M-line Python codebase (3,567 files) implementing a unified memory system for AI agents. The core architecture (13 modules under `hololoom/core/`) is well-designed with clean lazy-loading, protocol-based abstractions, and a solid `_CoreRedirectFinder` meta-path system. However, the repository suffers from significant organizational debt: root-level clutter, scattered test files, markdown sprawl, and legacy code that should be archived.

**Overall**: Strong architecture, weak housekeeping.

---

## What's Working Well

### 1. Core Architecture (hololoom/core/)
The 13-module core is cleanly structured with no circular import issues detected:
- `protocols/` — zero-dep contracts (Layer 0)
- `memory/`, `embedding/`, `policy/`, `convergence/`, `orchestrator/`, `warp/`, `fabric/`, `chrono/`, `resonance/`, `loom/`, `recursive/`, `reflection/`

Plus newer additions: `bus/`, `deep_thinking/`, `ritual/`, `runtime/` (not in original 13, may need taxonomy update).

### 2. Import System
`hololoom/__init__.py` implements an elegant lazy-loading + meta-path redirect system:
- `_CoreRedirectFinder` transparently maps `hololoom.memory` → `hololoom.core.memory`
- Module-level `__getattr__` defers all imports to first use
- No circular imports at import time
- `import hololoom` succeeds even without optional deps installed

### 3. Protocol-Based Design
Components define abstract protocols (PolicyEngine, KGStore, Retriever). Implementations can be swapped without touching orchestrator code. Graceful degradation via try/except ImportError throughout.

### 4. pyproject.toml
Well-configured with proper extras (`[nlp]`, `[voice]`, `[vision]`, `[server]`, `[ml]`, `[rl]`, `[production]`, `[all]`), tool configs for black/ruff/mypy/pytest, and correct package discovery.

### 5. Wave 5 Cleanup Complete
The `holoLoom/` (PascalCase) duplicate directory has been successfully removed. Package is correctly lowercase `hololoom/`.

---

## Issues Found

### Critical: No Dev Environment

Dependencies are not installed. `from hololoom import HoloLoom` fails with `ModuleNotFoundError: networkx`. Tests cannot be collected or run. This blocks all verification.

**Fix**: CI should install `pip install -e ".[dev]"` and run the test suite.

---

### High: Root-Level File Sprawl

**69 markdown files** at repository root. Most are ephemeral reports, week-by-week analysis notes, and planning artifacts:

| Category | Count | Examples |
|----------|-------|---------|
| Ephemeral reports | ~25 | `W3_*.md`, `W4_*.md`, `W8_*.md`, `W10_*.md`, `*_REPORT*.md` |
| Integration/feature docs | ~20 | `CRM_*.md`, `VERCEL_*.md`, `VOICE_*.md`, `PROMETHEUS_*.md` |
| Strategy/architecture | ~10 | `HOLOLOOM_STRATEGY_2026.md`, `GOVERNANCE_*.md` |
| Essential (keep at root) | ~5 | `README.md`, `CLAUDE.md`, `LICENSE`, `BUILD_PLAN.md`, `MODULE_TAXONOMY.md` |

**36 Python test/demo/debug files** scattered at root:
- `test_v2_*.py` (8 files), `test_*.py` (14 more), `smoke_test_*.py` (3), `demo_*.py` (3)
- `benchmark_models.py`, `fire_test.py`, `live_test_20.py`, `crm_demo*.py` (2)
- `debug_*.py` (2), `transcribe_audio.py`, `run_tests_by_file.py`

**Recommendation**:
1. Move ephemeral markdown to `docs/archive/` or `archive/reports/`
2. Move root test files to `tests/standalone/` or `tests/manual/`
3. Keep only `README.md`, `CLAUDE.md`, `LICENSE`, `BUILD_PLAN.md`, `MODULE_TAXONOMY.md`, `CHANGELOG.md` at root

---

### High: Markdown Inside Python Package

**437 markdown files** inside `hololoom/`. 23 are at the package root level. These get included in `pip install` distributions (per `package-data` config in pyproject.toml: `"*.md"`).

**Recommendation**: Move to `docs/` or exclude from package data.

---

### High: hololoom/ Directory Has 122 Items

The `hololoom/` package directory contains 78+ subdirectories and 40+ files. This is far too flat. The MODULE_TAXONOMY.md planned for "13 core + flat optional peers" but the directory has grown well beyond that with:

- 13 core modules (under `core/`)
- ~33 optional modules (as planned)
- ~20 additional directories not in the taxonomy (`domain_harness/`, `infrastructure/`, `motif/`, `pipeline/`, `tui/`, `weaverlet/`, `train_agent/`, etc.)
- 5 legacy weaving orchestrator files at package root
- 23 markdown files
- Various standalone Python files (`cli.py`, `cli_client.py`, `dashboard_server.py`, `studio_server.py`, `terminal_ui.py`, etc.)

---

### Medium: Legacy Weaving Orchestrators

Five orchestrator variant files at `hololoom/` root totaling ~4,300 lines:

| File | Lines | Status |
|------|-------|--------|
| `weaving_orchestrator.py` | 2,696 | Canonical — 9+ imports reference it |
| `weaving_orchestrator_bandit.py` | 606 | Experimental variant |
| `weaving_orchestrator_recursive.py` | 472 | Experimental variant |
| `weaving_orchestrator_refactored.py` | 304 | "Elegance Pass" variant |
| `weaving_orchestrator_llm.py` | 270 | LLM variant |

The production orchestrator is modularized in `hololoom/core/orchestrator/` (166+ imports). The root-level files appear to be development artifacts.

**Recommendation**: Archive `_bandit`, `_recursive`, `_refactored`, `_llm` variants. Verify if the base `weaving_orchestrator.py` is still needed or if `core/orchestrator/` has fully replaced it.

---

### Medium: MODULE_TAXONOMY.md Drift

Several modules listed as "unclear" have been resolved:
- `nested/`, `neural/`, `math/` — **don't exist** (consolidated in Waves 2-3)
- `shuttle/` — **active** (11 files, MCTS + Thompson Sampling)
- `weaving/` — **minimal** (3 files, likely deprecated)
- `synthesis/` — **active** (data synthesizer)
- `expansions/` — **active** (lazy-loaded research bundles)
- `input/` — **active** (9 files, audio/image/text processors)

New modules not in taxonomy: `domain_harness/`, `infrastructure/`, `motif/`, `pipeline/`, `tui/`, `weaverlet/`, `train_agent/`

Core has grown beyond the original 13: `bus/`, `deep_thinking/`, `ritual/`, `runtime/` are now under `core/`.

---

### Medium: Docker Compose Fragmentation

Three compose files with no clear guidance:
- `hololoom/docker-compose.yml` — full stack (neo4j, qdrant, postgres, redis)
- `docker-compose.lite.yml` — lightweight
- `docker-compose.quickstart.yml` — quickstart

All contain hardcoded credentials (e.g., `NEO4J_AUTH=neo4j/hololoom123`).

**Recommendation**: Use `.env.example` templates. Document which compose file to use when.

---

### Medium: unified_api.py Still Referenced

`hololoom/unified_api.py` is marked DEPRECATED but is still imported as a fallback in `__init__.py`. Should be fully removed once `hololoom/hololoom.py` (the HoloLoom class) is confirmed as the sole entry point.

---

### Low: conftest.py Coverage

Root `conftest.py` only handles 2 of the 36 root-level test files as "standalone scripts." The rest would hit import errors if pytest tried to collect them.

---

### Low: pyproject.toml URLs Point to "mythRL"

```toml
[project.urls]
"Bug Tracker" = "https://github.com/blakechasteen/mythRL/issues"
"Source Code" = "https://github.com/blakechasteen/mythRL"
```

These reference the old `mythRL` repo name. Should be updated to `hello-world` or whatever the canonical repo name is.

---

## Roadmap Progress Assessment

Per `docs/ROADMAP.md`, beta.1 status:

| Task | Status |
|------|--------|
| Merge HoloLoom classes | Done |
| Lazy WeavingOrchestrator shares AwarenessGraph | Done |
| Generic ingest via SpinnerProtocol | Done |
| Fix stale test collection errors | Done |
| Docker Compose PascalCase fixes | Done |
| Docker Compose tested end-to-end | **Not done** |
| Unit + integration suites green | **Cannot verify** (deps not installed) |

**v1.0.0 blockers**: CI not green (no CI running), API reference not generated, guides not verified against code, classifier still "4 - Beta".

---

## Recommended Actions (Priority Order)

### P0 — Unblock Development
1. **Set up CI**: Install deps, run `pytest hololoom/tests/ -v`, ensure collection has 0 errors
2. **Install dev environment**: `pip install -e ".[dev]"` and verify `from hololoom import HoloLoom` works

### P1 — Reduce Clutter
3. **Archive root markdown**: Move ~60 ephemeral `.md` files to `docs/archive/`
4. **Organize root test files**: Move 36 `.py` test/demo files to `tests/manual/` or `demos/`
5. **Archive legacy orchestrators**: Move 4 variant files to `archive/`

### P2 — Update Documentation
6. **Update MODULE_TAXONOMY.md**: Remove resolved "unclear" modules, add new ones
7. **Fix pyproject.toml URLs**: mythRL → current repo name
8. **Remove or fully deprecate unified_api.py**

### P3 — Harden
9. **Docker Compose**: Consolidate, use `.env.example`, document usage
10. **Package data**: Exclude `.md` files from pip distribution
11. **Update conftest.py**: Handle all root test files or relocate them

---

## Metrics

| Metric | Value |
|--------|-------|
| Total Python files | 3,567 |
| Total Python LOC | ~1,470,000 |
| Core modules (hololoom/core/) | 17 directories |
| Optional modules | ~55 directories |
| Root .md files | 69 |
| Root .py test/demo files | 36 |
| Package-internal .md files | 437 |
| hololoom/ directory items | 122 |
