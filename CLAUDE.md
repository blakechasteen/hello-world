# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

HoloLoom is a unified memory system for AI agents. The core API is two imports:

```python
from hololoom import HoloLoom, Memory

loom = HoloLoom()
memory = await loom.experience("content")
memories = await loom.recall("query")
await loom.reflect(memories, feedback={...})
```

Package name: `hololoom` (lowercase, PEP 8). Class name: `HoloLoom` (PascalCase).

## Repository Structure

```
hololoom/                  # Main package (lowercase)
  core/                    # 13 essential modules
    protocols/             # Layer 0: type contracts, DepartmentProtocol
    memory/                # KG, vector store, spring dynamics
    embedding/             # Matryoshka multi-scale embeddings
    policy/                # Thompson Sampling, neural policy
    convergence/           # Probability -> discrete actions
    orchestrator/          # Weaving orchestrator + nested stages
    warp/                  # Tensor manifold, Hofstadter
    fabric/                # Spacetime output + provenance
    chrono/                # Temporal windows
    resonance/             # Feature extraction (DotPlasma)
    loom/                  # Pattern card selection
    recursive/             # Self-improving loops
    reflection/            # Episodic buffer, PPO
  lite/                    # HoloLoomLite simplified API
  config/                  # Configuration system
  unified_api.py           # HoloLoom class definition
  __init__.py              # Lazy loading + _CoreRedirectFinder
apps/                      # Application layer (extracted from core)
  server/                  # FastAPI REST APIs
  chatops/                 # ChatOps integration
  departments/             # Multi-department architecture
  elle/                    # AR guide system
  sous/                    # Kitchen control loop
  bosspig/                 # BossPig AI
  trough/                  # Production QA
hololoom-ui/               # React frontend
docs/                      # Documentation
```

## Import System

IMPORTANT: `hololoom/__init__.py` installs a `sys.meta_path` finder (`_CoreRedirectFinder`) that transparently redirects `hololoom.memory` -> `hololoom.core.memory` (and all 13 core modules). This means:

- `from hololoom.memory.graph import KG` works (redirects to `hololoom.core.memory.graph`)
- Internal code in `hololoom/core/` should use **relative imports** to avoid circular redirects
- The 13 redirected modules: protocols, memory, embedding, policy, convergence, orchestrator, warp, fabric, chrono, resonance, loom, recursive, reflection

## Development Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests (3 tiers)
pytest hololoom/tests/unit/ -v          # Fast (<5s)
pytest hololoom/tests/integration/ -v   # Medium (<30s)
pytest hololoom/tests/e2e/ -v           # Slow (<2min)
pytest hololoom/tests/ -v               # All

# Lint
ruff check hololoom/
black --check hololoom/
mypy hololoom/
```

## Optional Dependencies

```bash
pip install hololoom[server]      # fastapi, uvicorn, websockets
pip install hololoom[voice]       # whisper, librosa
pip install hololoom[vision]      # opencv, ultralytics
pip install hololoom[nlp]         # spacy, sentence-transformers
pip install hololoom[ml]          # scipy, scikit-learn
pip install hololoom[production]  # neo4j, qdrant-client, redis
pip install hololoom[all]         # everything
```

## Key Patterns

- **Protocol-based design**: All components define abstract protocols (PolicyEngine, KGStore, Retriever). Swap implementations without touching orchestrator code.
- **Graceful degradation**: Optional deps (spaCy, sentence-transformers, scipy) degrade with warnings, never crash. Always use try/except ImportError.
- **Async pipeline**: Orchestrator uses async/await. Concurrent feature extraction, background memory management, non-blocking tool execution.
- **Lazy loading**: `hololoom/__init__.py` uses module-level `__getattr__` to defer all imports. No circular import issues at import time.
- **Deprecation shims**: Moved modules (departments, server, chatops) have shims at old paths that re-export from new locations.

## Code Style

- Line length: 100 (Black + Ruff)
- Python 3.10+ (type hints, match statements OK)
- `__init__.py` files: F401 (unused imports) suppressed — re-exports are intentional
- Prefer absolute imports from `hololoom.*` in non-core code
- Prefer relative imports within `hololoom/core/` subpackages

## Philosophy

**"Reliable Systems: Safety First"** — Graceful degradation over performance. Auto-fallback (HYBRID -> INMEMORY). Explicit cleanup via async context managers. Archive instead of delete. Never lose user data.

## Testing Baseline

20 pre-existing collection errors in test suite (referencing non-existent modules like EduVerse, xterminator). These are known and not regressions.

## Detailed Documentation

See @docs/ for architecture diagrams, API references, and feature guides.
See @docs/ROADMAP.md for the v1.0 → v2.0 roadmap.
See @BUILD_PLAN.md for the completed restructuring history.
See @MODULE_TAXONOMY.md for the full module classification.
