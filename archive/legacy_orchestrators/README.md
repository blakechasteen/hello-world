# Legacy Orchestrator Variants

These are backup copies of the weaving orchestrator variant files that live at
`hololoom/weaving_orchestrator_*.py`. They are legacy development artifacts
but are still actively imported by multiple modules.

## Status

- `weaving_orchestrator.py` (base) — 87 files import it. **Cannot be moved yet.**
- `weaving_orchestrator_bandit.py` — imported by `rag/`, `tests/integration/`
- `weaving_orchestrator_recursive.py` — imported by `agentic/`, `integrations/`
- `weaving_orchestrator_llm.py` — imported by `rag/`, `lite/`, `visualization/`
- `weaving_orchestrator_refactored.py` — imported by `tests/unit/`

## Future Work

These should be consolidated into `hololoom/core/orchestrator/` with deprecation
shims at the old import paths. This is a medium-sized refactor that requires
updating all importers.
