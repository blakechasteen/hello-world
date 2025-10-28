# Ruthless Cleanup Plan

## Executive Summary
**Current:** 236 files, 94K lines, sprawling structure
**Target:** 180 files, 80K lines, crystal clear boundaries
**ETA:** 3 hours of focused work

---

## Critical Issues (Fix Now)

### 1. Memory Backend Complexity ⚠️ HIGH PRIORITY
**Problem:** 29 files in `HoloLoom/memory/`, 8 different backend implementations

**Current Backends:**
```
✓ graph.py (NetworkX) - KEEP (default, always works)
✓ neo4j_graph.py - KEEP (production)
? hyperspace_backend.py - EVALUATE (research vs production?)
? mem0_adapter.py - MERGE into unified.py
? cache.py - MERGE into graph.py as caching layer
✗ migrate_to_neo4j.py - DELETE (one-time script)
✗ mcp_server_standalone.py - MOVE to tools/ or examples/
✗ reverse_query.py - DELETE (unused experiment?)
✗ query_enhancements.py - MERGE into protocol.py
```

**Action:**
```python
# Before: 8 backends, complex routing
backend = route_to_backend(config, query, fallback_chain, strategy, ...)

# After: 2 backends, simple fallback
backend = Neo4jBackend() if config.use_neo4j else NetworkXBackend()
```

**Files Eliminated:** 6-8 → 3-4 in memory/

---

### 2. Root Directory Clutter ⚠️ MEDIUM PRIORITY
**Problem:** 15+ files at `HoloLoom/` root, unclear entry points

**Current Root:**
```
HoloLoom/
├── weaving_shuttle.py ✓ MAIN ENTRY POINT
├── weaving_orchestrator.py ❓ REDUNDANT?
├── unified_api.py ✓ KEEP
├── config.py ✓ KEEP
├── bootstrap_system.py → MOVE to tools/
├── validate_pipeline.py → MOVE to tests/
├── visualize_bootstrap.py → MOVE to examples/
├── test_*.py (3 files) → MOVE to tests/
├── check_holoLoom.py → MOVE to tools/
├── synthesis_bridge.py → MOVE to synthesis/
├── matryoshka_interpreter.py → MOVE to embedding/
├── autospin.py → MOVE to spinningWheel/
└── conversational.py → MOVE to chatops/ OR DELETE
```

**Action:** Move 10 files out of root
**Target Root:** 5-6 core files only (shuttle, config, api, __init__)

---

### 3. Naming Inconsistency ⚠️ LOW PRIORITY (but annoying)
**Problem:** snake_case, camelCase, PascalCase mixed

**Fix Once:**
```
Rename:
  spinningWheel/ → spinning_wheel/
  darkTrace/ → dark_trace/
  Documentation/ → documentation/ (or docs/)
  Modules/ → modules/
  Utils/ → utils/
  Foundations/ → foundations/
```

**Skip if:** Too risky for working code (defer to v2.0)

---

### 4. Test Organization ⚠️ MEDIUM PRIORITY
**Problem:** Tests scattered everywhere

**Current:**
```
HoloLoom/
├── test_backends.py
├── test_smart_integration.py
├── test_unified_policy.py
├── tests/
│   ├── test_*.py (many)
├── examples/
│   └── ??? (are these tests or examples?)
```

**Action:**
```
tests/
├── unit/              # Fast, isolated
│   ├── test_memory.py
│   ├── test_policy.py
│   └── test_embeddings.py
├── integration/       # Multi-component
│   ├── test_weaving.py
│   └── test_backends.py
└── e2e/              # Full pipeline
    └── test_full_pipeline.py

examples/              # Runnable demos
└── simple_query_example.py
```

---

### 5. Dead Code Elimination ⚠️ HIGH PRIORITY

**Candidates for Deletion:**

```bash
# Check these files - if unused, DELETE:
git status | grep "^D"  # Already deleted but documented
  ✓ Orchestrator.py
  ✓ analytical_orchestrator.py
  ✓ smart_weaving_orchestrator.py

# Check import usage:
grep -r "from.*bootstrap_results" HoloLoom/
  → If empty, delete bootstrap_results/

grep -r "from.*darkTrace" HoloLoom/
  → If minimal usage, evaluate

grep -r "protocols\.py" HoloLoom/
  → If only importing from protocols/, delete protocols.py (duplicate)
```

**Heuristic:** If not imported in last 30 days → DELETE

---

## Architectural Principles (The "Why")

### 1. **One Way to Do It**
```python
# Bad: 8 ways to create memory backend
backend = NetworkXBackend()
backend = create_memory_backend(config)
backend = MemoryFactory.create(...)
backend = route_to_backend(...)
# ... etc

# Good: 1 way
backend = create_memory_backend(config)  # handles all cases internally
```

### 2. **Flat is Better Than Nested**
```python
# Bad:
from HoloLoom.memory.routing.strategy.adaptive import AdaptiveRouter

# Good:
from HoloLoom.memory import create_backend
```

### 3. **Entry Points are Sacred**
```
User-facing:
  - weaving_shuttle.py (main orchestrator)
  - unified_api.py (programmatic API)
  - config.py (configuration)

Everything else: implementation detail
```

### 4. **Tests Mirror Structure**
```
HoloLoom/memory/graph.py
tests/unit/test_memory_graph.py

HoloLoom/weaving_shuttle.py
tests/integration/test_weaving_shuttle.py
```

---

## Implementation Order

### Phase 1: Safe Cleanup (30 min)
```bash
# Move files (no code changes)
mkdir -p tools examples/{simple,advanced}
mv HoloLoom/bootstrap_system.py tools/
mv HoloLoom/visualize_bootstrap.py examples/
mv HoloLoom/test_*.py tests/
mv HoloLoom/check_holoLoom.py tools/

# Delete obvious dead code
rm -rf HoloLoom/bootstrap_results  # if unused
git commit -m "chore: organize project structure"
```

### Phase 2: Memory Simplification (1 hour)
```python
# File: HoloLoom/memory/__init__.py
# BEFORE: Complex factory with routing
def create_memory_backend(config):
    if config.backend == "NETWORKX": ...
    elif config.backend == "NEO4J": ...
    elif config.backend == "NEO4J_QDRANT": ...
    elif config.backend == "TRIPLE": ...
    elif config.backend == "HYPERSPACE": ...
    elif config.backend == "MEM0": ...
    # ... complex fallback chains

# AFTER: Simple choice
def create_memory_backend(config):
    """Create memory backend with automatic fallback."""
    if config.use_persistent_storage:
        try:
            return Neo4jBackend(config)  # Production
        except ConnectionError:
            logger.warning("Neo4j unavailable, falling back to NetworkX")

    return NetworkXBackend()  # Default, always works
```

**Files to merge/delete:**
- `mem0_adapter.py` → fold into `unified.py`
- `hyperspace_backend.py` → move to `research/` or delete if experimental
- `migrate_to_neo4j.py` → delete (one-time script)
- `reverse_query.py` → delete or move to examples
- `query_enhancements.py` → merge into `protocol.py`

### Phase 3: Test Reorganization (30 min)
```bash
mkdir -p tests/{unit,integration,e2e}

# Move unit tests
mv tests/test_memory_graph.py tests/unit/
mv tests/test_policy.py tests/unit/

# Move integration tests
mv HoloLoom/test_backends.py tests/integration/
mv HoloLoom/test_smart_integration.py tests/integration/

# Create e2e test
cat > tests/e2e/test_full_pipeline.py << 'EOF'
"""End-to-end test of complete weaving pipeline."""
async def test_full_weaving_cycle():
    shuttle = WeavingShuttle(config=Config.fast())
    query = Query(content="test")
    result = await shuttle.weave(query)
    assert result.response is not None
EOF
```

### Phase 4: Documentation Update (30 min)
```markdown
# Update README.md, CLAUDE.md to reflect new structure

## Quick Start (3 commands)
```bash
# 1. Install
pip install -e .

# 2. Run
python -m HoloLoom.weaving_shuttle

# 3. Test
pytest tests/
```

## Project Structure
```
HoloLoom/
├── weaving_shuttle.py   # Main entry point
├── config.py            # Configuration
├── unified_api.py       # Programmatic API
├── memory/              # Storage backends (2 implementations)
├── policy/              # Decision making
├── semantic_calculus/   # 244D semantic space
└── ...

tests/
├── unit/                # Fast, isolated
├── integration/         # Multi-component
└── e2e/                # Full pipeline

tools/                   # Developer utilities
examples/                # Runnable demos
```
```

### Phase 5: Validate (30 min)
```bash
# Run all tests
pytest tests/ -v

# Check imports
python -m HoloLoom.weaving_shuttle --help

# Verify examples work
python examples/simple_query_example.py
```

---

## Metrics

### Before
```
Files: 236
Lines: 94,241
Directories: 35
Root files: 15
Memory backends: 8
Test locations: 4
```

### After (Target)
```
Files: ~180 (-24%)
Lines: ~80,000 (-15%)
Directories: ~25
Root files: 5 (-67%)
Memory backends: 2 (-75%)
Test locations: 1
```

### Wins
- ✓ Faster CI/CD (less to test)
- ✓ Easier onboarding (clear structure)
- ✓ Fewer bugs (less code)
- ✓ Faster development (know where to look)

---

## Decision Framework

**When considering a file for deletion, ask:**

1. **Is it imported?** `grep -r "import.*filename" HoloLoom/`
   - If NO → DELETE

2. **Is it tested?** Check `tests/`
   - If NO → DELETE or add test

3. **Is it documented?** Check READMEs
   - If NO → DELETE or document

4. **Does it have a clear owner?**
   - If NO → DELETE or assign

5. **Could it be merged into another file?**
   - If YES → MERGE

**Default: DELETE unless clear value**

---

## Risk Mitigation

### Before Any Changes:
```bash
# Create cleanup branch
git checkout -b cleanup/ruthless-refactor

# Ensure tests pass
pytest tests/ -v

# Create backup
git tag backup-before-cleanup
```

### After Each Phase:
```bash
# Verify tests still pass
pytest tests/ -v

# Commit incrementally
git commit -m "phase X: ..."
```

### If Something Breaks:
```bash
# Rollback
git reset --hard backup-before-cleanup

# Or revert specific commit
git revert <commit-hash>
```

---

## ChatOps Roadmap Addition

```markdown
# HoloLoom/chatops/ROADMAP.md

## Phase 3: Semantic Learning Integration (Q1 2026)

### 3.1 Semantic State Tracking
- Track 244D semantic state of conversations
- Monitor Clarity, Warmth, Logic in real-time
- Alert when dimensions drop below thresholds

### 3.2 Multi-Task Learning
- Learn from conversation outcomes
- Tool effect signatures for ChatOps commands
- Goal-directed conversation steering

### 3.3 Interpretable Decisions
- "Why did the bot suggest this command?"
- "Increased Clarity from 0.7 → 0.85 by formatting code"
- Dashboard showing semantic trajectories

### 3.4 Demo Integration
- Add ChatOps panel to mega showcase
- Live semantic monitoring during conversations
- A/B test semantic vs vanilla bot
```

---

## Next Steps (Priority Order)

1. **NOW:** Memory backend simplification (biggest ROI)
   ```bash
   python test_memory_backend_simplification.py  # already exists!
   ```

2. **NEXT:** Root directory cleanup (easy wins)
3. **THEN:** Test reorganization (quality improvement)
4. **LATER:** Naming consistency (polish)

---

## Success Criteria

**Code Review Checklist:**
- [ ] Can new engineer find main entry point in <10 seconds?
- [ ] Is there exactly 1 way to create each component?
- [ ] Are tests organized by type (unit/integration/e2e)?
- [ ] Can you explain every file in root directory?
- [ ] Is `grep -r "import"` output clean and predictable?

**User Experience:**
- [ ] `pip install -e . && python -m HoloLoom.weaving_shuttle` works
- [ ] Examples run without modification
- [ ] Error messages point to correct files

**Maintainability:**
- [ ] No file >1000 lines
- [ ] No directory >15 files
- [ ] Test coverage >80%
- [ ] CI runs in <5 minutes

---

*"Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away."*
— Antoine de Saint-Exupéry

*"Code is like humor. When you have to explain it, it's bad."*
— Cory House

*"Any fool can write code that a computer can understand. Good programmers write code that humans can understand."*
— Martin Fowler