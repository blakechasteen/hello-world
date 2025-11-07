# mythRL Repository Cleanup Plan
**Generated:** November 2, 2025
**Analyst:** Claude Code Agent
**Scope:** Complete repository review and cleanup initiative

---

## EXECUTIVE SUMMARY

**Repository Health:** 7.2/10 (Good core architecture, needs organizational cleanup)

**Key Statistics:**
- **Total Python Files:** 489 (HoloLoom) + 352 (archive) = 841
- **Total Markdown Files:** 1,139
- **Test Files:** 113 (29,756 lines, ~800 tests)
- **Test Coverage:** ~40% (target: 60%)
- **Modified Files Pending:** 14 files, +1,945 insertions
- **Untracked Files:** 70+ files needing decisions
- **Duplicate Code:** 8,000-10,000 lines (~5-6% of codebase)
- **Documentation Issues:** 62 critical/high-priority issues

**Top 5 Critical Issues:**
1. 🔴 **Duplicate SpinningWheel Directories** - spinning_wheel/ vs spinningWheel/ causing confusion
2. 🔴 **Case Inconsistency** - HoloLoom vs HoloLoom (170+ files affected)
3. 🔴 **Test Coverage Gaps** - Critical modules lack tests (agentic, server, recursive)
4. 🔴 **Duplicate Memory Stores** - 12 implementations with 70-85% overlap
5. 🔴 **31 Root-Level Docs** - Too many files, hard to navigate

---

## PRIORITY MATRIX

### P0 - CRITICAL (Do Immediately - 8-12 hours)

| Issue | Impact | Effort | Files | Risk |
|-------|--------|--------|-------|------|
| Delete `nul` file | HIGH | 5 min | 1 | None |
| Consolidate spinningWheel | CRITICAL | 8-12 hrs | 23 | Medium |
| Fix case inconsistency | HIGH | 2-3 hrs | 170+ | Low |
| Commit untracked source | HIGH | 30 min | 48 | None |
| Archive session docs | MEDIUM | 30 min | 15 | None |

**Total P0 Effort:** 11.5-16 hours

---

### P1 - HIGH PRIORITY (Next Sprint - 30-45 hours)

| Issue | Impact | Effort | Lines Saved | Risk |
|-------|--------|--------|-------------|------|
| Consolidate memory stores | HIGH | 10-16 hrs | 1,800-2,200 | Medium |
| Add agentic/server tests | CRITICAL | 10-12 hrs | N/A | Low |
| Consolidate type definitions | HIGH | 6-10 hrs | 400-600 | Medium |
| Fix CLAUDE.md paths | HIGH | 1-2 hrs | N/A | None |
| Consolidate test directories | MEDIUM | 2-3 hrs | N/A | Low |

**Total P1 Effort:** 29-43 hours

---

### P2 - MEDIUM PRIORITY (Month 1 - 35-50 hours)

| Issue | Impact | Effort | Lines Saved |
|-------|--------|--------|-------------|
| Consolidate dashboards | MEDIUM | 8-12 hrs | 1,200-1,500 |
| Add recursive learning tests | HIGH | 8-10 hrs | N/A |
| Consolidate MCP servers | MEDIUM | 6-8 hrs | 500-700 |
| Consolidate test fixtures | MEDIUM | 4-6 hrs | 600-800 |
| Create missing READMEs | MEDIUM | 4-6 hrs | N/A |
| Fix broken doc links | MEDIUM | 2 hrs | N/A |
| Consolidate UI directories | MEDIUM | 2-3 hrs | N/A |

**Total P2 Effort:** 34-47 hours

---

### P3 - LOW PRIORITY (Month 2+ - 20-30 hours)

| Issue | Impact | Effort | Lines Saved |
|-------|--------|--------|-------------|
| Split large files | LOW | 4-6 hrs | N/A |
| Consolidate protocols | LOW | 4-5 hrs | 200-300 |
| Consolidate utilities | LOW | 3-4 hrs | 150-200 |
| Clean Python cache | LOW | 5 min | N/A |
| Organize demos | MEDIUM | 4-6 hrs | N/A |
| Review archive (352 files) | LOW | 3-4 hrs | N/A |
| Add property-based tests | MEDIUM | 6-8 hrs | N/A |

**Total P3 Effort:** 24-33 hours

---

## DETAILED ACTION PLAN

### Phase 1: Immediate Cleanup (P0 - Week 1)

#### Action 1.1: Delete Stray Files (5 minutes)
```bash
# Delete the nul file
git rm nul

# Clean Python cache
find HoloLoom -type d -name __pycache__ -exec rm -rf {} +
find HoloLoom -name "*.pyc" -delete

# Remove test databases
rm -f test_db.db
rm -f data/test_*.db data/demo_*.db
```

**Risk:** None
**Testing:** Verify git status clean

---

#### Action 1.2: Consolidate SpinningWheel Directories (8-12 hours)

**Current State:**
- `HoloLoom/spinning_wheel/` (14 files) - Original, modality-specific
- `HoloLoom/spinningWheel/` (9 files) - Newer, communication-specific

**Decision:** Merge to `HoloLoom/spinningWheel/` (matches weaving metaphor)

**Steps:**
1. Create unified structure:
   ```bash
   mkdir -p HoloLoom/spinningWheel/modalities
   mkdir -p HoloLoom/spinningWheel/comms
   ```

2. Move files:
   ```bash
   # Move modality spinners
   mv HoloLoom/spinning_wheel/audio.py HoloLoom/spinningWheel/modalities/
   mv HoloLoom/spinning_wheel/youtube.py HoloLoom/spinningWheel/modalities/
   mv HoloLoom/spinning_wheel/text.py HoloLoom/spinningWheel/modalities/
   mv HoloLoom/spinning_wheel/code.py HoloLoom/spinningWheel/modalities/
   mv HoloLoom/spinning_wheel/website.py HoloLoom/spinningWheel/modalities/

   # Communication spinners already in spinningWheel/comms
   # git_spinner.py, chat_history.py, matrix_spinner.py
   ```

3. Merge protocol definitions:
   ```bash
   # Combine spinning_wheel/base.py + spinningWheel/protocol.py
   # Keep best of both in spinningWheel/protocol.py
   ```

4. Update imports (automated):
   ```bash
   # Find all imports
   grep -r "from HoloLoom.spinning_wheel" HoloLoom/ --include="*.py"

   # Replace with new paths
   find HoloLoom -name "*.py" -type f -exec sed -i \
     's/from HoloLoom\.spinning_wheel/from HoloLoom.spinningWheel.modalities/g' {} \;
   ```

5. Delete old directory:
   ```bash
   git rm -r HoloLoom/spinning_wheel/
   ```

6. Run tests:
   ```bash
   pytest HoloLoom/tests/unit/test_*spinner*.py -v
   pytest HoloLoom/tests/integration/test_multimodal*.py -v
   ```

**Risk:** Medium (breaking changes to imports)
**Mitigation:** Comprehensive test suite, backward compatibility shim if needed
**Lines Saved:** 2,000-2,500

---

#### Action 1.3: Fix Case Inconsistency (2-3 hours)

**Problem:** 141 instances of `HoloLoom`, should be `HoloLoom`

**Steps:**
1. Backup first:
   ```bash
   git add -A
   git commit -m "chore: Backup before case fix"
   ```

2. Fix in documentation:
   ```bash
   find . -name "*.md" -type f -exec sed -i 's/\bholoLoom\b/HoloLoom/g' {} \;
   ```

3. Fix in Python files (if any):
   ```bash
   find HoloLoom -name "*.py" -type f -exec sed -i 's/\bholoLoom\b/HoloLoom/g' {} \;
   ```

4. Manual review of changes:
   ```bash
   git diff
   ```

5. Run full test suite:
   ```bash
   pytest HoloLoom/tests/ -v
   ```

**Risk:** Low (mostly documentation)
**Files Affected:** 170+
**Testing:** Full test suite must pass

---

#### Action 1.4: Commit Untracked Source Files (30 minutes)

**Files to Commit (48 files):**

**New Features:**
```bash
git add HoloLoom/spinningWheel/chat_history.py
git add HoloLoom/spinningWheel/git_spinner.py
git add HoloLoom/spinningWheel/matrix_spinner.py
git add HoloLoom/spinningWheel/protocol.py
git add HoloLoom/spinningWheel/utils.py
git add HoloLoom/spinningWheel/importance.py
git add HoloLoom/tools/handler_factory.py
git add HoloLoom/utils/timing.py
git add HoloLoom/documentation/typed_dicts.py
```

**Test Files:**
```bash
git add HoloLoom/tests/conftest.py
git add HoloLoom/tests/e2e/test_*.py
git add HoloLoom/tests/integration/test_warp_guardrails.py
git add HoloLoom/tests/unit/test_*.py
git add demos/demo_chat_history*.py
git add demos/git_spinner_example.py
```

**Documentation:**
```bash
git add HoloLoom/memory/README.md
git add HoloLoom/spinningWheel/README.md
git add HoloLoom/spinningWheel/*.md
git add HoloLoom/tests/README.md
```

**Commit:**
```bash
git commit -m "feat: Add spinningWheel chat history, git spinners, and comprehensive tests

New Features:
- Chat history spinner (Claude Desktop, browser histories)
- Git repository spinner with importance scoring
- Matrix chat spinner for real-time collaboration
- Spinner protocol standardization
- Handler factory for tool execution

Tests:
- 15+ new unit tests (spinner protocols, embeddings, config)
- 9 new E2E tests (performance, error handling, cache)
- Integration tests for warp guardrails
- conftest.py with comprehensive fixtures

Documentation:
- Memory system README
- SpinningWheel pipeline guide
- Test organization README

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Risk:** None
**Testing:** Verify tests pass

---

#### Action 1.5: Archive Session Documentation (30 minutes)

**Create Archive Structure:**
```bash
mkdir -p archive/session_docs_nov2025
```

**Move Session Docs:**
```bash
# Moonshot docs
mv MOONSHOT_*.md archive/session_docs_nov2025/
mv FINAL_MOONSHOT_*.md archive/session_docs_nov2025/

# Session summaries
mv SESSION_3_SUMMARY.md archive/session_docs_nov2025/
mv PERSISTENT_CHAT_STATUS.md archive/session_docs_nov2025/

# E2E docs
mv E2E_MOONSHOT_COMPLETE.md archive/session_docs_nov2025/
mv E2E_TEST_MONITORING_GUIDE.md archive/session_docs_nov2025/
mv E2E_TEST_PROGRESS.md archive/session_docs_nov2025/

# Phase docs
mv PHASE_3_COMPLETE.md archive/session_docs_nov2025/
mv POLISH_AND_EXPANSION_SUMMARY.md archive/session_docs_nov2025/

# Other completion docs
mv TEST_RESULTS_SUMMARY.md archive/session_docs_nov2025/
mv EXCEPTION_HANDLING_GUIDE.md archive/session_docs_nov2025/
mv CHAT_HISTORY_INTEGRATION.md archive/session_docs_nov2025/
```

**Commit:**
```bash
git add archive/session_docs_nov2025/
git commit -m "chore: Archive session documentation from Oct-Nov 2025"
```

**Risk:** None
**Files Moved:** 15

---

### Phase 2: High-Priority Cleanup (P1 - Week 2-3)

#### Action 2.1: Consolidate Memory Stores (10-16 hours)

**Problem:** 12 store implementations, 70-85% duplicate code

**Files to Consolidate:**
- Keep: `neo4j_store.py`, `mem0_store.py`, `hybrid_store.py`
- Archive: `neo4j_memory_store.py`, `mem0_memory_store.py`, `hybrid_neo4j_qdrant.py`

**Steps:**
1. Extract common patterns to `base_store.py`:
   - Connection management
   - Error handling
   - Memory serialization
   - Result fusion algorithms

2. Refactor kept stores to use base:
   ```python
   # memory/stores/base_store.py
   class BaseMemoryStore:
       def __init__(self, uri, credentials):
           self._connect(uri, credentials)

       def _connect(self, uri, credentials):
           # Common connection logic

       def _serialize_memory(self, memory):
           # Common serialization

       def _fuse_results(self, results):
           # Common fusion

   # memory/stores/neo4j_store.py
   class Neo4jStore(BaseMemoryStore):
       # Only Neo4j-specific logic
   ```

3. Update backend_factory to use new stores

4. Archive old implementations:
   ```bash
   mv HoloLoom/memory/stores/neo4j_memory_store.py archive/deprecated_modules/
   mv HoloLoom/memory/stores/mem0_memory_store.py archive/deprecated_modules/
   mv HoloLoom/memory/stores/hybrid_neo4j_qdrant.py archive/deprecated_modules/
   ```

5. Update all imports

6. Run integration tests:
   ```bash
   pytest HoloLoom/tests/integration/test_backends.py -v
   pytest HoloLoom/tests/integration/test_hybrid_memory.py -v
   ```

**Risk:** Medium (affects production backends)
**Testing:** Comprehensive integration tests required
**Lines Saved:** 1,800-2,200

---

#### Action 2.2: Add Critical Missing Tests (10-12 hours)

**Priority 0 Tests (Must Have):**

**1. Agentic Core Tests** (4 hours)
```python
# HoloLoom/tests/unit/test_agentic_core.py (300+ lines)
```
- Test all 4 reasoning modes (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)
- Test multi-query reasoning
- Test verification logic
- Test goal decomposition

**2. Server API Tests** (4 hours)
```python
# HoloLoom/tests/integration/test_agentic_api.py (400+ lines)
```
- Test /query endpoint with all modes
- Test /health, /stats, /audit-trail endpoints
- Test error handling (400, 500 responses)
- Test request validation
- Test WebSocket connections

**3. Convergence Engine Tests** (2 hours)
```python
# HoloLoom/tests/unit/test_convergence_engine.py (250+ lines)
```
- Test all collapse strategies
- Test Thompson Sampling updates
- Test probability distribution collapse

**4. Resonance Shed Tests** (2 hours)
```python
# HoloLoom/tests/unit/test_resonance_shed.py (300+ lines)
```
- Test feature thread lifting
- Test multi-modal fusion
- Test DotPlasma creation

**Implementation Plan:**
1. Create test files with fixtures
2. Write tests for happy paths
3. Add error case tests
4. Add edge case tests
5. Verify coverage >70% for each module

**Risk:** Low (adding tests, not changing code)
**Testing:** New tests must pass
**Coverage Increase:** 40% → 50%

---

#### Action 2.3: Consolidate Type Definitions (6-10 hours)

**Problem:** Types defined in 3 places:
- `modules/Types.py` (deprecated, 200 lines)
- `documentation/types.py` (canonical, 650+ lines)
- `protocols/types.py` (350+ lines)

**Strategy:**
1. Mark `modules/Types.py` as deprecated:
   ```python
   # modules/Types.py
   import warnings
   warnings.warn(
       "modules.Types is deprecated. Use HoloLoom.documentation.types instead.",
       DeprecationWarning,
       stacklevel=2
   )

   # Re-export from canonical location
   from HoloLoom.documentation.types import *
   ```

2. Merge unique types from `protocols/types.py`:
   - Keep protocol-specific types in `protocols/types.py`
   - Import shared types from `documentation/types.py`

3. Update 100+ import statements:
   ```bash
   # Find all imports
   grep -r "from.*modules\.Types import" HoloLoom/ --include="*.py"

   # Update to use documentation.types
   find HoloLoom -name "*.py" -exec sed -i \
     's/from.*modules\.Types import/from HoloLoom.documentation.types import/g' {} \;
   ```

4. Run full test suite:
   ```bash
   pytest HoloLoom/tests/ -v
   ```

**Risk:** Medium (changes imports across codebase)
**Testing:** Full test suite must pass
**Lines Saved:** 400-600

---

#### Action 2.4: Fix CLAUDE.md File Paths (1-2 hours)

**Changes Needed:**
- Fix 30+ instances of incorrect case
- Fix 10+ instances of wrong file paths
- Update spinner directory references
- Verify all code examples

**See:** Documentation Audit Report Section 12 for line-by-line fixes

**Risk:** None (documentation only)
**Testing:** Manual review of examples

---

#### Action 2.5: Consolidate Test Directories (2-3 hours)

**Current State:**
- `HoloLoom/tests/` (79 files, organized unit/integration/e2e)
- `tests/` (34 files, root-level)

**Strategy:**
1. Review root `tests/` files:
   ```bash
   ls tests/integration/
   ```

2. Categorize and move:
   - Integration tests → `HoloLoom/tests/integration/`
   - E2E tests → `HoloLoom/tests/e2e/`

3. Update pytest configuration:
   ```ini
   # pytest.ini
   [pytest]
   testpaths = HoloLoom/tests
   ```

4. Delete empty root `tests/` directory:
   ```bash
   rmdir tests/
   ```

5. Update all test path references in documentation

**Risk:** Low (just moving files)
**Testing:** Verify all tests still discoverable

---

### Phase 3: Medium-Priority Cleanup (P2 - Month 1)

#### Action 3.1-3.7: See P2 Matrix Above

Detailed plans for:
- Consolidate dashboards (8-12 hours)
- Add recursive learning tests (8-10 hours)
- Consolidate MCP servers (6-8 hours)
- Consolidate test fixtures (4-6 hours)
- Create missing READMEs (4-6 hours)
- Fix broken doc links (2 hours)
- Consolidate UI directories (2-3 hours)

---

### Phase 4: Low-Priority Cleanup (P3 - Month 2+)

#### Actions 4.1-4.7: See P3 Matrix Above

---

## MODIFIED FILES REVIEW

**14 Files Modified, +1,945 insertions**

**Commit Strategy:**

**Commit 1: Delete stray file**
```bash
git rm nul
git commit -m "chore: Remove stray nul file"
```

**Commit 2: Safety guardrails integration**
```bash
git add HoloLoom/loom/command.py
git add HoloLoom/warp/optimized.py
git add HoloLoom/warp/space.py
git add tests/integration/test_warp_drive_complete.py
git commit -m "feat: integrate safety guardrails across weaving architecture

- Safety guardrails in Loom Command, Warp Space, GPU Warp Space
- Complete provenance tracking for all warp operations
- Sparsity parameter for Warp Space (0.0-1.0 breathing control)
- Test coverage for guardrail metadata

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Commit 3: LLM integration enhancements**
```bash
git add HoloLoom/agentic/core.py
git add HoloLoom/weaving_orchestrator.py
git commit -m "feat: enhance LLM answer generation in agentic orchestrator

- LLM response generation in _direct_answer() with graceful fallback
- Context extraction from retrieved memories
- Packed context support in ToolExecutor
- Comprehensive error handling and logging

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Commit 4: Web dashboard**
```bash
git add HoloLoom/web_dashboard/agentic_server.py
git commit -m "feat: complete web dashboard with conversation management

- SQLite-based conversation persistence
- Project organization (folders/categorization)
- WebSocket real-time chat
- 4 reasoning modes (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)
- Search, favorites, tags, rename, delete
- Promptly integration (optional)
- 25+ action handlers

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Commit 5: Documentation updates**
```bash
git add CLAUDE.md
git add README.md
git add .claude/settings.local.json
git commit -m "docs: expand root file API documentation

- Complete hololoom.py API documentation (unified memory system)
- terminal_ui.py documentation (interactive CLI)
- weaving_orchestrator_llm.py documentation (LLM variant)
- Better module structure explanations (+238 lines)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Commit 6: Case compatibility**
```bash
git add HoloLoom.py
git add HoloLoom/__init__.py
git add HoloLoom/spinningWheel/__init__.py
git commit -m "fix: add case-insensitive import compatibility for Documentation module

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## GITIGNORE UPDATES

**Add to `.gitignore`:**
```gitignore
# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Test databases
*.db
test_*.db
demo_*.db
data/test_*.db
data/demo_*.db

# Test results
demo_test_results.json
test_results*.json

# Temporary files
nul
*.tmp
.DS_Store

# IDE
.vscode/
.idea/
*.swp
*.swo

# Build artifacts
dist/
build/
*.egg-info/
```

---

## SUCCESS METRICS

### Code Quality Metrics

| Metric | Before | After P0 | After P1 | After P2 | Target |
|--------|--------|----------|----------|----------|--------|
| **Duplicate Code** | 10,000 lines | 7,500 | 5,000 | 3,500 | <3,000 |
| **Root .md Files** | 31 | 16 | 10 | 7 | 5-7 |
| **Test Coverage** | 40% | 42% | 50% | 55% | 60% |
| **Untested Critical Modules** | 10 | 9 | 4 | 2 | 0 |
| **Case Inconsistencies** | 170 | 0 | 0 | 0 | 0 |
| **Duplicate Directories** | 3 sets | 1 set | 0 | 0 | 0 |
| **Broken Doc Links** | ~15 | ~10 | 0 | 0 | 0 |

### Time Investment vs. Impact

| Phase | Time | Code Saved | Coverage Gained | ROI |
|-------|------|------------|-----------------|-----|
| P0 | 11.5-16 hrs | 2,500 lines | +2% | Very High |
| P1 | 29-43 hrs | 2,800 lines | +10% | High |
| P2 | 34-47 hrs | 2,300 lines | +5% | Medium |
| P3 | 24-33 hrs | 500 lines | +5% | Medium |
| **Total** | **99-139 hrs** | **8,100 lines** | **+22%** | **High** |

---

## RISK ASSESSMENT

### Low Risk Actions (Safe to Execute)
- Delete stray files ✅
- Archive session docs ✅
- Commit untracked files ✅
- Fix case inconsistency ✅
- Add tests ✅
- Documentation fixes ✅

### Medium Risk Actions (Require Testing)
- Consolidate spinningWheel ⚠️
- Consolidate memory stores ⚠️
- Consolidate type definitions ⚠️
- Reorganize test directories ⚠️

### High Risk Actions (Require Careful Planning)
- None identified 👍

**Risk Mitigation:**
- Comprehensive test suite before each change
- Feature flags for gradual rollout (if needed)
- Git commits at each stage for easy rollback
- Code review for medium-risk changes

---

## TIMELINE

### Week 1 (P0): Foundation Cleanup
- **Days 1-2:** Consolidate spinningWheel, fix case consistency
- **Day 3:** Commit untracked files, archive session docs
- **Day 4:** Review and commit modified files
- **Day 5:** Documentation cleanup (CLAUDE.md paths)

### Weeks 2-3 (P1): Critical Improvements
- **Week 2:** Consolidate memory stores, add agentic tests
- **Week 3:** Consolidate types, add server tests, test directory cleanup

### Month 1 (P2): Quality Improvements
- **Weeks 4-5:** Dashboard consolidation, recursive tests, MCP servers
- **Weeks 6-7:** Test fixtures, READMEs, documentation links

### Month 2+ (P3): Polish
- **Ongoing:** File splitting, protocol consolidation, demo organization
- **Ongoing:** Property-based tests, benchmarks, load tests

---

## TEAM ASSIGNMENTS

**If working solo:** Follow phases sequentially
**If team of 2-3:** Parallelize phases

**Suggested Roles:**
- **Developer 1 (Senior):** P0-P1 code consolidation (spinningWheel, memory stores)
- **Developer 2 (Mid):** P1 testing (agentic, server, convergence)
- **Developer 3 (Junior/Docs):** P0-P2 documentation (CLAUDE.md, READMEs, links)

---

## MONITORING & REPORTING

**Weekly Checkpoints:**
- Lines of code removed (target: -8,000)
- Test coverage % (target: 40% → 60%)
- Documentation issues closed (target: 62 → 0)
- Duplicate code % (target: 5-6% → <2%)

**Tools:**
- `pytest --cov` for coverage tracking
- `radon cc` for complexity metrics
- `bandit` for security scanning
- Custom script for duplicate detection

---

## NEXT STEPS

### Immediate (Today)

1. **Review this plan** with stakeholders
2. **Make decision:** spinningWheel vs spinning_wheel naming
3. **Create feature branch:** `cleanup/phase-0`
4. **Execute Action 1.1:** Delete stray files

### Tomorrow

5. **Execute Actions 1.2-1.3:** Consolidate spinningWheel, fix case
6. **Run full test suite:** Verify nothing broke
7. **Commit changes:** Follow commit strategy above

### This Week

8. **Execute Actions 1.4-1.5:** Commit untracked, archive docs
9. **Begin Action 2.1:** Start memory store consolidation
10. **Daily standup:** Track progress, blockers

---

## APPENDICES

### A. Complete File Listing
See individual reports:
- **Repository Structure Analysis** (Section 1)
- **Modified Files Analysis** (Section 3)
- **Dead Code Report** (Section 4)
- **Code Duplication Report** (Section 5)
- **Test Coverage Report** (Section 6)
- **Documentation Audit** (Section 7)

### B. Command Reference

**Find duplicate code:**
```bash
# Using pylint
pylint --disable=all --enable=duplicate-code HoloLoom/

# Using CPD (Copy-Paste Detector)
pmd cpd --minimum-tokens 50 --files HoloLoom/ --language python
```

**Check test coverage:**
```bash
pytest --cov=HoloLoom --cov-report=html HoloLoom/tests/
open htmlcov/index.html
```

**Find broken links:**
```bash
# Check markdown links
find . -name "*.md" -exec grep -H '\[.*\](.*\.md)' {} \; | \
  while read line; do
    file=$(echo $line | cut -d: -f1)
    link=$(echo $line | grep -oP '\(.*\.md\)' | tr -d '()')
    [ -f "$link" ] || echo "BROKEN: $file -> $link"
  done
```

### C. Rollback Strategy

**If something breaks:**
```bash
# Rollback last commit
git revert HEAD

# Rollback to specific commit
git revert <commit-hash>

# Restore specific file
git checkout HEAD~1 -- path/to/file

# Create emergency patch branch
git checkout -b hotfix/rollback-cleanup
```

---

**APPROVAL SIGNATURES**

- [ ] Tech Lead: _______________
- [ ] QA Lead: _______________
- [ ] Documentation Lead: _______________

**Date Approved:** _______________

**Cleanup Initiative Status:** DRAFT → AWAITING APPROVAL

---

*Generated by Claude Code Agent on November 2, 2025*
*Next Review: After Phase P0 completion*
