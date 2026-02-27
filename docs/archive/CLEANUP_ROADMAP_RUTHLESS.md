# 🧹 Ruthless Cleanup Roadmap - Phase 2

**Branch**: `cleanUp/11-8-25`
**Status**: Phase 1 Complete (Documentation organized)
**Next Steps**: Deep structural cleanup

---

## ✅ Phase 1 Complete (Nov 8, 2025)

- [x] Organized 78 markdown files into docs/ structure
- [x] Reduced root directory clutter by 88%
- [x] Created docs/{sessions, features, research, guides, archived}/
- [x] Commit: `611a0e2`

---

## 📋 Phase 2: Directory Consolidation (Next)

### **Priority 1: Log Directory Cleanup**

**Problem**: Multiple log directories scattered across root
```
alignment_logs/          ← Alignment framework logs
demo_alignment_logs/     ← Demo alignment logs (DUPLICATE?)
demo_production_logs/    ← Production demo logs
monitoring/              ← Monitoring logs
```

**Action**:
```bash
# Consolidate into single logs/ directory
mkdir -p logs/{alignment,production,monitoring}
mv alignment_logs/* logs/alignment/
mv demo_alignment_logs/* logs/alignment/demo/
mv demo_production_logs/* logs/production/
mv monitoring/* logs/monitoring/
rm -rf alignment_logs demo_alignment_logs demo_production_logs monitoring
```

**Commit**: `chore: Consolidate log directories into logs/`

---

### **Priority 2: Dashboard Consolidation**

**Problem**: Two dashboard directories
```
dashboard/      ← Main dashboard (React/Vite)
dashboards/     ← Multiple dashboards? (investigate)
```

**Action**:
1. Compare contents of both directories
2. If dashboards/ has multiple apps, rename to `dashboards_archive/`
3. Keep `dashboard/` as primary UI
4. Document what each dashboard does

**Investigation Needed**:
```bash
ls dashboard/
ls dashboards/
diff -r dashboard/ dashboards/
```

**Commit**: `chore: Consolidate dashboard directories`

---

### **Priority 3: Archive Cleanup**

**Problem**: Archive has 25+ subdirectories (some may be redundant)

**Current Structure**:
```
archive/
├── old_demos/
├── old_dev/
├── old_projects/
├── old_tests/
├── session_docs/              ← REDUNDANT (we just created docs/sessions/)
├── session_docs_cleanup_nov7_2025/
├── session_docs_nov2025/
├── demo_outputs_nov7/
├── demo_states_nov7_2025/
├── obsolete_dirs_nov7_2025/
└── ... 15+ more
```

**Action - Ruthless Consolidation**:
```bash
# Group by type, not by date
mkdir -p archive/{code,sessions,demos,configs,tests,misc}

# Move session docs (remove date-based naming)
mv archive/session_docs* archive/sessions/

# Move demos
mv archive/old_demos archive/demo_outputs_nov7 archive/demo_states_nov7_2025 archive/demos/

# Move old code
mv archive/old_dev archive/old_projects archive/obsolete_dirs_nov7_2025 archive/code/

# Move configs
mv archive/deployment_configs_nov7 archive/install_scripts_nov7 archive/configs/

# Move tests
mv archive/old_tests archive/root_tests_nov7_2025 archive/tests/
```

**Commit**: `chore: Restructure archive by type, not date`

---

### **Priority 4: Test Directory Consolidation**

**Problem**: Tests scattered across repository
```
tests/                  ← Root-level tests (integration?)
hololoom/tests/        ← Main test suite
archive/old_tests/     ← Archived tests
archive/root_tests_nov7_2025/  ← More archived tests
```

**Action**:
1. Review `tests/` at root - are these still needed?
2. If integration tests, move to `hololoom/tests/integration/`
3. If duplicates of archived tests, delete
4. Keep all tests under `hololoom/tests/`

**Investigation**:
```bash
ls tests/
diff -r tests/ hololoom/tests/
```

**Commit**: `chore: Consolidate all tests under hololoom/tests/`

---

### **Priority 5: Output/Data Directory Cleanup**

**Problem**: Multiple output directories
```
synthesis_output/       ← Synthesis outputs
memory_data/           ← Memory persistence data
reflections/           ← Reflection buffer data
demo_alignment_logs/   ← Demo logs
hololoom.egg-info/     ← Build artifact (should be in .gitignore)
```

**Action**:
```bash
# Consolidate data directories
mkdir -p data/{memory,synthesis,reflections}
mv memory_data/* data/memory/ 2>/dev/null || true
mv synthesis_output/* data/synthesis/ 2>/dev/null || true
mv reflections/* data/reflections/ 2>/dev/null || true

# Remove build artifacts
rm -rf hololoom.egg-info/

# Update .gitignore
echo "hololoom.egg-info/" >> .gitignore
echo "data/" >> .gitignore  # Don't commit runtime data
```

**Commit**: `chore: Consolidate data directories and ignore build artifacts`

---

### **Priority 6: Config Directory Consolidation**

**Problem**: Configs scattered
```
config/                        ← Top-level configs
promptly-matrix-bot/config/   ← Promptly configs
ouroboros/                     ← Has configs?
darkTrace/                     ← Has configs?
```

**Action**:
1. Keep service-specific configs in their directories
2. Move global configs to root `config/`
3. Document what each config does

**Commit**: `docs: Document config file locations and purposes`

---

## 📋 Phase 3: Code Organization

### **Priority 7: Separate Standalone Projects**

**Projects that could be standalone repositories**:
```
trough/              ← AI slop detector (standalone tool)
ouroboros/           ← Drug interaction system (separate domain)
darkTrace/           ← Entity extraction (separate tool)
chronos/             ← Time-based analysis
promptly-matrix-bot/ ← Matrix bot (separate service)
mcp_server/          ← MCP integration
```

**Action**: For each project, decide:
- [ ] Keep in monorepo (if tightly coupled to HoloLoom)
- [ ] Move to separate repo (if standalone utility)
- [ ] Archive (if deprecated)

**Document**: Create `PROJECTS.md` listing each project and its status

---

### **Priority 8: UI/Frontend Consolidation**

**Problem**: Multiple UI directories
```
ui/                  ← Generic UI?
dashboard/           ← Main dashboard
squad/               ← VS Code extension (TypeScript)
```

**Action**:
1. Rename for clarity:
   - `dashboard/` → `web-dashboard/`
   - `ui/` → investigate contents
   - `squad/` → `vscode-extension/`

**Commit**: `refactor: Rename UI directories for clarity`

---

## 📋 Phase 4: .gitignore & Build Cleanup

### **Priority 9: Comprehensive .gitignore**

**Add to .gitignore**:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project-specific
data/
logs/
*.log
memory_data/
synthesis_output/
demo_*_logs/
reflections/

# Node
node_modules/
.vite/
dist/
```

**Commit**: `chore: Add comprehensive .gitignore`

---

### **Priority 10: Remove Git-Tracked Build Artifacts**

**Find and remove**:
```bash
# Find large files
git ls-files | xargs du -sh | sort -hr | head -20

# Find node_modules accidentally committed
find . -name "node_modules" -type d -not -path "./.venv/*"

# Find .vite cache
find . -name ".vite" -type d

# Remove from git but keep locally
git rm -r --cached node_modules/ dashboard/node_modules/
git rm -r --cached dashboard/.vite/
```

**Commit**: `chore: Remove build artifacts from git tracking`

---

## 📋 Phase 5: Documentation Cleanup (Round 2)

### **Priority 11: README Consolidation**

**Problem**: Multiple READMEs everywhere
```bash
find . -name "README*.md" | wc -l
```

**Action**:
1. Each project directory gets ONE README
2. Archive old READMEs to docs/archived/
3. Update main README with project structure

**Commit**: `docs: Consolidate and update README files`

---

### **Priority 12: Create Master Documentation Index**

**Create**: `docs/INDEX.md`

**Contents**:
```markdown
# HoloLoom Documentation Index

## Essential Reading
- [README](../README.md) - Project overview
- [QUICK_START_GUIDE](../QUICK_START_GUIDE.md) - Get started in 5 minutes
- [CLAUDE.md](../CLAUDE.md) - AI assistant instructions
- [ARCHITECTURE_VISUAL_MAP](../ARCHITECTURE_VISUAL_MAP.md) - System architecture

## Deep Dives
- [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE](../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md) - Complete system guide

## By Topic
### Features
- [Feature Documentation](./features/) - Completed feature docs

### Sessions
- [Session Summaries](./sessions/) - Development session notes

### Research
- [Research & Roadmaps](./research/) - Research notes and future plans

### Guides
- [Quick Start Guides](./guides/) - Topic-specific guides
```

**Commit**: `docs: Add master documentation index`

---

## 📋 Phase 6: Final Polish

### **Priority 13: Update Main README**

**Add sections**:
- Project structure diagram
- Link to docs/INDEX.md
- Contribution guidelines
- License information

**Commit**: `docs: Update main README with current structure`

---

### **Priority 14: Create PROJECTS.md**

**Document all subprojects**:
```markdown
# HoloLoom Subprojects

## Core System
- **hololoom/** - Main neural decision-making system

## Tools & Utilities
- **trough/** - AI slop/ML logic detector
- **ouroboros/** - Drug interaction safety system
- **darkTrace/** - Entity extraction pipeline

## Services
- **promptly-matrix-bot/** - Matrix bot for prompt optimization
- **mcp_server/** - Model Context Protocol server

## Infrastructure
- **dashboard/** - Web-based monitoring dashboard
- **squad/** - VS Code extension

## Archived/Deprecated
- **archive/** - Old code and documentation
```

**Commit**: `docs: Add subprojects documentation`

---

## 📊 Success Metrics

### **Before Cleanup**:
- Root markdown files: 78
- Top-level directories: 29
- Scattered logs: 4 directories
- Scattered tests: 3 locations
- Archive structure: 25+ date-based dirs

### **After Full Cleanup** (Target):
- Root markdown files: 9 (essential only)
- Top-level directories: ~15 (organized)
- Logs: 1 directory (`logs/`)
- Tests: 1 location (`hololoom/tests/`)
- Archive structure: 6 type-based dirs

### **Estimated Commits**: 14 commits across 6 phases

---

## 🚀 Execution Plan

### **Recommended Order**:
1. **Phase 2** (Directories) - Low risk, high impact
2. **Phase 4** (.gitignore) - Prevent future clutter
3. **Phase 3** (Code org) - Requires decisions
4. **Phase 5** (Docs) - Build on Phase 1
5. **Phase 6** (Polish) - Final touches

### **Time Estimate**:
- Phase 2: 30 minutes
- Phase 3: 1-2 hours (decision-making)
- Phase 4: 20 minutes
- Phase 5: 45 minutes
- Phase 6: 30 minutes

**Total**: ~3-4 hours of focused cleanup

---

## 🎯 Quick Wins (Do First)

If short on time, prioritize:
1. ✅ **Priority 1**: Log consolidation (5 min)
2. ✅ **Priority 5**: Output/data cleanup (10 min)
3. ✅ **Priority 9**: .gitignore update (5 min)
4. ✅ **Priority 10**: Remove build artifacts (10 min)

**Total Quick Wins**: 30 minutes for 80% of the impact

---

## 🔥 Ruthless Mode (Optional)

If you want to be REALLY ruthless:

### **Delete These If Unused**:
```bash
# Demo logs (regeneratable)
rm -rf demo_*_logs/

# Synthesis output (regeneratable)
rm -rf synthesis_output/

# Reflections (if not needed)
rm -rf reflections/

# Old experiments
rm -rf experiments/  # IF no valuable results

# Memory data (if can rebuild)
rm -rf memory_data/
```

### **Archive These Entire Projects** (if not actively used):
```bash
# Move to archive if not in active development
mv trough/ archive/code/trough/
mv darkTrace/ archive/code/darkTrace/
mv chronos/ archive/code/chronos/
```

---

## 📝 Notes

- **Branch**: All cleanup happens on `cleanUp/11-8-25`
- **Safety**: Master branch untouched until merge
- **Reversible**: Each commit can be reverted individually
- **Documentation**: Update CLAUDE.md after major changes

---

**Last Updated**: November 8, 2025
**Author**: Blake + Claude Code
**Status**: Ready to execute Phase 2
