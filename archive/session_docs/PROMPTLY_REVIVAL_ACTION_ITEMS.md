# Promptly Revival - Immediate Action Items

**Created:** November 21, 2025
**Purpose:** Quick checklist for getting started with Promptly revival

---

## Decision Point: Choose Integration Strategy

### Option A: Full Standalone Revival
**Effort:** 1-2 weeks | **Best For:** Separate product

**Quick Steps:**
1. Move entire Promptly from archive to root
2. Update dependencies
3. Test QUICK_TEST.py
4. Deploy web dashboard
5. Market as standalone product

---

### Option B: Selective Integration ⭐ **RECOMMENDED**
**Effort:** 2-3 weeks | **Best For:** Maximum value, minimum duplication

**Quick Steps:**
1. Week 1: Integrate MCP tools (800 lines) + Skills (600 lines)
2. Week 2: Integrate evaluation tools (A/B testing, LLM-as-judge, cost tracker)
3. Week 3: Integrate web dashboard + testing

**Result:**
- HoloLoom gains 5 new capabilities
- Promptly CLI remains for users
- No code duplication

---

### Option C: Hybrid Deployment
**Effort:** 3-4 weeks | **Best For:** Microservices architecture

**Quick Steps:**
1. Week 1: Design REST API
2. Week 2: Implement API client/server
3. Week 3: Integration testing
4. Week 4: Deploy both services

---

## Immediate Next Steps (If Option B Chosen)

### Step 1: Verify Current State (30 minutes)

**Check Archive Status:**
```bash
# Verify Promptly files exist
ls archive/old_projects/Promptly/

# Count Python files
find archive/old_projects/Promptly -name "*.py" | wc -l

# Check key files
ls archive/old_projects/Promptly/promptly/promptly.py
ls archive/old_projects/Promptly/promptly/integrations/mcp_server.py
ls archive/old_projects/Promptly/promptly/tools/ab_testing.py
```

**Check HoloLoom Test Suite:**
```bash
# Run baseline tests
cd HoloLoom/
pytest tests/ -v

# Check if directories exist for integration
ls HoloLoom/agentic/
ls HoloLoom/web_dashboard/
```

---

### Step 2: Create Feature Branch (5 minutes)

```bash
# Create and checkout feature branch
git checkout -b feature/promptly-integration

# Verify branch
git branch
```

---

### Step 3: Move Promptly to Root (15 minutes)

```bash
# Create promptly directory at root
mkdir -p promptly/

# Copy (not move yet, for safety) Promptly files
cp -r archive/old_projects/Promptly/promptly/* promptly/
cp -r archive/old_projects/Promptly/demos promptly/
cp -r archive/old_projects/Promptly/docs promptly/
cp -r archive/old_projects/Promptly/tests promptly/

# Copy root files
cp archive/old_projects/Promptly/README.md promptly/
cp archive/old_projects/Promptly/requirements.txt promptly/
cp archive/old_projects/Promptly/QUICK_TEST.py promptly/

# Verify structure
ls -la promptly/
```

---

### Step 4: Test Promptly CLI (10 minutes)

```bash
cd promptly/

# Install dependencies
pip install -r requirements.txt

# Run quick test
python QUICK_TEST.py

# Expected output: All 6 systems operational
```

---

### Step 5: Start Week 1 Integration (Day 1)

**Task 1.1: MCP Tools Integration**

```bash
# Create directory
mkdir -p HoloLoom/mcp_tools/

# Copy MCP server
cp promptly/integrations/mcp_server.py HoloLoom/mcp_tools/server.py

# Create __init__.py
touch HoloLoom/mcp_tools/__init__.py
```

**Update imports in `HoloLoom/mcp_tools/server.py`:**
- Change: `from tools.prompt_analytics import ...`
- To: `from HoloLoom.monitoring.cost_tracker import ...`

**Create README:**
```bash
touch HoloLoom/mcp_tools/README.md
```

**Create tests:**
```bash
mkdir -p HoloLoom/mcp_tools/tests/
touch HoloLoom/mcp_tools/tests/test_mcp_tools.py
```

---

## Weekly Milestones

### Week 1 Milestone: MCP Tools + Skills ✅

**Deliverables:**
- [ ] `HoloLoom/mcp_tools/server.py` (800 lines)
- [ ] 27 MCP tools working
- [ ] `HoloLoom/agentic/skills/` (600 lines)
- [ ] 13 skill templates loadable
- [ ] `HoloLoom/mcp_tools/README.md`
- [ ] `HoloLoom/agentic/skills/README.md`
- [ ] Tests passing: `test_mcp_tools.py`, `test_skills.py`
- [ ] Demo: `demos/demo_skills_system.py`

**Success Criteria:**
```bash
# All tests pass
pytest HoloLoom/mcp_tools/tests/ -v
pytest HoloLoom/agentic/skills/tests/ -v

# Demo works
PYTHONPATH=. python demos/demo_skills_system.py

# HoloLoom still works
pytest HoloLoom/tests/ -v
```

---

### Week 2 Milestone: Evaluation Tools ✅

**Deliverables:**
- [ ] `HoloLoom/evaluation/ab_testing.py` (450 lines)
- [ ] `HoloLoom/evaluation/llm_judge.py` (600 lines)
- [ ] `HoloLoom/monitoring/cost_tracker.py` (400 lines)
- [ ] `HoloLoom/evaluation/README.md`
- [ ] Tests passing: `test_ab_testing.py`, `test_llm_judge.py`, `test_cost_tracker.py`
- [ ] Demos: `demo_ab_testing.py`, `demo_llm_judge.py`

**Success Criteria:**
```bash
# All tests pass
pytest HoloLoom/evaluation/tests/ -v
pytest HoloLoom/monitoring/tests/ -v

# Demos work
PYTHONPATH=. python demos/demo_ab_testing.py
PYTHONPATH=. python demos/demo_llm_judge.py

# Integration works
pytest HoloLoom/tests/integration/test_evaluation.py -v
```

---

### Week 3 Milestone: Web Dashboard + Testing ✅

**Deliverables:**
- [ ] `HoloLoom/web_dashboard/promptly/analytics.py` (500 lines)
- [ ] `HoloLoom/web_dashboard/promptly/templates/`
- [ ] `HoloLoom/analytics/promptly_bridge.py`
- [ ] All integration tests passing
- [ ] Documentation updated (CLAUDE.md, etc.)
- [ ] `PROMPTLY_INTEGRATION_COMPLETE.md`

**Success Criteria:**
```bash
# Web dashboard runs
cd HoloLoom/web_dashboard/promptly/
python analytics.py
# → http://localhost:8001

# All tests pass
pytest HoloLoom/tests/ -v

# Promptly CLI still works
cd promptly/
python QUICK_TEST.py

# Documentation complete
cat PROMPTLY_INTEGRATION_COMPLETE.md
```

---

## File Relocation Map

### Files to Move to HoloLoom

| Source (Promptly) | Destination (HoloLoom) | Size |
|-------------------|------------------------|------|
| `integrations/mcp_server.py` | `mcp_tools/server.py` | 800 lines |
| `skill_templates_extended.py` | `agentic/skills/templates.py` | 600 lines |
| `package_manager.py` | `agentic/skills/manager.py` | 400 lines |
| `tools/ab_testing.py` | `evaluation/ab_testing.py` | 450 lines |
| `tools/llm_judge_enhanced.py` | `evaluation/llm_judge.py` | 600 lines |
| `tools/cost_tracker.py` | `monitoring/cost_tracker.py` | 400 lines |
| `web_dashboard_realtime.py` | `web_dashboard/promptly/analytics.py` | 500 lines |

**Total:** ~3,750 lines (includes tests + docs)

### Files to Keep in Promptly

| File | Size | Reason |
|------|------|--------|
| `promptly.py` | 1,200 lines | Version control (user-facing) |
| `team_collaboration.py` | 400 lines | Multi-user accounts (user-facing) |
| `recursive_loops.py` | 900 lines | User-facing loop CLI |
| `loop_composition.py` | 320 lines | User-facing DSL |
| `integrations/hololoom_bridge.py` | 450 lines | Bridge to HoloLoom memory |
| `tools/prompt_analytics.py` | 470 lines | Analytics database |

**Total:** ~3,740 lines remain in Promptly CLI

---

## Quick Commands Reference

### Testing Commands

```bash
# Test HoloLoom (baseline)
pytest HoloLoom/tests/ -v

# Test Promptly CLI
cd promptly/ && python QUICK_TEST.py

# Test specific module
pytest HoloLoom/mcp_tools/tests/ -v
pytest HoloLoom/agentic/skills/tests/ -v
pytest HoloLoom/evaluation/tests/ -v

# Test integration
pytest HoloLoom/tests/integration/test_promptly_integration.py -v

# Run demos
PYTHONPATH=. python demos/demo_skills_system.py
PYTHONPATH=. python demos/demo_ab_testing.py
PYTHONPATH=. python demos/demo_llm_judge.py
```

### Git Commands

```bash
# Create feature branch
git checkout -b feature/promptly-integration

# Stage changes
git add HoloLoom/mcp_tools/
git add HoloLoom/agentic/skills/
git add HoloLoom/evaluation/
git add promptly/

# Commit (after each feature)
git commit -m "feat: Integrate Promptly MCP tools (27 tools)"
git commit -m "feat: Integrate Promptly skills system (13 templates)"
git commit -m "feat: Integrate Promptly evaluation tools"

# Push branch
git push origin feature/promptly-integration

# Merge when complete
git checkout master
git merge feature/promptly-integration
```

### Documentation Commands

```bash
# Update CLAUDE.md
vim CLAUDE.md
# → Add "Promptly Integration" section

# Create module READMEs
vim HoloLoom/mcp_tools/README.md
vim HoloLoom/agentic/skills/README.md
vim HoloLoom/evaluation/README.md

# Create completion summary
vim PROMPTLY_INTEGRATION_COMPLETE.md
```

---

## Common Issues & Solutions

### Issue 1: Import Errors After Moving Files

**Problem:**
```python
ImportError: No module named 'tools.prompt_analytics'
```

**Solution:**
Update imports:
```python
# Old (Promptly)
from tools.prompt_analytics import PromptAnalytics

# New (HoloLoom)
from HoloLoom.monitoring.cost_tracker import CostTracker
```

---

### Issue 2: SQLite Database Path Errors

**Problem:**
```
sqlite3.OperationalError: unable to open database file
```

**Solution:**
Update database paths:
```python
# Old (Promptly)
db_path = ".promptly/promptly.db"

# New (HoloLoom)
db_path = Path.home() / ".hololoom" / "promptly.db"
```

---

### Issue 3: Flask App Conflicts

**Problem:**
```
Address already in use: Port 5000
```

**Solution:**
Use different port for Promptly dashboard:
```python
# Old
app.run(port=5000)

# New
app.run(port=8001)  # Avoid conflict with HoloLoom web_dashboard
```

---

### Issue 4: Tests Failing After Integration

**Problem:**
```
AssertionError: Expected 10 tools, got 37
```

**Solution:**
Update test assertions to account for new tools:
```python
# Old
assert len(tools) == 10

# New
assert len(tools) >= 10  # 10 original + 27 from Promptly
```

---

## Dependencies to Install

### HoloLoom Requirements (Add to requirements.txt)

```txt
# Promptly Integration
flask>=3.0.0
flask-socketio>=5.3.0
eventlet>=0.33.0
python-socketio>=5.10.0
```

### Promptly Requirements (Keep separate)

Promptly keeps its own `requirements.txt` for CLI usage.

---

## Quality Checklist

Before marking each feature complete:

### Code Quality
- [ ] No hardcoded paths
- [ ] All imports updated
- [ ] No circular dependencies
- [ ] Type hints preserved
- [ ] Docstrings complete

### Testing
- [ ] Unit tests written
- [ ] Integration tests written
- [ ] All tests passing
- [ ] No regressions in HoloLoom
- [ ] Demo script works

### Documentation
- [ ] README.md created for module
- [ ] API reference complete
- [ ] Usage examples provided
- [ ] CLAUDE.md updated

### Integration
- [ ] Works with existing HoloLoom features
- [ ] No breaking changes
- [ ] Backward compatible
- [ ] Feature flag (if needed)

---

## Success Criteria Summary

### Week 1 Success
✅ 27 MCP tools working in HoloLoom
✅ 13 skill templates loadable
✅ Tests passing
✅ HoloLoom still works

### Week 2 Success
✅ A/B testing compares strategies
✅ LLM-as-judge evaluates responses
✅ Cost tracking integrated
✅ Tests passing
✅ Demos work

### Week 3 Success
✅ Web dashboard shows HoloLoom metrics
✅ Real-time updates working
✅ All integration tests passing
✅ Documentation complete
✅ Promptly CLI still works

### Overall Success
✅ 5 new HoloLoom capabilities (MCP, skills, evaluation, cost tracking, dashboard)
✅ 0 code duplication
✅ 0 regressions
✅ 100% test coverage for new features
✅ Documentation complete

---

## Resources

### Reference Documents
- `PROMPTLY_REVIVAL_PLAN.md` - Complete revival plan (all 3 options)
- `PROMPTLY_INTEGRATION_ROADMAP.md` - Visual integration guide
- `archive/old_projects/Promptly/PROMPTLY_COMPREHENSIVE_REVIEW.md` - Full Promptly review

### Code Locations
- **Archive:** `archive/old_projects/Promptly/`
- **New Location:** `promptly/` (root level)
- **HoloLoom Integration:** `HoloLoom/mcp_tools/`, `HoloLoom/agentic/skills/`, `HoloLoom/evaluation/`

### Contact/Support
- Issues: GitHub issues
- Questions: Review PROMPTLY_COMPREHENSIVE_REVIEW.md

---

**Ready to start? Begin with Step 1: Verify Current State** ✅
