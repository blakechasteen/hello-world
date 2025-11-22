# Promptly Revival Plan

**Created:** November 21, 2025
**Status:** Proposal - Awaiting Approval
**Estimated Timeline:** 2-4 weeks depending on strategy chosen

---

## Executive Summary

Promptly is a production-ready prompt engineering platform with **17,000 lines of code** and **340 real executions tracked**. Analysis shows only **~35% of Promptly's functionality** was integrated into HoloLoom, leaving **65% unique features** that provide significant value:

**Unique Value Propositions:**
- Multi-user team collaboration (HoloLoom has multi-agent only)
- Real-time web dashboard with WebSocket updates
- 27 MCP tools for Claude Desktop integration
- 13 skill templates with package management
- Git-style version control for prompts (branches, commits, merges)
- A/B testing framework for prompt optimization
- LLM-as-judge with enhanced evaluation
- Complete analytics database (SQLite-based, 340 executions)

**Recommendation:** **Option B - Selective Integration** provides the best balance of effort, value, and maintainability.

---

## Current State Assessment

### Code Quality: ✅ Production Ready (99% functional)

**Total Code**: 17,088 lines across 50+ Python files
- Core Database: 1,200 lines (Git-style version control)
- Recursive Loops: 900 lines (6 loop types)
- Analytics: 470 lines (340 executions tracked)
- HoloLoom Bridge: 450 lines (neural memory integration)
- Team Collaboration: 400 lines (multi-user with roles)
- Web Dashboard: 500 lines (real-time WebSocket)
- MCP Tools: 800 lines (27 tools for Claude)
- Loop Composition: 320 lines (chain multiple loops)
- Skills System: 600 lines (13 templates)
- Rich CLI: 400 lines (beautiful terminal)

**Test Coverage**: 6/6 core systems operational
- QUICK_TEST.py validates all components
- tests/test_mcp_tools.py (MCP integration)
- tests/test_recursive_loops.py (6 loop types)

**Documentation**: 20+ guides (8,000+ lines)
- PROMPTLY_COMPREHENSIVE_REVIEW.md (1,200 lines)
- STATUS_AT_A_GLANCE.md (330 lines)
- Complete API documentation
- Multiple tutorials and quickstarts

### Dependencies Analysis

**Core Dependencies** (requirements.txt):
```txt
flask>=3.0.0
flask-socketio>=5.3.0
eventlet>=0.33.0
python-socketio>=5.10.0
requests>=2.31.0
```

**Setup.py Dependencies** (minimal):
```python
click>=8.0.0
PyYAML>=6.0
```

**Optional Dependencies** (for full features):
- `ollama` - Local LLM inference (HoloLoom compatible)
- `anthropic` - Claude API (HoloLoom compatible)
- `openai` - OpenAI API (HoloLoom compatible)
- `rich` - Terminal formatting (already in HoloLoom)
- `pyyaml` - Config files (standard library)

**Conflicts with HoloLoom**: ⚠️ Minor (1 conflict)
- Promptly uses SQLite for analytics/version control
- HoloLoom uses Neo4j/Qdrant for memory
- **Resolution**: Keep separate databases (no conflict)

### Architecture Compatibility

**Promptly's Architecture**:
```
┌─────────────────┐
│   Promptly CLI  │
└────────┬────────┘
         │
    ┌────┼────┐
    │    │    │
┌───▼┐ ┌─▼──┐ ┌──▼──┐
│Loop│ │Web │ │MCP  │
│    │ │Dash│ │Srvr │
└───┬┘ └─┬──┘ └──┬──┘
    │    │       │
    └────┼───────┘
         │
┌────────▼─────────┐
│  Storage Layer   │
│  • SQLite (3 DBs)│
│  • File System   │
│  • HoloLoom      │
└──────────────────┘
```

**Integration Points with HoloLoom**:
1. ✅ **HoloLoom Bridge** (`promptly/integrations/hololoom_bridge.py`)
   - Already implemented
   - Stores loop results in HoloLoom memory
   - Retrieves past patterns from knowledge graph
   - Meta-learning: "What loop type worked best?"

2. 🟡 **Shared LLM Clients** (Ollama, Anthropic, OpenAI)
   - Both systems use same API clients
   - Can share configuration

3. ❌ **No Direct Conflicts**
   - Promptly: User-facing prompt engineering
   - HoloLoom: Agent-facing reasoning system
   - Complementary, not competitive

---

## Revival Strategy Options

### Option A: Full Standalone Revival

**Approach**: Move Promptly out of archive, update dependencies, deploy as separate platform.

**Pros**:
- ✅ Preserves all 17,000 lines of code
- ✅ Maintains complete feature set
- ✅ Independent deployment and versioning
- ✅ No risk of breaking HoloLoom
- ✅ Can market as separate product

**Cons**:
- ⚠️ Duplicate LLM client code
- ⚠️ Separate documentation maintenance
- ⚠️ Users need to learn two systems
- ⚠️ Potential code divergence over time

**Effort**: 1-2 weeks
- Move files from archive to root or separate repo
- Update dependencies (Flask 3.0, socketio 5.3)
- Test all 6 core systems
- Update documentation paths
- Deploy web dashboard

**Best For**: If we want to market Promptly as a standalone product or keep it completely separate from HoloLoom.

---

### Option B: Selective Integration (RECOMMENDED)

**Approach**: Integrate Promptly's unique features (65%) into HoloLoom as new modules while keeping core separate.

**Features to Integrate**:

1. **MCP Tools** (800 lines) → `HoloLoom/mcp_tools/`
   - 27 tools for Claude Desktop
   - Complements HoloLoom's existing MCP ingestion
   - Effort: 3 days

2. **Skills System** (600 lines) → `HoloLoom/agentic/skills/`
   - 13 skill templates with package management
   - Natural fit with agentic reasoning
   - Effort: 2 days

3. **A/B Testing** (450 lines) → `HoloLoom/evaluation/ab_testing.py`
   - Compare prompt/strategy variants
   - Useful for policy optimization
   - Effort: 2 days

4. **LLM-as-Judge Enhanced** (600 lines) → `HoloLoom/evaluation/llm_judge.py`
   - Multi-criteria evaluation
   - Quality scoring
   - Effort: 2 days

5. **Web Dashboard** (500 lines) → `HoloLoom/web_dashboard/promptly/`
   - Real-time analytics
   - WebSocket updates
   - Effort: 4 days

**Features to Keep Separate** (in Promptly):

1. **Team Collaboration** (400 lines)
   - Multi-user accounts, teams, roles
   - User-facing, not agent-facing
   - Keep in Promptly CLI

2. **Version Control** (1,200 lines)
   - Git-style commits, branches, merges
   - SQLite-based
   - Keep in Promptly CLI

3. **Loop Composition DSL** (320 lines)
   - Custom parser for prompt chains
   - User-facing workflow tool
   - Keep in Promptly CLI

4. **Recursive Loops** (900 lines)
   - Already have `HoloLoom/recursive/` (more advanced)
   - Keep Promptly version for user-facing CLI

**Architecture After Integration**:
```
HoloLoom (Agent System)
├── agentic/
│   └── skills/          ← FROM PROMPTLY
├── evaluation/
│   ├── ab_testing.py    ← FROM PROMPTLY
│   └── llm_judge.py     ← FROM PROMPTLY
├── mcp_tools/           ← FROM PROMPTLY (27 tools)
└── web_dashboard/
    └── promptly/        ← FROM PROMPTLY (analytics)

Promptly (User System)
├── promptly/
│   ├── promptly.py      (version control)
│   ├── team_collaboration.py
│   ├── recursive_loops.py
│   └── loop_composition.py
└── web_dashboard_realtime.py
```

**Pros**:
- ✅ Best features integrated into HoloLoom
- ✅ No duplication of MCP/skills/evaluation code
- ✅ Promptly remains usable for users
- ✅ Clear separation: HoloLoom (agents) vs Promptly (users)
- ✅ Gradual migration path

**Cons**:
- ⚠️ Requires careful refactoring (2-3 weeks)
- ⚠️ Need to maintain integration tests
- ⚠️ Documentation split across two systems

**Effort**: 2-3 weeks
- Week 1: MCP tools, Skills system
- Week 2: A/B testing, LLM-as-judge
- Week 3: Web dashboard, integration tests

**Best For**: Maximizing value while minimizing maintenance burden. Gets best features into HoloLoom while preserving user-facing Promptly.

---

### Option C: Hybrid Deployment

**Approach**: Deploy Promptly as a separate service that communicates with HoloLoom via API.

**Architecture**:
```
┌──────────────┐     REST API     ┌──────────────┐
│   Promptly   │ ←──────────────→ │   HoloLoom   │
│  (User CLI)  │                  │  (Agents)    │
│              │                  │              │
│ • Teams      │                  │ • Memory     │
│ • Version Ctl│                  │ • Policy     │
│ • Analytics  │                  │ • Reasoning  │
│ • Dashboard  │                  │ • Learning   │
└──────────────┘                  └──────────────┘
```

**Integration Points**:
- Promptly calls HoloLoom API for memory storage
- HoloLoom triggers Promptly loops for refinement
- Shared authentication/authorization
- Event bus for notifications

**Pros**:
- ✅ Complete separation of concerns
- ✅ Both systems fully functional
- ✅ Can scale independently
- ✅ Clean API boundaries
- ✅ Microservices architecture

**Cons**:
- ⚠️ API overhead (network latency)
- ⚠️ Complex deployment (2 services)
- ⚠️ Requires API design and versioning
- ⚠️ More infrastructure (2 servers)

**Effort**: 3-4 weeks
- Week 1: API design, authentication
- Week 2: Promptly API client, HoloLoom endpoints
- Week 3: Integration testing
- Week 4: Deployment, monitoring

**Best For**: If we plan to scale Promptly and HoloLoom independently or want true microservices architecture.

---

## Comparison Matrix

| Criterion | Option A (Standalone) | Option B (Selective) | Option C (Hybrid) |
|-----------|----------------------|---------------------|-------------------|
| **Effort** | 1-2 weeks | 2-3 weeks | 3-4 weeks |
| **Code Reuse** | ❌ Low | ✅ High | 🟡 Medium |
| **Maintenance** | ⚠️ High (2 systems) | ✅ Low (unified) | ⚠️ High (2 services) |
| **User Experience** | ✅ Best (complete) | 🟡 Good (split) | ✅ Best (separate) |
| **Scalability** | 🟡 Medium | ✅ High | ✅ Highest |
| **Complexity** | ✅ Low | 🟡 Medium | ⚠️ High |
| **Risk** | ✅ Low | 🟡 Medium | ⚠️ High |

---

## Recommended Approach: Option B (Selective Integration)

**Rationale**:
1. **Maximizes Value**: Best features (MCP, skills, evaluation) go into HoloLoom
2. **Minimizes Duplication**: No duplicate code for shared functionality
3. **Clear Separation**: HoloLoom (agents) vs Promptly (users)
4. **Gradual Path**: Can integrate incrementally over 2-3 weeks
5. **Best ROI**: Most value for least effort

### Phase 1: Core Features (Week 1)

**1.1 MCP Tools Integration** (3 days)
- Move `promptly/integrations/mcp_server.py` → `HoloLoom/mcp_tools/`
- Move 27 tool implementations
- Update imports and dependencies
- Test with Claude Desktop
- Update CLAUDE.md documentation

**1.2 Skills System Integration** (2 days)
- Move `promptly/skill_templates_extended.py` → `HoloLoom/agentic/skills/`
- Move 13 skill templates to `HoloLoom/agentic/skills/templates/`
- Move `promptly/package_manager.py` → `HoloLoom/agentic/skills/manager.py`
- Create `SKILLS_README.md` in HoloLoom
- Test skill loading and execution

### Phase 2: Evaluation Tools (Week 2)

**2.1 A/B Testing Framework** (2 days)
- Move `promptly/tools/ab_testing.py` → `HoloLoom/evaluation/ab_testing.py`
- Integrate with policy engine for strategy comparison
- Add tests: `HoloLoom/evaluation/tests/test_ab_testing.py`
- Create demo: `demos/demo_ab_testing.py`

**2.2 LLM-as-Judge Enhanced** (2 days)
- Move `promptly/tools/llm_judge_enhanced.py` → `HoloLoom/evaluation/llm_judge.py`
- Integrate with agentic reasoning for quality scoring
- Add tests: `HoloLoom/evaluation/tests/test_llm_judge.py`
- Create demo: `demos/demo_llm_judge.py`

**2.3 Cost Tracking** (1 day)
- Move `promptly/tools/cost_tracker.py` → `HoloLoom/monitoring/cost_tracker.py`
- Integrate with weaving orchestrator
- Add Prometheus metrics export

### Phase 3: Web Dashboard (Week 3)

**3.1 Dashboard Integration** (3 days)
- Move `promptly/web_dashboard_realtime.py` → `HoloLoom/web_dashboard/promptly_analytics.py`
- Move templates to `HoloLoom/web_dashboard/templates/promptly/`
- Adapt to HoloLoom's data models (Spacetime, reflection buffer)
- Update WebSocket events for HoloLoom metrics

**3.2 Analytics Bridge** (1 day)
- Create `HoloLoom/analytics/promptly_bridge.py`
- Export HoloLoom executions to Promptly analytics format
- Enable cross-system analytics queries

**3.3 Integration Testing** (1 day)
- Test all integrated features end-to-end
- Verify no regressions in HoloLoom
- Verify Promptly CLI still works (separate)

---

## Promptly CLI Remains Separate

After integration, Promptly CLI remains a **user-facing prompt engineering tool**:

**Location**: `promptly/` (root level, not archive)

**Core Features** (65% unique):
- Team collaboration (multi-user accounts, shared prompts)
- Git-style version control (branches, commits, merges)
- Loop composition DSL (custom parser for chains)
- Recursive loops (user-facing refinement)
- Rich terminal UI
- SQLite database (prompts, evaluations, analytics)

**Usage**:
```bash
# Install
pip install -e promptly/

# Use CLI
promptly add sql-opt "Optimize: {query}"
promptly loop refine sql-opt --iterations=5
promptly analytics sql-opt
promptly share sql-opt --team=backend
```

**Integration with HoloLoom**:
- Promptly stores results in HoloLoom via `hololoom_bridge.py`
- HoloLoom can trigger Promptly loops for refinement
- Shared LLM configuration (Ollama, Anthropic, OpenAI)

---

## Migration Steps (Detailed)

### Step 1: Move Files from Archive (Day 1)

```bash
# Create Promptly directory at root
mkdir -p promptly/

# Move core Promptly files
mv archive/old_projects/Promptly/promptly/* promptly/

# Keep demos and docs
mv archive/old_projects/Promptly/demos/ promptly/demos/
mv archive/old_projects/Promptly/docs/ promptly/docs/

# Keep tests
mv archive/old_projects/Promptly/tests/ promptly/tests/

# Root level files
mv archive/old_projects/Promptly/README.md promptly/
mv archive/old_projects/Promptly/requirements.txt promptly/
mv archive/old_projects/Promptly/QUICK_TEST.py promptly/
```

### Step 2: Update Dependencies (Day 1)

```bash
# Install Promptly dependencies
cd promptly/
pip install -r requirements.txt

# Verify all imports work
python QUICK_TEST.py
```

### Step 3: Feature-by-Feature Integration (Week 1-3)

Follow Phase 1-3 plan above, integrating one feature at a time.

**For each feature**:
1. Copy files to HoloLoom
2. Update imports
3. Write integration tests
4. Update documentation
5. Create demo

### Step 4: Documentation Updates (Week 3)

**Update CLAUDE.md**:
- Add section: "Promptly Integration"
- Document MCP tools (27 tools)
- Document skills system (13 templates)
- Document evaluation tools (A/B testing, LLM-as-judge)
- Document web dashboard integration

**Create new docs**:
- `HoloLoom/mcp_tools/README.md` (MCP tools reference)
- `HoloLoom/agentic/skills/README.md` (Skills guide)
- `HoloLoom/evaluation/README.md` (Evaluation tools)

**Update existing docs**:
- `VISUAL_QUICK_START.md` (add Promptly features)
- `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md` (add Phase 6: Promptly Integration)

### Step 5: Testing (Week 3)

**Unit Tests**:
```bash
pytest HoloLoom/mcp_tools/tests/ -v
pytest HoloLoom/agentic/skills/tests/ -v
pytest HoloLoom/evaluation/tests/ -v
```

**Integration Tests**:
```bash
pytest HoloLoom/tests/integration/test_promptly_integration.py -v
```

**End-to-End Tests**:
```bash
# Test HoloLoom with Promptly features
PYTHONPATH=. python demos/demo_promptly_integration.py

# Test Promptly CLI still works
cd promptly/
python QUICK_TEST.py
```

---

## Dependency Management

### HoloLoom Dependencies (Add to requirements.txt)

```txt
# Promptly Integration
flask>=3.0.0
flask-socketio>=5.3.0
eventlet>=0.33.0
python-socketio>=5.10.0
```

### Promptly Dependencies (Keep separate)

Promptly keeps its own `requirements.txt` for CLI usage.

### Shared Dependencies

Both systems share:
- `ollama` (LLM client)
- `anthropic` (Claude API)
- `openai` (OpenAI API)
- `rich` (Terminal UI)
- `pyyaml` (Config)

---

## Timeline Estimate

### Option B (Selective Integration) - 2-3 Weeks

**Week 1: Core Features**
- Day 1-3: MCP tools integration (800 lines)
- Day 4-5: Skills system integration (600 lines)

**Week 2: Evaluation Tools**
- Day 1-2: A/B testing framework (450 lines)
- Day 3-4: LLM-as-judge enhanced (600 lines)
- Day 5: Cost tracking (400 lines)

**Week 3: Web Dashboard & Testing**
- Day 1-3: Dashboard integration (500 lines)
- Day 4: Analytics bridge
- Day 5: Integration testing, documentation

**Total Integration**: ~2,950 lines moved to HoloLoom
**Promptly Remaining**: ~14,000 lines (CLI, teams, version control)

---

## Success Criteria

### Phase 1 Success
- ✅ 27 MCP tools working in HoloLoom
- ✅ 13 skill templates loadable
- ✅ Skills can be executed from agentic system
- ✅ Tests passing

### Phase 2 Success
- ✅ A/B testing compares strategies correctly
- ✅ LLM-as-judge evaluates responses
- ✅ Cost tracking integrated with orchestrator
- ✅ Tests passing

### Phase 3 Success
- ✅ Web dashboard shows HoloLoom metrics
- ✅ Real-time WebSocket updates working
- ✅ Analytics bridge exports data correctly
- ✅ All integration tests passing

### Overall Success
- ✅ HoloLoom has 5 new capabilities (MCP tools, skills, A/B testing, LLM-as-judge, dashboard)
- ✅ Promptly CLI still functional for users
- ✅ No regressions in HoloLoom
- ✅ Documentation updated
- ✅ All tests passing (unit + integration + e2e)

---

## Risk Mitigation

### Risk 1: Breaking HoloLoom
**Mitigation**:
- Integrate one feature at a time
- Run full test suite after each integration
- Keep features in separate modules (`mcp_tools/`, `evaluation/`)
- Use feature flags to disable if issues

### Risk 2: Promptly CLI Breaks
**Mitigation**:
- Keep Promptly CLI separate (not in HoloLoom)
- Test `QUICK_TEST.py` after each integration
- Maintain separate requirements.txt
- Don't modify core Promptly files during integration

### Risk 3: Documentation Drift
**Mitigation**:
- Update docs immediately after each feature
- Create integration checklist
- Add examples to demos/
- Update CLAUDE.md with every change

### Risk 4: Dependency Conflicts
**Mitigation**:
- Flask/SocketIO only used for web dashboard
- SQLite doesn't conflict with Neo4j/Qdrant
- Keep Promptly's SQLite database separate
- No shared global state

---

## Open Questions

1. **Deployment Strategy**:
   - Should Promptly CLI be installable via pip?
   - Should we create a separate PyPI package for Promptly?
   - Or keep it as part of mythRL repo?

2. **Web Dashboard**:
   - Merge with existing `HoloLoom/web_dashboard/` or keep separate?
   - Use same Flask app or separate server?
   - Same port or different port?

3. **MCP Tools**:
   - Should all 27 tools be enabled by default?
   - Or opt-in via configuration?
   - How to handle tool conflicts with existing MCP ingestion?

4. **Skills System**:
   - Should skills be stored in database or file system?
   - How to handle skill versioning?
   - Integration with existing agentic reasoning modes?

5. **Marketing/Positioning**:
   - Is Promptly a product or internal tool?
   - Should we open-source Promptly separately?
   - Target audience: prompt engineers, developers, both?

---

## Next Steps

1. **Get Approval**: User confirms Option B approach
2. **Create GitHub Issue**: Track integration work
3. **Start Phase 1**: MCP tools + Skills (Week 1)
4. **Weekly Check-ins**: Review progress, adjust plan
5. **Final Review**: Week 3, test everything, update docs

---

## Conclusion

Promptly represents **17,000 lines of production-ready code** with **65% unique functionality** that complements HoloLoom perfectly:

- **HoloLoom**: Agent-facing reasoning, memory, learning
- **Promptly**: User-facing prompt engineering, teams, analytics

**Recommended Strategy**: **Option B (Selective Integration)** provides the best ROI by integrating valuable features (MCP, skills, evaluation, dashboard) into HoloLoom while preserving Promptly's unique user-facing capabilities.

**Timeline**: 2-3 weeks for complete integration
**Risk**: Low (incremental integration with testing)
**Value**: High (5 new HoloLoom capabilities + preserved Promptly CLI)

---

**Ready to proceed with Option B?**
