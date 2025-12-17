---
id: doc-onboarding-checklist
version: "1.0"
status: production
created: 2025-12-15
last-updated: 2025-12-15
audience:
  - role: [business-data, development, architect, researcher]
  - level: [beginner, intermediate, expert]
topics: [onboarding, setup, getting-started]
dependencies: []
teaches: [cli-basics, architecture-overview, first-workflow, governance-features]
duration-minutes: 180
next-review: 2026-03-15
---

# Team Onboarding Checklist

> **"Kubernetes for AI Agents"** - HoloLoom orchestrates multi-agent, multi-model AI workflows with enterprise-grade infrastructure.

## How to Use This Document

This checklist provides structured learning paths for new team members. Follow the path that matches your role:

| Your Role | Follow These Tags | Skip These Tags |
|-----------|------------------|-----------------|
| **Business Data** | `[ALL]`, `[BIZ]` | `[DEV]` |
| **Development** | `[ALL]`, `[DEV]` | `[BIZ]` |
| **Both/Architect** | All tags | None |

**Time Investment**: ~3 hours for Foundation (Days 1-3), then ongoing exploration.

---

## Phase 1: Foundation (Days 1-3)

### [ALL] Conceptual Understanding

**Day 1 - What is HoloLoom?** (~2 hours)

| Time | Activity | Resource | Outcome |
|------|----------|----------|---------|
| 30 min | Read positioning document | [docs/AGENT_HYPERVISOR.md](../AGENT_HYPERVISOR.md) | Explain "Kubernetes for AI Agents" |
| 30 min | Watch agent hypervisor demo | `PYTHONPATH=. python demos/demo_agent_hypervisor.py` | See 3 agents + 2 models in action |
| 30 min | Understand the 3 pillars | [CLAUDE.md](../../CLAUDE.md) (Overview section) | Articulate Amplifier + Governance + Memory |
| 30 min | Try the CLI | `python -m HoloLoom.cli agent list` | Run your first HoloLoom command |

**Checkpoint**: Can you explain to a colleague what HoloLoom does and why it matters?

<details>
<summary><strong>Intermediate Depth</strong> (15 min per task)</summary>

> **Why this works**: HoloLoom positions as infrastructure, not a framework. This is crucial because:
> - Frameworks get replaced (jQuery → React → Next.js)
> - Infrastructure persists (Linux, Kubernetes, PostgreSQL)
> - Enterprise needs governance, not just orchestration
>
> The "3 pillars" (Amplifier, Governance, Memory) each solve problems that foundation models don't:
> - **Amplifier**: 100x cache speedup, 40-90% token savings
> - **Governance**: Audit trails, budgets, safety guardrails
> - **Memory**: Persistent state that compounds over time

</details>

<details>
<summary><strong>Expert Depth</strong> (30 min)</summary>

> **Deep dive**: Compare HoloLoom's positioning to competitors:
>
> | Feature | CrewAI | LangGraph | AutoGPT | **HoloLoom** |
> |---------|--------|-----------|---------|--------------|
> | Multi-agent | Basic crews | Graphs | Single agent | **Federation + MCTS** |
> | Token budgets | No | No | No | **Per-agent limits** |
> | Audit trails | No | No | No | **Full provenance** |
> | Self-improving | No | No | No | **Thompson Sampling** |
>
> Read: [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md) for complete architectural context.

</details>

---

### [BIZ] Data Flow Understanding

**Day 2 - How Data Moves** (~3 hours)

| Time | Activity | Resource | Outcome |
|------|----------|----------|---------|
| 1 hr | RAG system overview | [HoloLoom/rag/README.md](../../HoloLoom/rag/README.md) | Understand 4-level RAG hierarchy |
| 1 hr | Memory architecture | [docs/architecture/ARCHITECTURE_VISUAL_MAP.md](../architecture/ARCHITECTURE_VISUAL_MAP.md) | Trace data through 9-step cycle |
| 1 hr | Run RAG demo | `PYTHONPATH=. python demos/demo_rag_qa_simple.py` | Ingest and query data |

**Checkpoint**: Can you explain how a user query flows from input to response?

<details>
<summary><strong>Intermediate Depth</strong></summary>

> **9-Step Weaving Cycle**:
> ```
> Query → Loom Command → Chrono Trigger → Yarn Graph → Resonance Shed
>       → DotPlasma → Warp Space → Convergence Engine → Spacetime
> ```
>
> Each step has a metaphorical name that maps to technical function:
> - **Loom Command**: Pattern selection (BARE/FAST/FUSED)
> - **Chrono Trigger**: Temporal windowing
> - **Yarn Graph**: Knowledge graph (NetworkX MultiDiGraph)
> - **Resonance Shed**: Feature extraction
> - **DotPlasma**: Feature tensor
> - **Warp Space**: Continuous manifold
> - **Convergence Engine**: Decision collapse
> - **Spacetime**: Final output with provenance

</details>

---

### [DEV] Architecture Deep Dive

**Day 2 - Technical Architecture** (~3 hours)

| Time | Activity | Resource | Outcome |
|------|----------|----------|---------|
| 1 hr | 9-step weaving cycle | [CLAUDE.md](../../CLAUDE.md) (Weaving Architecture) | Understand full pipeline |
| 1 hr | Step through orchestrator | Debug `WeavingOrchestrator.weave()` | Trace a query through code |
| 1 hr | Explore protocols | [HoloLoom/protocols/](../../HoloLoom/protocols/) | Understand protocol-based design |

**Checkpoint**: Can you trace a query through the full pipeline in the debugger?

<details>
<summary><strong>Intermediate Depth</strong></summary>

> **Key Files to Understand**:
> - `weaving_orchestrator.py` (3,476 lines) - Main orchestrator
> - `policy/unified.py` (1,247 lines) - Neural policy + Thompson Sampling
> - `memory/unified.py` - Unified memory interface
> - `config.py` - Configuration modes (BARE/FAST/FUSED)
>
> **Protocol Pattern**:
> ```python
> class PolicyEngine(Protocol):
>     def forward(self, features: Features, context: Context) -> ActionPlan: ...
> ```
> All major components use protocols for swappable implementations.

</details>

<details>
<summary><strong>Expert Depth</strong></summary>

> **Portal Orchestration Stages** (December 2025):
>
> The weaving cycle is modularized into pure function stages:
> - `steps_0_3.py` (349 lines) - Query setup, thread selection
> - `steps_4_6.py` (673 lines) - Parallel feature extraction
> - `steps_7_9.py` (514 lines) - Convergence, execution, output
>
> Steps 4-6 run in **parallel** via `asyncio.gather` for 40-120ms speedup.
>
> Read: [HoloLoom/orchestrator/stages/](../../HoloLoom/orchestrator/stages/) for implementation.

</details>

---

### [ALL] Environment Setup

**Day 3 - Get Running** (~2 hours)

| Time | Activity | Resource | Outcome |
|------|----------|----------|---------|
| 30 min | Clone and setup | [README.md](../../README.md) | Development environment working |
| 30 min | Run test suite | `pytest HoloLoom/tests/ -v` | All tests passing |
| 30 min | Explore codebase | `python -m HoloLoom.cli agent list` | Navigate confidently |
| 30 min | Make first edit | Fix a typo or add a comment | First commit to repo |

**Setup Commands**:
```bash
# Clone repository
git clone <repo-url>
cd mythRL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/verify_setup.py --level quick
```

**Checkpoint**: Can you run `pytest HoloLoom/tests/unit/ -v` with all tests passing?

---

### [ALL] Governance & Compliance

**Day 3 - Enterprise Features** (~2 hours)

| Time | Activity | Resource | Outcome |
|------|----------|----------|---------|
| 1 hr | Audit trail system | [HoloLoom/alignment/audit_trail.py](../../HoloLoom/alignment/audit_trail.py) | Understand provenance tracking |
| 30 min | Safety guardrails | [HoloLoom/alignment/README.md](../../HoloLoom/alignment/README.md) | Understand risk gating |
| 30 min | Run governance demo | Part of `demo_agent_hypervisor.py` | See safety in action |

**Checkpoint**: Can you explain HoloLoom's compliance features to an enterprise client?

<details>
<summary><strong>Key Governance Features</strong></summary>

> **4-Layer Alignment Framework** (0.103ms overhead):
> 1. **Safety Guardrails** - Risk-based action gating (LOW/MEDIUM/HIGH/CRITICAL)
> 2. **Deception Detection** - Goal transparency tracking
> 3. **Instrumental Convergence** - Power-seeking detection
> 4. **Audit Trail** - Complete decision provenance
>
> **Enterprise Requirements Met**:
> - GDPR: Full data provenance
> - HIPAA: Audit trails for healthcare
> - SOC2: Complete decision logging
> - Token budgets: Per-agent resource limits

</details>

---

## Phase 2: Exploration (Week 1-2)

### [BIZ] Query & Retrieval

- [ ] Run 10+ queries through CLI (`hololoom query "..."`)
- [ ] Compare FAST vs FUSED vs RESEARCH modes
- [ ] Document typical latencies and token usage
- [ ] Understand when to use each mode

**Mode Comparison**:
| Mode | Latency | Token Usage | Use Case |
|------|---------|-------------|----------|
| BARE | <50ms | Minimal | Simple lookups |
| FAST | <150ms | Moderate | Standard queries |
| FUSED | <300ms | Full | Complex reasoning |
| RESEARCH | No limit | Maximum | Deep exploration |

### [BIZ] Data Ingestion

- [ ] Ingest 3 different data sources (PDF, web, text)
- [ ] Use SpinningWheel adapters (see [HoloLoom/spinningWheel/](../../HoloLoom/spinningWheel/))
- [ ] Monitor memory graph growth
- [ ] Create data quality report

### [DEV] Testing & Quality

- [ ] Run unit tests (`HoloLoom/tests/unit/`)
- [ ] Run integration tests (`HoloLoom/tests/integration/`)
- [ ] Add test for one new edge case
- [ ] Review code coverage

### [DEV] Extension Points

- [ ] Create a new SpinningWheel adapter (inherit from `BaseSpinner`)
- [ ] Add a custom agent type
- [ ] Modify a chaining workflow pattern
- [ ] Understand extension patterns

---

## Phase 3: Application (Week 3-4)

### [BIZ] Business Use Cases

- [ ] Design 2 business workflows using Visual Workflow Builder
- [ ] Calculate ROI based on token savings (100x cache speedup)
- [ ] Document cost comparison (with vs without caching)

### [BIZ] Reporting & Metrics

- [ ] Set up Prometheus metrics dashboard
- [ ] Create weekly usage report template
- [ ] Document KPIs for stakeholder presentations

### [DEV] Feature Development

- [ ] Pick a feature from backlog
- [ ] Design using ADR format (Architecture Decision Record)
- [ ] Implement with tests
- [ ] Code review and merge

### [DEV] Production Deployment

- [ ] Set up Docker environment (`docker-compose up -d`)
- [ ] Configure Kubernetes manifests (see `k8s/`)
- [ ] Implement health checks
- [ ] Document deployment process

---

## Phase 4: Mastery (Month 2+)

### [BIZ] Leadership

- [ ] Train other team members on basics
- [ ] Create department-specific workflows
- [ ] Optimize token budgets for cost efficiency
- [ ] Contribute to documentation improvements

### [DEV] Ownership

- [ ] Own a subsystem (memory, policy, or chaining)
- [ ] Review PRs from other developers
- [ ] Contribute to Phase 6 (HAL production deployment)
- [ ] Mentor new team members

---

## Cross-Team Milestones

### Week 1 Checkpoint

Both teams should be able to:
- [ ] Run `hololoom agent list` successfully
- [ ] Execute a demo script end-to-end
- [ ] Explain the "3 pillars" (Amplifier, Governance, Memory)

### Week 2 Checkpoint

Both teams should be able to:
- [ ] Run a multi-agent workflow
- [ ] View audit trail logs
- [ ] Explain competitive positioning vs CrewAI/LangGraph

### Week 4 Checkpoint

Both teams should be able to:
- [ ] Create a new workflow using their skills
- [ ] Present a 5-minute demo to stakeholders
- [ ] Identify one area for improvement

### Month 2 Checkpoint

Both teams should be able to:
- [ ] Independently solve problems in their domain
- [ ] Train new team members on basics
- [ ] Contribute documentation improvements

---

## Verification

Run the setup verification script to confirm your environment:

```bash
# Quick verification (2-3 min, essential only)
python scripts/verify_setup.py --level quick

# Standard verification (5-10 min, all core checks)
python scripts/verify_setup.py --level standard

# Comprehensive verification (15+ min, edge cases)
python scripts/verify_setup.py --level comprehensive
```

See [scripts/verify_setup.py](../../scripts/verify_setup.py) for implementation.

---

## Glossary: HoloLoom Vocabulary

| Technical Concept | HoloLoom Name | Metaphor |
|-------------------|---------------|----------|
| Orchestrator | **Shuttle** | Loom shuttle carries thread |
| Knowledge Graph | **Yarn Graph** | Threads of knowledge |
| Pattern Selection | **Loom Command** | Pattern card on loom |
| Temporal Windows | **Chrono Trigger** | Time-bound weaving |
| Feature Extraction | **Resonance Shed** | Interference zone |
| Feature Tensor | **DotPlasma** | Flowing feature fluid |
| Continuous Manifold | **Warp Space** | Tensioned fabric |
| Decision Collapse | **Convergence Engine** | Probability to choice |
| Final Output | **Spacetime** | 4D woven fabric |

---

## Quick Reference

**Essential CLI Commands**:
```bash
# Query
hololoom query "What is Thompson Sampling?"
hololoom query --mode research "Compare bandit algorithms"

# Agent management
hololoom agent list
hololoom agent run --workflow research "Analyze topic"
hololoom agent status
hololoom agent logs --limit 10

# Cluster (distributed mode)
hololoom cluster status
hololoom cluster nodes
```

**Key Documentation**:
- Architecture: [ARCHITECTURE_VISUAL_MAP.md](../architecture/ARCHITECTURE_VISUAL_MAP.md)
- RAG System: [HoloLoom/rag/README.md](../../HoloLoom/rag/README.md)
- Alignment: [HoloLoom/alignment/README.md](../../HoloLoom/alignment/README.md)
- Full Reference: [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)

---

## Feedback

Found something unclear? Have suggestions?

1. Open an issue: `https://github.com/yourusername/hololoom/issues`
2. Update this document with improvements
3. Share feedback with the team

**Last Updated**: 2025-12-15
**Next Review**: 2026-03-15
