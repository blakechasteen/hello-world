# HoloLoom Quick Reference

> **"Kubernetes for AI Agents"** | v1.0 | Updated: 2025-12-15

---

## Three Pillars

| Pillar | What It Does |
|--------|--------------|
| **Amplifier** | 100x cache speedup, 40-90% token savings |
| **Governance** | Audit trails, budgets, safety guardrails |
| **Memory** | Persistent state that compounds over time |

---

## CLI Commands

### Query
```bash
hololoom query "What is X?"                    # Simple query (FAST mode)
hololoom query --mode research "Compare X and Y"  # Deep research
hololoom query --mode verify "Claim to verify"    # Verify a claim
```

### Agent Management
```bash
hololoom agent list                    # List available agents
hololoom agent run --workflow research "Topic"  # Run workflow
hololoom agent status                  # Check status
hololoom agent logs --limit 10         # View logs
```

### Cluster (Distributed)
```bash
hololoom cluster status    # Cluster health
hololoom cluster nodes     # List nodes
```

---

## Processing Modes

| Mode | Latency | Use Case |
|------|---------|----------|
| **BARE** | <50ms | Simple lookups |
| **FAST** | <150ms | Standard queries |
| **FUSED** | <300ms | Complex reasoning |
| **RESEARCH** | No limit | Deep exploration |

---

## Weaving Vocabulary

| HoloLoom Term | Technical Meaning |
|---------------|-------------------|
| **Shuttle** | Orchestrator |
| **Yarn Graph** | Knowledge Graph |
| **Loom Command** | Pattern Selection |
| **Chrono Trigger** | Temporal Windows |
| **Resonance Shed** | Feature Extraction |
| **DotPlasma** | Feature Tensor |
| **Warp Space** | Continuous Manifold |
| **Convergence Engine** | Decision Collapse |
| **Spacetime** | Final Output |

---

## Key Files

### Architecture
| File | Purpose |
|------|---------|
| `hololoom/weaving_orchestrator.py` | Main orchestrator (3,476 lines) |
| `hololoom/policy/unified.py` | Neural policy + Thompson Sampling |
| `hololoom/memory/unified.py` | Unified memory interface |
| `hololoom/config.py` | Configuration modes |

### Documentation
| File | Purpose |
|------|---------|
| `CLAUDE.md` | Developer reference |
| `docs/HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md` | Complete architectural map |
| `docs/architecture/ARCHITECTURE_VISUAL_MAP.md` | Visual diagrams |

---

## Governance Features

| Feature | Description | Overhead |
|---------|-------------|----------|
| Safety Guardrails | Risk-based action gating | 0.039ms |
| Deception Detection | Goal transparency tracking | 0.034ms |
| Audit Trail | Complete decision provenance | 0.015ms |
| Token Budgets | Per-agent resource limits | <1ms |

**Total alignment overhead: 0.103ms** (29x faster than target)

---

## Performance Specs

| Metric | Value |
|--------|-------|
| Query Cache | **100x speedup** |
| Token Savings | **40-90%** |
| Alignment Overhead | **0.103ms** |
| First Token (Phase 4) | **<100ms** |

---

## Setup Commands

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/verify_setup.py --level quick

# Start backends (optional, for production)
docker-compose up -d
```

---

## Server Commands

```bash
# Start Agentic API (queries, monitoring) on port 8000
uvicorn hololoom.server.agentic_api:app --port 8000

# Start Agent Manager (threads, swarms) on port 8002
uvicorn hololoom.server.agent_manager_api:app --port 8002
```

---

## Testing

```bash
pytest hololoom/tests/unit/ -v          # Unit (<5s)
pytest hololoom/tests/integration/ -v   # Integration (<30s)
pytest hololoom/tests/e2e/ -v           # End-to-end (<2min)
```

---

## Demo Commands

```bash
PYTHONPATH=. python demos/demo_agent_hypervisor.py   # Agent Hypervisor
PYTHONPATH=. python demos/demo_rag_qa_simple.py      # RAG Q&A
PYTHONPATH=. python demos/demo_multipass_simple.py   # Multi-pass refinement
```

---

## Competitive Positioning

| vs | HoloLoom Advantage |
|----|-------------------|
| **CrewAI** | Federation + MCTS vs basic crews |
| **LangGraph** | Full provenance vs graph-only |
| **AutoGPT** | Multi-agent + governance vs single agent |

---

## Quick Links

- **Issues**: https://github.com/yourusername/hololoom/issues
- **Docs**: `docs/`
- **Demos**: `demos/`
- **Developer Reference**: `CLAUDE.md`

---

*Generated from `docs/quick_ref.yaml` | Regenerate: `python scripts/generate_quick_ref.py`*
