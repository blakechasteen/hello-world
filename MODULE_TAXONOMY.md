# HoloLoom Module Taxonomy

**Date**: 2026-02-25
**Purpose**: Classify every directory into Core Infrastructure, Extension, App, or Tooling
**Goal**: Establish clear boundaries so the repo communicates its own architecture

---

## The Principle

> HoloLoom Core is infrastructure that adds reasoning, recursion, and structured
> memory to LLMs and Agents. Everything else is either an extension to that
> infrastructure, an app built on top of it, or tooling for development.

---

## Layer 0: Protocols & Types (foundation everything imports)

These have **zero HoloLoom dependencies**. They define the contracts.

| Directory | LOC | What It Is |
|-----------|-----|-----------|
| `protocols/` | 3,546 | Type definitions, interfaces, shared data types |
| `utils/` | 216 | General utilities |

---

## Layer 1: Core Infrastructure (the product)

These are the building blocks that make HoloLoom *HoloLoom*. An app built on
HoloLoom would use these. Without any one of them, the system is fundamentally
incomplete.

| Directory | LOC | What It Is | Why Core |
|-----------|-----|-----------|----------|
| `memory/` | 42,362 | Knowledge graph, vector store, awareness graph, caching | You can't reason without memory |
| `embedding/` | 3,546 | Matryoshka multi-scale embeddings | Everything needs vector representations |
| `policy/` | 3,185 | Thompson Sampling decision engine | Core decision-making loop |
| `convergence/` | 4,097 | Probability collapse → discrete actions | How decisions become actions |
| `orchestrator/` | 8,788 | 9-step weaving cycle | The central pipeline |
| `warp/` | 36,949 | Tensioned tensor manifold | Continuous math on discrete memory |
| `fabric/` | 1,546 | Spacetime output with provenance | Structured output format |
| `chrono/` | 672 | Temporal windows and timing | Time-awareness |
| `resonance/` | ~2,000 | Feature extraction (DotPlasma creation) | Feature fusion |
| `loom/` | 9,054 | Pattern card selection (BARE/FAST/FUSED) | Execution mode control |
| `recursive/` | 6,972 | 5-phase self-improving learning loops | Core differentiator |
| `reflection/` | 3,474 | Episodic buffer, PPO training | Learning from outcomes |
| `config.py` (root) | ~460 | BARE/FAST/FUSED configuration | System configuration |

**Subtotal: ~122,000 LOC** — This is the product.

---

## Layer 2: Core Extensions (optional power-ups to core)

These extend core capabilities but are **not required** for the basic pipeline to
work. An app could use HoloLoom without any of these.

| Directory | LOC | What It Is | Why Extension, Not Core |
|-----------|-----|-----------|------------------------|
| `agentic/` | 25,447 | Multi-step reasoning with verification | Core works with single-step |
| `alignment/` | 23,763 | Safety guardrails, audit trail | Core works without safety gates |
| `rag/` | 14,795 | Level 4 agentic RAG | Core has memory; RAG is a pattern on top |
| `context_packing/` | 6,844 | Beta wave context optimization | Optimization, not requirement |
| `routing/` | 8,754 | Smart query complexity classification | Core works without routing |
| `semantic_calculus/` | 12,501 | 244 interpretable semantic axes | Advanced analysis layer |
| `dark_trace/` | 66,315 | SAE interpretability suite | Observability, not execution |
| `prompting/` | 17,460 | MRF metaprompting framework | Prompt quality, not core pipeline |
| `search/` | 4,682 | Vector + BM25 hybrid search | Alternative retrieval method |
| `bandits/` | 4,285 | Extended exploration strategies | Core has Thompson; this extends |
| `context/` | 6,006 | Production hardening (circuit breakers, rate limits) | Operational, not functional |
| `memory_symphony/` | ~1,700 | Multi-system memory coordination | Optimization layer |
| `spinningWheel/` | 29,020 | 47 input adapters | Data ingestion, not reasoning |
| `integrations/` | 6,129 | LangChain bridge | Third-party integration |
| `chaining/` | 5,045 | 17 chain patterns | Composition patterns |
| `physics/` | 4,053 | Helmholtz free energy optimization | Advanced math backend |
| `planning/` | 3,535 | POMDP planning under uncertainty | Advanced planning |
| `causal/` | 3,150 | Pearl's do-calculus | Advanced reasoning |
| `explainability/` | ~3,000 | 7 XAI techniques | Observability |
| `reasoning/` | 2,631 | Multi-modal reasoning | Extended reasoning |
| `verification/` | ~2,000 | Chain of verification | Quality checking |

**Subtotal: ~250,000 LOC** — Power-ups. Should be independently installable.

---

## Layer 3: Apps (built ON HoloLoom, not part of it)

These are end-user-facing applications that consume HoloLoom as infrastructure.
**They should not live inside the `HoloLoom/` package.**

| Current Location | LOC | What It Is | Should Be |
|-----------------|-----|-----------|-----------|
| `apps/elle/` | 30,948 | AR guide companion | `apps/elle/` ✅ already correct |
| `HoloLoom/visualization/` (Jenny) | 25,925 | Adaptive viz runtime | `apps/jenny/` |
| `apps/trough/` + `apps/xterminator/` | 24,686 | Code quality QA system | `apps/trough/` ✅ already correct |
| `HoloLoom/chatops/` | 44,221 | Matrix.org chatbot | `apps/chatops/` |
| `HoloLoom/web_dashboard/` | 23,029 | Visual workflow builder | `apps/workflow_builder/` |
| `HoloLoom/departments/` | 38,131 | Multi-department B2B system | `apps/departments/` |
| `HoloLoom/redteam/` | 40,821 | Adversarial red-teaming | `apps/redteam/` |
| `apps/bosspig/` | 5,510 | Business doc slop detector | `apps/bosspig/` ✅ already correct |
| `apps/sous/` | 19,189 | Kitchen management AI | `apps/sous/` ✅ already correct |
| `HoloLoom/server/` | 22,804 | FastAPI server | `apps/server/` |
| `HoloLoom/collaboration/` | 7,616 | Multi-user workspaces | `apps/collaboration/` |
| `HoloLoom/dreamweaving/` | ~5,000 | Creative world building | `apps/dreamweaving/` |
| `HoloLoom/spatial/` | 16,102 | WebXR AR/VR integration | `apps/spatial/` |
| `HoloLoom/voice/` | 16,759 | STT/TTS voice commands | `apps/voice/` |
| `HoloLoom/vision/` | ~8,000 | YOLO, MiDaS, SLAM | `apps/vision/` |
| `HoloLoom/thirdeye/` | 13,883 | Scene understanding viz | `apps/thirdeye/` |
| `HoloLoom/federation/` | 20,379 | Distributed gossip + DHT | `apps/federation/` |
| `HoloLoom/eggroll/` | 4,561 | Distributed evolution | `apps/eggroll/` |

**Subtotal: ~370,000 LOC** — This is more than half the codebase.

**Key insight**: The majority of HoloLoom's code is apps, not infrastructure.

---

## Layer 4: Tooling & DevOps (supports development)

| Current Location | LOC | What It Is | Should Be |
|-----------------|-----|-----------|-----------|
| `HoloLoom/tests/` | 110,394 | Test suite | `tests/` (root) |
| `HoloLoom/tools/` | 6,422 | Dev utilities | `scripts/` or `tools/` |
| `infra/` | 11,934 | Docker, K8s, monitoring | `infra/` ✅ correct |
| `demos/` | ~10,000 | Demo scripts | `demos/` ✅ correct |
| `scripts/` | ~2,000 | Build/deploy scripts | `scripts/` ✅ correct |
| `HoloLoom/skills/` | 9,603 | Claude Code skill definitions | `skills/` (root) |
| `HoloLoom/mcp_tools/` | ~3,000 | MCP server tools | `tools/mcp/` |
| `HoloLoom/telemetry/` | ~2,000 | Metrics collection | Part of core or infra |
| `HoloLoom/tuning/` | ~1,500 | Hyperparameter tuning | `tools/tuning/` |
| `HoloLoom/performance/` | ~3,000 | Profiling & benchmarks | `tools/performance/` |
| `HoloLoom/datapig/` | 2,159 | Data quality assurance | `tools/datapig/` |
| `HoloLoom/cve/` | 3,928 | Vulnerability assessment | `tools/security/` |

---

## Unclear / Needs Discussion

These could go either way:

| Directory | LOC | Question |
|-----------|-----|---------|
| `HoloLoom/agents/` | 8,938 | Core extension or app? (MCTS multi-agent) |
| `HoloLoom/handoff/` | 3,894 | Core extension or app feature? (context handoffs) |
| `HoloLoom/model_extension/` | 15,155 | Core extension or tooling? |
| `HoloLoom/portal/` | 13,894 | Part of orchestrator or separate? |
| `HoloLoom/shuttle/` | ~3,000 | Legacy compat or active? |
| `HoloLoom/tapestry/` | ~3,000 | Session management - core or app? |
| `HoloLoom/saas/` | ~5,000 | SaaS deployment layer |
| `HoloLoom/lite/` | ~2,000 | Lightweight mode |
| `HoloLoom/neural/` | ~2,000 | Neural network primitives - part of policy? |
| `HoloLoom/ml/` | ~1,500 | ML utilities - part of embedding? |
| `HoloLoom/math/` | ~1,000 | Math utilities - part of warp? |
| `HoloLoom/input/` | ~1,000 | Input processing - part of spinningWheel? |
| `HoloLoom/synthesis/` | ~1,000 | Output synthesis - part of fabric? |
| `HoloLoom/weaving/` | ~1,000 | Weaving utilities - part of orchestrator? |
| `HoloLoom/writing/` | 4,720 | NLG system |
| `HoloLoom/llm/` | ~3,000 | LLM client abstraction |
| `HoloLoom/conscience/` | ~2,000 | Ethical reasoning |
| `HoloLoom/nested/` | ~1,000 | Nested contexts |
| `HoloLoom/expansions/` | ~1,000 | Context expansion |
| `HoloLoom/clustering/` | ~1,000 | Clustering algorithms |
| `HoloLoom/safety/` | ~2,000 | Overlaps with alignment? |
| `HoloLoom/promptly/` | ~2,000 | Overlaps with prompting? |
| `HoloLoom/ts_core/` | ~1,000 | TypeScript core? |
| `HoloLoom/lsp/` | ~1,000 | Language Server Protocol |
| `HoloLoom/workflows/` | ~1,000 | Workflow definitions |
| `holoLoom/` (root) | 16,742 | Duplicate package - merge into HoloLoom/ |

---

## The Numbers Tell the Story

| Layer | LOC | % of Codebase | Directory Count |
|-------|-----|--------------|-----------------|
| **Core Infrastructure** | ~122K | 13% | 13 dirs |
| **Core Extensions** | ~250K | 27% | 21 dirs |
| **Apps** | ~370K | 40% | 18 dirs |
| **Tooling & Tests** | ~150K | 16% | 12 dirs |
| **Unclear** | ~40K | 4% | 25 dirs |

**40% of the codebase is apps masquerading as infrastructure.**

That's why the repo feels unfocused — the core product is 13% of the code,
buried under layers of applications and extensions that all live at the same
directory level.

---

## Proposed Target Structure

```
hololoom/                          ← Renamed, lowercase (PEP 8)
├── core/                          ← Layer 1: The Product (~122K LOC)
│   ├── memory/
│   ├── embedding/
│   ├── policy/
│   ├── orchestrator/              ← Consolidate 6 files here
│   ├── warp/
│   ├── convergence/
│   ├── fabric/
│   ├── chrono/
│   ├── resonance/
│   ├── loom/
│   ├── recursive/
│   ├── reflection/
│   └── protocols/
│
├── extensions/                    ← Layer 2: Optional Power-ups (~250K LOC)
│   ├── agentic/
│   ├── alignment/
│   ├── rag/
│   ├── dark_trace/
│   ├── routing/
│   ├── semantic_calculus/
│   ├── context_packing/
│   ├── prompting/
│   ├── search/
│   ├── spinningWheel/
│   └── ...
│
apps/                              ← Layer 3: Separate from package
├── elle/
├── jenny/
├── trough/
├── chatops/
├── departments/
├── workflow_builder/
├── server/                        ← Or this could be top-level
└── ...

tools/                             ← Layer 4: Development support
├── scripts/
├── performance/
├── tuning/
└── security/

tests/                             ← All tests at root
├── unit/
├── integration/
└── e2e/

docs/                              ← Documentation
infra/                             ← Docker, K8s, monitoring
demos/                             ← Examples
```

This structure makes the architecture **self-documenting**. A new developer
immediately understands:
- `core/` is the product
- `extensions/` are optional add-ons
- `apps/` are things built with the product
- Everything else is support

---

## Migration Path

This doesn't have to happen all at once. Suggested order:

1. **Move apps out of HoloLoom/** (biggest clarity win, least breakage)
2. **Consolidate the "unclear" micro-modules** into their parent systems
3. **Split core/ vs extensions/** inside HoloLoom/
4. **Rename to lowercase** (breaking change, do last)
