# HoloLoom Module Taxonomy

**Date**: 2026-02-25
**Updated**: 2026-02-26 — Refined target structure (dropped `extensions/` wrapper)
**Updated**: 2026-04-02 — Audit against actual codebase: resolved unclear modules, added new core/optional modules
**Purpose**: Classify every directory into Core, Optional Module, App, or Tooling
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
| `bus/` | 3,822 | Unified event/message bus | System coordination *(added 2026-04)* |
| `deep_thinking/` | 1,865 | Deliberation engine for complex reasoning | Extended reasoning pipeline *(added 2026-04)* |
| `ritual/` | 2,815 | Ritual grammar, hook system, runtime | Workflow lifecycle hooks *(added 2026-04)* |
| `runtime/` | 1,428 | 6-layer lifecycle management | System startup/shutdown *(added 2026-04)* |
| `config.py` (root) | ~460 | BARE/FAST/FUSED configuration | System configuration |

**Subtotal: ~132,000 LOC** — This is the product. (17 modules, up from original 13.)

---

## Layer 2: Optional Modules (flat peers to core)

Everything that isn't core. These extend capabilities but the basic weaving
pipeline works without any of them. They don't import each other — each is
independently installable.

No `extensions/` wrapper. The grouping lives in `pyproject.toml` extras and
documentation, not the directory tree.

| Directory | LOC | What It Is | Why Not Core |
|-----------|-----|-----------|--------------|
| `agentic/` | 25,447 | Multi-step reasoning | Core works with single-step |
| `alignment/` | 23,763 | Safety guardrails, audit trail | Core works without safety gates |
| `rag/` | 14,795 | Level 4 agentic RAG | RAG is a pattern on top of memory |
| `dark_trace/` | 66,315 | SAE interpretability | Observability, not execution |
| `prompting/` | 17,460 | MRF metaprompting | Prompt quality, not core pipeline |
| `routing/` | 8,754 | Query complexity classification | Core works without routing |
| `semantic_calculus/` | 12,501 | 244 semantic axes | Advanced analysis layer |
| `context_packing/` | 6,844 | Beta wave optimization | Optimization, not requirement |
| `search/` | 4,682 | Vector + BM25 hybrid | Alternative retrieval method |
| `bandits/` | 4,285 | Extended exploration | Core has Thompson; this extends |
| `context/` | 6,006 | Circuit breakers, rate limits | Operational, not functional |
| `memory_symphony/` | ~1,700 | Multi-system coordination | Optimization layer |
| `spinningWheel/` | 29,020 | 47 input adapters | Data ingestion, not reasoning |
| `integrations/` | 6,129 | LangChain bridge | Third-party integration |
| `chaining/` | 5,045 | 17 chain patterns | Composition patterns |
| `physics/` | 4,053 | Helmholtz free energy | Advanced math backend |
| `planning/` | 3,535 | POMDP planning | Advanced planning |
| `causal/` | 3,150 | Pearl's do-calculus | Advanced reasoning |
| `explainability/` | ~3,000 | 7 XAI techniques | Observability |
| `reasoning/` | 2,631 | Multi-modal reasoning | Extended reasoning |
| `verification/` | ~2,000 | Chain of verification | Quality checking |
| `visualization/` | 25,925 | Jenny adaptive viz runtime | Implements viz protocol from core |
| `voice/` | 16,759 | STT/TTS voice commands | Input modality |
| `vision/` | ~8,000 | YOLO, MiDaS, SLAM | Input modality |
| `spatial/` | 16,102 | WebXR AR/VR | Output modality |
| `collaboration/` | 7,616 | Multi-user workspaces | Multi-user feature |
| `dreamweaving/` | ~5,000 | Creative world building | Creative feature |
| `departments/` | 38,131 | Multi-department B2B | Enterprise routing |
| `eggroll/` | 4,561 | Distributed evolution | Distributed compute |
| `federation/` | 20,379 | Gossip + DHT | Distributed infra |
| `server/` | 22,804 | FastAPI server | Deployment infra |
| `thirdeye/` | 13,883 | Scene understanding | Vision analysis |
| `redteam/` | 40,821 | Adversarial testing | Dev/testing tool |

**Subtotal: ~470,000 LOC** — Optional capabilities. Install what you need.

**Note on the previous version**: This list used to be split into "extensions"
(Layer 2) and "apps" (Layer 3). Most of what was called an "app" — voice,
vision, server, federation, redteam, visualization — is really an optional
module. The test: does it have its own users and UI? If not, it's a module.

---

## Layer 3: Apps (built ON HoloLoom, separate packages)

These have their own users, their own UIs, their own reason to exist.
**They should not live inside the `hololoom/` package.**

| Current Location | LOC | What It Is | Should Be |
|-----------------|-----|-----------|-----------|
| `apps/elle/` | 30,948 | AR guide companion | `apps/elle/` ✅ |
| `apps/trough/` + `apps/xterminator/` | 24,686 | Code quality QA | `apps/trough/` ✅ |
| `hololoom/chatops/` | 44,221 | Matrix.org chatbot | `apps/chatops/` |
| `hololoom/web_dashboard/` | 23,029 | Visual workflow builder | `apps/workflow_builder/` |
| `apps/bosspig/` | 5,510 | Business doc slop detector | `apps/bosspig/` ✅ |
| `apps/sous/` | 19,189 | Kitchen management AI | `apps/sous/` ✅ |

**Subtotal: ~150,000 LOC**

**The app test**: Does it have its own end users who don't know or care about
HoloLoom's internals? Does it have its own UI (CLI, web, chat)? Then it's an
app. ChatOps has Matrix users. Elle has AR users. Sous has kitchen staff.
These are products, not modules.

---

## Layer 4: Tooling & DevOps (supports development)

| Current Location | LOC | What It Is | Should Be |
|-----------------|-----|-----------|-----------|
| `hololoom/tests/` | 110,394 | Test suite | `tests/` (root) |
| `hololoom/tools/` | 6,422 | Dev utilities | `scripts/` or `tools/` |
| `infra/` | 11,934 | Docker, K8s, monitoring | `infra/` ✅ correct |
| `demos/` | ~10,000 | Demo scripts | `demos/` ✅ correct |
| `scripts/` | ~2,000 | Build/deploy scripts | `scripts/` ✅ correct |
| `hololoom/skills/` | 9,603 | Claude Code skill definitions | `skills/` (root) |
| `hololoom/mcp_tools/` | ~3,000 | MCP server tools | `tools/mcp/` |
| `hololoom/telemetry/` | ~2,000 | Metrics collection | Part of core or infra |
| `hololoom/tuning/` | ~1,500 | Hyperparameter tuning | `tools/tuning/` |
| `hololoom/performance/` | ~3,000 | Profiling & benchmarks | `tools/performance/` |
| `hololoom/datapig/` | 2,159 | Data quality assurance | `tools/datapig/` |
| `hololoom/cve/` | 3,928 | Vulnerability assessment | `tools/security/` |

---

## Resolved: Previously Unclear Modules (audited 2026-04-02)

### Now classified as Optional Modules (Layer 2)

| Directory | LOC | Classification | Notes |
|-----------|-----|---------------|-------|
| `hololoom/agents/` | 11,109 | Optional Module | MCTS multi-agent framework with memory learning |
| `hololoom/handoff/` | 3,868 | Optional Module | Context handoff and unified identity management |
| `hololoom/model_extension/` | 15,148 | Optional Module | Memory-augmented LLM wrappers |
| `hololoom/portal/` | 13,905 | Optional Module | Cross-platform bridge (node daemon, shuttle bot, wasm) |
| `hololoom/shuttle/` | 5,241 | Optional Module | Distributed inference client/server (active, not legacy) |
| `hololoom/tapestry/` | 3,557 | Optional Module | Session management and state persistence |
| `hololoom/saas/` | 4,711 | Optional Module | SaaS deployment layer with dashboards |
| `hololoom/lite/` | 3,414 | Optional Module | Lightweight HoloLoom mode (simplified API) |
| `hololoom/ml/` | 7,188 | Optional Module | ML trainers, evaluation, skills |
| `hololoom/input/` | 2,201 | Optional Module | Multi-modal input processing (audio, image) |
| `hololoom/synthesis/` | 1,500 | Optional Module | Pattern extraction and training data synthesis |
| `hololoom/weaving/` | 4,182 | Optional Module | Weaving protocols, policies, stage definitions |
| `hololoom/writing/` | 4,659 | Optional Module | NLG system (standard/creative/technical modes) |
| `hololoom/llm/` | 1,535 | Optional Module | Unified LLM client abstraction |
| `hololoom/conscience/` | 3,892 | Optional Module | Ethical reasoning and values alignment |
| `hololoom/expansions/` | 1,084 | Optional Module | Context expansion patterns |
| `hololoom/safety/` | 1,136 | Optional Module | Safety locks (complements alignment/) |
| `hololoom/promptly/` | 3,766 | Optional Module | DSPy bridge, prompt management |
| `hololoom/ts_core/` | 2,000 | Optional Module | Thompson Sampler core with model management |
| `hololoom/lsp/` | 1,119 | Optional Module | Language Server Protocol integration |
| `hololoom/workflows/` | 1,988 | Optional Module | Workflow executor and definitions |
| `hololoom/domain_harness/` | 9,363 | Optional Module | Domain-specific prompt harness *(new)* |
| `hololoom/weaverlet/` | 4,011 | Optional Module | Task controller and refinement loops *(new)* |
| `hololoom/tui/` | 1,824 | Optional Module | Terminal UI with graph display *(new)* |
| `hololoom/motif/` | 762 | Optional Module | Pattern/motif abstractions *(new)* |
| `hololoom/pipeline/` | 700 | Optional Module | Pipeline stage caching and resolver *(new)* |
| `hololoom/infrastructure/` | 2,454 | Tooling | Deployment support (SQL, Grafana, K8s) *(new)* |

### Removed (confirmed missing — consolidated in Waves 2-3)

| Directory | Merged Into |
|-----------|-------------|
| `hololoom/neural/` | `core/policy/` |
| `hololoom/math/` | `core/warp/` |
| `hololoom/nested/` | `core/orchestrator/` |
| `hololoom/clustering/` | `core/memory/` |
| `holoLoom/` (PascalCase) | `hololoom/` (Wave 5 rename complete) |

---

## The Numbers (Revised 2026-04-02)

| Layer | LOC | % of Codebase | Directory Count |
|-------|-----|--------------|-----------------|
| **Core** | ~132K | 14% | 17 dirs |
| **Optional Modules** | ~560K | 56% | 60 dirs |
| **Apps** | ~150K | 15% | 6 dirs |
| **Tooling & Tests** | ~150K | 15% | 13 dirs |
| **Unclear** | 0 | 0% | 0 dirs |

All previously "unclear" modules have been classified. Core grew from 13 → 17
modules (bus, deep_thinking, ritual, runtime). Optional modules absorbed the
27 formerly unclear modules plus 5 new ones. 4 modules were confirmed removed
(neural, math, nested, clustering — merged during Waves 2-3).

---

## Proposed Target Structure (Revised 2026-02-26)

The previous version of this doc proposed `core/` + `extensions/`. On review,
the `extensions/` wrapper doesn't earn its keep. `voice/` and `federation/` are
both just "optional" — the filesystem doesn't need to encode *why* they're
optional. That distinction belongs in `pyproject.toml` extras and docs.

**Two levels: core, and everything else.**

```
hololoom/                          ← The package (lowercase, PEP 8)
│
├── core/                          ← Always installed (~132K LOC, 17 modules)
│   ├── protocols/                 ← Zero-dep contracts (includes viz protocol)
│   ├── memory/                    ← Knowledge graph, vector store, awareness
│   ├── embedding/                 ← Matryoshka multi-scale
│   ├── policy/                    ← Thompson Sampling decision engine
│   ├── convergence/               ← Probability collapse → actions
│   ├── orchestrator/              ← 9-step weaving cycle
│   ├── warp/                      ← Tensioned tensor manifold
│   ├── fabric/                    ← Spacetime output with provenance
│   ├── chrono/                    ← Temporal windows
│   ├── resonance/                 ← Feature extraction (DotPlasma)
│   ├── loom/                      ← Pattern card selection
│   ├── recursive/                 ← Self-improving learning loops
│   ├── reflection/                ← Episodic buffer, PPO
│   ├── bus/                       ← Unified event/message bus
│   ├── deep_thinking/             ← Deliberation engine
│   ├── ritual/                    ← Ritual grammar & hooks
│   └── runtime/                   ← 6-layer lifecycle management
│
├── agentic/                       ← Flat optional peers (~250K LOC total)
├── alignment/                     ← None of these import each other.
├── rag/                           ← Install what you need.
├── dark_trace/
├── routing/
├── semantic_calculus/
├── context_packing/
├── prompting/
├── search/
├── bandits/
├── context/
├── memory_symphony/
├── spinningWheel/
├── integrations/
├── chaining/
├── physics/
├── planning/
├── causal/
├── explainability/
├── reasoning/
├── verification/
├── voice/
├── vision/
├── spatial/
├── visualization/                 ← Jenny (implements viz protocol from core)
├── collaboration/
├── dreamweaving/
├── departments/
├── eggroll/
├── federation/
├── server/
├── thirdeye/
└── redteam/                       ← Dev tool, but still just an optional module
```

**Apps** — separate packages, not inside `hololoom/`:

```
apps/
├── elle/                          ← AR guide companion
├── trough/ + xterminator/         ← Code quality QA
├── chatops/                       ← Matrix.org chatbot
├── workflow_builder/              ← Visual workflow builder
├── bosspig/                       ← Business doc slop detector
├── sous/                          ← Kitchen management AI
└── ...
```

**Support** — at the repo root:

```
tests/                             ← All tests
├── unit/
├── integration/
└── e2e/
tools/                             ← Dev utilities, tuning, security
infra/                             ← Docker, K8s, monitoring
docs/                              ← Documentation
demos/                             ← Examples
```

### Why This Structure

**What we dropped**: The `extensions/` directory. It was a category for the sake
of having a category. `voice/` and `federation/` and `rag/` are all just
"optional modules that don't import each other." They don't need a parent folder
to tell you that.

**Where the categories live instead**:

- **`pyproject.toml` extras** — `pip install hololoom[voice,vision]`,
  `pip install hololoom[server,federation]`
- **Documentation** — "Input modalities: voice, vision. Output: spatial,
  visualization. Infra: server, federation."
- **Not the directory tree** — A developer shouldn't have to open
  `extensions/` → `input_modalities/` → `voice/` to find the voice module.
  `from hololoom.voice import ...` is enough.

**The test**: If you can't explain why two modules are in the same folder
(beyond "they're both optional"), the folder shouldn't exist.

### How `core/` Earns Its Folder

Unlike `extensions/`, the `core/` grouping passes the test:

- Everything in `core/` is **always installed** — no extras needed
- Everything in `core/` is required for the **basic weaving pipeline** to work
- Removing any `core/` module breaks the system
- `core/` modules **do** import each other (orchestrator imports memory,
  policy, convergence, etc.)

The boundary is: "does the weaving cycle break without this?" Yes → core.
No → optional peer.

---

## Migration Path

Suggested order, each step independently valuable:

1. **Move apps out of `hololoom/`** — biggest clarity win, least breakage.
   chatops, departments, web_dashboard, server → `apps/`

2. **Consolidate micro-modules** — merge `neural/` into `policy/`, `math/`
   into `warp/`, `input/` into `spinningWheel/`, `synthesis/` into `fabric/`,
   `weaving/` into `orchestrator/`, etc. (the "unclear" list above)

3. **Create `core/`** — move the 13 core directories under `core/`. Update
   imports. This is the biggest refactor but makes the product visible.

4. **Flatten optional modules** — everything left in `hololoom/` that isn't
   `core/` stays as a flat peer. No `extensions/` wrapper.

5. **Rename to lowercase** — `hololoom/` → `hololoom/`. Breaking change,
   do last.

6. **Add `pyproject.toml` extras** — define install groups:
   ```toml
   [project.optional-dependencies]
   voice = ["whisper", "pyttsx3"]
   vision = ["ultralytics", "opencv-python"]
   server = ["fastapi", "uvicorn"]
   federation = ["kademlia"]
   all = ["hololoom[voice,vision,server,federation,...]"]
   ```
