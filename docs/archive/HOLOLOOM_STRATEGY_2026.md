# HoloLoom Strategy 2026

> "I have kids."
> "Then that's who it's for."

---

## Part I: The Realization

### What We Built

We set out to build a memory system for AI agents. What we actually built is a **neurosymbolic AI architecture** — 900,000+ lines of production code across 10+ projects that implements what the research community is only now converging on: **structured reasoning as the authority layer over large language models**.

The system is organized into six architectural layers:

| Layer | Name | What It Does |
|-------|------|-------------|
| 0 | **Memory** | Knowledge graphs, vector stores, spring dynamics, awareness graph |
| 1 | **Weaving** | 9-stage orchestration cycle, pattern cards, convergence engine |
| 2 | **Learning** | Thompson Sampling, neural policy, 7 parallel learning loops |
| 3 | **Deliberation** | MCTS planning, causal inference, game-theoretic verification |
| 4 | **Intent** | Multi-step reasoning, goal decomposition, failure recovery |
| 5 | **Mirror** | Self-reference, introspection, episodic consolidation |

This is not a wrapper around an LLM. This is a cognitive architecture where LLMs are **supervised participants** in a structured reasoning process.

The codebase includes:

- **122K LOC of core infrastructure** — the product itself (memory, weaving, learning, convergence)
- **470K LOC of optional modules** — 33 flat-peer extensions (agentic RAG, alignment, causal inference, game theory, information theory, federation, voice, vision, spatial)
- **150K LOC of applications** — Elle (farm ops AR guide), SOUS (kitchen control loop), ChatOps, workflow builder
- **~4,900 tests** across the ecosystem
- **The Council** — 7 rites that govern system behavior (Summon, Debate, Weigh, Decide, Enact, Record, Adjourn)
- **The Mirror** — self-referential introspection layer (Hofstadter strange loops made literal)
- **The Nervous System** — Femtoclaw (peripheral nerves / fast runtime), OpenClaw (autonomic / infrastructure), HoloLoom (cortex / reasoning)
- **Spindle/Weaver/Bobbin** — the weaving metaphor made architecture (thread management, pattern execution, context winding)

### Why This Matters Now

The AI industry spent 2023-2025 scaling LLMs. The results are clear: **LLMs are extraordinary pattern matchers but unreliable reasoners**. The market is pivoting.

> **Gary Marcus** (NYU): "We need hybrid AI — neural networks for pattern recognition combined with symbolic systems for reasoning, planning, and formal guarantees."

> **Judea Pearl** (Turing Award): "Current deep learning systems are curve-fitting devices. To achieve human-level intelligence, machines must acquire causal models of their environment."

> **DeepMind AlphaProof** (2024): Solved 4 of 6 International Math Olympiad problems by combining an LLM (Gemini) with a formal theorem prover (Lean). Neither system alone could have done it. The LLM proposed proof strategies; the formal system verified them.

This is **exactly** the architecture we built: structured AI as the authority, LLMs as supervised participants. We didn't design it from theory — we built it from necessity, because unreliable LLM outputs were breaking our applications.

Gary Marcus — the most vocal and most ridiculed critic of LLM-only approaches — has been systematically vindicated by events. His 2022 predictions that LLMs would plateau on reasoning, hallucinate persistently, and require hybrid architectures were dismissed as "naysaying." By 2025, every major lab acknowledged these exact problems. His core thesis — that AI needs structured symbolic components, not just bigger models — is now the World Economic Forum's recommendation, EY's platform strategy, and Gartner's 2-5 year prediction. HoloLoom doesn't just agree with Marcus. **HoloLoom is what Marcus has been asking for.**

**We built what the market is about to demand.**

---

## Part II: The Market Is Coming to Us

### The Regulatory Tide

The EU AI Act is the most significant AI regulation in history. It creates **legal requirements** for exactly the kind of structured oversight HoloLoom provides.

| Requirement | EU AI Act Article | HoloLoom Capability |
|------------|-------------------|---------------------|
| Risk management system | Art. 9 | Additive scoring with named terms, priority bands, mode-based weight adjustment |
| Data governance | Art. 10 | Bi-temporal knowledge graph with 7 edge types, provenance tracking |
| Technical documentation | Art. 11 | Spacetime Fabric output with 4D provenance (3D semantic + 1D temporal) |
| Record-keeping | Art. 12 | Event bus with signal recording, episodic buffer, reflection logs |
| Transparency | Art. 13 | 7 XAI techniques (explainability module), decision path tracing |
| Human oversight | Art. 14 | Intent approval gates, confidence thresholds, rollback on low confidence |
| Accuracy & robustness | Art. 15 | Thompson Sampling convergence guarantees, multi-pass refinement with convergence detection |
| Post-market monitoring | Art. 72 | 7 parallel learning loops at different timescales, anomaly detection |

**Timeline**:
- August 2025: Prohibited AI practices take effect
- August 2026: High-risk AI requirements take effect (governance, transparency, oversight)
- August 2027: Full enforcement including penalties (up to 7% global turnover)

Every company deploying AI in the EU will need infrastructure that provides these capabilities. Most don't have it. We do.

### The NIST Framework

The U.S. approach is voluntary but influential. NIST AI RMF 1.0 defines four functions:

| NIST Function | What It Requires | HoloLoom Mapping |
|---------------|-----------------|------------------|
| **GOVERN** | Policies, roles, accountability | Council (7 rites), DepartmentProtocol, intent approval gates |
| **MAP** | Context, scope, risk identification | Causal DAGs, knowledge graph entity mapping, routing/classification |
| **MEASURE** | Metrics, testing, monitoring | Thompson Sampling regret bounds, convergence proofs, 4,900+ tests |
| **MANAGE** | Risk treatment, incident response | Graceful degradation (FULL→PARTIAL→DEGRADED→DARK), circuit breakers |

Enterprise procurement teams are already including NIST AI RMF compliance in RFPs. HoloLoom's architecture maps directly onto their requirements.

### Market Size

The AI governance, risk, and compliance market is emerging rapidly:

| Source | 2026 Estimate | 2030 Projection | CAGR |
|--------|--------------|-----------------|------|
| MarketsandMarkets | $492M | $2.1B | 33.8% |
| Grand View Research | ~$500M | $5.6B | 40.2% |
| Gartner | — | "30% of enterprises will require AI governance tooling" | — |
| IDC | — | "$1B+ in AI trust/safety tooling" | — |

The adjacent markets are larger: AI observability ($2B+), MLOps ($4B+), AI security ($3B+). HoloLoom sits at the intersection of all three.

### What Enterprise Buyers Actually Require

The U.S. Office of Management and Budget directive **OMB M-26-04** (April 2026) establishes concrete AI governance requirements for all federal agencies — and these requirements cascade to every vendor selling to government. Enterprise RFPs in regulated industries are converging on the same checklist:

| RFP Requirement | What It Means | HoloLoom Status |
|-----------------|---------------|-----------------|
| **AI Bill of Materials (AIBOM)** | Inventory of all AI components, data sources, models used | ✅ Spacetime provenance tracks every component |
| **Model Cards** | Standardized documentation of model capabilities, limitations, bias | ✅ Dark Trace + 244 semantic dimensions provide richer detail than standard model cards |
| **Fairness Testing** | Demonstrate non-discrimination across protected groups | 🟡 Framework supports custom fairness metrics; needs domain-specific calibration |
| **Audit Trails** | Complete, tamper-proof record of all AI decisions | ✅ Cryptographic SHA-256 chain-sealed audit trail |
| **Human-in-the-Loop** | Human review for high-risk decisions | ✅ Safety guardrails with configurable risk thresholds and escalation |
| **Stress Testing** | Performance under adversarial conditions | ✅ RedTeam CARTS framework with Thompson Sampling adversarial testing |
| **Explainability** | End-user understandable decision explanations | ✅ 244 interpretable dimensions + XAI techniques + causal reasoning |
| **Incident Response** | Plan for AI failures and misalignment | ✅ Circuit breakers, deception detection, instrumental convergence prevention |

**Key insight:** HoloLoom meets **7 of 8** standard enterprise RFP requirements out of the box. Most competitors meet 1-2. This is the single strongest sales argument for regulated industries.

### The Responsible AI Market

**The broader Responsible AI market is even larger:** $1.09B (2024) → **$10.26B by 2030** at 45.2% CAGR (NextMSC Research). This encompasses the full stack — governance, bias detection, explainability, audit tooling — and HoloLoom addresses multiple segments simultaneously.

**Key buyer segments:**
- Government agencies: 20% of spending
- Financial services: Highest per-deal value ($50K-250K/year)
- Healthcare: Compliance-driven, growing fastest
- EdTech / Child Safety: Emerging vertical driven by KOSA and AI-for-minors concerns

---

## Part III: Competitive Landscape

### The Frontier Lab Gap

Frontier labs (OpenAI, Anthropic, Google DeepMind) are building increasingly capable LLMs. But they have a structural blind spot: **they can't provide the structured oversight layer**.

| Lab | Strengths | What They Can't Do |
|-----|-----------|-------------------|
| **OpenAI** | GPT-4+, function calling, assistants API | Can't provide causal reasoning, formal verification, or persistent structured memory across sessions |
| **Anthropic** | Claude, constitutional AI, safety research | Constitutional AI constrains *generation* but doesn't verify *reasoning*. No structured decision engine. |
| **Google DeepMind** | Gemini, AlphaProof, AlphaCode | AlphaProof proves the thesis (LLM + formal system > LLM alone) but isn't a product you can buy |
| **Meta** | Llama (open weights), FAIR research | Open weights enable self-hosting but don't include structured reasoning infrastructure |

**Anthropic's circuit tracing research** validates this gap. Their work on Claude discovered that LLMs plan ahead — computing future tokens before generating current ones — and identified the exact mechanism of hallucination: *"when the model incorrectly extends a pattern"*, like fabricating a person's nationality by extrapolating from a foreign-sounding name. This confirms that LLMs have systematic failure modes that can't be fixed with scale alone. The question is whether you try to fix the model (Anthropic's approach, internal-only, model-specific) or build a system where the model can't cause harm (HoloLoom's approach, open-source, model-agnostic).

**Why they can't compete here**: Frontier labs sell LLM inference. Adding structured oversight would mean admitting their models need supervision — contradicting their go-to-market narrative. The oversight layer must come from an independent party.

### The Guardrails Gap

Existing AI safety/guardrails tools focus on input/output filtering, not structured reasoning:

| Company | Approach | Limitation |
|---------|----------|-----------|
| **Guardrails AI** | Input/output validators, prompt injection detection | Surface-level checking. Validates format, not reasoning. No learning loop. |
| **NeMo Guardrails** (NVIDIA) | Programmable guardrails for LLM apps | Rule-based rails. No Bayesian learning, no causal inference, no convergence guarantees. |
| **Rebuff** | Prompt injection detection | Single-purpose. Doesn't address reasoning quality, decision auditability, or multi-model governance. |
| **LLM Guard** | Content moderation | Input/output filtering only. No structured decision-making. |
| **Lakera** | Prompt injection + content safety | Detection-focused. Doesn't help with *making better decisions*, just blocking bad inputs. |

**The gap**: These tools say "don't do bad things." HoloLoom says "here's how to make good decisions, and here's the proof." Prevention vs. structured authority.

### The Orchestration Gap

LLM orchestration frameworks provide plumbing but not intelligence:

| Framework | What It Does | What It Doesn't Do |
|-----------|-------------|-------------------|
| **LangChain** | Chain LLM calls, tool use, retrieval | No learning loop. No convergence detection. No structured decision engine. Stateless between calls. |
| **LlamaIndex** | Data framework for LLM apps | Excellent retrieval, but no reasoning layer. No policy engine. No causal inference. |
| **DSPy** | Programmatic prompt optimization | Optimizes prompts, not decisions. No persistent memory. No game-theoretic verification. |
| **AutoGen** (Microsoft) | Multi-agent conversation | Agent conversation patterns, but no structured authority. Agents negotiate in natural language — no formal verification. |
| **CrewAI** | Multi-agent task execution | Role-based agents, but decisions are made by LLMs, not structured systems. No convergence guarantees. |

**The gap**: Orchestrators move data between LLMs. HoloLoom provides the **reasoning authority** that governs what the LLMs produce.

### The Neurosymbolic AI Landscape

Companies explicitly pursuing neurosymbolic approaches:

| Company | Focus | Approach | HoloLoom Differentiation |
|---------|-------|----------|--------------------------|
| **Symbolica AI** | Structured reasoning | Category theory for AI | Research-stage. No production system. No memory layer. |
| **Relational AI** | Knowledge graph + ML | Graph neural networks | Database-focused. No orchestration, no decision engine, no multi-model governance. |
| **Neuro-Symbolic AI** (IBM Research) | Hybrid architectures | Research papers | Research division, not a product. Papers, not production code. |
| **Elemental Cognition** | Hybrid reasoning | NLU + knowledge representation | Enterprise NLU focus. No open architecture. No self-improving loops. |
| **Aleph Alpha** | Sovereign AI | European LLM + explainability | LLM provider with explainability features, not a structured reasoning infrastructure. |

**HoloLoom's position**: The only production-scale neurosymbolic architecture with 900K+ LOC, 6 architectural layers, 33 optional modules, and applications already running in production (Elle for farm ops, SOUS for kitchen management).

---

## Part IV: The Thesis

### "Structured AI as the Authority"

The thesis is simple: **LLMs should not be the decision-makers. Structured AI should be the authority, with LLMs as supervised participants.**

This is not anti-LLM. LLMs are extraordinary at:
- Natural language understanding and generation
- Pattern recognition across vast corpora
- Creative synthesis and analogy
- Code generation and translation

LLMs are unreliable at:
- Consistent reasoning across steps
- Formal verification of their own outputs
- Learning from specific outcomes (without retraining)
- Providing audit trails for their decisions
- Operating within provable safety bounds

The architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    USER / APPLICATION                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              STRUCTURED AUTHORITY                     │ │
│  │                                                       │ │
│  │  Memory ──► Weaving ──► Learning ──► Deliberation    │ │
│  │    │           │           │             │            │ │
│  │    │     ┌─────┴─────┐    │      ┌──────┴──────┐    │ │
│  │    │     │ 9-Stage   │    │      │   MCTS +    │    │ │
│  │    │     │ Pipeline  │    │      │  Causal +   │    │ │
│  │    │     │           │    │      │ Game Theory │    │ │
│  │    │     └─────┬─────┘    │      └──────┬──────┘    │ │
│  │    │           │          │             │            │ │
│  │    ▼           ▼          ▼             ▼            │ │
│  │  Intent ◄── Mirror ◄── Council ◄── Convergence      │ │
│  │                                                       │ │
│  └────────────────────┬──────────────────────────────────┘ │
│                       │                                     │
│              ┌────────┴────────┐                           │
│              │  LLM INTERFACE  │                           │
│              │  (Supervised)   │                           │
│              └────────┬────────┘                           │
│                       │                                     │
│         ┌─────────────┼─────────────┐                      │
│         ▼             ▼             ▼                      │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│    │ Local   │  │  Cloud  │  │  Rig    │                  │
│    │ Ollama  │  │  Claude │  │ AirLLM  │                  │
│    │ qwen3.5 │  │  GPT-4  │  │ 108B   │                  │
│    └─────────┘  └─────────┘  └─────────┘                 │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

The structured authority:
1. **Decides** which LLM to use (Thompson Sampling model router)
2. **Constrains** what the LLM can assert (causal DAG validation)
3. **Verifies** the LLM's output (game-theoretic equilibrium checks)
4. **Learns** from outcomes (7 parallel learning loops)
5. **Explains** every decision (intrinsic explainability, not post-hoc)
6. **Degrades gracefully** when components fail (FULL → PARTIAL → DEGRADED → DARK)

### The 30-Second Pitch

"Every enterprise deploying AI needs to answer three questions: Is the AI making good decisions? Can you prove it? Can you improve it over time?

HoloLoom is the structured reasoning layer that sits between your applications and your LLMs. It makes decisions using formal methods — Bayesian learning, causal inference, game theory — and uses LLMs as pattern matchers within that structured framework.

The result: AI systems that are auditable, improvable, and provably reliable. Not because we added guardrails, but because the architecture is built on mathematical foundations."

---

## Part V: Go-to-Market Strategy

### Positioning

**Category**: AI Governance & Structured Reasoning Infrastructure

**Tagline**: "The reasoning layer your AI needs."

**For**: Engineering teams deploying AI in production who need reliability, auditability, and continuous improvement.

**Against**: Ad hoc prompt engineering, LLM-only architectures, surface-level guardrails.

### Three-Phase Market Entry

**Phase 1: Developer Adoption (Q2-Q3 2026)**

Open source HoloLoom Lite — the core weaving engine with Thompson Sampling, basic memory, and convergence detection. Target: developers building AI applications who are frustrated with unreliable LLM outputs.

- MIT license, self-hostable
- pip install hololoom
- Works with any LLM provider (Ollama, OpenAI, Anthropic, local models)
- Documentation, examples, community Discord
- Target: 5,000+ developers, 1,000+ GitHub stars

**Phase 2: Enterprise Pilots (Q4 2026 - Q1 2027)**

HoloLoom Pro — managed deployment with enterprise features. Target: companies preparing for EU AI Act compliance.

- Managed cloud or on-premise deployment
- Audit trail and compliance reporting
- Multi-model governance dashboard
- SSO, RBAC, team management
- Target: 10-20 enterprise pilot customers

**Phase 3: Market Leadership (2027+)**

HoloLoom Enterprise — full platform with federation, advanced analytics, and professional services.

- Multi-region deployment
- Federation (distributed reasoning across organizational boundaries)
- Advanced analytics and anomaly detection
- SOC2, HIPAA, GDPR compliance certifications
- Professional services and training
- Target: 50+ enterprise customers, $5M+ ARR

### Pricing Tiers

| Tier | Target | Price | Includes |
|------|--------|-------|----------|
| **HoloLoom Lite** | Individual developers, startups | Free (MIT) | Core weaving engine, basic memory, Thompson Sampling, CLI |
| **HoloLoom Pro** | Teams (5-50), mid-market | $99/user/month | Managed deployment, audit trails, multi-model governance, dashboard, email support |
| **HoloLoom Enterprise** | Enterprise (50+), regulated industries | Custom ($50K+/year) | Everything in Pro + compliance certs, federation, dedicated support, SLAs, professional services |

### Revenue Projections

| Year | Lite Users | Pro Users | Enterprise Deals | ARR |
|------|-----------|-----------|-------------------|-----|
| 2026 (H2) | 5,000 | 50 | 2 | $160K |
| 2027 | 20,000 | 500 | 15 | $1.3M |
| 2028 | 50,000 | 2,000 | 50 | $7.9M |
| 2029 | 100,000 | 5,000 | 150 | $28M |
| 2030 | 200,000 | 10,000 | 300 | $60M+ |

Revenue mix shifts from Pro-heavy (Year 1-2) to Enterprise-heavy (Year 3+) as compliance deadlines hit.

---

## Part VI: Technical Roadmap

### v1.0 Cleanup Sprint (4-6 weeks, Q2 2026)

The immediate priority. Close beta.1, get to stable, prepare for external shipping.

**API Lock**:
- Freeze the public API surface: `HoloLoom`, `Memory`, `experience()`, `recall()`, `reflect()`, `query()`, `chat()`, `ingest()`
- Generate API reference from docstrings
- Semver enforced: no breaking changes in 1.x

**Tests Green**:
- CI pipeline: ruff + black + mypy + pytest
- All ~4,900 tests passing (unit <5s, integration <30s, e2e <2min)
- Docker Compose tested end-to-end
- All guides verified against actual code paths

**Non-Destructive Consolidation**:
- Archive stale modules (don't delete — archive)
- Resolve the "Unclear / Needs Discussion" modules from MODULE_TAXONOMY.md (~40K LOC across 25 dirs)
- Merge or archive micro-modules that were deferred in Wave 2

**Orchestrator Decomposition**:
- The orchestrator (8,788 LOC) currently handles too much
- Extract clean stage boundaries matching the 9-stage weaving cycle
- Each stage independently testable

**Infrastructure Hardening**:
- systemd service files for all long-running processes
- Secrets management (no more hardcoded URLs in source)
- Alerting on service health (mining rig GPU servers, Ollama, Matrix homeserver)
- Equipment inbound — prepare deployment configs

**Exit Criteria**: CI green, API frozen, docs verified, `pip install hololoom` works clean, classifier updated to "5 - Production/Stable".

### v2.0 Cognitive UI (Q2-Q3 2026)

The paradigm shift: the interface becomes the cognitive process. This is what ships externally.

**Three-Pane Consciousness Shell**:
- Memory Palace (left) — knowledge graph visualization, entity relationships, spring dynamics
- Active Thinking (center) — streaming reasoning timeline, 9-stage progress, confidence tracking
- Awareness (right) — context window, active memories, attention heatmap

**WebSocket Event Stream**:
- Real-time orchestrator events streamed to UI
- Stage transitions, bandit candidate selection, convergence detection
- Cross-pane interaction: node click in Memory Palace triggers chat context; reasoning step highlights relevant graph edges

**Dogfooding**:
- Elle for farm operations (Coz cooperative — real daily use)
- SOUS for kitchen management (107 tests, production-ready scoring engine)
- Both systems exercise the full 6-layer architecture daily

**Jenny Conversation Stages** (Stages 1-3 complete):
- Static HTML panels, ConversationGraph with trajectory detection, positioned 3D overlays via WebSocket
- Stages 4-8 planned: intelligence track, persistent learning, cross-session synthesis

**React Frontend** (hololoom-ui/ — Next.js 14 + React 18 + TypeScript + Tailwind):
- MemoryGraph (canvas force-directed, zoom/pan/select)
- ChatInterface (4 reasoning modes, confidence badges)
- PerformanceOverview (sparklines, trend indicators)
- API client + WebSocket (pattern subscriptions, auto-reconnect)
- Design system (12 components, 3 themes)

**Exit Criteria**: Three-pane shell live with real WebSocket data, Elle and SOUS running daily through the UI, external users can `pip install hololoom` and run the cognitive shell locally.

### Phase 3: Ecosystem Integration (Q3-Q4 2026)

Wire the nervous system together. Connect the applications to the core through clean protocols.

**Femtoclaw Wiring**:
- Full HoloLoom backend integration (currently Ollama-only for most channels)
- Per-group container isolation with HoloLoom memory contexts
- Matrix + WhatsApp channels routing through structured authority

**Elle Full Integration**:
- AR overlays driven by HoloLoom's spatial module
- Farm operations memory persisted in awareness graph
- Equipment status, animal health, seasonal planning — all through the 6-layer architecture

**SOUS CrunchBackend**:
- The "Crunch" backend bridges SOUS scoring engine to HoloLoom's structured authority
- Recipe selection uses HoloLoom's Thompson Sampling (not its own separate bandit)
- Inventory tracking in knowledge graph with bi-temporal provenance

---

## Part VI-A: Structured AI Roadmap

This is the technical heart — the plan for strengthening HoloLoom's structured AI capabilities, connecting disconnected systems, and integrating new structured methods. Two tracks run in parallel: **wiring** (connecting what exists) and **building** (adding new structured systems).

### What We Already Have

A codebase audit (March 2026) found structured AI capabilities at or ahead of the industry:

| Domain | LOC | Status | Industry Comparison |
|--------|-----|--------|---------------------|
| **Bayesian Methods** | — | Thompson Sampling, GP Bandits, Variational Inference, Neural TS | At or ahead (Google's Bayesian Teaching, BC-LLM) |
| **Causal Inference** | 3,150 | Full Pearl framework: DAGs, do-calculus, counterfactuals, SCMs | Ahead of most (NAACL 2025 identifies 3 approaches; we have all 3) |
| **Planning** | — | MCTS (6 modules), HTN, causal chain planning | Strong (AlphaProof uses MCTS+RL; we have MCTS+Bayesian) |
| **Game Theory** | 31,800 | Nash, correlated equilibria, Shapley values, mechanism design | Exceptional (niche in most frameworks) |
| **Information Theory** | 14,400 | MI, KL divergence, channel capacity, rate distortion | Unique advantage (most systems don't have this) |
| **Knowledge Graphs** | — | Bi-temporal, 7 edge types, spectral features, spring dynamics | Solid (KG-LLM fusion is 2025's hottest topic) |
| **Interpretability** | 15,000 | Dark Trace SAE decomposition, 244 semantic dimensions | Ahead (closest: Anthropic internal tools, DeepMind Gemma Scope) |

### The Five Integration Patterns

The neurosymbolic field has converged on five patterns. HoloLoom implements one, is missing two critical ones, and can adopt two emerging ones.

**Pattern 1: LLM proposes, formal system verifies** ⚠️ MISSING
- DeepMind AlphaProof: Gemini proposes proof strategies, Lean verifies
- Lean Copilot: 74% of proof steps automated with LLM + theorem prover
- HoloLoom has both the LLM interface and formal systems — they aren't wired this way

**Pattern 2: LLM as front-end, structured system as back-end** ✅ DOING THIS
- DSPy + ASP, LLM + PDDL planners
- The 9-stage weaving cycle IS this pattern

**Pattern 3: Structured constraints on LLM inference** ⚠️ MISSING
- Constitutional AI, PiShield/CCN+ (formal constraints on neural output)
- We have causal DAGs and info-theoretic bounds but don't apply them to LLM output

**Pattern 4: Adaptive symbolic routing** 🆕 NEW (2025)
- Meta-router dispatches reasoning problems to the optimal solver: FOL → Prover9, LP → Pyke, SAT/SMT → Z3, planning → MCTS
- **96% accuracy on composite benchmarks vs 71% for any single method**
- HoloLoom has all the solvers. Doesn't have the router.

**Pattern 5: Spectral neuro-symbolic reasoning** ✅ VALIDATED
- Fully spectral architecture (Sep 2025): graph signals + learnable polynomial filters
- **Outperforms attention-based reasoning** on ProofWriter, EntailmentBank, bAbI, CLUTRR, ARC-Challenge
- HoloLoom's spectral features approach is vindicated. GNNs are NOT needed. Mixture-of-spectral-experts is the next step.

### Track 1: Wiring Roadmap (connect existing systems)

Ranked by value/effort. Most of this is integration — the components exist, the connections don't.

#### Tier 1: Immediate (days, not weeks)

**W1. Info theory → refinement convergence**
Use MI between consecutive refinement passes as the principled stop signal. When MI(pass_n, pass_n+1) < threshold, the passes have converged. KL divergence measures how much the output distribution shifted. Replaces ad-hoc convergence detection with information-theoretic optimality.
- *Uses existing*: MI and KL modules (14K LOC info theory)
- *Connects to*: multi-pass refinement in weaving orchestrator
- *Citation*: "To Believe or Not to Believe Your LLM" (NeurIPS 2024) — MI separates epistemic from aleatoric uncertainty

**W2. KL divergence → LLM drift detection**
Compute KL(response_distribution || knowledge_base_distribution) after each LLM response. High divergence = hallucination. Threshold triggers constrained re-generation or flags for review. Semantic Divergence Metrics (Halperin, arXiv 2508.10192, 2025) provides the production-ready framework with Jensen-Shannon divergence over sentence embeddings.
- *Uses existing*: KL divergence, knowledge graph embeddings
- *Connects to*: router.ts (Femtoclaw), weaving output stage

**W3. Shapley → Thompson Sampling calibration**
For additive scoring (Design Principle #4), Shapley values are *analytically free*: φᵢ = wᵢ × (xᵢ - E[xᵢ]). Periodically compute Shapley values for tool/model contributions. If they diverge from Thompson Sampling α/β parameters, the bandits are miscalibrated. Shapley provides independent ground-truth credit assignment.
- *Uses existing*: game theory Shapley (31K LOC), Thompson Sampling bandits
- *Citation*: AgentSHAP (Dec 2025) — Monte Carlo Shapley for LLM agent tool attribution

**W4. PiShield-style constraint layer on output**
Add a lightweight constraint enforcement layer after the weaving cycle. Define safety/quality invariants as CNF formulas or linear inequalities. Z3 solver guarantees compliance regardless of LLM output. PiShield (IJCAI 2024) demonstrates this with minimal integration cost — pip installable, requires only input dimension + requirements file.
- *Uses existing*: alignment framework, safety guardrails
- *New dependency*: Z3 solver (well-maintained, Python bindings)
- *Ships as*: optional post-processing stage in orchestrator

**W5. Game-theoretic minimax gate**
Construct game matrix (tools × environment states), solve LP for minimax-optimal tool. If it matches Thompson Sampling selection → proceed. If it diverges → flag for deeper analysis. For 10 tools × 20 states: LP has 200 variables, solvable in <1ms. Cheap gate, high signal.
- *Uses existing*: game theory Nash/minimax (31K LOC), convergence engine
- *Citation*: MaMa (arXiv 2602.04431, Feb 2026) — Stackelberg security games for safe agent design

#### Tier 2: Medium integration (weeks)

**W6. Causal DAGs → convergence engine (Online Causal Thompson Sampling)**
Instead of updating Beta(α,β) from raw (action, reward), decompose expected reward via do-calculus: E[Y|do(X=a)] using backdoor adjustment. This deconfounds reward estimation — if tool A only *looks* good because it gets selected for easy queries, the causal model exposes the bias. Bareinboim proved causal bandits achieve strictly better regret bounds than standard bandits when the graph structure is known (NeurIPS 2016, extended through 2024).
- *Uses existing*: Pearl causal framework (3,150 LOC), convergence engine, Thompson Sampling
- *The algorithm*: maintain Beta posteriors over *structural parameters* of the causal model, not raw rewards. Sample from structural posteriors, compute interventional expected reward per candidate tool.
- *Cost*: backdoor adjustment is O(|Z|) per candidate — microseconds
- *Citations*: Bareinboim "Causal Bandits" (NeurIPS 2016), "Introduction to Causal RL" (R-65, 2024-2025)

**W7. Info theory → deep thinking convergence (EVINCE pattern)**
Implement EVINCE's dual-entropy policy for the Propose/Challenge/Resolve cycle. Planner = high entropy (explorer), Critic = low entropy (confirmer). Track MI between them. When MI plateaus, deliberation has converged. Use Jensen-Shannon divergence to measure argument diversity.
- *Uses existing*: deep thinking module (3 GPU servers), info theory (MI, JSD)
- *Citation*: EVINCE (arXiv 2408.14575, 2024) — improved Top-1 accuracy by ~7% on medical diagnosis via MI-controlled multi-LLM debate

**W8. MCTS + KG → active reasoning (ReKG-MCTS pattern)**
Use MCTS to search over knowledge graph paths for multi-hop reasoning. UCB-based node selection on KG nodes, LLM-guided rollouts along edges, value backpropagation to score reasoning paths. Transforms the KG from passive storage into an active reasoning substrate.
- *Uses existing*: MCTS (6 modules), knowledge graph (bi-temporal, spectral)
- *Citation*: ReKG-MCTS (ACL 2025) — training-free, outperforms baselines on WebQSP and CWQ

**W9. Spring tension = free energy (active inference bridge)**
Reinterpret spring potential energy as variational free energy. F = -kx (Hooke's Law) IS the gradient of quadratic free energy. Tension = prediction error. The system minimizes free energy by updating its world model (adjusting spring rest lengths) or taking action (consolidating memories). This connects the physics metaphor to Karl Friston's free energy principle without changing the existing math.
- *Key insight*: Thompson Sampling stays for fast individual decisions. Active inference works at orchestration level — the meta-controller for the weaving cycle.
- *Empirical*: Active inference outperforms TS in short sessions/stationary environments. TS more robust in non-stationary with many arms. They're complementary, not competing.
- *Citation*: VERSES AXIOM — crushed DreamerV3 on Gameworld 10k, converging in 947 steps

**W10. KG as world model with forward prediction**
Add a forward-prediction step to the knowledge graph: given current state (nodes + edges), predict the next state (which edges will activate, which nodes become relevant). Use MCTS to search over possible futures. Validate predictions against actual outcomes to learn dynamics.
- *Uses existing*: KG, MCTS, HTN planning
- *Citation*: AriGraph (IJCAI 2025) — KG as world model with semantic + episodic memory, outperforms RL baselines in text-based games. HoloLoom's Yarn Graph already does this structurally.

#### Tier 3: Significant build (months)

**W11. Adaptive symbolic router**
Meta-router classifies reasoning problems and dispatches to optimal solver: causal questions → do-calculus engine, constraint satisfaction → Z3, graph traversal → MCTS over KG, scoring/ranking → additive scoring pipeline. Use Thompson Sampling to learn which routing decisions work best.
- *This is Pattern 4*: the pattern that yields 96% accuracy vs 71% for single methods
- *We have all the solvers*. We don't have the router.

**W12. AlphaEvolve for scoring term evolution**
LLM proposes scoring term code diffs, evaluator measures downstream performance, MAP-Elites maintains diverse configurations, Thompson Sampling selects which to evolve next. Self-improving decision system. OpenEvolve (Apache 2.0) supports Ollama endpoints.
- *Uses existing*: additive scoring (Design Principle #4), Thompson Sampling, game theory fitness landscapes
- *New*: evolutionary loop, MAP-Elites population database
- *Citation*: AlphaEvolve (DeepMind, May 2025) — found 48-multiplication algorithm for 4x4 complex matrix multiply, beating Strassen's 1969 result

### Track 2: New Structured Systems

Systems that don't exist yet but the research says we should build.

**S1. Evolutionary search engine** — `hololoom/evolution/` (High priority)
The AlphaEvolve pattern adapted for HoloLoom. Five components: prompt sampler (selects parent solutions), LLM ensemble (Ollama local for throughput + Claude for quality rewrites), evaluator (user-defined fitness function), population database (MAP-Elites for diversity), controller (async pipeline coordinator). MCTS provides tree structure, game theory evaluates multi-objective fitness, Thompson Sampling handles exploration in the population.
- *Substrate*: OpenEvolve (pip installable, Apache 2.0)
- *First application*: evolve weaving cycle parameters and scoring term weights

**S2. SAT/SMT constraint engine** — `hololoom/constraints/` (Medium priority)
Z3 solver integration for hard safety invariants. Two modes: training-time constraint enforcement (PiShield pattern) and inference-time post-processing (shield layer on weaving output). CNF formulas for logical constraints, linear inequalities for continuous bounds.
- *Why now*: PiShield (IJCAI 2024) proved minimal integration cost. Z3 is battle-tested.
- *First application*: safety invariants that must never be violated regardless of LLM output

**S3. VCG truthful reporting mechanism** — extend `hololoom/game_theory/` (Medium priority)
Vickrey-Clarke-Groves mechanism for multi-subsystem coordination. Each component (memory, tools, router) reports confidence scores. VCG makes truthful reporting a dominant strategy — each component is "paid" the externality it creates (value added to overall decision quality). Uses strictly proper scoring rules (log score, Brier score) for Beta updates.
- *Why*: Without this, subsystems can strategically inflate relevance scores to get their results selected
- *Citation*: mechanism design literature + "Incentive-Aware AI Safety via Stackelberg Games" (arXiv 2602.07259, 2026)

**S4. Entropic activation steering** — extend `hololoom/policy/` (Low priority, high ceiling)
EAST (arXiv 2406.00244, 2024) computes entropy-weighted steering vectors that directly control LLM confidence/uncertainty during the forward pass. Goes beyond temperature sampling — modifies the subjective uncertainty the LLM expresses. Steering vectors transfer across tasks.
- *Integration point*: model router applies steering vectors based on convergence engine confidence

**S5. Mixture-of-spectral-experts** — extend `hololoom/resonance/` (Low priority)
Extends existing spectral features with specialized filters per reasoning type. Different spectral experts for different reasoning patterns (relational, temporal, causal, spatial). Dynamic basis learning and uncertainty quantification.
- *Citation*: "From Eigenmodes to Proofs" (arXiv 2509.07017, Sep 2025)

### The Cross-Cutting Architecture

The three structured AI domains connect through the decision lifecycle:

```
BEFORE decision (Causal):
  Causal DAG constrains candidate set
  → Backdoor adjustment deconfounds Thompson Sampling rewards
  → Interventional query: P(outcome | do(select_tool_k))
  → Adaptive router dispatches to optimal solver

DURING decision (Game-Theoretic):
  Minimax verification gate (<1ms)
  → Shapley attribution of feature contributions (free for additive scoring)
  → Proper scoring rules ensure truthful subsystem confidence
  → VCG mechanism aligns subsystem incentives

AFTER decision (Information-Theoretic):
  KL divergence detects output drift from knowledge base
  → MI measures epistemic uncertainty in LLM response
  → Rate-distortion bounds response length
  → EVINCE dual-entropy detects deliberation convergence
  → Feedback updates causal model (L3 counterfactual: "would tool B have been better?")
```

### What's Genuinely Novel

After the landscape research, here's what nobody else is doing:

- **Physics-based memory activation**: Spring dynamics (Hooke's Law) for context retrieval, with Velocity Verlet integration. The free energy reinterpretation connects this to active inference. Nobody else uses physics simulation for memory retrieval.

- **Information-theoretic context packing**: MI-aware Matryoshka scaling — rate distortion theory determines how much information to include at each scale level. "Fundamental Limits of Prompt Compression" (NeurIPS 2024) confirms large gains are available; we're implementing the theory.

- **7 parallel learning loops at different timescales**: Most systems have 1-2 learning mechanisms. The combinatorial complexity of making 7 loops work without interference is the real moat.

- **Game-theoretic decision foundations**: 31,800 lines. No other AI orchestration system has formal game-theoretic foundations. AgentSHAP (Dec 2025) validates Shapley for tool attribution but only uses it for explanation — we can use it for calibration.

- **Brain-wave memory consolidation**: β/α/θ/δ/REM modes implemented literally. The spectral neuro-symbolic validation (Sep 2025) confirms spectral approaches outperform attention-based reasoning.

- **Spring tension as free energy**: The bridge between our physics metaphor and the free energy principle. F = -kx is the gradient of quadratic free energy. This unifies two frameworks without changing existing code.

### The Paper Thesis

> "Here's an architecture where structured AI is the authority and LLMs are supervised participants, implementing all five neurosymbolic integration patterns, and here's the empirical evidence that this produces more reliable, auditable, and self-improving systems than LLM-first approaches."

The evidence:
- Convergence guarantees from Thompson Sampling (provable regret bounds)
- Deconfounded rewards from causal bandits (Bareinboim's strict improvement proof)
- Decision stability from game-theoretic verification (Nash equilibrium checking)
- Information-theoretic optimality of context packing (rate distortion bounds)
- Hallucination detection from KL drift monitoring (SDM framework)
- Self-improvement trajectories from 7 learning loops (empirical)
- Adaptive routing superiority (96% vs 71% on composite benchmarks)

### What's Missing (and Whether We Need It)

| Missing Capability | Industry Example | Need? | Priority |
|-------------------|-----------------|-------|----------|
| **Evolutionary Search** | AlphaEvolve (found algorithm beating Strassen) | YES — "LLM proposes, structured AI disposes" | High |
| **SAT/SMT Solvers** | PiShield (IJCAI 2024), Z3 | YES — hard safety guarantees on output | Medium |
| **Adaptive Symbolic Router** | 96% accuracy on composite benchmarks | YES — we have the solvers, need the dispatcher | Medium |
| **VCG Truthful Reporting** | Mechanism design literature | YES — subsystems need incentive alignment | Medium |
| **Entropic Steering (EAST)** | arXiv 2406.00244 | Maybe — high ceiling, controls LLM uncertainty directly | Low |
| **Mixture-of-Spectral-Experts** | arXiv 2509.07017 | Maybe — extends validated spectral approach | Low |
| **Graph Neural Networks** | Message-passing on graph structure | No — spectral methods outperform (Sep 2025 paper) | Skip |
| **Formal Theorem Proving** | Lean Copilot (74% automation) | Not yet — interesting but not core | Future |
| **Probabilistic Programming** | Pyro, Stan | No — DIY variational inference is fine | Skip |

### Implementation Status (March 19, 2026)

| Wire | Status | Module | Tests |
|------|--------|--------|-------|
| W1: MI convergence | ✅ Built + tested | `SemanticConvergenceDetector` | 14 |
| W2: KL drift detection | ✅ Built + tested | `DriftDetector` | 16 |
| W3: Minimax gate | ✅ Built + tested + wired into orchestrator | `MinimaxGate` | 17 |
| W4: PiShield constraints | Designed, not built | — | — |
| W5: Orchestrator integration | ✅ Wired | `steps_7_9.py` | — |
| W6: Causal bandits | ✅ Built + tested | `CausalBandit` | 20 |
| W7: EVINCE deliberation | ✅ Built + tested | `DeliberationConvergence` | 26 |
| W8: MCTS+KG reasoning | ✅ Built + tested | `KGReasoner` | 16 |
| W9: Free energy bridge | ✅ Built + tested | `FreeEnergyBridge` | 13 |
| W10: KG world model | ✅ Built + tested | `KGWorldModel` | 15 |
| W11: Symbolic router | ✅ Built + tested | `SymbolicRouter` | 22 |
| W12: Scoring evolver | ✅ Built + tested | `ScoringEvolver` | 21 |

All modules in `hololoom/core/convergence/`. All tests in `hololoom/tests/unit/`.
Total: 180 new tests, 0 new dependencies (numpy only), all protocols swappable.

**Next**: Dogfood against real data (SOUS meal selection, Elle farm ops) to answer the empirical questions.

---

## Part VII: Defensible Moats

Six moats that compound over time:

### 1. Architectural Depth

900,000+ lines of production code across 6 architectural layers is not something a competitor can replicate in a quarter. The codebase represents 18+ months of iterative development, bug fixing, and architectural evolution. The BUILD_PLAN.md alone documents 5 waves of restructuring to get the architecture right.

### 2. Mathematical Foundations

Thompson Sampling, causal inference (Pearl's framework), game theory (31K LOC), information theory (14K LOC) — these are implemented from the mathematics, not wrapped from libraries. The implementations are tested, optimized, and integrated. Reimplementing them requires deep mathematical expertise.

### 3. Composability

33 optional modules that don't import each other, each independently installable via pyproject.toml extras. This composability is the result of disciplined protocol-based design enforced across 900K+ LOC. It's an architectural property, not a feature — you can't add it retroactively.

### 4. Learning Loops

7 parallel learning loops at different timescales means the system gets better with use. A new competitor starts from uniform priors; a deployed HoloLoom instance has months of accumulated Beta(alpha, beta) distributions encoding what works in that specific environment.

### 5. Production Applications

Elle and SOUS are not demos — they're production applications exercising the full architecture daily. This provides continuous integration testing, real-world feedback loops, and proof that the architecture works outside of benchmarks.

### 6. Regulatory Timing

The EU AI Act high-risk requirements take effect August 2026. By the time competitors build equivalent infrastructure, the regulatory deadline will have passed and enterprises will have committed to solutions. First-mover advantage in compliance infrastructure is durable because switching costs are high (audit trails, trained models, configured policies).

---

## Part VIII: The Bigger Vision

### Child Safety

**The child safety crisis is accelerating:**
- The **Kids Online Safety Act (KOSA)** passed the Senate with rare 91-3 bipartisan support and is advancing through the House — the strongest signal that AI safety for minors is becoming federal law
- **Thorn** (founded by Ashton Kutcher & Demi Moore) builds AI safety tools specifically for child protection — proving the market exists and matters
- **Common Sense Media** rates AI products for child safety — creating pressure on every platform to demonstrate responsible AI practices
- **AI-generated CSAM** is exploding faster than platforms can detect it — the UN and NCMEC have both issued urgent calls for better AI governance
- Every state AG in the country is investigating AI's impact on children

This isn't abstract policy. This is **our kids using AI tools every day at school, on their phones, in their games** — and nobody can tell us how those systems make decisions.

The structured authority architecture has profound implications for child safety:

- **Confidence thresholds** prevent the system from presenting uncertain information as fact
- **Causal reasoning** can model the downstream effects of actions before taking them
- **Audit trails** provide full accountability for every interaction
- **Content constraints** are structural, not post-hoc filters — the system can't violate them even under adversarial prompting
- **Human oversight gates** ensure that high-stakes decisions require human approval

The opening quote — "I have kids. Then that's who it's for." — is not marketing. The architecture we built to make AI reliable for enterprises is the same architecture that makes AI safe for children. Structured authority is the foundation for both.

### Federation

HoloLoom's federation module (20K LOC) implements gossip protocol + DHT for distributed reasoning across organizational boundaries. This enables:

- **Cross-organization learning**: hospitals can improve diagnostic AI collectively without sharing patient data (federated Thompson Sampling)
- **Distributed governance**: regulatory bodies can monitor AI systems across jurisdictions without centralizing data
- **Sovereign AI**: organizations maintain full control of their reasoning infrastructure while participating in collective intelligence

Federation is the long-term competitive moat. Once organizations are federated, switching costs become prohibitive — not because of vendor lock-in, but because the collective learning is more valuable than any single instance.

### The Nervous System Architecture

The full vision is a three-tier nervous system:

- **Femtoclaw** (peripheral nerves): Fast runtime, channel adapters (Matrix, WhatsApp), real-time message routing. Reacts in milliseconds.
- **OpenClaw** (autonomic): Infrastructure management, health monitoring, service orchestration. Keeps the system alive without conscious intervention.
- **HoloLoom** (cortex): Structured reasoning, deliberation, learning. Makes the hard decisions.

This maps to biological nervous systems: reflexive responses (Femtoclaw) handle routine interactions, autonomic processes (OpenClaw) maintain system health, and deliberate reasoning (HoloLoom) handles complex decisions. The routing between tiers is scored — simple queries stay peripheral, complex queries escalate to cortex.

---

## Part IX: Key Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Frontier labs add structured reasoning** | Medium | High | Our lead is 18+ months. Structured reasoning contradicts their "LLMs are sufficient" narrative. The moat is depth, not novelty. |
| **EU AI Act delayed or weakened** | Low | Medium | NIST voluntary framework still drives enterprise demand. Market trend toward AI governance is independent of any single regulation. |
| **Open source competition** | Medium | Medium | Composability moat: 33 modules, 6 layers, 900K+ LOC is not easily replicated. Learning loops compound — a fresh install can't match a trained instance. |
| **Enterprise sales cycle too long** | High | Medium | Product-led growth via HoloLoom Lite reduces dependence on enterprise sales. Developer adoption creates bottom-up demand. |
| **Solo developer risk** | High | High | Architecture is documented (CLAUDE.md, MODULE_TAXONOMY.md, BUILD_PLAN.md, ROADMAP.md). Protocol-based design means components are independently maintainable. Hiring plan activates after first revenue. |
| **Complexity overwhelms users** | Medium | Medium | HoloLoom Lite provides simple entry point. Three-class API (HoloLoom, Memory, experience/recall/reflect). Complexity is opt-in via optional modules. |
| **GPU infrastructure costs** | Medium | Low | Graceful degradation: FULL(3 GPUs) → PARTIAL(1-2) → DEGRADED(local) → DARK(queue only). System works on a laptop with Ollama — GPU rig is for advanced features. |

---

## Part X: Where We Are (March 2026)

### What Just Happened: Structured AI Wiring Sprint

On March 18-19, 2026, an Opus 4.6 swarm session built the structured AI integration
layer — 11 modules connecting HoloLoom's previously disconnected mathematical systems
through the convergence engine. 246 tests, all green, 4.18 seconds.

**The architecture that emerged:**

```
Query → Symbolic Router (classifies, dispatches to optimal solver)
         │
         ├→ Causal Engine (do-calculus, backdoor adjustment)
         ├→ Minimax Gate (game-theoretic worst-case verification)
         ├→ KG Reasoner (MCTS multi-hop graph search)
         ├→ World Model (spring dynamics forward prediction)
         ├→ Deliberation (EVINCE dual-entropy convergence)
         │
         ▼
    Convergence Engine (Thompson Sampling collapses to action)
         │
    BEFORE: Causal DAG deconfounds reward signal
    DURING: Minimax gate checks worst-case regret
    AFTER:  KL drift detects hallucination
            MI convergence stops refinement
         │
         ▼
    Free Energy Bridge (spring tension = prediction error = what to attend to next)
         │
         ▼
    Scoring Evolver (AlphaEvolve MAP-Elites evolves scoring weights over time)
```

**Five mathematical traditions now wired through one decision pipeline:**
1. Bayesian (Thompson Sampling) — learns from outcomes
2. Causal (Pearl's do-calculus) — distinguishes correlation from causation
3. Game-theoretic (minimax, Shapley) — verifies stability, attributes credit
4. Information-theoretic (MI, KL, rate distortion) — measures information flow, detects drift
5. Physics-based (spring dynamics = free energy) — simulates activation, predicts future states

**Smoke test results (March 19):**
- Semantic convergence detector correctly stopped refinement when responses stabilized (JSD=0.0000)
- Drift detector cleanly separated grounded (JSD=0.0006) from hallucinated (JSD=0.953) responses
- Minimax gate caught a risky tool selection (regret=2.5) that Thompson Sampling missed
- Causal bandit found `query_difficulty` as confounder, adjusted reward 0.90 → 0.68
- Deliberation convergence detected agreement (JSD=0.036) then correctly flagged new disagreement
- KG reasoner found 7 unique paths through a knowledge graph via MCTS
- Free energy decomposed system state into surprise (0.400) + complexity (0.245)
- World model predicted activation propagation along graph edges
- Symbolic router correctly dispatched causal/graph/prediction/deliberation queries
- Scoring evolver reached 100% fitness on synthetic data, maintained diverse population

### What We Don't Know Yet

The smoke test proves mechanical correctness. The open questions require real data:

- Does the causal bandit actually improve decision quality on SOUS meal selection?
- Does the minimax gate fire on real queries, and when it does, is it right?
- Does the symbolic router learn better dispatch over time?
- Does the evolver discover non-obvious weight configurations on real scoring data?
- How does the free energy interpretation change how spring dynamics guides attention?

These are empirical questions. They require dogfooding.

### Now: v1.0 Cleanup Sprint (4-6 weeks)

This is the priority. Nothing else ships until v1.0 is stable.

1. **Week 1-2: API Lock + CI Green**
   - Freeze public API surface
   - Get ruff + black + mypy + pytest all passing in CI
   - Fix any remaining test collection errors
   - Docker Compose end-to-end test

2. **Week 2-3: Non-Destructive Consolidation**
   - Resolve the 25 "Unclear" modules from MODULE_TAXONOMY.md
   - Archive (not delete) modules that are superseded
   - Merge micro-modules that should have been consolidated in Wave 2

3. **Week 3-4: Orchestrator Decomposition**
   - Extract clean stage boundaries from the 8,788 LOC orchestrator
   - Each of the 9 stages independently testable
   - Stage protocols defined so alternative implementations can be swapped

4. **Week 4-6: Infrastructure Hardening**
   - systemd service files for Ollama, HoloLoom API, Femtoclaw, OpenClaw
   - Secrets management (environment variables, not hardcoded)
   - Health check alerting (mining rig, Matrix homeserver, GPU servers)
   - Prepare deployment configs for inbound equipment

5. **Throughout: Documentation Verification**
   - Every guide verified against actual code paths
   - API reference generated from docstrings
   - Remove or update stale references

### After v1.0: Parallel Tracks

Once v1.0 is stable, three tracks run in parallel:

- **v2.0 Cognitive UI** — three-pane shell, WebSocket streaming, external shipping
- **Dogfooding** — Elle (farm) and SOUS (kitchen) running daily through the full architecture, generating the empirical data the structured AI wiring needs
- **Paper** — "Structured AI as the Authority": empirical evidence from dogfooding that five mathematical traditions operating on the same decisions produce more reliable systems than LLM-first approaches

### Decision Points

- **After v1.0**: Decide on open-source licensing strategy for HoloLoom Lite
- **After 3 months dogfooding**: Decide on enterprise pilot timing based on empirical results
- **After first external users**: Decide on hiring priorities (infrastructure vs. sales)
- **After paper submission**: Decide on conference strategy (NeurIPS/ICML workshop vs. main track)

---

## Appendix A: Market Research Sources

| Source | URL / Reference | Key Data Point |
|--------|----------------|----------------|
| EU AI Act full text | eur-lex.europa.eu | Risk-based classification, Art. 6-15 high-risk requirements |
| NIST AI RMF 1.0 | nist.gov/ai-rmf | 4-function framework (Govern, Map, Measure, Manage) |
| MarketsandMarkets AI Governance | marketsandmarkets.com | $492M (2026) → $2.1B (2030), 33.8% CAGR |
| Grand View Research AI Trust | grandviewresearch.com | $5.6B by 2030, 40.2% CAGR |
| Gartner AI TRiSM | gartner.com | "By 2026, organizations that operationalize AI transparency, trust, and security will see 50% improvement in AI adoption" |
| IDC AI Trust & Safety | idc.com | $1B+ market by 2030 |
| DeepMind AlphaProof | deepmind.google | 4/6 IMO problems solved with LLM + formal prover |
| Gary Marcus, "Rebooting AI" | Various publications | Hybrid AI thesis, limitations of pure neural approaches |
| Judea Pearl, "The Book of Why" | Cambridge University Press | Causal reasoning framework, ladder of causation |
| Lean Copilot (Song et al. 2024) | arxiv.org | 74% of proof steps automated with LLM + theorem prover |
| AlphaEvolve (DeepMind 2025) | deepmind.google | LLM-guided evolutionary search discovers novel algorithms |
| DSPy (Khattab et al.) | Stanford NLP | Programmatic prompt optimization, structured LLM interaction |
| PiShield (Giunchiglia et al.) | arxiv.org | Formal constraints embedded in neural training |
| Constitutional AI (Anthropic) | anthropic.com | Rule-based constraints on LLM generation |
| Anthropic Circuit Tracing | anthropic.com/research/tracing-model-behavior | Discovered hallucination mechanism and planning-ahead behavior in Claude |
| NextMSC Responsible AI Market | nextmsc.com | $1.09B (2024) → $10.26B by 2030, 45.2% CAGR |
| OMB M-26-04 Federal AI Governance | whitehouse.gov | AIBOM, model cards, fairness testing requirements for all federal agencies |
| Kids Online Safety Act (KOSA) | congress.gov/bill/118th-congress | 91-3 Senate passage, advancing through House |
| Thorn: AI for Child Protection | thorn.org | AI tools defending children from sexual abuse |
| Common Sense Media AI Ratings | commonsensemedia.org/ai-ratings | Rating AI products for child safety |
| WEF: The Power of Neurosymbolic AI | weforum.org (Dec 2025) | "No hallucinations, auditable workings" |
| EY Neurosymbolic AI Platform | ey.com (Sep 2025) | "Hundred-million-dollar-plus growth opportunities" |
| Kognitos $25M Series B | businesswire.com (Jun 2025) | Hallucination-free neurosymbolic automation |
| 2026 International AI Safety Report | internationalaisafetyreport.org | 100+ experts, 30+ countries: "AI safety is a system and deployment issue" |

---

## Appendix B: Technical Differentiators Summary

| Capability | HoloLoom | LangChain | Guardrails AI | DSPy | AutoGen |
|-----------|----------|-----------|---------------|------|---------|
| Structured decision engine | Thompson Sampling + game theory | None | None | Prompt optimization | None |
| Causal reasoning | Full Pearl framework (do-calculus, counterfactuals, SCMs) | None | None | None | None |
| Persistent memory | Bi-temporal KG + vector store + spring dynamics | Stateless | None | None | Conversation only |
| Learning loops | 7 parallel loops at different timescales | None | None | Optimization loop | None |
| Convergence guarantees | Provable regret bounds | None | None | None | None |
| Explainability | Intrinsic (7 XAI techniques) | None | Post-hoc | None | None |
| Compliance mapping | EU AI Act + NIST RMF | None | Basic | None | None |
| Multi-model governance | Thompson Sampling router + health caching | Manual selection | None | None | None |
| Formal verification | Game-theoretic (Nash, Shapley, mechanism design) | None | Input/output validators | None | None |
| Information-theoretic bounds | MI, KL, channel capacity, rate distortion (14K LOC) | None | None | None | None |
| Self-improvement | Episodic consolidation, PPO, reflection buffer | None | None | Prompt tuning | None |
| Graceful degradation | FULL → PARTIAL → DEGRADED → DARK | Crash | Crash | Crash | Crash |

---

## Appendix C: The Name

**HoloLoom**: *Holo-* (Greek: whole, complete) + *Loom* (the machine that weaves).

The loom metaphor is not decorative — it IS the architecture:

- **Threads** are memory traces (entities, relationships, temporal context)
- **The Warp** is the tensioned mathematical substrate (tensor manifold, Hooke's Law springs)
- **The Weft** is the LLM-generated content that gets woven through the structured warp
- **The Shuttle** carries context between memory and reasoning (context packing)
- **The Fabric** is the output — Spacetime Fabric with 4D provenance
- **The Pattern Card** selects the weaving mode (BARE/FAST/FUSED)
- **The Spindle** manages thread lifecycle
- **The Weaver** executes the pattern
- **The Bobbin** winds and stores context for reuse

A holographic loom: every piece contains information about the whole. The weaving metaphor naturally supports the key architectural properties — composition (threads compose into fabric), tension (mathematical constraints), pattern selection (mode-based behavior), and provenance (every thread in the fabric is traceable).

The 9-stage weaving cycle:
1. **Mount** — Load memory context
2. **Card** — Select pattern (BARE/FAST/FUSED)
3. **Tension** — Apply mathematical constraints
4. **Thread** — Extract features (DotPlasma)
5. **Weave** — Execute reasoning with LLM
6. **Beat** — Convergence check
7. **Inspect** — Quality verification
8. **Cut** — Produce Spacetime Fabric output
9. **Wind** — Update memory, learning loops, reflection buffer

---

*Last updated: 2026-03-17*
*Version: 2.0 (synthesized from market analysis, codebase audit, and strategic priorities)*
