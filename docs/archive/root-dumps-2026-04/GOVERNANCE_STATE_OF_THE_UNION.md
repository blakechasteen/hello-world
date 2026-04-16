# HoloLoom Governance — State of the Union
**March 12, 2026**

---

## Executive Summary

HoloLoom has ~15,000+ lines of governance code across 11 systems. The alignment framework on the hot path is genuine, fast, and well-tested. But the honest picture reveals three structural problems:

1. **The hot path is heavy but shallow** — every query runs through 3 separate SafetyGuardrails evaluations and 5 AuditTrail entries, but the detection logic behind them is regex and keyword matching, not semantic analysis.
2. **86% of governance code is behind flags nobody flips** — the policy engine, RBAC, departments, adaptive learning, and deep thinking governance are all built but disconnected from production.
3. **Thompson Sampling has metastasized** — at least 17 independent implementations of the same Beta(α,β) prior exist across the codebase, with a shared base protocol (`bandits/ts_base.py`) that nothing uses.

**Grade: B- infrastructure, D+ detection intelligence, C overall.**
The plumbing is real. The intelligence inside it is placeholder. The wiring between systems is incomplete.

---

## Part I: What Actually Runs (The Hot Path)

### Architecture: Triple Safety Gate

Every query through `agentic_api.py` passes through **three separate SafetyGuardrails instances** and logs **five AuditTrail entries** minimum:

```
POST /query
  │
  ├─ [Gate 1] API-level SafetyGuardrails.evaluate()     ← state.safety_guardrails
  │   └─ AuditTrail.log_decision()                       ← state.audit_trail
  │
  ├─ orchestrator.reason(query)
  │   ├─ [Gate 2] AgenticSafetyAdapter.gate_reasoning()  ← orchestrator.guardrails
  │   │   └─ AuditTrail.log_decision()
  │   │
  │   └─ weave(query)
  │       ├─ ConscienceAdapter.gate_reasoning()           ← conscience layer
  │       │   └─ AuditTrail.log_decision()
  │       │
  │       └─ [Gate 3] SafetyGuardrails.evaluate()        ← step 8 tool gate
  │           └─ AuditTrail.log_decision()
  │
  └─ AuditTrail.log_decision()                           ← completion log
```

**Finding: Duplicate singleton instances.** The API layer creates its own SafetyGuardrails + AuditTrail + DeceptionDetector at startup. The orchestrator creates a *second* set via `config_init.py`. The AgenticOrchestrator creates a *third* DeceptionDetector. These instances share no state — deception goal tracking is fragmented across separate detector objects.

**Finding: Dead DeceptionDetector.** `state.deception_detector` is instantiated at `agentic_api.py:648` but **never called by any endpoint**. It sits on the state object doing nothing.

### Hot Path Components: Honest Assessment

| Component | Lines | Overhead | What It Actually Does |
|-----------|-------|----------|----------------------|
| **SafetyGuardrails** | 1,114 | 0.039ms | **Real gating.** HMAC-SHA256 override tokens, thread-safe state, epistemic confidence adjustment. The `evaluate()` method has genuine branching logic. CRITICAL risk always requires human approval (cannot be bypassed). |
| **Audit Trail** | 688 | 0.015ms | **Real logging.** SHA-256 hash chain (tamper-detection, not tamper-resistance — no HMAC key on the chain). Provenance DAG with DFS traversal. JSONL persistence. Query methods are O(n) linear scans. |
| **Deception Detection** | 556 | 0.034ms | **Placeholder.** "Deception score" is Jaccard word overlap between expected and actual strings. Code comments admit: "In practice, would use semantic embeddings." Goal transparency is structural bookkeeping, not alignment analysis. |
| **Instrumental Convergence** | 699 | — | **Placeholder detection, real plumbing.** Resource bounds mechanism is genuine (hard/soft/rate limits with time windows). But ships with **no default bounds** (removed Nov 2025). Power-seeking detection is keyword substring counting against ~20 pattern words. Self-modification detection matches 7 hardcoded substrings. |
| **Monitoring** | 854 | — | **Production quality.** Sliding window latency with p50/p95/p99. Prometheus export with proper HELP/TYPE comments. Alert cooldown (5min). But monitoring is observation, not governance — it records events, doesn't prevent them. |

### What's Strong

- **SafetyGuardrails.evaluate()** is the crown jewel — multi-layer gating, thread safety, epistemic confidence integration, proper HMAC override mechanism
- **Audit Trail** with hash chain is real infrastructure — tamper-evident logging with provenance DAG
- **Monitoring** is genuinely production-ready Prometheus integration
- **Total overhead < 0.103ms** — governance that's invisible to latency stays turned on

### What's Weak

- **Adversarial detection is regex-only** — 7 injection patterns, 3 jailbreak patterns, 3 exhaustion patterns, plus a 50K char length check. Trivially bypassable with Unicode substitution, misspellings, or indirect phrasing. No semantic analysis.
- **Deception detection is word overlap** — if a deceptive system uses the same words while lying, score = 0.0 ("no deception"). If an honest system paraphrases, score = high ("deception!"). The detector cannot distinguish deception from rephrasing.
- **Power-seeking detection is grep** — concatenates action descriptions, counts keyword hits ("allocate", "admin", "control"), divides by constants. Would be defeated by any non-trivial adversary.
- **Epistemic confidence thresholds (0.3, 0.6) are round numbers** — never calibrated against real usage
- **Risk-to-category mappings are first-draft assignments** — READ is LOW but QUERY is SAFE? Never tuned.

---

## Part II: What's Built But Gated

### Policy Engine (unified.py — 1,235 lines)

**Status: Architecturally impressive, functionally empty.**

The policy engine has a full neural network (transformer blocks, cross-attention, LoRA adapters, motif-gated MHA) that makes tool selection decisions. It IS called from the weaving orchestrator (`weaving_orchestrator.py:1683`).

**The problem: the neural network has random weights and is never trained.**

- No training loop exists anywhere in the policy directory
- No saved checkpoints, no `.pt` files, no model loading
- `PPOAgent.update()` returns hardcoded zeros — it's a test stub
- All Linear layers use default PyTorch random initialization
- The tool selection probabilities from the neural network are random noise passed through softmax

The Thompson Sampling bandit DOES update within a session (from coherence metrics), but **resets to uniform Beta(1,1) priors on every restart**. No persistence.

**Semantic nudging is mathematically a no-op.** The nudge applies `p_i' = p_i × (1 + nudge_weight × deficit)` uniformly to all tools. After renormalization, the relative ordering is identical. The code comment admits: "In a full implementation, this would use learned tool→semantic mappings."

~30% of `unified.py` (lines 867-1235) are test stubs — minimal implementations that exist solely so test files can import them.

**Verdict: A beautifully engineered empty vessel.** The architecture is ready for training. No training has occurred.

### Deep Thinking Governance (gate + tier + deliberation — ~1,200 lines)

**Status: Functional but isolated, with operational bugs.**

| Component | Lines | Assessment |
|-----------|-------|------------|
| **MilestoneGate** | 263 | Real LLM-based checkpoint review with Thompson priors. Calls `/v1/chat/completions` via httpx. |
| **TierManager** | 183 | Real GPU probing via async health checks. FULL/PARTIAL/DEGRADED/DARK. |
| **Deliberation Engine** | 424 | Propose→Challenge→Resolve 3-phase cycle. Working but has bugs. |
| **Deferred Queue** | 214 | Priority queue with staleness boost and JSON persistence. |
| **Sleeptime Scheduler** | 272 | Idle-time dream jobs. Aspirational. |

**Bugs that would break deployment:**

1. **GPU endpoint defaults contain literal string `"MINING_RIG_IP"`** — `config.py` does NOT read from environment variables. Health probes will fail DNS resolution and the system will be stuck in DEGRADED/DARK.

2. **Deferred queue is dead code** — DARK tier gate returns `deferred=True` but nobody calls `DeferredQueue.push()`. Deferred work is flagged and forgotten. No retroactive review occurs.

3. **Gate parse failure = silent proceed** — unparseable LLM response defaults to `verdict="proceed", confidence=0.3`. This is below `defer_below` (0.5) so it's flagged as deferred — which, per bug #2, means nothing happens.

4. **DARK tier crashes the deliberation engine** — `deliberate()` doesn't check for DARK tier. It calls `best_available_endpoint()` which returns `None`, then passes `None` to httpx, which crashes.

5. **No connection pooling** — `DeepClient` creates a new `httpx.AsyncClient` per request. Fine for occasional gate reviews, wasteful under load.

6. **Deliberation timing is zeroed** — `propose_elapsed_s = proposal.raw and 0` always evaluates to `0` (Python `and` semantics). Per-phase timing was never wired up.

7. **Bleed rate naming mismatch** — critic truncation uses `critic_to_synth_bleed` (0.3) instead of `plan_to_critic_bleed` (1.0). Numerically accidentally correct, but refactoring config values would silently break the anti-groupthink property.

**No heartbeat.py exists in `deep_thinking/`.** The MEMORY.md reference to a heartbeat bug fix belongs to a different module. The `require_approval` fix may have been applied elsewhere.

### RBAC & Policy Governance (policy_governance.py — 620 lines)

**Status: Complete, tested, and completely orphaned.**

A full RBAC system with 5 roles (ADMIN→RESTRICTED), topic governance (allow/forbid/restrict per agent), and 3 policy templates (development/production/enterprise). Has a 574-line test file. Tests pass.

**Never imported by the orchestrator, API, or agentic core.** Only imported by `collaborative_agents.py` (multi-agent system) and its own test file. Zero production callsites.

### Departments (~94 files, ~40K lines)

**Status: 10% load-bearing, 90% aspirational.**

The structural core works:
- `DepartmentProtocol` — real `@runtime_checkable Protocol` with 7 abstract methods
- `DepartmentRegistry` — real registry with 4 indexes, dependency graph, load-balanced routing, health checks
- `ContextDepartment` — real concrete implementation

**But:**
- Cross-department context is shared in-memory dicts, not message passing
- B2B tiers (Bronze/Silver/Gold/Platinum) are defined in `sla_definitions.py` but never enforced
- No customer database, no tier assignment, no SLA violation alerting
- `governance_config.json` **does not exist** (searched entire tree)
- ~36K lines in subdirectories (performance/, verification/, privacy/) are not demonstrably wired into any running code path

### Adaptive Learning (continuous_validator + adaptive_updater + pattern_miner — ~1,449 lines)

**Status: Governance scaffolding with no physical deployment mechanism.**

The orchestration logic is complete:
- SHADOW → AB_TEST → GRADUAL rollout with validation at each stage
- Regression detection (>2% overall drop, >5% per-complexity drop)
- Pattern versioning (keeps last 10 versions)
- Auto-rollback on regression

**But `_install_patterns()` is a stub that just logs.** The traffic split value is updated (0.0→0.10→0.50→1.0) but nothing reads it. `_get_current_patterns()` returns `[]` with a comment "to be implemented during integration." The rollback correctly manages state but the physical revert is unimplemented.

The hourly validation is NOT self-scheduling — `validate_hourly()` is a plain async method that must be called externally.

---

## Part III: The Thompson Sampling Proliferation

### 17+ Independent Implementations

| # | Location | Class Name | Sampling Over | Genuinely Different? |
|---|----------|-----------|---------------|---------------------|
| 1 | `core/policy/thompson_sampling.py` | `TSBandit` | Tools | Yes — numpy vectorized, 3 strategies |
| 2 | `apps/server/model_router.py` | `ModelBandit` | Models | **Yes** — geodesic certainty (Fisher-Rao), speed blending |
| 3 | `apps/xterminator/thompson_bandit.py` | `ThompsonBandit` | Fix strategies | Yes — convergence detection, persistence |
| 4 | `.claude/skills/domain/ritual/thompson_router.py` | `ThompsonPrior` | Ritual agents | Copy-paste variant |
| 5 | `core/deep_thinking/gate.py` | `VerdictPrior` | Gate verdicts | Simplified — no actual sampling |
| 6 | `context_packing/learning.py` | `ThompsonSampler` | MI budgets | Yes — budget scaling, not arm selection |
| 7 | `redteam/bandit.py` | `BanditArm` | Attack strategies | Asymmetric updates |
| 8 | `agentic/expert_router.py` | `ThompsonPrior` | Agents | Copy-paste of #4 |
| 9 | `shuttle/eggroll_shuttle.py` | `ThompsonSampler` | Hyperparameters | Standard |
| 10 | `collaboration/ux_learning.py` | `BetaPrior` | UX features | Standard |
| 11 | `core/warp/math/smart_operation_selector.py` | `ThompsonSamplingLearner` | Math operations | Standard |
| 12 | `alignment/constitutional_critique.py` | `PrincipleWeight` | Principle weights | Asymmetric (severity) |
| 13 | `alignment/automated_auditor.py` | `ThresholdPrior` | Anomaly thresholds | Standard |
| 14 | `conscience/core.py` | (inline) | Wisdom patterns | Inline `betavariate()` |
| 15 | `redteam/swarm/learning.py` | `ThompsonSamplingPrior` | Swarm attacks | Standard |
| 16 | `core/recursive/full_learning_loop.py` | `ThompsonPriors` | Recursive tools | Standard |
| 17 | `core/recursive/hot_patterns.py` | (inline) | Pattern selection | Heat-score variant |

Plus ~10 additional inline `random.betavariate()` call sites across the codebase.

### The Shared Base Nobody Uses

`hololoom/bandits/ts_base.py` (171 lines) defines a proper Protocol hierarchy: `ThompsonSampler` → `DiscreteSampler` / `ContextualSampler` / `ContinuousSampler`. It exists. Essentially nothing implements it.

### What's Actually Different vs Copy-Paste

**Genuinely novel (3):**
1. **Model Router** — Fisher-Rao geodesic certainty on Beta manifold, speed/quality blending, exploration bonus
2. **Core Policy TSBandit** — numpy vectorized sampling, 3 selection strategies
3. **Context Packing** — samples success rate to scale a continuous budget, not select a discrete arm

**Asymmetric updates (2):**
4. **Constitutional Critique** — success adds severity weight, failure adds 0.1
5. **RedTeam Bandit** — success weighted by severity, failure always +1.0

**Copy-paste with cosmetic differences (12):** The remaining implementations follow the same ~15-line pattern with different field names.

### Recommendation

A single concrete `BetaArm` dataclass with `sample()`, `expected_value()`, `update(reward, weight)`, `to_dict()`, `from_dict()` would replace at least 12 implementations. The 3 genuinely different ones would extend or compose it.

---

## Part IV: The Learning Loop — Is It Closed?

### Closed (production data flows back):

| Loop | Mechanism |
|------|-----------|
| **Hot pattern retrieval weights** | `record_usage()` → heat score update → `AdaptiveRetriever.update_weights()` |
| **Thompson priors (per session)** | Tool confidence → `ThompsonPriors.update()` → future tool selection |
| **Policy adapter weights** | Per-query Laplace-smoothed success rates → adapter preference |

### Open (training data goes nowhere):

| Gap | What's Missing |
|-----|---------------|
| **Pattern miner → classifier** | `_install_patterns()` is a stub. Mined patterns are never installed. |
| **Validation → correction** | Regressions detected and logged but no handler consumes alerts. |
| **Cross-session persistence** | Thompson priors reset on restart. `save_learning_state()` must be called explicitly. |
| **Model router ↔ recursive learning** | Completely separate Thompson implementations with no feedback path between them. |
| **Heat decay is non-compounding** | Multi-hour gaps apply only one 0.95× decay step, not 0.95^n. |

**Bottom line:** The loop is closed at the micro level (per-query bandit updates within a session), half-closed at the meso level (hot pattern retrieval weights), and open at the macro level (pattern deployment, cross-session persistence, model routing feedback).

---

## Part V: Structural Issues

### 1. Triple Evaluation, Fragmented State

Three SafetyGuardrails instances evaluate the same query independently. Two AuditTrail instances may write to different directories. Three DeceptionDetector instances track goals separately. This isn't defense-in-depth — it's accidental duplication from organic growth.

### 2. The Build-vs-Wire Gap

| System | Lines Built | Lines Wired | Wired % |
|--------|-------------|-------------|---------|
| Alignment framework | ~3,900 | ~3,900 | 100% |
| Policy engine | ~1,980 | ~0 (random weights) | 0% |
| Deep thinking | ~1,350 | ~0 (behind flag) | 0% |
| RBAC / policy governance | ~620 | ~0 (orphaned) | 0% |
| Departments | ~40,000 | ~3,700 | 9% |
| Adaptive learning | ~1,449 | ~0 (stubs) | 0% |
| Recursive learning | ~2,000 | ~500 (hot patterns) | 25% |
| **Total** | **~51,300** | **~8,100** | **16%** |

### 3. Detection Intelligence vs Infrastructure

The infrastructure layer (gating, logging, monitoring, hash chains, Prometheus export) is B+ quality. The detection layer (adversarial regex, Jaccard deception, keyword power-seeking) is D quality. The gap between them is the gap between "real governance" and "governance theater."

### 4. No Governance Dashboard

Multiple systems hide behind flags (`enable_deep_thinking`, `FullLearningEngine`, deployment strategies, conscience enable) but there's no centralized way to see what's on, what's off, or what's firing. Governance you can't observe is governance you don't have.

---

## Part VI: Recommendations

### Tier 0 — Fix What's Broken

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 0a | **Deduplicate governance singletons** — one SafetyGuardrails, one AuditTrail, one DeceptionDetector, passed through the stack | Small | Eliminates fragmented state, 3x→1x evaluation |
| 0b | **Fix deep thinking GPU endpoint config** — read `MINING_RIG_IP` from env var in `config.py` | Trivial | Unblocks deep thinking deployment |
| 0c | **Fix DARK tier crash** — add tier check before `deliberate()` | Trivial | Prevents crash on no-GPU |
| 0d | **Fix deliberation timing** — `proposal.raw and 0` → actual timing | Trivial | Enables performance monitoring |
| 0e | **Remove dead `state.deception_detector`** from agentic_api.py | Trivial | Reduces confusion |

### Tier 1 — Wire What's Built

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1a | **Create governance manifest** — one file showing what's enabled, what's gated, current thresholds | Small | Visibility into governance state |
| 1b | **Wire deferred queue to gate** — `gate.review()` should `queue.push()` on deferral | Small | Enables retroactive review |
| 1c | **Persist Thompson priors** — serialize on shutdown, restore on startup | Small | Cross-session learning |
| 1d | **Implement `_install_patterns()`** — actually swap classifier patterns | Medium | Closes macro learning loop |

### Tier 2 — Improve Detection Quality

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 2a | **Replace Jaccard deception with embedding similarity** — use existing semantic calculus | Medium | Real deception detection |
| 2b | **Replace keyword power-seeking with behavioral analysis** — track resource acquisition patterns over time | Medium | Real convergence guard |
| 2c | **Calibrate risk thresholds** — run historical queries, tune 0.3/0.6 boundaries | Small | Fewer false positives/negatives |
| 2d | **Add semantic adversarial detection** — embedding-based intent classification | Medium | Bypass-resistant safety |

### Tier 3 — Consolidate Architecture

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 3a | **Unify Thompson Sampling** — single `BetaArm` base, refactor 12+ copy-paste implementations | Medium | DRY, consistent learning primitive |
| 3b | **Audit departments for dead code** — 40K lines should earn their keep | Medium | Reduces maintenance burden |
| 3c | **Wire RBAC into production or archive it** — 620 lines guarding nothing | Small | Honest codebase |
| 3d | **Train the policy network or remove it** — random weights making production decisions | Large | Either real neural policy or honest bandits |

---

## The Bottom Line

HoloLoom's governance has **genuine infrastructure that works** — the safety gating pipeline, audit trail, and monitoring are real, fast, and well-tested. The system correctly blocks high-risk actions, maintains tamper-evident logs, and exports Prometheus metrics.

But it also has **detection logic that doesn't detect** (Jaccard similarity, keyword grep), **a neural policy making random decisions**, **17 copies of the same Bayesian primitive**, and **~43K lines of governance code that never runs**.

The path forward is not more building. It is:
1. **Fix** the broken things (duplicated singletons, dead code, config bugs)
2. **Wire** the good things that are disconnected (deferred queue, pattern deployment, Thompson persistence)
3. **Upgrade** the detection intelligence from string matching to semantic analysis
4. **Consolidate** the Thompson Sampling proliferation into a shared primitive

The foundation is strong. The house needs finishing.
