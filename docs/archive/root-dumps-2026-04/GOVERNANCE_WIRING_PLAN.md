# HoloLoom Governance Wiring Plan
**March 12, 2026 — Stop building, start wiring.**

---

## Dependency Graph

```
Phase 1: Crash Fixes ──────────────────┐
  (5x Small, no deps)                  │
                                        ├──→ Phase 3: Deferred Queue (needs 1.2)
Phase 2: Safety Singleton ─────────────┤
  (2x Medium, no deps)                 ├──→ Phase 5: Routing & RBAC (needs 2.1)
                                        │
Phase 4: TS Consolidation ─────────────┤    (independent)
  (S + L + M, no deps)                 │
                                        │
Phase 6: Config Cleanup ───────────────┘    (independent)
  (2x Small, no deps)
```

Phases 1, 2, 4, 6 can run in parallel. Phase 3 waits on 1.2. Phase 5 waits on 2.1.

---

## Phase 1: Crash Fixes (PR #1)

Zero-risk fixes to things that are currently broken. Each independently shippable.

### 1.1 — Fix GPU endpoint literal strings [S]

**File:** `hololoom/core/deep_thinking/config.py` lines 30-34

```python
# BEFORE (broken — literal string, fails DNS)
gpu_servers: list = field(default_factory=lambda: [
    "http://MINING_RIG_IP:8001",
    "http://MINING_RIG_IP:8002",
    "http://MINING_RIG_IP:8003",
])

# AFTER (reads env var, matches model_router.py pattern)
gpu_servers: list = field(default_factory=lambda: [
    f"http://{os.environ.get('MINING_RIG_IP', '127.0.0.1')}:{port}"
    for port in (8001, 8002, 8003)
])
```

### 1.2 — Fix DARK tier crash in deliberation engine [S]

**File:** `hololoom/core/deep_thinking/deliberation.py`, `_resolve_endpoint()` (~line 280)

When `best_available_endpoint()` returns `None`, return a synthetic deferred result instead of passing `None` to httpx. Mirror the pattern already used in `gate.py:140-146`.

### 1.3 — Fix deliberation timing bug [S]

**File:** `hololoom/core/deep_thinking/deliberation.py` line 152

```python
# BEFORE (always 0 — Python `and` semantics)
propose_elapsed_s=proposal.raw and 0

# AFTER (actual timing)
propose_elapsed_s=propose_elapsed  # captured from time.perf_counter() around propose call
```

Add `t0 = time.perf_counter()` before the propose call and `propose_elapsed = time.perf_counter() - t0` after.

### 1.4 — Fix hot pattern heat decay non-compounding [S]

**File:** `hololoom/core/recursive/hot_patterns.py`, `_apply_decay_if_needed()` lines 239-254

```python
# BEFORE (one decay step regardless of elapsed time)
if elapsed >= self.decay_interval:
    record.access_count = int(record.access_count * self.decay_rate)

# AFTER (compound decay for missed intervals)
if elapsed >= self.decay_interval:
    n_intervals = int(elapsed / self.decay_interval)
    decay_factor = self.decay_rate ** n_intervals
    record.access_count = max(0.0, record.access_count * decay_factor)
    record.success_count = max(0.0, record.success_count * decay_factor)
    self.last_decay = now
```

Keep counts as floats internally; cast to int at API boundaries only.

### 1.5 — Fix semantic nudging no-op [S]

**File:** `hololoom/core/policy/semantic_nudging.py` lines 436-445

```python
# BEFORE (proportional bonus — renormalizes to identity)
semantic_bonus = nudge_strength * base_prob

# AFTER (uniform additive bonus — actually shifts distribution)
semantic_bonus = nudge_strength / len(action_plan.tool_probs)
```

Uniform additive bonus redistributes probability mass from high-prob to low-prob tools after renormalization.

---

## Phase 2: Safety Singleton & DeceptionDetector Wiring (PR #2)

The highest-impact structural fix. Eliminates triple evaluation, unifies state.

### 2.1 — Consolidate SafetyGuardrails to single shared instance [M]

**Files:**
- `agentic_api.py` — already creates `state.safety_guardrails` at startup
- `config_init.py` — creates orchestrator-level guardrails (second instance)
- `agentic/core.py` — creates `safety_adapter` from orchestrator or fresh (third instance)

**Change:** `agentic_api.py` is the single source of truth. Pass `state.safety_guardrails` to the orchestrator constructor, which passes it to AgenticOrchestrator. Remove all `auto_create=True` fallback paths.

```
agentic_api.py startup:
  guardrails = SafetyGuardrails(...)     ← one instance
  audit_trail = AuditTrail(...)          ← one instance
  orchestrator = build_orchestrator(guardrails=guardrails, audit_trail=audit_trail)
    └─ WeavingOrchestrator(guardrails=guardrails)  ← receives, doesn't create
        └─ AgenticOrchestrator(guardrails=guardrails)  ← receives, doesn't create
```

**Result:** 3 evaluations → 1 evaluation. Fragmented state → unified state. All deception goals tracked in one detector.

### 2.2 — Wire DeceptionDetector into hot path (monitoring mode) [M] `[SAE READY]`

**File:** `agentic_api.py`, after safety gate (~line 950)

Currently `state.deception_detector` is created but never called. Wire it in as a **monitoring signal** (log, don't block):

```python
# After safety gate passes, before orchestrator.reason():
deception_score = state.deception_detector.check_response(query_text, response_text)
state.audit_trail.log_decision("deception_check", score=deception_score)
# Don't block — Jaccard similarity is too crude for enforcement
```

**`[SAE READY]`** — The `_calculate_deception_score()` method in `deception_detection.py` is the interface where SAE-based feature decomposition will replace Jaccard word overlap. Keep this interface stable: `check(expected, actual) → float`. When SAEs land, swap the implementation behind the same interface.

### 2.3 — Remove dead `state.deception_detector` duplication [S]

Remove the DeceptionDetector auto-creation in `AgenticOrchestrator.__init__()` (line 307). It should receive the shared instance from above, same as SafetyGuardrails.

---

## Phase 3: Deep Thinking Deferred Queue (PR #3)

*Depends on Phase 1.2 (DARK tier crash fix)*

Connects the already-built DeferredQueue to the gate that already marks things as deferred.

### 3.1 — Wire gate.review() → DeferredQueue.push() [M]

**Files:**
- `gate.py` — `review()` sets `verdict.deferred = True` but never queues
- `deferred.py` — complete DeferredQueue with `push()`, `drain()`, JSON persistence

**Change:** `MilestoneGate.__init__()` accepts optional `DeferredQueue`. When `verdict.deferred = True`, call `self.queue.push(DeferredJob(job_type="gate_review", payload=milestone_context))`.

### 3.2 — Wire tier recovery → DeferredQueue.drain() [M]

**File:** `tier.py` — tier transition callbacks

**Change:** Register a callback on DARK/DEGRADED → PARTIAL/FULL transition that calls `queue.drain(callback=resubmit_gate_review)`. The drain callback re-submits deferred gate reviews to the now-available GPU endpoint.

**Result:** Complete lifecycle: DARK tier defers → rig comes back → deferred work auto-processes.

---

## Phase 4: Thompson Sampling Consolidation (PR #4)

### 4.1 — Classify all 17+ implementations [S]

No code changes. Document in a `THOMPSON_SAMPLING_AUDIT.md`:

**Category A — genuinely different algorithms (keep as-is):**
- `core/policy/thompson_sampling.py` — TSBandit (numpy vectorized, 3 strategies)
- `apps/server/model_router.py` — ModelBandit (geodesic certainty, Fisher-Rao, speed blending)
- `context_packing/learning.py` — budget scaling (samples rate to scale continuous budget)

**Category B — copy-paste Beta priors (consolidate to use `ts_base` protocol):**
12+ implementations including convergence engine, recursive learning, conscience, expert router, ritual router, redteam bandit, eggroll shuttle, UX learning, smart operation selector, constitutional critique, automated auditor, chatops feedback.

### 4.2 — Make Category B conform to ts_base protocol [L]

**File:** `hololoom/bandits/ts_base.py` — the `ThompsonSampler` Protocol

For each Category B implementation:
1. Import `DiscreteSampler` from `ts_base`
2. Implement the protocol interface (`select()`, `update()`, `get_statistics()`)
3. Prefer composition: wrap a concrete `BetaArm` rather than reimplementing Beta sampling

Create a concrete `BetaArm` dataclass in `bandits/`:

```python
@dataclass
class BetaArm:
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    def expected_value(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool, weight: float = 1.0):
        if success:
            self.alpha += weight
        else:
            self.beta += weight

    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1))

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: dict) -> "BetaArm":
        return cls(alpha=d["alpha"], beta=d["beta"])
```

Priority order: convergence engine → recursive learning → context packing → others.

### 4.3 — Add persistence to core Thompson priors [M]

**Files:**
- `core/policy/thompson_sampling.py` — TSBandit (no save/load)
- `core/deep_thinking/gate.py` — VerdictPrior (no save/load)
- `core/recursive/full_learning_loop.py` — ThompsonPriors (no save/load)

**Reference:** `model_router.py:352-385` already has correct atomic save/load.

**Changes:**
1. Add `save(path)` / `load(path)` to each, following model_router pattern
2. Wire into orchestrator lifecycle: save on shutdown (`atexit` or SIGTERM handler), load on startup
3. Persistence dir: `./data/bandits/` (configurable via `HOLOLOOM_BANDIT_STATE_DIR` env var)

**Result:** Thompson priors survive restarts. Learning accumulates across sessions.

---

## Phase 5: Routing & RBAC Wiring (PR #5)

*Depends on Phase 2.1 (safety singleton)*

### 5.1 — Implement `_install_patterns()` [M]

**File:** `hololoom/routing/learning/adaptive_updater.py` lines 503-514

Currently a `logger.info()` stub. Change to actually update the classifier's pattern set:

1. Accept `AdaptiveMoonshotClassifier` ref in `AdaptiveUpdater.__init__()`
2. In `_install_patterns()`, call `classifier.update_patterns(patterns, shadow=shadow)`
3. In `_get_current_patterns()`, call `classifier.get_active_patterns()`
4. Check `query_classifier_adaptive.py` for the classifier's pattern storage API

**Result:** Closes the macro learning loop. SHADOW→AB_TEST→GRADUAL can physically deploy patterns.

### 5.2 — Wire continuous validator self-scheduling [S]

**File:** `hololoom/routing/learning/continuous_validator.py`

Add:
```python
async def start_background_validation(self, interval_s: float = 3600.0):
    self._running = True
    while self._running:
        await asyncio.sleep(interval_s)
        await self.validate_hourly()

def stop_background_validation(self):
    self._running = False
```

Wire into `agentic_api.py` startup as a background task.

### 5.3 — Wire PolicyGovernance RBAC into safety gate [M]

**Files:**
- `agents/policy_governance.py` — complete RBAC, zero callsites
- `agentic_api.py` — safety gating

**Change:** Layer RBAC before SafetyGuardrails:

```python
# In agentic_api.py startup:
state.governance = GovernancePolicy.from_template("production")

# In query endpoint, before SafetyGuardrails:
rbac_decision = state.governance.evaluate(agent_role, action)
if rbac_decision == PolicyDecision.DENY:
    return HTTP 403  # RBAC blocked
# Then proceed to SafetyGuardrails...
```

RBAC checks permissions (who can do what). SafetyGuardrails checks risk (is this action safe). Layered, not redundant.

---

## Phase 6: Config Cleanup & SAE Markers (PR #6)

### 6.1 — Fix bleed rate config naming [S]

**File:** `hololoom/core/deep_thinking/config.py`

Rename `critic_to_synth_bleed` → `plan_reasoning_bleed` (what it actually controls: how much plan reasoning the critic sees). Update all references in `deliberation.py`.

Current usage is numerically correct (0.3) but semantically misnamed. The rename prevents future refactoring bugs.

### 6.2 — Add SAE integration point markers [S] `[SAE READY]`

Comment-only changes documenting where SAEs slot in:

| File | Location | SAE Replaces |
|------|----------|-------------|
| `alignment/deception_detection.py` | `_calculate_deception_score()` | Jaccard word overlap → learned feature decomposition |
| `alignment/instrumental_convergence.py` | `detect_power_seeking()` | Keyword grep → interpretable feature activation patterns |
| `alignment/instrumental_convergence.py` | `detect_self_modification()` | Substring match → learned modification signatures |
| `core/policy/unified.py` | `NeuralCore` | Random weights → SAE-decomposed feature training |
| `alignment/safety_guardrails.py` | `AdversarialDetector.detect()` | Regex patterns → SAE adversarial feature detection |

**Interface contract for SAE integration:**
```python
class FeatureDetector(Protocol):
    """[SAE READY] Any detection method should conform to this interface.
    Current: regex/keyword/Jaccard implementations.
    Future: SAE feature activation from Dark Trace."""

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        """Returns score (0.0-1.0) and activated features."""
        ...

    def get_active_features(self) -> list[str]:
        """Returns interpretable feature names that fired."""
        ...
```

When SAEs are ready, they provide `detect()` with interpretable feature activations instead of regex matches. The governance stack calls the same interface — no rewiring needed.

---

## Summary

| Phase | PR | Steps | Size | Deps | What Ships |
|-------|----|-------|------|------|-----------|
| 1 | #1 | 5 | 5×S | None | No more crashes, correct math |
| 2 | #2 | 3 | 2×M + S | None | Single governance truth, deception monitoring |
| 3 | #3 | 2 | 2×M | 1.2 | Complete deferred queue lifecycle |
| 4 | #4 | 3 | S + L + M | None | Unified TS primitive, cross-session learning |
| 5 | #5 | 3 | 2×M + S | 2.1 | Closed learning loop, live RBAC |
| 6 | #6 | 2 | 2×S | None | Clean config, SAE-ready interfaces |

**17 steps. 6 PRs. Zero new architecture.**

### Before / After

| Metric | Before | After |
|--------|--------|-------|
| Safety evaluations per query | 3 (fragmented) | 1 (unified) |
| Audit trail instances | 2 (divergent) | 1 (shared) |
| DeceptionDetector callsites | 0 (dead code) | 1 (monitoring) |
| Thompson priors cross-session | Reset on restart | Persisted |
| Deferred queue flow | Flag and forget | Queue → drain on recovery |
| `_install_patterns()` | `logger.info()` stub | Physical deployment |
| RBAC enforcement | 0 callsites | Layered before safety gate |
| TS implementations | 17+ independent | 3 genuine + 12 using shared `BetaArm` |
| SAE integration points | 0 documented | 5 marked with stable interfaces |
| Hot path governance overhead | <0.103ms (unchanged) | <0.103ms (same — we're wiring, not adding) |

### SAE Back Pocket — When Ready

SAEs slot into 5 marked `[SAE READY]` interfaces without rewiring the governance stack:

1. **Deception detection** — replace `_calculate_deception_score()` Jaccard with SAE feature decomposition of response vs expected behavior
2. **Power-seeking detection** — replace keyword grep in `detect_power_seeking()` with interpretable feature activation patterns from agent action traces
3. **Self-modification detection** — replace 7 substring matches with learned modification signatures
4. **Adversarial detection** — replace regex patterns with SAE-based intent classification
5. **Policy network training** — use SAE-decomposed features as interpretable input to NeuralCore, giving the policy engine meaningful (not random) weights

The `FeatureDetector` protocol in Phase 6.2 is the contract. Implement it with SAE backends, swap the implementations, governance stack doesn't change.
