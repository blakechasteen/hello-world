# Thompson Sampling Audit

**Phase 4.1 — Governance Wiring Plan**
**Date**: 2026-03-12

## Overview

Classification of all Thompson Sampling implementations across the HoloLoom codebase into:

- **Category A**: Genuinely different algorithms (keep as-is)
- **Category B**: Copy-paste Beta priors (refactored to use `BetaArm` via composition)

## Canonical Implementation

**`hololoom/bandits/beta_arm.py`** — The single source of truth for Beta(alpha, beta) prior management.

Provides: `sample()`, `expected_value()`, `update()`, `variance()`, `to_dict()`, `from_dict()`, `total_observations`, `uncertainty`, `confidence`.

---

## Category A — Different Algorithms (Keep As-Is)

### 1. `hololoom/core/policy/thompson_sampling.py` — TSBandit

**Why Category A**: Numpy-vectorized multi-arm bandit operating on arrays of alpha/beta priors simultaneously. Uses `np.random.beta(self.alphas, self.betas)` across all arms in a single call. Not a single-arm Beta prior.

**Change made**: Added `save(path)`/`load(path)` persistence (Phase 4.3). No structural refactoring.

### 2. `hololoom/prompting/analytics/learning.py` — StrategyStats

**Why Category A**: Uses Gaussian approximation (`random.gauss`) instead of `betavariate` for sampling. Asymmetric update logic: success adds `quality_improvement` to alpha, failure adds `(threshold - quality_improvement)` to beta. Fundamentally different sampling and update semantics.

### 3. `hololoom/context/bandit.py` — ThompsonBandit (the orchestrator class)

**Why Category A**: Multi-arm bandit orchestrator that manages multiple `BanditArm` instances. The orchestrator itself doesn't hold Beta priors — it delegates to `BanditArm` instances (which are now Category B). The `select()` method samples from all arms and picks the max.

---

## Category B — Refactored to BetaArm Composition

All files below now delegate their Beta(alpha, beta) logic to `BetaArm` via a `_arm` field initialized in `__post_init__`. External APIs are preserved.

### Prior Session (5 files)

| File | Class | Notes |
|------|-------|-------|
| `hololoom/core/convergence/refinement_strategies.py` | `StrategyStats` | Standard delegation |
| `hololoom/core/recursive/full_learning_loop.py` | `ThompsonPriors` | + `save(path)`/`load(path)` persistence |
| `hololoom/core/deep_thinking/gate.py` | `VerdictPrior` | + `save(path)`/`load(path)` persistence |
| `hololoom/core/policy/thompson_sampling.py` | `TSBandit` | Category A (vectorized), but `save`/`load` added |
| `hololoom/agentic/expert_router.py` | `ThompsonPrior` | Standard delegation |

### Current Session (14 files)

| File | Class | Notes |
|------|-------|-------|
| `hololoom/collaboration/ux_learning.py` | `BetaPrior` | Standard delegation with `from_dict` |
| `hololoom/alignment/automated_auditor.py` | `ThresholdPrior` | Standard delegation |
| `hololoom/alignment/sandbagging_detection.py` | `StrategyPrior` | Delegates `uncertainty` to BetaArm |
| `hololoom/alignment/constitutional_critique.py` | `PrincipleWeightPrior` | Maps BetaArm sample to [0.5, 2.0] range |
| `hololoom/agentic/conscience_calibrator.py` | `CalibrationPrior` | Uses `_arm.confidence(saturation=100.0)` |
| `hololoom/semantic_calculus/clustering_thompson.py` | `BetaPrior` | Special `update(reward)`: increments both alpha and beta |
| `hololoom/visualization/jenny_mrf.py` | `PanelTypePrior` | Standard delegation |
| `hololoom/redteam/bandit.py` | `BanditArm` | Extra fields: strategy, total_pulls, total_rewards |
| `hololoom/redteam/learning/hierarchical_learning.py` | `HierarchicalArm` | Extra fields: level, arm_id, parent_id, children |
| `hololoom/redteam/learning/contextual_bandit.py` | `ContextualArm` | Extra fields: strategy, context |
| `hololoom/redteam/swarm/learning.py` | `ThompsonSamplingPrior` | Extra fields: strategy_id, context_stats |
| `hololoom/conscience/judgment.py` | `Wisdom` | Custom `confidence` formula (not BetaArm's) |
| `hololoom/model_extension/eval/learning_metrics.py` | `ThompsonSamplingState` | Read-only; keeps alpha/beta as fields for constructor compat |
| `hololoom/context/bandit.py` | `BanditArm` | ThompsonBandit.update() refactored to use arm.update() |

---

## Persistence (Phase 4.3)

Atomic `save(path)`/`load(path)` added to:

| Class | File |
|-------|------|
| `TSBandit` | `hololoom/core/policy/thompson_sampling.py` |
| `ThompsonPriors` | `hololoom/core/recursive/full_learning_loop.py` |
| `VerdictPrior` | `hololoom/core/deep_thinking/gate.py` |

Pattern: write to `.tmp` file, then `os.replace()` for atomic swap.

---

## Design Patterns Used

### Standard Delegation
```python
@dataclass
class MyPrior:
    _arm: "BetaArm" = field(default=None, repr=False)

    def __post_init__(self):
        from hololoom.bandits.beta_arm import BetaArm
        if self._arm is None:
            self._arm = BetaArm()

    @property
    def alpha(self) -> float:
        return self._arm.alpha

    @property
    def beta(self) -> float:
        return self._arm.beta
```

### Read-Only Metrics (ThompsonSamplingState)
Keeps `alpha`/`beta` as regular dataclass fields (not properties) because tests construct with `ThompsonSamplingState(alpha=10, beta=5)`. The `_arm` is created from init values in `__post_init__`.

### External Mutator (context/bandit.py)
`ThompsonBandit.update()` originally did `arm.alpha += weight` directly. Refactored to `arm.update(success, weight)` which delegates to `_arm.update()`.
