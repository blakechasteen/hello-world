# HoloLoom Policy Module

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/policy/`
**Total Code**: 3,082 lines across 6 Python files

---

## Overview

The Policy Module is HoloLoom's **neural decision engine**, combining transformer-based deep learning with Thompson Sampling bandits for optimal exploration/exploitation. It decides which tool to use (answer, search, notion_write, calc) based on query features and retrieved context.

**Key Innovation**: Unlike traditional RL agents, HoloLoom's policy uses **motif-gated attention** (linguistic patterns control neural pathways) and **Thompson Sampling** (Bayesian exploration) to make contextually aware, semantically guided decisions.

### Quick Start

```python
from HoloLoom.policy import create_policy, BanditStrategy
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

# Create embedder
emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])

# Create policy
policy = create_policy(
    mem_dim=384,
    emb=emb,
    scales=[96, 192, 384],
    bandit_strategy=BanditStrategy.EPSILON_GREEDY,
    epsilon=0.1  # 10% exploration
)

# Make decision
action_plan = await policy.decide(features, context)
print(f"Chosen Tool: {action_plan.chosen_tool}")
```

---

## Architecture

### File Structure

```
HoloLoom/policy/
├── __init__.py                  # 18 lines - Public exports
├── unified.py                   # 1,233 lines - Main policy engine
├── thompson_sampling.py         # 160 lines - Bandit strategies
├── semantic_nudging.py          # 662 lines - 244D semantic guidance
├── bayesian_policy.py           # 553 lines - Uncertainty quantification
└── gp_policy.py                 # 456 lines - Continuous optimization
```

### Core Components

#### 1. NeuralCore (Transformer-Based Decision Network)

**Architecture** (`unified.py:355-431`):
```
Input Features → Learnable Latent Queries (16 tokens)
    ↓
Transformer Blocks (2 layers × 4 heads)
    ├── Motif-Gated Multi-Head Attention
    ├── Cross-Attention to Retrieved Context
    └── LoRA-Style Feed-Forward (4 adapters)
    ↓
Readout Pooling → Tool Selection Head (4 tools)
```

**Key Innovations**:
- **Motif-Gated Attention**: Detected motifs (question→answer, cause→effect) control which attention heads activate
- **Cross-Attention to Memory**: Policy attends to retrieved context shards for context-sensitive decisions
- **LoRA-Style Adapters**: Different adapters for different execution modes (BARE/FAST/FUSED) enable specialization without retraining

```python
core = NeuralCore(
    d_model=384,        # Model dimension
    n_layers=2,         # Transformer layers
    n_heads=4,          # Attention heads
    n_motifs=8,         # Motif control signals
    n_adapters=4,       # LoRA adapters (general, farm, brewing, mirrorcore)
    n_tools=4           # Tools (answer, search, notion_write, calc)
)
```

#### 2. Thompson Sampling Bandit

**Purpose**: Bayesian exploration/exploitation using Beta distributions.

**How It Works**:
1. Maintain Beta(α, β) distribution per tool
2. Sample from each distribution: `θ_i ~ Beta(α_i, β_i)`
3. Select tool with highest sample: `argmax_i θ_i`
4. Update after reward: `α ← α + reward`, `β ← β + |penalty|`

**Strategies** (`thompson_sampling.py`):

| Strategy | Neural Weight | Bandit Weight | Use Case |
|----------|---------------|---------------|----------|
| **EPSILON_GREEDY** | 90% | 10% (explore) | Stable production, trust neural |
| **BAYESIAN_BLEND** | 70% | 30% | Balanced neural + prior |
| **PURE_THOMPSON** | 0% | 100% | Maximum exploration, ignore neural |

```python
# Thompson Sampling update equations
Success (reward > 0): α ← α + reward
Failure (reward < 0): β ← β + |reward|
Expected Reward: E[θ] = α / (α + β)
```

#### 3. UnifiedPolicy (Main Integration)

Combines neural core + Thompson Sampling + safety guardrails:

```python
@dataclass
class UnifiedPolicy:
    core: NeuralCore                     # Neural decision network
    bandit: TSBandit                     # Thompson Sampling
    guardrails: SafetyGuardrails         # Alignment framework
    emb: MatryoshkaEmbeddings            # Context encoder
    adapter_for_dim: Dict[int, int]      # Dimension → adapter
    bandit_strategy: BanditStrategy      # EPSILON_GREEDY/etc
    epsilon: float = 0.1                 # Exploration rate
```

**Main Method**:
```python
async def decide(
    self,
    features: Features,    # Ψ (spectral) + motifs + metrics
    context: Context       # Retrieved shards + KG subgraph
) -> ActionPlan
```

**Decision Flow**:
```
1. Encode context with embeddings
2. Neural forward pass → tool logits
3. Apply Thompson Sampling strategy
4. Select tool (exploit or explore)
5. Safety guardrails check
6. Return ActionPlan with metadata
```

---

## Advanced Extensions

### Bayesian Policy (Uncertainty Quantification)

**File**: `bayesian_policy.py`
**Purpose**: Quantify epistemic (model) and aleatoric (data) uncertainty

**Architecture**:
- Variational distributions over weights: `q(W) = N(μ_W, Σ_W)`
- MC sampling for predictive distributions (10 samples)
- ELBO training: `E_q[log p(y|x,w)] - KL(q||p)`

**Usage**:
```python
policy = create_policy(
    mem_dim=384,
    emb=emb,
    scales=[96, 192, 384],
    use_bayesian=True,
    bayesian_samples=10,
    bayesian_kl_weight=1.0
)

action_plan = await policy.decide(features, context)

# Uncertainty metrics
unc = action_plan.metadata['uncertainty']
print(f"Epistemic: {unc['epistemic']:.3f}")  # Model uncertainty
print(f"Aleatoric: {unc['aleatoric']:.3f}")  # Data uncertainty
print(f"Total: {unc['total']:.3f}")
```

**When to Use**:
- Research mode: Need confidence intervals on decisions
- Safety-critical: High epistemic → explore more
- Active learning: Query points with high uncertainty

### GP Policy (Continuous Optimization)

**File**: `gp_policy.py`
**Purpose**: Learn smooth functions over continuous action spaces

**Supported Acquisitions**:
- **GP-TS** (Thompson Sampling): Sample from GP posterior
- **GP-UCB** (Upper Confidence Bound): Optimize acquisition function

**Usage**:
```python
from HoloLoom.policy.gp_policy import create_gp_policy, GPConfig

policy = create_gp_policy(
    mem_dim=384,
    emb=emb,
    scales=[96, 192, 384],
    gp_config=GPConfig(
        acquisition="thompson",        # or "ucb"
        kernel_type="matern",           # or "rbf"
        action_space_dims=3,            # stiffness, damping, temp
        action_space_bounds={
            'stiffness': (0.05, 0.5),
            'damping': (0.5, 0.95),
            'temperature': (0.1, 2.0)
        },
        n_candidates_per_dim=5
    )
)

# GP learns optimal hyperparameters
action_plan = await policy.decide(features, context)
hyperparams = action_plan.metadata['gp']['hyperparams']
```

**When to Use**:
- Hyperparameter tuning (learning rate, temperature, etc.)
- Physics simulations (stiffness, damping)
- Smooth action spaces requiring gradient information

### Semantic Nudging (244D Guidance)

**File**: `semantic_nudging.py`
**Purpose**: Guide decisions toward semantic goals (Warmth, Clarity, Wisdom)

**Components**:
- `SemanticStateEncoder`: 244D semantic space → policy features
- `SemanticRewardShaper`: Potential-based reward shaping
- `SemanticNudgePolicy`: Wrapper applying semantic nudges

**Usage**:
```python
from HoloLoom.policy.semantic_nudging import (
    SemanticNudgePolicy,
    define_semantic_goals
)

# Define semantic goals
goals = define_semantic_goals('professional')
# {'Formality': 0.7, 'Clarity': 0.9, 'Directness': 0.8, ...}

# Wrap base policy
semantic_policy = SemanticNudgePolicy(
    base_policy=base_policy,
    semantic_spectrum=spectrum,
    semantic_goals=goals
)

# Decisions are semantically guided
action_plan = await semantic_policy.decide(
    features,
    context,
    semantic_state=semantic_state  # 244D projection
)
```

**Predefined Goals**:
- `'professional'`: High formality, clarity, directness
- `'creative'`: High warmth, playfulness, abstractness
- `'educational'`: High clarity, patience, supportiveness
- `'conversational'`: High warmth, casualness, relatability

---

## Usage Examples

### Example 1: Basic Policy

```python
import asyncio
from HoloLoom.policy import create_policy, BanditStrategy
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Features, Context

async def demo():
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    policy = create_policy(
        mem_dim=384,
        emb=emb,
        scales=[96, 192, 384],
        n_layers=2,
        n_heads=4,
        bandit_strategy=BanditStrategy.EPSILON_GREEDY,
        epsilon=0.1
    )

    features = Features(
        psi=np.array([0.1, 0.5, 0.2, 0.3, 0.4, 0.6]),
        motifs=["question→answer"],
        metrics={"coherence": 0.8},
        confidence=0.85
    )

    context = Context(
        hits=[],
        kg_sub=None,
        shard_texts=["Context about Thompson Sampling..."],
        relevance=0.7
    )

    action_plan = await policy.decide(features, context)

    print(f"Tool: {action_plan.chosen_tool}")
    print(f"Adapter: {action_plan.adapter}")
    print(f"Confidence: {action_plan.tool_probs.max():.3f}")

asyncio.run(demo())
```

### Example 2: Compare Strategies

```python
async def compare_strategies():
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    strategies = [
        (BanditStrategy.EPSILON_GREEDY, "90% Neural, 10% Explore"),
        (BanditStrategy.BAYESIAN_BLEND, "70% Neural, 30% Bandit"),
        (BanditStrategy.PURE_THOMPSON, "100% Thompson Sampling")
    ]

    for strategy, desc in strategies:
        print(f"\n=== {desc} ===")

        policy = create_policy(
            mem_dim=384,
            emb=emb,
            scales=[96, 192, 384],
            bandit_strategy=strategy
        )

        # Run 100 decisions
        for _ in range(100):
            action_plan = await policy.decide(features, context)

        # Show bandit statistics
        stats = policy.bandit.get_stats()
        for i, stat in stats.items():
            tool = policy.core.tools[i]
            print(f"{tool}: {stat['pulls']} pulls, mean={stat['mean']:.3f}")

asyncio.run(compare_strategies())
```

### Example 3: Bayesian with Uncertainty

```python
async def demo_bayesian():
    policy = create_policy(
        mem_dim=768,
        emb=emb,
        scales=[768],
        use_bayesian=True,
        bayesian_samples=10
    )

    action_plan = await policy.decide(features, context)

    unc = action_plan.metadata['uncertainty']

    print(f"Tool: {action_plan.chosen_tool}")
    print(f"Epistemic: {unc['epistemic']:.3f}")
    print(f"Aleatoric: {unc['aleatoric']:.3f}")

    # High epistemic → model uncertain, explore more
    if unc['epistemic'] > 0.5:
        print("⚠️  High model uncertainty - explore!")

asyncio.run(demo_bayesian())
```

### Example 4: Integration with Orchestrator

```python
# From weaving_orchestrator.py
from HoloLoom.policy import create_policy

class WeavingOrchestrator:
    def __init__(self, cfg: Config, shards, guardrails=None):
        # Create policy
        self.policy = create_policy(
            mem_dim=cfg.embedding_dim,
            emb=self.embedder,
            scales=cfg.matryoshka_sizes,
            bandit_strategy=BanditStrategy.EPSILON_GREEDY,
            guardrails=guardrails,
            cfg=cfg
        )

    async def weave(self, query: Query) -> Spacetime:
        # Extract features
        features = await self._extract_features(query)

        # Retrieve context
        context = await self._retrieve_context(query, features)

        # Policy decides tool
        action_plan = await self.policy.decide(features, context)

        # Execute
        result = await self._execute_tool(
            action_plan.chosen_tool,
            query,
            context
        )

        return result
```

---

## Configuration

### All Options

```python
policy = create_policy(
    # Required
    mem_dim=384,                    # Memory dimension
    emb=embedder,                   # MatryoshkaEmbeddings
    scales=[96, 192, 384],          # Embedding scales

    # Optional - Neural architecture
    device=None,                    # Auto-detect GPU/CPU
    n_layers=2,                     # Transformer depth
    n_heads=4,                      # Attention heads

    # Optional - Exploration
    bandit_strategy=BanditStrategy.EPSILON_GREEDY,
    epsilon=0.1,                    # Exploration rate

    # Optional - Safety
    guardrails=None,                # Pre-configured guardrails
    cfg=None,                       # Config for environment-aware safety

    # Optional - Bayesian extension
    use_bayesian=False,             # Enable Bayesian
    bayesian_samples=10,            # MC samples
    bayesian_kl_weight=1.0,         # KL regularization
    bayesian_prior_std=1.0          # Prior std dev
)
```

### Adapter Selection

Adapters are automatically selected based on memory dimension:

```python
adapter_for_dim = {
    96: 1,      # farm adapter (smallest scale)
    192: 2,     # brewing adapter (mid scale)
    384: 3      # mirrorcore adapter (largest scale)
}
```

Access via:
```python
action_plan = await policy.decide(features, context)
print(action_plan.adapter)  # "mirrorcore" (for mem_dim=384)
```

---

## Safety & Alignment

### Automatic Guardrails Integration

The policy automatically integrates with HoloLoom's alignment framework:

```python
# Guardrails are created automatically
policy = create_policy(...)  # Guardrails enabled by default

# High-risk actions are blocked
try:
    action_plan = await policy.decide(features, context)
except PermissionError as e:
    print(f"Action blocked: {e}")
```

### Decision Flow with Safety

```
1. Policy selects tool (neural + bandit)
2. Guardrails evaluate action safety
3. Risk assessment:
   - LOW → allowed
   - MEDIUM → allowed with logging
   - HIGH → requires approval (raises PermissionError)
   - CRITICAL → blocked (raises PermissionError)
4. Decision logged to audit trail
5. Return ActionPlan or raise error
```

### Metadata

```python
action_plan.metadata = {
    'bandit': {
        'mode': 'explore',              # or 'exploit'
        'strategy': 'epsilon_greedy',
        'selected_arm': 2,
        'total_pulls': [10, 15, 8, 12]
    },
    'guardrails': {
        'allowed': True,
        'safety_score': 0.92,
        'risk_category': 'LOW',
        'requires_approval': False
    }
}
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Deterministic Decision** | ~1-2ms | Single forward pass |
| **Bayesian Decision (10 samples)** | ~10-20ms | 10× overhead from MC sampling |
| **GP Decision** | ~5-10ms | Candidate set optimization |
| **Semantic Nudging** | +0.5ms | Minimal overhead |
| **Alignment Check** | +0.1ms | Safety guardrails |

**Memory**:
- NeuralCore: ~2.5MB (d_model=384, n_layers=2)
- Bandit statistics: ~1KB per tool
- Total: ~3MB typical

---

## Testing

**Unit Tests**:
```bash
pytest HoloLoom/tests/unit/test_unified_policy.py -v
pytest HoloLoom/tests/unit/test_bayesian_policy.py -v
```

**Demos**:
```bash
PYTHONPATH=. python demos/demo_bayesian_policy.py
PYTHONPATH=. python demos/semantic_micropolicy_nudge_demo.py
PYTHONPATH=. python demos/demo_gp_bandits.py
```

---

## API Reference

### Core Classes

#### `create_policy()`
Factory function for creating policy instances.

**Signature**:
```python
def create_policy(
    mem_dim: int,
    emb: MatryoshkaEmbeddings,
    scales: List[int],
    device: Optional[torch.device] = None,
    n_layers: int = 2,
    n_heads: int = 4,
    bandit_strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY,
    epsilon: float = 0.1,
    guardrails: Optional[SafetyGuardrails] = None,
    cfg: Optional[Config] = None,
    use_bayesian: bool = False,
    bayesian_samples: int = 10,
    bayesian_kl_weight: float = 1.0,
    bayesian_prior_std: float = 1.0
) -> PolicyEngine
```

**Returns**: `UnifiedPolicy` or `BayesianUnifiedPolicy`

#### `PolicyEngine.decide()`
Main decision method.

**Signature**:
```python
async def decide(
    self,
    features: Features,
    context: Context
) -> ActionPlan
```

**Returns**:
```python
@dataclass
class ActionPlan:
    chosen_tool: str           # "answer", "search", etc.
    tool_probs: np.ndarray     # [0.7, 0.2, 0.05, 0.05]
    adapter: str               # "mirrorcore", etc.
    metadata: Dict[str, Any]   # Bandit stats, guardrails, etc.
```

#### `TSBandit`
Thompson Sampling bandit implementation.

**Methods**:
```python
# Choose arm using Thompson Sampling
arm: int = bandit.choose()

# Get prior probabilities
priors: np.ndarray = bandit.get_priors()

# Select with strategy (epsilon-greedy, etc.)
tool_idx, debug = bandit.select_with_strategy(neural_probs)

# Update with reward
bandit.update(arm_idx, reward)

# Get statistics
stats: Dict[int, Dict] = bandit.get_stats()
```

### Enums

```python
class BanditStrategy(Enum):
    EPSILON_GREEDY = "epsilon_greedy"  # 90% neural, 10% explore
    BAYESIAN_BLEND = "bayesian_blend"  # 70% neural, 30% bandit
    PURE_THOMPSON = "pure_thompson"    # 100% Thompson Sampling
```

---

## Dependencies

**Internal**:
```python
from HoloLoom.documentation.types import Features, Context, ActionPlan
from HoloLoom.protocols import PolicyEngine
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.alignment.safety_guardrails import SafetyGuardrails
from HoloLoom.semantic_calculus import SemanticSpectrum  # Optional
```

**External**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
```

---

## Further Reading

- **Thompson Sampling**: Agrawal & Goyal (2012) - "Analysis of Thompson Sampling for the Multi-armed Bandit Problem"
- **Bayesian Neural Networks**: Blundell et al. (2015) - "Weight Uncertainty in Neural Networks"
- **Gaussian Process Bandits**: Srinivas et al. (2010) - "Gaussian Process Optimization in the Bandit Setting"
- **Attention Mechanisms**: Vaswani et al. (2017) - "Attention is All You Need"

---

## Summary

The HoloLoom Policy Module provides:

✅ **Transformer-based architecture** with motif-gated attention
✅ **Thompson Sampling** for optimal exploration/exploitation
✅ **5 exploration strategies** (epsilon-greedy, Bayesian blend, pure Thompson, GP-TS, GP-UCB)
✅ **Bayesian uncertainty quantification** (epistemic + aleatoric)
✅ **Continuous optimization** via Gaussian Process bandits
✅ **244D semantic guidance** for semantically appropriate decisions
✅ **Alignment framework integration** for safe decision making
✅ **Protocol-based design** for swappable implementations
✅ **Production ready** with comprehensive testing

**Total**: 3,082 lines of production code implementing state-of-the-art decision making.
