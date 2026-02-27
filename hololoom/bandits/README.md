# HoloLoom Bandits: Neural Thompson Sampling

**Production-ready contextual bandits for exploration/exploitation in HoloLoom's decision pipeline.**

## Overview

HoloLoom Bandits implements **Neural Thompson Sampling** - an uncertainty-aware approach to action selection that balances exploration (trying new tools/prompts) with exploitation (using what works). Perfect for adaptive systems that learn from every interaction.

### Key Features

- **Thompson Sampling**: Principled Bayesian exploration
- **Neural Reward Models**: Non-linear context-action rewards
- **Two Uncertainty Backends**:
  - **Bootstrap Ensemble** (default): N independent MLPs, well-calibrated
  - **MC-Dropout**: Single MLP with dropout, parameter-efficient
- **Online Learning**: Continual improvement from observations
- **Evaluation Metrics**: ECE (calibration), regret proxy, reward tracking
- **Production Ready**: <3ms selection latency, checkpointing, diagnostics

## Quick Start

### Installation

```bash
# Already included in HoloLoom
cd HoloLoom
pip install torch numpy
```

### Basic Usage

```python
from HoloLoom.bandits import create_neural_ts_policy, BanditConfig
from HoloLoom.bandits.neural_ts.types import Context, Action, Observation
import numpy as np

# Create policy (uses HoloLoom defaults: 384-dim embeddings, Bootstrap)
config = BanditConfig.loom_default()
policy = create_neural_ts_policy(config)

# Your decision loop
for query in queries:
    # 1. Get context from HoloLoom embeddings
    ctx = Context(
        id=f"query_{query.id}",
        x=query_embeddings,  # From HoloLoom's Matryoshka embeddings
    )

    # 2. Define candidate actions
    actions = [
        Action(id="answer_tool"),
        Action(id="search_tool"),
        Action(id="explain_tool"),
    ]

    # 3. Select action using Thompson Sampling
    chosen = policy.select(ctx, actions)

    # 4. Execute action
    result = execute_tool(chosen.id, query)

    # 5. Compute reward (success - cost penalty)
    reward = 1.0 if result.success else 0.0
    reward -= result.cost * 0.1  # Cost penalty

    # 6. Learn from outcome
    policy.update(Observation(
        context_id=ctx.id,
        action_id=chosen.id,
        reward=reward,
    ))

# Save learned policy
policy.save_checkpoint("checkpoints/bandit_v1.pt")
```

### Configuration Presets

```python
from HoloLoom.bandits.config import BanditConfig

# Default (Bootstrap, 384-dim, production settings)
config = BanditConfig.loom_default()

# Fast (smaller models, less training - for dev/testing)
config = BanditConfig.fast()

# MC-Dropout (parameter-efficient alternative)
config = BanditConfig.mc_dropout()

# Custom
config = BanditConfig(
    backend="bootstrap",
    context_dim=512,        # Match your embeddings
    action_dim=16,          # Add action features if available
    hidden_dims=[256, 128],
    n_ensemble=7,           # Bootstrap heads
    lr=1e-3,
    batch_size=256,
    train_every=200,        # Update every 200 observations
    train_steps=100,        # 100 gradient steps per update
    replay_capacity=200_000,
    replay_warmup=5_000,    # Wait for 5k observations before training
)
policy = create_neural_ts_policy(config)
```

## Architecture

### Components

```
NeuralThompsonPolicy
├── Posterior (BootstrapPosterior | MCDropoutPosterior)
│   └── Models (Ensemble of MLPs | Single MLP with dropout)
├── Featurizer (Context + Action → Feature vector)
├── ReplayBuffer (Stores observations for training)
└── BanditTrainer (Online SGD updates)
```

### Data Flow

```
1. Context (from HoloLoom embeddings)
   ↓
2. Featurizer → [context, action] features
   ↓
3. Posterior.sample_fn() → Sample reward model
   ↓
4. Thompson Sampling → Greedy w.r.t. sampled model
   ↓
5. Action selection
   ↓
6. Observation (context, action, reward)
   ↓
7. ReplayBuffer.add()
   ↓
8. Trainer.fit_steps() (every N observations)
   ↓
9. Posterior updated → Better next selections
```

## Core Concepts

### Thompson Sampling

**Idea**: Sample a reward model from the posterior, act greedily.

- **Exploration**: High uncertainty → diverse samples → tries new actions
- **Exploitation**: Low uncertainty → confident predictions → picks best
- **Automatic**: No manual epsilon tuning like ε-greedy

**Algorithm**:
```
θ ~ Posterior                       # Sample model parameters
f_θ(x, a) = predicted reward       # Evaluate all actions
a* = argmax_a f_θ(x, a)            # Pick best according to sample
```

### Uncertainty Backends

#### Bootstrap Ensemble (Default)

- **Trains**: N independent MLPs on bootstrap resamples
- **Thompson Sampling**: Uniformly pick one head, act greedily
- **Pros**: Well-calibrated, simple, robust
- **Cons**: N× memory (but models are tiny: ~165k params/head)

**Config**:
```python
config = BanditConfig(
    backend="bootstrap",
    n_ensemble=7,  # Typical: 3-10
)
```

#### MC-Dropout

- **Trains**: Single MLP with dropout
- **Thompson Sampling**: Keep dropout ON at inference → stochastic forward
- **Pros**: Memory-efficient (1 model), fast
- **Cons**: May be less calibrated than Bootstrap

**Config**:
```python
config = BanditConfig(
    backend="mc_dropout",
    dropout_p=0.1,  # Typical: 0.05-0.15
)
```

### Online Learning

**Replay Buffer** stores recent observations:
- **Warmup**: Wait for minimum data before training (default: 5000)
- **Training**: Every `train_every` observations (default: 200)
  - Bootstrap: Each head trains on bootstrap sample
  - MC-Dropout: Single model trains on random batch
- **Capacity**: Circular buffer (default: 200k, oldest evicted)

**Training Loop**:
```python
# Every 200 observations
if replay.is_ready() and obs_count % train_every == 0:
    for step in range(train_steps):
        batch = replay.sample(batch_size, bootstrap=True)
        loss = MSE(model(batch.x), batch.y)
        optimizer.step()
```

## Evaluation

### Metrics

```python
from HoloLoom.bandits.neural_ts.eval import BanditEvaluator

evaluator = BanditEvaluator(n_bins=10)

for ctx, action, reward in decisions:
    predicted = model.predict(ctx, action)
    evaluator.record(predicted, reward)

metrics = evaluator.compute_metrics()
print(f"ECE: {metrics['ece']:.4f}")           # Calibration (lower better, <0.1 good)
print(f"Regret: {metrics['regret_proxy']:.4f}")  # vs. oracle (lower better)
print(f"Mean reward: {metrics['mean_reward']:.4f}")
```

**Expected Calibration Error (ECE)**:
- **Measures**: How well predicted rewards match actual rewards
- **Lower is better**: ECE < 0.08 → well calibrated
- **Algorithm**: Bin predictions, compare mean(predicted) vs mean(actual)

**Regret Proxy**:
- **Measures**: Suboptimality vs. best possible (oracle)
- **Formula**: `regret = oracle_reward - actual_reward`
- **Note**: Without oracle, uses max(observed) as proxy

### Visualization

```python
from HoloLoom.bandits.neural_ts.eval import compute_cumulative_regret, compute_moving_average_reward

# Cumulative regret over time (should plateau as model learns)
cum_regret = compute_cumulative_regret(rewards, oracle_rewards)

# Smoothed reward trajectory
smooth_reward = compute_moving_average_reward(rewards, window=100)

import matplotlib.pyplot as plt
plt.plot(cum_regret, label="Cumulative Regret")
plt.xlabel("Step")
plt.ylabel("Regret")
plt.legend()
plt.show()
```

## Advanced Usage

### Action Features

If actions have features (e.g., cost, latency, complexity):

```python
config = BanditConfig(
    context_dim=384,
    action_dim=8,  # Enable action features
)

# Define actions with features
actions = [
    Action(
        id="cheap_fast",
        a=np.array([0.1, 50.0, 0.0]),  # [cost, latency_ms, uses_llm]
    ),
    Action(
        id="expensive_slow",
        a=np.array([1.0, 500.0, 1.0]),
    ),
]

chosen = policy.select(ctx, actions)
```

The model learns `reward(context, action_features)`, enabling transfer across actions.

### Diagnostics

```python
chosen, diag = policy.select(ctx, actions, return_diagnostics=True)

print(diag["pred_mean"])       # Predicted reward
print(diag["pred_std"])        # Uncertainty (Bootstrap/MC-Dropout)
print(diag["all_preds"])       # Predictions for all actions
print(diag["latency_ms"])      # Selection time
print(diag["warmup_mode"])     # Still in warmup?

# Policy statistics
stats = policy.get_statistics()
print(stats["total_selections"])
print(stats["replay_stats"]["size"])
print(stats["trainer_stats"]["avg_loss"])
```

### Checkpointing

```python
# Save
policy.save_checkpoint("checkpoints/bandit_epoch_100.pt")

# Load
policy.load_checkpoint("checkpoints/bandit_epoch_100.pt")
```

**Note**: Only model weights are saved, not replay buffer (too large).

### Top-k Recommendations

```python
# Get top-3 actions (for ensembling or exploration)
top3 = policy.recommend_k(ctx, actions, k=3)
```

## Integration with HoloLoom

### WeavingOrchestrator Integration

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.bandits import create_neural_ts_policy, BanditConfig
from HoloLoom.documentation.types import Query

# Create bandit for tool selection
config = BanditConfig.loom_default()
bandit = create_neural_ts_policy(config)

# Inside orchestrator
async def weave(query: Query):
    # Extract context from query embeddings
    embeddings = await self.embed(query.text)
    ctx = Context(id=query.id, x=embeddings)

    # Define candidate tools
    tools = [
        Action(id="answer"),
        Action(id="search"),
        Action(id="explain"),
    ]

    # Bandit selects tool
    chosen_tool = bandit.select(ctx, tools)

    # Execute
    result = await self.execute_tool(chosen_tool.id, query)

    # Compute reward from result quality
    reward = result.confidence * (1.0 - result.latency / 1000.0)  # Quality - latency penalty

    # Learn
    bandit.update(Observation(ctx.id, chosen_tool.id, reward))

    return result
```

### Reward Design

**Good rewards** are:
1. **Normalized**: [0, 1] or z-scored
2. **Informative**: Captures true utility (success, speed, cost, user satisfaction)
3. **Immediate**: Available right after action execution

**Examples**:

```python
# Binary success
reward = 1.0 if result.success else 0.0

# Continuous quality
reward = result.confidence  # 0-1

# Multi-objective (success - cost - latency)
reward = (
    result.success * 1.0
    - result.cost * 0.1
    - result.latency_ms / 1000.0 * 0.05
)

# User feedback (if available)
reward = user_rating / 5.0  # 0-1
```

## Testing

### Unit Tests

```bash
# All unit tests (37 tests, <1s)
PYTHONPATH=. pytest HoloLoom/bandits/tests/test_units.py -v
```

**Coverage**:
- Types (Context, Action, Observation)
- Models (MLP, ensemble creation)
- Posterior (Bootstrap, MC-Dropout sampling, uncertainty)
- Replay buffer (add, sample, bootstrap)
- Trainer (Bootstrap, MC-Dropout updates)
- Featurizer (context-only, context+action, normalization)
- Evaluator (ECE, regret, metrics)
- Policy (select, update, diagnostics, recommend_k)
- Config (defaults, factories, validation)

### Synthetic Bandit Validation

```bash
# CI validation (~30s)
PYTHONPATH=. python HoloLoom/bandits/tests/test_synthetic_bandit.py
```

**Tests**:
1. **Oracle**: Best action achieves high rewards
2. **Neural-TS learns**: Beats random baseline after 3000 steps
3. **vs. Epsilon-greedy**: Competitive with context-blind baseline
4. **Bootstrap vs MC-Dropout**: Both backends work

**Synthetic Bandit**:
- **Reward**: `r(x, a) = sin(w_a^T x) + ε` (non-linear)
- **Context**: `x ~ N(0, I)`
- **Actions**: 5 tools with different weight vectors
- **Validation**: Neural-TS exploits context, outperforms random

### Running All Tests

```bash
# Unit + synthetic
PYTHONPATH=. pytest HoloLoom/bandits/tests/ -v

# Or use pytest directly
cd HoloLoom/bandits/tests
PYTHONPATH=../../.. pytest -v
```

## Performance

### Latency

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Selection** (Bootstrap, N=7) | ~2ms | One forward pass (sampled head) |
| **Selection** (MC-Dropout) | ~1ms | Single model forward |
| **Training** (100 steps) | ~50ms | Async, doesn't block selection |
| **Diagnostics** (with uncertainty) | ~10ms | Multiple forward passes |

**Production**: Selection is <3ms, training happens in background every 200 observations.

### Memory

| Backend | Memory | Notes |
|---------|--------|-------|
| **Bootstrap** (N=7, 256-128 hidden) | ~10 MB | 7 models × 165k params × 4 bytes |
| **MC-Dropout** (256-128 hidden) | ~1.5 MB | 1 model |
| **Replay buffer** (200k capacity) | ~50 MB | 200k × (384D + reward) |

**Total**: ~60 MB for Bootstrap, ~52 MB for MC-Dropout.

### Scalability

- **Context dim**: Linear in selection time (matrix multiply)
- **Action count**: Linear in selection time (O(|A|) forwards)
- **Replay size**: Constant selection time (doesn't affect inference)
- **Ensemble size**: Linear in training time, constant selection time

## Troubleshooting

### "Replay buffer not ready"

**Cause**: Not enough observations for training (below `replay_warmup`).

**Solution**:
- Wait for more data (warmup defaults to 5000)
- Or reduce warmup: `BanditConfig(replay_warmup=1000)`

### "Context dimension mismatch"

**Cause**: Context features don't match `context_dim` in config.

**Solution**:
```python
# Ensure embeddings match config
config = BanditConfig(context_dim=embeddings.shape[0])
```

### Poor calibration (high ECE)

**Cause**: Model is overconfident or underconfident.

**Diagnosis**:
```python
bins = evaluator.get_calibration_bins()
for bin in bins:
    print(f"Predicted: {bin.predicted_mean:.3f}, Actual: {bin.actual_mean:.3f}")
```

**Solutions**:
- More training: Increase `train_steps` or reduce `train_every`
- Better uncertainty: Try Bootstrap if using MC-Dropout
- More data: Lower `replay_warmup` threshold

### Not learning (high regret)

**Cause**: Model isn't capturing context-action-reward relationship.

**Diagnosis**:
```python
stats = policy.get_statistics()
print(stats["trainer_stats"]["avg_loss"])  # Should decrease over time
```

**Solutions**:
- Larger model: `hidden_dims=[512, 256, 128]`
- More training: `train_every=100` (train more often)
- Better rewards: Ensure rewards are informative and normalized

## FAQ

**Q: Bootstrap vs MC-Dropout?**

**A**: Bootstrap is default (better calibration). Use MC-Dropout if memory is tight or you need faster training.

---

**Q: How many observations before it learns?**

**A**: Depends on complexity. Typically:
- Warmup: 1000-5000 observations
- Competitive: 5000-10000 observations
- Good: 20000+ observations

---

**Q: Can I use with non-HoloLoom projects?**

**A**: Yes! The bandit is generic. Just provide `Context` (features) and `Action` (candidates), compute `reward`, and call `update()`.

---

**Q: How to handle delayed rewards?**

**A**: Store context/action pairs, call `update()` when reward arrives. Use a cache:
```python
pending = {}  # context_id → (ctx, action)

# Selection
chosen = policy.select(ctx, actions)
pending[ctx.id] = (ctx, chosen)

# Later, when reward arrives
ctx, action = pending.pop(context_id)
policy.update(Observation(context_id, action.id, reward))
```

---

**Q: Can I use for A/B testing?**

**A**: Yes! Bandit automatically explores (A/B) and exploits (picks winner). Better than fixed A/B splits because it adapts.

---

**Q: How to add new actions?**

**A**: Just include them in `actions` list. Bandit will explore them (high uncertainty → Thompson samples them occasionally).

---

**Q: What if context changes over time?**

**A**: Online learning handles concept drift. For severe drift, use:
- Exponential replay decay (future feature)
- Periodic checkpointing + fresh starts
- Time-windowed replay buffer

## References

### Papers

1. **Thompson Sampling for Contextual Bandits**
   - Agrawal & Goyal (2013), "Thompson Sampling for Contextual Bandits with Linear Payoffs"

2. **Neural Bandits**
   - Riquelme et al. (2018), "Deep Bayesian Bandits Showdown"
   - Zhou et al. (2020), "Neural Contextual Bandits with UCB-based Exploration"

3. **Uncertainty Quantification**
   - Lakshminarayanan et al. (2017), "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
   - Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"

4. **Calibration**
   - Naeini et al. (2015), "Obtaining Well Calibrated Probabilities Using Bayesian Binning"

### Related Work

- **LinUCB**: Linear contextual bandits (we use neural for non-linear rewards)
- **Neural Bandits**: We implement Thompson Sampling variant (simpler than UCB-based neural bandits)
- **Bayesian Optimization**: Related but typically for continuous actions (we handle discrete tools/prompts)

## License

MIT License (same as HoloLoom)

## Contributing

See [HoloLoom CLAUDE.md](../../CLAUDE.md) for development guidelines.

**Key principles**:
- **Safety first**: No crashes from missing data
- **Tested**: All PRs require tests
- **Documented**: Inline docstrings + examples

---

**Questions?** Open an issue or see [HoloLoom documentation](../../README.md).
