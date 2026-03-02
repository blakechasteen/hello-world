**# HoloLoom Thompson Sampling Core - Unified TS Underlayer**

**Complete Thompson Sampling ecosystem for exploration/exploitation across all decision domains.**

## Overview

`HoloLoom/ts_core/` provides a **unified interface** for Thompson Sampling across:

1. **Discrete Multi-Armed Bandits** (Beta-Bernoulli) - A/B testing, agent routing
2. **Bayesian Linear Contextual Bandits** - News recommendation, linear rewards
3. **Neural Contextual Bandits** - Deep contextual bandits with embeddings
4. **Gaussian Process TS** - Continuous optimization, hyperparameter tuning
5. **Deep Kernel GP-TS** (future) - Neural embedding + GP kernel

**All models implement the same `ThompsonSampler` protocol** - swap algorithms without changing code.

## Quick Start

### One-Line Usage

```python
from HoloLoom.ts_core import create_thompson_sampler

# Discrete MAB (A/B testing)
sampler = create_thompson_sampler("discrete", n_arms=2)
arm = sampler.select()
sampler.update(arm, reward=1.0)

# Bayesian Linear (contextual)
sampler = create_thompson_sampler("linear", context_dim=50, n_actions=10)
action = sampler.select(context)
sampler.update(context, action, reward)

# Neural Bandit (deep contextual)
sampler = create_thompson_sampler("neural", context_dim=384, n_actions=5)
# See HoloLoom.bandits for full neural API

# GP-TS (continuous optimization)
sampler = create_thompson_sampler("gp", param_dim=10)
params = sampler.select()
sampler.update(params, reward)
```

## Models

### 1. Discrete Bernoulli (Beta-Bernoulli MAB)

**Use Cases**: A/B testing, agent routing, simple tool selection

**Model**:
```
Prior: θ_a ~ Beta(α, β)
Likelihood: r_a ~ Bernoulli(θ_a)
Posterior: θ_a | data ~ Beta(α + successes, β + failures)
```

**Example**:
```python
from HoloLoom.ts_core.models.discrete_bernoulli import DiscreteBernoulliTS

# A/B test
sampler = DiscreteBernoulliTS(n_arms=2, alpha_prior=1.0, beta_prior=1.0)

for trial in range(1000):
    arm = sampler.select()  # 0 or 1
    reward = run_experiment(arm)  # 0 or 1
    sampler.update(arm, reward)

stats = sampler.get_statistics()
print(f"Best arm: {stats['best_arm']}")
print(f"Posterior means: {stats['posterior_means']}")
```

**Features**:
- Closed-form updates (no optimization)
- Fast (~0.01ms per select)
- Perfect for binary rewards
- Well-calibrated uncertainty

**Configuration**:
```python
# Uniform prior (no bias)
sampler = DiscreteBernoulliTS(n_arms=5, alpha_prior=1.0, beta_prior=1.0)

# Optimistic prior (encourages exploration)
sampler = DiscreteBernoulliTS(n_arms=5, alpha_prior=10.0, beta_prior=1.0)

# Using factory
from HoloLoom.ts_core.models.discrete_bernoulli import create_discrete_ts
sampler = create_discrete_ts(n_arms=5, prior="optimistic")
```

### 2. Bayesian Linear (Gaussian Contextual Bandit)

**Use Cases**: News recommendation, ads, linear reward models

**Model**:
```
Prior: θ_a ~ N(μ_a, Σ_a)
Likelihood: r(x, a) = x^T θ_a + ε, ε ~ N(0, σ²)
Posterior: Gaussian (closed-form conjugate update)
```

**Example**:
```python
from HoloLoom.ts_core.models.bayes_linear import BayesianLinearTS

# News article recommendation
sampler = BayesianLinearTS(context_dim=50, n_actions=10)

for user in users:
    context = user.get_features()  # [50]
    article = sampler.select(context)  # 0-9
    clicked = user.clicked()  # True/False
    reward = 1.0 if clicked else 0.0
    sampler.update(context, article, reward)

stats = sampler.get_statistics()
print(f"Best action: {stats['best_action']}")
```

**Features**:
- Uses context (unlike discrete MAB)
- Closed-form updates (Gaussian conjugate)
- Fast (~1ms per select)
- Interpretable (linear weights)

**Configuration**:
```python
# Default prior
sampler = BayesianLinearTS(
    context_dim=50,
    n_actions=10,
    lambda_prior=1.0,  # Prior precision
    sigma_noise=1.0     # Observation noise
)

# Using factory
from HoloLoom.ts_core.models.bayes_linear import create_bayesian_linear_ts
sampler = create_bayesian_linear_ts(
    context_dim=50,
    n_actions=10,
    prior="tight"  # Less exploration
)
```

### 3. Neural Contextual Bandit

**Use Cases**: Tool selection with embeddings, non-linear rewards

**See**: [HoloLoom/bandits/README.md](../bandits/README.md) for full documentation.

**Example**:
```python
from HoloLoom.ts_core import create_thompson_sampler

# HoloLoom tool selection
sampler = create_thompson_sampler(
    "neural",
    context_dim=384,      # Matryoshka embeddings
    action_dim=0,         # Context-only
    hidden_dims=[256, 128],
    backend="bootstrap",  # or "mc_dropout"
    n_ensemble=7
)

# Use same API as above, but with Action objects
# (See bandits/README.md for details)
```

### 4. Gaussian Process Thompson Sampling

**Use Cases**: Hyperparameter tuning, continuous control, black-box optimization

**Model**:
```
Prior: f ~ GP(0, k(x, x'))
Likelihood: y = f(x) + ε, ε ~ N(0, σ²)
Posterior: GP(μ_post, k_post)
```

**Example**:
```python
from HoloLoom.ts_core.models.gp_ts import GaussianProcessTS

# Hyperparameter tuning
sampler = GaussianProcessTS(
    param_dim=5,
    kernel="rbf",
    length_scale=0.5,
    bounds=(0.0, 1.0)
)

for iteration in range(100):
    params = sampler.select(n_candidates=100)  # [5]
    reward = train_model_with_params(params)
    sampler.update(params, reward)

best_params = sampler.get_best_params()
```

**Kernels**:
- **RBF**: `k(x, x') = exp(-||x - x'||² / (2l²))` - smooth functions
- **Matern**: More flexible smoothness

**Configuration**:
```python
# RBF kernel (smooth)
sampler = GaussianProcessTS(
    param_dim=10,
    kernel="rbf",
    length_scale=1.0,
    noise_std=0.1,
    bounds=(0.0, 1.0)
)

# Using factory
from HoloLoom.ts_core.models.gp_ts import create_gp_ts
sampler = create_gp_ts(param_dim=10, kernel="matern")
```

**Note**: This is a simplified GP implementation. For production GP-TS, consider using GPyTorch or scikit-learn's GaussianProcessRegressor.

## Unified Factory

### create_thompson_sampler()

**One function to rule them all**:

```python
from HoloLoom.ts_core import create_thompson_sampler

# Discrete
sampler = create_thompson_sampler(
    "discrete",
    n_arms=5,
    alpha_prior=1.0,
    beta_prior=1.0
)

# Linear
sampler = create_thompson_sampler(
    "linear",
    context_dim=50,
    n_actions=10,
    lambda_prior=1.0,
    sigma_noise=1.0
)

# Neural
sampler = create_thompson_sampler(
    "neural",
    context_dim=384,
    action_dim=0,
    hidden_dims=[256, 128],
    backend="bootstrap"
)

# GP
sampler = create_thompson_sampler(
    "gp",
    param_dim=10,
    kernel="rbf",
    bounds=(0.0, 1.0)
)
```

### Convenience Functions

```python
from HoloLoom.ts_core.samplers import (
    create_discrete_mab,
    create_contextual_bandit,
    create_continuous_optimizer,
)

# Discrete MAB
sampler = create_discrete_mab(n_arms=5)

# Contextual bandit (linear or neural)
sampler = create_contextual_bandit(
    context_dim=50,
    n_actions=10,
    model="linear"  # or "neural"
)

# Continuous optimizer (GP-TS)
sampler = create_continuous_optimizer(param_dim=10)
```

## Choosing the Right Model

| Model | Context? | Continuous? | Speed | Use When |
|-------|----------|-------------|-------|----------|
| **Discrete Bernoulli** | No | No | ⚡⚡⚡ | A/B testing, simple MAB, binary rewards |
| **Bayesian Linear** | Yes | No | ⚡⚡ | Linear rewards, news/ads, interpretable |
| **Neural Bandit** | Yes | No | ⚡ | Non-linear rewards, embeddings, HoloLoom tools |
| **GP-TS** | N/A | Yes | ⚠️ | Hyperparameter tuning, continuous control |

**Rules of thumb**:
- **No context, discrete actions** → Discrete Bernoulli
- **Context, linear rewards** → Bayesian Linear
- **Context, non-linear rewards** → Neural Bandit
- **Continuous parameters** → GP-TS

## Testing

### Run All Tests

```bash
# All ts_core tests (22 tests)
PYTHONPATH=. pytest HoloLoom/ts_core/tests/test_ts_models.py -v
```

**Test Coverage**:
- ✅ Discrete Bernoulli: init, select, update, learning, factory
- ✅ Bayesian Linear: init, select, update, learning, factory
- ✅ GP-TS: init, select, update, optimization, factory
- ✅ Unified factory: all model types
- ✅ Convenience functions
- ✅ Integration: discrete vs linear comparison

**Output**:
```
======================== 22 passed in 11.85s ==========================
```

### Example Test

```python
def test_discrete_learning():
    """Test that discrete TS learns best arm."""
    sampler = DiscreteBernoulliTS(n_arms=3, seed=42)

    # True probabilities: arm 2 is best
    true_probs = [0.3, 0.5, 0.9]

    # Run bandit
    for _ in range(200):
        arm = sampler.select()
        reward = float(np.random.rand() < true_probs[arm])
        sampler.update(arm, reward)

    # Should identify arm 2 as best
    stats = sampler.get_statistics()
    assert stats["best_arm"] == 2
```

## Performance

| Model | Selection | Update | Memory |
|-------|-----------|--------|--------|
| Discrete | ~0.01ms | ~0.001ms | O(K) arms |
| Linear | ~1ms | ~2ms | O(KD²) |
| Neural | ~2ms | ~50ms (batched) | ~60MB |
| GP | ~10ms | ~1ms | O(N²) observations |

**Notes**:
- Discrete and Linear have closed-form updates (no optimization)
- Neural training is batched (every 200 observations)
- GP time grows quadratically with observations (use inducing points for scaling)

## Integration with HoloLoom

### Tool Selection

```python
from HoloLoom.ts_core import create_thompson_sampler

# Neural bandit for tool selection
tool_sampler = create_thompson_sampler(
    "neural",
    context_dim=384,  # Matryoshka embeddings
    action_dim=0,
    hidden_dims=[256, 128]
)

# Inside WeavingOrchestrator
async def weave(query):
    embeddings = await self.embed(query.text)
    ctx = Context(id=query.id, x=embeddings)

    tools = [Action(id="answer"), Action(id="search"), Action(id="explain")]
    chosen = tool_sampler.select(ctx, tools)

    result = await self.execute_tool(chosen.id, query)
    reward = result.confidence - result.cost * 0.1

    tool_sampler.update(Observation(ctx.id, chosen.id, reward))
    return result
```

### Agent Routing

```python
from HoloLoom.ts_core import create_discrete_mab

# Discrete MAB for routing queries to agents
router = create_discrete_mab(n_arms=5)  # 5 agents

for query in queries:
    agent_id = router.select()
    result = agents[agent_id].process(query)
    reward = 1.0 if result.success else 0.0
    router.update(agent_id, reward)
```

### Hyperparameter Tuning

```python
from HoloLoom.ts_core import create_continuous_optimizer

# GP-TS for tuning learning rates, temperatures, etc.
optimizer = create_continuous_optimizer(
    param_dim=3,  # [lr, temperature, dropout]
    kernel="rbf",
    bounds=(0.0, 1.0)
)

for iteration in range(50):
    params = optimizer.select()
    lr, temp, dropout = params

    reward = train_model(lr=lr, temp=temp, dropout=dropout)
    optimizer.update(params, reward)

best = optimizer.get_best_params()
```

## Advanced Usage

### Custom Priors (Discrete)

```python
# Informative prior (arm 0 is better a priori)
sampler = DiscreteBernoulliTS(
    n_arms=3,
    alpha_prior=1.0,
    beta_prior=1.0
)

# Manually set prior for arm 0
sampler.alpha[0] = 10.0  # Optimistic about arm 0
sampler.beta[0] = 1.0
```

### Diagnostics

```python
# Get detailed diagnostics
arm, diag = sampler.select(return_diagnostics=True)

print(diag["sampled_thetas"])    # Sampled success probabilities
print(diag["posterior_means"])   # E[θ_a] for each arm
print(diag["posterior_vars"])    # Var[θ_a]
```

### Reset

```python
# Reset posterior to prior
sampler.reset()  # All arms
sampler.reset(arm=2)  # Single arm
```

### Posterior Access

```python
# Discrete
params = sampler.get_posterior_params()
print(params[0])  # (alpha_0, beta_0)

# Linear
mu, Sigma = sampler.get_posterior_params(action=0)
print(mu.shape)  # [context_dim]
print(Sigma.shape)  # [context_dim, context_dim]
```

## FAQ

**Q: When to use discrete vs contextual?**

**A**: Discrete if rewards don't depend on context (same for all users/queries). Contextual if rewards vary by context (different users have different preferences).

---

**Q: Linear vs neural bandits?**

**A**: Linear if rewards are linear in context (interpretable, fast). Neural if non-linear (more flexible, requires more data).

---

**Q: How many observations before it learns?**

**A**:
- Discrete: ~50-200 per arm
- Linear: ~100-500 total
- Neural: ~1000-5000 (see [bandits/README.md](../bandits/README.md))
- GP: ~20-100 (depends on dimensionality)

---

**Q: Can I use with delayed rewards?**

**A**: Yes, cache observations and call `update()` when reward arrives:

```python
pending = {}  # context_id → (context, action)

# Selection
action = sampler.select(context)
pending[context_id] = (context, action)

# Later
context, action = pending.pop(context_id)
sampler.update(context, action, reward)
```

---

**Q: How to handle non-stationary rewards?**

**A**:
- Discrete: Use optimistic priors (α=10, β=1) or periodically reset poorly-performing arms
- Linear/Neural: Discount old data or use sliding window
- GP: Increase noise parameter or use time-varying kernels

---

**Q: What if I have millions of arms/actions?**

**A**: Use contextual bandits with action features:

```python
# Instead of discrete (infeasible with 1M arms)
sampler = BayesianLinearTS(
    context_dim=50,     # User features
    n_actions=1000000   # Too many!
)

# Use action features (linear)
sampler = BayesianLinearTS(
    context_dim=50 + 20,  # User + item features
    n_actions=1  # Single "match" action
)

# Or neural with embeddings
sampler = create_thompson_sampler(
    "neural",
    context_dim=50 + 20,  # Concat user and item embeddings
    action_dim=0
)
```

## References

### Papers

1. **Thompson Sampling**
   - Chapelle & Li (2011), "An Empirical Evaluation of Thompson Sampling"
   - Agrawal & Goyal (2013), "Thompson Sampling for Contextual Bandits with Linear Payoffs"

2. **Bayesian Linear Bandits**
   - Russo & Van Roy (2014), "Learning to Optimize via Posterior Sampling"

3. **Neural Bandits**
   - Riquelme et al. (2018), "Deep Bayesian Bandits Showdown"
   - See [bandits/README.md](../bandits/README.md) for more

4. **Gaussian Process TS**
   - Srinivas et al. (2010), "Gaussian Process Optimization in the Bandit Setting"
   - Kandasamy et al. (2018), "Neural Architecture Search with Bayesian Optimisation and Optimal Transport"

### Related Work

- **LinUCB**: Upper Confidence Bound variant (deterministic, less exploration)
- **ε-greedy**: Simple baseline (poor exploration)
- **UCB**: Optimistic exploration (deterministic)
- **Bayesian Optimization**: Related to GP-TS but typically for fewer, more expensive evaluations

## License

MIT License (same as HoloLoom)

## Contributing

See [HoloLoom CLAUDE.md](../../CLAUDE.md) for development guidelines.

**Key principles**:
- Protocol-based design (easy to swap implementations)
- Comprehensive testing (all PRs need tests)
- Graceful degradation (no crashes from edge cases)

---

**Questions?** See [HoloLoom documentation](../../README.md) or open an issue.
