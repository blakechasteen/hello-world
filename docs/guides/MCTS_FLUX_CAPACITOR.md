# MCTS Flux Capacitor

Monte Carlo Tree Search integrated with Thompson Sampling for tool selection decisions.

**Location:** `hololoom/core/convergence/mcts_engine.py`

## How It Works

MCTS explores decision trees by simulating outcomes. Combined with Thompson Sampling, it balances exploration (trying new tools) and exploitation (using proven tools).

```
Query arrives
    |
    v
MCTS Root Node
    |-- Tool A (visits: 120, reward: 0.82)
    |   |-- Sub-option A1
    |   |-- Sub-option A2
    |-- Tool B (visits: 45, reward: 0.91)
    |-- Tool C (visits: 8, reward: 0.67)
    |
    v
Selection: UCB1 score = reward + C * sqrt(ln(N) / n_i)
    |
    v
Tool B selected (high reward + enough visits)
```

## Architecture

### Selection (UCB1)

```python
ucb_score = mean_reward + C * sqrt(ln(total_visits) / node_visits)
```

- `C = sqrt(2)` balances exploration/exploitation
- High reward + few visits = explore this node
- High reward + many visits = exploit this node

### Thompson Sampling Integration

Each tool maintains a Beta distribution:

```python
# Tool A: Beta(alpha=82, beta=38) -> E[reward] = 0.68
# Tool B: Beta(alpha=91, beta=9)  -> E[reward] = 0.91

# Sample from each distribution
sample_a = np.random.beta(82, 38)  # ~0.65-0.75
sample_b = np.random.beta(91, 9)   # ~0.85-0.95

# Select highest sample (Thompson Sampling)
selected = max(tools, key=lambda t: sample(t.alpha, t.beta))
```

### Simulation

Run N simulations (default: 50) to estimate tool effectiveness:

```python
from hololoom.core.convergence.mcts_engine import MCTSEngine

engine = MCTSEngine(simulations=50, exploration_constant=1.414)
result = engine.search(root_state, available_tools)

print(f"Selected: {result.best_tool}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Visits: {result.visit_counts}")
```

## Integration with Weaving

MCTS runs during stage 5 (WarpSpace) of the weaving cycle:

```
1. LoomCommand    -> Pattern selection
2. ChronoTrigger  -> Temporal window
3. ResonanceShed  -> Feature extraction
4. SynthesisBridge -> Pattern enrichment
5. WarpSpace      -> MCTS + Thompson Sampling (here)
6. ConvergenceEngine -> Collapse to discrete action
7. Spacetime      -> Output with provenance
```

## Configuration

```python
from hololoom.config import Config

config = Config.fused()  # Uses MCTS (50 simulations)
config = Config.fast()   # Uses simplified selection (5 simulations)
config = Config.bare()   # No MCTS (direct Thompson Sampling)
```

## Performance

| Simulations | Latency | Decision Quality |
|-------------|---------|-----------------|
| 5 (FAST) | ~2ms | Good |
| 50 (FUSED) | ~15ms | Better |
| 200 | ~60ms | Best (diminishing returns) |
