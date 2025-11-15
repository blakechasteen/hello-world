# Convergence Engine - Decision Collapse

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/convergence/`
**Code**: 832 lines across 3 files

---

## Overview

The **Convergence Engine** collapses continuous probability distributions from the Policy Engine into discrete tool selections. It implements the critical transition from **continuous → discrete** that bridges neural decision-making with executable actions.

**Philosophy**: The convergence is the moment of decision—where flowing probabilities crystallize into a single chosen tool.

---

## Architecture

### File Structure

```
HoloLoom/convergence/
├── __init__.py          # 17 lines - Public exports
├── engine.py            # 406 lines - Main convergence engine
└── mcts_engine.py       # 409 lines - MCTS-based planning
```

### Core Concept: Collapse

**Continuous → Discrete Transition**:
```
Policy Engine outputs:
    tool_probs = [0.65, 0.25, 0.07, 0.03]  # Probabilities
        ↓ Convergence Engine ↓
    chosen_tool = "answer"  # Discrete selection (index 0)
```

---

## Collapse Strategies

### 1. ARGMAX (Deterministic)

**Most Confident**: Select tool with highest probability.

```python
from HoloLoom.convergence import ConvergenceEngine, CollapseStrategy

engine = ConvergenceEngine(strategy=CollapseStrategy.ARGMAX)

# Collapse distribution
tool_probs = np.array([0.65, 0.25, 0.07, 0.03])
chosen_idx = engine.collapse(tool_probs)

print(f"Chosen: {chosen_idx}")  # 0 (highest probability)
```

**When to use**: Production systems requiring consistency.

### 2. EPSILON_GREEDY (Exploration)

**Mostly Exploit, Sometimes Explore**: 90% argmax, 10% random.

```python
engine = ConvergenceEngine(
    strategy=CollapseStrategy.EPSILON_GREEDY,
    epsilon=0.1  # 10% exploration
)

# 90% chance: index 0 (argmax)
# 10% chance: random selection
chosen_idx = engine.collapse(tool_probs)
```

**When to use**: Online learning, need some exploration.

### 3. BAYESIAN_BLEND (Balanced)

**Blend Neural + Thompson Sampling**: Combine neural probs with bandit priors.

```python
from HoloLoom.policy.thompson_sampling import TSBandit

# Create bandit with priors
bandit = TSBandit(n_arms=4)

engine = ConvergenceEngine(
    strategy=CollapseStrategy.BAYESIAN_BLEND,
    bandit=bandit,
    blend_weight=0.7  # 70% neural, 30% bandit
)

# Blended decision
chosen_idx = engine.collapse(tool_probs)
# Incorporates both neural confidence and bandit priors
```

**When to use**: Want both neural intelligence and exploration.

### 4. PURE_THOMPSON (Maximum Exploration)

**Thompson Sampling Only**: Ignore neural probs, use bandit.

```python
engine = ConvergenceEngine(
    strategy=CollapseStrategy.PURE_THOMPSON,
    bandit=bandit
)

# Neural probs ignored, uses Thompson Sampling
chosen_idx = engine.collapse(tool_probs)
```

**When to use**: Research mode, maximum exploration.

### 5. MCTS (Tree Search)

**Monte Carlo Tree Search**: Plan ahead with lookahead.

```python
from HoloLoom.convergence.mcts_engine import MCTSEngine

# Create MCTS engine
engine = MCTSEngine(
    n_simulations=100,  # 100 rollouts
    exploration_weight=1.4,  # UCT constant
    max_depth=5  # Lookahead depth
)

# Plan with tree search
chosen_idx = await engine.plan(
    tool_probs,
    state=current_state,
    reward_fn=reward_function
)
```

**When to use**: Complex planning tasks, need lookahead.

---

## Usage Examples

### Example 1: Basic Collapse

```python
from HoloLoom.convergence import ConvergenceEngine, CollapseStrategy
import numpy as np

# Create engine
engine = ConvergenceEngine(strategy=CollapseStrategy.ARGMAX)

# Tool probabilities from policy
tool_probs = np.array([
    0.65,  # answer
    0.25,  # search
    0.07,  # notion_write
    0.03   # calc
])

# Collapse to discrete choice
chosen_idx = engine.collapse(tool_probs)
tools = ["answer", "search", "notion_write", "calc"]

print(f"Chosen tool: {tools[chosen_idx]}")  # "answer"
```

### Example 2: Epsilon-Greedy Exploration

```python
# 10% exploration
engine = ConvergenceEngine(
    strategy=CollapseStrategy.EPSILON_GREEDY,
    epsilon=0.1
)

# Run 100 times to see exploration
choices = []
for _ in range(100):
    chosen = engine.collapse(tool_probs)
    choices.append(chosen)

# Distribution:
# ~90 times: index 0 (argmax)
# ~10 times: random (exploration)
from collections import Counter
print(Counter(choices))
# {0: 91, 1: 5, 2: 3, 3: 1}
```

### Example 3: Bayesian Blend

```python
from HoloLoom.policy.thompson_sampling import TSBandit

# Create bandit with history
bandit = TSBandit(n_arms=4)
bandit.update(0, reward=0.8)  # answer worked well
bandit.update(1, reward=0.5)  # search was okay

# Blend neural + bandit
engine = ConvergenceEngine(
    strategy=CollapseStrategy.BAYESIAN_BLEND,
    bandit=bandit,
    blend_weight=0.7  # 70% neural, 30% bandit
)

# Blended probabilities
chosen = engine.collapse(tool_probs)
# Considers both neural confidence and bandit history
```

### Example 4: MCTS Planning

```python
from HoloLoom.convergence.mcts_engine import MCTSEngine

# Create MCTS engine
mcts = MCTSEngine(
    n_simulations=100,
    exploration_weight=1.4,
    max_depth=5
)

# Define reward function
def reward_fn(state, action):
    # Simulate reward
    if action == 0:  # answer
        return 0.8
    elif action == 1:  # search
        return 0.6
    return 0.3

# Plan with lookahead
chosen = await mcts.plan(
    tool_probs,
    state={'context': 'current_context'},
    reward_fn=reward_fn
)

print(f"MCTS chose: {tools[chosen]}")
# Considers future rewards, not just immediate
```

---

## Integration with Orchestrator

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.convergence import ConvergenceEngine, CollapseStrategy

# Create convergence engine
engine = ConvergenceEngine(strategy=CollapseStrategy.ARGMAX)

# Orchestrator uses engine for decision collapse
orchestrator = WeavingOrchestrator(
    convergence_engine=engine,
    # ... other config
)

# During weaving:
# 1. Policy outputs tool_probs: [0.65, 0.25, 0.07, 0.03]
# 2. Convergence Engine collapses to discrete: chosen_idx = 0
# 3. Tool executed: "answer"
```

---

## MCTS Engine (Advanced)

### Monte Carlo Tree Search

**Purpose**: Plan ahead with tree search, considering future rewards.

**Algorithm**:
```
1. Selection: Traverse tree using UCT (Upper Confidence Bound for Trees)
2. Expansion: Add new node to tree
3. Simulation: Rollout to terminal state
4. Backpropagation: Update values along path
```

**UCT Formula**:
```
UCT(node) = exploitation + exploration
          = Q(node) / N(node) + c * sqrt(log(N(parent)) / N(node))
```

**Usage**:
```python
from HoloLoom.convergence.mcts_engine import MCTSEngine, MCTSConfig

# Configure MCTS
config = MCTSConfig(
    n_simulations=100,       # Rollouts per decision
    exploration_weight=1.4,  # c in UCT formula
    max_depth=5,             # Lookahead depth
    discount_factor=0.99     # Future reward discount
)

engine = MCTSEngine(config)

# Plan action
action = await engine.plan(
    tool_probs,
    state=current_state,
    reward_fn=reward_function
)
```

**When to use**: Multi-step planning, games, sequential decision-making.

---

## API Reference

### Core Classes

#### `ConvergenceEngine.__init__()`
```python
def __init__(
    self,
    strategy: CollapseStrategy = CollapseStrategy.ARGMAX,
    epsilon: float = 0.1,
    bandit: Optional[TSBandit] = None,
    blend_weight: float = 0.7
)
```

#### `ConvergenceEngine.collapse()`
```python
def collapse(
    self,
    tool_probs: np.ndarray  # [n_tools] probabilities
) -> int  # Chosen tool index
```

#### `MCTSEngine.__init__()`
```python
def __init__(
    self,
    n_simulations: int = 100,
    exploration_weight: float = 1.4,
    max_depth: int = 5,
    discount_factor: float = 0.99
)
```

#### `MCTSEngine.plan()`
```python
async def plan(
    self,
    tool_probs: np.ndarray,
    state: Dict[str, Any],
    reward_fn: Callable
) -> int  # Chosen action
```

### Enums

```python
class CollapseStrategy(Enum):
    ARGMAX = "argmax"                    # Deterministic
    EPSILON_GREEDY = "epsilon_greedy"    # 90% exploit, 10% explore
    BAYESIAN_BLEND = "bayesian_blend"    # Neural + bandit
    PURE_THOMPSON = "pure_thompson"      # Bandit only
```

---

## Performance

| Strategy | Latency | Notes |
|----------|---------|-------|
| **ARGMAX** | <0.01ms | np.argmax |
| **EPSILON_GREEDY** | <0.01ms | Random check + argmax |
| **BAYESIAN_BLEND** | <0.5ms | Bandit sampling |
| **PURE_THOMPSON** | <0.5ms | Bandit sampling |
| **MCTS (100 sims)** | ~50ms | Tree search |

**Memory**: <1KB (negligible)

---

## Dependencies

**Internal**:
```python
from HoloLoom.policy.thompson_sampling import TSBandit
from HoloLoom.documentation.types import ActionPlan
```

**External**:
```python
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Dict, Any
```

---

## Summary

The Convergence Engine provides:

✅ **Continuous → discrete collapse** (probabilities → tool selection)
✅ **5 collapse strategies** (argmax, epsilon-greedy, Bayesian blend, Thompson, MCTS)
✅ **Thompson Sampling integration** (exploration via bandits)
✅ **MCTS planning** (lookahead with tree search)
✅ **Sub-millisecond latency** (<0.5ms for non-MCTS strategies)
✅ **Flexible exploration** (deterministic to maximum exploration)

The Convergence Engine is where decisions crystallize—the moment probabilities become actions.
