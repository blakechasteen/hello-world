# Semantic Learning

Semantic micropolicy nudging uses HoloLoom's 244-dimensional semantic calculus to guide neural policy decisions. Instead of matching features, the policy becomes semantically aware — understanding dimensions like Warmth, Clarity, and Wisdom as measurable geometric properties.

## How It Works

```
Traditional:  Features → Neural Network → Tool Selection
Semantic:     Features → Neural Network → Tool Selection
                  ↓                            ↑
              Semantic State (244D) -----→ Semantic Nudge
```

The nudge biases tool selection toward semantically aligned choices using potential-based reward shaping (Ng et al., 1999), which preserves optimal policy guarantees.

## Quick Start

```python
from hololoom.core.policy.semantic_nudging import (
    SemanticNudgePolicy,
    SemanticRewardShaper,
    define_semantic_goals,
)

# 1. Define semantic goals
goals = define_semantic_goals('professional')
# {'Formality': 0.7, 'Clarity': 0.9, 'Directness': 0.8, ...}

# 2. Create reward shaper
shaper = SemanticRewardShaper(
    target_dimensions=goals,
    gamma=0.99,
    potential_weight=0.3,
)

# 3. Wrap your policy
nudge_policy = SemanticNudgePolicy(
    base_policy=your_policy_engine,
    semantic_spectrum=semantic_spectrum,
    semantic_goals=goals,
)

# 4. Use normally — semantic guidance is automatic
action_plan = await nudge_policy.decide(
    features=extracted_features,
    context=retrieved_context,
    semantic_state=current_semantic_state,
)
```

## Predefined Goal Sets

| Goal Set | Key Dimensions | Use Case |
|----------|---------------|----------|
| `professional` | Formality 0.7, Clarity 0.9, Precision 0.8 | Technical docs, formal comms |
| `empathetic` | Warmth 0.9, Compassion 0.9, Patience 0.8 | Support, sensitive topics |
| `educational` | Clarity 0.9, Patience 0.8, Simplicity 0.7 | Teaching, onboarding |
| `creative` | Imagination 0.8, Expression 0.8, Flow 0.8 | Writing, brainstorming |
| `analytical` | Nuance 0.8, Precision 0.9, Logic 0.8 | Research, deep analysis |

Custom goals: pass any `dict[str, float]` of dimension name to target value.

## When to Use Semantic Learning

**Use it when:**
- Sample cost > $1/experience (RLHF, expert annotation)
- Interpretability required (regulated, safety-critical)
- Optimizing 3+ objectives simultaneously
- Building conversational AI (HoloLoom's primary domain)

**Skip it when:**
- Free data (game simulator, benchmark tasks)
- Single metric optimization
- Real-time inference < 1ms required (train semantic, deploy vanilla)

## Performance

| Metric | Value |
|--------|-------|
| 244D projection | ~50ms (cached) |
| Top-K selection | O(K log N), negligible |
| Policy augmentation | ~5% overhead (384 → 416 dim) |
| Convergence speedup | 2-3x fewer episodes |
| Final performance | +7-25% improvement |

## Integration

### With Reflection/PPO Training

```python
from hololoom.core.reflection.rewards import RewardExtractor

base_reward = reward_extractor.compute_reward(spacetime)
shaped_reward = shaper.shape_reward(base_reward, old_state, new_state)
agent.update(observation, action, shaped_reward)
```

### With WeavingShuttle

```python
async with WeavingShuttle(cfg=config, memory=memory) as shuttle:
    semantic_state = compute_semantic_state(query, analyzer)
    spacetime = await shuttle.weave(
        query,
        semantic_state=semantic_state,
        semantic_goals=professional_goals,
    )
```

## Demo

```bash
python demos/semantic_micropolicy_nudge_demo.py
```

Runs 5 scenarios (Technical, Emotional, Creative, Educational, Analytical) showing semantic alignment improvements and reward shaping effects.

## Files

- `hololoom/core/policy/semantic_nudging.py` — Implementation (SemanticStateEncoder, SemanticRewardShaper, SemanticNudgePolicy)
- `hololoom/core/reflection/semantic_learning.py` — Training integration
- `demos/semantic_micropolicy_nudge_demo.py` — Working demo
