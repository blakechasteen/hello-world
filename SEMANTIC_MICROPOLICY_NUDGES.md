# Semantic Micropolicy Nudges: 244D-Guided Decision Making

## Concept Overview

**Semantic Micropolicy Nudges** integrate the 244-dimensional semantic calculus with the neural policy engine to create **semantically-aware decision making**. Instead of selecting tools purely based on feature matching, the policy can be guided by understanding *where it is in semantic space* and *where it wants to go*.

## Core Innovation

The policy network currently operates on:
- **Features**: Motifs (linguistic patterns), embeddings (semantic vectors), spectral features (graph structure)
- **Context**: Retrieved memory shards, knowledge graph subgraphs
- **Objective**: Select best tool via neural network + Thompson Sampling

We add a new dimension: **Semantic Position and Trajectory**
- **Where are we?** Current position along 244 interpretable semantic dimensions
- **Where have we been?** Semantic trajectory history (velocity, acceleration)
- **Where should we go?** Target semantic regions for different interaction goals
- **How do we nudge?** Subtle biases toward desired semantic movements

## Architecture Components

### 1. Semantic State Encoder

Compresses the current 244D semantic position into a lower-dimensional representation suitable for policy input:

```python
class SemanticStateEncoder(nn.Module):
    """
    Encodes 244D semantic position into policy features.

    Architecture:
    - Input: 244D semantic projections
    - Sparse encoding (only top-K active dimensions)
    - Projection to policy dimension (384D)
    - Category aggregation (16 semantic categories)
    """

    def __init__(self, n_dims: int = 244, policy_dim: int = 384, top_k: int = 32):
        super().__init__()
        self.n_dims = n_dims
        self.top_k = top_k

        # Sparse projection: only attend to top-K dimensions
        self.sparse_proj = nn.Linear(top_k * 2, policy_dim // 2)  # value + velocity

        # Category aggregation: 16 categories from 244 dims
        self.category_proj = nn.Linear(16, policy_dim // 2)

        # Fusion
        self.fusion = nn.Linear(policy_dim, policy_dim)

    def forward(self, semantic_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode semantic state for policy.

        Args:
            semantic_state: Dict with:
                - 'position': [B, 244] current semantic projections
                - 'velocity': [B, 244] semantic velocity (rate of change)
                - 'categories': [B, 16] aggregated by category

        Returns:
            Encoded representation [B, policy_dim]
        """
        pos = semantic_state['position']  # [B, 244]
        vel = semantic_state['velocity']  # [B, 244]

        # Select top-K dimensions by absolute velocity (most active)
        topk_indices = torch.topk(torch.abs(vel), self.top_k, dim=-1).indices

        # Gather top-K positions and velocities
        topk_pos = torch.gather(pos, 1, topk_indices)  # [B, top_k]
        topk_vel = torch.gather(vel, 1, topk_indices)  # [B, top_k]

        # Concatenate position + velocity for sparse encoding
        sparse_features = torch.cat([topk_pos, topk_vel], dim=-1)  # [B, top_k*2]
        sparse_encoded = self.sparse_proj(sparse_features)  # [B, policy_dim//2]

        # Category aggregation
        category_features = semantic_state['categories']  # [B, 16]
        category_encoded = self.category_proj(category_features)  # [B, policy_dim//2]

        # Fuse
        fused = torch.cat([sparse_encoded, category_encoded], dim=-1)
        return self.fusion(fused)
```

### 2. Semantic Reward Shaping

Use semantic trajectory to create reward signals that guide policy toward desired semantic regions:

```python
class SemanticRewardShaper:
    """
    Shapes rewards based on semantic trajectory.

    Rewards movement toward desired semantic dimensions while
    preserving optimal policy (potential-based shaping).
    """

    def __init__(self, target_dimensions: Dict[str, float], gamma: float = 0.99):
        """
        Args:
            target_dimensions: Desired positions for key dimensions
                Example: {'Wisdom': 0.8, 'Compassion': 0.7, 'Clarity': 0.6}
            gamma: Discount factor for potential shaping
        """
        self.target_dims = target_dimensions
        self.gamma = gamma

    def compute_potential(self, semantic_state: Dict[str, float]) -> float:
        """
        Compute potential function value from semantic position.

        Potential increases as we approach target dimensions.

        Args:
            semantic_state: Current semantic projections (244D)

        Returns:
            Potential value (0-1)
        """
        # Compute distance to target for each desired dimension
        distances = []
        for dim_name, target_value in self.target_dims.items():
            if dim_name in semantic_state:
                current_value = semantic_state[dim_name]
                distance = abs(current_value - target_value)
                distances.append(distance)

        # Average distance (lower is better)
        avg_distance = np.mean(distances) if distances else 0.0

        # Convert to potential (0-1, higher near target)
        potential = 1.0 - np.clip(avg_distance / 2.0, 0.0, 1.0)

        return potential

    def shape_reward(
        self,
        base_reward: float,
        semantic_state_old: Dict[str, float],
        semantic_state_new: Dict[str, float]
    ) -> float:
        """
        Apply potential-based semantic reward shaping.

        F(s, a, s') = gamma * phi(s') - phi(s)
        Shaped reward = R + F

        This preserves the optimal policy while guiding toward
        semantically desirable regions.
        """
        potential_old = self.compute_potential(semantic_state_old)
        potential_new = self.compute_potential(semantic_state_new)

        shaping = self.gamma * potential_new - potential_old
        shaped_reward = base_reward + shaping

        return shaped_reward
```

### 3. Semantic Attention Gating

Use semantic dimensions to modulate attention heads in the policy network:

```python
class SemanticGatedMHA(nn.Module):
    """
    Multi-head attention with semantic dimension gating.

    Different semantic states activate different attention heads,
    allowing the policy to focus on different aspects of context
    based on where it is in semantic space.
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_semantic_categories: int = 16):
        super().__init__()
        self.mha = CustomMHA(d_model, n_heads)

        # Map semantic categories to attention gate activations
        self.semantic_gate_proj = nn.Sequential(
            nn.Linear(n_semantic_categories, n_heads * 2),
            nn.ReLU(),
            nn.Linear(n_heads * 2, n_heads),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        semantic_categories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with semantic gating.

        Args:
            x: Input tensor [B, T, D]
            semantic_categories: Category activations [B, 16]

        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute gates from semantic state
        gates = self.semantic_gate_proj(semantic_categories)  # [B, n_heads]

        # Apply gated attention
        out, attn = self.mha(x, gates)

        return out, attn
```

### 4. Semantic Goal-Directed Policy

Extend the policy to support semantic goals:

```python
class SemanticGoalPolicy:
    """
    Policy that can pursue semantic goals.

    Supports explicit semantic targets like:
    - "Increase Clarity while maintaining Warmth"
    - "Move toward Wisdom and Compassion"
    - "Reduce Complexity, increase Directness"
    """

    def __init__(
        self,
        base_policy: PolicyEngine,
        semantic_encoder: SemanticStateEncoder,
        reward_shaper: SemanticRewardShaper
    ):
        self.base_policy = base_policy
        self.semantic_encoder = semantic_encoder
        self.reward_shaper = reward_shaper

    async def decide_with_semantic_goal(
        self,
        features: Features,
        context: Context,
        semantic_state: Dict[str, float],
        semantic_goal: Optional[Dict[str, float]] = None
    ) -> ActionPlan:
        """
        Make decision with optional semantic goal guidance.

        Args:
            features: Extracted query features
            context: Retrieved context
            semantic_state: Current semantic position (244D)
            semantic_goal: Optional target dimensions to move toward

        Returns:
            ActionPlan with semantic guidance applied
        """
        # Encode semantic state
        semantic_features = self._encode_semantic_state(semantic_state)

        # Augment features with semantic information
        augmented_features = self._augment_features(features, semantic_features)

        # Get base decision
        action_plan = await self.base_policy.decide(augmented_features, context)

        # If semantic goal provided, adjust tool selection probabilities
        if semantic_goal:
            action_plan = self._apply_semantic_bias(
                action_plan,
                semantic_state,
                semantic_goal
            )

        return action_plan

    def _apply_semantic_bias(
        self,
        action_plan: ActionPlan,
        current_state: Dict[str, float],
        target_state: Dict[str, float]
    ) -> ActionPlan:
        """
        Apply subtle bias toward tools that move toward semantic goal.

        This is the "nudge" - we don't override the policy,
        just gently guide it toward semantically desirable directions.
        """
        # Estimate semantic trajectory for each tool
        tool_trajectories = self._estimate_tool_trajectories(current_state)

        # Compute alignment with goal for each tool
        alignments = {}
        for tool, trajectory in tool_trajectories.items():
            alignment = self._compute_goal_alignment(trajectory, target_state)
            alignments[tool] = alignment

        # Adjust action plan confidence with semantic bias
        # (subtle: 10% weight to semantic alignment)
        semantic_bias_weight = 0.1

        # Re-weight tool probabilities
        adjusted_probs = {}
        for tool, base_prob in action_plan.tool_probs.items():
            semantic_bonus = alignments.get(tool, 0.0) * semantic_bias_weight
            adjusted_probs[tool] = base_prob * (1.0 + semantic_bonus)

        # Renormalize
        total = sum(adjusted_probs.values())
        adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}

        # Update action plan
        action_plan.tool_probs = adjusted_probs

        return action_plan
```

## Use Cases

### 1. Conversational Tone Management

**Goal**: Maintain appropriate formality level while increasing clarity

```python
semantic_goals = {
    'Formality': 0.7,      # Professional tone
    'Clarity': 0.8,        # Very clear
    'Warmth': 0.6,         # Still friendly
    'Directness': 0.7      # Get to the point
}

reward_shaper = SemanticRewardShaper(target_dimensions=semantic_goals)
```

### 2. Narrative Arc Guidance

**Goal**: Guide story generation toward specific mythological patterns

```python
narrative_goals = {
    'Heroism': 0.8,        # Strong heroic themes
    'Quest': 0.7,          # Journey structure
    'Transformation': 0.9,  # Character growth
    'Conflict': 0.6,       # Moderate tension
    'Wisdom': 0.7          # Learned lessons
}
```

### 3. Ethical AI Behavior

**Goal**: Ensure responses align with ethical dimensions

```python
ethical_goals = {
    'Compassion': 0.9,     # Highly compassionate
    'Justice': 0.8,        # Fair and equitable
    'Respect': 0.9,        # Respectful tone
    'Authenticity': 0.8,   # Genuine responses
    'Responsibility': 0.9   # Accountable
}
```

### 4. Educational Adaptation

**Goal**: Adjust explanation complexity based on learner needs

```python
beginner_goals = {
    'Complexity': -0.5,    # Low complexity
    'Clarity': 0.9,        # Very clear
    'Patience': 0.8,       # Patient explanations
    'Encouragement': 0.7   # Supportive
}

expert_goals = {
    'Complexity': 0.7,     # Higher complexity OK
    'Nuance': 0.8,         # Subtle distinctions
    'Efficiency': 0.7,     # Concise
    'Precision': 0.9       # Very precise
}
```

## Implementation Strategy

### Phase 1: Semantic Feature Integration

1. Add semantic calculus to feature extraction pipeline
2. Compute 244D projections for each query
3. Aggregate into 16 semantic categories
4. Encode as additional policy features

### Phase 2: Reward Shaping

1. Implement potential-based semantic reward shaping
2. Define semantic goals for different interaction types
3. Track semantic trajectories in reflection buffer
4. Train policy with shaped rewards

### Phase 3: Attention Gating

1. Modify policy network to accept semantic category inputs
2. Add semantic gating to attention mechanisms
3. Learn which semantic states benefit from which attention patterns

### Phase 4: Goal-Directed Behavior

1. Implement semantic goal specification interface
2. Add trajectory estimation for different tools
3. Apply subtle biases toward goal-aligned tools
4. Validate that nudges improve interaction quality

## Performance Considerations

### Computational Cost

- **244D projection**: ~50ms (cached after first computation)
- **Top-K selection**: O(K log N) = negligible
- **Category aggregation**: O(1) with precomputed mapping
- **Policy augmentation**: ~5% overhead (384 -> 416 dim input)

**Total overhead**: ~10% with proper optimization

### Optimization Strategies

1. **Dimension caching**: Cache semantic axes after learning
2. **Sparse computation**: Only compute top-K active dimensions
3. **Batch processing**: Compute trajectories in batches
4. **Lazy evaluation**: Only compute semantic features when needed

## Evaluation Metrics

### Semantic Alignment Score

Measures how well the policy tracks desired semantic dimensions:

```
alignment = 1 - mean(|actual_dim - target_dim| for dim in goals)
```

### Semantic Smoothness

Measures trajectory stability (no wild semantic oscillations):

```
smoothness = 1 / (1 + mean(|acceleration| for dim in trajectory))
```

### Goal Achievement Rate

Percentage of interactions that reach target semantic regions:

```
achievement = count(distance_to_goal < threshold) / total_interactions
```

## Research Questions

1. **How much can semantic nudges improve interaction quality?**
   - Hypothesis: 10-20% improvement in user satisfaction scores

2. **Do semantic goals generalize across domains?**
   - Test if "Clarity + Warmth" goals work for both technical and emotional conversations

3. **Can we learn optimal semantic goals from user feedback?**
   - Use PPO to learn which semantic targets maximize reward

4. **Do semantic features improve exploration efficiency?**
   - Does semantic-aware policy converge faster than pure neural policy?

## Next Steps

1. **Prototype Implementation**: Build `SemanticPolicyEngine` with basic 244D integration
2. **Demo Creation**: Show semantic nudges in action on example queries
3. **Benchmark**: Compare semantic-aware vs vanilla policy on diverse tasks
4. **User Study**: Evaluate if semantic guidance improves perceived quality

## Conclusion

Semantic Micropolicy Nudges represent a bridge between:
- **Symbolic AI**: Interpretable semantic dimensions
- **Neural AI**: Deep learning policy networks
- **Hybrid Intelligence**: Best of both worlds

By making the policy *aware* of semantic space, we enable:
- More interpretable decisions
- Goal-directed behavior
- Ethical alignment
- Adaptive interaction styles

The 244D semantic calculus provides the vocabulary for the policy to understand *what it means* to be helpful, clear, warm, wise, or compassionate - not just as abstract concepts, but as measurable dimensions in a geometric space that can be optimized toward.