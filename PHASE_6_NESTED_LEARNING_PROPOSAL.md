# Phase 6: Nested Learning Meta-Architecture

**Status**: 🎯 Proposal (December 2025)
**Inspired By**: [Google Research: Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
**Philosophy**: "Architecture and optimization are not separate—they are nested learning problems at different frequencies"

---

## Executive Summary

Phase 6 transforms HoloLoom from a **fixed 9-step pipeline** into a **self-organizing nested learning system** where:

1. **Architecture becomes learnable**: The weaving cycle learns which steps to use/skip/repeat
2. **Multi-frequency optimization**: 5 nested levels (ultra-fast → very slow) with distinct update frequencies
3. **Hope-style recurrence**: Memory flows bidirectionally across cycles, enabling long-context reasoning
4. **Continuum Memory System**: Unified memory continuum from working memory → persistent graph
5. **Self-improving retrieval**: Memory learns its own organization and retrieval strategies

**Key Insight**: The 9-step weaving cycle was always an **abstraction for modularity**. Making it learnable is the natural evolution, not a departure.

---

## Table of Contents

1. [Motivation & Background](#motivation--background)
2. [Core Concepts](#core-concepts)
3. [Multi-Frequency Learning Hierarchy](#multi-frequency-learning-hierarchy)
4. [Learnable Weaving Cycle](#learnable-weaving-cycle)
5. [Hope-Style Recurrent Architecture](#hope-style-recurrent-architecture)
6. [Continuum Memory System (CMS)](#continuum-memory-system-cms)
7. [Self-Optimizing Memory](#self-optimizing-memory)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Migration Path](#migration-path)
10. [Performance Expectations](#performance-expectations)
11. [Research Questions](#research-questions)

---

## Motivation & Background

### What Is Nested Learning?

Nested Learning (Google Research, 2024) treats ML systems as **hierarchies of learning problems** operating at different frequencies:

- **Inner loops**: Fast updates (per-query, per-token)
- **Mid loops**: Medium updates (per-episode, per-batch)
- **Outer loops**: Slow updates (per-epoch, per-session)

Each level has its own **context flow** and **associative memory**, creating a continuum from working memory to long-term knowledge.

### Why HoloLoom Needs This

HoloLoom Phase 5 already implements:
- ✅ Multi-frequency learning (Thompson Sampling, policy weights, background learning)
- ✅ Continuum memory (DotPlasma → Hot Patterns → Reflection Buffer → Knowledge Graph)
- ✅ Deep optimizers (cross-attention, L2 regression, not simple dot products)

**What's Missing**:
- ❌ Architecture is **fixed** (always 9 steps in sequence)
- ❌ No **recurrence** across queries (each query starts from scratch)
- ❌ Memory organization is **hand-coded** (not learned from outcomes)
- ❌ No **ultra-fast** updates for sub-query agentic reasoning

**Phase 6 Goal**: Make HoloLoom a true Nested Learning system where architecture, memory, and optimization co-evolve.

---

## Core Concepts

### 1. Learnable Architecture

Current HoloLoom:
```python
# Fixed 9-step pipeline
async def weave(query):
    step1_result = await loom_command(query)
    step2_result = await chrono_trigger(step1_result)
    step3_result = await yarn_graph(step2_result)
    # ... always all 9 steps
    return spacetime
```

Phase 6:
```python
# Learnable step selection
async def weave(query):
    steps = self.meta_policy.select_steps(query)  # e.g., [1,2,4,6,8]
    context = QueryContext(query)

    for step_id in steps:
        context = await self.execute_step(step_id, context)

        # Learn whether to continue/stop/repeat
        if self.meta_policy.should_stop(context):
            break

    return context.to_spacetime()
```

**Key Idea**: The weaving cycle becomes a **meta-learning problem** where the system learns:
- Which steps are necessary for which query types
- When to skip expensive operations (e.g., Warp Space for simple queries)
- When to repeat steps (e.g., iterative refinement)
- When to stop early (confidence threshold met)

### 2. Recurrent Context Flow

Current HoloLoom: Each query is independent
```
Query A → [9 steps] → Spacetime A
Query B → [9 steps] → Spacetime B  (starts from scratch)
```

Phase 6: Context flows across queries (Hope-style)
```
Query A → [steps] → Hidden State H_A ─┐
                                        ├→ Shared Context
Query B → [steps] → Hidden State H_B ─┘
```

**Key Idea**: Maintain a **recurrent hidden state** that:
- Accumulates context from previous queries
- Enables "remembering" without explicit graph storage
- Supports multi-turn conversations naturally
- Creates emergent long-term dependencies

### 3. Multi-Frequency Optimization

Nested Learning organizes updates by frequency:

```
Ultra-Fast (1-10ms)  ← Agentic sub-queries, token-level decisions
Fast (100-200ms)     ← Thompson Sampling, tool selection
Medium (1-5s)        ← Policy weights, adapter selection
Slow (60s)           ← Background learning, pattern extraction
Very Slow (∞)        ← Knowledge Graph structure, architectural changes
```

Each level has:
- **Own learning rate**: Faster levels update more aggressively
- **Own context window**: Faster levels see less history
- **Own associative memory**: Faster levels use simpler lookups

### 4. Continuum Memory System (CMS)

Memory becomes a **spectrum** rather than discrete tiers:

```
Working Memory ←─────── CMS Continuum ───────→ Long-Term Memory
  (volatile)                                      (persistent)

   DotPlasma  →  Hot Cache  →  Reflection  →  Knowledge Graph
   <1ms decay    5%/hr decay    1000 items      ∞ capacity
```

**Key Innovation**: Memory modules update at different frequencies but share representations:
- Ultra-fast: Working memory (DotPlasma) for current query
- Fast: Hot cache (recently accessed patterns)
- Medium: Reflection buffer (episodic memory)
- Slow: Knowledge graph (semantic memory)

All communicate via **shared embedding space** (Matryoshka 244D).

---

## Multi-Frequency Learning Hierarchy

Phase 6 introduces **5 nested optimization levels**:

### Level 1: Ultra-Fast (1-10ms per update)

**Purpose**: Agentic sub-query reasoning, token-level decisions

**What Updates**:
- Sub-query routing (which agent/department to invoke)
- Token-level confidence (for streaming responses)
- Working memory activation (which features are relevant RIGHT NOW)

**Implementation**:
```python
class UltraFastOptimizer:
    """Updates every sub-query (~5-10ms)"""

    def __init__(self):
        self.working_memory_activations = torch.zeros(244)  # Current query context
        self.subquery_router = LightweightMLP(244, n_agents)

    async def route_subquery(self, subquery: str, context: Features) -> str:
        """Ultra-fast routing decision"""
        activations = self.working_memory_activations * 0.9  # Fast decay
        activations += self.encode_subquery(subquery) * 0.1

        agent_logits = self.subquery_router(activations)
        return self.select_agent(agent_logits)  # <1ms
```

**Example Use Cases**:
- Agentic reasoning: "Should I verify this claim?" → instant decision
- Multi-query research: "Which sub-question next?" → real-time routing
- Streaming: "Is this token confident enough?" → per-token gating

**Update Rule**:
```python
# Ultra-fast exponential moving average
working_memory ← 0.9 * working_memory + 0.1 * new_features
```

### Level 2: Fast (100-200ms per update)

**Purpose**: Tool selection, Thompson Sampling, query-level decisions

**What Updates** (existing Phase 5 systems):
- Thompson Sampling priors (α, β for each tool)
- Tool selection confidence
- Query complexity classification

**Implementation** (already exists):
```python
# Thompson Sampling updates (every query)
if confidence >= 0.75:
    alpha[tool] += confidence
else:
    beta[tool] += (1 - confidence)
```

**No changes needed** - Phase 5 already implements this level.

### Level 3: Medium (1-5s per update)

**Purpose**: Policy adapter weights, retrieval strategy learning

**What Updates** (existing + new):
- Policy adapter weights (BARE/FAST/FUSED selection)
- **NEW**: Retrieval strategy weights (BM25 vs semantic vs spectral)
- **NEW**: Step selection policy (which weaving steps to use)

**Implementation** (new for Phase 6):
```python
class MediumFrequencyOptimizer:
    """Updates every successful query outcome (~1-5s)"""

    def __init__(self):
        self.step_selection_policy = NeuralStepSelector()
        self.retrieval_strategy_weights = {
            'bm25': 0.33,
            'semantic': 0.33,
            'spectral': 0.34
        }

    async def update(self, query: Query, outcome: Spacetime):
        """Learn from query outcome"""
        # Update step selection
        steps_used = outcome.trace.steps_executed
        reward = outcome.confidence
        self.step_selection_policy.update(query, steps_used, reward)

        # Update retrieval weights
        retrieval_method = outcome.metadata.get('retrieval_method')
        self.retrieval_strategy_weights[retrieval_method] += 0.01 * reward
        self.normalize_weights()
```

**Example Learning**:
```
Query: "What is Thompson Sampling?"
Steps Used: [1, 2, 3, 4, 6, 8]  (skipped Warp Space, Convergence)
Confidence: 0.92
→ Learn: Simple queries don't need full 9 steps

Query: "Design a multi-agent system with safety guarantees"
Steps Used: [1, 2, 3, 4, 5, 6, 7, 8, 9]  (all steps)
Confidence: 0.87
→ Learn: Complex queries need full pipeline
```

### Level 4: Slow (60s per update)

**Purpose**: Background learning, pattern extraction, hot pattern decay

**What Updates** (existing Phase 5):
- Background learning thread (comprehensive statistics update)
- Hot pattern heat decay (5% per hour)
- Pattern learner (motif → tool → confidence patterns)

**No changes needed** - Phase 5 already implements this level.

### Level 5: Very Slow (∞ / manual per update)

**Purpose**: Knowledge Graph structure, architectural evolution

**What Updates**:
- Knowledge Graph entity relationships (manual or rare)
- **NEW**: Meta-architecture evolution (add/remove/modify steps)
- **NEW**: Feature dimension evolution (grow/prune Matryoshka dimensions)

**Implementation** (new for Phase 6):
```python
class VerySlowOptimizer:
    """Updates very rarely - architectural changes"""

    async def evolve_architecture(self, performance_history: List[Spacetime]):
        """Analyze long-term performance and propose architectural changes"""
        # E.g., after 10,000 queries, analyze:
        # - Which steps are rarely useful? → Consider removing
        # - Which queries struggle? → Consider adding new steps
        # - Which features dominate? → Consider growing those dimensions

        if self.should_add_step():
            new_step = self.propose_new_step()
            return ArchitecturalChange(action='ADD_STEP', step=new_step)
```

**Example Evolution**:
```
After 10,000 queries:
- Warp Space only helps on 5% of queries → Mark as "optional"
- Retrieval struggles on code queries → Add new "code-aware" step
- Dimensions 96-192 rarely used → Consider pruning
```

---

## Learnable Weaving Cycle

### Current Architecture

```python
# Fixed pipeline (always 9 steps)
class WeavingOrchestrator:
    async def weave(self, query: Query) -> Spacetime:
        # Step 1: Loom Command
        pattern_card = self.loom.select_pattern(query)

        # Step 2: Chrono Trigger
        temporal_window = self.chrono.create_window(pattern_card)

        # Step 3: Yarn Graph
        threads = self.yarn_graph.select_threads(temporal_window)

        # Step 4: Resonance Shed
        features = self.resonance.extract_features(threads)

        # Step 5: Warp Space
        tensioned = self.warp.tension(features)

        # Step 6: Convergence Engine
        decision = self.convergence.collapse(tensioned)

        # Step 7: Tool Execution
        result = await self.executor.execute(decision)

        # Step 8: Spacetime Fabric
        spacetime = self.fabric.weave(result)

        # Step 9: Reflection Buffer
        await self.reflection.store(spacetime)

        return spacetime
```

### Phase 6 Architecture

```python
# Learnable step selection
class NeuralWeavingOrchestrator:
    def __init__(self, cfg: Config):
        # Step executors (same as before)
        self.step_executors = {
            1: LoomCommand(),
            2: ChronoTrigger(),
            3: YarnGraph(),
            4: ResonanceShed(),
            5: WarpSpace(),
            6: ConvergenceEngine(),
            7: ToolExecutor(),
            8: SpacetimeFabric(),
            9: ReflectionBuffer()
        }

        # NEW: Meta-policy for step selection
        self.meta_policy = MetaWeavingPolicy(
            input_dim=244,  # Matryoshka features
            n_steps=9,
            hidden_dim=128
        )

        # NEW: Recurrent hidden state (Hope-style)
        self.hidden_state = None

        # Medium-frequency optimizer
        self.step_optimizer = MediumFrequencyOptimizer()

    async def weave(self, query: Query) -> Spacetime:
        """Learnable weaving with step selection"""
        # Encode query
        query_features = await self.encode_query(query)

        # Initialize context
        context = WeavingContext(
            query=query,
            features=query_features,
            hidden_state=self.hidden_state,  # Carry over from previous query
            steps_executed=[],
            confidence=0.0
        )

        # Meta-policy selects which steps to execute
        step_sequence = self.meta_policy.select_steps(
            query_features,
            self.hidden_state
        )

        # Execute selected steps
        for step_id in step_sequence:
            context = await self.execute_step(step_id, context)

            # Early stopping if confidence threshold met
            if context.confidence >= 0.9 and step_id >= 6:
                break

        # Update hidden state (Hope-style recurrence)
        self.hidden_state = context.hidden_state

        # Learn from outcome (medium-frequency)
        await self.step_optimizer.update(query, context.to_spacetime())

        return context.to_spacetime()

    async def execute_step(self, step_id: int, context: WeavingContext) -> WeavingContext:
        """Execute a single weaving step"""
        executor = self.step_executors[step_id]

        # Execute step
        result = await executor.execute(context)

        # Update context
        context.features = result.features
        context.hidden_state = result.hidden_state
        context.confidence = result.confidence
        context.steps_executed.append(step_id)

        return context
```

### Meta-Policy Architecture

```python
class MetaWeavingPolicy(nn.Module):
    """Neural policy for step selection"""

    def __init__(self, input_dim: int, n_steps: int, hidden_dim: int):
        super().__init__()
        self.n_steps = n_steps

        # Encode query + hidden state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Step selection head (which steps to execute)
        self.step_selector = nn.Linear(hidden_dim, n_steps)

        # Stopping policy (when to stop early)
        self.stop_policy = nn.Linear(hidden_dim, 1)

    def select_steps(
        self,
        query_features: torch.Tensor,
        hidden_state: Optional[torch.Tensor]
    ) -> List[int]:
        """Select which steps to execute"""
        # Encode
        if hidden_state is None:
            hidden_state = torch.zeros(query_features.shape[0], self.hidden_dim)

        x = torch.cat([query_features, hidden_state], dim=-1)
        encoded = self.encoder(x)

        # Step selection logits
        step_logits = self.step_selector(encoded)
        step_probs = torch.sigmoid(step_logits)

        # Select steps with prob > threshold (e.g., 0.5)
        selected_steps = []
        for i, prob in enumerate(step_probs.squeeze()):
            if prob > 0.5:
                selected_steps.append(i + 1)  # 1-indexed

        # Always include steps 1 (Loom Command) and 8 (Spacetime Fabric)
        if 1 not in selected_steps:
            selected_steps.insert(0, 1)
        if 8 not in selected_steps:
            selected_steps.append(8)

        return sorted(selected_steps)

    def should_stop(self, context: WeavingContext) -> bool:
        """Decide whether to stop early"""
        x = torch.cat([context.features, context.hidden_state], dim=-1)
        encoded = self.encoder(x)

        stop_logit = self.stop_policy(encoded)
        stop_prob = torch.sigmoid(stop_logit)

        # Stop if high confidence and prob > threshold
        return context.confidence >= 0.9 and stop_prob > 0.7
```

### Learning Algorithm

The meta-policy learns from query outcomes using **policy gradient** (REINFORCE):

```python
class MetaPolicyTrainer:
    def __init__(self, meta_policy: MetaWeavingPolicy):
        self.meta_policy = meta_policy
        self.optimizer = torch.optim.Adam(meta_policy.parameters(), lr=1e-4)

    def update(self, query: Query, steps_used: List[int], reward: float):
        """Update meta-policy from outcome"""
        # Reward = confidence - (latency_penalty * num_steps)
        # Encourages: high confidence with fewer steps

        query_features = encode_query(query)
        hidden_state = self.get_hidden_state()

        # Forward pass
        step_logits = self.meta_policy.step_selector(
            self.meta_policy.encoder(
                torch.cat([query_features, hidden_state], dim=-1)
            )
        )

        # Compute loss (policy gradient)
        loss = 0.0
        for i in range(self.meta_policy.n_steps):
            step_was_used = (i + 1) in steps_used
            step_prob = torch.sigmoid(step_logits[i])

            if step_was_used:
                # Encourage this step
                loss -= torch.log(step_prob) * reward
            else:
                # Discourage this step
                loss -= torch.log(1 - step_prob) * reward

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### Example Step Selection Patterns (Learned)

After training, the meta-policy learns query-specific shortcuts:

| Query Type | Steps Selected | Reasoning |
|------------|---------------|-----------|
| **Simple factual** | [1, 2, 3, 8] | Only needs: Pattern selection, temporal window, memory retrieval, output |
| **Code explanation** | [1, 2, 3, 4, 6, 8] | Needs features but not Warp Space |
| **Complex reasoning** | [1, 2, 3, 4, 5, 6, 7, 8, 9] | Full pipeline required |
| **Cached query** | [1, 8] | Skip everything, load from cache |
| **Multi-turn conversation** | [1, 2, 4, 6, 8] | Recurrent state carries context, skip graph |

**Performance Gains**:
- Simple queries: ~5 steps instead of 9 → **40% faster**
- Cached queries: ~2 steps → **80% faster**
- Complex queries: Still uses all 9 steps (no regression)

---

## Hope-Style Recurrent Architecture

### Problem: HoloLoom Forgets Across Queries

Current behavior:
```
User: "What is Thompson Sampling?"
HoloLoom: [9 steps, builds context] → Excellent answer

User: "Can you show me an example?" (refers to previous answer)
HoloLoom: [9 steps, starts from scratch] → Loses context
```

The knowledge graph helps, but requires explicit entity linking. **Human conversation doesn't work this way.**

### Solution: Recurrent Hidden State

Maintain a **hidden state** that flows across queries:

```python
class RecurrentWeavingOrchestrator:
    def __init__(self, cfg: Config):
        # ... standard components ...

        # NEW: Recurrent hidden state
        self.hidden_state = torch.zeros(1, 512)  # Larger than query features
        self.state_compressor = nn.GRU(
            input_size=244,   # Matryoshka features
            hidden_size=512,  # Hidden state
            num_layers=2
        )

    async def weave(self, query: Query) -> Spacetime:
        """Weave with recurrent context"""
        query_features = await self.encode_query(query)

        # Update hidden state (Hope-style)
        output, self.hidden_state = self.state_compressor(
            query_features.unsqueeze(0),
            self.hidden_state
        )

        # Hidden state now encodes:
        # - Current query
        # - Previous query context
        # - Long-term conversation history

        # Use hidden state in step selection
        steps = self.meta_policy.select_steps(query_features, self.hidden_state)

        # Execute weaving cycle
        context = await self.execute_steps(steps, query_features, self.hidden_state)

        return context.to_spacetime()
```

### What Gets Remembered

The hidden state compresses:

1. **Recent queries** (last 5-10 queries)
2. **User preferences** (e.g., "verbose" vs "concise" responses)
3. **Topic continuity** (e.g., still discussing "Thompson Sampling")
4. **Conversation flow** (e.g., "follow-up question" vs "new topic")

### Recurrent Update Rule

```python
# GRU update equations
z_t = σ(W_z · [h_{t-1}, x_t])          # Update gate
r_t = σ(W_r · [h_{t-1}, x_t])          # Reset gate
h_tilde = tanh(W_h · [r_t * h_{t-1}, x_t])  # Candidate state
h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde   # New hidden state
```

**Key properties**:
- **Decays gracefully**: Old context fades naturally via update gate
- **Selective memory**: Reset gate controls what to forget
- **No fixed window**: Unlike attention, can remember arbitrarily far back

### Integration with Knowledge Graph

Hidden state **complements** (not replaces) the knowledge graph:

| Memory Type | Storage | Access | Use Case |
|-------------|---------|--------|----------|
| **Hidden State** | 512D vector | Implicit (always active) | Conversation flow, recent context |
| **Knowledge Graph** | Entity relationships | Explicit retrieval | Long-term facts, structured knowledge |

Example:
```
User: "What is Thompson Sampling?"
→ Knowledge Graph: Retrieve {Thompson Sampling, Bayesian, Multi-Armed Bandit}
→ Hidden State: Encode "user is learning about exploration/exploitation"

User: "Show me an example"
→ Hidden State: "Still discussing Thompson Sampling, wants example"
→ Knowledge Graph: Retrieve code examples from memory
```

### Multi-Turn Conversation Demo

```python
# Turn 1
User: "What is Thompson Sampling?"
Hidden State: [0.0, 0.0, ..., 0.0]  # Cold start
→ Full weaving cycle, build context
Hidden State: [0.8, 0.3, ..., 0.5]  # Encodes topic

# Turn 2
User: "Can you show me Python code?"
Hidden State: [0.8, 0.3, ..., 0.5]  # Carries over
→ Meta-policy recognizes continuation, skips graph retrieval
→ Uses hidden state to infer "Python code for Thompson Sampling"
Hidden State: [0.7, 0.4, ..., 0.6]  # Updated

# Turn 3
User: "How does it compare to epsilon-greedy?"
Hidden State: [0.7, 0.4, ..., 0.6]  # Still active
→ Recognizes comparison query, retrieves both algorithms
Hidden State: [0.6, 0.5, ..., 0.5]  # Topic shift detected
```

---

## Continuum Memory System (CMS)

Phase 6 unifies HoloLoom's memory into a **continuous spectrum** rather than discrete tiers.

### Current Memory Architecture (Discrete Tiers)

```
DotPlasma (working memory)
    ↓ [gap]
Hot Patterns (access-based cache)
    ↓ [gap]
Reflection Buffer (episodic memory)
    ↓ [gap]
Knowledge Graph (semantic memory)
```

Each tier is **isolated** with manual handoffs.

### Phase 6: Continuum Memory System

```
Ultra-Fast ←────── Memory Continuum ────────→ Very Slow
  (volatile)                                   (persistent)

Working Memory → Hot Cache → Warm Cache → Reflection → Knowledge Graph
  <1ms decay     5%/hr        1%/hr        1000 items    ∞ capacity

  [────── Shared 244D Embedding Space ──────]
```

**Key Innovation**: All memory modules share the **same representation** (Matryoshka 244D):
- Working memory: Currently active features
- Hot cache: Recently accessed patterns
- Warm cache: Moderately accessed patterns
- Reflection buffer: Episodic outcomes
- Knowledge graph: Persistent entities

### Memory Access by Frequency

| Level | Memory Type | Access Pattern | Decay Rate | Use Case |
|-------|------------|----------------|------------|----------|
| **Ultra-Fast** | Working Memory | Current query only | <1ms (replaced every query) | Active reasoning |
| **Fast** | Hot Cache | Last 10-100 queries | 5% per hour | Frequently accessed knowledge |
| **Medium** | Warm Cache | Last 100-1000 queries | 1% per hour | Moderately accessed knowledge |
| **Slow** | Reflection Buffer | Last 1000 queries | No decay (FIFO) | Recent episodic memory |
| **Very Slow** | Knowledge Graph | All time | No decay | Persistent semantic memory |

### Implementation

```python
class ContinuumMemorySystem:
    """Unified memory continuum"""

    def __init__(self, embedding_dim: int = 244):
        # All memory modules share embedding space
        self.embedding_dim = embedding_dim

        # Ultra-fast: Working memory (current query)
        self.working_memory = torch.zeros(1, embedding_dim)

        # Fast: Hot cache (last ~100 queries)
        self.hot_cache = HotPatternCache(
            capacity=100,
            decay_rate=0.05  # 5% per hour
        )

        # Medium: Warm cache (last ~1000 queries)
        self.warm_cache = WarmPatternCache(
            capacity=1000,
            decay_rate=0.01  # 1% per hour
        )

        # Slow: Reflection buffer (last 1000 outcomes)
        self.reflection_buffer = ReflectionBuffer(capacity=1000)

        # Very slow: Knowledge graph (persistent)
        self.knowledge_graph = KG()

    async def retrieve(
        self,
        query_embedding: torch.Tensor,
        max_results: int = 10
    ) -> List[MemoryShard]:
        """Retrieve from memory continuum"""
        results = []

        # Level 1: Check working memory (current query context)
        if self.is_relevant(query_embedding, self.working_memory):
            results.append(self.working_memory_to_shard())

        # Level 2: Check hot cache (frequent patterns)
        hot_results = self.hot_cache.retrieve(query_embedding, k=5)
        results.extend(hot_results)

        # Level 3: Check warm cache (moderate patterns)
        if len(results) < max_results:
            warm_results = self.warm_cache.retrieve(query_embedding, k=5)
            results.extend(warm_results)

        # Level 4: Check reflection buffer (recent episodic)
        if len(results) < max_results:
            episodic_results = self.reflection_buffer.retrieve(query_embedding, k=3)
            results.extend(episodic_results)

        # Level 5: Fall back to knowledge graph (persistent)
        if len(results) < max_results:
            graph_results = await self.knowledge_graph.retrieve(
                query_embedding,
                k=max_results - len(results)
            )
            results.extend(graph_results)

        return results[:max_results]

    async def store(
        self,
        shard: MemoryShard,
        access_frequency: str = 'slow'  # ultra_fast, fast, medium, slow, very_slow
    ):
        """Store to appropriate memory level"""
        embedding = shard.embedding

        if access_frequency == 'ultra_fast':
            # Replace working memory
            self.working_memory = embedding

        elif access_frequency == 'fast':
            # Add to hot cache
            self.hot_cache.add(shard, heat=1.0)

        elif access_frequency == 'medium':
            # Add to warm cache
            self.warm_cache.add(shard, heat=0.5)

        elif access_frequency == 'slow':
            # Add to reflection buffer
            await self.reflection_buffer.store(shard)

        elif access_frequency == 'very_slow':
            # Add to knowledge graph (persistent)
            await self.knowledge_graph.add_from_shard(shard)

    async def update_activations(self, query_embedding: torch.Tensor):
        """Update all memory activations (multi-frequency)"""
        # Ultra-fast: Update working memory
        self.working_memory = query_embedding

        # Fast: Boost hot patterns
        self.hot_cache.boost_matching(query_embedding)

        # Medium: Decay warm patterns
        self.warm_cache.decay_step()

        # Slow: Update episodic statistics
        await self.reflection_buffer.update_statistics()

        # Very slow: No update (manual only)
```

### Heat Decay Visualization

```
Heat Score Over Time (5% decay per hour):

1.0 ┤●                                    Initial access
0.9 ┤ ●                                   1 hour
0.8 ┤  ●                                  2 hours
0.7 ┤   ●                                 3 hours
0.6 ┤    ●                                5 hours
0.5 ┤     ●                               7 hours (threshold)
0.4 ┤      ●                              10 hours
0.3 ┤       ●                             13 hours
0.2 ┤        ●                            17 hours
0.1 ┤          ●                          23 hours
0.0 ┤            ●                        Moved to cold storage
    └────────────────────────────────────
```

Patterns with heat < 0.5 move from hot → warm → cold → archived.

### Retrieval Latency by Level

| Memory Level | Latency | Cache Hit Rate | Use Case |
|--------------|---------|----------------|----------|
| Working Memory | <0.1ms | ~10% | Current query context |
| Hot Cache | ~1ms | ~30% | Recently accessed |
| Warm Cache | ~5ms | ~40% | Moderately accessed |
| Reflection Buffer | ~10ms | ~15% | Recent outcomes |
| Knowledge Graph | ~50ms | ~5% | Rare/deep knowledge |

**Total average latency**: ~10ms (weighted by hit rates)

---

## Self-Optimizing Memory

Phase 6 enables memory to **learn its own organization** from outcomes.

### Problem: Fixed Retrieval Strategies

Current HoloLoom uses **hand-coded** retrieval:
```python
# Fixed weights
results = 0.33 * bm25_results + 0.33 * semantic_results + 0.34 * spectral_results
```

But different queries need different strategies:
- **Code queries**: BM25 (keyword matching) works best
- **Concept queries**: Semantic similarity works best
- **Relationship queries**: Spectral features work best

### Solution: Learned Retrieval Policy

```python
class SelfOptimizingMemory:
    """Memory that learns its own retrieval strategies"""

    def __init__(self, kg: KG, embedding_dim: int = 244):
        self.kg = kg

        # Available retrieval strategies
        self.strategies = {
            'bm25': BM25Retriever(),
            'semantic': SemanticRetriever(),
            'spectral': SpectralRetriever(),
            'graph_walk': GraphWalkRetriever(),
            'temporal': TemporalRetriever()
        }

        # Learnable strategy selector
        self.strategy_policy = StrategySelectionPolicy(
            input_dim=embedding_dim,
            n_strategies=len(self.strategies),
            hidden_dim=128
        )

        # Track strategy performance
        self.strategy_stats = {
            name: {'successes': 0, 'failures': 0}
            for name in self.strategies
        }

    async def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 10
    ) -> List[MemoryShard]:
        """Retrieve using learned strategy"""
        # Strategy policy selects retrieval method(s)
        strategy_weights = self.strategy_policy.select(query_embedding)

        # Execute strategies in parallel
        results = {}
        for name, weight in strategy_weights.items():
            if weight > 0.1:  # Only execute if weight significant
                results[name] = await self.strategies[name].retrieve(
                    query_embedding,
                    k=k
                )

        # Blend results using learned weights
        blended = self.blend_results(results, strategy_weights)
        return blended[:k]

    async def learn_from_outcome(
        self,
        query_embedding: torch.Tensor,
        strategies_used: Dict[str, float],
        confidence: float
    ):
        """Update strategy policy from outcome"""
        # Reward = confidence (how well did retrieval work?)
        for strategy_name, weight in strategies_used.items():
            if confidence >= 0.75:
                self.strategy_stats[strategy_name]['successes'] += 1
            else:
                self.strategy_stats[strategy_name]['failures'] += 1

        # Update neural policy
        self.strategy_policy.update(
            query_embedding,
            strategies_used,
            reward=confidence
        )
```

### Strategy Selection Policy

```python
class StrategySelectionPolicy(nn.Module):
    """Neural policy for retrieval strategy selection"""

    def __init__(self, input_dim: int, n_strategies: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.strategy_head = nn.Linear(hidden_dim, n_strategies)

    def select(self, query_embedding: torch.Tensor) -> Dict[str, float]:
        """Select strategy weights"""
        encoded = self.encoder(query_embedding)
        logits = self.strategy_head(encoded)
        weights = torch.softmax(logits, dim=-1)

        # Convert to dict
        strategy_names = ['bm25', 'semantic', 'spectral', 'graph_walk', 'temporal']
        return {
            name: weight.item()
            for name, weight in zip(strategy_names, weights)
        }
```

### Example Learning Trajectory

```
Epoch 0 (cold start):
  All strategies weighted equally: {bm25: 0.2, semantic: 0.2, spectral: 0.2, ...}

After 100 queries:
  Query: "What is Thompson Sampling?"
  → Semantic works best (0.92 confidence)
  → Learn: Boost semantic for concept queries
  Weights: {semantic: 0.4, bm25: 0.2, spectral: 0.15, ...}

After 1000 queries:
  Query: "def thompson_sampling():" (code)
  → BM25 works best (0.88 confidence)
  → Learn: Boost BM25 for code queries
  Weights: {bm25: 0.45, semantic: 0.3, spectral: 0.1, ...}

After 10,000 queries:
  System has learned query-specific strategies:
  - Concept queries → semantic (0.5-0.7 weight)
  - Code queries → bm25 (0.6-0.8 weight)
  - Relationship queries → spectral + graph_walk (0.4 + 0.4 weight)
```

---

## Implementation Roadmap

Phase 6 is a **6-month project** (Jan 2026 - June 2026) broken into 4 sub-phases.

### Sub-Phase 6.1: Multi-Frequency Infrastructure (8 weeks)

**Goal**: Establish 5-level optimization hierarchy

**Tasks**:
1. ✅ **Week 1-2**: Ultra-fast optimizer
   - Working memory activation tracking
   - Sub-query routing policy (lightweight MLP)
   - Integration with agentic reasoning system

2. ✅ **Week 3-4**: Medium-frequency optimizer
   - Step selection policy (neural selector)
   - Retrieval strategy policy (strategy weights)
   - Outcome-based learning

3. ✅ **Week 5-6**: Very slow optimizer
   - Architectural evolution system
   - Long-term performance analysis
   - Automatic step addition/removal proposals

4. ✅ **Week 7-8**: Integration & testing
   - Wire all 5 levels together
   - Performance benchmarking
   - Ablation studies (disable each level, measure impact)

**Deliverables**:
- `HoloLoom/nested/ultra_fast.py` (200 lines)
- `HoloLoom/nested/medium_frequency.py` (300 lines)
- `HoloLoom/nested/very_slow.py` (400 lines)
- `HoloLoom/nested/hierarchy.py` (500 lines)
- Tests: `tests/unit/test_nested_hierarchy.py` (15 tests)
- Performance report: `PHASE_6_1_PERFORMANCE.md`

### Sub-Phase 6.2: Learnable Weaving Cycle (8 weeks)

**Goal**: Make step selection learnable

**Tasks**:
1. ✅ **Week 1-2**: Meta-policy architecture
   - `MetaWeavingPolicy` neural network
   - Step selection logic
   - Early stopping logic

2. ✅ **Week 3-4**: Weaving context refactor
   - `WeavingContext` data structure
   - Step executors protocol
   - Context flow between steps

3. ✅ **Week 5-6**: Learning algorithm
   - Policy gradient training (REINFORCE)
   - Reward shaping (confidence - latency penalty)
   - Exploration/exploitation (epsilon-greedy step selection)

4. ✅ **Week 7-8**: Evaluation & tuning
   - Compare fixed vs learnable pipeline
   - Measure speedups on simple/complex queries
   - Tune reward function

**Deliverables**:
- `HoloLoom/nested/meta_policy.py` (600 lines)
- `HoloLoom/nested/weaving_context.py` (200 lines)
- `HoloLoom/nested/neural_weaving_orchestrator.py` (800 lines)
- Tests: `tests/integration/test_learnable_weaving.py` (20 tests)
- Evaluation report: `PHASE_6_2_EVALUATION.md`

### Sub-Phase 6.3: Hope-Style Recurrence (8 weeks)

**Goal**: Add recurrent hidden state for multi-turn conversations

**Tasks**:
1. ✅ **Week 1-2**: Hidden state architecture
   - GRU state compressor
   - Hidden state initialization
   - State reset logic (new conversation vs continuation)

2. ✅ **Week 3-4**: Integration with meta-policy
   - Pass hidden state to step selector
   - Update hidden state during weaving
   - Carry state across queries

3. ✅ **Week 5-6**: Conversation flow detection
   - Topic continuity scoring
   - Follow-up question detection
   - Context switching detection

4. ✅ **Week 7-8**: Multi-turn evaluation
   - Build multi-turn conversation benchmark
   - Compare with/without recurrence
   - Measure context retention across 5-10 turns

**Deliverables**:
- `HoloLoom/nested/recurrent_orchestrator.py` (700 lines)
- `HoloLoom/nested/conversation_flow.py` (300 lines)
- Multi-turn benchmark: `benchmarks/multi_turn_conversations.py`
- Tests: `tests/e2e/test_multi_turn.py` (25 tests)
- Evaluation report: `PHASE_6_3_MULTITURN_EVAL.md`

### Sub-Phase 6.4: Continuum Memory & Self-Optimization (8 weeks)

**Goal**: Unify memory continuum and enable self-optimizing retrieval

**Tasks**:
1. ✅ **Week 1-2**: Continuum Memory System
   - `ContinuumMemorySystem` class
   - Hot/warm cache implementation
   - Multi-level retrieval logic

2. ✅ **Week 3-4**: Heat decay & promotion/demotion
   - Exponential decay (5% per hour for hot, 1% per hour for warm)
   - Automatic promotion (cold → warm → hot)
   - Automatic demotion (hot → warm → cold)

3. ✅ **Week 5-6**: Self-optimizing retrieval
   - `StrategySelectionPolicy` neural network
   - Strategy blending logic
   - Learning from retrieval outcomes

4. ✅ **Week 7-8**: End-to-end integration
   - Replace old memory systems with CMS
   - Backward compatibility layer
   - Full system testing

**Deliverables**:
- `HoloLoom/nested/continuum_memory.py` (800 lines)
- `HoloLoom/nested/self_optimizing_memory.py` (600 lines)
- Migration guide: `PHASE_6_MIGRATION_GUIDE.md`
- Tests: `tests/integration/test_continuum_memory.py` (30 tests)
- Final report: `PHASE_6_COMPLETE.md`

---

## Migration Path

Phase 6 is **backward compatible** with existing HoloLoom code.

### Stage 1: Opt-In Feature Flag (Weeks 1-8)

```python
from HoloLoom.config import Config

# Phase 5 (existing)
config = Config.fused()
orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

# Phase 6 (opt-in)
config = Config.fused()
config.enable_nested_learning = True  # NEW FLAG
orchestrator = NeuralWeavingOrchestrator(cfg=config, shards=shards)
```

**No breaking changes** - Phase 5 orchestrator still works.

### Stage 2: Gradual Rollout (Weeks 9-16)

Enable nested learning features incrementally:

```python
config = Config.fused()
config.enable_nested_learning = True

# Enable specific features
config.nested_learning_config = {
    'multi_frequency': True,       # Sub-phase 6.1
    'learnable_steps': False,      # Sub-phase 6.2 (disabled for now)
    'recurrent_state': False,      # Sub-phase 6.3 (disabled for now)
    'continuum_memory': False      # Sub-phase 6.4 (disabled for now)
}
```

### Stage 3: Full Cutover (Week 32)

After 32 weeks of development + testing:

```python
# Phase 6 becomes default
from HoloLoom import HoloLoom  # Automatically uses NeuralWeavingOrchestrator

# Phase 5 available via legacy flag
config.use_legacy_orchestrator = True  # Explicit opt-in to old behavior
```

### Backward Compatibility Guarantees

| API | Phase 5 | Phase 6 | Notes |
|-----|---------|---------|-------|
| `WeavingOrchestrator.weave(query)` | ✅ | ✅ | Same signature |
| `Config.fused()` | ✅ | ✅ | Same config object |
| `Spacetime` output | ✅ | ✅ | Same data structure |
| `MemoryShard` input | ✅ | ✅ | Same input format |
| `ReflectionBuffer` | ✅ | ✅ | Integrated into CMS |
| `Knowledge Graph` | ✅ | ✅ | Wrapped by CMS |

**Breaking changes**: None (all Phase 6 features are additive)

---

## Performance Expectations

### Latency Improvements

| Query Type | Phase 5 Latency | Phase 6 Latency | Speedup |
|------------|----------------|-----------------|---------|
| **Simple factual** | 150ms (9 steps) | 60ms (4 steps) | **2.5×** |
| **Cached query** | 150ms (9 steps) | 30ms (2 steps) | **5.0×** |
| **Code explanation** | 180ms (9 steps) | 100ms (6 steps) | **1.8×** |
| **Complex reasoning** | 300ms (9 steps) | 300ms (9 steps) | **1.0×** (no regression) |
| **Multi-turn follow-up** | 150ms (cold start) | 80ms (uses hidden state) | **1.9×** |

**Average speedup**: ~2.0× across typical query distribution

### Memory Retrieval Improvements

| Metric | Phase 5 | Phase 6 (CMS) | Improvement |
|--------|---------|---------------|-------------|
| **Hot cache hit rate** | N/A | 30% | +30% (new tier) |
| **Avg retrieval latency** | 50ms | 10ms | **5.0×** (multi-tier) |
| **Memory utilization** | Static | Dynamic | Heat-based optimization |
| **Strategy accuracy** | Fixed weights | Learned weights | +15% confidence |

### Learning Efficiency

| Level | Update Frequency | Overhead | Impact |
|-------|-----------------|----------|--------|
| **Ultra-fast** | Every sub-query (~5ms) | <0.5ms | Sub-query routing |
| **Fast** | Every query (~150ms) | <1ms | Tool selection |
| **Medium** | Every outcome (~1s) | <5ms | Step/strategy learning |
| **Slow** | Every 60s | ~50ms (async) | Background learning |
| **Very slow** | Manual | 0ms | No runtime overhead |

**Total per-query overhead**: ~6.5ms (acceptable for 2.0× speedup)

### Memory Footprint

| Component | Phase 5 | Phase 6 | Delta |
|-----------|---------|---------|-------|
| **Working memory** | N/A | 244 floats (~1KB) | +1KB |
| **Hidden state** | N/A | 512 floats (~2KB) | +2KB |
| **Hot cache** | N/A | 100 entries (~2MB) | +2MB |
| **Warm cache** | N/A | 1000 entries (~20MB) | +20MB |
| **Meta-policy** | N/A | ~500K params (~2MB) | +2MB |

**Total increase**: ~26MB (negligible for modern systems)

---

## Research Questions

Phase 6 opens several research directions:

### 1. Optimal Step Sequences

**Question**: What are the **minimal step sequences** for each query type?

**Hypothesis**: Most queries need <6 steps, not all 9.

**Experiment**:
- Train meta-policy on 10,000 diverse queries
- Analyze learned step patterns
- Identify "canonical sequences" (e.g., [1,2,3,8] for simple queries)

**Expected Findings**:
- Simple queries: 3-5 steps
- Complex queries: 7-9 steps
- Multi-turn: 2-4 steps (uses hidden state)

### 2. Recurrent State Capacity

**Question**: How much context can the hidden state hold?

**Hypothesis**: 512D GRU can remember ~10 query turns effectively.

**Experiment**:
- Build multi-turn benchmark with 1-20 turn conversations
- Measure context retention across turns
- Compare 256D, 512D, 1024D hidden states

**Expected Findings**:
- 256D: ~5 turns
- 512D: ~10 turns
- 1024D: ~15 turns (diminishing returns)

### 3. Memory Continuum Boundaries

**Question**: What are the **optimal heat thresholds** for hot/warm/cold?

**Hypothesis**: 5% decay (hot) and 1% decay (warm) are reasonable, but may need tuning.

**Experiment**:
- Vary decay rates: 1%, 2%, 5%, 10%, 20%
- Measure cache hit rates and latency
- Find optimal balance

**Expected Findings**:
- Too fast decay (20%): Low hit rate, frequent re-computation
- Too slow decay (1%): Stale patterns dominate
- Optimal: 3-7% for hot, 0.5-2% for warm

### 4. Self-Optimizing Retrieval Convergence

**Question**: How long does retrieval policy take to converge?

**Hypothesis**: 1,000-10,000 queries needed for stable strategy weights.

**Experiment**:
- Track strategy weights over 50,000 queries
- Measure convergence (change in weights < 0.01)
- Identify query types that converge fastest

**Expected Findings**:
- Simple queries converge fast (~1,000 queries)
- Complex queries converge slower (~10,000 queries)
- Strategy weights stabilize after ~20,000 queries

### 5. Architectural Evolution

**Question**: Can the system **learn to add/remove steps** automatically?

**Hypothesis**: After 100,000 queries, system proposes meaningful architectural changes.

**Experiment**:
- Run very slow optimizer on large-scale deployment
- Collect architectural change proposals
- Human review of proposals (are they sensible?)

**Expected Findings**:
- Proposes removing rarely-used steps (e.g., Warp Space for 95% of queries)
- Proposes adding domain-specific steps (e.g., code-aware retrieval)
- Proposes feature dimension changes (e.g., grow dimension 128-192 for code queries)

---

## Success Metrics

Phase 6 will be considered successful if:

### Performance Metrics

✅ **2.0× average speedup** on typical query distribution
- Simple queries: 2-3× faster
- Complex queries: No regression (<5% slowdown acceptable)

✅ **30%+ hot cache hit rate** within 1 hour of warm-up

✅ **5× faster retrieval** via multi-tier memory continuum

✅ **<10ms per-query overhead** for all learning updates combined

### Quality Metrics

✅ **No confidence regression**: Phase 6 confidence ≥ Phase 5 confidence

✅ **Improved multi-turn**: 1.5-2× better context retention across 5+ turns

✅ **Learned strategies outperform fixed weights** by 10-15% confidence

### System Metrics

✅ **Backward compatible**: All Phase 5 code works without changes

✅ **Memory footprint < 50MB** for all new components

✅ **Convergence < 10,000 queries** for retrieval strategy learning

✅ **46+ tests passing** (unit + integration + e2e)

---

## Next Steps

### Immediate Actions (Week 1)

1. **Create Phase 6 project structure**
   ```bash
   mkdir -p HoloLoom/nested
   mkdir -p HoloLoom/nested/tests
   mkdir -p benchmarks/phase6
   ```

2. **Prototype ultra-fast optimizer**
   - Implement `UltraFastOptimizer` class
   - Integrate with agentic reasoning system
   - Basic tests (5-10 unit tests)

3. **Design meta-policy architecture**
   - Sketch `MetaWeavingPolicy` neural network
   - Define input/output formats
   - Reward function design

4. **Benchmark current performance**
   - Run Phase 5 on diverse query set
   - Record latencies, confidence, step usage
   - Establish baseline for comparison

### Medium-Term (Weeks 2-8)

- Complete Sub-Phase 6.1 (multi-frequency infrastructure)
- Begin Sub-Phase 6.2 (learnable weaving cycle)
- Build multi-turn conversation benchmark
- Weekly performance reviews

### Long-Term (Months 2-6)

- Complete all 4 sub-phases
- Large-scale deployment testing
- Publish research findings
- Community feedback & iteration

---

## Conclusion

Phase 6 transforms HoloLoom into a **self-organizing nested learning system** that:

1. **Learns its own architecture** (which steps to use)
2. **Remembers across queries** (Hope-style recurrence)
3. **Optimizes at 5 frequencies** (ultra-fast → very slow)
4. **Unifies memory continuum** (working memory → knowledge graph)
5. **Self-optimizes retrieval** (learned strategy selection)

This is the **natural evolution** of HoloLoom's original vision: the 9-step weaving cycle was always an abstraction for modularity. Making it learnable doesn't abandon the vision—it completes it.

**Key Insight**: Architecture and optimization are not separate concerns. They are nested learning problems at different frequencies.

---

**Document Status**: 🎯 Proposal Draft
**Word Count**: ~9,500 words
**Estimated Reading Time**: 35-40 minutes
**Next Review**: After community feedback

**Feedback Welcome**: blake@hololoom.ai

---

## Appendix A: Code Structure

```
HoloLoom/
├── nested/                        # NEW: Phase 6 nested learning
│   ├── __init__.py
│   ├── ultra_fast.py             # Ultra-fast optimizer (200 lines)
│   ├── medium_frequency.py       # Medium-frequency optimizer (300 lines)
│   ├── very_slow.py              # Very slow optimizer (400 lines)
│   ├── hierarchy.py              # 5-level hierarchy (500 lines)
│   ├── meta_policy.py            # Step selection policy (600 lines)
│   ├── weaving_context.py        # Context flow (200 lines)
│   ├── neural_weaving_orchestrator.py  # Learnable orchestrator (800 lines)
│   ├── recurrent_orchestrator.py # Hope-style recurrence (700 lines)
│   ├── conversation_flow.py      # Multi-turn detection (300 lines)
│   ├── continuum_memory.py       # CMS (800 lines)
│   ├── self_optimizing_memory.py # Strategy learning (600 lines)
│   └── tests/
│       ├── test_nested_hierarchy.py
│       ├── test_learnable_weaving.py
│       ├── test_multi_turn.py
│       └── test_continuum_memory.py
│
├── weaving_orchestrator.py       # Phase 5 orchestrator (kept for compatibility)
├── config.py                      # Add enable_nested_learning flag
└── ... (existing modules)

benchmarks/
├── phase6/
│   ├── multi_turn_conversations.py
│   ├── step_selection_analysis.py
│   └── retrieval_strategy_comparison.py
```

**Total new code**: ~6,000 lines
**Total new tests**: ~100 tests
**Existing code modified**: ~500 lines (mostly config + integration)

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Nested Learning** | ML paradigm treating architecture and optimization as hierarchical learning problems at different frequencies |
| **Hope Architecture** | Google Research's proof-of-concept recurrent architecture for Nested Learning |
| **CMS** | Continuum Memory System - unified memory spectrum from volatile to persistent |
| **Meta-Policy** | Neural policy for selecting which weaving steps to execute |
| **Recurrent Hidden State** | Vector encoding conversation context across queries (Hope-style) |
| **Ultra-Fast Optimizer** | Optimization level updating every sub-query (~5-10ms) |
| **Very Slow Optimizer** | Optimization level updating rarely (manual or after 100k+ queries) |
| **Heat Decay** | Exponential decay of memory activation over time (5% per hour) |
| **Strategy Selection Policy** | Neural policy for choosing retrieval strategies |
| **Weaving Context** | Data structure flowing between weaving steps |

---

## Appendix C: References

1. **Google Research: Nested Learning**
   https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

2. **HoloLoom Phase 5: Recursive Learning**
   `RECURSIVE_LEARNING_COMPLETE.md`

3. **HoloLoom Master Scope & Sequence**
   `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md`

4. **Thompson Sampling**
   Agrawal, S., & Goyal, N. (2012). Analysis of Thompson Sampling for the Multi-armed Bandit Problem.

5. **Matryoshka Embeddings**
   Kusupati, A., et al. (2022). Matryoshka Representation Learning.

6. **PPO (Proximal Policy Optimization)**
   Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.

---

**End of Proposal**
