# Spring-Based Memory Activation for HoloLoom

**Status**: Proposed Architecture
**Date**: 2025-10-29
**Author**: Design exploration based on physics-driven memory recall

## Executive Summary

Memory recall can be modeled as a **spring network energy minimization problem**. When a query activates specific memory nodes, activation energy spreads through elastic connections (springs) until the system reaches equilibrium. This reveals semantically related memories through physical dynamics rather than heuristic similarity metrics.

## The Core Insight

**Traditional Memory Retrieval**:
```python
# Cosine similarity - static, one-shot
similarities = [cosine_sim(query, memory) for memory in memories]
top_k = sorted(similarities)[:k]
```

**Spring-Based Activation** (proposed):
```python
# Dynamic energy propagation through graph
activate_node(query_node, activation=1.0)
while not_converged:
    propagate_activation_via_springs()
    apply_damping()
    decay_activation()
active_memories = [m for m in memories if m.activation > threshold]
```

## Physics Model

### Hooke's Law for Memory

For each connection between memories `i` and `j`:

```
F_ij = -k * (activation_i - activation_j) - c * velocity_i
```

Where:
- **k** (stiffness): Connection strength (edge weight in KG)
- **Δx** (displacement): Activation difference between connected memories
- **c** (damping): Prevents oscillation, models forgetting
- **v** (velocity): Rate of activation change

### Energy Landscape

The memory network has a potential energy surface:

```
E = Σ (1/2 * k * (a_i - a_j)^2)  [spring potential energy]
  + Σ (1/2 * m * v_i^2)          [kinetic energy]
  + Σ decay * a_i                [dissipation]
```

Query activation creates a **high-energy state**. The system relaxes toward equilibrium, and the settled state reveals which memories are activated.

## Mapping to HoloLoom Architecture

### Current Architecture

```
Query → Embedding → BM25/Cosine Similarity → Top-K Retrieval
                    ↓
              Knowledge Graph (static structure)
```

### Proposed Spring-Augmented Architecture

```
Query → Initial Activation
          ↓
      Spring Network Propagation
          ↓
      [Yarn Graph edges = springs]
          ↓
      Equilibrium State = Retrieved Memories
```

### Integration Points

#### 1. **Yarn Graph → Spring Network**

**Current** (`HoloLoom/memory/graph.py`):
```python
class YarnGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        # Edges have weights but no dynamics
```

**Proposed**:
```python
class YarnGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_activation = {}  # Current activation level
        self.node_velocity = {}    # Activation velocity (for dynamics)

    def propagate_activation(self, steps=100, dt=0.016):
        """Spring-based spreading activation"""
        for _ in range(steps):
            for node in self.graph.nodes():
                force = 0

                # Sum spring forces from neighbors
                for neighbor in self.graph.neighbors(node):
                    edge = self.graph[node][neighbor]
                    stiffness = edge.get('weight', 1.0)  # Use edge weight as k
                    activation_diff = (
                        self.node_activation[neighbor] -
                        self.node_activation[node]
                    )
                    force += stiffness * activation_diff

                # Damping
                damping = 0.85
                force -= damping * self.node_velocity[node]

                # Update dynamics
                self.node_velocity[node] += force * dt
                self.node_activation[node] += self.node_velocity[node] * dt

                # Decay (forgetting)
                self.node_activation[node] *= 0.98

                # Clamp [0, 1]
                self.node_activation[node] = max(0, min(1,
                    self.node_activation[node]))

        # Return activated nodes above threshold
        return [n for n, a in self.node_activation.items() if a > 0.1]
```

#### 2. **WeavingOrchestrator Integration**

**Current** (`HoloLoom/weaving_orchestrator.py:L234-250`):
```python
async def _retrieve_context(self, query: Query) -> List[MemoryShard]:
    # Static retrieval
    if self.memory:
        results = await self.memory.retrieve(query.text, k=5)
        return results
    else:
        return self.shards[:5]
```

**Proposed**:
```python
async def _retrieve_context(self, query: Query) -> List[MemoryShard]:
    if self.cfg.use_spring_activation:  # New config flag
        # 1. Find query-relevant nodes via embedding
        query_nodes = await self.memory.find_nodes(query.text, k=3)

        # 2. Activate those nodes
        for node in query_nodes:
            self.memory.graph.node_activation[node.id] = 1.0

        # 3. Let activation propagate through springs
        activated_nodes = self.memory.graph.propagate_activation(
            steps=self.cfg.spring_iterations,
            dt=0.016
        )

        # 4. Retrieve shards for activated nodes
        return await self.memory.get_shards(activated_nodes)
    else:
        # Fallback to traditional retrieval
        return await self.memory.retrieve(query.text, k=5)
```

#### 3. **Configuration Extension**

**Add to** (`HoloLoom/config.py`):
```python
@dataclass
class Config:
    # ... existing fields ...

    # Spring activation parameters
    use_spring_activation: bool = False
    spring_stiffness: float = 0.15       # k (connection strength multiplier)
    spring_damping: float = 0.85         # c (velocity damping)
    spring_decay: float = 0.98           # activation decay per step
    spring_iterations: int = 100         # propagation steps
    spring_activation_threshold: float = 0.1  # min activation to retrieve
```

## Advantages Over Static Retrieval

### 1. **Transitive Relationships**

Static cosine similarity: `Query → Memory` (direct only)

Spring activation: `Query → Memory1 → Memory2 → Memory3` (multi-hop)

**Example**:
```
Query: "How does Thompson Sampling work?"

Direct match: [Thompson Sampling node]

Spring propagation:
  Thompson Sampling (1.0)
    ↓ (strong spring)
  Bandit Algorithms (0.7)
    ↓ (medium spring)
  Exploration vs Exploitation (0.4)
    ↓ (weak spring)
  Regret Bounds (0.2)
```

All four memories retrieved, even though "Regret Bounds" has low direct similarity to query.

### 2. **Context-Sensitive Retrieval**

Spring stiffness can vary based on:
- **Edge type** (IS_A vs MENTIONS vs USES)
- **Recency** (recent memories have stiffer springs)
- **Usage frequency** (often-accessed paths strengthen)

```python
# Adaptive spring stiffness
edge['stiffness'] = (
    base_stiffness
    * edge_type_multiplier[edge['type']]
    * recency_factor(edge['timestamp'])
    * usage_factor(edge['access_count'])
)
```

### 3. **Energy-Based Confidence**

Activation level = confidence in relevance:
```python
results = [
    (memory, activation)
    for memory, activation in activated_memories
    if activation > threshold
]
# Naturally sorted by activation (physics handles ranking)
```

### 4. **Handles Multi-Modal Queries**

Activate multiple seed nodes simultaneously:
```python
# Query: "Show me performant queries using caching"
seed_nodes = [
    find_node("performance"),  # concept
    find_node("cache"),        # concept
    find_node("query_log_123") # instance
]

for node in seed_nodes:
    node.activation = 1.0

propagate()  # Springs find intersection of these concepts
```

## Computational Complexity

**Traditional Retrieval**: O(N) for N memories (cosine similarity)

**Spring Activation**: O(E * I) where:
- E = number of edges in KG (typically O(N) for sparse graphs)
- I = iterations (typically 50-200)

**Optimization**: Stop early if energy change < epsilon (often converges in <50 iterations)

```python
def propagate_activation(self, max_steps=200, epsilon=1e-4):
    prev_energy = float('inf')

    for step in range(max_steps):
        self._update_step()

        # Check convergence
        energy = self._compute_energy()
        if abs(energy - prev_energy) < epsilon:
            break  # Converged!

        prev_energy = energy

    return step  # Actual iterations needed
```

## Implementation Roadmap

### Phase 1: Proof of Concept (2-3 days)
- [ ] Add spring dynamics to `YarnGraph`
- [ ] Implement `propagate_activation()` method
- [ ] Create demo notebook showing spring vs static retrieval

### Phase 2: Orchestrator Integration (3-4 days)
- [ ] Add config flags for spring activation
- [ ] Integrate into `_retrieve_context()`
- [ ] Write tests comparing retrieval quality

### Phase 3: Optimization (5-7 days)
- [ ] Early stopping (convergence detection)
- [ ] Sparse matrix operations (for large graphs)
- [ ] GPU acceleration (PyTorch for parallel spring updates)
- [ ] Adaptive stiffness based on edge metadata

### Phase 4: Hybrid Mode (3-5 days)
- [ ] Combine spring activation + embedding similarity
- [ ] Use embeddings for seed node selection
- [ ] Use springs for context expansion
- [ ] A/B testing framework

## Related Work

### Academic Foundations

1. **Spreading Activation Theory** (Collins & Loftus, 1975)
   - Human memory modeled as activation spreading through semantic networks
   - Basis for ACT-R cognitive architecture

2. **Hopfield Networks** (Hopfield, 1982)
   - Energy-based neural networks
   - Memories as attractors in energy landscape

3. **Graph Neural Networks with Message Passing** (Gilmer et al., 2017)
   - Iterative message passing ≈ spring activation
   - Node features updated via neighbor aggregation

4. **PageRank** (Page et al., 1998)
   - Random walk on graph ≈ diffusion
   - Spring activation is directed diffusion with physics

### Differences from Existing Approaches

**vs BM25**: Spring activation considers graph structure, not just term frequency

**vs Embedding Similarity**: Transitive relationships via multi-hop propagation

**vs Graph Neural Networks**: No training needed, uses explicit KG structure

**vs PageRank**: Physics-based (springs) instead of probabilistic (random walk)

## Demo

Open `demos/spring_memory_recall.html` to see interactive visualization:

1. **Click "Thompson Sampling"** memory on right panel
2. **Watch activation spread** through spring connections
3. **Observe**:
   - Direct neighbors activate strongly (stiff springs)
   - Distant memories activate weakly (loose springs)
   - System reaches equilibrium (energy minimized)
4. **Adjust parameters**:
   - Higher stiffness → faster propagation
   - Lower damping → more oscillation
   - Higher decay → faster forgetting

## Conclusion

Spring-based memory activation offers:
- **Richer retrieval** via transitive relationships
- **Physical interpretability** (energy, forces, equilibrium)
- **Adaptive dynamics** (stiffness based on edge metadata)
- **Natural ranking** (activation level = confidence)

Integration with HoloLoom's existing Yarn Graph is straightforward - edges become springs, retrieval becomes energy minimization.

**Next Step**: Implement Phase 1 POC and compare retrieval quality on benchmark queries.

---

**References**:
- [1] Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing.
- [2] Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities.
- [3] Gilmer, J., et al. (2017). Neural Message Passing for Quantum Chemistry.
- [4] Page, L., et al. (1998). The PageRank Citation Ranking: Bringing Order to the Web.
