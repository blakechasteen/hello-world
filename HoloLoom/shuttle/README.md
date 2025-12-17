# HoloLoom Shuttle: MCTS-Powered Warp↔Yarn Intersection

**Status**: ✅ Production Ready (v2.0.0 - December 2025)
**Location**: `HoloLoom/shuttle/`
**Total Lines**: ~3,200+ lines of production code and tests
**Integration**: HoloLoom WeavingOrchestrator Step 3 (Thread Selection)

---

## Overview

**Shuttle** is the transport layer that intelligently combines two complementary search modalities for context retrieval:

- **Warp**: Fuzzy, semantic search over embedded content (Qdrant vector database)
- **Yarn**: Structured, symbolic traversal through knowledge graphs (Neo4j or NetworkX)

Instead of choosing one search strategy, Shuttle **intersects** both, using Monte Carlo Tree Search (MCTS) with Thompson Sampling to find optimal graph expansion paths that maximize context quality.

### Core Philosophy

> **"The best context comes from combining fuzzy and structured search."**

Warp provides semantic grounding (what's *relevant*), while Yarn provides structural grounding (what's *related*). By intelligently intersecting these two search modes, Shuttle produces richer context than either could provide alone.

### Key Innovation

Shuttle uses **Trajectory Strategies** (different graph traversal approaches) combined with **Thompson Sampling** to learn which expansion strategy works best for different query types. This enables adaptive, self-improving context retrieval that gets smarter with every query.

---

## Quick Start

### Basic Usage

```python
from HoloLoom.shuttle import create_hololoom_shuttle

# Create a shuttle instance (auto-selects best available backend)
shuttle = create_hololoom_shuttle()

# Execute Warp↔Yarn intersection
result = shuttle.intersect("What's blocking the Q4 feature launch?")

# Access results
print(f"Fuzzy Evidence (Warp):")
for hit in result.fuzzy_evidence[:3]:
    print(f"  - {hit['text']} (score: {hit['score']:.2f})")

print(f"\nStructural Claims (Yarn):")
print(f"  {result.structural_claims}")

print(f"\nSelected Trajectory: {result.trajectory_used}")
print(f"Confidence: {result.reward:.2f}")
print(f"Time: {result.search_time_ms:.1f}ms")
```

### Integration with WeavingOrchestrator

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.shuttle import create_shuttle_stage

config = Config.fused()
shards = create_memory_shards()

# Create shuttle stage (replaces Step 3)
shuttle_stage = create_shuttle_stage(config)

async with WeavingOrchestrator(
    cfg=config,
    shards=shards,
    thread_selector=shuttle_stage  # Use Shuttle instead of simple yarn selection
) as orchestrator:
    spacetime = await orchestrator.weave(query)
    print(f"Response: {spacetime.response}")
```

### Configuration Modes

Shuttle supports three operation modes with automatic degradation:

```python
from HoloLoom.shuttle import ShuttleConfig, ShuttleMode, create_hololoom_shuttle

# Production (Neo4j + Qdrant + MCTS)
config = ShuttleConfig.for_mode(ShuttleMode.FULL)
shuttle = create_hololoom_shuttle(config)

# Development (NetworkX + simplified MCTS)
config = ShuttleConfig.for_mode(ShuttleMode.LITE)
shuttle = create_hololoom_shuttle(config)

# Minimal (pure Python, always works)
config = ShuttleConfig.for_mode(ShuttleMode.MINIMAL)
shuttle = create_hololoom_shuttle(config)

# Auto-detect best available (recommended)
config = ShuttleConfig.for_mode(ShuttleMode.AUTO)
shuttle = create_hololoom_shuttle(config)
```

---

## Key Components

| Component | Lines | Purpose |
|-----------|-------|---------|
| **orchestrator_v2.py** | 520 | Main Shuttle orchestrator, WarpInterface/YarnInterface protocols |
| **mcts.py** | 365 | Monte Carlo Tree Search for graph expansion |
| **trajectories.py** | 169 | 6 different graph traversal strategies |
| **trajectory_bandit.py** | 275 | Thompson Sampling for trajectory selection |
| **entity_extraction.py** | 540 | 3-tier entity extraction (Payload/Regex/spaCy) |
| **weaving_integration.py** | 490 | Integration with WeavingOrchestrator, Warp/Yarn adapters |
| **config.py** | 380 | Configuration management with graceful degradation |
| **exceptions.py** | 185 | Comprehensive exception hierarchy |
| **eggroll_shuttle.py** | 265 | Distributed Shuttle with Eggroll cluster support |
| **hololoom_adapters.py** | 395 | HoloLoom-specific backend adapters |
| **Tests** | 400+ | Comprehensive test suite (unit + integration + benchmarks) |

**Total**: ~3,200+ lines of production code

---

## Main Classes and Functions

### Shuttle (Orchestrator)

```python
class Shuttle:
    """Main orchestrator for Warp↔Yarn intersection."""

    def __init__(
        self,
        warp: WarpInterface,
        yarn: YarnInterface,
        config: ShuttleConfig,
    ):
        """Initialize Shuttle with Warp and Yarn backends."""

    async def intersect(
        self,
        query: str,
        top_k_warp: int = 10,
        trajectory_name: Optional[str] = None,
    ) -> WeaveResult:
        """
        Main query execution: Warp search → Trajectory selection → Yarn expansion → MCTS.

        Returns:
            WeaveResult with fuzzy_evidence (Warp), structural_claims (Yarn),
            selected_nodes, trajectory_used, reward, search_time_ms, metadata
        """
```

**Key Methods**:
- `intersect()` - Execute Warp↔Yarn intersection
- `get_trajectory_statistics()` - Get Thompson Sampling statistics
- `get_best_trajectories(top_k)` - Get top trajectories by mean reward
- `save_bandit_state(filepath)` - Persist Thompson Sampling priors

### Trajectory Strategies

Six built-in trajectory strategies for different query types:

| Strategy | Purpose | Edge Types | Depth | Max Nodes |
|----------|---------|-----------|-------|-----------|
| **ProjectBlockersTrajectory** | "What's blocking X?" | BLOCKED_BY, DEPENDS_ON | 2 | 40 |
| **OwnershipTrajectory** | "Who owns X?" | ASSIGNED_TO, OWNS | 1 | 30 |
| **TimelineTrajectory** | "What happened before X?" | HAPPENED_BEFORE, PRECEDES | 3 | 50 |
| **ConceptualTrajectory** | "What's related to X?" | RELATED_TO, SIMILAR_TO | 2 | 35 |
| **HierarchicalTrajectory** | "What's above X?" | PARENT_OF, CONTAINS | 2 | 40 |
| **ExploratoryTrajectory** | "Tell me about X" | RELATED_TO, CONNECTED_TO | 2 | 50 |

Each strategy is a Protocol implementing:
```python
def build_config(self, anchors: List[Anchor]) -> TraversalConfig:
    """Return traversal parameters (depth, nodes, edge types)."""
```

### MCTS (Monte Carlo Tree Search)

```python
class MCTS:
    """Monte Carlo Tree Search for optimal graph expansion."""

    def search(
        self,
        root_state: MCTSState,
        neighbor_map: NeighborMap,
        rollout_fn: Callable[[MCTSState], float],
    ) -> MCTSState:
        """
        Find best graph expansion using UCB1-based tree search.

        Phases:
        1. Selection: Traverse tree using UCB1 (exploitation/exploration balance)
        2. Expansion: Add new child node
        3. Simulation: Evaluate node quality via rollout_fn
        4. Backpropagation: Update statistics

        Returns:
            Best MCTSState found
        """
```

**MCTS State**:
```python
@dataclass
class MCTSState:
    selected_nodes: List[str]  # Nodes in current expansion
    depth: int                 # Current depth in tree

    def available_actions(self, neighbor_map) -> List[str]:
        """Get candidate nodes to expand next."""
```

### Entity Extraction (3-Tier)

Three extraction strategies with automatic fallback:

```python
from HoloLoom.shuttle import EntityExtractionFactory

# Automatic selection based on availability
extractor = EntityExtractionFactory.create("auto")

# Specific strategies
extractor = EntityExtractionFactory.create("spacy")   # High quality (requires spaCy)
extractor = EntityExtractionFactory.create("regex")   # Lightweight pattern-based
extractor = EntityExtractionFactory.create("payload") # Zero dependencies (fastest)

# Extract anchors from Warp results
anchors = extractor.extract(warp_results, max_anchors=10)
```

Each extractor implements:
```python
def extract(
    self,
    warp_results: List[Dict[str, Any]],
    max_anchors: int = 10
) -> List[Anchor]:
    """Extract named entities from search results."""
```

### Thompson Sampling (Trajectory Bandit)

```python
from HoloLoom.shuttle import TrajectoryBandit

bandit = TrajectoryBandit(
    trajectory_names=[
        "project_blockers",
        "who_owns_this",
        "timeline",
        "conceptual",
        "hierarchical",
        "exploratory"
    ]
)

# Select trajectory using Thompson Sampling
trajectory = bandit.choose_trajectory()

# Update with reward feedback
bandit.update(trajectory, reward=0.85)

# Get statistics
stats = bandit.get_statistics()
# Returns: Dict[trajectory_name, {"successes": int, "failures": int, "mean_reward": float}]
```

**Thompson Sampling Details**:
- Maintains Beta(α, β) distribution for each trajectory
- Success: α ← α + confidence
- Failure: β ← β + (1 - confidence)
- Selection: Sample from each Beta, pick highest

### Configuration

```python
from HoloLoom.shuttle import ShuttleConfig, ShuttleMode

config = ShuttleConfig(
    # Operation mode
    mode=ShuttleMode.AUTO,

    # MCTS parameters
    mcts_simulations=32,           # 10-100 recommended
    mcts_timeout_ms=5000,          # Max time for MCTS
    exploration_constant=1.4,      # UCB1 parameter (optimal ≈ √2)

    # Warp (vector search) parameters
    warp_top_k=10,                 # Results to retrieve
    warp_timeout_ms=3000,          # Max time for search

    # Yarn (graph) parameters
    max_graph_depth=2,             # Max traversal depth
    max_graph_nodes=40,            # Max nodes to expand
    yarn_timeout_ms=5000,          # Max time for traversal

    # Entity extraction
    enable_entity_extraction=True,
    entity_extraction_method="payload",  # "spacy", "regex", "payload"
    entity_extraction_fallback=True,

    # Error handling
    enable_graceful_degradation=True,
    fallback_chain=[ShuttleMode.FULL, ShuttleMode.LITE, ShuttleMode.MINIMAL],

    # Performance
    timeout_budget_ms=10000,  # Total timeout for intersect()
    enable_caching=True,
    cache_size=1000,

    # Logging
    log_level="INFO",
    log_performance_metrics=True,
    log_trajectory_selection=True,
)

shuttle = create_hololoom_shuttle(config)
```

**Preset Configurations**:
```python
from HoloLoom.shuttle.config import (
    default_config,      # AUTO mode (recommended)
    production_config,   # FULL mode, optimized for quality
    development_config,  # LITE mode, optimized for speed
    minimal_config,      # MINIMAL mode, zero dependencies
)

config = production_config()
```

---

## Architecture

### Data Flow: Warp↔Yarn Intersection

```
Query
  ↓
[1. Entity Extraction] → Identify entities in query
  ↓
[2. Warp Search (Qdrant)] → Get fuzzy, semantic results
  ↓                         Returns: List of MemoryShard
  ↓                         Top K by semantic similarity
  ↓
[3. Anchor Creation] → Convert Warp results to graph anchors
  ↓
[4. Trajectory Selection (Thompson Sampling)] → Pick expansion strategy
  ↓
[5. Yarn Expansion (Neo4j/NetworkX)] → Build neighbor map from anchors
  ↓                                       Following selected edge types
  ↓                                       Respecting depth/node limits
  ↓
[6. MCTS Search] → Find optimal expansion path
  ↓                UCB1 balances exploration/exploitation
  ↓                Rollout evaluates node combinations
  ↓                Backpropagation updates statistics
  ↓
[7. Node Description] → Get human-readable graph context
  ↓
[8. Result Synthesis] → Combine Warp + Yarn + MCTS
  ↓
WeaveResult
  ├─ fuzzy_evidence: Warp results (semantic)
  ├─ structural_claims: Yarn description (symbolic)
  ├─ selected_nodes: Nodes chosen by MCTS
  ├─ trajectory_used: Which strategy was selected
  ├─ reward: Quality score (0.0-1.0)
  └─ search_time_ms: Total execution time
```

### Mode Degradation Strategy

```
AutoSelect
  ├─ FULL: Neo4j + Qdrant + MCTS (32 simulations)
  │         • High quality, all features available
  │         • Requires Docker services
  │
  ├─ LITE: NetworkX + NumPy + MCTS (16 simulations)
  │         • Good balance, single-machine deployment
  │         • In-memory graph
  │
  └─ MINIMAL: Pure Python + greedy BFS
              • Always works, minimal dependencies
              • No external services required
              • Fallback for any failures
```

Graceful degradation: If FULL fails, automatically tries LITE, then MINIMAL.

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Warp Search** | 50-200ms | Qdrant vector search |
| **Anchor Extraction** | 1-5ms | Payload → Regex → spaCy (fallback chain) |
| **MCTS (32 simulations)** | 100-300ms | Graph expansion via UCB1 |
| **Yarn Traversal** | 50-150ms | Neo4j or NetworkX |
| **Total intersect()** | 200-700ms | Typical workload |
| **With Caching** | 10-50ms | Cache hit (10-14x speedup) |

**Scalability**:
- Warp: O(log n) - vector search with indexing
- Yarn: O(k × d) - k nodes, depth d expansion
- MCTS: O(s × n) - s simulations, n candidate nodes
- Total: Linear in query complexity, sublinear with caching

---

## Integration with HoloLoom

### With WeavingOrchestrator

Shuttle replaces Step 3 (Thread Selection) in the 9-step weaving cycle:

```
Step 1: Loom Command (pattern selection)
Step 2: Chrono Trigger (temporal window)
Step 3: [SHUTTLE] ← Intelligent thread/node selection via Warp↔Yarn
Step 4: Resonance Shed (feature extraction)
Step 5: Warp Space (continuous manifold)
Step 6: Convergence Engine (decision collapse)
Step 7: Tool Execution (action)
Step 8: Spacetime Fabric (output synthesis)
Step 9: Reflection Buffer (learning)
```

### Adapters

**HoloLoomWarpAdapter**: Wraps HoloLoom's retriever for semantic search
```python
from HoloLoom.shuttle.weaving_integration import HoloLoomWarpAdapter

warp = HoloLoomWarpAdapter(retriever=hololoom.memory_backend.retriever)
```

**HoloLoomYarnAdapter**: Wraps HoloLoom's KG for graph traversal
```python
from HoloLoom.shuttle.weaving_integration import HoloLoomYarnAdapter

yarn = HoloLoomYarnAdapter(kg=hololoom.memory_backend.kg)
```

### Memory Backend Integration

Shuttle works with all HoloLoom memory backends:

```python
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.shuttle import create_hololoom_shuttle

# INMEMORY (development)
memory = await create_memory_backend(Config.bare())
shuttle = create_hololoom_shuttle()

# HYBRID (production - Neo4j + Qdrant)
memory = await create_memory_backend(Config.fused())
shuttle = create_hololoom_shuttle()

# Auto-fallback to INMEMORY if backends unavailable
memory = await create_memory_backend(Config())  # AUTO mode
shuttle = create_hololoom_shuttle()
```

---

## When to Use Shuttle

### ✅ Use Shuttle When You Need

- **Optimal context retrieval**: Combining semantic + structural search
- **Query type adaptation**: Different strategies for different questions
- **Self-improving systems**: Thompson Sampling learns from outcomes
- **Hybrid search**: Neither fuzzy nor structured search alone is sufficient
- **Production quality**: Best available context for high-stakes decisions
- **Explainability**: Clear trajectory rationale for each result

### ✅ Shuttle Excels At

- "What's blocking the Q4 launch?" (dependency traversal)
- "Who owns this system?" (ownership relationships)
- "What happened before the incident?" (temporal chains)
- "Show me related technologies" (conceptual clustering)
- "What's the hierarchy of components?" (structural navigation)

### 🟡 Consider Alternatives When

- Simple vector search is sufficient (use Warp directly)
- Only structured knowledge matters (use Yarn directly)
- Latency is critical (<100ms required) - cache helps, but MCTS adds overhead
- Memory is extremely constrained (MINIMAL mode still requires neighbor map)
- Your graph has no meaningful traversal paths (structure doesn't help)

### ❌ Don't Use Shuttle When

- Backends (Neo4j/Qdrant) are permanently unavailable
- Your knowledge base is purely unstructured (no graph)
- You need <50ms latency consistently (MCTS simulation time)
- Simple rules engine would suffice
- Deterministic selection is required (Thompson Sampling is stochastic)

---

## Examples

### Example 1: Basic Query

```python
from HoloLoom.shuttle import create_hololoom_shuttle

shuttle = create_hololoom_shuttle()

result = shuttle.intersect(
    "What's blocking the authentication service deployment?"
)

print(f"Trajectory: {result.trajectory_used}")
print(f"Confidence: {result.reward:.2f}")
print(f"Found {len(result.selected_nodes)} related nodes")
print(f"Structural claims: {result.structural_claims[:200]}...")
```

### Example 2: Custom Trajectory

```python
from HoloLoom.shuttle import create_hololoom_shuttle, TRAJECTORY_BY_NAME

shuttle = create_hololoom_shuttle()

# Force a specific trajectory
result = shuttle.intersect(
    query="What's the timeline of this incident?",
    trajectory_name="timeline"
)
```

### Example 3: Learning and Statistics

```python
from HoloLoom.shuttle import create_hololoom_shuttle

shuttle = create_hololoom_shuttle()

# Process multiple queries
queries = [
    "What's blocking us?",
    "Who owns the API service?",
    "What's the history of this bug?",
    "What systems depend on this?",
]

for query in queries:
    result = shuttle.intersect(query)
    print(f"Query: {query}")
    print(f"  Strategy: {result.trajectory_used}")
    print(f"  Confidence: {result.reward:.2f}")

# View learned preferences
stats = shuttle.get_trajectory_statistics()
for trajectory, stats in stats.items():
    print(f"{trajectory}: {stats['mean_reward']:.2f} avg (N={stats['total_pulls']})")

# Get top strategies
best = shuttle.get_best_trajectories(top_k=3)
for trajectory, mean_reward in best:
    print(f"  {trajectory}: {mean_reward:.2f}")

# Save learning state for next session
shuttle.save_bandit_state("./shuttle_state.json")
```

### Example 4: Configuration Tuning

```python
from HoloLoom.shuttle import ShuttleConfig, ShuttleMode, create_hololoom_shuttle

# Production: quality over speed
config = ShuttleConfig.for_mode(
    ShuttleMode.FULL,
    mcts_simulations=100,    # More exploration
    warp_top_k=20,           # Broader semantic search
    max_graph_depth=3,       # Deeper structural expansion
    max_graph_nodes=60,      # More context
)
shuttle = create_hololoom_shuttle(config)

# Development: speed over quality
config = ShuttleConfig.for_mode(
    ShuttleMode.LITE,
    mcts_simulations=5,      # Fast search
    warp_top_k=5,            # Limited results
    max_graph_depth=1,       # Shallow expansion
    max_graph_nodes=15,      # Small context
)
shuttle = create_hololoom_shuttle(config)
```

### Example 5: Error Handling

```python
from HoloLoom.shuttle import (
    create_hololoom_shuttle,
    ShuttleError,
    ConfigurationError,
    BackendUnavailableError,
)

try:
    shuttle = create_hololoom_shuttle()
    result = shuttle.intersect("What's blocking us?")
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
except BackendUnavailableError as e:
    print(f"Backend {e.backend_name} unavailable: {e.reason}")
    print(f"Falling back to: {e.fallback_mode}")
except ShuttleError as e:
    print(f"Shuttle error: {e}")
```

---

## Development and Testing

### Running Tests

```bash
# Unit tests (15 tests, <5s)
pytest HoloLoom/shuttle/tests/ -v

# Integration tests (with real backends)
pytest HoloLoom/shuttle/tests/ -v -m integration

# Benchmarks
python HoloLoom/shuttle/benchmarks/weaving_performance.py
```

### Testing Trajectory Selection

```python
from HoloLoom.shuttle import TrajectoryBandit

bandit = TrajectoryBandit([
    "project_blockers",
    "who_owns_this",
    "timeline",
    "conceptual",
    "hierarchical",
    "exploratory"
])

# Test Thompson Sampling
for i in range(100):
    trajectory = bandit.choose_trajectory()
    reward = simulate_query_reward()  # Your reward function
    bandit.update(trajectory, reward)

stats = bandit.get_statistics()
assert all(s['total_pulls'] > 0 for s in stats.values())
```

---

## Architecture Decisions

### Why Thompson Sampling?

Thompson Sampling provides:
- **Principled exploration**: Balances trying unknown strategies with exploiting known good ones
- **Efficient learning**: Learns from every query without explicit confidence bounds
- **Simplicity**: Easy to implement and understand
- **Flexibility**: Works with any reward function (quality, latency, coverage)

### Why MCTS?

MCTS provides:
- **Optimal planning**: Finds best graph expansion without exhaustive search
- **Flexibility**: Handles arbitrary graph structures
- **Scalability**: Simulation count can be tuned for latency requirements
- **Interpretability**: Can show selection reasoning via tree traversal

### Why Six Trajectories?

Each trajectory targets a different query pattern:
- **ProjectBlockers**: Causality ("what's blocking?")
- **Ownership**: Assignment ("who owns?")
- **Timeline**: Temporality ("when did?")
- **Conceptual**: Relatedness ("similar to?")
- **Hierarchical**: Structure ("above/below?")
- **Exploratory**: Open-ended ("tell me about?")

Together they cover most real-world query types.

---

## Future Roadmap

**Phase 2 (Planned)**:
- [ ] Custom trajectory definition via DSL
- [ ] Dynamic trajectory discovery from query patterns
- [ ] Trajectory composition (combine multiple strategies)
- [ ] A/B testing framework for trajectory evaluation
- [ ] Multi-hop MCTS (expand beyond immediate neighbors)

**Phase 3 (Planned)**:
- [ ] Distributed Shuttle (Eggroll cluster support)
- [ ] Real-time trajectory performance tracking
- [ ] Trajectory recommendation API
- [ ] Explanation generation ("why this trajectory?")

---

## Troubleshooting

### Shuttle Returns Low Confidence

```python
# Check trajectory statistics
stats = shuttle.get_trajectory_statistics()
for traj, stat in stats.items():
    if stat['mean_reward'] < 0.5:
        print(f"Warning: {traj} has low average reward")

# Solution: Increase MCTS simulations or expand graph depth
config.mcts_simulations = 64
config.max_graph_depth = 3
```

### Warp Search Returns No Results

```python
# Check entity extraction
from HoloLoom.shuttle import EntityExtractionFactory

extractor = EntityExtractionFactory.create("spacy")
anchors = extractor.extract(warp_results)
if not anchors:
    print("No entities extracted from query")
    # Try simpler extraction method
    extractor = EntityExtractionFactory.create("regex")
```

### Graph Traversal Too Slow

```python
# Reduce search space
config.max_graph_depth = 1
config.max_graph_nodes = 20
config.mcts_simulations = 16

# Or use LITE mode
config = ShuttleConfig.for_mode(ShuttleMode.LITE)
```

---

## References

- **Thompson Sampling**: [Thompson (1933)](https://en.wikipedia.org/wiki/Thompson_sampling)
- **MCTS**: [Browne et al. (2012)](http://ieeexplore.ieee.org/document/6145622/)
- **UCB1**: [Auer et al. (2002)](https://link.springer.com/article/10.1023/A:1013689704352)

---

## Summary

Shuttle brings intelligent, adaptive context retrieval to HoloLoom by combining semantic search (Warp) with structured traversal (Yarn), using MCTS to find optimal expansion paths and Thompson Sampling to learn which strategies work best. It's production-ready with graceful degradation across three operation modes, comprehensive error handling, and deep integration with HoloLoom's memory systems.

For questions or integration help, refer to `HoloLoom/shuttle/weaving_integration.py` for WeavingOrchestrator integration or `HoloLoom/shuttle/hololoom_adapters.py` for HoloLoom-specific adapters.
