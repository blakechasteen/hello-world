# Awareness Graph

**Status**: Production Ready (November 2025)
**Location**: `HoloLoom/memory/awareness_graph.py`
**Lines**: ~556 lines

Living memory system with semantic topology in 228-dimensional semantic space.

---

## Overview

The Awareness Graph is HoloLoom's consciousness layer - a dynamic memory system that tracks what the system "knows" and how that knowledge relates semantically. Unlike static knowledge graphs, the Awareness Graph:

- **Lives**: Activation levels decay, memories strengthen with use
- **Perceives**: Projects queries into 228D semantic space with 16 interpretable axes
- **Resonates**: Detects semantic shifts and coherence across activated memories
- **Spreads**: Activation propagates through connected memories

**Core Philosophy**: Memory isn't just storage - it's an active, living topology that shapes how we understand and respond.

---

## Quick Start

```python
from HoloLoom.memory.awareness_graph import AwarenessGraph
from HoloLoom.memory.awareness_types import ActivationStrategy

# Create awareness graph
graph = AwarenessGraph(
    graph_backend=kg,           # NetworkX or Neo4j backend
    semantic_calculus=calculus, # 228D semantic projection
    vector_store=store          # Optional vector store
)

# Perceive a query (project into semantic space)
perception = await graph.perceive("What is Thompson Sampling?")
print(f"Position: {perception.position[:3]}...")  # 228D vector
print(f"Dominant: {perception.dominant_dimensions}")  # Top semantic axes
print(f"Shift: {perception.shift_detected}")  # Topic change?

# Remember something (store with semantic metadata)
node_id = await graph.remember(
    content="Thompson Sampling balances exploration and exploitation",
    metadata={"source": "user", "confidence": 0.9}
)

# Activate related memories
activated = await graph.activate(
    query="Explain bandit algorithms",
    strategy=ActivationStrategy.BALANCED,
    k=10
)
for memory in activated:
    print(f"{memory.id}: activation={memory.activation:.2f}")
```

---

## Core Components

### 1. Semantic Perception

When you call `perceive()`, the query is projected into 228-dimensional semantic space:

```python
@dataclass
class SemanticPerception:
    position: np.ndarray          # 228D semantic position
    velocity: np.ndarray          # Rate of semantic change
    dominant_dimensions: List[str] # Top interpretable axes
    shift_detected: bool          # Significant topic change?
    coherence: float              # 0.0-1.0 semantic consistency
    timestamp: float              # When perceived
```

**16 Interpretable Axes** (first 16 of 228):
- Warmth, Valence, Formality, Urgency
- Technicality, Certainty, Abstraction, Specificity
- Temporality, Objectivity, Complexity, Scope
- Directness, Emotionality, Actionability, Novelty

### 2. Activation Strategies

Control how activation spreads through the memory graph:

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| **PRECISE** | Tight activation, minimal spread | Factual queries |
| **BALANCED** | Moderate spread (default) | General queries |
| **EXPLORATORY** | Wide activation spread | Research/brainstorming |
| **DEEP** | Multi-hop traversal | Complex reasoning |

```python
from HoloLoom.memory.awareness_types import ActivationStrategy

# Precise: Only closest semantic matches
activated = await graph.activate(query, strategy=ActivationStrategy.PRECISE)

# Exploratory: Cast a wide net
activated = await graph.activate(query, strategy=ActivationStrategy.EXPLORATORY)
```

### 3. Edge Types

Semantic relationships in the graph topology:

| Edge Type | Purpose | Example |
|-----------|---------|---------|
| **TEMPORAL** | Time-based ordering | "A happened before B" |
| **SEMANTIC_RESONANCE** | Semantic similarity | "A is similar to B" |
| **CAUSAL** | Cause-effect | "A causes B" |
| **REFERENCE** | Citation/mention | "A references B" |

```python
from HoloLoom.memory.awareness_types import EdgeType, EdgeMetadata

# Create semantic edge
edge = EdgeMetadata(
    edge_type=EdgeType.SEMANTIC_RESONANCE,
    weight=0.85,
    timestamp=time.time(),
    metadata={"similarity": 0.85}
)
```

### 4. Activation Budget

Control resource consumption during activation:

```python
from HoloLoom.memory.awareness_types import ActivationBudget

budget = ActivationBudget(
    max_nodes=100,        # Maximum nodes to activate
    max_depth=3,          # Maximum hop distance
    time_limit_ms=50.0,   # Timeout in milliseconds
    min_activation=0.1    # Minimum activation threshold
)

activated = await graph.activate(
    query="Complex topic",
    strategy=ActivationStrategy.DEEP,
    budget=budget
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AwarenessGraph                           │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Graph     │  │  Semantic   │  │    Vector Store     │ │
│  │  Backend    │  │  Calculus   │  │    (Optional)       │ │
│  │ NetworkX/   │  │   228D      │  │   Qdrant/FAISS     │ │
│  │   Neo4j     │  │ Projection  │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          │                                  │
│                ┌─────────▼─────────┐                       │
│                │  Activation Field │                       │
│                │   (Spreading)     │                       │
│                └───────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

**Components**:
- **Graph Backend**: Stores nodes and edges (NetworkX for dev, Neo4j for production)
- **Semantic Calculus**: Projects into 228D space with velocity tracking
- **Vector Store**: Optional similarity search (Qdrant, FAISS)
- **Activation Field**: Manages activation levels and spreading

---

## Key Operations

### perceive(query) → SemanticPerception

Project a query into semantic space:

```python
perception = await graph.perceive("What causes inflation?")

# Access semantic position
position = perception.position  # 228D numpy array

# Check for topic shift from previous query
if perception.shift_detected:
    print("Topic changed significantly")

# See dominant semantic dimensions
for dim in perception.dominant_dimensions[:3]:
    print(f"  {dim}")  # e.g., "Technicality", "Urgency"
```

### remember(content, metadata) → node_id

Store content with semantic metadata:

```python
node_id = await graph.remember(
    content="Thompson Sampling uses Beta distributions for uncertainty",
    metadata={
        "source": "documentation",
        "confidence": 0.95,
        "timestamp": time.time()
    }
)

# Content is automatically:
# 1. Embedded into vector space
# 2. Connected to semantically similar nodes
# 3. Assigned initial activation level
```

### activate(query, strategy, k) → List[Memory]

Activate relevant memories with spreading:

```python
memories = await graph.activate(
    query="How do bandits handle exploration?",
    strategy=ActivationStrategy.BALANCED,
    k=10
)

for mem in memories:
    print(f"ID: {mem.id}")
    print(f"  Content: {mem.content[:50]}...")
    print(f"  Activation: {mem.activation:.2f}")
    print(f"  Hops: {mem.hop_distance}")
```

---

## Metrics & Monitoring

### AwarenessMetrics

Track system health and performance:

```python
from HoloLoom.memory.awareness_types import AwarenessMetrics

metrics = graph.get_metrics()

print(f"Total nodes: {metrics.total_nodes}")
print(f"Active nodes: {metrics.active_nodes}")
print(f"Mean activation: {metrics.mean_activation:.2f}")
print(f"Coherence: {metrics.coherence:.2f}")
print(f"Entropy: {metrics.entropy:.2f}")
```

**Key Metrics**:
- **active_nodes**: Nodes with activation > threshold
- **mean_activation**: Average activation across all nodes
- **coherence**: How well-connected active nodes are (0.0-1.0)
- **entropy**: Diversity of activation distribution

---

## Integration with HoloLoom

The Awareness Graph integrates with the weaving pipeline:

```python
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    # Experience creates memories in awareness graph
    await loom.experience("Thompson Sampling is Bayesian")

    # Recall activates and retrieves
    memories = await loom.recall("What is Thompson Sampling?")

    # Get awareness metrics
    metrics = loom.get_metrics()
    print(f"Active: {metrics['activation']['active_nodes']}")
    print(f"Coherence: {metrics['coherence']['global_coherence']:.2f}")
```

### WeavingOrchestrator Integration

The orchestrator uses awareness for perception injection:

```python
# Step 5.6 in weaving cycle
awareness_context = await awareness_graph.perceive(query)
if awareness_context.shift_detected:
    # Topic changed - may need different retrieval strategy
    pass
```

---

## Configuration

### Graph Backend Selection

```python
from HoloLoom.memory.graph import KG
from HoloLoom.memory.neo4j_graph import Neo4jKG

# Development (in-memory)
kg = KG()

# Production (persistent)
kg = Neo4jKG(uri="bolt://localhost:7687", user="neo4j", password="...")

graph = AwarenessGraph(graph_backend=kg, ...)
```

### Activation Tuning

```python
# Custom activation budget for complex queries
budget = ActivationBudget(
    max_nodes=200,      # More nodes for research
    max_depth=5,        # Deeper traversal
    time_limit_ms=100,  # Longer timeout
    min_activation=0.05 # Lower threshold
)
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| **perceive()** | ~2ms | 228D projection |
| **remember()** | ~5ms | Store + index |
| **activate()** | ~10-50ms | Depends on strategy |
| **get_metrics()** | <1ms | Cached computation |

**Scaling**:
- NetworkX: Up to ~100K nodes in memory
- Neo4j: Millions of nodes with persistence

---

## Best Practices

1. **Use BALANCED strategy by default** - covers most use cases
2. **Set activation budgets** - prevents runaway traversal
3. **Monitor coherence** - low coherence may indicate fragmented knowledge
4. **Track entropy** - high entropy means diverse activation (good for exploration)
5. **Use Neo4j in production** - persistence + scale

---

## Related Documentation

- [SEMANTIC_DIMENSIONS.md](../semantic_calculus/SEMANTIC_DIMENSIONS.md) - 228D semantic space
- [SPRING_DYNAMICS.md](SPRING_DYNAMICS.md) - Physics-based activation spreading
- [MULTI_WAVE_ENGINE.md](MULTI_WAVE_ENGINE.md) - Brain wave consolidation
- [Memory README](README.md) - Memory system overview

---

## API Reference

### AwarenessGraph

```python
class AwarenessGraph:
    async def perceive(query: str) -> SemanticPerception
    async def remember(content: str, metadata: dict) -> str
    async def activate(query: str, strategy: ActivationStrategy, k: int) -> List[Memory]
    def get_metrics() -> AwarenessMetrics
```

### Data Types

```python
# awareness_types.py exports
SemanticPerception  # Perception result
ActivationStrategy  # PRECISE, BALANCED, EXPLORATORY, DEEP
ActivationBudget    # Resource limits
AwarenessMetrics    # System metrics
EdgeType            # TEMPORAL, SEMANTIC_RESONANCE, CAUSAL, REFERENCE
EdgeMetadata        # Edge data structure
```
