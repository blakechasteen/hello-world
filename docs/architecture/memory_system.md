# HoloLoom Memory System Architecture

**Created: November 2025** - Elegance Pass Documentation
**Purpose:** Visual reference for the 11 integrated memory systems

## Memory System Overview

```mermaid
graph TB
    subgraph "Core Memory (3 systems)"
        Vector[Vector Memory<br/>BM25 + Semantic]
        KG[Knowledge Graph<br/>NetworkX]
        Yarn[Yarn Graph<br/>Discrete Threads]
    end

    subgraph "Dynamic Memory (4 systems)"
        Awareness[Awareness Graph<br/>Activation Tracking]
        Spring[Spring Dynamics<br/>Physics-based]
        Wave[Multi-Wave Engine<br/>Temporal Waves]
        Warp[Warp Space<br/>Tensor Field]
    end

    subgraph "Specialized Memory (4 systems)"
        Photo[Photo Memory<br/>CLIP Embeddings]
        Visual[Visual Compression<br/>Graph→Image]
        Cache[Query Cache<br/>100x Speedup]
        Reflection[Reflection Buffer<br/>Learning]
    end

    Query[User Query] --> Vector
    Query --> KG
    Vector --> Awareness
    KG --> Yarn
    Yarn --> Warp
    Awareness --> Wave
    Spring --> Wave
    Warp --> Decision[Decision Engine]

    style Vector fill:#d4edda
    style KG fill:#d4edda
    style Yarn fill:#d4edda
    style Awareness fill:#cfe2ff
    style Spring fill:#cfe2ff
    style Wave fill:#cfe2ff
    style Warp fill:#cfe2ff
    style Photo fill:#fff3cd
    style Visual fill:#fff3cd
    style Cache fill:#fff3cd
    style Reflection fill:#fff3cd
```

## Core Memory Systems

### 1. Vector Memory (cache.py)

**Purpose:** BM25 + semantic similarity retrieval
**Technology:** Hybrid search (keyword + vector)

```mermaid
graph LR
    Query[Query] --> BM25[BM25<br/>Keyword Search]
    Query --> Semantic[Semantic<br/>Vector Search]

    BM25 --> Scores1[BM25 Scores]
    Semantic --> Scores2[Vector Scores]

    Scores1 --> Fusion{Score<br/>Fusion}
    Scores2 --> Fusion

    Fusion --> Ranked[Ranked Results]
```

**Features:**
- BM25 keyword matching (Okapi BM25)
- Semantic similarity (cosine distance)
- Hybrid score fusion (0.4 BM25 + 0.6 semantic)
- Multi-scale retrieval (96/192/384D)

---

### 2. Knowledge Graph (graph.py)

**Purpose:** Entity relationships with typed edges
**Technology:** NetworkX MultiDiGraph

```mermaid
graph LR
    Entity1[Entity A] -->|IS_A| Entity2[Entity B]
    Entity1 -->|USES| Entity3[Entity C]
    Entity2 -->|PART_OF| Entity4[Entity D]
    Entity3 -->|LEADS_TO| Entity5[Entity E]

    style Entity1 fill:#e1f5ff
    style Entity2 fill:#d4edda
    style Entity3 fill:#fff3cd
    style Entity4 fill:#cfe2ff
    style Entity5 fill:#f8d7da
```

**Edge Types:**
- **IS_A:** Taxonomy (inheritance)
- **USES:** Functional relationships
- **MENTIONS:** Reference relationships
- **LEADS_TO:** Causal chains
- **PART_OF:** Composition
- **IN_TIME:** Temporal ordering
- **OCCURRED_AT:** Event relationships

---

### 3. Yarn Graph (yarn_graph.py)

**Purpose:** Persistent symbolic memory (discrete threads)
**Technology:** Dict-based (production: Neo4j)

```mermaid
graph TB
    Storage[(Yarn Graph<br/>Storage)] --> Select{Temporal<br/>Selection}
    Window[Temporal Window] --> Select
    Select --> Threads[Memory Threads]
    Threads --> Tension{Tension<br/>Operation}
    Tension --> Warp[Warp Space<br/>Continuous]

    style Storage fill:#f8d7da
    style Threads fill:#f8d7da
    style Warp fill:#d1e7dd
```

**Operations:**
- Thread selection (temporal + relevance)
- Thread addition (new memories)
- Thread removal (cleanup)
- Tensioning (discrete → continuous)

---

## Dynamic Memory Systems

### 4. Awareness Graph (awareness_graph.py)

**Purpose:** Memory activation tracking with spreading activation
**Technology:** Graph-based activation propagation

```mermaid
graph TB
    Query[Query] --> Activate1[Initial<br/>Activation]
    Activate1 --> Node1[Node A<br/>α=1.0]
    Node1 --> Spread{Spreading<br/>Activation}

    Spread --> Node2[Node B<br/>α=0.8]
    Spread --> Node3[Node C<br/>α=0.6]
    Node2 --> Node4[Node D<br/>α=0.5]

    Node1 -.Decay.-> Node1_decay[α=0.95]
    Node2 -.Decay.-> Node2_decay[α=0.76]

    style Node1 fill:#d4edda
    style Node2 fill:#fff3cd
    style Node3 fill:#cfe2ff
    style Node4 fill:#e2e3e5
```

**Metrics:**
- **Activation:** Node activation levels (0.0-1.0)
- **Coherence:** How well-connected active nodes are
- **Temporal decay:** Inactive memories fade over time
- **Spreading:** Activation propagates across edges

---

### 5. Spring Dynamics (spring_dynamics.py)

**Purpose:** Physics-based memory connectivity
**Technology:** Hooke's law for memory relationships

```mermaid
graph LR
    A[Memory A] ---|Spring k=1.0| B[Memory B]
    B ---|Spring k=0.5| C[Memory C]
    A ---|Spring k=0.8| C

    A --> Force1[Force: -kΔx]
    B --> Force2[Force: -kΔx]
    C --> Force3[Force: -kΔx]

    Force1 --> Layout[Graph Layout]
    Force2 --> Layout
    Force3 --> Layout
```

**Physics Model:**
- **Spring constant (k):** Relationship strength
- **Displacement (Δx):** Distance between memories
- **Force (F):** F = -k × Δx (Hooke's law)
- **Equilibrium:** Natural clustering of related memories

---

### 6. Multi-Wave Engine (multi_wave_engine.py)

**Purpose:** Temporal wave propagation across memory graph
**Technology:** Multi-frequency wave interference

```mermaid
graph TB
    Source[Activation<br/>Source] --> Fast[Fast Wave<br/>λ=short]
    Source --> Slow[Slow Wave<br/>λ=long]

    Fast --> Propagate1{Propagate<br/>Network}
    Slow --> Propagate2{Propagate<br/>Network}

    Propagate1 --> Interference{Wave<br/>Interference}
    Propagate2 --> Interference

    Interference --> Pattern[Interference<br/>Pattern]
    Pattern --> Priority[Memory<br/>Priority]

    style Fast fill:#d1e7dd
    style Slow fill:#cfe2ff
    style Pattern fill:#fff3cd
```

**Wave Characteristics:**
- **Fast waves:** Rapid propagation, short wavelength
- **Slow waves:** Gradual propagation, long wavelength
- **Interference:** Constructive/destructive patterns
- **Priority:** Memories at interference peaks prioritized

---

### 7. Warp Space (warp/space.py)

**Purpose:** Tensioned tensor field for continuous math
**Technology:** Discrete → continuous transformation

```mermaid
graph LR
    Discrete[Discrete Threads<br/>Yarn Graph] --> Tension{Tension<br/>Operation}
    Tension --> Continuous[Continuous<br/>Tensor Field]

    Continuous --> Compute[Tensor<br/>Operations]
    Compute --> Result[Computation<br/>Result]

    Result --> Collapse{Collapse<br/>Operation}
    Collapse --> Discrete2[Discrete<br/>Decision]

    style Discrete fill:#f8d7da
    style Continuous fill:#d1e7dd
    style Discrete2 fill:#f8d7da
```

**Lifecycle:**
1. **Tension:** Discrete threads → continuous manifold
2. **Compute:** Tensor operations on manifold
3. **Collapse:** Continuous → discrete decision
4. **Detension:** Release back to discrete form

---

## Specialized Memory Systems

### 8. Photo Memory (photo_tokens.py)

**Purpose:** CLIP embeddings for images
**Technology:** Vision transformer embeddings

```mermaid
graph LR
    Image[Image] --> CLIP[CLIP Encoder]
    CLIP --> Embedding[512D Vector]
    Embedding --> Store[(Photo Store)]

    Query[Text Query] --> CLIP_Text[CLIP Text Encoder]
    CLIP_Text --> Query_Emb[512D Vector]

    Query_Emb --> Similarity{Cosine<br/>Similarity}
    Store --> Similarity
    Similarity --> Results[Similar Images]

    style Image fill:#e1f5ff
    style Embedding fill:#d1e7dd
    style Results fill:#d4edda
```

**Operations:**
- **Ingest:** Image → CLIP embedding → store
- **Search:** Text → CLIP embedding → similarity search
- **Retrieve:** Top-k similar images

---

### 9. Visual Compression (visual_compression.py)

**Purpose:** Graph→image compression (5-20x token savings)
**Technology:** Knowledge graph visualization

```mermaid
graph TB
    KG[Knowledge Graph] --> Layout{Force-Directed<br/>Layout}
    Layout --> Nodes[Node Positions]
    Layout --> Edges[Edge Paths]

    Nodes --> Render{Render<br/>to PNG}
    Edges --> Render

    Render --> PNG[PNG Image]
    PNG --> LLM[LLM Context]

    KG -.Text: 10000 tokens.-> LLM_Text[Text Context]
    PNG -.Image: 500 tokens.-> LLM

    style PNG fill:#d4edda
    style LLM fill:#d4edda
```

**Compression Benefits:**
- **5-20x token savings** vs. text representation
- **Visual structure** preserved
- **Automatic compression** when sources > threshold
- **Configurable threshold** (default: 10 sources)

---

### 10. Query Cache (query_cache.py)

**Purpose:** 100x speedup for repeated queries
**Technology:** LRU cache with TTL

```mermaid
graph LR
    Query[Query] --> Hash{Hash<br/>Query}
    Hash --> Lookup{Cache<br/>Lookup}

    Lookup --> |Hit| Cached[Cached Result<br/>~1ms]
    Lookup --> |Miss| Compute[Compute Result<br/>~150ms]

    Compute --> Store[Store in Cache]
    Store --> Return[Return Result]
    Cached --> Return

    style Cached fill:#d4edda
    style Compute fill:#fff3cd
```

**Performance:**
- **Cache hit:** <1ms (hash lookup)
- **Cache miss:** ~150ms (full computation)
- **Speedup:** 100-150x for repeated queries
- **TTL:** 5 minutes (configurable)
- **LRU eviction:** Oldest entries removed when full

---

### 11. Reflection Buffer (reflection/buffer.py)

**Purpose:** Episodic buffer for learning
**Technology:** Sliding window with pattern extraction

```mermaid
graph TB
    Spacetime[Spacetime Result] --> Buffer[(Reflection<br/>Buffer)]
    Feedback[User Feedback] --> Buffer

    Buffer --> Extract{Extract<br/>Patterns}
    Extract --> Quality[Pattern Quality]
    Extract --> Tools[Tool Success]
    Extract --> Timing[Timing Metrics]

    Quality --> Learn{Learning<br/>Loop}
    Tools --> Learn
    Timing --> Learn

    Learn --> Thompson[Thompson Priors<br/>α, β]
    Learn --> Policy[Policy Weights]
    Learn --> Retrieval[Hot Patterns]

    style Buffer fill:#e1f5ff
    style Learn fill:#fff3cd
    style Thompson fill:#d4edda
    style Policy fill:#d4edda
    style Retrieval fill:#d4edda
```

**Learning Signals:**
- **Pattern quality:** Which patterns work best
- **Tool success:** Thompson Sampling priors (α/β)
- **Timing metrics:** Performance optimization
- **Retrieval effectiveness:** Hot pattern feedback

---

## Memory Backend Configurations

```mermaid
graph TB
    Config[Config] --> Backend{Memory<br/>Backend}

    Backend --> |INMEMORY| NX[NetworkX<br/>In-Memory]
    Backend --> |HYBRID| Hybrid[Neo4j + Qdrant<br/>with Fallback]
    Backend --> |HYPERSPACE| Hyper[Advanced<br/>Multipass]

    NX --> |Always Works| Dev[Development]
    Hybrid --> |Prod Ready| Prod[Production]
    Hyper --> |Research| Research[Research]

    style NX fill:#d4edda
    style Hybrid fill:#cfe2ff
    style Hyper fill:#fff3cd
```

**Backend Options:**

| Backend | Storage | Features | Use Case |
|---------|---------|----------|----------|
| **INMEMORY** | NetworkX | Basic graph | Development, testing |
| **HYBRID** | Neo4j + Qdrant | Persistent + auto-fallback | Production |
| **HYPERSPACE** | Advanced | Gated multipass | Research |

---

## Memory Integration Flow

```mermaid
sequenceDiagram
    participant Q as Query
    participant V as Vector Memory
    participant K as Knowledge Graph
    participant Y as Yarn Graph
    participant A as Awareness Graph
    participant W as Warp Space
    participant D as Decision

    Q->>V: Search similar
    Q->>K: Find relationships
    V->>A: Activate nodes
    K->>Y: Select threads
    Y->>W: Tension threads
    A->>W: Propagate activation
    W->>D: Collapsed decision
    D->>K: Update graph
    D->>A: Update activation
```

**Integration Points:**
1. **Query** triggers Vector + KG search
2. **Vector Memory** activates Awareness Graph
3. **KG** selects threads from Yarn Graph
4. **Yarn** tensions into Warp Space
5. **Warp** feeds Decision Engine
6. **Decision** updates KG and Awareness

---

## Performance Characteristics

| System | Latency | Operation |
|--------|---------|-----------|
| **Vector Memory** | ~20ms | BM25 + semantic search |
| **Knowledge Graph** | ~10ms | Subgraph extraction |
| **Yarn Graph** | ~5ms | Thread selection |
| **Awareness Graph** | ~3ms | Activation spread |
| **Spring Dynamics** | ~15ms | Physics simulation |
| **Multi-Wave** | ~8ms | Wave propagation |
| **Warp Space** | ~50ms | Tensioning + compute |
| **Photo Memory** | ~50ms | CLIP similarity |
| **Visual Compression** | ~150ms | Graph → PNG |
| **Query Cache** | <1ms | Hash lookup (hit) |
| **Reflection Buffer** | <1ms | Pattern extraction |

**Total Retrieval Path:** ~100-150ms (FULL mode)

---

## Revision History

- **2025-11-17:** Created during Elegance Pass refactoring
- **Author:** Claude Code
- **Purpose:** Comprehensive visual documentation of memory systems
