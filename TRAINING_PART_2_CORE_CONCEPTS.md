# HoloLoom Training Guide: Part 2 - Core Concepts Deep Dive

**Target Audience:** Developers who understand Part 1 basics and want deep architectural understanding
**Prerequisites:** Part 1 of this training guide
**Reading Time:** 45-60 minutes
**Last Updated:** November 2025

---

## Introduction: From Basic to Deep

Part 1 taught you "what" HoloLoom does. Part 2 teaches you "how" it does it.

In this guide, you'll learn:
- The 9-layer weaving architecture (from input to learning)
- How data transforms at each stage
- When and why to use different execution modes
- How memory backends work and when to choose each
- The protocol-based design pattern that makes HoloLoom flexible
- How to configure the system for your needs

Let's start with the foundation: the 9-layer architecture.

---

## Section 1: The 9-Layer Architecture

### Overview

HoloLoom's power comes from its **layered design**. Each layer does one thing well and passes data to the next layer. This separation enables:

- **Swappable components** (replace memory backend without touching the orchestrator)
- **Understandable complexity** (each layer is simple; power emerges from combination)
- **Graceful degradation** (optional dependencies fail gracefully)

Let's walk through each layer:

### Diagram 1: Complete 9-Layer Data Transformation Flowchart

To understand how data flows through HoloLoom, visualize the complete transformation from input to output:

```
┌─────────────────────────────────────────────┐
│  INPUT: Query                                │
│  Type: Query(text="What is Thompson?")       │
│  Size: ~50 bytes                             │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 1: Input Processing (SpinningWheel)  │
│  Output: ProcessedInput                      │
│  Size: ~200 bytes (normalized text)          │
│  Time: ~2-5ms                                │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 2: Pattern Selection (LoomCommand)    │
│  Output: PatternCard (mode=FAST)             │
│  Size: ~100 bytes                            │
│  Time: ~1-2ms                                │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 3: Temporal Control (ChronoTrigger)   │
│  Output: TemporalWindow (timeout=200ms)      │
│  Size: ~150 bytes                            │
│  Time: <1ms                                  │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 4: Memory Retrieval (YarnGraph)       │
│  Output: List[MemoryShard] (n=6)             │
│  Size: ~6KB (6 shards × 1KB each)            │
│  Time: ~35-50ms                              │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 5: Feature Extraction (ResonanceShed) │
│  Output: DotPlasma (Features)                │
│  • Motifs: 3-5 patterns                      │
│  • Embeddings: [96D, 192D, 384D]             │
│  • Spectral: 5 graph features                │
│  Size: ~2KB                                  │
│  Time: ~25-35ms                              │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 6: Warp Space Tensioning              │
│  Output: TensionedThreads (Continuous)       │
│  Size: ~3KB (continuous manifold)            │
│  Time: ~8-12ms                               │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 7: Decision Collapse                  │
│  (ConvergenceEngine)                         │
│  Output: ActionPlan                          │
│  • Tool: "answer"                            │
│  • Confidence: 0.92                          │
│  Size: ~500 bytes                            │
│  Time: ~5-10ms                               │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 8: Tool Execution                     │
│  Output: ToolResult                          │
│  Size: ~1KB (response text)                  │
│  Time: ~30-50ms                              │
└────────────────┬────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  LAYER 9: Spacetime Construction             │
│  Output: Spacetime (Complete Fabric)         │
│  • Result + Trace + Metadata                 │
│  Size: ~10KB (complete provenance)           │
│  Time: ~5ms                                  │
└─────────────────────────────────────────────┘

SUMMARY:
├─ Total latency: ~155ms (FAST mode)
├─ Total data: 50 bytes → 10KB (200× expansion)
├─ Retrieval: 40% of time
├─ Features: 30% of time
├─ Decision: 7% of time
└─ Tool execution: 20% of time
```

**Key observations:**
- Memory retrieval (Layer 4) dominates latency (~35-50ms)
- Feature extraction (Layer 5) is second most expensive (~25-35ms)
- Data grows 200× through processing (compression happens in learning loop)
- Each layer is independent and can be profiled separately

---

### Layer 1: Input Processing (SpinningWheel)

**What it does:** Converts raw input (text, images, audio, video) into a unified `MemoryShard` format.

**Why it exists:** Different input types need different parsing. Instead of having the orchestrator handle text differently than images, we have specialized "spinners" that understand each format.

**Key files/classes:**
- `HoloLoom/spinningWheel/base.py` - `BaseSpinner` protocol
- `HoloLoom/spinningWheel/audio.py` - `AudioSpinner` (transcripts)
- `HoloLoom/spinningWheel/youtube.py` - `YouTubeSpinner` (YouTube videos)
- `HoloLoom/spinningWheel/autospin.py` - `InputRouter` (auto-detect input type)

**Data transformation:**
```
Raw Input (e.g., YouTube URL)
    ↓
[YouTubeSpinner]
    ↓
MemoryShard {
    id: "yt_abc123",
    content: "transcript text",
    modality: "audio",
    source: "youtube",
    entities: ["speaker", "topic"],  # Optional enrichment
    timestamp: 2025-11-16
}
```

**Example:**
```python
from HoloLoom.spinningWheel import YouTubeSpinner, YouTubeSpinnerConfig

config = YouTubeSpinnerConfig(chunk_duration=60.0)  # 1-min chunks
spinner = YouTubeSpinner(config)

shards = await spinner.spin({
    'url': 'https://youtube.com/watch?v=dQw4w9WgXcQ',
    'languages': ['en']
})
# Returns: List[MemoryShard] with 10-20 shards for a long video
```

### Layer 2: Pattern Selection (Loom Command)

**What it does:** Analyzes the query to determine complexity, then selects an execution template.

**Why it exists:** Simple queries ("What is 2+2?") waste time if they use the full 9-layer pipeline. By detecting complexity upfront, we route to the fastest sufficient solution.

**Key files/classes:**
- `HoloLoom/loom/command.py` - `LoomCommand`, `PatternCard`, `PatternSpec`
- `HoloLoom/routing/query_classifier.py` - `QueryClassifier` (complexity detection)

**Data transformation:**
```
Query: "What is Thompson Sampling?"
    ↓
[QueryClassifier.classify()]
    ↓
ComplexityLevel = MEDIUM  # Fact-based, needs context
    ↓
[LoomCommand.select_pattern()]
    ↓
PatternCard = FAST_PATTERN {
    embedding_scales: [768],
    retrieval_k: 6,
    enable_spectral: true,
    layers: 5  # vs 9 for FUSED
}
```

**Decision logic:**
- **SIMPLE queries** (<50ms):
  - "What is 2+2?" → BARE mode (regex, no context)
  - "What time is it?" → BARE mode (doesn't need knowledge graph)

- **STANDARD queries** (100-200ms):
  - "Explain recursion" → FAST mode (needs context, moderate depth)
  - "What is Thompson Sampling?" → FAST mode

- **COMPLEX queries** (200-500ms+):
  - "Compare Thompson Sampling with UCB" → FUSED mode (needs multi-angle analysis)
  - "How does your system work?" → FUSED mode

### Layer 3: Temporal Control (Chrono Trigger)

**What it does:** Creates a time window for this query and sets hard limits (timeout, max iterations).

**Why it exists:** Without boundaries, queries could run forever. Chrono Trigger enforces time discipline.

**Key files/classes:**
- `HoloLoom/chrono/trigger.py` - `ChronoTrigger`, `TemporalWindow`, `ExecutionLimits`

**Data transformation:**
```
PatternCard (FAST_PATTERN)
    ↓
[ChronoTrigger.fire()]
    ↓
TemporalWindow {
    start: 2025-11-16T10:30:00Z,
    deadline: 2025-11-16T10:30:00.5Z,  # 500ms timeout
    threads_active: [motif, embedding, memory],  # Which to activate
    max_depth: 2,  # Recursion limit
    decay_factor: 0.95  # Per-layer quality decay
}
```

**Example:**
```python
from HoloLoom.chrono.trigger import ChronoTrigger, ExecutionLimits

trigger = ChronoTrigger()
window = trigger.fire(
    pattern_card=pattern,
    base_timeout_ms=500
)

# window.exceeded_deadline() checks if we're over time
if window.exceeded_deadline():
    return current_best_result  # Stop and return what we have
```

### Layer 4: Memory Retrieval (Yarn Graph)

**What it does:** Queries the knowledge graph to retrieve relevant entities and relationships.

**Why it exists:** The orchestrator needs context about the query topic. The knowledge graph stores previous memories (learned facts).

**Key files/classes:**
- `HoloLoom/memory/graph.py` - `KG` (Yarn Graph, NetworkX-based)
- `HoloLoom/memory/cache.py` - `MemoryManager` (BM25 + semantic retrieval)
- `HoloLoom/awareness/awareness_graph.py` - `AwarenessGraph` (activation tracking)

**Data transformation:**
```
TemporalWindow + Query("Thompson Sampling")
    ↓
[MemoryManager.retrieve()]
    ↓
Retrieved MemoryShards {
    shard_0: {
        content: "Thompson Sampling is a Bayesian strategy...",
        confidence: 0.92,
        relevance: 0.88
    },
    shard_1: {
        content: "Exploration-exploitation tradeoff...",
        confidence: 0.85,
        relevance: 0.79
    },
    ... (up to K=6 shards)
}
```

**Memory backend options (covered in Section 4):**
- `INMEMORY`: NetworkX (development, ~10ms)
- `HYBRID`: Neo4j+Qdrant (production, ~50ms, auto-fallback)
- `HYPERSPACE`: Gated multipass (research, ~150ms)

### Layer 5: Feature Extraction (Resonance Shed)

**What it does:** Lifts three feature "threads" from the input and retrieved memories:
1. **Motif Thread** (symbolic): Keywords, entities ("Thompson", "Sampling", "exploration")
2. **Embedding Thread** (continuous): Dense vectors (768-dimensional)
3. **Spectral Thread** (topological): Graph properties (Laplacian eigenvalues, SVD components)

**Why it exists:** Different features capture different aspects of meaning. Neural policies make better decisions with richer features.

**Key files/classes:**
- `HoloLoom/resonance/shed.py` - `ResonanceShed`
- `HoloLoom/motif/base.py` - Motif detection (regex + spaCy)
- `HoloLoom/embedding/spectral.py` - Matryoshka embeddings + spectral features
- `HoloLoom/embedding/zero_copy.py` - Zero-copy embedding optimization (37x faster)

**Data transformation:**
```
Retrieved memories
    ↓
[ResonanceShed.lift_features()]
    ├── Motif Thread
    │   └─ ["thompson", "sampling", "bandit", "exploration"]
    │
    ├── Embedding Thread
    │   └─ tensor([0.23, -0.45, 0.67, ...])  # 768-dim
    │
    └── Spectral Thread
        └─ {
            laplacian_eigs: [2.1, 1.5, 0.9, 0.3],
            svd_components: [[0.2, 0.8], [...]],
            density: 0.15
           }
         ↓
      DotPlasma (unified feature fluid)
        └─ Features {
            symbolic: motifs,
            continuous: embeddings,
            topological: spectral
           }
```

**Performance characteristics:**
- **Motif extraction:** ~5-10ms (regex), ~20-30ms (spaCy)
- **Embedding:** ~5ms (warm cache), 37ms (cold, without zero-copy)
- **Spectral features:** ~10-20ms (depends on graph size)
- **Phase 5 optimization:** Compositional cache provides 10-300x speedup for repeated patterns

### Layer 6: Continuous Mathematics (Warp Space)

**What it does:** Takes discrete features (graph, motifs) and "tensions" them into a continuous manifold where tensor operations are possible.

**Why it exists:** Neural networks work with continuous tensors. Graphs are discrete. Warp Space bridges this gap.

**Key files/classes:**
- `HoloLoom/warp/space.py` - `WarpSpace`, `TensionedThread`

**Data transformation:**
```
DotPlasma (discrete + continuous features)
    ↓
[WarpSpace.tension()]
    ├─ Discrete graph → Continuous tensor
    ├─ Motifs → Embedding projections
    └─ Spectral features → Tensor operations
    ↓
TensionedManifold (continuous)
    └─ Continuous space where neural math happens

[WarpSpace.compute()]
    └─ Apply transformer blocks, attention, etc.

[WarpSpace.collapse()]
    └─ Return to discrete representation

[WarpSpace.detension()]
    └─ Restored graph (ready for next layer)
```

**Lifecycle:**
```python
async with WarpSpace(features=dot_plasma) as space:
    # tension: discrete → continuous
    tensioned = await space.tension()

    # compute: neural operations
    result = await space.compute(
        query_embedding=query_emb,
        context_embeddings=context_embs
    )

    # collapse: continuous → probabilities
    action_probs = await space.collapse()

    # detension: back to graph form
    graph_result = await space.detension()
    # Automatic cleanup on exit
```

### Layer 7: Decision Collapse (Convergence Engine)

**What it does:** Collapses probability distributions to discrete tool selection using one of 4 strategies.

**Why it exists:** The neural policy outputs probabilities for each tool. We need to pick one. Different strategies (ARGMAX, EPSILON_GREEDY, BAYESIAN_BLEND, PURE_THOMPSON) give different exploration/exploitation tradeoffs.

**Key files/classes:**
- `HoloLoom/convergence/engine.py` - `ConvergenceEngine`, `CollapseStrategy`
- `HoloLoom/policy/unified.py` - `NeuralPolicy`, `ThompsonBandit`

**Data transformation:**
```
Warp Space output: action_probs = [0.45, 0.30, 0.15, 0.10]
                                   (answer|search|calc|write)
    ↓
[ConvergenceEngine.collapse()]
    │
    ├─ ARGMAX: Select max prob → "answer" (greedy)
    │
    ├─ EPSILON_GREEDY (ε=0.1):
    │   └─ With 90% prob: select "answer"
    │   └─ With 10% prob: random explore → maybe "search"
    │
    ├─ BAYESIAN_BLEND (70% neural, 30% bandit):
    │   └─ Blend neural probs with Thompson posterior
    │
    └─ PURE_THOMPSON:
        └─ Sample from Thompson bandit posterior
        └─ Exploration-exploitation via uncertainty
    ↓
ActionPlan {
    tool: "answer",
    confidence: 0.45,
    strategy: "epsilon_greedy",
    exploration: false  # Exploiting (vs exploring)
}
```

**Bandit learning example:**
```python
# Thompson Sampling tracks success rates
bandit = ThompsonBandit(n_tools=4)

# Track outcomes
await bandit.update("answer", success=True, confidence=0.92)
await bandit.update("search", success=False, confidence=0.45)

# Next decision uses learned priors
posterior = bandit.sample()  # Bayesian posterior
# Tools with higher success rates more likely to be selected
```

### Layer 8: Tool Execution & Provenance (Spacetime)

**What it does:** Executes the selected tool and records complete provenance.

**Why it exists:** We want to know why a decision was made (for explainability, debugging, learning).

**Key files/classes:**
- `HoloLoom/fabric/spacetime.py` - `Spacetime`, `WeavingTrace`

**Data transformation:**
```
ActionPlan {tool: "answer", ...}
    ↓
[ToolExecutor.execute()]
    ├─ Load tool ("answer")
    ├─ Pass features + retrieved memories
    └─ Run tool logic
    ↓
ToolResult {response: "...", confidence: 0.87}
    ↓
[Spacetime.weave()]
    └─ Create 4D artifact:
        ├─ 3D: Semantic space (entities, features)
        ├─ 1D: Temporal trace (complete lineage)
        ├─ Confidence: 0.87
        └─ Metadata: {tool, strategy, time_ms, cache_hit}
    ↓
Spacetime (woven fabric)
    └─ Serializable, reproducible, debuggable
        ├─ response: "Thompson Sampling is..."
        ├─ trace: [query→features→decision→result]
        └─ metadata: {layers_used, latency_ms: 142}
```

**Complete traceability:**
```python
spacetime = await orchestrator.weave(query)

# View complete computational lineage
trace = spacetime.trace
print(f"Query: {trace.query}")
print(f"Pattern: {trace.pattern_card}")
print(f"Memory retrieved: {trace.retrieved_count} shards")
print(f"Features extracted: {trace.feature_types}")
print(f"Decision: {trace.tool_selected}")
print(f"Total time: {trace.total_duration_ms}ms")
```

### Layer 9: Learning & Reflection (Reflection Buffer)

**What it does:** Stores successful interactions and continuously improves the system.

**Why it exists:** Machine learning requires feedback. Reflection Buffer creates a learning loop.

**Key files/classes:**
- `HoloLoom/reflection/buffer.py` - `ReflectionBuffer`
- `HoloLoom/reflection/ppo_trainer.py` - PPO policy training
- `HoloLoom/reflection/semantic_learning.py` - Multi-task semantic learning

**Data transformation:**
```
Spacetime {response, confidence, tool, ...}
    ↓
User feedback (optional): {helpful: true, accurate: true}
    ↓
[ReflectionBuffer.store()]
    └─ Create learning signals (6 types)

Learning signals feed three systems:
    ├─ Thompson Bandit (success rates per tool)
    ├─ PPO Trainer (policy gradient updates)
    └─ Semantic Learner (fact consolidation)
    ↓
Next iteration uses improved policy
    └─ Better tool selection
    └─ Better feature extraction
    └─ Better decisions overall
```

**Background learning loop:**
```python
async with ReflectionBuffer(capacity=1000) as buffer:
    # Store interaction
    await buffer.store(
        spacetime=spacetime,
        feedback={"helpful": True, "confident": True}
    )

    # Every N stores, update policy
    if buffer.size() % 10 == 0:
        await buffer.train_policies()  # PPO update
        await buffer.update_bandits()  # Thompson update
```

---

## Section 2: Data Flow Through the System

### Complete Lifecycle Example

Let's trace a real query through all 9 layers:

**Query:** "What is Thompson Sampling?"

#### Step 1: Input Processing
```
User Query: "What is Thompson Sampling?"
    ↓
[Layer 1: SpinningWheel]
    ├─ InputRouter detects: text input
    ├─ TextSpinner processes input
    └─ Output: MemoryShard { content: "What is Thompson Sampling?", ... }
```

#### Step 2: Pattern Selection
```
Input MemoryShard
    ↓
[Layer 2: Loom Command]
    ├─ QueryClassifier analyzes:
    │   • Keyword count: 4
    │   • Query type: FACTUAL
    │   • Complexity signals: medium
    ├─ Decision: FAST mode (100-200ms)
    └─ Output: PatternCard = FAST_PATTERN
        {
            embedding_scales: [768],
            retrieval_k: 6,
            enable_spectral: true,
            n_layers: 5
        }
```

#### Step 3: Temporal Control
```
PatternCard
    ↓
[Layer 3: Chrono Trigger]
    ├─ Base timeout: 200ms (FAST mode)
    ├─ Create TemporalWindow
    └─ Output: TemporalWindow
        {
            start: 2025-11-16T10:30:00.000Z,
            deadline: 2025-11-16T10:30:00.200Z,
            threads_active: [motif, embedding, memory, spectral],
            max_depth: 2
        }
```

#### Step 4: Memory Retrieval
```
TemporalWindow + Query
    ↓
[Layer 4: Yarn Graph]
    ├─ MemoryManager.retrieve()
    ├─ BM25 + semantic search
    ├─ Find related memories from previous interactions
    └─ Output: Retrieved MemoryShards
        {
            shard_0: "Thompson Sampling balances exploration-exploitation...",
            shard_1: "UCB (Upper Confidence Bound) is an alternative...",
            shard_2: "Bayesian optimization uses posterior sampling...",
            ... (up to 6 shards)
        }
```

#### Step 5: Feature Extraction
```
Retrieved memories
    ↓
[Layer 5: Resonance Shed]
    ├─ Motif Thread:
    │   ["thompson", "sampling", "bandit", "exploration", "exploitation"]
    ├─ Embedding Thread:
    │   [0.23, -0.45, 0.67, ...] (768-dim vector)
    ├─ Spectral Thread:
    │   laplacian eigenvalues, SVD components
    └─ Output: DotPlasma (features unified)
        {
            motifs: [...],
            embeddings: tensor(...),
            spectral: {...}
        }
```

#### Step 6: Continuous Mathematics
```
DotPlasma
    ↓
[Layer 6: Warp Space]
    ├─ tension(): Convert to continuous manifold
    ├─ compute(): Apply transformer attention
    │   ├─ Query embedding: "thompson sampling"
    │   ├─ Key embeddings: retrieved memories
    │   └─ Attention scores: which memories are relevant?
    ├─ Produce logits: [answer: 0.45, search: 0.30, calc: 0.15, write: 0.10]
    └─ collapse(): Return to discrete space
```

#### Step 7: Decision Collapse
```
Tool logits: [answer: 0.45, search: 0.30, calc: 0.15, write: 0.10]
    ↓
[Layer 7: Convergence Engine]
    ├─ Strategy: EPSILON_GREEDY (ε=0.1)
    ├─ Roll random: 0.05 (< 0.1)
    ├─ Exploit (with 90% prob): Select max logit
    ├─ Selected tool: "answer"
    └─ Output: ActionPlan
        {
            tool: "answer",
            confidence: 0.45,
            strategy: "epsilon_greedy",
            exploration: false
        }
```

#### Step 8: Execution & Provenance
```
ActionPlan
    ↓
[Layer 8: ToolExecutor]
    ├─ Load "answer" tool
    ├─ Pass: query, retrieved memories, features
    ├─ Execute logic: Generate response using retrieved context
    └─ Output: ToolResult
        {
            response: "Thompson Sampling is a Bayesian strategy...",
            confidence: 0.87,
            sources: [shard_0, shard_1],
            time_ms: 42
        }
```

**Spacetime Fabric:**
```python
Spacetime {
    response: "Thompson Sampling is a Bayesian approach to exploration...",
    confidence: 0.87,
    tool: "answer",
    trace: {
        query_text: "What is Thompson Sampling?",
        pattern: "FAST",
        memory_retrieved: 6,
        features_extracted: 3,
        decision_strategy: "epsilon_greedy",
        time_breakdown: {
            retrieval_ms: 35,
            feature_extraction_ms: 15,
            decision_ms: 8,
            execution_ms: 42,
            total_ms: 100
        }
    },
    metadata: {
        cache_hit: False,
        layers_used: [1, 2, 3, 4, 5, 6, 7, 8],
        version: "2025-11-01"
    }
}
```

#### Step 9: Learning
```
Spacetime + User feedback (implicit: user didn't correct)
    ↓
[Layer 9: Reflection Buffer]
    ├─ Store interaction
    ├─ Create learning signals:
    │   ├─ Tool "answer" succeeded (confidence: 0.87)
    │   ├─ Pattern "FAST" was appropriate
    │   ├─ Memories were relevant
    │   └─ Decision confidence was high
    ├─ Update Thompson Bandit for "answer" tool
    ├─ Update PPO policy weights
    └─ Next similar query will:
        ├─ Be more likely to use "answer" tool
        ├─ Retrieve similar memories
        └─ Have higher confidence
```

### Type System

HoloLoom uses strong typing at layer boundaries:

```python
# Layer 1 output
MemoryShard = {
    id: str,
    content: str,
    modality: str,  # "text", "image", "audio"
    entities: List[str],
    timestamp: datetime
}

# Layer 2 output
PatternCard = {
    pattern_name: str,  # "BARE", "FAST", "FUSED"
    embedding_scales: List[int],
    retrieval_k: int,
    enable_spectral: bool
}

# Layer 5 output
Features = {
    motifs: List[str],
    embeddings: np.ndarray,  # (n, 768) shape
    spectral: Dict[str, Any]
}

# Layer 7 output
ActionPlan = {
    tool: str,
    confidence: float,
    strategy: str,  # "argmax", "epsilon_greedy", etc.
    exploration: bool
}

# Layer 8 output (final)
Spacetime = {
    response: str,
    confidence: float,
    tool: str,
    trace: WeavingTrace,
    metadata: Dict[str, Any]
}
```

---

## Section 3: The Three Execution Modes

HoloLoom adapts its processing depth based on query complexity. You control this with three modes:

### Diagram 2: BARE/FAST/FUSED Mode Comparison Matrix

The three execution modes offer different tradeoffs. Use this matrix to understand when each mode is appropriate:

```
┌──────────────────┬─────────────────┬─────────────────┬──────────────────┐
│   Feature        │      BARE       │      FAST       │      FUSED       │
├──────────────────┼─────────────────┼─────────────────┼──────────────────┤
│ Latency          │   50-100ms      │   100-200ms     │   200-500ms      │
│ Memory Usage     │   <1MB          │   5-10MB        │   10-20MB        │
│ Quality (avg)    │   ★★★☆☆         │   ★★★★☆         │   ★★★★★          │
├──────────────────┼─────────────────┼─────────────────┼──────────────────┤
│ Motif Detection  │   Regex only    │   Hybrid        │   Full NLP       │
│ Embedding Scale  │   Single (768D) │   Single (768D) │   Multi-scale    │
│ Graph Traversal  │   1-hop         │   2-hop         │   3-hop          │
│ Cache Enabled    │   ✗             │   ✓             │   ✓              │
│ Zero-Copy Emb    │   ✗             │   ✓             │   ✓              │
│ Thompson Sample  │   ✗             │   ✓             │   ✓              │
│ Spectral Feats   │   ✗             │   ✓             │   ✓              │
├──────────────────┼─────────────────┼─────────────────┼──────────────────┤
│ Accuracy (simple)│   85%           │   92%           │   93%            │
│ Accuracy (complex)│  45%           │   85%           │   95%            │
├──────────────────┼─────────────────┼─────────────────┼──────────────────┤
│ Best For         │  Speed-critical │  Production     │  Research        │
│                  │  queries        │  balanced       │  quality-critical│
│                  │  simple facts   │  standard       │  deep analysis   │
├──────────────────┼─────────────────┼─────────────────┼──────────────────┤
│ Example Query    │  "What time?"   │  "Explain X?"   │  "Compare X & Y" │
│                  │  "2+2?"         │  "What is TS?"  │  "Design a..."   │
└──────────────────┴─────────────────┴─────────────────┴──────────────────┘

SELECTION GUIDE:
• Need sub-100ms latency? → BARE
• Production deployment? → FAST (recommended for ~95% of queries)
• Research or quality critical? → FUSED
```

**Cross-reference:** See Section 4 for memory backend selection, which also impacts overall performance.

### BARE Mode: Maximum Speed

**When to use:** Simple, factual queries with clear answers.

**Performance:** ~50-100ms, <1MB memory

**Features enabled:**
- ✓ Regex-only motif detection (no spaCy)
- ✓ Single embedding scale (768-dim)
- ✗ Spectral features (disabled)
- ✓ Simple policy (no adapters)
- ✓ Minimal retrieval

**Example queries:**
- "What time is it?" (system question)
- "What is 2+2?" (arithmetic)
- "Spell 'python'" (trivial)
- "List the vowels" (simple)

**Configuration:**
```python
from HoloLoom.config import Config, ExecutionMode

config = Config.bare()
# Or manually:
config = Config(
    mode=ExecutionMode.BARE,
    n_transformer_layers=1,
    n_attention_heads=2,
    enable_semantic_calculus=False,
    spectral_k_eigen=0  # Disabled
)
```

**Code flow:**
```python
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="What is 2+2?"))
    # Time breakdown:
    # - Motif: 2ms
    # - Retrieval: 15ms
    # - Features: 5ms
    # - Decision: 3ms
    # - Execution: 10ms
    # Total: ~35ms
```

### FAST Mode: The Sweet Spot

**When to use:** Most production queries. Balanced speed and quality.

**Performance:** ~100-200ms, ~5-10MB memory

**Features enabled:**
- ✓ Hybrid motif detection (regex + spaCy if available)
- ✓ Matryoshka embeddings (768-dim)
- ✓ Spectral features (graph Laplacian, SVD)
- ✓ Neural policy with LoRA adapters
- ✓ Multi-memory retrieval (BM25 + semantic)
- ✓ Phase 5: Compositional cache (10-300x speedup for repeated queries)
- ✓ Zero-copy embeddings (37x faster scale extraction)

**Example queries:**
- "Explain recursion"
- "What is Thompson Sampling?"
- "How does photosynthesis work?"
- "Compare Python and JavaScript"

**Configuration:**
```python
config = Config.fast()
# Automatically enables:
# - Spectral features
# - Neural policy
# - Phase 5 linguistic gate + compositional cache
# - Zero-copy embeddings
```

**Code flow:**
```python
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(
        Query(text="What is Thompson Sampling?")
    )
    # Time breakdown:
    # - Motif (hybrid): 25ms
    # - Retrieval: 50ms
    # - Features (spectral): 30ms
    # - Decision: 8ms
    # - Execution: 42ms
    # Total: ~155ms

    # On repeated query:
    # - Compositional cache hit: 1ms (100x speedup!)
```

### FUSED Mode: Maximum Quality

**When to use:** Complex queries needing deep understanding, research, investigation.

**Performance:** ~200-500ms, ~10-20MB memory

**Features enabled:**
- ✓ Full hybrid motif detection
- ✓ All Matryoshka scales (if configured)
- ✓ Complete spectral features
- ✓ Full neural policy (all adapters)
- ✓ Deep memory retrieval
- ✓ Multi-hop graph traversal
- ✓ Phase 5: Compositional cache + full linguistic analysis
- ✓ All learning signals active
- ✓ Optional: Riemannian embeddings, wavelets, diffusion maps

**Example queries:**
- "Compare Thompson Sampling with UCB. What are the key tradeoffs?"
- "Explain how HoloLoom's architecture enables self-improvement."
- "Research the history and evolution of bandit algorithms."
- "Design a system that uses Thompson Sampling. What challenges might arise?"

**Configuration:**
```python
config = Config.fused()
# This is the default
# Enables absolutely everything for maximum quality
```

**Code flow:**
```python
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(
        Query(text="Compare Thompson Sampling with UCB...")
    )
    # Time breakdown:
    # - Motif (full): 35ms
    # - Retrieval (multi-hop): 120ms
    # - Features (all types): 100ms
    # - Decision (full network): 15ms
    # - Execution (deep): 80ms
    # Total: ~350ms

    # But on repeated query:
    # - Compositional cache: 2ms (100-200x speedup!)
```

### Decision Tree: Which Mode?

```
Query received
    │
    ├─ Is it a system question?
    │  (time, date, basic arithmetic)
    │  └─→ BARE ⚡
    │
    ├─ Is it a standard knowledge question?
    │  (explain concept, simple facts, comparisons)
    │  └─→ FAST ⚡⚡
    │
    └─ Is it complex/research-level?
       (deep analysis, design, reasoning)
       └─→ FUSED ⚡⚡⚡
```

### Performance Comparison Table

| Metric | BARE | FAST | FUSED |
|--------|------|------|-------|
| **Latency** | 35-50ms | 100-200ms | 200-500ms |
| **Memory** | <1MB | 5-10MB | 10-20MB |
| **Motif Detection** | Regex only | Hybrid | Full |
| **Spectral Features** | ✗ | ✓ | ✓ |
| **Neural Policy** | Simple | Full | Full |
| **Retrieval Depth** | Shallow | Medium | Deep |
| **Accuracy (simple)** | 85% | 92% | 93% |
| **Accuracy (complex)** | 45% | 85% | 95% |
| **Use BARE when** | Speed critical | Never | (Avoid) |
| **Use FAST when** | Production default | ✓ | (Use unless complex) |
| **Use FUSED when** | (Use FAST) | Complex queries | ✓ |

---

## Section 4: Memory Backends Explained

HoloLoom supports three memory backends, each optimized for different scenarios:

### Diagram 3: Memory Backend Auto-Fallback Chain

HoloLoom implements automatic graceful degradation for memory backends:

```
PRODUCTION DEPLOYMENT STRATEGY
═════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│  TIER 1: Primary Backend (HYBRID)                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Neo4j (Graph Database)        +  Qdrant (Vector DB)    │  │
│  │ • Persistent entity storage       • Fast semantic search │  │
│  │ • 10M+ entities                   • Multi-scale embeddings│ │
│  │ • Port: 7687                      • Port: 6333           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Status: Try to connect to Docker services                   │
└───────────────────────┬──────────────────────────────────────┘
                        │
                ┌───────┴───────┐
                │               │
           SUCCESS ✓        FAILURE ✗
              (95%)           (5%)
                │               │
                ▼               ▼
        ┌───────────────┐  ┌─────────────────────────────────┐
        │   Use HYBRID  │  │  Automatic Fallback Triggered   │
        │  (persistent) │  │                                 │
        │               │  │  • Log warning                  │
        │ • Neo4j ✓     │  │  • Keep config set to HYBRID    │
        │ • Qdrant ✓    │  │  • Switch to INMEMORY runtime   │
        │ • Latency:    │  │  • Continue normally (no crash!)│
        │   40-80ms     │  │                                 │
        │ • Scale:      │  │  TIER 2: Fallback Backend       │
        │   10M+        │  │  ┌─────────────────────────────┐│
        │               │  │  │ NetworkX (In-Memory Graph)  ││
        │               │  │  │ • Zero dependencies         ││
        │               │  │  │ • Fast startup              ││
        │               │  │  │ • Latency: 10-20ms          ││
        │               │  │  │ • Ephemeral (lost on exit)  ││
        │               │  │  └─────────────────────────────┘│
        │               │  │                                 │
        │               │  │ Key Benefit: Zero crashes!      │
        │               │  │ System continues operating.     │
        │               │  │ Data restored when Docker ready.│
        └───────────────┘  └─────────────────────────────────┘

FALLBACK LOGIC:
1. Load config: memory_backend = HYBRID
2. Try to connect:
   - Neo4j @ bolt://localhost:7687
   - Qdrant @ http://localhost:6333
3. If either fails:
   - Emit warning: "Docker service unavailable, using INMEMORY"
   - Continue with NetworkX backend
4. When Docker restored:
   - Data can be re-ingested
   - Or use separate persistence layer
```

**Key characteristics:**
- **Automatic**: No code changes needed
- **Transparent**: System operates normally with fallback
- **Non-destructive**: No data loss (just ephemeral instead of persistent)
- **Production-safe**: Never crash due to infrastructure issues

### INMEMORY: NetworkX (Development & Testing)

**What it is:** In-memory graph using NetworkX. Everything stays in RAM, disappears on shutdown.

**Performance:**
- Retrieval latency: ~10-20ms
- Memory per 1000 entities: ~2-5MB
- Persistence: None (ephemeral)
- Scaling limit: ~50k entities (then slowdown)

**When to use:**
- ✓ Local development
- ✓ Testing and debugging
- ✓ Demos and prototypes
- ✓ CI/CD pipelines
- ✗ Production (no persistence)
- ✗ Large datasets (>50k entities)

**Configuration:**
```python
from HoloLoom.config import Config, MemoryBackend

config = Config.fast()
config.memory_backend = MemoryBackend.INMEMORY

# Create empty shards for testing
from HoloLoom.documentation.types import MemoryShard

shards = [
    MemoryShard(
        id="test_0",
        content="Thompson Sampling...",
        modality="text"
    ),
    # ...more shards
]

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="..."))
    # Shards stored in memory, lost on exit
```

### HYBRID: Neo4j + Qdrant (Production)

**What it is:** Combined graph database (Neo4j) + vector database (Qdrant) with automatic fallback to INMEMORY if unavailable.

**Architecture:**
- **Neo4j** (graph database)
  - Stores entities and relationships
  - Enables pattern queries and traversals
  - 10M+ entity scalability

- **Qdrant** (vector database)
  - Stores dense embeddings
  - Fast semantic similarity search
  - Handles multi-scale Matryoshka embeddings

**Performance:**
- Retrieval latency: ~40-80ms (with services), ~10-20ms (fallback to INMEMORY)
- Persistence: Permanent (survives shutdown)
- Scaling: 10M+ entities easily
- Memory: Distributed across services

**When to use:**
- ✓ Production deployment
- ✓ Large knowledge bases (>100k shards)
- ✓ Persistent storage needed
- ✓ Multi-instance deployments
- ✗ Zero dependencies (requires Docker)
- ✗ Sub-50ms latency critical (use INMEMORY if needed)

**Setup:**
```bash
# Start Neo4j and Qdrant
docker-compose up -d

# Verify running
curl http://localhost:7687  # Neo4j
curl http://localhost:6333  # Qdrant
```

**Configuration:**
```python
from HoloLoom.config import Config, MemoryBackend

config = Config.fast()
config.memory_backend = MemoryBackend.HYBRID
config.neo4j_uri = "bolt://localhost:7687"
config.neo4j_username = "neo4j"
config.neo4j_password = "hololoom123"
config.qdrant_host = "localhost"
config.qdrant_port = 6333

async with WeavingOrchestrator(cfg=config) as orchestrator:
    # If Neo4j/Qdrant unavailable, automatically falls back to INMEMORY
    spacetime = await orchestrator.weave(Query(text="..."))
```

### HYPERSPACE: Gated Multipass (Research)

**What it is:** Advanced backend with multi-layer retrieval and gated access patterns. Useful for research and advanced scenarios.

**Performance:**
- Retrieval latency: ~100-200ms
- Complexity: Multi-hop graph traversal
- Use case: Finding indirect relationships

**When to use:**
- ✓ Research projects
- ✓ Finding hidden connections
- ✗ Production (slow)
- ✗ Budget-constrained (computationally expensive)

**Configuration:**
```python
config = Config.fused()
config.memory_backend = MemoryBackend.HYPERSPACE
config.hyperspace_depth = 3  # Max recursion depth
config.hyperspace_thresholds = [0.6, 0.75, 0.85]
config.hyperspace_breadth = 10  # Links per level
```

### Comparison Table

| Feature | INMEMORY | HYBRID | HYPERSPACE |
|---------|----------|--------|-----------|
| **Setup** | Zero | Docker | Docker + config |
| **Latency** | 10-20ms | 40-80ms | 100-200ms |
| **Persistence** | ✗ | ✓ | ✓ |
| **Scale** | ~50k | 10M+ | 10M+ |
| **Memory usage** | High | Distributed | Medium-High |
| **Dependencies** | None | Docker services | Docker services |
| **Auto-fallback** | N/A | ✓ (to INMEMORY) | Optional |
| **Best for** | Dev/test | Production | Research |

### Choosing a Backend

```python
# Development
config.memory_backend = MemoryBackend.INMEMORY

# Production (with fallback)
config.memory_backend = MemoryBackend.HYBRID

# Research (advanced features)
config.memory_backend = MemoryBackend.HYPERSPACE
```

---

## Section 5: The Protocol-Based Design

### Why Protocols?

HoloLoom uses Python protocols (similar to interfaces in other languages) to define component boundaries. This enables:

1. **Swappable implementations**: Replace Neo4j with PostgreSQL without touching orchestrator
2. **Type safety**: Protocol violations caught at development time
3. **Clear contracts**: Components know exactly what to expect
4. **Testable**: Easy to mock for unit testing

### Diagram 4: Protocol Swapping Before/After

Protocols enable clean, flexible architecture:

```
═══════════════════════════════════════════════════════════════════════

BEFORE: Tightly Coupled Architecture (Old Way)
─────────────────────────────────────────────

┌──────────────────────────┐
│   WeavingOrchestrator    │
│                          │
│ imports:                 │
│ • Neo4jGraph (hard-coded)│
│ • MemoryManager (direct) │
│ • NeuralPolicy (direct)  │
└──────────────┬───────────┘
               │
        ┌──────┴──────┬──────────┐
        ▼             ▼          ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │Neo4jGraph│  │MemoryMgr │  │NeuralPol │
  │(hard)    │  │(hard)    │  │(hard)    │
  └──────────┘  └──────────┘  └──────────┘

PROBLEMS:
✗ To test: Must start Neo4j (slow, complex setup)
✗ To swap: Edit orchestrator code (risky, couples concerns)
✗ To integrate: Create custom Neo4j variant (code duplication)
✗ Type safety: Mypy can't catch implementation mismatches


═══════════════════════════════════════════════════════════════════════

AFTER: Protocol-Based Architecture (New Way)
──────────────────────────────────────────────

┌──────────────────────────────────────────┐
│   WeavingOrchestrator                    │
│                                          │
│ uses protocols:                          │
│ • KGStore (interface)                    │
│ • Retriever (interface)                  │
│ • PolicyEngine (interface)               │
└──────────────┬──────────────────────────┘
               │
        ┌──────┼──────┬────────────┐
        │      │      │            │
        ▼      ▼      ▼            ▼
    ╔═════════════════════════════════╗
    ║    Abstract Protocols (Traits)   ║
    ║  • KGStore                       ║
    ║  • Retriever                     ║
    ║  • PolicyEngine                  ║
    ╚═════════════════════════════════╝
        │      │      │            │
        ▼      ▼      ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │NetworkXKG│ │Neo4jKG   │ │HyperKG   │
   │(impl 1)  │ │(impl 2)  │ │(impl 3)  │
   └──────────┘ └──────────┘ └──────────┘

BENEFITS:
✓ To test: Swap mock implementation (fast, simple)
✓ To swap: Pass different implementation (no code change)
✓ To integrate: Implement protocol (clear contract)
✓ Type safety: Mypy validates protocol conformance


═══════════════════════════════════════════════════════════════════════

EXAMPLE: Swapping for Testing
─────────────────────────────

# Production code (unchanged)
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(query)


# Test code (swap implementation)
class MockKG:  # Implements KGStore protocol
    async def add_edge(self, source, target, rel, weight):
        pass

    async def get_subgraph(self, nodes, depth):
        return self

    async def query_similar(self, emb, k, threshold):
        return []  # Return empty for predictability


# Same orchestrator code, different backend!
async with WeavingOrchestrator(
    cfg=config,
    shards=shards,
    kg=MockKG()  # ← Swap one line!
) as orchestrator:
    spacetime = await orchestrator.weave(query)


RESULT:
• No changes to orchestrator
• Test runs 100x faster (no DB)
• Type-safe (Mypy validates MockKG implements KGStore)
• Clear contract (protocol defines what's needed)
```

### Key Protocols

#### 1. PolicyEngine Protocol

**What it defines:** How decision-making components work.

```python
from typing import Protocol

class PolicyEngine(Protocol):
    """Core decision-making interface."""

    async def select_tool(
        self,
        context: Features,
        available_tools: List[str],
        temperature: float = 1.0
    ) -> ActionPlan:
        """
        Select which tool to use given context.

        Returns: ActionPlan with selected tool + confidence
        """
        ...

    async def update(
        self,
        tool: str,
        outcome: bool,
        confidence: float
    ) -> None:
        """
        Learn from outcome of last decision.

        Args:
            tool: Tool that was selected
            outcome: True if successful
            confidence: Confidence level (0-1)
        """
        ...
```

**Implementations:**
- `NeuralPolicy` (transformer-based, default)
- `ThompsonBandit` (exploration via uncertainty)
- `SimplePolicy` (rule-based, for testing)

**Example: Swapping policies**
```python
# Default (neural)
policy = create_policy(
    mem_dim=384,
    emb=embeddings,
    scales=[768]
)

# For testing, use simple rule-based policy
class SimpleTestPolicy:
    async def select_tool(self, context, tools, temp=1.0):
        # Always select first tool
        return ActionPlan(tool=tools[0], confidence=1.0)

# Orchestrator doesn't care which implementation!
async with WeavingOrchestrator(
    cfg=config,
    shards=shards,
    policy=SimpleTestPolicy()  # Swap easily
) as orchestrator:
    pass
```

#### 2. KGStore Protocol

**What it defines:** How knowledge graph storage works.

```python
class KGStore(Protocol):
    """Knowledge graph storage interface."""

    async def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0
    ) -> None:
        """Add an entity relationship."""
        ...

    async def query_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.5
    ) -> List[Entity]:
        """Find similar entities by embedding."""
        ...

    async def get_subgraph(
        self,
        center_nodes: List[str],
        depth: int = 2
    ) -> 'KGStore':
        """Extract subgraph around nodes."""
        ...
```

**Implementations:**
- `NetworkXKG` (in-memory, development)
- `Neo4jKG` (production, persistent)
- `HyperKG` (research, multi-level)

#### 3. Retriever Protocol

**What it defines:** How memory retrieval works.

```python
class Retriever(Protocol):
    """Memory retrieval interface."""

    async def retrieve(
        self,
        query: Union[str, np.ndarray],
        k: int = 6,
        filters: Dict = None
    ) -> List[MemoryShard]:
        """
        Retrieve relevant memories.

        Args:
            query: Text or embedding
            k: Number of results
            filters: Optional metadata filters

        Returns: Top-K similar shards
        """
        ...

    async def ingest(
        self,
        shard: MemoryShard
    ) -> None:
        """Store a memory shard."""
        ...
```

**Implementations:**
- `BM25Retriever` (keyword search)
- `SemanticRetriever` (embedding similarity)
- `HybridRetriever` (BM25 + semantic)

### How Protocols Enable Flexibility

**Example: Testing with mocks**

```python
# Real implementation (slow, requires DB)
real_kg = Neo4jKG(uri="bolt://localhost:7687")

# Test implementation (fast, in-memory)
class MockKG:
    def __init__(self):
        self.edges = {}

    async def add_edge(self, source, target, rel, weight):
        self.edges[(source, target)] = (rel, weight)

    async def get_subgraph(self, nodes, depth):
        return self  # Return self for testing

# Orchestrator works with either!
async with WeavingOrchestrator(
    cfg=config,
    kg=MockKG()  # Swap for testing
) as orchestrator:
    spacetime = await orchestrator.weave(query)
```

---

## Section 6: Configuration System

### Three Factory Methods

HoloLoom provides three pre-configured modes accessible via factory methods:

```python
from HoloLoom.config import Config

# Fastest
config = Config.bare()

# Balanced (recommended for production)
config = Config.fast()

# Highest quality
config = Config.fused()
```

### Diagram 5: Configuration Decision Tree

Use this flowchart to select the right configuration for your scenario:

```
                    START
                      │
                      ▼
        ┌─────────────────────────────┐
        │  What's your priority?      │
        └──┬──────────┬────────┬──────┘
           │          │        │
      Speed      Balance    Quality
           │          │        │
           ▼          ▼        ▼
      ┌────────┐ ┌────────┐ ┌────────┐
      │  BARE  │ │  FAST  │ │ FUSED  │
      └───┬────┘ └───┬────┘ └───┬────┘
          │          │          │
          ▼          ▼          ▼
   ┌────────────┐ ┌──────────┐ ┌────────────┐
   │ <100ms     │ │Production│ │  Research  │
   │ Simple     │ │ Balanced │ │   Quality  │
   │ Regex      │ │ Hybrid   │ │   Fused    │
   │ 1-hop KG   │ │ features │ │   Multi-hop│
   │ INMEMORY   │ │ Neural   │ │   Full NLP │
   │ No cache   │ │ Cache ✓  │ │   HYPERSP. │
   └────────────┘ └──────────┘ └────────────┘

FAST Path (Most Common):

START
  │
  ├─ Need < 100ms? ──────────→ YES ──→ BARE
  │                                     │
  │                                     └─ Then use:
  │                                        • INMEMORY backend
  │                                        • Timeout: 100ms
  │                                        • Minimal retrieval
  │
  ├─ Production query? ──────→ YES ──→ FAST ← (Default)
  │                                    │
  │                                    └─ Then use:
  │                                       • HYBRID backend
  │                                       • Timeout: 500ms
  │                                       • Full features
  │
  └─ Need maximum quality? ──→ YES ──→ FUSED
                                       │
                                       └─ Then use:
                                          • HYPERSPACE backend
                                          • Timeout: no limit
                                          • All features
```

### Key Configuration Parameters

#### Execution Control
```python
config = Config.fast()

# Mode selection
config.mode = ExecutionMode.FAST  # or BARE, FUSED

# Timeout controls
config.pipeline_timeout = 0.5  # Max 500ms total
config.retrieval_timeout = 0.2  # Max 200ms for retrieval
```

#### Memory Configuration
```python
# Backend selection
config.memory_backend = MemoryBackend.HYBRID  # INMEMORY, HYBRID, HYPERSPACE

# Neo4j connection (for HYBRID)
config.neo4j_uri = "bolt://localhost:7687"
config.neo4j_username = "neo4j"
config.neo4j_password = "hololoom123"

# Qdrant connection (for HYBRID)
config.qdrant_host = "localhost"
config.qdrant_port = 6333
```

#### Retrieval Settings
```python
# How many memories to retrieve
config.retrieval_k = 6

# Balance between BM25 (keyword) and semantic search
config.bm25_weight = 0.15  # 15% BM25, 85% semantic
```

#### Feature Extraction
```python
# Graph analysis (spectral features)
config.spectral_k_eigen = 4  # Number of graph eigenvalues
config.svd_components = 2    # Number of SVD topics

# Embedding scales (Matryoshka)
config.scales = [768]  # Single scale for speed
# config.scales = [128, 256, 384, 768]  # Multi-scale for quality
```

#### Policy Settings
```python
# Tool selection strategy
from HoloLoom.documentation.types import BanditStrategy

config.bandit_strategy = BanditStrategy.EPSILON_GREEDY
config.epsilon = 0.1  # 10% exploration rate

# Exploration/exploitation
config.blend_neural_weight = 0.7  # 70% neural, 30% bandit (if BAYESIAN_BLEND)
```

#### Adapter Configuration
```python
# LoRA adapters for different domains
config.n_adapters = 4
# Adapters: general, farm, brewing, mirrorcore

# Neural network architecture
config.n_transformer_layers = 2
config.n_attention_heads = 4
```

### Performance vs Quality Tradeoff

```python
# Maximum speed (sacrifice quality)
config = Config.bare()
# Latency: ~50ms
# Quality: 70% (simple queries only)

# Balanced (recommended)
config = Config.fast()
# Latency: ~150ms
# Quality: 90%

# Maximum quality (sacrifice speed)
config = Config.fused()
# Latency: ~350ms
# Quality: 98%
```

### Customization Examples

**Example 1: Production with High Reliability**
```python
config = Config.fast()
config.memory_backend = MemoryBackend.HYBRID  # Persistent
config.pipeline_timeout = 1.0  # 1 second timeout
config.retrieval_k = 10  # Get more context
config.enable_semantic_calculus = True  # Extra analysis
```

**Example 2: Speed-Critical Application**
```python
config = Config.bare()
config.memory_backend = MemoryBackend.INMEMORY  # No I/O
config.pipeline_timeout = 0.1  # 100ms max
config.retrieval_k = 3  # Minimal retrieval
config.n_transformer_layers = 1  # Minimal network
```

**Example 3: Research/Exploration**
```python
config = Config.fused()
config.memory_backend = MemoryBackend.HYPERSPACE  # Multi-level
config.pipeline_timeout = 10.0  # No rush
config.retrieval_k = 20  # Maximum context
config.enable_semantic_calculus = True
config.use_wavelets = True  # Advanced features
config.use_riemannian = True  # Geometric embeddings
```

### Diagram 6: Configuration Validation Checklist

HoloLoom validates configuration automatically. Use this checklist when customizing:

```
Configuration Validation Pipeline:
═══════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────┐
│ Step 1: Mode Consistency Check                             │
├────────────────────────────────────────────────────────────┤
│ Verifies: BARE mode doesn't enable optional dependencies   │
│                                                            │
│ ✓ BARE mode with spaCy disabled              → PASS       │
│ ✗ BARE mode with spectral features           → FAIL       │
│                                                            │
│ Example:                                                   │
│ if config.mode == BARE and config.spectral_k_eigen > 0:  │
│     raise ValueError("BARE mode cannot use spectral")     │
└────────────┬───────────────────────────────────────────────┘
             ▼
┌────────────────────────────────────────────────────────────┐
│ Step 2: Memory Backend Availability Check                  │
├────────────────────────────────────────────────────────────┤
│ Verifies: Selected backend is available or fallback exists │
│                                                            │
│ ✓ HYBRID with fallback enabled                 → PASS     │
│ ✗ HYBRID with no Docker and no fallback        → FAIL     │
│                                                            │
│ Example:                                                   │
│ if config.memory_backend == HYBRID:                       │
│     try_connect_to_services()                             │
│     if not connected and not fallback:                    │
│         raise ConfigError("No backend available")         │
└────────────┬───────────────────────────────────────────────┘
             ▼
┌────────────────────────────────────────────────────────────┐
│ Step 3: Embedding Scale Alignment Check                    │
├────────────────────────────────────────────────────────────┤
│ Verifies: Embedding scales match retrieval expectations    │
│                                                            │
│ ✓ Scales: [96, 192, 384, 768] ascending      → PASS       │
│ ✗ Scales: [768, 384, 192, 96] descending     → FAIL       │
│ ✗ Scales: [768, 768, 768] duplicated         → FAIL       │
│                                                            │
│ Example:                                                   │
│ if config.scales != sorted(config.scales):                │
│     raise ValueError("Scales must be in ascending order")  │
└────────────┬───────────────────────────────────────────────┘
             ▼
┌────────────────────────────────────────────────────────────┐
│ Step 4: Timeout Sanity Check                               │
├────────────────────────────────────────────────────────────┤
│ Verifies: Timeouts are reasonable for execution mode       │
│                                                            │
│ ✓ BARE mode with timeout 100ms                → PASS      │
│ ✗ BARE mode with timeout 2000ms               → WARNING   │
│ ✗ Any mode with timeout 10000ms+              → WARNING   │
│                                                            │
│ Example:                                                   │
│ expected_latency = estimate_latency(config.mode)           │
│ if config.pipeline_timeout < expected_latency * 0.8:      │
│     warn(f"Timeout may be too aggressive")                 │
└────────────┬───────────────────────────────────────────────┘
             ▼
┌────────────────────────────────────────────────────────────┐
│ Step 5: Dependency Availability Check                      │
├────────────────────────────────────────────────────────────┤
│ Verifies: Required Python packages are available           │
│                                                            │
│ ✓ spaCy installed and model available          → PASS     │
│ ✗ spaCy required but not installed             → WARNING  │
│ ✗ sentence-transformers required but missing   → WARNING  │
│                                                            │
│ Example:                                                   │
│ if config.use_spacy:                                       │
│     try: import spacy; spacy.load('en_core')             │
│     except: warn("spaCy unavailable, falling back")        │
└────────────┬───────────────────────────────────────────────┘
             ▼
         ✓ CONFIG VALID ✓

         All checks passed!
         System ready for initialization.
```

### Configuration Validation

HoloLoom validates configuration automatically:

```python
config = Config.fused()

# Validates in __post_init__:
# - Scales are sorted ✓
# - Fusion weights sum to ~1.0 ✓
# - Hyperspace thresholds match depth ✓
# - Timeouts are reasonable ✓

# If invalid:
# ValueError: scales must be in ascending order
# UserWarning: Fusion weights sum to 1.2, normalizing...
```

### Serialization

Save and load configurations:

```python
# Serialize to dict
config = Config.fast()
data = config.to_dict()

# Save to file
import json
with open('config.json', 'w') as f:
    json.dump(data, f)

# Load from file
with open('config.json') as f:
    data = json.load(f)

restored_config = Config.from_dict(data)
```

---

## Summary: Pulling It All Together

### The Complete Picture

```
User Query
    ↓
[1] INPUT (SpinningWheel) → MemoryShard
    ↓
[2] PATTERN (LoomCommand) → PatternCard
    ↓
[3] TEMPORAL (ChronoTrigger) → TemporalWindow
    ↓
[4] MEMORY (YarnGraph) → Retrieved shards
    ↓
[5] FEATURES (ResonanceShed) → DotPlasma
    ↓
[6] MATH (WarpSpace) → Continuous manifold
    ↓
[7] DECISION (ConvergenceEngine) → ActionPlan
    ↓
[8] EXECUTION (ToolExecutor) → Spacetime
    ↓
[9] LEARNING (ReflectionBuffer) → Updated system
    ↓
Response delivered, system improved
```

### Key Concepts

| Concept | Purpose | Performance |
|---------|---------|-------------|
| **9-Layer Architecture** | Separation of concerns | Maintainable |
| **BARE/FAST/FUSED Modes** | Speed/quality tradeoff | Flexible |
| **Memory Backends** | INMEMORY/HYBRID/HYPERSPACE | Scalable |
| **Protocol-Based Design** | Swappable components | Extensible |
| **Spacetime Fabric** | Complete provenance | Debuggable |
| **Reflection Loop** | Continuous improvement | Self-improving |

### When to Use What

| Scenario | Config | Backend | Why |
|----------|--------|---------|-----|
| Local dev | Config.bare() | INMEMORY | Fast iteration |
| Production API | Config.fast() | HYBRID | Speed + reliability |
| Research | Config.fused() | HYPERSPACE | Maximum quality |
| Testing | Config.bare() | INMEMORY | Fast, no setup |

### Next Steps

Having completed Part 2, you now understand:

✓ How the 9-layer architecture works
✓ How data flows through the system
✓ When to use each execution mode
✓ How memory backends work
✓ Why protocols matter
✓ How to configure HoloLoom

**Next:** Part 3 will cover **Advanced Techniques** including:
- Building custom spinners (input adapters)
- Implementing custom policies
- Integration patterns
- Performance optimization
- Production deployment

---

## Appendix: File Reference

### Core Architecture Files

| File | Lines | Purpose |
|------|-------|---------|
| `/HoloLoom/weaving_orchestrator.py` | 1,963 | Main orchestrator (9-step cycle) |
| `/HoloLoom/config.py` | 530 | Configuration system |
| `/HoloLoom/loom/command.py` | ~300 | Pattern selection |
| `/HoloLoom/chrono/trigger.py` | ~300 | Temporal control |
| `/HoloLoom/memory/graph.py` | ~500 | Yarn Graph (NetworkX) |
| `/HoloLoom/resonance/shed.py` | ~400 | Feature extraction |
| `/HoloLoom/warp/space.py` | ~450 | Continuous mathematics |
| `/HoloLoom/convergence/engine.py` | ~380 | Decision collapse |
| `/HoloLoom/fabric/spacetime.py` | ~350 | Provenance tracking |
| `/HoloLoom/reflection/buffer.py` | ~400 | Learning loop |

### Memory Backend Files

| File | Purpose |
|------|---------|
| `/HoloLoom/memory/backend_factory.py` | Backend creation |
| `/HoloLoom/memory/cache.py` | BM25 + semantic retrieval |
| `/HoloLoom/memory/neo4j_graph.py` | Neo4j integration |
| `/HoloLoom/memory/hyperspace_backend.py` | Hyperspace backend |

### Input Adapter Files

| File | Purpose |
|------|---------|
| `/HoloLoom/spinningWheel/base.py` | Base spinner protocol |
| `/HoloLoom/spinningWheel/audio.py` | Audio/text processing |
| `/HoloLoom/spinningWheel/youtube.py` | YouTube transcription |

---

## Glossary

**ActionPlan:** Output of decision layer, specifies which tool to use and confidence level.

**Convergence Engine:** Layer 7 component that converts probability distributions to discrete decisions.

**DotPlasma:** Unified feature representation combining motifs, embeddings, and spectral features.

**Loom Command:** Layer 2 component that selects execution pattern (BARE/FAST/FUSED).

**MemoryShard:** Standardized unit of stored knowledge.

**Spacetime:** 4D fabric containing complete provenance of computation.

**Warp Space:** Layer 6 continuous mathematical manifold where tensor operations occur.

**Yarn Graph:** Knowledge graph storing entities and relationships (layer 4).

---

**End of Part 2**

Read Part 3 for advanced techniques and production patterns.
