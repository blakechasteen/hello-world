# HoloLoom 9-Step Weaving Cycle Architecture

**Created: November 2025** - Elegance Pass Documentation
**Purpose:** Visual reference for the complete weaving cycle

## Complete Weaving Cycle

```mermaid
graph TB
    Start([User Query]) --> Step1[1. Loom Command]

    Step1 --> |Pattern Card Selection| Step2[2. Chrono Trigger]
    Step2 --> |Temporal Window| Step3[3. Yarn Graph]
    Step3 --> |Thread Selection| Step4[4. Resonance Shed]
    Step4 --> |Feature Extraction| Step5[5. Warp Space]
    Step5 --> |Tensor Operations| Step6[6. Convergence Engine]
    Step6 --> |Tool Selection| Step7[7. Tool Execution]
    Step7 --> |Results| Step8[8. Spacetime Fabric]
    Step8 --> |Provenance| Step9[9. Reflection Buffer]
    Step9 --> End([Response + Learning])

    style Start fill:#e1f5ff
    style End fill:#d4edda
    style Step1 fill:#fff3cd
    style Step2 fill:#fff3cd
    style Step3 fill:#cfe2ff
    style Step4 fill:#cfe2ff
    style Step5 fill:#f8d7da
    style Step6 fill:#f8d7da
    style Step7 fill:#d1e7dd
    style Step8 fill:#d1e7dd
    style Step9 fill:#e2e3e5
```

## Detailed Step Breakdown

### Step 1: Loom Command (Pattern Selection)

**Purpose:** Select processing pattern (BARE/FAST/FUSED)
**Input:** User query
**Output:** Pattern card (processing configuration)

```mermaid
graph LR
    Query[Query] --> Analyze{Analyze<br/>Complexity}
    Analyze --> |Simple| Bare[BARE Mode]
    Analyze --> |Standard| Fast[FAST Mode]
    Analyze --> |Complex| Fused[FUSED Mode]

    Bare --> |Config| Pattern[Pattern Card]
    Fast --> |Config| Pattern
    Fused --> |Config| Pattern
```

**Decision Criteria:**
- Query length and structure
- Keyword complexity indicators
- Historical performance data
- User preference overrides

---

### Step 2: Chrono Trigger (Temporal Window)

**Purpose:** Create time-based filtering window
**Input:** Pattern card, current time
**Output:** Temporal window with bounds

```mermaid
graph LR
    Time[Current Time] --> Window{Create<br/>Window}
    Pattern[Pattern Card] --> Window
    Window --> Bounds[Temporal Bounds]
    Bounds --> Decay[Recency Decay]
    Decay --> Filter[Memory Filter]
```

**Components:**
- Start time (temporal window start)
- End time (temporal window end)
- Episode filter (specific memory episodes)
- Recency weighting (decay over time)

---

### Step 3: Yarn Graph (Thread Selection)

**Purpose:** Select relevant memory threads
**Input:** Temporal window, query
**Output:** List of memory shards (threads)

```mermaid
graph TB
    YarnGraph[(Yarn Graph<br/>Memory Store)] --> Filter{Temporal<br/>Filter}
    Window[Temporal Window] --> Filter
    Filter --> Threads[Selected Threads]
    Query[Query] --> Relevance{Relevance<br/>Scoring}
    Threads --> Relevance
    Relevance --> Final[Filtered Threads]
```

**Filtering Logic:**
- Temporal bounds matching
- Semantic relevance scoring
- Graph relationship traversal
- Activation spreading

---

### Step 4: Resonance Shed (Feature Extraction)

**Purpose:** Extract multi-modal features (DotPlasma)
**Input:** Selected threads, query
**Output:** Features object (motifs, embeddings, spectral)

```mermaid
graph TB
    Threads[Memory Threads] --> Motifs[Motif Detector]
    Threads --> Embeddings[Matryoshka<br/>Embeddings]
    Threads --> Spectral[Spectral<br/>Features]

    Motifs --> DotPlasma[DotPlasma<br/>Feature Fluid]
    Embeddings --> DotPlasma
    Spectral --> DotPlasma

    DotPlasma --> Fusion{Multi-Modal<br/>Fusion}
    Fusion --> Features[Features Object]
```

**Feature Types:**
- **Motifs:** Symbolic patterns (regex/NLP)
- **Embeddings:** Multi-scale vectors (96/192/384D)
- **Spectral:** Graph topology (Laplacian eigenvalues)

---

### Step 5: Warp Space (Tensor Operations)

**Purpose:** Tension threads into continuous manifold
**Input:** Features, memory threads
**Output:** Tensioned warp space

```mermaid
graph LR
    Discrete[Discrete Threads] --> Tension{Tension<br/>Operation}
    Features[Features] --> Tension
    Tension --> Warp[Warp Space<br/>Continuous Manifold]
    Warp --> Compute[Tensor<br/>Operations]
    Compute --> Ready[Ready for<br/>Decision]
```

**Operations:**
- Thread tensioning (discrete → continuous)
- Tensor field computation
- Manifold operations
- Detensioning (continuous → discrete)

---

### Step 6: Convergence Engine (Tool Selection)

**Purpose:** Collapse probability distribution to tool choice
**Input:** Warp space, policy network
**Output:** Tool selection + confidence

```mermaid
graph TB
    Policy[Policy Network] --> Predict{Predict<br/>Distribution}
    WarpSpace[Warp Space] --> Predict
    Predict --> Probs[Tool Probabilities]

    Probs --> Strategy{Collapse<br/>Strategy}
    Strategy --> |ARGMAX| Tool1[Highest Prob]
    Strategy --> |EPSILON_GREEDY| Tool2[90% Best<br/>10% Explore]
    Strategy --> |BAYESIAN_BLEND| Tool3[Neural + Bandit]
    Strategy --> |PURE_THOMPSON| Tool4[Thompson Only]

    Tool1 --> Selected[Selected Tool]
    Tool2 --> Selected
    Tool3 --> Selected
    Tool4 --> Selected
```

**Strategies:**
- ARGMAX: Deterministic (highest probability)
- EPSILON_GREEDY: 90% exploit, 10% explore
- BAYESIAN_BLEND: 70% neural + 30% bandit
- PURE_THOMPSON: Full Thompson Sampling

---

### Step 7: Tool Execution

**Purpose:** Execute selected tool with context
**Input:** Tool selection, query, context
**Output:** Tool results

```mermaid
graph LR
    Tool[Selected Tool] --> Route{Route to<br/>Handler}
    Context[Context] --> Route

    Route --> |answer| LLM[LLM Generator]
    Route --> |search| Search[Search Engine]
    Route --> |notion_write| Notion[Notion API]
    Route --> |calc| Calc[Calculator]

    LLM --> Results[Tool Results]
    Search --> Results
    Notion --> Results
    Calc --> Results
```

**Tools:**
- **answer:** Generate response using LLM with context
- **search:** Search external knowledge bases
- **notion_write:** Write to Notion database
- **calc:** Perform calculations

---

### Step 8: Spacetime Fabric (Provenance Weaving)

**Purpose:** Weave results with complete lineage
**Input:** Tool results, trace data
**Output:** Spacetime object (4D fabric)

```mermaid
graph TB
    Results[Tool Results] --> Weave{Weave<br/>Fabric}
    Trace[Execution Trace] --> Weave
    Metadata[Metadata] --> Weave

    Weave --> Response[Response Text]
    Weave --> Confidence[Confidence Score]
    Weave --> Provenance[Provenance Trail]
    Weave --> Metrics[Performance Metrics]

    Response --> Spacetime[Spacetime Object]
    Confidence --> Spacetime
    Provenance --> Spacetime
    Metrics --> Spacetime
```

**Spacetime Components:**
- **Response:** Generated response text
- **Confidence:** Decision confidence [0, 1]
- **Trace:** Full computational lineage
- **Metadata:** Tool used, pattern, complexity, timings

---

### Step 9: Reflection Buffer (Learning)

**Purpose:** Learn from outcome for continuous improvement
**Input:** Spacetime, feedback (optional)
**Output:** Learning signals

```mermaid
graph LR
    Spacetime[Spacetime] --> Extract{Extract<br/>Signals}
    Feedback[User Feedback] --> Extract

    Extract --> Patterns[Pattern Quality]
    Extract --> Tools[Tool Success]
    Extract --> Timing[Timing Metrics]

    Patterns --> Update{Update<br/>Systems}
    Tools --> Update
    Timing --> Update

    Update --> Thompson[Thompson Priors]
    Update --> Policy[Policy Weights]
    Update --> Retrieval[Retrieval Weights]
```

**Learning Signals:**
- Pattern quality (which patterns work best)
- Tool success rates (Thompson Sampling priors)
- Timing metrics (performance optimization)
- Retrieval effectiveness (hot pattern feedback)

---

## Progressive Complexity (3-5-7-9 System)

```mermaid
graph TB
    Query[Query] --> Assess{Assess<br/>Complexity}

    Assess --> |Simple| Lite[LITE: 3 Steps]
    Assess --> |Standard| Fast[FAST: 5 Steps]
    Assess --> |Complex| Full[FULL: 7 Steps]
    Assess --> |Research| Research[RESEARCH: 9 Steps]

    Lite --> |Extract, Route, Execute| L_Result[Result<br/>< 50ms]
    Fast --> |+ Pattern, Temporal| F_Result[Result<br/>< 150ms]
    Full --> |+ Decision, Synthesis| Full_Result[Result<br/>< 300ms]
    Research --> |+ Advanced Warp, Tracing| R_Result[Result<br/>No Limit]

    style Lite fill:#d4edda
    style Fast fill:#cfe2ff
    style Full fill:#fff3cd
    style Research fill:#f8d7da
```

**Complexity Levels:**

| Level | Steps | Target | Use Case |
|-------|-------|--------|----------|
| **LITE** | 3 | <50ms | Simple lookups, cached queries |
| **FAST** | 5 | <150ms | Standard queries, real-time apps |
| **FULL** | 7 | <300ms | Complex queries, production systems |
| **RESEARCH** | 9 | No limit | Research, debugging, quality max |

---

## Module Architecture (Refactored)

```mermaid
graph TB
    subgraph "Weaving Orchestrator Main"
        Orchestrator[WeavingOrchestrator<br/>< 1000 lines]
    end

    subgraph "Executors"
        ToolExec[ToolExecutor<br/>332 lines]
    end

    subgraph "Memory"
        Yarn[YarnGraph<br/>155 lines]
    end

    subgraph "Strategies"
        Lite[LiteStrategy]
        Fast[FastStrategy]
        Full[FullStrategy]
        Res[ResearchStrategy]
    end

    subgraph "Core Components"
        Loom[Loom Command]
        Chrono[Chrono Trigger]
        Resonance[Resonance Shed]
        Warp[Warp Space]
        Convergence[Convergence Engine]
        Spacetime[Spacetime Fabric]
        Reflection[Reflection Buffer]
    end

    Orchestrator --> ToolExec
    Orchestrator --> Yarn
    Orchestrator --> Strategies
    Orchestrator --> Core

    style Orchestrator fill:#e1f5ff
    style ToolExec fill:#d4edda
    style Yarn fill:#d4edda
    style Strategies fill:#fff3cd
    style Core fill:#cfe2ff
```

**Refactoring Benefits:**
- **Before:** 3,476 lines in single file
- **After:** Modular architecture with <1,000 lines per file
- **Maintainability:** Each component has single responsibility
- **Testability:** Independent unit testing per module
- **Clarity:** Clear separation of concerns

---

## Performance Characteristics

```mermaid
gantt
    title Weaving Cycle Timing (FULL Mode)
    dateFormat X
    axisFormat %L ms

    section Stages
    Pattern Selection :0, 5
    Temporal Window :5, 10
    Thread Selection :10, 30
    Feature Extraction :30, 100
    Warp Space :100, 150
    Convergence :150, 170
    Tool Execution :170, 280
    Spacetime Weave :280, 295
    Reflection :295, 300
```

**Stage Breakdown (FULL Mode, ~300ms):**
1. Pattern Selection: 5ms
2. Temporal Window: 5ms
3. Thread Selection: 20ms
4. Feature Extraction: 70ms
5. Warp Space: 50ms
6. Convergence: 20ms
7. Tool Execution: 110ms
8. Spacetime Weave: 15ms
9. Reflection: 5ms

---

## Data Flow

```mermaid
flowchart LR
    Input[User Query] --> Discrete1[Discrete<br/>Symbolic]
    Discrete1 --> |Yarn Graph| Threads[Memory Threads]
    Threads --> Continuous[Continuous<br/>DotPlasma]
    Continuous --> |Warp Space| Manifold[Tensor Manifold]
    Manifold --> Decision[Decision<br/>Distribution]
    Decision --> Discrete2[Discrete<br/>Tool Choice]
    Discrete2 --> |Execution| Results[Tool Results]
    Results --> Fabric[Spacetime<br/>Fabric]
    Fabric --> Output[Response]

    style Input fill:#e1f5ff
    style Discrete1 fill:#f8d7da
    style Continuous fill:#d1e7dd
    style Discrete2 fill:#f8d7da
    style Output fill:#d4edda
```

**Representation Transitions:**
- **Query:** Natural language text
- **Threads:** Discrete symbolic memory (Yarn Graph)
- **DotPlasma:** Continuous feature fluid (Resonance Shed)
- **Manifold:** Tensioned tensor field (Warp Space)
- **Decision:** Probability distribution (Policy)
- **Tool:** Discrete action choice (Convergence)
- **Spacetime:** 4D woven fabric (output)

---

## Revision History

- **2025-11-17:** Created during Elegance Pass refactoring
- **Author:** Claude Code
- **Purpose:** Comprehensive visual documentation of HoloLoom architecture
