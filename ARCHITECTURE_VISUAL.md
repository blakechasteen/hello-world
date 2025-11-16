# HoloLoom Visual Architecture

**Last Updated**: 2025-11-15

This document provides comprehensive visual diagrams of the HoloLoom architecture, organized by major system components and data flows.

---

## Table of Contents

1. [9-Step Weaving Cycle](#1-9-step-weaving-cycle)
2. [Component Relationship Map](#2-component-relationship-map)
3. [Data Flow Architecture](#3-data-flow-architecture)
4. [Memory System Architecture](#4-memory-system-architecture)
5. [Learning Systems Timeline](#5-learning-systems-timeline)
6. [Integration Points](#6-integration-points)
7. [Phase Architectures](#7-phase-architectures)
8. [Backend Architecture](#8-backend-architecture)
9. [RAG System Architecture](#9-rag-system-architecture)
10. [Alignment Framework](#10-alignment-framework)

---

## 1. 9-Step Weaving Cycle

The complete weaving cycle from query input to fabric output with reflection learning.

```mermaid
graph TB
    subgraph "Input Layer"
        Q[Query Input]
    end

    subgraph "Step 1: Pattern Selection"
        LC[Loom Command]
        PC[Pattern Card<br/>BARE/FAST/FUSED]
        Q --> LC
        LC --> PC
    end

    subgraph "Step 2: Temporal Control"
        CT[Chrono Trigger]
        TW[Temporal Window]
        PC --> CT
        CT --> TW
    end

    subgraph "Step 3: Thread Selection"
        YG[Yarn Graph<br/>Knowledge Graph]
        TS[Selected Threads]
        TW --> YG
        YG --> TS
    end

    subgraph "Step 4: Feature Extraction"
        RS[Resonance Shed]
        DP[DotPlasma<br/>Feature Fluid]
        TS --> RS
        RS --> DP
    end

    subgraph "Step 5: Continuous Manifold"
        WS[Warp Space]
        TT[Tensioned Threads]
        DP --> WS
        WS --> TT
    end

    subgraph "Step 6: Decision Collapse"
        CE[Convergence Engine]
        TD[Tool Decision]
        TT --> CE
        CE --> TD
    end

    subgraph "Step 7: Action Execution"
        TE[Tool Executor]
        TR[Tool Result]
        TD --> TE
        TE --> TR
    end

    subgraph "Step 8: Fabric Weaving"
        ST[Spacetime Fabric]
        WV[Woven Output<br/>+ Trace]
        TR --> ST
        ST --> WV
    end

    subgraph "Step 9: Learning Loop"
        RB[Reflection Buffer]
        LL[Learning & Adaptation]
        WV --> RB
        RB --> LL
        LL -.Feedback.-> YG
        LL -.Feedback.-> CE
    end

    subgraph "Output Layer"
        OUT[Response + Provenance]
        WV --> OUT
    end

    style Q fill:#e1f5ff
    style PC fill:#fff4e1
    style TW fill:#ffe1f5
    style TS fill:#e1ffe1
    style DP fill:#f5e1ff
    style TT fill:#ffe1e1
    style TD fill:#e1fff5
    style TR fill:#fff5e1
    style WV fill:#e1e1ff
    style LL fill:#ffe1e1
    style OUT fill:#e1f5ff
```

**Legend**:
- **Blue**: Input/Output layers
- **Yellow**: Pattern selection & temporal control
- **Pink**: Thread management
- **Green**: Feature extraction
- **Purple**: Continuous mathematics
- **Red**: Decision making & execution
- **Cyan**: Output synthesis
- **Orange**: Learning & feedback

---

## 2. Component Relationship Map

High-level view of all major HoloLoom components and their relationships.

```mermaid
graph LR
    subgraph "Core Orchestration"
        WO[Weaving Orchestrator]
        CFG[Config<br/>BARE/FAST/FUSED]
    end

    subgraph "Memory Systems"
        YG[Yarn Graph<br/>KG]
        VC[Vector Cache<br/>BM25 + Semantic]
        AG[Awareness Graph]
        PB[Photo Tokens<br/>CLIP]
        CB[Compositional Cache]
    end

    subgraph "Feature Extraction"
        ME[Matryoshka Embeddings<br/>96/192/384D]
        SP[Spectral Features<br/>Graph + SVD]
        MT[Motif Detector<br/>Regex/spaCy]
        ZC[Zero-Copy Layer]
    end

    subgraph "Decision Making"
        UP[Unified Policy<br/>Neural + Bandit]
        TS[Thompson Sampling]
        CE[Convergence Engine]
        SG[Safety Guardrails]
    end

    subgraph "Learning Systems"
        RB[Reflection Buffer]
        PL[Pattern Learner]
        AL[Adaptive Learning<br/>Phase 3]
        RL[Recursive Learning<br/>Phase 5]
    end

    subgraph "Input Adapters"
        SW[SpinningWheel]
        AS[Audio Spinner]
        YS[YouTube Spinner]
    end

    subgraph "External Integration"
        API[FastAPI Server]
        VSC[VS Code Extension]
        WEB[Web Dashboard]
        MCP[MCP Protocol]
    end

    subgraph "RAG System"
        SRAG[SimpleRAG]
        MRAG[MultimodalRAG]
        VQA[Visual Q&A]
    end

    %% Core connections
    WO --> CFG
    WO --> YG
    WO --> VC
    WO --> ME
    WO --> UP
    WO --> RB

    %% Memory connections
    YG --> SP
    YG --> AG
    VC --> ME
    PB --> ME
    CB --> ME

    %% Feature connections
    ME --> ZC
    ME --> SP
    MT --> UP
    SP --> UP

    %% Decision connections
    UP --> TS
    UP --> CE
    CE --> SG

    %% Learning connections
    RB --> PL
    PL --> AL
    AL --> RL
    RL -.Feedback.-> UP
    RL -.Feedback.-> YG

    %% Input connections
    SW --> AS
    SW --> YS
    AS --> WO
    YS --> WO

    %% External connections
    API --> WO
    VSC --> API
    WEB --> API
    MCP --> WO

    %% RAG connections
    SRAG --> WO
    MRAG --> SRAG
    MRAG --> VQA
    VQA --> PB

    style WO fill:#ff6b6b
    style CFG fill:#4ecdc4
    style YG fill:#95e1d3
    style ME fill:#f38181
    style UP fill:#aa96da
    style RB fill:#fcbad3
    style API fill:#ffffd2
    style SRAG fill:#a8e6cf
```

**Legend**:
- **Red**: Core orchestration
- **Cyan**: Configuration
- **Green**: Memory systems
- **Pink**: Feature extraction
- **Purple**: Decision making
- **Rose**: Learning systems
- **Yellow**: External integration
- **Light Green**: RAG system

---

## 3. Data Flow Architecture

Complete data transformation pipeline from raw input to final response.

```mermaid
graph TB
    subgraph "Layer 1: Raw Input"
        RI[Raw Input<br/>Text/Audio/Image]
    end

    subgraph "Layer 2: Spinning"
        SW[SpinningWheel]
        MS[MemoryShard<br/>Structured Data]
        RI --> SW
        SW --> MS
    end

    subgraph "Layer 3: Query Processing"
        QP[Query Parser]
        QC[Query Classifier<br/>Complexity Detection]
        MS --> QP
        QP --> QC
    end

    subgraph "Layer 4: Feature Extraction"
        MT[Motif Detection<br/>Symbolic]
        EMB[Embeddings<br/>96/192/384D]
        SPEC[Spectral<br/>Graph/SVD]
        QC --> MT
        QC --> EMB
        QC --> SPEC
    end

    subgraph "Layer 5: Feature Fusion"
        FUS[Feature Fusion<br/>DotPlasma]
        MT --> FUS
        EMB --> FUS
        SPEC --> FUS
    end

    subgraph "Layer 6: Context Retrieval"
        RET[Retrieval<br/>BM25 + Semantic]
        KG[Knowledge Graph<br/>Subgraph Expansion]
        FUS --> RET
        FUS --> KG
    end

    subgraph "Layer 7: Context Assembly"
        CTX[Context Assembly<br/>Memories + Features]
        RET --> CTX
        KG --> CTX
    end

    subgraph "Layer 8: Decision Making"
        POL[Policy Network<br/>Transformer]
        BAN[Thompson Sampling<br/>Exploration]
        CTX --> POL
        CTX --> BAN
        POL --> BAN
    end

    subgraph "Layer 9: Action Selection"
        ACT[Action/Tool Selection<br/>Argmax/Sample]
        BAN --> ACT
    end

    subgraph "Layer 10: Execution"
        EXE[Tool Execution]
        RES[Result]
        ACT --> EXE
        EXE --> RES
    end

    subgraph "Layer 11: Response Synthesis"
        SYN[Response Synthesis]
        RESP[Final Response]
        RES --> SYN
        CTX --> SYN
        SYN --> RESP
    end

    subgraph "Layer 12: Learning"
        REFL[Reflection]
        FEED[Feedback Learning]
        RESP --> REFL
        REFL --> FEED
        FEED -.Update.-> KG
        FEED -.Update.-> POL
        FEED -.Update.-> BAN
    end

    style RI fill:#e3f2fd
    style MS fill:#bbdefb
    style QC fill:#90caf9
    style FUS fill:#64b5f6
    style CTX fill:#42a5f5
    style ACT fill:#2196f3
    style RES fill:#1976d2
    style RESP fill:#0d47a1
    style FEED fill:#ff5722
```

**Data Dimensions at Each Layer**:
1. Raw Input: Variable (text/bytes)
2. MemoryShard: Structured dict
3. Query: {text, context, metadata}
4. Features: {motifs: List, embeddings: [96/192/384], spectral: [10-50]}
5. DotPlasma: Concatenated feature vector
6. Context: {memories: List[MemoryShard], features: Features}
7. Policy Input: [batch, seq_len, mem_dim]
8. Action Logits: [batch, n_tools]
9. Tool Selection: int (tool index)
10. Result: {output: str, metadata: dict}
11. Spacetime: {response: str, trace: WeavingTrace, confidence: float}
12. Learning Signal: {success: bool, reward: float, patterns: List}

---

## 4. Memory System Architecture

The 11 integrated memory systems in HoloLoom.

```mermaid
graph TB
    subgraph "Primary Storage"
        YG[Yarn Graph<br/>NetworkX MultiDiGraph<br/>Entity Relationships]
        VC[Vector Cache<br/>BM25 + Semantic<br/>Document Retrieval]
    end

    subgraph "Awareness Layer"
        AG[Awareness Graph<br/>Activation Tracking<br/>Coherence Metrics]
        SD[Spring Dynamics<br/>Node Activation<br/>Temporal Decay]
    end

    subgraph "Multimodal Memory"
        PT[Photo Tokens<br/>CLIP Embeddings<br/>Visual Similarity]
        OCR[OCR Integration<br/>DeepSeek/Tesseract<br/>Text Extraction]
    end

    subgraph "Performance Layer"
        CB[Compositional Cache<br/>Parse/Merge/Semantic<br/>50-300x Speedup]
        ZC[Zero-Copy Embeddings<br/>Memory-Mapped<br/>37x Faster]
        QC[Query Cache<br/>100x Repeated Queries]
    end

    subgraph "Learning Memory"
        RB[Reflection Buffer<br/>Episodic Memory<br/>Outcome Learning]
        PL[Pattern Memory<br/>Hot Patterns<br/>Usage Tracking]
    end

    subgraph "Multi-Wave Engine"
        MWE[Multi-Wave Retrieval<br/>Coarse → Fine<br/>Progressive Refinement]
    end

    %% Primary connections
    YG --> AG
    VC --> QC

    %% Awareness connections
    AG --> SD
    SD -.Decay.-> YG

    %% Multimodal connections
    PT --> OCR
    PT --> VC

    %% Performance connections
    CB --> VC
    ZC --> VC
    QC --> VC

    %% Learning connections
    RB --> PL
    PL -.Patterns.-> YG

    %% Multi-wave connections
    MWE --> YG
    MWE --> VC
    MWE --> CB

    style YG fill:#81c784
    style VC fill:#64b5f6
    style AG fill:#ba68c8
    style SD fill:#9575cd
    style PT fill:#f06292
    style CB fill:#ffd54f
    style ZC fill:#ffb74d
    style QC fill:#ff8a65
    style RB fill:#90a4ae
    style PL fill:#a1887f
    style MWE fill:#4db6ac
```

**Memory Characteristics**:

| System | Storage Type | Size | Access Time | Use Case |
|--------|-------------|------|-------------|----------|
| Yarn Graph | NetworkX | 10K-1M nodes | ~5ms | Entity relationships |
| Vector Cache | In-memory vectors | 100K docs | ~50ms | Semantic search |
| Awareness Graph | Activation map | Same as YG | <1ms | Activation tracking |
| Spring Dynamics | Timestamped activations | Same as YG | <1ms | Temporal decay |
| Photo Tokens | CLIP embeddings | 10K images | ~50ms | Visual similarity |
| Compositional Cache | LRU cache | 50K entries | <1ms | Parse reuse |
| Zero-Copy | Memory-mapped | 10K vectors | <0.1ms | Scale extraction |
| Query Cache | LRU cache | 1K queries | <0.01ms | Repeated queries |
| Reflection Buffer | Circular buffer | 1K episodes | ~1ms | Outcome learning |
| Pattern Memory | Hash map | 10K patterns | <1ms | Pattern matching |
| Multi-Wave | Virtual layer | N/A | +20ms | Progressive retrieval |

---

## 5. Learning Systems Timeline

The 7 learning systems organized by frequency and latency characteristics.

```mermaid
gantt
    title Learning Systems by Frequency & Overhead
    dateFormat X
    axisFormat %s

    section Per-Query (<5ms)
    Thompson Sampling (0.5ms)           :0, 1
    Policy Weight Updates (0.5ms)       :0, 1
    Heat Tracking (0.5ms)               :0, 1
    Provenance Logging (1ms)            :0, 1

    section Per-Success (~3ms)
    Pattern Extraction (1ms)            :active, 0, 1
    Reflection Storage (2ms)            :active, 0, 2

    section Hourly (50ms async)
    Pattern Mining (20ms)               :milestone, 0, 20
    Continuous Validation (20ms)        :milestone, 0, 20
    Pattern Deployment (10ms)           :milestone, 0, 10

    section Daily (100ms async)
    Performance Reports (50ms)          :crit, 0, 50
    Metrics Export (30ms)               :crit, 0, 30
    Alert Generation (20ms)             :crit, 0, 20

    section Weekly (offline)
    Model Retraining                    :done, 0, 1000
    Architecture Search                 :done, 0, 1000
```

**Learning System Details**:

```mermaid
graph TB
    subgraph "Real-Time Learning (Per Query)"
        TS[Thompson Sampling<br/>Bayesian Updates<br/>α,β per tool]
        PW[Policy Weights<br/>Laplace Smoothing<br/>Success rate tracking]
        HT[Heat Tracking<br/>Access frequency<br/>Exponential decay]
        PV[Provenance<br/>Scratchpad logging<br/>Audit trail]
    end

    subgraph "Success-Based Learning (High Confidence)"
        PE[Pattern Extraction<br/>Motif → Tool<br/>Quality scoring]
        RS[Reflection Storage<br/>Episode buffer<br/>Outcome learning]
    end

    subgraph "Background Learning (Hourly)"
        PM[Pattern Mining<br/>n-gram → regex<br/>Quality: 95%+ precision]
        CV[Continuous Validation<br/>Hourly checks<br/>Regression detection]
        PD[Pattern Deployment<br/>Shadow/A-B/Gradual<br/>Safe rollout]
    end

    subgraph "Monitoring (Daily)"
        PR[Performance Reports<br/>Daily/weekly<br/>Recommendations]
        ME[Metrics Export<br/>Prometheus<br/>Grafana dashboards]
        AG[Alert Generation<br/>Slack/email<br/>Regression alerts]
    end

    subgraph "Offline Learning (Weekly+)"
        MR[Model Retraining<br/>PPO updates<br/>New checkpoints]
        AS[Architecture Search<br/>Hyperparameter tuning<br/>Ablation studies]
    end

    %% Flow
    TS --> PE
    PW --> PE
    HT --> PE
    PV --> RS

    PE --> PM
    RS --> PM

    PM --> CV
    CV --> PD

    PD --> PR
    PR --> ME
    ME --> AG

    AG --> MR
    MR --> AS

    style TS fill:#4caf50
    style PW fill:#8bc34a
    style HT fill:#cddc39
    style PV fill:#ffeb3b
    style PE fill:#ffc107
    style RS fill:#ff9800
    style PM fill:#ff5722
    style CV fill:#f44336
    style PD fill:#e91e63
    style PR fill:#9c27b0
    style ME fill:#673ab7
    style AG fill:#3f51b5
    style MR fill:#2196f3
    style AS fill:#03a9f4
```

**Frequency Summary**:
- **Per-Query**: 4 systems, <5ms total
- **Per-Success**: 2 systems, ~3ms (10-20% of queries)
- **Hourly**: 3 systems, ~50ms (async, background)
- **Daily**: 3 systems, ~100ms (async, background)
- **Weekly**: 2 systems, offline (hours)

**Total Production Overhead**: ~5-8ms per query (real-time only)

---

## 6. Integration Points

External integration architecture showing all APIs and protocols.

```mermaid
graph TB
    subgraph "HoloLoom Core"
        WO[Weaving Orchestrator]
        SRAG[SimpleRAG]
        MRAG[MultimodalRAG]
        AL[Adaptive Learning]
    end

    subgraph "API Layer"
        FAPI[FastAPI Server<br/>Port 8000<br/>REST API]
        WS[WebSocket<br/>Real-time streaming]
        WORK[Workflow Executor<br/>Port 8001<br/>Visual workflows]
    end

    subgraph "VS Code Extension"
        SQ[Squad Extension<br/>TypeScript]
        HL[HoloLoom Bridge<br/>HTTP Client]
        UI[Sidebar UI<br/>Webview]
    end

    subgraph "Web Dashboard"
        WB[Workflow Builder<br/>Drag & Drop]
        VIS[Visualizations<br/>Tufte charts]
        MON[Monitoring<br/>Real-time metrics]
    end

    subgraph "MCP Protocol"
        MCP[Model Context Protocol<br/>Standardized AI API]
        TOOLS[Tool Definitions<br/>JSON Schema]
        PROM[Prompt Templates]
    end

    subgraph "LLM Integration"
        OLL[Ollama<br/>Local models]
        ANT[Anthropic<br/>Claude API]
        OAI[OpenAI<br/>GPT-4 API]
    end

    subgraph "External Services"
        NEO[Neo4j<br/>Graph DB<br/>Port 7687]
        QD[Qdrant<br/>Vector DB<br/>Port 6333]
        PROM2[Prometheus<br/>Metrics<br/>Port 9090]
        GRAF[Grafana<br/>Dashboards<br/>Port 3000]
    end

    %% Core to API
    WO --> FAPI
    SRAG --> FAPI
    MRAG --> FAPI
    AL --> FAPI

    FAPI --> WS
    WO --> WORK

    %% VS Code
    SQ --> HL
    HL --> FAPI
    HL --> WS
    UI --> HL

    %% Web Dashboard
    WB --> WORK
    VIS --> FAPI
    MON --> WS

    %% MCP
    MCP --> WO
    TOOLS --> MCP
    PROM --> MCP

    %% LLM
    WO --> OLL
    WO --> ANT
    WO --> OAI

    %% External
    WO --> NEO
    WO --> QD
    AL --> PROM2
    MON --> PROM2
    MON --> GRAF

    style WO fill:#ff6b6b
    style FAPI fill:#4ecdc4
    style SQ fill:#95e1d3
    style WB fill:#f38181
    style MCP fill:#aa96da
    style NEO fill:#fcbad3
    style QD fill:#ffffd2
```

**API Endpoints**:

### FastAPI Server (Port 8000)

```
POST   /query              # Main query endpoint
POST   /query/batch        # Batch queries
GET    /health             # Health check
GET    /stats              # System statistics
GET    /audit-trail        # Audit log
POST   /memory/store       # Store memory
GET    /memory/search      # Search memories
POST   /reflect            # Submit feedback
```

### Workflow Executor (Port 8001)

```
POST   /api/workflow/execute      # Execute workflow
POST   /api/workflow/validate     # Validate workflow
GET    /api/agents/list           # List available agents
WS     /ws/workflow/{id}          # Real-time workflow updates
```

### MCP Tools

```json
{
  "tools": [
    {
      "name": "hololoom_query",
      "description": "Query HoloLoom with agentic reasoning",
      "parameters": {
        "query": "string",
        "mode": "direct|verify|research|plan_execute",
        "max_steps": "integer"
      }
    },
    {
      "name": "hololoom_memory_search",
      "description": "Search knowledge graph",
      "parameters": {
        "query": "string",
        "k": "integer"
      }
    }
  ]
}
```

---

## 7. Phase Architectures

### Phase 3: Adaptive Learning System

```mermaid
graph TB
    subgraph "Production Traffic"
        Q[Queries]
        CLS[Adaptive Classifier]
        LOG[JSONL Logs]
        Q --> CLS
        CLS --> LOG
    end

    subgraph "Pattern Mining (Hourly)"
        PM[PatternMiner<br/>n-gram → regex<br/>Quality scoring]
        PAT[High-Quality Patterns<br/>Precision >95%<br/>Support >10]
        LOG --> PM
        PM --> PAT
    end

    subgraph "Continuous Validation (Hourly)"
        CV[ContinuousValidator<br/>Holdout validation<br/>Regression detection]
        REG[Regression Alerts<br/>Drop >2%<br/>Severity levels]
        PAT --> CV
        CV --> REG
    end

    subgraph "Adaptive Deployment (Days 1-7)"
        AU[AdaptiveUpdater<br/>Safe deployment<br/>Version control]
        SHAD[Day 1-2: SHADOW<br/>0% traffic<br/>Silent monitoring]
        AB[Day 3: A/B TEST<br/>10/90 split<br/>Statistical comparison]
        GRAD[Day 4-7: GRADUAL<br/>10%→50%→100%<br/>Progressive rollout]

        AU --> SHAD
        SHAD --> AB
        AB --> GRAD
    end

    subgraph "Monitoring (Daily)"
        PR[PerformanceReporter<br/>Daily/weekly reports<br/>Prometheus metrics]
        DASH[Grafana Dashboards<br/>Accuracy trends<br/>Pattern performance]
        ALERT[Slack/Email Alerts<br/>Critical regressions<br/>Recommendations]

        PR --> DASH
        PR --> ALERT
    end

    %% Flow
    CV --> AU
    AU -.Deploy.-> CLS

    GRAD --> PR
    REG --> ALERT

    %% Rollback
    REG -.Rollback.-> AU
    AU -.Revert.-> CLS

    style PM fill:#4caf50
    style CV fill:#2196f3
    style AU fill:#ff9800
    style PR fill:#9c27b0
    style SHAD fill:#81c784
    style AB fill:#64b5f6
    style GRAD fill:#ffb74d
    style REG fill:#f44336
```

**Phase 3 Components**:
- **PatternMiner**: 425 lines, ~500ms hourly
- **ContinuousValidator**: 469 lines, ~2-5s hourly
- **AdaptiveUpdater**: 682 lines, ~100ms deployment
- **PerformanceReporter**: 627 lines, ~50ms daily

**Total Overhead**: <1ms per query (logging only), ~3-6s hourly (async)

### Phase 5: Compositional Cache

```mermaid
graph TB
    subgraph "Input"
        TXT[Text Input<br/>"the big red ball"]
    end

    subgraph "Tier 1: Parse Cache (10-50x)"
        XB[X-bar Parser<br/>Universal Grammar]
        PC[Parse Cache<br/>LRU 10K entries]
        TREE[Parse Tree<br/>NP[Det[the] Adj[big] Adj[red] N[ball]]]

        TXT --> XB
        XB --> PC
        PC --> TREE
    end

    subgraph "Tier 2: Merge Cache (5-10x)"
        MC[Merge Cache<br/>Compositional reuse<br/>LRU 50K entries]
        PHRASES["Cached Phrases:<br/>ball (reused)<br/>red ball (reused)<br/>big red ball (new)"]

        TREE --> MC
        MC --> PHRASES
    end

    subgraph "Tier 3: Semantic Cache (3-10x)"
        SC[Semantic Cache<br/>244D projections<br/>LRU 10K entries]
        EMB[Cached Embeddings<br/>ball: [0.1, 0.3, ...]<br/>red ball: [0.2, 0.5, ...]<br/>big red ball: [0.15, 0.4, ...]]

        PHRASES --> SC
        SC --> EMB
    end

    subgraph "Compositional Reuse"
        NEW["New Query:<br/>a big red ball"]
        REUSE["Reuse from cache:<br/>big red ball (100% match)<br/>Zero computation!"]

        NEW -.Match.-> MC
        MC -.Hit.-> REUSE
    end

    subgraph "Performance"
        COLD[Cold Cache<br/>~150ms<br/>Full parsing]
        WARM[Warm Cache<br/>~0.5ms<br/>300x speedup!]
    end

    TXT --> COLD
    REUSE --> WARM

    style XB fill:#4caf50
    style PC fill:#8bc34a
    style MC fill:#ffc107
    style SC fill:#ff9800
    style REUSE fill:#2196f3
    style WARM fill:#00bcd4
```

**Phase 5 Components**:
- **X-bar Parser**: Principled phrase structure
- **Parse Cache**: 10K entries, <1ms lookup
- **Merge Cache**: 50K entries, compositional reuse
- **Semantic Cache**: 10K entries, 244D vectors

**Total Speedup**: 50-300x (multiplicative, hot paths)

---

## 8. Backend Architecture

Memory backend hierarchy with auto-fallback chain.

```mermaid
graph TB
    subgraph "Configuration"
        CFG[Config.memory_backend]
        INM[INMEMORY]
        HYB[HYBRID]
        HYP[HYPERSPACE]
    end

    subgraph "INMEMORY Backend"
        NX[NetworkX<br/>MultiDiGraph<br/>In-memory only]
        NX_VEC[Dict Vectors<br/>In-memory search]
        NX --> NX_VEC
    end

    subgraph "HYBRID Backend (Production)"
        NEO[Neo4j<br/>Graph Database<br/>Port 7687]
        QD[Qdrant<br/>Vector Database<br/>Port 6333]
        FALL[Auto-Fallback<br/>If unavailable]

        NEO --> FALL
        QD --> FALL
        FALL -.Fallback.-> NX
    end

    subgraph "HYPERSPACE Backend (Research)"
        GATE[Gated Multipass<br/>Advanced retrieval]
        WAVE[Multi-Wave Engine<br/>Coarse → Fine]
        SPECT[Spectral Features<br/>Graph Laplacian]

        GATE --> WAVE
        WAVE --> SPECT
        SPECT --> NEO
        SPECT --> QD
    end

    CFG --> INM
    CFG --> HYB
    CFG --> HYP

    INM --> NX
    HYB --> NEO
    HYB --> QD
    HYP --> GATE

    style NX fill:#4caf50
    style NEO fill:#2196f3
    style QD fill:#ff9800
    style GATE fill:#9c27b0
    style FALL fill:#f44336
```

**Backend Comparison**:

| Backend | Storage | Latency | Scalability | Use Case |
|---------|---------|---------|-------------|----------|
| **INMEMORY** | NetworkX + Dict | ~5ms | 10K nodes | Development, testing |
| **HYBRID** | Neo4j + Qdrant | ~20ms | 10M+ nodes | Production (recommended) |
| **HYPERSPACE** | Neo4j + Qdrant + Advanced | ~50ms | 10M+ nodes | Research, complex queries |

**Auto-Fallback Chain**:
```
HYBRID → Check Neo4j available → Yes: Use Neo4j
                                → No: Check Qdrant available → Yes: Use Qdrant
                                                              → No: Fallback to INMEMORY
```

**Docker Setup**:
```yaml
services:
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/password

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"  # HTTP
      - "6334:6334"  # gRPC
```

---

## 9. RAG System Architecture

Complete RAG system with multimodal capabilities.

```mermaid
graph TB
    subgraph "RAG API Layer"
        SRAG[SimpleRAG<br/>Level 2-4 RAG<br/>Zero-config]
        MRAG[MultimodalRAG<br/>Text + Images<br/>CLIP + OCR]
    end

    subgraph "Reasoning Modes"
        DIR[DIRECT<br/>Single-pass<br/>~150ms]
        VER[VERIFY<br/>Answer + verify<br/>~600ms]
        RES[RESEARCH<br/>Multi-query<br/>~900ms]
        PLAN[PLAN_EXECUTE<br/>Goal decomposition<br/>~750ms]
    end

    subgraph "Text Path"
        ING[ingest()<br/>Any modality]
        QRY[query()<br/>Agentic reasoning]
        BAT[batch_query()<br/>Efficient batch]

        ING --> QRY
        QRY --> BAT
    end

    subgraph "Visual Path"
        PHOTO[ingest_photo()<br/>CLIP encoding]
        VQA[query_with_image()<br/>OCR + CLIP]
        REL[get_related_photos()<br/>Similarity search]

        PHOTO --> VQA
        VQA --> REL
    end

    subgraph "HoloLoom Integration"
        WO[Weaving Orchestrator<br/>Full pipeline]
        MEM[Memory Systems<br/>Yarn Graph + Vector]
        EMB[Matryoshka Embeddings<br/>Multi-scale]

        WO --> MEM
        WO --> EMB
    end

    subgraph "LLM Integration"
        OLL[Ollama<br/>Local (default)]
        ANT[Anthropic<br/>Claude 3.5]
        OAI[OpenAI<br/>GPT-4]
    end

    subgraph "Performance"
        CACHE[Query Cache<br/>100x speedup]
        COMP[Visual Compression<br/>5-20x tokens]
        DASH[RAG Dashboard<br/>5 panels]
    end

    %% Connections
    SRAG --> DIR
    SRAG --> VER
    SRAG --> RES
    SRAG --> PLAN

    MRAG --> SRAG
    MRAG --> PHOTO
    MRAG --> VQA

    SRAG --> ING
    SRAG --> QRY

    MRAG --> REL

    QRY --> WO
    VQA --> WO

    WO --> OLL
    WO --> ANT
    WO --> OAI

    QRY --> CACHE
    VQA --> COMP

    SRAG --> DASH

    style SRAG fill:#4caf50
    style MRAG fill:#8bc34a
    style DIR fill:#2196f3
    style VER fill:#03a9f4
    style RES fill:#00bcd4
    style PLAN fill:#0097a7
    style WO fill:#ff6b6b
    style CACHE fill:#ffc107
    style COMP fill:#ff9800
```

**RAG Features**:

1. **Level 2: Hybrid RAG** - BM25 + semantic search
2. **Level 3: Graph RAG** - Yarn Graph entity relationships
3. **Level 4: Agentic RAG** - Multi-step reasoning (4 modes)
4. **Multimodal**: Text + images with CLIP + OCR
5. **Performance**: Query cache (100x), visual compression (5-20x)

**RAG Dashboard** (5 Panels):
```mermaid
graph LR
    subgraph "Panel 1"
        RQ[Retrieval Quality<br/>Sources per query<br/>Trend analysis]
    end

    subgraph "Panel 2"
        LW[Latency Waterfall<br/>Stage timing<br/>Bottleneck detection]
    end

    subgraph "Panel 3"
        CE[Cache Effectiveness<br/>Hit rate gauge<br/>Recommendations]
    end

    subgraph "Panel 4"
        CT[Confidence Trajectory<br/>Confidence over time<br/>Anomaly detection]
    end

    subgraph "Panel 5"
        SA[Source Attribution<br/>Knowledge graph<br/>Entity relationships]
    end

    style RQ fill:#4caf50
    style LW fill:#2196f3
    style CE fill:#ffc107
    style CT fill:#ff9800
    style SA fill:#9c27b0
```

---

## 10. Alignment Framework

Production-ready safety and monitoring system.

```mermaid
graph TB
    subgraph "Input Layer"
        ACT[Action Request<br/>Tool + Context]
    end

    subgraph "Module 1: Safety Guardrails (0.039ms)"
        SG[Safety Guardrails]
        RISK[Risk Assessment<br/>LOW/MEDIUM/HIGH/CRITICAL]
        ADV[Adversarial Detection<br/>Pattern matching]
        HIL[Human-in-Loop<br/>High-risk escalation]

        ACT --> SG
        SG --> RISK
        SG --> ADV
        RISK --> HIL
    end

    subgraph "Module 2: Deception Detection (0.034ms)"
        DD[Deception Detector]
        GOAL[Goal Transparency<br/>Declared vs actual]
        PROBE[Behavioral Probes<br/>Consistency checks]
        HIDDEN[Hidden Goal Detection<br/>Drift monitoring]

        ACT --> DD
        DD --> GOAL
        DD --> PROBE
        DD --> HIDDEN
    end

    subgraph "Module 3: Instrumental Convergence (0.015ms)"
        IC[IC Prevention]
        POWER[Power-Seeking<br/>Resource acquisition]
        SELF[Self-Preservation<br/>Override attempts]
        SCOPE[Scope Creep<br/>Unauthorized expansion]

        ACT --> IC
        IC --> POWER
        IC --> SELF
        IC --> SCOPE
    end

    subgraph "Module 4: Audit Trail (0.015ms)"
        AT[Audit Trail]
        PROV[Complete Provenance<br/>Decision lineage]
        SEARCH[Temporal Search<br/>Query by time/action]
        EXPORT[Export Logs<br/>Compliance/debugging]

        ACT --> AT
        AT --> PROV
        AT --> SEARCH
        AT --> EXPORT
    end

    subgraph "Decision Gate"
        GATE{All Checks<br/>Passed?}
        ALLOW[Execute Action]
        BLOCK[Block + Log]

        HIL --> GATE
        HIDDEN --> GATE
        SCOPE --> GATE

        GATE -->|Yes| ALLOW
        GATE -->|No| BLOCK
    end

    subgraph "Monitoring"
        MON[Live Monitoring<br/>Prometheus metrics]
        ALERT[Alerts<br/>Slack/Email]
        DASH[Grafana Dashboard<br/>Safety trends]

        BLOCK --> ALERT
        ALLOW --> MON
        MON --> DASH
    end

    ALLOW --> AT
    BLOCK --> AT

    style SG fill:#4caf50
    style DD fill:#2196f3
    style IC fill:#ff9800
    style AT fill:#9c27b0
    style GATE fill:#f44336
    style ALLOW fill:#8bc34a
    style BLOCK fill:#e53935
```

**Alignment Metrics** (Prometheus):
```
alignment_actions_total{status="allowed|blocked"}
alignment_risk_level{level="low|medium|high|critical"}
alignment_deception_score{threshold="0.8"}
alignment_power_seeking_detected{count}
alignment_audit_events_total{action_type}
```

**Performance Characteristics**:
- **Total Overhead**: 0.103 ms per query (29x faster than 3ms target)
- **Safety Guardrails**: 0.039 ms
- **Deception Detection**: 0.034 ms
- **IC Prevention**: 0.015 ms
- **Audit Trail**: 0.015 ms

**Test Coverage**: 46 functional tests + 13 performance benchmarks

---

## Summary: System Scale

**Code Statistics** (November 2025):

| Component | Files | Lines | Tests |
|-----------|-------|-------|-------|
| Core Orchestration | 8 | 4,665 | 24 |
| Memory Systems | 24 | 8,500 | 31 |
| Policy & Decision | 12 | 3,200 | 18 |
| RAG System | 11 | 6,811 | 45 |
| Alignment Framework | 8 | 4,200 | 59 |
| Adaptive Learning (Phase 3) | 6 | 3,148 | 13 |
| Recursive Learning (Phase 5) | 5 | 4,700 | 22 |
| Visualization | 10 | 5,400 | 39 |
| **TOTAL** | **84** | **40,624** | **251** |

**Performance Profile**:

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Simple query (LITE) | ~45ms | 22 QPS |
| Standard query (FAST) | ~150ms | 6.7 QPS |
| Complex query (FULL) | ~300ms | 3.3 QPS |
| Research query | ~900ms | 1.1 QPS |
| Repeated query (cached) | <1ms | 1000+ QPS |
| Alignment checks | 0.1ms | N/A (per action) |
| Learning overhead | 5-8ms | N/A (per query) |

**Production Deployment**:
- **Docker services**: Neo4j, Qdrant, Prometheus, Grafana
- **APIs**: FastAPI (8000), Workflow Executor (8001)
- **Monitoring**: Prometheus metrics, Grafana dashboards, Slack alerts
- **Safety**: Alignment framework with <0.11ms overhead
- **Learning**: Background learning with <8ms per-query overhead

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-15
**Maintained By**: HoloLoom Team
