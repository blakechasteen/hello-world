# HoloLoom: Comprehensive System Summary

**Generated**: 2025-11-03
**Repository**: c:\Users\blake\OneDrive\Documents\mythRL
**Status**: Production Ready (v1.0+)

---

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [Core Architecture](#core-architecture)
3. [Main Entry Points](#main-entry-points)
4. [Memory Systems](#memory-systems)
5. [Learning & Intelligence](#learning--intelligence)
6. [Input/Output Systems](#inputoutput-systems)
7. [Integration & Deployment](#integration--deployment)
8. [Test Infrastructure](#test-infrastructure)
9. [Performance Characteristics](#performance-characteristics)
10. [Quick Start Guide](#quick-start-guide)

---

## Executive Overview

**HoloLoom** is a production-grade neural decision-making and memory system built on a sophisticated **weaving metaphor**. It implements a canonical 9-step processing cycle that transforms queries into intelligent responses with complete computational provenance.

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Code** | 302+ Python files, 100,000+ lines |
| **Execution Modes** | 3 (LITE/FAST/FULL/RESEARCH) |
| **Architecture Layers** | 9-layer weaving cycle |
| **Performance** | 10-300× speedups via compositional caching |
| **Test Coverage** | 387+ tests, ~40% coverage |
| **Input Adapters** | 28+ spinners (YouTube, PDF, audio, etc.) |
| **External Integrations** | FastAPI, MCP, VS Code, Docker |
| **Design Philosophy** | "Reliable Systems: Safety First" |

### Core Innovation

HoloLoom's breakthrough is implementing a complete **weaving metaphor** as first-class abstractions:
- **Yarn Graph**: Discrete symbolic memory (entities + relationships)
- **Warp Space**: Continuous tensor field for computation
- **Spacetime Fabric**: 4D output (3D semantic + 1D temporal trace)
- **Complete Provenance**: Every decision fully traceable

---

## Core Architecture

### The 9-Step Weaving Cycle

```
┌─────────────────────────────────────────────────────────────┐
│              THE 9-STEP WEAVING CYCLE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Loom Command      → Pattern Card selection (BARE/FAST/FUSED)
│  2. Chrono Trigger    → Temporal window creation
│  3. Yarn Graph        → Thread selection from memory
│  4. Resonance Shed    → Feature extraction, DotPlasma creation
│  5. Warp Space        → Continuous manifold tensioning
│  6. Convergence Engine → Discrete decision collapse
│  7. Tool Execution    → Action with results
│  8. Spacetime Fabric  → Provenance and complete trace
│  9. Reflection Buffer → Learning from outcome
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Weaving Metaphor Components

#### 1. Yarn Graph (Discrete Memory)
- **Implementation**: NetworkX MultiDiGraph
- **Location**: [HoloLoom/memory/graph.py](HoloLoom/memory/graph.py)
- **Purpose**: Persistent symbolic memory as discrete thread structure
- **Features**:
  - Typed edges (IS_A, USES, MENTIONS, LEADS_TO, PART_OF, etc.)
  - Entity and relationship storage
  - Subgraph extraction for context
  - Path finding between concepts

#### 2. Loom Command (Pattern Selection)
- **Implementation**: PatternCard selector
- **Location**: [HoloLoom/loom/command.py](HoloLoom/loom/command.py)
- **Purpose**: Choose execution template (BARE/FAST/FUSED)
- **Features**:
  - Configures scales, features, timeouts
  - Determines which warp threads to lift
  - Quality vs speed tradeoffs

#### 3. Chrono Trigger (Temporal Control)
- **Implementation**: Temporal windows and execution limits
- **Location**: [HoloLoom/chrono/trigger.py](HoloLoom/chrono/trigger.py)
- **Purpose**: Manage all time-dependent aspects
- **Features**:
  - Recency weighting
  - Episode filtering
  - Thread decay over time
  - Execution timeouts and halt conditions

#### 4. Resonance Shed (Feature Extraction)
- **Implementation**: Multi-modal feature fusion
- **Location**: [HoloLoom/resonance/shed.py](HoloLoom/resonance/shed.py)
- **Purpose**: Combine multiple extraction modalities
- **Features**:
  - Motif detection (symbolic patterns)
  - Embeddings (semantic vectors)
  - Spectral features (graph topology)
  - Creates DotPlasma (flowing feature representation)

#### 5. DotPlasma (Feature Fluid)
- **Alias**: `DotPlasma = Features`
- **Location**: [HoloLoom/documentation/types.py](HoloLoom/documentation/types.py)
- **Purpose**: Malleable medium between extraction and decision
- **Contains**:
  - Embeddings (continuous vectors)
  - Motifs (symbolic patterns)
  - Spectral features (topological signals)

#### 6. Warp Space (Continuous Manifold)
- **Implementation**: Tensioned tensor field
- **Location**: [HoloLoom/warp/space.py](HoloLoom/warp/space.py)
- **Purpose**: Temporary manifold for tensor operations
- **Lifecycle**: tension() → compute() → collapse()
- **Features**:
  - Multi-scale embedding operations
  - Spectral computation
  - Context expansion
  - Detensions back to Yarn Graph after computation

#### 7. Convergence Engine (Decision Collapse)
- **Implementation**: Thompson Sampling + Neural Network
- **Location**: [HoloLoom/convergence/engine.py](HoloLoom/convergence/engine.py)
- **Purpose**: Collapse continuous → discrete decisions
- **Strategies**:
  - ARGMAX: Pure exploitation
  - EPSILON_GREEDY: 90% exploitation, 10% exploration
  - BAYESIAN_BLEND: 70% neural, 30% bandit priors
  - PURE_THOMPSON: 100% Thompson Sampling

#### 8. Spacetime Fabric (Woven Output)
- **Implementation**: 4D structured output
- **Location**: [HoloLoom/fabric/spacetime.py](HoloLoom/fabric/spacetime.py)
- **Purpose**: Complete computational lineage
- **Dimensions**:
  - 3D: Semantic space positioning
  - 1D: Temporal trace (chronological provenance)
- **Features**:
  - Full WeavingTrace with stage timings
  - Decision metadata
  - Memory access logs
  - Serializable for persistence

#### 9. Reflection Buffer (Learning Loop)
- **Implementation**: Episodic memory of outcomes
- **Location**: [HoloLoom/memory/cache.py](HoloLoom/memory/cache.py)
- **Purpose**: Continuous system improvement
- **Features**:
  - Pattern extraction from successful queries
  - Thompson Sampling updates
  - Policy weight adaptation
  - ReflectionMetrics tracking

---

## Main Entry Points

### 1. Simple API: HoloLoom (10/10 Layer)

**Best for**: Most users - single unified interface

**Location**: [HoloLoom/hololoom.py](HoloLoom/hololoom.py) (471 lines)

```python
from HoloLoom import HoloLoom

# Initialize
async with HoloLoom() as loom:
    # Three core operations
    mem = await loom.experience("Thompson Sampling balances exploration")
    memories = await loom.recall("What did I learn about sampling?")
    await loom.reflect(memories, feedback={"helpful": True})

    # Get metrics
    metrics = loom.get_metrics()
    print(f"Active memories: {metrics['activation']['active_nodes']}")
```

**Features**:
- Single entry point - everything is a memory operation
- AwarenessGraph for memory activation tracking
- MatryoshkaEmbeddings for semantic encoding
- Optional InputRouter for multimodal support
- Graceful degradation when dependencies unavailable

---

### 2. Complete Control: WeavingOrchestrator

**Best for**: Advanced users - full pipeline control

**Location**: [HoloLoom/weaving_orchestrator.py](HoloLoom/weaving_orchestrator.py) (1,963 lines)

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query

config = Config.fused()
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    # Execute complete 9-step cycle
    spacetime = await orchestrator.weave(Query(text="Your question"))
    print(spacetime.response)
    print(spacetime.trace)  # Full computational lineage
```

**Capabilities**:
- Implements complete 9-step cycle
- mythRL protocol integration (3-5-7-9 system)
- Alignment framework support
- Full provenance tracing
- Lifecycle management via async context managers

**mythRL Progressive Complexity**:
- **LITE (3 steps)**: Extract → Route → Execute (<50ms)
- **FAST (5 steps)**: + Pattern Selection + Temporal Windows (<150ms)
- **FULL (7 steps)**: + Decision Engine + Synthesis Bridge (<300ms)
- **RESEARCH (9 steps)**: + Advanced WarpSpace + Full Tracing (no limit)

---

### 3. Unified API

**Best for**: Applications - higher-level operations

**Location**: [HoloLoom/unified_api.py](HoloLoom/unified_api.py) (729 lines)

```python
from HoloLoom.unified_api import HoloLoom

# Create with options
loom = await HoloLoom.create(
    pattern="fast",           # BARE, FAST, FUSED
    memory_backend="simple",
    enable_synthesis=True
)

# Query
response = await loom.query("Your question")

# Chat (conversational)
response = await loom.chat("Follow-up question")

# Ingest data
await loom.ingest_text("Knowledge base content")
await loom.ingest_youtube("VIDEO_ID")
```

---

### 4. Interactive Terminal UI

**Best for**: Development and testing

**Location**: [HoloLoom/terminal_ui.py](HoloLoom/terminal_ui.py) (751 lines)

```python
from HoloLoom.terminal_ui import TerminalUI

# Start interactive session
ui = TerminalUI()
await ui.run()
```

**Features**:
- Rich terminal formatting (colors, tables, progress bars)
- Real-time pipeline visualization
- 9-step trace display
- Interactive pattern selection
- Conversation history with awareness context

---

## Memory Systems

### Three-Tier Backend Architecture

#### 1. INMEMORY (Development)
- **Implementation**: NetworkX MultiDiGraph in-memory
- **Speed**: <10ms all operations
- **Persistence**: None (data lost on exit)
- **Dependencies**: Zero (always works)
- **Best for**: Development, testing, quick iteration

#### 2. HYBRID (Production - Recommended)
- **Architecture**: Neo4j (graph) + Qdrant (vectors)
- **Speed**: ~50ms for full retrieval
- **Persistence**: Docker-backed
- **Auto-Fallback**: Intelligent degradation chain
- **Best for**: Production deployment with reliability

**Fallback Chain (Never Crashes)**:
```
Neo4j + Qdrant (production)
    ↓ (if Neo4j fails)
Neo4j only (degraded)
    ↓ (if Qdrant fails)
Qdrant only (degraded)
    ↓ (if both fail)
NetworkX (emergency fallback - system continues working)
```

#### 3. HYPERSPACE (Research)
- **Algorithm**: Recursive gated multipass memory crawling
- **Complexity**: 4 progressive passes with thresholds (0.6 → 0.75 → 0.85 → 0.9)
- **Speed**: ~150ms total including graph traversal
- **Features**: Matryoshka importance gating
- **Best for**: Research mode with maximum capability

### Knowledge Graph (YarnGraph)

**Edge Types** (Semantic relationships):
- `IS_A` - Taxonomy
- `USES` - Functional
- `MENTIONS` - Reference
- `LEADS_TO` - Causal
- `PART_OF` - Composition
- `IN_TIME` - Temporal
- `OCCURRED_AT` - Event location

**Key Operations**:
- `add_edge()` - Add relationship
- `get_neighbors()` - Find adjacent (1+ hops)
- `subgraph_for_entities()` - Extract context
- `get_paths()` - Find reasoning paths
- `connect_entity_to_time()` - Temporal threading

### Configuration

```python
from HoloLoom.config import Config, MemoryBackend

config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID  # Auto-falls back to INMEMORY

# Create persistent backend
memory = await create_memory_backend(config)

async with WeavingOrchestrator(cfg=config, memory=memory) as orchestrator:
    spacetime = await orchestrator.weave(query)
    # Data persists across sessions (if Neo4j/Qdrant available)
```

---

## Learning & Intelligence

### Recursive Learning System (6 Phases)

**Philosophy**: *"Great answers aren't written, they're refined."*

**Location**: [HoloLoom/recursive/](HoloLoom/recursive/)

#### Phase 1: Scratchpad Integration (Provenance Tracking)
**File**: [scratchpad_integration.py](HoloLoom/recursive/scratchpad_integration.py) (717 lines)

**Purpose**: Record complete reasoning history for every query

**Components**:
- **ProvenanceTracker**: Extracts WeavingTrace → ScratchpadEntry
- **ScratchpadOrchestrator**: Automatic provenance logging
- **RecursiveRefiner**: Auto-refine low-confidence results (<0.75)

**Maps**: Thought → Action → Observation → Score (TAOS pattern)

#### Phase 2: Loop Engine Integration (Pattern Learning)
**File**: [loop_integration.py](HoloLoom/recursive/loop_integration.py) (661 lines)

**Purpose**: Learn patterns from successful queries

**Components**:
- **LearnedPattern**: Captures successful reasoning paths
- **PatternExtractor**: Extracts patterns from high-confidence results (≥0.75)
- **PatternLearner**: Maintains pattern library with auto-pruning
- **LearningLoopEngine**: Auto-learns from every query

**Query Classification**: Factual, procedural, analytical, comparative, exploratory, general

#### Phase 3: Hot Pattern Feedback (Usage-Based Adaptation)
**File**: [hot_patterns.py](HoloLoom/recursive/hot_patterns.py) (450+ lines)

**Purpose**: Track access frequency and adapt retrieval

**Heat Score Algorithm**:
```
heat = access_count × success_rate × avg_confidence × (0.95 ^ hours_since_last_access)
```

**Components**:
- **UsageRecord**: Tracks access patterns
- **HotPatternTracker**: Identifies frequently accessed knowledge
- **AdaptiveRetriever**: Adjusts weights (hot patterns get 2× boost, cold get 0.5× penalty)
- **HotPatternFeedbackEngine**: Full integration

#### Phase 4: Advanced Refinement (Multi-Strategy Quality)
**File**: [advanced_refinement.py](HoloLoom/recursive/advanced_refinement.py) (380+ lines)

**Purpose**: Apply sophisticated refinement strategies

**Refinement Strategies**:
- **REFINE**: Iterative expansion
- **CRITIQUE**: Self-improvement
- **VERIFY**: Multi-source check (3 passes)
- **ELEGANCE**: Clarity → Simplicity → Beauty (3 passes)
- **HOFSTADTER**: Recursive self-reference

**Quality Metrics**: 0.7 × confidence + 0.2 × context_richness + 0.1 × completeness

#### Phase 5: Full Learning Loop (Background Learning)
**File**: [full_learning_loop.py](HoloLoom/recursive/full_learning_loop.py) (626 lines)

**Purpose**: Continuous background learning with Thompson Sampling

**Components**:
- **ThompsonPriors**: Beta distributions for each tool
- **PolicyWeights**: Learned adapter weights
- **BackgroundLearner**: Async learning (every 60s)
- **FullLearningEngine**: Complete integration

**Update Rules**:
```
Success (confidence ≥ 0.75): α ← α + confidence
Failure (confidence < 0.75): β ← β + (1 - confidence)
Expected reward: E[X] = α / (α + β)
```

**Usage**:
```python
from HoloLoom.recursive import FullLearningEngine

async with FullLearningEngine(
    cfg=config,
    shards=shards,
    enable_background_learning=True
) as engine:
    spacetime = await engine.weave(query, enable_refinement=True)
    stats = engine.get_learning_statistics()
```

### Alignment Framework (Safety & Transparency)

**Philosophy**: *"Safe by default, transparent by design"*

**Location**: [HoloLoom/alignment/](HoloLoom/alignment/)
**Status**: Production ready (v1.0.0, November 2025)
**Performance**: 0.103 ms overhead (29× faster than 3ms target)

#### Component 1: Safety Guardrails
**File**: [safety_guardrails.py](HoloLoom/alignment/safety_guardrails.py) (526 lines)

**Risk Levels**: SAFE, LOW, MEDIUM, HIGH, CRITICAL

**Features**:
- Risk-based action gating
- Adversarial pattern detection
- Human-in-the-loop escalation for high-risk actions
- Configurable policies by environment

#### Component 2: Deception Detection
**File**: [deception_detection.py](HoloLoom/alignment/deception_detection.py) (~300 lines)

**Probe Types**: CONSISTENCY, CAPABILITY, GOAL_ALIGNMENT, REWARD_HACKING, HONESTY

**Features**:
- Behavioral consistency checks
- Goal-action alignment analysis
- Hidden capability detection
- Reward hacking prevention

#### Component 3: Instrumental Convergence Prevention
**File**: [instrumental_convergence.py](HoloLoom/alignment/instrumental_convergence.py) (~400 lines)

**Detects**:
- Power-seeking behavior
- Resource acquisition patterns
- Self-preservation behaviors

**Resource Bounds**: CPU, Memory, Storage, API calls, Network, Tool executions

#### Component 4: Audit Trail
**File**: [audit_trail.py](HoloLoom/alignment/audit_trail.py) (~400 lines)

**Features**:
- Complete decision provenance
- Queryable history (by type, time range, risk level)
- JSON serialization/deserialization
- Integration with monitoring systems

**Usage**:
```python
from HoloLoom.alignment import SafetyGuardrails, AuditTrail

guardrails = SafetyGuardrails(enable_human_in_loop=True)
audit_trail = AuditTrail()

# Gate action
gate_result = await guardrails.gate_action(action, context)

if gate_result.allowed:
    spacetime = await orchestrator.weave(query)
    await audit_trail.log_decision(
        query=query.text,
        action=action,
        outcome="success",
        safety_score=gate_result.safety_score
    )
```

### Agentic Reasoning System

**Location**: [HoloLoom/agentic/core.py](HoloLoom/agentic/core.py)

#### 4 Reasoning Modes

| Mode | Purpose | Latency | Queries |
|------|---------|---------|---------|
| **DIRECT** | Single-pass answer | ~150ms | 1 |
| **VERIFY** | Answer + verification | ~600ms | 3-5 |
| **RESEARCH** | Multi-query exploration | ~900ms | 5+ |
| **PLAN_EXECUTE** | Goal decomposition | ~750ms | 3-7 |

**Usage**:
```python
from HoloLoom.agentic import AgenticOrchestrator, ReasoningMode

async with AgenticOrchestrator(cfg=config, shards=shards) as orchestrator:
    result = await orchestrator.reason(
        query="What are the tradeoffs of Thompson Sampling?",
        mode=ReasoningMode.RESEARCH,
        max_steps=5
    )

    print(result.response)
    print(result.confidence)
    print(result.steps_taken)
    print(result.verification)  # If mode=VERIFY
```

---

## Input/Output Systems

### SpinningWheel Input Adapters (28+ Spinners)

**Location**: [HoloLoom/spinningWheel/](HoloLoom/spinningWheel/)

**Philosophy**: *"If you need to configure it, we failed."* - Ruthlessly elegant API

#### Primary API
```python
from HoloLoom.spinningWheel import spin, spin_batch

# Ingest ANYTHING into memory
memory = await spin("text string")
memory = await spin("https://example.com")
memory = await spin("/path/to/file.pdf")
memory = await spin({"json": "data"})
memory = await spin([text, image, audio])  # Multi-modal
```

#### Available Spinners

**Modality-Based**:
1. **TextSpinner** - Raw text, documents
2. **AudioSpinner** - Audio with transcription
3. **ImageSpinner** - OCR, receipt extraction
4. **CodeSpinner** - Code with syntax analysis
5. **WebsiteSpinner** - Recursive crawling

**Format-Specific**:
6. **YouTubeSpinner** - Video transcripts with timecodes
7. **SpreadsheetSpinner** - Excel, CSV, Google Sheets
8. **PDFSpinner** - Documents with table extraction
9. **CodebaseSpinner** - Repository analysis (AST parsing)
10. **GitSpinner** - Repository history with importance scoring
11. **EmailSpinner** - IMAP/mbox with threading
12. **WhisperSpinner** - Audio transcription
13. **URLSpinner** - Web content extraction
14. **MatrixSpinner** - Matrix/Element chat exports
15. **ChatHistorySpinner** - Discord, Slack, Telegram

**Specialized**:
- BrowserHistorySpinner
- RecursiveCrawler
- GroceryReceiptSpinner
- MultiModalSpinner (auto-detection)

#### Importance Scoring (9 Signals)

Each spinner scores data by:
- **Length Signal** (0.15) - Longer = more substantive
- **Technical Signal** (0.20) - Domain-specific terms
- **Structural Signal** (0.10) - Well-formatted
- **Authority Signal** (0.20) - Source credibility
- **Recency Signal** (0.10) - Time decay
- **Engagement Signal** (0.15) - Reactions, replies
- **Reference Signal** (0.10) - Citations, backlinks
- **Noise Penalty** (-1.0 to 0.0) - Spam, duplicates
- **Custom Signals** - Spinner-specific

Final importance = weighted sum, clamped to [0.0, 1.0]

### Tufte-Style Visualization System

**Location**: [HoloLoom/visualization/](HoloLoom/visualization/)

**Philosophy**: *"Above all else show the data"* - Edward Tufte principles

#### Primary API
```python
from HoloLoom.visualization import auto, render, save

# Perfect dashboard from any data
dashboard = auto(spacetime_result)
html = render(dashboard)
save(dashboard, "dashboard.html")
```

#### 7 Core Visualizations

1. **Small Multiples** - Query comparison
   - Enable side-by-side comparison with consistent scales
   - Highlight best/worst performers
   - Inline sparklines for trends

2. **Stage Waterfall** - Pipeline timing
   - Sequential pipeline visualization
   - Automatic bottleneck detection (>40% of total)
   - Status indicators (SUCCESS, WARNING, ERROR)

3. **Confidence Trajectory** - Time series tracking
   - Confidence over query sequences
   - Cache hit/miss markers
   - Automatic anomaly detection (4 types)

4. **Cache Effectiveness Gauge** - Performance metrics
   - Radial gauge visualization
   - 5 effectiveness ratings (EXCELLENT → CRITICAL)
   - Actionable recommendations

5. **Data Density Table** - High information density
   - Maximum info per square inch
   - Inline sparklines in cells
   - ~60-70% data-ink ratio

6. **Knowledge Graph Network** - Force-directed graph
   - Fruchterman-Reingold layout
   - 7 semantic edge types with colors
   - Path highlighting for reasoning chains

7. **Semantic Space** - 3D projection
   - t-SNE or UMAP of 244D space
   - Query trajectory visualization
   - Memory distribution

#### Dashboard Auto-Construction

**DashboardConstructor**:
- Analyzes data type and complexity
- Auto-selects optimal visualizations
- Arranges panels (METRIC/FLOW/RESEARCH/ADAPTIVE layouts)
- Adds insights and annotations
- Handles responsive sizing

**Panel Types** (13): METRIC, TIMELINE, TRAJECTORY, NETWORK, HEATMAP, DISTRIBUTION, TEXT, SCATTER, LINE, BAR, INSIGHT, SPARKLINE, WATERFALL

**Panel Sizes**: TINY, COMPACT, SMALL, MEDIUM, LARGE, FULL_WIDTH, HERO

---

## Integration & Deployment

### FastAPI Server

**Location**: [HoloLoom/server/agentic_api.py](HoloLoom/server/agentic_api.py) (531 lines)

**Main Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/query` | POST | Main agentic reasoning |
| `/stats` | GET | Server statistics |
| `/audit-trail` | GET | Retrieve audit logs |
| `/memories/add` | POST | Store new memories |

**Startup**:
```bash
# Development
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --reload --port 8000

# Production
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### MCP (Model Context Protocol) Servers

#### A. Standard Memory Server
**File**: [HoloLoom/memory/mcp_server.py](HoloLoom/memory/mcp_server.py) (600+ lines)

**Tools**:
- `experience(content)` - Form memories
- `recall(query, limit)` - Retrieve memories
- `reflect(memories, feedback)` - Learn
- `search(query, k)` - Keyword/semantic search
- `chat(message)` - Conversational interface

**Conversation Intelligence**:
- Automatic signal vs noise filtering
- Importance scoring (0.0-1.0)
- Auto-spin important turns to persistent memory

#### B. RAG Server
**File**: [HoloLoom/memory/mcp_rag_server.py](HoloLoom/memory/mcp_rag_server.py) (753 lines)

**Pipeline**:
1. Semantic routing (classify intent)
2. HyDE rewriting (hypothetical documents)
3. Semantic chunking (boundary-based)
4. Hybrid retrieval (dense + sparse fusion)
5. Re-ranking (cross-encoder)

**Tools**:
- `rag_query` - Full RAG pipeline
- `ingest_document` - Semantic chunking + storage
- `rewrite_query` - HyDE expansion
- `hybrid_search` - Dense + sparse with weights

**Usage**:
```bash
# Start MCP server
python -m HoloLoom.memory.mcp_server

# Configure in Claude Desktop
{
  "mcpServers": {
    "hololoom": {
      "command": "python",
      "args": ["-m", "HoloLoom.memory.mcp_server"]
    }
  }
}
```

### VS Code Extension (Squad)

**Location**: [squad/src/](squad/src/)

**Technology**: TypeScript + Axios HTTP client

**Components**:
- **HoloLoomBridge** ([HoloLoomBridge.ts](squad/src/HoloLoomBridge.ts), 144 lines) - Core communication
- **Extension Commands** - VS Code integration
- **AgentPanel** - UI panel for chat
- **CodeContextProvider** - Extract editor context

**Usage**:
```typescript
const bridge = new HoloLoomBridge('http://localhost:8000');

const result = await bridge.query(
  "Explain this TypeScript code",
  codeContext,
  'verify',
  5
);

console.log(result.response);
console.log(result.verification.verified);
```

### Docker Deployment

#### Development Docker Compose
**File**: [docker-compose.yml](docker-compose.yml) (59 lines)

**Services**:
- **Neo4j (5.15.0)**: Graph database (ports 7474, 7687)
- **Qdrant (1.7.4)**: Vector database (ports 6333, 6334)

#### Production Docker Compose
**File**: [docker-compose.production.yml](docker-compose.production.yml) (138 lines)

**Additional Services**:
- **HoloLoom Application** (port 8000)
- **Prometheus** (port 9090, optional)
- **Grafana** (port 3000, optional)

**Startup**:
```bash
# Development
docker-compose up -d

# Production with monitoring
docker-compose -f docker-compose.production.yml --profile monitoring up -d

# Check health
curl http://localhost:8000/health
```

---

## Test Infrastructure

### Test Statistics

| Metric | Value |
|--------|-------|
| **Total Test Files** | 121 |
| **Unit Tests** | 27 files, 657 functions |
| **Integration Tests** | 77 files, 77+ classes |
| **E2E Tests** | 10 files, 143 functions |
| **Alignment Tests** | 4 files, 60 functions |
| **Demo Scripts** | 138 executable |
| **Total Test Code** | ~16,000+ lines |
| **Status** | 387+ tests passing ✅ |
| **Coverage** | ~40% |

### Test Organization

#### Unit Tests (<500ms budget)
**Location**: [HoloLoom/tests/unit/](HoloLoom/tests/unit/)

**Key Tests**:
- `test_unified_policy.py` - 60+ assertions (Thompson Sampling, PPO)
- `test_embedding_spectral.py` - Matryoshka embeddings
- `test_memory_graph.py` - 80+ assertions (NetworkX)
- `test_memory_cache.py` - 70+ assertions (BM25 + semantic)

#### Integration Tests (<2s budget)
**Location**: [HoloLoom/tests/integration/](HoloLoom/tests/integration/)

**Categories**:
- Backend integration (6 files)
- System integration (15+ files)
- Feature integration (20+ files)
- Input adapters (8+ files)
- Visualization (8+ files)

#### E2E Tests (<30s budget)
**Location**: [HoloLoom/tests/e2e/](HoloLoom/tests/e2e/)

**Key Tests**:
- `test_full_pipeline.py` - BARE/FAST/FUSED modes
- `test_concurrent_queries.py` - 100 parallel queries
- `test_performance_profile.py` - Latency/memory benchmarks
- `test_memory_growth.py` - 500-query leak detection
- `test_edge_cases.py` - Unicode, 50K chars, emoji

#### Alignment Tests
**Location**: [HoloLoom/alignment/tests/](HoloLoom/alignment/tests/)

**Files**:
- `test_alignment.py` - 46 functional tests
- `test_performance.py` - 13 benchmarks (<3ms overhead ✅ 0.103ms actual)

### Testing Philosophy

All tests validate **"Reliable Systems: Safety First"**:
1. Graceful degradation (20 tests)
2. Thread safety (20 tests)
3. Timeout protection (all tests)
4. Complete provenance (all tests)
5. Performance budgets (enforced)

---

## Performance Characteristics

### Latency by Mode

| Mode | Typical Latency | Features | Use Case |
|------|-----------------|----------|----------|
| **BARE** | 45-60ms | Regex motifs, single scale | Real-time, latency-critical |
| **FAST** | 100-200ms | Hybrid motifs, spectral | Standard queries |
| **FUSED** | 200-500ms | Full features, multi-scale | Complex, high-quality |
| **RESEARCH** | 500ms+ | All features, full tracing | Analysis, debugging |

### Phase 5 Speedups (Compositional Caching)

- **Compositional Cache Only**: 10-50× (parse cache hits)
- **Merge Cache**: 5-10× (compositional reuse)
- **Semantic Cache**: 3-10× (embedding caching)
- **Combined (Hot Path)**: 50-300× multiplicative
- **Production (90-99% hit rate)**: 10-17× typical

### Memory Backend Performance

| Operation | INMEMORY | HYBRID | HYPERSPACE |
|-----------|----------|--------|------------|
| store() | <1ms | 5ms | 10ms |
| recall(10) | 10ms | 20ms | 30ms |
| recall(50) | 20ms | 50ms | 150ms |
| subgraph | 5ms | 15ms | 25ms |

### Per-Query Overhead

| Component | Latency | When |
|-----------|---------|------|
| Base Orchestrator | 100-150ms | Always |
| Phase 1 (Scratchpad) | <1ms | Every query |
| Phase 2 (Pattern learning) | <1ms | High-confidence only |
| Phase 3 (Hot tracking) | <0.5ms | Every query |
| Phase 4 (Refinement) | ~150ms | Low-confidence only (10-20%) |
| Phase 5 (Background learning) | <3ms | Async (every 60s) |
| Alignment framework | <0.1ms | Every query |

**Typical Production**: 85% of queries: 100-155ms (base + <3% overhead)

---

## Quick Start Guide

### Installation

```bash
# Clone repository
git clone <repository-url>
cd mythRL

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy gymnasium matplotlib

# Optional for full features
pip install spacy sentence-transformers scipy networkx ollama
python -m spacy download en_core_web_sm
```

### Quick Examples

#### 1. Simple Memory Operations
```python
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    # Form memories
    await loom.experience("Thompson Sampling balances exploration and exploitation")

    # Recall memories
    memories = await loom.recall("What did I learn about sampling?")

    # Reflect on feedback
    await loom.reflect(memories, feedback={"helpful": True})
```

#### 2. Full Weaving Cycle
```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query

config = Config.fast()
shards = create_test_shards()

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="Explain Thompson Sampling"))
    print(spacetime.response)
    print(spacetime.confidence)
    print(spacetime.trace)  # Complete provenance
```

#### 3. Ingest YouTube Video
```python
from HoloLoom.spinningWheel import spin

# Automatic transcription and chunking
memory = await spin("https://youtube.com/watch?v=VIDEO_ID")
```

#### 4. Start FastAPI Server
```bash
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --reload --port 8000
```

#### 5. Start Docker Services
```bash
docker-compose up -d
```

### Configuration Modes

```python
from HoloLoom.config import Config

# Fastest (minimal processing)
cfg_bare = Config.bare()

# Balanced (recommended)
cfg_fast = Config.fast()

# Highest quality (full features)
cfg_fused = Config.fused()
```

### Enable Advanced Features

```python
config = Config.fused()

# Phase 5 compositional caching (10-300× speedup)
config.enable_linguistic_gate = True
config.use_compositional_cache = True

# Recursive learning
config.enable_recursive_learning = True

# Alignment framework
config.enable_alignment = True

# Semantic calculus
config.enable_semantic_calculus = True
```

---

## Key File Locations

### Core System
- [HoloLoom/hololoom.py](HoloLoom/hololoom.py) - Simple API (471 lines)
- [HoloLoom/weaving_orchestrator.py](HoloLoom/weaving_orchestrator.py) - Main orchestrator (1,963 lines)
- [HoloLoom/config.py](HoloLoom/config.py) - Configuration (460 lines)
- [HoloLoom/unified_api.py](HoloLoom/unified_api.py) - Unified API (729 lines)
- [HoloLoom/terminal_ui.py](HoloLoom/terminal_ui.py) - Interactive UI (751 lines)

### Memory
- [HoloLoom/memory/graph.py](HoloLoom/memory/graph.py) - Knowledge graph
- [HoloLoom/memory/backend_factory.py](HoloLoom/memory/backend_factory.py) - Backend creation (231 lines)
- [HoloLoom/memory/cache.py](HoloLoom/memory/cache.py) - BM25 + semantic

### Learning & Intelligence
- [HoloLoom/recursive/full_learning_loop.py](HoloLoom/recursive/full_learning_loop.py) - Phase 5 (626 lines)
- [HoloLoom/alignment/safety_guardrails.py](HoloLoom/alignment/safety_guardrails.py) - Safety (526 lines)
- [HoloLoom/agentic/core.py](HoloLoom/agentic/core.py) - Agentic reasoning

### Input/Output
- [HoloLoom/spinningWheel/](HoloLoom/spinningWheel/) - Input adapters (28+ spinners)
- [HoloLoom/visualization/](HoloLoom/visualization/) - Tufte visualizations (7 types)

### Integration
- [HoloLoom/server/agentic_api.py](HoloLoom/server/agentic_api.py) - FastAPI server (531 lines)
- [HoloLoom/memory/mcp_server.py](HoloLoom/memory/mcp_server.py) - MCP server (600+ lines)
- [squad/src/HoloLoomBridge.ts](squad/src/HoloLoomBridge.ts) - VS Code bridge (144 lines)

---

## Documentation Index

### Getting Started
- **CLAUDE.md** - Developer quick reference (this file)
- **HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md** - Complete architectural map (25,000+ lines)
- **CURRENT_STATUS_AND_NEXT_STEPS.md** - What works now, what to build next
- **ARCHITECTURE_VISUAL_MAP.md** - Visual diagrams

### Core Systems
- **MEMORY_SYSTEMS_SUMMARY.md** - Memory backend guide
- **RECURSIVE_LEARNING_COMPLETE.md** - 6-phase learning system
- **ALIGNMENT_CONFIG_GUIDE.md** - Safety framework setup
- **AGENTIC_SEARCH_COMPLETE.md** - Multi-query reasoning

### Advanced Features
- **PHASE_5_COMPLETE.md** - Compositional caching (10-300× speedup)
- **TUFTE_VISUALIZATION_ROADMAP.md** - Visualization system
- **EXPERIMENTS_GUIDE.md** - Automated testing framework

### Integration
- **DOCKER_MEMORY_SETUP.md** - Docker deployment
- **UNIFIED_MEMORY_INTEGRATION.md** - Memory backend integration
- **HoloLoom/server/README.md** - API server guide

### Testing
- **TEST_REPORT_WAVE_1.md** - Test results
- **WAVE_1_VALIDATION_SUMMARY.md** - Validation summary

---

## Conclusion

HoloLoom represents a complete, production-ready neural decision-making system built on sound theoretical foundations (Chomsky linguistics, cognitive science, reinforcement learning) with pragmatic engineering.

### Core Principles

1. **Weaving Metaphor**: Symbolic ↔ Continuous seamless transition
2. **Complete Provenance**: Every decision fully traceable
3. **Graceful Degradation**: Never crashes, always works
4. **Self-Improving**: Learns from every interaction
5. **Safety First**: All decisions gated and audited

### Why HoloLoom?

- **For Developers**: Clean API, comprehensive docs, extensive tests
- **For Researchers**: Complete provenance, extensible protocols, research modes
- **For Production**: Docker deployment, alignment framework, performance optimization
- **For Users**: Simple interface, intelligent responses, continuous improvement

### Next Steps

1. Read [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md) for complete architecture
2. Run demos in `demos/` directory
3. Start with simple API ([HoloLoom/hololoom.py](HoloLoom/hololoom.py))
4. Explore advanced features (recursive learning, alignment, agentic)
5. Deploy to production (Docker + FastAPI)

---

**Generated by**: Claude Code + 6 parallel Explore agents
**Date**: 2025-11-03
**Repository**: c:\Users\blake\OneDrive\Documents\mythRL
**Total Analysis**: ~63,000 tokens across core architecture, memory systems, learning systems, I/O systems, integration points, and test infrastructure
