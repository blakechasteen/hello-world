# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📚 Comprehensive Documentation

**New to HoloLoom?** Start here:

1. **[HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)** (25,000+ lines)
   - Complete architectural map from first principles to production
   - Learning sequence for beginners → researchers
   - All 5 phases explained with context
   - Future roadmap (Phases 6-10)
   - **Start here for the big picture!**

2. **[CURRENT_STATUS_AND_NEXT_STEPS.md](CURRENT_STATUS_AND_NEXT_STEPS.md)**
   - What works right now (snapshot)
   - What needs work (prioritized tasks)
   - Recommended next actions
   - Quick decision guide
   - **Use this to know what to build next**

3. **[ARCHITECTURE_VISUAL_MAP.md](ARCHITECTURE_VISUAL_MAP.md)**
   - Visual diagrams of the 9-layer system
   - Data flow illustrations
   - Component relationships
   - Quick reference to key files
   - **Best for visual learners**

4. **This file (CLAUDE.md)** - Developer quick reference (below)

---

## Reliable Systems: Safety First

**"Reliable Systems: Safety First"** is our guiding development philosophy. Before optimizing for performance, features, or elegance, we prioritize:

- **Graceful degradation**: Systems should never crash due to missing optional dependencies
- **Automatic fallbacks**: When production backends fail, fall back to working alternatives (e.g., HYBRID → INMEMORY)
- **Proper lifecycle management**: All resources get explicit cleanup through async context managers
- **Comprehensive testing**: Unit, integration, and end-to-end tests organized by speed for fast feedback
- **Clear error messages**: When things fail, developers should immediately understand why and how to fix it
- **Type safety**: Protocol-based design with clear interfaces prevents integration errors
- **Data persistence safety**: Never lose user data - archive instead of delete, checkpoint frequently

This principle permeates every architectural decision in HoloLoom. We'd rather ship a slower but reliable system than a fast but fragile one.

## Repository Overview

**HoloLoom** is a Python-based neural decision-making system that combines:
- Multi-scale embeddings (Matryoshka representations)
- Knowledge graph memory with spectral features
- Unified policy engine with Thompson Sampling exploration
- PPO reinforcement learning for agent training
- Input adapters ("SpinningWheel") for processing audio, text, and other modalities

The system is designed around a "weaving" metaphor: independent "warp thread" modules are coordinated by an "orchestrator" (the shuttle) to produce responses.

## Development Commands

### Environment Setup

Create and activate virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy gymnasium matplotlib
```

Optional dependencies for full features:
```bash
pip install spacy sentence-transformers scipy networkx ollama
python -m spacy download en_core_web_sm
```

### Testing

**Test Organization** (Phase 1+2 Cleanup - Oct 2025):
Tests are organized into three tiers for fast feedback loops:

```bash
# Unit Tests (Fast - <5s) - Isolated component testing
pytest HoloLoom/tests/unit/ -v

# Integration Tests (Medium - <30s) - Multi-component testing
pytest HoloLoom/tests/integration/ -v

# End-to-End Tests (Slow - <2min) - Full pipeline testing
pytest HoloLoom/tests/e2e/ -v

# Run all tests
pytest HoloLoom/tests/ -v
```

**Key Tests:**
- `tests/unit/test_unified_policy.py` - Neural components (MLP, attention, ICM/RND, PPO)
- `tests/integration/test_backends.py` - Memory backend integration
- `tests/e2e/test_full_pipeline.py` - Complete weaving cycle (BARE/FAST/FUSED modes)

**Memory Backend Validation:**
```bash
python test_memory_backend_simplification.py
```
Validates 3-backend architecture (INMEMORY/HYBRID/HYPERSPACE) and auto-fallback.

### Training Example

Run a short CartPole training session:
```bash
PYTHONPATH=. .venv/bin/python -c "from holoLoom.train_agent import PPOTrainer; t=PPOTrainer(env_name='CartPole-v1', total_timesteps=2000, steps_per_update=256, n_epochs=1, batch_size=32, log_dir='./logs/test_run_small'); t.train()"
```

Training checkpoints are saved to the specified `log_dir`.

### Running the Orchestrator

Example usage of the full HoloLoom orchestrator:
```bash
PYTHONPATH=. .venv/bin/python holoLoom/orchestrator.py
```

This runs a demo showing query → features → context → decision → response pipeline.

## Architecture

### Core Design Philosophy

**"Warp Thread" Modules**: Each major component (motif detection, embedding, memory, policy) is independent and protocol-based. They don't import from each other, only from shared types (`holoLoom/documentation/types.py`).

**"Shuttle" Orchestrator**: The `orchestrator.py` is the only module that imports from all others. It weaves components together into the full processing pipeline.

### Weaving Architecture

HoloLoom implements a complete weaving metaphor as first-class abstractions:

#### 1. Yarn Graph (holoLoom/memory/graph.py)
The persistent symbolic memory - discrete thread structure stored as a NetworkX MultiDiGraph.
- **Alias**: `YarnGraph = KG`
- Entities and relationships form the "threads" of memory
- Remains discrete until "tensioned" into Warp Space

#### 2. Loom Command (holoLoom/loom/command.py)
Pattern card selector that chooses execution template (BARE/FAST/FUSED).
- **Classes**: `LoomCommand`, `PatternCard`, `PatternSpec`
- Determines which warp threads to lift and how densely to weave
- Configures scales, features, timeouts for entire cycle

#### 3. Chrono Trigger (holoLoom/chrono/trigger.py)
Temporal control system managing time-dependent aspects.
- **Classes**: `ChronoTrigger`, `TemporalWindow`, `ExecutionLimits`
- Controls when threads activate (temporal windows)
- Manages execution timing, rhythm (heartbeat), halt conditions
- Handles thread decay and system evolution over time

#### 4. Resonance Shed (holoLoom/resonance/shed.py)
Feature interference zone where extraction threads combine.
- **Classes**: `ResonanceShed`, `FeatureThread`
- Lifts feature threads (motif, embedding, spectral)
- Creates interference patterns through multi-modal fusion
- Produces DotPlasma (flowing feature representation)

#### 5. DotPlasma (holoLoom/documentation/types.py)
The "feature fluid" - flowing continuous representation.
- **Alias**: `DotPlasma = Features`
- Malleable medium between extraction and decision
- Contains motifs (symbolic), embeddings (continuous), spectral (topological)

#### 6. Warp Space (holoLoom/warp/space.py)
Tensioned tensor field for continuous mathematics.
- **Classes**: `WarpSpace`, `TensionedThread`
- Temporary manifold where activated threads undergo tensor operations
- Lifecycle: tension() → compute() → collapse()
- Detensions back to discrete Yarn Graph after computation

#### 7. Convergence Engine (holoLoom/convergence/engine.py)
Decision collapse from continuous → discrete.
- **Classes**: `ConvergenceEngine`, `CollapseStrategy`, `ThompsonBandit`
- Collapses probability distributions to discrete tool selections
- Strategies: ARGMAX, EPSILON_GREEDY, BAYESIAN_BLEND, PURE_THOMPSON
- Thompson Sampling for exploration/exploitation balance

#### 8. Spacetime (holoLoom/fabric/spacetime.py)
Woven fabric - structured output with complete lineage.
- **Classes**: `Spacetime`, `WeavingTrace`, `FabricCollection`
- 4-dimensional output: 3D semantic space + 1D temporal trace
- Full computational provenance for debugging and reflection learning
- Serializable for persistence and analysis

#### 9. Reflection Buffer (holoLoom/memory/cache.py)
Learning loop - stores outcomes for improvement.
- **Alias**: `ReflectionBuffer = MemoryManager`
- Episodic buffer of recent interactions
- Provides signals for system evolution and adaptation

#### Complete Weaving Cycle

```
1. Loom Command selects Pattern Card (BARE/FAST/FUSED)
2. Chrono Trigger fires, creates TemporalWindow
3. Yarn Graph threads selected based on temporal window
4. Resonance Shed lifts feature threads, creates DotPlasma
5. Warp Space tensions threads into continuous manifold
6. Convergence Engine collapses to discrete tool selection
7. Tool executes, results woven into Spacetime fabric
8. Reflection Buffer learns from outcome
9. Chrono Trigger detensions, cycle completes
```

This architecture enables:
- **Symbolic ↔ Continuous**: Seamless transition between discrete and continuous representations
- **Temporal Control**: Fine-grained timing and decay mechanisms
- **Multi-Modal Fusion**: Interference patterns from diverse feature types
- **Provenance**: Complete computational lineage for every output
- **Evolution**: System learns and adapts from reflection

### Key Components

#### 1. Weaving Orchestrator (`HoloLoom/weaving_orchestrator.py`)

**UPDATED (Task 1.2 - Oct 27, 2025):** The Shuttle architecture has been integrated into the canonical `WeavingOrchestrator`.

The WeavingOrchestrator implements the full 9-step weaving cycle with mythRL protocol-based architecture:

1. **Loom Command** → Pattern Card selection (BARE/FAST/FUSED)
2. **Chrono Trigger** → Temporal window creation
3. **Yarn Graph** → Thread selection from memory
4. **Resonance Shed** → Feature extraction, DotPlasma creation
5. **Warp Space** → Continuous manifold tensioning
6. **Convergence Engine** → Discrete decision collapse
7. **Tool Execution** → Action with results
8. **Spacetime Fabric** → Provenance and trace
9. **Reflection Buffer** → Learning from outcome

**mythRL Progressive Complexity (3-5-7-9 System):**
- **LITE (3 steps)**: Extract → Route → Execute (<50ms) - simple queries
- **FAST (5 steps)**: + Pattern Selection + Temporal Windows (<150ms) - standard queries
- **FULL (7 steps)**: + Decision Engine + Synthesis Bridge (<300ms) - complex queries
- **RESEARCH (9 steps)**: + Advanced WarpSpace + Full Tracing (no limit) - research mode

**Protocol-Based Design:**
- `PatternSelectionProtocol`: Processing pattern selection
- `FeatureExtractionProtocol`: Multi-scale Matryoshka extraction
- `WarpSpaceProtocol`: Mathematical manifold operations
- `DecisionEngineProtocol`: Strategic multi-criteria optimization

**Key Features:**
- Auto-complexity detection based on query characteristics
- Performance caching (QueryCache) for repeated queries
- Reflection loop for continuous improvement
- Lifecycle management with async context managers
- Backward compatibility: `WeavingShuttle` is an alias to `WeavingOrchestrator`

Usage:
```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query

config = Config.fused()
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="What is Thompson Sampling?"))
    # Automatic cleanup on exit
```

#### 2. Policy Engine (`holoLoom/policy/unified.py`)
Neural decision-making with three bandit exploration strategies:
- **Epsilon-Greedy** (default): 90% neural exploitation, 10% Thompson Sampling exploration
- **Bayesian Blend**: Combines neural predictions (70%) with bandit priors (30%)
- **Pure Thompson**: Uses only Thompson Sampling (ignores neural network)

The policy uses:
- Transformer blocks with cross-attention to context memory
- Motif-gated multi-head attention
- LoRA-style adapters for different execution modes (bare/fast/fused)
- Thompson Sampling bandit for exploration/exploitation balance

Key fix from code review: Bandit now updates statistics for the **actually selected tool** (previously disconnected).

#### 3. Configuration (`holoLoom/config.py`)
Three execution modes:
- **BARE**: Minimal processing (regex motifs, single scale, simple policy) - fastest
- **FAST**: Balanced (hybrid motifs, 2 scales, neural policy) - good tradeoff
- **FUSED**: Full processing (all features, 3 scales, multi-scale retrieval) - highest quality

Access via factory methods:
```python
from holoLoom.config import Config
cfg_fast = Config.fast()
cfg_fused = Config.fused()
```

#### 4. Memory Systems

**Vector Memory** (`holoLoom/memory/cache.py`): BM25 + semantic similarity retrieval
**Knowledge Graph** (`holoLoom/memory/graph.py`): NetworkX-based entity relationships with:
- Typed edges (IS_A, USES, MENTIONS, etc.)
- Subgraph extraction for context expansion
- Path finding between entities
- Spectral graph features for policy input

#### 5. Embeddings (`holoLoom/embedding/spectral.py`)
Matryoshka embeddings at multiple scales (96, 192, 384 dimensions) with:
- Multi-scale fusion for retrieval
- Spectral features: graph Laplacian eigenvalues, SVD topic components
- Optional sentence-transformers backend (degrades gracefully without it)

#### 6. SpinningWheel (`holoLoom/spinningWheel/`)
Input adapters that convert raw data → `MemoryShard` objects:
- **AudioSpinner**: Processes transcripts, task lists, summaries
- **YouTubeSpinner**: Extracts YouTube video transcripts with optional chunking
  - Supports multiple URL formats (full URL, youtu.be, video ID)
  - Language preference with automatic fallback
  - Time-based chunking for long videos
  - Preserves timestamps and video metadata
  - See `HoloLoom/spinningWheel/README_YOUTUBE.md` for details
- Optional Ollama enrichment for entity/motif extraction
- Standardized output format feeds directly into orchestrator

#### 7. Training (`holoLoom/train_agent`)
PPO trainer for RL environments with:
- GAE (Generalized Advantage Estimation)
- Optional ICM/RND curiosity modules
- Checkpoint saving/loading
- Configurable network architectures

### Module Structure (Phase 1+2 Cleanup - Oct 2025)

**Clean Root Directory** (6 core files only):
```
HoloLoom/
├── __init__.py                # Package entry point
├── config.py                  # Configuration (BARE/FAST/FUSED modes)
├── unified_api.py             # Programmatic API
├── weaving_shuttle.py         # Main entry point (async context manager)
├── weaving_orchestrator.py    # Full 9-step weaving cycle
└── protocols.py               # DEPRECATED (use protocols/ directory)
```

**Organized Subdirectories:**
```
HoloLoom/
├── tests/                     # All tests (Phase 2)
│   ├── unit/                  # Fast isolated tests (<5s)
│   ├── integration/           # Multi-component tests (<30s)
│   └── e2e/                   # Full pipeline tests (<2min)
│
├── tools/                     # Developer utilities (Phase 1)
│   ├── bootstrap_system.py
│   ├── validate_pipeline.py
│   ├── visualize_bootstrap.py
│   └── archive/               # Archived dead code (safety net)
│
├── memory/                    # Storage backends (13 files, was 17)
│   ├── backend_factory.py    # Create backends (231 lines, was 550)
│   ├── graph.py              # NetworkX (default, always works)
│   ├── neo4j_graph.py        # Production backend
│   ├── hyperspace_backend.py # Research backend
│   ├── protocol.py           # Memory protocols (120 lines, was 787)
│   └── unified.py            # Unified interface
│
├── policy/                    # Decision making
│   ├── unified.py            # Neural core + Thompson Sampling
│   └── semantic_nudging.py   # Semantic goal guidance
│
├── protocols/                 # Protocol definitions (Phase 2)
│   ├── __init__.py           # Public exports
│   ├── core.py               # Core protocol definitions
│   └── types.py              # Shared data types
│
├── semantic_calculus/         # 244D semantic space
│   ├── dimensions.py         # EXTENDED_244_DIMENSIONS
│   ├── integrator.py         # SemanticSpectrum
│   └── dimension_selector.py
│
├── reflection/                # Learning & improvement
│   ├── buffer.py             # ReflectionBuffer
│   ├── ppo_trainer.py        # PPO training
│   └── semantic_learning.py  # Multi-task learner (6 signals)
│
├── embedding/                 # Multi-scale embeddings
│   ├── spectral.py           # Matryoshka + spectral features
│   └── matryoshka_interpreter.py  # (moved from root)
│
├── spinningWheel/             # Input adapters
│   ├── audio.py              # Audio/transcript processing
│   ├── youtube.py            # YouTube transcription
│   └── autospin.py           # (moved from root)
│
├── chatops/                   # Conversational features
│   ├── core/chatops_bridge.py
│   ├── conversational.py     # (moved from root)
│   └── ROADMAP.md            # ChatOps + Semantic Learning plan
│
└── [other feature dirs...]    # loom/, warp/, resonance/, etc.
```

**Key Changes:**
- ✅ Root: 17 → 6 files (-65%)
- ✅ Memory: 17 → 13 files (-24%)
- ✅ Tests: Organized into unit/integration/e2e
- ✅ Backend factory: 550 → 231 lines (-58%)
- ✅ Protocols: 787 → 120 lines (-84%)
- ✅ Dead code: Archived to tools/archive/
- ✅ All tests passing

## Important Patterns

### Protocol-Based Design
All major components define protocols (abstract interfaces):
- `PolicyEngine` for decision making
- `KGStore` for knowledge graphs
- `Retriever` for memory systems

This enables swapping implementations without changing orchestrator code.

### Graceful Degradation
Optional dependencies (spaCy, sentence-transformers, BM25, SciPy) degrade with warnings:
- Motif detection falls back to regex-only
- Embeddings use fallback implementations
- Spectral features are skipped if unavailable

Never crash due to missing optional dependencies.

### Async Pipeline
The orchestrator uses `async/await` for the main processing pipeline, enabling:
- Concurrent feature extraction and retrieval
- Background memory management tasks
- Non-blocking tool execution

### Import Path Requirements

**CRITICAL**: When running holoLoom modules, set `PYTHONPATH=.` from repository root or run with proper path:
```bash
# Correct - from repository root
PYTHONPATH=. python holoLoom/test_unified_policy.py

# Also correct - cd into directory
cd holoLoom && python test_unified_policy.py
```

The codebase uses absolute imports like `from holoLoom.policy.unified import ...`, which require the repository root to be on the Python path.

### Testing Strategy

The test suite (`test_unified_policy.py`) validates components in isolation:
1. Building blocks (MLP, attention)
2. Curiosity modules (ICM, RND)
3. Policy variants (deterministic, categorical, gaussian)
4. PPO agent (GAE, updates, checkpointing)
5. Full end-to-end pipeline

Tests are designed to run without external dependencies (no actual RL environments).

## Known Issues

From `documentation/CODE_REVIEW.md`:

1. **Background task lifecycle**: MemoryManager spawns fire-and-forget tasks without shutdown hooks. Add explicit lifecycle management.

2. **Type duplication**: Orchestrator defines inline types that duplicate `modules/Types.py`. Consolidate to shared types.

3. **Empty Features module**: `modules/Features.py` is currently empty but imported elsewhere. Either implement or remove.

## Development Tips

1. **Start with BARE mode** for fastest iteration:
   ```python
   cfg = Config.bare()
   ```

2. **Check bandit statistics** to understand exploration behavior:
   ```python
   stats = policy.bandit.get_stats()
   ```

3. **Use spectral features** for richer context in FUSED mode:
   - Graph Laplacian eigenvalues capture knowledge structure
   - SVD components provide topic-level signals

4. **Monitor adapter selection** via action plan metadata:
   ```python
   action_plan = await orchestrator.process(query)
   print(f"Adapter: {action_plan.adapter}")
   ```

5. **Training checkpoints** are saved as `.pt` files with full state dicts. Load with:
   ```python
   agent.load('logs/checkpoint.pt')
   ```

6. **Use async context managers for lifecycle management** (recommended):
   ```python
   async with WeavingShuttle(cfg=config, shards=shards) as shuttle:
       spacetime = await shuttle.weave(query)
       # Automatic cleanup on exit
   ```

7. **Use dynamic memory backends for persistent storage**:
   ```python
   memory = await create_memory_backend(config)
   async with WeavingShuttle(cfg=config, memory=memory) as shuttle:
       spacetime = await shuttle.weave(query)
   ```

## Phase 5: Universal Grammar + Compositional Cache

**Implemented: Oct 2025** - Phase 5 provides 10-300× speedup through linguistic intelligence and compositional caching.

### Overview

Phase 5 integrates three breakthrough technologies:
1. **Universal Grammar Chunking**: X-bar theory for principled phrase structure analysis
2. **Compositional Cache**: 3-tier caching (parse/merge/semantic) with phrase-level reuse
3. **Linguistic Matryoshka Gate**: Pre-filtering and progressive refinement

### Performance Benefits

- **Parse Cache**: 10-50× speedup for X-bar structure caching
- **Merge Cache**: 5-10× speedup through compositional reuse
- **Semantic Cache**: 3-10× speedup for 244D projections
- **Total Speedup**: 50-300× multiplicative speedup (hot paths)
- **Production**: 10-17× expected speedup with 90-99% cache hit rates

### Configuration

Enable Phase 5 in your config:

```python
from HoloLoom.config import Config

# Basic Phase 5 (compositional cache only)
config = Config.fused()
config.enable_linguistic_gate = True
config.linguistic_mode = "disabled"  # Cache only, no pre-filtering
config.use_compositional_cache = True
config.parse_cache_size = 10000
config.merge_cache_size = 50000

# Advanced Phase 5 (full linguistic filtering)
config = Config.fused()
config.enable_linguistic_gate = True
config.linguistic_mode = "both"  # Pre-filter + embedding features
config.use_compositional_cache = True
config.linguistic_weight = 0.3
config.prefilter_similarity_threshold = 0.3
config.prefilter_keep_ratio = 0.7
```

### Usage

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

# Create config with Phase 5 enabled
config = Config.fused()
config.enable_linguistic_gate = True
config.linguistic_mode = "both"

# Create orchestrator
shards = create_memory_shards()
async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
    # First query (cold cache)
    spacetime = await shuttle.weave(Query(text="What is passive voice?"))
    # Duration: ~150ms (cold)

    # Repeated query (warm cache)
    spacetime = await shuttle.weave(Query(text="What is passive voice?"))
    # Duration: ~0.5ms (hot) - 300× speedup!
```

### Linguistic Filter Modes

- **disabled**: Compositional cache only (no linguistic pre-filtering)
- **prefilter**: Filter candidates by syntactic compatibility before embedding
- **embedding**: Add linguistic features to embeddings
- **both**: Pre-filter + embedding features (recommended for production)

### Demo

Run the Phase 5 integration demo:

```bash
PYTHONPATH=. python demos/phase5_orchestrator_integration.py
```

This demonstrates:
- Baseline performance without Phase 5
- Compositional cache performance (cache only)
- Full linguistic filtering performance
- Warm cache performance (100-300× speedup)

### Key Features

**Compositional Reuse**: Different queries share building blocks
- "the big red ball" caches "ball", "red ball", "big red ball"
- "a big red ball" reuses "big red ball" composition
- Cross-query optimization for massive speedups

**Universal Grammar**: Principled phrase structure
- X-bar theory (XP → Spec + X' → X' + Comp)
- Hierarchical phrase detection (NP, VP, PP, CP, TP)
- Syntactic compatibility scoring

**Graceful Fallback**: No breaking changes
- Phase 5 automatically falls back if spaCy not available
- Disabled by default (opt-in via config)
- Backward compatible with all existing code

### Documentation

See comprehensive Phase 5 documentation:
- `CHOMSKY_LINGUISTIC_INTEGRATION.md` - Linguistic foundations (992 lines)
- `LINGUISTIC_MATRYOSHKA_INTEGRATION.md` - Matryoshka gate integration (551 lines)
- `PHASE_5_UG_COMPOSITIONAL_CACHE.md` - Architecture and design (782 lines)
- `PHASE_5_COMPLETE.md` - Implementation summary (592 lines)
- `PHASE_5_INTEGRATION_COMPLETE.md` - Final integration notes

## Unified Memory Integration

HoloLoom supports both static shards and dynamic memory backends for persistent storage.

### Memory Sources

**Static Shards** (backward compatible):
```python
shards = create_test_shards()
shuttle = WeavingShuttle(cfg=config, shards=shards)
```

**Dynamic Backends** (persistent storage):
```python
from HoloLoom.memory.backend_factory import create_memory_backend

memory = await create_memory_backend(config)
shuttle = WeavingShuttle(cfg=config, memory=memory)
```

### Backend Options (Simplified - Oct 2025)

**Task 1.3 Simplification:** Reduced from 10+ backends to 3 core options.

Configure via `Config.memory_backend`:
- **INMEMORY**: NetworkX in-memory graph (development, always works)
- **HYBRID**: Neo4j + Qdrant with auto-fallback (production, recommended)
- **HYPERSPACE**: Advanced gated multipass (research only)

**Auto-Fallback:** HYBRID automatically falls back to INMEMORY if Docker services unavailable.
**Migration:** All legacy backend enums removed. See `MEMORY_SIMPLIFICATION_REVIEW.md` for details.

### Docker Setup

Start Neo4j + Qdrant backends:
```bash
docker-compose up -d
```

See `DOCKER_MEMORY_SETUP.md` for complete setup guide and `UNIFIED_MEMORY_INTEGRATION.md` for implementation details.

### Production Example

```python
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend

config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID

# Create persistent backend (auto-falls back to INMEMORY if no Docker)
memory = await create_memory_backend(config)

# Use with shuttle
async with WeavingShuttle(cfg=config, memory=memory) as shuttle:
    spacetime = await shuttle.weave(query)
    # Data persists across sessions (if Neo4j/Qdrant available)
```

## Lifecycle Management

HoloLoom implements proper resource management through async context managers:

### Using Context Managers (Recommended)

```python
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config

config = Config.fast()
shards = create_memory_shards()

# Recommended: Automatic cleanup
async with WeavingShuttle(cfg=config, shards=shards, enable_reflection=True) as shuttle:
    spacetime = await shuttle.weave(query)
    await shuttle.reflect(spacetime, feedback={"helpful": True})
    # Resources automatically cleaned up on exit
```

### Manual Cleanup

If context managers aren't suitable (e.g., long-lived services), use explicit cleanup:

```python
shuttle = WeavingShuttle(cfg=config, shards=shards)
try:
    spacetime = await shuttle.weave(query)
finally:
    await shuttle.close()  # IMPORTANT: Always close!
```

### Background Task Tracking

Background tasks are automatically tracked and cancelled on shutdown:

```python
async with WeavingShuttle(cfg=config, shards=shards) as shuttle:
    # Spawn tracked background tasks
    task = shuttle.spawn_background_task(some_async_work())

    # Do weaving
    spacetime = await shuttle.weave(query)

    # Background tasks cancelled automatically on exit
```

### What Gets Cleaned Up

1. **Background tasks**: Cancelled with 5-second timeout
2. **Reflection buffer**: Metrics flushed to disk
3. **Database connections**: Neo4j/Qdrant clients closed (when implemented)
4. **File handles**: Proper closing of persistent storage

### ReflectionBuffer Lifecycle

The reflection buffer also supports lifecycle management:

```python
async with ReflectionBuffer(capacity=1000, persist_path="./reflections") as buffer:
    await buffer.store(spacetime, feedback=feedback)
    # Metrics automatically flushed on exit
```

## Recursive Learning System

**Status**: ✅ All 5 Phases Complete (October 29, 2025)
**Location**: `HoloLoom/recursive/`
**Total Code**: ~4,700 lines across 5 phases

The Recursive Learning System is a self-improving knowledge architecture that learns from every interaction, adapts continuously, and maintains complete provenance of all decisions.

### Overview

The system implements 5 phases of recursive learning:

1. **Phase 1: Scratchpad Integration** - Provenance tracking
2. **Phase 2: Loop Engine Integration** - Pattern learning
3. **Phase 3: Hot Pattern Feedback** - Usage-based adaptation
4. **Phase 4: Advanced Refinement** - Multi-strategy refinement
5. **Phase 5: Full Learning Loop** - Background learning with Thompson Sampling

### Philosophy

**"Great answers aren't written, they're refined."**

The system embraces multiple passes on quality dimensions:
- **ELEGANCE**: Clarity → Simplicity → Beauty
- **VERIFY**: Accuracy → Completeness → Consistency

### Phase 1: Scratchpad Integration (990 lines)

Tracks complete provenance of every decision:

```python
from HoloLoom.recursive import weave_with_scratchpad

spacetime, scratchpad = await weave_with_scratchpad(
    Query(text="What is Thompson Sampling?"),
    Config.fast(),
    shards=shards,
    enable_refinement=True
)

# View complete reasoning history
print(scratchpad.get_history())
```

**Features**:
- Automatic thought → action → observation → score tracking
- Full audit trail for debugging
- Triggers refinement when confidence < threshold

### Phase 2: Pattern Learning (850 lines)

Learns from successful queries:

```python
from HoloLoom.recursive import LearningLoopEngine

async with LearningLoopEngine(cfg=config, shards=shards) as engine:
    spacetime = await engine.weave_and_learn(query)

    # System automatically learns patterns from high-confidence results
    patterns = engine.pattern_learner.get_hot_patterns()
```

**Features**:
- Extracts motif → tool → confidence patterns
- Classifies queries (factual, procedural, analytical)
- Auto-prunes stale patterns
- Learns what works over time

### Phase 3: Hot Pattern Feedback (780 lines)

Adapts retrieval based on usage:

```python
from HoloLoom.recursive import HotPatternFeedbackEngine

async with HotPatternFeedbackEngine(cfg=config, shards=shards) as engine:
    spacetime = await engine.weave(query)

    # View hot patterns (most accessed knowledge)
    hot = engine.hot_tracker.get_hot_patterns(limit=10)
```

**Heat Score Algorithm**:
```
heat = access_count × success_rate × avg_confidence
     × (0.95 ^ hours_since_last_access)
```

**Features**:
- Tracks access frequency of knowledge elements
- Hot patterns get 2x boost, cold patterns get 0.5x penalty
- Exponential decay (5% per hour)
- Adaptive retrieval weights

### Phase 4: Advanced Refinement (680 lines)

Multiple refinement strategies with quality tracking:

```python
from HoloLoom.recursive import AdvancedRefiner, RefinementStrategy

refiner = AdvancedRefiner(orchestrator, enable_learning=True)

result = await refiner.refine(
    query=query,
    initial_spacetime=low_confidence_result,
    strategy=RefinementStrategy.ELEGANCE,  # Or None for auto-select
    max_iterations=3,
    quality_threshold=0.9
)

print(result.summary())
# Output: Strategy: elegance, Iterations: 3, Quality: 0.65 → 0.94
```

**Available Strategies**:

| Strategy | Focus | Passes |
|----------|-------|--------|
| REFINE | Context expansion | Iterative |
| CRITIQUE | Self-improvement | 1 pass |
| VERIFY | Accuracy → Completeness → Consistency | 3 passes |
| ELEGANCE | Clarity → Simplicity → Beauty | 3 passes |
| HOFSTADTER | Recursive self-reference | Iterative |

**Quality Scoring**:
```
quality = 0.7 × confidence + 0.2 × context_richness + 0.1 × response_completeness
```

**Features**:
- Auto-strategy selection based on query characteristics
- Quality trajectory tracking across iterations
- Learns which strategies work best for which queries
- Multi-pass refinement for complex quality dimensions

### Phase 5: Full Learning Loop (750 lines)

Background learning with Bayesian updates:

```python
from HoloLoom.recursive import FullLearningEngine

async with FullLearningEngine(
    cfg=config,
    shards=shards,
    enable_background_learning=True,
    learning_update_interval=60.0  # Update every 60 seconds
) as engine:
    # Process queries - system learns automatically
    spacetime = await engine.weave(
        query,
        enable_refinement=True,
        refinement_threshold=0.75
    )

    # View comprehensive statistics
    stats = engine.get_learning_statistics()

    # Save learning state
    engine.save_learning_state("./learning_state")
```

**Thompson Sampling Updates**:
```
Success (confidence ≥ 0.75): α ← α + confidence
Failure (confidence < 0.75): β ← β + (1 - confidence)

Expected Reward: E[X] = α / (α + β)
```

**Policy Weight Updates**:
```
weight = (successes + 1) / (total + 2)  # Laplace smoothing
```

**Features**:
- Background learning thread (async, every 60s)
- Thompson Sampling priors adapt to tool performance
- Policy adapter weights adjust based on outcomes
- Complete learning state persistence

### Usage Examples

**Simple (Phase 1 only)**:
```python
from HoloLoom.recursive import weave_with_scratchpad

spacetime, scratchpad = await weave_with_scratchpad(
    Query(text="Explain recursion"),
    Config.fast(),
    shards=shards
)
```

**With Learning (Phases 1-3)**:
```python
from HoloLoom.recursive import HotPatternFeedbackEngine

async with HotPatternFeedbackEngine(cfg=config, shards=shards) as engine:
    spacetime = await engine.weave(query)
```

**Full System (All 5 Phases)**:
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

### Performance Characteristics

| Operation | Overhead | When |
|-----------|----------|------|
| Provenance extraction | <1ms | Every query |
| Pattern extraction | <1ms | High-confidence only |
| Heat tracking | <0.5ms | Every query |
| Thompson/Policy update | <0.5ms | Every query |
| Refinement | ~150ms × iterations | Low-confidence only (10-20%) |
| Background learning | ~50ms | Every 60s (async) |

**Total Per-Query Overhead**: <3ms (excluding refinement)

### Key Benefits

1. **Self-Improving**: Gets better with every query
2. **Quality-Aware**: Detects low confidence and refines automatically
3. **Adaptive**: Thompson Sampling + policy weights + retrieval weights all adapt
4. **Complete Provenance**: Full audit trail with scratchpad
5. **Minimal Overhead**: <3ms per query

### Documentation

- **RECURSIVE_LEARNING_COMPLETE.md**: Complete system overview
- **PHASES_4_5_COMPLETE.md**: Phase 4-5 implementation details
- **MULTIPASS_REFINEMENT.md**: Multi-pass philosophy and usage
- **demos/demo_multipass_simple.py**: Visual demonstration

### Running Demos

```bash
# Multi-pass refinement demonstration
python demos/demo_multipass_simple.py

# Full 5-phase system (requires HoloLoom integration)
PYTHONPATH=. python demos/demo_full_recursive_learning.py
```

## Common Workflows

### Adding a New Tool
1. Add tool name to `NeuralCore.tools` list in `policy/unified.py`
2. Update `n_tools` parameter in config
3. Implement execution logic in `ToolExecutor.execute()` in `orchestrator.py`
4. Retrain policy or adjust tool selection weights

### Creating a Custom Adapter
1. Add adapter to `adapter_bank` dict in policy factory
2. Create corresponding LoRA adapter weights in `LoRALikeFFN`
3. Map adapter to execution context in orchestrator

### Adding a New Spinner (Input Adapter)
1. Inherit from `BaseSpinner` in `spinningWheel/base.py`
2. Implement `async def spin(raw_data) -> List[MemoryShard]`
3. Add to factory in `spinningWheel/__init__.py`
4. Parse raw data format and extract entities/motifs

Example: See `HoloLoom/spinningWheel/youtube.py` for a complete implementation
```python
from HoloLoom.spinningWheel import transcribe_youtube

# Quick usage
shards = await transcribe_youtube('VIDEO_ID', chunk_duration=60.0)

# Or use the spinner directly
from HoloLoom.spinningWheel import YouTubeSpinner, YouTubeSpinnerConfig

config = YouTubeSpinnerConfig(chunk_duration=60.0, enable_enrichment=True)
spinner = YouTubeSpinner(config)
shards = await spinner.spin({'url': 'VIDEO_ID', 'languages': ['en']})
```

### Tuning Exploration Strategy
Change bandit strategy when creating policy:
```python
from holoLoom.policy.unified import BanditStrategy
policy = create_policy(
    mem_dim=384,
    emb=emb,
    scales=[96, 192, 384],
    bandit_strategy=BanditStrategy.BAYESIAN_BLEND,
    epsilon=0.15  # 15% exploration for epsilon-greedy
)
```

### Tufte-Style Visualizations (October 2025)

HoloLoom implements Edward Tufte's visualization principles: **"Above all else show the data."**

**Philosophy**: Maximize information density, minimize decoration ("chartjunk"). Show meaning first.

**Available Visualizations**:

1. **Small Multiples** (`HoloLoom/visualization/small_multiples.py`)
   - Compare multiple queries side-by-side
   - Consistent scales for fair comparison
   - Highlights best/worst automatically (★ and ⚠)
   - Inline sparklines show trends
   - Usage:
   ```python
   from HoloLoom.visualization.small_multiples import render_small_multiples

   queries = [
       {'query_text': 'Query A', 'latency_ms': 95, 'confidence': 0.92,
        'threads_count': 3, 'cached': True, 'trend': [105, 102, 98, 96, 95],
        'timestamp': 1698595200.0, 'tool_used': 'answer'},
       # ... more queries
   ]
   html = render_small_multiples(queries, layout='grid', max_columns=4)
   ```

2. **Data Density Tables** (`HoloLoom/visualization/density_table.py`)
   - Maximum information per square inch
   - Inline sparklines, delta indicators, bottleneck detection
   - Tight spacing, small fonts, monospace numbers
   - Usage:
   ```python
   from HoloLoom.visualization.density_table import render_stage_timing_table

   stages = [
       {'name': 'Retrieval', 'duration_ms': 50.5,
        'trend': [45, 47, 48, 50, 50.5], 'delta': +2.5},
       # ... more stages
   ]
   html = render_stage_timing_table(stages, total_duration=150.0)
   ```

3. **Tufte Sparklines** (enhanced in `html_renderer.py`)
   - Word-sized graphics (100x30px)
   - Show trends inline with metrics
   - Auto-normalization, endpoint indicators
   - See [TUFTE_SPARKLINES_PHASE_2_1_COMPLETE.md](TUFTE_SPARKLINES_PHASE_2_1_COMPLETE.md) for details

4. **Stage Waterfall Charts** (`HoloLoom/visualization/stage_waterfall.py`)
   - Sequential pipeline timing with horizontal stacked bars
   - Automatic bottleneck detection (stages >40% of total time)
   - Status indicators (success, warning, error, skipped)
   - Inline sparklines for historical trends
   - Parallel execution visualization support
   - Usage:
   ```python
   from HoloLoom.visualization.stage_waterfall import render_pipeline_waterfall

   # After weaving
   spacetime = await orchestrator.weave(query)

   # Render waterfall from trace
   html = render_pipeline_waterfall(
       spacetime.trace.stage_durations,
       stage_trends=historical_trends,  # Optional
       title=f"Pipeline: {query.text[:50]}"
   )

   # Or create custom stages
   from HoloLoom.visualization.stage_waterfall import WaterfallStage, StageStatus

   stages = [
       WaterfallStage(name='Retrieval', start_ms=0, duration_ms=50.5,
                      status=StageStatus.SUCCESS, trend=[45, 47, 48, 50, 50.5]),
       WaterfallStage(name='Decision', start_ms=50.5, duration_ms=30.0)
   ]
   html = renderer.render(stages)
   ```

5. **Confidence Trajectory** (`HoloLoom/visualization/confidence_trajectory.py`)
   - Time series confidence tracking with anomaly detection
   - Automatic anomaly detection (4 types: sudden drop, prolonged low, high variance, cache miss cluster)
   - Cache effectiveness visualization (hit/miss markers)
   - Statistical context (mean ± std bands, trend analysis)
   - Comprehensive programmatic API for automated tool calling
   - Usage:
   ```python
   from HoloLoom.visualization.confidence_trajectory import render_confidence_trajectory

   # Simple usage - just confidence scores
   confidences = [0.92, 0.88, 0.65, 0.87, 0.91]
   html = render_confidence_trajectory(confidences)

   # With cache markers
   cached = [True, True, False, False, True]
   html = render_confidence_trajectory(confidences, cached=cached)

   # Complete usage with anomaly detection
   query_texts = [
       "What is Thompson Sampling?",
       "How does it work?",
       "Show me an example",
       "What are the tradeoffs?",
       "How to implement?"
   ]
   html = render_confidence_trajectory(
       confidences,
       cached=cached,
       query_texts=query_texts,
       title='Session Analysis',
       subtitle='User session from 2025-10-29',
       detect_anomalies=True
   )

   # Integration with HoloLoom
   results = []
   for query in queries:
       spacetime = await orchestrator.weave(query)
       results.append(spacetime)

   # Extract and visualize
   confidences = [s.confidence for s in results]
   cached = [s.metadata.get('cache_hit', False) for s in results]
   html = render_confidence_trajectory(confidences, cached=cached)
   ```

   **Anomaly Types**:
   - SUDDEN_DROP: Confidence drops >0.2 in single step (red markers)
   - PROLONGED_LOW: Confidence <threshold for >3 consecutive queries (amber markers)
   - HIGH_VARIANCE: Std dev >0.15 in rolling window (amber markers)
   - CACHE_MISS_CLUSTER: 3+ cache misses in rolling window (indigo markers)

   **API Reference**: See [CONFIDENCE_TRAJECTORY_API.md](HoloLoom/visualization/CONFIDENCE_TRAJECTORY_API.md) (1000+ lines comprehensive documentation)

6. **Cache Effectiveness Gauge** (`HoloLoom/visualization/cache_gauge.py`)
   - Radial gauge showing cache performance metrics
   - 5 effectiveness ratings (excellent, good, fair, poor, critical)
   - Performance metrics (hit rate, latencies, time saved, speedup)
   - Actionable recommendations based on performance
   - Simple programmatic API for monitoring dashboards
   - Usage:
   ```python
   from HoloLoom.visualization.cache_gauge import render_cache_gauge

   # Simple usage
   html = render_cache_gauge(
       hit_rate=0.75,
       total_queries=100,
       cache_hits=75
   )

   # Complete usage with all parameters
   html = render_cache_gauge(
       hit_rate=0.75,
       total_queries=100,
       cache_hits=75,
       avg_cached_latency_ms=15.0,
       avg_uncached_latency_ms=120.0,
       title='Production Cache Performance',
       subtitle='Last 24 hours',
       show_details=True,
       show_recommendations=True
   )

   # Integration with HoloLoom
   total = 0
   hits = 0

   for query in queries:
       spacetime = await orchestrator.weave(query)
       total += 1
       if spacetime.metadata.get('cache_hit'):
           hits += 1

   html = render_cache_gauge(
       hit_rate=hits / total,
       total_queries=total,
       cache_hits=hits
   )
   ```

   **Effectiveness Ratings**:
   - EXCELLENT (Green): Hit rate >80%, speedup >4x
   - GOOD (Light Green): Hit rate 60-80%, speedup >2x
   - FAIR (Amber): Hit rate 40-60% or speedup >2x
   - POOR (Red): Hit rate 20-40%, low speedup
   - CRITICAL (Dark Red): Hit rate <20%

7. **Knowledge Graph Network** (`HoloLoom/visualization/knowledge_graph.py`)
   - Force-directed graph layout (Fruchterman-Reingold algorithm)
   - Node sizing by degree/importance (8-24px)
   - Semantic edge type colors (7 relationship types)
   - Interactive tooltips with node details
   - Path highlighting for reasoning chains
   - Direct integration with HoloLoom.memory.graph.KG
   - Zero dependencies (pure HTML/CSS/SVG)
   - Usage:
   ```python
   from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_kg
   from HoloLoom.memory.graph import KG, KGEdge

   # Create knowledge graph
   kg = KG()
   kg.add_edges([
       KGEdge("attention", "transformer", "USES", 1.0),
       KGEdge("transformer", "neural_network", "IS_A", 1.0),
       KGEdge("BERT", "transformer", "IS_A", 1.0),
       KGEdge("GPT", "transformer", "IS_A", 1.0)
   ])

   # Render network
   html = render_knowledge_graph_from_kg(
       kg,
       title="Transformer Architecture",
       subtitle="Entity relationships in neural network domain"
   )

   # With path highlighting (for reasoning chains)
   highlighted_path = ["query", "retrieval", "reasoning", "decision"]
   html = render_knowledge_graph_from_kg(
       kg,
       title="Reasoning Pipeline",
       highlighted_path=highlighted_path
   )

   # Also supports NetworkX MultiDiGraph directly
   from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_networkx
   import networkx as nx

   G = nx.MultiDiGraph()
   # ... add edges ...
   html = render_knowledge_graph_from_networkx(G, title="My Graph")
   ```

   **Edge Types with Semantic Colors**:
   - IS_A (Blue): Taxonomy relationships
   - USES (Green): Functional relationships
   - MENTIONS (Gray): Reference relationships
   - LEADS_TO (Orange): Causal relationships
   - PART_OF (Purple): Composition relationships
   - IN_TIME (Cyan): Temporal relationships
   - OCCURRED_AT (Teal): Event relationships

   **Force-Directed Layout**:
   - Repulsion: All nodes repel (inverse square law)
   - Attraction: Connected nodes attract (spring force)
   - Cooling: Gradual stabilization over 300 iterations
   - Natural clustering of related entities

**Demos**: See `demos/output/tufte_advanced_demo.html`, `demos/output/stage_waterfall_demo.html`, `demos/output/confidence_trajectory_demo.html`, `demos/output/cache_gauge_demo.html`, and `demos/output/knowledge_graph_demo.html`
**Roadmap**: [TUFTE_VISUALIZATION_ROADMAP.md](TUFTE_VISUALIZATION_ROADMAP.md) (600+ lines, 8 phases planned)
**Tests**: `test_tufte_advanced.py` (5/5 passing), `test_stage_waterfall.py` (7/7 passing), `test_confidence_trajectory.py` (9/9 passing), `test_cache_gauge.py` (8/8 passing), `test_knowledge_graph.py` (10/10 passing)

**Key Principles**:
- Maximize data-ink ratio (~60-70% vs ~30% traditional)
- Small multiples enable comparison
- High data density (16-24x more visible data)
- Meaning first (critical info highlighted)
- Zero external dependencies (pure HTML/CSS/SVG)
