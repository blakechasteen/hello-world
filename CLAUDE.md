# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
