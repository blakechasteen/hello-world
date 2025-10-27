# HoloLoom AI Agent Instructions

HoloLoom is a neural decision-making system built around a **weaving metaphor** - independent "warp thread" modules are coordinated by orchestrators to produce responses with full computational provenance.

## Quick Start for AI Agents

**First time here?** Start with these essentials:

1. **PYTHONPATH is critical**: Always run `$env:PYTHONPATH = "."; python ...` from repo root
2. **Architecture**: Read the 9-step weaving cycle below - it's the system's core abstraction
3. **Entry points**: Use `HoloLoom.weaving_orchestrator` or `HoloLoom.unified_api` for most tasks
4. **Testing**: Run `$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py` to verify setup
5. **Demos**: Check `demos/01_quickstart.py` for working examples

**Common first tasks:**
- Adding a feature? Check "Extending Tool Selection" or "Adding a New Spinner"
- Debugging? Jump to "Debugging Workflows" section
- Understanding architecture? Read "Architecture Philosophy" and "9-Step Weaving Cycle"
- Production deployment? See "Production Deployment" section

## Architecture Philosophy

### The Weaving Metaphor (First-Class Design)
- **Yarn Graph**: Discrete symbolic memory (NetworkX MultiDiGraph)
- **Warp Space**: Continuous tensor field where threads are "tensioned" for computation
- **DotPlasma**: Flowing feature representation between extraction and decision
- **Spacetime**: 4D woven output (3D semantic + 1D temporal) with complete lineage
- **Shuttle**: The orchestrator that weaves threads together

### Protocol-Based Design
Components define protocols (abstract interfaces) for swappable implementations:
- `PolicyEngine` for decision making
- `KGStore` for knowledge graphs  
- `Retriever` for memory systems

**Never** have modules import from each other - only from shared types in `HoloLoom/Documentation/types.py`. The orchestrator is the ONLY module that imports from all others.

### 9-Step Weaving Cycle
1. **LoomCommand** - Selects pattern card (BARE/FAST/FUSED)
2. **ChronoTrigger** - Creates temporal window for thread selection
3. **Yarn Graph** - Selects memory threads based on temporal window
4. **ResonanceShed** - Extracts features (motifs, embeddings, spectral) → DotPlasma
5. **Warp Space** - Tensions discrete threads into continuous manifold
6. **Memory Retrieval** - Gathers context for policy input
7. **ConvergenceEngine** - Collapses probabilities to discrete tool selection
8. **Tool Execution** - Executes selected tool
9. **Spacetime Fabric** - Weaves output with complete trace

## Critical Development Patterns

### PYTHONPATH Requirements
**ALWAYS** set `PYTHONPATH=.` when running from repository root:

```powershell
# Windows PowerShell
$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py

# Testing
$env:PYTHONPATH = "."; python HoloLoom/weaving_orchestrator.py

# Demos
$env:PYTHONPATH = "."; python demos/01_quickstart.py
```

The codebase uses absolute imports like `from HoloLoom.policy.unified import ...` which require the repo root on the Python path.

### Async/Await Pattern
The orchestrator uses `async/await` throughout:
- Feature extraction and retrieval run concurrently
- Spinners (input adapters) are async
- Memory operations are async
- Tool execution is async

```python
# Correct async pattern
result = await orchestrator.weave(query)
shards = await spinner.spin(raw_data)
context = await memory.retrieve(query)

# Never block on async operations
```

### Graceful Degradation
Optional dependencies degrade gracefully with warnings, **never crash**:
- spaCy → regex-only motif detection
- sentence-transformers → fallback embeddings
- scipy → skip spectral features
- Neo4j → in-memory NetworkX graph

```python
# Check optional availability before using
if HAS_SPACY:
    # Use spaCy NLP
else:
    # Fall back to regex
```

### Multi-Scale Embeddings (Matryoshka)
Embeddings operate at 3 scales: 96d (coarse), 192d (medium), 384d (fine)
- **BARE** mode: Single scale (96d)
- **FAST** mode: Two scales (96d, 192d)  
- **FUSED** mode: All three scales (96d, 192d, 384d)

```python
# Access embeddings at different scales
emb_96 = embedder.encode(text, output_dim=96)
emb_384 = embedder.encode(text, output_dim=384)

# Multi-scale fusion for retrieval
scales = [96, 192, 384]
fused_features = spectral_fusion.fuse_scales(embeddings, scales)
```

## Key Components

### Configuration (`HoloLoom/config.py`)
Three execution modes via factory methods:
```python
from HoloLoom.config import Config

cfg_bare = Config.bare()    # ~50ms, minimal processing
cfg_fast = Config.fast()    # ~150ms, balanced (default)
cfg_fused = Config.fused()  # ~300ms, full processing
```

### Memory Backends (`HoloLoom/memory/`)
Configurable via `MemoryBackend` enum:
- **NETWORKX**: In-memory (fast prototyping)
- **NEO4J**: Persistent graph (production)
- **QDRANT**: Vector similarity search
- **NEO4J_QDRANT**: Hybrid graph + vector (common production)
- **HYPERSPACE**: Gated multipass with importance filtering

```python
# Configure backend
config.memory_backend = MemoryBackend.NEO4J_QDRANT
```

### SpinningWheel (Input Adapters)
Converters that transform raw data → `MemoryShard` objects:
- **TextSpinner**: Plain text
- **WebsiteSpinner**: Web scraping
- **YouTubeSpinner**: Video transcripts with chunking
- **AudioSpinner**: Transcripts, task lists, summaries

```python
# All spinners follow same pattern
shards = await spinner.spin(raw_data)
# Returns List[MemoryShard] ready for orchestrator
```

### Policy Engine (`HoloLoom/policy/unified.py`)
Neural decision-making with Thompson Sampling:
- **Epsilon-Greedy**: 90% neural exploit, 10% Thompson explore (default)
- **Bayesian Blend**: 70% neural + 30% bandit priors
- **Pure Thompson**: 100% Thompson Sampling exploration

```python
from HoloLoom.policy.unified import BanditStrategy

policy = create_policy(
    bandit_strategy=BanditStrategy.EPSILON_GREEDY,
    epsilon=0.10
)
```

### Synthesis Bridge (`HoloLoom/synthesis_bridge.py`)
Integrates pattern extraction into weaving cycle:
- Entity extraction from queries
- Reasoning type detection
- Pattern mining
- **Integration point**: Between ResonanceShed and WarpSpace

## Common Workflows

### Running Tests
```powershell
# Full test suite (18 tests)
$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py

# Integration tests
$env:PYTHONPATH = "."; python HoloLoom/chatops/test_bot_simple.py

# Validate pipeline
$env:PYTHONPATH = "."; python HoloLoom/validate_pipeline.py
```

### Training PPO Agent
```powershell
$env:PYTHONPATH = "."; python -c "from HoloLoom.train_agent import PPOTrainer; t=PPOTrainer(env_name='CartPole-v1', total_timesteps=2000, steps_per_update=256, n_epochs=1, batch_size=32, log_dir='./logs/test_run'); t.train()"
```

### Running Demos
```powershell
# Quickstart
$env:PYTHONPATH = "."; python demos/01_quickstart.py

# Web ingestion
$env:PYTHONPATH = "."; python demos/02_web_to_memory.py

# Conversational
$env:PYTHONPATH = "."; python demos/03_conversational.py

# Complete weaving
$env:PYTHONPATH = "."; python demos/complete_weaving_demo.py
```

### Using the Unified API
```python
from HoloLoom.unified_api import HoloLoom

# One-shot query
loom = await HoloLoom.create(pattern="fast")
result = await loom.query("What is HoloLoom?")
print(f"Response: {result.response}")

# Conversational
await loom.chat("What is the weaving metaphor?")
await loom.chat("Tell me more")

# Data ingestion
await loom.ingest_text("Knowledge base content...")
await loom.ingest_web("https://example.com")
await loom.ingest_youtube("VIDEO_ID", languages=['en'])
```

### Adding a New Spinner
```python
from HoloLoom.spinningWheel.base import BaseSpinner
from HoloLoom.Documentation.types import MemoryShard

class CustomSpinner(BaseSpinner):
    async def spin(self, raw_data: Dict) -> List[MemoryShard]:
        # Parse raw_data format
        # Extract entities/motifs
        # Return standardized MemoryShard objects
        return shards
```

### Extending Tool Selection
1. Add tool name to `NeuralCore.tools` list in `policy/unified.py`
2. Update `n_tools` parameter in config
3. Implement execution logic in orchestrator's tool executor
4. Retrain policy or adjust weights

## File Organization

```
HoloLoom/
├── weaving_orchestrator.py    # Main weaving cycle (imports all modules)
├── config.py                  # Execution modes (BARE/FAST/FUSED)
├── policy/unified.py          # Neural core + Thompson Sampling
├── embedding/spectral.py      # Multi-scale Matryoshka embeddings
├── memory/                    # Memory backends (protocol-based)
│   ├── cache.py              # Vector/BM25 retrieval
│   ├── graph.py              # NetworkX KG
│   ├── neo4j_graph.py        # Neo4j persistent KG
│   └── stores/               # Backend implementations
├── motif/base.py             # Pattern detection (regex + NLP)
├── spinningWheel/            # Input adapters
│   ├── audio.py
│   ├── youtube.py
│   └── base.py
├── loom/command.py           # Pattern card selection
├── chrono/trigger.py         # Temporal control
├── resonance/shed.py         # Feature extraction
├── warp/space.py             # Tensioned manifold
├── convergence/engine.py     # Decision collapse
├── fabric/spacetime.py       # Output with trace
├── synthesis_bridge.py       # Pattern extraction integration
└── Documentation/types.py    # Shared types (Query, Context, Features)

demos/                        # Working examples
tests/                        # Test suites
```

## Important Conventions

### Import Order
1. Standard library
2. Third-party packages
3. HoloLoom modules (absolute imports)
4. Optional dependencies with try/except and graceful fallback

### Error Handling
- Return `Spacetime` objects with error details instead of raising exceptions
- Log warnings for degraded functionality
- Never crash due to missing optional dependencies

### Type Hints
Use comprehensive type hints with protocols:
```python
from typing import Protocol, Optional, List
from HoloLoom.Documentation.types import Query, Context, Features

async def process(query: Query) -> Spacetime:
    ...
```

### Docstrings
Use NumPy-style docstrings with clear sections:
```python
def weave(query: Query) -> Spacetime:
    """
    Execute complete weaving cycle.
    
    Parameters
    ----------
    query : Query
        Input query object
        
    Returns
    -------
    Spacetime
        Woven fabric with complete trace
    """
```

### Code Patterns to Avoid

**❌ DON'T: Cross-module imports**
```python
# BAD - modules importing from each other
from HoloLoom.policy.unified import PolicyEngine
from HoloLoom.memory.cache import MemoryManager
```

**✅ DO: Import only from types**
```python
# GOOD - import from shared types
from HoloLoom.Documentation.types import Query, Context, Features
```

**❌ DON'T: Await synchronous functions**
```python
# BAD - embedder.encode() is not async
embedding = await embedder.encode(text)
```

**✅ DO: Check function signatures**
```python
# GOOD - embedder.encode() is synchronous
embedding = embedder.encode(text)
```

**❌ DON'T: Hardcode execution modes**
```python
# BAD - hardcoded scales
embedder = MatryoshkaEmbeddings(scales=[96, 192, 384])
```

**✅ DO: Use pattern specs**
```python
# GOOD - derive from pattern card
pattern_spec = loom_command.select_pattern(query)
embedder = MatryoshkaEmbeddings(scales=pattern_spec.scales)
```

### Module Interaction Rules

**Orchestrator (Hub)**: Can import from ALL modules
**Feature Modules (Spokes)**: Can ONLY import from `Documentation/types.py`
**Shared Types**: Never imports from feature modules

```
           ┌─────────────────────────────┐
           │   weaving_orchestrator.py   │
           │    (ONLY hub module)        │
           └─────────────┬───────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ policy/ │    │ memory/ │    │ warp/   │
    │         │    │         │    │         │
    └────┬────┘    └────┬────┘    └────┬────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                ┌────────▼────────┐
                │ Documentation/  │
                │    types.py     │
                │  (shared only)  │
                └─────────────────┘
```

## Known Issues & Gotchas

1. **Background Tasks**: MemoryManager spawns fire-and-forget tasks without shutdown hooks - use explicit lifecycle management
2. **ResonanceShed**: `embedder.encode()` is synchronous, not async - don't await it
3. **Thread Count**: Extract from `dot_plasma['threads']` before calling `shed.lower()` which clears threads
4. **Bandit Updates**: Always update bandit statistics for the actually-selected tool, not predicted tool
5. **Embedding Dimensions**: Create pattern-specific embedders for each query to match scale configuration

## Production Deployment

### Docker Setup
```bash
# Start backends
cd config
docker-compose up -d neo4j qdrant

# Verify connectivity
python HoloLoom/test_backends.py
```

### Environment Variables
```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Qdrant
QDRANT_URL=http://localhost:6333

# Matrix bot
MATRIX_HOMESERVER=https://matrix.org
MATRIX_USER=@bot:matrix.org
MATRIX_PASSWORD=bot_password
```

## Testing Strategy

Tests validate components in isolation then integration:
1. Building blocks (MLP, attention)
2. Curiosity modules (ICM, RND)  
3. Policy variants (deterministic, categorical, gaussian)
4. PPO agent (GAE, updates, checkpointing)
5. Full end-to-end weaving pipeline

Run tests before committing changes:
```powershell
$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py
```

## Resources

- **CLAUDE.md**: Complete developer guide with all architecture details
- **README.md**: Project overview and quick start
- **QUICKSTART.md**: Getting started with Terminal UI and Matrix bot
- **WEAVING_ARCHITECTURE_COMPLETE.md**: Implementation details of weaving cycle
- **demos/**: Working examples of all major features

## Quick Reference

**Three execution modes:**
- `Config.bare()` - Fast, minimal (50ms)
- `Config.fast()` - Balanced, default (150ms)
- `Config.fused()` - Full quality (300ms)

**Three bandit strategies:**
- `EPSILON_GREEDY` - 90% exploit, 10% explore
- `BAYESIAN_BLEND` - 70% neural + 30% bandit
- `PURE_THOMPSON` - 100% Thompson Sampling

**Five memory backends:**
- `NETWORKX` - In-memory, fast prototyping
- `NEO4J` - Persistent graph
- `QDRANT` - Vector similarity
- `NEO4J_QDRANT` - Hybrid (most common)
- `HYPERSPACE` - Gated multipass

**Critical reminder:** Always use `PYTHONPATH=.` from repository root!

## Related Components

### Promptly Framework (`Promptly/`)
Meta-prompt composition and testing framework integrated with HoloLoom:
- **Core**: Prompt chaining, loop composition, recursive loops
- **Tools**: LLM-as-judge, A/B testing, cost tracking, analytics
- **Integrations**: HoloLoom bridge (`integrations/hololoom_bridge.py`), MCP server
- **Skill System**: Package manager for reusable prompt skills

```python
from Promptly.promptly.integrations.hololoom_bridge import HoloLoomBridge

# Bridge Promptly skills to HoloLoom
bridge = HoloLoomBridge(hololoom_config="fast")
result = await bridge.execute_skill("analyze_code", code_text)
```

### Matrix Bot / ChatOps (`HoloLoom/chatops/`)
Production-ready Matrix bot with full HoloLoom integration:

**Core Commands:**
- `!weave <query>` - Execute full weaving cycle with MCTS
- `!memory add <text>` - Add to knowledge base
- `!memory search <query>` - Semantic search
- `!memory stats` - Memory statistics
- `!analyze <text>` - MCTS analysis
- `!stats` - System statistics
- `!help` - Command help

**Deployment:**
```powershell
# Environment variables
$env:MATRIX_HOMESERVER = "https://matrix.org"
$env:MATRIX_USER = "@bot:matrix.org"
$env:MATRIX_PASSWORD = "password"

# Run bot
$env:PYTHONPATH = "."; python HoloLoom/chatops/run_bot.py --hololoom-mode fast
```

**Key Files:**
- `run_bot.py` - Main launcher
- `handlers/hololoom_handlers.py` - Command handlers
- `docs/README.md` - Full documentation

### mythRL_core (`mythRL_core/`)
Domain-specific RL modules (automotive, entity resolution, summarization):
- Specialized environments for training
- Domain adapters for HoloLoom policy
- Pre-trained models for specific tasks

## Debugging Workflows

### Quick Troubleshooting Index

| Error Message / Symptom | Section | Quick Fix |
|------------------------|---------|-----------|
| `ModuleNotFoundError: No module named 'HoloLoom'` | Import Errors | Set `$env:PYTHONPATH = "."` |
| `RuntimeWarning: coroutine was never awaited` | Async/Await Bugs | Add `await` before async calls |
| `TypeError: object async_generator can't be used in 'await'` | Async/Await Bugs | Remove `await` (embedder.encode is sync) |
| Neo4j/Qdrant connection failed | Backend Connection | Run `docker-compose up -d` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Dimension Mismatch | Match embedder scales to policy config |
| No results from memory search | Empty Memory Results | Verify embeddings exist on shards |
| Bot always picks same tool | Thompson Sampling | Increase epsilon or use PURE_THOMPSON |
| Docker compose exit code 1 | Backend Connection | Check `docker-compose logs` for errors |

### Common Issues & Solutions

#### 1. Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'HoloLoom'`
**Solution**:
```powershell
# Always set PYTHONPATH from repo root
cd C:\Users\blake\Documents\mythRL
$env:PYTHONPATH = "."
python HoloLoom/your_script.py
```

#### 2. Async/Await Bugs
**Symptom**: `RuntimeWarning: coroutine was never awaited`
**Solution**:
```python
# WRONG - Missing await
result = orchestrator.weave(query)

# CORRECT - Await async functions
result = await orchestrator.weave(query)
```

**Symptom**: `TypeError: object async_generator can't be used in 'await'`
**Solution**:
```python
# WRONG - Awaiting synchronous embedder
embedding = await embedder.encode(text)

# CORRECT - embedder.encode() is synchronous
embedding = embedder.encode(text)
```

#### 3. Memory Backend Connection Failures
**Symptom**: Neo4j or Qdrant connection errors
**Solution**:
```powershell
# Start Docker backends
cd HoloLoom
docker-compose up -d

# Verify services running
docker-compose ps

# Check logs for errors
docker-compose logs -f neo4j
docker-compose logs -f qdrant

# Test connection
$env:PYTHONPATH = "."; python HoloLoom/test_backends.py
```

**Graceful fallback**: System automatically falls back to in-memory if backends unavailable

#### 4. Dimension Mismatch Errors
**Symptom**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
**Solution**:
```python
# WRONG - Using config scales with different embedder
embedder = MatryoshkaEmbeddings(scales=[96])
policy = create_policy(scales=[96, 192, 384])  # Mismatch!

# CORRECT - Match embedder scales to policy
pattern_spec = loom_command.select_pattern(query)
embedder = MatryoshkaEmbeddings(scales=pattern_spec.scales)
policy = create_policy(scales=pattern_spec.scales)
```

#### 5. Empty Results from Memory
**Symptom**: No context retrieved despite having data
**Solution**:
```python
# Check memory stats
stats = await memory.get_stats()
print(f"Total memories: {stats['total']}")

# Verify embeddings are being created
shards = await spinner.spin(text)
for shard in shards:
    assert len(shard.embedding) > 0, "Missing embedding!"

# Check retrieval directly
results = await memory.search(query="test", limit=10)
print(f"Retrieved: {len(results)} results")
```

#### 6. Thompson Sampling Not Exploring
**Symptom**: Bot always selects same tool
**Solution**:
```python
# Check bandit statistics
stats = policy.bandit.get_stats()
for tool, (alpha, beta, mean) in stats.items():
    print(f"{tool}: α={alpha:.1f}, β={beta:.1f}, mean={mean:.3f}")

# Increase exploration with epsilon-greedy
policy = create_policy(
    bandit_strategy=BanditStrategy.EPSILON_GREEDY,
    epsilon=0.20  # 20% exploration
)

# Or use pure Thompson for max exploration
policy = create_policy(
    bandit_strategy=BanditStrategy.PURE_THOMPSON
)
```

### Debugging Commands

```powershell
# Run specific test
$env:PYTHONPATH = "."; python -m pytest HoloLoom/tests/test_policy.py::test_epsilon_greedy -v

# Enable debug logging
$env:PYTHONPATH = "."; python -c "import logging; logging.basicConfig(level=logging.DEBUG); from HoloLoom.weaving_orchestrator import main; import asyncio; asyncio.run(main())"

# Check Docker services
docker-compose ps
docker-compose logs --tail=50 neo4j
docker-compose logs --tail=50 qdrant

# Test memory backend connectivity
$env:PYTHONPATH = "."; python HoloLoom/test_backends.py

# Validate complete pipeline
$env:PYTHONPATH = "."; python HoloLoom/validate_pipeline.py

# Run Matrix bot in test mode
$env:PYTHONPATH = "."; python HoloLoom/chatops/test_bot_simple.py
```

### Performance Profiling

```python
import time
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query

async def profile_weaving():
    orchestrator = WeavingOrchestrator(config=Config.fast())
    
    start = time.perf_counter()
    result = await orchestrator.weave(Query(text="Test query"))
    duration = time.perf_counter() - start
    
    # Check trace for stage timings
    trace = result.trace
    print(f"Total: {duration*1000:.1f}ms")
    print(f"  Pattern selection: {trace.pattern_selection_ms:.1f}ms")
    print(f"  Feature extraction: {trace.feature_extraction_ms:.1f}ms")
    print(f"  Memory retrieval: {trace.memory_retrieval_ms:.1f}ms")
    print(f"  Decision: {trace.decision_ms:.1f}ms")
    print(f"  Tool execution: {trace.tool_execution_ms:.1f}ms")
```

## Advanced Architecture Details

### Discrete ↔ Continuous Transformations

The weaving metaphor enables seamless transitions:

**Discrete (Yarn Graph)**:
- Entities, relationships, motifs
- Symbolic knowledge representation
- NetworkX MultiDiGraph storage

**Continuous (Warp Space)**:
- Tensor operations on activated threads
- Multi-scale embeddings (96d/192d/384d)
- Spectral features from graph topology

**Lifecycle**:
```python
# 1. Start with discrete threads
threads = yarn_graph.select_threads(temporal_window)

# 2. Tension into continuous manifold
warp_space = WarpSpace(threads)
await warp_space.tension(threads, yarn_graph.shards)

# 3. Perform continuous operations
features = warp_space.compute_features()

# 4. Collapse back to discrete
collapse_result = convergence.collapse(neural_probs)
warp_updates = warp_space.collapse()

# 5. Update discrete memory
yarn_graph.apply_updates(warp_updates)
```

### Thompson Sampling Integration

Bayesian exploration with Beta distributions:

```python
class ThompsonBandit:
    def __init__(self, n_tools: int):
        # Beta(α, β) for each tool
        self.alpha = np.ones(n_tools)  # Successes + 1
        self.beta = np.ones(n_tools)   # Failures + 1
    
    def sample(self) -> np.ndarray:
        # Sample from Beta distributions
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, tool_idx: int, reward: float):
        # Update based on outcome
        if reward > 0.5:
            self.alpha[tool_idx] += 1  # Success
        else:
            self.beta[tool_idx] += 1   # Failure
```

**Three strategies**:
1. **Epsilon-Greedy**: `sampled_tool = thompson_sample() if rand() < ε else neural_argmax()`
2. **Bayesian Blend**: `tool_probs = 0.7 * neural_probs + 0.3 * thompson_priors()`
3. **Pure Thompson**: `tool_probs = thompson_sample()` (ignores neural network)

### MCTS Flux Capacitor

Monte Carlo Tree Search for decision exploration:

```python
class MCTSConvergenceEngine:
    def __init__(self, n_simulations: int = 100):
        self.n_simulations = n_simulations
    
    def collapse(self, features, context) -> CollapseResult:
        # Build decision tree
        root = self._build_tree(features, context)
        
        # Run simulations
        for _ in range(self.n_simulations):
            node = self._select(root)  # UCB1 selection
            reward = self._simulate(node)  # Rollout
            self._backpropagate(node, reward)  # Update
        
        # Select best action
        best_tool = max(root.children, key=lambda n: n.visit_count)
        return CollapseResult(tool=best_tool.action, confidence=...)
```

**UCB1 Formula**: `score = mean_reward + C * sqrt(ln(parent_visits) / node_visits)`

### Memory Architecture

**Hybrid Backend (Neo4j + Qdrant)**:
```python
class HybridMemoryStore:
    def __init__(self):
        self.neo4j = Neo4jMemoryStore()  # Graph relationships
        self.qdrant = QdrantMemoryStore()  # Vector search
    
    async def add(self, shard: MemoryShard):
        # Store in both backends
        graph_id = await self.neo4j.add_node(shard)
        vector_id = await self.qdrant.add_vector(shard.embedding, shard)
        
        # Link IDs for consistency
        await self._link_stores(graph_id, vector_id)
    
    async def search(self, query: str, limit: int) -> List[MemoryShard]:
        # Vector search for candidates
        candidates = await self.qdrant.search(query, limit=limit*2)
        
        # Graph expansion for context
        expanded = await self.neo4j.expand_subgraph(candidates)
        
        # Rank and return
        return self._rank_results(expanded, query)[:limit]
```

### Spacetime Provenance

Every output includes complete computational lineage:

```python
@dataclass
class Spacetime:
    response: str
    tool_used: str
    confidence: float
    trace: WeavingTrace
    
@dataclass  
class WeavingTrace:
    # Pattern selection
    pattern_card: PatternCard
    pattern_selection_ms: float
    
    # Feature extraction
    motifs_detected: List[str]
    embedding_scales_used: List[int]
    spectral_features: Dict[str, float]
    feature_extraction_ms: float
    
    # Memory
    threads_activated: List[str]
    context_shards_retrieved: int
    memory_retrieval_ms: float
    
    # Decision
    neural_predictions: Dict[str, float]
    bandit_samples: Dict[str, float]
    collapse_strategy: str
    mcts_simulations: Optional[int]
    decision_ms: float
    
    # Execution
    tool_result: Dict[str, Any]
    tool_execution_ms: float
    
    # Totals
    duration_ms: float
    timestamp: datetime
```

Use traces for:
- **Debugging**: See exactly what happened at each stage
- **Optimization**: Identify bottlenecks
- **Learning**: Analyze successful vs failed decisions
- **Explainability**: Show users why decisions were made

## Summary

HoloLoom is a production-ready neural decision-making system with:
- Complete weaving metaphor architecture (9 stages)
- Multi-scale embeddings with protocol-based design
- Hybrid memory (graph + vector)
- Thompson Sampling and MCTS decision-making
- Full computational provenance via Spacetime
- Matrix bot for ChatOps
- Promptly integration for meta-prompting

**Start here**: Run demos, review CLAUDE.md, check Spacetime traces!

**Critical reminder:** Always use `PYTHONPATH=.` from repository root!
