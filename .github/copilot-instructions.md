# mythRL AI Agent Instructions

mythRL is a revolutionary neural decision-making system built around a **Shuttle-centric architecture** - where the Shuttle contains creative orchestration intelligence that progressively scales from simple queries to research-grade analysis.

## Quick Start for AI Agents

**First time here?** Start with these essentials:

1. **PYTHONPATH is critical**: Always run `$env:PYTHONPATH = "."; python ...` from repo root
2. **Architecture**: Read the Shuttle-centric design below - it's the system's core innovation
3. **Entry points**: Use `MythRLShuttle` for new architecture or legacy `HoloLoom.unified_api` for compatibility
4. **Testing**: Run `$env:PYTHONPATH = "."; python dev/protocol_modules_mythrl.py` to verify new architecture
5. **Demos**: Check `multipass_memory_demo.py` for advanced memory crawling examples

**Common first tasks:**
- Understanding new architecture? Read "Shuttle-Centric Architecture" and "3-5-7-9 Progressive Complexity"
- Adding features? Check "Protocol + Modules = mythRL" pattern
- Memory exploration? See "Recursive Gated Multipass Memory Crawling"
- Legacy compatibility? Use existing HoloLoom components with adapter patterns

## Revolutionary Architecture: Shuttle-Centric Design

### The Shuttle: Creative Orchestrator (NEW)
The **Shuttle** contains internal creative intelligence, not just coordination:
- **Synthesis Bridge**: Pattern integration across modules (internal logic)
- **Temporal Windows**: Time-aware processing coordination (internal logic)
- **Spacetime Tracing**: Full computational provenance (internal logic)
- **Routing System**: Intelligent module activation (internal logic)
- **Multipass Memory Crawling**: Recursive gated knowledge exploration (internal logic)

### Protocol + Modules = mythRL
- **Protocols**: Clean interface contracts (swappable implementations)
- **Modules**: Domain-specific implementations that grow in sophistication
- **Shuttle**: Creative orchestrator with internal intelligence

### 3-5-7-9 Progressive Complexity System
```
LITE (3 steps):    Extract → Route → Execute                    <50ms
FAST (5 steps):    + Pattern Selection + Temporal Windows      <150ms
FULL (7 steps):    + Decision Engine + Synthesis Bridge        <300ms
RESEARCH (9 steps): + Advanced WarpSpace + Full Tracing        No Limit
```

## Recursive Gated Multipass Memory Crawling

### Revolutionary Memory Intelligence
**Integrated into Shuttle's internal logic** - not a separate module:

1. **Gated Retrieval**: Initial broad exploration → focused expansion
2. **Matryoshka Importance Gating**: 0.6 → 0.75 → 0.85 → 0.9 thresholds by depth
3. **Graph Traversal**: Follow entity relationships contextually
4. **Multipass Fusion**: Intelligent result combination with score fusion

### Crawling Configuration by Complexity
- **LITE**: 1 pass, threshold 0.7, 5-10 items
- **FAST**: 2 passes, thresholds [0.6, 0.75], 8-20 items  
- **FULL**: 3 passes, thresholds [0.6, 0.75, 0.85], 12-35 items
- **RESEARCH**: 4 passes, thresholds [0.5, 0.65, 0.8, 0.9], 20-50 items

### Performance Excellence
- **Sub-2ms response times** even with deep graph traversal
- **Up to 17 protocol calls** for complex research queries
- **Intelligent deduplication** and composite scoring
- **Full computational provenance** with spacetime coordinates

## Critical Development Patterns

### PYTHONPATH Requirements (Unchanged)
**ALWAYS** set `PYTHONPATH=.` from repository root:

```powershell
# Windows PowerShell - use this pattern everywhere
$env:PYTHONPATH = "."; python dev/protocol_modules_mythrl.py
$env:PYTHONPATH = "."; python multipass_memory_demo.py
```

### New Architecture Usage
```python
from dev.protocol_modules_mythrl import MythRLShuttle, DemoMemoryBackend

# Create Shuttle and register protocols
shuttle = MythRLShuttle()
shuttle.register_protocol('memory_backend', DemoMemoryBackend())
shuttle.register_protocol('pattern_selection', PatternSelectionImpl())

# Weave with progressive complexity
result = await shuttle.weave("analyze complex bee colony relationships")
print(f"Complexity: {result.complexity_level.name}")
print(f"Confidence: {result.confidence}")
print(f"Crawl depth: {result.provenance.shuttle_events}")
```

### Protocol Implementation Pattern
```python
class CustomProtocol(Protocol):
    async def core_method(self, data: Dict, complexity: ComplexityLevel) -> Dict:
        """Implement based on complexity level."""
        ...

class CustomImplementation:
    async def core_method(self, data: Dict, complexity: ComplexityLevel) -> Dict:
        if complexity == ComplexityLevel.LITE:
            return lightweight_processing(data)
        elif complexity == ComplexityLevel.RESEARCH:
            return advanced_processing(data)
```

## Key Components

### MythRLShuttle (NEW)
```python
shuttle = MythRLShuttle()
result = await shuttle.weave(query, context)
# Returns MythRLResult with full provenance
```

### Protocol Implementations
- **PatternSelectionProtocol**: Grows from basic → advanced → emergent
- **DecisionEngineProtocol**: Skip → intelligent → multi-criteria
- **MemoryBackendProtocol**: InMemory → Neo4j → Qdrant → Hybrid + multipass crawling
- **FeatureExtractionProtocol**: 96d → multi-scale → full-spectrum
- **WarpSpaceProtocol**: Basic → standard → advanced → experimental (NON-NEGOTIABLE)
- **ToolExecutionProtocol**: Direct → intelligent → optimized

### Legacy HoloLoom Compatibility
```python
# Legacy approach still works
from HoloLoom.unified_api import HoloLoom
loom = await HoloLoom.create(pattern="fast")
result = await loom.query("What is HoloLoom?")
```

## File Organization

```
mythRL/
├── dev/
│   ├── protocol_modules_mythrl.py    # NEW: Shuttle-centric architecture
│   └── shuttle_centric_architecture.py  # Momentum-based gating
├── multipass_memory_demo.py         # NEW: Advanced memory crawling demo
├── SHUTTLE_ARCHITECTURE_PLAN.md     # NEW: Implementation roadmap
├── HoloLoom/                        # Legacy components (still supported)
│   ├── weaving_orchestrator.py     # Original 9-step weaving
│   ├── config.py                   # BARE/FAST/FUSED modes
│   ├── policy/unified.py           # Neural core + Thompson Sampling
│   ├── memory/                     # Legacy backends
│   └── ...                         # Other legacy components
└── demos/                          # Working examples
```

## Common Workflows

### New Architecture Demo
```powershell
$env:PYTHONPATH = "."; python dev/protocol_modules_mythrl.py         # Core demo
$env:PYTHONPATH = "."; python multipass_memory_demo.py               # Memory crawling
```

### Legacy Compatibility
```powershell
$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py         # 18 tests
$env:PYTHONPATH = "."; python demos/01_quickstart.py                  # Basic demo
```

### Add New Protocol Implementation
```python
class AdvancedPatternSelection:
    async def select_pattern(self, query: str, context: Dict, complexity: ComplexityLevel) -> Dict:
        # Implementation that scales with complexity
        return pattern_result

shuttle.register_protocol('pattern_selection', AdvancedPatternSelection())
```

### Memory Backend with Multipass Crawling
```python
class GraphMemoryBackend:
    async def retrieve_with_threshold(self, query: str, threshold: float, limit: int = 10) -> List[Dict]:
        # Threshold-based retrieval for gated crawling
        ...
    
    async def get_related(self, item_id: str, limit: int = 10) -> List[Dict]:
        # Graph traversal for multipass crawling
        ...

shuttle.register_protocol('memory_backend', GraphMemoryBackend())
```

## Important Conventions

**Architecture Philosophy**: Shuttle = Creative Intelligence, Protocols = Clean Interfaces, Modules = Swappable Implementations
**Progressive Complexity**: Always consider 3-5-7-9 scaling in implementations
**Internal Intelligence**: Synthesis, temporal, tracing, routing logic belongs in Shuttle
**Protocol Compliance**: Implement all required methods for each protocol

### New Architecture Benefits

✅ **Creative Orchestration**: Shuttle contains actual intelligence, not just routing
✅ **Progressive Scaling**: 3-5-7-9 complexity levels with natural performance targets
✅ **Intelligent Memory**: Recursive gated multipass crawling with graph traversal
✅ **Clean Modularity**: Protocol-based swappable implementations
✅ **Full Provenance**: Complete computational tracing with spacetime coordinates
✅ **Performance Excellence**: Sub-2ms responses even for complex research queries

### Legacy vs New Comparison

| Feature | Legacy HoloLoom | New mythRL Shuttle |
|---------|-----------------|-------------------|
| Architecture | 9-step weaving pipeline | Shuttle-centric with internal intelligence |
| Complexity | Fixed pipeline | 3-5-7-9 progressive scaling |
| Memory | Simple retrieval | Recursive gated multipass crawling |
| Orchestration | External coordination | Internal creative intelligence |
| Performance | ~150ms (FAST mode) | <50ms (LITE) to research-grade |
| Modularity | Module imports | Protocol-based swappable implementations |

## Migration Guide

### From Legacy to New Architecture
1. **Understand Shuttle**: Read `dev/protocol_modules_mythrl.py`
2. **Protocol Mapping**: Map existing components to new protocols
3. **Progressive Implementation**: Start with LITE, scale to RESEARCH
4. **Memory Enhancement**: Implement multipass crawling in memory backends

### Adapter Pattern for Legacy Components
```python
class LegacyToProtocolAdapter:
    def __init__(self, legacy_component):
        self.legacy = legacy_component
    
    async def protocol_method(self, data: Dict, complexity: ComplexityLevel) -> Dict:
        # Adapt legacy method to new protocol
        return await self.legacy.old_method(data)
```

## Performance Targets

### New Architecture Performance
- **LITE (3 steps)**: <50ms - Perfect for simple queries, greetings
- **FAST (5 steps)**: <150ms - Search patterns with temporal awareness  
- **FULL (7 steps)**: <300ms - Complex analysis with decision engine
- **RESEARCH (9 steps)**: No limit - Maximum capability deployment

### Memory Crawling Performance
- **1-4 crawl passes** based on complexity
- **Sub-2ms total time** including graph traversal
- **Up to 50 items retrieved** with intelligent deduplication
- **17+ protocol calls** for deep research queries

## Debugging the New Architecture

### Quick Troubleshooting
| Issue | Solution |
|-------|----------|
| `protocol_modules_mythrl not found` | Run from repo root with `$env:PYTHONPATH = "."` |
| Poor memory crawling performance | Check protocol implementation of `get_related()` |
| Complexity assessment wrong | Review `_assess_complexity_needed()` in Shuttle |
| Protocol not registered | Use `shuttle.register_protocol(name, implementation)` |

### Architecture Validation
```powershell
$env:PYTHONPATH = "."; python dev/protocol_modules_mythrl.py         # Test all protocols
$env:PYTHONPATH = "."; python multipass_memory_demo.py               # Test memory crawling
```

## Summary

mythRL represents a **revolutionary advancement** in neural decision-making systems:

### Core Innovations
- **Shuttle-Centric Architecture**: Creative orchestrator with internal intelligence
- **3-5-7-9 Progressive Complexity**: Natural scaling from simple to research-grade
- **Recursive Gated Multipass Memory Crawling**: Intelligent knowledge exploration
- **Protocol + Modules Pattern**: Clean, swappable, scalable implementations
- **Sub-2ms Performance**: Even for complex multi-hop reasoning

### Legacy Compatibility
- Full backward compatibility with existing HoloLoom components
- Adapter patterns for seamless migration
- Existing demos and tests continue to work

**Start here for new development**: Use `dev/protocol_modules_mythrl.py` and the Shuttle-centric architecture!

**Critical reminder:** Always use `PYTHONPATH=.` from repository root!

## Critical Development Patterns

### PYTHONPATH Requirements
**ALWAYS** set `PYTHONPATH=.` from repository root (absolute imports require this):

```powershell
# Windows PowerShell - use this pattern everywhere
$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py
$env:PYTHONPATH = "."; python demos/01_quickstart.py
```

### Async/Await Pattern
Orchestrator uses async throughout. Spinners, memory ops, and tool execution are async.

**CRITICAL**: `embedder.encode()` is synchronous - never await it!

### Graceful Degradation
Optional deps degrade with warnings: spaCy→regex, sentence-transformers→fallback, scipy→skip, Neo4j→NetworkX

### Multi-Scale Embeddings (Matryoshka)
BARE: 96d (~50ms) | FAST: 96d+192d (~150ms, default) | FUSED: 96d+192d+384d (~300ms)

## Key Components

### Configuration
```python
from HoloLoom.config import Config
Config.bare()    # 50ms | Config.fast()    # 150ms | Config.fused()   # 300ms
```

### Memory Backends
```python
config.memory_backend = MemoryBackend.NETWORKX      # In-memory
config.memory_backend = MemoryBackend.NEO4J_QDRANT  # Production
config.memory_backend = MemoryBackend.HYPERSPACE    # Gated multipass
```

### SpinningWheel
All spinners: `async def spin(raw_data) -> List[MemoryShard]`

### Policy Engine
```python
from HoloLoom.policy.unified import BanditStrategy
policy = create_policy(bandit_strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.10)
# EPSILON_GREEDY (90% exploit), BAYESIAN_BLEND (70/30), PURE_THOMPSON (100% explore)
```

### Synthesis Bridge
Integrates pattern extraction between ResonanceShed and WarpSpace

## Common Workflows

### Tests, Training, Demos
```powershell
$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py         # 18 tests
$env:PYTHONPATH = "."; python demos/01_quickstart.py                  # Basic
$env:PYTHONPATH = "."; python -c "from HoloLoom.train_agent import PPOTrainer; PPOTrainer(env_name='CartPole-v1', total_timesteps=2000).train()"
```

### Unified API
```python
from HoloLoom.unified_api import HoloLoom
loom = await HoloLoom.create(pattern="fast")
result = await loom.query("What is HoloLoom?")
await loom.chat("Tell me more")
await loom.ingest_text("Knowledge...")
```

### Add Spinner
```python
class CustomSpinner(BaseSpinner):
    async def spin(self, raw_data: Dict) -> List[MemoryShard]:
        return shards
```

### Extend Tools
1. Add to `NeuralCore.tools` in `policy/unified.py`
2. Update `n_tools` in config
3. Implement in orchestrator

## File Organization

```
HoloLoom/
├── weaving_orchestrator.py    # Main cycle (imports all modules)
├── config.py                  # BARE/FAST/FUSED modes
├── policy/unified.py          # Neural core + Thompson Sampling
├── embedding/spectral.py      # Matryoshka embeddings
├── memory/                    # Backends (protocol-based)
├── spinningWheel/            # Input adapters
├── loom/command.py           # Pattern selection
├── chrono/trigger.py         # Temporal control
├── resonance/shed.py         # Feature extraction
├── warp/space.py             # Tensor manifold
├── convergence/engine.py     # Decision collapse
├── fabric/spacetime.py       # Output with trace
└── Documentation/types.py    # Shared types only

demos/                        # Working examples
```

## Important Conventions

**Import Order**: stdlib → third-party → HoloLoom (absolute) → optional (try/except)
**Error Handling**: Return `Spacetime` with errors, never crash
**Type Hints**: Use protocols from `Documentation/types.py`
**Docstrings**: NumPy-style

### Code Patterns to Avoid

❌ Cross-module imports | ✅ Import only from `Documentation/types.py`
❌ `await embedder.encode()` | ✅ `embedder.encode()` (sync)
❌ Hardcoded scales | ✅ Derive from pattern card

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

1. **Background Tasks**: MemoryManager spawns fire-and-forget tasks without shutdown hooks
2. **ResonanceShed**: `embedder.encode()` is synchronous, not async
3. **Thread Count**: Extract from `dot_plasma['threads']` before `shed.lower()` clears them
4. **Bandit Updates**: Update statistics for actually-selected tool, not predicted
5. **Embedding Dimensions**: Create pattern-specific embedders per query

## Production Deployment

```bash
cd config && docker-compose up -d neo4j qdrant  # Start
python HoloLoom/test_backends.py                 # Verify
```

**Env vars**: `NEO4J_URI`, `QDRANT_URL`, `MATRIX_HOMESERVER` (for bot)

## Related Components

### Promptly Framework
Meta-prompt composition: chaining, loops, LLM-as-judge, A/B testing, HoloLoom bridge

### Matrix Bot / ChatOps
Commands: `!weave`, `!memory add/search/stats`, `!analyze`, `!stats`, `!help`
```powershell
$env:PYTHONPATH = "."; python HoloLoom/chatops/run_bot.py --hololoom-mode fast
```

### mythRL_core
Domain adapters: automotive, entity resolution, summarization

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
**Solution**: Set `$env:PYTHONPATH = "."` from repo root

#### 2. Async/Await Bugs
**Symptom**: `RuntimeWarning: coroutine was never awaited`
**Solution**: Add `await` before async calls

**Symptom**: `TypeError: object async_generator can't be used in 'await'`
**Solution**: `embedder.encode()` is synchronous - don't await it

#### 3. Memory Backend Connection Failures
**Solution**: `cd HoloLoom && docker-compose up -d && docker-compose ps`
Auto-falls back to in-memory if unavailable.

#### 4. Dimension Mismatch Errors
**Solution**: Derive embedder scales from pattern card
```python
pattern_spec = loom_command.select_pattern(query)
embedder = MatryoshkaEmbeddings(scales=pattern_spec.scales)
```

#### 5. Empty Memory Results
**Solution**: `stats = await memory.get_stats()` and verify embeddings exist

#### 6. Thompson Sampling Not Exploring
**Solution**: Increase `epsilon=0.20` or use `BanditStrategy.PURE_THOMPSON`

### Debugging Commands

```powershell
$env:PYTHONPATH = "."; python HoloLoom/test_unified_policy.py
$env:PYTHONPATH = "."; python -m pytest HoloLoom/tests/test_policy.py::test_epsilon_greedy -v
docker-compose ps && docker-compose logs --tail=50 neo4j
```

### Performance Profiling

```python
start = time.perf_counter()
result = await orchestrator.weave(Query(text="Test"))
print(f"Total: {(time.perf_counter() - start)*1000:.1f}ms")
print(f"  Extraction: {result.trace.feature_extraction_ms:.1f}ms")
```

## Advanced Architecture Details

### Discrete ↔ Continuous Transformations

**Lifecycle**: Discrete (Yarn Graph) → Tension (Warp Space) → Compute → Collapse → Update (Yarn Graph)

```python
threads = yarn_graph.select_threads(temporal_window)
warp_space = WarpSpace(threads)
await warp_space.tension(threads, yarn_graph.shards)
features = warp_space.compute_features()
collapse_result = convergence.collapse(neural_probs)
yarn_graph.apply_updates(warp_space.collapse())
```

### Thompson Sampling Integration

```python
class ThompsonBandit:
    def __init__(self, n_tools: int):
        self.alpha = np.ones(n_tools)  # Successes + 1
        self.beta = np.ones(n_tools)   # Failures + 1
    
    def sample(self) -> np.ndarray:
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, tool_idx: int, reward: float):
        if reward > 0.5: self.alpha[tool_idx] += 1
        else: self.beta[tool_idx] += 1
```

### MCTS Flux Capacitor

**UCB1**: `score = mean_reward + C * sqrt(ln(parent_visits) / node_visits)`

MCTS: simulations → UCB1 selection → rollout → backprop → best action

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
