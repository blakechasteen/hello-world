# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

Run the unified policy test suite (18 tests covering neural components):
```bash
PYTHONPATH=. .venv/bin/python holoLoom/test_unified_policy.py
```

This tests: MLP blocks, attention, ICM/RND curiosity modules, hierarchical policies, PPO agent, and full end-to-end pipeline.

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

#### 1. Orchestrator (`holoLoom/orchestrator.py`)
Central coordinator that implements the query processing pipeline:
1. Feature extraction (motifs + embeddings + spectral features)
2. Memory retrieval (context shards from knowledge graph)
3. Policy decision (tool selection via neural network + Thompson Sampling)
4. Tool execution
5. Response assembly

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
- Optional Ollama enrichment for entity/motif extraction
- Standardized output format feeds directly into orchestrator

#### 7. Training (`holoLoom/train_agent`)
PPO trainer for RL environments with:
- GAE (Generalized Advantage Estimation)
- Optional ICM/RND curiosity modules
- Checkpoint saving/loading
- Configurable network architectures

### Module Structure

```
holoLoom/
├── __init__.py              # Package entry point
├── config.py                # Execution modes, system config
├── orchestrator.py          # Central coordinator (imports all modules)
├── train_agent              # PPO training script
├── test_unified_policy.py   # Comprehensive test suite
│
├── policy/                  # Decision making
│   ├── __init__.py
│   └── unified.py          # Neural core + Thompson Sampling
│
├── embedding/               # Multi-scale embeddings
│   ├── __init__.py
│   └── spectral.py         # Matryoshka + spectral features
│
├── memory/                  # Knowledge storage
│   ├── __init__.py
│   ├── cache.py            # Vector/BM25 retrieval
│   └── graph.py            # Knowledge graph (NetworkX)
│
├── motif/                   # Pattern detection
│   ├── __init__.py
│   ├── base.py             # Regex + optional NLP
│   └── types.py
│
├── spinningWheel/           # Input adapters
│   ├── __init__.py
│   ├── base.py
│   ├── audio.py
│   └── utils/
│
├── documentation/           # Shared types
│   ├── types.py            # Query, Context, Features, etc.
│   └── CODE_REVIEW.md
│
└── modules/                 # Additional components
    ├── Features.py
    └── Types.py
```

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

## Promptly Integration

HoloLoom now includes bidirectional integration with **Promptly**, a prompt management framework with versioning, branching, and evaluation capabilities.

### Integration Overview

The integration enables:
- **Promptly → HoloLoom**: Load prompts from Promptly repositories as HoloLoom MemoryShards
- **HoloLoom → Promptly**: Use HoloLoom's neural policy to evaluate Promptly prompts
- **Unified Configuration**: Single config file (`integration_config.yaml`) for both systems
- **Seamless Workflow**: End-to-end prompt engineering with neural evaluation

### Key Integration Files

```
HoloLoom/integrations/
├── __init__.py
└── promptly.py              # PromptlyLoader, PromptlyMemoryAdapter

Promptly/promptly/integrations/
├── __init__.py
└── hololoom.py              # HoloLoomEvaluator, HoloLoomSpinner

integration_config.yaml       # Unified configuration
examples/hololoom_promptly_demo.py  # Integration demo
INTEGRATION.md               # Complete integration guide
```

### Quick Start

#### Load Promptly Prompts into HoloLoom

```python
from HoloLoom.integrations.promptly import PromptlyLoader

# Initialize loader
loader = PromptlyLoader(promptly_repo_path='/path/to/repo')

# Load all prompts as HoloLoom MemoryShards
shards = loader.prompts_to_shards(branch='main')

# Filter by tags
prompts = loader.load_prompts(
    branch='main',
    tags=['production', 'summarization'],
    min_version=2
)
```

#### Evaluate Prompts with HoloLoom Neural Policy

```python
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

# Initialize evaluator
evaluator = HoloLoomEvaluator(config_mode='fast')

# Evaluate a prompt
result = evaluator.evaluate_prompt(
    prompt_name='summarizer',
    prompt_content='Summarize: {text}',
    test_inputs={'text': 'Sample input text'}
)

print(f"Score: {result.score:.3f}")
print(f"Tool selection: {result.tool_selection}")
```

#### Use HoloLoom Evaluator in Promptly Tests

```python
from Promptly.promptly.promptly import Promptly
from Promptly.promptly.integrations.hololoom import create_evaluator_function

promptly = Promptly()
hololoom_eval = create_evaluator_function(config_mode='fast')

test_cases = [
    {
        'inputs': {'text': 'Test input'},
        'expected': 'expected output',
        'evaluator': hololoom_eval  # Use HoloLoom for scoring
    }
]

results = promptly.eval_prompt('my_prompt', test_cases)
```

### Integration Configuration

Edit `integration_config.yaml` to customize:

```yaml
integration:
  enabled: true
  auto_load_prompts: false

hololoom:
  mode: fast  # bare | fast | fused
  memory:
    max_prompts: 100
    extract_entities: true
    extract_motifs: true
  policy:
    bandit_strategy: epsilon_greedy
    epsilon: 0.1

promptly:
  default_branch: main
  evaluation:
    use_hololoom_evaluator: false
    quality_threshold: 0.7

evaluators:
  default: promptly_default  # or hololoom_neural
```

### Running the Integration Demo

```bash
# Run all integration demos
python examples/hololoom_promptly_demo.py

# Run specific demo
python -c "from examples.hololoom_promptly_demo import demo_2_hololoom_evaluator; demo_2_hololoom_evaluator()"
```

### Common Integration Use Cases

#### 1. Prompt Quality Scoring

Automatically score prompt templates using HoloLoom's neural policy:

```python
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

evaluator = HoloLoomEvaluator(config_mode='fused')

variants = [
    "Summarize: {text}",
    "Provide a brief summary of: {text}",
    "Extract the key points from: {text}",
]

scores = []
for variant in variants:
    result = evaluator.evaluate_prompt('variant', variant, {'text': 'test'})
    scores.append((variant, result.score))

best = max(scores, key=lambda x: x[1])
print(f"Best variant: {best[0]} (score: {best[1]:.3f})")
```

#### 2. Prompt Repository as Knowledge Base

Use Promptly repository as a knowledge base for HoloLoom:

```python
from HoloLoom.integrations.promptly import PromptlyLoader

loader = PromptlyLoader()
shards = loader.prompts_to_shards(branch='production')

# Use shards as context for HoloLoom decisions
# loader.load_into_memory(memory_manager, branch='production')
```

#### 3. A/B Testing with Neural Evaluation

Compare prompt versions using HoloLoom:

```python
from Promptly.promptly.promptly import Promptly
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

promptly = Promptly()
evaluator = HoloLoomEvaluator()

v1 = promptly.get('my_prompt', version=1)
v2 = promptly.get('my_prompt', version=2)

result_v1 = evaluator.evaluate_prompt(v1['name'], v1['content'], test_data)
result_v2 = evaluator.evaluate_prompt(v2['name'], v2['content'], test_data)

winner = 'v1' if result_v1.score > result_v2.score else 'v2'
print(f"Winner: {winner}")
```

### Integration API Reference

#### HoloLoom Side (`HoloLoom/integrations/promptly.py`)

- **PromptlyLoader**: Loads prompts from Promptly repositories
  - `load_prompts(branch, tags, min_version)`: Load with filtering
  - `prompts_to_shards(prompts, branch)`: Convert to MemoryShards
  - `load_into_memory(memory_manager, branch, tags)`: Load into HoloLoom memory

- **PromptlyMemoryAdapter**: Converts Promptly prompts to MemoryShards
  - `prompt_to_shard(prompt_data)`: Convert single prompt
  - `prompts_to_shards(prompts)`: Convert multiple prompts

#### Promptly Side (`Promptly/promptly/integrations/hololoom.py`)

- **HoloLoomEvaluator**: Evaluates prompts using neural policy
  - `evaluate_prompt(prompt_name, prompt_content, test_inputs)`: Evaluate single prompt
  - `batch_evaluate(prompts, test_inputs)`: Evaluate multiple prompts

- **HoloLoomSpinner**: Converts prompts to MemoryShards
  - `prompt_to_shard(prompt_data)`: Convert single prompt
  - `spin_prompts(promptly_instance, branch)`: Convert all from repo

### Graceful Degradation

The integration gracefully degrades when components are unavailable:

```python
from Promptly.promptly.integrations.hololoom import HOLOLOOM_AVAILABLE
from HoloLoom.integrations.promptly import PROMPTLY_AVAILABLE

if HOLOLOOM_AVAILABLE:
    # Use HoloLoom features
    pass
else:
    # Fall back to default behavior
    pass
```

### Documentation

See **INTEGRATION.md** for complete documentation including:
- Detailed API reference
- Configuration options
- Performance tuning
- Error handling
- Advanced use cases
- Troubleshooting guide
