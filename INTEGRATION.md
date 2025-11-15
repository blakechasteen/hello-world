# HoloLoom-Promptly Integration Guide

## Overview

This document describes the bidirectional integration between **HoloLoom** (neural decision-making system) and **Promptly** (prompt management framework).

The integration enables:

1. **Promptly → HoloLoom**: Load prompts from Promptly repositories as HoloLoom memory shards
2. **HoloLoom → Promptly**: Use HoloLoom's neural policy to evaluate Promptly prompts
3. **Unified Configuration**: Single config file for both systems
4. **Seamless Workflow**: End-to-end prompt engineering with neural evaluation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Integration Layer                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Promptly Side                      HoloLoom Side            │
│  ─────────────                      ─────────────            │
│                                                               │
│  HoloLoomEvaluator ─────────────▶ Neural Policy              │
│  (evaluate prompts)                (score prompts)           │
│                                                               │
│  HoloLoomSpinner ───────────────▶ MemoryShard                │
│  (convert prompts)                 (memory format)           │
│                                                               │
│  PromptlyLoader ◀───────────────── Knowledge Graph           │
│  (load to memory)                  (graph store)             │
│                                                               │
│  PromptlyMemoryAdapter ◀────────── MemoryManager             │
│  (format conversion)               (retrieval)               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

Both systems must be installed and accessible:

```bash
# HoloLoom should be at /path/to/repo/HoloLoom
# Promptly should be at /path/to/repo/Promptly

# Verify installation
python -c "from HoloLoom.Documentation.types import MemoryShard; print('HoloLoom OK')"
python -c "from Promptly.promptly.promptly import Promptly; print('Promptly OK')"
```

### Integration Setup

1. **Configuration**: Copy and customize `integration_config.yaml`:

```bash
cp integration_config.yaml integration_config.yaml.local
# Edit integration_config.yaml.local with your settings
```

2. **Verify Integration**:

```bash
# Run integration demo
python examples/hololoom_promptly_demo.py
```

## Usage Guide

### 1. Loading Promptly Prompts into HoloLoom

Use HoloLoom's `PromptlyLoader` to load prompts as memory shards:

```python
from HoloLoom.integrations.promptly import PromptlyLoader
from Promptly.promptly.promptly import Promptly

# Initialize Promptly
promptly = Promptly(root_dir='/path/to/repo')

# Create loader
loader = PromptlyLoader(promptly_instance=promptly)

# Load all prompts as shards
shards = loader.prompts_to_shards(branch='main')

print(f"Loaded {len(shards)} prompts as HoloLoom MemoryShards")

# Load into HoloLoom memory manager (if available)
# loader.load_into_memory(memory_manager, branch='main')
```

**Filtering prompts:**

```python
# Load only prompts with specific tags
prompts = loader.load_prompts(
    branch='main',
    tags=['production', 'summarization'],
    min_version=2
)
shards = loader.adapter.prompts_to_shards(prompts)
```

**Quick loading:**

```python
from HoloLoom.integrations.promptly import quick_load_prompts

shards = quick_load_prompts(
    promptly_repo_path='/path/to/repo',
    branch='main'
)
```

### 2. Evaluating Prompts with HoloLoom

Use Promptly's `HoloLoomEvaluator` to score prompts using neural policy:

```python
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

# Initialize evaluator
evaluator = HoloLoomEvaluator(config_mode='fast')

# Evaluate a single prompt
result = evaluator.evaluate_prompt(
    prompt_name='summarizer',
    prompt_content='Summarize this text: {text}',
    test_inputs={'text': 'Sample input text here'}
)

print(f"Score: {result.score:.3f}")
print(f"Tool selection: {result.tool_selection}")
print(f"Metadata: {result.metadata}")
```

**Batch evaluation:**

```python
prompts = [
    {'name': 'prompt1', 'content': 'Template 1: {input}'},
    {'name': 'prompt2', 'content': 'Template 2: {input}'},
    {'name': 'prompt3', 'content': 'Template 3: {input}'},
]

results = evaluator.batch_evaluate(
    prompts,
    test_inputs={'input': 'test data'}
)

for result in results:
    print(f"{result.prompt_name}: {result.score:.3f}")
```

### 3. Using HoloLoom Evaluator in Promptly Workflow

Integrate HoloLoom evaluation into Promptly's test suite:

```python
from Promptly.promptly.promptly import Promptly
from Promptly.promptly.integrations.hololoom import create_evaluator_function

# Create Promptly instance
promptly = Promptly()

# Create HoloLoom evaluator function
hololoom_eval = create_evaluator_function(config_mode='fast')

# Define test cases with HoloLoom evaluator
test_cases = [
    {
        'id': 'test-1',
        'inputs': {'text': 'Machine learning is advancing rapidly.'},
        'expected': 'positive insight',
        'evaluator': hololoom_eval  # Use HoloLoom for scoring
    },
    {
        'id': 'test-2',
        'inputs': {'text': 'The system failed to process the request.'},
        'expected': 'negative outcome',
        'evaluator': hololoom_eval
    }
]

# Run evaluation
results = promptly.eval_prompt('my_prompt', test_cases)

# Results will include HoloLoom neural policy scores
for result in results:
    print(f"Test: {result['test_case']['id']}")
    print(f"Score: {result['score']:.3f}")
```

### 4. Memory Integration

Load Promptly prompts into HoloLoom memory system:

```python
from HoloLoom.integrations.promptly import PromptlyLoader
from HoloLoom.memory.cache import MemoryManager  # If available

# Initialize components
loader = PromptlyLoader(promptly_repo_path='/path/to/repo')
memory = MemoryManager()

# Load prompts directly into memory
count = loader.load_into_memory(
    memory_manager=memory,
    branch='main',
    tags=['production']
)

print(f"Loaded {count} prompts into HoloLoom memory")
```

### 5. Prompt Metadata and Versioning

Access detailed prompt metadata:

```python
from HoloLoom.integrations.promptly import PromptlyLoader

loader = PromptlyLoader()

# Get metadata for specific prompt
metadata = loader.get_prompt_metadata('my_prompt', version=3)

if metadata:
    print(f"Name: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Branch: {metadata.branch}")
    print(f"Tags: {metadata.tags}")
    print(f"Performance: {metadata.performance_metrics}")
```

## Configuration

The `integration_config.yaml` file provides unified configuration for both systems.

### Key Configuration Sections

#### Integration Settings

```yaml
integration:
  enabled: true
  auto_load_prompts: false  # Auto-load on startup
  auto_sync_results: false  # Sync eval results back
```

#### HoloLoom Settings

```yaml
hololoom:
  mode: fast  # bare | fast | fused
  scales: [96, 192, 384]
  memory:
    max_prompts: 100
    extract_entities: true
    extract_motifs: true
  policy:
    bandit_strategy: epsilon_greedy
    epsilon: 0.1
```

#### Promptly Settings

```yaml
promptly:
  repo_path: null  # Use current directory
  default_branch: main
  evaluation:
    use_hololoom_evaluator: false
    quality_threshold: 0.7
    store_results: true
```

#### Evaluator Configuration

```yaml
evaluators:
  default: promptly_default  # or hololoom_neural
  hololoom_neural:
    config_mode: fast
    use_cache: true
    batch_size: 10
```

### Loading Configuration

```python
from HoloLoom.integrations.promptly import load_integration_config

config = load_integration_config('/path/to/integration_config.yaml')

# Access settings
hololoom_mode = config['hololoom']['mode']
default_evaluator = config['evaluators']['default']
```

## API Reference

### Promptly Side (Promptly/promptly/integrations/hololoom.py)

#### `HoloLoomEvaluator`

Evaluates prompts using HoloLoom's neural policy.

**Methods:**
- `__init__(config_mode='fast', custom_config=None)`: Initialize evaluator
- `evaluate_prompt(prompt_name, prompt_content, test_inputs)`: Evaluate single prompt
- `batch_evaluate(prompts, test_inputs)`: Evaluate multiple prompts

**Returns:** `EvaluationResult` with score, tool_selection, metadata

#### `HoloLoomSpinner`

Converts Promptly prompts to HoloLoom MemoryShards.

**Methods:**
- `__init__()`: Initialize spinner
- `prompt_to_shard(prompt_data)`: Convert single prompt
- `spin_prompts(promptly_instance, branch)`: Convert all prompts from repo

**Returns:** `List[MemoryShard]`

#### `create_evaluator_function(config_mode)`

Factory function to create Promptly-compatible evaluator.

**Returns:** Function compatible with Promptly test cases

### HoloLoom Side (HoloLoom/integrations/promptly.py)

#### `PromptlyLoader`

Loads prompts from Promptly repositories.

**Methods:**
- `__init__(promptly_repo_path, promptly_instance)`: Initialize loader
- `load_prompts(branch, tags, min_version)`: Load with filtering
- `load_all_prompts(branch)`: Load all prompts
- `prompts_to_shards(prompts, branch)`: Convert to MemoryShards
- `load_into_memory(memory_manager, branch, tags)`: Load into HoloLoom memory
- `get_prompt_metadata(prompt_name, version)`: Get detailed metadata

#### `PromptlyMemoryAdapter`

Converts Promptly prompts to MemoryShards.

**Methods:**
- `__init__(extract_entities, extract_motifs)`: Initialize adapter
- `prompt_to_shard(prompt_data)`: Convert single prompt
- `prompts_to_shards(prompts)`: Convert multiple prompts

#### `quick_load_prompts(promptly_repo_path, branch)`

Convenience function for quick loading.

**Returns:** `List[MemoryShard]`

## Use Cases

### 1. Prompt Quality Scoring

Automatically score prompt templates using HoloLoom's neural policy:

```python
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

evaluator = HoloLoomEvaluator(config_mode='fused')  # High-quality mode

# Test multiple variants
variants = [
    "Summarize: {text}",
    "Provide a brief summary of: {text}",
    "Extract the key points from: {text}",
]

scores = []
for variant in variants:
    result = evaluator.evaluate_prompt(
        'summarizer_variant',
        variant,
        {'text': 'test data'}
    )
    scores.append((variant, result.score))

# Select best variant
best = max(scores, key=lambda x: x[1])
print(f"Best variant: {best[0]} (score: {best[1]:.3f})")
```

### 2. Prompt Repository as Knowledge Base

Use Promptly repository as a knowledge base for HoloLoom:

```python
from HoloLoom.integrations.promptly import PromptlyLoader
from HoloLoom.Orchestrator import Orchestrator

loader = PromptlyLoader()
shards = loader.prompts_to_shards(branch='production')

# Use shards as context for HoloLoom decisions
# (Integration with Orchestrator's memory system)
```

### 3. A/B Testing with Neural Evaluation

Compare prompt versions using HoloLoom evaluation:

```python
from Promptly.promptly.promptly import Promptly
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

promptly = Promptly()
evaluator = HoloLoomEvaluator()

# Get different versions
v1 = promptly.get('my_prompt', version=1)
v2 = promptly.get('my_prompt', version=2)

# Evaluate both
test_data = {'input': 'sample data'}
result_v1 = evaluator.evaluate_prompt(v1['name'], v1['content'], test_data)
result_v2 = evaluator.evaluate_prompt(v2['name'], v2['content'], test_data)

print(f"Version 1 score: {result_v1.score:.3f}")
print(f"Version 2 score: {result_v2.score:.3f}")

winner = 'v1' if result_v1.score > result_v2.score else 'v2'
print(f"Winner: {winner}")
```

### 4. Automated Prompt Optimization

Continuously improve prompts based on HoloLoom feedback:

```python
from Promptly.promptly.promptly import Promptly
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

promptly = Promptly()
evaluator = HoloLoomEvaluator()

def optimize_prompt(base_prompt, iterations=5):
    """Iteratively improve prompt based on HoloLoom scores."""
    best_score = 0
    best_variant = base_prompt

    for i in range(iterations):
        # Generate variant (simplified - use LLM in production)
        variant = f"{base_prompt} [variant {i}]"

        # Evaluate
        result = evaluator.evaluate_prompt(
            f'optimization_test_{i}',
            variant,
            {'input': 'test'}
        )

        if result.score > best_score:
            best_score = result.score
            best_variant = variant

            # Save to Promptly
            promptly.add(
                'optimized_prompt',
                best_variant,
                metadata={'score': result.score, 'iteration': i}
            )

    return best_variant, best_score

optimized, score = optimize_prompt("Analyze: {input}")
print(f"Optimized prompt: {optimized}")
print(f"Final score: {score:.3f}")
```

## Performance Considerations

### Caching

Enable caching for repeated evaluations:

```yaml
evaluators:
  hololoom_neural:
    use_cache: true
```

### Batch Processing

Use batch evaluation for large prompt sets:

```python
evaluator = HoloLoomEvaluator()
results = evaluator.batch_evaluate(prompts, test_inputs)  # More efficient
```

### Async Loading

Enable async loading in config:

```yaml
performance:
  async_loading: true
  parallel_workers: 4
```

### Execution Modes

Choose appropriate HoloLoom mode:

- **bare**: Fastest, minimal features
- **fast**: Balanced (recommended)
- **fused**: Highest quality, slower

## Error Handling

The integration gracefully degrades when components are unavailable:

```python
# Both integrations check availability
from Promptly.promptly.integrations.hololoom import HOLOLOOM_AVAILABLE
from HoloLoom.integrations.promptly import PROMPTLY_AVAILABLE

if HOLOLOOM_AVAILABLE:
    # Use HoloLoom features
    pass
else:
    # Fall back to default behavior
    pass
```

## Troubleshooting

### Import Errors

If you see `ImportError: No module named 'HoloLoom'`:

1. Verify both systems are installed
2. Check Python path includes repository root
3. Use absolute imports

### Promptly Not Initialized

If you see `"Not a promptly repository"`:

```bash
cd /path/to/repo
python -c "from Promptly.promptly.promptly import Promptly; Promptly().init()"
```

### Configuration Not Found

If integration config is not found:

1. Verify `integration_config.yaml` exists in repo root
2. Pass explicit path to `load_integration_config()`

### Evaluation Failures

If HoloLoom evaluation fails:

1. Check HoloLoom dependencies (torch, numpy, etc.)
2. Verify execution mode is valid (bare/fast/fused)
3. Check logs for specific errors

## Examples

See `examples/hololoom_promptly_demo.py` for complete working examples:

```bash
# Run all demos
python examples/hololoom_promptly_demo.py

# Run specific demo
python -c "from examples.hololoom_promptly_demo import demo_2_hololoom_evaluator; demo_2_hololoom_evaluator()"
```

## Future Enhancements

Potential improvements for future versions:

1. **Bidirectional Sync**: Auto-sync evaluation results back to Promptly
2. **Chain Integration**: Use HoloLoom policy for Promptly chain routing
3. **Memory Persistence**: Save HoloLoom memory with Promptly prompts
4. **Optimization Loop**: Automated prompt improvement based on HoloLoom feedback
5. **Multi-Model Evaluation**: Compare HoloLoom scores with other evaluators
6. **Visualization**: Dashboard showing prompt quality scores over time

## Contributing

To extend the integration:

1. Add new features to integration modules
2. Update `integration_config.yaml` with new settings
3. Add examples to demo script
4. Update this documentation

## License

This integration follows the licenses of HoloLoom and Promptly projects.

## Support

For issues or questions:

- HoloLoom: See HoloLoom documentation
- Promptly: See Promptly documentation
- Integration: Check this guide and example code
