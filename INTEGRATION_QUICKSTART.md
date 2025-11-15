# HoloLoom-Promptly Integration Quick Start

## 5-Minute Setup

### 1. Verify Both Systems Are Available

```bash
python -c "from HoloLoom.Documentation.types import MemoryShard; print('✓ HoloLoom OK')"
python -c "from Promptly.promptly.promptly import Promptly; print('✓ Promptly OK')"
```

### 2. Run the Demo

```bash
cd /home/user/hello-world
python examples/hololoom_promptly_demo.py
```

## Most Common Use Cases

### Use Case 1: Evaluate a Prompt Template

**Problem:** You have a prompt template and want to know how good it is.

**Solution:** Use HoloLoom's neural policy to score it.

```python
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

# Initialize evaluator (fast mode = balanced performance)
evaluator = HoloLoomEvaluator(config_mode='fast')

# Your prompt template
prompt = "Summarize the following text in 2-3 sentences: {text}"

# Evaluate with sample input
result = evaluator.evaluate_prompt(
    prompt_name='summarizer_v1',
    prompt_content=prompt,
    test_inputs={'text': 'Your sample text here...'}
)

# Check the score (0.0 to 1.0, higher is better)
print(f"Quality score: {result.score:.3f}")
```

### Use Case 2: Compare Two Prompt Versions

**Problem:** You have two versions of a prompt and want to know which is better.

**Solution:** Evaluate both and compare scores.

```python
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

evaluator = HoloLoomEvaluator(config_mode='fast')

v1 = "Summarize: {text}"
v2 = "Provide a concise summary of the following: {text}"

test_data = {'text': 'Machine learning is transforming industries...'}

r1 = evaluator.evaluate_prompt('v1', v1, test_data)
r2 = evaluator.evaluate_prompt('v2', v2, test_data)

print(f"Version 1: {r1.score:.3f}")
print(f"Version 2: {r2.score:.3f}")
print(f"Winner: {'v1' if r1.score > r2.score else 'v2'}")
```

### Use Case 3: Load Promptly Prompts into HoloLoom

**Problem:** You have prompts in Promptly and want to use them in HoloLoom.

**Solution:** Use PromptlyLoader to convert them to MemoryShards.

```python
from HoloLoom.integrations.promptly import PromptlyLoader

# Initialize loader
loader = PromptlyLoader(promptly_repo_path='/home/user/hello-world')

# Load all prompts from main branch
shards = loader.prompts_to_shards(branch='main')

print(f"Loaded {len(shards)} prompts as HoloLoom MemoryShards")

# Use shards in HoloLoom
for shard in shards:
    print(f"- {shard.id}: {shard.text[:50]}...")
```

### Use Case 4: Filter Prompts by Tag

**Problem:** You only want production-ready prompts with specific tags.

**Solution:** Use the filter parameters.

```python
from HoloLoom.integrations.promptly import PromptlyLoader

loader = PromptlyLoader()

# Load only prompts tagged 'production' and 'summarization'
prompts = loader.load_prompts(
    branch='main',
    tags=['production', 'summarization'],
    min_version=2  # Only version 2 or higher
)

print(f"Found {len(prompts)} matching prompts")
```

### Use Case 5: Use HoloLoom in Promptly Tests

**Problem:** You want to use HoloLoom's neural policy in your Promptly test suite.

**Solution:** Create a HoloLoom evaluator function.

```python
from Promptly.promptly.promptly import Promptly
from Promptly.promptly.integrations.hololoom import create_evaluator_function

# Create Promptly instance
promptly = Promptly()

# Create HoloLoom evaluator function
hololoom_eval = create_evaluator_function(config_mode='fast')

# Define test cases
test_cases = [
    {
        'id': 'test-1',
        'inputs': {'text': 'Sample input text'},
        'expected': 'expected output',
        'evaluator': hololoom_eval  # Use HoloLoom!
    }
]

# Run evaluation
results = promptly.eval_prompt('my_prompt', test_cases)

# Check results
for result in results:
    print(f"Test {result['test_case']['id']}: {result['score']:.3f}")
```

### Use Case 6: Batch Evaluate Multiple Prompts

**Problem:** You have many prompts to evaluate quickly.

**Solution:** Use batch evaluation.

```python
from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

evaluator = HoloLoomEvaluator(config_mode='fast')

# List of prompts to evaluate
prompts = [
    {'name': 'summarizer', 'content': 'Summarize: {text}'},
    {'name': 'classifier', 'content': 'Classify: {text}'},
    {'name': 'extractor', 'content': 'Extract entities from: {text}'},
]

# Batch evaluate
results = evaluator.batch_evaluate(
    prompts,
    test_inputs={'text': 'Sample text'}
)

# Show results
for result in results:
    print(f"{result.prompt_name}: {result.score:.3f}")
```

## Configuration Basics

### Minimal Configuration

Create `integration_config.yaml`:

```yaml
integration:
  enabled: true

hololoom:
  mode: fast  # fast is recommended

promptly:
  default_branch: main
```

### Production Configuration

```yaml
integration:
  enabled: true
  auto_load_prompts: true

hololoom:
  mode: fused  # highest quality
  memory:
    max_prompts: 500
    extract_entities: true
    extract_motifs: true

promptly:
  default_branch: production
  filter:
    tags: ['production']
    min_version: 3
  evaluation:
    use_hololoom_evaluator: true
    quality_threshold: 0.8

evaluators:
  default: hololoom_neural
```

## Execution Modes

Choose the right mode for your needs:

| Mode   | Speed      | Quality    | Use When                   |
|--------|------------|------------|----------------------------|
| bare   | Fastest    | Basic      | Rapid iteration, testing   |
| fast   | Balanced   | Good       | Production (recommended)   |
| fused  | Slower     | Best       | Final evaluation, A/B test |

```python
# Fast mode (recommended)
evaluator = HoloLoomEvaluator(config_mode='fast')

# High-quality mode for important decisions
evaluator = HoloLoomEvaluator(config_mode='fused')

# Quick testing mode
evaluator = HoloLoomEvaluator(config_mode='bare')
```

## Error Handling

### Check if Integration is Available

```python
from Promptly.promptly.integrations.hololoom import HOLOLOOM_AVAILABLE
from HoloLoom.integrations.promptly import PROMPTLY_AVAILABLE

if HOLOLOOM_AVAILABLE:
    # Use HoloLoom features
    evaluator = HoloLoomEvaluator()
else:
    # Fall back to default
    print("HoloLoom not available, using default evaluator")
```

### Handle Evaluation Failures

```python
try:
    result = evaluator.evaluate_prompt(name, content, inputs)
    print(f"Score: {result.score}")
except Exception as e:
    print(f"Evaluation failed: {e}")
    # Fall back to default scoring
```

## Common Patterns

### Pattern 1: Find Best Prompt Variant

```python
variants = [
    "Summarize: {text}",
    "Provide a summary of: {text}",
    "Extract key points from: {text}",
]

evaluator = HoloLoomEvaluator(config_mode='fused')
test_data = {'text': 'Sample input'}

scores = []
for variant in variants:
    result = evaluator.evaluate_prompt('test', variant, test_data)
    scores.append((variant, result.score))

best = max(scores, key=lambda x: x[1])
print(f"Best: {best[0]} (score: {best[1]:.3f})")
```

### Pattern 2: Quality Gate

```python
def meets_quality_threshold(prompt, threshold=0.7):
    evaluator = HoloLoomEvaluator(config_mode='fast')
    result = evaluator.evaluate_prompt('test', prompt, {'text': 'test'})
    return result.score >= threshold

if meets_quality_threshold(my_prompt):
    print("✓ Prompt approved")
else:
    print("✗ Prompt needs improvement")
```

### Pattern 3: Progressive Improvement

```python
def improve_prompt(base_prompt, iterations=5):
    """Iteratively improve prompt based on scores."""
    evaluator = HoloLoomEvaluator(config_mode='fused')
    best_score = 0
    best_variant = base_prompt

    for i in range(iterations):
        # Generate variant (simplified - use LLM in production)
        variant = f"{base_prompt} (variant {i})"

        result = evaluator.evaluate_prompt(f'iter_{i}', variant, {'text': 'test'})

        if result.score > best_score:
            best_score = result.score
            best_variant = variant
            print(f"Iteration {i}: New best score {best_score:.3f}")

    return best_variant, best_score

optimized, score = improve_prompt("Analyze: {text}")
```

## Troubleshooting

### Problem: Import errors

**Error:** `ImportError: No module named 'HoloLoom'`

**Solution:**
```bash
# Ensure repo root is in Python path
export PYTHONPATH=/home/user/hello-world:$PYTHONPATH
python your_script.py
```

### Problem: Promptly not initialized

**Error:** `"Not a promptly repository"`

**Solution:**
```python
from Promptly.promptly.promptly import Promptly
promptly = Promptly(root_dir='/home/user/hello-world')
promptly.init()
```

### Problem: Config file not found

**Error:** `Integration config not found`

**Solution:**
```python
# Use explicit path
from HoloLoom.integrations.promptly import load_integration_config
config = load_integration_config('/home/user/hello-world/integration_config.yaml')
```

### Problem: Low scores for all prompts

**Possible causes:**
1. Using 'bare' mode (try 'fast' or 'fused')
2. Test inputs don't match prompt format
3. HoloLoom dependencies missing (install torch, numpy)

**Solution:**
```python
# Use higher-quality mode
evaluator = HoloLoomEvaluator(config_mode='fused')

# Ensure test inputs match prompt placeholders
prompt = "Summarize: {text}"
inputs = {'text': 'your text here'}  # Must have 'text' key
```

## Performance Tips

### Tip 1: Use Batch Evaluation

```python
# SLOW: Individual evaluations
for prompt in prompts:
    result = evaluator.evaluate_prompt(...)

# FAST: Batch evaluation
results = evaluator.batch_evaluate(prompts, test_inputs)
```

### Tip 2: Cache Results

```python
# Enable caching in config
evaluators:
  hololoom_neural:
    use_cache: true
```

### Tip 3: Choose Appropriate Mode

```python
# For testing: use bare (fastest)
evaluator = HoloLoomEvaluator(config_mode='bare')

# For production: use fast (balanced)
evaluator = HoloLoomEvaluator(config_mode='fast')

# For A/B testing: use fused (highest quality)
evaluator = HoloLoomEvaluator(config_mode='fused')
```

## Next Steps

1. **Read Full Documentation**: See `INTEGRATION.md` for complete details
2. **Run Demo**: Execute `python examples/hololoom_promptly_demo.py`
3. **Customize Config**: Edit `integration_config.yaml` for your needs
4. **Try Examples**: Start with the common patterns above
5. **Integrate with Your Workflow**: Add to your existing Promptly or HoloLoom code

## Quick Reference

### Import Cheatsheet

```python
# HoloLoom side (load prompts)
from HoloLoom.integrations.promptly import (
    PromptlyLoader,
    PromptlyMemoryAdapter,
    quick_load_prompts
)

# Promptly side (evaluate prompts)
from Promptly.promptly.integrations.hololoom import (
    HoloLoomEvaluator,
    HoloLoomSpinner,
    create_evaluator_function
)

# Configuration
from HoloLoom.integrations.promptly import load_integration_config
```

### One-Liners

```python
# Quick evaluate
result = HoloLoomEvaluator().evaluate_prompt('test', 'Summarize: {text}', {'text': 'data'})

# Quick load
shards = quick_load_prompts('/path/to/repo', branch='main')

# Quick config
config = load_integration_config()
```

## Support

- **Full Documentation**: `INTEGRATION.md`
- **Configuration Guide**: `integration_config.yaml` (with comments)
- **Examples**: `examples/hololoom_promptly_demo.py`
- **Project Instructions**: `CLAUDE.md` (Promptly Integration section)
