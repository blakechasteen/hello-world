# Promptly Plugin System - Implementation Summary

## Overview

A complete plugin architecture has been added to Promptly, enabling extensibility through:
1. **Custom Evaluators** - Pluggable scoring and evaluation logic
2. **Custom Storage Backends** - Alternative persistence implementations
3. **Custom Chain Step Processors** - Extensible chain execution logic

## Files Created

### Core Plugin System

#### 1. `/home/user/hello-world/Promptly/promptly/plugins/base.py`
**Protocol definitions for all plugin types**

Contains:
- `EvaluatorPlugin` protocol - Interface for evaluators
- `StorageBackend` protocol - Interface for storage backends
- `ChainStepProcessor` protocol - Interface for chain processors
- `BaseEvaluator` class - Base class for evaluator plugins
- `BaseStorageBackend` class - Base class for storage plugins
- `BaseChainStepProcessor` class - Base class for processor plugins

**Key Features:**
- Runtime-checkable protocols using `@runtime_checkable`
- Abstract base classes for inheritance-based implementation
- Full type hints for IDE support
- Comprehensive docstrings

#### 2. `/home/user/hello-world/Promptly/promptly/plugins/__init__.py`
**Plugin loader and registry system**

Contains:
- `PluginRegistry` - Central registry for all plugins
- `PluginLoader` - Loads plugins from various sources
- Global registry and loader instances
- Convenience functions: `get_evaluator()`, `get_storage_backend()`, etc.
- `list_plugins()` - Lists all available plugins

**Key Features:**
- Automatic plugin discovery
- Support for loading from directories
- Module-based plugin loading
- Built-in plugin auto-registration

### Evaluator Plugins

#### 3. `/home/user/hello-world/Promptly/promptly/plugins/evaluators/keyword.py`
**Keyword-based evaluator plugin**

**Features:**
- Simple word overlap scoring
- Required keywords (must be present)
- Optional keywords (bonus points)
- Forbidden keywords (penalties)
- Minimum frequency checking
- Case-sensitive/insensitive modes
- Detailed metrics output

**Usage Example:**
```python
from plugins import get_evaluator

evaluator = get_evaluator("keyword")

# Simple usage
score = evaluator.evaluate(
    actual="The cat sat on the mat",
    expected="cat mat sat"
)

# Advanced usage with context
score = evaluator.evaluate(
    actual="Python is powerful",
    expected="",
    context={
        'required_keywords': ['python', 'powerful'],
        'optional_keywords': ['language'],
        'forbidden_keywords': ['slow', 'bad'],
        'min_length': 10,
        'max_length': 500
    }
)
```

#### 4. `/home/user/hello-world/Promptly/promptly/plugins/evaluators/semantic.py`
**Semantic similarity evaluator plugin**

**Features:**
- Multiple backends:
  - TF-IDF cosine similarity (default, no dependencies)
  - sentence-transformers (optional)
  - OpenAI embeddings (optional)
- Graceful fallback if dependencies unavailable
- Configurable similarity thresholds
- Detailed metrics including word counts

**Classes:**
- `SemanticSimilarityEvaluator` - Main semantic evaluator
- `ExactMatchEvaluator` - Bonus exact string matching evaluator

**Usage Example:**
```python
from plugins import get_evaluator

# Default TF-IDF backend
evaluator = get_evaluator("semantic", backend="tfidf")
score = evaluator.evaluate(
    actual="Machine learning enables computers to learn",
    expected="AI systems can be trained on data"
)

# With sentence-transformers
evaluator = get_evaluator(
    "semantic",
    backend="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)
```

#### 5. `/home/user/hello-world/Promptly/promptly/plugins/evaluators/__init__.py`
Exports: `KeywordEvaluator`, `SemanticSimilarityEvaluator`

### Storage Backend Plugins

#### 6. `/home/user/hello-world/Promptly/promptly/plugins/storage/sqlite.py`
**SQLite storage backend (refactored from original)**

**Features:**
- Database-backed persistence
- Full ACID transaction support
- Efficient querying
- Branch management
- Version history tracking
- Evaluation results storage
- Chain definitions storage

**Usage Example:**
```bash
# CLI
promptly init --storage sqlite

# Python
storage = get_storage_backend("sqlite")
storage.init_storage("/path/to/promptly.db")
```

#### 7. `/home/user/hello-world/Promptly/promptly/plugins/storage/json_file.py`
**JSON file storage backend**

**Features:**
- Human-readable JSON files
- Git-friendly structure
- No external dependencies
- Easy to inspect and edit manually
- File-per-commit architecture

**File Structure:**
```
.promptly/
  config.json              # Configuration
  branches/
    main.json             # Branch metadata
  prompts/
    abc123.json           # Individual prompt commits
  chains/
    pipeline.json         # Chain definitions
  evaluations/
    eval_timestamp.json   # Evaluation results
```

**Usage Example:**
```bash
# CLI
promptly init --storage json

# Python
storage = get_storage_backend("json")
storage.init_storage("/path/to/.promptly")
```

#### 8. `/home/user/hello-world/Promptly/promptly/plugins/storage/__init__.py`
Exports: `SQLiteStorage`, `JSONStorage`

### Chain Processors

#### 9. `/home/user/hello-world/Promptly/promptly/plugins/processors/__init__.py`
**Placeholder for future chain step processors**

Ready for custom processor implementations.

### Integration Layer

#### 10. `/home/user/hello-world/Promptly/promptly/promptly_plugins.py`
**Refactored Promptly class with plugin support**

**New Features:**
- `__init__()` accepts `storage_backend` parameter
- `init()` accepts `storage_backend` parameter
- `eval_prompt()` accepts `evaluator_name` parameter
- Automatic backend configuration persistence
- Plugin-based storage abstraction

**New CLI Commands:**
```bash
# Initialize with backend selection
promptly init --storage sqlite
promptly init --storage json

# List available plugins
promptly plugins

# Evaluate with custom evaluator
promptly eval run my_prompt tests.json --evaluator keyword
promptly eval run my_prompt tests.json --evaluator semantic
promptly eval run my_prompt tests.json --evaluator exact_match
```

### Documentation

#### 11. `/home/user/hello-world/Promptly/promptly/plugins/README.md`
**Comprehensive plugin development guide**

**Contents:**
- Plugin system overview
- Protocol definitions for each plugin type
- Built-in plugin documentation
- Creating custom plugins step-by-step
- Testing custom plugins
- Advanced examples:
  - Custom evaluator with external API
  - Custom storage with Redis caching
  - Sentiment analysis evaluator
- Best practices
- Plugin discovery mechanisms
- Troubleshooting guide
- Contributing guidelines

### Examples

#### 12. `/home/user/hello-world/Promptly/promptly/examples/plugin_usage.py`
**Comprehensive plugin usage examples**

**Examples Included:**
1. Keyword evaluator - basic and advanced usage
2. Semantic similarity evaluator
3. Exact match evaluator
4. Comparing multiple evaluators
5. Using different storage backends
6. Advanced context-based evaluation
7. Listing all available plugins

**Run Examples:**
```bash
cd Promptly/promptly
python examples/plugin_usage.py
```

## Usage Examples

### Example 1: Using Custom Evaluator

**Test file (tests.json):**
```json
[
  {
    "inputs": {"text": "Analyze this text"},
    "expected": "analysis summary",
    "context": {
      "required_keywords": ["analysis", "summary"],
      "optional_keywords": ["detailed", "thorough"]
    }
  }
]
```

**Command:**
```bash
promptly eval run my_prompt tests.json --evaluator keyword
```

**Python API:**
```python
from promptly import Promptly

promptly = Promptly()

test_cases = [
    {
        'inputs': {'text': 'Test input'},
        'expected': 'expected output',
        'context': {
            'required_keywords': ['test', 'output']
        }
    }
]

results = promptly.eval_prompt(
    'my_prompt',
    test_cases,
    evaluator_name='keyword'
)

for result in results:
    print(f"Score: {result['score']:.2f}")
    print(f"Metrics: {result['metrics']}")
```

### Example 2: Using Custom Storage Backend

**Initialize with JSON storage:**
```bash
promptly init --storage json
```

**Python API:**
```python
from promptly import Promptly

# Initialize with JSON backend
promptly = Promptly(storage_backend="json")
promptly.init()

# Add prompts - stored as JSON files
promptly.add(
    "my_prompt",
    "Summarize: {text}",
    metadata={"version": "1.0"}
)

# Retrieve prompt
prompt = promptly.get("my_prompt")
print(prompt['content'])
```

**Result Structure (.promptly/ directory):**
```
.promptly/
  backend.json          # {"storage_backend": "json"}
  config.json           # {"current_branch": "main"}
  branches/
    main.json          # Branch metadata
  prompts/
    abc123def456.json  # Prompt commit
  chains/
    my_chain.json      # Chain definition
```

### Example 3: Creating Custom Evaluator Plugin

**File: `Promptly/promptly/plugins/my_evaluator.py`**
```python
from plugins.base import BaseEvaluator

class LengthEvaluator(BaseEvaluator):
    """Evaluates based on text length"""

    def __init__(self, target_length: int = 100, tolerance: float = 0.2):
        super().__init__(
            name="length",
            description="Length-based evaluator"
        )
        self.target_length = target_length
        self.tolerance = tolerance

    def evaluate(self, actual: str, expected: str, context=None) -> float:
        if not actual:
            return 0.0

        actual_length = len(actual)
        min_length = self.target_length * (1 - self.tolerance)
        max_length = self.target_length * (1 + self.tolerance)

        if min_length <= actual_length <= max_length:
            return 1.0
        else:
            diff = abs(actual_length - self.target_length)
            max_diff = self.target_length * self.tolerance
            return max(0.0, 1.0 - (diff / max_diff))
```

**Register and use:**
```python
from plugins import get_registry

registry = get_registry()
registry.register_evaluator(LengthEvaluator)

# Use it
evaluator = get_evaluator("length", target_length=50)
score = evaluator.evaluate("This is a test", "")
```

## Plugin Development Guide Summary

### Creating an Evaluator Plugin

1. **Inherit from `BaseEvaluator`**
2. **Implement required methods:**
   - `__init__()` - Set name and description
   - `evaluate()` - Return score 0.0-1.0
   - `get_metrics()` (optional) - Return detailed metrics
3. **Register plugin:**
   - Automatic if in `plugins/evaluators/`
   - Manual via `registry.register_evaluator()`

### Creating a Storage Backend Plugin

1. **Inherit from `BaseStorageBackend`**
2. **Implement required methods:**
   - `init_storage()` - Initialize storage
   - `save_prompt()` - Persist prompts
   - `get_prompt()` - Retrieve prompts
   - `list_prompts()` - List all prompts
   - `create_branch()` - Branch management
   - `get_current_branch()` / `set_current_branch()`
   - `get_commit_history()` - Version history
   - `save_evaluation()` - Store eval results
   - `save_chain()` / `get_chain()` - Chain management
   - `close()` - Cleanup
3. **Register plugin:**
   - Automatic if in `plugins/storage/`
   - Manual via `registry.register_storage_backend()`

### Creating a Chain Processor Plugin

1. **Inherit from `BaseChainStepProcessor`**
2. **Implement required methods:**
   - `__init__()` - Set name and description
   - `process()` - Process step
   - `pre_process()` (optional) - Pre-processing hook
   - `post_process()` (optional) - Post-processing hook
3. **Register plugin:**
   - Manual via `registry.register_chain_processor()`

## Testing Plugins

**Run built-in examples:**
```bash
cd /home/user/hello-world/Promptly/promptly
python examples/plugin_usage.py
```

**Expected output:**
- Keyword evaluator examples with scores
- Semantic similarity comparisons
- Exact match tests
- Evaluator comparisons
- Storage backend listings
- Context-based evaluation demonstrations

## Integration with Existing Promptly

The plugin system is **backward compatible**. Existing code continues to work:

- Default storage backend: SQLite (same as before)
- Default evaluator: keyword (simple word overlap)
- All existing CLI commands work unchanged

**New capabilities added:**
- `--storage` flag on `init` command
- `--evaluator` flag on `eval run` command
- `plugins` command to list available plugins

## Key Benefits

1. **Extensibility** - Add new evaluators and storage backends without modifying core code
2. **Modularity** - Plugins are self-contained and independently testable
3. **Flexibility** - Choose the right tool for the job (SQLite vs JSON, keyword vs semantic)
4. **Maintainability** - Clear separation of concerns via protocols
5. **Testability** - Each plugin can be tested in isolation
6. **Backward Compatibility** - Existing functionality preserved

## Next Steps

1. **Add more evaluator plugins:**
   - BLEU score evaluator
   - ROUGE score evaluator
   - Perplexity evaluator
   - Custom domain-specific evaluators

2. **Add more storage backends:**
   - PostgreSQL backend
   - MongoDB backend
   - S3/cloud storage backend
   - Redis caching layer

3. **Implement chain processors:**
   - Data transformation processors
   - Filtering processors
   - Aggregation processors

4. **Testing:**
   - Unit tests for each plugin
   - Integration tests for plugin system
   - Performance benchmarks

5. **Documentation:**
   - Video tutorials
   - More code examples
   - API reference docs

## Files Summary

**Total Files Created: 12**

**Plugin System Core:**
- `plugins/base.py` - Protocol definitions
- `plugins/__init__.py` - Registry and loader

**Evaluator Plugins:**
- `plugins/evaluators/__init__.py`
- `plugins/evaluators/keyword.py`
- `plugins/evaluators/semantic.py`

**Storage Backend Plugins:**
- `plugins/storage/__init__.py`
- `plugins/storage/sqlite.py`
- `plugins/storage/json_file.py`

**Chain Processors:**
- `plugins/processors/__init__.py`

**Integration:**
- `promptly_plugins.py` - Refactored Promptly with plugin support

**Documentation & Examples:**
- `plugins/README.md` - Comprehensive plugin development guide
- `examples/plugin_usage.py` - Working examples

## Conclusion

The Promptly plugin system provides a robust, extensible architecture that enables:

- Custom evaluation logic tailored to specific use cases
- Flexible storage options for different deployment scenarios
- Easy extension without modifying core code
- Clear, documented APIs for plugin development

All plugins follow consistent patterns, are fully documented, and include working examples.
