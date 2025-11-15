# Promptly Plugin System

The Promptly plugin system allows you to extend Promptly with custom functionality in three key areas:

1. **Evaluators** - Custom scoring and evaluation logic
2. **Storage Backends** - Alternative storage implementations
3. **Chain Step Processors** - Custom processing for chain execution

## Plugin Types

### 1. Evaluator Plugins

Evaluators score the quality of model outputs against expected results.

#### Protocol Definition

```python
from plugins.base import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            name="my_evaluator",  # Unique identifier
            description="Description of what this evaluator does"
        )

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate actual output against expected output

        Args:
            actual: The actual output from the model
            expected: The expected output or reference
            context: Optional context with additional information

        Returns:
            float: Score between 0.0 and 1.0
        """
        # Your scoring logic here
        return 0.5

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optional: Return detailed metrics"""
        score = self.evaluate(actual, expected, context)
        return {
            "score": score,
            "my_custom_metric": 42
        }
```

#### Example: Keyword Evaluator

```python
from plugins.evaluators.keyword import KeywordEvaluator

# Basic usage - simple word overlap
evaluator = KeywordEvaluator()
score = evaluator.evaluate(
    actual="The cat sat on the mat",
    expected="cat mat sat"
)

# Advanced usage - with context
score = evaluator.evaluate(
    actual="Python is a programming language",
    expected="",
    context={
        'required_keywords': ['python', 'programming'],
        'optional_keywords': ['language', 'code'],
        'forbidden_keywords': ['java', 'javascript'],
        'min_length': 10,
        'max_length': 500
    }
)
```

#### Example: Semantic Similarity Evaluator

```python
from plugins.evaluators.semantic import SemanticSimilarityEvaluator

# Default TF-IDF backend (no dependencies)
evaluator = SemanticSimilarityEvaluator(backend="tfidf")

# With sentence-transformers (requires installation)
evaluator = SemanticSimilarityEvaluator(
    backend="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)

# With OpenAI embeddings (requires API key)
evaluator = SemanticSimilarityEvaluator(
    backend="openai",
    api_key="your-api-key"
)

score = evaluator.evaluate(
    actual="Machine learning enables computers to learn from data",
    expected="AI systems can be trained on datasets"
)
```

### 2. Storage Backend Plugins

Storage backends handle persistence of prompts, chains, and evaluations.

#### Protocol Definition

```python
from plugins.base import BaseStorageBackend

class MyStorage(BaseStorageBackend):
    def __init__(self):
        super().__init__(
            name="my_storage",
            description="My custom storage backend"
        )

    def init_storage(self, storage_path: str) -> None:
        """Initialize storage at the given path"""
        pass

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Save a prompt, return commit hash"""
        return "commit_hash"

    def get_prompt(self, name: str, branch: str = "main",
                   version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt"""
        return None

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """List all prompts on a branch"""
        return []

    # ... implement other required methods
```

#### Built-in Storage Backends

**SQLite Storage** (default):
- Database-backed storage
- Full transaction support
- Efficient querying
- Good for production use

```bash
promptly init --storage sqlite
```

**JSON File Storage**:
- File-based storage
- Human-readable
- Git-friendly
- Good for version control

```bash
promptly init --storage json
```

File structure with JSON backend:
```
.promptly/
  config.json              # Configuration
  branches/
    main.json             # Branch metadata
    feature.json
  prompts/
    abc123.json           # Prompt commits
    def456.json
  chains/
    pipeline1.json        # Chain definitions
  evaluations/
    test1_2024-01-15.json # Evaluation results
```

### 3. Chain Step Processor Plugins

Process individual steps in prompt chains with custom logic.

#### Protocol Definition

```python
from plugins.base import BaseChainStepProcessor

class MyProcessor(BaseChainStepProcessor):
    def __init__(self):
        super().__init__(
            name="my_processor",
            description="Custom chain step processor"
        )

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chain step

        Args:
            step_input: Input data for this step
            step_config: Configuration for this step

        Returns:
            Dict: Output data to pass to next step
        """
        # Your processing logic
        return {"result": "processed"}

    def pre_process(self, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Optional pre-processing hook"""
        return step_input

    def post_process(self, step_output: Dict[str, Any]) -> Dict[str, Any]:
        """Optional post-processing hook"""
        return step_output
```

## Using Plugins in Promptly

### CLI Usage

**Initialize with custom storage:**
```bash
promptly init --storage json
```

**Evaluate with custom evaluator:**
```bash
promptly eval run my_prompt tests.json --evaluator semantic
promptly eval run my_prompt tests.json --evaluator keyword
```

**List available plugins:**
```bash
promptly plugins
```

### Python API Usage

```python
from promptly import Promptly
from plugins import get_evaluator, get_storage_backend

# Initialize with custom storage
promptly = Promptly(storage_backend="json")
promptly.init()

# Use custom evaluator
evaluator = get_evaluator("semantic", backend="tfidf")
score = evaluator.evaluate(actual, expected)

# Evaluate prompts with plugin
results = promptly.eval_prompt(
    "my_prompt",
    test_cases,
    evaluator_name="keyword"
)
```

## Creating Custom Plugins

### Step 1: Create Plugin File

Create a Python file in the plugins directory:

```
promptly/plugins/my_custom_plugin.py
```

### Step 2: Implement Protocol

```python
from plugins.base import BaseEvaluator

class SentimentEvaluator(BaseEvaluator):
    """Evaluates if output matches expected sentiment"""

    def __init__(self):
        super().__init__(
            name="sentiment",
            description="Sentiment-based evaluator"
        )

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        # Simple sentiment scoring
        positive_words = ['good', 'great', 'excellent', 'amazing']
        negative_words = ['bad', 'poor', 'terrible', 'awful']

        actual_lower = actual.lower()
        expected_sentiment = expected.lower()

        if expected_sentiment == "positive":
            score = sum(1 for word in positive_words if word in actual_lower) / len(positive_words)
        elif expected_sentiment == "negative":
            score = sum(1 for word in negative_words if word in actual_lower) / len(negative_words)
        else:
            score = 0.5

        return min(1.0, max(0.0, score))
```

### Step 3: Register Plugin

Plugins are automatically discovered if placed in the plugins directory. Or manually register:

```python
from plugins import get_registry

registry = get_registry()
registry.register_evaluator(SentimentEvaluator)
```

### Step 4: Use Plugin

```bash
promptly eval run my_prompt tests.json --evaluator sentiment
```

## Testing Custom Plugins

```python
def test_my_evaluator():
    from plugins import get_evaluator

    evaluator = get_evaluator("my_evaluator")

    score = evaluator.evaluate(
        actual="test output",
        expected="expected output"
    )

    assert 0.0 <= score <= 1.0
    print(f"Score: {score}")

    metrics = evaluator.get_metrics(
        actual="test output",
        expected="expected output"
    )

    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    test_my_evaluator()
```

## Advanced Examples

### Custom Evaluator with External API

```python
from plugins.base import BaseEvaluator
import requests

class APIEvaluator(BaseEvaluator):
    """Evaluates using external API"""

    def __init__(self, api_url: str, api_key: str):
        super().__init__(
            name="api_evaluator",
            description="External API-based evaluator"
        )
        self.api_url = api_url
        self.api_key = api_key

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        response = requests.post(
            self.api_url,
            json={
                "actual": actual,
                "expected": expected
            },
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

        result = response.json()
        return result.get("score", 0.0)
```

### Custom Storage with Caching

```python
from plugins.base import BaseStorageBackend
import redis
import json

class RedisStorage(BaseStorageBackend):
    """Redis-backed storage with caching"""

    def __init__(self):
        super().__init__(
            name="redis",
            description="Redis storage backend with caching"
        )
        self.redis_client = None

    def init_storage(self, storage_path: str) -> None:
        # Parse redis connection string
        self.redis_client = redis.Redis.from_url(storage_path)

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        commit_hash = self._generate_hash(prompt_data)

        # Save to Redis
        key = f"prompt:{prompt_data['name']}:{prompt_data['branch']}"
        self.redis_client.set(key, json.dumps(prompt_data))

        return commit_hash

    # ... implement other methods
```

## Best Practices

1. **Always validate inputs** - Check that actual/expected are non-None
2. **Handle errors gracefully** - Return sensible defaults on errors
3. **Document your plugins** - Include docstrings and examples
4. **Test thoroughly** - Write unit tests for your plugins
5. **Version your plugins** - Track versions for compatibility
6. **Use type hints** - Make your plugins easier to use
7. **Provide defaults** - Make plugins work out of the box

## Plugin Discovery

Plugins can be loaded from:

1. **Built-in plugins** - Automatically loaded from `plugins/evaluators/` and `plugins/storage/`
2. **Custom directory** - Use `PluginLoader.load_from_directory(path)`
3. **Python modules** - Use `PluginLoader.load_plugin_module(module_path)`

Example custom plugin loading:

```python
from plugins import get_loader

loader = get_loader()

# Load from directory
loader.load_from_directory(Path("/path/to/custom/plugins"))

# Load specific module
loader.load_plugin_module("my_company.promptly_plugins")
```

## Troubleshooting

**Plugin not found:**
- Ensure plugin file is in correct directory
- Check plugin class inherits from Base class or implements protocol
- Verify plugin has unique name

**Import errors:**
- Check all dependencies are installed
- Use graceful fallbacks for optional dependencies

**Type errors:**
- Ensure methods match protocol signatures
- Use proper type hints

## Contributing Plugins

To contribute a plugin to Promptly:

1. Create plugin following protocol
2. Add comprehensive documentation
3. Include unit tests
4. Add example usage
5. Submit pull request

## Resources

- [Protocol Documentation](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Example Plugins](/plugins/evaluators/)
- [Plugin Registry Source](/plugins/__init__.py)

---

For questions or support, please open an issue on GitHub.
