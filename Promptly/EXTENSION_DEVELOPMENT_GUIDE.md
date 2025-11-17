# Promptly Extension Development Guide

**For Plugin Developers and Extension Authors**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Plugin Types](#plugin-types)
3. [Creating Custom Evaluators](#creating-custom-evaluators)
4. [Creating Storage Backends](#creating-storage-backends)
5. [Creating Chain Processors](#creating-chain-processors)
6. [Testing Guide](#testing-guide)
7. [Publishing Guide](#publishing-guide)
8. [Best Practices](#best-practices)

---

## Introduction

Promptly's plugin architecture allows developers to extend functionality with custom:
- **Evaluators** - Custom evaluation logic for prompts
- **Storage Backends** - Alternative storage systems
- **Chain Step Processors** - Custom processing logic for chains

### Plugin Architecture

```
┌─────────────────────────────────────────┐
│          Promptly Core Engine            │
├─────────────────────────────────────────┤
│                                          │
│  ┌────────────┐  ┌────────────┐        │
│  │  Protocol  │  │  Protocol  │        │
│  │Definitions │  │ Discovery  │        │
│  └────────────┘  └────────────┘        │
│                                          │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐   ┌────▼────┐  ┌────▼────┐
│Eval   │   │Storage  │  │Processor│
│Plugin │   │Plugin   │  │Plugin   │
└───────┘   └─────────┘  └─────────┘
```

---

## Plugin Types

### Evaluator Plugins

**Purpose:** Assess prompt output quality

**Use Cases:**
- Custom scoring algorithms
- Domain-specific evaluation
- Multi-metric assessment
- Integration with external services

**Protocol:** `EvaluatorPlugin` or `BaseEvaluator`

### Storage Backend Plugins

**Purpose:** Alternative data persistence

**Use Cases:**
- NoSQL databases (MongoDB, Redis)
- Cloud storage (S3, GCS)
- Git-based storage
- Custom versioning systems

**Protocol:** `StorageBackend` or `BaseStorageBackend`

### Chain Step Processor Plugins

**Purpose:** Custom chain processing logic

**Use Cases:**
- Data transformation
- External API integration
- Conditional logic
- Loop and retry mechanisms

**Protocol:** `ChainStepProcessor` or `BaseChainStepProcessor`

---

## Creating Custom Evaluators

### Basic Evaluator

```python
from promptly.plugins.base import BaseEvaluator
from typing import Dict, Any, Optional

class LengthEvaluator(BaseEvaluator):
    """Evaluates output based on length constraints"""

    def __init__(self, target_length: int = 100, tolerance: float = 0.2):
        super().__init__(
            name="length",
            description="Evaluates output length against target"
        )
        self.target_length = target_length
        self.tolerance = tolerance

    def evaluate(
        self,
        actual: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate based on length proximity to target

        Args:
            actual: The actual output from the model
            expected: The expected output (unused in this evaluator)
            context: Optional context with 'target_length' override

        Returns:
            float: Score between 0.0 and 1.0
        """
        # Allow context to override instance target
        target = context.get('target_length', self.target_length) if context else self.target_length

        actual_length = len(actual)
        diff = abs(actual_length - target)
        max_diff = target * self.tolerance

        # Calculate score based on how close to target
        if diff <= max_diff:
            score = 1.0 - (diff / max_diff) * 0.5  # Max penalty 50%
        else:
            score = max(0.0, 0.5 - (diff - max_diff) / target)

        return score

    def get_metrics(
        self,
        actual: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return detailed metrics"""
        score = self.evaluate(actual, expected, context)
        target = context.get('target_length', self.target_length) if context else self.target_length

        actual_length = len(actual)
        diff = abs(actual_length - target)

        return {
            'score': score,
            'actual_length': actual_length,
            'target_length': target,
            'difference': diff,
            'within_tolerance': diff <= (target * self.tolerance)
        }
```

### Advanced Evaluator with External API

```python
import requests
from promptly.plugins.base import BaseEvaluator
from typing import Dict, Any, Optional

class GPTJudgeEvaluator(BaseEvaluator):
    """Uses GPT-4 as a judge for evaluation"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(
            name="gpt_judge",
            description="Uses GPT-4 as evaluation judge"
        )
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def evaluate(
        self,
        actual: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Evaluate using GPT-4 as judge"""

        # Build evaluation prompt
        eval_prompt = f"""
        Evaluate the following output against the expected output.
        Rate from 0.0 to 1.0 based on:
        - Accuracy
        - Completeness
        - Relevance

        Expected: {expected}
        Actual: {actual}

        Respond with only a number between 0.0 and 1.0.
        """

        # Call GPT-4
        response = requests.post(
            self.base_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': self.model,
                'messages': [
                    {'role': 'user', 'content': eval_prompt}
                ],
                'temperature': 0.0
            }
        )

        result = response.json()
        score_text = result['choices'][0]['message']['content'].strip()

        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except ValueError:
            return 0.5  # Default score if parsing fails

    def get_metrics(
        self,
        actual: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get detailed evaluation metrics"""
        score = self.evaluate(actual, expected, context)

        return {
            'score': score,
            'model': self.model,
            'evaluator': 'gpt_judge',
            'actual_length': len(actual),
            'expected_length': len(expected)
        }
```

### Composite Evaluator

```python
from promptly.plugins.base import BaseEvaluator
from typing import Dict, Any, Optional, List

class CompositeEvaluator(BaseEvaluator):
    """Combines multiple evaluators"""

    def __init__(self, evaluators: List[BaseEvaluator], weights: List[float] = None):
        super().__init__(
            name="composite",
            description="Combines multiple evaluators with weighted average"
        )
        self.evaluators = evaluators

        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(evaluators)] * len(evaluators)

        assert len(weights) == len(evaluators), "Weights must match evaluators"
        assert abs(sum(weights) - 1.0) < 0.001, "Weights must sum to 1.0"

        self.weights = weights

    def evaluate(
        self,
        actual: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Combine scores from all evaluators"""
        scores = []

        for evaluator in self.evaluators:
            score = evaluator.evaluate(actual, expected, context)
            scores.append(score)

        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, self.weights))
        return final_score

    def get_metrics(
        self,
        actual: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get metrics from all evaluators"""
        composite_score = self.evaluate(actual, expected, context)

        individual_metrics = {}
        for evaluator in self.evaluators:
            metrics = evaluator.get_metrics(actual, expected, context)
            individual_metrics[evaluator.name] = metrics

        return {
            'score': composite_score,
            'individual_scores': individual_metrics,
            'weights': dict(zip([e.name for e in self.evaluators], self.weights))
        }
```

---

## Creating Storage Backends

### Basic Storage Backend

```python
from promptly.plugins.base import BaseStorageBackend
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

class JSONFileStorage(BaseStorageBackend):
    """Simple JSON file-based storage"""

    def __init__(self):
        super().__init__(
            name="json_file",
            description="JSON file storage backend"
        )
        self.storage_path = None
        self.data = {
            'prompts': {},
            'branches': {},
            'config': {}
        }

    def init_storage(self, storage_path: str) -> None:
        """Initialize storage at path"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing data
        data_file = self.storage_path / 'data.json'
        if data_file.exists():
            with open(data_file, 'r') as f:
                self.data = json.load(f)

    def _save_data(self):
        """Persist data to disk"""
        data_file = self.storage_path / 'data.json'
        with open(data_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Save prompt"""
        commit_hash = prompt_data['commit_hash']
        name = prompt_data['name']
        branch = prompt_data['branch']

        # Store in nested structure
        key = f"{branch}:{name}:{commit_hash}"
        self.data['prompts'][key] = prompt_data

        self._save_data()
        return commit_hash

    def get_prompt(
        self,
        name: str,
        branch: str = "main",
        version: Optional[int] = None,
        commit_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve prompt"""

        if commit_hash:
            key = f"{branch}:{name}:{commit_hash}"
            return self.data['prompts'].get(key)

        # Find latest version
        matching = [
            p for k, p in self.data['prompts'].items()
            if p['name'] == name and p['branch'] == branch
        ]

        if version:
            matching = [p for p in matching if p['version'] == version]

        if not matching:
            return None

        # Return latest
        return max(matching, key=lambda p: p['version'])

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """List all prompts on branch"""
        prompts = {}

        for prompt_data in self.data['prompts'].values():
            if prompt_data['branch'] != branch:
                continue

            name = prompt_data['name']
            if name not in prompts or prompt_data['version'] > prompts[name]['version']:
                prompts[name] = prompt_data

        return list(prompts.values())

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create new branch"""
        # Copy prompts from source branch
        source_prompts = self.list_prompts(from_branch)

        for prompt in source_prompts:
            new_prompt = prompt.copy()
            new_prompt['branch'] = branch_name
            self.save_prompt(new_prompt)

        # Save branch metadata
        self.data['branches'][branch_name] = {
            'name': branch_name,
            'from_branch': from_branch
        }
        self._save_data()

    def get_current_branch(self) -> str:
        """Get current branch"""
        return self.data['config'].get('current_branch', 'main')

    def set_current_branch(self, branch_name: str) -> None:
        """Set current branch"""
        self.data['config']['current_branch'] = branch_name
        self._save_data()

    def get_commit_history(
        self,
        name: Optional[str] = None,
        branch: str = "main",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get commit history"""
        commits = []

        for prompt_data in self.data['prompts'].values():
            if prompt_data['branch'] != branch:
                continue
            if name and prompt_data['name'] != name:
                continue

            commits.append({
                'commit_hash': prompt_data['commit_hash'],
                'name': prompt_data['name'],
                'version': prompt_data['version'],
                'branch': prompt_data['branch'],
                'created_at': prompt_data.get('created_at')
            })

        # Sort by version descending
        commits.sort(key=lambda c: c['version'], reverse=True)
        return commits[:limit]

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        if 'evaluations' not in self.data:
            self.data['evaluations'] = []

        self.data['evaluations'].append(eval_data)
        self._save_data()

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save chain definition"""
        if 'chains' not in self.data:
            self.data['chains'] = {}

        self.data['chains'][chain_data['name']] = chain_data
        self._save_data()

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get chain definition"""
        return self.data.get('chains', {}).get(name)

    def close(self) -> None:
        """Cleanup"""
        self._save_data()
```

### Advanced Storage: PostgreSQL Backend

```python
from promptly.plugins.base import BaseStorageBackend
from typing import Dict, List, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import json

class PostgreSQLStorage(BaseStorageBackend):
    """PostgreSQL storage backend for production"""

    def __init__(self):
        super().__init__(
            name="postgresql",
            description="PostgreSQL storage backend"
        )
        self.conn = None

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize PostgreSQL connection

        storage_path format: postgresql://user:pass@host:port/dbname
        """
        self.conn = psycopg2.connect(storage_path)
        self._create_schema()

    def _create_schema(self):
        """Create database schema"""
        with self.conn.cursor() as cursor:
            # Prompts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    branch VARCHAR(255) NOT NULL DEFAULT 'main',
                    version INTEGER NOT NULL DEFAULT 1,
                    parent_id INTEGER REFERENCES prompts(id),
                    commit_hash VARCHAR(64) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    UNIQUE(name, branch, version)
                )
            """)

            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompts_name_branch
                ON prompts(name, branch)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompts_commit_hash
                ON prompts(commit_hash)
            """)

            # Branches table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS branches (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    head_commit VARCHAR(64) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Config table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key VARCHAR(255) PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Initialize main branch
            cursor.execute("""
                INSERT INTO branches (name, head_commit)
                VALUES ('main', 'init')
                ON CONFLICT DO NOTHING
            """)

            cursor.execute("""
                INSERT INTO config (key, value)
                VALUES ('current_branch', 'main')
                ON CONFLICT DO NOTHING
            """)

            self.conn.commit()

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Save prompt to PostgreSQL"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO prompts (
                    name, content, branch, version, parent_id, commit_hash, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING commit_hash
            """, (
                prompt_data['name'],
                prompt_data['content'],
                prompt_data['branch'],
                prompt_data['version'],
                prompt_data.get('parent_id'),
                prompt_data['commit_hash'],
                json.dumps(prompt_data.get('metadata', {}))
            ))

            commit_hash = cursor.fetchone()[0]
            self.conn.commit()
            return commit_hash

    def get_prompt(
        self,
        name: str,
        branch: str = "main",
        version: Optional[int] = None,
        commit_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve prompt from PostgreSQL"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if commit_hash:
                cursor.execute("""
                    SELECT * FROM prompts
                    WHERE name = %s AND commit_hash = %s
                """, (name, commit_hash))
            elif version:
                cursor.execute("""
                    SELECT * FROM prompts
                    WHERE name = %s AND branch = %s AND version = %s
                """, (name, branch, version))
            else:
                cursor.execute("""
                    SELECT * FROM prompts
                    WHERE name = %s AND branch = %s
                    ORDER BY version DESC
                    LIMIT 1
                """, (name, branch))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    # ... implement other methods similarly
```

---

## Creating Chain Processors

### Basic Processor

```python
from promptly.plugins.base import BaseChainStepProcessor
from typing import Dict, Any

class TransformProcessor(BaseChainStepProcessor):
    """Transforms data between chain steps"""

    def __init__(self):
        super().__init__(
            name="transform",
            description="Transforms data between chain steps"
        )

    def process(
        self,
        step_input: Dict[str, Any],
        step_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process the step

        Config options:
        - transform_type: 'extract', 'map', 'filter', 'aggregate'
        - field: field name to transform
        - method: transformation method
        """
        transform_type = step_config.get('transform_type', 'extract')

        if transform_type == 'extract':
            return self._extract(step_input, step_config)
        elif transform_type == 'map':
            return self._map(step_input, step_config)
        elif transform_type == 'filter':
            return self._filter(step_input, step_config)
        elif transform_type == 'aggregate':
            return self._aggregate(step_input, step_config)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def _extract(self, input_data: Dict, config: Dict) -> Dict:
        """Extract specific fields"""
        fields = config.get('fields', [])
        result = {field: input_data.get(field) for field in fields}
        return result

    def _map(self, input_data: Dict, config: Dict) -> Dict:
        """Map values using function"""
        field = config.get('field')
        mapping = config.get('mapping', {})

        value = input_data.get(field)
        mapped_value = mapping.get(value, value)

        return {**input_data, field: mapped_value}

    def _filter(self, input_data: Dict, config: Dict) -> Dict:
        """Filter data based on condition"""
        condition = config.get('condition', {})
        field = condition.get('field')
        operator = condition.get('operator', 'eq')
        value = condition.get('value')

        actual = input_data.get(field)

        operators = {
            'eq': lambda a, b: a == b,
            'ne': lambda a, b: a != b,
            'gt': lambda a, b: a > b,
            'lt': lambda a, b: a < b,
        }

        if operators[operator](actual, value):
            return input_data
        else:
            return {}  # Filtered out

    def _aggregate(self, input_data: Dict, config: Dict) -> Dict:
        """Aggregate list values"""
        field = config.get('field')
        method = config.get('method', 'sum')

        values = input_data.get(field, [])

        if method == 'sum':
            result = sum(values)
        elif method == 'avg':
            result = sum(values) / len(values) if values else 0
        elif method == 'max':
            result = max(values) if values else None
        elif method == 'min':
            result = min(values) if values else None
        else:
            result = values

        return {**input_data, f'{field}_aggregated': result}
```

---

## Testing Guide

### Unit Testing

```python
import unittest
from my_plugin import LengthEvaluator

class TestLengthEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = LengthEvaluator(target_length=100, tolerance=0.2)

    def test_exact_length(self):
        """Test with exact target length"""
        actual = "a" * 100
        score = self.evaluator.evaluate(actual, "")
        self.assertAlmostEqual(score, 1.0)

    def test_within_tolerance(self):
        """Test within tolerance range"""
        actual = "a" * 110  # 10% over
        score = self.evaluator.evaluate(actual, "")
        self.assertGreater(score, 0.5)

    def test_outside_tolerance(self):
        """Test outside tolerance range"""
        actual = "a" * 200  # 100% over
        score = self.evaluator.evaluate(actual, "")
        self.assertLess(score, 0.5)

    def test_context_override(self):
        """Test context overriding target"""
        actual = "a" * 50
        score = self.evaluator.evaluate(
            actual,
            "",
            context={'target_length': 50}
        )
        self.assertAlmostEqual(score, 1.0)

    def test_metrics(self):
        """Test detailed metrics"""
        actual = "a" * 110
        metrics = self.evaluator.get_metrics(actual, "")

        self.assertIn('score', metrics)
        self.assertIn('actual_length', metrics)
        self.assertIn('target_length', metrics)
        self.assertEqual(metrics['actual_length'], 110)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
import unittest
from promptly import Promptly
from my_plugin import LengthEvaluator

class TestEvaluatorIntegration(unittest.TestCase):

    def setUp(self):
        self.p = Promptly()
        self.p.init()
        self.p.add("test_prompt", "Test: {input}")

        self.evaluator = LengthEvaluator(target_length=50)

    def test_with_promptly(self):
        """Test evaluator with Promptly"""
        test_cases = [
            {
                'inputs': {'input': 'test'},
                'expected': '',
                'evaluator': lambda a, e: self.evaluator.evaluate(a, e)
            }
        ]

        # Mock model that returns 50-char output
        def mock_model(prompt):
            return "a" * 50

        results = self.p.eval_prompt("test_prompt", test_cases, model_func=mock_model)

        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]['score'], 0.9)
```

---

## Publishing Guide

### Package Structure

```
my-promptly-plugin/
├── setup.py
├── README.md
├── LICENSE
├── requirements.txt
├── my_plugin/
│   ├── __init__.py
│   ├── evaluator.py
│   └── version.py
├── tests/
│   ├── __init__.py
│   ├── test_evaluator.py
│   └── test_integration.py
└── examples/
    └── usage_example.py
```

### setup.py

```python
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="promptly-length-evaluator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Length-based evaluator plugin for Promptly",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/promptly-length-evaluator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires='>=3.7',
    install_requires=[
        "promptly>=1.0.0",
    ],
    entry_points={
        'promptly.plugins.evaluators': [
            'length = my_plugin:LengthEvaluator',
        ],
    },
)
```

### Publishing to PyPI

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Or test on TestPyPI first
python -m twine upload --repository testpypi dist/*
```

---

## Best Practices

### 1. Follow Protocol Specifications
- Implement all required methods
- Use proper type hints
- Return expected types

### 2. Add Comprehensive Documentation
- Docstrings for all classes and methods
- Usage examples in README
- API reference

### 3. Write Tests
- Unit tests for all methods
- Integration tests with Promptly
- Edge case testing
- Performance testing for critical operations

### 4. Handle Errors Gracefully
- Validate inputs
- Provide helpful error messages
- Don't crash on unexpected input

### 5. Version Your Plugin
- Use semantic versioning
- Maintain changelog
- Document breaking changes

### 6. Optimize Performance
- Cache expensive operations
- Use efficient algorithms
- Profile critical paths

### 7. Support Configuration
- Allow customization via config
- Provide sensible defaults
- Document all options

### 8. Add Examples
- Provide working examples
- Show common use cases
- Include troubleshooting tips

---

## Need Help?

- **Protocol Documentation**: See `/Promptly/promptly/plugins/base.py`
- **Built-in Examples**: See `/Promptly/promptly/plugins/evaluators/`
- **Testing Examples**: See `/Promptly/promptly/plugins/tests/`
- **Community Forum**: https://forum.promptly.dev
- **Discord**: https://discord.gg/promptly

**Happy plugin development!** 🔌
