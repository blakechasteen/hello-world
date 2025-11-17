# Custom Evaluators Integration Guide

Complete guide for integrating custom evaluators into your Promptly workflows.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Integration](#basic-integration)
3. [Advanced Patterns](#advanced-patterns)
4. [Production Deployment](#production-deployment)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring & Observability](#monitoring--observability)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

```bash
# Install Promptly
pip install promptly

# Install optional dependencies
pip install jsonschema pyyaml sqlparse lxml redis openai anthropic
```

### Project Structure

```
your-project/
├── evaluators/
│   ├── __init__.py
│   ├── custom_evaluators/  # Custom evaluator package
│   └── config.yaml         # Evaluator configuration
├── prompts/
│   └── my_prompt.txt
├── tests/
│   └── test_evaluators.py
└── main.py
```

---

## Basic Integration

### Step 1: Import and Initialize

```python
from promptly import Promptly
from custom_evaluators import CustomerServiceEvaluator

# Initialize Promptly
promptly = Promptly(storage_path='./prompts')

# Initialize evaluator
evaluator = CustomerServiceEvaluator(min_length=50, max_length=500)
```

### Step 2: Create Evaluation Function

```python
def evaluate_response(actual, expected, context=None):
    """
    Wrapper function for Promptly integration.

    Args:
        actual: The actual LLM output
        expected: The expected/reference output
        context: Optional context dictionary

    Returns:
        Dict with score and metadata
    """
    metrics = evaluator.get_metrics(actual, expected, context)

    return {
        'score': metrics['score'],
        'passed': metrics['passed'],
        'details': {
            'rules_passed': metrics['rules_passed'],
            'total_rules': metrics['total_rules'],
            'rule_results': metrics['rule_results']
        }
    }
```

### Step 3: Use in Promptly Tests

```python
# Define test cases
test_cases = [
    {
        'inputs': {'customer_query': 'My order is late'},
        'expected': 'Professional empathetic response with resolution',
        'evaluator': evaluate_response
    },
    {
        'inputs': {'customer_query': 'How do I return an item?'},
        'expected': 'Clear return instructions',
        'evaluator': evaluate_response
    }
]

# Run evaluation
results = promptly.eval_prompt('customer_service_prompt', test_cases)

# Print results
for i, result in enumerate(results):
    print(f"\nTest {i+1}:")
    print(f"  Score: {result['score']:.2f}")
    print(f"  Passed: {result['passed']}")
    print(f"  Details: {result['details']}")
```

---

## Advanced Patterns

### Pattern 1: Multi-Stage Evaluation Pipeline

```python
from custom_evaluators import (
    JSONValidator,
    CustomerServiceEvaluator,
    BrandVoiceEvaluator,
    HumanInTheLoopEvaluator
)

class EvaluationPipeline:
    """
    Multi-stage evaluation with progressive filtering.
    """

    def __init__(self):
        self.stages = [
            ('structure', JSONValidator(required_fields=['response']), 0.8, True),
            ('quality', CustomerServiceEvaluator(), 0.7, False),
            ('brand', BrandVoiceEvaluator(), 0.6, False),
            ('human', HumanInTheLoopEvaluator(), 0.5, False)
        ]

    def evaluate(self, actual, expected, context=None):
        """
        Run evaluation through all stages.

        Args:
            actual: Output to evaluate
            expected: Expected output
            context: Optional context

        Returns:
            Dict with results from all stages
        """
        results = {
            'stages': [],
            'overall_score': 0.0,
            'passed': True,
            'stopped_at': None
        }

        total_weight = 0.0
        weighted_score = 0.0

        for stage_name, evaluator, threshold, stop_on_fail in self.stages:
            # Run stage evaluation
            score = evaluator.evaluate(actual, expected, context)
            passed = score >= threshold

            stage_result = {
                'name': stage_name,
                'score': score,
                'threshold': threshold,
                'passed': passed
            }

            results['stages'].append(stage_result)

            # Update weighted score
            weight = 1.0
            weighted_score += score * weight
            total_weight += weight

            # Check if should stop
            if not passed and stop_on_fail:
                results['passed'] = False
                results['stopped_at'] = stage_name
                results['overall_score'] = weighted_score / total_weight
                return results

        # Calculate final score
        results['overall_score'] = weighted_score / total_weight
        results['passed'] = results['overall_score'] >= 0.7

        return results

# Usage
pipeline = EvaluationPipeline()

test_output = '{"response": "Hello! I can help you with that."}'
result = pipeline.evaluate(test_output, "")

print(f"Overall Score: {result['overall_score']:.2f}")
print(f"Passed: {result['passed']}")

for stage in result['stages']:
    print(f"  {stage['name']}: {stage['score']:.2f} ({'✓' if stage['passed'] else '✗'})")
```

### Pattern 2: Consensus Evaluation with Fallback

```python
from custom_evaluators.consensus import (
    MultiModelConsensusEvaluator,
    ConsensusStrategy,
    OpenAIProvider,
    AnthropicProvider,
    MockProvider
)

class RobustConsensusEvaluator:
    """
    Consensus evaluator with automatic fallback.
    """

    def __init__(self, openai_key=None, anthropic_key=None):
        # Primary providers (real LLMs)
        primary_providers = []

        if openai_key:
            try:
                primary_providers.append(
                    OpenAIProvider(model="gpt-4", api_key=openai_key)
                )
            except Exception as e:
                print(f"Warning: OpenAI provider failed: {e}")

        if anthropic_key:
            try:
                primary_providers.append(
                    AnthropicProvider(model="claude-3-sonnet-20240229", api_key=anthropic_key)
                )
            except Exception as e:
                print(f"Warning: Anthropic provider failed: {e}")

        # Fallback providers (mock for testing)
        fallback_providers = [
            MockProvider("fallback_1", bias=0.7),
            MockProvider("fallback_2", bias=0.75)
        ]

        # Use primary if available, otherwise fallback
        providers = primary_providers if primary_providers else fallback_providers

        self.evaluator = MultiModelConsensusEvaluator(
            providers=providers,
            strategy=ConsensusStrategy.MEDIAN,
            disagreement_threshold=0.3,
            cache_enabled=True
        )

        self.using_fallback = len(primary_providers) == 0

    def evaluate(self, actual, expected, context=None):
        """Evaluate with automatic fallback"""
        metrics = self.evaluator.get_metrics(actual, expected, context)

        # Add warning if using fallback
        if self.using_fallback:
            metrics['warning'] = 'Using fallback evaluators (mock providers)'

        return metrics

# Usage
import os

evaluator = RobustConsensusEvaluator(
    openai_key=os.getenv('OPENAI_API_KEY'),
    anthropic_key=os.getenv('ANTHROPIC_API_KEY')
)

result = evaluator.evaluate("Test output", "Expected")
print(f"Score: {result['score']:.3f}")
if 'warning' in result:
    print(f"⚠️  {result['warning']}")
```

### Pattern 3: Domain-Specific Evaluation Factory

```python
from custom_evaluators.domain_specific import (
    MedicalTerminologyEvaluator,
    LegalCitationEvaluator,
    CodeSecurityEvaluator,
    BrandVoiceEvaluator
)

class DomainEvaluatorFactory:
    """
    Factory for creating domain-specific evaluators.
    """

    @staticmethod
    def create(domain, **kwargs):
        """
        Create evaluator for specific domain.

        Args:
            domain: Domain name (medical, legal, code, marketing)
            **kwargs: Domain-specific configuration

        Returns:
            Domain-specific evaluator instance
        """
        evaluators = {
            'medical': MedicalTerminologyEvaluator,
            'legal': LegalCitationEvaluator,
            'code': CodeSecurityEvaluator,
            'marketing': BrandVoiceEvaluator
        }

        if domain not in evaluators:
            raise ValueError(f"Unknown domain: {domain}")

        return evaluators[domain](**kwargs)

    @staticmethod
    def create_from_config(config_dict):
        """Create evaluator from configuration dictionary"""
        domain = config_dict.pop('domain')
        return DomainEvaluatorFactory.create(domain, **config_dict)

# Usage
medical_eval = DomainEvaluatorFactory.create('medical')
code_eval = DomainEvaluatorFactory.create('code', language='python')

# From config
config = {
    'domain': 'marketing',
    'brand_guidelines': {
        'voice_attributes': ['friendly', 'professional'],
        'tone': 'conversational'
    }
}
marketing_eval = DomainEvaluatorFactory.create_from_config(config)
```

---

## Production Deployment

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Promptly   │  │  API Server  │  │   Web UI        │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘   │
└─────────┼──────────────────┼───────────────────┼────────────┘
          │                  │                   │
┌─────────▼──────────────────▼───────────────────▼────────────┐
│                  Evaluation Service                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Evaluation Pipeline Orchestrator                │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Business │ │Consensus │ │Structured│ │    HITL      │  │
│  │  Logic   │ │          │ │          │ │              │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
          │                  │                   │
┌─────────▼──────────────────▼───────────────────▼────────────┐
│                    Infrastructure                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │  Redis   │ │PostgreSQL│ │  OpenAI  │ │  Monitoring  │  │
│  │  Cache   │ │          │ │   API    │ │              │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy evaluators
COPY custom_evaluators/ ./custom_evaluators/
COPY config.yaml .

# Copy application
COPY main.py .

# Environment variables
ENV PYTHONPATH=/app
ENV REDIS_HOST=redis
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Run application
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - postgres
    volumes:
      - ./prompts:/app/prompts
      - ./logs:/app/logs

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=evaluations
      - POSTGRES_USER=promptly
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: promptly-evaluators
spec:
  replicas: 3
  selector:
    matchLabels:
      app: promptly-evaluators
  template:
    metadata:
      labels:
        app: promptly-evaluators
    spec:
      containers:
      - name: evaluator
        image: promptly-evaluators:latest
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: evaluator-service
spec:
  selector:
    app: promptly-evaluators
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

---

## Performance Optimization

### 1. Caching Strategy

```python
from custom_evaluators.consensus import MultiModelConsensusEvaluator
import redis

class OptimizedConsensusEvaluator:
    """
    Consensus evaluator with Redis caching.
    """

    def __init__(self):
        # Redis cache
        self.redis = redis.Redis(host='localhost', port=6379, db=0)

        # Evaluator with in-memory cache
        self.evaluator = MultiModelConsensusEvaluator(
            cache_enabled=True,
            cache_ttl=3600
        )

    def evaluate(self, actual, expected, context=None):
        """Evaluate with two-level caching"""
        # Try Redis cache first
        cache_key = f"eval:{hash(actual)}:{hash(expected)}"
        cached = self.redis.get(cache_key)

        if cached:
            return float(cached)

        # Evaluate
        score = self.evaluator.evaluate(actual, expected, context)

        # Cache in Redis
        self.redis.setex(cache_key, 3600, score)

        return score
```

### 2. Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
from custom_evaluators import CustomerServiceEvaluator

class BatchEvaluator:
    """
    Process evaluations in batches with parallelization.
    """

    def __init__(self, max_workers=4):
        self.evaluator = CustomerServiceEvaluator()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def evaluate_batch(self, items):
        """
        Evaluate multiple items in parallel.

        Args:
            items: List of (actual, expected) tuples

        Returns:
            List of scores
        """
        futures = [
            self.executor.submit(self.evaluator.evaluate, actual, expected)
            for actual, expected in items
        ]

        return [f.result() for f in futures]

    def close(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)

# Usage
batch_eval = BatchEvaluator(max_workers=8)

items = [
    ("response 1", "expected 1"),
    ("response 2", "expected 2"),
    # ... more items
]

scores = batch_eval.evaluate_batch(items)
print(f"Average score: {sum(scores)/len(scores):.2f}")

batch_eval.close()
```

### 3. Lazy Loading

```python
class LazyEvaluatorLoader:
    """
    Load evaluators only when needed.
    """

    def __init__(self):
        self._evaluators = {}

    def get_evaluator(self, evaluator_type):
        """Get or create evaluator"""
        if evaluator_type not in self._evaluators:
            if evaluator_type == 'customer_service':
                from custom_evaluators import CustomerServiceEvaluator
                self._evaluators[evaluator_type] = CustomerServiceEvaluator()
            elif evaluator_type == 'consensus':
                from custom_evaluators.consensus import MultiModelConsensusEvaluator
                self._evaluators[evaluator_type] = MultiModelConsensusEvaluator()
            # ... more evaluator types

        return self._evaluators[evaluator_type]

# Usage
loader = LazyEvaluatorLoader()

# Only loads when first needed
cs_eval = loader.get_evaluator('customer_service')
```

---

## Monitoring & Observability

### 1. Metrics Collection

```python
from datetime import datetime
import json

class EvaluationMetrics:
    """
    Collect and store evaluation metrics.
    """

    def __init__(self, storage_path='./metrics.jsonl'):
        self.storage_path = storage_path

    def record(self, evaluator_name, score, duration, metadata=None):
        """Record evaluation metric"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'evaluator': evaluator_name,
            'score': score,
            'duration_ms': duration * 1000,
            'metadata': metadata or {}
        }

        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(metric) + '\n')

    def get_stats(self, evaluator_name=None, since=None):
        """Get aggregated statistics"""
        metrics = []

        with open(self.storage_path, 'r') as f:
            for line in f:
                metric = json.loads(line)

                # Filter by evaluator
                if evaluator_name and metric['evaluator'] != evaluator_name:
                    continue

                # Filter by time
                if since and datetime.fromisoformat(metric['timestamp']) < since:
                    continue

                metrics.append(metric)

        if not metrics:
            return {}

        scores = [m['score'] for m in metrics]
        durations = [m['duration_ms'] for m in metrics]

        return {
            'count': len(metrics),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'avg_duration_ms': sum(durations) / len(durations)
        }

# Usage
import time

metrics = EvaluationMetrics()

# Record evaluation
start = time.time()
score = evaluator.evaluate("test", "expected")
duration = time.time() - start

metrics.record('customer_service', score, duration, {'version': '1.0'})

# Get statistics
stats = metrics.get_stats(evaluator_name='customer_service')
print(f"Average score: {stats['avg_score']:.2f}")
print(f"Average duration: {stats['avg_duration_ms']:.2f}ms")
```

### 2. Logging Integration

```python
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def log_evaluation(func):
    """Decorator to log evaluations"""
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(self, actual, expected, context=None):
        logger.info(f"Starting evaluation: {self.name}")

        try:
            score = func(self, actual, expected, context)
            logger.info(f"Evaluation complete: {self.name}, score={score:.3f}")
            return score
        except Exception as e:
            logger.error(f"Evaluation failed: {self.name}, error={str(e)}")
            raise

    return wrapper

# Apply to evaluator
class LoggedCustomerServiceEvaluator(CustomerServiceEvaluator):
    @log_evaluation
    def evaluate(self, actual, expected, context=None):
        return super().evaluate(actual, expected, context)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors

**Problem:**
```
ImportError: No module named 'custom_evaluators'
```

**Solution:**
```python
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_evaluators import CustomerServiceEvaluator
```

#### Issue 2: Dependency Missing

**Problem:**
```
ModuleNotFoundError: No module named 'jsonschema'
```

**Solution:**
```bash
pip install jsonschema

# Or install all optional dependencies
pip install jsonschema pyyaml sqlparse lxml redis openai anthropic
```

#### Issue 3: Redis Connection Failed

**Problem:**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution:**
```python
from custom_evaluators.hitl import ReviewQueue

# Fallback to file backend if Redis unavailable
try:
    queue = ReviewQueue(backend="redis")
except Exception:
    print("Redis unavailable, using file backend")
    queue = ReviewQueue(backend="file")
```

#### Issue 4: API Rate Limits

**Problem:**
```
openai.error.RateLimitError: Rate limit exceeded
```

**Solution:**
```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, backoff_factor=2):
    """Decorator for exponential backoff retry"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    print(f"Retry {attempt+1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
        return wrapper
    return decorator

# Apply to provider
class RateLimitedOpenAIProvider(OpenAIProvider):
    @retry_with_backoff(max_retries=5)
    def evaluate(self, prompt, actual, expected):
        return super().evaluate(prompt, actual, expected)
```

---

## Best Practices

### 1. Configuration Management

Use environment-specific configs:

```python
import os
import yaml

def load_config(env='development'):
    """Load environment-specific configuration"""
    config_file = f'config.{env}.yaml'

    if not os.path.exists(config_file):
        config_file = 'config.yaml'  # Fallback to default

    with open(config_file) as f:
        return yaml.safe_load(f)

# Usage
env = os.getenv('APP_ENV', 'development')
config = load_config(env)
```

### 2. Error Handling

Always handle evaluator errors gracefully:

```python
def safe_evaluate(evaluator, actual, expected, default_score=0.5):
    """Evaluate with error handling"""
    try:
        return evaluator.evaluate(actual, expected)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        return default_score
```

### 3. Testing

Test evaluators independently:

```python
import unittest

class TestCustomerServiceEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = CustomerServiceEvaluator()

    def test_good_response(self):
        good = "Hello! I'll help you..."
        score = self.evaluator.evaluate(good, "")
        self.assertGreater(score, 0.7)

    def test_bad_response(self):
        bad = "Can't help."
        score = self.evaluator.evaluate(bad, "")
        self.assertLess(score, 0.5)
```

---

## Support

For additional help:
- Documentation: `README.md`
- Examples: Run each evaluator file directly
- Tests: `python test_evaluators.py`
- Issues: Contact support or file an issue

---

**Last Updated:** January 2024
**Version:** 1.0.0
