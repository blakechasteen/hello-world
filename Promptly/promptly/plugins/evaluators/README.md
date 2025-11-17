# Advanced Evaluators Plugin System

Sophisticated evaluation methods for Promptly prompt engineering and testing.

## Overview

This directory contains 21+ advanced evaluators organized into four categories:

### 1. LLM-Based Evaluators (`llm.py`)
- **OpenAIEvaluator** - GPT-4/GPT-3.5 as judge
- **AnthropicEvaluator** - Claude as judge
- **OllamaEvaluator** - Local LLMs via Ollama
- **LLMPairwiseEvaluator** - A/B comparison evaluator
- **LLMJudgeEvaluator** - Base class with caching

### 2. NLP Metrics (`nlp_metrics.py`)
- **BLEUEvaluator** - Machine translation quality
- **ROUGEEvaluator** - Summarization quality (ROUGE-1, ROUGE-2, ROUGE-L)
- **CosineSimilarityEvaluator** - Semantic similarity via embeddings
- **PerplexityEvaluator** - Language model quality
- **NamedEntityOverlapEvaluator** - Entity extraction quality

### 3. Custom Metrics (`custom.py`)
- **LengthEvaluator** - Length-based scoring (chars/words/sentences)
- **ReadabilityEvaluator** - Flesch-Kincaid, ARI, Flesch Reading Ease
- **SentimentEvaluator** - Sentiment polarity and subjectivity
- **ToxicityEvaluator** - Toxic/offensive content detection
- **JSONSchemaEvaluator** - JSON validation against schema

### 4. Composite Evaluators (`composite.py`)
- **WeightedAverageEvaluator** - Combine scores with weights
- **VotingEnsembleEvaluator** - Majority/unanimous voting
- **CascadingEvaluator** - Sequential with short-circuit
- **MinMaxEvaluator** - Min/max aggregation
- **ConditionalEvaluator** - Adaptive evaluator selection
- **ThresholdedEvaluator** - Apply thresholds to base evaluators

## Quick Start

```python
from Promptly.promptly.plugins.evaluators import *

# Simple evaluation
evaluator = BLEUEvaluator(max_n=4)
score = evaluator.evaluate(
    actual="The cat is on the mat",
    expected="There is a cat on the mat"
)

# Composite pipeline
pipeline = WeightedAverageEvaluator([
    (BLEUEvaluator(), 0.4),
    (ROUGEEvaluator(), 0.3),
    (ReadabilityEvaluator(), 0.3)
])
score = pipeline.evaluate(actual, expected)
```

## Features

✅ **Graceful Degradation** - All evaluators work without optional dependencies
✅ **Caching** - LLM evaluators cache results to reduce API calls
✅ **Composable** - Build complex pipelines from simple evaluators
✅ **Configurable** - Extensive configuration options with sensible defaults
✅ **Fast** - Pure Python metrics run in <1ms
✅ **Comprehensive** - 21+ evaluators covering diverse quality dimensions

## Installation

### Minimal (Core Functionality)
No additional dependencies required! All evaluators work with Python standard library.

### Full Features

```bash
# LLM evaluators
pip install openai anthropic ollama

# NLP metrics
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm

# Transformers
pip install transformers torch

# Sentiment & toxicity
pip install textblob detoxify

# JSON validation
pip install jsonschema
```

## Examples

See comprehensive examples in:
- **Demo Script:** `examples/advanced_evaluators_demo.py`
- **Documentation:** `docs/ADVANCED_EVALUATORS.md`

### Example: Content Moderation Pipeline

```python
# Fast pre-filter
fast_filter = CascadingEvaluator([
    (LengthEvaluator(min_length=5, max_length=1000, unit='words'), 0.5),
    (ToxicityEvaluator(backend='simple'), 0.7)
], stop_on_failure=True)

# Comprehensive validation
comprehensive = WeightedAverageEvaluator([
    (ToxicityEvaluator(backend='detoxify'), 0.5),
    (SentimentEvaluator(), 0.3),
    (ReadabilityEvaluator(), 0.2)
])

# Two-stage moderation
if fast_filter.evaluate(text, "") > 0.5:
    score = comprehensive.evaluate(text, "")
    approved = score >= 0.6
```

### Example: A/B Testing

```python
# Multi-dimensional comparison
evaluator = WeightedAverageEvaluator([
    (BLEUEvaluator(max_n=4), 0.4),
    (ReadabilityEvaluator(metric='flesch_reading_ease'), 0.3),
    (LengthEvaluator(target_length=100, unit='words'), 0.3)
])

score_a = evaluator.evaluate(variant_a, reference)
score_b = evaluator.evaluate(variant_b, reference)

winner = 'A' if score_a > score_b else 'B'
```

## Performance Benchmarks

| Evaluator Type | Latency | Notes |
|----------------|---------|-------|
| Pure Python (BLEU, ROUGE, Length) | <1ms | No dependencies |
| TF-IDF Similarity | ~0.5ms | Keyword-based |
| Sentence Transformers | ~50ms (CPU), ~10ms (GPU) | Semantic |
| Transformers (Sentiment, Toxicity) | ~100ms (CPU), ~20ms (GPU) | Deep learning |
| LLM Judges (API) | ~1500ms | Cached: <0.01ms |
| LLM Judges (Local) | ~800ms (CPU), ~200ms (GPU) | Llama3-8B |

## API Reference

All evaluators implement the `EvaluatorPlugin` protocol:

```python
class EvaluatorPlugin(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def evaluate(self, actual: str, expected: str,
                 context: Optional[Dict[str, Any]] = None) -> float: ...

    def get_metrics(self, actual: str, expected: str,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
```

## Best Practices

1. **Choose the right evaluator for your use case:**
   - Translation/generation → BLEU, ROUGE
   - Summarization → ROUGE-L, Length, Readability
   - QA → ExactMatch, F1 (NER), Semantic
   - Creative writing → LLM judges, Sentiment
   - Code/API → JSONSchema, ExactMatch

2. **Use composite strategies for comprehensive evaluation:**
   - Fast pre-filtering with cascading
   - Balanced scoring with weighted average
   - Conservative gates with voting ensemble

3. **Optimize for cost and latency:**
   - Use cheap evaluators first (cascading)
   - Cache LLM evaluations
   - Use local LLMs for development

4. **Handle missing dependencies gracefully:**
   - All evaluators provide fallbacks
   - Check backend availability in production

## Documentation

- **Full Documentation:** `docs/ADVANCED_EVALUATORS.md`
- **Examples:** `examples/advanced_evaluators_demo.py`
- **Source Code:** Individual evaluator files in this directory

## Contributing

To add a new evaluator:

1. Inherit from `BaseEvaluator` (in `../base.py`)
2. Implement `evaluate()` and optionally `get_metrics()`
3. Add graceful degradation for optional dependencies
4. Include docstrings and type hints
5. Add to `__init__.py` and update documentation
6. Create examples and tests

## License

Part of the Promptly prompt engineering framework.
