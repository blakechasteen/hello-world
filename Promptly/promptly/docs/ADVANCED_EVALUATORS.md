# Advanced Evaluators Documentation

Comprehensive guide to Promptly's advanced evaluation plugin system.

## Table of Contents

1. [Overview](#overview)
2. [LLM-Based Evaluators](#llm-based-evaluators)
3. [NLP Metrics Evaluators](#nlp-metrics-evaluators)
4. [Custom Metrics Evaluators](#custom-metrics-evaluators)
5. [Composite Evaluators](#composite-evaluators)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Best Practices](#best-practices)
8. [Use Cases](#use-cases)

---

## Overview

Promptly's advanced evaluator plugins provide sophisticated methods for assessing prompt quality, output correctness, and response characteristics. The system includes:

- **21+ evaluator types** covering diverse quality dimensions
- **Graceful degradation** when dependencies are unavailable
- **Composable architecture** for building custom evaluation pipelines
- **Caching mechanisms** for expensive operations (LLM calls)
- **Flexible configuration** with sensible defaults

### Quick Start

```python
from Promptly.promptly.plugins.evaluators import *

# Simple evaluation
evaluator = BLEUEvaluator(max_n=4)
score = evaluator.evaluate(
    actual="The cat is on the mat",
    expected="There is a cat on the mat"
)
print(f"BLEU-4 score: {score:.3f}")

# Composite evaluation
pipeline = WeightedAverageEvaluator([
    (BLEUEvaluator(), 0.4),
    (ROUGEEvaluator(), 0.3),
    (ReadabilityEvaluator(), 0.3)
])
score = pipeline.evaluate(actual, expected)
```

---

## LLM-Based Evaluators

Use Large Language Models as judges for nuanced quality assessment.

### OpenAIEvaluator

Uses GPT-4 or GPT-3.5 to evaluate outputs with sophisticated reasoning.

**Features:**
- Configurable evaluation criteria
- Custom rubrics
- Temperature control for consistency
- Automatic response caching (100 evaluations by default)

**Configuration:**

```python
evaluator = OpenAIEvaluator(
    api_key="sk-...",              # Or set OPENAI_API_KEY env var
    model="gpt-4-turbo-preview",   # gpt-4, gpt-3.5-turbo
    criteria=["accuracy", "clarity", "completeness"],
    temperature=0.1,               # Lower = more consistent
    cache_size=100                 # Cache up to 100 evaluations
)
```

**Custom Rubric:**

```python
rubric = """Evaluate the output on a scale of 0.0 to 1.0 based on:
- Technical accuracy (40%)
- Clarity of explanation (30%)
- Completeness (30%)

Respond with JSON: {"score": <float>, "reasoning": "<text>"}"""

evaluator = OpenAIEvaluator(rubric=rubric)
```

**Performance:**
- **Latency:** 1-3 seconds per evaluation (API call)
- **Cost:** ~$0.01-0.03 per evaluation (GPT-4)
- **Cache hit rate:** 70-90% in typical workflows
- **Consistency:** High (temperature=0.1 recommended)

### AnthropicEvaluator

Uses Claude for evaluation with similar capabilities to OpenAI.

```python
evaluator = AnthropicEvaluator(
    api_key="sk-ant-...",          # Or set ANTHROPIC_API_KEY
    model="claude-3-5-sonnet-20241022",
    criteria=["factual accuracy", "coherence"],
    temperature=0.1
)
```

**Performance:**
- **Latency:** 1-2 seconds per evaluation
- **Cost:** ~$0.015 per evaluation (Claude 3.5 Sonnet)
- **Quality:** Excellent for reasoning and nuance

### OllamaEvaluator

Uses locally-hosted LLMs for private, offline evaluation.

```python
evaluator = OllamaEvaluator(
    model="llama3",                # mistral, mixtral, phi, etc.
    host="http://localhost:11434",
    criteria=["accuracy", "conciseness"],
    temperature=0.1
)
```

**Advantages:**
- ✅ No API costs
- ✅ Private/offline operation
- ✅ Low latency (local inference)

**Performance:**
- **Latency:** 0.5-2 seconds (depends on hardware)
- **Quality:** Good (competitive with GPT-3.5)
- **Requirements:** 8GB+ RAM for Llama3-8B

### LLMPairwiseEvaluator

Compares two outputs to determine which is better.

```python
base_evaluator = OpenAIEvaluator(model="gpt-4-turbo-preview")
pairwise = LLMPairwiseEvaluator(base_evaluator)

result = pairwise.compare(
    output_a="Paris is the capital of France.",
    output_b="The capital of France is Paris, known for the Eiffel Tower.",
    reference="What is the capital of France?"
)

print(f"Winner: {result['winner']}")      # 'A', 'B', or 'tie'
print(f"Confidence: {result['confidence']:.2f}")
print(f"Reasoning: {result['reasoning']}")
```

**Use Cases:**
- A/B testing prompt variants
- Preference learning datasets
- Ranking outputs

---

## NLP Metrics Evaluators

Standard NLP evaluation metrics with zero-dependency fallbacks.

### BLEUEvaluator

Bilingual Evaluation Understudy - measures n-gram overlap.

**Best for:** Machine translation, text generation

```python
evaluator = BLEUEvaluator(
    max_n=4,      # BLEU-4 (1-4 gram)
    smooth=True   # Smoothing for zero counts
)

score = evaluator.evaluate(
    actual="the cat is on the mat",
    expected="there is a cat on the mat"
)
```

**Multiple References:**

```python
score = evaluator.evaluate(
    actual="the cat sits on the mat",
    expected="reference1",
    context={
        'references': [
            "the cat is on the mat",
            "there is a cat on the mat",
            "a cat sits on a mat"
        ]
    }
)
```

**Interpretation:**
- **0.0-0.3:** Poor quality
- **0.3-0.5:** Moderate quality
- **0.5-0.7:** Good quality
- **0.7-1.0:** Excellent quality

**Performance:** ~0.1ms per evaluation

### ROUGEEvaluator

Recall-Oriented Understudy for Gisting Evaluation.

**Best for:** Summarization evaluation

```python
# ROUGE-1 (unigram overlap)
rouge1 = ROUGEEvaluator(variant='rouge-1', beta=1.0)

# ROUGE-2 (bigram overlap)
rouge2 = ROUGEEvaluator(variant='rouge-2', beta=1.0)

# ROUGE-L (longest common subsequence)
rougel = ROUGEEvaluator(variant='rouge-l', beta=1.0)
```

**Beta Parameter:**
- `beta=1.0`: F1 score (balanced precision/recall)
- `beta>1.0`: Favors recall
- `beta<1.0`: Favors precision

**Performance:** ~0.2ms per evaluation

### CosineSimilarityEvaluator

Semantic similarity via embedding vectors.

```python
# TF-IDF backend (no dependencies)
evaluator = CosineSimilarityEvaluator(backend="tfidf")

# Sentence transformers (requires sentence-transformers)
evaluator = CosineSimilarityEvaluator(
    backend="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)
```

**Backends:**
- **tfidf:** Fast, no dependencies, keyword-based
- **sentence-transformers:** Semantic understanding, requires library
- **openai:** High quality, requires API key

**Performance:**
- TF-IDF: ~0.5ms
- Sentence-transformers: ~50ms (CPU), ~10ms (GPU)

### PerplexityEvaluator

Measures language model quality via perplexity.

```python
evaluator = PerplexityEvaluator(model_name="gpt2")

score = evaluator.evaluate(
    actual="The quick brown fox jumps over the lazy dog.",
    expected=""  # Not used (intrinsic metric)
)
```

**Models:** Any HuggingFace causal LM (gpt2, gpt-neo, llama, etc.)

**Interpretation:**
- Lower perplexity = better fit to language model
- Score normalized to 0-1 range (higher = better)

**Performance:** ~100ms per evaluation (GPU), ~500ms (CPU)

### NamedEntityOverlapEvaluator

Measures entity extraction quality via F1 score.

```python
evaluator = NamedEntityOverlapEvaluator(
    entity_types=['PERSON', 'ORG', 'LOC'],  # Filter types
    case_sensitive=False
)

metrics = evaluator.get_metrics(
    actual="Apple Inc. is in Cupertino. Tim Cook is CEO.",
    expected="Apple is based in Cupertino. CEO is Tim Cook."
)

print(f"F1 Score: {metrics['score']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"Missing entities: {metrics['missing_entities']}")
```

**Backends:**
- **spaCy:** High quality (requires model)
- **Pattern-based:** Fallback (capitalized words)

**Performance:** ~50ms per evaluation (spaCy)

---

## Custom Metrics Evaluators

Specialized metrics for specific quality dimensions.

### LengthEvaluator

Controls output length with flexible constraints.

```python
# Target length (Gaussian scoring)
evaluator = LengthEvaluator(
    target_length=100,
    unit="chars"  # 'chars', 'words', 'sentences'
)

# Min/max constraints (hard bounds)
evaluator = LengthEvaluator(
    min_length=50,
    max_length=200,
    unit="words"
)
```

**Scoring:**
- **Target mode:** Gaussian decay from target (±20% tolerance)
- **Min/max mode:** 1.0 if within bounds, penalty if outside

**Performance:** <0.1ms

### ReadabilityEvaluator

Measures text readability using standard metrics.

```python
# Flesch Reading Ease (0-100, higher = easier)
evaluator = ReadabilityEvaluator(
    metric="flesch_reading_ease",
    target_score=70.0  # 7th-8th grade level
)

# Flesch-Kincaid Grade Level (US grade)
evaluator = ReadabilityEvaluator(
    metric="flesch_kincaid_grade",
    target_score=8.0  # 8th grade
)

# Automated Readability Index
evaluator = ReadabilityEvaluator(metric="ari")
```

**Interpretation:**

**Flesch Reading Ease:**
- 90-100: 5th grade (very easy)
- 60-70: 8th-9th grade (standard)
- 30-50: College (difficult)

**Grade Level Metrics:**
- Lower = easier to read
- Target 6-10 for general audience

**Performance:** ~0.5ms

### SentimentEvaluator

Analyzes sentiment polarity and subjectivity.

```python
# Simple lexicon-based (no dependencies)
evaluator = SentimentEvaluator(
    backend="simple",
    target_polarity="positive"  # 'positive', 'negative', 'neutral'
)

# TextBlob (requires textblob)
evaluator = SentimentEvaluator(backend="textblob")

# Transformers (requires transformers)
evaluator = SentimentEvaluator(backend="transformers")
```

**Metrics:**
- **Polarity:** -1 (negative) to +1 (positive)
- **Subjectivity:** 0 (objective) to 1 (subjective)

**Performance:**
- Simple: <0.1ms
- TextBlob: ~10ms
- Transformers: ~100ms (CPU)

### ToxicityEvaluator

Detects toxic, offensive, or harmful content.

```python
# Simple keyword-based
evaluator = ToxicityEvaluator(
    backend="simple",
    threshold=0.5  # Toxicity threshold
)

# Detoxify (requires detoxify library)
evaluator = ToxicityEvaluator(backend="detoxify")

metrics = evaluator.get_metrics(text, "")
print(f"Toxicity: {metrics['toxicity_analysis']['toxicity']:.3f}")
print(f"Is toxic: {metrics['is_toxic']}")
```

**Score Interpretation:**
- 1.0 = non-toxic (safe)
- 0.0 = toxic (harmful)

**Performance:**
- Simple: <0.1ms
- Detoxify: ~200ms (CPU)

### JSONSchemaEvaluator

Validates JSON output against a schema.

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

evaluator = JSONSchemaEvaluator(schema=schema)

score = evaluator.evaluate(
    actual='{"name": "John", "age": 30, "email": "john@example.com"}',
    expected=""
)
```

**Returns:**
- 1.0: Valid JSON matching schema
- 0.0: Invalid JSON or schema mismatch

**Performance:** <1ms

---

## Composite Evaluators

Combine multiple evaluators using various strategies.

### WeightedAverageEvaluator

Combines scores with configurable weights.

```python
evaluator = WeightedAverageEvaluator([
    (BLEUEvaluator(), 0.4),
    (ROUGEEvaluator(), 0.3),
    (ReadabilityEvaluator(), 0.3)
], normalize_weights=True)  # Automatically normalize to sum=1.0

score = evaluator.evaluate(actual, expected)
metrics = evaluator.get_metrics(actual, expected)

# Access individual scores
for name, data in metrics['individual_scores'].items():
    print(f"{name}: {data['score']:.3f} (weight: {data['weight']})")
```

**Use Cases:**
- Balancing multiple quality dimensions
- Custom scoring functions
- Multi-objective optimization

### VotingEnsembleEvaluator

Binary pass/fail decisions via voting.

```python
evaluator = VotingEnsembleEvaluator(
    evaluators=[bleu, rouge, cosine],
    threshold=0.5,           # Score threshold per evaluator
    voting_strategy="majority",  # 'majority', 'unanimous', 'any'
    min_votes=None          # Custom vote threshold
)

score = evaluator.evaluate(actual, expected)  # 1.0 = pass, 0.0 = fail
```

**Strategies:**
- **majority:** More than half must pass
- **unanimous:** All must pass
- **any:** At least one must pass

**Use Cases:**
- Quality gates
- Multi-stage validation
- Conservative evaluation

### CascadingEvaluator

Sequential evaluation with short-circuiting.

```python
evaluator = CascadingEvaluator([
    (LengthEvaluator(min_length=5, max_length=100, unit="words"), 0.5),
    (ToxicityEvaluator(), 0.8),
    (BLEUEvaluator(), 0.3)
], stop_on_failure=True, aggregate="min")

score = evaluator.evaluate(actual, expected)
metrics = evaluator.get_metrics(actual, expected)

if metrics['failed_stage'] is not None:
    print(f"Failed at stage: {metrics['failed_stage']}")
```

**Aggregation Methods:**
- **min:** Minimum score across stages
- **max:** Maximum score
- **average:** Mean score
- **product:** Multiplicative combination

**Use Cases:**
- Fast rejection of bad outputs
- Staged validation pipelines
- Resource optimization

### MinMaxEvaluator

Returns min or max score across evaluators.

```python
# Conservative evaluation (weakest link)
conservative = MinMaxEvaluator([bleu, rouge, cosine], mode="min")

# Optimistic evaluation (best performance)
optimistic = MinMaxEvaluator([bleu, rouge, cosine], mode="max")
```

### ConditionalEvaluator

Adaptive evaluator selection based on conditions.

```python
def select_evaluator(actual, expected, context):
    word_count = len(actual.split())
    if word_count < 10:
        return 'short'
    elif word_count < 50:
        return 'medium'
    else:
        return 'long'

evaluator = ConditionalEvaluator(
    evaluator_map={
        'short': ExactMatchEvaluator(),
        'medium': BLEUEvaluator(max_n=2),
        'long': BLEUEvaluator(max_n=4)
    },
    condition_func=select_evaluator,
    default_evaluator='medium'
)
```

### ThresholdedEvaluator

Applies thresholds to base evaluators.

```python
# Binary: 1.0 if >= threshold, else 0.0
binary = ThresholdedEvaluator(
    base_evaluator=CosineSimilarityEvaluator(),
    threshold=0.5,
    mode="binary"
)

# Scale above: map [threshold, 1.0] -> [0.0, 1.0]
scaled = ThresholdedEvaluator(
    base_evaluator=CosineSimilarityEvaluator(),
    threshold=0.5,
    mode="scale_above"
)
```

---

## Performance Benchmarks

Benchmarks measured on:
- CPU: Intel i7-10700K (8 cores)
- GPU: NVIDIA RTX 3080
- Text length: ~100 words

### Latency (ms per evaluation)

| Evaluator | CPU | GPU | Notes |
|-----------|-----|-----|-------|
| BLEUEvaluator | 0.1 | - | Pure Python |
| ROUGEEvaluator | 0.2 | - | Pure Python |
| CosineSimilarityEvaluator (TF-IDF) | 0.5 | - | No dependencies |
| CosineSimilarityEvaluator (transformers) | 50 | 10 | all-MiniLM-L6-v2 |
| PerplexityEvaluator | 500 | 100 | GPT-2 |
| NamedEntityOverlapEvaluator | 50 | - | spaCy en_core_web_sm |
| LengthEvaluator | 0.05 | - | Trivial computation |
| ReadabilityEvaluator | 0.5 | - | Pure Python |
| SentimentEvaluator (simple) | 0.1 | - | Lexicon-based |
| SentimentEvaluator (transformers) | 100 | 20 | DistilBERT |
| ToxicityEvaluator (simple) | 0.1 | - | Keyword-based |
| ToxicityEvaluator (detoxify) | 200 | 40 | Full model |
| JSONSchemaEvaluator | 0.5 | - | jsonschema library |
| OpenAIEvaluator | 1500 | - | API call (cached: 0.01) |
| AnthropicEvaluator | 1200 | - | API call (cached: 0.01) |
| OllamaEvaluator | 800 | 200 | Llama3-8B local |

### Throughput (evaluations/second)

| Category | Throughput | Notes |
|----------|------------|-------|
| Pure Python metrics | 5000-10000 | BLEU, ROUGE, Length |
| Simple ML | 100-1000 | TF-IDF, readability |
| Local transformers (CPU) | 10-20 | Sentence-transformers |
| Local transformers (GPU) | 50-100 | Batching helps |
| LLM judges (API) | 0.5-1 | Rate limits apply |
| LLM judges (local) | 1-5 | Llama3 on RTX 3080 |

### Cost Analysis (per 1000 evaluations)

| Evaluator | Cost | Notes |
|-----------|------|-------|
| All pure Python | $0 | Free |
| OpenAI GPT-4 | $15-30 | ~$0.015-0.03 per eval |
| OpenAI GPT-3.5 | $0.50-1.00 | ~$0.0005-0.001 per eval |
| Anthropic Claude 3.5 | $15-20 | ~$0.015-0.02 per eval |
| Ollama (local) | $0 | One-time hardware cost |
| HuggingFace models | $0 | Free (compute cost only) |

### Cache Effectiveness

LLM evaluators include caching:

```python
evaluator = OpenAIEvaluator(cache_size=100)

# First evaluation: API call (~1.5s)
score1 = evaluator.evaluate(text_a, reference)

# Second evaluation (same inputs): cached (~0.01ms)
score2 = evaluator.evaluate(text_a, reference)

# Check cache statistics
metrics = evaluator.get_metrics(text_a, reference)
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

**Typical cache hit rates:**
- Development/testing: 80-95%
- Production (unique inputs): 5-20%
- A/B testing (repeated comparisons): 60-80%

---

## Best Practices

### 1. Choose the Right Evaluator

| Use Case | Recommended Evaluators |
|----------|------------------------|
| Machine translation | BLEU, ROUGE, CosineSimilarity |
| Summarization | ROUGE-L, Length, Readability |
| Question answering | ExactMatch, F1 (NER), Semantic |
| Creative writing | LLM judges, Sentiment, Readability |
| Code generation | ExactMatch, JSONSchema |
| Content moderation | Toxicity, Sentiment cascading |
| API responses | JSONSchema, Length |

### 2. Composite Strategies

**Fast pre-filtering:**
```python
pipeline = CascadingEvaluator([
    (LengthEvaluator(min_length=10, max_length=500), 0.5),  # Fast
    (ToxicityEvaluator(), 0.8),                              # Fast
    (OpenAIEvaluator(), 0.7)                                 # Slow but accurate
], stop_on_failure=True)
```

**Balanced evaluation:**
```python
pipeline = WeightedAverageEvaluator([
    (BLEUEvaluator(), 0.3),        # Precision
    (ROUGEEvaluator(), 0.3),       # Recall
    (ReadabilityEvaluator(), 0.2), # User experience
    (LengthEvaluator(), 0.2)       # Constraints
])
```

**Conservative quality gate:**
```python
gate = VotingEnsembleEvaluator(
    evaluators=[bleu, rouge, semantic],
    threshold=0.6,
    voting_strategy="unanimous"  # All must pass
)
```

### 3. Optimize for Cost and Latency

**Minimize LLM calls:**
```python
# Use cheap evaluators first
fast_check = CascadingEvaluator([
    (LengthEvaluator(min_length=20), 0.5),
    (ToxicityEvaluator(), 0.8),
    (BLEUEvaluator(), 0.4)
], stop_on_failure=True)

# Only use LLM for outputs that pass
if fast_check.evaluate(actual, expected) > 0.5:
    llm_score = OpenAIEvaluator().evaluate(actual, expected)
```

**Use local LLMs for development:**
```python
# Development: fast local evaluation
dev_evaluator = OllamaEvaluator(model="llama3")

# Production: high-quality API
prod_evaluator = OpenAIEvaluator(model="gpt-4-turbo-preview")
```

### 4. Handle Missing Dependencies

All evaluators degrade gracefully:

```python
# Will use TF-IDF if sentence-transformers unavailable
semantic = CosineSimilarityEvaluator(backend="sentence-transformers")

# Will use simple lexicon if textblob unavailable
sentiment = SentimentEvaluator(backend="textblob")

# Always provide fallback in composite evaluators
pipeline = WeightedAverageEvaluator([
    (CosineSimilarityEvaluator(backend="sentence-transformers"), 0.6),
    (BLEUEvaluator(), 0.4)  # Fallback if transformers fail
])
```

### 5. Leverage Caching

```python
# Configure cache size based on workload
evaluator = OpenAIEvaluator(
    cache_size=500  # Increase for repeated evaluations
)

# Monitor cache effectiveness
metrics = evaluator.get_metrics(actual, expected)
if metrics['cache_hit_rate'] < 0.3:
    print("Warning: Low cache hit rate, consider increasing cache_size")
```

---

## Use Cases

### Use Case 1: Prompt Engineering Workflow

Evaluate prompt variants during development:

```python
from Promptly.promptly.plugins.evaluators import *

# Define evaluation pipeline
pipeline = WeightedAverageEvaluator([
    (BLEUEvaluator(max_n=4), 0.3),
    (ROUGEEvaluator(variant='rouge-l'), 0.3),
    (ReadabilityEvaluator(metric='flesch_reading_ease'), 0.2),
    (LengthEvaluator(target_length=50, unit='words'), 0.2)
])

# Test variants
variants = [
    "Explain quantum computing in simple terms.",
    "Describe quantum computing for beginners.",
    "What is quantum computing? Explain simply."
]

reference = "Quantum computing uses quantum bits (qubits) that can exist in superposition, enabling parallel computation."

results = []
for variant in variants:
    # Assume we have a function that generates output from the prompt
    output = generate_output(variant)
    score = pipeline.evaluate(output, reference)
    results.append((variant, score))

# Select best variant
best_variant, best_score = max(results, key=lambda x: x[1])
print(f"Best prompt: {best_variant}")
print(f"Score: {best_score:.3f}")
```

### Use Case 2: Content Moderation Pipeline

Multi-stage validation for user-generated content:

```python
# Stage 1: Fast pre-filtering
fast_filter = CascadingEvaluator([
    (LengthEvaluator(min_length=5, max_length=1000, unit='words'), 0.5),
    (ToxicityEvaluator(backend='simple', threshold=0.5), 0.7)
], stop_on_failure=True, aggregate='min')

# Stage 2: Comprehensive validation (only if Stage 1 passes)
comprehensive = WeightedAverageEvaluator([
    (ToxicityEvaluator(backend='detoxify'), 0.5),
    (SentimentEvaluator(backend='transformers'), 0.3),
    (ReadabilityEvaluator(), 0.2)
])

def moderate_content(user_input):
    # Fast rejection
    fast_score = fast_filter.evaluate(user_input, "")
    if fast_score < 0.5:
        return {'approved': False, 'reason': 'Failed fast filter', 'score': fast_score}

    # Comprehensive check
    comprehensive_score = comprehensive.evaluate(user_input, "")
    approved = comprehensive_score >= 0.6

    return {
        'approved': approved,
        'score': comprehensive_score,
        'reason': 'Passed' if approved else 'Failed comprehensive check'
    }

# Example
result = moderate_content("This is a helpful and constructive comment.")
print(f"Approved: {result['approved']}, Score: {result['score']:.3f}")
```

### Use Case 3: A/B Testing with Statistical Rigor

Compare prompt variants with multiple dimensions:

```python
# Define evaluation dimensions
evaluators = {
    'quality': BLEUEvaluator(max_n=4),
    'readability': ReadabilityEvaluator(metric='flesch_kincaid_grade'),
    'length': LengthEvaluator(target_length=100, unit='words'),
    'sentiment': SentimentEvaluator(target_polarity='positive')
}

# Test cases
test_cases = [
    ("How does photosynthesis work?", "Plants convert light to energy"),
    ("Explain machine learning", "ML learns patterns from data"),
    # ... more test cases
]

# Compare variants
variant_a_scores = {k: [] for k in evaluators}
variant_b_scores = {k: [] for k in evaluators}

for question, reference in test_cases:
    output_a = generate_with_variant_a(question)
    output_b = generate_with_variant_b(question)

    for name, evaluator in evaluators.items():
        variant_a_scores[name].append(evaluator.evaluate(output_a, reference))
        variant_b_scores[name].append(evaluator.evaluate(output_b, reference))

# Analyze results
import numpy as np

for dimension in evaluators:
    mean_a = np.mean(variant_a_scores[dimension])
    mean_b = np.mean(variant_b_scores[dimension])
    std_a = np.std(variant_a_scores[dimension])
    std_b = np.std(variant_b_scores[dimension])

    print(f"{dimension}:")
    print(f"  Variant A: {mean_a:.3f} ± {std_a:.3f}")
    print(f"  Variant B: {mean_b:.3f} ± {std_b:.3f}")
    print(f"  Winner: {'A' if mean_a > mean_b else 'B'}")
    print()
```

### Use Case 4: Custom Evaluation Criteria

Build domain-specific evaluators:

```python
# Custom medical text evaluator
class MedicalTextEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            name="medical_text",
            description="Evaluator for medical text quality"
        )
        self.readability = ReadabilityEvaluator(metric='flesch_kincaid_grade')
        self.ner = NamedEntityOverlapEvaluator(entity_types=['MEDICAL'])
        self.length = LengthEvaluator(min_length=50, max_length=300, unit='words')

    def evaluate(self, actual, expected, context=None):
        # Custom logic
        readability_score = self.readability.evaluate(actual, expected)
        ner_score = self.ner.evaluate(actual, expected)
        length_score = self.length.evaluate(actual, expected)

        # Medical texts should be readable (6-8 grade level)
        readability_weight = 0.4
        # Accurate medical entities are critical
        ner_weight = 0.4
        # Appropriate length
        length_weight = 0.2

        return (readability_score * readability_weight +
                ner_score * ner_weight +
                length_score * length_weight)

# Use custom evaluator
evaluator = MedicalTextEvaluator()
score = evaluator.evaluate(medical_text, reference)
```

### Use Case 5: Regression Testing

Ensure prompt changes don't degrade quality:

```python
# Baseline evaluation
baseline_prompt = "Summarize the following text: {text}"
new_prompt = "Provide a concise summary of: {text}"

test_suite = load_test_cases()  # List of (input, expected_output)

# Evaluate baseline
baseline_evaluator = WeightedAverageEvaluator([
    (BLEUEvaluator(), 0.5),
    (ROUGEEvaluator(), 0.5)
])

baseline_scores = []
for input_text, expected in test_suite:
    output = generate(baseline_prompt.format(text=input_text))
    score = baseline_evaluator.evaluate(output, expected)
    baseline_scores.append(score)

baseline_mean = np.mean(baseline_scores)
baseline_std = np.std(baseline_scores)

# Evaluate new prompt
new_scores = []
for input_text, expected in test_suite:
    output = generate(new_prompt.format(text=input_text))
    score = baseline_evaluator.evaluate(output, expected)
    new_scores.append(score)

new_mean = np.mean(new_scores)
new_std = np.std(new_scores)

# Statistical comparison
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(baseline_scores, new_scores)

print(f"Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
print(f"New: {new_mean:.3f} ± {new_std:.3f}")
print(f"Improvement: {(new_mean - baseline_mean):.3f} ({(new_mean - baseline_mean) / baseline_mean * 100:.1f}%)")
print(f"Statistical significance: p={p_value:.4f}")

if p_value < 0.05 and new_mean > baseline_mean:
    print("✓ Statistically significant improvement!")
elif p_value < 0.05 and new_mean < baseline_mean:
    print("✗ Statistically significant regression!")
else:
    print("~ No significant difference")
```

---

## Troubleshooting

### ImportError: Missing dependencies

All evaluators degrade gracefully, but for full functionality:

```bash
# LLM evaluators
pip install openai anthropic ollama

# NLP metrics
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm

# Transformers-based
pip install transformers torch

# Sentiment & toxicity
pip install textblob detoxify

# JSON validation
pip install jsonschema
```

### Slow evaluation performance

1. **Use faster backends:**
   ```python
   # Slow: sentence-transformers
   slow = CosineSimilarityEvaluator(backend="sentence-transformers")

   # Fast: TF-IDF
   fast = CosineSimilarityEvaluator(backend="tfidf")
   ```

2. **Enable caching:**
   ```python
   evaluator = OpenAIEvaluator(cache_size=1000)
   ```

3. **Use cascading for early rejection:**
   ```python
   pipeline = CascadingEvaluator([
       (fast_check, 0.5),  # Reject bad outputs early
       (expensive_check, 0.7)  # Only run if fast check passes
   ], stop_on_failure=True)
   ```

### Inconsistent LLM evaluations

1. **Lower temperature:**
   ```python
   evaluator = OpenAIEvaluator(temperature=0.0)  # Deterministic
   ```

2. **Use multiple samples and average:**
   ```python
   scores = [evaluator.evaluate(actual, expected) for _ in range(3)]
   final_score = np.mean(scores)
   ```

3. **Provide detailed rubrics:**
   ```python
   rubric = """Rate 0.0-1.0 based on:
   - Factual accuracy (must be 100% correct)
   - Completeness (covers all key points)
   - Clarity (easy to understand)

   Be strict and consistent. Return JSON: {"score": X.X, "reasoning": "..."}"""
   evaluator = OpenAIEvaluator(rubric=rubric)
   ```

---

## Appendix: Quick Reference

### Import All Evaluators

```python
from Promptly.promptly.plugins.evaluators import (
    # LLM-based
    OpenAIEvaluator, AnthropicEvaluator, OllamaEvaluator,
    LLMPairwiseEvaluator,

    # NLP metrics
    BLEUEvaluator, ROUGEEvaluator, CosineSimilarityEvaluator,
    PerplexityEvaluator, NamedEntityOverlapEvaluator,

    # Custom metrics
    LengthEvaluator, ReadabilityEvaluator, SentimentEvaluator,
    ToxicityEvaluator, JSONSchemaEvaluator,

    # Composite
    WeightedAverageEvaluator, VotingEnsembleEvaluator,
    CascadingEvaluator, MinMaxEvaluator, ConditionalEvaluator,
    ThresholdedEvaluator,

    # Original
    KeywordEvaluator, SemanticSimilarityEvaluator, ExactMatchEvaluator
)
```

### One-Liners

```python
# Quick BLEU score
score = BLEUEvaluator(max_n=4).evaluate(actual, expected)

# Fast semantic similarity
score = CosineSimilarityEvaluator(backend="tfidf").evaluate(text1, text2)

# Readability check
score = ReadabilityEvaluator(metric="flesch_reading_ease").evaluate(text, "")

# JSON validation
score = JSONSchemaEvaluator(schema=my_schema).evaluate(json_text, "")

# Composite pipeline
score = WeightedAverageEvaluator([
    (BLEUEvaluator(), 0.5),
    (ROUGEEvaluator(), 0.5)
]).evaluate(actual, expected)
```

---

**For more examples, see:** `Promptly/promptly/examples/advanced_evaluators_demo.py`

**For API reference, see:** Plugin source code in `Promptly/promptly/plugins/evaluators/`
