# HoloLoom Synthesis: Signal → Training Data → Intelligence

**Transform filtered conversations into high-quality training data for AI.**

The synthesis pipeline extracts learnable patterns from your filtered conversations (signal >= 0.4 importance) and converts them into training examples ready for fine-tuning or few-shot prompting.

## The Vision

> "I was thinking I could synthesize enough data from meaningful interactions to make a pretty damn smart AI"

**The Math:**
- YOUR filtered conversations: **60-80% signal**
- Random internet text: **1-5% signal**
- **Advantage: 12-60x better signal-to-noise ratio**

This means training on YOUR brain's reasoning patterns, not random internet noise.

## Quick Start

```python
from HoloLoom.synthesis import (
    MemoryEnricher, PatternExtractor, DataSynthesizer, SynthesisConfig
)

# Sample conversation (already filtered, importance >= 0.4)
raw_memory = {
    'id': 'conv_1',
    'text': 'User: What is Thompson Sampling?\n\nSystem: Thompson Sampling is...',
    'timestamp': '2025-10-23T10:15:00',
    'importance': 0.88,
    'metadata': {
        'user_input': 'What is Thompson Sampling?',
        'system_output': 'Thompson Sampling is...'
    }
}

# Step 1: Enrich (extract entities, relationships, reasoning type)
enricher = MemoryEnricher()
enriched = enricher.enrich(raw_memory)

# Step 2: Extract patterns (Q&A, reasoning chains, causal, decisions, etc.)
extractor = PatternExtractor(min_confidence=0.5)
patterns = extractor.extract_patterns([enriched])

# Step 3: Synthesize training examples
config = SynthesisConfig(
    include_reasoning=True,
    include_context=True,
    min_confidence=0.5
)
synthesizer = DataSynthesizer(config)
examples = synthesizer.synthesize(patterns)

# Step 4: Export to JSONL for fine-tuning
synthesizer.export_jsonl(examples, 'training_data.jsonl', format='alpaca')
```

## Architecture

```
Filtered Conversations (importance >= 0.4)
    ↓
STEP 1: Memory Enrichment
    - Extract entities, relationships, topics
    - Classify reasoning type (question, answer, decision, etc.)
    - Add structure to raw text
    ↓
EnrichedMemory objects
    ↓
STEP 2: Pattern Extraction
    - Mine Q&A pairs
    - Extract reasoning chains
    - Find causal relationships
    - Identify decisions, comparisons, procedures
    ↓
Pattern objects (with confidence scores)
    ↓
STEP 3: Data Synthesis
    - Convert patterns to TrainingExample objects
    - Add context (entities, topics)
    - Format as instruction-response pairs
    ↓
TrainingExample objects
    ↓
STEP 4: Export
    - Alpaca format (instruction, input, output)
    - ChatML format (messages array)
    - Raw format (all fields)
    ↓
JSONL files ready for fine-tuning
```

## Module Overview

### 1. EnrichedMemory (`enriched_memory.py`)

Adds structure to raw conversation memories.

**Classes:**
- `ReasoningType`: Enum (QUESTION, ANSWER, DECISION, FACT, etc.)
- `EnrichedMemory`: Structured memory with entities, relationships, reasoning type
- `MemoryEnricher`: Extracts structure from raw memories

**Example:**
```python
enricher = MemoryEnricher()
enriched = enricher.enrich(raw_memory)

print(enriched.reasoning_type)  # ReasoningType.ANSWER
print(enriched.entities)        # ['Thompson', 'Sampling', 'Bayesian', 'Beta']
print(enriched.topics)          # ['policy', 'bandit']
print(enriched.relationships)   # [('Thompson Sampling', 'uses', 'Beta distributions')]
```

### 2. PatternExtractor (`pattern_extractor.py`)

Mines learnable patterns from enriched memories.

**Pattern Types:**
- `QA_PAIR`: Question-answer pairs
- `REASONING_CHAIN`: Multi-step reasoning (premise → steps → conclusion)
- `CAUSAL`: Cause-effect relationships
- `DECISION`: Context → decision + reasoning
- `COMPARISON`: Item A vs Item B
- `PROCEDURE`: Task → steps
- `DEFINITION`: Term → definition

**Example:**
```python
extractor = PatternExtractor(min_confidence=0.5)
patterns = extractor.extract_patterns(enriched_memories)

for pattern in patterns:
    print(f"{pattern.pattern_type.value}: confidence={pattern.confidence:.2f}")
    print(f"  Content: {pattern.content}")
```

**Example Patterns:**

Q&A Pair:
```python
Pattern(
    pattern_type=PatternType.QA_PAIR,
    content={
        'question': 'What is Thompson Sampling?',
        'answer': 'Thompson Sampling is a Bayesian approach...',
        'entities': ['Thompson', 'Sampling', 'Bayesian'],
        'topics': ['policy', 'bandit']
    },
    confidence=0.88
)
```

Reasoning Chain:
```python
Pattern(
    pattern_type=PatternType.REASONING_CHAIN,
    content={
        'premise': 'The policy engine processes query features',
        'steps': [
            'Extracts motifs and embeddings',
            'Retrieves relevant context',
            'Applies transformer with cross-attention',
            'Outputs logits for each tool'
        ],
        'conclusion': 'Tool selection uses Thompson Sampling for exploration'
    },
    confidence=0.75
)
```

### 3. DataSynthesizer (`data_synthesizer.py`)

Converts patterns to training examples in multiple formats.

**Classes:**
- `TrainingExample`: Single training example with conversion methods
- `SynthesisConfig`: Configuration (reasoning depth, context inclusion, thresholds)
- `DataSynthesizer`: Converts patterns to examples and exports to JSONL

**Formats:**

**Alpaca Format:**
```json
{
  "instruction": "What is Thompson Sampling?",
  "input": "Entities: Thompson, Sampling, Bayesian\nTopics: policy\n",
  "output": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem..."
}
```

**ChatML Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant..."},
    {"role": "user", "content": "What is Thompson Sampling?\n\nEntities: Thompson, Sampling..."},
    {"role": "assistant", "content": "Thompson Sampling is a Bayesian approach..."}
  ]
}
```

**Raw Format:**
```json
{
  "instruction": "What is Thompson Sampling?",
  "input": "Entities: Thompson, Sampling, Bayesian\nTopics: policy\n",
  "output": "Thompson Sampling is a Bayesian approach...",
  "system": "You are a helpful AI assistant...",
  "metadata": {
    "pattern_type": "qa_pair",
    "confidence": 0.88,
    "source": ["conv_1"]
  }
}
```

## Configuration

### SynthesisConfig

```python
config = SynthesisConfig(
    include_reasoning=True,      # Include chain-of-thought steps
    include_context=True,         # Add entities/topics to input
    min_confidence=0.4,           # Minimum pattern confidence
    max_examples_per_pattern=1,  # Max examples per pattern
    system_prompt="You are a helpful AI assistant trained on high-quality conversations."
)
```

**Parameters:**
- `include_reasoning`: If True, reasoning chain examples include all steps. If False, only conclusion.
- `include_context`: If True, adds entities/topics as input context for better grounding.
- `min_confidence`: Patterns below this threshold are filtered out.
- `max_examples_per_pattern`: Limit examples per pattern (future: data augmentation).
- `system_prompt`: System prompt for ChatML format.

## End-to-End Example

See [`example_synthesis_pipeline.py`](../../../example_synthesis_pipeline.py) for a complete demonstration.

**Input:** 7 filtered conversations (importance >= 0.4)

**Output:**
- 8 training examples
- 87.5% high confidence (>= 0.7)
- Average length: 352 chars
- Pattern types: Q&A pairs, definitions
- Exported in 3 formats: Alpaca, ChatML, Raw

**Quality Metrics:**
```
Total Examples: 8
Average Confidence: 0.78
High Confidence (>= 0.7): 7/8 (87.5%)
Pattern Diversity: 2 types (qa_pair, definition)
```

## Integration with Conversational AutoLoom

The synthesis pipeline integrates seamlessly with the conversational memory system:

```python
from HoloLoom.conversational import conversational_loom

# Create conversational system
loom = await conversational_loom("Initial knowledge...")

# Chat - important turns auto-spun to memory
await loom.chat("What is Thompson Sampling?")  # importance: 0.88 → REMEMBERED
await loom.chat("thanks")                       # importance: 0.18 → FORGOTTEN
await loom.chat("How does the policy work?")   # importance: 0.75 → REMEMBERED

# Later: Mine patterns from accumulated signal
from HoloLoom.synthesis import MemoryEnricher, PatternExtractor, DataSynthesizer

# Get only important memories (filtered signal)
important_memories = loom.get_history(min_importance=0.4)

# Convert to raw memory dicts
raw_memories = [
    {
        'id': f"conv_{turn.turn_id}",
        'text': turn.to_text(),
        'timestamp': turn.timestamp,
        'importance': turn.importance_score,
        'metadata': {
            'user_input': turn.user_input,
            'system_output': turn.system_output
        }
    }
    for turn in important_memories
]

# Synthesis pipeline
enricher = MemoryEnricher()
enriched = [enricher.enrich(m) for m in raw_memories]

extractor = PatternExtractor(min_confidence=0.5)
patterns = extractor.extract_patterns(enriched)

synthesizer = DataSynthesizer()
examples = synthesizer.synthesize(patterns)

# Export training data
synthesizer.export_jsonl(examples, 'my_conversations.jsonl', format='alpaca')
```

## Statistics and Analysis

```python
# Get synthesis statistics
stats = synthesizer.export_statistics(examples)

print(f"Total Examples: {stats['total_examples']}")
print(f"Average Length: {stats['avg_length']:.0f} chars")
print(f"Average Confidence: {stats['avg_confidence']:.2f}")
print(f"High Confidence: {stats['high_confidence_count']}/{stats['total_examples']}")

print("\nBy Pattern Type:")
for ptype, count in stats['pattern_types'].items():
    print(f"  {ptype}: {count}")
```

## Use Cases

### 1. Fine-Tuning Local Models

Export in Alpaca format for fine-tuning with Axolotl, LLaMA Factory, etc:

```bash
# Export training data
python example_synthesis_pipeline.py

# Use with Axolotl
axolotl train config.yml --data synthesis_output/training_data_alpaca.jsonl
```

### 2. Few-Shot Prompting

Use high-confidence examples as few-shot demonstrations:

```python
# Get only high-confidence examples
high_conf = [ex for ex in examples if ex.metadata['confidence'] >= 0.8]

# Format as few-shot prompt
few_shot_prompt = "\n\n".join([
    f"Q: {ex.instruction}\nA: {ex.output}"
    for ex in high_conf[:3]
])

# Use in prompt
prompt = f"""{few_shot_prompt}

Q: {new_question}
A: """
```

### 3. Domain-Specific Datasets

Build specialized datasets from your conversations:

```python
# Filter by topic
policy_examples = [
    ex for ex in examples
    if 'policy' in ex.metadata.get('topics', [])
]

# Export domain-specific dataset
synthesizer.export_jsonl(policy_examples, 'policy_training.jsonl', format='chatml')
```

### 4. Curriculum Learning

Order examples by complexity for progressive training:

```python
# Sort by length (proxy for complexity)
by_length = sorted(examples, key=lambda ex: len(ex.output))

# Export in curriculum order
synthesizer.export_jsonl(by_length, 'curriculum.jsonl', format='alpaca')
```

## Quality Comparison

**YOUR Filtered Conversations:**
- Pre-filtered: importance >= 0.4 (signal only)
- Domain-specific technical exchanges
- YOUR reasoning patterns captured
- Estimated signal: **60-80%**

**Typical Web-Scraped Data:**
- Random internet text
- Noise: ads, spam, low-quality content
- Generic patterns, not personalized
- Estimated signal: **1-5%**

**Advantage:**
- **12-60x better signal-to-noise ratio**
- Training on YOUR brain's patterns
- Domain-specific knowledge preserved
- Self-supervised learning from interaction

## Future Enhancements

**Pattern Augmentation:**
- Paraphrase questions for data augmentation
- Generate variations of reasoning chains
- Synthetic negative examples for contrastive learning

**Advanced Extraction:**
- Multi-turn dialogue patterns
- Code snippets and usage examples
- Error-correction pairs

**Quality Metrics:**
- Coherence scoring
- Factuality checking
- Diversity metrics

**Integration:**
- Automatic periodic synthesis from conversational memory
- Real-time pattern extraction
- Incremental fine-tuning

## Philosophy

**Signal vs Noise Baby!** 🎯

Not all conversation is worth training on:
- "What is Thompson Sampling?" → **SIGNAL** (Q&A pattern, high confidence)
- "thanks" → **NOISE** (filtered out, never reaches synthesis)
- "How does the policy work?" → **SIGNAL** (reasoning chain, extractable)
- "ok" → **NOISE** (filtered out)

The synthesis pipeline operates on pre-filtered signal, extracting only the learnable patterns from meaningful exchanges. This creates training data that captures YOUR reasoning, YOUR domain knowledge, YOUR problem-solving approach.

This is the difference between:
- An AI trained on random internet text
- An AI trained on YOUR filtered, high-quality conversations

**The alchemy: Signal → Patterns → Training Data → Intelligence**
