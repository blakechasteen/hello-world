# Synthesis Quickstart Guide

**Transform your conversations into training data in 5 minutes.**

## Prerequisites

```bash
# No additional dependencies required!
# Uses only Python stdlib
```

## 1-Minute Demo

```bash
# Run the complete pipeline demonstration
python example_synthesis_pipeline.py
```

**What you'll see:**
- 7 filtered conversations → 8 training examples
- Pattern extraction (Q&A pairs, definitions)
- Export in 3 formats (Alpaca, ChatML, Raw)
- Quality metrics (87.5% high confidence)

**Output files:**
```
synthesis_output/
├── training_data_alpaca.jsonl
├── training_data_chatml.jsonl
└── training_data_raw.jsonl
```

## 5-Minute Integration

### Step 1: Have Filtered Conversations

Use the conversational system (automatic signal filtering):

```python
from HoloLoom.conversational import conversational_loom

# Create conversational system
loom = await conversational_loom("Your knowledge base...")

# Chat - important turns auto-remembered
await loom.chat("What is Thompson Sampling?")  # importance: 0.88 ✓
await loom.chat("thanks")                       # importance: 0.18 ✗
await loom.chat("How does the policy work?")   # importance: 0.75 ✓

# Check stats
stats = loom.get_stats()
print(f"Signal rate: {stats['remember_rate']:.1%}")
```

### Step 2: Mine Training Data

Extract patterns and synthesize:

```python
from HoloLoom.synthesis import MemoryEnricher, PatternExtractor, DataSynthesizer

# Get filtered signal (important conversations only)
important_turns = loom.get_history(min_importance=0.4)

# Convert to raw memory format
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
    for turn in important_turns
]

# Enrich memories (extract entities, relationships, reasoning type)
enricher = MemoryEnricher()
enriched_memories = [enricher.enrich(m) for m in raw_memories]

# Extract patterns (Q&A, reasoning chains, causal, etc.)
extractor = PatternExtractor(min_confidence=0.5)
patterns = extractor.extract_patterns(enriched_memories)

# Synthesize training examples
synthesizer = DataSynthesizer()
examples = synthesizer.synthesize(patterns)

# Export to JSONL
synthesizer.export_jsonl(examples, 'training.jsonl', format='alpaca')

# Check quality
stats = synthesizer.export_statistics(examples)
print(f"Generated {stats['total_examples']} examples")
print(f"Average confidence: {stats['avg_confidence']:.2f}")
```

### Step 3: Use Training Data

**For fine-tuning:**
```bash
# With Axolotl
axolotl train config.yml --data training.jsonl

# With LLaMA Factory
python train.py --data training.jsonl --model llama3
```

**For few-shot prompting:**
```python
# Load high-confidence examples
import json

with open('training.jsonl') as f:
    examples = [json.loads(line) for line in f]

high_conf = [ex for ex in examples if ex.get('metadata', {}).get('confidence', 0) >= 0.8]

# Use as few-shot demonstrations
few_shot = "\n\n".join([
    f"Q: {ex['instruction']}\nA: {ex['output']}"
    for ex in high_conf[:3]
])

prompt = f"""{few_shot}

Q: {new_question}
A: """
```

## Common Workflows

### Periodic Synthesis (Daily/Weekly)

```python
# Run this daily/weekly to mine accumulated conversations

from datetime import datetime, timedelta

# Get recent conversations (last 7 days)
week_ago = datetime.now() - timedelta(days=7)
recent_turns = [
    turn for turn in loom.get_history(min_importance=0.4)
    if datetime.fromisoformat(turn.timestamp) >= week_ago
]

# Mine and export
# [synthesis code from above]

print(f"Weekly synthesis: {len(examples)} new training examples")
```

### Incremental Synthesis

```python
# Synthesize after each conversation batch

batch_size = 10
conversation_count = 0

for user_input, system_output in conversation_stream:
    await loom.chat(user_input)
    conversation_count += 1

    # Synthesize every 10 conversations
    if conversation_count % batch_size == 0:
        # [synthesis code]
        print(f"Batch {conversation_count // batch_size}: {len(examples)} examples")
```

### Domain-Specific Datasets

```python
# Filter by topic
policy_examples = [
    ex for ex in examples
    if 'policy' in ex.metadata.get('topics', [])
]

# Filter by confidence
high_quality = [
    ex for ex in examples
    if ex.metadata.get('confidence', 0) >= 0.8
]

# Export specialized dataset
synthesizer.export_jsonl(policy_examples, 'policy_dataset.jsonl', format='chatml')
```

## Configuration

### Importance Threshold

Control what gets remembered:

```python
# More permissive (remember more)
loom = await conversational_loom(
    "Knowledge...",
    importance_threshold=0.2  # 80% remember rate
)

# More selective (remember less)
loom = await conversational_loom(
    "Knowledge...",
    importance_threshold=0.7  # 20% remember rate
)

# Balanced (default)
loom = await conversational_loom(
    "Knowledge...",
    importance_threshold=0.4  # 60% remember rate
)
```

### Pattern Confidence

Control synthesis quality:

```python
# High quality only
extractor = PatternExtractor(min_confidence=0.7)
patterns = extractor.extract_patterns(enriched)
# → Fewer patterns, higher confidence

# Include medium quality
extractor = PatternExtractor(min_confidence=0.5)
patterns = extractor.extract_patterns(enriched)
# → More patterns, balanced quality

# All patterns
extractor = PatternExtractor(min_confidence=0.0)
patterns = extractor.extract_patterns(enriched)
# → Most patterns, some low quality
```

### Synthesis Config

Customize training data format:

```python
from HoloLoom.synthesis import SynthesisConfig

config = SynthesisConfig(
    include_reasoning=True,      # Include chain-of-thought
    include_context=True,         # Add entities/topics
    min_confidence=0.5,           # Pattern threshold
    system_prompt="Custom system prompt..."
)

synthesizer = DataSynthesizer(config)
examples = synthesizer.synthesize(patterns)
```

## Output Formats

### Alpaca (Instruction Tuning)

```json
{
  "instruction": "What is Thompson Sampling?",
  "input": "Entities: Thompson, Sampling, Bayesian\nTopics: policy\n",
  "output": "Thompson Sampling is a Bayesian approach..."
}
```

**Use with:** Axolotl, Stanford Alpaca, LLaMA Factory

### ChatML (Chat Models)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant..."},
    {"role": "user", "content": "What is Thompson Sampling?\n\nEntities: Thompson, Sampling..."},
    {"role": "assistant", "content": "Thompson Sampling is a Bayesian approach..."}
  ]
}
```

**Use with:** ChatGPT fine-tuning, OpenAI API format, Mistral

### Raw (Full Metadata)

```json
{
  "instruction": "What is Thompson Sampling?",
  "input": "Entities: Thompson, Sampling, Bayesian\nTopics: policy\n",
  "output": "Thompson Sampling is a Bayesian approach...",
  "system": "You are a helpful AI assistant...",
  "metadata": {
    "pattern_type": "qa_pair",
    "confidence": 0.88,
    "source": ["conv_1"],
    "topics": ["policy", "bandit"],
    "entities": ["Thompson", "Sampling", "Bayesian"]
  }
}
```

**Use with:** Custom pipelines, analysis, filtering

## Quality Metrics

```python
stats = synthesizer.export_statistics(examples)

# Check quality
assert stats['avg_confidence'] >= 0.7, "Low confidence examples"
assert stats['total_examples'] >= 10, "Not enough data"
assert len(stats['pattern_types']) >= 2, "Low diversity"

print(f"✓ Quality check passed:")
print(f"  {stats['total_examples']} examples")
print(f"  {stats['avg_confidence']:.2f} avg confidence")
print(f"  {len(stats['pattern_types'])} pattern types")
```

## Troubleshooting

### No Patterns Extracted

**Problem:** `extractor.extract_patterns()` returns empty list

**Solutions:**
1. Lower `min_confidence` threshold
2. Check if memories have `user_input` and `system_output` in metadata
3. Verify enrichment is working (check entities, topics)

### Low Confidence Examples

**Problem:** Average confidence < 0.5

**Solutions:**
1. Increase importance threshold to 0.6+ (more selective filtering)
2. Filter by specific topics to focus on your best conversations
3. Use `min_confidence=0.7` in extractor to only keep high-quality patterns

### No Training Examples Generated

**Problem:** `synthesizer.synthesize()` returns empty list

**Solutions:**
1. Check if patterns list is empty
2. Verify `min_confidence` in SynthesisConfig isn't too high
3. Ensure patterns have required fields (question/answer for Q&A, etc.)

## Examples

All examples are in the repository root:

- **`example_synthesis_pipeline.py`** - Complete end-to-end demo
- **`example_auto_synthesis.py`** - Periodic and incremental synthesis
- **`example_conversational.py`** - Signal vs noise filtering

Run them to see the system in action!

## Next Steps

1. **Try the demos:**
   ```bash
   python example_synthesis_pipeline.py
   python example_auto_synthesis.py
   ```

2. **Integrate with your conversations:**
   - Use `ConversationalAutoLoom` for automatic filtering
   - Periodically mine training data
   - Export and fine-tune

3. **Customize for your domain:**
   - Add domain-specific keywords to importance scorer
   - Adjust thresholds for your use case
   - Create specialized datasets

4. **Read the full documentation:**
   - `HoloLoom/synthesis/README.md` - Synthesis pipeline guide
   - `SYNTHESIS_VISION.md` - Complete vision and architecture
   - `CONVERSATIONAL_README.md` - Conversational filtering guide

## The Bottom Line

**Three lines to synthesize training data:**

```python
enriched = [MemoryEnricher().enrich(m) for m in raw_memories]
patterns = PatternExtractor(min_confidence=0.5).extract_patterns(enriched)
examples = DataSynthesizer().synthesize(patterns)
```

**That's it.** Signal → Patterns → Training Data → Intelligence 🎯
