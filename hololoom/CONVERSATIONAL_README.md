# Conversational AutoLoom: Signal vs Noise Memory

**Automatic episodic memory that learns from conversation, filtering signal from noise.**

The ConversationalAutoLoom automatically spins important conversation turns into memory while discarding trivial exchanges. After each input/output sequence, it scores importance and decides what to remember.

## Quick Start

```python
from HoloLoom.conversational import conversational_loom
from HoloLoom.Documentation.types import Query

# Create conversational system
loom = await conversational_loom("Initial knowledge base...")

# Chat - important turns are automatically remembered!
response1 = await loom.chat("What is Thompson Sampling?")
# ^ Important Q&A → REMEMBERED (spun into memory)

response2 = await loom.chat("thanks")
# ^ Trivial acknowledgment → FORGOTTEN (filtered as noise)

response3 = await loom.chat("How does the policy engine work?")
# ^ Important question → REMEMBERED
# ^ Can now reference the Thompson Sampling conversation!

# Check what was remembered
stats = loom.get_stats()
print(f"Remembered: {stats['remembered_turns']}/{stats['total_turns']}")
```

## How It Works

### 1. Importance Scoring (0.0 - 1.0)

Each conversation turn is scored based on:

**SIGNAL Indicators (increase score):**
- Questions (`?`)
- Information-dense keywords (what, how, why, explain, etc.)
- Domain-specific terms (thompson, policy, embedding, etc.)
- Substantive content (longer messages)
- References to entities and facts
- High-confidence tool executions

**NOISE Indicators (decrease score):**
- Greetings and pleasantries (hi, thanks, bye)
- Acknowledgments only (ok, sure, got it)
- Very short exchanges
- Error messages
- Low-information content

### 2. Threshold Filtering

Default threshold: **0.4** (balanced)

- Score >= 0.4 → **REMEMBERED** (spun into MemoryShard)
- Score < 0.4 → **FORGOTTEN** (filtered as noise)

### 3. Automatic Memory Building

Important turns are automatically:
1. Converted to text representation
2. Spun by TextSpinner
3. Added to knowledge base
4. Available for future context retrieval

## Example Scores

```
Turn: "hi"
Score: 0.10 → FORGOTTEN (noise)

Turn: "What is HoloLoom?"
Score: 0.72 → REMEMBERED (signal)

Turn: "ok"
Score: 0.00 → FORGOTTEN (noise)

Turn: "How does Thompson Sampling work in the policy engine?"
Score: 0.95 → REMEMBERED (strong signal)

Turn: "thanks"
Score: 0.00 → FORGOTTEN (noise)
```

## API Reference

### ConversationalAutoLoom

#### Creation

```python
from HoloLoom.conversational import ConversationalAutoLoom
from HoloLoom.config import Config

# From text
loom = await ConversationalAutoLoom.from_text(
    text="Initial knowledge...",
    config=Config.fast(),
    importance_threshold=0.4  # Default threshold
)

# From file
loom = await ConversationalAutoLoom.from_file(
    file_path="knowledge_base.md",
    importance_threshold=0.5  # More selective
)

# Quick helper
from HoloLoom.conversational import conversational_loom
loom = await conversational_loom("Knowledge...")
```

#### Methods

**chat(user_input, auto_remember=True)**
- Main chat interface
- Returns response dict with `_meta` field containing importance score
- Automatically remembers if important (when auto_remember=True)

```python
response = await loom.chat("What is Thompson Sampling?")

print(response['_meta'])
# {
#   'turn_id': 0,
#   'importance': 0.85,
#   'remembered': True,
#   'timestamp': '2025-10-23T...'
# }
```

**get_stats()**
- Returns conversation statistics

```python
stats = loom.get_stats()
# {
#   'total_turns': 10,
#   'remembered_turns': 6,
#   'forgotten_turns': 4,
#   'avg_importance': 0.58,
#   'current_memory_shards': 15,
#   'history_size': 10,
#   'remember_rate': 0.6
# }
```

**get_history(limit=None, min_importance=0.0)**
- Get conversation history
- Optionally filter by minimum importance

```python
# Get last 5 turns
recent = loom.get_history(limit=5)

# Get only important turns
important = loom.get_history(min_importance=0.6)
```

**print_history(limit=10)**
- Pretty-print conversation history

```python
loom.print_history(limit=5)
# Turn 0 [✓] (Importance: 0.85)
#   User: What is Thompson Sampling?...
#   System: Thompson Sampling is a Bayesian approach...
```

**clear_history(keep_memory=True)**
- Clear conversation history
- Optionally keep the spun memory shards

## Customization

### Adjust Threshold

```python
# More permissive (remembers more)
loom = await conversational_loom(
    "Knowledge...",
    importance_threshold=0.2  # Remember almost everything
)

# More selective (remembers less)
loom = await conversational_loom(
    "Knowledge...",
    importance_threshold=0.7  # Only important exchanges
)

# Very selective
loom = await conversational_loom(
    "Knowledge...",
    importance_threshold=0.9  # Only critical information
)
```

### Custom Scoring Function

```python
def custom_scorer(user_input: str, system_output: str, metadata: dict) -> float:
    """Custom importance scoring."""
    score = 0.5

    # Boost urgency
    if 'urgent' in user_input.lower():
        score += 0.4

    # Boost client-specific terms
    client_terms = ['acme', 'project_x', 'deployment']
    if any(term in user_input.lower() for term in client_terms):
        score += 0.3

    # Penalize verbose outputs (likely errors)
    if len(system_output) > 1000:
        score -= 0.2

    return max(0.0, min(1.0, score))

# Use custom scorer
loom = ConversationalAutoLoom(
    orchestrator=base_orch,
    custom_scorer=custom_scorer
)
```

### Domain-Specific Keywords

Modify the `ImportanceScorer` to include your domain terms:

```python
# In ImportanceScorer.score_turn()
domain_terms = [
    'policy', 'thompson', 'embedding',  # Default HoloLoom terms
    'acme', 'deployment', 'client_name',  # Your terms
    'project_x', 'production', 'incident'  # More custom terms
]
```

## Threshold Guide

| Threshold | Description | Remember Rate | Use Case |
|-----------|-------------|---------------|----------|
| 0.2 | Very permissive | ~80% | Testing, want to capture everything |
| 0.4 | **Balanced** (default) | ~60% | General conversation, good signal/noise ratio |
| 0.6 | Selective | ~40% | Focus on important exchanges only |
| 0.8 | Very selective | ~20% | Critical information only, minimal memory usage |

## Memory Evolution Example

```
Initial State:
  - 10 shards from documentation

Conversation:
  Turn 1: "What is HoloLoom?" (Score: 0.72)
    → REMEMBERED
    → Memory: 11 shards

  Turn 2: "ok" (Score: 0.15)
    → FORGOTTEN
    → Memory: 11 shards (unchanged)

  Turn 3: "How does Thompson Sampling work?" (Score: 0.85)
    → REMEMBERED
    → Memory: 12 shards

  Turn 4: "thanks" (Score: 0.18)
    → FORGOTTEN
    → Memory: 12 shards (unchanged)

  Turn 5: "Tell me about the policy engine" (Score: 0.78)
    → REMEMBERED
    → Memory: 13 shards

Final State:
  - 13 shards (10 original + 3 from conversation)
  - 5 turns, 3 remembered (60% signal)
  - 2 noise exchanges filtered
```

## Benefits

### 1. Self-Building Knowledge Base
- Learns from conversations automatically
- No manual curation needed
- Past exchanges inform future responses

### 2. Efficient Memory Usage
- Only stores important information
- Filters noise (greetings, acknowledgments)
- Keeps memory lean and focused

### 3. Context-Aware Responses
- Can reference previous conversation
- Builds on prior exchanges
- Maintains conversation continuity

### 4. Customizable Filtering
- Adjust threshold for your use case
- Add domain-specific keywords
- Custom scoring functions

## Integration

Works seamlessly with:
- **TextSpinner** - Spins conversation turns
- **AutoSpinOrchestrator** - Underlying orchestration
- **HoloLoom Policy** - All execution modes (bare/fast/fused)

## Statistics and Monitoring

Track conversation quality:

```python
stats = loom.get_stats()

print(f"Signal-to-Noise Ratio: {stats['remember_rate']:.1%}")
print(f"Total Exchanges: {stats['total_turns']}")
print(f"Knowledge Captured: {stats['remembered_turns']}")
print(f"Noise Filtered: {stats['forgotten_turns']}")
print(f"Memory Size: {stats['current_memory_shards']} shards")
print(f"Avg Importance: {stats['avg_importance']:.2f}")
```

## Examples

See:
- `example_conversational.py` - Full demonstration of signal vs noise filtering
- `HoloLoom/conversational.py` - Implementation details

## Architecture

```
User Input
    ↓
ConversationalAutoLoom.chat()
    ↓
Process Query (via AutoSpinOrchestrator)
    ↓
Score Importance (ImportanceScorer)
    ↓
Threshold Check (>= 0.4?)
    ↓
If Important → Spin to MemoryShard → Add to Knowledge Base
If Noise → Discard
    ↓
Return Response (with importance metadata)
```

## Philosophy

**Signal vs Noise Baby!** 🎯

Not all conversation is worth remembering:
- "What is Thompson Sampling?" → **SIGNAL** (remember!)
- "thanks" → **NOISE** (forget!)
- "How does the policy engine work?" → **SIGNAL** (remember!)
- "ok" → **NOISE** (forget!)

The system automatically filters noise and captures signal, creating a high-quality episodic memory that learns from meaningful exchanges while staying efficient and focused.

This is the difference between a chatbot that forgets everything and an AI that builds knowledge from conversation.