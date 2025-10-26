# Better Mem0 Integration Pattern

## TL;DR

Instead of the complex HoloLoom hybrid system, use this simpler pattern:

```python
from mem0_simple_integration import UserMemory

# Initialize (falls back gracefully if no API key)
memory = UserMemory(provider="openai")  # or "ollama"

# Store
memory.remember("blake", "I prefer organic bee treatments")

# Retrieve
context = memory.recall("blake", "What treatments does Blake use?")
print(context)  # Relevant memories
```

## Why This Is Better

### The Problem with Complex Integration

The full HoloLoom + mem0 hybrid system ([HoloLoom/memory/mem0_adapter.py](HoloLoom/memory/mem0_adapter.py)) is:
- **Heavy**: Requires PyTorch, NetworkX, embeddings, knowledge graphs
- **Complex**: 575 lines of integration code
- **Coupled**: Tightly integrated into HoloLoom's architecture
- **Fragile**: Import path issues, dependency conflicts

### The Better Approach

Use mem0 as a **lightweight user preference layer** that sits ALONGSIDE your code:

```python
# Your existing agent/workflow
class MyAgent:
    def __init__(self, user_memory: UserMemory):
        self.memory = user_memory  # Optional!

    def answer(self, user_id: str, question: str) -> str:
        # Enrich with user context
        user_prefs = self.memory.recall(user_id, question)

        # Your existing logic
        answer = self.generate_answer(question)

        # Add personalization
        if user_prefs:
            answer += f"\n\nPersonalized for you:\n{user_prefs}"

        return answer
```

## Implementation

See [mem0_simple_integration.py](mem0_simple_integration.py) for the full implementation.

### Key Features

1. **Graceful Fallback**: Works even if mem0 fails (uses in-memory dict)
2. **Simple API**: Just `remember()` and `recall()`
3. **No Dependencies**: Works standalone, no HoloLoom needed
4. **Provider Flexibility**: OpenAI (simple) or Ollama (local)
5. **User-Specific**: Automatic per-user tracking

### Example: Enrich Existing Workflow

```python
from mem0_simple_integration import UserMemory

memory = UserMemory(provider="openai")

def my_existing_function(user_id, query):
    # Get user preferences
    user_context = memory.recall(user_id, query)

    # Your existing logic here
    result = do_your_thing(query)

    # Optionally enrich with context
    if user_context:
        result = f"{result}\n\nBased on your preferences:\n{user_context}"

    # Remember this interaction
    memory.remember(user_id, f"Queried: {query}")

    return result
```

## Setup Options

### Option 1: OpenAI (Simple, Reliable)

```bash
export OPENAI_API_KEY=sk-...
```

```python
memory = UserMemory(provider="openai")
```

**Pros:**
- Just works
- No configuration
- Fast and reliable

**Cons:**
- Requires API key
- Cloud-based (not local)
- Costs money

### Option 2: Ollama (Local, Free)

```bash
# Start Ollama
ollama serve

# Pull models (one-time)
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

```python
memory = UserMemory(provider="ollama")
```

**Pros:**
- Fully local
- Free
- Private

**Cons:**
- Requires Ollama running
- May have compatibility issues
- Slower than OpenAI

### Option 3: Fallback Mode (No Setup)

If neither OpenAI nor Ollama is available, the system automatically falls back to an in-memory dict:

```python
# No API key, no Ollama? No problem!
memory = UserMemory(provider="openai")  # Will fail gracefully

# Still works, just uses in-memory storage
memory.remember("blake", "I like bees")
context = memory.recall("blake", "What does Blake like?")
# Returns: "- I like bees"
```

## Comparison: Simple vs Complex

| Feature | Simple Pattern | HoloLoom Hybrid |
|---------|---------------|-----------------|
| Lines of code | ~200 | ~575 |
| Dependencies | mem0 only | PyTorch, NetworkX, scipy, etc. |
| Import complexity | 1 file | 10+ modules |
| Setup time | < 1 minute | 10-30 minutes |
| Learning curve | 5 minutes | 1-2 hours |
| Fallback mode | ✓ | ✗ |
| Works standalone | ✓ | ✗ (needs HoloLoom) |
| User-specific | ✓ | ✓ |
| Automatic extraction | ✓ | ✓ |
| Multi-scale retrieval | ✗ | ✓ |
| Knowledge graph sync | ✗ | ✓ |

## When to Use Which

### Use Simple Pattern When:
- Adding user memory to an existing workflow
- Prototyping quickly
- Don't need knowledge graph features
- Want minimal dependencies
- Need graceful fallback

### Use HoloLoom Hybrid When:
- Need multi-scale embeddings
- Want knowledge graph integration
- Building a complex AI system
- Have time for deep integration
- Need spectral features

## Migration Path

Start simple, upgrade if needed:

1. **Start**: Use `mem0_simple_integration.py`
2. **Test**: Verify it works with your workflow
3. **Evaluate**: Do you need KG/multi-scale features?
4. **If yes**: Migrate to HoloLoom hybrid
5. **If no**: Keep it simple!

## Usage Examples

### Example 1: Chat Bot

```python
memory = UserMemory(provider="openai")

def chat(user_id, message):
    # Get user context
    context = memory.recall(user_id, message)

    # Generate response (your logic)
    response = generate_response(message, context)

    # Remember this exchange
    memory.remember(user_id, message)

    return response
```

### Example 2: Task Assistant

```python
memory = UserMemory(provider="openai")

def schedule_task(user_id, task_description):
    # Check user preferences
    prefs = memory.recall(user_id, "scheduling preferences")

    # Apply preferences
    time = infer_time(task_description, prefs)

    # Remember for next time
    memory.remember(user_id, f"Scheduled: {task_description} at {time}")

    return f"Task scheduled at {time}"
```

### Example 3: Recommendation Engine

```python
memory = UserMemory(provider="openai")

def recommend(user_id, query):
    # Get user history and preferences
    history = memory.get_all(user_id)

    # Your recommendation logic
    items = find_items(query)
    ranked = rank_by_preferences(items, history)

    # Remember what they searched for
    memory.remember(user_id, f"Searched for: {query}")

    return ranked[:5]
```

## Troubleshooting

### "No OpenAI API key"

Either:
1. Set environment variable: `export OPENAI_API_KEY=sk-...`
2. Pass to constructor: `UserMemory(provider="openai", api_key="sk-...")`
3. Use Ollama: `UserMemory(provider="ollama")`
4. Accept fallback mode (works, just no persistence)

### "Ollama connection failed"

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### "Dimension mismatch" with Ollama

This is a known issue with certain Ollama model combinations. Solutions:
1. Use OpenAI provider instead
2. Accept fallback mode
3. Wait for mem0 update with better Ollama support

## Files

- **[mem0_simple_integration.py](mem0_simple_integration.py)**: Main implementation
- **[HOW_TO_USE_MEM0.md](HOW_TO_USE_MEM0.md)**: Full mem0 guide
- **[HoloLoom/memory/mem0_adapter.py](HoloLoom/memory/mem0_adapter.py)**: Complex hybrid system (for reference)
- **[HoloLoom/Documentation/MEM0_QUICKSTART.md](HoloLoom/Documentation/MEM0_QUICKSTART.md)**: Hybrid system quickstart

## Bottom Line

**For most use cases, the simple pattern is better:**
- Faster to implement
- Easier to understand
- Fewer dependencies
- More reliable (fallback mode)
- Easier to debug

**Only use the complex hybrid if you specifically need:**
- Multi-scale retrieval
- Knowledge graph features
- Spectral embeddings
- Deep HoloLoom integration

## Next Steps

1. **Try it**: Run [mem0_simple_integration.py](mem0_simple_integration.py)
2. **Integrate**: Add `UserMemory` to your workflow
3. **Test**: Verify it works with fallback mode
4. **Configure**: Add OpenAI key or Ollama when ready
5. **Iterate**: Expand usage as needed