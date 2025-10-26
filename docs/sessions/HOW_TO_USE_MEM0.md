# How to Use Mem0 with HoloLoom

## Quick Summary

Mem0ai is **already integrated** into HoloLoom! You just need to configure it.

## What Mem0 Does

1. **Intelligent Extraction**: Automatically decides what's important to remember from conversations
2. **User-Specific Memory**: Tracks preferences/facts per user (e.g., `user_id="blake"`)
3. **Semantic Search**: Finds relevant memories even if worded differently
4. **Automatic Filtering**: Only keeps meaningful facts, filters out noise

## Option 1: Use with Ollama (Fully Local, No API Keys)

### Setup

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Code Example

```python
from mem0 import Memory

# Configure for Ollama
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0.1,
            "max_tokens": 1500,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest"
        }
    },
    "version": "v1.1"
}

# Initialize
m = Memory.from_config(config)

# Store a conversation
messages = [
    {"role": "user", "content": "I prefer organic treatments for bees"},
    {"role": "assistant", "content": "Got it! Noted your organic preference."}
]
result = m.add(messages, user_id="blake")
print(f"Extracted {len(result['results'])} memories")

# Search for memories
results = m.search("What treatments does Blake like?", user_id="blake")
for mem in results['results']:
    print(f"- {mem['memory']} (score: {mem['score']:.2f})")

# Get all memories for a user
all_mems = m.get_all(user_id="blake")
for mem in all_mems['results']:
    print(f"- {mem['memory']}")
```

## Option 2: Use with OpenAI (Requires API Key)

### Setup

```bash
export OPENAI_API_KEY=sk-...
```

### Code Example

```python
from mem0 import Memory

# Initialize (uses OpenAI by default)
m = Memory()

# Same API as above
messages = [{"role": "user", "content": "..."}, ...]
m.add(messages, user_id="blake")
results = m.search("query", user_id="blake")
```

## Option 3: HoloLoom Integration (Hybrid Memory)

### Setup

See: `HoloLoom/examples/hybrid_memory_example.py`

### Code Example

```python
from HoloLoom.memory.cache import create_memory_manager
from HoloLoom.memory.mem0_adapter import create_hybrid_memory, Mem0Config
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Query, MemoryShard, Features

# 1. Create your memory shards
shards = [
    MemoryShard(
        id="shard_001",
        text="Blake prefers organic treatments",
        episode="user_preferences",
        entities=["Blake", "organic"],
        motifs=["USER_PREFERENCE"]
    ),
]

# 2. Initialize HoloLoom memory
emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
hololoom_memory = create_memory_manager(shards, emb)

# 3. Configure mem0 (local Ollama mode)
mem0_config = Mem0Config(
    enabled=True,
    extraction_enabled=True,  # Use intelligent extraction
    graph_sync_enabled=True,  # Sync to knowledge graph
    mem0_weight=0.3,          # 30% mem0, 70% HoloLoom
    hololoom_weight=0.7
)

# 4. Create hybrid manager
hybrid = create_hybrid_memory(hololoom_memory, mem0_config)

# 5. Store interactions
query = Query(text="What treatments do I prefer?")
results = {'response': "You prefer organic treatments"}
features = Features(psi=[0.1]*384, motifs=[], metrics={}, metadata={})

await hybrid.store(query, results, features, user_id="blake")

# 6. Retrieve with fusion (mem0 + HoloLoom)
context = await hybrid.retrieve(
    Query(text="How should I treat mites?"),
    user_id="blake",
    k=5
)

print(f"Retrieved {len(context.shards)} memories")
for shard in context.shards:
    print(f"- {shard.text}")
```

## Running the Full Demo

```bash
# Make sure PyTorch is installed (now done!)
pip install torch numpy

# Run the hybrid demo
python HoloLoom/examples/hybrid_memory_example.py
```

## Configuration Options

### Mem0Config Parameters

```python
mem0_config = Mem0Config(
    enabled=True,                      # Turn mem0 on/off
    extraction_enabled=True,           # Use LLM to extract facts
    graph_sync_enabled=True,           # Sync entities to KG
    user_tracking_enabled=True,        # Per-user memories

    # Fusion weights (must sum to ~1.0)
    mem0_weight=0.3,                   # 30% mem0
    hololoom_weight=0.7,               # 70% HoloLoom

    # Limits
    max_memories_per_query=5,          # Max mem0 results
    memory_relevance_threshold=0.5,    # Min score threshold
)
```

### Common Configurations

**Development (HoloLoom only, no mem0):**
```python
mem0_config = Mem0Config(enabled=False)
```

**Balanced (default):**
```python
mem0_config = Mem0Config(enabled=True, mem0_weight=0.3, hololoom_weight=0.7)
```

**Mem0-heavy (personalization focus):**
```python
mem0_config = Mem0Config(enabled=True, mem0_weight=0.7, hololoom_weight=0.3)
```

**High-quality filtering:**
```python
mem0_config = Mem0Config(
    enabled=True,
    memory_relevance_threshold=0.7,  # Only keep very relevant memories
    max_memories_per_query=3          # Limit results
)
```

## Troubleshooting

### "OpenAI API key required"

**Problem:** Mem0 defaults to OpenAI embeddings

**Solution:** Either:
1. Use Ollama config (see Option 1 above)
2. Set `OPENAI_API_KEY` environment variable
3. Disable mem0: `Mem0Config(enabled=False)`

### "Cannot import HoloLoom modules"

**Problem:** Python path issue

**Solution:** Run from repository root:
```bash
cd c:\Users\blake\Documents\mythRL
python HoloLoom/examples/hybrid_memory_example.py
```

### "Ollama connection failed"

**Problem:** Ollama not running

**Solution:**
```bash
ollama serve  # In one terminal
python your_script.py  # In another terminal
```

## Key Concepts

| Component | What It Does |
|-----------|--------------|
| `Memory()` | Standalone mem0 client |
| `Memory.from_config()` | Mem0 with custom config (Ollama, etc.) |
| `m.add()` | Store conversation, extract facts |
| `m.search()` | Search memories by semantic similarity |
| `m.get_all()` | Get all memories for a user |
| `HybridMemoryManager` | Combines HoloLoom + mem0 |
| `hybrid.store()` | Store in both systems |
| `hybrid.retrieve()` | Fused retrieval (weighted combination) |

## Next Steps

1. **Run the demo**: `python HoloLoom/examples/hybrid_memory_example.py`
2. **Read the docs**: See `HoloLoom/Documentation/MEM0_QUICKSTART.md`
3. **Customize**: Tune fusion weights for your use case
4. **Integrate**: Add to your orchestrator workflow

## Resources

- **Mem0 Docs**: https://docs.mem0.ai/
- **HoloLoom Integration**: `HoloLoom/memory/mem0_adapter.py`
- **Example**: `HoloLoom/examples/hybrid_memory_example.py`
- **Quickstart**: `HoloLoom/Documentation/MEM0_QUICKSTART.md`