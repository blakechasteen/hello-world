# Mem0 Integration Quickstart

This guide will help you get started with the HoloLoom + mem0 hybrid memory system in under 10 minutes.

## Prerequisites

1. **Install mem0**:
   ```bash
   pip install mem0ai
   ```

2. **Verify HoloLoom is installed**:
   ```bash
   python -c "from HoloLoom.memory.cache import MemoryManager; print('✓ HoloLoom ready')"
   ```

3. **Optional: Get mem0 API key** (for managed platform):
   - Sign up at [https://app.mem0.ai/](https://app.mem0.ai/)
   - Get your API key from settings
   - Set environment variable: `export MEM0_API_KEY=your_key_here`

## Quick Start

### 1. Basic Setup

```python
from HoloLoom.memory.cache import create_memory_manager
from HoloLoom.memory.mem0_adapter import create_hybrid_memory, Mem0Config
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Query, MemoryShard, Features

# Initialize HoloLoom components
shards = [...]  # Your memory shards
emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
hololoom_memory = create_memory_manager(shards, emb)

# Configure mem0 integration
mem0_config = Mem0Config(
    enabled=True,              # Enable mem0
    extraction_enabled=True,   # Use intelligent extraction
    graph_sync_enabled=True,   # Sync to knowledge graph
)

# Create hybrid memory manager
hybrid_memory = create_hybrid_memory(
    hololoom_memory=hololoom_memory,
    mem0_config=mem0_config
)
```

### 2. Store Memories

```python
import asyncio

async def store_interaction():
    # Create a query
    query = Query(text="How should I winterize my beehives?")
    
    # Simulate results from your agent
    results = {
        'response': "Ensure 60-80 lbs honey stores per hive...",
        'tool': 'answer',
        'confidence': 0.92
    }
    
    features = Features(
        psi=[0.1] * 384,  # Your embeddings
        motifs=["SEASONAL", "PREPARATION"],
        metrics={},
        metadata={}
    )
    
    # Store in both HoloLoom and mem0
    await hybrid_memory.store(
        query=query,
        results=results,
        features=features,
        user_id="blake"  # User-specific tracking
    )
    
    print("✓ Memory stored with intelligent extraction")

asyncio.run(store_interaction())
```

### 3. Retrieve Memories

```python
async def retrieve_memory():
    query = Query(text="What are my winter preparation steps?")
    
    # Retrieve using fused approach (mem0 + HoloLoom)
    context = await hybrid_memory.retrieve(
        query=query,
        user_id="blake",
        k=5
    )
    
    # Use the retrieved context
    print(f"Retrieved {len(context.shards)} relevant memories")
    for shard in context.shards[:3]:
        print(f"- {shard.text[:80]}...")
    
    # Check fusion metadata
    if hasattr(context, 'metadata'):
        print(f"\nFusion info:")
        print(f"  Mem0 memories: {context.metadata.get('mem0_count')}")
        print(f"  HoloLoom memories: {context.metadata.get('hololoom_count')}")

asyncio.run(retrieve_memory())
```

### 4. Get User Profile

```python
async def show_user_profile():
    profile = await hybrid_memory.get_user_profile(user_id="blake")
    
    if profile['available']:
        print(f"User: {profile['user_id']}")
        print(f"Total memories: {profile['memory_count']}")
        print("\nRecent memories:")
        for mem in profile['memories'][:5]:
            print(f"  - {mem['memory']}")

asyncio.run(show_user_profile())
```

## Configuration Options

### Mem0Config Parameters

```python
mem0_config = Mem0Config(
    # Core settings
    enabled=True,                      # Enable/disable mem0
    api_key=None,                      # API key for managed platform
    
    # Feature flags
    extraction_enabled=True,           # Use LLM extraction
    graph_sync_enabled=True,           # Sync entities to KG
    user_tracking_enabled=True,        # Track user-specific memories
    
    # Fusion weights
    mem0_weight=0.3,                   # 30% mem0 in fused retrieval
    hololoom_weight=0.7,               # 70% HoloLoom
    
    # Memory limits
    max_memories_per_query=5,          # Max mem0 memories to retrieve
    memory_relevance_threshold=0.5,    # Min score for mem0 memories
)
```

### Common Configurations

#### Development (Local mem0)
```python
mem0_config = Mem0Config(
    enabled=True,
    api_key=None,  # Uses local/default config
)
```

#### Production (Managed Platform)
```python
import os

mem0_config = Mem0Config(
    enabled=True,
    api_key=os.environ.get('MEM0_API_KEY'),
    extraction_enabled=True,
    graph_sync_enabled=True,
)
```

#### HoloLoom-Only (No mem0)
```python
mem0_config = Mem0Config(
    enabled=False,  # Disables mem0, uses HoloLoom only
)
```

#### Mem0-Heavy (70% mem0, 30% HoloLoom)
```python
mem0_config = Mem0Config(
    enabled=True,
    mem0_weight=0.7,
    hololoom_weight=0.3,
)
```

## Integration with Orchestrator

Add hybrid memory to your HoloLoom orchestrator:

```python
from HoloLoom.Orchestrator import HoloLoomOrchestrator
from HoloLoom.config import Config
from HoloLoom.memory.mem0_adapter import create_hybrid_memory, Mem0Config

# Create config
config = Config.fused()

# Create shards and embedder
shards = [...]
emb = MatryoshkaEmbeddings(sizes=config.scales)

# Create HoloLoom memory
from HoloLoom.memory.cache import create_memory_manager
hololoom_memory = create_memory_manager(shards, emb)

# Create hybrid memory
mem0_config = Mem0Config(enabled=True)
hybrid_memory = create_hybrid_memory(hololoom_memory, mem0_config)

# Modify orchestrator to use hybrid memory
# (This requires modifying orchestrator to accept custom memory manager)
```

## Troubleshooting

### Mem0 Not Installed
```
RuntimeError: Mem0 integration enabled but mem0ai not installed
```
**Solution**: `pip install mem0ai`

### Import Errors
```
ImportError: cannot import name 'Memory' from 'mem0'
```
**Solution**: Ensure you have the latest version: `pip install --upgrade mem0ai`

### API Key Issues (Managed Platform)
```
Error: Invalid API key
```
**Solution**: 
1. Check your API key at https://app.mem0.ai/
2. Set environment variable: `export MEM0_API_KEY=your_key`
3. Or pass directly to config: `Mem0Config(api_key="your_key")`

### No User Memories
```
profile['memory_count'] == 0
```
**Solution**: 
- Ensure you're using the same `user_id` for store and retrieve
- Check that `extraction_enabled=True`
- Verify mem0 is actually storing memories (check logs)

## Examples

### Full Working Example

See `HoloLoom/examples/hybrid_memory_example.py` for a complete demo.

Run it:
```bash
python HoloLoom/examples/hybrid_memory_example.py
```

### Domain-Specific Example (Beekeeping)

```python
async def beekeeping_example():
    # Setup (abbreviated)
    hybrid_memory = create_hybrid_memory(...)
    
    # Store a hive inspection
    query = Query(text="How is Hive Jodi doing?")
    results = {
        'response': "Hive Jodi has 8 frames of brood and good honey stores.",
        'tool': 'answer'
    }
    await hybrid_memory.store(query, results, features, user_id="blake")
    
    # Later, retrieve user's hive status
    query = Query(text="Show me status of my hives")
    context = await hybrid_memory.retrieve(query, user_id="blake")
    
    # Mem0 will remember:
    # - Blake's hive names (Jodi, Aurora, Luna)
    # - Last inspection notes
    # - User's preferences (organic treatments, etc.)
```

## Performance Tips

1. **Use Fast Mode for High Throughput**:
   ```python
   context = await hybrid_memory.retrieve(query, user_id="blake", fast=True)
   ```

2. **Tune Fusion Weights**:
   - More mem0 weight = more personalization
   - More HoloLoom weight = more semantic similarity

3. **Limit Max Memories**:
   ```python
   mem0_config = Mem0Config(max_memories_per_query=3)  # Faster retrieval
   ```

4. **Set Relevance Threshold**:
   ```python
   mem0_config = Mem0Config(memory_relevance_threshold=0.7)  # Higher quality
   ```

## Next Steps

1. **Read the full analysis**: See `MEM0_INTEGRATION_ANALYSIS.md` for architectural details
2. **Customize extraction**: Configure mem0's LLM prompts for your domain
3. **Monitor performance**: Use mem0's dashboard (managed platform) or logs
4. **Optimize fusion weights**: Experiment with different `mem0_weight` / `hololoom_weight` ratios
5. **Implement memory decay**: Use mem0's filtering to prevent memory bloat

## Resources

- [Mem0 Documentation](https://docs.mem0.ai/)
- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [Mem0 Research Paper](https://mem0.ai/research)
- [HoloLoom Memory Architecture](../memory/cache.py)
- [Integration Analysis](./MEM0_INTEGRATION_ANALYSIS.md)

## Support

- **HoloLoom Issues**: Open issue in your repository
- **Mem0 Issues**: [GitHub Issues](https://github.com/mem0ai/mem0/issues)
- **Mem0 Community**: [Discord](https://mem0.dev/DiG)
