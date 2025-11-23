# 5-Minute Quickstart

Get HoloLoom running in 5 minutes with this minimal setup guide.

---

## Prerequisites

- Python 3.9+
- pip

## Installation

```bash
# Clone repository
git clone https://github.com/blakewoolbright/mythRL
cd mythRL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch numpy networkx
```

## Your First Query

### Option 1: Simple API (Recommended for beginners)

```python
from HoloLoom import HoloLoom
import asyncio

async def main():
    # Create HoloLoom instance
    async with HoloLoom() as loom:
        # Form a memory
        await loom.experience("Thompson Sampling balances exploration and exploitation")

        # Recall information
        memories = await loom.recall("What did I learn about sampling?")

        # Print results
        for memory in memories:
            print(f"Memory: {memory.content}")
            print(f"Confidence: {memory.confidence:.2f}")

asyncio.run(main())
```

**Output:**
```
Memory: Thompson Sampling balances exploration and exploitation
Confidence: 0.92
```

### Option 2: Department API (More control)

```python
from HoloLoom.departments import get_department
import asyncio

async def main():
    # Get RAG department
    rag_dept = get_department("rag")

    # Create query request
    request = {
        "task_type": "question_answering",
        "parameters": {
            "query": "What is Thompson Sampling?",
            "max_sources": 5
        }
    }

    # Process request
    response = await rag_dept.process(request)

    # Print results
    print(f"Answer: {response['result']['answer']}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Sources: {len(response['result']['sources'])}")

asyncio.run(main())
```

### Option 3: Full Weaving Cycle (Advanced)

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard
import asyncio

async def main():
    # Create configuration
    config = Config.fast()  # BARE (fastest) | FAST (balanced) | FUSED (highest quality)

    # Create memory shards
    shards = [
        MemoryShard(
            content="Thompson Sampling balances exploration and exploitation",
            source="quickstart",
            timestamp=1698595200.0
        )
    ]

    # Create orchestrator
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Process query
        spacetime = await orchestrator.weave(Query(text="What is Thompson Sampling?"))

        # Print results
        print(f"Response: {spacetime.response}")
        print(f"Confidence: {spacetime.confidence:.2f}")
        print(f"Tool used: {spacetime.metadata.get('tool_used')}")

asyncio.run(main())
```

---

## Configuration Modes

HoloLoom has 3 execution modes:

| Mode | Latency | Quality | Use Case |
|------|---------|---------|----------|
| **BARE** | ~50ms | Good | Simple queries, development |
| **FAST** | ~150ms | Better | Production (recommended) |
| **FUSED** | ~300ms | Best | Complex queries, research |

```python
from HoloLoom.config import Config

config_bare = Config.bare()   # Fastest
config_fast = Config.fast()   # Balanced (recommended)
config_fused = Config.fused() # Highest quality
```

---

## Production Setup (Optional)

For persistence and production features:

```bash
# Start Docker services (Neo4j + Qdrant)
docker-compose up -d

# Configure backend
python -c "
from HoloLoom.config import Config, MemoryBackend
config = Config.fast()
config.memory_backend = MemoryBackend.HYBRID  # Uses Neo4j + Qdrant
"
```

---

## What's Next?

- [Your First Query](first-query.md) - Detailed tutorial
- [Department Overview](../guides/departments/README.md) - Understand the architecture
- [Workflow Examples](../examples/workflows/cross-department.md) - Real-world patterns
- [Production Deployment](../guides/production/deployment.md) - Deploy to production

---

## Troubleshooting

### Import Error: "No module named 'HoloLoom'"

**Solution**: Set PYTHONPATH
```bash
export PYTHONPATH=.  # From repository root
python your_script.py
```

### Memory Error: "Backend unavailable"

**Solution**: HoloLoom auto-falls back to INMEMORY backend. For production, start Docker:
```bash
docker-compose up -d
```

### Slow Queries (>1 second)

**Solution**: Enable query cache
```python
config = Config.fast()
config.enable_query_cache = True
```

---

**Next**: [Installation Guide](installation.md) for complete setup instructions
