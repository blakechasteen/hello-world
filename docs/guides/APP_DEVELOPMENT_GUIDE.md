# App Development Guide

How to build domain-specific applications on the HoloLoom framework.

## Architecture

```
apps/                          # Application layer
  bosspig/                     # BossPig AI
  elle/                        # AR guide system
  sous/                        # Kitchen control loop
  trough/                      # Production QA
  your_app/                    # Your app here

hololoom/core/                 # Framework layer (don't modify)
  memory/                      # Knowledge graph, vector store
  embedding/                   # Multi-scale embeddings
  policy/                      # Thompson Sampling, neural policy
  orchestrator/                # Weaving orchestrator
  protocols/                   # Type contracts
  ...                          # 13 modules total
```

Apps consume the framework through its public API. They never modify core modules.

## Creating an App

### 1. Basic Structure

```
apps/your_app/
  __init__.py
  main.py
  config.py
```

### 2. Use the Public API

```python
from hololoom import HoloLoom, Memory

class YourApp:
    def __init__(self):
        self.loom = HoloLoom()

    async def process(self, input_text: str):
        # Store knowledge
        await self.loom.experience(input_text)

        # Recall relevant memories
        memories = await self.loom.recall(input_text)

        # Reflect on outcomes
        await self.loom.reflect(memories, feedback={"quality": 0.9})

        return memories
```

### 3. Advanced: Direct Component Access

For apps that need fine-grained control:

```python
from hololoom.core.memory.graph import KG
from hololoom.core.embedding.spectral import MatryoshkaEmbedding
from hololoom.core.policy.unified import create_policy
from hololoom.core.orchestrator import WeavingOrchestrator
from hololoom.config import Config

# Build custom pipeline
kg = KG()
embedder = MatryoshkaEmbedding([96, 192, 384])
policy = create_policy(mem_dim=384, emb=embedder, scales=[96, 192, 384])
orchestrator = WeavingOrchestrator(config=Config.fast())
```

## Protocol-Based Extension

HoloLoom uses protocols (abstract interfaces) throughout. Swap implementations without touching orchestrator code:

```python
from hololoom.core.protocols import PolicyEngine, KGStore, Retriever

class CustomStore(KGStore):
    """Your custom storage backend."""
    async def store(self, memory: Memory) -> str: ...
    async def retrieve(self, query: str, limit: int) -> list[Memory]: ...
```

## Patterns

### Graceful Degradation

Always wrap optional imports:

```python
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    nlp = None
    logger.warning("spaCy not available, using fallback")
```

### Async Pipeline

Use async/await for non-blocking operations:

```python
async def process_batch(items: list[str]):
    tasks = [loom.experience(item) for item in items]
    return await asyncio.gather(*tasks)
```

### Configuration

Extend the existing config system:

```python
from hololoom.config import Config

class AppConfig(Config):
    app_specific_setting: str = "default"
    custom_threshold: float = 0.8
```

## Example Apps

See existing apps for patterns:
- `apps/sous/` — Kitchen control loop with GPS/MFS scoring
- `apps/bosspig/` — Domain-specific AI assistant
- `apps/trough/` — Production quality assurance
