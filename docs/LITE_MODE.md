# HoloLoom Lite

**Simplified API for quick startup and minimal dependencies.**

**Status**: Production Ready (December 2025)
**Location**: `hololoom/lite/`
**Lines**: ~1,600 across core + 4 UI modes

## Overview

HoloLoom Lite provides a streamlined entry point with:

- **5 core methods** instead of 100+ (simpler mental model)
- **~70% faster startup** via lazy loading (~500ms vs ~2-3s)
- **~75% smaller footprint** (~40k lines vs 165k+)
- **4 required packages** (torch, sentence-transformers, numpy, networkx)
- **INMEMORY backend** (no Docker required)
- **SafetyGuardrails** enabled by default

## Quick Start

### Programmatic API

```python
from hololoom import HoloLoomLite

async with HoloLoomLite() as loom:
    # 1. Store memories
    await loom.experience("Thompson Sampling balances exploration")

    # 2. Retrieve memories
    memories = await loom.recall("What is Thompson Sampling?")

    # 3. Learn from feedback
    await loom.reflect(memories, {"helpful": True})

    # 4. Agentic reasoning
    answer = await loom.reason("Explain Thompson Sampling", mode="verify")
    print(f"Answer: {answer.text}")
    print(f"Confidence: {answer.confidence:.2%}")

    # 5. Safety checks
    safety = await loom.check_safety("execute_code", {"code": "print('hi')"})
    print(f"Safe: {safety.confidence > 0.7}")
```

### Command-Line UIs

```bash
# Simple REPL (no dependencies)
python -m hololoom.lite repl

# Rich terminal (requires: pip install rich)
python -m hololoom.lite terminal

# Web chat (requires: pip install fastapi uvicorn)
python -m hololoom.lite web

# Desktop app (requires: pip install gradio)
python -m hololoom.lite desktop
```

## Core API (5 Methods)

### 1. experience(content, context=None)

Store content in memory.

```python
mem = await loom.experience("Python uses indentation for blocks")
print(f"Stored: {mem.id}")
```

**Parameters**:
- `content` (str): Text content to store
- `context` (dict, optional): Metadata

**Returns**: `Memory` object with `id`, `text`, `timestamp`, `context`

### 2. recall(query, limit=5)

Retrieve memories related to query.

```python
memories = await loom.recall("What is Python?", limit=3)
for mem in memories:
    print(f"- {mem.text}")
```

**Parameters**:
- `query` (str): Search query
- `limit` (int): Maximum memories to return

**Returns**: List of `Memory` objects

### 3. reflect(memories, feedback=None)

Learn from feedback on recalled memories.

```python
memories = await loom.recall("Python")
await loom.reflect(memories, {"helpful": True, "relevance": 0.9})
```

**Parameters**:
- `memories` (list): Memories to reflect on
- `feedback` (dict, optional): Feedback signals

### 4. reason(query, mode="direct")

Agentic reasoning with multiple modes.

```python
result = await loom.reason("Compare Thompson vs UCB", mode="verify")
print(f"Answer: {result.text}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Verified: {result.metadata.get('verified')}")
```

**Parameters**:
- `query` (str): Question or request
- `mode` (str): Reasoning mode

**Modes**:

| Mode | Latency | Description |
|------|---------|-------------|
| `direct` | ~150ms | Single-pass answer |
| `verify` | ~600ms | Answer with verification |
| `research` | ~900ms | Multi-query exploration |
| `plan_execute` | ~750ms | Goal decomposition |

**Returns**: `LiteResult` with `text`, `confidence`, `sources`, `metadata`

### 5. check_safety(action, context=None)

Evaluate action safety.

```python
result = await loom.check_safety("delete_data", {"scope": "all"})
if result.confidence > 0.7:
    print("Safe to proceed")
else:
    print(f"Risky: {result.metadata.get('reason')}")
```

**Parameters**:
- `action` (str): Action to evaluate
- `context` (dict, optional): Additional context

**Returns**: `LiteResult` with confidence (1.0 = safe, 0.0 = unsafe)

## LiteResult Dataclass

All `reason()`, `query()`, and `check_safety()` return `LiteResult`:

```python
@dataclass
class LiteResult:
    text: str                    # Response text
    confidence: float = 0.0      # 0.0-1.0 confidence
    sources: List[str] = None    # Source memory IDs
    metadata: Dict[str, Any] = None  # Additional info
```

## Configuration

### Default (Recommended)

```python
from hololoom import HoloLoomLite

# Uses Config.lite() preset by default
loom = HoloLoomLite()
```

### Custom Configuration

```python
from hololoom import HoloLoomLite
from hololoom.config import Config

# Customize the lite config
config = Config.lite()
config.working_memory_size = 100

loom = HoloLoomLite(config=config, enable_safety=True)
```

### Config.lite() Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `scales` | [384] | Single embedding scale |
| `n_transformer_layers` | 1 | Minimal layers |
| `n_attention_heads` | 2 | Minimal heads |
| `enable_safety_guardrails` | True | Safety enabled |
| `fast_mode` | True | Optimized paths |
| `memory_backend` | INMEMORY | No Docker needed |
| `working_memory_size` | 50 | Small working set |
| `pipeline_timeout` | 3.0s | Quick timeouts |
| `retrieval_k` | 5 | Top-5 retrieval |

## UI Modes

### 1. Simple REPL

**No dependencies** - uses standard library only.

```bash
python -m hololoom.lite repl
```

**Commands**:
- `/help` - Show help
- `/history` - Show conversation history
- `/memories` - Show stored memories
- `/clear` - Clear history
- `/experience <text>` - Store explicit memory
- `/reason <mode> <query>` - Agentic reasoning
- `/quit` - Exit

### 2. Rich Terminal

**Requires**: `pip install rich`

```bash
python -m hololoom.lite terminal
```

**Features**:
- Colored output with syntax highlighting
- Progress spinners during processing
- Formatted tables for memories
- Conversation panels
- Same commands as REPL

### 3. Web Chat

**Requires**: `pip install fastapi uvicorn`

```bash
python -m hololoom.lite web
# Open http://localhost:8080
```

**Features**:
- Chat bubble interface
- Mode selector (Direct/Verify/Research)
- Dark theme
- RESTful API endpoints

**Endpoints**:
- `GET /` - Chat UI
- `POST /query` - Query endpoint
- `GET /memories` - List memories
- `GET /metrics` - System metrics
- `GET /health` - Health check

### 4. Desktop App

**Requires**: `pip install gradio`

```bash
python -m hololoom.lite desktop
```

**Features**:
- Chat interface with history
- Mode selector (Direct/Verify/Research/Plan-Execute)
- Metrics panel
- Memory browser
- Store memory tab
- Copy button on responses

**With sharing** (public URL):
```python
from hololoom.lite.ui import launch
launch("desktop", share=True)
```

## Lazy Loading

Optional features are lazy loaded on first use:

```python
async with HoloLoomLite() as loom:
    # Core memory is initialized immediately
    await loom.experience("Hello")  # Uses core memory

    # Agentic is lazy loaded on first reason() call
    await loom.reason("query")  # Loads agentic module

    # RAG is lazy loaded on first query() call
    await loom.query("question")  # Loads RAG module

    # Guardrails are lazy loaded on first check_safety()
    await loom.check_safety("action")  # Loads guardrails
```

**Lazy loaded modules**:
- `hololoom.agentic` - Multi-query reasoning
- `hololoom.rag` - RAG Q&A integration
- `hololoom.alignment` - Safety guardrails

## Error Handling

All operations gracefully degrade:

```python
async with HoloLoomLite() as loom:
    # If agentic fails to load, falls back to recall
    result = await loom.reason("query")
    if result.metadata.get("mode") == "fallback":
        print("Agentic unavailable, used fallback")

    # If RAG fails, falls back to reason()
    result = await loom.query("question")

    # If guardrails fail, returns "unavailable" status
    result = await loom.check_safety("action")
    if result.metadata.get("status") == "unavailable":
        print("Guardrails not loaded")
```

## Performance Comparison

| Metric | Full HoloLoom | HoloLoom Lite | Improvement |
|--------|---------------|---------------|-------------|
| **Import time** | ~2-3s | ~0.5s | **4-6x faster** |
| **RAM at import** | ~500MB | ~150MB | **70% less** |
| **First query** | ~500ms | ~200ms | **2.5x faster** |
| **Lines of code** | 165k+ | ~40k | **75% smaller** |
| **Dependencies** | 15+ | 4 | **73% fewer** |

## When to Use

### Use HoloLoom Lite when:

- Quick prototyping and demos
- Simple applications with basic memory needs
- Learning HoloLoom concepts
- Resource-constrained environments
- No Docker available (INMEMORY backend)
- Need fast startup times

### Use Full HoloLoom when:

- Production deployments with persistence
- Complex multi-agent workflows
- Advanced features (Dark Trace, Federation, etc.)
- Need HYBRID backend (Neo4j + Qdrant)
- Maximum quality over speed

## Migration

### From Full HoloLoom

```python
# Before (Full HoloLoom)
from hololoom import hololoom
from hololoom.weaving_orchestrator import WeavingOrchestrator

async with HoloLoom() as loom:
    await loom.experience("content")
    memories = await loom.recall("query")

# After (HoloLoom Lite)
from hololoom import HoloLoomLite

async with HoloLoomLite() as loom:
    await loom.experience("content")  # Same API!
    memories = await loom.recall("query")  # Same API!
```

### To Full HoloLoom

```python
# From Lite
from hololoom import HoloLoomLite
loom = HoloLoomLite()

# To Full (when you need more features)
from hololoom import hololoom
from hololoom.config import Config

loom = HoloLoom()  # Use full system
# or
config = Config.fused()  # Full features
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `lite/core.py` | ~590 | Main HoloLoomLite class |
| `lite/__init__.py` | ~90 | Package exports + CLI |
| `lite/__main__.py` | ~10 | CLI entry point |
| `lite/ui/__init__.py` | ~95 | UI launcher |
| `lite/ui/repl.py` | ~190 | Simple REPL |
| `lite/ui/terminal.py` | ~285 | Rich terminal |
| `lite/ui/web.py` | ~445 | FastAPI web chat |
| `lite/ui/desktop.py` | ~250 | Gradio desktop app |
| **Total** | **~1,955** | |

## Demo

```bash
# Run comprehensive demo
PYTHONPATH=. python demos/demo_lite.py
```

**Demo covers**:
1. Basic experience/recall/reflect
2. Agentic reasoning modes
3. Safety guardrails
4. Startup performance
5. RAG query integration

## Troubleshooting

### Import Error: torch/sentence-transformers

```bash
pip install torch sentence-transformers
```

### Rich Terminal Not Working

```bash
pip install rich
# Falls back to simple REPL if rich unavailable
```

### Web Chat Error

```bash
pip install fastapi uvicorn
```

### Desktop App Error

```bash
pip install gradio
```

### Memory Issues

```python
# Use smaller working memory
config = Config.lite()
config.working_memory_size = 20
loom = HoloLoomLite(config=config)
```

## See Also

- [VISUAL_QUICK_START.md](getting-started/VISUAL_QUICK_START.md) - Visual guide
- [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md) - Full architecture
- [demos/demo_lite.py](../demos/demo_lite.py) - Comprehensive demo
