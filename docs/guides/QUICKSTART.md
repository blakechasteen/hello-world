# Quickstart

Get HoloLoom running and weave your first query.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e "."
```

For optional features, see [SETUP_REQUIREMENTS.md](SETUP_REQUIREMENTS.md).

## Core API

```python
from hololoom import HoloLoom, Memory

loom = HoloLoom()

# Store a memory
memory = await loom.experience("Thompson Sampling balances exploration and exploitation")

# Recall memories
memories = await loom.recall("What is Thompson Sampling?")

# Reflect on outcomes
await loom.reflect(memories, feedback={"helpful": True})
```

## Run the Demo

```bash
python demos/complete_weaving_demo.py
```

Runs the full 7-stage weaving pipeline: LoomCommand, ChronoTrigger, ResonanceShed, SynthesisBridge, WarpSpace, ConvergenceEngine, Spacetime.

## Performance Modes

| Mode | Latency | Quality | Use Case |
|------|---------|---------|----------|
| BARE | ~50ms | Good | High-volume queries |
| FAST | ~150ms | Better | General use (default) |
| FUSED | ~300ms | Best | Complex reasoning |

```python
from hololoom.config import Config

config = Config.bare()   # or .fast() or .fused()
```

## Memory Backends

HoloLoom defaults to in-memory storage. For persistence:

```bash
# Start Neo4j + Qdrant via Docker
cd config && docker-compose up -d
```

See [DOCKER_MEMORY_SETUP.md](DOCKER_MEMORY_SETUP.md) for details.

## Tests

```bash
pytest hololoom/tests/unit/ -v          # Fast (<5s)
pytest hololoom/tests/integration/ -v   # Medium (<30s)
pytest hololoom/tests/e2e/ -v           # Slow (<2min)
```

## Next Steps

- [SETUP_REQUIREMENTS.md](SETUP_REQUIREMENTS.md) — Detailed dependency guide
- [MEMORY_BACKEND_SYSTEM.md](MEMORY_BACKEND_SYSTEM.md) — Backend architecture
- [MCP_SERVER_SETUP.md](MCP_SERVER_SETUP.md) — Claude Desktop integration
- [APP_DEVELOPMENT_GUIDE.md](APP_DEVELOPMENT_GUIDE.md) — Building apps on HoloLoom
