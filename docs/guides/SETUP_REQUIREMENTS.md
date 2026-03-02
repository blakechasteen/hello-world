# Setup Requirements

What HoloLoom needs at each level of deployment.

## Quick Install

```bash
pip install -e "."          # Core only (torch, numpy, networkx, httpx, pydantic)
pip install -e ".[all]"     # Everything
```

## Optional Extras

```bash
pip install hololoom[nlp]         # spacy, sentence-transformers
pip install hololoom[voice]       # whisper, librosa
pip install hololoom[vision]      # opencv, ultralytics
pip install hololoom[server]      # fastapi, uvicorn, websockets
pip install hololoom[ml]          # scipy, scikit-learn
pip install hololoom[rl]          # gymnasium
pip install hololoom[production]  # qdrant-client, neo4j, redis
pip install hololoom[viz]         # matplotlib, plotly
pip install hololoom[dev]         # pytest, black, ruff, mypy
pip install hololoom[all]         # everything above
```

## Three Levels

### Minimal (get started in 5 minutes)

- Python 3.10+
- Core deps: `torch`, `numpy`, `networkx`, `httpx`, `pydantic`
- No external services needed
- Uses in-memory backend

```bash
pip install -e "."
python -c "from hololoom import HoloLoom; print('OK')"
```

**Limitations:** No semantic embeddings (uses fallback), no NLP entity extraction, no persistence.

### Recommended (full features, no services)

```bash
pip install -e ".[nlp,ml,viz,rl]"
python -m spacy download en_core_web_sm
```

**Adds:** Matryoshka embeddings (96d/192d/384d), NLP entity extraction, spectral graph features, visualization.

### Production (persistent storage)

```bash
pip install -e ".[nlp,ml,production,server]"
cd config && docker-compose up -d
```

**Adds:** Neo4j (graph DB, port 7474/7687), Qdrant (vector DB, port 6333), FastAPI server.

See [DOCKER_MEMORY_SETUP.md](DOCKER_MEMORY_SETUP.md) for Docker setup details.

## Hardware Requirements

| Level | CPU | RAM | Disk | GPU |
|-------|-----|-----|------|-----|
| Minimal | 2+ cores | 4GB | 2GB | - |
| Recommended | 4+ cores | 8GB | 10GB | Optional |
| Production | 8+ cores | 16GB+ | 50GB+ | Recommended (8GB+ VRAM) |

Tested on Linux (Ubuntu 20.04+), macOS 11+, Windows 10/11 (WSL2 recommended for Docker).

## Graceful Degradation

All optional dependencies degrade with warnings, never crashes:

```python
# If spacy is missing:
# WARNING: spaCy not available, using regex fallback for entity extraction

# If sentence-transformers is missing:
# WARNING: Using random embeddings (install sentence-transformers for semantic search)
```

## Environment Variables

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="hololoom123"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export OLLAMA_HOST="http://localhost:11434"
```

## Verification

```bash
# Core
python -c "from hololoom import HoloLoom; print('Core OK')"

# Embeddings
python -c "from hololoom.core.embedding.spectral import MatryoshkaEmbedding; print('Embeddings OK')"

# Knowledge Graph
python -c "from hololoom.core.memory.graph import KG; print('KG OK')"

# Tests
pytest hololoom/tests/unit/ -v
```
