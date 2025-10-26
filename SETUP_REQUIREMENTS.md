# What HoloLoom Needs to Work

A comprehensive guide to getting HoloLoom operational, from minimal setup to full-featured deployment.

---

## Quick Answer: Three Levels

### 🟢 **Minimal** (Get Started in 5 Minutes)
- Python 3.9+
- 4 core packages: torch, numpy, networkx, gymnasium
- **No external services needed**
- Uses in-memory backend

### 🟡 **Recommended** (Full HoloLoom Features)
- Minimal setup +
- 6 additional packages: spacy, sentence-transformers, scipy, matplotlib, ollama
- spaCy language model (en_core_web_sm)
- **Still no external services**

### 🔴 **Production** (All Features + Persistence)
- Recommended setup +
- Neo4j database (via Docker)
- Optional: Qdrant vector database
- Optional: Mem0 for memory management

---

## Detailed Setup Guide

### Level 1: Minimal Setup ✅

**What You Get:**
- Core HoloLoom orchestrator
- Basic embeddings (no transformer models)
- In-memory knowledge graph
- Policy engine with Thompson Sampling
- All demos work (with 'simple' backend)

**Installation:**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install torch>=2.0.0 numpy>=1.24.0 networkx>=3.0 gymnasium>=0.28.0

# Test it works
PYTHONPATH=. python demos/01_quickstart.py
```

**Dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Neural networks, tensors, GPU acceleration |
| numpy | >=1.24.0 | Numerical operations, arrays |
| networkx | >=3.0 | Knowledge graph structure |
| gymnasium | >=0.28.0 | RL environments (for training) |

**Limitations:**
- No semantic embeddings (uses fallback)
- No NLP entity extraction (uses regex only)
- No spectral features
- No persistence (data lost on restart)

---

### Level 2: Recommended Setup ⭐

**What You Get:**
- Everything from Minimal +
- Semantic embeddings (Matryoshka 96d/192d/384d)
- NLP entity extraction (spaCy)
- Spectral graph features
- Local LLM support (Ollama)
- Visualization tools

**Installation:**
```bash
# All minimal dependencies first
pip install torch numpy networkx gymnasium

# Add full features
pip install spacy>=3.5.0 \
            sentence-transformers>=2.2.0 \
            scipy>=1.10.0 \
            matplotlib>=3.7.0 \
            ollama>=0.1.0

# Download spaCy language model
python -m spacy download en_core_web_sm

# Test full features
PYTHONPATH=. python HoloLoom/unified_api.py
```

**Additional Dependencies:**
| Package | Version | Purpose | Optional? |
|---------|---------|---------|-----------|
| spacy | >=3.5.0 | NLP, entity extraction | Graceful fallback |
| sentence-transformers | >=2.2.0 | Semantic embeddings | Graceful fallback |
| scipy | >=1.10.0 | Spectral features, optimization | Graceful fallback |
| matplotlib | >=3.7.0 | Visualization | Optional |
| ollama | >=0.1.0 | Local LLM | Optional |

**Still No External Services Needed!** Everything runs locally.

**What This Unlocks:**
- Multi-scale semantic search
- Entity/relationship extraction from text
- Graph spectral analysis (eigenvalues, SVD)
- Local LLM enrichment (if Ollama running)

---

### Level 3: Production Setup 🚀

**What You Get:**
- Everything from Recommended +
- Persistent storage (survives restarts)
- Graph database (Neo4j) for relationships
- Vector database (Qdrant) for semantic search
- Hybrid memory strategies
- Multi-database queries

**Installation:**

#### Step 1: Core Dependencies
```bash
# All recommended dependencies first
pip install torch numpy networkx gymnasium spacy sentence-transformers scipy matplotlib

# Add database drivers
pip install neo4j>=5.14.0 \
            qdrant-client>=1.7.0 \
            mem0ai>=0.0.1
```

#### Step 2: Start External Services (Docker)
```bash
# Navigate to config directory
cd config

# Start Neo4j (and optionally Qdrant)
docker-compose up -d

# Verify Neo4j is running
# Open browser: http://localhost:7474
# Login: neo4j / hololoom123
```

**Docker Compose Provides:**
- **Neo4j (HoloLoom)**: Port 7474 (browser), 7687 (bolt)
  - Credentials: neo4j / hololoom123
  - APOC plugin enabled
  - Persistent volumes

- **Neo4j (Beekeeping)**: Port 7475 (browser), 7688 (bolt)
  - Credentials: neo4j / beekeeper123
  - Dedicated domain graph

**Configuration:**
```python
# In your code
from HoloLoom.memory.stores import Neo4jStore, QdrantStore, HybridNeo4jQdrant

# Neo4j only
store = Neo4jStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="hololoom123"
)

# Hybrid (best quality)
store = HybridNeo4jQdrant(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="hololoom123",
    qdrant_host="localhost",
    qdrant_port=6333
)
```

#### Step 3: Test Production Setup
```bash
PYTHONPATH=. python demos/06_hybrid_memory.py
```

---

## SpinningWheel (Data Ingestion) Requirements

For different data sources:

### YouTube Transcription
```bash
pip install youtube-transcript-api>=0.6.0

# Test
PYTHONPATH=. python -c "from HoloLoom.spinningWheel import transcribe_youtube; import asyncio; asyncio.run(transcribe_youtube('dQw4w9WgXcQ'))"
```

### Web Scraping
```bash
pip install beautifulsoup4 requests

# Already included in minimal setup via HoloLoom dependencies
```

### Audio Processing
```bash
# Add audio processing dependencies as needed
pip install librosa soundfile
```

---

## Promptly Framework Requirements

For the prompt engineering framework:

```bash
cd Promptly/promptly
pip install -r requirements.txt

# Core: click, PyYAML
# Optional: anthropic, openai (for cloud LLMs)
```

**For Ollama Support:**
```bash
# Install Ollama (https://ollama.ai)
# Then install Python client
pip install ollama

# Pull a model
ollama pull llama2
```

---

## Warp Drive Mathematical Framework

Warp Drive is included with core HoloLoom, but for GPU acceleration:

```bash
# PyTorch with CUDA support (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Performance Boost:**
- CPU: 19-87ms (5-50 threads)
- GPU: 5-20ms (batches of 10-100) = **10-50x faster**

---

## Environment Variables

Optional configuration via environment:

```bash
# Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="hololoom123"

# Qdrant
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"

# Ollama
export OLLAMA_HOST="http://localhost:11434"

# HoloLoom
export HOLOLOOM_MODE="FUSED"  # BARE, FAST, or FUSED
export PYTHONPATH="."  # Always set this when running
```

---

## System Requirements

### Hardware

**Minimal:**
- CPU: 2+ cores
- RAM: 4GB
- Disk: 2GB free

**Recommended:**
- CPU: 4+ cores
- RAM: 8GB
- Disk: 10GB free
- GPU: Optional (CUDA-capable, 4GB+ VRAM)

**Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Disk: 50GB+ (for Neo4j data)
- GPU: Recommended (8GB+ VRAM for large models)
- SSD storage (for Neo4j performance)

### Operating System

Tested on:
- ✅ Linux (Ubuntu 20.04+, Debian 11+)
- ✅ macOS (11+)
- ✅ Windows 10/11 (with WSL2 recommended for Docker)

---

## Complete Installation Script

Save as `install.sh`:

```bash
#!/bin/bash

# HoloLoom Complete Installation Script

echo "🔧 Setting up HoloLoom..."

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Choose installation level
echo "Choose installation level:"
echo "1) Minimal (core only, no external services)"
echo "2) Recommended (full features, no external services) ⭐"
echo "3) Production (full features + Neo4j/Qdrant)"

read -p "Enter choice [1-3]: " choice

case $choice in
  1)
    echo "📦 Installing minimal dependencies..."
    pip install torch>=2.0.0 numpy>=1.24.0 networkx>=3.0 gymnasium>=0.28.0
    ;;
  2)
    echo "📦 Installing recommended dependencies..."
    pip install torch>=2.0.0 numpy>=1.24.0 networkx>=3.0 gymnasium>=0.28.0 \
                spacy>=3.5.0 sentence-transformers>=2.2.0 scipy>=1.10.0 \
                matplotlib>=3.7.0 ollama>=0.1.0
    python -m spacy download en_core_web_sm
    ;;
  3)
    echo "📦 Installing production dependencies..."
    pip install torch>=2.0.0 numpy>=1.24.0 networkx>=3.0 gymnasium>=0.28.0 \
                spacy>=3.5.0 sentence-transformers>=2.2.0 scipy>=1.10.0 \
                matplotlib>=3.7.0 ollama>=0.1.0 neo4j>=5.14.0
    python -m spacy download en_core_web_sm

    echo "🐳 Starting Docker services..."
    cd config && docker-compose up -d && cd ..

    echo "⏳ Waiting for Neo4j to be ready..."
    sleep 10
    ;;
esac

# Set PYTHONPATH
export PYTHONPATH="."

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtualenv: source .venv/bin/activate"
echo "  2. Run a demo: PYTHONPATH=. python demos/01_quickstart.py"
echo "  3. Read CLAUDE.md for architecture overview"
```

Make executable: `chmod +x install.sh`

Run: `./install.sh`

---

## Verification Tests

After installation, verify everything works:

### Test 1: Core System
```bash
PYTHONPATH=. python -c "
from HoloLoom import HoloLoom
import asyncio

async def test():
    loom = await HoloLoom.create()
    print('✅ Core system working')

asyncio.run(test())
"
```

### Test 2: Embeddings
```bash
PYTHONPATH=. python -c "
from HoloLoom.embedding.spectral import MatryoshkaEmbedding

emb = MatryoshkaEmbedding([96, 192, 384])
vec = emb.embed('Hello world')
print(f'✅ Embeddings working: {vec.shape}')
"
```

### Test 3: Knowledge Graph
```bash
PYTHONPATH=. python -c "
from HoloLoom.memory.graph import KG

kg = KG()
kg.add_entity('entity_1', 'Person', {'name': 'Alice'})
print(f'✅ Knowledge graph working: {kg.entity_count()} entities')
"
```

### Test 4: Policy Engine
```bash
PYTHONPATH=. python HoloLoom/test_unified_policy.py
```

### Test 5: Warp Drive
```bash
PYTHONPATH=. python tests/test_warp_drive_complete.py
```

### Test 6: Neo4j (Production Only)
```bash
PYTHONPATH=. python -c "
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    'bolt://localhost:7687',
    auth=('neo4j', 'hololoom123')
)

with driver.session() as session:
    result = session.run('RETURN 1 as num')
    print(f'✅ Neo4j connected: {result.single()[0]}')

driver.close()
"
```

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'HoloLoom'**
```bash
# Always set PYTHONPATH from repository root
export PYTHONPATH=.
# Or run with: PYTHONPATH=. python your_script.py
```

**2. PyTorch ImportError**
```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch>=2.0.0
```

**3. spaCy Model Not Found**
```bash
# Download the language model
python -m spacy download en_core_web_sm

# Verify
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy working')"
```

**4. Neo4j Connection Refused**
```bash
# Check Docker is running
docker ps

# Start Neo4j if not running
cd config && docker-compose up -d

# Check logs
docker logs hololoom-neo4j
```

**5. CUDA Out of Memory (GPU)**
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU

# Or use smaller models
```

**6. Permission Denied (Docker volumes)**
```bash
# Fix volume permissions
sudo chown -R $(whoami):$(whoami) ~/.docker
```

---

## What Each Component Needs

### HoloLoom Core
- ✅ torch, numpy, networkx, gymnasium
- Optional: spacy, sentence-transformers, scipy

### Warp Drive
- ✅ torch (core tensor operations)
- Optional: CUDA for GPU acceleration
- All included with HoloLoom core

### Promptly
- ✅ click, PyYAML
- Optional: anthropic, openai, ollama

### SpinningWheel
- ✅ Core: Built-in with HoloLoom
- Optional: youtube-transcript-api (YouTube)
- Optional: beautifulsoup4, requests (web scraping)

### Memory Systems
- ✅ Simple: No dependencies (in-memory)
- Optional: neo4j (graph database)
- Optional: qdrant-client (vector database)
- Optional: mem0ai (memory management)

---

## Quick Reference

### Minimal Command Sequence
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy networkx gymnasium
export PYTHONPATH=.
python demos/01_quickstart.py
```

### Recommended Command Sequence
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy networkx gymnasium spacy sentence-transformers scipy
python -m spacy download en_core_web_sm
export PYTHONPATH=.
python HoloLoom/unified_api.py
```

### Production Command Sequence
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy networkx gymnasium spacy sentence-transformers scipy neo4j
python -m spacy download en_core_web_sm
cd config && docker-compose up -d && cd ..
export PYTHONPATH=.
python demos/06_hybrid_memory.py
```

---

## Summary: What You Actually Need

**To run HoloLoom:**
- Python 3.9+
- 4 packages: `torch numpy networkx gymnasium`
- PYTHONPATH=. environment variable

**That's it!** Everything else is optional enhancements.

The system uses **graceful degradation** - if optional dependencies are missing, it falls back to simpler implementations with warnings, not crashes.

**Recommendation:** Start with Level 2 (Recommended) setup for best experience without external services.

---

*Last updated: October 26, 2025*
