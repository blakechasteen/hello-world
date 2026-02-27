# Installation Guide

Complete setup instructions for HoloLoom, from basic installation to production deployment.

---

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.9 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB for dependencies, 10GB+ for production data

### Production Requirements
- **RAM**: 16GB+ (for Neo4j + Qdrant)
- **Disk**: 50GB+ SSD
- **Docker**: 20.10+ (for backend services)
- **Kubernetes**: 1.24+ (optional, for cluster deployment)

---

## Basic Installation

### 1. Clone Repository

```bash
git clone https://github.com/blakewoolbright/mythRL
cd mythRL
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Core Dependencies

```bash
# Core requirements (required)
pip install torch numpy networkx

# Basic functionality
pip install gymnasium matplotlib
```

### 4. Verify Installation

```bash
# Test import
python -c "from hololoom import hololoom; print('HoloLoom installed successfully!')"
```

---

## Optional Dependencies

Install additional features as needed:

### NLP Features
```bash
# spaCy for linguistic analysis
pip install spacy
python -m spacy download en_core_web_sm

# Sentence transformers for embeddings
pip install sentence-transformers
```

### Full-Text Search
```bash
pip install rank-bm25
```

### Web Features
```bash
# API server
pip install fastapi uvicorn

# Web scraping
pip install beautifulsoup4 requests
```

### Audio/Video Processing
```bash
# YouTube transcription
pip install yt-dlp

# Audio processing
pip install librosa soundfile
```

### Production Features
```bash
# Monitoring
pip install prometheus-client

# Database clients
pip install neo4j qdrant-client

# Configuration
pip install pydantic python-dotenv
```

---

## Production Setup

### 1. Docker Backend Services

**Start Neo4j (Knowledge Graph) + Qdrant (Vector Database)**:

```bash
# Start services
docker-compose up -d

# Verify services
docker ps
# Should show: neo4j (ports 7474, 7687) and qdrant (ports 6333, 6334)
```

**docker-compose.yml** (create if missing):
```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.9.0
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/hololoom123
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

  qdrant:
    image: qdrant/qdrant:v1.5.0
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  neo4j_data:
  qdrant_data:
```

### 2. Configure Backend

Create `.env` file in repository root:

```bash
# Memory Backend
MEMORY_BACKEND=HYBRID  # INMEMORY | HYBRID | HYPERSPACE

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=hololoom123

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Performance
QUERY_CACHE_SIZE=10000
ENABLE_ZERO_COPY_EMBEDDINGS=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/hololoom.log

# Alignment
ENABLE_ALIGNMENT=true
ENABLE_AUDIT_TRAIL=true
```

### 3. Verify Production Setup

```python
from hololoom.config import Config, MemoryBackend
from hololoom.memory.backend_factory import create_memory_backend
import asyncio

async def verify():
    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID

    try:
        memory = await create_memory_backend(config)
        print("✓ HYBRID backend connected (Neo4j + Qdrant)")
    except Exception as e:
        print(f"✗ Backend error: {e}")
        print("  Falling back to INMEMORY backend")

asyncio.run(verify())
```

---

## Development Setup

### 1. Install Development Dependencies

```bash
pip install pytest pytest-asyncio black mypy
```

### 2. Run Tests

```bash
# All tests
pytest hololoom/tests/ -v

# Unit tests only (fast)
pytest hololoom/tests/unit/ -v

# Integration tests
pytest hololoom/tests/integration/ -v

# End-to-end tests
pytest hololoom/tests/e2e/ -v
```

### 3. Code Formatting

```bash
# Format code
black hololoom/

# Type checking
mypy hololoom/ --ignore-missing-imports
```

---

## Platform-Specific Notes

### Windows

**Set PYTHONPATH**:
```cmd
set PYTHONPATH=.
python your_script.py
```

**PowerShell**:
```powershell
$env:PYTHONPATH="."
python your_script.py
```

### macOS

**Install Homebrew dependencies**:
```bash
brew install python@3.11
brew install docker docker-compose
```

### Linux (Ubuntu/Debian)

**Install system dependencies**:
```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3-pip
sudo apt-get install docker.io docker-compose
```

---

## Kubernetes Deployment (Optional)

For cluster deployment, see [Production Deployment Guide](../guides/production/deployment.md).

**Quick start**:
```bash
# Create namespace
kubectl create namespace hololoom

# Deploy services
kubectl apply -f deploy/k8s/

# Verify deployment
kubectl get pods -n hololoom
```

---

## Troubleshooting

### Common Issues

**1. Import Error: "No module named 'HoloLoom'"**

**Solution**: Set PYTHONPATH from repository root
```bash
export PYTHONPATH=.
python your_script.py
```

**2. Docker Connection Error**

**Solution**: Verify Docker is running
```bash
docker ps
# If error, start Docker daemon

# Verify services
docker-compose ps
```

**3. Neo4j Authentication Error**

**Solution**: Reset password
```bash
docker exec -it mythrl-neo4j-1 cypher-shell
# Default: neo4j/neo4j
# Change password when prompted
```

**4. Qdrant Connection Timeout**

**Solution**: Check firewall/ports
```bash
# Test connectivity
curl http://localhost:6333/health
# Should return: {"status": "ok"}
```

**5. Out of Memory (OOM) Error**

**Solution**: Increase Docker memory limit
```bash
# Docker Desktop: Settings → Resources → Memory → 8GB+
```

---

## Verification Checklist

After installation, verify:

- [ ] HoloLoom imports successfully
- [ ] Docker services running (if HYBRID backend)
- [ ] Neo4j accessible at http://localhost:7474
- [ ] Qdrant accessible at http://localhost:6333
- [ ] Tests passing: `pytest hololoom/tests/unit/ -v`
- [ ] Simple query works (see [Quickstart](quickstart.md))

---

## Next Steps

- [Your First Query](first-query.md) - Hello World tutorial
- [Configuration Guide](configuration.md) - BARE/FAST/FUSED modes
- [Department Overview](../guides/departments/README.md) - System architecture
- [Production Deployment](../guides/production/deployment.md) - Deploy to production

---

## Getting Help

- [Troubleshooting Guide](../guides/production/troubleshooting.md)
- [GitHub Issues](https://github.com/blakewoolbright/mythRL/issues)
- [Discussions](https://github.com/blakewoolbright/mythRL/discussions)

---

**Last Updated**: November 2025 | **Documentation Version**: 1.1.0
