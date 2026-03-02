# HoloLoom Backend Setup Guide

Complete guide to setting up Neo4j and Qdrant backends for production-grade memory storage.

## Quick Start (Docker)

### 1. Install Docker Desktop
- Download from https://www.docker.com/products/docker-desktop
- Install and start Docker Desktop

### 2. Start All Backends

```bash
cd HoloLoom
docker-compose up -d
```

This starts:
- **Neo4j**: Graph database on `bolt://localhost:7687`, Web UI at `http://localhost:7474`
- **Qdrant**: Vector database on `http://localhost:6333`, Dashboard at `http://localhost:6333/dashboard`

### 3. Verify Services

```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Install Python Dependencies

```bash
pip install neo4j qdrant-client sentence-transformers
```

### 5. Test Connection

```python
from HoloLoom.memory.stores.hybrid_neo4j_qdrant import HybridNeo4jQdrant

# Initialize hybrid store
store = HybridNeo4jQdrant(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="hololoom123",
    qdrant_url="http://localhost:6333"
)

# Store a memory
memory_id = store.add(
    text="Test memory for HoloLoom",
    context={"type": "test", "source": "setup_guide"}
)

print(f"✓ Memory stored: {memory_id}")

# Search
results = store.search(query="test memory", limit=5)
print(f"✓ Found {len(results)} results")
```

---

## Detailed Setup

### Option 1: Docker (Recommended)

#### Neo4j Container

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/hololoom123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  neo4j:5.14.0
```

**Access:**
- Web Interface: http://localhost:7474
- Bolt Connection: bolt://localhost:7687
- Username: `neo4j`
- Password: `hololoom123`

#### Qdrant Container

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

**Access:**
- REST API: http://localhost:6333
- gRPC: localhost:6334
- Dashboard: http://localhost:6333/dashboard

### Option 2: Local Installation

#### Neo4j Desktop

1. Download Neo4j Desktop: https://neo4j.com/download/
2. Install and create a new database
3. Install APOC plugin (for graph algorithms)
4. Set password to `hololoom123`
5. Start database

#### Qdrant Local

```bash
# Using Cargo (Rust)
cargo install qdrant

# Or download binary
wget https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xvf qdrant-x86_64-unknown-linux-gnu.tar.gz
./qdrant
```

### Option 3: Cloud Services

#### Neo4j AuraDB (Free Tier)

1. Go to https://neo4j.com/cloud/aura/
2. Create free instance
3. Save credentials
4. Update connection URI in code

#### Qdrant Cloud

1. Go to https://qdrant.tech/
2. Create free cluster
3. Get API key and URL
4. Update in code:

```python
store = QdrantMemoryStore(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)
```

---

## Configuration

### Environment Variables

Create `.env` file in HoloLoom directory:

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=hololoom123
NEO4J_DATABASE=neo4j

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional, for cloud

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Load in Python

```python
from dotenv import load_dotenv
load_dotenv()

from HoloLoom.memory.neo4j_graph import Neo4jConfig
config = Neo4jConfig.from_env()
```

---

## Usage Examples

### 1. Neo4j Only (Graph Relationships)

```python
from HoloLoom.memory.stores.neo4j_store import Neo4jMemoryStore

store = Neo4jMemoryStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="hololoom123"
)

# Store with relationships
store.add(
    text="Inspected Hive Jodi - 8 frames of brood",
    context={
        "entity": "Hive Jodi",
        "action": "inspection",
        "location": "apiary"
    }
)

# Query relationships
results = store.search(
    query="What happened with Hive Jodi?",
    strategy="graph"  # Use graph traversal
)
```

### 2. Qdrant Only (Semantic Search)

```python
from HoloLoom.memory.stores.qdrant_store import QdrantMemoryStore

store = QdrantMemoryStore(
    url="http://localhost:6333",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    scales=[96, 192, 384]  # Multi-scale embeddings
)

# Store with automatic embedding
store.add(
    text="Python best practices for error handling",
    context={"topic": "programming", "language": "python"}
)

# Semantic search
results = store.search(
    query="how to handle exceptions in Python",
    limit=5
)
```

### 3. Hybrid (Best of Both)

```python
from HoloLoom.memory.stores.hybrid_neo4j_qdrant import HybridNeo4jQdrant

store = HybridNeo4jQdrant(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="hololoom123",
    qdrant_url="http://localhost:6333"
)

# Store once, searchable both ways
memory_id = store.add(
    text="Implemented OAuth2 authentication for user API",
    context={
        "project": "user-service",
        "feature": "authentication",
        "tech": ["OAuth2", "JWT", "Redis"]
    }
)

# Search with fusion (combines graph + vectors)
results = store.search(
    query="authentication implementation",
    strategy="fused",  # Combines both backends
    limit=10
)
```

### 4. Unified Memory (Full HoloLoom)

```python
from HoloLoom.memory.unified import UnifiedMemory

# Auto-detects and uses all available backends
memory = UnifiedMemory(user_id="blake")

# Simple storage
mem_id = memory.store(
    "Completed Phase 1 of HoloLoom integration",
    context={"project": "HoloLoom", "phase": 1},
    importance=0.8
)

# Smart recall (uses best strategy automatically)
memories = memory.recall(
    "What did I accomplish with HoloLoom?",
    limit=5
)

for mem in memories:
    print(f"[{mem.relevance:.2f}] {mem.text}")
```

---

## Verification & Testing

### Test Neo4j Connection

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "hololoom123")
)

with driver.session() as session:
    result = session.run("RETURN 'Connected!' as message")
    print(result.single()["message"])

driver.close()
```

### Test Qdrant Connection

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collections = client.get_collections()
print(f"✓ Qdrant connected: {len(collections.collections)} collections")
```

### Run Full Test Suite

```bash
cd HoloLoom
python -m pytest tests/test_backends.py -v
```

---

## Troubleshooting

### Neo4j Issues

**"Connection refused"**
```bash
# Check if running
docker ps | grep neo4j

# Check logs
docker logs neo4j

# Restart
docker restart neo4j
```

**"Authentication failed"**
- Default password: `hololoom123`
- Reset: Delete docker volume and recreate

### Qdrant Issues

**"Failed to connect"**
```bash
# Check status
curl http://localhost:6333/health

# Check collections
curl http://localhost:6333/collections
```

**"Out of memory"**
- Increase Docker memory limit in Docker Desktop settings
- Use smaller embedding model (96d instead of 384d)

### Python Import Errors

```bash
# Install all dependencies
pip install neo4j qdrant-client sentence-transformers torch

# Or use HoloLoom requirements
pip install -r HoloLoom/requirements.txt
```

---

## Performance Tuning

### Neo4j Optimization

```cypher
// Create indexes for faster queries
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp);

// Enable query logging
CALL dbms.setConfigValue('dbms.logs.query.enabled', 'true');
```

### Qdrant Optimization

```python
# Use smaller scales for faster search
store = QdrantMemoryStore(
    scales=[96],  # Single scale - fastest
    # scales=[96, 192, 384]  # Multi-scale - best quality
)

# Adjust HNSW parameters
from qdrant_client.models import HnswConfigDiff

client.update_collection(
    collection_name="memories_96",
    hnsw_config=HnswConfigDiff(
        m=16,  # Number of edges per node
        ef_construct=100  # Search depth during construction
    )
)
```

---

## Next Steps

1. ✅ Start backends with docker-compose
2. ✅ Install Python dependencies
3. ✅ Test connections
4. ✅ Run example code
5. 📊 Integrate with Promptly
6. 🚀 Deploy to production

See [PROMPTLY_INTEGRATION.md](../Promptly/PROMPTLY_INTEGRATION.md) for Promptly-specific integration guide.
