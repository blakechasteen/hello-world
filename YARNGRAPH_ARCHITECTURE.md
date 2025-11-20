# YarnGraph Architecture: From YouTube to Knowledge

This document explains how your YouTube transcripts become a queryable knowledge graph.

## The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: YouTube Video                                           │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: YouTubeSpinner (What You Just Built!)                 │
│  ─────────────────────────────────────────────────────────────  │
│  • Extract transcript using youtube-transcript-api             │
│  • Parse segments with timestamps                               │
│  • Extract basic entities                                        │
│  • Create MemoryShards                                          │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: MemoryShard Structure                                  │
│  ─────────────────────────────────────────────────────────────  │
│  {                                                               │
│    id: "youtube_VIDEO_ID_chunk_001",                           │
│    text: "In this video we discuss...",                         │
│    episode: "youtube_VIDEO_ID",                                │
│    entities: ["Python", "Neural Networks"],                     │
│    motifs: [],                                                   │
│    metadata: {                                                   │
│      source: "youtube",                                          │
│      video_id: "VIDEO_ID",                                      │
│      chunk_start: 0.0,                                           │
│      chunk_end: 60.0,                                            │
│      url: "https://youtube.com/watch?v=..."                     │
│    }                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Dual Storage (YarnGraph Foundation)                    │
└─────────────────────────────────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        ┌───────────────┐     ┌──────────────┐
        │   Neo4j       │     │   Qdrant     │
        │   (Graph)     │     │   (Vectors)  │
        └───────────────┘     └──────────────┘
                │                     │
                └──────────┬──────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: HoloLoom Orchestrator                                  │
│  ─────────────────────────────────────────────────────────────  │
│  • Receives user query                                           │
│  • Semantic search in Qdrant → Find relevant chunks            │
│  • Graph traversal in Neo4j → Expand context                   │
│  • Motif detection → Extract patterns                           │
│  • Policy engine → Decide response strategy                     │
│  • Generate answer using retrieved context                      │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Intelligent Response                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Storage Architecture

### Neo4j (YarnGraph - Symbolic Memory)

The **discrete thread structure** - stores entities and relationships.

```cypher
// Graph Structure
(Video {id: "dQw4w9WgXcQ", url: "...", duration: 212.5})
  |
  +--[HAS_SEGMENT {order: 0}]--> (Segment {id: "...chunk_001", text: "...", start: 0, end: 60})
  |                                   |
  |                                   +--[MENTIONS {timestamp: 5.2}]--> (Entity {name: "Python"})
  |                                   |
  |                                   +--[MENTIONS {timestamp: 15.8}]--> (Entity {name: "Machine Learning"})
  |
  +--[HAS_SEGMENT {order: 1}]--> (Segment {id: "...chunk_002", text: "...", start: 60, end: 120})
                                      |
                                      +--[MENTIONS {timestamp: 75.3}]--> (Entity {name: "Python"})

// Relationships between entities
(Entity {name: "Python"}) -[CO_OCCURS_WITH {count: 5}]-> (Entity {name: "Machine Learning"})
```

**Why Neo4j?**
- Find all videos that mention "Python"
- Discover related entities: "If they searched X, also show Y"
- Context expansion: "What else was discussed around timestamp 30s?"
- Graph traversal: "What topics connect Video A and Video B?"

### Qdrant (Vector Store - Semantic Memory)

The **continuous representation** - stores embeddings for semantic search.

```json
{
  "points": [
    {
      "id": "youtube_VIDEO_ID_chunk_001",
      "vector": [0.234, -0.456, 0.789, ...],  // 384-dim embedding
      "payload": {
        "text": "In this video we discuss Python...",
        "video_id": "dQw4w9WgXcQ",
        "url": "https://youtube.com/watch?v=...",
        "chunk_start": 0.0,
        "chunk_end": 60.0,
        "entities": ["Python", "Machine Learning"]
      }
    }
  ]
}
```

**Why Qdrant?**
- Semantic search: "Find videos about 'deep learning'" (even if they say "neural networks")
- Similarity: "Find videos similar to this one"
- Fast: Vector search is O(log n) with HNSW index
- Multi-scale: Supports Matryoshka embeddings (96, 192, 384 dims)

## The "Weaving" Metaphor

From CLAUDE.md - this is the core HoloLoom philosophy:

### Yarn Graph (Discrete Threads)
```
[Video] --> [Segment] --> [Entity]
  ^                          ^
  |                          |
Discrete symbols          Named concepts
```

### Warp Space (Tensioned Continuous)
```
When queried, threads are "tensioned" into continuous space:

Discrete Graph     →    Tension    →    Continuous Manifold
[Entities]         →    [Embed]    →    Vector[384]
[Relationships]    →    [Weights]  →    Attention Matrix
```

### Complete Query Flow

```
1. User Query: "What is discussed in the Python tutorial?"

2. Orchestrator processes:
   - Extract features (motifs, embeddings)
   - Query Qdrant: Find semantically similar chunks
   - Query Neo4j: Expand context via graph

3. YarnGraph → Warp Space:
   - Selected entities/segments "tension" into vectors
   - Discrete knowledge → continuous representation

4. Policy Engine decides:
   - Which tools to use
   - How to combine context
   - Response strategy

5. Response generated:
   - "The video discusses Python basics, focusing on..."
   - Includes: sources, timestamps, confidence scores
```

## Setting Up the Stack

### 1. Neo4j (Graph Database)

```bash
# Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Or Neo4j Desktop
# Download from: https://neo4j.com/download/
```

### 2. Qdrant (Vector Database)

```bash
# Docker
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  qdrant/qdrant:latest

# Or local installation
pip install qdrant-client
```

### 3. Create Collections

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Create collection for video transcripts
client.create_collection(
    collection_name="youtube_transcripts",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
```

## Integration Example

### Step 1: Transcribe and Save
```bash
python transcribe_and_save.py "https://youtube.com/watch?v=VIDEO_ID" --format json
```

### Step 2: Prepare for YarnGraph
```bash
python integrate_to_yarngraph.py transcripts/VIDEO_ID_*.json
```

### Step 3: Insert into Neo4j (Example)
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Insert video node
with driver.session() as session:
    session.run("""
        CREATE (v:Video {
            id: $video_id,
            url: $url,
            duration: $duration
        })
    """, video_id="...", url="...", duration=212.5)
```

### Step 4: Insert into Qdrant (Example)
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Transcript text here...")

# Insert into Qdrant
client = QdrantClient("localhost", port=6333)
client.upsert(
    collection_name="youtube_transcripts",
    points=[{
        "id": "youtube_VIDEO_ID_chunk_001",
        "vector": embedding.tolist(),
        "payload": {
            "text": "Transcript text...",
            "video_id": "VIDEO_ID",
            "url": "..."
        }
    }]
)
```

### Step 5: Query via HoloLoom
```python
from HoloLoom.orchestrator import Orchestrator
from HoloLoom.config import Config

# Initialize orchestrator (it connects to Neo4j + Qdrant)
config = Config.fused()
orchestrator = Orchestrator(config)

# Query
response = await orchestrator.process("What topics are covered in the Python tutorial?")
print(response.text)
print(f"Sources: {response.context.sources}")
```

## Why This Architecture?

### Hybrid Search = Best of Both Worlds

**Neo4j (Symbolic)**
- ✅ Exact entity matching
- ✅ Relationship traversal
- ✅ Explainable paths
- ✅ Structured queries

**Qdrant (Semantic)**
- ✅ Fuzzy/semantic matching
- ✅ Handles synonyms
- ✅ Cross-lingual search
- ✅ Similarity ranking

**Together** = Powerful, intelligent search that understands both structure and meaning!

## Next Steps

1. **Save your transcripts** ✅ (You have this now!)
2. **Set up Neo4j + Qdrant** (Docker makes this easy)
3. **Create integration pipeline** (Use `integrate_to_yarngraph.py` as template)
4. **Connect to HoloLoom** (Orchestrator handles queries)
5. **Query your knowledge!** 🎉

## Resources

- **Neo4j Cypher**: https://neo4j.com/docs/cypher-manual/
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **HoloLoom Architecture**: See `CLAUDE.md` in repo
- **Sentence Transformers**: https://www.sbert.net/

---

**The Foundation You Built:**
Your YouTube transcriber is the **input adapter** that feeds the entire system.
Now the knowledge graph (YarnGraph) can store, connect, and intelligently retrieve that knowledge! 🧵✨
