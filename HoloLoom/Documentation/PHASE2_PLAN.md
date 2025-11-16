# HoloLoom Phase 2 Enhancement Plan

## Overview

Phase 2 adds production-ready features for enterprise deployment:
- Async database operations (Neo4j, Qdrant)
- Intelligent clustering for memory organization
- MCP server integration for tool ecosystem
- Modern web interface with drag-and-drop
- Interactive chat with streaming responses

**Timeline**: 5 major components
**Estimated LOC**: ~2,500 new lines
**Dependencies**: neo4j, qdrant-client, fastapi, websockets

---

## 1. Async Neo4j/Qdrant Stores

### Architecture

```
HoloLoom/memory/
├── neo4j_store.py      # Async Neo4j knowledge graph
├── qdrant_store.py     # Async vector store
└── unified_store.py    # Hybrid store combining both
```

### Features

**Neo4jStore (Async)**
- Connection pooling with aiohttp
- Cypher query execution with async context managers
- Batch operations for bulk ingestion
- Transaction support for consistency
- Graph algorithms: PageRank, community detection

**QdrantStore (Async)**
- Collection management
- Vector search with filters
- Batch upsert for efficiency
- Payload-based retrieval
- Multi-scale embeddings support

**UnifiedStore**
- Combines graph structure (Neo4j) with vector search (Qdrant)
- Graph provides symbolic relationships
- Vectors provide semantic similarity
- Hybrid retrieval: graph traversal → vector re-ranking

### Dependencies

```bash
pip install neo4j qdrant-client asyncio-mqtt
```

### Configuration

```python
NEO4J_CONFIG = {
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': 'password',
    'database': 'hololoom'
}

QDRANT_CONFIG = {
    'host': 'localhost',
    'port': 6333,
    'collection': 'hololoom_embeddings',
    'vector_size': 384
}
```

---

## 2. Embedding-Based Clustering

### Architecture

```
HoloLoom/clustering/
├── cluster_engine.py   # Main clustering logic
├── algorithms.py       # HDBSCAN, K-means, Spectral
└── visualizer.py       # UMAP/t-SNE visualization
```

### Features

**ClusterEngine**
- Multi-algorithm support: HDBSCAN (density), K-means (centroid), Spectral (graph)
- Hierarchical clustering for multi-scale organization
- Automatic cluster labeling using LLMs
- Cluster quality metrics (silhouette, Davies-Bouldin)

**Use Cases**
1. **Memory Organization**: Cluster similar memories for efficient retrieval
2. **Topic Discovery**: Identify themes in conversation history
3. **Anomaly Detection**: Find outlier queries/responses
4. **Adaptive Scaling**: Adjust BARE/FAST/FUSED based on cluster density

### Algorithm Selection

| Algorithm | Best For | Speed | Quality |
|-----------|----------|-------|---------|
| HDBSCAN | Varying density, noise handling | Medium | High |
| K-means | Known cluster count, speed | Fast | Medium |
| Spectral | Graph-based, complex shapes | Slow | High |

### Dependencies

```bash
pip install hdbscan umap-learn scikit-learn plotly
```

---

## 3. MCP Server for Tool Integration

### Architecture

```
HoloLoom/mcp/
├── server.py           # MCP server implementation
├── tools/
│   ├── search.py       # Web search tool
│   ├── calculator.py   # Math tool
│   ├── code_exec.py    # Code execution sandbox
│   └── notion.py       # Notion integration
└── protocol.py         # MCP protocol handlers
```

### Features

**MCP Server**
- Implements Model Context Protocol spec
- Tool discovery and registration
- Streaming responses for long-running operations
- Authentication and rate limiting
- Tool execution with timeout/sandboxing

**Tool Registry**
- Dynamic tool registration
- JSON schema validation for inputs
- Automatic OpenAPI documentation
- Usage analytics per tool

**Integration with ConvergenceEngine**
- Tools register as available actions
- Thompson Sampling explores tool effectiveness
- Automatic tool selection based on query

### MCP Protocol Example

```python
# Tool registration
@mcp_server.tool("search_web")
async def search_web(query: str, max_results: int = 5):
    """Search the web for information"""
    results = await web_search_api(query, limit=max_results)
    return {"results": results}

# Client usage
result = await convergence_engine.execute_tool(
    tool="search_web",
    params={"query": "Thompson Sampling algorithm", "max_results": 3}
)
```

### Dependencies

```bash
pip install fastapi uvicorn pydantic websockets
```

---

## 4. Drag-and-Drop File Ingestion

### Architecture

```
HoloLoom/ingestion/
├── file_processor.py   # Main file handler
├── parsers/
│   ├── pdf.py          # PDF extraction
│   ├── docx.py         # Word documents
│   ├── audio.py        # Audio transcription
│   └── code.py         # Code file parsing
└── chunker.py          # Smart text chunking
```

### Features

**FileProcessor**
- Auto-detect file type by extension/MIME
- Streaming upload for large files (>100MB)
- Progress callbacks for UI
- Parallel processing for batches
- Automatic error recovery

**Smart Chunking**
- Semantic chunking (respect sentence/paragraph boundaries)
- Overlap for context preservation (10-20%)
- Metadata preservation (page numbers, timestamps)
- Chunk size optimization (384-512 tokens)

**Parsers**
- **PDF**: pypdf2 + OCR fallback (pytesseract)
- **DOCX**: python-docx
- **Audio**: whisper.cpp for local transcription
- **Code**: Tree-sitter for syntax-aware chunking

### Web Interface

```html
<!-- Drag-and-drop zone -->
<div id="dropzone" class="border-dashed border-2 p-8">
  <p>Drag files here or click to upload</p>
  <input type="file" multiple accept=".pdf,.docx,.mp3,.wav,.py,.js" />
</div>

<!-- Progress display -->
<div id="progress">
  <div class="progress-bar" style="width: 0%"></div>
  <p id="status">Waiting for files...</p>
</div>
```

### Dependencies

```bash
pip install pypdf2 python-docx faster-whisper pytesseract pillow
```

---

## 5. Interactive Chat Interface

### Architecture

```
HoloLoom/web/
├── app.py              # FastAPI server
├── static/
│   ├── index.html      # Main UI
│   ├── chat.js         # Chat logic
│   └── styles.css      # Tailwind CSS
├── templates/
│   └── chat.html       # Chat template
└── websocket_handler.py # Real-time streaming
```

### Features

**Chat Interface**
- Real-time streaming responses (WebSocket)
- Markdown rendering with syntax highlighting
- Code block copy buttons
- Message history with pagination
- Export conversations as JSON/PDF

**Advanced UI Elements**
- **Weaving Visualization**: Show computational trace in real-time
  - Feature extraction progress
  - Thread activation animation
  - Tool selection confidence bars
- **Pattern Card Selector**: Toggle BARE/FAST/FUSED modes
- **Context Window**: Display active threads and memories
- **Feedback Loop**: Thumbs up/down for quality signals

**Backend (FastAPI)**
- WebSocket endpoint for streaming
- REST API for history/export
- Session management
- Rate limiting per user

### UI Stack

```javascript
// Frontend
- Vanilla JS (no framework overhead)
- Tailwind CSS for styling
- Marked.js for Markdown
- Highlight.js for code syntax

// Real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/chat');
ws.onmessage = (event) => {
  const chunk = JSON.parse(event.data);
  appendToMessage(chunk.text);
  updateTrace(chunk.trace);
};
```

### Dependencies

```bash
pip install fastapi uvicorn jinja2 websockets markdown
```

---

## Implementation Order

### Phase 2A: Backend Infrastructure (Week 1)
1. ✅ Async Neo4j store with connection pooling
2. ✅ Async Qdrant store with multi-scale support
3. ✅ UnifiedStore combining both
4. ✅ Tests for all stores

### Phase 2B: Intelligence Layer (Week 2)
5. ✅ ClusterEngine with HDBSCAN
6. ✅ Cluster quality metrics
7. ✅ Integration with MemoryManager
8. ✅ Cluster-based retrieval optimization

### Phase 2C: Tool Ecosystem (Week 3)
9. ✅ MCP server skeleton
10. ✅ Basic tools (search, calc, notion)
11. ✅ ConvergenceEngine integration
12. ✅ Tool analytics and monitoring

### Phase 2D: Ingestion Pipeline (Week 4)
13. ✅ FileProcessor with auto-detection
14. ✅ PDF, DOCX, audio parsers
15. ✅ Smart chunking algorithm
16. ✅ Batch processing with progress

### Phase 2E: User Interface (Week 5)
17. ✅ FastAPI chat server
18. ✅ WebSocket streaming
19. ✅ React-free minimal UI
20. ✅ Weaving visualization
21. ✅ E2E testing

---

## Testing Strategy

### Unit Tests
- Each store with mocked databases
- Clustering algorithms on synthetic data
- MCP tool execution in sandbox
- File parsers with sample files

### Integration Tests
- Neo4j + Qdrant hybrid retrieval
- End-to-end chat flow
- File upload → processing → indexing
- Tool execution from chat

### Performance Tests
- 10k vectors in Qdrant (latency < 50ms)
- 100k nodes in Neo4j (traversal < 100ms)
- Concurrent chat sessions (100 users)
- Large file processing (1GB PDF < 5min)

---

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.13
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/hololoom123

  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports: ["6333:6333"]

  hololoom:
    build: .
    ports: ["8000:8000"]
    depends_on: [neo4j, qdrant]
    environment:
      NEO4J_URI: bolt://neo4j:7687
      QDRANT_HOST: qdrant
```

### Environment Variables

```bash
# .env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=hololoom123
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_HOST=http://localhost:11434
CHAT_PORT=8000
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query latency (FUSED) | < 500ms | p95 response time |
| Concurrent users | 100+ | Load test with Locust |
| Memory efficiency | < 2GB RAM | 10k indexed documents |
| Cluster quality | Silhouette > 0.5 | Benchmark datasets |
| File processing | 100 pages/min | PDF throughput |
| UI responsiveness | < 100ms | First contentful paint |

---

## Risk Mitigation

### Database Failures
- Fallback to in-memory stores if Neo4j/Qdrant unavailable
- Graceful degradation to existing cache.py

### Clustering Overhead
- Cache cluster assignments for 1 hour
- Skip clustering if < 100 vectors
- Use K-means for speed when needed

### MCP Tool Failures
- Timeout after 30s per tool
- Circuit breaker pattern after 3 failures
- Fallback to alternative tools

### File Parsing Errors
- Skip problematic files with warning
- OCR fallback for image-based PDFs
- User notification for unsupported formats

---

## Future Enhancements (Phase 3)

- Multi-modal embeddings (images, audio)
- Distributed deployment (Kubernetes)
- Real-time collaboration (multiplayer chat)
- Custom tool builder (no-code)
- Mobile app (React Native)

---

**Ready to begin implementation**: All architecture decisions documented and dependencies identified.
