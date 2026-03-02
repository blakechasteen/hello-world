# Claude Desktop Architecture

How HoloLoom integrates with Claude Desktop through two complementary skills and an MCP server.

## Two-Skill Architecture

```
Claude Desktop
    |
    |-- spinning-wheel (Data Ingestion)
    |       Websites, docs, browser history -> memory
    |
    |-- loom (Memory & Orchestration)
    |       Query -> recall -> synthesize -> respond
    |
    |-- MCP Server (HoloLoom-memory)
            Store, recall, weave, analytics
```

### SpinningWheel Skill

Handles multi-modal data ingestion:
- **TextSpinner** — Plain text documents
- **WebsiteSpinner** — Web pages with recursive crawling
- **YouTubeSpinner** — Video transcripts
- **AudioSpinner** — Audio files

Workflow: scrape/process content -> chunk into shards -> store via MCP tools.

### Loom Skill

Handles intelligent retrieval and synthesis:
- Assess query complexity
- Select retrieval strategy (semantic, graph, temporal, fused)
- Retrieve relevant memories
- Synthesize response with provenance citations

## MCP Server

The MCP server (`hololoom.mcp_tools.server`) exposes HoloLoom's API as tools:

| Tool | Description |
|------|-------------|
| `hololoom_experience` | Store a memory |
| `hololoom_recall` | Search memories |
| `hololoom_weave` | Full weaving pipeline |
| `hololoom_analytics_summary` | System metrics |
| `memory_health` | Backend status check |
| `ingest_webpage` | Scrape and store web page |
| `process_text` | Process and store text |
| `chat` | Conversational interface |

## Data Flow

```
User: "Add this article to memory: https://example.com"
    |
    v
Claude loads spinning-wheel skill
    |
    v
Skill guides Claude to call: ingest_webpage(url)
    |
    v
MCP server: scrape -> chunk -> embed -> store
    |
    v
Claude: "Stored 12 chunks from the article"

User: "What did I learn about embeddings?"
    |
    v
Claude loads loom skill
    |
    v
Skill guides Claude to call: hololoom_recall(query)
    |
    v
MCP server: embed query -> multi-scale search -> rank
    |
    v
Claude synthesizes answer with citations
```

## Memory Backend

```
MCP Server
    |
    |-- Neo4j (graph relationships)
    |-- Qdrant (vector similarity)
    |-- InMemory (session cache)
    |
    v
Hybrid retrieval: graph + vector + temporal
```

See [MCP_SERVER_SETUP.md](MCP_SERVER_SETUP.md) for setup instructions.
See [MEMORY_BACKEND_SYSTEM.md](MEMORY_BACKEND_SYSTEM.md) for backend details.
