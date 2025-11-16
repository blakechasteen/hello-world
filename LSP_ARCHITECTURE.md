# LSP Architecture and Data Flow

**Version**: 1.0.0
**Updated**: November 2025
**Audience**: Developers, Architects, Contributors

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Diagram](#component-diagram)
3. [Data Flow Examples](#data-flow-examples)
4. [Protocol Flow](#protocol-flow)
5. [Module Organization](#module-organization)
6. [Integration Points](#integration-points)
7. [Performance Architecture](#performance-architecture)
8. [Deployment Architecture](#deployment-architecture)

---

## System Architecture

### High-Level System

```
┌─────────────────────────────────────────────────────┐
│            Text Editors (50+ LSP clients)           │
│  ┌──────────┐  ┌───────┐  ┌──────┐  ┌──────────┐  │
│  │ VS Code  │  │Neovim │  │Emacs │  │Sublime..│  │
│  └────┬─────┘  └───┬───┘  └──┬───┘  └────┬─────┘  │
└───────┼────────────┼────────┼──────────┼──────────┘
        │            │        │          │
        └────────────┼────────┼──────────┘
                     │        │
        ┌────────────┴────────┴──────────┐
        │  LSP Protocol (JSON-RPC)       │
        │  over stdio/TCP connections    │
        └────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│     HoloLoom LSP Server (Single Python Process)    │
│  ┌─────────────────────────────────────────────┐   │
│  │ LSP Message Router (pygls)                  │   │
│  │ - Parse JSON-RPC messages                   │   │
│  │ - Route to appropriate handler              │   │
│  │ - Format responses back to JSON             │   │
│  └────────────┬────────────────────────────────┘   │
│               │                                     │
│  ┌────────────┴────────────────────────────────┐   │
│  │ LSP Handlers                                │   │
│  │ ├─ initialize/shutdown                      │   │
│  │ ├─ textDocument/completion                  │   │
│  │ ├─ textDocument/hover                       │   │
│  │ ├─ textDocument/definition                  │   │
│  │ ├─ textDocument/didOpen                     │   │
│  │ ├─ textDocument/didChange                   │   │
│  │ └─ textDocument/didClose                    │   │
│  └────────────┬────────────────────────────────┘   │
│               │                                     │
│  ┌────────────┴────────────────────────────────┐   │
│  │ Document Manager                           │   │
│  │ - Track open documents                      │   │
│  │ - Cache file content                        │   │
│  │ - Handle incremental changes                │   │
│  └────────────┬────────────────────────────────┘   │
│               │                                     │
└───────────────┼─────────────────────────────────────┘
                │
                │ Direct function calls (async)
                │
┌───────────────┴─────────────────────────────────────┐
│    HoloLoom Core Services                          │
│  ┌─────────────────────────────────────────────┐   │
│  │ Weaving Orchestrator                        │   │
│  │ ├─ recall(query, k) → memories[]            │   │
│  │ ├─ get_graph() → KG data                    │   │
│  │ ├─ detect_logic(code) → issues[]            │   │
│  │ └─ remember(content) → memory_id            │   │
│  └────────────┬────────────────────────────────┘   │
│               │                                     │
│  ┌────────────┴────────────────────────────────┐   │
│  │ Memory System                               │   │
│  │ ├─ Vector DB (Qdrant)                       │   │
│  │ │  ├─ Semantic embeddings (384d)            │   │
│  │ │  └─ BM25 keyword index                    │   │
│  │ ├─ Knowledge Graph (Neo4j/NetworkX)         │   │
│  │ │  ├─ Entity nodes (functions, classes)     │   │
│  │ │  └─ Relationship edges (CALLS, USES)      │   │
│  │ └─ Cache Layer (in-memory)                  │   │
│  │    ├─ Document cache (10 files max)         │   │
│  │    ├─ Completion cache (1000 entries)       │   │
│  │    └─ Graph cache (hot entities)            │   │
│  └────────────┬────────────────────────────────┘   │
│               │                                     │
│  ┌────────────┴────────────────────────────────┐   │
│  │ Configuration & Settings                    │   │
│  │ ├─ BARE/FAST/FUSED modes                    │   │
│  │ ├─ Performance targets (latency)            │   │
│  │ └─ Feature flags (enable/disable)           │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Logical Layers

```
┌──────────────────────────────────────┐
│      Presentation Layer (5)          │
│  LSP Message Formatting & Routing    │
├──────────────────────────────────────┤
│      Application Layer (4)           │
│  LSP Handlers (completion, hover)    │
├──────────────────────────────────────┤
│      Service Layer (3)               │
│  Document Manager, Cache Layer       │
├──────────────────────────────────────┤
│      Domain Layer (2)                │
│  Orchestrator, Core Algorithms       │
├──────────────────────────────────────┤
│      Infrastructure Layer (1)        │
│  Memory backends, Graph, Vector DB   │
└──────────────────────────────────────┘
```

---

## Component Diagram

### Hierarchical View

```
HoloLoom LSP Server (lsprotocol + pygls)
│
├─ Message Handling Layer
│  ├─ JSON-RPC Parser
│  ├─ Message Router
│  └─ Response Formatter
│
├─ Handler Layer (4 core + 10 future)
│  ├─ initialize
│  ├─ textDocument/completion
│  ├─ textDocument/hover
│  ├─ textDocument/definition
│  └─ workspace/symbol (future)
│
├─ State Management
│  ├─ DocumentManager
│  │  ├─ open_documents dict
│  │  ├─ apply_change()
│  │  └─ get_content()
│  │
│  └─ CacheLayer
│     ├─ completion_cache
│     ├─ hover_cache
│     └─ graph_cache
│
└─ Backend Integration
   ├─ Orchestrator wrapper
   │  ├─ recall()
   │  ├─ get_graph()
   │  └─ detect_logic()
   │
   └─ Memory Access
      ├─ VectorDB (Qdrant)
      ├─ KnowledgeGraph (Neo4j/NetworkX)
      └─ FileSystem (cache)
```

### Data Flow Between Components

```
Request from Editor
    ↓
[Message Router] - Identify which handler
    ↓
[LSP Handler] - Execute business logic
    ├─ Check cache (fast path)
    │  └─ Hit? Return cached result
    │
    └─ Cache miss
       └─ [Document Manager] - Get file context
          └─ [Orchestrator] - Query HoloLoom
             ├─ [Semantic Search]
             │  └─ VectorDB query
             │
             ├─ [Graph Lookup]
             │  └─ KnowledgeGraph query
             │
             └─ [Combine Results]
                └─ Rank by confidence
                   ↓
       [Format Response] - Convert to LSP format
          ↓
       [Cache Result] - Store for next time
          ↓
Response to Editor
```

---

## Data Flow Examples

### Example 1: Code Completion Request

```
1. USER TYPES "auth" → presses Ctrl+Space

2. EDITOR (VS Code)
   Sends to server:
   {
     "jsonrpc": "2.0",
     "id": 42,
     "method": "textDocument/completion",
     "params": {
       "textDocument": {"uri": "file:///home/user/code.py"},
       "position": {"line": 10, "character": 18},
       "context": {"triggerKind": 1}
     }
   }

3. LSP SERVER - message_router.handle(msg)
   ├─ Parse JSON-RPC message
   ├─ Identify method: textDocument/completion
   ├─ Extract params: uri, position, context
   └─ Route to handler: on_completion(params)

4. LSP HANDLER - on_completion()
   ├─ Extract context:
   │  ├─ uri = "file:///home/user/code.py"
   │  ├─ line = 10, column = 18
   │  └─ Get surrounding 10 lines from cache
   │
   ├─ Build query string:
   │  └─ "auth" + surrounding context
   │
   ├─ Check completion_cache
   │  └─ Hit? Return cached items
   │
   └─ Cache miss:
      ├─ Call orchestrator.recall(query, k=10)
      │
      └─ MEMORY SYSTEM PROCESSES:
         ├─ VectorDB query
         │  ├─ Embed "auth context"
         │  ├─ Search 384-d semantic space
         │  └─ Return top 10 similarities
         │
         ├─ BM25 keyword search
         │  ├─ Find exact/partial matches for "auth"
         │  └─ Return keyword results
         │
         └─ Combine results:
            ├─ Merge semantic + keyword
            ├─ Deduplicate
            ├─ Rank by confidence score
            └─ Return top 10 memories

5. FORMAT RESPONSE - handler.format_completions(memories)
   ├─ For each memory:
   │  ├─ Extract label (entity name)
   │  ├─ Extract kind (function/class/module)
   │  ├─ Get signature (from graph metadata)
   │  ├─ Get documentation (from graph)
   │  ├─ Set sort order (by confidence)
   │  └─ Create CompletionItem
   │
   └─ Return as LSP CompletionList:
      [
        {
          "label": "authenticate",
          "kind": 6,
          "detail": "function authenticate(username, password)",
          "sortText": "0_authenticate",
          "confidence": 0.95
        },
        {
          "label": "authenticateWithMFA",
          "kind": 6,
          "detail": "function authenticateWithMFA(...)",
          "sortText": "1_authenticateWithMFA",
          "confidence": 0.85
        },
        ...
      ]

6. CACHE RESULT - cache_layer.store_completions(uri, query, items)
   └─ Save for 5 minutes (TTL)

7. SEND RESPONSE - server.send_response(id=42, result=items)
   └─ Convert to JSON-RPC and send back to editor

8. EDITOR (VS Code) displays:
   ☑ authenticate
   ☐ authenticateWithMFA
   ☐ auth_module
```

**Total Latency**: ~95ms
- Message routing: 1ms
- Cache check: <1ms
- VectorDB query: 45ms
- BM25 search: 20ms
- Formatting: 15ms
- Network: ~15ms

---

### Example 2: Hover Information Request

```
1. USER HOVERS over "authenticate" function

2. EDITOR (Neovim)
   Sends:
   {
     "method": "textDocument/hover",
     "params": {
       "textDocument": {"uri": "file:///auth.py"},
       "position": {"line": 2, "character": 18}
     }
   }

3. LSP HANDLER - on_hover(params)
   ├─ Extract position: line=2, char=18
   ├─ Get document content from cache
   ├─ Find token at position: "authenticate"
   │
   ├─ Check hover_cache
   │  └─ Hit? Return cached hover
   │
   └─ Cache miss:
      ├─ Query knowledge graph: "find node 'authenticate'"
      │
      └─ KNOWLEDGE GRAPH LOOKUP:
         ├─ Search nodes by name: "authenticate"
         ├─ Get node properties:
         │  ├─ type: "function"
         │  ├─ file: "src/auth.ts"
         │  ├─ line: 2
         │  ├─ signature: "function authenticate(username, password)"
         │  └─ docstring: "Authenticate user..."
         │
         ├─ Get related entities:
         │  ├─ Incoming edges (CALLED_BY):
         │  │  ├─ handleLogin (handlers.ts:15)
         │  │  └─ apiLogin (api.ts:45)
         │  │
         │  ├─ Outgoing edges (CALLS):
         │  │  ├─ db.users.findOne
         │  │  └─ compare (password func)
         │  │
         │  └─ Related (semantic):
         │     ├─ logout
         │     ├─ verifyToken
         │     └─ authenticateWithMFA
         │
         └─ Compile semantic context:
            └─ "User authentication function"

4. SYNTHESIZE HOVER CONTENT
   ├─ Code block:
   │  └─ "function authenticate(username, password)"
   │
   ├─ Description:
   │  └─ "Authenticates a user with credentials..."
   │
   ├─ Location:
   │  └─ "src/auth.ts:2"
   │
   ├─ Related code:
   │  ├─ "Calls: db.users.findOne, compare"
   │  ├─ "Called by: handleLogin, apiLogin"
   │  └─ "Related: logout, verifyToken"
   │
   └─ Format as Markdown

5. FORMAT RESPONSE - Create LSP Hover object
   {
     "contents": {
       "kind": "markdown",
       "value": "```typescript\nfunction authenticate(username: string, password: string)\n```\n\n---\n\n...[full markdown documentation]..."
     },
     "range": {
       "start": {"line": 2, "character": 9},
       "end": {"line": 2, "character": 21}
     }
   }

6. CACHE RESULT
   └─ hover_cache.set(("file.py", (2, 18)), hover_info, ttl=10min)

7. SEND TO EDITOR
   └─ Neovim displays hover window with info

8. USER SEES:
   ┌─────────────────────────────────┐
   │ function authenticate(...)      │
   │                                 │
   │ Authenticates a user...         │
   │                                 │
   │ Location: src/auth.ts:2         │
   │                                 │
   │ Related: logout(), verifyToken()│
   └─────────────────────────────────┘
```

**Total Latency**: ~65ms
- Graph query: 30ms
- Markdown synthesis: 20ms
- Formatting: 10ms
- Network: ~5ms

---

### Example 3: Go to Definition Request

```
1. USER CLICKS "Go to Definition" or presses gd

2. EDITOR (Neovim)
   Sends:
   {
     "method": "textDocument/definition",
     "params": {
       "textDocument": {"uri": "file:///main.py"},
       "position": {"line": 12, "character": 24}
     }
   }

3. LSP HANDLER - on_definition(params)
   ├─ Get token at cursor: "authenticate"
   │
   ├─ Query KG: find_node("authenticate")
   │
   └─ KNOWLEDGE GRAPH LOOKUP (CACHED):
      ├─ Search index: "authenticate"
      ├─ Get node ID from index
      └─ Retrieve properties:
         ├─ file: "src/auth.ts"
         ├─ line: 2
         ├─ column: 18
         └─ end_line: 10

4. FORMAT RESPONSE - Create LSP Location
   {
     "uri": "file:///home/user/project/src/auth.ts",
     "range": {
       "start": {"line": 1, "character": 18},
       "end": {"line": 1, "character": 30}
     }
   }

5. SEND RESPONSE

6. EDITOR NAVIGATES
   └─ Opens auth.ts and jumps to line 2
```

**Total Latency**: ~35ms (fastest operation)
- Graph index lookup: 10ms
- Format response: 5ms
- Network: ~20ms

---

## Protocol Flow

### Initialization Sequence

```
1. EDITOR starts LSP server
   └─ Server: pygls.start_tcp/start_io()

2. SERVER initializes
   └─ Creates:
      ├─ LanguageServer instance
      ├─ DocumentManager
      ├─ CacheLayer
      ├─ Configuration
      └─ Logging

3. CLIENT connects (editor)
   └─ Sends initialize request

4. SERVER handles initialize()
   ├─ Extract workspace folders
   ├─ Load HoloLoom config
   ├─ Initialize orchestrator (lazy)
   ├─ Declare capabilities (initialize response)
   │  └─ What features server supports
   └─ Wait for initialized() notification

5. CLIENT receives capabilities
   └─ Knows what LSP features to offer

6. CLIENT sends initialized notification
   └─ Signal server is ready

7. SERVER handles initialized()
   └─ Maybe load workspace (future)

8. SERVER ready, CLIENT ready
   └─ Can now send feature requests
```

### Request-Response Cycle

```
CLIENT sends request:
{
  "jsonrpc": "2.0",
  "id": <unique_id>,
  "method": "<method_name>",
  "params": <object>
}
    ↓
SERVER receives via stdin/TCP
    ↓
[Message Router]
    ├─ Parse JSON
    ├─ Validate JSON-RPC format
    └─ Route to handler
       └─ @server.feature("<method_name>")
          async def handler(params)
    ↓
[Handler Logic]
    ├─ Process request
    └─ Return result or error
    ↓
SERVER sends response:
{
  "jsonrpc": "2.0",
  "id": <same_id>,
  "result": <response_object>,
  "error": null  // or error object
}
    ↓
CLIENT receives response
    ↓
CLIENT uses result (e.g., display completion)
```

### File Lifecycle

```
USER opens file:
    ↓
CLIENT sends didOpen notification
    ├─ Contains: uri, content, version
    └─ Server receives:
       ├─ Document Manager caches content
       ├─ Optionally ingest to knowledge graph
       └─ Trigger initial analysis (future)

USER types:
    ↓
CLIENT sends didChange notification
    ├─ Incremental or full content
    └─ Server receives:
       ├─ Apply changes to cache
       ├─ Mark document dirty
       └─ Trigger async diagnostics (future)

USER saves:
    ↓
CLIENT sends didSave notification
    └─ Server receives (future: persist changes)

USER closes file:
    ↓
CLIENT sends didClose notification
    └─ Server:
       ├─ Remove from document cache
       ├─ Flush any pending updates
       └─ Clear diagnostics (future)
```

---

## Module Organization

### File Structure

```
HoloLoom/lsp/
├── __init__.py                 # Package exports
│
├── server.py                   # Main LSP server
│   ├─ LanguageServer class
│   ├─ @server.feature handlers
│   ├─ Capability declaration
│   └─ Logging setup
│
├── handlers.py                 # LSP request handlers
│   ├─ on_initialize()
│   ├─ on_completion()
│   ├─ on_hover()
│   ├─ on_definition()
│   └─ on_text_document_did_*()
│
├── document_manager.py         # Document tracking
│   ├─ DocumentManager class
│   ├─ Document class
│   ├─ apply_change()
│   └─ get_context()
│
├── cache_layer.py              # Caching (future)
│   ├─ CacheLayer class
│   ├─ completion_cache
│   ├─ hover_cache
│   └─ TTL management
│
├── formatter.py                # LSP response formatting
│   ├─ format_completion_list()
│   ├─ format_hover()
│   ├─ format_location()
│   └─ format_diagnostic()
│
└── tests/                      # Test suite
    ├─ __init__.py
    ├─ test_handlers.py
    ├─ test_protocol.py
    ├─ test_performance.py
    └─ test_integration.py
```

### Key Classes

**LanguageServer (pygls)**:
- Main LSP server class
- Lifecycle: start, handle messages, shutdown
- Features: register handlers via @server.feature()

**DocumentManager**:
- Tracks open documents in editor
- Maintains cache of file contents
- Handles incremental updates

**CacheLayer**:
- Completion cache (5min TTL)
- Hover cache (10min TTL)
- Graph cache (30min TTL)

**Formatter**:
- Convert internal results to LSP types
- Handle type conversions (Python → LSP JSON)

---

## Integration Points

### With HoloLoom Orchestrator

```
LSP Handler
    ↓
├─ completion → orchestrator.recall(query, k=10)
│              └─ Returns: List[Memory]
│
├─ hover     → orchestrator.get_graph()
│              └─ Returns: KnowledgeGraphData
│
├─ definition→ memory.get_graph().find_node(name)
│              └─ Returns: EntityLocation
│
└─ references→ memory.get_graph().find_edges(CALLED_BY)
               └─ Returns: List[Location]
```

### With Configuration System

```
LSP Server startup:
    ↓
config = Config.fast()  # or Config.fused()
    ├─ Sets performance targets
    ├─ Enables/disables features
    ├─ Configures logging
    └─ Sets memory limits
    ↓
orchestrator = await WeavingOrchestrator(config)
    └─ Applies config to all modules
```

### With Memory Backends

```
orchestrator.recall()
    ↓
├─ Check cache (in-memory)
├─ Query VectorDB (Qdrant)
│  └─ Semantic search
├─ Query KG (Neo4j or NetworkX)
│  └─ Entity relationships
└─ Combine + rank results
```

---

## Performance Architecture

### Latency Budget

```
LSP Request → Editor (typical: <150ms)

Breakdown:
├─ Message parsing:       1ms
├─ Handler routing:       1ms
├─ Cache check:          <1ms
├─ Semantic query:       45ms (VectorDB)
├─ KG query:            20ms (if needed)
├─ Result formatting:    15ms
├─ Caching result:        5ms
├─ Network I/O:         20ms
└─ TOTAL:              ~107ms ✓ (under 150ms target)

Hot path (cache hit):    ~5ms
```

### Caching Strategy

```
3-Tier Cache:

Tier 1: Query-level cache
├─ Key: (uri, query_string)
├─ TTL: 5 minutes
├─ Size: 1000 entries
└─ Hit rate: 70%

Tier 2: Entity-level cache
├─ Key: entity_id
├─ TTL: 30 minutes
├─ Size: 5000 entries
└─ Hit rate: 80%

Tier 3: Workspace-level cache
├─ Key: workspace_uri
├─ TTL: 1 hour
├─ Size: Unlimited
└─ Hit rate: 90%
```

### Memory Management

```
Per-open-file:
├─ Document cache:  ~100KB (typical file)
├─ AST cache:       ~50KB
└─ Completion items:~10KB
└─ Total: ~160KB per file

Typical workload (10 files):
├─ Document cache:  ~1.6MB
├─ Query cache:     ~50MB (1000 entries)
├─ Graph cache:     ~100MB (hot entities)
└─ Total: ~150MB

Optimization:
├─ LRU eviction (oldest first)
├─ Priority eviction (least frequently used)
└─ Manual cache clear (workspace change)
```

### Async Architecture

```
Server uses asyncio for non-blocking I/O:

receive_request()
    ├─ Parse JSON (blocking, <1ms)
    │
    └─ Route to async handler
        ├─ Handler starts task
        │  └─ Returns immediately
        │
        └─ Server can receive next request
           while processing previous

Benefits:
├─ Multiple simultaneous requests
├─ No thread complexity (GIL not an issue)
├─ Scales to 50+ concurrent clients
└─ Clean exception handling
```

---

## Deployment Architecture

### Single-Machine Deployment

```
One machine:
├─ HoloLoom LSP Server (Python)
│  └─ pid: 1234
│
├─ HoloLoom backends (optional)
│  ├─ Neo4j (KG) - docker
│  └─ Qdrant (VectorDB) - docker
│
└─ Editors (client-side)
   ├─ VS Code
   ├─ Neovim
   └─ Emacs

Communication:
├─ Server → Editors: JSON-RPC over stdio/TCP
└─ Server → Backends: localhost:6333/7687 (docker)
```

### Multi-Machine Deployment (Future)

```
Developer Machine:
├─ Editors (VS Code, Neovim, etc.)
└─ LSP Client (in editor)

LSP Server Machine:
├─ HoloLoom LSP Server (TCP port 8080)
└─ Orchestrator (async)

Backend Machine:
├─ Neo4j (KG)
├─ Qdrant (VectorDB)
└─ File cache (Redis optional)

Communication:
├─ Editors → Server: JSON-RPC over TCP/TLS
├─ Server → Backends: gRPC or HTTP
└─ All machines: Same network or VPN
```

### Docker Deployment

```
docker-compose.yml:
├─ lsp-server
│  ├─ Image: python:3.11
│  ├─ Port: 8080
│  └─ Mounts: /workspace
│
├─ neo4j
│  ├─ Port: 7687
│  └─ Volume: /var/lib/neo4j
│
└─ qdrant
   ├─ Port: 6333
   └─ Volume: /var/lib/qdrant

Usage:
docker-compose up -d
# Server ready on localhost:8080
```

---

## Sequence Diagrams

### Completion Request Sequence

```
Editor            LSP Server         HoloLoom       Memory
   │                  │                 │              │
   ├─ completion ────→│                 │              │
   │                  │                 │              │
   │            Cache │                 │              │
   │            Check │                 │              │
   │                  │                 │              │
   │                  ├─ recall()──────→│              │
   │                  │                 │              │
   │                  │                 ├─ query ─────→│
   │                  │                 │              │
   │                  │                 │   results ←──│
   │                  │                 │              │
   │                  │        format ←─│              │
   │                  │                 │              │
   │                  ├─ format ─────────────┐         │
   │                  │                      ↓         │
   │← completions ────│                                │
   │                  │                                │
```

### Hover Information Sequence

```
Editor            LSP Server      HoloLoom KG      Memory
   │                  │              │               │
   ├─ hover ─────────→│              │               │
   │                  │              │               │
   │                  ├─ find_node()→│               │
   │                  │              │               │
   │                  │      entity ←│               │
   │                  │              │               │
   │                  ├─ get_relations()─────────────→│
   │                  │              │               │
   │                  │              │     rels ←───│
   │                  │              │               │
   │                  ├─ format to MD ─────┐         │
   │                  │                    ↓         │
   │← hover ─────────│                                │
   │                  │                                │
```

---

## Future Architecture Enhancements

### Phase 4.1: Advanced Features

```
New handlers to add:
├─ textDocument/references
│  └─ Query: KG.find_edges(CALLED_BY)
│
├─ textDocument/documentSymbol
│  └─ Query: parse_ast() + KG lookup
│
├─ textDocument/formatting
│  └─ Call: black/prettier/...
│
├─ textDocument/rename
│  └─ Replace all occurrences in workspace
│
├─ textDocument/signatureHelp
│  └─ Extract parameters from signature
│
└─ textDocument/semanticTokens
   └─ Parse tokens + add type info
```

### Phase 4.2: Streaming Responses

```
For long-running operations:

completion(params)
    ├─ Start: send partial results
    │
    ├─ Stream 1: top 5 results (50ms)
    ├─ Stream 2: next 5 results (100ms)
    ├─ Stream 3: remaining (150ms)
    │
    └─ End: all results received

Benefits:
├─ User sees results immediately
├─ Can start typing while more results come
└─ Perceived responsiveness > actual latency
```

### Phase 4.3: Incremental Workspace Analysis

```
Instead of analyzing all files on startup:

Initialize:
└─ Lightweight analysis (5s)

On didOpen:
└─ Analyze new file (100ms)

On didChange:
└─ Analyze changed regions (50ms)

Background:
└─ Analyze remaining files (low priority)

Benefits:
├─ Fast startup
├─ Responsive editing
└─ Complete knowledge graph eventually
```

---

## Conclusion

The LSP architecture provides a clean separation between:
1. **Protocol** (JSON-RPC, standard)
2. **Application** (Handlers, LSP-specific logic)
3. **Domain** (Orchestrator, HoloLoom-specific)
4. **Infrastructure** (Memory backends)

This layering enables:
- **Easy testing**: Mock any layer
- **Performance optimization**: Profile each layer
- **Feature addition**: Add handlers without touching core
- **Integration**: Plug in new memory backends

The result is a maintainable, scalable, universal code intelligence system.

---

**Last Updated**: November 2025
**For detailed information**, see:
- `PHASE_4_LSP_SERVER_SUMMARY.md` - Complete implementation guide
- `LSP_PROTOCOL_SPEC.md` - Full LSP endpoint specifications
- `LSP_QUICK_START.md` - 5-minute setup guides
