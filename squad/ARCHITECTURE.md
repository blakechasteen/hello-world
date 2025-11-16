# Squad Architecture

**Technical architecture and design decisions**

---

## Overview

Squad is a VS Code extension that provides agentic AI assistance powered by HoloLoom's multi-step reasoning engine. The architecture follows a client-server pattern with clear separation of concerns.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VS Code Process                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Extension Host Process                     │ │
│  │                                                         │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │ │
│  │  │  extension  │  │  HoloLoom    │  │   Context    │ │ │
│  │  │    .ts      │──│   Bridge     │──│   Provider   │ │ │
│  │  └─────────────┘  └──────┬───────┘  └──────────────┘ │ │
│  │         │                 │                            │ │
│  │         │                 │ HTTP                       │ │
│  │         │                 │ (localhost:8000)           │ │
│  │         │                 ▼                            │ │
│  │         │          ┌─────────────┐                     │ │
│  │         │          │   axios     │                     │ │
│  │         │          │   client    │                     │ │
│  │         │          └─────────────┘                     │ │
│  │         │                                               │ │
│  │         ▼                                               │ │
│  │  ┌─────────────┐                                       │ │
│  │  │  Webview    │                                       │ │
│  │  │  Process    │  (Agent Panel - HTML/CSS/JS)          │ │
│  │  │             │  - Reasoning steps                    │ │
│  │  │             │  - Confidence scores                  │ │
│  │  │             │  - Verification results               │ │
│  │  └─────────────┘                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ HTTP POST/GET
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Python Process                            │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  FastAPI Server                          ││
│  │                                                          ││
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐  ││
│  │  │  /query    │  │   /chat    │  │    /stats       │  ││
│  │  │  endpoint  │  │  endpoint  │  │   endpoint      │  ││
│  │  └─────┬──────┘  └─────┬──────┘  └─────────────────┘  ││
│  │        │                │                               ││
│  │        └────────┬───────┘                               ││
│  │                 │                                       ││
│  │        ┌────────▼──────────┐                           ││
│  │        │  Request Handler  │                           ││
│  │        │  - Validation     │                           ││
│  │        │  - Mode selection │                           ││
│  │        │  - Error handling │                           ││
│  │        └────────┬──────────┘                           ││
│  │                 │                                       ││
│  └─────────────────┼───────────────────────────────────────┘│
│                    │                                        │
│  ┌─────────────────▼────────────────────────────────────┐  │
│  │           WeavingOrchestrator                         │  │
│  │           (HoloLoom Core)                             │  │
│  │                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │
│  │  │   Pattern    │  │   Feature    │  │  Decision │  │  │
│  │  │  Selection   │──│  Extraction  │──│   Engine  │  │  │
│  │  └──────────────┘  └──────────────┘  └───────────┘  │  │
│  │                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │
│  │  │   Memory     │  │   Safety     │  │  Response │  │  │
│  │  │   Retrieval  │  │  Guardrails  │  │ Generator │  │  │
│  │  └──────────────┘  └──────────────┘  └───────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### TypeScript Extension

#### `extension.ts` (Main Entry Point)

**Responsibilities:**
- Extension activation/deactivation
- Command registration
- Status bar management
- Health check monitoring

**Key Functions:**
```typescript
activate(context)          // Extension entry point
registerCommands(context)  // Register all commands
executeQuery(...)          // Execute query with progress
updateServerStatus()       // Check server health
```

**Lifecycle:**
1. VS Code starts
2. `activate()` called
3. Commands registered
4. Status bar created
5. Health check started
6. Extension ready

---

#### `HoloLoomBridge.ts` (HTTP Client)

**Responsibilities:**
- HTTP communication with server
- Request/response serialization
- Error handling
- Timeout management

**Key Methods:**
```typescript
healthCheck()                    // GET /health
query(text, context, mode, ...)  // POST /query
chat(message, context)           // POST /chat
getStats()                       // GET /stats
```

**Communication Protocol:**
- **Transport:** HTTP/1.1
- **Format:** JSON
- **Timeout:** 60 seconds
- **Retry:** None (fail fast)

---

#### `AgentPanel.ts` (Webview UI)

**Responsibilities:**
- Render reasoning steps
- Display confidence scores
- Show verification results
- Real-time updates

**Architecture:**
```
┌──────────────────────────────┐
│     Webview (Isolated)       │
│                              │
│  ┌────────────────────────┐  │
│  │  HTML/CSS/JavaScript   │  │
│  │  (No external deps)    │  │
│  └────────────────────────┘  │
│             ▲                │
│             │ postMessage    │
│             │                │
└─────────────┼────────────────┘
              │
    ┌─────────▼─────────┐
    │  Extension Code   │
    │  (AgentPanel.ts)  │
    └───────────────────┘
```

**Data Flow:**
1. Extension calls `displayResult(result)`
2. WebView receives message via `postMessage`
3. JavaScript updates DOM
4. User sees updated UI

---

#### `CodeContextProvider.ts` (Context Extraction)

**Responsibilities:**
- Extract current file content
- Get selected text
- Collect diagnostics
- Identify language

**Context Structure:**
```typescript
interface CodeContext {
    currentFile?: string;     // Full file text
    fileName?: string;        // File path
    languageId?: string;      // e.g., "typescript"
    selection?: string;       // Selected text
    diagnostics?: Diagnostic[]; // Errors/warnings
    workspace?: string;       // Workspace path
}
```

---

### Python Server

#### FastAPI Application

**Endpoints:**

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/health` | GET | Health check | <5ms |
| `/query` | POST | Main query processing | 150-900ms |
| `/chat` | POST | Simple chat | ~150ms |
| `/stats` | GET | Server statistics | <5ms |

**Middleware:**
- CORS (for development)
- JSON error handling
- Request logging

**Lifecycle:**
```python
startup():
    - Initialize config
    - Create orchestrator
    - Load memory shards
    - Ready!

handle_query(request):
    - Validate request
    - Create Query object
    - Call orchestrator.weave()
    - Format response
    - Return

shutdown():
    - Close orchestrator
    - Cleanup resources
```

---

#### HoloLoom Integration

**Weaving Cycle:**
```
Query → WeavingOrchestrator
   ↓
1. Pattern Selection (BARE/FAST/FUSED)
   ↓
2. Chrono Trigger (temporal window)
   ↓
3. Yarn Graph (memory threads)
   ↓
4. Resonance Shed (feature extraction)
   ↓
5. Warp Space (tensor operations)
   ↓
6. Convergence Engine (decision)
   ↓
7. Spacetime (result with trace)
   ↓
Response ← Extract & Format
```

**Configuration:**
```python
config = Config.fast()  # FAST mode
config.enable_alignment = True  # Safety checks
```

---

## Data Flow

### Query Processing Flow

```
┌─────────────┐
│    User     │
│  Types      │
│  Ctrl+Q     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  VS Code UI     │
│  - Input box    │
│  - Validation   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  extension.ts   │
│  - Get context  │
│  - Progress bar │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ HoloLoomBridge  │
│ - Serialize     │
│ - HTTP POST     │
└──────┬──────────┘
       │
       │ HTTP
       ▼
┌─────────────────┐
│  FastAPI Server │
│  - Deserialize  │
│  - Validate     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Orchestrator   │
│  - Process      │
│  - Reason       │
│  - Generate     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Response       │
│  - Format       │
│  - Add metadata │
└──────┬──────────┘
       │
       │ JSON
       ▼
┌─────────────────┐
│ HoloLoomBridge  │
│ - Deserialize   │
│ - Validate      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Agent Panel    │
│  - Display      │
│  - Show steps   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│     User        │
│  Sees result    │
└─────────────────┘
```

---

## Security

### Threat Model

**Threats:**
1. Malicious code injection via queries
2. Unauthorized access to server
3. Sensitive data in context
4. XSS in webview

**Mitigations:**

1. **Input Validation:**
   ```python
   # All inputs validated with Pydantic
   class QueryRequest(BaseModel):
       text: str  # Required, auto-validated
       mode: str = "verify"  # Enum validation
   ```

2. **Server-side Authorization:**
   - localhost-only by default
   - CORS restricted
   - No authentication (assumes trusted network)

3. **Context Sanitization:**
   ```typescript
   // Remove sensitive patterns
   const sanitized = context.replace(/password.*=.*/gi, '[REDACTED]');
   ```

4. **Webview CSP:**
   ```typescript
   // Content Security Policy in webview
   meta http-equiv="Content-Security-Policy"
   content="default-src 'none'; style-src 'unsafe-inline';"
   ```

---

## Performance

### Optimization Strategies

#### Client-Side

1. **Debouncing:**
   ```typescript
   // Don't spam server with requests
   const debouncedQuery = debounce(executeQuery, 500);
   ```

2. **Caching:**
   ```typescript
   // Cache recent results
   const cache = new LRU<string, AgenticResult>(100);
   ```

3. **Progressive Loading:**
   ```typescript
   // Show partial results immediately
   showPartialResult(step);  // Each step
   ```

#### Server-Side

1. **Async Processing:**
   ```python
   # Non-blocking I/O
   async def handle_query(request):
       result = await orchestrator.weave(query)
   ```

2. **Connection Pooling:**
   ```python
   # Reuse connections
   async with orchestrator:  # Context manager
       result = await orchestrator.weave(query)
   ```

3. **Response Streaming:**
   ```python
   # Stream results (future enhancement)
   async def stream_response():
       for step in reasoning_steps:
           yield step
   ```

### Performance Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| Health check | <10ms | ~5ms |
| Query DIRECT | <200ms | ~150ms |
| Query VERIFY | <700ms | ~600ms |
| Query RESEARCH | <1000ms | ~900ms |
| Extension activation | <500ms | ~200ms |

---

## Error Handling

### Error Propagation

```
┌──────────────┐
│  User Error  │  (e.g., bad input)
└──────┬───────┘
       │ ValidationError
       ▼
┌──────────────┐
│  Extension   │  → Show warning message
└──────────────┘


┌──────────────┐
│Server Error  │  (e.g., 500)
└──────┬───────┘
       │ HTTP 500
       ▼
┌──────────────┐
│    Bridge    │  → Parse error message
└──────┬───────┘
       │ Error object
       ▼
┌──────────────┐
│  Extension   │  → Show error + action buttons
└──────────────┘
       │
       ▼
┌──────────────┐
│     User     │  → Click "Open Terminal" or "Settings"
└──────────────┘
```

### Error Recovery

1. **Connection Errors:**
   - Offer to start server
   - Show terminal command
   - Link to settings

2. **Server Errors:**
   - Display detailed message
   - Log to console
   - Suggest troubleshooting steps

3. **Client Errors:**
   - Validate inputs early
   - Show helpful messages
   - Prevent invalid requests

---

## Testing Strategy

### Test Pyramid

```
         ┌─────────────┐
         │   Manual    │  (E2E testing in VS Code)
         │  Testing    │
         └─────────────┘
              ▲
         ┌─────────────┐
         │ Integration │  (test_squad.py)
         │    Tests    │
         └─────────────┘
              ▲
         ┌─────────────┐
         │    Unit     │  (Future)
         │   Tests     │
         └─────────────┘
```

### Current Coverage

**Integration Tests (5):**
- ✅ Health check
- ✅ Query DIRECT
- ✅ Query VERIFY
- ✅ Chat
- ✅ Stats

**Manual Tests:**
- Command execution
- UI responsiveness
- Error handling
- Edge cases

**Future:**
- Unit tests for each module
- UI automation tests
- Performance benchmarks
- Load testing

---

## Deployment

### Development Deployment

```bash
# Start server
PYTHONPATH=/home/user/hello-world python server.py

# Launch extension
# Press F5 in VS Code
```

### Production Deployment

```bash
# Build VSIX
vsce package

# Install extension
code --install-extension squad-0.1.0.vsix

# Configure server URL in settings
# (if not localhost)
```

---

## Future Enhancements

### Planned Features

1. **Streaming Responses:**
   - Server-sent events
   - Real-time step updates
   - Cancellable queries

2. **Offline Mode:**
   - Local model support
   - Cached responses
   - Degraded functionality

3. **Multi-workspace:**
   - Shared server instance
   - Workspace-specific memory
   - Cross-workspace queries

4. **Advanced UI:**
   - Inline code suggestions
   - Diff view for refactorings
   - Code lens annotations
   - Diagnostic provider

---

## Design Decisions

### Why FastAPI?

**Pros:**
- Async by default
- Automatic validation (Pydantic)
- Auto-generated docs
- Fast performance

**Alternatives considered:**
- Flask (too synchronous)
- Django (too heavy)
- Direct socket (too low-level)

---

### Why HTTP vs WebSocket?

**HTTP chosen because:**
- Simple request/response pattern
- No connection management
- Easy to debug with curl
- Good enough performance

**WebSocket future enhancement:**
- Real-time updates
- Bidirectional streaming
- Lower latency

---

### Why Webview for Agent Panel?

**Pros:**
- Full HTML/CSS control
- Rich visualizations
- No external dependencies
- Security isolation

**Cons:**
- More complex messaging
- Resource usage

**Alternatives considered:**
- TreeView (too limited)
- Output channel (no formatting)
- Custom editor (overkill)

---

## Monitoring & Observability

### Logging

**Client-side:**
```typescript
console.log('[Squad]', message);  // Development
// Extension host console
```

**Server-side:**
```python
logger.info(f"Query: {query[:100]}...")
logger.error("Error", exc_info=True)
```

### Metrics (Future)

- Query latency histogram
- Error rate
- Cache hit rate
- Resource usage

---

## References

- **VS Code Extension API:** https://code.visualstudio.com/api
- **FastAPI:** https://fastapi.tiangolo.com
- **HoloLoom:** `/home/user/hello-world/CLAUDE.md`
- **Pydantic:** https://docs.pydantic.dev

---

**Last Updated:** November 16, 2025
