# LSP Architecture Guide

**Published: 2025-11-17**
**Last Updated: 2025-11-17**

Comprehensive documentation of the Promptly LSP (Language Server Protocol) architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Responsibilities](#component-responsibilities)
4. [Message Flow](#message-flow)
5. [LSP Methods](#lsp-methods)
6. [Error Handling](#error-handling)
7. [Extension Points](#extension-points)
8. [Testing Strategy](#testing-strategy)

---

## Overview

### What is LSP?

**LSP (Language Server Protocol)** is a standardized protocol for communication between editors and language servers. It's used by VS Code, vim, Neovim, Emacs, and many other editors.

### Why LSP?

**Benefits over HTTP API:**
- ✅ **Persistent connection** - One connection for entire session
- ✅ **Bidirectional** - Server can send notifications to client
- ✅ **Auto-managed** - VS Code lifecycle management
- ✅ **Auto-reconnect** - Built-in recovery
- ✅ **Type-safe** - Enforced message schema
- ✅ **Standard** - Works with any LSP-compatible client
- ✅ **Faster** - Binary protocol, less overhead

### Promptly LSP Features

Promptly implements a **custom LSP server** that provides:
- Memory operations (remember, recall)
- Knowledge graph queries
- Workspace indexing
- CodeLens suggestions
- Inline completion hints

---

## Architecture Diagram

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VS Code Editor                             │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │            Promptly Extension (TypeScript)                 │ │
│  │                                                              │ │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐ │ │
│  │  │ Chat Command │  │   Sidebar   │  │    CodeLens      │ │ │
│  │  │  Handlers    │  │   Provider  │  │    Provider      │ │ │
│  │  └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘ │ │
│  │         │                 │                   │            │ │
│  │         └─────────────────┼───────────────────┘            │ │
│  │                           │                                 │ │
│  │                   ┌───────▼────────┐                        │ │
│  │                   │   LSP Client   │                        │ │
│  │                   │ (vscode-lc)    │                        │ │
│  │                   └───────┬────────┘                        │ │
│  └────────────────────────────┼──────────────────────────────┘ │
│                               │                                  │
│                        LSP Protocol (JSON-RPC)                  │
│                        (stdio/TCP, bidirectional)               │
│                               │                                  │
│  ┌────────────────────────────▼──────────────────────────────┐ │
│  │        LSP Server Process (Python)                        │ │
│  │      (Auto-started & managed by VS Code)                  │ │
│  │                                                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ LSP Protocol │  │   HoloLoom   │  │   Memory Index   │ │ │
│  │  │   Handler    │  │   Integration│  │   & Cache        │ │ │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │ │
│  │         │                 │                   │            │ │
│  │         └─────────────────┼───────────────────┘            │ │
│  │                           │                                 │ │
│  │                   ┌───────▼────────┐                        │ │
│  │                   │  HoloLoom      │                        │ │
│  │                   │  Memory System │                        │ │
│  │                   └────────────────┘                        │ │
│  │                           │                                 │ │
│  │              ┌────────────┴────────────┐                    │ │
│  │              │                         │                    │ │
│  │      ┌───────▼───────┐        ┌───────▼──────┐             │ │
│  │      │  Yarn Graph   │        │  Embeddings  │             │ │
│  │      │  (NetworkX)   │        │  (Vectors)   │             │ │
│  │      └───────────────┘        └──────────────┘             │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Message Flow (Request-Response)

```
VS Code Extension                LSP Server
      │                                 │
      │─── hololoom/remember ──────────>│
      │    { content: "..." }           │
      │                                 │ HoloLoom.experience()
      │                                 │ Update embeddings
      │                                 │ Update graph
      │<──── Response ─────────────────│
      │    { success: true }            │
      │                                 │
      └─────────────────────────────────┘
```

### Bidirectional Communication (Notifications)

```
VS Code Extension                LSP Server
      │                                 │
      │<────── hololoom/status ───────│
      │        { status: "ready" }      │
      │                                 │
      │                                 │
      │<────── hololoom/indexing ─────│
      │        { files: 150, done: 45 } │
      │                                 │
      └─────────────────────────────────┘
```

---

## Component Responsibilities

### 1. VS Code Extension (TypeScript)

**Files:** `src/extension.ts`, `src/chatView.ts`, `src/commands/*`, `src/views/*`, `src/providers/*`

**Responsibilities:**
- Initialize LSP client on startup
- Register VS Code commands
- Register language features (CodeLens, hover, etc.)
- Send requests to LSP server
- Handle responses and update UI
- Display status bar indicators
- Manage sidebar webview

**Key Methods:**
```typescript
// Initialize LSP client
const client = new LanguageClient(...);
await client.start();

// Send request
const result = await client.sendRequest('hololoom/remember', {
    content: 'note',
    context: {...}
});

// Listen for notifications
client.onNotification('hololoom/indexing', (params) => {
    updateProgressBar(params.progress);
});
```

### 2. LSP Client (vscode-languageclient)

**Library:** `vscode-languageclient` v9.0.0+

**Responsibilities:**
- Manage LSP protocol communication
- Serialize/deserialize JSON-RPC messages
- Auto-reconnect on connection loss
- Manage language server process lifecycle
- Provide type-safe request/response handling

**Connection Types:**
- **Stdio:** Default (language server on stdin/stdout)
- **TCP:** For remote servers

### 3. LSP Server (Python)

**Implementation:** Custom Python LSP server using `pylsp-jsonrpc` or similar

**Responsibilities:**
- Parse incoming LSP messages
- Route to appropriate handler
- Integrate with HoloLoom
- Execute memory operations
- Send responses back to client
- Send proactive notifications

**Key Handler Methods:**
```python
class HoloLoomLSPServer:
    async def remember(self, params):
        # Store in HoloLoom

    async def recall(self, params):
        # Query HoloLoom

    async def query(self, params):
        # Advanced reasoning
```

### 4. HoloLoom Integration

**Files:** `/path/to/HoloLoom/` (external package)

**Responsibilities:**
- Manage memory graph
- Manage embeddings
- Execute queries
- Return results to LSP server

**Integration Points:**
```python
from HoloLoom import HoloLoom

# In LSP server
hololoom = HoloLoom()

# Use in handlers
memory = await hololoom.experience(content)
results = await hololoom.recall(query)
```

---

## Message Flow

### 1. Simple Request-Response

#### Remember Operation

**Sequence:**
```
1. User types: /remember "Learning about LSP"
2. Extension sends: POST hololoom/remember with content
3. LSP Server receives message
4. Server calls: HoloLoom.experience(content)
5. HoloLoom returns: memory_id and embeddings
6. Server sends: Response { success: true }
7. Extension receives: Response in callback
8. UI updates: Shows "✅ Saved"
```

**JSON-RPC Message Format:**

Request:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "hololoom/remember",
  "params": {
    "content": "Learning about LSP",
    "context": {
      "workspace": "my-project",
      "file": "main.ts"
    }
  }
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "success": true,
    "memory_id": "uuid-12345",
    "confidence": 0.95
  }
}
```

### 2. Streaming Request-Response

#### Workspace Index Operation

**Sequence:**
```
1. User: HoloLoom: Index Workspace (command)
2. Extension sends: hololoom/indexWorkspace request
3. Server receives request, starts async indexing
4. Server streams progress notifications:
   - { files: 10, done: 2 }
   - { files: 10, done: 5 }
   - { files: 10, done: 10 }
5. Server sends final response: { success: true, total: 10 }
6. Extension updates progress bar in realtime
```

**Notification Messages (server → client):**
```json
{
  "jsonrpc": "2.0",
  "method": "hololoom/indexing",
  "params": {
    "files_total": 10,
    "files_done": 5,
    "current_file": "src/main.ts"
  }
}
```

### 3. CodeLens Flow

**Sequence:**
```
1. User opens file with // NOTE: comments
2. VS Code calls CodeLens provider
3. Provider: Iterate over comments in document
4. For each comment: Send hololoom/suggestCodeLens
5. LSP server queries knowledge graph
6. Server returns: Related memories + metadata
7. Provider creates CodeLens objects
8. VS Code displays CodeLens inline
9. User clicks CodeLens
10. Click handler sends hololoom/getCodeLens details
```

**CodeLens Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "hololoom/suggestCodeLens",
  "params": {
    "text": "NOTE: Using Thompson Sampling",
    "file": "policy.ts",
    "line": 42
  }
}
```

**CodeLens Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "suggestions": [
      {
        "text": "3 related notes",
        "confidence": 0.87,
        "command": "promptly.showSuggestions"
      }
    ]
  }
}
```

---

## LSP Methods

Complete reference of all LSP methods provided by Promptly server.

### Memory Operations

#### `hololoom/remember`

**Description:** Store content in HoloLoom memory

**Request:**
```typescript
{
  content: string;           // Text to remember
  context?: {
    workspace?: string;      // Workspace name
    file?: string;          // Current file path
    timestamp?: string;     // ISO timestamp
    tags?: string[];        // Optional tags
  };
}
```

**Response:**
```typescript
{
  success: boolean;         // true if saved
  memory_id?: string;       // Unique ID of saved memory
  confidence?: number;      // Confidence score 0-1
  error?: string;          // Error message if failed
}
```

#### `hololoom/recall`

**Description:** Query memories by semantic similarity

**Request:**
```typescript
{
  query: string;            // Search query
  k?: number;              // Top K results (default: 5)
  min_confidence?: number; // Minimum confidence 0-1
}
```

**Response:**
```typescript
{
  memories: Array<{
    id: string;            // Memory ID
    content: string;       // Memory text
    confidence: number;    // 0-1 score
    timestamp?: string;    // ISO timestamp
    source?: string;       // Where memory came from
  }>;
}
```

#### `hololoom/query`

**Description:** Advanced reasoning with multi-step queries

**Request:**
```typescript
{
  text: string;            // Query text
  mode?: "direct" | "verify" | "research" | "plan_execute";
  max_steps?: number;      // Max reasoning steps
  context?: string[];      // Additional context
}
```

**Response:**
```typescript
{
  response: string;        // Generated answer
  confidence?: number;     // Answer confidence 0-1
  steps_taken?: number;    // Number of reasoning steps
  sources?: string[];      // Source memories used
}
```

### Workspace Operations

#### `hololoom/indexWorkspace`

**Description:** Index entire workspace for code understanding

**Request:**
```typescript
{
  workspace_path: string;  // Root path to index
  languages?: string[];    // e.g. ["python", "typescript"]
  exclude_patterns?: string[]; // e.g. ["**/node_modules/**"]
}
```

**Response:**
```typescript
{
  success: boolean;
  files_indexed?: number;
  entities_created?: number;
  error?: string;
}
```

**Notifications (during indexing):**
```typescript
// Notification: hololoom/indexing
{
  files_total: number;
  files_done: number;
  current_file?: string;
  status?: "indexing" | "complete" | "failed";
}
```

#### `hololoom/incrementalIndex`

**Description:** Index specific files (for file watcher)

**Request:**
```typescript
{
  file_paths: string[];    // Files to index
  workspace_path: string;  // Workspace root
}
```

**Response:**
```typescript
{
  success: boolean;
  files_updated?: number;
  entities_updated?: number;
}
```

### CodeLens Operations

#### `hololoom/suggestCodeLens`

**Description:** Get suggestions for inline code comments

**Request:**
```typescript
{
  text: string;            // Comment text
  file: string;           // File path
  line: number;           // Line number
}
```

**Response:**
```typescript
{
  suggestions: Array<{
    text: string;         // Display text
    command: string;      // VS Code command
    arguments?: any[];    // Command args
    confidence: number;   // 0-1 score
  }>;
  tooltip?: string;       // Hover tooltip
}
```

#### `hololoom/getCodeLens`

**Description:** Get detailed info for CodeLens item

**Request:**
```typescript
{
  id: string;             // CodeLens ID
}
```

**Response:**
```typescript
{
  title: string;
  details?: string;
  references?: Array<{
    file: string;
    line: number;
    text: string;
  }>;
}
```

### Knowledge Graph Operations

#### `hololoom/getKnowledgeGraph`

**Description:** Get knowledge graph for visualization

**Request:**
```typescript
{
  query?: string;         // Optional filter
  max_nodes?: number;     // Max nodes to return
  include_edges?: boolean; // Include relationship edges
}
```

**Response:**
```typescript
{
  nodes: Array<{
    id: string;
    label: string;
    type: string;         // Entity type
    confidence?: number;
  }>;
  edges: Array<{
    source: string;
    target: string;
    label: string;        // Relationship type
    weight?: number;      // Edge weight
  }>;
}
```

#### `hololoom/searchKnowledgeGraph`

**Description:** Search knowledge graph by entity

**Request:**
```typescript
{
  query: string;          // Entity to search
  max_results?: number;
  include_paths?: boolean; // Include relationship paths
}
```

**Response:**
```typescript
{
  results: Array<{
    entity: string;
    matches: string[];
    relationships: Array<{
      entity: string;
      type: string;
      confidence: number;
    }>;
  }>;
}
```

### Server Control

#### `hololoom/status`

**Description:** Get server status and configuration

**Request:** (empty)

**Response:**
```typescript
{
  status: "ready" | "initializing" | "error";
  version?: string;
  hololoom_version?: string;
  python_version?: string;
  uptime_seconds?: number;
}
```

#### `hololoom/shutdown`

**Description:** Graceful server shutdown

**Request:** (empty)

**Response:**
```typescript
{
  success: boolean;
}
```

---

## Error Handling

### Error Response Format

All errors follow JSON-RPC error specification:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": {
      "detail": "Python module not found"
    }
  }
}
```

### Error Codes

| Code | Name | Meaning |
|------|------|---------|
| `-32700` | Parse error | JSON not valid |
| `-32600` | Invalid Request | Request structure invalid |
| `-32601` | Method not found | Unknown LSP method |
| `-32602` | Invalid params | Parameters don't match schema |
| `-32603` | Internal error | Server error |
| `-32000` to `-32099` | Custom | Server-specific errors |

### Common Error Scenarios

#### Python Not Installed

**Symptom:**
```
Error: Python executable not found at /usr/bin/python3
```

**Solution:**
- Install Python 3.8+
- Configure `hololoom.lsp.pythonPath` in settings
- Restart VS Code

#### HoloLoom Not Installed

**Symptom:**
```
Error: Module 'HoloLoom' not found
```

**Solution:**
- Install HoloLoom: `pip install HoloLoom`
- Or configure `hololoom.lsp.hololoomPath` in settings
- Restart VS Code

#### Memory Index Corrupted

**Symptom:**
```
Error: Knowledge graph index corrupted
```

**Solution:**
- Clear cache: `rm -rf ~/.cache/hololoom`
- Rebuild index: Run `hololoom/indexWorkspace` command
- If persists, re-initialize: Uninstall/reinstall HoloLoom

---

## Extension Points

### Adding New LSP Methods

To add a new LSP method to Promptly:

**1. Define in LSP Server (Python):**

```python
# File: hololoom_lsp_server.py
class HoloLoomLSPServer:
    async def handle_my_method(self, params):
        # Implementation
        return result

    def register_handlers(self):
        self.server.feature('hololoom/myMethod')(
            self.handle_my_method
        )
```

**2. Call from Extension (TypeScript):**

```typescript
// File: src/commands/myCommand.ts
export class MyCommand {
    async execute() {
        const result = await this.lspClient.sendRequest(
            'hololoom/myMethod',
            { /* params */ }
        );
        return result;
    }
}
```

**3. Register in VS Code:**

```typescript
// File: src/extension.ts
export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('promptly.myCommand', async () => {
            const cmd = new MyCommand(lspClient);
            await cmd.execute();
        })
    );
}
```

### Adding Custom Notifications

**Server Side:**

```python
# Send notification to client
self.server.notify('hololoom/myNotification', {
    'status': 'progress',
    'value': 50
})
```

**Client Side:**

```typescript
client.onNotification('hololoom/myNotification', (params) => {
    console.log('Server notification:', params);
});
```

---

## Testing Strategy

### Unit Tests

**What to test:**
- LSP message parsing/serialization
- Request handler logic
- Error handling
- HoloLoom integration

**Example:**
```typescript
test('remember() saves content to memory', async () => {
    const client = createMockLSPClient();
    const result = await client.sendRequest('hololoom/remember', {
        content: 'test'
    });
    expect(result.success).toBe(true);
});
```

### Integration Tests

**What to test:**
- End-to-end LSP communication
- Server auto-start and connection
- Multi-message sequences

**Example:**
```typescript
test('full workflow: remember → recall', async () => {
    await client.sendRequest('hololoom/remember', {
        content: 'Learn LSP'
    });
    const results = await client.sendRequest('hololoom/recall', {
        query: 'LSP'
    });
    expect(results.memories).toHaveLength(1);
});
```

### E2E Tests

**What to test:**
- Full VS Code extension
- Real LSP server process
- Real HoloLoom backend

**Example:**
```typescript
test('sidebar: capture → search → display', async () => {
    // Open sidebar
    const sidebar = await openHoloLoomSidebar();

    // Capture memory
    await sidebar.rememberNote('Testing LSP');

    // Search
    const results = await sidebar.search('LSP');

    // Verify display
    expect(sidebar.hasResult('Testing LSP')).toBe(true);
});
```

---

## Performance Considerations

### Latency Targets

| Operation | Target | Current |
|-----------|--------|---------|
| Remember | <100ms | ~65ms ✅ |
| Recall | <150ms | ~78ms ✅ |
| CodeLens | <100ms | ~40ms ✅ |
| Index workspace | <5000ms | ~4200ms ✅ |

### Connection Optimization

- **Connection pooling:** One persistent connection (not per-request)
- **Message batching:** Combine multiple requests when possible
- **Caching:** Cache frequent queries on client side
- **Compression:** LSP supports binary protocol for smaller messages

### Memory Optimization

- **Lazy loading:** Don't load full knowledge graph until needed
- **Pagination:** Return results in chunks for large queries
- **Cleanup:** Release unused resources on disconnect

---

## Debugging

### Enable Debug Logs

```json
{
  "hololoom.lsp.logLevel": "debug"
}
```

### View LSP Messages

```
Ctrl+Shift+U → "HoloLoom Language Server"
Scroll through JSON-RPC messages
```

### Trace LSP Protocol

Enable detailed tracing:

```json
{
  "[python]": {
    "editor.formatOnSave": true
  },
  "lsp": {
    "trace": "verbose"
  }
}
```

---

## References

- [Language Server Protocol Specification](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)
- [vscode-languageclient Documentation](https://github.com/microsoft/vscode-languageserver-node/tree/main/client)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

---

**Next:** See [MIGRATION_HTTP_TO_LSP.md](MIGRATION_HTTP_TO_LSP.md) for migration details or [SETUP_LSP.md](SETUP_LSP.md) for setup instructions.
