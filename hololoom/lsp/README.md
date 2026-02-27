# HoloLoom LSP Server

Language Server Protocol (LSP) implementation for HoloLoom neural memory system.

This server provides semantic code intelligence for any editor that supports LSP:
- **VSCode**, **Neovim**, **Emacs**, **Sublime**, **Vim**, and more

**Status**: Skeleton v0.1.0 (2025-11-16)
**Maintainer**: HoloLoom Contributors

## Quick Start

### 1. Install Dependencies

```bash
# Install pygls (Python LSP framework)
pip install pygls

# Verify installation
python -c "import pygls; print('pygls installed:', pygls.__version__)"
```

### 2. Run the Server

```bash
# Start on stdio (standard input/output for editors)
PYTHONPATH=. python -m hololoom.lsp.server

# Or with explicit log level
PYTHONPATH=. python -m hololoom.lsp.server --log-level INFO

# Or start on TCP port (for testing/debugging)
PYTHONPATH=. python -m hololoom.lsp.server --port 8080
```

The server will start and wait for LSP client connections.

### 3. Connect an LSP Client

#### Option A: VSCode (using LSP Client extension)

1. Install the **LSP Client** extension in VSCode
2. Open settings and configure:

```json
{
  "lsp": {
    "hololoom": {
      "command": "python",
      "args": ["-m", "hololoom.lsp.server"],
      "languages": ["python"],
      "initializationOptions": {}
    }
  }
}
```

3. Open a Python file - the server should auto-connect

#### Option B: Neovim (using nvim-lspconfig)

```lua
-- ~/.config/nvim/init.lua
local lspconfig = require('lspconfig')

lspconfig.hololoom.setup {
    cmd = {"python", "-m", "hololoom.lsp.server"},
    filetypes = {"python"},
}
```

#### Option C: Emacs (using lsp-mode)

```elisp
;; ~/.emacs.d/init.el
(lsp-register-client
 (make-lsp-client
  :new-connection (lsp-stdio-connection
                   '("python" "-m" "hololoom.lsp.server"))
  :major-modes '(python-mode)
  :server-id 'hololoom-lsp))
```

#### Option D: Command Line (for testing)

Use an LSP test client:

```bash
# Terminal 1: Start the server on TCP
PYTHONPATH=. python -m hololoom.lsp.server --port 8080

# Terminal 2: Test with a simple LSP client
python -c "
import json
import socket

# Connect to server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 8080))

# Send initialize request
init_msg = {
    'jsonrpc': '2.0',
    'id': 1,
    'method': 'initialize',
    'params': {
        'processId': None,
        'rootUri': 'file:///tmp',
        'capabilities': {}
    }
}

# Send LSP message (simplified)
print('Sending initialize request...')
sock.close()
"
```

## Testing

### Unit Tests (stub)

```bash
# Create tests/test_lsp_server.py
PYTHONPATH=. python -m pytest hololoom/lsp/tests/ -v
```

### Integration Testing

```bash
# 1. Start server in background
PYTHONPATH=. python -m hololoom.lsp.server --log-level DEBUG &
SERVER_PID=$!

# 2. Run LSP client tests
python tests/test_lsp_integration.py

# 3. Cleanup
kill $SERVER_PID
```

### Manual Testing

Open server logs while using the LSP features:

```bash
# Terminal 1: Start server with debug logging
PYTHONPATH=. python -m hololoom.lsp.server --log-level DEBUG

# Terminal 2: Open your editor
# Trigger completion, hover, etc.
# Watch the server output in Terminal 1
```

## Architecture

### Design Philosophy

**"Protocol-based, memory-integrated"**

The LSP server follows HoloLoom's architecture:

1. **Protocol-Based**: Server implements LSP protocol (standardized)
2. **Memory-Integrated**: Uses HoloLoom's orchestrator and memory backends
3. **Graceful Degradation**: Falls back if optional dependencies unavailable
4. **Async-First**: All handlers are async for non-blocking I/O

### Request/Response Flow

```
Editor (VSCode/Neovim/etc.)
    ↓ [LSP Protocol over stdio/TCP]
HoloLoom LSP Server (pygls)
    ├─ Initialize: Declare capabilities
    ├─ Completion: Query HoloLoom memories
    ├─ Hover: Fetch knowledge graph info
    ├─ Definition: Find entity locations
    └─ Symbol Search: Query semantic index
    ↓
HoloLoom Orchestrator (future)
    ├─ Memory backends (KG, vectors)
    ├─ Alignment framework
    └─ Recursive learning
    ↓
Editor [Completion items, hover text, definitions, etc.]
```

### Key Components

| File | Purpose | Lines |
|------|---------|-------|
| `server.py` | Main LSP server implementation | ~350 |
| `__init__.py` | Package exports | ~30 |
| `README.md` | This documentation | ~300 |

### Handler Stubs

The following LSP handlers are implemented as stubs (ready for integration):

#### `textDocument/completion` (Completion)
- **When**: User presses Ctrl+Space or types `.`
- **Returns**: List of completion items from hololoom memories
- **Status**: Placeholder returns hardcoded items
- **TODO**: Query HoloLoom for context-aware completions

#### `textDocument/hover` (Hover)
- **When**: User hovers over a symbol
- **Returns**: Markdown documentation from knowledge graph
- **Status**: Placeholder returns sample markdown
- **TODO**: Extract symbol and query KG for entity info

#### `textDocument/definition` (Go to Definition)
- **When**: User clicks "Go to Definition" (Ctrl+Click)
- **Returns**: Location(s) of symbol definition
- **Status**: Placeholder returns current document
- **TODO**: Query KG for definition locations, support multi-definition

#### `workspace/symbol` (Symbol Search)
- **When**: User searches symbols (Ctrl+T in VSCode)
- **Returns**: List of matching symbols
- **Status**: Placeholder returns hardcoded symbols
- **TODO**: Query semantic index for ranked results

### Logging

The server uses structured logging with:
- **Timestamp**: ISO format (YYYY-MM-DD HH:MM:SS)
- **Level**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Name**: Component name (hololoom-lsp)
- **Message**: Descriptive message

Example logs:

```
2025-11-16 10:30:45 [INFO] hololoom-lsp: Server initializing for root URI: file:///home/user/project
2025-11-16 10:30:45 [INFO] hololoom-lsp: Client: VSCode
2025-11-16 10:30:45 [INFO] hololoom-lsp: Server capabilities declared
2025-11-16 10:30:46 [DEBUG] hololoom-lsp: Completion requested at file:///home/user/project/example.py:5:10
2025-11-16 10:30:46 [DEBUG] hololoom-lsp: Returning 3 completion items
```

## Features (Planned)

### Phase 1: Skeleton (Current)
- ✅ LSP server initialization
- ✅ Handler stubs for key endpoints
- ✅ Logging and error handling
- ✅ Command-line configuration
- ⏳ Basic documentation

### Phase 2: HoloLoom Integration
- Integrate orchestrator for memory queries
- Query knowledge graph for hover/definition
- Implement semantic completion
- Add entity ranking by confidence

### Phase 3: Advanced Features
- Multi-hop reasoning (workspace/symbol)
- Alignment framework integration (diagnostics)
- Incremental sync (efficiency)
- Custom text document commands

### Phase 4: Editor Extensions
- VS Code extension with UI enhancements
- Neovim plugin with custom keybindings
- Semantic search sidebar
- Memory visualization panels

## Configuration

### Command-Line Arguments

```bash
python -m hololoom.lsp.server [OPTIONS]

Options:
  --port PORT           TCP port to listen on (default: stdio)
  --host HOST          Host to bind to (default: 127.0.0.1)
  --log-level LEVEL    Logging level (default: INFO)
                        Choices: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Environment Variables

```bash
# Set default log level
export HOLOLOOM_LSP_LOG_LEVEL=DEBUG

# Set config directory (future)
export HOLOLOOM_CONFIG=/etc/hololoom/

# Enable profiling (future)
export HOLOLOOM_LSP_PROFILE=1
```

### Editor Configuration

#### VSCode

Add to `.vscode/settings.json`:

```json
{
  "[python]": {
    "defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "linting.enabled": false,
    "formatting.enabled": false
  },
  "lsp": {
    "hololoom": {
      "command": "python",
      "args": ["-m", "hololoom.lsp.server", "--log-level", "DEBUG"],
      "languages": ["python"],
      "initializationOptions": {},
      "trace.server": "verbose"
    }
  }
}
```

#### Neovim

Add to `init.lua`:

```lua
require('lspconfig').hololoom.setup {
    cmd = {"python", "-m", "hololoom.lsp.server"},
    filetypes = {"python"},
    root_dir = require('lspconfig').util.root_pattern(".git", "setup.py", "pyproject.toml"),
    single_file_support = true,
    settings = {
        python = {
            analysis = {
                diagnosticSeverityOverrides = {
                    reportGeneralTypeIssues = "none",
            }
        }
    }
}
```

## Troubleshooting

### Server Won't Start

**Error**: `ModuleNotFoundError: No module named 'pygls'`

**Solution**: Install pygls:
```bash
pip install pygls
```

**Error**: `PYTHONPATH not set`

**Solution**: Run with PYTHONPATH:
```bash
PYTHONPATH=. python -m hololoom.lsp.server
```

### No Completions Appearing

**Likely cause**: Handlers are still stubs (returning placeholder data)

**Solution**: Wait for Phase 2 integration with HoloLoom orchestrator

**Temporary workaround**: Check server logs:
```bash
PYTHONPATH=. python -m hololoom.lsp.server --log-level DEBUG
```

### Editor Won't Connect

**Check**: Server is actually running on TCP port:
```bash
netstat -an | grep 8080  # Or lsof -i :8080
```

**Check**: Firewall isn't blocking port:
```bash
telnet 127.0.0.1 8080
```

**Check**: Editor configuration has correct command/port

### Performance Issues

**Monitor**: CPU and memory usage
```bash
ps aux | grep "hololoom.lsp"
```

**Profile**: With log-level DEBUG
```bash
PYTHONPATH=. python -m hololoom.lsp.server --log-level DEBUG 2>&1 | tee lsp.log
```

## Development

### Adding a New Handler

1. Define handler function with `@server.feature()` decorator:

```python
@server.feature("textDocument/YOUR_FEATURE")
async def your_handler(params: YourParams) -> YourResult:
    """Handle YOUR_FEATURE request."""
    logger.debug(f"YOUR_FEATURE requested...")

    # TODO: Implement

    return YourResult(...)
```

2. Test by triggering from editor or LSP client

3. Update capabilities in `initialize()` if needed:

```python
server_capabilities = ServerCapabilities(
    your_feature_provider=True,
)
```

### Integration Checklist

When integrating HoloLoom components:

- [ ] Import orchestrator/memory modules
- [ ] Lazy-load in `on_initialized()` (not `initialize()`)
- [ ] Handle graceful degradation if imports fail
- [ ] Add error logging for all exceptions
- [ ] Test with actual editor/LSP client
- [ ] Update documentation
- [ ] Add integration tests

### Testing Handlers Locally

```python
# In Python shell
from hololoom.lsp.server import server, initialize, completion
from pygls.lsp import InitializeParams, CompletionParams, Position

# Mock params
init_params = InitializeParams(
    process_id=None,
    root_uri='file:///tmp',
    capabilities={}
)

# Call handler directly
result = await initialize(init_params)
print(result)

# Test completion
comp_params = CompletionParams(
    text_document={'uri': 'file:///test.py'},
    position=Position(line=0, character=0)
)
items = await completion(comp_params)
print([item.label for item in items.items])
```

## Performance Targets

| Operation | Target Latency | Status |
|-----------|----------------|--------|
| Initialization | <100ms | ✅ |
| Completion | <200ms | ⏳ (stub: ~1ms) |
| Hover | <150ms | ⏳ (stub: ~1ms) |
| Definition | <100ms | ⏳ (stub: ~1ms) |
| Symbol search | <500ms | ⏳ (stub: ~1ms) |
| Memory query | <200ms | ⏳ (not yet integrated) |

**Note**: Stub handlers are instant. Real HoloLoom integration will determine actual latency.

## References

- **LSP Specification**: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/
- **pygls Documentation**: https://pygls.readthedocs.io/
- **HoloLoom Architecture**: See `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md`
- **Weaving Orchestrator**: See `hololoom/weaving_orchestrator.py`

## License

HoloLoom LSP Server is part of the HoloLoom project. See repository LICENSE file.

## Contributing

Contributions welcome! See the HoloLoom contribution guidelines.

**Quick contribution guide**:
1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## Changelog

### v0.1.0 (2025-11-16)
- ✅ Initial skeleton implementation
- ✅ Basic LSP handlers (initialize, completion, hover, definition, symbol)
- ✅ Logging and error handling
- ✅ Command-line configuration
- ✅ Documentation and examples
- ⏳ HoloLoom integration (Phase 2)
