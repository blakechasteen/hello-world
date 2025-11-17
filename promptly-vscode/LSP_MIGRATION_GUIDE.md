# HoloLoom LSP Client Migration Guide

**Status**: ✅ Complete (Phase 5 Wave 3 - November 2025)

This guide documents the migration of the Promptly VS Code extension from HTTP API calls to the Language Server Protocol (LSP) client.

## Overview

The extension now uses LSP for communication with HoloLoom's neural memory system, providing:
- **Real-time code completion** from knowledge graph
- **Hover information** with entity context
- **Go-to-definition** via graph relationships
- **Workspace symbol search** (semantic + graph-based)
- **Automatic graceful fallback** to HTTP API if LSP unavailable

## Architecture Changes

### Before (HTTP API Only)

```
VS Code Extension → HTTP API → HoloLoom Server (port 8000)
```

- All commands used axios for HTTP POST requests
- No real-time IDE integration
- Limited to explicit user commands

### After (LSP + HTTP Fallback)

```
VS Code Extension → LSP Client → HoloLoom LSP Server (stdio)
                  ↓ (fallback)
                  → HTTP API → HoloLoom Server (port 8000)
```

- LSP client for real-time IDE features (completion, hover, definition)
- Automatic fallback to HTTP API if LSP server unavailable
- Both modes work simultaneously (LSP for IDE features, HTTP for legacy endpoints)

## Key Components

### 1. LSP Client (`src/lsp/client.ts`)

**Responsibilities**:
- Start/stop HoloLoom LSP server process
- Auto-detect HoloLoom installation path
- Auto-detect Python interpreter
- Manage LSP client lifecycle
- Provide request/notification APIs

**Key Features**:
- Automatic server startup on extension activation
- Graceful degradation if server unavailable
- Configuration-based customization
- Output channel for debugging

**API**:
```typescript
const client = new HoloLoomLSPClient(context);
await client.start();              // Start LSP server
await client.stop();               // Stop gracefully
await client.restart();            // Restart server
client.isRunning();                // Check status
await client.sendRequest(method, params);  // Send LSP request
client.sendNotification(method, params);   // Send notification
```

### 2. Updated Commands (`src/commands/hololoomCommands.ts`)

**Migration Pattern**:
```typescript
// Try LSP first
if (this.lspClient && this.lspClient.isRunning()) {
    try {
        const result = await this.lspClient.sendRequest('method', params);
        return formatResult(result);
    } catch (error) {
        // Fall through to HTTP API
    }
}

// Fallback to HTTP API
const response = await axios.post(url, params);
return formatResult(response.data);
```

**Updated Methods**:
- `remember()` - Uses `hololoom/remember` custom request
- `recall()` - Uses `workspace/symbol` for semantic search
- Both maintain HTTP fallback for compatibility

### 3. CodeLens Provider (`src/providers/codeLensProvider.ts`)

**Changes**:
- Uses `textDocument/completion` for context-aware suggestions
- Falls back to HTTP API if LSP unavailable
- Higher confidence scores for LSP results (0.8 vs 0.6)

**Benefits**:
- Real-time suggestions as you type
- Context-aware recommendations
- Faster response times (no HTTP overhead)

### 4. Sidebar Provider (`src/views/sidebarProvider.ts`)

**Changes**:
- Receives LSP client instance in constructor
- Passes LSP client to HoloLoomCommands
- All operations use LSP-first pattern

### 5. Extension Activation (`src/extension.ts`)

**Startup Sequence**:
1. Create LSP client instance
2. Start LSP server asynchronously (non-blocking)
3. Pass LSP client to all components
4. Register LSP management commands

**New Commands**:
- `promptly.restartLSP` - Restart LSP server
- `promptly.lspStatus` - Show LSP connection status

**Deactivation**:
- Gracefully stops LSP server
- Cleans up resources

## Configuration

### VS Code Settings

Add to your workspace or user settings:

```json
{
  // Enable/disable LSP client
  "hololoom.lsp.enabled": true,

  // Custom Python interpreter (auto-detected if not set)
  "hololoom.lsp.pythonPath": "/path/to/python",

  // Custom HoloLoom path (auto-detected if not set)
  "hololoom.lsp.hololoomPath": "/path/to/hololoom",

  // LSP server log level
  "hololoom.lsp.logLevel": "INFO"
}
```

### Auto-Detection

The client automatically detects:
1. **HoloLoom Path**:
   - Workspace folders (checks for `HoloLoom/__init__.py`)
   - Parent directory of workspace
   - Fallback to configuration setting

2. **Python Interpreter**:
   - Python extension's active interpreter
   - `python3` on Unix-like systems
   - `python` on Windows
   - Fallback to configuration setting

## Usage

### For Users

**Normal Operation**:
1. Open VS Code in a workspace with HoloLoom installed
2. Extension automatically starts LSP server
3. Use commands as normal (`Ctrl+Shift+P` → `HoloLoom: ...`)
4. Get real-time completions and hover information

**Troubleshooting**:
1. Check LSP status: `Ctrl+Shift+P` → `HoloLoom: Show LSP Status`
2. Restart LSP server: `Ctrl+Shift+P` → `HoloLoom: Restart LSP Server`
3. View logs: `View` → `Output` → Select "HoloLoom LSP" from dropdown
4. If LSP fails, extension automatically falls back to HTTP API

**Manual Configuration**:
If auto-detection fails, manually configure paths:
1. Open settings: `File` → `Preferences` → `Settings`
2. Search for "HoloLoom LSP"
3. Set `hololoom.lsp.hololoomPath` to your HoloLoom installation
4. Set `hololoom.lsp.pythonPath` to your Python interpreter
5. Restart LSP: `Ctrl+Shift+P` → `HoloLoom: Restart LSP Server`

### For Developers

**Extending LSP Integration**:

```typescript
import { getLSPClient } from './extension';

// Get active LSP client
const client = getLSPClient();

if (client && client.isRunning()) {
    // Use LSP
    const result = await client.sendRequest('custom/method', params);
} else {
    // Use HTTP fallback
    const response = await axios.post(url, params);
}
```

**Adding New LSP Handlers**:

1. Add handler to `HoloLoom/lsp/server.py`:
```python
@server.feature('custom/method')
async def custom_handler(params):
    # Implementation
    return result
```

2. Use in extension:
```typescript
const result = await client.sendRequest('custom/method', params);
```

## Testing

### Run Tests

```bash
cd promptly-vscode
npm test
```

**Test Environment Variables**:
- `SKIP_LSP_TESTS=1` - Skip LSP integration tests (if server unavailable)

### Manual Testing

1. **Start LSP Server Manually** (for debugging):
```bash
cd /path/to/hololoom
python -m HoloLoom.lsp.server --log-level DEBUG
```

2. **Reload Extension**:
   - `Ctrl+Shift+P` → `Developer: Reload Window`

3. **Check Logs**:
   - `View` → `Output` → "HoloLoom LSP"

### Test Scenarios

- ✅ LSP client starts automatically on activation
- ✅ LSP client stops gracefully on deactivation
- ✅ LSP client restarts successfully
- ✅ Completion requests return results
- ✅ Hover requests show entity information
- ✅ Definition navigation works
- ✅ Workspace symbol search works
- ✅ HTTP fallback works when LSP unavailable
- ✅ Multiple concurrent requests handled
- ✅ Server errors handled gracefully

## Migration Checklist

For developers migrating other components:

- [ ] Import `HoloLoomLSPClient` from `../lsp/client`
- [ ] Add LSP client parameter to constructor
- [ ] Implement LSP-first pattern with HTTP fallback
- [ ] Test with LSP server running
- [ ] Test with LSP server stopped (fallback mode)
- [ ] Update tests to handle both modes
- [ ] Document LSP-specific behavior

## Troubleshooting

### Common Issues

**1. LSP Server Fails to Start**

**Symptoms**: Error message "Failed to start LSP server", HTTP fallback used

**Solutions**:
- Check HoloLoom installation: `python -m HoloLoom.lsp.server --help`
- Check Python path in settings
- Check HoloLoom path in settings
- View logs: Output → "HoloLoom LSP"

**2. LSP Server Crashes**

**Symptoms**: Features stop working, "LSP: Not connected" message

**Solutions**:
- Restart LSP: `Ctrl+Shift+P` → `HoloLoom: Restart LSP Server`
- Check LSP server logs for errors
- Update HoloLoom to latest version

**3. Auto-Detection Fails**

**Symptoms**: "HoloLoom installation not found" warning

**Solutions**:
- Open workspace in HoloLoom repository root
- Manually configure `hololoom.lsp.hololoomPath` in settings
- Ensure `HoloLoom/__init__.py` exists

**4. Python Interpreter Not Found**

**Symptoms**: "python: command not found" error

**Solutions**:
- Install Python extension for VS Code
- Manually configure `hololoom.lsp.pythonPath` in settings
- Ensure Python is in PATH

### Debug Mode

Enable debug logging:

1. Open settings: `File` → `Preferences` → `Settings`
2. Search for "HoloLoom LSP Log Level"
3. Set to "DEBUG"
4. Restart LSP server
5. View detailed logs in Output panel

## Performance Comparison

### HTTP API
- **Latency**: ~50-150ms per request (network overhead)
- **Features**: Limited to explicit commands
- **Availability**: Requires separate server process

### LSP
- **Latency**: ~5-20ms per request (stdio communication)
- **Features**: Real-time IDE integration (completion, hover, definition)
- **Availability**: Auto-started with extension

### Hybrid Approach
- **Best of both worlds**: LSP for IDE features, HTTP for complex operations
- **Graceful degradation**: HTTP fallback if LSP unavailable
- **Zero breaking changes**: Existing functionality preserved

## Future Enhancements

Planned for Phase 5 Wave 4:

1. **Enhanced LSP Features**:
   - Code actions (refactoring suggestions)
   - Diagnostics (linting with alignment framework)
   - Formatting (code style suggestions)

2. **Workspace Indexing**:
   - Automatic background indexing via LSP
   - Progress notifications
   - Incremental updates on file changes

3. **Graph Visualization**:
   - Interactive knowledge graph in webview
   - Entity relationships visualization
   - Click-to-navigate from graph to code

4. **Performance Optimizations**:
   - Request caching
   - Debouncing for frequent requests
   - Streaming results for large queries

## References

- [LSP Specification](https://microsoft.github.io/language-server-protocol/)
- [HoloLoom LSP Server](../../HoloLoom/lsp/server.py)
- [HoloLoom LSP Architecture](../../LSP_ARCHITECTURE.md)
- [Phase 5 Wave 3 Roadmap](../../MASTER_ROADMAP_PHASES_5_8.md)

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. View LSP server logs: `python -m HoloLoom.lsp.server --log-level DEBUG`
3. View extension logs: Output → "HoloLoom LSP"
4. File issue on GitHub with logs attached
