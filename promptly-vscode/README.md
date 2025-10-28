# Promptly VS Code Extension

**Version 0.2.0** - Now with Execution Engine! 🚀

Execute skills, chain workflows, and run recursive loops directly from VS Code.

## Features

### ✅ Prompt Library (v0.1.0)
- Browse prompts by branch
- View prompt content
- Refresh library

### 🆕 Execution Engine (v0.2.0)
- **⚡ Skill Execution**: Run individual prompts with user input
- **🔗 Chain Execution**: Sequential workflows with data flow
- **🔄 Loop Execution**: Recursive reasoning with 6 loop types
  - Refine, Critique, Decompose, Verify, Explore, Hofstadter
- **📊 Real-time Streaming**: WebSocket progress updates
- **📈 Quality Tracking**: Iteration-by-iteration improvement metrics

## Quick Start

### 1. Start Python Bridge
```bash
cd Promptly
python promptly/vscode_bridge.py
```

### 2. Open Execution Panel
- Click Promptly icon in activity bar
- Click **Play** (▶) button
- Start executing!

### 3. Try Your First Execution
See [EXECUTION_QUICKSTART.md](EXECUTION_QUICKSTART.md) for a 5-minute tutorial.

## Documentation

- **[Execution Quick Start](EXECUTION_QUICKSTART.md)** - Get running in 5 minutes
- **[Execution Guide](EXECUTION_GUIDE.md)** - Comprehensive documentation
- **[Architecture](PROTOTYPE_COMPLETE.md)** - Technical details

## What Works

### Python Bridge (FastAPI)
- **Server**: `Promptly/promptly/vscode_bridge.py`
- **Port**: 8765
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /prompts` - List all prompts with metadata
  - `GET /prompts/{name}` - Get specific prompt content

### TypeScript Extension
- **Sidebar tree view**: Groups prompts by branch
- **Commands**:
  - `promptly.refresh` - Refresh prompt library
  - `promptly.viewPrompt` - Open prompt in editor
- **Bridge client**: Auto-starts Python server, polls for data

## Testing Results

```bash
# Start Python bridge
cd Promptly && python promptly/vscode_bridge.py

# Test endpoints
curl http://localhost:8765/health
# {"status":"healthy","promptly_available":true}

curl http://localhost:8765/prompts
# {"prompts":[{"name":"test_prompt","branch":"main","tags":["test","assistant"],"created":"2025-10-27 07:13:55"}]}

curl http://localhost:8765/prompts/test_prompt
# {"content":"You are a helpful assistant.","metadata":{...}}
```

## Quick Start

1. **Install dependencies**:
```bash
cd promptly-vscode
npm install
npm run compile
```

2. **Start Python bridge**:
```bash
cd ../Promptly
python promptly/vscode_bridge.py
```

3. **Open in VS Code**:
- Open `promptly-vscode` folder in VS Code
- Press F5 to launch Extension Development Host
- Promptly sidebar should appear in activity bar

## Architecture Validated

✅ **Python ↔ TypeScript communication** works via FastAPI REST API
✅ **Process management** - Extension spawns/kills Python server
✅ **Data flow** - Prompts flow from SQLite → FastAPI → TypeScript → VS Code UI
✅ **UI rendering** - Tree view with branches and prompts displays correctly

## Performance Optimizations

### Python Bridge
- **In-memory caching** with TTL (30s for lists, 60s for individual prompts)
- **Structured logging** with DEBUG/INFO/ERROR levels
- **Error tracking** with full stack traces
- **Cache hit rate** logging for performance monitoring

### TypeScript Client
- **Health monitoring** with 30-second interval checks
- **Graceful degradation** when bridge is unhealthy
- **Request/response interceptors** for logging and error handling
- **Connection pooling** via axios (keepalive, max redirects)
- **Process lifecycle management** with proper cleanup

### Error Handling
- **503 Service Unavailable** when Promptly core not initialized
- **404 Not Found** with friendly warnings for missing prompts
- **Automatic retries** during server startup (20 attempts over 10 seconds)
- **Health status** exposed via `getHealthStatus()` method

## Next Steps (Full v1.2)

If prototype is approved:
1. Add execute panel (run prompts with variables)
2. Add analytics dashboard (webview)
3. Add inline editor commands
4. Polish UI/UX
5. Package for marketplace

**Time estimate**: 4 weeks for full build (see V1.2_VSCODE_EXTENSION_PLAN.md)

## Files Created

- `src/extension.ts` - Main entry point
- `src/api/PromptlyBridge.ts` - Python bridge client
- `src/promptLibrary/PromptTreeProvider.ts` - Sidebar tree view
- `Promptly/promptly/vscode_bridge.py` - FastAPI server
- `Promptly/promptly/__init__.py` - Package init for imports

**Prototype completed**: 2025-10-27
**Status**: Ready for validation