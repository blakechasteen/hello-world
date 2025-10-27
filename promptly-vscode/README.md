# Promptly VS Code Extension - Prototype

Quick prototype (v0.1.0) validating the technical approach for VS Code integration.

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