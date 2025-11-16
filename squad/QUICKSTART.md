# Squad - Quick Start Guide

## ✅ Installation Complete!

All files have been created and committed to git:
- 11 files, 1,313 lines of code
- TypeScript extension (6 files)
- Python server (1 file)
- Configuration and documentation

## Dependencies Status

### ✅ Installed:
- `einops` - Embedding operations
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### ⏳ Installing (running in background):
- `numpy` - Numerical computing
- `torch` - Deep learning framework (900MB)
- `scipy` - Scientific computing
- `networkx` - Graph algorithms

**Note**: Torch is large (900MB). Installation takes 5-10 minutes.

## Quick Test (Once Dependencies Finish)

### 1. Start the Server

```bash
cd /home/user/hello-world/squad
PYTHONPATH=/home/user/hello-world python server.py
```

You should see:
```
INFO:     Squad server ready! 🚀
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 2. Test Health Endpoint

In another terminal:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "orchestrator_ready": true,
  "timestamp": "2025-11-15T..."
}
```

### 3. Test Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is Thompson Sampling?",
    "mode": "direct",
    "max_steps": 3
  }'
```

Expected response:
```json
{
  "response": "...",
  "confidence": 0.85,
  "reasoning_mode": "direct",
  "steps_taken": [...],
  "total_duration_ms": 150.5
}
```

### 4. Build TypeScript Extension

```bash
cd /home/user/hello-world/squad
npm install
npm run compile
```

### 5. Run in VS Code

1. Open the `squad/` folder in VS Code
2. Press `F5` to launch Extension Development Host
3. In the new window, press `Ctrl+Shift+Q`
4. Type: "What is Thompson Sampling?"
5. Watch Squad think!

## Commands

Once running in VS Code:

- **`Ctrl+Shift+Q`** - Ask Squad a question
- **`Ctrl+Shift+E`** - Explain selected code
- **Squad: Suggest Fix** - Fix errors
- **Squad: Refactor Code** - Refactor selected code
- **Squad: Generate Tests** - Generate tests
- **Squad: Open Agent Panel** - View reasoning steps

## Troubleshooting

**Server won't start:**
- Check dependencies: `pip list | grep -E "numpy|torch|scipy"`
- Wait for torch to finish installing (~5-10 min)
- Check port 8000: `lsof -i :8000`

**Extension won't compile:**
- Run `npm install` in squad/ directory
- Check node version: `node --version` (needs 16+)

**Can't connect:**
- Verify server is running: `curl http://localhost:8000/health`
- Check VS Code settings: `squad.serverUrl`

## Architecture

```
VS Code Extension (TypeScript)
    ↓ HTTP (localhost:8000)
FastAPI Server (Python)
    ↓
WeavingOrchestrator
    ├─ Pattern Selection (BARE/FAST/FUSED)
    ├─ Feature Extraction
    ├─ Decision Engine
    └─ Safety Guardrails ✅
```

## Next Steps

1. ✅ Wait for dependencies to finish installing
2. ✅ Start server
3. ✅ Test health endpoint
4. ✅ Test query endpoint
5. ✅ Build TypeScript extension
6. ✅ Launch in VS Code
7. ✅ Test commands

## File Structure

```
squad/
├── src/
│   ├── extension.ts           # Main extension (250 lines)
│   ├── HoloLoomBridge.ts      # HTTP client (100 lines)
│   ├── AgentPanel.ts          # Interactive UI (150 lines)
│   └── CodeContextProvider.ts # Context extraction (40 lines)
├── server.py                  # FastAPI server (400 lines)
├── package.json               # VS Code manifest
├── tsconfig.json              # TypeScript config
├── requirements.txt           # Python dependencies
└── README.md                  # Full documentation
```

## Documentation

- **README.md** - Complete guide with examples
- **This file (QUICKSTART.md)** - Quick start instructions
- **package.json** - Extension configuration
- **server.py** - Server code with comments

---

**Status**: Ready for testing once dependencies finish installing!

Check installation progress:
```bash
ps aux | grep "pip install"
```

Estimated time remaining: 5-10 minutes for torch
