# Quick Start: Agentic Intelligence System

## 🚨 Port Configuration

**Important**: The server uses **port 8001** (not 8000) to avoid conflicts.

## 🚀 Start the System (2 Commands)

### Terminal 1: Start Backend Server

```powershell
# From repository root
python start_agentic_server.py
```

You should see:
```
Starting HoloLoom Agentic API on port 8001...
Features:
  ✓ Real LLM calls (Ollama/Anthropic)
  ✓ Persistent memory (Neo4j + Qdrant)
  ✓ Agentic reasoning (4 modes)
  ✓ WebSocket streaming for UI

API available at: http://localhost:8001
Docs available at: http://localhost:8001/docs
```

### Terminal 2: Start Chat UI

```powershell
# From repository root
python ui/agentic_learner_ui.py
```

You should see:
```
Running on local URL:  http://localhost:7860
```

### Open Browser

Open: **http://localhost:7860**

## 🎯 Test the System

Try these queries in the UI:

**VERIFY Mode** (detects contradictions):
```
Is Thompson Sampling always the best exploration strategy?
```

**RESEARCH Mode** (multi-step investigation):
```
Compare epsilon-greedy vs Thompson Sampling for my use case
```

**PLAN_EXECUTE Mode** (structured planning):
```
Help me implement a new retrieval backend
```

## 🔧 Troubleshooting

### "Connection refused" in UI

The backend server isn't running. Start it with:
```powershell
python start_agentic_server.py
```

### "Port 8001 already in use"

Find and kill the process:
```powershell
# PowerShell
netstat -ano | findstr ":8001"
# Note the PID, then:
taskkill /PID <pid> /F
```

Or edit `start_agentic_server.py` and change `port=8001` to `port=8002`.

### "No LLM available"

Install Ollama (free, local):
1. Download from https://ollama.ai
2. Run: `ollama pull llama3.2:3b`
3. Restart server

Or use Anthropic Claude (paid, cloud):
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python start_agentic_server.py
```

## 📊 Features You Get

| Feature | Status |
|---------|--------|
| **4 Reasoning Modes** | ✅ DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE |
| **Real LLM Calls** | ✅ Ollama (local) or Anthropic (cloud) |
| **Persistent Memory** | ✅ Neo4j + Qdrant (auto-fallback) |
| **Contradiction Detection** | ✅ Automatic in VERIFY mode |
| **Audit Trail** | ✅ Complete provenance logging |
| **Chat Interface** | ✅ Beautiful Gradio UI |

## 🧪 Testing Without UI

Use curl to test the API directly:

```powershell
# Health check
curl http://localhost:8001/health

# Query with VERIFY mode
curl -X POST http://localhost:8001/query `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"What is Thompson Sampling?\", \"mode\": \"verify\", \"max_steps\": 5}'
```

## 📖 Next Steps

- Read [COMPLETE_INTEGRATION_GUIDE.md](COMPLETE_INTEGRATION_GUIDE.md) for full documentation
- Read [AGENTIC_SYSTEM_COMPLETE.md](AGENTIC_SYSTEM_COMPLETE.md) for architecture details
- Try the demos: `python demos/demo_agentic_reasoning.py`

## 🎉 You're Ready!

The system is now fully integrated with:
- Real LLM calls (not stubs)
- Persistent memory (loads from backends)
- Beautiful chat UI
- 4 reasoning modes with verification

Enjoy! 🚀
