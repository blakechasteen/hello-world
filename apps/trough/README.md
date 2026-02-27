# Squad - AI Coding Assistant

**Agentic coding assistant powered by HoloLoom's multi-step reasoning**

Squad is a VS Code extension that brings HoloLoom's powerful agentic reasoning capabilities directly into your editor. Get AI assistance that doesn't just answer - it verifies, researches, and plans.

## Features

- **🤖 Agentic Reasoning**: Multi-step reasoning with verification loops
- **🔍 Code Understanding**: Explains code with full context awareness
- **🛠️ Smart Fixes**: Suggests fixes for errors and warnings
- **📊 Reasoning Transparency**: See every step of Squad's thinking process
- **⚡ Fast & Reliable**: Powered by HoloLoom's Phase 5 compositional cache (291× speedup)

## Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
cd ..  # Go to mythRL root
pip install fastapi uvicorn pydantic

# TypeScript dependencies
cd squad
npm install
```

### 2. Start the Server

```bash
# From squad/ directory
python server.py
```

You should see:
```
INFO:     Squad server ready! 🚀
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 3. Build the Extension

```bash
# From squad/ directory
npm run compile
```

### 4. Run in VS Code

1. Open the `squad/` folder in VS Code
2. Press `F5` to launch Extension Development Host
3. In the new window, press `Ctrl+Shift+Q` to ask Squad a question!

## Usage

### Commands

- **Squad: Ask Question** (`Ctrl+Shift+Q`): Ask Squad anything about your code
- **Squad: Explain Selection** (`Ctrl+Shift+E`): Explain selected code
- **Squad: Suggest Fix**: Get AI suggestions for fixing errors
- **Squad: Open Agent Panel**: Open the interactive agent panel

### Reasoning Modes

Squad supports 4 reasoning modes:

1. **Direct** - Single-pass answer (fastest)
2. **Verify** - Answer + verification loop (recommended)
3. **Research** - Multi-query exploration
4. **Plan & Execute** - Goal decomposition for complex tasks

### Example Workflows

**Explain Code:**
```typescript
// Select this code
function fibonacci(n: number): number {
  return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2);
}
```
Right-click → "Squad: Explain Selection"

**Fix Errors:**
Open a file with errors → Command Palette → "Squad: Suggest Fix"

**Ask Questions:**
`Ctrl+Shift+Q` → "How does Thompson Sampling work?"

## Architecture

```
┌─────────────────────┐
│   VS Code UI        │
│   - Commands        │
│   - Agent Panel     │
│   - Context Menu    │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│   FastAPI Server    │
│   - /query          │
│   - /chat           │
│   - /stats          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  HoloLoom Agentic   │
│  - ReasoningMode    │
│  - Verification     │
│  - Multi-step       │
└─────────────────────┘
```

## Configuration

Open VS Code Settings and search for "Squad":

- `squad.serverUrl`: HoloLoom server URL (default: `http://localhost:8000`)
- `squad.reasoningMode`: Default reasoning mode (default: `verify`)
- `squad.maxSteps`: Maximum reasoning steps (default: `5`)
- `squad.showReasoningSteps`: Show intermediate steps (default: `true`)

## Development

### Project Structure

```
squad/
├── src/
│   ├── extension.ts           # Entry point
│   ├── HoloLoomBridge.ts      # Python communication
│   ├── CodeContextProvider.ts # Context extraction
│   └── AgentPanel.ts          # Webview UI
├── server.py                  # FastAPI server
├── package.json               # VS Code manifest
└── tsconfig.json              # TypeScript config
```

### Running in Development

Terminal 1 - Start server:
```bash
python server.py
```

Terminal 2 - Watch TypeScript:
```bash
npm run watch
```

VS Code - Press `F5` to launch extension

### Testing the Server

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is Thompson Sampling?",
    "mode": "verify",
    "max_steps": 3
  }'
```

## Troubleshooting

**Server won't start:**
- Check Python dependencies: `pip install fastapi uvicorn pydantic`
- Verify you're in the mythRL root for imports
- Check port 8000 isn't in use

**Extension doesn't compile:**
- Run `npm install` first
- Check TypeScript version: `tsc --version` (should be 5.x)

**Can't connect to server:**
- Verify server is running on port 8000
- Check `squad.serverUrl` setting in VS Code
- Look for errors in Output panel → "Squad"

## Next Steps

- [ ] Add code ingestion (feed codebase into memory)
- [ ] Implement code completions (inline suggestions)
- [ ] Add diagnostic provider (auto-fix suggestions)
- [ ] Create CodeSpinner for workspace analysis
- [ ] Add conversation history persistence

## Learn More

- [HoloLoom Documentation](../README.md)
- [Agentic Architecture](../hololoom/agentic/README.md)
- [Alignment & Safety](../ALIGNMENT_SAFETY_BRIEF.md)

---

**Built with ❤️ using HoloLoom's recursive learning system**
