# Quick Test Guide

Fast guide to test the Promptly VS Code extension prototype.

## Prerequisites

✅ Node.js installed
✅ Python 3.x installed
✅ VS Code installed

## 1-Minute Setup

```bash
# Install TypeScript dependencies
cd promptly-vscode
npm install
npm run compile

# Initialize Promptly (if not already done)
cd ../Promptly
python -c "from promptly import Promptly; p = Promptly(); p.init()"

# Add test prompts
python -c "from promptly import Promptly; p = Promptly(); p.add('greeting', 'You are a friendly assistant.', {'tags': ['test']})"
python -c "from promptly import Promptly; p = Promptly(); p.add('coder', 'You are an expert Python developer.', {'tags': ['code', 'python']})"
python -c "from promptly import Promptly; p = Promptly(); p.add('writer', 'You are a creative writing assistant.', {'tags': ['creative', 'writing']})"
```

## Test the Bridge Server (Standalone)

```bash
# Start the Python bridge
cd Promptly
python promptly/vscode_bridge.py

# In another terminal, test endpoints
curl http://localhost:8765/health
curl http://localhost:8765/prompts
curl http://localhost:8765/prompts/greeting
```

Expected output:
- Health: `{"status":"healthy","promptly_available":true,...}`
- Prompts: List of 3 prompts with tags
- Greeting: Full prompt content + metadata

## Test in VS Code

### Method 1: F5 Launch (Recommended)
1. Open `promptly-vscode` folder in VS Code
2. Press **F5** (or Run → Start Debugging)
3. Extension Development Host window opens
4. Look for "Promptly" icon in activity bar (left sidebar)
5. Click to expand tree view
6. See branches and prompts
7. Click any prompt to view content

### Method 2: Manual Install
1. Package extension: `vsce package` (requires `npm install -g vsce`)
2. Install .vsix: Extensions → ... → Install from VSIX
3. Reload window
4. Same steps as Method 1

## What You Should See

### Sidebar Tree View
```
PROMPTLY
└── 📁 main (3 prompts)
    ├── 📄 greeting (test)
    ├── 📄 coder (code, python)
    └── 📄 writer (creative, writing)
```

### Click Behavior
- Click prompt → Opens in read-only editor
- Shows full content in Markdown format
- Hover tooltip shows metadata (branch, tags, created)

### Toolbar
- 🔄 Refresh button → Reloads prompt list

## Troubleshooting

### "Promptly not available" error
```bash
# Check if Promptly is initialized
cd Promptly
python -c "from promptly import Promptly; p = Promptly(); print(p.list_prompts())"
```

### Bridge server won't start
```bash
# Check port 8765 is free
netstat -ano | findstr :8765

# Kill any processes using port 8765
# Windows: taskkill /PID <pid> /F
# Linux/Mac: kill -9 <pid>
```

### Extension not appearing
- Check VS Code output panel (View → Output → Promptly)
- Check Debug Console for errors
- Verify TypeScript compiled: `npm run compile`

### Prompts not loading
- Check bridge server logs (terminal running vscode_bridge.py)
- Check browser DevTools console (Help → Toggle Developer Tools)
- Verify health endpoint: `curl http://localhost:8765/health`

## Performance Check

Run these commands while extension is active:

```bash
# Test cache performance
time curl http://localhost:8765/prompts  # First call (cold)
time curl http://localhost:8765/prompts  # Second call (cached, should be faster)

# Test individual prompt cache
time curl http://localhost:8765/prompts/greeting  # First call
time curl http://localhost:8765/prompts/greeting  # Cached
```

Expected:
- First call: ~50ms
- Cached call: ~5-10ms (90% improvement)

## Demo Script

Show off the prototype:

1. **Start**: "Here's Promptly integrated into VS Code"
2. **Sidebar**: Point out tree view with branches
3. **Click**: Open a prompt, show content in editor
4. **Tags**: Hover to show tooltip with metadata
5. **Refresh**: Click refresh button, show instant update
6. **Performance**: Show cache logs in terminal (< 1ms hits)
7. **Error handling**: Try to get non-existent prompt, show 404 handling

## Next Steps

After testing:
- ✅ Validate technical approach works
- ✅ Gather feedback on UX
- ✅ Decide: Ship v0.1 or build full v1.2?

---

**Questions?** Check [PROTOTYPE_COMPLETE.md](PROTOTYPE_COMPLETE.md) for detailed docs.
