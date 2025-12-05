# HoloLoom + Promptly Setup Guide

Complete setup guide for getting the Promptly VS Code extension working with HoloLoom.

## Prerequisites

- **VS Code**: Version 1.80.0 or later
- **Node.js**: Version 16+ (for extension development)
- **Python**: Version 3.9+ (for HoloLoom server)
- **Git**: For version control

## Step-by-Step Setup

### 1. Install Python Dependencies

```bash
# Navigate to repository root
cd hello-world

# Create and activate virtual environment
python3 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install torch numpy fastapi uvicorn pydantic

# Optional: Full HoloLoom features
pip install sentence-transformers networkx scipy spacy
python -m spacy download en_core_web_sm
```

### 2. Start HoloLoom Server

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Start the server (from repository root)
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --reload --port 8000
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     HoloLoom Agentic API server...
INFO:     Rate limiter: 60 requests/minute per IP
INFO:     Memory backend: HYBRID
INFO:     HoloLoom server ready!
```

**Verify server is running**:
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy", "version": "1.0.0"}
```

### 3. Build VS Code Extension

```bash
# Navigate to extension directory
cd promptly-vscode

# Install Node.js dependencies
npm install

# Compile TypeScript
npm run compile
```

**Expected output**:
```
✔ Compiled successfully
```

### 4. Run Extension in Development Mode

**Option A: F5 Launch (Recommended)**

1. Open `promptly-vscode` folder in VS Code
2. Press `F5` (or Run → Start Debugging)
3. New VS Code window opens with extension loaded
4. Look for 🧠 icon in Activity Bar (left sidebar)

**Option B: Watch Mode (for development)**

```bash
# Terminal 1: Watch mode (auto-recompile)
npm run watch

# Then press F5 in VS Code
# After making changes, press "Reload Window" in Extension Development Host
```

### 5. Verify Installation

**Check Extension Activation**:
1. In the Extension Development Host window
2. View → Output
3. Select "Promptly" from dropdown
4. Should see: "Promptly extension activated"

**Check HoloLoom Connection**:
1. Click 🧠 icon in Activity Bar
2. Sidebar should open with "HoloLoom" title
3. Try capturing a note: "Test memory"
4. Should see success message

**Check CodeLens**:
1. Create a new file: `test.ts`
2. Add comment: `// NOTE: Testing CodeLens functionality`
3. Wait 2-3 seconds
4. Should see inline annotation: `📝 Capture this NOTE` or `💡 X related notes`

## Troubleshooting

### Server Won't Start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
source .venv/bin/activate
pip install fastapi uvicorn
```

**Error**: `Address already in use`

**Solution**:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :8000   # Windows (note PID, then taskkill /PID)

# Or use different port
uvicorn HoloLoom.server.agentic_api:app --reload --port 8001
```

### Extension Won't Compile

**Error**: `Cannot find module 'vscode'`

**Solution**:
```bash
npm install
npm run compile
```

**Error**: `TypeScript errors in compilation`

**Solution**:
```bash
# Clean build
rm -rf out/
npm run compile
```

### Extension Not Appearing

**Check VS Code Version**:
- Help → About
- Must be 1.80.0 or later

**Reload Extension**:
- Command Palette (`Ctrl/Cmd + Shift + P`)
- Type: "Reload Window"

**Check Extension Host Logs**:
- Help → Toggle Developer Tools
- Console tab
- Look for errors

### HoloLoom Connection Failed

**Check Server Status**:
```bash
curl http://localhost:8000/health
```

**Check Extension Settings**:
1. File → Preferences → Settings (`Ctrl/Cmd + ,`)
2. Search: "promptly.hololoomUrl"
3. Verify: `http://localhost:8000`

**Check Firewall**:
- Ensure port 8000 is not blocked
- Try disabling firewall temporarily

**Check Server Logs**:
- Terminal running uvicorn
- Look for error messages

### CodeLens Not Showing

**Enable CodeLens in VS Code**:
1. Settings (`Ctrl/Cmd + ,`)
2. Search: "codelens"
3. Check: "Editor > Code Lens: Enabled"

**Verify Comment Pattern**:
```typescript
// NOTE: This should work ✓
//NOTE: This won't work (missing space) ✗
// note: This won't work (lowercase) ✗
```

**Force Refresh**:
- Command Palette → "Reload Window"

### Sidebar Not Appearing

**Check Activity Bar**:
- View → Appearance → Activity Bar (should be checked)

**Look for 🧠 Icon**:
- Should be at bottom of Activity Bar
- If not visible, extension didn't activate

**Check Extension Logs**:
- View → Output → Select "Promptly"
- Should see: "Promptly extension activated"

## Configuration

### Extension Settings

**Access Settings**:
- File → Preferences → Settings (`Ctrl/Cmd + ,`)
- Search: "Promptly"

**Available Settings**:

| Setting | Default | Description |
|---------|---------|-------------|
| `promptly.hololoomUrl` | `http://localhost:8000` | HoloLoom server URL |
| `promptly.claudeApiKey` | (empty) | Anthropic API key (optional) |
| `promptly.enableAutocomplete` | `true` | Slash command autocomplete |

### HoloLoom Server Configuration

**Config File**: `HoloLoom/config.py`

**Common Settings**:
```python
from HoloLoom.config import Config, MemoryBackend

config = Config.fast()  # BARE, FAST, or FUSED

# Memory backend
config.memory_backend = MemoryBackend.HYBRID  # INMEMORY, HYBRID, HYPERSPACE

# Performance
config.enable_zero_copy_embeddings = True  # 37x faster
config.zero_copy_cache_size = 10000  # Cache size

# Features
config.enable_agentic_reasoning = True
config.enable_alignment = True
```

**Apply Custom Config**:

Edit `HoloLoom/server/agentic_api.py` line 346:
```python
state.config = Config.fused()  # Change to your config
```

## Development Workflow

### Making Changes to Extension

1. **Edit TypeScript files** in `promptly-vscode/src/`

2. **Compile**:
   ```bash
   npm run compile
   # Or watch mode: npm run watch
   ```

3. **Reload Extension**:
   - In Extension Development Host window
   - Command Palette → "Reload Window"
   - Or close and press F5 again

### Making Changes to Server

1. **Edit Python files** in `HoloLoom/`

2. **Server auto-reloads** (if running with `--reload`)

3. **Test changes**:
   ```bash
   curl -X POST http://localhost:8000/api/remember \
     -H "Content-Type: application/json" \
     -d '{"content": "Test note", "context": {}}'
   ```

### Debugging Extension

**Set Breakpoints**:
1. Open `.ts` file in `promptly-vscode/src/`
2. Click left gutter to set breakpoint
3. Press F5 to launch
4. Trigger breakpoint in Extension Development Host

**View Logs**:
- Extension Output: View → Output → "Promptly"
- Developer Tools: Help → Toggle Developer Tools → Console

**Common Debug Points**:
- `extension.ts:8` - Extension activation
- `sidebarProvider.ts:49` - Sidebar messages
- `codeLensProvider.ts:36` - CodeLens provision

### Debugging Server

**View Logs**:
- Terminal running uvicorn shows all requests

**Test Endpoints**:
```bash
# Health check
curl http://localhost:8000/health

# Remember
curl -X POST http://localhost:8000/api/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Test", "context": {}}'

# Recall
curl -X POST http://localhost:8000/api/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "k": 5}'
```

## Testing

### Manual Testing Checklist

**Sidebar**:
- [ ] Sidebar opens when clicking 🧠 icon
- [ ] Quick Capture saves notes
- [ ] Search returns results
- [ ] Today's Notes shows captured notes

**CodeLens**:
- [ ] Shows on `// NOTE:` comments
- [ ] Shows on `// TODO:` comments
- [ ] Shows on `// FIXME:` comments
- [ ] Shows on `# NOTE:` (Python files)
- [ ] Clicking shows related memories
- [ ] "Capture this TODO" works

**Chat**:
- [ ] Opens with `Ctrl+Alt+P`
- [ ] `/remember` command works
- [ ] `/recall` command works
- [ ] `/help` shows commands
- [ ] Git commands work (in git repo)

**Integration**:
- [ ] Server health check passes
- [ ] Notes persist across restarts
- [ ] Search finds previously captured notes
- [ ] CodeLens updates after capturing notes

### Automated Testing

```bash
# Extension tests (TODO: not yet implemented)
cd promptly-vscode
npm test

# Server tests
cd HoloLoom
pytest HoloLoom/server/tests/
```

## Production Deployment

### Package Extension

```bash
cd promptly-vscode

# Install vsce
npm install -g vsce

# Package
vsce package
# Creates: promptly-0.1.0.vsix
```

### Install Packaged Extension

1. VS Code → Extensions (`Ctrl+Shift+X`)
2. Click `...` (More Actions)
3. "Install from VSIX..."
4. Select `promptly-0.1.0.vsix`

### Deploy Server

**Local Network**:
```bash
uvicorn HoloLoom.server.agentic_api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Production Server** (with SSL):
```bash
uvicorn HoloLoom.server.agentic_api:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem \
  --workers 4
```

**Docker** (TODO: create Dockerfile):
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "HoloLoom.server.agentic_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Next Steps

1. **Capture your first real note** in the sidebar
2. **Add some code comments** with `// NOTE:` and watch CodeLens appear
3. **Search your memories** to verify retrieval works
4. **Explore the chat** with slash commands (`/help`)
5. **Read the architecture docs** in `../HoloLoom/CLAUDE.md`

## Support

**Issues**: GitHub Issues

**Logs**:
- Extension: View → Output → "Promptly"
- Server: Terminal running uvicorn

**Community**: (Coming soon)

---

Happy coding with perfect memory! 🧠✨
