# Trough Quick Start Guide 🐷

**"From Slop to Sparkle in 5 Minutes!"**

Get your piglets munching on AI slop in under 5 minutes.

---

## Prerequisites

- Python 3.8+ installed
- Node.js 16+ installed
- VS Code 1.85+ installed
- HoloLoom repository cloned

---

## Step 1: Start the Piglet Server (2 minutes)

```bash
# Navigate to mythRL directory
cd mythRL

# Install Python dependencies (if not already done)
pip install fastapi uvicorn networkx

# Start the server
PYTHONPATH=. uvicorn hololoom.server.agentic_api:app --reload --port 8000
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Application startup complete.
```

**Test it works**:
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","service":"HoloLoom Agentic API"}
```

---

## Step 2: Install Trough Extension (2 minutes)

```bash
# Navigate to trough directory
cd trough

# Install dependencies
npm install

# Compile TypeScript
npm run compile
```

**Expected output**:
```
> trough@1.0.0 compile
> tsc -p ./
```

(No errors = success!)

---

## Step 3: Launch Extension (30 seconds)

### Option A: Development Mode

1. Open `trough/` folder in VS Code
2. Press `F5` to launch Extension Development Host
3. New VS Code window opens with Trough loaded

### Option B: Package and Install

```bash
# Install vsce (VS Code Extension packager)
npm install -g @vscode/vsce

# Package extension
vsce package

# Install the .vsix file
code --install-extension trough-1.0.0.vsix
```

---

## Step 4: First Test - Pig Out! (30 seconds)

### Create Test File

Create `test.py` with broken AI-generated code:

```python
def authenticate_user(username, password):
    # These functions don't exist (hallucinations)!
    user = fetch_user_from_database(username)
    if verify_password_hash(password, user.hash):
        return create_session_token(user)
    return None
```

### Run Trough

1. **Select all the code** (Ctrl+A)
2. **Press `Ctrl+Shift+P`**
3. **Watch piglets munch!** 🐷

You'll see:
```
Trough: We dine!
  ❌ Found 3 hallucinations
  → Piglet 1/5 munching away...
```

4. **Review the diff** when prompted
5. **Click "Apply Fix"**
6. **Success message**: "Thy trough sparkles, Sire!" ✨

---

## Step 5: Index Your Workspace (Optional but Recommended)

For hallucination detection to know what's real in YOUR codebase:

1. **Open command palette** (`Ctrl+Shift+P`)
2. **Run**: `Trough: Index Workspace`
3. **Select**: `all` (or choose specific languages)
4. **Wait for**: `✅ Pen mucked: 150 files indexed`

Now Trough knows your entire codebase!

---

## Verify Everything Works

### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

**Expected**: `{"status":"healthy",...}`

### Test 2: Index Endpoint

```bash
curl -X POST http://localhost:8000/ingest/workspace \
  -H "Content-Type: application/json" \
  -d '{"workspace_path": ".", "languages": ["python"]}'
```

**Expected**: JSON with `total_files`, `total_entities`, etc.

### Test 3: Hallucination Detection

Create `test_hallucination.py`:
```python
x = nonexistent_function()  # Should be detected
```

Run `Trough: Detect Hallucinations` command.

**Expected**: Warning about `nonexistent_function` not found.

### Test 4: Verification

Create `test_syntax.ts`:
```typescript
function foo() {
    return bar(  // Missing closing paren
}
```

Run `Trough: Verify Code` command.

**Expected**: Syntax error shown in diagnostics.

---

## Configuration (Optional)

Open VS Code settings (`Ctrl+,`) and search for "trough":

```json
{
  // Piglet server URL (default: http://localhost:8000)
  "trough.serverUrl": "http://localhost:8000",

  // Maximum fix iterations (default: 5, range: 1-10)
  "trough.maxPiglets": 5,

  // Reasoning mode (default: verify)
  // Options: direct, verify, research, plan_execute
  "trough.reasoningMode": "verify"
}
```

---

## Troubleshooting

### "Connection refused" errors

**Problem**: Extension can't reach server

**Fix**:
1. Check server is running: `curl http://localhost:8000/health`
2. Verify `trough.serverUrl` matches your server URL
3. Try restarting server

### "No active editor" warning

**Problem**: Trying to run commands without open file

**Fix**: Open a code file first, then run Trough commands

### TypeScript compilation errors

**Problem**: `npm run compile` fails

**Fix**:
1. Delete `node_modules` and `package-lock.json`
2. Run `npm install` again
3. Run `npm run compile`

### Piglets not munching (stuck)

**Problem**: Fix loop hangs

**Fix**:
1. Check HoloLoom server logs for errors
2. Restart server
3. Reduce `trough.maxPiglets` to 3

---

## Next Steps

### Learn More

- Read full [TROUGH_README.md](TROUGH_README.md) for complete documentation
- Check [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) for architecture details
- Run integration tests: `pytest tests/test_ai_slop_fixer_integration.py -v`

### Try Advanced Features

- **Ask Questions**: `Trough: Ask Question` (Ctrl+Shift+Q)
- **Explain Selection**: Select code → `Trough: Explain Selection`
- **Generate Tests**: `Trough: Generate Tests`

### Configure for Your Team

- Share `trough.serverUrl` for team server
- Set `trough.maxPiglets` based on your patience level
- Index workspace on project setup

---

## Common Workflows

### Workflow 1: Fix AI-Generated Code

```
1. Paste AI-generated code into editor
2. Ctrl+Shift+P → "Trough: Pig Out!"
3. Review diff
4. Apply fix
5. ✨ Thy trough sparkles!
```

### Workflow 2: Verify Before Commit

```
1. Make changes to file
2. Ctrl+Shift+P → "Trough: Verify Code"
3. Fix any errors shown
4. Commit clean code
```

### Workflow 3: Understand Unfamiliar Code

```
1. Select mysterious code block
2. Ctrl+Shift+Q → Ask: "What does this do?"
3. Read Trough's explanation
4. Profit!
```

---

## Performance Tips

### Speed Up Indexing

- Index only languages you use: Select `python` or `typescript`, not `all`
- Exclude `node_modules`, `.venv`, `build/` directories
- Re-index only when codebase changes significantly

### Speed Up Fixes

- Use `verify` mode (default) instead of `research` for faster reasoning
- Reduce `trough.maxPiglets` to 3 for quick iterations
- Index workspace first so hallucination detection is faster

### Reduce Server Load

- Keep server running (don't restart between uses)
- Use caching (HoloLoom has built-in query cache)
- Close extension development host when not testing

---

## You're Ready! 🐷

Your piglets are ready to feast on AI slop and leave thy trough sparkling!

**Quick Reference**:
- `Ctrl+Shift+P`: Pig Out! (fix AI slop)
- `Ctrl+Shift+Q`: Ask Question
- `Ctrl+Shift+P` → Type "Trough" to see all commands

**Remember**: Great code isn't written, it's devoured with honor! ✨

---

**Need Help?**

- Check [TROUGH_README.md](TROUGH_README.md) for detailed docs
- File issues on GitHub
- Ask the piglets - they're always happy to help! 🐷
