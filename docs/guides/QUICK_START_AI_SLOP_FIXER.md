# Quick Start: Squad AI Slop Fixer

Get up and running in 5 minutes.

## Step 1: Start the Server (2 minutes)

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL

# Start HoloLoom server
PYTHONPATH=. python -m uvicorn hololoom.server.agentic_api:app --reload --port 8000
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 2: Install Extension (1 minute)

Open VS Code:

```bash
cd squad
npm install
npm run compile
```

Press `F5` to launch extension development host.

## Step 3: Index Your Workspace (1 minute)

In the VS Code development host:

1. Open your project folder
2. Press `Ctrl+Shift+P` → Type "Squad: Index Workspace"
3. Select languages to index (e.g., "python" or "typescript")
4. Wait ~10-30 seconds

You'll see: `✅ Indexed workspace: Ingested 47 files with 342 entities`

## Step 4: Test with Broken Code (1 minute)

Create a test file with intentionally broken AI-generated code:

**Python Example** (`test_broken.py`):
```python
def process_user_data(user_id):
    # These functions don't exist (hallucinations)
    user = fetch_user_from_api(user_id)
    validated = validate_user_schema(user)
    transformed = apply_data_transforms(validated)
    return save_to_database(transformed)
```

**TypeScript Example** (`test_broken.ts`):
```typescript
function getUserProfile(id: number): User {
    const response = await fetch(`/api/users/${id}`);  // ❌ Can't use await without async
    return response.json();  // ❌ Wrong return type
}
```

## Step 5: Fix It! (30 seconds)

1. **Select the broken code** (or keep cursor anywhere in file)
2. Press `Ctrl+Shift+F` (or `Cmd+Shift+F` on Mac)
3. Watch Squad fix it:
   ```
   Squad: Fixing AI Slop
   Iteration 1/5... ⏳
   Iteration 2/5... ⏳
   ✅ Code fixed in 2 iterations!
   ```
4. Click **"Show Diff"** to review changes
5. Click **"Apply Fix"** to apply

## Commands Quick Reference

| Command | Shortcut | Use When |
|---------|----------|----------|
| **Fix AI Slop** | `Ctrl+Shift+F` | AI code is broken |
| **Verify Code** | - | Check for errors |
| **Detect Hallucinations** | - | Find fake functions |
| **Index Workspace** | - | First time setup |

## What to Expect

### First Run (Indexing)
```
Squad: Indexing workspace...
✅ Indexed workspace: Ingested 47 files with 342 entities
```

### Fixing Broken Code
```
Squad: Fixing AI Slop
Iteration 1/5...
  - Found 3 hallucinations
  - Found 5 type errors
  - Attempting fix...

Iteration 2/5...
  - Found 0 hallucinations ✅
  - Found 2 type errors
  - Attempting fix...

Iteration 3/5...
  - Found 0 hallucinations ✅
  - Found 0 type errors ✅

✅ Code fixed in 3 iterations!
```

### Review and Apply
```
[Show Diff] [Apply Fix] [Cancel]
```

## Troubleshooting

### ❌ Server not responding

**Check**: Is HoloLoom server running?
```bash
# Should show JSON response
curl http://localhost:8000/health
```

### ❌ Extension not loading

**Solution**:
1. Check VS Code output console
2. Ensure TypeScript compiled: `npm run compile`
3. Restart VS Code window

### ❌ "Codebase indexer not initialized"

**Solution**: Run `Squad: Index Workspace` first

### ❌ "mypy not installed" or "tsc not available"

**Solution**: Install compilers
```bash
# Python
pip install mypy pylint

# TypeScript
npm install -g typescript eslint
```

## Example Workflow

Here's a real-world example:

1. **ChatGPT generates code** for you (with hallucinations and errors)
2. **Paste into VS Code**
3. **Press `Ctrl+Shift+F`**
4. **Squad detects**:
   - 3 hallucinated function names
   - 5 type errors
   - 2 missing imports
5. **Squad fixes**:
   - Replaces hallucinations with real functions from your codebase
   - Fixes all type errors
   - Adds correct imports
6. **You review diff** and apply
7. **Code works!** ✨

## Next Steps

- Read full docs: [AI_SLOP_FIXER_README.md](./AI_SLOP_FIXER_README.md)
- Try other commands: `Squad: Detect Hallucinations`
- Index multiple projects
- Explore the web dashboard (coming soon)

---

**Happy fixing!** 🚀
