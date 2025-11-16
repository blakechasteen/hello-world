# Squad Quick Start Guide

Get started with Squad in 5 minutes!

## Step 1: Start HoloLoom Backend

```bash
# Terminal 1
cd HoloLoom/server
python agentic_api.py
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     HoloLoom server ready!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 2: Install Squad Extension

### Option A: From VSIX (Recommended)
```bash
cd squad
npm install
npm run compile
vsce package  # Creates squad-1.0.0.vsix
code --install-extension squad-1.0.0.vsix
```

### Option B: Development Mode
1. Open `squad/` folder in VS Code
2. Press F5 to launch Extension Development Host
3. Use extension in the new window

## Step 3: Verify Connection

1. Check status bar (bottom right): Should show "✓ Squad: Connected"
2. If you see "✗ Squad: Server Offline", restart backend from Step 1

## Step 4: Try Your First Command

### Example: Explain Code

1. Open a Python/TypeScript/JavaScript file
2. Select a function or class:
   ```python
   def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n-1) + fibonacci(n-2)
   ```
3. Right-click → **Squad: Explain This Code** (or press `Ctrl+Alt+E`)
4. View explanation in side panel

**You should see:**
- Detailed explanation of the algorithm
- Key concepts (recursion, base case)
- Performance analysis (O(2^n) time)
- Improvement suggestions (memoization)

### Example: Generate Tests

1. Select a function:
   ```typescript
   function add(a: number, b: number): number {
       return a + b;
   }
   ```
2. Right-click → **Squad: Generate Unit Tests** (or press `Ctrl+Alt+T`)
3. Choose framework: `jest`
4. View generated tests in new tab:
   ```typescript
   describe('add', () => {
       test('should add two positive numbers', () => {
           expect(add(2, 3)).toBe(5);
       });
       // ... more tests
   });
   ```

### Example: Find Similar Code

1. Select a code snippet:
   ```python
   for item in items:
       if condition(item):
           result.append(process(item))
   ```
2. Right-click → **Squad: Find Similar Code** (or press `Ctrl+Alt+F`)
3. View similar patterns across workspace

## Step 5: Explore Commands

Open Command Palette (`Ctrl+Shift+P`) and type "Squad" to see all commands:

- ✨ **Squad: Explain This Code** - Understand complex code
- 🔍 **Squad: Find Similar Code** - Discover patterns
- ✅ **Squad: Generate Unit Tests** - Auto-create tests
- 📝 **Squad: Add Documentation** - Generate docstrings
- 🔧 **Squad: Suggest Refactorings** - Improve code quality
- 👀 **Squad: Review Changes** - Analyze git diff
- 📊 **Squad: Show Statistics** - View usage stats
- 🗄️ **Squad: Index Workspace** - Build knowledge graph
- 🗑️ **Squad: Clear Cache** - Clear embeddings cache

## Step 6: Configure Settings

Open Settings (`Ctrl+,`) and search for "Squad":

**Essential Settings:**
```json
{
  "squad.hololoomUrl": "http://localhost:8000",  // Backend URL
  "squad.enableCache": true,                     // Fast responses
  "squad.reasoningMode": "verify",               // Balance speed/quality
}
```

**Optional Settings:**
```json
{
  "squad.autoIndexWorkspace": true,     // Auto-index on startup
  "squad.cacheMaxSize": 10000,          // 10k cached embeddings
  "squad.maxContextLines": 100,         // Context limit
  "squad.showInlineHints": true         // Inline suggestions
}
```

## Tips & Tricks

### 1. Keyboard Shortcuts
- `Ctrl+Alt+E` - Explain code
- `Ctrl+Alt+F` - Find similar
- `Ctrl+Alt+T` - Generate tests

### 2. Cache Performance
- First query: ~300ms
- Cached query: <1ms (100x faster!)
- View stats: `Squad: Show Statistics`

### 3. Context Menu
Right-click selected code to access all Squad features quickly.

### 4. Workspace Indexing
For large projects, index once:
```
Ctrl+Shift+P → "Squad: Index Workspace"
```
This builds a knowledge graph for better context understanding.

### 5. Reasoning Modes
Adjust quality/speed tradeoff:
- **direct**: Fastest (~150ms)
- **verify**: Balanced (~300ms) ← Default
- **research**: Thorough (~600ms)
- **plan_execute**: Multi-step (~750ms)

## Common Issues

### ❌ "Squad: Server Offline"
**Fix**: Start backend:
```bash
cd HoloLoom/server
python agentic_api.py
```

### ❌ "No active editor"
**Fix**: Open a code file and select text before running commands

### ❌ "Rate limit exceeded"
**Fix**: Wait a moment. Default: 60 requests/minute.

### ❌ Tree-sitter parser not found
**Fix**: Rebuild extension:
```bash
cd squad
npm install
npm run compile
```

## Next Steps

1. **Read full README**: [README.md](README.md)
2. **View architecture**: [Architecture section](README.md#-architecture)
3. **Explore examples**: [Examples section](README.md#-examples)
4. **Check troubleshooting**: [Troubleshooting section](README.md#-troubleshooting)

## Getting Help

- 📖 **Documentation**: [CLAUDE.md](../CLAUDE.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/squad/issues)
- 💬 **Community**: [Discord](#)

---

**Happy coding with Squad! 🚀**
