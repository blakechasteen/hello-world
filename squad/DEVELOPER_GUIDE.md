# Squad Developer Guide

**Extending and contributing to Squad**

---

## Table of Contents

1. [Architecture](#architecture)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Adding Commands](#adding-commands)
5. [Extending the Server](#extending-the-server)
6. [Testing](#testing)
7. [Debugging](#debugging)
8. [Contributing](#contributing)

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────┐
│           VS Code Extension (TypeScript)     │
│                                              │
│  ┌──────────────┐        ┌───────────────┐ │
│  │  Commands    │───────▶│  HoloLoom     │ │
│  │  (6 total)   │        │  Bridge       │ │
│  └──────────────┘        └───────┬───────┘ │
│                                   │ HTTP    │
│  ┌──────────────┐        ┌───────▼───────┐ │
│  │ Agent Panel  │◀───────│  Context      │ │
│  │  (Webview)   │        │  Provider     │ │
│  └──────────────┘        └───────────────┘ │
└──────────────────────────┬──────────────────┘
                           │
                           │ HTTP (localhost:8000)
                           ▼
┌─────────────────────────────────────────────┐
│         FastAPI Server (Python)              │
│                                              │
│  ┌──────────────┐        ┌───────────────┐ │
│  │  /query      │───────▶│  Weaving      │ │
│  │  /chat       │        │  Orchestrator │ │
│  │  /stats      │        └───────┬───────┘ │
│  └──────────────┘                │         │
│                                   │         │
│  ┌──────────────┐        ┌───────▼───────┐ │
│  │  Response    │◀───────│  HoloLoom     │ │
│  │  Formatting  │        │  Core         │ │
│  └──────────────┘        └───────────────┘ │
└─────────────────────────────────────────────┘
```

### Component Responsibilities

**TypeScript Extension:**
- Command registration and handling
- User input collection
- HTTP communication with server
- Result visualization (Agent Panel)
- Status bar management

**Python Server:**
- Query processing with HoloLoom
- Reasoning mode selection
- Response formatting
- Error handling
- Health monitoring

---

## Development Setup

### Prerequisites

```bash
# Node.js 16+ and Python 3.11+
node --version  # Should be 16+
python --version  # Should be 3.11+
```

### Initial Setup

```bash
# 1. Clone repository
cd /home/user/hello-world/squad

# 2. Install TypeScript dependencies
npm install

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Compile TypeScript
npm run compile

# 5. Start the server
PYTHONPATH=/home/user/hello-world python server.py
```

### Development Workflow

```bash
# Terminal 1: Watch TypeScript changes
npm run watch

# Terminal 2: Start server with reload
PYTHONPATH=/home/user/hello-world uvicorn server:app --reload

# VS Code: Press F5 to launch Extension Development Host
```

---

## Project Structure

```
squad/
├── src/                      # TypeScript source
│   ├── extension.ts          # Main extension entry point
│   ├── HoloLoomBridge.ts     # HTTP client
│   ├── AgentPanel.ts         # Webview UI
│   └── CodeContextProvider.ts # Context extraction
│
├── out/                      # Compiled JavaScript (gitignored)
│   └── *.js
│
├── .vscode/                  # VS Code configuration
│   ├── launch.json           # Debug configurations
│   ├── tasks.json            # Build tasks
│   ├── settings.json         # Workspace settings
│   └── extensions.json       # Recommended extensions
│
├── server.py                 # FastAPI server
├── test_squad.py             # Automated tests
├── start_and_test.sh         # Test automation script
│
├── package.json              # Extension manifest
├── tsconfig.json             # TypeScript config
└── requirements.txt          # Python dependencies
```

---

## Adding Commands

### Step 1: Register Command

Edit `package.json`:

```json
{
  "contributes": {
    "commands": [
      {
        "command": "squad.myNewCommand",
        "title": "Squad: My New Command",
        "icon": "$(symbol-method)"
      }
    ],
    "keybindings": [
      {
        "command": "squad.myNewCommand",
        "key": "ctrl+shift+n",
        "mac": "cmd+shift+n"
      }
    ]
  }
}
```

### Step 2: Implement Command

Edit `src/extension.ts`:

```typescript
function registerCommands(context: vscode.ExtensionContext) {
    // ... existing commands ...

    // Add new command
    context.subscriptions.push(
        vscode.commands.registerCommand('squad.myNewCommand', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            // Get input
            const input = await vscode.window.showInputBox({
                prompt: 'Enter your input',
                placeHolder: 'Type something...'
            });

            if (!input) return;

            // Get context
            const codeContext = contextProvider.getCurrentContext();

            // Execute query
            await executeQuery(input, codeContext, 'direct');
        })
    );
}
```

### Step 3: Compile and Test

```bash
npm run compile
# Press F5 to test in Extension Development Host
```

---

## Extending the Server

### Adding a New Endpoint

Edit `server.py`:

```python
@app.post("/my-endpoint")
async def handle_my_endpoint(request: MyRequest):
    """
    My custom endpoint
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Not initialized")

    try:
        # Your logic here
        result = await orchestrator.weave(query)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

### Adding a New Request Model

```python
class MyRequest(BaseModel):
    """My custom request"""
    input: str
    options: Optional[Dict[str, Any]] = None
```

### Testing the Endpoint

```bash
curl -X POST http://localhost:8000/my-endpoint \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "options": {}}'
```

---

## Testing

### Running Tests

```bash
# Automated test suite
python test_squad.py

# Or use the convenience script
./start_and_test.sh
```

### Writing New Tests

Edit `test_squad.py`:

```python
def test_my_feature(self) -> bool:
    """Test my new feature"""
    self.log("Testing my feature...")

    try:
        response = requests.post(
            f"{self.base_url}/my-endpoint",
            json={"input": "test"},
            timeout=30
        )

        if response.status_code == 200:
            self.log("✅ Test passed", "SUCCESS")
            self.results.append({
                "test": "my_feature",
                "status": "pass"
            })
            return True
        else:
            self.log("❌ Test failed", "ERROR")
            return False

    except Exception as e:
        self.log(f"❌ Error: {e}", "ERROR")
        return False
```

### Test Coverage

Current tests:
- ✅ Health check
- ✅ Query DIRECT mode
- ✅ Query VERIFY mode
- ✅ Chat endpoint
- ✅ Stats endpoint

Add tests for:
- New endpoints
- Error conditions
- Edge cases
- Performance benchmarks

---

## Debugging

### Debugging the Extension

1. **Set breakpoints** in TypeScript code
2. **Press F5** to launch debugger
3. **Use the extension** in the new window
4. **Breakpoints will trigger** in original VS Code window

**Debug Console:**
```typescript
console.log('Debug message');  // Shows in Debug Console
```

### Debugging the Server

1. **Add debugpy to server:**
```python
import debugpy
debugpy.listen(5678)
```

2. **Start server with debugging:**
```bash
python server.py
```

3. **Attach debugger** in VS Code:
   - Run → "Attach to Server"
   - Set breakpoints in Python code

**Logging:**
```python
logger.info("Info message")
logger.error("Error message", exc_info=True)
```

### Common Issues

**Extension doesn't activate:**
- Check `activationEvents` in package.json
- Check for TypeScript compilation errors
- Reload window: `Ctrl+Shift+P` → "Reload Window"

**Server returns 500 errors:**
- Check server logs in terminal
- Add try-except blocks
- Test with curl first

**Changes not reflecting:**
- TypeScript: Run `npm run compile`
- Server: Restart with `--reload` flag
- Extension: Reload Extension Development Host

---

## Contributing

### Code Style

**TypeScript:**
- Use 4-space indentation
- Use async/await (not callbacks)
- Type all parameters and returns
- Document public functions

```typescript
/**
 * Execute a query with Squad
 * @param question The question to ask
 * @param context Code context
 * @param mode Reasoning mode
 */
async function executeQuery(
    question: string,
    context: CodeContext,
    mode: string = 'verify'
): Promise<void> {
    // Implementation
}
```

**Python:**
- Follow PEP 8
- Use type hints
- Document with docstrings
- Use async/await

```python
async def handle_query(request: QueryRequest) -> QueryResponse:
    """
    Handle query request with agentic reasoning

    Args:
        request: The query request

    Returns:
        QueryResponse with results

    Raises:
        HTTPException: If query fails
    """
    # Implementation
```

### Git Workflow

1. **Create feature branch:**
```bash
git checkout -b feature/my-feature
```

2. **Make changes and test:**
```bash
npm run compile
python test_squad.py
```

3. **Commit with descriptive messages:**
```bash
git commit -m "Add feature: description

- What changed
- Why it changed
- How to test"
```

4. **Push and create PR:**
```bash
git push origin feature/my-feature
```

---

## Performance Optimization

### TypeScript

**Minimize HTTP calls:**
```typescript
// ❌ Bad: Multiple calls
const result1 = await bridge.query(q1);
const result2 = await bridge.query(q2);

// ✅ Good: Batch if possible
const results = await bridge.batchQuery([q1, q2]);
```

**Cache results:**
```typescript
const cache = new Map<string, AgenticResult>();

if (cache.has(question)) {
    return cache.get(question);
}
```

### Python

**Use async for I/O:**
```python
# ✅ Good: Non-blocking
async with orchestrator.weave(query) as result:
    return result
```

**Profile slow queries:**
```python
import time
start = time.time()
result = await orchestrator.weave(query)
duration = time.time() - start
logger.info(f"Query took {duration:.2f}s")
```

---

## Release Process

### Version Bumping

1. **Update version** in `package.json`
2. **Update CHANGELOG.md**
3. **Commit changes:**
```bash
git commit -m "Release v0.2.0"
git tag v0.2.0
```

### Building VSIX

```bash
npm install -g @vscode/vsce
vsce package
# Creates squad-0.2.0.vsix
```

### Testing Release

```bash
code --install-extension squad-0.2.0.vsix
# Test in clean VS Code instance
```

---

## Useful Resources

- **VS Code Extension API:** https://code.visualstudio.com/api
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **HoloLoom Docs:** `/home/user/hello-world/CLAUDE.md`
- **TypeScript Handbook:** https://www.typescriptlang.org/docs/

---

**Happy developing!** 🚀
