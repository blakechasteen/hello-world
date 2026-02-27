# Squad AI Slop Fixer

**"Take AI-built slop and make it work."**

A VS Code extension powered by HoloLoom that fixes broken AI-generated code through iterative verification, hallucination detection, and intelligent refactoring.

## The Problem

AI code assistants often generate code that:
- ❌ References non-existent functions/classes (hallucinations)
- ❌ Has type errors everywhere
- ❌ Missing imports or wrong API usage
- ❌ Looks plausible but doesn't compile/run
- ❌ Close to correct but fundamentally broken

**You need a system that understands your codebase and can fix AI slop automatically.**

## The Solution

Squad combines:
1. **Codebase Ingestion** - Indexes your entire project into a knowledge graph
2. **Hallucination Detection** - Catches references to non-existent entities
3. **Real Verification** - Runs actual compilers (tsc, mypy, etc.)
4. **Iterative Fixing** - Uses HoloLoom's agentic reasoning to fix code until it works

## Features

### 🎯 Fix AI Slop (The Killer Feature)

**Command**: `Squad: Fix AI Slop` (`Ctrl+Shift+F`)

Automatically fixes broken AI-generated code in up to 5 iterations:

```
Iteration 1:
  ❌ Found 3 hallucinations (nonexistent_function, FakeClass, wrong_import)
  ❌ Found 5 type errors
  → Fix and retry...

Iteration 2:
  ✅ No hallucinations
  ❌ Found 2 type errors
  → Fix and retry...

Iteration 3:
  ✅ No hallucinations
  ✅ No errors
  ✨ SUCCESS!
```

### 🔍 Deep Codebase Analysis

**Command**: `Squad: Index Workspace`

Indexes your entire workspace:
- Extracts all functions, classes, methods, imports
- Builds knowledge graph with relationships
- Enables semantic search over code entities
- Powers hallucination detection

### 🚨 Hallucination Detection

**Command**: `Squad: Detect Hallucinations`

Detects when code references non-existent entities:
```python
# AI generated this:
result = authenticate_user(username, password)  # ❌ Hallucination!

# Squad suggests:
# "authenticate_user not found. Did you mean:"
#   - verify_user (similarity: 0.85)
#   - auth_user (similarity: 0.78)
#   - check_credentials (similarity: 0.65)
```

### ✅ Real Code Verification

**Command**: `Squad: Verify Code`

Runs actual compilers and linters:
- **Python**: AST syntax check, mypy type checking, pylint
- **TypeScript**: tsc compiler, eslint
- **JavaScript**: eslint

Shows errors inline with VS Code diagnostics.

## Installation & Setup

### 1. Start HoloLoom Server

```bash
cd mythRL
PYTHONPATH=. uvicorn hololoom.server.agentic_api:app --reload --port 8000
```

### 2. Install VS Code Extension

```bash
cd squad
npm install
npm run compile
```

Then press `F5` to launch extension in debug mode.

### 3. Configure Server URL

In VS Code settings:
```json
{
  "squad.serverUrl": "http://localhost:8000",
  "squad.reasoningMode": "verify"
}
```

### 4. Index Your Workspace

1. Open your project in VS Code
2. Run command: `Squad: Index Workspace`
3. Select languages to index (Python, TypeScript, JavaScript)
4. Wait for indexing to complete (~10-30 seconds for medium projects)

## Usage

### Basic Workflow

1. **Paste AI-generated broken code** into editor
2. **Select the code** (or keep cursor in file for full-file fixing)
3. **Run**: `Squad: Fix AI Slop` (`Ctrl+Shift+F`)
4. **Watch** Squad iterate through fixes
5. **Review diff** and apply when ready

### Example: Fixing Hallucinated Code

**Before** (AI generated):
```python
def process_data(data):
    # Hallucination: these functions don't exist!
    cleaned = sanitize_input(data)
    validated = validate_schema(cleaned)
    result = transform_data(validated)
    return save_to_database(result)
```

**After 2 iterations** (Squad fixed):
```python
def process_data(data):
    # Squad found real functions in codebase
    cleaned = clean_user_input(data)  # ✅ Real function
    validated = check_data_schema(cleaned)  # ✅ Real function
    result = apply_transforms(validated)  # ✅ Real function
    return store_in_db(result)  # ✅ Real function
```

### Example: Fixing Type Errors

**Before** (TypeScript with type errors):
```typescript
function fetchUser(id: number): User {
    const response = await fetch(`/api/users/${id}`);  // ❌ Can't use await
    return response.json();  // ❌ Wrong return type
}
```

**After 1 iteration** (Squad fixed):
```typescript
async function fetchUser(id: number): Promise<User> {  // ✅ Added async + Promise
    const response = await fetch(`/api/users/${id}`);  // ✅ Now valid
    return response.json() as User;  // ✅ Correct type
}
```

## Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `Squad: Fix AI Slop` | `Ctrl+Shift+F` | **Main feature** - Iteratively fix broken code |
| `Squad: Index Workspace` | - | Index codebase for hallucination detection |
| `Squad: Verify Code` | - | Run compilers/linters on current file |
| `Squad: Detect Hallucinations` | - | Find references to non-existent entities |
| `Squad: Explain Selection` | - | Explain selected code |
| `Squad: Refactor Code` | - | Refactor with various strategies |
| `Squad: Generate Tests` | - | Generate unit tests |

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│      VS Code Extension (TypeScript)     │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │ FixSlopCommand                  │  │
│  │  - Orchestrates fix loop        │  │
│  │  - Manages iterations           │  │
│  │  - Shows progress/results       │  │
│  └───────────┬─────────────────────┘  │
│              │                          │
│  ┌───────────▼─────────────────────┐  │
│  │ VerificationService             │  │
│  │  - Detect hallucinations        │  │
│  │  - Run verification             │  │
│  │  - Show diagnostics             │  │
│  └───────────┬─────────────────────┘  │
│              │                          │
│  ┌───────────▼─────────────────────┐  │
│  │ HoloLoomBridge                  │  │
│  │  - HTTP client to server        │  │
│  └───────────┬─────────────────────┘  │
└──────────────┼──────────────────────────┘
               │ HTTP (localhost:8000)
               ▼
┌─────────────────────────────────────────┐
│    HoloLoom Python Server (FastAPI)    │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │ CodebaseIndexer                 │  │
│  │  - Parse Python/TS/JS files     │  │
│  │  - Extract entities             │  │
│  │  - Build knowledge graph        │  │
│  └─────────────────────────────────┘  │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │ HallucinationDetector           │  │
│  │  - Extract references           │  │
│  │  - Check existence in KG        │  │
│  │  - Find similar entities        │  │
│  └─────────────────────────────────┘  │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │ CodeVerifier                    │  │
│  │  - Run tsc, mypy, pylint        │  │
│  │  - Parse compiler output        │  │
│  │  - Return structured errors     │  │
│  └─────────────────────────────────┘  │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │ AgenticOrchestrator             │  │
│  │  - Agentic reasoning (VERIFY)   │  │
│  │  - Context-aware fixes          │  │
│  │  - Iterative refinement         │  │
│  └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Fix AI Slop Algorithm

```python
def fix_ai_slop(code, max_iterations=5):
    for iteration in range(max_iterations):
        # Step 1: Detect hallucinations
        hallucinations = detect_hallucinations(code)

        # Step 2: Run verification (compilers)
        errors = verify_code(code)

        # Step 3: Check if done
        if not hallucinations and not errors:
            return SUCCESS

        # Step 4: Build fix prompt with context
        prompt = build_fix_prompt(code, hallucinations, errors)

        # Step 5: Ask HoloLoom to fix
        fixed_code = hololoom.reason(prompt, mode=VERIFY)

        # Step 6: Update code for next iteration
        code = fixed_code

    return PARTIAL_SUCCESS  # Some issues may remain
```

## Performance

- **Codebase Indexing**: ~10-30 seconds for medium projects (5k-20k LOC)
- **Hallucination Detection**: ~50-200ms per file
- **Verification**: ~500ms-2s (depends on compiler)
- **Full Fix Loop**: ~10-30 seconds (3-5 iterations average)

## Supported Languages

| Language | Syntax Check | Type Check | Linter | Hallucination Detection |
|----------|--------------|------------|--------|------------------------|
| **Python** | ✅ (AST) | ✅ (mypy) | ✅ (pylint) | ✅ |
| **TypeScript** | ✅ (tsc) | ✅ (tsc) | ✅ (eslint) | ✅ |
| **JavaScript** | ✅ (eslint) | ❌ | ✅ (eslint) | ✅ |

## API Reference

### Python Server Endpoints

```python
# Ingest workspace
POST /ingest/workspace
{
  "workspace_path": "/path/to/project",
  "languages": ["python", "typescript"],
  "exclude_patterns": ["**/node_modules/**"]
}

# Detect hallucinations
POST /detect/hallucinations
{
  "code": "def foo(): return bar()",
  "language": "python"
}

# Verify code
POST /verify/code
{
  "code": "def foo(): return bar()",
  "language": "python",
  "check_syntax": true,
  "check_types": true
}

# Search codebase
POST /codebase/search
{
  "query": "authenticate",
  "entity_type": "function",
  "fuzzy": true
}
```

### VS Code Extension API

```typescript
// Fix AI slop
const result = await fixSlopCommand.fix(editor, {
    maxIterations: 5,
    verifyTypes: true,
    detectHallucinations: true
});

// Verify code
const verification = await verificationService.verifyCode(
    code,
    language,
    filePath
);

// Detect hallucinations
const hallucinations = await verificationService.detectHallucinations(
    code,
    language,
    filePath
);
```

## Limitations

1. **Language Support**: Currently Python, TypeScript, JavaScript only
2. **Compiler Dependency**: Requires compilers installed (`tsc`, `mypy`, etc.)
3. **Context Size**: Very large files (>10k LOC) may hit LLM context limits
4. **External Libraries**: Can't verify imports from external libraries without docs
5. **Max Iterations**: Limits to 5 iterations to prevent infinite loops

## Troubleshooting

### "Hallucination detector not initialized"

**Solution**: Run `Squad: Index Workspace` first

### "mypy not installed"

**Solution**: Install mypy: `pip install mypy`

### "tsc not available"

**Solution**: Install TypeScript: `npm install -g typescript`

### Server not responding

**Solution**: Check HoloLoom server is running on port 8000

## Future Enhancements

- [ ] Support more languages (Java, C++, Rust, Go)
- [ ] Integrate with language servers (LSP)
- [ ] Auto-index on workspace open
- [ ] Incremental indexing (only changed files)
- [ ] Better diff visualization
- [ ] Test generation integration
- [ ] Git commit integration
- [ ] Learning from fixes (pattern library)

## Credits

Built with:
- **HoloLoom** - Neural decision-making system
- **VS Code Extension API**
- **FastAPI** - Python web server
- **NetworkX** - Knowledge graph
- **TypeScript Compiler API**
- **Python AST**

## License

Part of the HoloLoom project.

---

**"The best AI assistant doesn't just generate code - it makes it work."**
