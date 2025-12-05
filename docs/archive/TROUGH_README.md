# Trough 🐷

**"Eats slop for breakfast!"**

A VS Code extension powered by HoloLoom that transforms broken AI-generated code into pristine, working perfection. Watch as noble piglets devour your slop with honor and leave thy trough sparkling clean.

---

## The Problem

AI code assistants often serve up code slop that:
- ❌ References non-existent functions/classes (hallucinations - but stinky ones!)
- ❌ Has type errors scattered like crumbs
- ❌ Missing imports or wrong API usage
- ❌ Looks plausible but doesn't compile/run
- ❌ Close to correct but fundamentally broken

**You need piglets who can feast on this slop and leave you with something pristine.**

---

## The Solution

Trough deploys honorable piglets who:
1. **Muck the Pen** - Index your entire codebase into a knowledge graph
2. **Sniff for Stink** - Detect hallucinations (non-existent references)
3. **Taste Test** - Run actual compilers (tsc, mypy, eslint)
4. **Munch with Gusto** - Fix code iteratively until pristine

---

## Features

### 🐷 Pig Out! (The Killer Feature)

**Command**: `Trough: Pig Out!` (`Ctrl+Shift+P`)

Watch piglets devour your slop in up to 5 glorious iterations:

```
🐷 Iteration 1/5: Munching away...
  ❌ Found 3 stinky hallucinations (nonexistent_function, FakeClass, wrong_import)
  ❌ Found 5 type errors
  → Sending more piglets...

🐷 Iteration 2/5: Munching away...
  ✅ No more stink!
  ❌ Found 2 type errors
  → Sending more piglets...

🐷 Iteration 3/5: Munching away...
  ✅ No hallucinations
  ✅ No errors
  ✨ Licked thy platter clean!
```

### 🧹 Muck the Pen

**Command**: `Trough: Index Workspace`

Catalog your entire pantry:
- Extract all functions, classes, methods, imports
- Build knowledge graph of relationships
- Enable semantic search over code entities
- Power hallucination detection

**Progress**: `Trough: Mucking the pen...`
**Success**: `✅ Pen mucked: 150 files indexed`

### 👃 Sniff for Stink

**Command**: `Trough: Detect Hallucinations`

Piglets have refined noses - they detect stinky code:

```python
# AI generated this slop:
result = authenticate_user(username, password)  # 👃 Something stinky!

# Trough suggests:
# "authenticate_user not found. Did you mean:"
#   - verify_user (similarity: 0.85)
#   - auth_user (similarity: 0.78)
#   - check_credentials (similarity: 0.65)
```

**Progress**: `Trough: Something stinky.... yum!`

### 👅 Taste Test

**Command**: `Trough: Verify Code`

Quality control through actual tasting (compilation):
- **Python**: AST syntax check, mypy type checking, pylint
- **TypeScript**: tsc compiler, eslint
- **JavaScript**: eslint

**Progress**: `Trough: Taste testing...`

Shows errors inline with VS Code diagnostics.

---

## Installation & Setup

### 1. Start the Piglet Server

```bash
cd mythRL
PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --reload --port 8000
```

### 2. Install VS Code Extension

```bash
cd trough
npm install
npm run compile
```

Then press `F5` to launch the extension development host.

### 3. Configure Trough

Open VS Code settings and configure:

```json
{
  "trough.serverUrl": "http://localhost:8000",
  "trough.maxPiglets": 5,
  "trough.reasoningMode": "verify"
}
```

---

## Usage

### Quick Start: Pig Out on AI Slop

1. **Generate broken AI code** (or paste some slop)
2. **Select the code** (or leave cursor in file for full file)
3. **Press `Ctrl+Shift+P`** → `Trough: Pig Out!`
4. **Watch piglets work their magic** 🐷
5. **Apply the fix** when presented with clean code

### Index Your Workspace First

For best results, let piglets learn your codebase:

1. **Open command palette** (`Ctrl+Shift+P`)
2. **Run** `Trough: Index Workspace`
3. **Select languages** to index (Python, TypeScript, JavaScript, or All)
4. **Wait** for "Pen mucked!" message

Now hallucination detection knows what's real!

### Manual Verification

**Detect Hallucinations Only**:
- `Ctrl+Shift+P` → `Trough: Detect Hallucinations`
- Piglets sniff out fake references

**Verify Code Only**:
- `Ctrl+Shift+P` → `Trough: Verify Code`
- Run actual compilers without fixing

---

## How It Works

### The 4-System Architecture

1. **Codebase Ingestion** (`HoloLoom/agentic/codebase_ingestion.py`)
   - AST-based parsing for Python
   - Regex-based parsing for TypeScript/JavaScript
   - NetworkX knowledge graph construction
   - ~50-100 files/second indexing speed

2. **Hallucination Detector** (`HoloLoom/agentic/hallucination_detector.py`)
   - Reference extraction (function calls, imports, class usage)
   - Existence verification against knowledge graph
   - Levenshtein distance similarity suggestions
   - Built-in function filtering (doesn't flag `print`, `len`, etc.)

3. **Code Verifier** (`HoloLoom/agentic/code_verification.py`)
   - Python: AST + mypy + pylint
   - TypeScript: tsc + eslint
   - Structured error parsing
   - Graceful degradation (works even if linters unavailable)

4. **Fix Loop** (`trough/src/FixSlopCommand.ts`)
   - Up to 5 iterations (configurable via `trough.maxPiglets`)
   - Hallucination detection → Verification → HoloLoom fix → Repeat
   - Shows diff before applying
   - Complete audit trail

### Iteration Example

```typescript
// AI-generated slop:
async function fetchUserData(id: number): User {
    const response = await fetchFromAPI(id);  // ❌ Doesn't exist!
    return response.user.data;  // ❌ Wrong type!
}

// After Iteration 1 (piglets munching...):
async function fetchUserData(id: number): User {
    const response = await getUserById(id);  // ✅ Fixed hallucination
    return response.user.data;  // ❌ Still wrong type
}

// After Iteration 2 (more munching...):
async function fetchUserData(id: number): User {
    const response = await getUserById(id);  // ✅ Real function
    return response.data;  // ✅ Correct type
}

// Result: ✨ Thy trough sparkles, Sire!
```

---

## Configuration

### Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `trough.serverUrl` | `http://localhost:8000` | Trough piglet server URL |
| `trough.maxPiglets` | `5` | Maximum fix iterations (1-10) |
| `trough.reasoningMode` | `verify` | Default piglet reasoning mode |

### Keyboard Shortcuts

| Command | Windows/Linux | Mac | Description |
|---------|---------------|-----|-------------|
| Pig Out! | `Ctrl+Shift+P` | `Cmd+Shift+P` | Fix AI slop |
| Ask Question | `Ctrl+Shift+Q` | `Cmd+Shift+Q` | Ask Trough anything |

---

## Delightful Phrases

Trough piglets speak with courtly manners! Every interaction uses randomized phrases:

### 🍽️ Eating Phrases (Progress)
- "We dine!"
- "Feast time!"
- "Let us eat cake!"
- "Watch your fingers!"
- "Oink oink, nom nom!"
- "Snouts to the trough!"
- "Release the hogs!"
- ...and more!

### ✨ Success Phrases (Completion)
- "Licked thy platter clean!"
- "Thy trough sparkles, Sire!"
- "Not a crumb remains, my Lord!"
- "Devoured with honor!"
- "Thy feast is complete!"
- "Sparkle-mode activated, my Liege!"
- ...and more courtly proclamations!

---

## Performance

Tested on medium Python/TypeScript projects:

| Operation | Duration | Notes |
|-----------|----------|-------|
| Index 50 Python files (5k LOC) | 15s | One-time cost |
| Index 50 TypeScript files (8k LOC) | 20s | One-time cost |
| Detect hallucinations | 100ms | Per file |
| Run Python verification | 500ms | mypy + pylint |
| Run TypeScript verification | 1s | tsc compile |
| Complete fix cycle (3 iterations) | 12s | End-to-end |

---

## Troubleshooting

### Piglets won't start

**Problem**: Extension activates but commands don't work

**Solutions**:
1. Check HoloLoom server is running: `curl http://localhost:8000/health`
2. Verify `trough.serverUrl` setting matches your server
3. Check browser console for errors (`Help` → `Toggle Developer Tools`)

### "Pen mucked" but hallucinations still detected

**Problem**: Indexed workspace but still getting false positives

**Solutions**:
1. Re-index workspace: `Trough: Index Workspace`
2. Check file extensions match your code (`.py`, `.ts`, `.js`)
3. Verify parser supports your language (Python/TypeScript/JavaScript only)

### Verification says "tsc not found"

**Problem**: TypeScript compiler not available

**Solutions**:
1. Install TypeScript globally: `npm install -g typescript`
2. Or install locally in project: `npm install --save-dev typescript`
3. System gracefully falls back to syntax checking only

### Piglets stuck munching

**Problem**: Fix iterations not completing

**Solutions**:
1. Check HoloLoom server logs for errors
2. Reduce `trough.maxPiglets` to 3 (faster iterations)
3. Try `verify` mode instead of `research` (faster reasoning)

---

## Contributing

Trough is powered by HoloLoom's agentic reasoning system. To contribute:

1. **Backend** (Python): `HoloLoom/agentic/` and `HoloLoom/server/`
2. **Frontend** (TypeScript): `trough/src/`
3. **Tests**: `tests/test_ai_slop_fixer_integration.py`

Run tests:
```bash
PYTHONPATH=. pytest tests/test_ai_slop_fixer_integration.py -v
```

---

## Architecture Diagrams

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Trough VS Code Extension                 │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ FixSlop      │  │ Verification  │  │  HoloLoom       │  │
│  │ Command      │→ │ Service       │→ │  Bridge         │  │
│  └──────────────┘  └───────────────┘  └─────────────────┘  │
│         ↓                                      ↓             │
└─────────────────────────────────────────────────────────────┘
                              ↓
                      HTTP (localhost:8000)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    HoloLoom FastAPI Server                   │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ Codebase     │  │ Hallucination │  │  Code           │  │
│  │ Indexer      │  │ Detector      │  │  Verifier       │  │
│  └──────────────┘  └───────────────┘  └─────────────────┘  │
│         ↓                  ↓                    ↓            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Agentic Orchestrator (VERIFY mode)          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Fix Loop Flow

```
┌─────────────────────────────────────────────────────────┐
│ Iteration 1: Munch, munch, munch... mmmm!              │
├─────────────────────────────────────────────────────────┤
│ 1. Extract code from editor                            │
│ 2. Detect hallucinations (Reference Extractor)         │
│ 3. Run verification (tsc / mypy / pylint)              │
│ 4. If clean → SUCCESS! ✨                              │
│ 5. If errors → Build fix prompt with context          │
│ 6. Send to HoloLoom (VERIFY mode)                      │
│ 7. Apply suggested fixes                               │
│ 8. Repeat (max 5 iterations)                           │
└─────────────────────────────────────────────────────────┘
```

---

## License

MIT License - See LICENSE file

---

## Credits

Built with love by the HoloLoom team 🐷

**Core Technologies**:
- HoloLoom Agentic Reasoning
- VS Code Extension API
- FastAPI
- NetworkX (Knowledge Graphs)
- AST Parsing (Python)
- TypeScript Compiler API

---

## What's Next?

### Planned Features

- 🌍 **More Languages**: Java, Rust, Go, C++
- 🔌 **LSP Integration**: Real-time error detection
- 🧠 **Pattern Learning**: Learn from successful fixes
- 📁 **Multi-File Refactoring**: Cross-file hallucination detection
- 👥 **Team Collaboration**: Share indexed codebases

### Join the Feast!

Star us on GitHub, report bugs, suggest features. The piglets are always hungry for improvement! 🐷✨

---

**Trough** - *Because AI slop deserves to be eaten with honor.*
