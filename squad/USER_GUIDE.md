# Squad User Guide

**Your AI-powered coding assistant with agentic reasoning**

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Commands](#commands)
3. [Reasoning Modes](#reasoning-modes)
4. [Configuration](#configuration)
5. [Tips & Tricks](#tips--tricks)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- VS Code 1.85.0 or higher
- Python 3.11+
- HoloLoom dependencies installed

### Installation

1. **Install the extension:**
   ```bash
   # From squad/ directory
   code --install-extension squad-0.1.0.vsix
   ```

2. **Start the Squad server:**
   ```bash
   cd /home/user/hello-world/squad
   PYTHONPATH=/home/user/hello-world python server.py
   ```

3. **Verify connection:**
   - Look for `✅ Squad` in the status bar (bottom right)
   - If you see `⚠️ Squad`, the server isn't running

---

## Commands

### 1. **Ask Question** (`Ctrl+Shift+Q`)

Ask Squad anything about your code or programming concepts.

**Example:**
```
Ctrl+Shift+Q → "What is Thompson Sampling?"
```

**Use when:**
- You need to understand a concept
- You want code examples
- You have general programming questions

---

### 2. **Explain Selection** (`Ctrl+Shift+E`)

Get a detailed explanation of selected code.

**How to use:**
1. Select code in your editor
2. Press `Ctrl+Shift+E` (or right-click → "Squad: Explain Selection")
3. View explanation in the Agent Panel

**Use when:**
- Reading unfamiliar code
- Reviewing pull requests
- Learning new patterns

---

### 3. **Suggest Fix**

Get AI-powered suggestions to fix errors and warnings.

**How to use:**
1. Open a file with errors (red squiggly lines)
2. Run: `Ctrl+Shift+P` → "Squad: Suggest Fix"
3. Review suggestions in the Agent Panel

**Use when:**
- You have TypeScript/JavaScript errors
- You see linter warnings
- You need quick fixes

---

### 4. **Refactor Code**

Intelligent code transformations with verification.

**Available refactorings:**
- Extract function
- Simplify logic
- Add error handling
- Optimize performance
- Add type annotations
- Custom (describe your own)

**How to use:**
1. Select code to refactor
2. Run: `Ctrl+Shift+P` → "Squad: Refactor Code"
3. Choose refactoring type
4. Review and apply changes

---

### 5. **Generate Tests**

Auto-generate tests for your code.

**Test types:**
- Unit tests
- Integration tests
- Edge cases
- All of the above

**How to use:**
1. Select function/class to test
2. Run: `Ctrl+Shift+P` → "Squad: Generate Tests"
3. Choose test type
4. Tests appear in Agent Panel

---

### 6. **Open Agent Panel**

View Squad's reasoning process in real-time.

**What you see:**
- Response text
- Confidence score (0-100%)
- Reasoning mode used
- Step-by-step thought process
- Duration (milliseconds)
- Verification results (if available)

**How to open:**
- Click `✅ Squad` in status bar
- Run: `Ctrl+Shift+P` → "Squad: Open Agent Panel"

---

## Reasoning Modes

Squad uses 4 different reasoning modes depending on your query:

### **DIRECT** (~150ms)
- **Best for:** Simple factual questions
- **Example:** "What is async/await?"
- **Speed:** Fastest
- **Confidence:** Good for straightforward queries

### **VERIFY** (~600ms) [Default]
- **Best for:** Claims needing verification
- **Example:** "Is this implementation correct?"
- **Speed:** Medium
- **Confidence:** Higher due to verification loop

### **RESEARCH** (~900ms)
- **Best for:** Open-ended exploration
- **Example:** "What are the tradeoffs of React vs Vue?"
- **Speed:** Slower
- **Confidence:** Comprehensive analysis

### **PLAN_EXECUTE** (~750ms)
- **Best for:** Multi-step tasks
- **Example:** "Refactor this code to use async/await"
- **Speed:** Medium-slow
- **Confidence:** High for complex tasks

---

## Configuration

### Settings

Access via: `Ctrl+,` → Search "squad"

#### `squad.serverUrl`
- **Type:** string
- **Default:** `http://localhost:8000`
- **Description:** Squad server URL

#### `squad.reasoningMode`
- **Type:** string
- **Options:** `direct`, `verify`, `research`, `plan_execute`
- **Default:** `verify`
- **Description:** Default reasoning mode

#### `squad.maxSteps`
- **Type:** number
- **Default:** `5`
- **Range:** 1-10
- **Description:** Maximum reasoning steps

#### `squad.showReasoningSteps`
- **Type:** boolean
- **Default:** `true`
- **Description:** Show Agent Panel automatically

---

## Tips & Tricks

### 🚀 Productivity Tips

1. **Use keyboard shortcuts:**
   - `Ctrl+Shift+Q` for quick questions
   - `Ctrl+Shift+E` for code explanations

2. **Start with DIRECT mode** for simple queries
   - Configure in settings if you want fast responses by default

3. **Watch the status bar:**
   - `✅ Squad` = Ready to use
   - `⚠️ Squad` = Server down (click to troubleshoot)

4. **Keep Agent Panel open:**
   - Dock it to the side
   - See all reasoning steps
   - Learn how Squad thinks

### 💡 Best Practices

1. **Be specific in questions:**
   - ❌ "Fix my code"
   - ✅ "Explain why this async function isn't awaiting properly"

2. **Select relevant code:**
   - Don't select entire files for explanations
   - Select the specific function/block you're interested in

3. **Use the right command:**
   - Questions → Ask Question
   - Code understanding → Explain Selection
   - Errors → Suggest Fix
   - Improvements → Refactor Code

---

## Troubleshooting

### Status Bar Shows ⚠️ Squad

**Problem:** Server not responding

**Solutions:**
1. Click the status bar → "Open Terminal"
2. Or manually start server:
   ```bash
   cd /home/user/hello-world/squad
   PYTHONPATH=/home/user/hello-world python server.py
   ```

---

### Commands Not Appearing

**Problem:** Extension not activated

**Solutions:**
1. Reload VS Code: `Ctrl+Shift+P` → "Reload Window"
2. Check extension is enabled: Extensions → Search "Squad"
3. Check VS Code version (needs 1.85.0+)

---

### Low Confidence Scores

**Problem:** Squad shows ❌ or ⚠️ confidence

**Solutions:**
1. **Try a different mode:** Switch to `verify` or `research`
2. **Add more context:** Select more code or provide details
3. **Rephrase question:** Be more specific

---

### Slow Responses

**Problem:** Queries taking >5 seconds

**Solutions:**
1. **Switch to DIRECT mode** for faster responses
2. **Reduce max_steps** in settings (try 3 instead of 5)
3. **Check server load:** One query at a time works best

---

### Error: "Cannot connect to server"

**Problem:** Network connection failed

**Solutions:**
1. **Verify server is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check port 8000 isn't in use:**
   ```bash
   lsof -i :8000
   ```

3. **Update server URL** in settings if using different port

---

## Getting Help

### Resources

- **README:** `/home/user/hello-world/squad/README.md`
- **Quick Start:** `/home/user/hello-world/squad/QUICKSTART.md`
- **Developer Guide:** `/home/user/hello-world/squad/DEVELOPER_GUIDE.md`

### Support

- Check server logs for errors
- Run test suite: `./start_and_test.sh`
- Open Agent Panel to see detailed error messages

---

## Example Workflows

### Workflow 1: Understanding New Code

```
1. Open file with unfamiliar code
2. Select a function
3. Ctrl+Shift+E (Explain Selection)
4. Read explanation in Agent Panel
5. Ask follow-up: Ctrl+Shift+Q → "How does X work?"
```

### Workflow 2: Fixing Errors

```
1. TypeScript shows errors (red squiggles)
2. Run: "Squad: Suggest Fix"
3. Review suggested fix
4. Apply changes
5. Verify: errors disappear
```

### Workflow 3: Code Review

```
1. Pull colleague's branch
2. Open changed files
3. For each change:
   - Select code
   - Ctrl+Shift+E
   - Review Squad's analysis
4. Ask questions: Ctrl+Shift+Q
5. Provide feedback
```

---

**Happy coding with Squad!** 🤖✨
