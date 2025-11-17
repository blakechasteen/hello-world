# LSP Server Setup Guide (v2.0.0)

**Published: 2025-11-17**
**Last Updated: 2025-11-17**

A complete guide to setting up the Promptly + HoloLoom LSP integration.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Starting the Extension](#starting-the-extension)
5. [Verification & Testing](#verification--testing)
6. [Common Setup Problems](#common-setup-problems)

---

## Prerequisites

Before installing Promptly with LSP, ensure you have:

### 1. VS Code

**Minimum:** VS Code 1.80.0 or newer
**Recommended:** Latest stable version

Check your version:
- Open VS Code
- Press `Ctrl+Shift+P` → type "About"
- Look for version number

### 2. Python 3.8+

**Minimum:** Python 3.8
**Recommended:** Python 3.11+

Check if installed:
```bash
python3 --version
# Output should be 3.8 or newer
```

**Install Python:**
- **macOS:** `brew install python@3.11`
- **Ubuntu/Debian:** `sudo apt-get install python3.11 python3.11-venv python3.11-dev`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)
- **Fedora/RHEL:** `sudo dnf install python3.11 python3.11-devel`

### 3. Node.js 16+

**Required for:** Building the VS Code extension

Check if installed:
```bash
node --version
npm --version
```

**Install Node:**
- **All platforms:** Download from [nodejs.org](https://nodejs.org/)
- **macOS:** `brew install node`
- **Ubuntu/Debian:** `sudo apt-get install nodejs npm`

### 4. HoloLoom (Python package)

**Required for:** LSP server runtime

Check if installed:
```bash
python3 -c "import HoloLoom; print(HoloLoom.__version__)"
```

**If not installed:**

Option A (from PyPI):
```bash
pip install HoloLoom
```

Option B (from source):
```bash
git clone https://github.com/yourrepo/HoloLoom.git
cd HoloLoom
pip install -e .
```

Option C (development mode with all features):
```bash
git clone https://github.com/yourrepo/HoloLoom.git
cd HoloLoom
pip install -e ".[dev,torch,spacy]"
python -m spacy download en_core_web_sm
```

---

## Installation Steps

### Step 1: Clone/Download Promptly Repository

```bash
# Clone the repository
git clone https://github.com/yourrepo/HoloLoom.git
cd HoloLoom/promptly-vscode

# Or, if you already have it
cd promptly-vscode
```

### Step 2: Install Node Dependencies

```bash
# Install npm packages
npm install

# Expected output:
# added 45 packages, and audited 46 packages
```

### Step 3: Compile TypeScript

```bash
# Compile TypeScript to JavaScript
npm run compile

# Expected output:
# Successfully compiled X files
```

### Step 4: (Optional) Development Setup

If you plan to modify the extension:

```bash
# Install TypeScript compiler
npm install -g typescript

# Enable watch mode
npm run watch
# This watches for file changes and auto-compiles
```

---

## Configuration

### Default Configuration (Auto-Detect)

The extension works with **zero configuration** in most cases:

1. Opens VS Code
2. Extension auto-detects Python on PATH
3. Extension auto-detects HoloLoom package
4. LSP server starts automatically
5. Status bar shows "HoloLoom LSP: Connected" ✅

### Custom Configuration

If auto-detection fails, configure manually:

**Open VS Code Settings:**
- Windows/Linux: `Ctrl+,`
- Mac: `Cmd+,`

**Search for "HoloLoom"** and configure:

#### `hololoom.lsp.enabled` (boolean)
- **Default:** `true`
- **Description:** Enable LSP client for HoloLoom
- **Example:** `true`

```json
"hololoom.lsp.enabled": true
```

#### `hololoom.lsp.pythonPath` (string)
- **Default:** Auto-detect from PATH
- **Description:** Full path to Python 3.8+ executable
- **Example:** `"/usr/local/bin/python3.11"`

```json
"hololoom.lsp.pythonPath": "/usr/local/bin/python3.11"
```

**Find your Python path:**
```bash
# macOS/Linux
which python3
# Output: /usr/local/bin/python3

# or
python3 -c "import sys; print(sys.executable)"
# Output: /usr/local/bin/python3.11
```

#### `hololoom.lsp.hololoomPath` (string)
- **Default:** Auto-detect from Python site-packages
- **Description:** Full path to HoloLoom installation
- **Example:** `"/home/user/projects/HoloLoom"`

```json
"hololoom.lsp.hololoomPath": "/home/user/projects/HoloLoom"
```

**Find HoloLoom path:**
```bash
# Show installation path
python3 -c "import HoloLoom; print(HoloLoom.__file__)"
# Output: /usr/local/lib/python3.11/site-packages/HoloLoom/__init__.py

# Extract directory
python3 -c "import HoloLoom; import os; print(os.path.dirname(HoloLoom.__file__))"
# Output: /usr/local/lib/python3.11/site-packages/HoloLoom
```

#### `hololoom.lsp.logLevel` (string)
- **Default:** `"info"`
- **Options:** `"debug"`, `"info"`, `"warning"`, `"error"`
- **Description:** Verbosity of LSP server logs

```json
"hololoom.lsp.logLevel": "debug"  // For troubleshooting
```

#### `hololoom.lsp.serverArgs` (array)
- **Default:** `[]`
- **Description:** Additional arguments for LSP server
- **Example:** `["--cache-size", "1000"]`

```json
"hololoom.lsp.serverArgs": ["--cache-size", "5000"]
```

### Complete Configuration Example

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.hololoomPath": "/home/user/projects/HoloLoom",
  "hololoom.lsp.logLevel": "info",
  "hololoom.lsp.serverArgs": ["--cache-size", "2000"]
}
```

---

## Starting the Extension

### 1. Development Mode (Testing)

If you want to test changes before packaging:

```bash
# In promptly-vscode directory
npm run watch
```

Then in VS Code:
- Press **F5** to start Extension Development Host
- New VS Code window opens with extension loaded
- Extension automatically connects to LSP server

### 2. Production Mode (Using Released Extension)

**Option A: Install from VS Code Marketplace**
1. Open VS Code
2. Extensions panel: `Ctrl+Shift+X`
3. Search for "Promptly"
4. Click "Install"
5. Reload VS Code when prompted

**Option B: Install from .vsix File**
```bash
# Build extension
npm run vscode:prepublish

# Install .vsix (from VS Code)
code --install-extension promptly-2.0.0.vsix
```

---

## Verification & Testing

### 1. Check LSP Connection Status

**Status Bar Check:**
- Look at bottom-right of VS Code
- Should show "🧠 HoloLoom LSP: Connected" ✅
- If shows "Disconnected" ❌, see troubleshooting

**Output Panel Check:**
```
1. Open: Ctrl+Shift+U
2. Select: "HoloLoom Language Server" from dropdown
3. Look for: "LSP server started successfully"
```

### 2. Test Memory Operations

**Test /remember:**
```
1. Press Ctrl+Alt+P (open chat)
2. Type: /remember "I'm testing the LSP migration"
3. Press Enter
4. Should show: ✅ Saved to HoloLoom memory
```

**Test /recall:**
```
1. Press Ctrl+Alt+P (open chat)
2. Type: /recall "testing"
3. Press Enter
4. Should show: results with your memory
```

### 3. Test Sidebar

**Open Sidebar:**
1. Click 🧠 brain icon in Activity Bar (left)
2. Or: Ctrl+Shift+P → "Show HoloLoom"

**Test Quick Capture:**
1. Type text in "Quick Capture" box
2. Click "💾 Remember"
3. Should show success message

**Test Search:**
1. Type in "Search Memory" box
2. Click "Search"
3. Should show results

### 4. Test CodeLens

**Add a comment with inline suggestions:**
```python
# NOTE: Testing LSP CodeLens integration
def hello():
    pass
```

**Verify:**
- Should see 💡 "CodeLens" indicator above comment
- Click to see suggestions
- Click "📝 Capture" to save as memory

### 5. Full Integration Test

```bash
# Run all tests
npm run test

# Expected: All tests pass (some may be marked as skipped if LSP unavailable)
```

---

## Common Setup Problems

### Problem 1: "Python not found" or "Python executable not found"

**Symptom:**
- Output shows: `Error: Python executable not found`
- Extension shows "❌ HoloLoom LSP: Disconnected"

**Solution:**

1. Verify Python is installed:
```bash
python3 --version
```

2. If not installed, install it:
   - **macOS:** `brew install python@3.11`
   - **Ubuntu:** `sudo apt-get install python3.11`
   - **Windows:** Download from python.org

3. Configure Python path in VS Code settings:
```json
{
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11"
}
```

4. Reload VS Code:
   - Press `Ctrl+Shift+P` → "Developer: Reload Window"

### Problem 2: "HoloLoom module not found"

**Symptom:**
- Output shows: `ModuleNotFoundError: No module named 'HoloLoom'`

**Solution:**

1. Check if installed:
```bash
python3 -c "import HoloLoom"
```

2. If error, install it:
```bash
# From PyPI (easiest)
pip install HoloLoom

# From source (for latest)
git clone https://github.com/yourrepo/HoloLoom.git
cd HoloLoom
pip install -e .
```

3. Verify installation:
```bash
python3 -c "import HoloLoom; print(HoloLoom.__version__)"
```

4. Restart VS Code:
```
Ctrl+Shift+P → "Developer: Reload Window"
```

### Problem 3: "Connection refused" or Network errors

**Symptom:**
- Output shows: `Connection refused` or `Network error`
- Status bar shows disconnected

**Solution:**

1. Check for process limits:
```bash
ulimit -n
# If < 256, increase it
ulimit -n 4096
```

2. Kill any conflicting processes:
```bash
# Old HTTP API server (if still running)
pkill -f "python.*agentic_api"

# Old LSP servers
pkill -f "hololoom.*lsp"
```

3. Restart VS Code:
   - Completely close VS Code
   - Reopen it
   - Extension should auto-connect

### Problem 4: "Extension failed to activate"

**Symptom:**
- VS Code shows notification: "Failed to activate extension Promptly"
- No HoloLoom features work

**Solution:**

1. Check extension logs:
   - Output panel: `Ctrl+Shift+U` → "Extension Host"
   - Look for error messages

2. Clear extension cache:
```bash
# macOS/Linux
rm -rf ~/.vscode/extensions/promptly-*

# Windows
rmdir %APPDATA%\.vscode\extensions\promptly-* /s

# Then reinstall
code --install-extension promptly-2.0.0.vsix
```

3. Verify dependencies:
```bash
npm list
```

4. Reinstall dependencies if needed:
```bash
rm -rf node_modules package-lock.json
npm install
npm run compile
```

### Problem 5: "npm install" fails

**Symptom:**
- `npm install` shows errors
- Missing packages

**Solution:**

1. Check npm version:
```bash
npm --version
# Should be 8.0+ (usually comes with Node)
```

2. Clear npm cache:
```bash
npm cache clean --force
```

3. Try again:
```bash
rm -rf node_modules package-lock.json
npm install
```

4. If still fails, check for network issues:
```bash
# Test npm registry connection
npm ping
```

### Problem 6: Sidebar not showing or blank

**Symptom:**
- Sidebar opens but shows no content
- No "Quick Capture" or "Search" boxes

**Solution:**

1. Check LSP connection:
   - Verify status bar shows "Connected" ✅
   - If not, solve connection problem first

2. Clear extension cache:
```bash
Ctrl+Shift+P → "Developer: Reload Window"
```

3. Check sidebar logs:
   - `Ctrl+Shift+U` → "HoloLoom Language Server"
   - Look for webview errors

4. Reinstall extension:
```bash
# Uninstall
code --uninstall-extension promptly

# Reinstall
code --install-extension promptly-2.0.0.vsix
```

### Problem 7: Commands work very slowly (>500ms)

**Symptom:**
- /remember takes >1 second
- Sidebar searches are slow
- Cursor freezes while waiting

**Solution:**

1. Warm up the connection:
   - First request may take ~50-100ms to establish LSP connection
   - Subsequent requests should be ~30-50ms
   - Run 2-3 test commands first

2. Check system load:
```bash
# macOS/Linux
top

# Windows
tasklist /v | sort /+55 /r
```

3. Check for large workspace indexing:
   - Sidebar may be slow if workspace is 1000+ files
   - Wait for indexing to complete

4. Restart LSP server:
   - `Ctrl+Shift+P` → "HoloLoom: Restart Language Server"
   - Wait for "Connected" status

### Problem 8: "vscode-languageclient not found"

**Symptom:**
- Error: `Cannot find module 'vscode-languageclient'`

**Solution:**

```bash
# Install missing dependency
npm install vscode-languageclient@latest

# Recompile
npm run compile

# Clear VS Code cache
rm -rf ~/.vscode/extensions/promptly-*

# Reinstall extension
npm run vscode:prepublish
code --install-extension out/promptly-2.0.0.vsix
```

---

## Next Steps

Once setup is complete:

1. **Read Quick Start:** [README.md](../README.md)
2. **Learn Commands:** See `/help` in chat for all commands
3. **Understand Architecture:** [LSP_ARCHITECTURE.md](LSP_ARCHITECTURE.md)
4. **Troubleshoot:** [MIGRATION_HTTP_TO_LSP.md](MIGRATION_HTTP_TO_LSP.md#troubleshooting)

---

## Support

**Questions?**
1. Check [Common Setup Problems](#common-setup-problems)
2. See [MIGRATION_HTTP_TO_LSP.md](MIGRATION_HTTP_TO_LSP.md) troubleshooting
3. Check logs: `Ctrl+Shift+U` → "HoloLoom Language Server"
4. Report on GitHub with logs attached

**Getting Logs for Bug Report:**
```bash
# Export logs to file
Ctrl+Shift+U → right-click → "Save As"

# Or from terminal
code --log=trace > /tmp/vscode.log 2>&1
```

---

**Ready to go?** Open a file, press `Ctrl+Alt+P`, and start using Promptly! 🧠
