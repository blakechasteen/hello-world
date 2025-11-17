# LSP Configuration Examples

**Published: 2025-11-17**
**Last Updated: 2025-11-17**

Comprehensive configuration examples for Promptly v2.0.0 LSP integration.

---

## Table of Contents

1. [Basic Configuration](#basic-configuration)
2. [Custom Python Path](#custom-python-path)
3. [Custom HoloLoom Path](#custom-hololoom-path)
4. [Debug Configuration](#debug-configuration)
5. [Production Configuration](#production-configuration)
6. [Workspace Settings](#workspace-settings)
7. [CI/CD Configuration](#cicd-configuration)
8. [Troubleshooting Configurations](#troubleshooting-configurations)

---

## Basic Configuration

### Minimal Setup (Auto-Detect)

The extension works with **zero configuration** - VS Code auto-detects Python and HoloLoom:

**VS Code settings.json:**
```json
{
  "hololoom.lsp.enabled": true
}
```

**How it works:**
1. VS Code looks for Python on PATH
2. Python looks for HoloLoom in site-packages
3. LSP server starts automatically
4. Status bar shows "Connected" ✅

**When it works:**
- Python installed via homebrew/apt/official installer
- HoloLoom installed via `pip install`
- Both on system PATH
- Standard installation locations

---

## Custom Python Path

### Use Non-Standard Python Installation

If Python is in an unusual location or you want to use a specific version:

**VS Code settings.json:**
```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11"
}
```

**Find your Python path:**

```bash
# Method 1: which command
which python3
# Output: /usr/local/bin/python3

# Method 2: Full path
python3 -c "import sys; print(sys.executable)"
# Output: /usr/local/bin/python3.11

# Method 3: In virtualenv
source ~/myenv/bin/activate
which python
# Output: /home/user/myenv/bin/python
```

### Examples by Platform

#### macOS (with Homebrew)

```json
{
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11"
}
```

Or with newer Homebrew (Apple Silicon):
```json
{
  "hololoom.lsp.pythonPath": "/opt/homebrew/bin/python3.11"
}
```

#### Ubuntu/Debian

```json
{
  "hololoom.lsp.pythonPath": "/usr/bin/python3.11"
}
```

#### Windows

```json
{
  "hololoom.lsp.pythonPath": "C:\\Users\\YourName\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"
}
```

#### Python Virtual Environment

```json
{
  "hololoom.lsp.pythonPath": "/home/user/myproject/.venv/bin/python3"
}
```

#### Conda Environment

```json
{
  "hololoom.lsp.pythonPath": "/opt/miniconda3/envs/myenv/bin/python"
}
```

### Verify Configuration

After setting `pythonPath`, verify it works:

```bash
# Check Python version
/usr/local/bin/python3.11 --version

# Check HoloLoom is available in that Python
/usr/local/bin/python3.11 -c "import HoloLoom; print(HoloLoom.__version__)"
```

---

## Custom HoloLoom Path

### Use Custom HoloLoom Installation

If HoloLoom is installed from source in a custom location:

**VS Code settings.json:**
```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.hololoomPath": "/home/user/projects/HoloLoom"
}
```

### Find HoloLoom Path

#### From Installed Package

```bash
python3 -c "import HoloLoom; import os; print(os.path.dirname(HoloLoom.__file__))"
# Output: /usr/local/lib/python3.11/site-packages/HoloLoom
```

#### From Source Installation

```bash
cd /path/to/HoloLoom
pwd  # Current directory is the path
# Output: /home/user/projects/HoloLoom
```

### Examples

#### Development Installation (from source)

```json
{
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.hololoomPath": "/home/user/dev/HoloLoom"
}
```

Verify:
```bash
ls /home/user/dev/HoloLoom/HoloLoom/__init__.py  # Should exist
```

#### System Package Installation

```json
{
  "hololoom.lsp.pythonPath": "/usr/bin/python3.11",
  "hololoom.lsp.hololoomPath": "/usr/local/lib/python3.11/site-packages/HoloLoom"
}
```

#### Multiple Projects (different versions)

**Project A (.vscode/settings.json):**
```json
{
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.10",
  "hololoom.lsp.hololoomPath": "/home/user/projectA/.venv/lib/python3.10/site-packages/HoloLoom"
}
```

**Project B (.vscode/settings.json):**
```json
{
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.hololoomPath": "/home/user/projectB/.venv/lib/python3.11/site-packages/HoloLoom"
}
```

---

## Debug Configuration

### Full Debug Logging

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.logLevel": "debug",
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.serverArgs": [
    "--log-level", "debug",
    "--trace", "verbose"
  ]
}
```

### Trace LSP Protocol

Enable detailed JSON-RPC message logging:

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.logLevel": "debug"
}
```

Then view in Output panel:
```
Ctrl+Shift+U → "HoloLoom Language Server"
```

You'll see messages like:
```
[hololoom/remember] →
  { "content": "Learning LSP" }
[hololoom/remember] ←
  { "success": true, "memory_id": "..." }
```

### Save Logs to File

```bash
# Start VS Code with logging
code --log=debug /path/to/project

# View logs
cat ~/.config/Code/logs/*/window*.log
```

### Profile Performance

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.logLevel": "debug",
  "hololoom.lsp.serverArgs": [
    "--profile",
    "--profile-output", "/tmp/hololoom-profile.json"
  ]
}
```

---

## Production Configuration

### Optimized for Performance

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.logLevel": "warning",
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.serverArgs": [
    "--cache-size", "10000",
    "--max-workers", "4",
    "--timeout", "30"
  ]
}
```

**Settings:**
- `logLevel: warning` - Less overhead, only errors logged
- `cache-size: 10000` - Large cache for frequent queries
- `max-workers: 4` - Parallel request handling
- `timeout: 30` - 30-second timeout for operations

### Team Workspace Configuration

For shared team projects, add to `.vscode/settings.json`:

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.hololoomPath": "${workspaceFolder}/.hololoom",
  "hololoom.lsp.serverArgs": [
    "--workspace-memory",
    "--shared-cache",
    "--team-mode"
  ]
}
```

### High-Traffic Server

For server with many concurrent users:

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.logLevel": "error",
  "hololoom.lsp.serverArgs": [
    "--cache-size", "50000",
    "--max-workers", "8",
    "--timeout", "60",
    "--memory-pool", "2048MB",
    "--connection-pool", "100"
  ]
}
```

---

## Workspace Settings

### Project-Specific Settings

Store in `.vscode/settings.json` (committed to repo):

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.pythonPath": "${workspaceFolder}/.venv/bin/python",
  "hololoom.lsp.hololoomPath": "${workspaceFolder}/vendor/HoloLoom"
}
```

**Available variables:**
- `${workspaceFolder}` - Root directory
- `${workspaceFolderBasename}` - Folder name
- `${env:VARIABLE}` - Environment variable

### Per-Language Configuration

For projects with mixed language support:

```json
{
  "hololoom.lsp.enabled": true,
  "[python]": {
    "hololoom.lsp.pythonPath": "/usr/bin/python3.11"
  },
  "[javascript]": {
    "hololoom.lsp.pythonPath": "/usr/local/bin/python3.10"
  }
}
```

### User vs Workspace Settings

**User settings (`~/.config/Code/settings.json`):**
- Global defaults for all projects
- Applied to every workspace

```json
{
  "hololoom.lsp.logLevel": "info",
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11"
}
```

**Workspace settings (`.vscode/settings.json`):**
- Project-specific overrides
- Committed to version control

```json
{
  "hololoom.lsp.pythonPath": "${workspaceFolder}/.venv/bin/python",
  "hololoom.lsp.hololoomPath": "${workspaceFolder}/HoloLoom"
}
```

**Precedence:** Workspace settings override user settings

### Exclude from Settings Sync

If using VS Code Settings Sync, exclude sensitive paths:

```json
{
  "settingsSync.ignoredSettings": [
    "hololoom.lsp.pythonPath",
    "hololoom.lsp.hololoomPath"
  ]
}
```

---

## CI/CD Configuration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test with Promptly LSP

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install HoloLoom
        run: pip install HoloLoom

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install extension dependencies
        run: |
          cd promptly-vscode
          npm install

      - name: Compile TypeScript
        run: |
          cd promptly-vscode
          npm run compile

      - name: Run tests
        env:
          HOLOLOOM_LSP_PYTHONPATH: ${{ runner.tool_cache }}/PyPy/3.11/x64
        run: |
          cd promptly-vscode
          npm test
```

### GitLab CI

```yaml
# .gitlab-ci.yml
test-lsp:
  image: node:18

  before_script:
    - apt-get update && apt-get install -y python3.11 python3-pip
    - pip install HoloLoom
    - cd promptly-vscode
    - npm install

  script:
    - npm run compile
    - npm test

  artifacts:
    reports:
      junit: test-results.xml
```

### Environment Variables

Set in CI/CD system:

```bash
# Python path for LSP server
HOLOLOOM_LSP_PYTHONPATH=/usr/bin/python3.11

# HoloLoom path
HOLOLOOM_LSP_HOLOLOOMPATH=/opt/HoloLoom

# Log level for debugging
HOLOLOOM_LSP_LOGLEVEL=debug
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM node:18-bullseye

# Install Python and HoloLoom
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install HoloLoom

# Set up extension
WORKDIR /app/promptly-vscode
COPY . .
RUN npm install
RUN npm run compile

# Run tests
CMD ["npm", "test"]
```

---

## Troubleshooting Configurations

### Python Version Mismatch

**Symptom:** "Python version 3.8 required, found 3.7"

**Solution:**
```json
{
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.9"
}
```

Verify:
```bash
/usr/local/bin/python3.9 --version
# Should show 3.8 or newer
```

### HoloLoom Import Errors

**Symptom:** "ModuleNotFoundError: No module named 'HoloLoom'"

**Solution:**

```json
{
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.hololoomPath": "/home/user/HoloLoom"
}
```

Verify:
```bash
/usr/local/bin/python3.11 -c "import HoloLoom"
# Should succeed without error
```

### Virtual Environment Issues

**Symptom:** LSP can't find HoloLoom installed in venv

**Solution:**
```json
{
  "hololoom.lsp.pythonPath": "/home/user/myenv/bin/python3"
}
```

Verify venv is activated:
```bash
source /home/user/myenv/bin/activate
python -c "import HoloLoom"
```

### Slow LSP Startup

**Symptom:** "Waiting for language server to start..." takes >5 seconds

**Solution:**

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.logLevel": "warning",
  "hololoom.lsp.serverArgs": [
    "--lazy-init",
    "--preload-core-only"
  ]
}
```

**Or check system load:**
```bash
top  # or Activity Monitor on macOS
# Look for high CPU/memory usage
```

### Connection Refused on Startup

**Symptom:** "Connection refused" error immediately on start

**Solution:**

1. Kill any orphaned processes:
```bash
pkill -f "hololoom.*lsp"
pkill -f "python.*agentic"
```

2. Restart VS Code:
```
Cmd+Shift+P → "Developer: Reload Window"
```

3. Check process limits:
```bash
ulimit -n
# If < 256, increase: ulimit -n 4096
```

### Permission Denied Errors

**Symptom:** "Permission denied" when accessing Python or HoloLoom

**Solution:**

```bash
# Check permissions
ls -la /usr/local/bin/python3.11
# Should be readable and executable

# Make executable if needed
chmod +x /usr/local/bin/python3.11
```

Or use a path in your home directory:

```json
{
  "hololoom.lsp.pythonPath": "/home/user/.local/bin/python3.11"
}
```

---

## Quick Reference

### Minimal Config

```json
{
  "hololoom.lsp.enabled": true
}
```

### Development Config

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.logLevel": "debug",
  "hololoom.lsp.serverArgs": ["--profile"]
}
```

### Production Config

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.logLevel": "warning",
  "hololoom.lsp.serverArgs": [
    "--cache-size", "10000",
    "--max-workers", "4"
  ]
}
```

### Team Config

```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.pythonPath": "${workspaceFolder}/.venv/bin/python",
  "hololoom.lsp.hololoomPath": "${workspaceFolder}/vendor/HoloLoom"
}
```

---

**Need help?** See [SETUP_LSP.md](SETUP_LSP.md) or [MIGRATION_HTTP_TO_LSP.md](MIGRATION_HTTP_TO_LSP.md).
