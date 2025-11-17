# HoloLoom LSP User Guide

**Quick reference for using the HoloLoom Language Server Protocol client in VS Code**

## What's New?

The Promptly extension now uses the Language Server Protocol (LSP) for faster, smarter integration with HoloLoom's neural memory system.

### Benefits

- ⚡ **10x faster** response times (5-20ms vs 50-150ms)
- 🧠 **Smart completions** as you type
- 💡 **Hover information** on entities
- 🔍 **Go-to-definition** navigation
- 🔄 **Automatic fallback** to HTTP API if needed

## Getting Started

### 1. Installation

The LSP client is built into the extension. No additional setup required!

### 2. First Launch

When you open VS Code:
1. Extension automatically detects HoloLoom installation
2. LSP server starts in background
3. You'll see "✓ HoloLoom LSP connected" status message

That's it! You're ready to use HoloLoom.

### 3. Basic Usage

**Remember something**:
- `Ctrl+Shift+P` → `Promptly: Remember (HoloLoom)`
- Type your note
- Press Enter

**Recall information**:
- `Ctrl+Shift+P` → `Promptly: Recall (HoloLoom)`
- Type your query
- See results in chat window

**Get smart suggestions**:
- Type `// TODO:` in any file
- Get related notes automatically (CodeLens feature)
- Click to view details

## Commands

### HoloLoom Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `HoloLoom: Remember` | - | Save a note to memory |
| `HoloLoom: Recall` | - | Search memory |
| `HoloLoom: Show Knowledge Graph` | - | Visualize entity relationships |
| `HoloLoom: Index Workspace` | - | Index entire workspace |
| `HoloLoom: Show Indexing Status` | - | View indexing progress |

### LSP Management Commands

| Command | Description |
|---------|-------------|
| `HoloLoom: Show LSP Status` | Check LSP connection status |
| `HoloLoom: Restart LSP Server` | Restart LSP server |

## Configuration

### Basic Settings

Open settings (`File` → `Preferences` → `Settings`) and search for "HoloLoom":

```json
{
  // Enable/disable LSP client
  "hololoom.lsp.enabled": true,

  // LSP server log level (for debugging)
  "hololoom.lsp.logLevel": "INFO"
}
```

### Advanced Settings

Only needed if auto-detection fails:

```json
{
  // Custom Python interpreter path
  "hololoom.lsp.pythonPath": "/usr/bin/python3",

  // Custom HoloLoom installation path
  "hololoom.lsp.hololoomPath": "/home/user/hololoom"
}
```

## Features

### 1. Smart Completions

**What it does**: Suggests relevant code/notes as you type

**How to use**:
1. Start typing in any file
2. Wait for completion suggestions
3. Press `Enter` to accept

**Example**:
```python
# Start typing...
def calculate_thompson_  # ← Suggests: thompson_sampling()
```

### 2. Hover Information

**What it does**: Shows context about entities when you hover

**How to use**:
1. Hover over any identifier (function, class, variable)
2. See definition, type, documentation
3. Click links to navigate

**Example**:
```python
result = thompson_sampling(...)
         ^^^^^^^^^^^^^^^^ ← Hover shows function signature and docs
```

### 3. Go-to-Definition

**What it does**: Navigate to where entities are defined

**How to use**:
1. Right-click on identifier
2. Select "Go to Definition" (or press `F12`)
3. Jump to definition location

**Example**:
```python
obj = MyClass()  # ← F12 on MyClass jumps to class definition
```

### 4. Workspace Symbol Search

**What it does**: Find entities across entire workspace

**How to use**:
1. Press `Ctrl+T` (or `Cmd+T` on Mac)
2. Type entity name
3. Select from results to navigate

**Example**:
```
Ctrl+T → Type "thompson" → See all related entities
```

### 5. CodeLens Suggestions

**What it does**: Shows related notes inline with code

**How to use**:
1. Add comments like `// TODO:`, `// NOTE:`, `// FIXME:`
2. See "💡 N related notes" above comment
3. Click to view suggestions

**Example**:
```python
# TODO: Implement Thompson Sampling
#       ↑ Shows "💡 3 related notes"
```

### 6. Sidebar Memory View

**What it does**: Quick access to memory search and capture

**How to use**:
1. Click HoloLoom icon in activity bar (🧠)
2. Use "Quick Capture" to save notes
3. Use "Search Memory" to find information
4. View "Today's Notes"

## Status Indicators

### Status Bar

Look for these messages in the bottom-right status bar:

- `✓ HoloLoom LSP connected` - LSP working normally
- `⚠ HoloLoom LSP: Not connected` - Using HTTP fallback

### Output Logs

View detailed logs:
1. `View` → `Output`
2. Select "HoloLoom LSP" from dropdown
3. See server startup, requests, errors

## Common Workflows

### Workflow 1: Capture Development Decisions

1. Make a design decision
2. `Ctrl+Shift+P` → `Promptly: Remember`
3. Type: "Decided to use PostgreSQL for auth storage because..."
4. Press Enter
5. Later, recall with: "What database did we choose?"

### Workflow 2: Track TODOs

1. Add TODO comment: `// TODO: Refactor this function`
2. See CodeLens: "📝 Capture this TODO"
3. Click to save to HoloLoom
4. Search later: `Ctrl+Shift+P` → `Recall` → "pending refactoring"

### Workflow 3: Navigate Complex Codebase

1. Press `Ctrl+T`
2. Type entity name (e.g., "UserAuth")
3. See all definitions and references
4. Click to navigate

### Workflow 4: Quick Reference

1. Hover over unfamiliar function
2. Read documentation and type info
3. Press `F12` to see implementation
4. Use breadcrumbs to navigate back

## Troubleshooting

### LSP Not Working?

**Step 1: Check Status**
- `Ctrl+Shift+P` → `HoloLoom: Show LSP Status`
- Look for connection status

**Step 2: View Logs**
- `View` → `Output` → "HoloLoom LSP"
- Look for error messages

**Step 3: Restart LSP**
- `Ctrl+Shift+P` → `HoloLoom: Restart LSP Server`
- Wait for "✓ connected" message

**Step 4: Check Configuration**
- Open settings
- Search "HoloLoom LSP"
- Verify `hololoom.lsp.hololoomPath` points to correct location

### Still Not Working?

**Use HTTP Fallback**:
The extension automatically falls back to HTTP API if LSP fails. You can still use all commands, just slightly slower.

**Get Help**:
1. Enable debug logging: Set `hololoom.lsp.logLevel` to `DEBUG`
2. Restart LSP server
3. Copy logs from Output panel
4. File issue on GitHub with logs

## FAQ

**Q: Do I need to start the LSP server manually?**
A: No, it starts automatically when you open VS Code.

**Q: What if HoloLoom is not in my workspace?**
A: Configure `hololoom.lsp.hololoomPath` in settings to point to your HoloLoom installation.

**Q: Can I disable LSP and use only HTTP API?**
A: Yes, set `hololoom.lsp.enabled: false` in settings.

**Q: Does LSP work with remote development (SSH, WSL, Docker)?**
A: Yes, as long as HoloLoom is installed in the remote environment.

**Q: How do I know if LSP is working?**
A: Look for "✓ HoloLoom LSP connected" in status bar. Also try hovering over code - if you see information, LSP is working.

**Q: What's the difference between LSP and HTTP API?**
A: LSP provides real-time IDE features (completion, hover, definition). HTTP API is for explicit commands (remember, recall). Both work together.

**Q: Can I use LSP without the HTTP API server?**
A: Yes! LSP runs independently. HTTP API is only a fallback.

**Q: How much memory does LSP use?**
A: Typically 50-100MB. The server is lightweight and optimized for IDE use.

## Tips & Tricks

### Tip 1: Use Keyboard Shortcuts

- `F12` - Go to definition
- `Ctrl+T` - Symbol search
- `Alt+F12` - Peek definition (inline view)
- `Shift+F12` - Find all references

### Tip 2: Customize Logging

For debugging, enable verbose logging:
```json
{
  "hololoom.lsp.logLevel": "DEBUG"
}
```

### Tip 3: Index Before Searching

For best results, index your workspace first:
- `Ctrl+Shift+P` → `HoloLoom: Index Workspace`
- Wait for completion
- Search with `Ctrl+T`

### Tip 4: Use Sidebar for Quick Access

Click HoloLoom icon (🧠) in activity bar for quick access to:
- Quick Capture
- Search Memory
- Today's Notes

### Tip 5: Combine with Chat

Use chat for complex queries:
- `Ctrl+Alt+P` (or `Cmd+Alt+P`) - Open Promptly chat
- Use `/recall` command for searches
- Use `/remember` command to save notes

## Next Steps

1. **Try it out**: Open a file and hover over code
2. **Capture notes**: Save design decisions as you make them
3. **Search memory**: Recall information when you need it
4. **Explore graph**: Visualize knowledge connections
5. **Read more**: See [LSP_MIGRATION_GUIDE.md](LSP_MIGRATION_GUIDE.md) for technical details

## Support

Need help? Check these resources:

1. **This guide** - Quick reference for common tasks
2. **Migration Guide** - [LSP_MIGRATION_GUIDE.md](LSP_MIGRATION_GUIDE.md) - Technical details
3. **Output logs** - `View` → `Output` → "HoloLoom LSP"
4. **GitHub Issues** - File bug reports or feature requests

Happy coding with HoloLoom! 🧠✨
