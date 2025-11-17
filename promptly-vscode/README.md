# Promptly + HoloLoom - Your IDE with Perfect Memory

**Your IDE that remembers everything you've ever coded, decided, or learned.**

Promptly integrates **HoloLoom**, a neural memory system, directly into VS Code. Capture thoughts, search past decisions, and get AI-powered insights without leaving your editor.

## ✨ Features

### 🧠 **HoloLoom Sidebar** (NEW!)

- **Quick Capture**: Save notes, decisions, and discoveries with one click
- **Today's Notes**: View everything you've captured today
- **Semantic Search**: Find relevant memories using natural language
- **Knowledge Graph**: All notes connected by relationships

### 💡 **Inline Code Intelligence (CodeLens)** (NEW!)

- **Smart Annotations**: See related notes inline with `// NOTE:`, `// TODO:`, `// FIXME:` comments
- **Auto-Linking**: Automatically finds relevant knowledge from your memory
- **One-Click Capture**: Turn comments into persistent memories

### ⚡ Slash Commands with Smart Autocomplete

Type `/` in the chat and see beautiful autocomplete with:
- **Category badges** (Git/Claude/Memory/Utility)
- **Usage examples** (e.g., `/gc "message"`)
- **Descriptions** for each command
- **Arrow key navigation** (↑↓)
- **Tab to complete**

### 🎯 Available Commands

**Git:**
- `/gs` - Git status
- `/gl` - Git log (last 10 commits)
- `/gc "message"` - Git commit
- `/gp` - Git push
- `/gd` - Git diff

**Claude:**
- `/review` - Review current file with Claude
- `/explain` - Explain code selection
- `/refactor "task"` - Get refactoring suggestions

**HoloLoom Memory:**
- `/remember "note"` - Save to knowledge graph
- `/recall "query"` - Query memories
- `/context` - Show current context

**Utility:**
- `/help` - Show all commands
- `/clear` - Clear chat

### 🧠 Natural Language

Don't remember slash commands? Just chat naturally:

```
You: show me what changed
Bot: [runs git status]

You: review this code
Bot: [runs code review]

You: remember we're using PostgreSQL
Bot: ✅ Saved to memory
```

## 🚀 Quick Start

### For Users (Install Extension)

1. **Install Extension**
   - Open VS Code
   - Extensions: `Ctrl+Shift+X`
   - Search: "Promptly"
   - Click "Install"

2. **Configure (Optional)**
   - Press `Ctrl+,` → Search "HoloLoom"
   - Set options if needed (usually auto-detects)
   - See [docs/SETUP_LSP.md](docs/SETUP_LSP.md) for details

3. **Verify Connection**
   - Check status bar (bottom-right) for "🧠 HoloLoom LSP: Connected" ✅
   - If not connected, see [docs/SETUP_LSP.md#troubleshooting](docs/SETUP_LSP.md#common-setup-problems)

4. **Start Using Features**
   - **Open Sidebar:** Click 🧠 brain icon in Activity Bar
   - **Open Chat:** Press `Ctrl+Alt+P`
   - **Quick Capture:** Type in sidebar → Click "💾 Remember"
   - **Search Memory:** Type in sidebar → Click "Search"
   - **Use Commands:** In chat, type `/` then choose command

### For Developers (Development Mode)

1. **Install dependencies:**
   ```bash
   cd promptly-vscode
   npm install
   ```

2. **Compile:**
   ```bash
   npm run compile
   ```

3. **Run (Development Mode):**
   - Press **F5** in VS Code
   - New window opens with extension loaded
   - Press **Ctrl+Alt+P** to open chat
   - Type `/` to see autocomplete magic! ✨

## ⚙️ Configuration

Promptly v2.0.0+ uses LSP (Language Server Protocol) which auto-detects Python and HoloLoom.

**No configuration needed in most cases!** VS Code auto-starts the LSP server.

### Basic Configuration

Press `Ctrl+,` → search "HoloLoom":

- **`hololoom.lsp.enabled`**: Enable LSP integration (default: `true`)
- **`hololoom.lsp.pythonPath`**: Custom Python path (optional, auto-detected)
- **`hololoom.lsp.hololoomPath`**: Custom HoloLoom path (optional, auto-detected)
- **`hololoom.lsp.logLevel`**: Log level (default: `"info"`)

### Other Settings

- **`promptly.claudeApiKey`**: Anthropic API key (optional)
- **`promptly.enableAutocomplete`**: Show/hide autocomplete (default: `true`)

### Examples

**Minimal (auto-detect everything):**
```json
{
  "hololoom.lsp.enabled": true
}
```

**Custom Python path:**
```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11"
}
```

**Custom HoloLoom location:**
```json
{
  "hololoom.lsp.enabled": true,
  "hololoom.lsp.pythonPath": "/usr/local/bin/python3.11",
  "hololoom.lsp.hololoomPath": "/home/user/HoloLoom"
}
```

See [docs/LSP_CONFIG_EXAMPLES.md](docs/LSP_CONFIG_EXAMPLES.md) for more examples.

## 🎮 Usage

### Open HoloLoom Sidebar (NEW!)
- **Click** the 🧠 brain icon in the Activity Bar (left sidebar)
- Or **Command Palette**: `View: Show HoloLoom`

**Quick Capture**:
1. Type your note in the "Quick Capture" box
2. Click "💾 Remember"
3. Done! Your note is saved to the knowledge graph

**Search Memories**:
1. Type a question in "Search Memory" box
2. Click "Search"
3. See results with confidence scores

**Example Captures**:
- "We decided to use PostgreSQL for authentication"
- "React Suspense causes issues with SSR in Next.js 13"
- "Team agreed to ship feature X by Nov 20"

### Use Inline Suggestions (CodeLens) (NEW!)

Add comments to your code and get automatic suggestions:

```typescript
// NOTE: Using Thompson Sampling for exploration/exploitation balance
// ↑ CodeLens shows: 💡 3 related notes (click to view)

// TODO: Add rate limiting to auth endpoints
// ↑ CodeLens shows: 📝 Capture this TODO (click to save)
```

**Supported comment patterns**:
- `// NOTE:`, `// TODO:`, `// FIXME:` (JavaScript, TypeScript, C, C++)
- `# NOTE:`, `# TODO:`, `# FIXME:` (Python, Ruby, Shell)
- `/* NOTE: ... */` (Block comments)
- `<!-- NOTE: ... -->` (HTML, XML)

### Open Chat
- **Keyboard**: `Ctrl+Alt+P` (Windows/Linux) or `Cmd+Alt+P` (Mac)
- **Status Bar**: Click "Promptly" icon (bottom-right)
- **Command Palette**: `Promptly: Open Chat`

### Use Slash Commands
1. Type `/` in chat
2. Autocomplete appears with suggestions
3. Use ↑↓ arrows to navigate
4. Press Tab to complete
5. Press Enter to send

### Example Session
```
/gs
→ ✅ Working tree clean

/gc "Add awesome feature"
→ ✅ Committed: Add awesome feature

/review
→ ✅ Code review complete! (opens in new tab)

/remember "We chose PostgreSQL for auth"
→ ✅ Saved to HoloLoom memory

/recall "database"
→ HoloLoom Recall Results:
   1. We chose PostgreSQL for auth
      Confidence: 95% | 2 minutes ago
```

## 🔗 Integration with HoloLoom

Promptly v2.0.0 uses **LSP (Language Server Protocol)** to communicate with HoloLoom.

**No manual server startup needed!** VS Code auto-starts the LSP server automatically.

### How It Works

1. Extension initializes LSP client on startup
2. LSP client spawns Python subprocess with HoloLoom
3. LSP server manages all communication with HoloLoom memory
4. Server auto-restarts if it crashes

### Requirements

- **Python 3.8+** installed and on PATH
- **HoloLoom package** installed: `pip install HoloLoom`

### Verify Connection

- Check status bar: "🧠 HoloLoom LSP: Connected" ✅
- View logs: `Ctrl+Shift+U` → "HoloLoom Language Server"

### Troubleshooting Connection

If LSP doesn't connect:
1. Check Python installed: `python3 --version`
2. Check HoloLoom installed: `python3 -c "import HoloLoom"`
3. Configure paths in settings (see [Configuration](#configuration))
4. Check logs in Output panel for errors

See [docs/SETUP_LSP.md](docs/SETUP_LSP.md) for detailed setup and troubleshooting.

## 🛠️ Development

### Watch Mode
```bash
npm run watch
```

Then press **Reload Window** in the Extension Development Host after changes.

### File Structure
```
promptly-vscode/
├── src/
│   ├── extension.ts              # Main entry point
│   ├── chatView.ts               # Chat UI + autocomplete
│   └── commands/
│       ├── gitCommands.ts        # Git integration
│       ├── claudeCommands.ts     # Claude API
│       └── hololoomCommands.ts   # HoloLoom client
├── package.json                  # Extension manifest
└── tsconfig.json                 # TypeScript config
```

## 🎨 Autocomplete Features

The autocomplete is **smart and beautiful**:

1. **Category Badges**: Color-coded by category
   - 🟡 Git (orange)
   - 🟢 Claude (green)
   - 🔵 Memory (blue)
   - ⚫ Utility (gray)

2. **Usage Examples**: Shows exact syntax
   - `/gc "message"` (shows you need quotes)
   - `/review` (no arguments needed)

3. **Keyboard Navigation**:
   - `↑↓` - Navigate suggestions
   - `Tab` - Complete selected
   - `Enter` - Send command
   - `Esc` - Close autocomplete

## 🔍 Troubleshooting

### "HoloLoom LSP: Disconnected" in status bar

**Problem:** Extension shows "Disconnected" instead of "Connected"

**Solutions:**
1. Check Python: `python3 --version` (must be 3.8+)
2. Check HoloLoom: `python3 -c "import HoloLoom"`
3. Configure paths in settings (see [Configuration](#configuration))
4. Restart VS Code: `Ctrl+Shift+P` → "Reload Window"
5. Check logs: `Ctrl+Shift+U` → "HoloLoom Language Server"

See [docs/SETUP_LSP.md](docs/SETUP_LSP.md) for complete troubleshooting guide.

### Autocomplete not appearing?
- Make sure you typed `/` at the start
- Check `promptly.enableAutocomplete` is true
- Try typing `/he` to trigger it

### Commands working slowly (>100ms)?
- First request may take ~50-100ms to establish LSP connection
- Subsequent requests should be <50ms
- Run 2-3 test commands to "warm up" connection
- Check system load: `top`

### Claude commands not working?
- Set `promptly.claudeApiKey` in VS Code settings
- Or use natural language (routes to HoloLoom instead)
- Check API key is valid at https://console.anthropic.com

### Git commands failing?
- Make sure you're in a git repository
- Check terminal access (extension needs to run git commands)

## 🎯 Tips & Tricks

1. **Autocomplete is context-aware**: It shows different suggestions based on what you've typed
2. **Tab completes the whole command**: Including quotes and placeholders
3. **Natural language is powerful**: If you forget the slash command, just ask!
4. **Commands work on current context**: `/review` reviews your active file automatically
5. **Status bar shows availability**: Green = HoloLoom connected, Gray = standalone mode

## 📋 Keyboard Shortcuts

| Action | Windows/Linux | Mac |
|--------|---------------|-----|
| Open Chat | `Ctrl+Alt+P` | `Cmd+Alt+P` |
| Send Message | `Enter` | `Enter` |
| Show Autocomplete | `/` | `/` |
| Navigate Suggestions | `↑` `↓` | `↑` `↓` |
| Complete | `Tab` | `Tab` |
| Cancel | `Esc` | `Esc` |

## 📚 Documentation

**v2.0.0 Migration from HTTP to LSP?** Start here:

- **[MIGRATION_HTTP_TO_LSP.md](docs/MIGRATION_HTTP_TO_LSP.md)** - Complete migration guide
  - Before/after examples
  - Breaking changes
  - Troubleshooting tips

- **[SETUP_LSP.md](docs/SETUP_LSP.md)** - Installation and configuration
  - Prerequisites
  - Step-by-step setup
  - Common setup problems

- **[BREAKING_CHANGES.md](BREAKING_CHANGES.md)** - What changed in v2.0.0
  - Impact on users and developers
  - Deprecation timeline
  - Migration checklists

- **[LSP_ARCHITECTURE.md](docs/LSP_ARCHITECTURE.md)** - How LSP works
  - Architecture diagrams
  - Component responsibilities
  - Message flow details

- **[LSP_CONFIG_EXAMPLES.md](docs/LSP_CONFIG_EXAMPLES.md)** - Configuration examples
  - Basic setup
  - Custom paths
  - Production configurations

- **[CHANGELOG.md](CHANGELOG.md)** - Version history
  - What's new in each release
  - Upgrade instructions

## 🚀 What's Next?

Future features:
- **v2.1.0:** Enhanced LSP features and performance improvements
- **v3.0.0:** LSP-only codebase (HTTP API removed)
- **Multi-file operations** (review entire folders)
- **Workflow automation** (custom slash command chains)
- **Team collaboration** (shared HoloLoom memories)
- **Matrix bot integration** (chat from anywhere)
- **Custom slash commands** (user-defined via config)

See [CHANGELOG.md](CHANGELOG.md) for complete roadmap.

## 📝 License

MIT - See LICENSE file
