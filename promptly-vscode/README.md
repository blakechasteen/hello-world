# Promptly VS Code Extension

**Chat-native development with slash commands inside VS Code!**

## ✨ Features

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

Settings (`Ctrl+,` → search "Promptly"):

- **`promptly.hololoomUrl`**: HoloLoom server URL (default: `http://localhost:8000`)
- **`promptly.claudeApiKey`**: Anthropic API key (optional)
- **`promptly.enableAutocomplete`**: Show/hide autocomplete (default: `true`)

## 🎮 Usage

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

Start the HoloLoom server:

```bash
cd HoloLoom/server
python agentic_api.py
```

The extension connects to `http://localhost:8000` by default.

### Server Endpoints Used:
- `POST /query` - Natural language queries
- `POST /api/remember` - Save memories
- `POST /api/recall` - Query memories

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

**Autocomplete not appearing?**
- Make sure you typed `/` at the start
- Check `promptly.enableAutocomplete` is true
- Try typing `/he` to trigger it

**HoloLoom connection failed?**
- Check server is running: `http://localhost:8000/health`
- Update URL in settings if using different port
- Server logs will show connection attempts

**Claude commands not working?**
- Set `promptly.claudeApiKey` in VS Code settings
- Or use natural language (routes to HoloLoom instead)
- Check API key is valid at https://console.anthropic.com

**Git commands failing?**
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

## 🚀 What's Next?

Future features:
- **Multi-file operations** (review entire folders)
- **Workflow automation** (custom slash command chains)
- **Team collaboration** (shared HoloLoom memories)
- **Matrix bot integration** (chat from anywhere)
- **Custom slash commands** (user-defined via config)

## 📝 License

MIT - See LICENSE file
