# HoloLoom VS Code Extension

Intelligent knowledge weaving with multi-scale memory, recursive learning, and transparent AI reasoning - directly in your editor.

## Features

### 🗨️ Chat Panel
- Query HoloLoom with context from your code
- 4 reasoning modes: Direct, Verify, Research, Plan & Execute
- See confidence scores and source attributions

### 📝 Quick Commands
- **Query Selection** (`Ctrl+Shift+Q`): Ask about selected text
- **Explain Selection** (`Ctrl+Shift+E`): Get explanations for code
- **Remember Selection** (`Ctrl+Shift+R`): Save to memory
- **Open Chat** (`Ctrl+Shift+H`): Open the chat panel

### 🧠 Memory Browser
- Search and browse your knowledge base
- Insert memories into your code
- Manage stored knowledge

### 🔌 Status Bar
- Connection status indicator
- Quick reconnect option

## Requirements

- HoloLoom server running (default: `http://localhost:8000`)
- Node.js 18+ for development

## Getting Started

1. Start the HoloLoom server:
   ```bash
   cd mythRL
   PYTHONPATH=. uvicorn HoloLoom.server.agentic_api:app --port 8000
   ```

2. Install the extension (from source):
   ```bash
   cd hololoom-ui/apps/vscode
   npm install
   npm run compile
   ```

3. Press `F5` to launch the Extension Development Host

4. Open the HoloLoom panel from the Activity Bar (left side)

## Extension Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `hololoom.apiEndpoint` | HoloLoom API server URL | `http://localhost:8000` |
| `hololoom.apiTimeout` | Request timeout (ms) | `30000` |
| `hololoom.defaultReasoningMode` | Default reasoning mode | `verify` |
| `hololoom.includeFileContext` | Include file context in queries | `true` |
| `hololoom.maxContextLines` | Max context lines to include | `100` |
| `hololoom.showStatusBar` | Show status bar indicator | `true` |
| `hololoom.autoConnect` | Auto-connect on startup | `true` |

## Reasoning Modes

| Mode | Latency | Description |
|------|---------|-------------|
| **Direct** | ~150ms | Single-pass answer for simple queries |
| **Verify** | ~600ms | Answer with verification for accuracy |
| **Research** | ~900ms | Multi-query exploration for depth |
| **Plan & Execute** | ~750ms | Goal decomposition for complex tasks |

## Commands

All commands are available from the Command Palette (`Ctrl+Shift+P`):

- `HoloLoom: Open Chat Panel` - Open the chat view
- `HoloLoom: Query Selection` - Query with selected text
- `HoloLoom: Explain Selection` - Explain selected code
- `HoloLoom: Remember Selection` - Save selection to memory
- `HoloLoom: Open Memory Browser` - Browse knowledge base
- `HoloLoom: Reconnect to Server` - Reconnect to HoloLoom

## Context Menu

Right-click in the editor to access HoloLoom commands:
- Query Selection
- Explain Selection (when text selected)
- Remember Selection (when text selected)

## Development

```bash
# Install dependencies
npm install

# Compile
npm run compile

# Watch mode
npm run watch

# Package extension
npm run package

# Lint
npm run lint
```

## Architecture

```
src/
├── extension.ts           # Main entry point
├── api/
│   └── holoLoomClient.ts  # API client wrapper
├── providers/
│   ├── chatViewProvider.ts    # Chat webview
│   ├── memoryViewProvider.ts  # Memory browser
│   └── statusBarProvider.ts   # Status bar
├── commands/
│   └── index.ts           # Command handlers
├── utils/
│   ├── config.ts          # Configuration
│   └── context.ts         # Editor context
└── webview/
    ├── chat/              # React chat (future)
    └── memory/            # React memory (future)
```

## Troubleshooting

### Extension not connecting
1. Ensure HoloLoom server is running on the configured port
2. Check the Output panel (View > Output > HoloLoom)
3. Try `HoloLoom: Reconnect to Server` command

### Slow responses
- Try `Direct` mode for faster responses
- Check server performance
- Reduce `maxContextLines` in settings

## License

MIT - Part of the HoloLoom project.
