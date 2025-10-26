# Promptly VS Code Extension

**AI-powered prompt engineering directly in VS Code**

Compose, test, and execute AI prompts with loop composition, analytics, and skill management - all without leaving your editor.

---

## ✨ Features

### 🚀 Quick Actions
- **Execute Selection** - Send selected code to AI with one click
- **Inline Prompts** - Right-click code for instant AI actions
- **Prompt Panel** - Full-featured prompt composer in sidebar
- **Keyboard Shortcuts** - `Ctrl+Shift+E` to execute selection

### 🎯 Code Actions
- **Analyze Code** - Find bugs, issues, and improvements
- **Improve Code** - Get refactoring suggestions
- **Explain Code** - Understand complex code
- **Document Code** - Generate documentation
- **Find Bugs** - Detect potential issues

### 📚 Skill Management
- Browse skill library in sidebar
- Create custom skills with templates
- Load and execute skills
- Import community skills

### 📊 Analytics Dashboard
- Track prompt executions
- Monitor token usage and costs
- View execution history
- Export analytics data

### 🔄 Loop Composition
- Design complex prompt workflows
- Chain multiple prompts
- Recursive processing
- Visual flow editor

---

## 🎯 Quick Start

### Installation

```bash
# Install from VS Code Marketplace
ext install promptly-vscode

# Or install from VSIX
code --install-extension promptly-vscode-1.0.0.vsix
```

### Setup

1. **Configure Backend**
   ```
   File → Preferences → Settings → Promptly
   ```
   - Choose backend: Ollama, OpenAI, or Anthropic
   - Set API keys (if needed)
   - Configure model

2. **Open Promptly Panel**
   - Click Promptly icon in Activity Bar
   - Or: `Ctrl+Shift+P` → "Promptly: Open Panel"

3. **Execute First Prompt**
   - Select some code
   - Right-click → "Send Selection to Promptly"
   - Choose action (Analyze, Improve, Explain, etc.)

---

## 📖 Usage Examples

### Execute Selection

1. Select code in editor
2. Press `Ctrl+Shift+E` or right-click
3. Choose "Send Selection to Promptly"
4. Pick an action or enter custom prompt
5. View results in new editor tab

### Use Quick Actions

Right-click selected code and choose:
- **Analyze Code** - Get comprehensive analysis
- **Improve Code** - Receive refactoring suggestions
- **Explain Code** - Understand what it does
- **Find Bugs** - Detect potential issues

### Create a Skill

1. Open Skills view in Promptly sidebar
2. Click `+` icon
3. Enter skill name
4. Edit skill template in opened file
5. Save and use!

### View Analytics

1. Open Analytics view in Promptly sidebar
2. See execution stats, costs, token usage
3. Click items for detailed history
4. Export data with export button

---

## ⚙️ Configuration

### Settings

```json
{
  // Backend configuration
  "promptly.backend": "ollama",  // ollama | openai | anthropic
  "promptly.ollamaUrl": "http://localhost:11434",
  "promptly.model": "llama3.2:3b",
  "promptly.apiKey": "",  // For OpenAI/Anthropic

  // Features
  "promptly.autoSave": true,
  "promptly.showCost": true,
  "promptly.enableAnalytics": true
}
```

### Keyboard Shortcuts

| Shortcut | Command |
|----------|---------|
| `Ctrl+Shift+P` | Open Prompt Panel |
| `Ctrl+Shift+E` | Execute Selection |
| `F1` → "Promptly" | Show all commands |

---

## 🏗️ Architecture

```
vscode-extension/
├── src/
│   ├── extension.ts          # Main entry point
│   ├── panels/
│   │   └── PromptPanel.ts    # Webview panel
│   └── providers/
│       ├── SkillsProvider.ts # Skills tree view
│       ├── HistoryProvider.ts # History tree view
│       ├── AnalyticsProvider.ts # Analytics view
│       └── CodeActionProvider.ts # Inline actions
├── webview/
│   └── prompt-panel.html     # Panel UI
├── media/
│   ├── icon.svg              # Extension icon
│   └── styles.css            # Panel styles
└── package.json              # Manifest
```

---

## 🔌 Integration

### With Promptly Python
Extension communicates with Promptly Python backend via:
- HTTP API (for Ollama)
- MCP Server (Model Context Protocol)
- Direct Python calls (if installed)

### With Claude Desktop
Use Promptly MCP server with Claude Desktop:
```json
{
  "mcpServers": {
    "promptly": {
      "command": "python",
      "args": ["path/to/promptly/mcp_server.py"]
    }
  }
}
```

---

## 📦 Building from Source

### Prerequisites
- Node.js 18+
- npm or yarn
- VS Code 1.80+

### Build Steps

```bash
# Install dependencies
cd vscode-extension
npm install

# Compile TypeScript
npm run compile

# Watch mode (development)
npm run watch

# Package extension
npm run package  # Creates promptly-vscode-1.0.0.vsix

# Install locally
code --install-extension promptly-vscode-1.0.0.vsix
```

### Development

```bash
# Open in VS Code
code vscode-extension/

# Press F5 to launch Extension Development Host
# Make changes, reload window to test
```

---

## 🎨 Features in Detail

### Prompt Panel

Full-featured prompt composer:
- Syntax highlighting
- Template variables
- Prompt history
- Quick actions
- Cost estimation
- Real-time preview

### Skills System

Create reusable prompt skills:
```yaml
name: summarize-code
description: Summarize code functionality
template: |
  Summarize the following code:

  {code}

  Focus on:
  - Main purpose
  - Key functions
  - Dependencies
parameters:
  - code
```

### Analytics

Track and optimize:
- Executions per day/week/month
- Token usage trends
- Cost breakdown by model
- Most used skills
- Success/failure rates

---

## 🛠️ Troubleshooting

### Extension Not Activating
- Check VS Code version >= 1.80
- Reload window: `Ctrl+Shift+P` → "Reload Window"
- Check Output panel → "Promptly" for errors

### Backend Connection Issues
- Verify Ollama running: `curl http://localhost:11434`
- Check API keys for OpenAI/Anthropic
- Test in Promptly Python CLI first

### Commands Not Showing
- Ensure extension activated
- Try: `Ctrl+Shift+P` → "Promptly"
- Check keybindings aren't conflicting

---

## 📚 Resources

- **Documentation**: [/docs/VSCODE_EXTENSION_DESIGN.md](../docs/VSCODE_EXTENSION_DESIGN.md)
- **Python Integration**: [/promptly/integrations/](../promptly/integrations/)
- **MCP Server**: [/promptly/integrations/mcp_server.py](../promptly/integrations/mcp_server.py)
- **Issues**: GitHub Issues

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes in `vscode-extension/`
4. Test with `F5` (Extension Development Host)
5. Submit pull request

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 🎉 What's Next

**Coming Soon:**
- [ ] Visual loop composer
- [ ] Multi-file context
- [ ] Inline diff view
- [ ] Team collaboration
- [ ] Prompt marketplace
- [ ] Custom themes

---

**Built with ❤️ for prompt engineers**

Transform your coding workflow with AI-powered prompts!
