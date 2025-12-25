# Promptly VS Code Extension

VS Code extension for managing prompts, chains, and skills with the Promptly CLI.

## Features

### Prompt Management
- **Add/Edit Prompts**: Create and modify prompts with version control
- **Version History**: View and restore previous versions
- **Branch Support**: Create and switch between branches
- **Insert to Editor**: Insert prompts at cursor position
- **Execute Prompts**: Run prompts with LLM and see results

### Chain Execution
- **Visual Chain Builder**: Create multi-step prompt chains
- **Run Chains**: Execute chains with variable inputs
- **Step-by-Step Visualization**: See chain execution flow

### Skill Management
- **Reusable Skills**: Create parameterized prompt templates
- **Input/Output Schemas**: Define structured inputs and outputs
- **Run Skills**: Execute skills with structured inputs

### LLM Judge Integration
- **Evaluate Prompts**: Score prompts using LLM-based evaluation
- **12 Criteria**: Clarity, accuracy, relevance, and more
- **Improvement Suggestions**: Get actionable feedback

## Installation

### Prerequisites

1. **Python 3.8+** with Promptly CLI installed:
   ```bash
   pip install -e /path/to/promptly
   ```

2. **Node.js 16+** for building the extension

### Build from Source

```bash
cd promptly/vscode

# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Package extension
npm run package
```

### Install Extension

1. Open VS Code
2. Press `Ctrl+Shift+P` → "Extensions: Install from VSIX..."
3. Select the generated `.vsix` file

Or for development:
```bash
# Open extension in VS Code
code --extensionDevelopmentPath=/path/to/promptly/vscode
```

## Usage

### Command Palette

All commands are available via Command Palette (`Ctrl+Shift+P`):

- `Promptly: Add Prompt` - Create a new prompt
- `Promptly: Get Prompt` - View a prompt
- `Promptly: Execute Prompt` - Run a prompt with LLM
- `Promptly: Insert Prompt` - Insert prompt at cursor
- `Promptly: Create Chain` - Create a prompt chain
- `Promptly: Run Chain` - Execute a chain
- `Promptly: Create Skill` - Create a reusable skill
- `Promptly: Run Skill` - Execute a skill
- `Promptly: Evaluate Prompt` - Run LLM Judge on a prompt

### Keyboard Shortcuts

- `Ctrl+Shift+I` - Insert prompt at cursor
- `Ctrl+Shift+E` - Execute prompt

### Sidebar Views

Three tree views in the Explorer sidebar:
- **Prompts** - All prompts with version history
- **Chains** - Prompt chains with step details
- **Skills** - Reusable skills with schemas

### Configuration

Settings available in VS Code Settings (`Ctrl+,`):

| Setting | Description | Default |
|---------|-------------|---------|
| `promptly.pythonPath` | Path to Python executable | `python` |
| `promptly.scope` | Default scope (local/global) | `local` |
| `promptly.llmBackend` | LLM backend (auto/ollama/claude) | `auto` |
| `promptly.ollamaModel` | Ollama model for execution | `llama3.2:3b` |
| `promptly.autoRefresh` | Auto-refresh tree views | `true` |

## Development

### Project Structure

```
promptly/vscode/
├── src/
│   ├── extension.ts          # Extension entry point
│   ├── PromptlyBridge.ts     # Python CLI bridge
│   └── providers/
│       ├── PromptTreeProvider.ts   # Prompt tree view
│       ├── ChainTreeProvider.ts    # Chain tree view
│       └── SkillTreeProvider.ts    # Skill tree view
├── package.json              # Extension manifest
└── tsconfig.json             # TypeScript config
```

### Building

```bash
# Watch mode for development
npm run watch

# Single compile
npm run compile

# Package for distribution
npm run package
```

### Testing

```bash
# Run tests
npm test

# Run with coverage
npm run test:coverage
```

## Troubleshooting

### Python Not Found

If you see "Python not found" errors:
1. Set `promptly.pythonPath` to your Python executable path
2. Ensure Promptly is installed: `pip show promptly`

### No Prompts Shown

1. Check if Promptly database exists: `~/.promptly/prompts.db` or `.promptly/prompts.db`
2. Try `promptly list` in terminal to verify CLI works
3. Check Output panel → "Promptly" for error messages

### LLM Execution Fails

1. Ensure Ollama is running: `ollama serve`
2. Check model is available: `ollama list`
3. Set `promptly.ollamaModel` to an installed model

## License

MIT License - see LICENSE file
