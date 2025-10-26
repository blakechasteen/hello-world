# Promptly VS Code Extension - Architecture Design

## Overview

VS Code extension that integrates Promptly's recursive intelligence directly into the editor for seamless prompt engineering and execution.

## Core Features

### 1. Prompt Editor
- Syntax highlighting for prompts
- Variable autocomplete (`{variable_name}`)
- Template snippets
- Live preview
- Version diff viewer

### 2. Skill Manager
- Create/edit skills in VS Code
- Attach files to skills
- Install from template library
- Skill explorer sidebar

### 3. Execution Panel
- Execute prompts/skills inline
- Choose backend (Ollama/Claude API)
- View results in editor
- Export to file

### 4. Loop Composer
- Visual pipeline builder
- Drag-and-drop loop steps
- Configure iterations/thresholds
- Execute compositions
- View step-by-step results

### 5. Analytics Dashboard
- Embedded webview
- Same UI as web dashboard
- Prompt performance charts
- Recommendations panel

### 6. Quick Actions
- Command palette integration
- Right-click context menus
- Keyboard shortcuts
- Status bar indicators

## Architecture

### Extension Structure

```
promptly-vscode/
├── package.json                 # Extension manifest
├── tsconfig.json               # TypeScript config
├── README.md
├── CHANGELOG.md
├── LICENSE
│
├── src/
│   ├── extension.ts            # Entry point
│   ├── commands/               # Command handlers
│   │   ├── promptCommands.ts   # Add/get/list prompts
│   │   ├── skillCommands.ts    # Skill management
│   │   ├── executeCommands.ts  # Execution handlers
│   │   └── loopCommands.ts     # Composition commands
│   │
│   ├── providers/              # VS Code providers
│   │   ├── promptProvider.ts   # Prompt tree view
│   │   ├── skillProvider.ts    # Skill tree view
│   │   ├── hoverProvider.ts    # Variable hover info
│   │   └── completionProvider.ts # Autocomplete
│   │
│   ├── webviews/               # Webview panels
│   │   ├── analyticsPanel.ts   # Analytics dashboard
│   │   ├── composerPanel.ts    # Loop composer
│   │   └── executionPanel.ts   # Execution results
│   │
│   ├── api/                    # Promptly integration
│   │   ├── promptlyClient.ts   # Core API client
│   │   ├── mcpClient.ts        # MCP server client
│   │   └── types.ts            # TypeScript types
│   │
│   ├── ui/                     # UI components
│   │   ├── statusBar.ts        # Status bar items
│   │   ├── quickPick.ts        # Quick pick menus
│   │   └── notifications.ts    # Toast messages
│   │
│   └── utils/                  # Utilities
│       ├── config.ts           # Settings management
│       ├── logger.ts           # Logging
│       └── validation.ts       # Input validation
│
├── resources/                  # Static resources
│   ├── icons/                  # Extension icons
│   │   ├── promptly.svg
│   │   ├── prompt.svg
│   │   ├── skill.svg
│   │   └── loop.svg
│   │
│   └── snippets/               # Code snippets
│       └── promptly.json
│
├── media/                      # Webview assets
│   ├── css/
│   │   └── dashboard.css
│   ├── js/
│   │   ├── analytics.js
│   │   └── composer.js
│   └── images/
│
└── test/                       # Tests
    ├── suite/
    │   ├── extension.test.ts
    │   ├── commands.test.ts
    │   └── providers.test.ts
    └── runTest.ts
```

## VS Code Extension API Usage

### Tree View Providers

```typescript
// Prompts sidebar
export class PromptTreeProvider implements vscode.TreeDataProvider<PromptItem> {
  async getChildren(element?: PromptItem): Promise<PromptItem[]> {
    // Fetch prompts from Promptly
  }

  getTreeItem(element: PromptItem): vscode.TreeItem {
    // Return tree item with icon, label, commands
  }
}

// Skills sidebar
export class SkillTreeProvider implements vscode.TreeDataProvider<SkillItem> {
  // Similar to PromptTreeProvider
}
```

### Command Handlers

```typescript
// Register commands
export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand('promptly.addPrompt', addPrompt),
    vscode.commands.registerCommand('promptly.executePrompt', executePrompt),
    vscode.commands.registerCommand('promptly.composeLoops', composeLoops),
    vscode.commands.registerCommand('promptly.showAnalytics', showAnalytics)
  );
}

// Command implementation
async function executePrompt(promptName?: string) {
  // Get prompt from tree or input
  // Show quick pick for backend
  // Execute and show results
}
```

### Webview Panels

```typescript
export class AnalyticsDashboardPanel {
  private panel: vscode.WebviewPanel;

  constructor(extensionUri: vscode.Uri) {
    this.panel = vscode.window.createWebviewPanel(
      'promptlyAnalytics',
      'Promptly Analytics',
      vscode.ViewColumn.Two,
      {
        enableScripts: true,
        retainContextWhenHidden: true
      }
    );

    this.panel.webview.html = this.getHtmlContent();
  }

  private getHtmlContent(): string {
    // Load dashboard HTML with embedded API
  }
}
```

### Language Features

```typescript
// Hover provider for variables
export class PromptHoverProvider implements vscode.HoverProvider {
  provideHover(
    document: vscode.TextDocument,
    position: vscode.Position
  ): vscode.ProviderResult<vscode.Hover> {
    const range = document.getWordRangeAtPosition(position, /\{[a-zA-Z_]+\}/);
    if (range) {
      const variable = document.getText(range);
      return new vscode.Hover(`Prompt variable: ${variable}`);
    }
  }
}

// Completion provider for variables
export class PromptCompletionProvider implements vscode.CompletionItemProvider {
  provideCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position
  ): vscode.ProviderResult<vscode.CompletionItem[]> {
    // Suggest variable names
    return [
      new vscode.CompletionItem('{task}', vscode.CompletionItemKind.Variable),
      new vscode.CompletionItem('{input}', vscode.CompletionItemKind.Variable)
    ];
  }
}
```

## Configuration Schema

```json
{
  "promptly.backend": {
    "type": "string",
    "enum": ["ollama", "claude_api"],
    "default": "ollama",
    "description": "Default execution backend"
  },
  "promptly.ollamaUrl": {
    "type": "string",
    "default": "http://localhost:11434",
    "description": "Ollama server URL"
  },
  "promptly.ollamaModel": {
    "type": "string",
    "default": "llama3.2:3b",
    "description": "Default Ollama model"
  },
  "promptly.claudeApiKey": {
    "type": "string",
    "default": "",
    "description": "Claude API key (stored in keychain)"
  },
  "promptly.mcpServerPath": {
    "type": "string",
    "default": "",
    "description": "Path to Promptly MCP server"
  },
  "promptly.showAnalyticsOnStartup": {
    "type": "boolean",
    "default": false,
    "description": "Show analytics dashboard on startup"
  }
}
```

## Command Palette Commands

```
Promptly: Add Prompt
Promptly: Get Prompt
Promptly: List Prompts
Promptly: Execute Prompt
Promptly: Create Skill
Promptly: Install Skill Template
Promptly: Execute Skill
Promptly: Compose Loops
Promptly: Show Analytics
Promptly: Refine Iteratively
Promptly: Hofstadter Loop
Promptly: Show Recommendations
```

## UI Components

### 1. Sidebar View

```
PROMPTLY
├─ PROMPTS
│  ├─ sql_optimizer (v2)
│  ├─ code_reviewer (v3)
│  └─ ui_designer (v1)
│
├─ SKILLS
│  ├─ sql_optimizer
│  │  ├─ optimizer.md
│  │  └─ optimize.py
│  ├─ code_reviewer
│  └─ ui_designer
│
└─ ANALYTICS
   ├─ Summary
   ├─ Top Performers
   └─ Recommendations
```

### 2. Status Bar Items

```
[Promptly] [Backend: Ollama] [Model: llama3.2:3b] [Executions: 42]
```

### 3. Quick Pick Menus

**Execute Prompt:**
```
> sql_optimizer (v2) - Optimize SQL queries
  code_reviewer (v3) - Review code for quality
  ui_designer (v1) - Design accessible UIs
```

**Choose Backend:**
```
> Ollama (llama3.2:3b) - Local, free, fast
  Claude API (claude-3-5-sonnet) - Cloud, paid, powerful
```

### 4. Webview Panels

**Analytics Dashboard:**
- Same as web dashboard
- Embedded in VS Code
- Communication via postMessage API

**Loop Composer:**
```
┌─────────────────────────────────┐
│ Loop Composition Builder        │
├─────────────────────────────────┤
│                                 │
│  [Critique] → [Refine] → [Verify]
│                                 │
│  ┌───────────────────────────┐  │
│  │ Step 1: Critique          │  │
│  │ Iterations: [1         ]  │  │
│  │ Description: Analyze code │  │
│  └───────────────────────────┘  │
│                                 │
│  [+ Add Step]  [Execute]        │
└─────────────────────────────────┘
```

## API Client Integration

### Promptly Client

```typescript
export class PromptlyClient {
  constructor(private mcpServerPath: string) {}

  async addPrompt(name: string, content: string): Promise<string> {
    // Call MCP tool: promptly_add
  }

  async executePrompt(name: string, inputs: any): Promise<ExecutionResult> {
    // Call MCP tool: promptly_execute_prompt
  }

  async composeLoops(task: string, steps: CompositionStep[]): Promise<CompositionResult> {
    // Call MCP tool: promptly_compose_loops
  }

  async getAnalyticsSummary(): Promise<AnalyticsSummary> {
    // Call MCP tool: promptly_analytics_summary
  }
}
```

### MCP Server Communication

Two options:

**Option 1: Spawn MCP Server Process**
```typescript
import { spawn } from 'child_process';

const mcpServer = spawn('python', [mcpServerPath]);
// Communicate via stdio
```

**Option 2: Direct Python Integration**
```typescript
// Use python-shell npm package
import { PythonShell } from 'python-shell';

const result = await PythonShell.run('mcp_server.py', options);
```

**Option 3: HTTP Bridge**
```typescript
// Create HTTP wrapper around MCP server
// Use axios for requests
```

## User Workflows

### Workflow 1: Execute Prompt

1. User opens command palette (Ctrl+Shift+P)
2. Types "Promptly: Execute Prompt"
3. Selects prompt from quick pick
4. Enters variable values
5. Chooses backend
6. Results shown in new editor tab

### Workflow 2: Create Skill

1. User right-clicks in sidebar
2. Selects "Create New Skill"
3. Enters skill name and description
4. Adds files (markdown, code, config)
5. Skill appears in sidebar tree

### Workflow 3: Compose Loops

1. User opens "Promptly: Compose Loops"
2. Loop composer webview opens
3. Drags loop types to canvas
4. Configures each step
5. Clicks "Execute"
6. Results shown step-by-step

### Workflow 4: View Analytics

1. User clicks analytics icon in sidebar
2. Analytics dashboard opens in webview
3. Auto-refreshes every 30s
4. Click prompt to see details
5. View recommendations

## Key Features

### 1. Inline Execution
```typescript
// Select text in editor
const selection = editor.selection;
const text = editor.document.getText(selection);

// Execute as prompt
const result = await client.executePrompt('code_reviewer', { code: text });

// Insert result below selection
editor.edit(editBuilder => {
  editBuilder.insert(selection.end, '\n\n' + result.output);
});
```

### 2. Git Integration
- Track prompt versions in git
- Diff viewer for changes
- Branch support

### 3. Export/Import
- Export prompts/skills as .promptly package
- Import from VS Code marketplace
- Share with team

### 4. Collaborative Features
- Share prompts via gist
- Team templates
- Skill marketplace

## Technology Stack

### Core
- **Language:** TypeScript
- **Build:** webpack
- **Testing:** Mocha + Chai
- **Linting:** ESLint + Prettier

### VS Code API
- TreeView API
- Webview API
- Commands API
- Language Features API
- Configuration API

### UI Libraries
- **Webviews:** Same HTML/CSS/JS as web dashboard
- **Icons:** Codicons (VS Code's icon font)
- **Charts:** Chart.js (for analytics)

## Development Setup

```bash
# Clone template
npm install -g yo generator-code
yo code

# Project structure
cd promptly-vscode
npm install

# Development
npm run watch    # Compile TypeScript
F5              # Launch extension in debug mode

# Testing
npm test

# Package
vsce package    # Create .vsix file
```

## Packaging & Distribution

### package.json

```json
{
  "name": "promptly",
  "displayName": "Promptly - Recursive Intelligence",
  "description": "AI-powered prompt engineering with recursive loops",
  "version": "0.1.0",
  "publisher": "promptly",
  "engines": {
    "vscode": "^1.85.0"
  },
  "categories": [
    "AI",
    "Programming Languages",
    "Snippets"
  ],
  "activationEvents": [
    "onView:promptlyPrompts",
    "onCommand:promptly.addPrompt"
  ],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [...],
    "viewsContainers": {...},
    "views": {...},
    "configuration": {...}
  }
}
```

### Publishing

```bash
# Get publisher token from VS Code Marketplace
vsce login promptly

# Publish
vsce publish
```

## Security Considerations

1. **API Keys**
   - Store in VS Code secret storage
   - Never log or display
   - Use keychain on macOS/Windows

2. **MCP Server**
   - Spawn as child process
   - Validate all inputs
   - Sandbox execution

3. **Webviews**
   - Enable CSP (Content Security Policy)
   - Sanitize user input
   - Use nonces for scripts

## Performance Optimization

1. **Lazy Loading**
   - Load webviews on demand
   - Defer analytics fetching
   - Cache prompt list

2. **Debouncing**
   - Debounce autocomplete
   - Throttle analytics refresh

3. **Background Tasks**
   - Run long executions in background
   - Show progress notifications
   - Allow cancellation

## Future Enhancements

### Phase 1 (MVP)
- [x] Basic prompt management
- [x] Skill creation
- [x] Simple execution
- [x] Analytics dashboard

### Phase 2 (Advanced)
- [ ] Loop composer UI
- [ ] Inline execution
- [ ] Git integration
- [ ] Export/import

### Phase 3 (Collaborative)
- [ ] Skill marketplace
- [ ] Team sharing
- [ ] Remote execution
- [ ] Cloud sync

### Phase 4 (AI Features)
- [ ] Prompt suggestions
- [ ] Auto-optimization
- [ ] Smart completions
- [ ] Quality predictions

## Alternatives Considered

### 1. Web-Only (No Extension)
**Pros:** Easier to build, cross-platform
**Cons:** Less integrated, no editor features

### 2. JetBrains Plugin
**Pros:** Reaches IntelliJ users
**Cons:** Smaller market than VS Code

### 3. CLI Only
**Pros:** Simplest, most portable
**Cons:** Less discoverable, steeper learning curve

**Decision:** VS Code extension provides best UX for developers

## Success Metrics

- Downloads per month
- Active users
- Prompt executions via extension
- User ratings
- GitHub stars
- Community contributions

## Summary

VS Code extension brings Promptly's recursive intelligence directly into developers' workflow with:
- Seamless prompt/skill management
- Inline execution with results
- Visual loop composition
- Real-time analytics
- Professional UI/UX

**Total Estimated Development Time:** 4-6 weeks
**Lines of Code:** ~8,000
**Extension Size:** ~500KB
