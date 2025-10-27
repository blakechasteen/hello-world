# VS Code Extension - MVP Started ✓

## Overview

VS Code extension structure created and ready for development. All architecture in place, needs TypeScript implementation.

## Files Created

### Core Files
- ✓ `package.json` - Extension manifest with all commands, views, config
- ✓ `src/extension.ts` - Main entry point with activation logic

### Structure (Ready for Implementation)

```
vscode-extension/
├── package.json              ✓ Complete
├── tsconfig.json            ⏳ Needed
├── webpack.config.js        ⏳ Needed
├── README.md                ⏳ Needed
│
├── src/
│   ├── extension.ts         ✓ Complete
│   │
│   ├── commands/            ⏳ To implement
│   │   ├── index.ts
│   │   ├── promptCommands.ts
│   │   ├── skillCommands.ts
│   │   └── loopCommands.ts
│   │
│   ├── providers/           ⏳ To implement
│   │   ├── promptProvider.ts
│   │   ├── skillProvider.ts
│   │   └── analyticsProvider.ts
│   │
│   ├── webviews/            ⏳ To implement
│   │   ├── analyticsPanel.ts
│   │   └── composerPanel.ts
│   │
│   ├── api/                 ⏳ To implement
│   │   ├── promptlyClient.ts
│   │   └── types.ts
│   │
│   └── utils/               ⏳ To implement
│       ├── config.ts
│       └── logger.ts
│
└── resources/               ⏳ To create
    └── icon.svg
```

## What's Included

### Commands (8 total)
1. **promptly.refreshPrompts** - Refresh all data
2. **promptly.addPrompt** - Create new prompt
3. **promptly.executePrompt** - Run prompt with LLM
4. **promptly.createSkill** - Create new skill
5. **promptly.installTemplate** - Install from template library
6. **promptly.showAnalytics** - Open analytics dashboard
7. **promptly.composeLoops** - Build loop pipeline
8. **promptly.exportCharts** - Export analytics as PNG

### Views (3 sidebar panels)
1. **Prompts** - Tree view of all prompts
2. **Skills** - Tree view of all skills
3. **Analytics** - Quick stats summary

### Configuration Options
- `promptly.backend` - ollama or claude_api
- `promptly.ollamaUrl` - Ollama server URL
- `promptly.ollamaModel` - Default model
- `promptly.mcpServerPath` - Path to MCP server
- `promptly.showAnalyticsOnStartup` - Auto-open analytics
- `promptly.autoRefresh` - Refresh every 30s

### Features Implemented in extension.ts
- ✓ Tree view registration
- ✓ Command registration
- ✓ Configuration loading
- ✓ Status bar item
- ✓ Auto-refresh timer
- ✓ Analytics on startup

## Next Implementation Steps

### Phase 1: Core Infrastructure (2-3 days)
1. Create `tsconfig.json` and `webpack.config.js`
2. Implement `promptlyClient.ts` - API communication
3. Create TypeScript types in `types.ts`
4. Add error handling and logging utilities

### Phase 2: Tree Providers (3-4 days)
1. **PromptTreeProvider**
   - List all prompts
   - Show version, branch
   - Click to execute

2. **SkillTreeProvider**
   - List all skills
   - Show attached files
   - Expand to see files

3. **AnalyticsTreeProvider**
   - Show top 3 prompts
   - Success rate
   - Quick stats

### Phase 3: Commands (4-5 days)
1. **Add/Execute Prompts**
   - Input boxes for name/content
   - Variable substitution
   - Backend selection
   - Results in new editor

2. **Skill Management**
   - Create skill wizard
   - Template installer
   - File attachment

3. **Loop Composition**
   - Step-by-step builder
   - Preview pipeline
   - Execute and show results

### Phase 4: Webviews (5-6 days)
1. **Analytics Dashboard**
   - Embed web dashboard HTML
   - Communication via postMessage
   - Live data from API
   - Chart export

2. **Loop Composer** (future)
   - Visual pipeline builder
   - Drag-and-drop interface
   - Live preview

### Phase 5: Polish (2-3 days)
1. Icons and branding
2. Welcome page
3. Keyboard shortcuts
4. Error messages
5. Loading states
6. Testing

## Development Setup

### Prerequisites
```bash
npm install -g yo generator-code
node --version  # v20+
npm --version   # v10+
```

### Initialize Project
```bash
cd vscode-extension
npm install
```

### Development
```bash
npm run watch   # Compile TypeScript
# Press F5 in VS Code to launch Extension Development Host
```

### Package
```bash
npm run package
vsce package    # Creates .vsix file
```

### Publish
```bash
vsce login promptly
vsce publish
```

## API Client Design

### PromptlyClient Class

```typescript
export class PromptlyClient {
    constructor(private mcpServerPath: string) {}

    async listPrompts(): Promise<Prompt[]> {
        // Call MCP: promptly_list
    }

    async addPrompt(name: string, content: string): Promise<string> {
        // Call MCP: promptly_add
    }

    async executePrompt(name: string, inputs: any): Promise<ExecutionResult> {
        // Call MCP: promptly_execute_prompt
    }

    async getAnalyticsSummary(): Promise<AnalyticsSummary> {
        // Call MCP: promptly_analytics_summary
    }

    async composeLoops(task: string, steps: Step[]): Promise<CompositionResult> {
        // Call MCP: promptly_compose_loops
    }
}
```

### Communication Options

**Option 1: Spawn Python Process** (Recommended)
```typescript
import { spawn } from 'child_process';

const mcpServer = spawn('python', [mcpServerPath], {
    stdio: ['pipe', 'pipe', 'pipe']
});

// Send JSON-RPC over stdio
```

**Option 2: HTTP Bridge**
- Create Flask wrapper around MCP
- Use axios for HTTP requests
- Simpler but requires server running

## UI/UX Design

### Sidebar Tree View
```
PROMPTLY
├─ 📝 PROMPTS
│  ├─ sql_optimizer (v2) ▶️
│  ├─ code_reviewer (v3) ▶️
│  └─ ui_designer (v1) ▶️
│
├─ 🎯 SKILLS
│  ├─ sql_optimizer
│  │  ├─ optimizer.md
│  │  └─ optimize.py
│  └─ code_reviewer
│
└─ 📊 ANALYTICS
   ├─ Total: 340 executions
   ├─ Success: 95.2%
   └─ Top: sql_optimizer
```

### Status Bar
```
[Promptly] [Backend: Ollama] [Model: llama3.2:3b]
```

### Command Palette
```
> Promptly: Execute Prompt
  Pick a prompt: sql_optimizer
  Enter task: SELECT * FROM users
  Choose backend: Ollama (llama3.2:3b)
  ✓ Executed in 14.2s
```

## Integration Points

### With Promptly MCP Server
- Uses all 27 MCP tools
- Async communication
- Error handling
- Progress indicators

### With Web Dashboard
- Embeds HTML in webview
- Shares analytics data
- Same Chart.js visualizations
- Export functionality

### With Editor
- Execute selected text as prompt
- Insert results at cursor
- Create prompts from selection
- Diff view for versions

## Estimated Timeline

**MVP (Core Functionality):** 3-4 weeks
- Week 1: Infrastructure + Tree Providers
- Week 2: Commands + Basic Execution
- Week 3: Analytics Dashboard Webview
- Week 4: Polish + Testing

**Full Feature Set:** 6-8 weeks
- MVP + Loop Composer
- Advanced webviews
- Keyboard shortcuts
- Marketplace polish

## Success Criteria

### MVP Complete When:
- [x] Extension structure created
- [ ] Can list prompts in sidebar
- [ ] Can execute prompts
- [ ] Analytics dashboard opens
- [ ] Skills can be created
- [ ] Templates can be installed
- [ ] Package builds successfully

### Production Ready When:
- [ ] All 27 MCP tools accessible
- [ ] Tests passing
- [ ] Error handling complete
- [ ] Icons and branding done
- [ ] README with examples
- [ ] Published to marketplace

## Next Actions

### Immediate (Today)
1. Create `tsconfig.json` and `webpack.config.js`
2. Implement `PromptlyClient` class
3. Create basic types in `types.ts`

### This Week
1. Implement PromptTreeProvider
2. Implement basic commands
3. Test with local MCP server

### This Month
1. Complete all tree providers
2. All commands working
3. Analytics dashboard embedded
4. Ready for beta testing

## Resources Needed

### NPM Packages
- `@types/vscode` - VS Code API types
- `@types/node` - Node.js types
- `webpack` + `ts-loader` - Bundling
- `axios` - HTTP client (if using HTTP bridge)

### Documentation
- VS Code Extension API docs
- Chart.js documentation
- MCP protocol spec

### Testing
- Manual testing in Extension Development Host
- Unit tests with Mocha
- Integration tests with MCP server

---

*Started: 2025-10-26*
*Status: MVP Structure Complete, Implementation In Progress*
*ETA: 3-4 weeks to functional MVP*
