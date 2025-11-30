# Promptly VS Code Extension - Slash Commands Implementation

## Vision: Chat-Native Development Inside VS Code

Instead of switching between VS Code → Terminal → Browser, bring everything into VS Code with intuitive slash commands.

```
VS Code Editor
    ↓ (Type slash command)
/gs                  → Git status inline
/review              → Claude reviews current file
/remember "note"     → Save to HoloLoom
/recall "query"      → Query HoloLoom knowledge graph
/optimize            → Optimize selected prompt
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│           VS Code Extension (TypeScript)        │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │     Slash Command Handler                │  │
│  │  - /gs, /gc, /gp (git)                  │  │
│  │  - /review, /explain (Claude)           │  │
│  │  - /remember, /recall (HoloLoom)        │  │
│  └──────────────────┬───────────────────────┘  │
│                     │                           │
│  ┌──────────────────▼───────────────────────┐  │
│  │     Command Router                       │  │
│  │  Routes to appropriate backend           │  │
│  └──────────────────┬───────────────────────┘  │
└────────────────────┼─────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌───▼──────┐
    │  Git    │ │ Claude  │ │ HoloLoom │
    │ (local) │ │  API    │ │  Server  │
    └─────────┘ └─────────┘ └──────────┘
```

## Quick Start (30 Minutes)

### Step 1: Generate Extension Scaffold (5 min)

```bash
cd mythRL
npm install -g yo generator-code
yo code

# Answer prompts:
? What type of extension? New Extension (TypeScript)
? Extension name? Promptly
? Identifier? promptly
? Description? Chat-native development with slash commands
? Initialize git? No (we're already in a repo)
? Package manager? npm
```

This creates:
```
promptly-vscode/
├── src/
│   ├── extension.ts        # Main entry point
│   └── commands/
│       ├── gitCommands.ts
│       ├── claudeCommands.ts
│       └── hololoomCommands.ts
├── package.json
├── tsconfig.json
└── README.md
```

### Step 2: Configure Slash Commands (10 min)

Edit `package.json`:

```json
{
  "name": "promptly",
  "displayName": "Promptly",
  "description": "Chat-native development with slash commands",
  "version": "0.1.0",
  "engines": {
    "vscode": "^1.80.0"
  },
  "categories": ["Other"],
  "activationEvents": ["onStartupFinished"],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "promptly.gitStatus",
        "title": "Promptly: Git Status",
        "shortTitle": "/gs"
      },
      {
        "command": "promptly.gitCommit",
        "title": "Promptly: Git Commit",
        "shortTitle": "/gc"
      },
      {
        "command": "promptly.gitPush",
        "title": "Promptly: Git Push",
        "shortTitle": "/gp"
      },
      {
        "command": "promptly.review",
        "title": "Promptly: Review Code",
        "shortTitle": "/review"
      },
      {
        "command": "promptly.explain",
        "title": "Promptly: Explain Code",
        "shortTitle": "/explain"
      },
      {
        "command": "promptly.remember",
        "title": "Promptly: Remember (HoloLoom)",
        "shortTitle": "/remember"
      },
      {
        "command": "promptly.recall",
        "title": "Promptly: Recall (HoloLoom)",
        "shortTitle": "/recall"
      },
      {
        "command": "promptly.chat",
        "title": "Promptly: Open Chat",
        "shortTitle": "/chat"
      }
    ],
    "keybindings": [
      {
        "command": "promptly.chat",
        "key": "ctrl+shift+p",
        "mac": "cmd+shift+p"
      }
    ],
    "configuration": {
      "title": "Promptly",
      "properties": {
        "promptly.hololoomUrl": {
          "type": "string",
          "default": "http://localhost:8000",
          "description": "HoloLoom server URL"
        },
        "promptly.claudeApiKey": {
          "type": "string",
          "default": "",
          "description": "Anthropic API key for Claude"
        },
        "promptly.matrixServer": {
          "type": "string",
          "default": "https://matrix.org",
          "description": "Matrix server (optional)"
        }
      }
    }
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./"
  },
  "devDependencies": {
    "@types/vscode": "^1.80.0",
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0"
  },
  "dependencies": {
    "axios": "^1.6.0",
    "@anthropic-ai/sdk": "^0.9.0"
  }
}
```

### Step 3: Implement Chat Interface (5 min)

Create `src/chatView.ts`:

```typescript
import * as vscode from 'vscode';

export class PromptlyChatView {
    private panel: vscode.WebviewPanel | undefined;

    public show() {
        if (this.panel) {
            this.panel.reveal();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'promptlyChat',
            'Promptly Chat',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        this.panel.webview.html = this.getHtmlContent();

        // Handle messages from webview
        this.panel.webview.onDidReceiveMessage(
            async (message) => {
                await this.handleChatMessage(message);
            }
        );

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });
    }

    private async handleChatMessage(message: any) {
        const command = message.text.trim();

        // Parse slash commands
        if (command.startsWith('/')) {
            await this.executeSlashCommand(command);
        } else {
            // Natural language - route to HoloLoom or Ollama
            await this.handleNaturalLanguage(command);
        }
    }

    private async executeSlashCommand(command: string) {
        const parts = command.split(' ');
        const cmd = parts[0];
        const args = parts.slice(1);

        switch (cmd) {
            case '/gs':
            case '/git-status':
                await vscode.commands.executeCommand('promptly.gitStatus');
                break;

            case '/gc':
            case '/git-commit':
                await vscode.commands.executeCommand('promptly.gitCommit', args.join(' '));
                break;

            case '/gp':
            case '/git-push':
                await vscode.commands.executeCommand('promptly.gitPush');
                break;

            case '/review':
                await vscode.commands.executeCommand('promptly.review');
                break;

            case '/explain':
                await vscode.commands.executeCommand('promptly.explain');
                break;

            case '/remember':
                await vscode.commands.executeCommand('promptly.remember', args.join(' '));
                break;

            case '/recall':
                await vscode.commands.executeCommand('promptly.recall', args.join(' '));
                break;

            case '/help':
                this.showHelp();
                break;

            default:
                this.sendMessage(`Unknown command: ${cmd}. Type /help for available commands.`);
        }
    }

    private async handleNaturalLanguage(query: string) {
        // Route to HoloLoom server
        this.sendMessage('🤔 Thinking...');

        try {
            const config = vscode.workspace.getConfiguration('promptly');
            const hololoomUrl = config.get<string>('hololoomUrl');

            const response = await fetch(`${hololoomUrl}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: query,
                    mode: 'verify',
                    max_steps: 3
                })
            });

            const result = await response.json();

            this.sendMessage(result.response, {
                confidence: result.confidence,
                verified: result.verification?.verified
            });

        } catch (error) {
            this.sendMessage(`❌ Error: ${error}`);
        }
    }

    private sendMessage(text: string, metadata?: any) {
        this.panel?.webview.postMessage({
            type: 'response',
            text,
            metadata
        });
    }

    private showHelp() {
        const helpText = `
**Promptly Slash Commands**

**Git:**
- \`/gs\` or \`/git-status\` - Show git status
- \`/gc "message"\` - Git commit
- \`/gp\` - Git push
- \`/gl\` - Git log

**Code Review:**
- \`/review\` - Review current file with Claude
- \`/explain\` - Explain current file
- \`/refactor "task"\` - Refactor current file

**Memory (HoloLoom):**
- \`/remember "note"\` - Save to HoloLoom
- \`/recall "query"\` - Query HoloLoom knowledge graph
- \`/context\` - Show current context

**Misc:**
- \`/help\` - This help
- \`/clear\` - Clear chat

**Natural Language:**
Just type normally and Promptly will understand!
        `;

        this.sendMessage(helpText);
    }

    private getHtmlContent(): string {
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Promptly Chat</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        .message {
            margin: 10px 0;
            padding: 8px;
            border-radius: 4px;
        }
        .user {
            background: var(--vscode-input-background);
            text-align: right;
        }
        .bot {
            background: var(--vscode-editor-background);
        }
        .metadata {
            font-size: 0.9em;
            color: var(--vscode-descriptionForeground);
            margin-top: 4px;
        }
        #input-container {
            display: flex;
            padding: 10px;
            border-top: 1px solid var(--vscode-panel-border);
        }
        #input {
            flex: 1;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border-radius: 4px;
        }
        #send {
            margin-left: 8px;
            padding: 8px 16px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #send:hover {
            background: var(--vscode-button-hoverBackground);
        }
    </style>
</head>
<body>
    <div id="messages"></div>
    <div id="input-container">
        <input type="text" id="input" placeholder="Type a command (/help for help)..." />
        <button id="send">Send</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');

        function addMessage(text, sender = 'bot', metadata) {
            const div = document.createElement('div');
            div.className = \`message \${sender}\`;
            div.textContent = text;

            if (metadata) {
                const meta = document.createElement('div');
                meta.className = 'metadata';
                meta.textContent = \`Confidence: \${metadata.confidence?.toFixed(2) || 'N/A'}\`;
                div.appendChild(meta);
            }

            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function send() {
            const text = input.value.trim();
            if (!text) return;

            addMessage(text, 'user');
            vscode.postMessage({ type: 'command', text });
            input.value = '';
        }

        sendBtn.addEventListener('click', send);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') send();
        });

        // Handle responses
        window.addEventListener('message', (event) => {
            const message = event.data;
            if (message.type === 'response') {
                addMessage(message.text, 'bot', message.metadata);
            }
        });

        // Welcome message
        addMessage('👋 Hi! I\\'m Promptly. Type /help for commands or just chat naturally.', 'bot');
    </script>
</body>
</html>
        `;
    }
}
```

### Step 4: Implement Command Handlers (10 min)

Create `src/commands/gitCommands.ts`:

```typescript
import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class GitCommands {
    async status() {
        try {
            const { stdout } = await execAsync('git status --short', {
                cwd: vscode.workspace.rootPath
            });

            const output = stdout || 'Working tree clean';

            vscode.window.showInformationMessage('Git Status', {
                modal: false,
                detail: output
            });

            return output;

        } catch (error) {
            vscode.window.showErrorMessage(`Git status failed: ${error}`);
            return '';
        }
    }

    async commit(message?: string) {
        if (!message) {
            message = await vscode.window.showInputBox({
                prompt: 'Enter commit message',
                placeHolder: 'feat: Add new feature'
            });
        }

        if (!message) return;

        try {
            // Stage all changes
            await execAsync('git add .', { cwd: vscode.workspace.rootPath });

            // Commit
            const { stdout } = await execAsync(`git commit -m "${message}"`, {
                cwd: vscode.workspace.rootPath
            });

            vscode.window.showInformationMessage(`✅ Committed: ${message}`);
            return stdout;

        } catch (error) {
            vscode.window.showErrorMessage(`Commit failed: ${error}`);
            return '';
        }
    }

    async push() {
        try {
            const { stdout } = await execAsync('git push', {
                cwd: vscode.workspace.rootPath
            });

            vscode.window.showInformationMessage('✅ Pushed to remote');
            return stdout;

        } catch (error) {
            vscode.window.showErrorMessage(`Push failed: ${error}`);
            return '';
        }
    }

    async log() {
        try {
            const { stdout } = await execAsync('git log --oneline -10', {
                cwd: vscode.workspace.rootPath
            });

            const panel = vscode.window.createWebviewPanel(
                'gitLog',
                'Git Log',
                vscode.ViewColumn.Beside,
                {}
            );

            panel.webview.html = `
                <html>
                <body style="font-family: monospace; padding: 20px;">
                    <h3>Recent Commits</h3>
                    <pre>${stdout}</pre>
                </body>
                </html>
            `;

            return stdout;

        } catch (error) {
            vscode.window.showErrorMessage(`Git log failed: ${error}`);
            return '';
        }
    }
}
```

Create `src/commands/claudeCommands.ts`:

```typescript
import * as vscode from 'vscode';
import Anthropic from '@anthropic-ai/sdk';

export class ClaudeCommands {
    private client: Anthropic | null = null;

    constructor() {
        const config = vscode.workspace.getConfiguration('promptly');
        const apiKey = config.get<string>('claudeApiKey');

        if (apiKey) {
            this.client = new Anthropic({ apiKey });
        }
    }

    async review() {
        if (!this.client) {
            vscode.window.showWarningMessage('Claude API key not set. Configure in settings.');
            return;
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active file to review');
            return;
        }

        const code = editor.document.getText();
        const language = editor.document.languageId;
        const fileName = editor.document.fileName;

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Reviewing code with Claude...',
            cancellable: false
        }, async () => {
            try {
                const message = await this.client!.messages.create({
                    model: 'claude-sonnet-4',
                    max_tokens: 4096,
                    messages: [{
                        role: 'user',
                        content: `Review this ${language} code for:
1. Security vulnerabilities
2. Code quality issues
3. Performance problems
4. Best practices

File: ${fileName}

\`\`\`${language}
${code}
\`\`\`

Provide a concise review with specific line references.`
                    }]
                });

                const review = message.content[0].text;

                // Show in new document
                const doc = await vscode.workspace.openTextDocument({
                    content: review,
                    language: 'markdown'
                });

                await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);

            } catch (error) {
                vscode.window.showErrorMessage(`Review failed: ${error}`);
            }
        });
    }

    async explain() {
        if (!this.client) {
            vscode.window.showWarningMessage('Claude API key not set');
            return;
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selection = editor.selection;
        const code = selection.isEmpty
            ? editor.document.getText()
            : editor.document.getText(selection);

        const language = editor.document.languageId;

        try {
            const message = await this.client.messages.create({
                model: 'claude-sonnet-4',
                max_tokens: 2048,
                messages: [{
                    role: 'user',
                    content: `Explain this ${language} code concisely:

\`\`\`${language}
${code}
\`\`\`

What does it do? How does it work? Any gotchas?`
                }]
            });

            const explanation = message.content[0].text;

            vscode.window.showInformationMessage(explanation, {
                modal: true,
                detail: 'Code Explanation'
            });

        } catch (error) {
            vscode.window.showErrorMessage(`Explanation failed: ${error}`);
        }
    }
}
```

Create `src/commands/hololoomCommands.ts`:

```typescript
import * as vscode from 'vscode';

export class HoloLoomCommands {
    private baseUrl: string;

    constructor() {
        const config = vscode.workspace.getConfiguration('promptly');
        this.baseUrl = config.get<string>('hololoomUrl') || 'http://localhost:8000';
    }

    async remember(content: string) {
        try {
            const response = await fetch(`${this.baseUrl}/api/remember`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content })
            });

            if (response.ok) {
                vscode.window.showInformationMessage('✅ Saved to HoloLoom memory');
            } else {
                throw new Error(`HTTP ${response.status}`);
            }

        } catch (error) {
            vscode.window.showErrorMessage(`Failed to save: ${error}`);
        }
    }

    async recall(query: string) {
        try {
            const response = await fetch(`${this.baseUrl}/api/recall`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            const result = await response.json();

            // Show results in panel
            const panel = vscode.window.createWebviewPanel(
                'hololoomRecall',
                'HoloLoom Recall',
                vscode.ViewColumn.Beside,
                {}
            );

            panel.webview.html = this.getRecallHtml(result);

        } catch (error) {
            vscode.window.showErrorMessage(`Recall failed: ${error}`);
        }
    }

    private getRecallHtml(result: any): string {
        const memories = result.memories || [];

        return `
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
        }
        .memory {
            margin: 10px 0;
            padding: 10px;
            border-left: 3px solid var(--vscode-focusBorder);
            background: var(--vscode-editor-background);
        }
        .confidence {
            color: var(--vscode-descriptionForeground);
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h2>HoloLoom Recall Results</h2>
    ${memories.map((m: any) => `
        <div class="memory">
            <div>${m.content}</div>
            <div class="confidence">Confidence: ${(m.confidence * 100).toFixed(0)}%</div>
        </div>
    `).join('')}
</body>
</html>
        `;
    }
}
```

### Step 5: Wire It All Together

Edit `src/extension.ts`:

```typescript
import * as vscode from 'vscode';
import { PromptlyChatView } from './chatView';
import { GitCommands } from './commands/gitCommands';
import { ClaudeCommands } from './commands/claudeCommands';
import { HoloLoomCommands } from './commands/hololoomCommands';

export function activate(context: vscode.ExtensionContext) {
    console.log('Promptly extension activated');

    const chatView = new PromptlyChatView();
    const gitCommands = new GitCommands();
    const claudeCommands = new ClaudeCommands();
    const hololoomCommands = new HoloLoomCommands();

    // Register commands
    context.subscriptions.push(
        // Chat
        vscode.commands.registerCommand('promptly.chat', () => {
            chatView.show();
        }),

        // Git
        vscode.commands.registerCommand('promptly.gitStatus', () => {
            return gitCommands.status();
        }),

        vscode.commands.registerCommand('promptly.gitCommit', (message?: string) => {
            return gitCommands.commit(message);
        }),

        vscode.commands.registerCommand('promptly.gitPush', () => {
            return gitCommands.push();
        }),

        vscode.commands.registerCommand('promptly.gitLog', () => {
            return gitCommands.log();
        }),

        // Claude
        vscode.commands.registerCommand('promptly.review', () => {
            return claudeCommands.review();
        }),

        vscode.commands.registerCommand('promptly.explain', () => {
            return claudeCommands.explain();
        }),

        // HoloLoom
        vscode.commands.registerCommand('promptly.remember', (content?: string) => {
            if (!content) {
                vscode.window.showInputBox({
                    prompt: 'What should I remember?'
                }).then(input => {
                    if (input) hololoomCommands.remember(input);
                });
            } else {
                return hololoomCommands.remember(content);
            }
        }),

        vscode.commands.registerCommand('promptly.recall', (query?: string) => {
            if (!query) {
                vscode.window.showInputBox({
                    prompt: 'What should I recall?'
                }).then(input => {
                    if (input) hololoomCommands.recall(input);
                });
            } else {
                return hololoomCommands.recall(query);
            }
        })
    );

    // Status bar item
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = "$(comment-discussion) Promptly";
    statusBar.command = 'promptly.chat';
    statusBar.tooltip = "Open Promptly Chat";
    statusBar.show();
    context.subscriptions.push(statusBar);
}

export function deactivate() {}
```

## Usage

### Development Mode
```bash
cd promptly-vscode
npm install
npm run compile

# Press F5 in VS Code to launch extension development host
```

### Using Slash Commands

Once installed, open the Promptly chat panel:

1. **Via Command Palette**: `Ctrl+Shift+P` → "Promptly: Open Chat"
2. **Via Status Bar**: Click "Promptly" in bottom-right
3. **Via Keybinding**: `Ctrl+Shift+P` (configurable)

Then use slash commands:

```
/gs                              → Git status
/gc "Add awesome feature"        → Git commit
/gp                              → Git push

/review                          → Review current file
/explain                         → Explain current code
/refactor "use async/await"      → Refactor suggestions

/remember "We chose PostgreSQL"  → Save to HoloLoom
/recall "What database?"         → Query HoloLoom

/help                            → Show all commands
```

## Advanced: IntelliSense for Slash Commands

Add autocomplete for slash commands:

```typescript
// In extension.ts

const completionProvider = vscode.languages.registerCompletionItemProvider(
    { scheme: 'promptly' },
    {
        provideCompletionItems() {
            const commands = [
                { label: '/gs', detail: 'Git status' },
                { label: '/gc', detail: 'Git commit' },
                { label: '/gp', detail: 'Git push' },
                { label: '/review', detail: 'Review code' },
                { label: '/explain', detail: 'Explain code' },
                { label: '/remember', detail: 'Save to HoloLoom' },
                { label: '/recall', detail: 'Query HoloLoom' },
            ];

            return commands.map(cmd => {
                const item = new vscode.CompletionItem(cmd.label, vscode.CompletionItemKind.Function);
                item.detail = cmd.detail;
                return item;
            });
        }
    },
    '/' // Trigger on '/'
);

context.subscriptions.push(completionProvider);
```

## Next Steps

1. **Test locally**: Press F5 to launch extension
2. **Add keybindings**: Make `/gs`, `/review` even faster
3. **Publish**: `vsce publish` to VS Code marketplace
4. **Integrate with Matrix bot**: Extension can talk to your Matrix bot too!

This gives you **native VS Code integration** with the same slash command UX!
