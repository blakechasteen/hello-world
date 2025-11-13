import * as vscode from 'vscode';

interface SlashCommand {
    command: string;
    aliases: string[];
    description: string;
    usage: string;
    category: string;
}

export class PromptlyChatView {
    private panel: vscode.WebviewPanel | undefined;
    private context: vscode.ExtensionContext;

    // Slash command definitions for autocomplete
    private commands: SlashCommand[] = [
        // Git commands
        { command: '/gs', aliases: ['/git-status'], description: 'Show git status', usage: '/gs', category: 'Git' },
        { command: '/gl', aliases: ['/git-log'], description: 'Show git log', usage: '/gl', category: 'Git' },
        { command: '/gc', aliases: ['/git-commit'], description: 'Create git commit', usage: '/gc "message"', category: 'Git' },
        { command: '/gp', aliases: ['/git-push'], description: 'Push to remote', usage: '/gp', category: 'Git' },
        { command: '/gd', aliases: ['/git-diff'], description: 'Show git diff', usage: '/gd', category: 'Git' },

        // Claude commands
        { command: '/review', aliases: ['/r'], description: 'Review current file with Claude', usage: '/review', category: 'Claude' },
        { command: '/explain', aliases: ['/e'], description: 'Explain current code', usage: '/explain', category: 'Claude' },
        { command: '/refactor', aliases: [], description: 'Refactor current file', usage: '/refactor "task"', category: 'Claude' },

        // HoloLoom commands
        { command: '/remember', aliases: ['/save'], description: 'Save to HoloLoom memory', usage: '/remember "note"', category: 'Memory' },
        { command: '/recall', aliases: ['/search'], description: 'Query HoloLoom memory', usage: '/recall "query"', category: 'Memory' },
        { command: '/context', aliases: [], description: 'Show current context', usage: '/context', category: 'Memory' },

        // Utility
        { command: '/help', aliases: ['/h', '/?'], description: 'Show this help', usage: '/help', category: 'Utility' },
        { command: '/clear', aliases: ['/cls'], description: 'Clear chat history', usage: '/clear', category: 'Utility' },
    ];

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }

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
                retainContextWhenHidden: true,
                localResourceRoots: []
            }
        );

        this.panel.webview.html = this.getHtmlContent();

        // Handle messages from webview
        this.panel.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.type) {
                    case 'command':
                        await this.handleChatMessage(message.text);
                        break;
                    case 'autocomplete':
                        this.sendAutocomplete(message.text);
                        break;
                }
            }
        );

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });

        // Send welcome message
        setTimeout(() => {
            this.sendBotMessage('👋 Hi! I\'m Promptly. Type / to see available commands or just chat naturally.\n\nTry:\n• `/gs` - git status\n• `/review` - review current file\n• `/remember "note"` - save to memory');
        }, 100);
    }

    private async handleChatMessage(text: string) {
        const command = text.trim();

        // Parse slash commands
        if (command.startsWith('/')) {
            await this.executeSlashCommand(command);
        } else {
            // Natural language - route to HoloLoom
            await this.handleNaturalLanguage(command);
        }
    }

    private async executeSlashCommand(command: string) {
        const parts = command.split(' ');
        const cmd = parts[0].toLowerCase();
        const args = parts.slice(1).join(' ');

        switch (cmd) {
            case '/gs':
            case '/git-status':
                await vscode.commands.executeCommand('promptly.gitStatus');
                break;

            case '/gl':
            case '/git-log':
                await vscode.commands.executeCommand('promptly.gitLog');
                break;

            case '/gc':
            case '/git-commit':
                // Extract message from quotes or use all args
                const commitMsg = args.match(/"([^"]+)"/)?.[ 1] || args;
                if (!commitMsg) {
                    this.sendBotMessage('❌ Usage: /gc "your commit message"');
                } else {
                    await vscode.commands.executeCommand('promptly.gitCommit', commitMsg);
                }
                break;

            case '/gp':
            case '/git-push':
                await vscode.commands.executeCommand('promptly.gitPush');
                break;

            case '/review':
            case '/r':
                await vscode.commands.executeCommand('promptly.review');
                break;

            case '/explain':
            case '/e':
                await vscode.commands.executeCommand('promptly.explain');
                break;

            case '/remember':
            case '/save':
                const rememberContent = args.match(/"([^"]+)"/)?.[ 1] || args;
                if (!rememberContent) {
                    this.sendBotMessage('❌ Usage: /remember "what to remember"');
                } else {
                    await vscode.commands.executeCommand('promptly.remember', rememberContent);
                }
                break;

            case '/recall':
            case '/search':
                const recallQuery = args.match(/"([^"]+)"/)?.[ 1] || args;
                if (!recallQuery) {
                    this.sendBotMessage('❌ Usage: /recall "what to find"');
                } else {
                    await vscode.commands.executeCommand('promptly.recall', recallQuery);
                }
                break;

            case '/help':
            case '/h':
            case '/?':
                this.showHelp();
                break;

            case '/clear':
            case '/cls':
                this.panel?.webview.postMessage({ type: 'clear' });
                this.sendBotMessage('Chat cleared.');
                break;

            default:
                this.sendBotMessage(`❌ Unknown command: ${cmd}\n\nType /help to see available commands.`);
        }
    }

    private async handleNaturalLanguage(query: string) {
        this.sendBotMessage('🤔 Thinking...');

        try {
            const config = vscode.workspace.getConfiguration('promptly');
            const hololoomUrl = config.get<string>('hololoomUrl');

            const axios = require('axios');
            const response = await axios.post(`${hololoomUrl}/query`, {
                text: query,
                mode: 'verify',
                max_steps: 3
            });

            const result = response.data;

            let message = result.response || result.answer || 'No response';

            if (result.confidence) {
                message += `\n\n_Confidence: ${(result.confidence * 100).toFixed(0)}%_`;
            }

            this.sendBotMessage(message);

        } catch (error: any) {
            if (error.code === 'ECONNREFUSED') {
                this.sendBotMessage('❌ HoloLoom server not running at http://localhost:8000\n\nStart it with:\n```\ncd HoloLoom/server\npython agentic_api.py\n```');
            } else {
                this.sendBotMessage(`❌ Error: ${error.message}`);
            }
        }
    }

    private sendAutocomplete(text: string) {
        const lastWord = text.split(' ').pop() || '';

        if (!lastWord.startsWith('/')) {
            return;
        }

        const matches = this.commands.filter(cmd =>
            cmd.command.startsWith(lastWord) ||
            cmd.aliases.some(alias => alias.startsWith(lastWord))
        );

        this.panel?.webview.postMessage({
            type: 'autocomplete',
            suggestions: matches.map(cmd => ({
                command: cmd.command,
                description: cmd.description,
                usage: cmd.usage,
                category: cmd.category
            }))
        });
    }

    public sendBotMessage(text: string, metadata?: any) {
        this.panel?.webview.postMessage({
            type: 'response',
            text,
            metadata
        });
    }

    private showHelp() {
        const helpByCategory: { [key: string]: SlashCommand[] } = {};

        this.commands.forEach(cmd => {
            if (!helpByCategory[cmd.category]) {
                helpByCategory[cmd.category] = [];
            }
            helpByCategory[cmd.category].push(cmd);
        });

        let helpText = '**Promptly Slash Commands**\n\n';

        for (const [category, cmds] of Object.entries(helpByCategory)) {
            helpText += `**${category}:**\n`;
            cmds.forEach(cmd => {
                const aliases = cmd.aliases.length > 0 ? ` (${cmd.aliases.join(', ')})` : '';
                helpText += `• \`${cmd.usage}\`${aliases} - ${cmd.description}\n`;
            });
            helpText += '\n';
        }

        helpText += '**Natural Language:**\nJust type normally and Promptly will understand!\n\n';
        helpText += '_Pro tip: Start typing / to see autocomplete suggestions_';

        this.sendBotMessage(helpText);
    }

    private getHtmlContent(): string {
        const config = vscode.workspace.getConfiguration('promptly');
        const enableAutocomplete = config.get<boolean>('enableAutocomplete', true);

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
            background: var(--vscode-editor-background);
        }
        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 6px;
            max-width: 80%;
            animation: fadeIn 0.2s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user {
            background: var(--vscode-input-background);
            margin-left: auto;
            text-align: right;
        }
        .bot {
            background: var(--vscode-editor-inlineValuesBackground);
            border-left: 3px solid var(--vscode-focusBorder);
        }
        .bot code {
            background: var(--vscode-textCodeBlock-background);
            padding: 2px 4px;
            border-radius: 3px;
            font-family: var(--vscode-editor-font-family);
        }
        .metadata {
            font-size: 0.85em;
            color: var(--vscode-descriptionForeground);
            margin-top: 5px;
            font-style: italic;
        }
        #input-container {
            display: flex;
            padding: 10px;
            border-top: 1px solid var(--vscode-panel-border);
            position: relative;
        }
        #input {
            flex: 1;
            padding: 10px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border-radius: 4px;
            font-family: var(--vscode-font-family);
            font-size: 14px;
        }
        #input:focus {
            outline: 1px solid var(--vscode-focusBorder);
        }
        #send {
            margin-left: 8px;
            padding: 10px 20px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
        }
        #send:hover {
            background: var(--vscode-button-hoverBackground);
        }
        #autocomplete {
            position: absolute;
            bottom: 60px;
            left: 10px;
            right: 10px;
            background: var(--vscode-editorSuggestWidget-background);
            border: 1px solid var(--vscode-editorSuggestWidget-border);
            border-radius: 4px;
            max-height: 300px;
            overflow-y: auto;
            display: none;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .autocomplete-item {
            padding: 8px 12px;
            cursor: pointer;
            border-bottom: 1px solid var(--vscode-editorSuggestWidget-border);
        }
        .autocomplete-item:hover {
            background: var(--vscode-list-hoverBackground);
        }
        .autocomplete-item:last-child {
            border-bottom: none;
        }
        .autocomplete-command {
            font-weight: 600;
            color: var(--vscode-editorSuggestWidget-foreground);
            font-family: var(--vscode-editor-font-family);
        }
        .autocomplete-description {
            font-size: 0.9em;
            color: var(--vscode-descriptionForeground);
            margin-top: 2px;
        }
        .autocomplete-usage {
            font-size: 0.85em;
            color: var(--vscode-textLink-foreground);
            font-family: var(--vscode-editor-font-family);
            margin-top: 2px;
        }
        .category-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: 600;
            margin-right: 6px;
        }
        .category-git { background: #f0ad4e; color: #000; }
        .category-claude { background: #5cb85c; color: #fff; }
        .category-memory { background: #5bc0de; color: #000; }
        .category-utility { background: #777; color: #fff; }
    </style>
</head>
<body>
    <div id="messages"></div>
    <div id="input-container">
        <div id="autocomplete"></div>
        <input type="text" id="input" placeholder="Type / for commands or just chat naturally..." autocomplete="off" />
        <button id="send">Send</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const autocompleteDiv = document.getElementById('autocomplete');

        const enableAutocomplete = ${enableAutocomplete};
        let selectedSuggestionIndex = -1;
        let currentSuggestions = [];

        function addMessage(text, sender = 'bot', metadata) {
            const div = document.createElement('div');
            div.className = \`message \${sender}\`;

            // Simple markdown rendering
            text = text.replace(/\`\`\`([^\`]+)\`\`\`/g, '<pre><code>$1</code></pre>');
            text = text.replace(/\`([^\`]+)\`/g, '<code>$1</code>');
            text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            text = text.replace(/_([^_]+)_/g, '<em>$1</em>');
            text = text.replace(/\n/g, '<br>');

            div.innerHTML = text;

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
            hideAutocomplete();
        }

        function showAutocomplete(suggestions) {
            if (!enableAutocomplete || suggestions.length === 0) {
                hideAutocomplete();
                return;
            }

            currentSuggestions = suggestions;
            selectedSuggestionIndex = -1;

            autocompleteDiv.innerHTML = suggestions.map((sug, index) => {
                const categoryClass = \`category-\${sug.category.toLowerCase()}\`;
                return \`
                    <div class="autocomplete-item" data-index="\${index}">
                        <span class="category-badge \${categoryClass}">\${sug.category}</span>
                        <span class="autocomplete-command">\${sug.command}</span>
                        <div class="autocomplete-description">\${sug.description}</div>
                        <div class="autocomplete-usage">\${sug.usage}</div>
                    </div>
                \`;
            }).join('');

            autocompleteDiv.style.display = 'block';

            // Add click handlers
            autocompleteDiv.querySelectorAll('.autocomplete-item').forEach(item => {
                item.addEventListener('click', () => {
                    const index = parseInt(item.dataset.index);
                    selectSuggestion(index);
                });
            });
        }

        function hideAutocomplete() {
            autocompleteDiv.style.display = 'none';
            currentSuggestions = [];
            selectedSuggestionIndex = -1;
        }

        function selectSuggestion(index) {
            if (index >= 0 && index < currentSuggestions.length) {
                const suggestion = currentSuggestions[index];
                input.value = suggestion.usage;
                hideAutocomplete();
                input.focus();
            }
        }

        function highlightSuggestion(index) {
            const items = autocompleteDiv.querySelectorAll('.autocomplete-item');
            items.forEach((item, i) => {
                if (i === index) {
                    item.style.background = 'var(--vscode-list-activeSelectionBackground)';
                } else {
                    item.style.background = '';
                }
            });
        }

        sendBtn.addEventListener('click', send);

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            } else if (e.key === 'ArrowDown' && currentSuggestions.length > 0) {
                e.preventDefault();
                selectedSuggestionIndex = Math.min(selectedSuggestionIndex + 1, currentSuggestions.length - 1);
                highlightSuggestion(selectedSuggestionIndex);
            } else if (e.key === 'ArrowUp' && currentSuggestions.length > 0) {
                e.preventDefault();
                selectedSuggestionIndex = Math.max(selectedSuggestionIndex - 1, -1);
                highlightSuggestion(selectedSuggestionIndex);
            } else if (e.key === 'Tab' && currentSuggestions.length > 0) {
                e.preventDefault();
                selectSuggestion(selectedSuggestionIndex >= 0 ? selectedSuggestionIndex : 0);
            } else if (e.key === 'Escape') {
                hideAutocomplete();
            }
        });

        input.addEventListener('input', () => {
            const text = input.value;

            if (enableAutocomplete && text.includes('/')) {
                vscode.postMessage({ type: 'autocomplete', text });
            } else {
                hideAutocomplete();
            }
        });

        // Handle responses
        window.addEventListener('message', (event) => {
            const message = event.data;

            if (message.type === 'response') {
                addMessage(message.text, 'bot', message.metadata);
            } else if (message.type === 'autocomplete') {
                showAutocomplete(message.suggestions);
            } else if (message.type === 'clear') {
                messagesDiv.innerHTML = '';
            }
        });

        // Focus input on load
        input.focus();
    </script>
</body>
</html>
        `;
    }
}