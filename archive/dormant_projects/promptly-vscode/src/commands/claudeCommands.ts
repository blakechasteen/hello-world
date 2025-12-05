import * as vscode from 'vscode';

export class ClaudeCommands {
    private apiKey: string | undefined;

    constructor() {
        const config = vscode.workspace.getConfiguration('promptly');
        this.apiKey = config.get<string>('claudeApiKey');
    }

    async review(): Promise<string> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return '❌ No active file to review';
        }

        const code = editor.document.getText();
        const language = editor.document.languageId;
        const fileName = editor.document.fileName.split(/[\\/]/).pop();

        if (!this.apiKey) {
            return '❌ Claude API key not set.\n\nSet it in VS Code settings: `promptly.claudeApiKey`\n\nOr use HoloLoom server: Just ask "review this code" naturally!';
        }

        try {
            const Anthropic = require('@anthropic-ai/sdk');
            const client = new Anthropic({ apiKey: this.apiKey });

            const message = await client.messages.create({
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

            // Open in new document
            const doc = await vscode.workspace.openTextDocument({
                content: review,
                language: 'markdown'
            });

            await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);

            return `✅ **Code review complete!**\n\nReview opened in new tab →`;

        } catch (error: any) {
            return `❌ Review failed: ${error.message}`;
        }
    }

    async explain(): Promise<string> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return '❌ No active file to explain';
        }

        const selection = editor.selection;
        const code = selection.isEmpty
            ? editor.document.getText()
            : editor.document.getText(selection);

        const language = editor.document.languageId;

        if (!this.apiKey) {
            return '❌ Claude API key not set.\n\nSet it in VS Code settings or ask naturally via HoloLoom!';
        }

        try {
            const Anthropic = require('@anthropic-ai/sdk');
            const client = new Anthropic({ apiKey: this.apiKey });

            const message = await client.messages.create({
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

            return `**Code Explanation:**\n\n${explanation}`;

        } catch (error: any) {
            return `❌ Explanation failed: ${error.message}`;
        }
    }
}