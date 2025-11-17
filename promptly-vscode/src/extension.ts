import * as vscode from 'vscode';
import { PromptlyChatView } from './chatView';
import { GitCommands } from './commands/gitCommands';
import { ClaudeCommands } from './commands/claudeCommands';
import { HoloLoomCommands } from './commands/hololoomCommands';
import { HoloLoomSidebarProvider } from './views/sidebarProvider';
import { GraphViewProvider } from './views/graphViewProvider';
import { HoloLoomCodeLensProvider, registerCodeLensCommands } from './providers/codeLensProvider';
import { WorkspaceWatcher, registerWorkspaceCommands } from './watchers/workspaceWatcher';
import { HoloLoomLSPClient } from './lsp/client';

// Global LSP client instance
let lspClient: HoloLoomLSPClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('Promptly extension activated');

    // Initialize LSP client
    lspClient = new HoloLoomLSPClient(context);

    // Start LSP client asynchronously (non-blocking)
    lspClient.start().then(success => {
        if (success) {
            console.log('HoloLoom LSP client started successfully');
        } else {
            console.log('HoloLoom LSP client failed to start - using HTTP API fallback');
        }
    }).catch(error => {
        console.error('Error starting LSP client:', error);
    });

    const chatView = new PromptlyChatView(context);
    const gitCommands = new GitCommands();
    const claudeCommands = new ClaudeCommands();
    const hololoomCommands = new HoloLoomCommands(lspClient);

    // Register HoloLoom sidebar
    const sidebarProvider = new HoloLoomSidebarProvider(context.extensionUri, lspClient);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            HoloLoomSidebarProvider.viewType,
            sidebarProvider
        )
    );

    // Register knowledge graph viewer
    const graphViewProvider = new GraphViewProvider(context);

    // Register CodeLens provider for inline suggestions
    const codeLensProvider = new HoloLoomCodeLensProvider(lspClient);
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider(
            { scheme: 'file' }, // All file types
            codeLensProvider
        )
    );

    // Register CodeLens commands
    registerCodeLensCommands(context, lspClient);

    // Start workspace watcher (auto-indexing)
    const workspaceWatcher = new WorkspaceWatcher();
    workspaceWatcher.start();
    context.subscriptions.push({
        dispose: () => workspaceWatcher.stop()
    });

    // Register workspace commands
    registerWorkspaceCommands(context, workspaceWatcher);

    // Register commands
    context.subscriptions.push(
        // Chat
        vscode.commands.registerCommand('promptly.chat', () => {
            chatView.show();
        }),

        // Git
        vscode.commands.registerCommand('promptly.gitStatus', async () => {
            const result = await gitCommands.status();
            chatView.sendBotMessage(result);
        }),

        vscode.commands.registerCommand('promptly.gitCommit', async (message?: string) => {
            const result = await gitCommands.commit(message);
            chatView.sendBotMessage(result);
        }),

        vscode.commands.registerCommand('promptly.gitPush', async () => {
            const result = await gitCommands.push();
            chatView.sendBotMessage(result);
        }),

        vscode.commands.registerCommand('promptly.gitLog', async () => {
            const result = await gitCommands.log();
            chatView.sendBotMessage(result);
        }),

        // Claude
        vscode.commands.registerCommand('promptly.review', async () => {
            const result = await claudeCommands.review();
            if (result) chatView.sendBotMessage(result);
        }),

        vscode.commands.registerCommand('promptly.explain', async () => {
            const result = await claudeCommands.explain();
            if (result) chatView.sendBotMessage(result);
        }),

        // HoloLoom
        vscode.commands.registerCommand('promptly.remember', async (content?: string) => {
            if (!content) {
                content = await vscode.window.showInputBox({
                    prompt: 'What should I remember?',
                    placeHolder: 'e.g., We decided to use PostgreSQL for auth'
                });
            }
            if (content) {
                const result = await hololoomCommands.remember(content);
                chatView.sendBotMessage(result);
            }
        }),

        vscode.commands.registerCommand('promptly.recall', async (query?: string) => {
            if (!query) {
                query = await vscode.window.showInputBox({
                    prompt: 'What should I recall?',
                    placeHolder: 'e.g., What database did we choose?'
                });
            }
            if (query) {
                const result = await hololoomCommands.recall(query);
                chatView.sendBotMessage(result);
            }
        }),

        // Knowledge Graph
        vscode.commands.registerCommand('promptly.showGraph', async () => {
            await graphViewProvider.show();
        }),

        // LSP Client Management
        vscode.commands.registerCommand('promptly.restartLSP', async () => {
            if (lspClient) {
                vscode.window.showInformationMessage('Restarting HoloLoom LSP server...');
                const success = await lspClient.restart();
                if (success) {
                    vscode.window.showInformationMessage('HoloLoom LSP server restarted successfully');
                } else {
                    vscode.window.showErrorMessage('Failed to restart HoloLoom LSP server');
                }
            }
        }),

        vscode.commands.registerCommand('promptly.lspStatus', async () => {
            if (lspClient && lspClient.isRunning()) {
                vscode.window.showInformationMessage('HoloLoom LSP: Connected');
            } else {
                vscode.window.showWarningMessage('HoloLoom LSP: Not connected (using HTTP API fallback)');
            }
        })
    );

    // Status bar item
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = "$(comment-discussion) Promptly";
    statusBar.command = 'promptly.chat';
    statusBar.tooltip = "Open Promptly Chat (Ctrl+Alt+P)";
    statusBar.show();
    context.subscriptions.push(statusBar);

    // Show welcome message on first activation
    const hasShownWelcome = context.globalState.get('hasShownWelcome');
    if (!hasShownWelcome) {
        vscode.window.showInformationMessage(
            'Welcome to Promptly! Press Ctrl+Alt+P to open chat and type /help',
            'Open Chat'
        ).then(selection => {
            if (selection === 'Open Chat') {
                chatView.show();
            }
        });
        context.globalState.update('hasShownWelcome', true);
    }
}

export async function deactivate() {
    // Gracefully stop LSP client
    if (lspClient) {
        console.log('Deactivating HoloLoom LSP client...');
        await lspClient.stop();
        lspClient = undefined;
    }
}

// Export LSP client for use in other modules
export function getLSPClient(): HoloLoomLSPClient | undefined {
    return lspClient;
}