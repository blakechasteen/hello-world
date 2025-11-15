/**
 * Squad VS Code Extension
 * Agentic coding assistant powered by HoloLoom
 */

import * as vscode from 'vscode';
import { HoloLoomBridge } from './HoloLoomBridge';
import { AgentPanel } from './AgentPanel';
import { CodeContextProvider } from './CodeContextProvider';

let bridge: HoloLoomBridge;
let agentPanel: AgentPanel | undefined;
let contextProvider: CodeContextProvider;

export async function activate(context: vscode.ExtensionContext) {
    console.log('Squad extension activating...');

    // Initialize components
    contextProvider = new CodeContextProvider();

    const config = vscode.workspace.getConfiguration('squad');
    const serverUrl = config.get<string>('serverUrl', 'http://localhost:8000');

    bridge = new HoloLoomBridge(serverUrl);

    // Check server health
    const healthy = await bridge.healthCheck();
    if (!healthy) {
        vscode.window.showWarningMessage(
            'Squad server not responding. Make sure to start the server with: python squad/server.py'
        );
    } else {
        vscode.window.showInformationMessage('Squad is ready! 🤖');
    }

    // Register commands
    registerCommands(context);

    // Status bar
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = '$(robot) Squad';
    statusBarItem.command = 'squad.openPanel';
    statusBarItem.tooltip = 'Squad - Agentic AI Assistant';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    console.log('Squad extension activated!');
}

function registerCommands(context: vscode.ExtensionContext) {
    // Ask Question
    context.subscriptions.push(
        vscode.commands.registerCommand('squad.ask', async () => {
            const question = await vscode.window.showInputBox({
                prompt: 'What would you like to know?',
                placeHolder: 'Ask Squad anything about your code...'
            });

            if (!question) return;

            const codeContext = contextProvider.getCurrentContext();
            await executeQuery(question, codeContext, 'direct');
        })
    );

    // Explain Selection
    context.subscriptions.push(
        vscode.commands.registerCommand('squad.explainSelection', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const selectedCode = editor.document.getText(editor.selection);
            const language = editor.document.languageId;
            const question = `Explain this ${language} code:\n\n${selectedCode}`;

            const codeContext = contextProvider.getCurrentContext();
            await executeQuery(question, codeContext, 'verify');
        })
    );

    // Suggest Fix
    context.subscriptions.push(
        vscode.commands.registerCommand('squad.suggestFix', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
            if (diagnostics.length === 0) {
                vscode.window.showInformationMessage('No issues detected');
                return;
            }

            const selectedCode = editor.selection.isEmpty
                ? editor.document.getText()
                : editor.document.getText(editor.selection);

            const question = `Fix these issues:\n${
                diagnostics.map(d => `- ${d.message}`).join('\n')
            }\n\nCode:\n${selectedCode}`;

            const codeContext = contextProvider.getCurrentContext();
            codeContext.diagnostics = diagnostics;

            await executeQuery(question, codeContext, 'plan_execute');
        })
    );

    // Refactor Code
    context.subscriptions.push(
        vscode.commands.registerCommand('squad.refactor', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const refactorType = await vscode.window.showQuickPick([
                'Extract function',
                'Simplify logic',
                'Add error handling',
                'Optimize performance',
                'Add type annotations',
                'Custom...'
            ], {
                placeHolder: 'Select refactoring type'
            });

            if (!refactorType) return;

            let instruction = refactorType;
            if (refactorType === 'Custom...') {
                const custom = await vscode.window.showInputBox({
                    prompt: 'What refactoring would you like?'
                });
                if (!custom) return;
                instruction = custom;
            }

            const selectedCode = editor.document.getText(editor.selection);
            const language = editor.document.languageId;
            const question = `Refactor this ${language} code: ${instruction}\n\nCode:\n${selectedCode}`;

            const codeContext = contextProvider.getCurrentContext();
            await executeQuery(question, codeContext, 'plan_execute');
        })
    );

    // Generate Tests
    context.subscriptions.push(
        vscode.commands.registerCommand('squad.generateTests', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selectedCode = editor.selection.isEmpty
                ? editor.document.getText()
                : editor.document.getText(editor.selection);

            const language = editor.document.languageId;

            const testType = await vscode.window.showQuickPick([
                'Unit tests',
                'Integration tests',
                'Edge cases',
                'All of the above'
            ], {
                placeHolder: 'What type of tests?'
            });

            if (!testType) return;

            const question = `Generate ${testType.toLowerCase()} for this ${language} code:\n\n${selectedCode}`;
            const codeContext = contextProvider.getCurrentContext();

            await executeQuery(question, codeContext, 'plan_execute');
        })
    );

    // Open Panel
    context.subscriptions.push(
        vscode.commands.registerCommand('squad.openPanel', () => {
            if (!agentPanel) {
                agentPanel = new AgentPanel(context.extensionUri);
            }
            agentPanel.show();
        })
    );
}

async function executeQuery(
    question: string,
    codeContext: any,
    mode: string = 'verify'
) {
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Squad: Thinking...`,
        cancellable: false
    }, async () => {
        try {
            const config = vscode.workspace.getConfiguration('squad');
            const maxSteps = config.get<number>('maxSteps', 5);
            const showSteps = config.get<boolean>('showReasoningSteps', true);

            const result = await bridge.query(question, codeContext, mode, maxSteps);

            if (showSteps) {
                if (!agentPanel) {
                    agentPanel = new AgentPanel(vscode.Uri.file(__dirname));
                }
                agentPanel.show();
                agentPanel.displayResult(result);
            }

            // Show quick notification
            const confidenceText = `${(result.confidence * 100).toFixed(0)}%`;
            vscode.window.showInformationMessage(
                `✅ Squad: ${confidenceText} confidence (${result.reasoning_mode} mode, ${result.total_duration_ms.toFixed(0)}ms)`
            );

        } catch (error: any) {
            vscode.window.showErrorMessage(`Squad error: ${error.message}`);
        }
    });
}

export async function deactivate() {
    if (bridge) {
        await bridge.stop();
    }
    console.log('Squad extension deactivated');
}
