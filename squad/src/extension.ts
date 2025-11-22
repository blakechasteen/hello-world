/**
 * Squad VS Code Extension
 * Agentic coding assistant powered by HoloLoom
 */

import * as vscode from 'vscode';
import { HoloLoomBridge } from './HoloLoomBridge';
import { AgentPanel } from './AgentPanel';
import { CodeContextProvider } from './CodeContextProvider';
import { MCPServer } from './MCPServer';

let bridge: HoloLoomBridge;
let agentPanel: AgentPanel | undefined;
let contextProvider: CodeContextProvider;
let statusBarItem: vscode.StatusBarItem;
let mcpServer: MCPServer;

async function updateServerStatus() {
    const healthy = await bridge.healthCheck();
    if (healthy) {
        statusBarItem.text = '$(check) Squad';
        statusBarItem.tooltip = 'Squad - Connected and ready';
        statusBarItem.backgroundColor = undefined;
        statusBarItem.show();
    } else {
        statusBarItem.text = '$(warning) Squad';
        statusBarItem.tooltip = 'Squad - Server not responding. Click to start server.';
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        statusBarItem.show();
    }
}

export async function activate(context: vscode.ExtensionContext) {
    console.log('Squad extension activating...');

    // Initialize components
    contextProvider = new CodeContextProvider();

    const config = vscode.workspace.getConfiguration('squad');
    const serverUrl = config.get<string>('serverUrl', 'http://localhost:8000');

    bridge = new HoloLoomBridge(serverUrl);

    // Initialize MCP Server
    const mcpPort = config.get<number>('mcpPort', 9001);
    const enableMCP = config.get<boolean>('enableMCP', true);

    if (enableMCP) {
        try {
            mcpServer = new MCPServer(contextProvider, bridge, mcpPort);
            await mcpServer.start();
            console.log(`MCP Server started on ws://localhost:${mcpPort}`);
        } catch (error: any) {
            console.error(`Failed to start MCP Server: ${error.message}`);
            vscode.window.showWarningMessage(
                `Could not start Claude Code MCP Server on port ${mcpPort}. Matrix integration will be unavailable.`
            );
        }
    }

    // Status bar
    statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.command = 'squad.openPanel';
    context.subscriptions.push(statusBarItem);

    // Check server health and update status
    await updateServerStatus();

    // Register commands
    registerCommands(context);

    // Periodic health check (every 30 seconds)
    const healthCheckInterval = setInterval(updateServerStatus, 30000);
    context.subscriptions.push({
        dispose: () => clearInterval(healthCheckInterval)
    });

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
    }, async (progress) => {
        try {
            // Progress: Starting
            progress.report({ increment: 0, message: 'Connecting to server...' });

            const config = vscode.workspace.getConfiguration('squad');
            const maxSteps = config.get<number>('maxSteps', 5);
            const showSteps = config.get<boolean>('showReasoningSteps', true);

            // Progress: Query started
            progress.report({ increment: 20, message: `Processing query (${mode} mode)...` });

            const startTime = Date.now();
            const result = await bridge.query(question, codeContext, mode, maxSteps);
            const duration = Date.now() - startTime;

            // Progress: Query complete
            progress.report({ increment: 60, message: 'Formatting results...' });

            if (showSteps) {
                if (!agentPanel) {
                    agentPanel = new AgentPanel(vscode.Uri.file(__dirname));
                }
                agentPanel.show();
                agentPanel.displayResult(result);
            }

            // Progress: Done
            progress.report({ increment: 20, message: 'Complete!' });

            // Show result notification with confidence indicator
            const confidenceText = `${(result.confidence * 100).toFixed(0)}%`;
            const confidenceIcon = result.confidence >= 0.8 ? '✅' :
                                  result.confidence >= 0.5 ? '⚠️' : '❌';

            vscode.window.showInformationMessage(
                `${confidenceIcon} Squad: ${confidenceText} confidence (${result.reasoning_mode} mode, ${duration}ms)`
            );

        } catch (error: any) {
            // Enhanced error handling with specific messages
            let errorMessage = 'Squad error: ';

            if (error.code === 'ECONNREFUSED') {
                errorMessage += 'Cannot connect to server. Make sure the Squad server is running on port 8000.';
                const action = await vscode.window.showErrorMessage(
                    errorMessage,
                    'Open Terminal',
                    'Settings'
                );

                if (action === 'Open Terminal') {
                    const terminal = vscode.window.createTerminal('Squad Server');
                    terminal.show();
                    terminal.sendText('cd /home/user/hello-world/squad && PYTHONPATH=/home/user/hello-world python server.py');
                } else if (action === 'Settings') {
                    vscode.commands.executeCommand('workbench.action.openSettings', 'squad');
                }
            } else if (error.response?.status === 503) {
                errorMessage += 'Server is starting up. Please wait a moment and try again.';
                vscode.window.showWarningMessage(errorMessage);
            } else if (error.response?.status === 500) {
                errorMessage += `Server error: ${error.response?.data?.detail || 'Internal server error'}`;
                vscode.window.showErrorMessage(errorMessage);
            } else {
                errorMessage += error.message || 'Unknown error occurred';
                vscode.window.showErrorMessage(errorMessage);
            }
        }
    });
}

export async function deactivate() {
    // Stop MCP Server
    if (mcpServer) {
        await mcpServer.stop();
    }

    // Stop HoloLoom bridge
    if (bridge) {
        await bridge.stop();
    }

    console.log('Squad extension deactivated');
}
