/**
 * Squad VS Code Extension
 * Agentic coding assistant powered by HoloLoom
 */

import * as vscode from 'vscode';
import { HoloLoomBridge } from './HoloLoomBridge';
import { AgentPanel } from './AgentPanel';
import { CodeContextProvider } from './CodeContextProvider';
import { MCPServer } from './MCPServer';
import { AgentMonitorTreeProvider } from './AgentMonitorTreeProvider';
import { SkillsManager, createSkillsManager } from './SkillsManager';
import { SkillsTreeProvider } from './SkillsTreeProvider';
import { SkillsExplorerPanel } from './SkillsExplorerPanel';

let bridge: HoloLoomBridge;
let agentPanel: AgentPanel | undefined;
let contextProvider: CodeContextProvider;
let statusBarItem: vscode.StatusBarItem;
let mcpServer: MCPServer;
let agentMonitorProvider: AgentMonitorTreeProvider;
let skillsManager: SkillsManager;
let skillsTreeProvider: SkillsTreeProvider;

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

    // Initialize Agent Monitor Tree View
    agentMonitorProvider = new AgentMonitorTreeProvider(serverUrl);
    const treeView = vscode.window.createTreeView('hololoomAgents', {
        treeDataProvider: agentMonitorProvider,
        showCollapseAll: true
    });
    context.subscriptions.push(treeView);
    context.subscriptions.push(agentMonitorProvider);

    // Initialize Skills Manager and Tree View
    skillsManager = createSkillsManager(context);
    skillsTreeProvider = new SkillsTreeProvider(skillsManager);
    const skillsTreeView = vscode.window.createTreeView('hololoomSkills', {
        treeDataProvider: skillsTreeProvider,
        showCollapseAll: true
    });
    context.subscriptions.push(skillsTreeView);
    context.subscriptions.push(skillsTreeProvider);
    context.subscriptions.push(skillsManager);

    // Register commands
    registerCommands(context);
    registerSkillsCommands(context);

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

    // Agent Monitor Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('hololoom.refreshAgentMonitor', async () => {
            await agentMonitorProvider.forceRefresh();
            vscode.window.showInformationMessage('Agent monitor refreshed');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('hololoom.showAgentDetails', async (session: any) => {
            const panel = vscode.window.createWebviewPanel(
                'agentDetails',
                `Agent: ${session.agent_id}`,
                vscode.ViewColumn.One,
                {
                    enableScripts: true
                }
            );

            panel.webview.html = getAgentDetailsHtml(session);
        })
    );
}

function registerSkillsCommands(context: vscode.ExtensionContext) {
    // Open Skills Explorer
    context.subscriptions.push(
        vscode.commands.registerCommand('skills.openExplorer', () => {
            SkillsExplorerPanel.show(skillsManager, context.extensionUri);
        })
    );

    // Refresh Skills
    context.subscriptions.push(
        vscode.commands.registerCommand('skills.refresh', () => {
            skillsTreeProvider.refresh();
        })
    );

    // Invoke Skill
    context.subscriptions.push(
        vscode.commands.registerCommand('skills.invoke', async (skillId?: string) => {
            // If no skillId provided, show quick pick
            if (!skillId) {
                const skill = await skillsManager.selectSkill({ installed: true });
                if (!skill) return;
                skillId = skill.skill_id;
            }

            // Get input for the skill
            const input = await vscode.window.showInputBox({
                prompt: `Enter input for skill "${skillId}"`,
                placeHolder: 'Skill input...'
            });

            if (!input) return;

            const result = await skillsManager.invokeSkill(skillId, { query: input });

            if (result.success) {
                vscode.window.showInformationMessage(
                    `Skill completed in ${result.duration_ms}ms`
                );
                // Could show output in a panel here
            } else {
                vscode.window.showErrorMessage(`Skill error: ${result.error}`);
            }
        })
    );

    // Install Skill
    context.subscriptions.push(
        vscode.commands.registerCommand('skills.install', async (skillId?: string) => {
            if (!skillId) {
                const skill = await skillsManager.selectSkill({ installed: false });
                if (!skill) return;
                skillId = skill.skill_id;
            }
            await skillsManager.installSkill(skillId);
        })
    );

    // Remove Skill
    context.subscriptions.push(
        vscode.commands.registerCommand('skills.remove', async (item?: any) => {
            const skillId = item?.data?.skill?.skill_id;
            if (skillId) {
                await skillsManager.removeSkill(skillId);
            }
        })
    );

    // Enable Skill
    context.subscriptions.push(
        vscode.commands.registerCommand('skills.enable', async (item?: any) => {
            const skillId = item?.data?.skill?.skill_id;
            if (skillId) {
                await skillsManager.setSkillEnabled(skillId, true);
                skillsTreeProvider.refresh();
            }
        })
    );

    // Disable Skill
    context.subscriptions.push(
        vscode.commands.registerCommand('skills.disable', async (item?: any) => {
            const skillId = item?.data?.skill?.skill_id;
            if (skillId) {
                await skillsManager.setSkillEnabled(skillId, false);
                skillsTreeProvider.refresh();
            }
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

function getAgentDetailsHtml(session: any): string {
    const tree = session.tree || {};
    const duration = session.total_duration_ms
        ? `${session.total_duration_ms.toFixed(0)}ms`
        : 'In progress';

    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Agent Details</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-foreground);
                    background: var(--vscode-editor-background);
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 {
                    font-size: 24px;
                    margin-bottom: 10px;
                    border-bottom: 2px solid var(--vscode-panel-border);
                    padding-bottom: 10px;
                }
                .metadata {
                    display: grid;
                    grid-template-columns: 150px 1fr;
                    gap: 10px;
                    margin: 20px 0;
                    padding: 15px;
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-radius: 5px;
                }
                .metadata-label {
                    font-weight: bold;
                    color: var(--vscode-textLink-foreground);
                }
                .status {
                    display: inline-block;
                    padding: 3px 10px;
                    border-radius: 3px;
                    font-size: 12px;
                    font-weight: bold;
                }
                .status-completed { background: #4CAF50; color: white; }
                .status-running { background: #2196F3; color: white; }
                .status-failed { background: #f44336; color: white; }
                .tree {
                    margin: 20px 0;
                }
                .tree-node {
                    margin: 10px 0 10px 20px;
                    padding: 10px;
                    border-left: 3px solid var(--vscode-textLink-foreground);
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-radius: 3px;
                }
                .confidence {
                    display: inline-block;
                    padding: 2px 8px;
                    background: var(--vscode-badge-background);
                    color: var(--vscode-badge-foreground);
                    border-radius: 3px;
                    font-size: 11px;
                    margin-left: 10px;
                }
                .files {
                    margin: 10px 0;
                    padding: 10px;
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border-radius: 5px;
                }
                .file {
                    padding: 5px;
                    font-family: monospace;
                    font-size: 12px;
                }
            </style>
        </head>
        <body>
            <h1>Agent Reasoning Details</h1>

            <div class="metadata">
                <div class="metadata-label">Agent ID:</div>
                <div><code>${session.agent_id}</code></div>

                <div class="metadata-label">Project:</div>
                <div>${session.project}</div>

                <div class="metadata-label">Query:</div>
                <div>${session.query}</div>

                <div class="metadata-label">Mode:</div>
                <div>${session.mode}</div>

                <div class="metadata-label">Status:</div>
                <div><span class="status status-${session.status}">${session.status.toUpperCase()}</span></div>

                <div class="metadata-label">Duration:</div>
                <div>${duration}</div>

                <div class="metadata-label">Steps:</div>
                <div>${session.current_step} / ${session.total_steps}</div>
            </div>

            ${session.files && session.files.length > 0 ? `
                <h2>Files</h2>
                <div class="files">
                    ${session.files.map((f: string) => `<div class="file">📄 ${f}</div>`).join('')}
                </div>
            ` : ''}

            ${tree.node_id ? `
                <h2>Reasoning Tree</h2>
                <div class="tree">
                    ${renderTreeNode(tree)}
                </div>
            ` : '<p><em>Reasoning tree not yet available</em></p>'}
        </body>
        </html>
    `;
}

function renderTreeNode(node: any, depth: number = 0): string {
    if (!node || !node.node_id) return '';

    const confidence = node.confidence
        ? `<span class="confidence">${(node.confidence * 100).toFixed(0)}% conf</span>`
        : '';

    const epistemic = node.epistemic_confidence
        ? `<span class="confidence">${(node.epistemic_confidence * 100).toFixed(0)}% epistemic</span>`
        : '';

    let html = `
        <div class="tree-node">
            <strong>${node.step_type}</strong> ${confidence} ${epistemic}
            ${node.query ? `<br><em>${node.query}</em>` : ''}
            ${node.finding ? `<br>${node.finding.substring(0, 200)}${node.finding.length > 200 ? '...' : ''}` : ''}
        </div>
    `;

    if (node.children && node.children.length > 0) {
        for (const child of node.children) {
            html += renderTreeNode(child, depth + 1);
        }
    }

    return html;
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
