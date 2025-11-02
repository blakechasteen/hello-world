import * as vscode from 'vscode';
import { PromptTreeProvider } from './providers/promptProvider';
import { SkillTreeProvider } from './providers/skillProvider';
import { AnalyticsTreeProvider } from './providers/analyticsProvider';
import { AnalyticsDashboardPanel } from './webviews/analyticsPanel';
import { PromptlyClient } from './api/promptlyClient';
import * as commands from './commands';

let promptTreeProvider: PromptTreeProvider;
let skillTreeProvider: SkillTreeProvider;
let analyticsTreeProvider: AnalyticsTreeProvider;
let promptlyClient: PromptlyClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('Promptly extension is now active!');

    // Initialize Promptly client
    const config = vscode.workspace.getConfiguration('promptly');
    const mcpServerPath = config.get<string>('mcpServerPath', '');
    promptlyClient = new PromptlyClient(mcpServerPath);

    // Initialize tree providers
    promptTreeProvider = new PromptTreeProvider(promptlyClient);
    skillTreeProvider = new SkillTreeProvider(promptlyClient);
    analyticsTreeProvider = new AnalyticsTreeProvider(promptlyClient);

    // Register tree views
    vscode.window.registerTreeDataProvider('promptlyPrompts', promptTreeProvider);
    vscode.window.registerTreeDataProvider('promptlySkills', skillTreeProvider);
    vscode.window.registerTreeDataProvider('promptlyAnalytics', analyticsTreeProvider);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('promptly.refreshPrompts', () => {
            promptTreeProvider.refresh();
            skillTreeProvider.refresh();
            analyticsTreeProvider.refresh();
        }),

        vscode.commands.registerCommand('promptly.addPrompt', () =>
            commands.addPrompt(promptlyClient, promptTreeProvider)
        ),

        vscode.commands.registerCommand('promptly.executePrompt', (item) =>
            commands.executePrompt(promptlyClient, item)
        ),

        vscode.commands.registerCommand('promptly.createSkill', () =>
            commands.createSkill(promptlyClient, skillTreeProvider)
        ),

        vscode.commands.registerCommand('promptly.installTemplate', () =>
            commands.installTemplate(promptlyClient, skillTreeProvider)
        ),

        vscode.commands.registerCommand('promptly.showAnalytics', () => {
            AnalyticsDashboardPanel.createOrShow(context.extensionUri, promptlyClient);
        }),

        vscode.commands.registerCommand('promptly.composeLoops', () =>
            commands.composeLoops(promptlyClient)
        ),

        vscode.commands.registerCommand('promptly.exportCharts', () =>
            commands.exportCharts(promptlyClient)
        )
    );

    // Show analytics on startup if configured
    if (config.get<boolean>('showAnalyticsOnStartup', false)) {
        AnalyticsDashboardPanel.createOrShow(context.extensionUri, promptlyClient);
    }

    // Status bar
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = "$(beaker) Promptly";
    statusBar.tooltip = "Click to show analytics";
    statusBar.command = 'promptly.showAnalytics';
    statusBar.show();
    context.subscriptions.push(statusBar);

    // Auto-refresh if enabled
    if (config.get<boolean>('autoRefresh', true)) {
        setInterval(() => {
            promptTreeProvider.refresh();
            skillTreeProvider.refresh();
            analyticsTreeProvider.refresh();
        }, 30000); // Every 30 seconds
    }
}

export function deactivate() {
    console.log('Promptly extension deactivated');
}
