import * as vscode from 'vscode';
import { PromptTreeProvider } from './promptLibrary/PromptTreeProvider';
import { PromptlyBridge } from './api/PromptlyBridge';

let bridge: PromptlyBridge;

export async function activate(context: vscode.ExtensionContext) {
    console.log('Promptly extension activating...');

    // Initialize Python bridge
    bridge = new PromptlyBridge();
    await bridge.start();

    // Create tree data provider
    const promptTreeProvider = new PromptTreeProvider(bridge);

    // Register tree view
    const treeView = vscode.window.createTreeView('promptLibrary', {
        treeDataProvider: promptTreeProvider,
        showCollapseAll: true
    });

    // Register refresh command
    const refreshCommand = vscode.commands.registerCommand('promptly.refresh', async () => {
        promptTreeProvider.refresh();
        vscode.window.showInformationMessage('Prompt library refreshed');
    });

    // Register view prompt command
    const viewPromptCommand = vscode.commands.registerCommand('promptly.viewPrompt', async (item) => {
        if (item && item.promptName) {
            const promptData = await bridge.getPrompt(item.promptName);
            if (promptData) {
                const doc = await vscode.workspace.openTextDocument({
                    content: promptData.content,
                    language: 'markdown'
                });
                await vscode.window.showTextDocument(doc, { preview: true });
            }
        }
    });

    context.subscriptions.push(treeView, refreshCommand, viewPromptCommand);

    vscode.window.showInformationMessage('Promptly is ready!');
    console.log('Promptly extension activated');
}

export async function deactivate() {
    if (bridge) {
        await bridge.stop();
    }
}