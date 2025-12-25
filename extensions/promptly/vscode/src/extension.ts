import * as vscode from 'vscode';
import { PromptlyBridge } from './PromptlyBridge';
import { PromptTreeProvider, PromptItem } from './providers/PromptTreeProvider';
import { ChainTreeProvider } from './providers/ChainTreeProvider';
import { SkillTreeProvider } from './providers/SkillTreeProvider';

let bridge: PromptlyBridge;
let promptProvider: PromptTreeProvider;
let chainProvider: ChainTreeProvider;
let skillProvider: SkillTreeProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('Promptly extension activated');

    // Initialize bridge to Python CLI
    const config = vscode.workspace.getConfiguration('promptly');
    const pythonPath = config.get<string>('pythonPath', 'python');
    bridge = new PromptlyBridge(pythonPath);

    // Initialize tree providers
    promptProvider = new PromptTreeProvider(bridge);
    chainProvider = new ChainTreeProvider(bridge);
    skillProvider = new SkillTreeProvider(bridge);

    // Register tree views
    vscode.window.registerTreeDataProvider('promptlyPrompts', promptProvider);
    vscode.window.registerTreeDataProvider('promptlyChains', chainProvider);
    vscode.window.registerTreeDataProvider('promptlySkills', skillProvider);

    // Register commands
    context.subscriptions.push(
        // Refresh
        vscode.commands.registerCommand('promptly.refresh', () => {
            promptProvider.refresh();
            chainProvider.refresh();
            skillProvider.refresh();
            vscode.window.showInformationMessage('Promptly: Refreshed');
        }),

        // Add Prompt
        vscode.commands.registerCommand('promptly.addPrompt', async () => {
            const name = await vscode.window.showInputBox({
                prompt: 'Prompt name',
                placeHolder: 'e.g., greeting, system_prompt, code_review'
            });
            if (!name) { return; }

            const content = await vscode.window.showInputBox({
                prompt: 'Prompt content (use {{variable}} for placeholders)',
                placeHolder: 'Hello, {{name}}! Welcome to {{project}}.'
            });
            if (!content) { return; }

            const scope = config.get<string>('scope', 'local');
            const result = await bridge.addPrompt(name, content, scope);

            if (result.success) {
                vscode.window.showInformationMessage(`Added prompt: ${name} (v${result.version})`);
                promptProvider.refresh();
            } else {
                vscode.window.showErrorMessage(`Failed to add prompt: ${result.error}`);
            }
        }),

        // Get Prompt
        vscode.commands.registerCommand('promptly.getPrompt', async () => {
            const prompts = await bridge.listPrompts();
            if (!prompts.length) {
                vscode.window.showWarningMessage('No prompts found');
                return;
            }

            const items = prompts.map(p => ({
                label: p.name,
                description: `v${p.version} (${p.branch})`,
                detail: p.updated
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a prompt'
            });

            if (selected) {
                const prompt = await bridge.getPrompt(selected.label);
                if (prompt) {
                    const doc = await vscode.workspace.openTextDocument({
                        content: prompt.content,
                        language: 'markdown'
                    });
                    await vscode.window.showTextDocument(doc);
                }
            }
        }),

        // Edit Prompt
        vscode.commands.registerCommand('promptly.editPrompt', async (item?: PromptItem) => {
            let name = item?.name;
            if (!name) {
                const prompts = await bridge.listPrompts();
                const items = prompts.map(p => p.name);
                name = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select prompt to edit'
                });
            }
            if (!name) { return; }

            const prompt = await bridge.getPrompt(name);
            if (!prompt) {
                vscode.window.showErrorMessage(`Prompt '${name}' not found`);
                return;
            }

            const content = await vscode.window.showInputBox({
                prompt: `Edit content for '${name}'`,
                value: prompt.content
            });

            if (content !== undefined) {
                const scope = config.get<string>('scope', 'local');
                const result = await bridge.addPrompt(name, content, scope);
                if (result.success) {
                    vscode.window.showInformationMessage(`Updated: ${name} (v${result.version})`);
                    promptProvider.refresh();
                }
            }
        }),

        // Delete Prompt
        vscode.commands.registerCommand('promptly.deletePrompt', async (item?: PromptItem) => {
            let name = item?.name;
            if (!name) {
                const prompts = await bridge.listPrompts();
                const items = prompts.map(p => p.name);
                name = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select prompt to delete'
                });
            }
            if (!name) { return; }

            const confirm = await vscode.window.showWarningMessage(
                `Delete prompt '${name}'?`,
                { modal: true },
                'Delete'
            );

            if (confirm === 'Delete') {
                const result = await bridge.deletePrompt(name);
                if (result.success) {
                    vscode.window.showInformationMessage(`Deleted: ${name}`);
                    promptProvider.refresh();
                }
            }
        }),

        // Insert Prompt at Cursor
        vscode.commands.registerCommand('promptly.insertPrompt', async (item?: PromptItem) => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            let name = item?.name;
            if (!name) {
                const prompts = await bridge.listPrompts();
                const items = prompts.map(p => ({
                    label: p.name,
                    description: `v${p.version}`
                }));
                const selected = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select prompt to insert'
                });
                name = selected?.label;
            }
            if (!name) { return; }

            const prompt = await bridge.getPrompt(name);
            if (prompt) {
                await editor.edit(editBuilder => {
                    editBuilder.insert(editor.selection.active, prompt.content);
                });
            }
        }),

        // Execute Prompt
        vscode.commands.registerCommand('promptly.executePrompt', async (item?: PromptItem) => {
            let name = item?.name;
            if (!name) {
                const prompts = await bridge.listPrompts();
                const items = prompts.map(p => p.name);
                name = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select prompt to execute'
                });
            }
            if (!name) { return; }

            // Get variables from user
            const prompt = await bridge.getPrompt(name);
            if (!prompt) { return; }

            // Find {{variables}} in content
            const varMatches = prompt.content.match(/\{\{(\w+)\}\}/g) || [];
            const variables: Record<string, string> = {};

            for (const match of varMatches) {
                const varName = match.replace(/\{\{|\}\}/g, '');
                if (!variables[varName]) {
                    const value = await vscode.window.showInputBox({
                        prompt: `Enter value for ${varName}`,
                        placeHolder: varName
                    });
                    if (value !== undefined) {
                        variables[varName] = value;
                    }
                }
            }

            // Execute
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Executing: ${name}`,
                cancellable: false
            }, async () => {
                const result = await bridge.executePrompt(name!, variables);
                if (result.success) {
                    // Show result in new document
                    const doc = await vscode.workspace.openTextDocument({
                        content: `# Prompt: ${name}\n\n${result.response}`,
                        language: 'markdown'
                    });
                    await vscode.window.showTextDocument(doc, { preview: true });
                } else {
                    vscode.window.showErrorMessage(`Execution failed: ${result.error}`);
                }
            });
        }),

        // Create Branch
        vscode.commands.registerCommand('promptly.createBranch', async () => {
            const name = await vscode.window.showInputBox({
                prompt: 'New branch name',
                placeHolder: 'e.g., experiments, feature-x'
            });
            if (!name) { return; }

            const result = await bridge.createBranch(name);
            if (result.success) {
                vscode.window.showInformationMessage(`Created branch: ${name}`);
            } else {
                vscode.window.showErrorMessage(`Failed: ${result.error}`);
            }
        }),

        // Checkout Branch
        vscode.commands.registerCommand('promptly.checkoutBranch', async () => {
            const branches = await bridge.listBranches();
            const selected = await vscode.window.showQuickPick(branches, {
                placeHolder: 'Select branch to checkout'
            });

            if (selected) {
                const result = await bridge.checkoutBranch(selected);
                if (result.success) {
                    vscode.window.showInformationMessage(`Switched to: ${selected}`);
                    promptProvider.refresh();
                }
            }
        }),

        // Show History
        vscode.commands.registerCommand('promptly.showHistory', async (item?: PromptItem) => {
            let name = item?.name;
            if (!name) {
                const prompts = await bridge.listPrompts();
                const items = prompts.map(p => p.name);
                name = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select prompt to view history'
                });
            }
            if (!name) { return; }

            const history = await bridge.getHistory(name);
            if (history.length === 0) {
                vscode.window.showWarningMessage(`No history for '${name}'`);
                return;
            }

            const items = history.map(h => ({
                label: `v${h.version}`,
                description: h.hash.substring(0, 12),
                detail: h.created
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: `History for '${name}'`
            });

            if (selected) {
                const version = parseInt(selected.label.substring(1));
                const prompt = await bridge.getPrompt(name, version);
                if (prompt) {
                    const doc = await vscode.workspace.openTextDocument({
                        content: `# ${name} v${version}\n\n${prompt.content}`,
                        language: 'markdown'
                    });
                    await vscode.window.showTextDocument(doc);
                }
            }
        }),

        // Create Chain
        vscode.commands.registerCommand('promptly.createChain', async () => {
            const name = await vscode.window.showInputBox({
                prompt: 'Chain name',
                placeHolder: 'e.g., summarize_flow, code_review_pipeline'
            });
            if (!name) { return; }

            const prompts = await bridge.listPrompts();
            const promptNames = prompts.map(p => p.name);

            const steps: string[] = [];
            while (true) {
                const selected = await vscode.window.showQuickPick(
                    [...promptNames, '$(check) Done'],
                    { placeHolder: `Add step ${steps.length + 1} (or Done)` }
                );

                if (!selected || selected === '$(check) Done') {
                    break;
                }
                steps.push(selected);
            }

            if (steps.length > 0) {
                const result = await bridge.createChain(name, steps);
                if (result.success) {
                    vscode.window.showInformationMessage(`Created chain: ${name} (${steps.length} steps)`);
                    chainProvider.refresh();
                }
            }
        }),

        // Run Chain
        vscode.commands.registerCommand('promptly.runChain', async (item?: any) => {
            let name = item?.name;
            if (!name) {
                const chains = await bridge.listChains();
                const items = chains.map(c => c.name);
                name = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select chain to run'
                });
            }
            if (!name) { return; }

            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Running chain: ${name}`,
                cancellable: false
            }, async () => {
                const result = await bridge.runChain(name!);
                if (result.success) {
                    const doc = await vscode.workspace.openTextDocument({
                        content: `# Chain: ${name}\n\n${JSON.stringify(result.outputs, null, 2)}`,
                        language: 'json'
                    });
                    await vscode.window.showTextDocument(doc);
                } else {
                    vscode.window.showErrorMessage(`Chain failed: ${result.error}`);
                }
            });
        }),

        // Create Skill
        vscode.commands.registerCommand('promptly.createSkill', async () => {
            const name = await vscode.window.showInputBox({
                prompt: 'Skill name',
                placeHolder: 'e.g., code_review, data_analysis'
            });
            if (!name) { return; }

            const description = await vscode.window.showInputBox({
                prompt: 'Description',
                placeHolder: 'What does this skill do?'
            });

            const template = await vscode.window.showInputBox({
                prompt: 'Prompt template',
                placeHolder: 'Analyze the following: {{input}}'
            });
            if (!template) { return; }

            const result = await bridge.createSkill(name, description || '', template);
            if (result.success) {
                vscode.window.showInformationMessage(`Created skill: ${name}`);
                skillProvider.refresh();
            }
        }),

        // Run Skill
        vscode.commands.registerCommand('promptly.runSkill', async (item?: any) => {
            let name = item?.name;
            if (!name) {
                const skills = await bridge.listSkills();
                const items = skills.map(s => s.name);
                name = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select skill to run'
                });
            }
            if (!name) { return; }

            const input = await vscode.window.showInputBox({
                prompt: 'Input for skill',
                placeHolder: 'Enter input...'
            });

            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Running skill: ${name}`,
                cancellable: false
            }, async () => {
                const result = await bridge.runSkill(name!, { input: input || '' });
                if (result.success) {
                    const doc = await vscode.workspace.openTextDocument({
                        content: `# Skill: ${name}\n\n${result.response}`,
                        language: 'markdown'
                    });
                    await vscode.window.showTextDocument(doc);
                } else {
                    vscode.window.showErrorMessage(`Skill failed: ${result.error}`);
                }
            });
        }),

        // Evaluate Prompt
        vscode.commands.registerCommand('promptly.evalPrompt', async (item?: PromptItem) => {
            let name = item?.name;
            if (!name) {
                const prompts = await bridge.listPrompts();
                const items = prompts.map(p => p.name);
                name = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select prompt to evaluate'
                });
            }
            if (!name) { return; }

            const criteria = await vscode.window.showQuickPick(
                ['quality', 'clarity', 'safety', 'completeness', 'relevance'],
                { placeHolder: 'Select evaluation criteria' }
            );

            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Evaluating: ${name}`,
                cancellable: false
            }, async () => {
                const result = await bridge.evalPrompt(name!, criteria || 'quality');
                if (result.success) {
                    vscode.window.showInformationMessage(
                        `${name}: ${result.score}/10 - ${result.feedback}`
                    );
                } else {
                    vscode.window.showErrorMessage(`Evaluation failed: ${result.error}`);
                }
            });
        }),

        // Export Prompt
        vscode.commands.registerCommand('promptly.exportPrompt', async (item?: PromptItem) => {
            let name = item?.name;
            if (!name) {
                const prompts = await bridge.listPrompts();
                const items = prompts.map(p => p.name);
                name = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select prompt to export'
                });
            }
            if (!name) { return; }

            const result = await bridge.exportPrompt(name);
            if (result.success) {
                const uri = await vscode.window.showSaveDialog({
                    defaultUri: vscode.Uri.file(`${name}.yaml`),
                    filters: { 'YAML': ['yaml', 'yml'], 'JSON': ['json'] }
                });

                if (uri) {
                    await vscode.workspace.fs.writeFile(uri, Buffer.from(result.data));
                    vscode.window.showInformationMessage(`Exported to: ${uri.fsPath}`);
                }
            }
        }),

        // Import Prompt
        vscode.commands.registerCommand('promptly.importPrompt', async () => {
            const uri = await vscode.window.showOpenDialog({
                filters: { 'YAML/JSON': ['yaml', 'yml', 'json'] },
                canSelectMany: false
            });

            if (uri && uri[0]) {
                const content = await vscode.workspace.fs.readFile(uri[0]);
                const scope = config.get<string>('scope', 'local');
                const result = await bridge.importPrompt(content.toString(), scope);

                if (result.success) {
                    vscode.window.showInformationMessage(`Imported: v${result.version}`);
                    promptProvider.refresh();
                } else {
                    vscode.window.showErrorMessage(`Import failed: ${result.error}`);
                }
            }
        }),

        // Show Info
        vscode.commands.registerCommand('promptly.showInfo', async () => {
            const info = await bridge.getInfo();
            const content = `# Promptly Configuration

- **Global Directory**: ${info.globalDir}
- **Local Directory**: ${info.localDir || 'Not set'}
- **Global Prompts**: ${info.globalPrompts}
- **Local Prompts**: ${info.localPrompts}
- **Current Branch**: ${info.currentBranch}
`;
            const doc = await vscode.workspace.openTextDocument({
                content,
                language: 'markdown'
            });
            await vscode.window.showTextDocument(doc);
        })
    );

    // Status bar
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = "$(note) Promptly";
    statusBar.tooltip = "Promptly - Personal Prompt Management";
    statusBar.command = 'promptly.showInfo';
    statusBar.show();
    context.subscriptions.push(statusBar);

    // Auto-refresh on file system changes
    if (config.get<boolean>('autoRefresh', true)) {
        const watcher = vscode.workspace.createFileSystemWatcher('**/.promptly/**');
        watcher.onDidChange(() => promptProvider.refresh());
        watcher.onDidCreate(() => promptProvider.refresh());
        watcher.onDidDelete(() => promptProvider.refresh());
        context.subscriptions.push(watcher);
    }

    // Initial refresh
    promptProvider.refresh();
    chainProvider.refresh();
    skillProvider.refresh();
}

export function deactivate() {
    console.log('Promptly extension deactivated');
}
