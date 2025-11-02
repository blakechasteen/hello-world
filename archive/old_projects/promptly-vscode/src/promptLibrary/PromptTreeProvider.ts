import * as vscode from 'vscode';
import { PromptlyBridge } from '../api/PromptlyBridge';

export class PromptTreeProvider implements vscode.TreeDataProvider<PromptTreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<PromptTreeItem | undefined | null | void> = new vscode.EventEmitter<PromptTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<PromptTreeItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor(private bridge: PromptlyBridge) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: PromptTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: PromptTreeItem): Promise<PromptTreeItem[]> {
        if (!element) {
            // Root level: show branches
            const prompts = await this.bridge.listPrompts();

            // Group by branch
            const branches = new Map<string, any[]>();
            for (const prompt of prompts) {
                const branch = prompt.branch || 'main';
                if (!branches.has(branch)) {
                    branches.set(branch, []);
                }
                branches.get(branch)!.push(prompt);
            }

            // Create branch items
            const items: PromptTreeItem[] = [];
            for (const [branch, branchPrompts] of branches) {
                items.push(new PromptTreeItem(
                    branch,
                    vscode.TreeItemCollapsibleState.Expanded,
                    'branch',
                    branchPrompts
                ));
            }

            return items;
        } else if (element.contextValue === 'branch') {
            // Branch level: show prompts
            return element.children.map(prompt =>
                new PromptTreeItem(
                    prompt.name,
                    vscode.TreeItemCollapsibleState.None,
                    'prompt',
                    [],
                    prompt
                )
            );
        }

        return [];
    }
}

export class PromptTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly contextValue: string,
        public readonly children: any[] = [],
        public readonly promptData?: any
    ) {
        super(label, collapsibleState);

        if (contextValue === 'branch') {
            this.iconPath = new vscode.ThemeIcon('git-branch');
            this.description = `${children.length} prompts`;
            this.tooltip = this.label;
        } else if (contextValue === 'prompt') {
            this.iconPath = new vscode.ThemeIcon('file-text');
            this.command = {
                command: 'promptly.viewPrompt',
                title: 'View Prompt',
                arguments: [{ promptName: promptData.name }]
            };

            if (promptData.tags && promptData.tags.length > 0) {
                this.description = promptData.tags.join(', ');
            }

            // Set tooltip with full details
            this.tooltip = `${promptData.name}\nBranch: ${promptData.branch}\nTags: ${promptData.tags?.join(', ') || 'none'}`;
        }
    }
}