/**
 * PromptTreeProvider - Tree view provider for Promptly prompts
 *
 * Displays prompts in the VS Code sidebar with version and branch info.
 */

import * as vscode from 'vscode';
import { PromptlyBridge, Prompt } from '../PromptlyBridge';

export class PromptItem extends vscode.TreeItem {
    constructor(
        public readonly prompt: Prompt,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(prompt.name, collapsibleState);

        this.tooltip = this.buildTooltip();
        this.description = `v${prompt.version} (${prompt.branch})`;
        this.contextValue = 'prompt';

        // Set icon based on prompt type/metadata
        this.iconPath = new vscode.ThemeIcon('file-text');

        // Command to show prompt on click
        this.command = {
            command: 'promptly.getPrompt',
            title: 'View Prompt',
            arguments: [this]
        };
    }

    private buildTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`**${this.prompt.name}**\n\n`);
        md.appendMarkdown(`- Version: ${this.prompt.version}\n`);
        md.appendMarkdown(`- Branch: ${this.prompt.branch}\n`);
        md.appendMarkdown(`- Hash: \`${this.prompt.commit_hash?.slice(0, 8) || 'N/A'}\`\n`);
        md.appendMarkdown(`- Updated: ${this.prompt.updated_at}\n\n`);

        if (this.prompt.content) {
            const preview = this.prompt.content.slice(0, 200);
            md.appendCodeblock(preview + (this.prompt.content.length > 200 ? '...' : ''), 'text');
        }

        return md;
    }
}

export class PromptVersionItem extends vscode.TreeItem {
    constructor(
        public readonly promptName: string,
        public readonly version: number,
        public readonly commitHash: string,
        public readonly message: string,
        public readonly createdAt: string
    ) {
        super(`v${version}`, vscode.TreeItemCollapsibleState.None);

        this.description = message || commitHash.slice(0, 8);
        this.tooltip = `Version ${version}\nHash: ${commitHash}\nCreated: ${createdAt}`;
        this.contextValue = 'promptVersion';
        this.iconPath = new vscode.ThemeIcon('git-commit');

        this.command = {
            command: 'promptly.getPromptVersion',
            title: 'View Version',
            arguments: [this.promptName, this.version]
        };
    }
}

export class BranchItem extends vscode.TreeItem {
    constructor(
        public readonly branchName: string,
        public readonly isCurrent: boolean
    ) {
        super(branchName, vscode.TreeItemCollapsibleState.None);

        this.description = isCurrent ? '(current)' : '';
        this.contextValue = isCurrent ? 'currentBranch' : 'branch';
        this.iconPath = new vscode.ThemeIcon(isCurrent ? 'check' : 'git-branch');

        if (!isCurrent) {
            this.command = {
                command: 'promptly.checkoutBranch',
                title: 'Checkout Branch',
                arguments: [this.branchName]
            };
        }
    }
}

export class PromptTreeProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<vscode.TreeItem | undefined | null | void> =
        new vscode.EventEmitter<vscode.TreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<vscode.TreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private prompts: Prompt[] = [];
    private branches: string[] = [];
    private currentBranch: string = 'main';
    private showBranches: boolean = false;

    constructor(private bridge: PromptlyBridge) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    toggleBranchView(): void {
        this.showBranches = !this.showBranches;
        this.refresh();
    }

    getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: vscode.TreeItem): Promise<vscode.TreeItem[]> {
        if (!element) {
            // Root level
            return this.getRootItems();
        }

        if (element instanceof PromptItem) {
            // Show versions for this prompt
            return this.getPromptVersions(element.prompt.name);
        }

        return [];
    }

    private async getRootItems(): Promise<vscode.TreeItem[]> {
        const items: vscode.TreeItem[] = [];

        // Add branch section header if showing branches
        if (this.showBranches) {
            const branchResult = await this.bridge.listBranches();
            const currentBranchResult = await this.bridge.getCurrentBranch();

            if (branchResult.success && branchResult.data) {
                this.branches = branchResult.data;
            }
            if (currentBranchResult.success) {
                this.currentBranch = currentBranchResult.data || 'main';
            }

            // Add branch items
            for (const branch of this.branches) {
                items.push(new BranchItem(branch, branch === this.currentBranch));
            }

            // Add separator
            const separator = new vscode.TreeItem('─────────────', vscode.TreeItemCollapsibleState.None);
            separator.contextValue = 'separator';
            items.push(separator);
        }

        // Fetch prompts
        const result = await this.bridge.listPrompts();
        if (result.success && result.data) {
            this.prompts = result.data;

            for (const prompt of this.prompts) {
                items.push(new PromptItem(prompt, vscode.TreeItemCollapsibleState.Collapsed));
            }
        }

        if (items.length === 0) {
            const emptyItem = new vscode.TreeItem('No prompts found', vscode.TreeItemCollapsibleState.None);
            emptyItem.contextValue = 'empty';
            emptyItem.iconPath = new vscode.ThemeIcon('info');
            items.push(emptyItem);
        }

        return items;
    }

    private async getPromptVersions(promptName: string): Promise<vscode.TreeItem[]> {
        const result = await this.bridge.getPromptHistory(promptName, 10);
        if (!result.success || !result.data) {
            return [];
        }

        return result.data.map(v => new PromptVersionItem(
            promptName,
            v.version,
            v.commit_hash,
            v.message,
            v.created_at
        ));
    }

    /**
     * Get prompt by name from cached list
     */
    getPrompt(name: string): Prompt | undefined {
        return this.prompts.find(p => p.name === name);
    }

    /**
     * Get all cached prompts
     */
    getPrompts(): Prompt[] {
        return this.prompts;
    }
}
