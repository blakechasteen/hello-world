/**
 * ChainTreeProvider - Tree view provider for Promptly chains
 *
 * Displays prompt chains in the VS Code sidebar with step details.
 */

import * as vscode from 'vscode';
import { PromptlyBridge, Chain, ChainStep } from '../PromptlyBridge';

export class ChainItem extends vscode.TreeItem {
    constructor(
        public readonly chain: Chain,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(chain.name, collapsibleState);

        this.tooltip = this.buildTooltip();
        this.description = `${chain.steps.length} steps`;
        this.contextValue = 'chain';
        this.iconPath = new vscode.ThemeIcon('link');

        // Command to run chain on click
        this.command = {
            command: 'promptly.runChain',
            title: 'Run Chain',
            arguments: [this]
        };
    }

    private buildTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`**${this.chain.name}**\n\n`);
        md.appendMarkdown(`Steps: ${this.chain.steps.length}\n\n`);

        md.appendMarkdown(`**Pipeline:**\n`);
        this.chain.steps.forEach((step, i) => {
            const arrow = i < this.chain.steps.length - 1 ? ' →' : '';
            md.appendMarkdown(`${i + 1}. \`${step.prompt_name}\`${step.output_var ? ` (→ ${step.output_var})` : ''}${arrow}\n`);
        });

        return md;
    }
}

export class ChainStepItem extends vscode.TreeItem {
    constructor(
        public readonly chainName: string,
        public readonly step: ChainStep,
        public readonly stepIndex: number,
        public readonly isLast: boolean
    ) {
        super(`${stepIndex + 1}. ${step.prompt_name}`, vscode.TreeItemCollapsibleState.None);

        this.description = step.output_var ? `→ ${step.output_var}` : '';
        this.tooltip = this.buildTooltip();
        this.contextValue = 'chainStep';
        this.iconPath = new vscode.ThemeIcon(isLast ? 'debug-stop' : 'debug-continue');

        // Command to view the prompt
        this.command = {
            command: 'promptly.getPrompt',
            title: 'View Prompt',
            arguments: [{ prompt: { name: step.prompt_name } }]
        };
    }

    private buildTooltip(): string {
        let tip = `Step ${this.stepIndex + 1}: ${this.step.prompt_name}`;
        if (this.step.output_var) {
            tip += `\nOutput variable: ${this.step.output_var}`;
        }
        if (this.step.condition) {
            tip += `\nCondition: ${this.step.condition}`;
        }
        return tip;
    }
}

export class ChainTreeProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<vscode.TreeItem | undefined | null | void> =
        new vscode.EventEmitter<vscode.TreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<vscode.TreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private chains: Chain[] = [];

    constructor(private bridge: PromptlyBridge) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: vscode.TreeItem): Promise<vscode.TreeItem[]> {
        if (!element) {
            // Root level - list all chains
            return this.getRootItems();
        }

        if (element instanceof ChainItem) {
            // Show steps for this chain
            return this.getChainSteps(element.chain);
        }

        return [];
    }

    private async getRootItems(): Promise<vscode.TreeItem[]> {
        const result = await this.bridge.listChains();

        if (!result.success || !result.data) {
            const emptyItem = new vscode.TreeItem('No chains found', vscode.TreeItemCollapsibleState.None);
            emptyItem.contextValue = 'empty';
            emptyItem.iconPath = new vscode.ThemeIcon('info');
            return [emptyItem];
        }

        this.chains = result.data;

        if (this.chains.length === 0) {
            const emptyItem = new vscode.TreeItem('No chains found', vscode.TreeItemCollapsibleState.None);
            emptyItem.contextValue = 'empty';
            emptyItem.iconPath = new vscode.ThemeIcon('info');
            return [emptyItem];
        }

        return this.chains.map(chain =>
            new ChainItem(chain, vscode.TreeItemCollapsibleState.Collapsed)
        );
    }

    private getChainSteps(chain: Chain): vscode.TreeItem[] {
        return chain.steps.map((step, index) =>
            new ChainStepItem(
                chain.name,
                step,
                index,
                index === chain.steps.length - 1
            )
        );
    }

    /**
     * Get chain by name from cached list
     */
    getChain(name: string): Chain | undefined {
        return this.chains.find(c => c.name === name);
    }

    /**
     * Get all cached chains
     */
    getChains(): Chain[] {
        return this.chains;
    }
}
