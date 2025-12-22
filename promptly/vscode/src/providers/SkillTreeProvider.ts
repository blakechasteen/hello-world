/**
 * SkillTreeProvider - Tree view provider for Promptly skills
 *
 * Displays skills in the VS Code sidebar with schema information.
 */

import * as vscode from 'vscode';
import { PromptlyBridge, Skill } from '../PromptlyBridge';

export class SkillItem extends vscode.TreeItem {
    constructor(
        public readonly skill: Skill,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(skill.name, collapsibleState);

        this.tooltip = this.buildTooltip();
        this.description = skill.description.slice(0, 50) + (skill.description.length > 50 ? '...' : '');
        this.contextValue = 'skill';
        this.iconPath = new vscode.ThemeIcon('symbol-method');

        // Command to run skill on click
        this.command = {
            command: 'promptly.runSkill',
            title: 'Run Skill',
            arguments: [this]
        };
    }

    private buildTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`**${this.skill.name}**\n\n`);
        md.appendMarkdown(`${this.skill.description}\n\n`);

        // Show input schema
        if (this.skill.input_schema && Object.keys(this.skill.input_schema).length > 0) {
            md.appendMarkdown(`**Inputs:**\n`);
            for (const [key, value] of Object.entries(this.skill.input_schema)) {
                md.appendMarkdown(`- \`${key}\`: ${typeof value === 'object' ? JSON.stringify(value) : value}\n`);
            }
            md.appendMarkdown('\n');
        }

        // Show output schema
        if (this.skill.output_schema && Object.keys(this.skill.output_schema).length > 0) {
            md.appendMarkdown(`**Outputs:**\n`);
            for (const [key, value] of Object.entries(this.skill.output_schema)) {
                md.appendMarkdown(`- \`${key}\`: ${typeof value === 'object' ? JSON.stringify(value) : value}\n`);
            }
        }

        return md;
    }
}

export class SkillDetailItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly detail: string,
        public readonly icon: string = 'symbol-property'
    ) {
        super(label, vscode.TreeItemCollapsibleState.None);

        this.description = detail;
        this.contextValue = 'skillDetail';
        this.iconPath = new vscode.ThemeIcon(icon);
    }
}

export class SkillInputItem extends vscode.TreeItem {
    constructor(
        public readonly skillName: string,
        public readonly inputName: string,
        public readonly inputSpec: any
    ) {
        super(inputName, vscode.TreeItemCollapsibleState.None);

        const type = typeof inputSpec === 'object' ? inputSpec.type || 'any' : inputSpec;
        this.description = type;
        this.contextValue = 'skillInput';
        this.iconPath = new vscode.ThemeIcon('symbol-parameter');

        if (typeof inputSpec === 'object') {
            const md = new vscode.MarkdownString();
            md.appendMarkdown(`**${inputName}**\n\n`);
            md.appendCodeblock(JSON.stringify(inputSpec, null, 2), 'json');
            this.tooltip = md;
        } else {
            this.tooltip = `${inputName}: ${type}`;
        }
    }
}

export class SkillTreeProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<vscode.TreeItem | undefined | null | void> =
        new vscode.EventEmitter<vscode.TreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<vscode.TreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private skills: Skill[] = [];

    constructor(private bridge: PromptlyBridge) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: vscode.TreeItem): Promise<vscode.TreeItem[]> {
        if (!element) {
            // Root level - list all skills
            return this.getRootItems();
        }

        if (element instanceof SkillItem) {
            // Show details for this skill
            return this.getSkillDetails(element.skill);
        }

        return [];
    }

    private async getRootItems(): Promise<vscode.TreeItem[]> {
        const result = await this.bridge.listSkills();

        if (!result.success || !result.data) {
            const emptyItem = new vscode.TreeItem('No skills found', vscode.TreeItemCollapsibleState.None);
            emptyItem.contextValue = 'empty';
            emptyItem.iconPath = new vscode.ThemeIcon('info');
            return [emptyItem];
        }

        this.skills = result.data;

        if (this.skills.length === 0) {
            const emptyItem = new vscode.TreeItem('No skills found', vscode.TreeItemCollapsibleState.None);
            emptyItem.contextValue = 'empty';
            emptyItem.iconPath = new vscode.ThemeIcon('info');
            return [emptyItem];
        }

        return this.skills.map(skill =>
            new SkillItem(skill, vscode.TreeItemCollapsibleState.Collapsed)
        );
    }

    private getSkillDetails(skill: Skill): vscode.TreeItem[] {
        const items: vscode.TreeItem[] = [];

        // Add description
        items.push(new SkillDetailItem('Description', skill.description, 'info'));

        // Add input parameters
        if (skill.input_schema && Object.keys(skill.input_schema).length > 0) {
            const inputsHeader = new vscode.TreeItem('Inputs', vscode.TreeItemCollapsibleState.None);
            inputsHeader.iconPath = new vscode.ThemeIcon('arrow-right');
            items.push(inputsHeader);

            for (const [name, spec] of Object.entries(skill.input_schema)) {
                items.push(new SkillInputItem(skill.name, name, spec));
            }
        }

        // Add output parameters
        if (skill.output_schema && Object.keys(skill.output_schema).length > 0) {
            const outputsHeader = new vscode.TreeItem('Outputs', vscode.TreeItemCollapsibleState.None);
            outputsHeader.iconPath = new vscode.ThemeIcon('arrow-left');
            items.push(outputsHeader);

            for (const [name, spec] of Object.entries(skill.output_schema)) {
                items.push(new SkillInputItem(skill.name, name, spec));
            }
        }

        // Add created date
        items.push(new SkillDetailItem('Created', skill.created_at, 'calendar'));

        return items;
    }

    /**
     * Get skill by name from cached list
     */
    getSkill(name: string): Skill | undefined {
        return this.skills.find(s => s.name === name);
    }

    /**
     * Get all cached skills
     */
    getSkills(): Skill[] {
        return this.skills;
    }
}
