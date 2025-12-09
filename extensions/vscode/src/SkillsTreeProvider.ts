/**
 * SkillsTreeProvider - Tree view provider for Skills sidebar
 *
 * Displays installed skills grouped by category with actions.
 */

import * as vscode from 'vscode';
import { SkillsManager, Skill, InstalledSkill, SkillEvent } from './SkillsManager';

type SkillTreeItemType = 'category' | 'skill' | 'action';

interface SkillTreeItemData {
    type: SkillTreeItemType;
    category?: string;
    skill?: Skill;
    installed?: InstalledSkill;
    action?: string;
}

export class SkillTreeItem extends vscode.TreeItem {
    constructor(
        public readonly data: SkillTreeItemData,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(
            data.type === 'category'
                ? data.category!
                : data.type === 'skill'
                ? data.skill!.name
                : data.action!,
            collapsibleState
        );
        this.setupTreeItem();
    }

    private setupTreeItem(): void {
        switch (this.data.type) {
            case 'category':
                this.iconPath = new vscode.ThemeIcon('folder');
                this.contextValue = 'skillCategory';
                break;
            case 'skill':
                this.setupSkill();
                break;
            case 'action':
                this.setupAction();
                break;
        }
    }

    private setupSkill(): void {
        const skill = this.data.skill!;
        const installed = this.data.installed;

        const status = installed?.enabled ? '' : ' (disabled)';
        this.description = `v${skill.version}${status}`;

        this.tooltip = new vscode.MarkdownString();
        this.tooltip.appendMarkdown(`### ${skill.name}\n\n`);
        this.tooltip.appendMarkdown(`${skill.description}\n\n`);
        this.tooltip.appendMarkdown(`**Author:** ${skill.author}\n\n`);
        this.tooltip.appendMarkdown(`**Version:** ${skill.version}\n`);

        if (!installed?.enabled) {
            this.iconPath = new vscode.ThemeIcon('circle-slash');
        } else {
            const icons: Record<string, string> = {
                domain: 'library',
                utility: 'tools',
                integration: 'plug',
                workflow: 'workflow'
            };
            this.iconPath = new vscode.ThemeIcon(icons[skill.category] || 'extensions');
        }

        this.contextValue = installed?.enabled ? 'skillInstalled' : 'skillDisabled';
        this.command = {
            command: 'skills.invoke',
            title: 'Invoke Skill',
            arguments: [skill.skill_id]
        };
    }

    private setupAction(): void {
        if (this.data.action === 'browse') {
            this.label = 'Browse Marketplace';
            this.iconPath = new vscode.ThemeIcon('search');
            this.command = { command: 'skills.openExplorer', title: 'Browse' };
        }
    }
}

export class SkillsTreeProvider implements vscode.TreeDataProvider<SkillTreeItem>, vscode.Disposable {
    private _onDidChangeTreeData = new vscode.EventEmitter<void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private skillsByCategory: Map<string, { skill: Skill; installed: InstalledSkill }[]> = new Map();
    private eventSubscription: vscode.Disposable;

    constructor(private skillsManager: SkillsManager) {
        this.eventSubscription = skillsManager.onSkillEvent(() => this.refresh());
        this.refresh();
    }

    async refresh(): Promise<void> {
        this.skillsByCategory.clear();

        const installed = this.skillsManager.getInstalledSkills();
        const skills = await this.skillsManager.getInstalledSkillDetails();

        for (const skill of skills) {
            const info = installed.find(i => i.skill_id === skill.skill_id);
            if (!info) continue;

            const cat = skill.category || 'other';
            if (!this.skillsByCategory.has(cat)) {
                this.skillsByCategory.set(cat, []);
            }
            this.skillsByCategory.get(cat)!.push({ skill, installed: info });
        }

        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: SkillTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: SkillTreeItem): Promise<SkillTreeItem[]> {
        if (!element) {
            const items: SkillTreeItem[] = [
                new SkillTreeItem({ type: 'action', action: 'browse' }, vscode.TreeItemCollapsibleState.None)
            ];

            for (const [cat, skills] of this.skillsByCategory) {
                const item = new SkillTreeItem(
                    { type: 'category', category: cat.charAt(0).toUpperCase() + cat.slice(1) },
                    vscode.TreeItemCollapsibleState.Expanded
                );
                item.description = `(${skills.length})`;
                items.push(item);
            }

            if (this.skillsByCategory.size === 0) {
                const empty = new SkillTreeItem(
                    { type: 'action', action: 'browse' },
                    vscode.TreeItemCollapsibleState.None
                );
                empty.label = 'No skills installed - Browse Marketplace';
            }

            return items;
        }

        if (element.data.type === 'category') {
            const catKey = element.data.category!.toLowerCase();
            const skills = this.skillsByCategory.get(catKey) || [];
            return skills.map(({ skill, installed }) =>
                new SkillTreeItem({ type: 'skill', skill, installed }, vscode.TreeItemCollapsibleState.None)
            );
        }

        return [];
    }

    dispose(): void {
        this._onDidChangeTreeData.dispose();
        this.eventSubscription.dispose();
    }
}