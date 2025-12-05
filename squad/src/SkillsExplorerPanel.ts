/**
 * SkillsExplorerPanel - Webview panel for browsing and installing skills
 */

import * as vscode from 'vscode';
import { SkillsManager, Skill } from './SkillsManager';

export class SkillsExplorerPanel {
    public static currentPanel: SkillsExplorerPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private disposables: vscode.Disposable[] = [];

    private constructor(
        panel: vscode.WebviewPanel,
        private skillsManager: SkillsManager,
        private extensionUri: vscode.Uri
    ) {
        this.panel = panel;
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        this.panel.webview.onDidReceiveMessage(
            message => this.handleMessage(message),
            null,
            this.disposables
        );
        this.updateContent();
    }

    public static show(skillsManager: SkillsManager, extensionUri: vscode.Uri): void {
        if (SkillsExplorerPanel.currentPanel) {
            SkillsExplorerPanel.currentPanel.panel.reveal();
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'skillsExplorer',
            'Skills Marketplace',
            vscode.ViewColumn.One,
            { enableScripts: true, retainContextWhenHidden: true }
        );

        SkillsExplorerPanel.currentPanel = new SkillsExplorerPanel(panel, skillsManager, extensionUri);
    }

    private async handleMessage(message: any): Promise<void> {
        switch (message.command) {
            case 'getSkills':
                await this.sendSkills(message.category, message.search);
                break;
            case 'install':
                await this.installSkill(message.skillId);
                break;
            case 'remove':
                await this.removeSkill(message.skillId);
                break;
            case 'viewDetails':
                await this.showSkillDetails(message.skillId);
                break;
        }
    }

    private async sendSkills(category?: string, search?: string): Promise<void> {
        try {
            const skills = await this.skillsManager.getAvailableSkills({ category, search });
            const installed = this.skillsManager.getInstalledSkills();
            const installedIds = new Set(installed.map(s => s.skill_id));

            this.panel.webview.postMessage({
                command: 'skillsLoaded',
                skills: skills.map(s => ({ ...s, installed: installedIds.has(s.skill_id) }))
            });
        } catch (error: any) {
            this.panel.webview.postMessage({ command: 'error', message: error.message });
        }
    }

    private async installSkill(skillId: string): Promise<void> {
        try {
            await this.skillsManager.installSkill(skillId);
            await this.sendSkills();
        } catch (error: any) {
            vscode.window.showErrorMessage(`Install failed: ${error.message}`);
        }
    }

    private async removeSkill(skillId: string): Promise<void> {
        try {
            await this.skillsManager.removeSkill(skillId);
            await this.sendSkills();
        } catch (error: any) {
            vscode.window.showErrorMessage(`Remove failed: ${error.message}`);
        }
    }

    private async showSkillDetails(skillId: string): Promise<void> {
        try {
            const detail = await this.skillsManager.getSkillDetail(skillId);
            this.panel.webview.postMessage({ command: 'skillDetail', skill: detail });
        } catch (error: any) {
            vscode.window.showErrorMessage(`Failed to load details: ${error.message}`);
        }
    }

    private async updateContent(): Promise<void> {
        this.panel.webview.html = this.getHtml();
        await this.sendSkills();
    }

    private getHtml(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skills Marketplace</title>
    <style>
        :root {
            --bg: var(--vscode-editor-background);
            --fg: var(--vscode-foreground);
            --border: var(--vscode-panel-border);
            --accent: var(--vscode-textLink-foreground);
            --card-bg: var(--vscode-editor-inactiveSelectionBackground);
            --btn-bg: var(--vscode-button-background);
            --btn-fg: var(--vscode-button-foreground);
            --input-bg: var(--vscode-input-background);
            --input-border: var(--vscode-input-border);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: var(--vscode-font-family); background: var(--bg); color: var(--fg); padding: 20px; }

        .header { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
        .search-box { flex: 1; min-width: 200px; }
        .search-box input {
            width: 100%; padding: 8px 12px; background: var(--input-bg);
            border: 1px solid var(--input-border); border-radius: 4px; color: var(--fg);
        }
        .category-filters { display: flex; gap: 8px; flex-wrap: wrap; }
        .category-btn {
            padding: 6px 12px; background: var(--card-bg); border: 1px solid var(--border);
            border-radius: 4px; color: var(--fg); cursor: pointer; font-size: 13px;
        }
        .category-btn.active { background: var(--accent); color: white; border-color: var(--accent); }

        .skills-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }

        .skill-card {
            background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
            padding: 16px; display: flex; flex-direction: column; gap: 12px;
        }
        .skill-header { display: flex; justify-content: space-between; align-items: flex-start; }
        .skill-title { font-size: 16px; font-weight: 600; }
        .skill-version { font-size: 12px; color: var(--vscode-descriptionForeground); }
        .skill-category {
            font-size: 11px; padding: 2px 8px; border-radius: 4px;
            background: var(--accent); color: white; text-transform: uppercase;
        }
        .skill-description { font-size: 13px; color: var(--vscode-descriptionForeground); line-height: 1.4; }
        .skill-meta { display: flex; gap: 16px; font-size: 12px; color: var(--vscode-descriptionForeground); }
        .skill-actions { display: flex; gap: 8px; margin-top: auto; }
        .btn {
            padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 13px;
            border: none; display: flex; align-items: center; gap: 4px;
        }
        .btn-primary { background: var(--btn-bg); color: var(--btn-fg); }
        .btn-secondary { background: transparent; border: 1px solid var(--border); color: var(--fg); }
        .btn-danger { background: var(--vscode-inputValidation-errorBackground); color: var(--fg); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .loading { text-align: center; padding: 40px; color: var(--vscode-descriptionForeground); }
        .error { color: var(--vscode-errorForeground); padding: 20px; text-align: center; }
        .installed-badge { color: var(--vscode-gitDecoration-addedResourceForeground); font-size: 12px; }

        .detail-overlay {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7); display: none; justify-content: center; align-items: center; padding: 20px;
        }
        .detail-overlay.active { display: flex; }
        .detail-panel {
            background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
            max-width: 600px; width: 100%; max-height: 80vh; overflow-y: auto; padding: 24px;
        }
        .detail-close { float: right; cursor: pointer; font-size: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <div class="search-box">
            <input type="text" id="search" placeholder="Search skills..." />
        </div>
        <div class="category-filters">
            <button class="category-btn active" data-category="">All</button>
            <button class="category-btn" data-category="domain">Domain</button>
            <button class="category-btn" data-category="utility">Utility</button>
            <button class="category-btn" data-category="integration">Integration</button>
            <button class="category-btn" data-category="workflow">Workflow</button>
        </div>
    </div>

    <div id="content" class="skills-grid"></div>

    <div id="detail-overlay" class="detail-overlay">
        <div class="detail-panel">
            <span class="detail-close" onclick="closeDetail()">&times;</span>
            <div id="detail-content"></div>
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let currentCategory = '';
        let currentSearch = '';
        let skills = [];

        // Event listeners
        document.getElementById('search').addEventListener('input', (e) => {
            currentSearch = e.target.value;
            loadSkills();
        });

        document.querySelectorAll('.category-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentCategory = btn.dataset.category;
                loadSkills();
            });
        });

        // Load skills
        function loadSkills() {
            document.getElementById('content').innerHTML = '<div class="loading">Loading skills...</div>';
            vscode.postMessage({ command: 'getSkills', category: currentCategory, search: currentSearch });
        }

        // Render skills
        function renderSkills() {
            const container = document.getElementById('content');
            if (skills.length === 0) {
                container.innerHTML = '<div class="loading">No skills found</div>';
                return;
            }

            container.innerHTML = skills.map(skill => \`
                <div class="skill-card">
                    <div class="skill-header">
                        <div>
                            <div class="skill-title">\${skill.name}</div>
                            <div class="skill-version">v\${skill.version} by \${skill.author}</div>
                        </div>
                        <span class="skill-category">\${skill.category}</span>
                    </div>
                    <div class="skill-description">\${skill.description}</div>
                    <div class="skill-meta">
                        <span>⭐ \${skill.rating?.toFixed(1) || 'N/A'}</span>
                        <span>📥 \${skill.downloads || 0}</span>
                        \${skill.installed ? '<span class="installed-badge">✓ Installed</span>' : ''}
                    </div>
                    <div class="skill-actions">
                        <button class="btn btn-secondary" onclick="viewDetails('\${skill.skill_id}')">Details</button>
                        \${skill.installed
                            ? \`<button class="btn btn-danger" onclick="removeSkill('\${skill.skill_id}')">Remove</button>\`
                            : \`<button class="btn btn-primary" onclick="installSkill('\${skill.skill_id}')">Install</button>\`
                        }
                    </div>
                </div>
            \`).join('');
        }

        // Actions
        function installSkill(skillId) {
            vscode.postMessage({ command: 'install', skillId });
        }

        function removeSkill(skillId) {
            vscode.postMessage({ command: 'remove', skillId });
        }

        function viewDetails(skillId) {
            vscode.postMessage({ command: 'viewDetails', skillId });
        }

        function closeDetail() {
            document.getElementById('detail-overlay').classList.remove('active');
        }

        // Message handler
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.command) {
                case 'skillsLoaded':
                    skills = message.skills;
                    renderSkills();
                    break;
                case 'skillDetail':
                    showDetail(message.skill);
                    break;
                case 'error':
                    document.getElementById('content').innerHTML = \`<div class="error">\${message.message}</div>\`;
                    break;
            }
        });

        function showDetail(skill) {
            document.getElementById('detail-content').innerHTML = \`
                <h2>\${skill.name}</h2>
                <p style="color: var(--vscode-descriptionForeground); margin: 8px 0;">v\${skill.version} by \${skill.author}</p>
                <p style="margin: 16px 0;">\${skill.description}</p>
                \${skill.documentation ? \`<div style="margin: 16px 0;"><h3>Documentation</h3><pre style="white-space: pre-wrap; background: var(--card-bg); padding: 12px; border-radius: 4px; margin-top: 8px;">\${skill.documentation}</pre></div>\` : ''}
                \${skill.requirements?.length ? \`<div style="margin: 16px 0;"><h3>Requirements</h3><ul>\${skill.requirements.map(r => \`<li>\${r}</li>\`).join('')}</ul></div>\` : ''}
            \`;
            document.getElementById('detail-overlay').classList.add('active');
        }

        // Initial load
        loadSkills();
    </script>
</body>
</html>`;
    }

    private dispose(): void {
        SkillsExplorerPanel.currentPanel = undefined;
        this.panel.dispose();
        this.disposables.forEach(d => d.dispose());
    }
}