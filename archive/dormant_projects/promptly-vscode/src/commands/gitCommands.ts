import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class GitCommands {
    private getCwd(): string {
        return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
    }

    async status(): Promise<string> {
        try {
            const { stdout } = await execAsync('git status --short', {
                cwd: this.getCwd()
            });

            if (!stdout.trim()) {
                return '✅ Working tree clean';
            }

            return `**Git Status:**\n\`\`\`\n${stdout}\`\`\``;

        } catch (error: any) {
            return `❌ Git status failed: ${error.message}`;
        }
    }

    async commit(message?: string): Promise<string> {
        if (!message) {
            message = await vscode.window.showInputBox({
                prompt: 'Enter commit message',
                placeHolder: 'feat: Add new feature'
            });
        }

        if (!message) {
            return '❌ Commit cancelled';
        }

        try {
            // Stage all changes
            await execAsync('git add .', { cwd: this.getCwd() });

            // Commit
            const { stdout } = await execAsync(`git commit -m "${message.replace(/"/g, '\\"')}"`, {
                cwd: this.getCwd()
            });

            return `✅ **Committed:** ${message}\n\`\`\`\n${stdout}\`\`\``;

        } catch (error: any) {
            return `❌ Commit failed: ${error.message}`;
        }
    }

    async push(): Promise<string> {
        try {
            const { stdout, stderr } = await execAsync('git push', {
                cwd: this.getCwd()
            });

            return `✅ **Pushed to remote**\n\`\`\`\n${stdout || stderr}\`\`\``;

        } catch (error: any) {
            return `❌ Push failed: ${error.message}\n\n_Tip: Make sure you have push access and remote is configured_`;
        }
    }

    async log(): Promise<string> {
        try {
            const { stdout } = await execAsync('git log --oneline -10', {
                cwd: this.getCwd()
            });

            return `**Recent Commits:**\n\`\`\`\n${stdout}\`\`\``;

        } catch (error: any) {
            return `❌ Git log failed: ${error.message}`;
        }
    }

    async diff(): Promise<string> {
        try {
            const { stdout } = await execAsync('git diff', {
                cwd: this.getCwd()
            });

            if (!stdout.trim()) {
                return 'No changes in working tree';
            }

            return `**Git Diff:**\n\`\`\`diff\n${stdout}\`\`\``;

        } catch (error: any) {
            return `❌ Git diff failed: ${error.message}`;
        }
    }
}