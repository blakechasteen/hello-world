/**
 * Squad Commands - Main command implementations
 *
 * All commands accessible via Command Palette and context menu
 */

import * as vscode from 'vscode';
import { HoloLoomBridge } from '../lib/HoloLoomBridge';
import { CodeContextExtractor } from '../lib/CodeContextExtractor';
import { CacheManager } from '../lib/CacheManager';
import { AgenticResponse } from '../types';

export class SquadCommands {
    private bridge: HoloLoomBridge;
    private extractor: CodeContextExtractor;
    private cache: CacheManager;
    private outputChannel: vscode.OutputChannel;

    constructor(
        bridge: HoloLoomBridge,
        extractor: CodeContextExtractor,
        cache: CacheManager
    ) {
        this.bridge = bridge;
        this.extractor = extractor;
        this.cache = cache;
        this.outputChannel = vscode.window.createOutputChannel('Squad');
    }

    /**
     * Explain selected code
     */
    async explainCode() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('Please select code to explain');
            return;
        }

        try {
            const code = editor.document.getText(selection);
            const context = await this.extractor.extractContext(editor);

            // Check cache first
            const cacheKey = `explain:${this.hashCode(code)}`;
            let result = this.cache.get(cacheKey);

            if (!result) {
                // Show progress
                result = await vscode.window.withProgress({
                    location: vscode.ProgressLocation.Notification,
                    title: 'Squad: Explaining code...',
                    cancellable: false
                }, async () => {
                    const response = await this.bridge.explainCode(code, context);
                    this.cache.set(cacheKey, response, editor.document.fileName);
                    return response;
                });
            }

            await this.showResponse('Code Explanation', result);

        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Find similar code
     */
    async findSimilar() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('Please select code to find similar patterns');
            return;
        }

        try {
            const code = editor.document.getText(selection);
            const context = await this.extractor.extractContext(editor);

            const result = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Squad: Searching for similar code...',
                cancellable: false
            }, async () => {
                return await this.bridge.findSimilar(code, context);
            });

            await this.showResponse('Similar Code', result);

        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Generate unit tests
     */
    async generateTests() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('Please select code to generate tests for');
            return;
        }

        try {
            const code = editor.document.getText(selection);
            const context = await this.extractor.extractContext(editor);

            // Ask for test framework preference
            const framework = await vscode.window.showQuickPick(
                ['pytest', 'unittest', 'jest', 'mocha', 'auto-detect'],
                { placeHolder: 'Select test framework' }
            );

            if (!framework) return;

            const result = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Squad: Generating tests...',
                cancellable: false
            }, async () => {
                return await this.bridge.generateTests(
                    code,
                    context,
                    framework === 'auto-detect' ? undefined : framework
                );
            });

            // Create new file with tests
            const testFileName = this.getTestFileName(editor.document.fileName, framework);
            const doc = await vscode.workspace.openTextDocument({
                content: result.response,
                language: editor.document.languageId
            });

            await vscode.window.showTextDocument(doc);
            vscode.window.showInformationMessage(`Tests generated! Save as: ${testFileName}`);

        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Add documentation
     */
    async addDocumentation() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('Please select code to document');
            return;
        }

        try {
            const code = editor.document.getText(selection);
            const context = await this.extractor.extractContext(editor);

            const result = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Squad: Generating documentation...',
                cancellable: false
            }, async () => {
                return await this.bridge.addDocumentation(code, context);
            });

            // Insert documentation above selection
            await editor.edit(editBuilder => {
                const line = selection.start.line;
                const insertPosition = new vscode.Position(line, 0);
                const indent = this.getIndentation(editor.document, line);
                const formattedDoc = this.formatDocumentation(result.response, indent);

                editBuilder.insert(insertPosition, formattedDoc + '\n');
            });

            vscode.window.showInformationMessage('Documentation added!');

        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Suggest refactorings
     */
    async refactorCode() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('Please select code to refactor');
            return;
        }

        try {
            const code = editor.document.getText(selection);
            const context = await this.extractor.extractContext(editor);

            const result = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Squad: Analyzing code...',
                cancellable: false
            }, async () => {
                return await this.bridge.suggestRefactorings(code, context);
            });

            await this.showResponse('Refactoring Suggestions', result);

        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Review changes (git diff)
     */
    async reviewChanges() {
        try {
            // Get git diff
            const terminal = vscode.window.createTerminal({ name: 'Squad Git', hideFromUser: true });
            // TODO: Properly capture git diff output
            // For now, show input box for diff

            const diff = await vscode.window.showInputBox({
                prompt: 'Paste git diff or changes to review',
                placeHolder: 'diff --git a/file.py b/file.py...',
                ignoreFocusOut: true
            });

            if (!diff) return;

            const editor = vscode.window.activeTextEditor;
            const context = editor ? await this.extractor.extractContext(editor) : {};

            const result = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Squad: Reviewing changes...',
                cancellable: false
            }, async () => {
                return await this.bridge.reviewChanges(diff, context);
            });

            await this.showResponse('Code Review', result);

        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Index workspace
     */
    async indexWorkspace() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders || workspaceFolders.length === 0) {
            vscode.window.showErrorMessage('No workspace folder open');
            return;
        }

        try {
            const workspacePath = workspaceFolders[0].uri.fsPath;

            const result = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Squad: Indexing workspace...',
                cancellable: false
            }, async (progress) => {
                progress.report({ message: 'Scanning files...' });

                const stats = await this.bridge.indexWorkspace(workspacePath);

                progress.report({ message: 'Completed!' });
                return stats;
            });

            vscode.window.showInformationMessage(
                `Indexed ${result.filesIndexed} files with ${result.entitiesFound} entities in ${(result.duration / 1000).toFixed(1)}s`
            );

        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Clear cache
     */
    async clearCache() {
        const choice = await vscode.window.showWarningMessage(
            'Clear all Squad cache?',
            { modal: true },
            'Clear All',
            'Clear Old (>7 days)',
            'Cancel'
        );

        if (choice === 'Clear All') {
            this.cache.clear();
            vscode.window.showInformationMessage('Cache cleared!');
        } else if (choice === 'Clear Old (>7 days)') {
            this.cache.clearOlderThan(7);
            vscode.window.showInformationMessage('Old cache entries cleared!');
        }
    }

    /**
     * Show statistics
     */
    async showStats() {
        try {
            const [cacheStats, serverStats] = await Promise.all([
                this.cache.getStats(),
                this.bridge.getStats()
            ]);

            const message = `
**Squad Statistics**

**Cache:**
- Entries: ${cacheStats.totalEntries} / ${cacheStats.maxSize}
- Hit Rate: ${cacheStats.hitRate}%
- Size: ${(cacheStats.size / 1024 / 1024).toFixed(2)} MB

**Server:**
- Queries: ${serverStats.total_queries || 0}
- Success Rate: ${serverStats.success_rate || 100}%
- Avg Latency: ${serverStats.avg_latency_ms || 0}ms
            `;

            const doc = await vscode.workspace.openTextDocument({
                content: message,
                language: 'markdown'
            });

            await vscode.window.showTextDocument(doc);

        } catch (error) {
            this.handleError(error);
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    private async showResponse(title: string, response: AgenticResponse) {
        const content = `# ${title}\n\n${response.response}\n\n---\n**Confidence:** ${(response.confidence * 100).toFixed(1)}%\n**Mode:** ${response.reasoning_mode}\n**Duration:** ${response.total_duration_ms.toFixed(0)}ms`;

        const doc = await vscode.workspace.openTextDocument({
            content,
            language: 'markdown'
        });

        await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    }

    private getTestFileName(fileName: string, framework: string): string {
        const path = require('path');
        const baseName = path.basename(fileName, path.extname(fileName));
        const ext = path.extname(fileName);

        if (framework.includes('jest') || framework.includes('mocha')) {
            return `${baseName}.test${ext}`;
        } else if (framework.includes('pytest') || framework.includes('unittest')) {
            return `test_${baseName}${ext}`;
        } else {
            return `${baseName}_test${ext}`;
        }
    }

    private getIndentation(document: vscode.TextDocument, line: number): string {
        const lineText = document.lineAt(line).text;
        const match = lineText.match(/^(\s*)/);
        return match ? match[1] : '';
    }

    private formatDocumentation(doc: string, indent: string): string {
        return doc.split('\n').map(line => indent + line).join('\n');
    }

    private hashCode(str: string): string {
        const crypto = require('crypto');
        return crypto.createHash('sha256').update(str).digest('hex').substring(0, 16);
    }

    private handleError(error: any) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`Squad Error: ${message}`);
        this.outputChannel.appendLine(`[ERROR] ${new Date().toISOString()}: ${message}`);
        this.outputChannel.show();
    }

    dispose() {
        this.outputChannel.dispose();
    }
}
