/**
 * EdWINManager.ts
 * ===============
 *
 * EdWIN 💡 - The Research Companion
 *
 * Features:
 * - Elle AR guidance (context-aware help)
 * - GraphRAG navigation
 * - Voice chat integration
 * - Scene analysis
 *
 * Created: 2025-01-20
 */

import * as vscode from 'vscode';
import axios, { AxiosInstance } from 'axios';

interface EdWINConfig {
    backendUrl: string;
    guidance: {
        autoTrigger: boolean;
        complexity: 'simple' | 'moderate' | 'complex';
    };
    graph: {
        autoVisualize: boolean;
        maxDepth: number;
    };
}

interface SceneAnalysis {
    complexity: string;
    focus_areas: string[];
    entities: Array<{ name: string; type: string; line: number }>;
    relationships: any[];
    suggestions: string[];
}

interface GuideResponse {
    guidance: string;
    suggested_actions: Array<{
        action: string;
        description: string;
        confidence: number;
    }>;
    focus_highlights: Array<{
        line: number;
        type: string;
        description: string;
    }>;
    related_concepts: string[];
    confidence: number;
}

export class EdWINManager {
    private client: AxiosInstance;
    private config: EdWINConfig;
    private outputChannel: vscode.OutputChannel;
    private statusBarItem: vscode.StatusBarItem;
    private decorationType: vscode.TextEditorDecorationType;

    constructor(config: EdWINConfig) {
        this.config = config;
        this.client = axios.create({
            baseURL: config.backendUrl,
            timeout: 30000
        });

        this.outputChannel = vscode.window.createOutputChannel('EdWIN 💡');
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 98);
        this.statusBarItem.text = '💡 EdWIN: Ready';
        this.statusBarItem.show();

        this.decorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('editor.findMatchHighlightBackground'),
            borderRadius: '3px'
        });
    }

    async analyzeScene(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        try {
            this.statusBarItem.text = '💡 EdWIN: Analyzing...';

            const response = await this.client.post<SceneAnalysis>('/elle/scene', {
                code: editor.document.getText(),
                language: editor.document.languageId,
                file_name: editor.document.fileName
            });

            this.showScenePanel(response.data);
            this.statusBarItem.text = '💡 EdWIN: Ready';

        } catch (error: any) {
            this.handleError('analyzeScene', error);
        }
    }

    async getGuidance(intent: string = 'seeking_guidance'): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        try {
            this.statusBarItem.text = '💡 EdWIN: Getting guidance...';

            const response = await this.client.post<GuideResponse>('/elle/guide', {
                scene: {
                    code: editor.document.getText(),
                    language: editor.document.languageId,
                    file_name: editor.document.fileName
                },
                intent
            });

            this.showGuidancePanel(response.data);
            this.highlightFocusAreas(editor, response.data.focus_highlights);

            this.statusBarItem.text = '💡 EdWIN: Ready';

        } catch (error: any) {
            this.handleError('getGuidance', error);
        }
    }

    async exploreGraph(): Promise<void> {
        try {
            const entity = await vscode.window.showInputBox({
                prompt: 'Enter entity to explore',
                placeHolder: 'e.g., MyClass'
            });

            if (!entity) return;

            this.statusBarItem.text = '💡 EdWIN: Exploring graph...';

            const response = await this.client.get(`/graph/relationships/${entity}`, {
                params: { max_depth: this.config.graph.maxDepth }
            });

            this.showGraphPanel(response.data);
            this.statusBarItem.text = '💡 EdWIN: Ready';

        } catch (error: any) {
            this.handleError('exploreGraph', error);
        }
    }

    private highlightFocusAreas(editor: vscode.TextEditor, highlights: any[]): void {
        const ranges = highlights.map(h => {
            const line = Math.max(0, h.line - 1);
            const position = new vscode.Position(line, 0);
            return new vscode.Range(position, position.translate(0, 100));
        });

        editor.setDecorations(this.decorationType, ranges);

        setTimeout(() => {
            editor.setDecorations(this.decorationType, []);
        }, 5000);
    }

    private showScenePanel(data: SceneAnalysis): void {
        const panel = vscode.window.createWebviewPanel(
            'edwinScene',
            'EdWIN: Scene Analysis',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    .complexity { display: inline-block; padding: 5px 10px; background: #ffa500; border-radius: 3px; }
                    .entity { margin: 5px 0; padding: 8px; background: #f0f0f0; border-left: 3px solid #007acc; }
                    .suggestion { padding: 8px; margin: 5px 0; background: #e8f4f8; }
                </style>
            </head>
            <body>
                <h1>💡 EdWIN Scene Analysis</h1>
                <p><span class="complexity">${data.complexity}</span></p>

                <h2>Focus Areas</h2>
                <ul>${data.focus_areas.map(f => `<li>${f}</li>`).join('')}</ul>

                <h2>Entities (${data.entities.length})</h2>
                ${data.entities.map(e => `
                    <div class="entity">
                        <strong>${e.type}</strong>: ${e.name} (line ${e.line})
                    </div>
                `).join('')}

                <h2>Suggestions</h2>
                ${data.suggestions.map(s => `<div class="suggestion">💡 ${s}</div>`).join('')}
            </body>
            </html>
        `;
    }

    private showGuidancePanel(data: GuideResponse): void {
        const panel = vscode.window.createWebviewPanel(
            'edwinGuidance',
            'EdWIN: Guidance',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    .guidance { background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                    .action { padding: 10px; margin: 5px 0; background: #f0f0f0; border-left: 3px solid #28a745; cursor: pointer; }
                    .concept { display: inline-block; margin: 5px; padding: 5px 10px; background: #e0e0e0; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1>💡 EdWIN Guidance</h1>
                <p>Confidence: ${(data.confidence * 100).toFixed(0)}%</p>

                <div class="guidance">
                    <p>${data.guidance.replace(/\n/g, '<br>')}</p>
                </div>

                <h2>Suggested Actions</h2>
                ${data.suggested_actions.map(a => `
                    <div class="action">
                        <strong>${a.action}</strong> (${(a.confidence * 100).toFixed(0)}%)<br>
                        ${a.description}
                    </div>
                `).join('')}

                <h2>Related Concepts</h2>
                ${data.related_concepts.map(c => `<span class="concept">${c}</span>`).join('')}
            </body>
            </html>
        `;
    }

    private showGraphPanel(data: any): void {
        const panel = vscode.window.createWebviewPanel(
            'edwinGraph',
            'EdWIN: Knowledge Graph',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    .relationship { margin: 10px 0; padding: 10px; background: #f0f0f0; border-left: 3px solid #007acc; }
                </style>
            </head>
            <body>
                <h1>💡 EdWIN Knowledge Graph</h1>
                <p><strong>Entity:</strong> ${data.entity}</p>
                <p><strong>Relationships:</strong> ${data.total || 0}</p>

                <h2>Related Entities</h2>
                ${(data.relationships || []).map((r: any) => `
                    <div class="relationship">
                        ${r.from} <strong>→ ${r.type} →</strong> ${r.to}
                    </div>
                `).join('')}
            </body>
            </html>
        `;
    }

    private handleError(method: string, error: any): void {
        const message = error.response?.data?.detail || error.message || 'Unknown error';
        this.outputChannel.appendLine(`✗ Error in ${method}: ${message}`);
        vscode.window.showErrorMessage(`EdWIN: ${message}`);
        this.statusBarItem.text = '💡 EdWIN: Error';
        setTimeout(() => { this.statusBarItem.text = '💡 EdWIN: Ready'; }, 3000);
    }

    dispose(): void {
        this.outputChannel.dispose();
        this.statusBarItem.dispose();
        this.decorationType.dispose();
    }
}
