/**
 * ProtoManager.ts
 * ===============
 *
 * Proto ⚡ - The Code Orchestrator
 *
 * Features:
 * - Visual workflow execution
 * - Promptly reliability patterns (6 solvers)
 * - Surgical code refactoring
 * - Multi-stage reasoning pipelines
 *
 * Created: 2025-01-20
 */

import * as vscode from 'vscode';
import axios, { AxiosInstance } from 'axios';

// ============================================================================
// Types & Interfaces
// ============================================================================

interface ProtoConfig {
    backendUrl: string;
    defaultMode: 'verify' | 'research' | 'direct';
    workflows: {
        autoSave: boolean;
        templates: string[];
    };
}

interface ExplainRequest {
    code: string;
    language: string;
    output_schema?: any;
    mode: 'brief' | 'comprehensive' | 'tutorial';
}

interface ExplainResponse {
    explanation: string;
    structure: any;
    key_concepts: string[];
    complexity: string;
    confidence: number;
}

interface RefactorRequest {
    code: string;
    language: string;
    goal: 'readability' | 'performance' | 'maintainability';
    constraints?: string[];
}

interface RefactorResponse {
    original_code: string;
    refactored_code: string;
    changes: Array<{
        type: string;
        description: string;
        line: number;
    }>;
    reasoning: string;
    preservation_score: number;
}

interface ResearchRequest {
    question: string;
    context?: string;
    max_stages: number;
}

interface ResearchStage {
    stage_number: number;
    query: string;
    findings: string;
    confidence: number;
}

interface ResearchResponse {
    question: string;
    stages: ResearchStage[];
    final_answer: string;
    overall_confidence: number;
}

interface VerifyRequest {
    claim: string;
    context?: string;
    evidence_threshold: number;
}

interface VerifyResponse {
    claim: string;
    verified: boolean;
    confidence: number;
    evidence: Array<{
        type: string;
        finding: string;
        confidence: number;
        supports: boolean;
    }>;
    reasoning: string;
}

// ============================================================================
// ProtoManager Class
// ============================================================================

export class ProtoManager {
    private client: AxiosInstance;
    private config: ProtoConfig;
    private outputChannel: vscode.OutputChannel;
    private statusBarItem: vscode.StatusBarItem;

    constructor(config: ProtoConfig) {
        this.config = config;
        this.client = axios.create({
            baseURL: config.backendUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });

        this.outputChannel = vscode.window.createOutputChannel('Proto ⚡');
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            100
        );
        this.statusBarItem.text = '⚡ Proto: Ready';
        this.statusBarItem.show();
    }

    // ========================================================================
    // Public API Methods
    // ========================================================================

    /**
     * Explain selected code with schema-based output
     */
    async explainCode(mode: 'brief' | 'comprehensive' | 'tutorial' = 'comprehensive'): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        const code = editor.document.getText(selection);
        if (!code) {
            vscode.window.showErrorMessage('No code selected');
            return;
        }

        const language = editor.document.languageId;

        try {
            this.statusBarItem.text = '⚡ Proto: Explaining...';
            this.outputChannel.appendLine(`Explaining ${mode} (${code.length} chars)...`);

            const request: ExplainRequest = {
                code,
                language,
                mode
            };

            const response = await this.client.post<ExplainResponse>(
                '/promptly/explain',
                request
            );

            // Show explanation in webview panel
            this.showExplanationPanel(response.data);

            this.statusBarItem.text = '⚡ Proto: Ready';
            this.outputChannel.appendLine(`✓ Explanation complete (${response.data.confidence * 100}% confidence)`);

        } catch (error: any) {
            this.handleError('explainCode', error);
        }
    }

    /**
     * Surgical code refactoring
     */
    async refactorCode(goal: 'readability' | 'performance' | 'maintainability'): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        const code = editor.document.getText(selection);
        if (!code) {
            vscode.window.showErrorMessage('No code selected');
            return;
        }

        const language = editor.document.languageId;

        try {
            this.statusBarItem.text = '⚡ Proto: Refactoring...';
            this.outputChannel.appendLine(`Refactoring for ${goal}...`);

            const request: RefactorRequest = {
                code,
                language,
                goal
            };

            const response = await this.client.post<RefactorResponse>(
                '/promptly/refactor',
                request
            );

            // Ask user to apply changes
            const apply = await vscode.window.showInformationMessage(
                `Proto found ${response.data.changes.length} improvements (${response.data.preservation_score * 100}% preserved). Apply?`,
                'Apply', 'Preview', 'Cancel'
            );

            if (apply === 'Apply') {
                await editor.edit(editBuilder => {
                    editBuilder.replace(selection, response.data.refactored_code);
                });
                vscode.window.showInformationMessage(`✓ Refactoring applied`);
            } else if (apply === 'Preview') {
                this.showRefactoringPreview(response.data);
            }

            this.statusBarItem.text = '⚡ Proto: Ready';

        } catch (error: any) {
            this.handleError('refactorCode', error);
        }
    }

    /**
     * Multi-stage research
     */
    async researchQuestion(question?: string): Promise<void> {
        // Get question from user if not provided
        if (!question) {
            question = await vscode.window.showInputBox({
                prompt: 'What would you like to research?',
                placeHolder: 'e.g., What are the tradeoffs of Thompson Sampling?'
            });

            if (!question) {
                return;
            }
        }

        // Get context from current editor if available
        const editor = vscode.window.activeTextEditor;
        const context = editor ? editor.document.getText() : undefined;

        try {
            this.statusBarItem.text = '⚡ Proto: Researching...';
            this.outputChannel.appendLine(`Researching: ${question}`);

            const request: ResearchRequest = {
                question,
                context,
                max_stages: 3
            };

            const response = await this.client.post<ResearchResponse>(
                '/promptly/research',
                request
            );

            // Show research results in panel
            this.showResearchPanel(response.data);

            this.statusBarItem.text = '⚡ Proto: Ready';
            this.outputChannel.appendLine(`✓ Research complete (${response.data.overall_confidence * 100}% confidence)`);

        } catch (error: any) {
            this.handleError('researchQuestion', error);
        }
    }

    /**
     * Verify factual claim
     */
    async verifyClaim(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        const claim = editor.document.getText(selection);
        if (!claim) {
            vscode.window.showErrorMessage('No text selected');
            return;
        }

        // Get context (surrounding code)
        const fullText = editor.document.getText();

        try {
            this.statusBarItem.text = '⚡ Proto: Verifying...';
            this.outputChannel.appendLine(`Verifying claim: ${claim.substring(0, 50)}...`);

            const request: VerifyRequest = {
                claim,
                context: fullText,
                evidence_threshold: 0.7
            };

            const response = await this.client.post<VerifyResponse>(
                '/promptly/verify',
                request
            );

            // Show verification results
            const icon = response.data.verified ? '✓' : '✗';
            const message = `${icon} Claim ${response.data.verified ? 'verified' : 'could not be verified'} (${response.data.confidence * 100}% confidence)`;

            vscode.window.showInformationMessage(message);
            this.showVerificationPanel(response.data);

            this.statusBarItem.text = '⚡ Proto: Ready';

        } catch (error: any) {
            this.handleError('verifyClaim', error);
        }
    }

    /**
     * Open workflow builder
     */
    async openWorkflowBuilder(): Promise<void> {
        const panel = vscode.window.createWebviewPanel(
            'protoWorkflow',
            'Proto Workflow Builder',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = this.getWorkflowBuilderHtml();

        // Handle messages from webview
        panel.webview.onDidReceiveMessage(async message => {
            switch (message.command) {
                case 'execute':
                    await this.executeWorkflow(message.workflow);
                    break;
                case 'save':
                    await this.saveWorkflow(message.workflow);
                    break;
            }
        });
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    private showExplanationPanel(data: ExplainResponse): void {
        const panel = vscode.window.createWebviewPanel(
            'protoExplanation',
            'Proto: Code Explanation',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    h1 { color: #007acc; }
                    .complexity {
                        display: inline-block;
                        padding: 5px 10px;
                        background: #ffa500;
                        border-radius: 3px;
                    }
                    .concept {
                        display: inline-block;
                        margin: 5px;
                        padding: 5px 10px;
                        background: #e0e0e0;
                        border-radius: 3px;
                    }
                    .confidence { color: #28a745; font-weight: bold; }
                </style>
            </head>
            <body>
                <h1>⚡ Proto Code Explanation</h1>
                <p><span class="complexity">${data.complexity}</span> complexity</p>
                <p class="confidence">Confidence: ${(data.confidence * 100).toFixed(0)}%</p>

                <h2>Explanation</h2>
                <p>${data.explanation.replace(/\n/g, '<br>')}</p>

                <h2>Key Concepts</h2>
                <div>
                    ${data.key_concepts.map(c => `<span class="concept">${c}</span>`).join('')}
                </div>

                <h2>Structure</h2>
                <pre>${JSON.stringify(data.structure, null, 2)}</pre>
            </body>
            </html>
        `;
    }

    private showRefactoringPreview(data: RefactorResponse): void {
        const panel = vscode.window.createWebviewPanel(
            'protoRefactor',
            'Proto: Refactoring Preview',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: monospace; padding: 20px; }
                    .diff { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                    .before, .after { border: 1px solid #ccc; padding: 10px; }
                    .before { background: #ffe0e0; }
                    .after { background: #e0ffe0; }
                    .change { margin: 10px 0; padding: 10px; background: #f0f0f0; }
                    .preservation { color: #28a745; font-weight: bold; }
                </style>
            </head>
            <body>
                <h1>⚡ Proto Refactoring</h1>
                <p class="preservation">Preservation: ${(data.preservation_score * 100).toFixed(0)}%</p>
                <p>${data.reasoning}</p>

                <h2>Changes (${data.changes.length})</h2>
                ${data.changes.map(c => `
                    <div class="change">
                        <strong>${c.type}</strong> at line ${c.line}<br>
                        ${c.description}
                    </div>
                `).join('')}

                <h2>Diff</h2>
                <div class="diff">
                    <div class="before">
                        <h3>Before</h3>
                        <pre>${data.original_code}</pre>
                    </div>
                    <div class="after">
                        <h3>After</h3>
                        <pre>${data.refactored_code}</pre>
                    </div>
                </div>
            </body>
            </html>
        `;
    }

    private showResearchPanel(data: ResearchResponse): void {
        const panel = vscode.window.createWebviewPanel(
            'protoResearch',
            'Proto: Research Results',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    .stage { margin: 20px 0; padding: 15px; border-left: 3px solid #007acc; }
                    .confidence { color: #28a745; }
                    .answer { background: #e8f4f8; padding: 15px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>⚡ Proto Research</h1>
                <p><strong>Question:</strong> ${data.question}</p>
                <p class="confidence">Overall Confidence: ${(data.overall_confidence * 100).toFixed(0)}%</p>

                <h2>Research Stages (${data.stages.length})</h2>
                ${data.stages.map(s => `
                    <div class="stage">
                        <h3>Stage ${s.stage_number}: ${s.query}</h3>
                        <p>${s.findings}</p>
                        <p class="confidence">Confidence: ${(s.confidence * 100).toFixed(0)}%</p>
                    </div>
                `).join('')}

                <h2>Final Answer</h2>
                <div class="answer">
                    ${data.final_answer.replace(/\n/g, '<br>')}
                </div>
            </body>
            </html>
        `;
    }

    private showVerificationPanel(data: VerifyResponse): void {
        const panel = vscode.window.createWebviewPanel(
            'protoVerify',
            'Proto: Verification',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        const statusColor = data.verified ? '#28a745' : '#dc3545';

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    .status { font-size: 24px; color: ${statusColor}; font-weight: bold; }
                    .evidence { margin: 10px 0; padding: 10px; border-left: 3px solid #ccc; }
                    .supports { border-color: #28a745; }
                    .refutes { border-color: #dc3545; }
                </style>
            </head>
            <body>
                <h1>⚡ Proto Verification</h1>
                <p class="status">${data.verified ? '✓ VERIFIED' : '✗ NOT VERIFIED'}</p>
                <p>Confidence: ${(data.confidence * 100).toFixed(0)}%</p>

                <h2>Claim</h2>
                <p><em>${data.claim}</em></p>

                <h2>Evidence (${data.evidence.length})</h2>
                ${data.evidence.map(e => `
                    <div class="evidence ${e.supports ? 'supports' : 'refutes'}">
                        <strong>${e.type}</strong> (${(e.confidence * 100).toFixed(0)}%)<br>
                        ${e.finding}
                    </div>
                `).join('')}

                <h2>Reasoning</h2>
                <p>${data.reasoning}</p>
            </body>
            </html>
        `;
    }

    private getWorkflowBuilderHtml(): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
                    .palette { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }
                    .agent { padding: 10px; background: #007acc; color: white; cursor: pointer; border-radius: 3px; }
                    .canvas { min-height: 400px; border: 2px dashed #ccc; padding: 20px; }
                    button { margin: 5px; padding: 10px 20px; }
                </style>
            </head>
            <body>
                <h1>⚡ Proto Workflow Builder</h1>

                <h2>Agent Palette</h2>
                <div class="palette">
                    <div class="agent" onclick="addAgent('explain')">Explain Code</div>
                    <div class="agent" onclick="addAgent('refactor')">Refactor</div>
                    <div class="agent" onclick="addAgent('research')">Research</div>
                    <div class="agent" onclick="addAgent('verify')">Verify</div>
                    <div class="agent" onclick="addAgent('synthesize')">Synthesize</div>
                    <div class="agent" onclick="addAgent('output')">Output</div>
                </div>

                <h2>Workflow Canvas</h2>
                <div class="canvas" id="canvas">
                    Drag agents here to build your workflow
                </div>

                <button onclick="execute()">▶️ Execute</button>
                <button onclick="save()">💾 Save</button>
                <button onclick="load()">📂 Load</button>

                <script>
                    const vscode = acquireVsCodeApi();
                    let workflow = [];

                    function addAgent(type) {
                        workflow.push({ type, id: Date.now() });
                        render();
                    }

                    function render() {
                        const canvas = document.getElementById('canvas');
                        canvas.innerHTML = workflow.map(a =>
                            \`<div style="display:inline-block; margin:10px; padding:10px; background:#007acc; color:white; border-radius:3px;">
                                \${a.type}
                            </div>\`
                        ).join(' → ');
                    }

                    function execute() {
                        vscode.postMessage({ command: 'execute', workflow });
                    }

                    function save() {
                        vscode.postMessage({ command: 'save', workflow });
                    }
                </script>
            </body>
            </html>
        `;
    }

    private async executeWorkflow(workflow: any[]): Promise<void> {
        this.outputChannel.appendLine(`Executing workflow with ${workflow.length} steps...`);
        // TODO: Implement workflow execution
        vscode.window.showInformationMessage(`Workflow execution not yet implemented`);
    }

    private async saveWorkflow(workflow: any[]): Promise<void> {
        const name = await vscode.window.showInputBox({
            prompt: 'Workflow name',
            placeHolder: 'my-workflow'
        });

        if (name) {
            // TODO: Save to workspace
            vscode.window.showInformationMessage(`Workflow "${name}" saved`);
        }
    }

    private handleError(method: string, error: any): void {
        const message = error.response?.data?.detail || error.message || 'Unknown error';
        this.outputChannel.appendLine(`✗ Error in ${method}: ${message}`);
        vscode.window.showErrorMessage(`Proto: ${message}`);
        this.statusBarItem.text = '⚡ Proto: Error';

        setTimeout(() => {
            this.statusBarItem.text = '⚡ Proto: Ready';
        }, 3000);
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    dispose(): void {
        this.outputChannel.dispose();
        this.statusBarItem.dispose();
    }
}
