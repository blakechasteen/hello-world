/**
 * Agent Panel
 * Interactive webview showing Squad's reasoning process
 */

import * as vscode from 'vscode';
import { AgenticResult } from './HoloLoomBridge';

export class AgentPanel {
    private panel: vscode.WebviewPanel | undefined;
    private extensionUri: vscode.Uri;

    constructor(extensionUri: vscode.Uri) {
        this.extensionUri = extensionUri;
    }

    show() {
        if (this.panel) {
            this.panel.reveal();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'squadAgentPanel',
            'Squad Reasoning',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        this.panel.webview.html = this.getWebviewContent();

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });
    }

    displayResult(result: AgenticResult) {
        if (!this.panel) {
            this.show();
        }

        this.panel?.webview.postMessage({
            type: 'result',
            data: result
        });
    }

    private getWebviewContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Squad Reasoning</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: var(--vscode-textLink-foreground);
        }
        .result-container {
            margin: 20px 0;
            padding: 15px;
            background-color: var(--vscode-editor-inactiveSelectionBackground);
            border-radius: 5px;
        }
        .step {
            margin: 10px 0;
            padding: 10px;
            background-color: var(--vscode-input-background);
            border-left: 3px solid var(--vscode-textLink-foreground);
        }
        .confidence {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .confidence-high { background-color: #28a745; color: white; }
        .confidence-medium { background-color: #ffc107; color: black; }
        .confidence-low { background-color: #dc3545; color: white; }
        .response {
            margin: 15px 0;
            padding: 15px;
            background-color: var(--vscode-textBlockQuote-background);
            border-left: 4px solid var(--vscode-textLink-foreground);
            white-space: pre-wrap;
        }
        .metadata {
            font-size: 0.9em;
            color: var(--vscode-descriptionForeground);
            margin-top: 10px;
        }
        .verification {
            margin: 15px 0;
            padding: 15px;
            background-color: var(--vscode-inputValidation-infoBackground);
            border-radius: 5px;
        }
        .contradictions {
            color: var(--vscode-inputValidation-errorForeground);
        }
        .evidence {
            color: var(--vscode-inputValidation-infoForeground);
        }
    </style>
</head>
<body>
    <h1>🤖 Squad Agent Panel</h1>
    <div id="content">
        <p style="color: var(--vscode-descriptionForeground);">
            Waiting for query results...
        </p>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'result') {
                displayResult(message.data);
            }
        });

        function displayResult(result) {
            const content = document.getElementById('content');

            const confidenceClass = result.confidence >= 0.8 ? 'confidence-high' :
                                  result.confidence >= 0.5 ? 'confidence-medium' :
                                  'confidence-low';

            let html = \`
                <div class="result-container">
                    <h2>Response</h2>
                    <div class="response">\${escapeHtml(result.response)}</div>

                    <div class="metadata">
                        <strong>Confidence:</strong>
                        <span class="confidence \${confidenceClass}">
                            \${(result.confidence * 100).toFixed(1)}%
                        </span>
                        <br>
                        <strong>Mode:</strong> \${result.reasoning_mode}
                        <br>
                        <strong>Duration:</strong> \${result.total_duration_ms.toFixed(1)}ms
                        <br>
                        <strong>Total Queries:</strong> \${result.total_queries}
                    </div>

                    <h3>Reasoning Steps</h3>
            \`;

            for (const step of result.steps_taken) {
                const stepConf = step.confidence ?
                    \`<span class="confidence \${step.confidence >= 0.8 ? 'confidence-high' :
                        step.confidence >= 0.5 ? 'confidence-medium' : 'confidence-low'}">
                        \${(step.confidence * 100).toFixed(1)}%
                    </span>\` : '';

                html += \`
                    <div class="step">
                        <strong>\${step.type}</strong> \${stepConf}
                        \${step.query ? '<br>Query: ' + escapeHtml(step.query) : ''}
                        \${step.finding ? '<br>Finding: ' + escapeHtml(step.finding) : ''}
                    </div>
                \`;
            }

            if (result.verification) {
                html += \`
                    <h3>Verification</h3>
                    <div class="verification">
                        <strong>Verified:</strong> \${result.verification.verified ? '✅ Yes' : '❌ No'}
                        <br>
                        <strong>Verification Confidence:</strong> \${(result.verification.confidence * 100).toFixed(1)}%
                \`;

                if (result.verification.contradictions.length > 0) {
                    html += \`<br><br><strong class="contradictions">Contradictions:</strong><ul>\`;
                    for (const c of result.verification.contradictions) {
                        html += \`<li class="contradictions">\${escapeHtml(c)}</li>\`;
                    }
                    html += \`</ul>\`;
                }

                if (result.verification.supporting_evidence.length > 0) {
                    html += \`<br><strong class="evidence">Supporting Evidence:</strong><ul>\`;
                    for (const e of result.verification.supporting_evidence) {
                        html += \`<li class="evidence">\${escapeHtml(e)}</li>\`;
                    }
                    html += \`</ul>\`;
                }

                html += \`</div>\`;
            }

            html += \`</div>\`;

            content.innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>`;
    }
}
