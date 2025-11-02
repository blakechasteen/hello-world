/**
 * Agent Panel
 * Webview panel for Squad agent interaction and reasoning visualization
 */

import * as vscode from 'vscode';
import { HoloLoomBridge, AgenticResult } from './HoloLoomBridge';

export class AgentPanel {
    private panel: vscode.WebviewPanel | undefined;
    private bridge: HoloLoomBridge;
    private extensionUri: vscode.Uri;

    constructor(extensionUri: vscode.Uri, bridge: HoloLoomBridge) {
        this.extensionUri = extensionUri;
        this.bridge = bridge;
    }

    show() {
        if (this.panel) {
            this.panel.reveal(vscode.ViewColumn.Beside);
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'squadAgent',
            'Squad Agent',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        this.panel.webview.html = this.getWebviewContent();

        this.panel.webview.onDidReceiveMessage(
            async message => {
                if (message.command === 'query') {
                    await this.handleQuery(message.text, message.mode);
                }
            }
        );

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });
    }

    displayResult(result: AgenticResult) {
        if (!this.panel) return;

        this.panel.webview.postMessage({
            command: 'displayResult',
            result
        });
    }

    private async handleQuery(text: string, mode: string) {
        try {
            this.panel?.webview.postMessage({ command: 'queryStart', text });
            const result = await this.bridge.query(text, undefined, mode);
            this.displayResult(result);
        } catch (error) {
            this.panel?.webview.postMessage({
                command: 'error',
                error: String(error)
            });
        }
    }

    private getWebviewContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Squad Agent</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
        }
        .query-input {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        #queryText {
            flex: 1;
            padding: 8px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            border-radius: 4px;
        }
        select, button {
            padding: 8px 16px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .result {
            margin: 20px 0;
            padding: 15px;
            background: var(--vscode-editor-inactiveSelectionBackground);
            border-radius: 8px;
            border-left: 4px solid var(--vscode-textLink-foreground);
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .confidence {
            color: var(--vscode-testing-iconPassed);
        }
        .confidence.low {
            color: var(--vscode-testing-iconFailed);
        }
        .steps {
            margin-top: 10px;
            padding: 10px;
            background: var(--vscode-editor-background);
            border-radius: 4px;
        }
        .step {
            margin: 5px 0;
            padding: 8px;
            background: var(--vscode-input-background);
            border-radius: 4px;
            font-size: 0.9em;
        }
        .step-type {
            font-weight: bold;
            color: var(--vscode-symbolIcon-functionForeground);
        }
        .thinking {
            text-align: center;
            padding: 20px;
            color: var(--vscode-descriptionForeground);
        }
    </style>
</head>
<body>
    <h1>🤖 Squad Agent</h1>
    <div class="query-input">
        <input type="text" id="queryText" placeholder="Ask Squad anything..." />
        <select id="modeSelect">
            <option value="direct">Direct</option>
            <option value="verify" selected>Verify</option>
            <option value="research">Research</option>
            <option value="plan_execute">Plan & Execute</option>
        </select>
        <button onclick="sendQuery()">Ask</button>
    </div>
    <div id="thinking" class="thinking" style="display: none;">Squad is thinking...</div>
    <div id="results"></div>

    <script>
        const vscode = acquireVsCodeApi();

        function sendQuery() {
            const text = document.getElementById('queryText').value;
            const mode = document.getElementById('modeSelect').value;
            if (!text) return;

            vscode.postMessage({ command: 'query', text: text, mode: mode });
            document.getElementById('queryText').value = '';
        }

        document.getElementById('queryText').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendQuery();
        });

        window.addEventListener('message', event => {
            const message = event.data;

            if (message.command === 'queryStart') {
                document.getElementById('thinking').style.display = 'block';
            } else if (message.command === 'displayResult') {
                document.getElementById('thinking').style.display = 'none';
                displayResult(message.result);
            } else if (message.command === 'error') {
                document.getElementById('thinking').style.display = 'none';
                alert('Error: ' + message.error);
            }
        });

        function displayResult(result) {
            const resultsDiv = document.getElementById('results');
            const resultElement = document.createElement('div');
            resultElement.className = 'result';

            const confidenceClass = result.confidence >= 0.7 ? 'confidence' : 'confidence low';

            let html = \`
                <div class="result-header">
                    <span>Mode: \${result.reasoning_mode}</span>
                    <span class="\${confidenceClass}">Confidence: \${(result.confidence * 100).toFixed(0)}%</span>
                </div>
                <div class="response">
                    <strong>Response:</strong>
                    <p>\${result.response || 'Processing...'}</p>
                </div>
            \`;

            if (result.steps_taken && result.steps_taken.length > 0) {
                html += '<div class="steps"><strong>Reasoning Steps:</strong>';
                result.steps_taken.forEach(step => {
                    html += \`
                        <div class="step">
                            <span class="step-type">\${step.type}</span>
                            \${step.query ? '<div>Query: ' + step.query + '</div>' : ''}
                            \${step.confidence !== undefined ? '<div>Confidence: ' + (step.confidence * 100).toFixed(0) + '%</div>' : ''}
                        </div>
                    \`;
                });
                html += '</div>';
            }

            html += \`
                <div style="margin-top: 10px; font-size: 0.8em; color: var(--vscode-descriptionForeground);">
                    Queries: \${result.total_queries} | Duration: \${result.total_duration_ms.toFixed(0)}ms
                </div>
            \`;

            resultElement.innerHTML = html;
            resultsDiv.insertBefore(resultElement, resultsDiv.firstChild);
        }
    </script>
</body>
</html>`;
    }
}
