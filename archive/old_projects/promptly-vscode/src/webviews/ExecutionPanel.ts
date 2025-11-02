import * as vscode from 'vscode';
import { ExecutionClient, ExecutionEvent, ExecutionStatus } from '../api/ExecutionClient';

export class ExecutionPanel {
    private static currentPanel: ExecutionPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private readonly extensionUri: vscode.Uri;
    private disposables: vscode.Disposable[] = [];
    private executionClient: ExecutionClient;

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri, client: ExecutionClient) {
        this.panel = panel;
        this.extensionUri = extensionUri;
        this.executionClient = client;

        // Set the webview's initial html content
        this.update();

        // Listen for when the panel is disposed
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);

        // Handle messages from the webview
        this.panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'executeSkill':
                        this.handleExecuteSkill(message.data);
                        break;
                    case 'executeChain':
                        this.handleExecuteChain(message.data);
                        break;
                    case 'executeLoop':
                        this.handleExecuteLoop(message.data);
                        break;
                    case 'getStatus':
                        this.handleGetStatus(message.executionId);
                        break;
                }
            },
            null,
            this.disposables
        );

        // Connect WebSocket for real-time updates
        this.executionClient.connectWebSocket();
    }

    public static createOrShow(extensionUri: vscode.Uri, client: ExecutionClient) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        // If we already have a panel, show it
        if (ExecutionPanel.currentPanel) {
            ExecutionPanel.currentPanel.panel.reveal(column);
            return;
        }

        // Otherwise, create a new panel
        const panel = vscode.window.createWebviewPanel(
            'promptlyExecution',
            'Promptly Execution',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
                retainContextWhenHidden: true
            }
        );

        ExecutionPanel.currentPanel = new ExecutionPanel(panel, extensionUri, client);
    }

    private async handleExecuteSkill(data: any) {
        try {
            const response = await this.executionClient.executeSkill(data);

            // Register event handler for this execution
            this.executionClient.onExecutionEvent(response.execution_id, (event) => {
                this.panel.webview.postMessage({
                    type: 'executionEvent',
                    event: event
                });
            });

            // Send initial response
            this.panel.webview.postMessage({
                type: 'executionStarted',
                execution: response
            });

        } catch (error: any) {
            this.panel.webview.postMessage({
                type: 'executionError',
                error: error.message
            });
        }
    }

    private async handleExecuteChain(data: any) {
        try {
            const response = await this.executionClient.executeChain(data);

            this.executionClient.onExecutionEvent(response.execution_id, (event) => {
                this.panel.webview.postMessage({
                    type: 'executionEvent',
                    event: event
                });
            });

            this.panel.webview.postMessage({
                type: 'executionStarted',
                execution: response
            });

        } catch (error: any) {
            this.panel.webview.postMessage({
                type: 'executionError',
                error: error.message
            });
        }
    }

    private async handleExecuteLoop(data: any) {
        try {
            const response = await this.executionClient.executeLoop(data);

            this.executionClient.onExecutionEvent(response.execution_id, (event) => {
                this.panel.webview.postMessage({
                    type: 'executionEvent',
                    event: event
                });
            });

            this.panel.webview.postMessage({
                type: 'executionStarted',
                execution: response
            });

        } catch (error: any) {
            this.panel.webview.postMessage({
                type: 'executionError',
                error: error.message
            });
        }
    }

    private async handleGetStatus(executionId: string) {
        try {
            const status = await this.executionClient.getStatus(executionId);
            this.panel.webview.postMessage({
                type: 'statusUpdate',
                status: status
            });
        } catch (error: any) {
            this.panel.webview.postMessage({
                type: 'statusError',
                error: error.message
            });
        }
    }

    public dispose() {
        ExecutionPanel.currentPanel = undefined;

        // Disconnect WebSocket
        this.executionClient.disconnectWebSocket();

        // Clean up resources
        this.panel.dispose();

        while (this.disposables.length) {
            const disposable = this.disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }

    private update() {
        const webview = this.panel.webview;
        this.panel.webview.html = this.getHtmlForWebview(webview);
    }

    private getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Promptly Execution</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        h2 {
            margin-bottom: 20px;
            color: var(--vscode-editor-foreground);
        }

        .mode-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .mode-button {
            flex: 1;
            padding: 15px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            cursor: pointer;
            border-radius: 4px;
            text-align: center;
            transition: all 0.2s;
        }

        .mode-button:hover {
            background: var(--vscode-button-hoverBackground);
        }

        .mode-button.active {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border-color: var(--vscode-focusBorder);
        }

        .mode-icon {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .mode-name {
            font-weight: bold;
        }

        .execution-form {
            display: none;
            margin-bottom: 20px;
        }

        .execution-form.active {
            display: block;
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }

        input[type="text"],
        textarea,
        select {
            width: 100%;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border-radius: 4px;
            font-family: var(--vscode-font-family);
        }

        textarea {
            min-height: 120px;
            resize: vertical;
        }

        .chain-builder {
            border: 1px solid var(--vscode-input-border);
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
        }

        .chain-skill {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 8px;
        }

        .chain-skill input {
            flex: 1;
        }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }

        .btn-primary {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }

        .btn-primary:hover {
            background: var(--vscode-button-hoverBackground);
        }

        .btn-secondary {
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
        }

        .btn-danger {
            background: #e74c3c;
            color: white;
        }

        .loop-type-selector {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 15px;
        }

        .loop-type {
            padding: 12px;
            border: 1px solid var(--vscode-input-border);
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .loop-type:hover {
            background: var(--vscode-list-hoverBackground);
        }

        .loop-type.selected {
            border-color: var(--vscode-focusBorder);
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }

        .slider-group {
            margin-bottom: 15px;
        }

        .slider-value {
            float: right;
            color: var(--vscode-descriptionForeground);
        }

        input[type="range"] {
            width: 100%;
        }

        .execution-status {
            display: none;
            border: 1px solid var(--vscode-input-border);
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }

        .execution-status.active {
            display: block;
        }

        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-indicator.running {
            background: #3498db;
            animation: pulse 1.5s infinite;
        }

        .status-indicator.completed {
            background: #2ecc71;
        }

        .status-indicator.failed {
            background: #e74c3c;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--vscode-progressBar-background);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 10px;
        }

        .progress-fill {
            height: 100%;
            background: var(--vscode-button-background);
            transition: width 0.3s;
        }

        .output-box {
            background: var(--vscode-editor-background);
            border: 1px solid var(--vscode-input-border);
            padding: 12px;
            border-radius: 4px;
            margin-top: 10px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: var(--vscode-editor-font-family);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 15px;
        }

        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            background: var(--vscode-input-background);
            border-radius: 4px;
        }

        .stat-label {
            color: var(--vscode-descriptionForeground);
        }

        .stat-value {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>🔮 Promptly Execution</h2>

        <div class="mode-selector">
            <div class="mode-button active" data-mode="skill">
                <div class="mode-icon">⚡</div>
                <div class="mode-name">Skill</div>
            </div>
            <div class="mode-button" data-mode="chain">
                <div class="mode-icon">🔗</div>
                <div class="mode-name">Chain</div>
            </div>
            <div class="mode-button" data-mode="loop">
                <div class="mode-icon">🔄</div>
                <div class="mode-name">Loop</div>
            </div>
        </div>

        <!-- Skill Execution Form -->
        <div id="skill-form" class="execution-form active">
            <div class="form-group">
                <label>Skill Name</label>
                <input type="text" id="skill-name" placeholder="Enter skill name...">
            </div>
            <div class="form-group">
                <label>User Input</label>
                <textarea id="skill-input" placeholder="Enter your input here..."></textarea>
            </div>
            <button class="btn btn-primary" onclick="executeSkill()">▶ Execute Skill</button>
        </div>

        <!-- Chain Execution Form -->
        <div id="chain-form" class="execution-form">
            <div class="form-group">
                <label>Chain Builder</label>
                <div class="chain-builder" id="chain-builder">
                    <div class="chain-skill">
                        <span>1.</span>
                        <input type="text" placeholder="Skill name..." class="chain-skill-input">
                        <button class="btn btn-danger btn-sm" onclick="removeChainSkill(this)">✕</button>
                    </div>
                </div>
                <button class="btn btn-secondary" onclick="addChainSkill()">+ Add Skill</button>
            </div>
            <div class="form-group">
                <label>Initial Input</label>
                <textarea id="chain-input" placeholder="Enter initial input..."></textarea>
            </div>
            <button class="btn btn-primary" onclick="executeChain()">▶ Execute Chain</button>
        </div>

        <!-- Loop Execution Form -->
        <div id="loop-form" class="execution-form">
            <div class="form-group">
                <label>Skill Name</label>
                <input type="text" id="loop-skill" placeholder="Enter skill name...">
            </div>
            <div class="form-group">
                <label>Loop Type</label>
                <div class="loop-type-selector">
                    <div class="loop-type selected" data-type="refine">
                        <strong>⚡ Refine</strong><br>
                        <small>Iterative improvement</small>
                    </div>
                    <div class="loop-type" data-type="critique">
                        <strong>🔍 Critique</strong><br>
                        <small>Self-evaluation</small>
                    </div>
                    <div class="loop-type" data-type="decompose">
                        <strong>🧩 Decompose</strong><br>
                        <small>Divide and conquer</small>
                    </div>
                    <div class="loop-type" data-type="verify">
                        <strong>✓ Verify</strong><br>
                        <small>Generate → verify → improve</small>
                    </div>
                    <div class="loop-type" data-type="explore">
                        <strong>🌟 Explore</strong><br>
                        <small>Multiple approaches</small>
                    </div>
                    <div class="loop-type" data-type="hofstadter">
                        <strong>∞ Hofstadter</strong><br>
                        <small>Meta-level thinking</small>
                    </div>
                </div>
            </div>
            <div class="slider-group">
                <label>
                    Max Iterations
                    <span class="slider-value" id="max-iterations-value">5</span>
                </label>
                <input type="range" id="max-iterations" min="1" max="10" value="5"
                    oninput="document.getElementById('max-iterations-value').textContent = this.value">
            </div>
            <div class="slider-group">
                <label>
                    Quality Threshold
                    <span class="slider-value" id="quality-threshold-value">0.90</span>
                </label>
                <input type="range" id="quality-threshold" min="0.5" max="1.0" step="0.05" value="0.9"
                    oninput="document.getElementById('quality-threshold-value').textContent = parseFloat(this.value).toFixed(2)">
            </div>
            <div class="form-group">
                <label>User Input</label>
                <textarea id="loop-input" placeholder="Enter your input here..."></textarea>
            </div>
            <button class="btn btn-primary" onclick="executeLoop()">▶ Execute Loop</button>
        </div>

        <!-- Execution Status -->
        <div id="execution-status" class="execution-status">
            <div class="status-header">
                <div>
                    <span class="status-indicator" id="status-indicator"></span>
                    <strong id="status-text">Queued</strong>
                </div>
                <small id="execution-id"></small>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
            </div>
            <div id="current-step">Initializing...</div>

            <div class="stats-grid" id="stats-grid" style="display: none;">
                <div class="stat-item">
                    <span class="stat-label">Progress:</span>
                    <span class="stat-value" id="stat-progress">0%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Status:</span>
                    <span class="stat-value" id="stat-status">-</span>
                </div>
                <div class="stat-item" id="stat-iterations-item" style="display: none;">
                    <span class="stat-label">Iterations:</span>
                    <span class="stat-value" id="stat-iterations">-</span>
                </div>
                <div class="stat-item" id="stat-quality-item" style="display: none;">
                    <span class="stat-label">Quality:</span>
                    <span class="stat-value" id="stat-quality">-</span>
                </div>
            </div>

            <div class="output-box" id="output-box" style="display: none;"></div>
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let currentExecutionId = null;
        let selectedLoopType = 'refine';

        // Mode switching
        document.querySelectorAll('.mode-button').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;

                // Update button states
                document.querySelectorAll('.mode-button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Update form visibility
                document.querySelectorAll('.execution-form').forEach(f => f.classList.remove('active'));
                document.getElementById(mode + '-form').classList.add('active');
            });
        });

        // Loop type selection
        document.querySelectorAll('.loop-type').forEach(type => {
            type.addEventListener('click', () => {
                document.querySelectorAll('.loop-type').forEach(t => t.classList.remove('selected'));
                type.classList.add('selected');
                selectedLoopType = type.dataset.type;
            });
        });

        // Chain builder functions
        function addChainSkill() {
            const builder = document.getElementById('chain-builder');
            const skillCount = builder.querySelectorAll('.chain-skill').length + 1;
            const skillDiv = document.createElement('div');
            skillDiv.className = 'chain-skill';
            skillDiv.innerHTML = \`
                <span>\${skillCount}.</span>
                <input type="text" placeholder="Skill name..." class="chain-skill-input">
                <button class="btn btn-danger btn-sm" onclick="removeChainSkill(this)">✕</button>
            \`;
            builder.appendChild(skillDiv);
        }

        function removeChainSkill(btn) {
            const builder = document.getElementById('chain-builder');
            if (builder.querySelectorAll('.chain-skill').length > 1) {
                btn.parentElement.remove();
                // Renumber skills
                builder.querySelectorAll('.chain-skill').forEach((skill, i) => {
                    skill.querySelector('span').textContent = (i + 1) + '.';
                });
            }
        }

        // Execution functions
        function executeSkill() {
            const skillName = document.getElementById('skill-name').value;
            const userInput = document.getElementById('skill-input').value;

            if (!skillName || !userInput) {
                alert('Please fill in all fields');
                return;
            }

            vscode.postMessage({
                command: 'executeSkill',
                data: {
                    skill_name: skillName,
                    user_input: userInput
                }
            });

            showExecutionStatus();
        }

        function executeChain() {
            const skillInputs = Array.from(document.querySelectorAll('.chain-skill-input'));
            const skillNames = skillInputs.map(input => input.value).filter(v => v.trim());
            const initialInput = document.getElementById('chain-input').value;

            if (skillNames.length === 0 || !initialInput) {
                alert('Please fill in all fields');
                return;
            }

            vscode.postMessage({
                command: 'executeChain',
                data: {
                    skill_names: skillNames,
                    initial_input: initialInput
                }
            });

            showExecutionStatus();
        }

        function executeLoop() {
            const skillName = document.getElementById('loop-skill').value;
            const userInput = document.getElementById('loop-input').value;
            const maxIterations = parseInt(document.getElementById('max-iterations').value);
            const qualityThreshold = parseFloat(document.getElementById('quality-threshold').value);

            if (!skillName || !userInput) {
                alert('Please fill in all fields');
                return;
            }

            vscode.postMessage({
                command: 'executeLoop',
                data: {
                    skill_name: skillName,
                    user_input: userInput,
                    loop_type: selectedLoopType,
                    max_iterations: maxIterations,
                    quality_threshold: qualityThreshold
                }
            });

            showExecutionStatus();
        }

        function showExecutionStatus() {
            document.getElementById('execution-status').classList.add('active');
            document.getElementById('stats-grid').style.display = 'grid';
            document.getElementById('status-indicator').className = 'status-indicator running';
            document.getElementById('status-text').textContent = 'Running';
        }

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;

            switch (message.type) {
                case 'executionStarted':
                    currentExecutionId = message.execution.execution_id;
                    document.getElementById('execution-id').textContent = 'ID: ' + currentExecutionId;
                    updateStatus({
                        status: 'running',
                        progress: 0,
                        current_step: 'Starting execution...'
                    });
                    break;

                case 'executionEvent':
                    handleExecutionEvent(message.event);
                    break;

                case 'executionError':
                    document.getElementById('status-indicator').className = 'status-indicator failed';
                    document.getElementById('status-text').textContent = 'Error';
                    document.getElementById('current-step').textContent = message.error;
                    break;
            }
        });

        function handleExecutionEvent(event) {
            if (event.event === 'status_update') {
                updateStatus({
                    progress: event.progress,
                    current_step: event.step
                });
            } else if (event.event === 'iteration_update') {
                document.getElementById('stat-iterations-item').style.display = 'flex';
                document.getElementById('stat-quality-item').style.display = 'flex';
                document.getElementById('stat-iterations').textContent = event.iteration;
                document.getElementById('stat-quality').textContent = event.quality?.toFixed(2);
                updateStatus({
                    progress: event.progress
                });
            } else if (event.event === 'completed') {
                updateStatus({
                    status: 'completed',
                    progress: 1,
                    current_step: 'Complete!',
                    output: event.output
                });
                document.getElementById('status-indicator').className = 'status-indicator completed';
                document.getElementById('status-text').textContent = 'Completed';
            } else if (event.event === 'failed') {
                document.getElementById('status-indicator').className = 'status-indicator failed';
                document.getElementById('status-text').textContent = 'Failed';
                document.getElementById('current-step').textContent = 'Error: ' + event.error;
            }
        }

        function updateStatus(status) {
            if (status.progress !== undefined) {
                document.getElementById('progress-fill').style.width = (status.progress * 100) + '%';
                document.getElementById('stat-progress').textContent = Math.round(status.progress * 100) + '%';
            }
            if (status.current_step) {
                document.getElementById('current-step').textContent = status.current_step;
            }
            if (status.status) {
                document.getElementById('stat-status').textContent = status.status;
            }
            if (status.output) {
                const outputBox = document.getElementById('output-box');
                outputBox.style.display = 'block';
                outputBox.textContent = status.output;
            }
        }
    </script>
</body>
</html>`;
    }
}
