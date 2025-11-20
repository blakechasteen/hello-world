import * as vscode from 'vscode';
import { HoloLoomCommands } from '../commands/hololoomCommands';

export class HoloLoomSidebarProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'promptly.hololoomSidebar';

    private _view?: vscode.WebviewView;
    private hololoomCommands: HoloLoomCommands;

    constructor(
        private readonly _extensionUri: vscode.Uri,
    ) {
        this.hololoomCommands = new HoloLoomCommands();
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'search':
                    await this.handleSearch(data.query);
                    break;
                case 'capture':
                    await this.handleCapture(data.content);
                    break;
                case 'getTodaysNotes':
                    await this.getTodaysNotes();
                    break;
            }
        });

        // Load today's notes on initialization
        this.getTodaysNotes();
    }

    private async handleSearch(query: string) {
        if (!query) return;

        try {
            const result = await this.hololoomCommands.recall(query);

            if (this._view) {
                this._view.webview.postMessage({
                    type: 'searchResults',
                    results: result
                });
            }
        } catch (error: any) {
            vscode.window.showErrorMessage(`Search failed: ${error.message}`);
        }
    }

    private async handleCapture(content: string) {
        if (!content) return;

        try {
            const result = await this.hololoomCommands.remember(content);

            if (this._view) {
                this._view.webview.postMessage({
                    type: 'captureSuccess',
                    message: 'Note captured successfully!'
                });
            }

            // Refresh today's notes
            await this.getTodaysNotes();

            vscode.window.showInformationMessage('Note captured to HoloLoom!');
        } catch (error: any) {
            vscode.window.showErrorMessage(`Capture failed: ${error.message}`);
        }
    }

    private async getTodaysNotes() {
        try {
            const today = new Date().toISOString().split('T')[0];
            const result = await this.hololoomCommands.recall(`notes from ${today}`);

            if (this._view) {
                this._view.webview.postMessage({
                    type: 'todaysNotes',
                    notes: result
                });
            }
        } catch (error: any) {
            // Silently fail - server might not be running yet
            console.error('Failed to load today\'s notes:', error);
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HoloLoom</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    font-size: var(--vscode-font-size);
                    color: var(--vscode-foreground);
                    padding: 16px;
                }

                h2 {
                    margin-top: 0;
                    font-size: 18px;
                    font-weight: 600;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    padding-bottom: 8px;
                }

                h3 {
                    font-size: 14px;
                    font-weight: 600;
                    margin-top: 20px;
                    margin-bottom: 8px;
                }

                section {
                    margin-bottom: 24px;
                }

                input[type="text"], textarea {
                    width: 100%;
                    padding: 8px;
                    margin-bottom: 8px;
                    background: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 2px;
                    font-family: inherit;
                    font-size: inherit;
                }

                textarea {
                    min-height: 80px;
                    resize: vertical;
                }

                button {
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    padding: 8px 16px;
                    cursor: pointer;
                    border-radius: 2px;
                    font-size: inherit;
                }

                button:hover {
                    background: var(--vscode-button-hoverBackground);
                }

                button:active {
                    opacity: 0.8;
                }

                .note-item {
                    padding: 8px;
                    margin-bottom: 8px;
                    background: var(--vscode-editor-background);
                    border-left: 3px solid var(--vscode-textLink-foreground);
                    border-radius: 2px;
                }

                .note-content {
                    font-size: 13px;
                    margin-bottom: 4px;
                }

                .note-meta {
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                }

                .search-result {
                    padding: 8px;
                    margin-bottom: 8px;
                    background: var(--vscode-editor-background);
                    border-radius: 2px;
                }

                .confidence-badge {
                    display: inline-block;
                    padding: 2px 6px;
                    background: var(--vscode-badge-background);
                    color: var(--vscode-badge-foreground);
                    font-size: 10px;
                    border-radius: 2px;
                    margin-left: 4px;
                }

                .empty-state {
                    text-align: center;
                    padding: 20px;
                    color: var(--vscode-descriptionForeground);
                    font-size: 12px;
                }

                .status-message {
                    padding: 8px;
                    margin-bottom: 8px;
                    border-radius: 2px;
                    font-size: 12px;
                }

                .status-success {
                    background: rgba(0, 128, 0, 0.2);
                    color: var(--vscode-terminal-ansiGreen);
                }

                .status-error {
                    background: rgba(255, 0, 0, 0.2);
                    color: var(--vscode-errorForeground);
                }
            </style>
        </head>
        <body>
            <h2>🧠 HoloLoom</h2>

            <!-- Quick Capture -->
            <section id="capture-section">
                <h3>✨ Quick Capture</h3>
                <textarea id="capture-input" placeholder="Capture a thought, decision, or note..."></textarea>
                <button onclick="captureNote()">💾 Remember</button>
                <div id="capture-status"></div>
            </section>

            <!-- Today's Notes -->
            <section id="today-section">
                <h3>📝 Today's Notes</h3>
                <div id="todays-notes">
                    <div class="empty-state">Loading...</div>
                </div>
            </section>

            <!-- Search -->
            <section id="search-section">
                <h3>🔍 Search Memory</h3>
                <input type="text" id="search-input" placeholder="What did I learn about...?" />
                <button onclick="searchMemory()">Search</button>
                <div id="search-results"></div>
            </section>

            <script>
                const vscode = acquireVsCodeApi();

                // Capture note
                function captureNote() {
                    const content = document.getElementById('capture-input').value.trim();
                    if (!content) return;

                    vscode.postMessage({
                        type: 'capture',
                        content: content
                    });
                }

                // Search memory
                function searchMemory() {
                    const query = document.getElementById('search-input').value.trim();
                    if (!query) return;

                    document.getElementById('search-results').innerHTML =
                        '<div class="empty-state">Searching...</div>';

                    vscode.postMessage({
                        type: 'search',
                        query: query
                    });
                }

                // Handle Enter key
                document.getElementById('search-input').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') searchMemory();
                });

                // Handle messages from extension
                window.addEventListener('message', event => {
                    const message = event.data;

                    switch (message.type) {
                        case 'todaysNotes':
                            displayTodaysNotes(message.notes);
                            break;
                        case 'searchResults':
                            displaySearchResults(message.results);
                            break;
                        case 'captureSuccess':
                            showCaptureSuccess(message.message);
                            break;
                    }
                });

                function displayTodaysNotes(notes) {
                    const container = document.getElementById('todays-notes');

                    if (!notes || notes.length === 0) {
                        container.innerHTML = '<div class="empty-state">No notes yet today</div>';
                        return;
                    }

                    // Parse if it's a markdown string (from recall command)
                    let notesList = [];
                    if (typeof notes === 'string') {
                        // Simple parsing of markdown results
                        const matches = notes.match(/\\*\\*\\d+\\.\\*\\* (.+?)\\n/g);
                        if (matches) {
                            notesList = matches.map(m => ({
                                content: m.replace(/\\*\\*\\d+\\.\\*\\* /, '').trim()
                            }));
                        }
                    } else {
                        notesList = notes;
                    }

                    if (notesList.length === 0) {
                        container.innerHTML = '<div class="empty-state">No notes yet today</div>';
                        return;
                    }

                    container.innerHTML = notesList.map((note, i) => \`
                        <div class="note-item">
                            <div class="note-content">\${note.content || note.text || 'No content'}</div>
                            <div class="note-meta">Note #\${i + 1}</div>
                        </div>
                    \`).join('');
                }

                function displaySearchResults(results) {
                    const container = document.getElementById('search-results');

                    if (!results || results.length === 0) {
                        container.innerHTML = '<div class="empty-state">No results found</div>';
                        return;
                    }

                    // Parse if it's a markdown string
                    let resultsList = [];
                    if (typeof results === 'string') {
                        const matches = results.match(/\\*\\*\\d+\\.\\*\\* (.+?)\\n\\s+_(.+?)_/gs);
                        if (matches) {
                            resultsList = matches.map(m => {
                                const parts = m.match(/\\*\\*\\d+\\.\\*\\* (.+?)\\n\\s+_Confidence: (\\d+)%/);
                                return {
                                    content: parts ? parts[1] : m,
                                    confidence: parts ? parseInt(parts[2]) / 100 : 0.5
                                };
                            });
                        }
                    } else {
                        resultsList = results;
                    }

                    container.innerHTML = resultsList.map((result, i) => \`
                        <div class="search-result">
                            <div class="note-content">
                                \${result.content || result.text || 'No content'}
                                <span class="confidence-badge">\${Math.round((result.confidence || 0.5) * 100)}%</span>
                            </div>
                            <div class="note-meta">
                                \${result.source || result.timestamp || \`Result #\${i + 1}\`}
                            </div>
                        </div>
                    \`).join('');
                }

                function showCaptureSuccess(message) {
                    const status = document.getElementById('capture-status');
                    status.innerHTML = \`<div class="status-message status-success">\${message}</div>\`;

                    // Clear input
                    document.getElementById('capture-input').value = '';

                    // Clear status after 3 seconds
                    setTimeout(() => {
                        status.innerHTML = '';
                    }, 3000);
                }

                // Request today's notes on load
                vscode.postMessage({ type: 'getTodaysNotes' });
            </script>
        </body>
        </html>`;
    }
}
