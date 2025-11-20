import * as vscode from 'vscode';
import axios from 'axios';

/**
 * Knowledge Graph Visualization Provider
 *
 * Displays HoloLoom's knowledge graph as an interactive force-directed network.
 *
 * Features:
 * - D3.js force-directed layout (via HoloLoom's visualization system)
 * - Click nodes to see details
 * - Search and filter nodes
 * - Highlight recent/active nodes
 * - Export to HTML/PNG
 *
 * Architecture:
 * 1. Fetches graph HTML from HoloLoom API (`GET /api/graph/html`)
 * 2. Displays in webview panel
 * 3. Supports interactive features via message passing
 */
export class GraphViewProvider {
    private panel: vscode.WebviewPanel | undefined;
    private config = vscode.workspace.getConfiguration('promptly');

    constructor(private context: vscode.ExtensionContext) {}

    public async show() {
        const column = vscode.ViewColumn.Two;

        if (this.panel) {
            // Panel already exists, reveal it
            this.panel.reveal(column);
        } else {
            // Create new panel
            this.panel = vscode.window.createWebviewPanel(
                'hololoomGraph',
                'HoloLoom Knowledge Graph',
                column,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true,
                    localResourceRoots: [this.context.extensionUri]
                }
            );

            // Handle panel disposal
            this.panel.onDidDispose(() => {
                this.panel = undefined;
            });

            // Load graph
            await this.loadGraph();

            // Handle messages from webview
            this.panel.webview.onDidReceiveMessage(async (message) => {
                switch (message.type) {
                    case 'refresh':
                        await this.loadGraph();
                        break;
                    case 'export':
                        await this.exportGraph(message.format);
                        break;
                    case 'nodeClicked':
                        await this.handleNodeClick(message.nodeId);
                        break;
                    case 'search':
                        await this.handleSearch(message.query);
                        break;
                }
            });
        }
    }

    private async loadGraph(maxNodes: number = 50, highlightRecent: boolean = true) {
        if (!this.panel) return;

        try {
            // Show loading state
            this.panel.webview.html = this.getLoadingHtml();

            // Fetch graph HTML from HoloLoom
            const hololoomUrl = this.config.get<string>('hololoomUrl', 'http://localhost:8000');
            const response = await axios.get(`${hololoomUrl}/api/graph/html`, {
                params: {
                    max_nodes: maxNodes,
                    title: 'HoloLoom Knowledge Graph',
                    highlight_recent: highlightRecent
                }
            });

            // Inject message passing for interactivity
            const enhancedHtml = this.injectMessagePassing(response.data);

            // Display graph
            this.panel.webview.html = enhancedHtml;

        } catch (error: any) {
            console.error('Failed to load graph:', error);

            // Show error state
            this.panel.webview.html = this.getErrorHtml(error.message);

            vscode.window.showErrorMessage(
                `Failed to load knowledge graph: ${error.message}. Make sure HoloLoom server is running.`
            );
        }
    }

    private injectMessagePassing(originalHtml: string): string {
        // Inject VS Code API for message passing
        const vscodeScript = `
            <script>
                const vscode = acquireVsCodeApi();

                // Override click handlers to send messages to extension
                document.addEventListener('DOMContentLoaded', () => {
                    // Add refresh button
                    const controlBar = document.createElement('div');
                    controlBar.style.cssText = 'position: fixed; top: 10px; right: 10px; z-index: 1000; display: flex; gap: 10px;';

                    const refreshBtn = document.createElement('button');
                    refreshBtn.textContent = '🔄 Refresh';
                    refreshBtn.onclick = () => vscode.postMessage({ type: 'refresh' });
                    controlBar.appendChild(refreshBtn);

                    const exportBtn = document.createElement('button');
                    exportBtn.textContent = '💾 Export HTML';
                    exportBtn.onclick = () => vscode.postMessage({ type: 'export', format: 'html' });
                    controlBar.appendChild(exportBtn);

                    document.body.appendChild(controlBar);

                    // Add search box
                    const searchBox = document.createElement('div');
                    searchBox.style.cssText = 'position: fixed; top: 10px; left: 10px; z-index: 1000;';
                    searchBox.innerHTML = \`
                        <input type="text"
                               id="graph-search"
                               placeholder="Search nodes..."
                               style="padding: 5px; border-radius: 4px; border: 1px solid #ccc;"
                        />
                    \`;
                    document.body.appendChild(searchBox);

                    const searchInput = document.getElementById('graph-search');
                    searchInput?.addEventListener('input', (e) => {
                        const query = (e.target as HTMLInputElement).value;
                        vscode.postMessage({ type: 'search', query });

                        // Basic client-side filtering
                        const circles = document.querySelectorAll('circle');
                        circles.forEach(circle => {
                            const label = circle.getAttribute('data-label') || '';
                            const matches = label.toLowerCase().includes(query.toLowerCase());
                            circle.style.opacity = matches || query === '' ? '1.0' : '0.2';
                        });
                    });

                    // Enhance node clicks
                    const circles = document.querySelectorAll('circle');
                    circles.forEach(circle => {
                        circle.addEventListener('click', (e) => {
                            const nodeId = circle.getAttribute('data-id');
                            if (nodeId) {
                                vscode.postMessage({ type: 'nodeClicked', nodeId });
                            }
                        });
                    });
                });
            </script>
        `;

        // Inject before closing </body> tag
        return originalHtml.replace('</body>', `${vscodeScript}</body>`);
    }

    private async handleNodeClick(nodeId: string) {
        // Show details panel
        const panel = vscode.window.createWebviewPanel(
            'nodeDetails',
            `Node: ${nodeId}`,
            vscode.ViewColumn.Three,
            { enableScripts: true }
        );

        try {
            // Fetch node details from HoloLoom
            const hololoomUrl = this.config.get<string>('hololoomUrl', 'http://localhost:8000');
            const response = await axios.post(`${hololoomUrl}/api/recall`, {
                query: nodeId,
                k: 5
            });

            const memories = response.data.memories || [];

            // Display details
            panel.webview.html = this.getNodeDetailsHtml(nodeId, memories);

        } catch (error: any) {
            console.error('Failed to fetch node details:', error);
            panel.webview.html = this.getErrorHtml(error.message);
        }
    }

    private async handleSearch(query: string) {
        // Search is handled client-side for now
        // Could enhance with server-side search in the future
        console.log('Search query:', query);
    }

    private async exportGraph(format: 'html' | 'png') {
        if (!this.panel) return;

        try {
            if (format === 'html') {
                // Get current HTML content
                const hololoomUrl = this.config.get<string>('hololoomUrl', 'http://localhost:8000');
                const response = await axios.get(`${hololoomUrl}/api/graph/html`);

                // Ask user for save location
                const uri = await vscode.window.showSaveDialog({
                    filters: { 'HTML Files': ['html'] },
                    defaultUri: vscode.Uri.file('knowledge-graph.html')
                });

                if (uri) {
                    // Write to file
                    const fs = require('fs');
                    fs.writeFileSync(uri.fsPath, response.data);

                    vscode.window.showInformationMessage(`Graph exported to ${uri.fsPath}`);
                }
            } else if (format === 'png') {
                vscode.window.showInformationMessage(
                    'PNG export not yet implemented. Use browser screenshot for now.'
                );
            }

        } catch (error: any) {
            console.error('Export failed:', error);
            vscode.window.showErrorMessage(`Export failed: ${error.message}`);
        }
    }

    private getLoadingHtml(): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    }
                    .spinner {
                        border: 4px solid rgba(0, 0, 0, 0.1);
                        border-left-color: #007acc;
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        animation: spin 1s linear infinite;
                    }
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                </style>
            </head>
            <body>
                <div>
                    <div class="spinner"></div>
                    <p style="text-align: center; margin-top: 20px;">Loading knowledge graph...</p>
                </div>
            </body>
            </html>
        `;
    }

    private getErrorHtml(message: string): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        padding: 20px;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    }
                    .error {
                        background: #fee;
                        border: 1px solid #fcc;
                        border-radius: 4px;
                        padding: 15px;
                        color: #c00;
                    }
                    button {
                        margin-top: 10px;
                        padding: 8px 16px;
                        background: #007acc;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                </style>
            </head>
            <body>
                <div class="error">
                    <h3>❌ Failed to Load Graph</h3>
                    <p>${message}</p>
                    <button onclick="location.reload()">Retry</button>
                </div>
                <h4>Troubleshooting:</h4>
                <ul>
                    <li>Make sure HoloLoom server is running (port 8000)</li>
                    <li>Check your <code>promptly.hololoomUrl</code> setting</li>
                    <li>Verify network connectivity</li>
                </ul>
            </body>
            </html>
        `;
    }

    private getNodeDetailsHtml(nodeId: string, memories: any[]): string {
        const memoryList = memories.map((mem, i) => `
            <div style="margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px;">
                <strong>Memory ${i + 1}</strong> (Confidence: ${(mem.confidence * 100).toFixed(1)}%)
                <p>${mem.text}</p>
                ${mem.metadata ? `<small>Source: ${mem.metadata.source || 'unknown'}</small>` : ''}
            </div>
        `).join('');

        return `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        padding: 20px;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    }
                    h2 {
                        margin-top: 0;
                        color: #007acc;
                    }
                </style>
            </head>
            <body>
                <h2>🧠 ${nodeId}</h2>
                <h3>Related Memories:</h3>
                ${memories.length > 0 ? memoryList : '<p>No memories found for this node.</p>'}
            </body>
            </html>
        `;
    }
}
