import * as vscode from 'vscode';
import { HoloLoomClient, MemoryItem } from '../api/holoLoomClient';

export class MemoryViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'hololoom.memoryView';

  private view?: vscode.WebviewView;
  private memories: MemoryItem[] = [];
  private searchQuery: string = '';
  private isLoading: boolean = false;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly client: HoloLoomClient
  ) {}

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this.view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.extensionUri],
    };

    webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);

    // Handle messages from the webview
    webviewView.webview.onDidReceiveMessage(async (data) => {
      switch (data.type) {
        case 'search':
          await this.handleSearch(data.query);
          break;
        case 'loadRecent':
          await this.loadRecentMemories();
          break;
        case 'delete':
          await this.handleDelete(data.id);
          break;
        case 'insert':
          this.insertMemory(data.content);
          break;
      }
    });

    // Load initial memories
    this.loadRecentMemories();
  }

  /**
   * Search memories
   */
  private async handleSearch(query: string): Promise<void> {
    this.searchQuery = query;
    this.setLoading(true);

    try {
      if (query.trim()) {
        this.memories = await this.client.searchMemory({ query, limit: 20 });
      } else {
        await this.loadRecentMemories();
      }
      this.sendMemories();
    } catch (error) {
      vscode.window.showErrorMessage(
        `Failed to search memories: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    } finally {
      this.setLoading(false);
    }
  }

  /**
   * Load recent memories
   */
  private async loadRecentMemories(): Promise<void> {
    this.setLoading(true);

    try {
      this.memories = await this.client.getRecentMemories(20);
      this.sendMemories();
    } catch (error) {
      // Silently fail on initial load - server might not be running
      console.log('Failed to load recent memories:', error);
      this.memories = [];
      this.sendMemories();
    } finally {
      this.setLoading(false);
    }
  }

  /**
   * Delete a memory
   */
  private async handleDelete(id: string): Promise<void> {
    try {
      await this.client.deleteMemory(id);
      this.memories = this.memories.filter((m) => m.id !== id);
      this.sendMemories();
      vscode.window.showInformationMessage('Memory deleted');
    } catch (error) {
      vscode.window.showErrorMessage(
        `Failed to delete memory: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Insert memory content into active editor
   */
  private insertMemory(content: string): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('No active editor');
      return;
    }

    editor.edit((editBuilder) => {
      editBuilder.insert(editor.selection.active, content);
    });
  }

  /**
   * Refresh memories
   */
  public async refresh(): Promise<void> {
    if (this.searchQuery) {
      await this.handleSearch(this.searchQuery);
    } else {
      await this.loadRecentMemories();
    }
  }

  /**
   * Send memories to webview
   */
  private sendMemories(): void {
    this.view?.webview.postMessage({
      type: 'memories',
      memories: this.memories,
    });
  }

  /**
   * Set loading state
   */
  private setLoading(loading: boolean): void {
    this.isLoading = loading;
    this.view?.webview.postMessage({
      type: 'loading',
      loading,
    });
  }

  /**
   * Get the HTML content for the webview
   */
  private getHtmlForWebview(webview: vscode.Webview): string {
    const nonce = this.getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <title>HoloLoom Memory</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      color: var(--vscode-foreground);
      background-color: var(--vscode-sideBar-background);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .search-area {
      padding: 12px;
      border-bottom: 1px solid var(--vscode-panel-border);
    }

    .search-input {
      width: 100%;
      padding: 8px;
      border: 1px solid var(--vscode-input-border);
      background-color: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      border-radius: 4px;
    }

    .search-input:focus {
      outline: none;
      border-color: var(--vscode-focusBorder);
    }

    .memories {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
    }

    .memory-item {
      padding: 12px;
      margin-bottom: 8px;
      background-color: var(--vscode-editor-background);
      border: 1px solid var(--vscode-panel-border);
      border-radius: 6px;
    }

    .memory-item:hover {
      border-color: var(--vscode-focusBorder);
    }

    .memory-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }

    .memory-type {
      font-size: 11px;
      padding: 2px 6px;
      background-color: var(--vscode-badge-background);
      color: var(--vscode-badge-foreground);
      border-radius: 3px;
    }

    .memory-time {
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
    }

    .memory-content {
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 100px;
      overflow: hidden;
    }

    .memory-actions {
      display: flex;
      gap: 8px;
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--vscode-panel-border);
    }

    .action-btn {
      padding: 4px 8px;
      font-size: 11px;
      border: none;
      background-color: transparent;
      color: var(--vscode-textLink-foreground);
      cursor: pointer;
      border-radius: 3px;
    }

    .action-btn:hover {
      background-color: var(--vscode-toolbar-hoverBackground);
    }

    .action-btn.delete {
      color: var(--vscode-errorForeground);
    }

    .loading {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 24px;
      color: var(--vscode-descriptionForeground);
    }

    .loading-spinner {
      width: 16px;
      height: 16px;
      border: 2px solid var(--vscode-progressBar-background);
      border-top-color: var(--vscode-progressBar-foreground);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .empty-state {
      padding: 24px;
      text-align: center;
      color: var(--vscode-descriptionForeground);
    }

    .empty-state h4 {
      margin-bottom: 8px;
    }

    .refresh-btn {
      margin-top: 8px;
      padding: 4px 12px;
      font-size: 12px;
      border: 1px solid var(--vscode-button-border);
      background-color: var(--vscode-button-secondaryBackground);
      color: var(--vscode-button-secondaryForeground);
      border-radius: 4px;
      cursor: pointer;
    }

    .refresh-btn:hover {
      background-color: var(--vscode-button-secondaryHoverBackground);
    }
  </style>
</head>
<body>
  <div class="search-area">
    <input type="text" class="search-input" id="search" placeholder="Search memories...">
  </div>
  <div class="memories" id="memories">
    <div class="loading" id="loading">
      <div class="loading-spinner"></div>
      <span>Loading...</span>
    </div>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    let searchTimeout = null;

    const memoriesEl = document.getElementById('memories');
    const loadingEl = document.getElementById('loading');
    const searchEl = document.getElementById('search');

    // Search with debounce
    searchEl.addEventListener('input', (e) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        vscode.postMessage({
          type: 'search',
          query: e.target.value
        });
      }, 300);
    });

    // Handle messages from extension
    window.addEventListener('message', (event) => {
      const data = event.data;

      switch (data.type) {
        case 'memories':
          renderMemories(data.memories);
          break;
        case 'loading':
          loadingEl.style.display = data.loading ? 'flex' : 'none';
          break;
      }
    });

    function renderMemories(memories) {
      if (memories.length === 0) {
        memoriesEl.innerHTML = \`
          <div class="empty-state">
            <h4>No memories found</h4>
            <p>Start adding memories or try a different search.</p>
            <button class="refresh-btn" onclick="refresh()">Refresh</button>
          </div>
        \`;
        return;
      }

      memoriesEl.innerHTML = memories.map(mem => \`
        <div class="memory-item">
          <div class="memory-header">
            <span class="memory-type">\${mem.type || 'memory'}</span>
            <span class="memory-time">\${formatTime(mem.timestamp)}</span>
          </div>
          <div class="memory-content">\${escapeHtml(mem.content)}</div>
          <div class="memory-actions">
            <button class="action-btn" onclick="insertMemory('\${escapeJs(mem.content)}')">Insert</button>
            <button class="action-btn" onclick="copyMemory('\${escapeJs(mem.content)}')">Copy</button>
            <button class="action-btn delete" onclick="deleteMemory('\${mem.id}')">Delete</button>
          </div>
        </div>
      \`).join('');
    }

    function formatTime(timestamp) {
      const date = new Date(timestamp);
      const now = new Date();
      const diff = now - date;

      if (diff < 60000) return 'Just now';
      if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
      if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
      return date.toLocaleDateString();
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    function escapeJs(text) {
      return text.replace(/'/g, "\\\\'").replace(/\\n/g, '\\\\n');
    }

    function insertMemory(content) {
      vscode.postMessage({ type: 'insert', content: content.replace(/\\\\n/g, '\\n') });
    }

    function copyMemory(content) {
      navigator.clipboard.writeText(content.replace(/\\\\n/g, '\\n'));
    }

    function deleteMemory(id) {
      vscode.postMessage({ type: 'delete', id });
    }

    function refresh() {
      vscode.postMessage({ type: 'loadRecent' });
    }

    // Load initial memories
    vscode.postMessage({ type: 'loadRecent' });
  </script>
</body>
</html>`;
  }

  private getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
      text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
  }
}
