import * as vscode from 'vscode';
import { HoloLoomClient, QueryResponse, ReasoningMode } from '../api/holoLoomClient';
import { getEditorContext, formatContextForPrompt } from '../utils/context';
import { getConfig } from '../utils/config';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  mode?: ReasoningMode;
  confidence?: number;
  sources?: Array<{ id: string; content: string; relevance: number }>;
}

export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'hololoom.chatView';

  private view?: vscode.WebviewView;
  private messages: Message[] = [];
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
        case 'query':
          await this.handleQuery(data.text, data.mode);
          break;
        case 'cancel':
          this.client.cancelRequest();
          this.setLoading(false);
          break;
        case 'clear':
          this.clearMessages();
          break;
        case 'getMessages':
          this.sendMessages();
          break;
      }
    });
  }

  /**
   * Send a query with optional context
   */
  public async sendQuery(text: string, mode?: ReasoningMode): Promise<void> {
    await this.handleQuery(text, mode);
  }

  /**
   * Send a query with editor context
   */
  public async sendQueryWithContext(
    text: string,
    mode?: ReasoningMode
  ): Promise<void> {
    const context = getEditorContext();
    let fullQuery = text;

    if (context) {
      const contextStr = formatContextForPrompt(context);
      fullQuery = `${text}\n\n${contextStr}`;
    }

    await this.handleQuery(fullQuery, mode);
  }

  /**
   * Handle a query request
   */
  private async handleQuery(text: string, mode?: ReasoningMode): Promise<void> {
    if (this.isLoading) {
      return;
    }

    const config = getConfig();
    const effectiveMode = mode || config.defaultReasoningMode;

    // Add user message
    const userMessage: Message = {
      id: this.generateId(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
      mode: effectiveMode,
    };
    this.addMessage(userMessage);

    // Set loading state
    this.setLoading(true);

    try {
      // Get editor context if enabled
      let context;
      if (config.includeFileContext) {
        const editorContext = getEditorContext();
        if (editorContext) {
          context = {
            fileName: editorContext.file,
            languageId: editorContext.language,
            selection: editorContext.selection,
            cursorPosition: editorContext.lineNumber != null
              ? { line: editorContext.lineNumber, column: 0 }
              : undefined,
          };
        }
      }

      // Send query to HoloLoom
      const response = await this.client.query({
        text,
        mode: effectiveMode,
        context,
      });

      // Add assistant message
      const assistantMessage: Message = {
        id: this.generateId(),
        role: 'assistant',
        content: response.response,
        timestamp: Date.now(),
        mode: response.mode,
        confidence: response.confidence,
        sources: response.sources?.map((s) => ({ id: s.id, content: s.text, relevance: s.relevance })),
      };
      this.addMessage(assistantMessage);
    } catch (error) {
      // Add error message
      const errorMessage: Message = {
        id: this.generateId(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: Date.now(),
      };
      this.addMessage(errorMessage);
    } finally {
      this.setLoading(false);
    }
  }

  /**
   * Add a message and update the webview
   */
  private addMessage(message: Message): void {
    this.messages.push(message);
    this.sendMessages();
  }

  /**
   * Clear all messages
   */
  private clearMessages(): void {
    this.messages = [];
    this.sendMessages();
  }

  /**
   * Send messages to the webview
   */
  private sendMessages(): void {
    this.view?.webview.postMessage({
      type: 'messages',
      messages: this.messages,
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
   * Generate a unique ID
   */
  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  /**
   * Get the HTML content for the webview
   */
  private getHtmlForWebview(webview: vscode.Webview): string {
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, 'dist', 'webview', 'chat.js')
    );

    const nonce = this.getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <title>HoloLoom Chat</title>
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

    #app {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
    }

    .message {
      margin-bottom: 16px;
      padding: 12px;
      border-radius: 8px;
    }

    .message.user {
      background-color: var(--vscode-input-background);
      margin-left: 24px;
    }

    .message.assistant {
      background-color: var(--vscode-editor-background);
      margin-right: 24px;
      border: 1px solid var(--vscode-panel-border);
    }

    .message-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--vscode-descriptionForeground);
    }

    .message-content {
      white-space: pre-wrap;
      line-height: 1.5;
    }

    .message-meta {
      margin-top: 8px;
      display: flex;
      gap: 12px;
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
    }

    .confidence {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .confidence-bar {
      width: 50px;
      height: 4px;
      background-color: var(--vscode-progressBar-background);
      border-radius: 2px;
      overflow: hidden;
    }

    .confidence-fill {
      height: 100%;
      background-color: var(--vscode-progressBar-foreground);
    }

    .sources {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--vscode-panel-border);
    }

    .sources-title {
      font-size: 11px;
      color: var(--vscode-descriptionForeground);
      margin-bottom: 4px;
    }

    .source {
      font-size: 12px;
      padding: 4px 8px;
      background-color: var(--vscode-badge-background);
      border-radius: 4px;
      margin-right: 4px;
      margin-bottom: 4px;
      display: inline-block;
    }

    .input-area {
      padding: 12px;
      border-top: 1px solid var(--vscode-panel-border);
      background-color: var(--vscode-sideBar-background);
    }

    .mode-selector {
      display: flex;
      gap: 4px;
      margin-bottom: 8px;
    }

    .mode-btn {
      padding: 4px 8px;
      font-size: 11px;
      border: 1px solid var(--vscode-input-border);
      background-color: transparent;
      color: var(--vscode-foreground);
      cursor: pointer;
      border-radius: 4px;
    }

    .mode-btn.active {
      background-color: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border-color: var(--vscode-button-background);
    }

    .input-row {
      display: flex;
      gap: 8px;
    }

    textarea {
      flex: 1;
      padding: 8px;
      border: 1px solid var(--vscode-input-border);
      background-color: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      border-radius: 4px;
      resize: none;
      font-family: inherit;
      font-size: inherit;
    }

    textarea:focus {
      outline: none;
      border-color: var(--vscode-focusBorder);
    }

    button.send {
      padding: 8px 16px;
      background-color: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }

    button.send:hover {
      background-color: var(--vscode-button-hoverBackground);
    }

    button.send:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .loading {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px;
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
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: var(--vscode-descriptionForeground);
      padding: 24px;
      text-align: center;
    }

    .empty-state h3 {
      margin-bottom: 8px;
    }

    .empty-state p {
      font-size: 12px;
      max-width: 300px;
    }
  </style>
</head>
<body>
  <div id="app">
    <div class="messages" id="messages">
      <div class="empty-state">
        <h3>HoloLoom Chat</h3>
        <p>Ask questions about your codebase, get explanations, or save knowledge for later.</p>
      </div>
    </div>
    <div class="loading" id="loading" style="display: none;">
      <div class="loading-spinner"></div>
      <span>Thinking...</span>
    </div>
    <div class="input-area">
      <div class="mode-selector">
        <button class="mode-btn active" data-mode="direct">Direct</button>
        <button class="mode-btn" data-mode="verify">Verify</button>
        <button class="mode-btn" data-mode="research">Research</button>
        <button class="mode-btn" data-mode="plan_execute">Plan</button>
      </div>
      <div class="input-row">
        <textarea id="input" rows="2" placeholder="Ask HoloLoom..."></textarea>
        <button class="send" id="send">Send</button>
      </div>
    </div>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    let currentMode = 'direct';
    let isLoading = false;

    // DOM elements
    const messagesEl = document.getElementById('messages');
    const loadingEl = document.getElementById('loading');
    const inputEl = document.getElementById('input');
    const sendBtn = document.getElementById('send');
    const modeBtns = document.querySelectorAll('.mode-btn');

    // Mode selection
    modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentMode = btn.dataset.mode;
      });
    });

    // Send message
    function sendMessage() {
      const text = inputEl.value.trim();
      if (!text || isLoading) return;

      vscode.postMessage({
        type: 'query',
        text,
        mode: currentMode
      });

      inputEl.value = '';
    }

    sendBtn.addEventListener('click', sendMessage);
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Handle messages from extension
    window.addEventListener('message', (event) => {
      const data = event.data;

      switch (data.type) {
        case 'messages':
          renderMessages(data.messages);
          break;
        case 'loading':
          isLoading = data.loading;
          loadingEl.style.display = data.loading ? 'flex' : 'none';
          sendBtn.disabled = data.loading;
          break;
      }
    });

    function renderMessages(messages) {
      if (messages.length === 0) {
        messagesEl.innerHTML = \`
          <div class="empty-state">
            <h3>HoloLoom Chat</h3>
            <p>Ask questions about your codebase, get explanations, or save knowledge for later.</p>
          </div>
        \`;
        return;
      }

      messagesEl.innerHTML = messages.map(msg => {
        const modeLabel = msg.mode ? \`<span class="mode">\${msg.mode}</span>\` : '';
        const confidenceBar = msg.confidence !== undefined ? \`
          <div class="confidence">
            <span>\${Math.round(msg.confidence * 100)}%</span>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: \${msg.confidence * 100}%"></div>
            </div>
          </div>
        \` : '';

        const sources = msg.sources && msg.sources.length > 0 ? \`
          <div class="sources">
            <div class="sources-title">Sources:</div>
            \${msg.sources.slice(0, 3).map(s => \`<span class="source">\${s.id}</span>\`).join('')}
          </div>
        \` : '';

        return \`
          <div class="message \${msg.role}">
            <div class="message-header">
              <span>\${msg.role === 'user' ? 'You' : 'HoloLoom'}</span>
              \${modeLabel}
            </div>
            <div class="message-content">\${escapeHtml(msg.content)}</div>
            \${msg.role === 'assistant' ? \`
              <div class="message-meta">
                \${confidenceBar}
              </div>
              \${sources}
            \` : ''}
          </div>
        \`;
      }).join('');

      // Scroll to bottom
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    // Request initial messages
    vscode.postMessage({ type: 'getMessages' });
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
