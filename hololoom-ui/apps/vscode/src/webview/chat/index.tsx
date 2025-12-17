/**
 * Chat Webview Entry Point
 *
 * This is a placeholder for the React-based chat webview.
 * Currently, the chat UI is implemented inline in chatViewProvider.ts
 * for simplicity. This file can be expanded for more complex UI needs.
 */

import React from 'react';
import { createRoot } from 'react-dom/client';

// VS Code API
declare const acquireVsCodeApi: () => {
  postMessage: (message: unknown) => void;
  getState: () => unknown;
  setState: (state: unknown) => void;
};

const vscode = acquireVsCodeApi();

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  mode?: string;
  confidence?: number;
  sources?: Array<{ id: string; content: string; relevance: number }>;
}

function ChatApp() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState('');
  const [mode, setMode] = React.useState('direct');
  const [isLoading, setIsLoading] = React.useState(false);

  React.useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const data = event.data;
      switch (data.type) {
        case 'messages':
          setMessages(data.messages);
          break;
        case 'loading':
          setIsLoading(data.loading);
          break;
      }
    };

    window.addEventListener('message', handleMessage);
    vscode.postMessage({ type: 'getMessages' });

    return () => window.removeEventListener('message', handleMessage);
  }, []);

  const sendMessage = () => {
    if (!input.trim() || isLoading) return;

    vscode.postMessage({
      type: 'query',
      text: input,
      mode,
    });

    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-app">
      <div className="messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h3>HoloLoom Chat</h3>
            <p>Ask questions about your codebase, get explanations, or save knowledge for later.</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-header">
                <span>{msg.role === 'user' ? 'You' : 'HoloLoom'}</span>
                {msg.mode && <span className="mode">{msg.mode}</span>}
              </div>
              <div className="message-content">{msg.content}</div>
              {msg.role === 'assistant' && msg.confidence !== undefined && (
                <div className="message-meta">
                  <div className="confidence">
                    <span>{Math.round(msg.confidence * 100)}%</span>
                    <div className="confidence-bar">
                      <div
                        className="confidence-fill"
                        style={{ width: `${msg.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {isLoading && (
        <div className="loading">
          <div className="loading-spinner" />
          <span>Thinking...</span>
        </div>
      )}

      <div className="input-area">
        <div className="mode-selector">
          {['direct', 'verify', 'research', 'plan_execute'].map((m) => (
            <button
              key={m}
              className={`mode-btn ${mode === m ? 'active' : ''}`}
              onClick={() => setMode(m)}
            >
              {m === 'plan_execute' ? 'Plan' : m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
        <div className="input-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask HoloLoom..."
            rows={2}
          />
          <button className="send" onClick={sendMessage} disabled={isLoading}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

// Mount the app
const container = document.getElementById('app');
if (container) {
  const root = createRoot(container);
  root.render(<ChatApp />);
}
