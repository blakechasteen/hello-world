'use client';

import { useState, useRef, useEffect } from 'react';
import { Navigation } from '../../components/Navigation';
import { Button, Badge, Card } from '@hololoom/design-system';
import {
  ChatMessage,
  ChatInput,
  SourcePanel,
  ReasoningModeSelector,
  ConfidenceIndicator,
} from '../../components/chat';
import { usePromptlyChat } from '@hololoom/api-client';
import type { ReasoningMode as ApiReasoningMode } from '@hololoom/api-client';

export type ReasoningMode = 'direct' | 'verify' | 'research' | 'plan_execute';

export interface Source {
  id: string;
  content: string;
  relevance: number;
  type: 'memory' | 'knowledge_graph' | 'cache';
  metadata?: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  confidence?: number;
  sources?: Source[];
  reasoningMode?: ReasoningMode;
  verified?: boolean;
  steps?: string[];
  isStreaming?: boolean;
  model?: string;
  refined?: boolean;
}

export default function ChatPage() {
  const {
    messages: promptlyMessages,
    sendMessage: promptlySend,
    isLoading,
    error,
    clearConversation,
  } = usePromptlyChat();

  const [inputValue, setInputValue] = useState('');
  const [reasoningMode, setReasoningMode] = useState<ReasoningMode>('direct');
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [showSourcePanel, setShowSourcePanel] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Adapt promptly messages to this page's Message type
  const messages: Message[] = [
    {
      id: 'welcome',
      role: 'system',
      content: 'Welcome to HoloLoom. I\'m ready to help you explore and reason over your knowledge base. What would you like to know?',
      timestamp: Date.now(),
    },
    ...promptlyMessages.map((m) => ({
      id: m.id,
      role: m.role as Message['role'],
      content: m.content,
      timestamp: m.timestamp.getTime(),
      confidence: m.metadata?.confidence,
      model: m.metadata?.model,
      refined: m.metadata?.refined,
      reasoningMode: reasoningMode,
    })),
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [promptlyMessages]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;
    const text = inputValue.trim();
    setInputValue('');
    await promptlySend(text, reasoningMode as ApiReasoningMode);
  };

  const handleMessageClick = (message: Message) => {
    if (message.role === 'assistant') {
      setSelectedMessage(message);
      setShowSourcePanel(true);
    }
  };

  return (
    <div className="min-h-screen bg-bg-primary flex flex-col">
      <Navigation />

      <div className="flex-1 flex">
        {/* Main Chat Area */}
        <main className={`flex-1 flex flex-col transition-all duration-300 ${showSourcePanel ? 'mr-96' : ''}`}>
          {/* Header */}
          <div className="border-b border-border-primary px-6 py-4">
            <div className="max-w-3xl mx-auto flex items-center justify-between">
              <div>
                <h1 className="text-xl font-semibold text-fg-primary">HoloLoom Chat</h1>
                <p className="text-sm text-fg-tertiary">
                  RAG-powered conversational interface
                </p>
              </div>
              <ReasoningModeSelector
                mode={reasoningMode}
                onChange={setReasoningMode}
              />
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6">
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onClick={() => handleMessageClick(message)}
                  isSelected={selectedMessage?.id === message.id}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div className="border-t border-border-primary px-6 py-4">
            <div className="max-w-3xl mx-auto">
              <ChatInput
                value={inputValue}
                onChange={setInputValue}
                onSend={handleSend}
                isLoading={isLoading}
                reasoningMode={reasoningMode}
              />
              <div className="mt-2 flex items-center justify-between text-xs text-fg-tertiary">
                <span>
                  Mode: <span className="text-cosmic-nebula font-medium">{formatModeName(reasoningMode)}</span>
                </span>
                <span>
                  Press <kbd className="px-1.5 py-0.5 bg-bg-secondary rounded text-fg-secondary">Enter</kbd> to send
                </span>
              </div>
            </div>
          </div>
        </main>

        {/* Source Panel */}
        {showSourcePanel && selectedMessage && (
          <SourcePanel
            message={selectedMessage}
            onClose={() => {
              setShowSourcePanel(false);
              setSelectedMessage(null);
            }}
          />
        )}
      </div>
    </div>
  );
}

function formatModeName(mode: ReasoningMode): string {
  const names: Record<ReasoningMode, string> = {
    direct: 'Direct Answer',
    verify: 'Verify & Confirm',
    research: 'Deep Research',
    plan_execute: 'Plan & Execute',
  };
  return names[mode];
}

