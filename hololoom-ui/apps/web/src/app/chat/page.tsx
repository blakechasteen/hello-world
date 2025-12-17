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
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'system',
      content: 'Welcome to HoloLoom. I\'m ready to help you explore and reason over your knowledge base. What would you like to know?',
      timestamp: Date.now(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [reasoningMode, setReasoningMode] = useState<ReasoningMode>('direct');
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [showSourcePanel, setShowSourcePanel] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: inputValue,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate streaming response
    const assistantMessageId = `assistant-${Date.now()}`;
    const streamingMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      reasoningMode,
      isStreaming: true,
    };

    setMessages((prev) => [...prev, streamingMessage]);

    // Simulate response based on reasoning mode
    await simulateResponse(assistantMessageId, inputValue, reasoningMode);
    setIsLoading(false);
  };

  const simulateResponse = async (
    messageId: string,
    query: string,
    mode: ReasoningMode
  ) => {
    const mockResponses: Record<ReasoningMode, { content: string; confidence: number; sources: Source[]; steps?: string[]; verified?: boolean }> = {
      direct: {
        content: `Based on my knowledge, here's what I found about "${query}":\n\nThompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation by sampling from posterior distributions of expected rewards.\n\nIt maintains Beta(α, β) priors for each option and updates them based on observed outcomes.`,
        confidence: 0.89,
        sources: generateMockSources(3),
      },
      verify: {
        content: `I've verified this response through multiple sources:\n\n**Claim Analysis:**\n1. ✓ Thompson Sampling uses Bayesian priors - Verified in 3 sources\n2. ✓ Balances exploration/exploitation - Confirmed by algorithm definition\n3. ✓ Updates based on outcomes - Core mechanism validated\n\n**Answer:**\nThompson Sampling is a well-established algorithm for online decision making under uncertainty.`,
        confidence: 0.95,
        sources: generateMockSources(5),
        verified: true,
      },
      research: {
        content: `**Research Summary: "${query}"**\n\n**Key Findings:**\n1. Thompson Sampling was first described by William Thompson in 1933\n2. It's optimal in the sense of minimizing Bayesian regret\n3. Modern applications include A/B testing, recommendation systems, and RL\n\n**Related Topics:**\n- Multi-armed bandits\n- Bayesian optimization\n- Exploration strategies\n\n**Open Questions:**\n- Computational scaling for large action spaces\n- Non-stationary reward distributions`,
        confidence: 0.87,
        sources: generateMockSources(8),
        steps: [
          'Analyzed query for key concepts',
          'Retrieved 8 relevant memory nodes',
          'Explored 3 knowledge graph paths',
          'Synthesized findings from multiple sources',
        ],
      },
      plan_execute: {
        content: `**Execution Plan for: "${query}"**\n\n**Step 1: Define Scope** ✓\nIdentified key concepts: Thompson Sampling, Bayesian methods, bandits\n\n**Step 2: Gather Information** ✓\nRetrieved 6 relevant memories and 4 graph nodes\n\n**Step 3: Analyze Relationships** ✓\nMapped connections between concepts\n\n**Step 4: Generate Response** ✓\nSynthesized comprehensive answer\n\n**Result:**\nThompson Sampling is a probability-matching strategy that has become a gold standard for online learning problems.`,
        confidence: 0.92,
        sources: generateMockSources(6),
        steps: [
          'Define Scope',
          'Gather Information',
          'Analyze Relationships',
          'Generate Response',
        ],
      },
    };

    const response = mockResponses[mode];
    const words = response.content.split(' ');

    // Simulate streaming
    for (let i = 0; i <= words.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 30));
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? {
                ...msg,
                content: words.slice(0, i).join(' '),
                isStreaming: i < words.length,
              }
            : msg
        )
      );
    }

    // Final update with metadata
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId
          ? {
              ...msg,
              content: response.content,
              confidence: response.confidence,
              sources: response.sources,
              steps: response.steps,
              verified: response.verified,
              isStreaming: false,
            }
          : msg
      )
    );
  };

  const handleMessageClick = (message: Message) => {
    if (message.role === 'assistant' && message.sources) {
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

function generateMockSources(count: number): Source[] {
  const types: Source['type'][] = ['memory', 'knowledge_graph', 'cache'];
  const contents = [
    'Thompson Sampling is a heuristic for choosing actions that addresses the exploration-exploitation dilemma.',
    'The algorithm maintains a probability distribution over the expected reward of each action.',
    'Beta distributions are commonly used as priors for Bernoulli bandits.',
    'Thompson Sampling achieves logarithmic regret bounds in many settings.',
    'The approach naturally balances exploration and exploitation through random sampling.',
    'Modern implementations use various optimizations for computational efficiency.',
    'Applications include clinical trials, A/B testing, and recommendation systems.',
    'The algorithm was first proposed by William R. Thompson in 1933.',
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: `source-${i}`,
    content: contents[i % contents.length],
    relevance: 0.95 - i * 0.08,
    type: types[i % types.length],
    metadata: {
      timestamp: Date.now() - i * 86400000,
      accessCount: Math.floor(Math.random() * 50) + 1,
    },
  }));
}
