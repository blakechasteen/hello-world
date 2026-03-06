'use client';

import { useState, useRef, useEffect, FormEvent, KeyboardEvent } from 'react';
import {
  Button,
  Input,
  Avatar,
  Badge,
  ConfidenceIndicator,
  ReasoningBadge,
  LoadingDots,
  WeavingIndicator,
  SafetyBadge,
} from '@hololoom/design-system';
import { useConversation } from '@hololoom/api-client';
import type { ReasoningMode } from '@hololoom/api-client';

interface ChatInterfaceProps {
  className?: string;
  pendingReference?: { nodeId: string; content: string } | null;
  onClearReference?: () => void;
}

export function ChatInterface({ className, pendingReference, onClearReference }: ChatInterfaceProps) {
  const {
    messages,
    sendMessage,
    isLoading,
    error,
    clearConversation,
  } = useConversation();
  const [currentProgress, setCurrentProgress] = useState<{ step: number; message?: string; details?: string } | null>(null);

  const [input, setInput] = useState('');
  const [selectedMode, setSelectedMode] = useState<ReasoningMode>('direct');
  const [showModeSelector, setShowModeSelector] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentProgress]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    sendMessage(input.trim(), { mode: selectedMode });
    setInput('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const reasoningModes: { mode: ReasoningMode; label: string; description: string }[] = [
    { mode: 'direct', label: 'Direct', description: 'Single-pass answer (~150ms)' },
    { mode: 'verify', label: 'Verify', description: 'With verification step (~600ms)' },
    { mode: 'research', label: 'Research', description: 'Multi-query exploration (~900ms)' },
    { mode: 'plan_execute', label: 'Plan & Execute', description: 'Goal decomposition (~750ms)' },
  ];

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cosmic-nebula/20 to-cosmic-aurora/20 flex items-center justify-center mb-4">
              <span className="text-2xl">🌌</span>
            </div>
            <h3 className="text-lg font-medium text-fg-primary mb-2">
              Start a Conversation
            </h3>
            <p className="text-sm text-fg-tertiary max-w-sm">
              Ask questions, explore your knowledge, or run complex research queries.
              HoloLoom will weave context from your memories to provide intelligent answers.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${
              message.role === 'user' ? 'flex-row-reverse' : ''
            }`}
          >
            <Avatar
              variant={message.role === 'user' ? 'user' : 'ai'}
              size="sm"
              className="flex-shrink-0"
            />
            <div
              className={`flex flex-col gap-2 max-w-[80%] ${
                message.role === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              <div
                className={`rounded-2xl px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-cosmic-nebula/20 text-fg-primary rounded-tr-md'
                    : 'bg-bg-secondary text-fg-primary rounded-tl-md'
                }`}
              >
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              </div>

              {/* Message metadata for AI responses */}
              {message.role === 'assistant' && message.metadata && (
                <div className="flex flex-wrap items-center gap-2 px-1">
                  {message.metadata.confidence !== undefined && (
                    <ConfidenceIndicator
                      value={message.metadata.confidence}
                      showLabel
                      size="sm"
                    />
                  )}
                  {message.metadata.reasoningMode && (
                    <ReasoningBadge
                      mode={message.metadata.reasoningMode as ReasoningMode}
                      size="sm"
                    />
                  )}
                  {message.metadata.safetyLevel && (
                    <SafetyBadge
                      level={message.metadata.safetyLevel as 'safe' | 'caution' | 'warning' | 'danger'}
                      size="sm"
                    />
                  )}
                  {message.metadata.sourcesCount !== undefined && (
                    <Badge variant="secondary" size="sm">
                      {message.metadata.sourcesCount} sources
                    </Badge>
                  )}
                </div>
              )}

              {/* Timestamp */}
              <span className="text-xs text-fg-tertiary px-1">
                {new Date(message.timestamp).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            </div>
          </div>
        ))}

        {/* Loading state with progress */}
        {isLoading && (
          <div className="flex gap-3">
            <Avatar variant="ai" size="sm" className="flex-shrink-0" />
            <div className="flex flex-col gap-2">
              <div className="bg-bg-secondary rounded-2xl rounded-tl-md px-4 py-3">
                {currentProgress ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <WeavingIndicator step={currentProgress.step} />
                      <span className="text-sm text-fg-secondary">
                        {currentProgress.message || 'Processing...'}
                      </span>
                    </div>
                    {currentProgress.details && (
                      <p className="text-xs text-fg-tertiary">
                        {currentProgress.details}
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <LoadingDots size="sm" />
                    <span className="text-sm text-fg-secondary">Thinking...</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="flex items-center justify-between bg-safety-danger/10 border border-safety-danger/30 rounded-lg px-4 py-3">
            <div className="flex items-center gap-2">
              <SafetyBadge level="danger" size="sm" showLabel={false} />
              <span className="text-sm text-safety-danger">{error.message}</span>
            </div>
            <Button variant="ghost" size="sm" onClick={clearConversation}>
              Dismiss
            </Button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-border-primary p-4">
        {/* Mode selector */}
        <div className="mb-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-fg-tertiary">Reasoning Mode:</span>
            <button
              onClick={() => setShowModeSelector(!showModeSelector)}
              className="flex items-center gap-1 text-xs text-cosmic-nebula hover:text-cosmic-aurora transition-colors"
            >
              <ReasoningBadge mode={selectedMode} size="sm" />
              <ChevronIcon className={showModeSelector ? 'rotate-180' : ''} />
            </button>
          </div>

          {showModeSelector && (
            <div className="grid grid-cols-2 gap-2 p-2 bg-bg-secondary rounded-lg">
              {reasoningModes.map(({ mode, label, description }) => (
                <button
                  key={mode}
                  onClick={() => {
                    setSelectedMode(mode);
                    setShowModeSelector(false);
                  }}
                  className={`flex flex-col items-start p-2 rounded-lg text-left transition-colors ${
                    selectedMode === mode
                      ? 'bg-cosmic-nebula/20 border border-cosmic-nebula/30'
                      : 'hover:bg-bg-tertiary'
                  }`}
                >
                  <span className="text-sm font-medium text-fg-primary">{label}</span>
                  <span className="text-xs text-fg-tertiary">{description}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Reference pill from graph selection */}
        {pendingReference && (
          <div className="flex items-center gap-2 px-4 py-1 text-sm">
            <span className="px-2 py-0.5 rounded-full bg-bg-tertiary text-cosmic-nebula text-xs">
              @{pendingReference.content}
            </span>
            <button onClick={onClearReference} className="text-fg-tertiary hover:text-fg-secondary text-xs">
              ×
            </button>
          </div>
        )}

        {/* Input form */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask HoloLoom anything..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button type="submit" disabled={!input.trim() || isLoading}>
            {isLoading ? <LoadingDots size="sm" /> : <SendIcon />}
          </Button>
        </form>

        {/* Quick actions */}
        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-fg-tertiary">
              Press <kbd className="px-1.5 py-0.5 bg-bg-secondary rounded text-xs">Enter</kbd> to send
            </span>
          </div>
          {messages.length > 0 && (
            <Button variant="ghost" size="sm" onClick={clearConversation}>
              Clear chat
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

// Icon components
function SendIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m22 2-7 20-4-9-9-4Z" />
      <path d="M22 2 11 13" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`transition-transform ${className}`}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}
