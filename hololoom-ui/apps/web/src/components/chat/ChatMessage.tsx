'use client';

import { Badge } from '@hololoom/design-system';
import type { Message, ReasoningMode } from '../../app/chat/page';

interface ChatMessageProps {
  message: Message;
  onClick?: () => void;
  isSelected?: boolean;
}

export function ChatMessage({ message, onClick, isSelected }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const isAssistant = message.role === 'assistant';

  if (isSystem) {
    return (
      <div className="flex justify-center">
        <div className="bg-bg-secondary/50 rounded-lg px-4 py-2 max-w-md text-center">
          <p className="text-sm text-fg-tertiary">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-2xl ${isUser ? 'order-2' : 'order-1'}`}
        onClick={isAssistant && message.sources ? onClick : undefined}
      >
        {/* Avatar */}
        <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
              isUser
                ? 'bg-cosmic-nebula text-white'
                : 'bg-gradient-to-br from-cosmic-aurora to-cosmic-nebula text-white'
            }`}
          >
            {isUser ? (
              <UserIcon />
            ) : (
              <HoloLoomIcon />
            )}
          </div>

          <div className={`flex-1 ${isUser ? 'text-right' : ''}`}>
            {/* Header */}
            <div className={`flex items-center gap-2 mb-1 ${isUser ? 'justify-end' : ''}`}>
              <span className="text-sm font-medium text-fg-primary">
                {isUser ? 'You' : 'HoloLoom'}
              </span>
              <span className="text-xs text-fg-tertiary">
                {formatTime(message.timestamp)}
              </span>
              {message.reasoningMode && (
                <Badge variant="secondary" size="sm">
                  {formatModeName(message.reasoningMode)}
                </Badge>
              )}
              {message.verified && (
                <Badge variant="success" size="sm">
                  <VerifiedIcon />
                  Verified
                </Badge>
              )}
            </div>

            {/* Content */}
            <div
              className={`rounded-2xl px-4 py-3 ${
                isUser
                  ? 'bg-cosmic-nebula text-white rounded-tr-sm'
                  : `bg-bg-secondary text-fg-primary rounded-tl-sm ${
                      isSelected ? 'ring-2 ring-cosmic-nebula' : ''
                    } ${message.sources ? 'cursor-pointer hover:bg-bg-tertiary transition-colors' : ''}`
              }`}
            >
              <div className="prose prose-sm max-w-none">
                <MessageContent content={message.content} />
              </div>

              {message.isStreaming && (
                <span className="inline-block w-2 h-4 bg-fg-tertiary animate-pulse ml-1" />
              )}
            </div>

            {/* Footer - Confidence & Sources */}
            {isAssistant && !message.isStreaming && (
              <div className={`mt-2 flex items-center gap-3 ${isUser ? 'justify-end' : ''}`}>
                {message.confidence !== undefined && (
                  <ConfidencePill confidence={message.confidence} />
                )}
                {message.sources && message.sources.length > 0 && (
                  <button
                    onClick={onClick}
                    className="flex items-center gap-1 text-xs text-fg-tertiary hover:text-cosmic-nebula transition-colors"
                  >
                    <SourceIcon />
                    {message.sources.length} sources
                  </button>
                )}
                {message.steps && message.steps.length > 0 && (
                  <span className="flex items-center gap-1 text-xs text-fg-tertiary">
                    <StepsIcon />
                    {message.steps.length} steps
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageContent({ content }: { content: string }) {
  // Parse markdown-like formatting
  const lines = content.split('\n');

  return (
    <div className="space-y-2">
      {lines.map((line, i) => {
        if (line.startsWith('**') && line.endsWith('**')) {
          return (
            <p key={i} className="font-semibold text-fg-primary">
              {line.slice(2, -2)}
            </p>
          );
        }
        if (line.startsWith('- ')) {
          return (
            <p key={i} className="pl-4 before:content-['•'] before:mr-2 before:text-fg-tertiary">
              {line.slice(2)}
            </p>
          );
        }
        if (line.match(/^\d+\.\s/)) {
          return (
            <p key={i} className="pl-4">
              {line}
            </p>
          );
        }
        if (line.trim() === '') {
          return <div key={i} className="h-2" />;
        }
        return <p key={i}>{line}</p>;
      })}
    </div>
  );
}

function ConfidencePill({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100);
  let colorClass = 'text-safety-safe';
  if (confidence < 0.7) colorClass = 'text-safety-risky';
  if (confidence < 0.5) colorClass = 'text-safety-critical';

  return (
    <div className="flex items-center gap-1.5">
      <div className="w-12 h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            confidence >= 0.7
              ? 'bg-safety-safe'
              : confidence >= 0.5
              ? 'bg-safety-risky'
              : 'bg-safety-critical'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className={`text-xs font-medium ${colorClass}`}>
        {percentage}%
      </span>
    </div>
  );
}

function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatModeName(mode: ReasoningMode): string {
  const names: Record<ReasoningMode, string> = {
    direct: 'Direct',
    verify: 'Verify',
    research: 'Research',
    plan_execute: 'Plan',
  };
  return names[mode];
}

function UserIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function HoloLoomIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v4" />
      <path d="M12 18v4" />
      <path d="m4.93 4.93 2.83 2.83" />
      <path d="m16.24 16.24 2.83 2.83" />
      <path d="M2 12h4" />
      <path d="M18 12h4" />
      <path d="m4.93 19.07 2.83-2.83" />
      <path d="m16.24 7.76 2.83-2.83" />
    </svg>
  );
}

function VerifiedIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function SourceIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}

function StepsIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="8" y1="6" x2="21" y2="6" />
      <line x1="8" y1="12" x2="21" y2="12" />
      <line x1="8" y1="18" x2="21" y2="18" />
      <line x1="3" y1="6" x2="3.01" y2="6" />
      <line x1="3" y1="12" x2="3.01" y2="12" />
      <line x1="3" y1="18" x2="3.01" y2="18" />
    </svg>
  );
}
