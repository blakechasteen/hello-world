/**
 * StreamingResponse Component
 *
 * Displays streaming text response with optional markdown rendering,
 * typing cursor animation, and token-by-token animation support.
 */

import React, { useMemo, useEffect, useRef, useCallback } from 'react';
import { clsx } from 'clsx';
import { marked } from 'marked';
import type { StreamingResponseProps } from '../types';

// ============================================================================
// Markdown Configuration
// ============================================================================

// Configure marked for safe rendering
marked.setOptions({
  breaks: true,
  gfm: true,
});

// ============================================================================
// Token Renderer Sub-component
// ============================================================================

interface TokenRendererProps {
  tokens: string[];
  onTokenRender?: (token: string, index: number) => void;
  animated?: boolean;
}

const TokenRenderer: React.FC<TokenRendererProps> = React.memo(({
  tokens,
  onTokenRender,
  animated = true,
}) => {
  const lastRenderedRef = useRef(0);

  useEffect(() => {
    if (onTokenRender && tokens.length > lastRenderedRef.current) {
      for (let i = lastRenderedRef.current; i < tokens.length; i++) {
        onTokenRender(tokens[i], i);
      }
      lastRenderedRef.current = tokens.length;
    }
  }, [tokens, onTokenRender]);

  if (!animated) {
    return <>{tokens.join('')}</>;
  }

  return (
    <>
      {tokens.map((token, index) => (
        <span
          key={index}
          className="token animate-fade-in"
          style={{
            animationDelay: `${index * 10}ms`,
          }}
        >
          {token}
        </span>
      ))}
    </>
  );
});

TokenRenderer.displayName = 'TokenRenderer';

// ============================================================================
// Cursor Component
// ============================================================================

interface CursorProps {
  char?: string;
  visible?: boolean;
}

const Cursor: React.FC<CursorProps> = React.memo(({ char = '▋', visible = true }) => {
  if (!visible) return null;

  return (
    <span
      className="cursor inline-block animate-blink"
      aria-hidden="true"
    >
      {char}
    </span>
  );
});

Cursor.displayName = 'Cursor';

// ============================================================================
// Markdown Renderer Component
// ============================================================================

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = React.memo(({ content, className }) => {
  const htmlContent = useMemo(() => {
    try {
      return marked.parse(content);
    } catch (error) {
      console.error('Markdown parsing error:', error);
      return content;
    }
  }, [content]);

  return (
    <div
      className={clsx('prose prose-sm dark:prose-invert max-w-none', className)}
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
});

MarkdownRenderer.displayName = 'MarkdownRenderer';

// ============================================================================
// Main Component
// ============================================================================

export const StreamingResponse: React.FC<StreamingResponseProps> = ({
  text,
  isStreaming,
  markdown = true,
  showCursor = true,
  cursorChar = '▋',
  tokenAnimation = false,
  onTokenRender,
  className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<string>('');

  // Track new tokens for animation
  const tokens = useMemo(() => {
    if (!tokenAnimation) return [];

    // Simple tokenization - split on whitespace boundaries
    const newTokens: string[] = [];
    let remaining = text.slice(textRef.current.length);

    // Split into tokens (words + spaces)
    const matches = remaining.match(/\S+|\s+/g) || [];
    newTokens.push(...matches);

    return newTokens;
  }, [text, tokenAnimation]);

  // Update text ref after render
  useEffect(() => {
    textRef.current = text;
  }, [text]);

  // Auto-scroll to bottom while streaming
  useEffect(() => {
    if (isStreaming && containerRef.current) {
      const container = containerRef.current;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;

      if (isNearBottom) {
        container.scrollTop = container.scrollHeight;
      }
    }
  }, [text, isStreaming]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
    if (event.key === 'Escape' && isStreaming) {
      // Could emit cancel event here
      event.preventDefault();
    }
  }, [isStreaming]);

  // Empty state
  if (!text && !isStreaming) {
    return (
      <div
        className={clsx(
          'streaming-response text-gray-400 dark:text-gray-500 italic',
          className
        )}
        role="status"
        aria-live="polite"
      >
        Waiting for response...
      </div>
    );
  }

  // Loading state (streaming but no text yet)
  if (!text && isStreaming) {
    return (
      <div
        className={clsx(
          'streaming-response flex items-center gap-2 text-gray-500',
          className
        )}
        role="status"
        aria-live="polite"
        aria-busy="true"
      >
        <LoadingDots />
        <span className="sr-only">Loading response...</span>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={clsx(
        'streaming-response',
        'relative overflow-auto',
        className
      )}
      role="log"
      aria-live="polite"
      aria-busy={isStreaming}
      aria-atomic="false"
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      {markdown ? (
        <div className="relative">
          <MarkdownRenderer content={text} />
          {showCursor && isStreaming && (
            <Cursor char={cursorChar} visible={isStreaming} />
          )}
        </div>
      ) : tokenAnimation ? (
        <div className="whitespace-pre-wrap font-mono text-sm">
          <TokenRenderer
            tokens={tokens}
            onTokenRender={onTokenRender}
            animated={tokenAnimation}
          />
          {showCursor && isStreaming && (
            <Cursor char={cursorChar} visible={isStreaming} />
          )}
        </div>
      ) : (
        <div className="whitespace-pre-wrap">
          {text}
          {showCursor && isStreaming && (
            <Cursor char={cursorChar} visible={isStreaming} />
          )}
        </div>
      )}

      {/* Screen reader announcements */}
      <div className="sr-only" aria-live="assertive">
        {isStreaming ? 'Response is streaming...' : 'Response complete.'}
      </div>
    </div>
  );
};

StreamingResponse.displayName = 'StreamingResponse';

export default StreamingResponse;

// ============================================================================
// Loading Dots Component
// ============================================================================

const LoadingDots: React.FC = () => (
  <span className="loading-dots inline-flex gap-1" aria-hidden="true">
    <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
    <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
    <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
  </span>
);

// ============================================================================
// Connected Component (uses store)
// ============================================================================

import { useHoloLoomStore, selectAccumulatedText, selectIsStreaming } from '../store';

export interface ConnectedStreamingResponseProps extends Omit<StreamingResponseProps, 'text' | 'isStreaming'> {}

export const ConnectedStreamingResponse: React.FC<ConnectedStreamingResponseProps> = (props) => {
  const text = useHoloLoomStore(selectAccumulatedText);
  const isStreaming = useHoloLoomStore(selectIsStreaming);

  return (
    <StreamingResponse
      text={text}
      isStreaming={isStreaming}
      {...props}
    />
  );
};

ConnectedStreamingResponse.displayName = 'ConnectedStreamingResponse';

// ============================================================================
// CSS Animations (inject into head or use Tailwind)
// ============================================================================

const styleId = 'hololoom-streaming-styles';

if (typeof document !== 'undefined' && !document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    @keyframes fade-in {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    @keyframes blink {
      0%, 50% { opacity: 1; }
      51%, 100% { opacity: 0; }
    }

    .animate-fade-in {
      animation: fade-in 150ms ease-out forwards;
    }

    .animate-blink {
      animation: blink 1s step-end infinite;
    }

    .token {
      opacity: 0;
    }
  `;
  document.head.appendChild(style);
}
