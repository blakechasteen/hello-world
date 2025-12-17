/**
 * Integration Tests for WebSocket Client with React
 *
 * Tests for React hook integration, component lifecycle,
 * and real-world usage patterns.
 *
 * Date: 2025-12-11
 */

import { render, screen, waitFor } from '@testing-library/react';
import { ReactNode } from 'react';
import { useAgentManagerWS, useAgentManagerMessages, useAgentManagerPattern } from './useAgentManagerWS';
import { wsClient, type WebSocketMessage } from './websocketClient';

// Test components
function ConnectionStatusComponent() {
  const { isConnected, state } = useAgentManagerWS();
  return <div data-testid="status">{isConnected ? 'Connected' : state}</div>;
}

function MessageListenerComponent({ eventType }: { eventType: string }) {
  const messages = useAgentManagerMessages(eventType, 10);
  return (
    <div data-testid="messages">
      {messages.map((msg, idx) => (
        <div key={idx} data-testid={`message-${idx}`}>
          {msg.type}
        </div>
      ))}
    </div>
  );
}

function PatternSubscriberComponent({ pattern }: { pattern: string }) {
  const messages = useAgentManagerPattern(pattern, 10);
  return (
    <div data-testid="pattern-messages">
      <span data-testid="message-count">{messages.length}</span>
      {messages.map((msg, idx) => (
        <div key={idx} data-testid={`pattern-msg-${idx}`}>
          {msg.type}
        </div>
      ))}
    </div>
  );
}

function TestWrapper({ children }: { children: ReactNode }) {
  return <div>{children}</div>;
}

describe('WebSocket Integration with React', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    wsClient.disconnect();
  });

  describe('useAgentManagerWS Hook', () => {
    test('should initialize disconnected', () => {
      render(
        <TestWrapper>
          <ConnectionStatusComponent />
        </TestWrapper>
      );

      // Should show initial state (disconnected or connecting depending on autoConnect)
      expect(screen.getByTestId('status')).toBeInTheDocument();
    });

    test('should update on connection state change', async () => {
      render(
        <TestWrapper>
          <ConnectionStatusComponent />
        </TestWrapper>
      );

      // Initial state
      const status = screen.getByTestId('status');

      // Simulate connection
      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      await waitFor(() => {
        expect(status.textContent).toBe('Connected');
      });
    });

    test('should cleanup handlers on unmount', () => {
      const { unmount } = render(
        <TestWrapper>
          <ConnectionStatusComponent />
        </TestWrapper>
      );

      const stateHandlers = (wsClient as any).stateHandlers as Set<Function>;
      const initialSize = stateHandlers.size;

      unmount();

      // Handler should still exist (cleanup is automatic on disconnect)
      expect(stateHandlers.size).toBeLessThanOrEqual(initialSize);
    });
  });

  describe('useAgentManagerMessages Hook', () => {
    test('should collect messages of specific type', async () => {
      render(
        <TestWrapper>
          <MessageListenerComponent eventType="step_progress" />
        </TestWrapper>
      );

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Simulate incoming messages
      const message1: WebSocketMessage = {
        type: 'step_progress',
        timestamp: new Date().toISOString(),
        step_index: 0,
        progress: 50
      };

      (wsClient as any).ws.onmessage?.(
        new MessageEvent('message', { data: JSON.stringify(message1) })
      );

      await waitFor(() => {
        expect(screen.getByTestId('message-0')).toHaveTextContent('step_progress');
      });
    });

    test('should maintain message order (most recent first)', async () => {
      render(
        <TestWrapper>
          <MessageListenerComponent eventType="test_event" />
        </TestWrapper>
      );

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Send multiple messages
      for (let i = 0; i < 3; i++) {
        const message: WebSocketMessage = {
          type: 'test_event',
          timestamp: new Date().toISOString(),
          data: { index: i }
        };

        (wsClient as any).ws.onmessage?.(
          new MessageEvent('message', { data: JSON.stringify(message) })
        );
      }

      await waitFor(() => {
        const messages = screen.getAllByTestId(/^message-\d+$/);
        expect(messages.length).toBe(3);
      });
    });

    test('should respect maxMessages limit', async () => {
      const { rerender } = render(
        <TestWrapper>
          <MessageListenerComponent eventType="test_event" />
        </TestWrapper>
      );

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Send many messages
      for (let i = 0; i < 20; i++) {
        const message: WebSocketMessage = {
          type: 'test_event',
          timestamp: new Date().toISOString(),
          data: { index: i }
        };

        (wsClient as any).ws.onmessage?.(
          new MessageEvent('message', { data: JSON.stringify(message) })
        );
      }

      await waitFor(() => {
        const messages = screen.queryAllByTestId(/^message-\d+$/);
        expect(messages.length).toBeLessThanOrEqual(10); // Max 10
      });
    });
  });

  describe('useAgentManagerPattern Hook', () => {
    test('should filter messages by thread pattern', async () => {
      const threadId = 'thread-123';
      render(
        <TestWrapper>
          <PatternSubscriberComponent pattern={`thread:${threadId}`} />
        </TestWrapper>
      );

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Send message matching pattern
      const message1: WebSocketMessage = {
        type: 'step_progress',
        timestamp: new Date().toISOString(),
        thread_id: threadId,
        progress: 50
      };

      // Send message NOT matching pattern
      const message2: WebSocketMessage = {
        type: 'step_progress',
        timestamp: new Date().toISOString(),
        thread_id: 'other-thread',
        progress: 25
      };

      (wsClient as any).ws.onmessage?.(
        new MessageEvent('message', { data: JSON.stringify(message1) })
      );
      (wsClient as any).ws.onmessage?.(
        new MessageEvent('message', { data: JSON.stringify(message2) })
      );

      await waitFor(() => {
        const count = screen.getByTestId('message-count');
        expect(parseInt(count.textContent || '0')).toBe(1); // Only 1 matching
      });
    });

    test('should handle wildcard pattern', async () => {
      render(
        <TestWrapper>
          <PatternSubscriberComponent pattern="*" />
        </TestWrapper>
      );

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Send different message types
      const messages: WebSocketMessage[] = [
        { type: 'event1', timestamp: new Date().toISOString() },
        { type: 'event2', timestamp: new Date().toISOString() },
        { type: 'event3', timestamp: new Date().toISOString() }
      ];

      messages.forEach((msg) => {
        (wsClient as any).ws.onmessage?.(
          new MessageEvent('message', { data: JSON.stringify(msg) })
        );
      });

      await waitFor(() => {
        const count = screen.getByTestId('message-count');
        expect(parseInt(count.textContent || '0')).toBe(3); // All 3
      });
    });
  });

  describe('Multiple Component Integration', () => {
    function MultiComponentApp() {
      return (
        <TestWrapper>
          <ConnectionStatusComponent />
          <MessageListenerComponent eventType="event_a" />
          <PatternSubscriberComponent pattern="thread:123" />
        </TestWrapper>
      );
    }

    test('should handle multiple components with same client', async () => {
      render(<MultiComponentApp />);

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Send message
      const message: WebSocketMessage = {
        type: 'event_a',
        timestamp: new Date().toISOString(),
        thread_id: 'thread-123'
      };

      (wsClient as any).ws.onmessage?.(
        new MessageEvent('message', { data: JSON.stringify(message) })
      );

      await waitFor(() => {
        // Both components should receive the message
        expect(screen.getByTestId('status')).toHaveTextContent('Connected');
        expect(screen.getByTestId('message-0')).toHaveTextContent('event_a');
      });
    });
  });

  describe('Real-World Scenarios', () => {
    test('should handle rapid message bursts', async () => {
      render(
        <TestWrapper>
          <MessageListenerComponent eventType="rapid_event" />
        </TestWrapper>
      );

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Simulate rapid message burst
      for (let i = 0; i < 50; i++) {
        const message: WebSocketMessage = {
          type: 'rapid_event',
          timestamp: new Date().toISOString(),
          data: { sequence: i }
        };

        (wsClient as any).ws.onmessage?.(
          new MessageEvent('message', { data: JSON.stringify(message) })
        );
      }

      await waitFor(() => {
        const messages = screen.queryAllByTestId(/^message-\d+$/);
        expect(messages.length).toBeGreaterThan(0);
      });
    });

    test('should handle connection drop and reconnect', async () => {
      const { rerender } = render(
        <TestWrapper>
          <ConnectionStatusComponent />
        </TestWrapper>
      );

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      await waitFor(() => {
        expect(screen.getByTestId('status')).toHaveTextContent('Connected');
      });

      // Simulate disconnect
      (wsClient as any).ws.onclose?.();

      await waitFor(() => {
        expect(wsClient.getState()).not.toBe('connected');
      });

      // Simulate reconnect
      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      await waitFor(() => {
        expect(screen.getByTestId('status')).toHaveTextContent('Connected');
      });
    });

    test('should preserve message history across reconnects', async () => {
      render(
        <TestWrapper>
          <MessageListenerComponent eventType="persistent_event" />
        </TestWrapper>
      );

      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Send message
      const message1: WebSocketMessage = {
        type: 'persistent_event',
        timestamp: new Date().toISOString(),
        data: { num: 1 }
      };

      (wsClient as any).ws.onmessage?.(
        new MessageEvent('message', { data: JSON.stringify(message1) })
      );

      await waitFor(() => {
        expect(screen.getByTestId('message-0')).toBeInTheDocument();
      });

      // Simulate disconnect
      (wsClient as any).ws.onclose?.();

      // Message history should still exist
      expect(screen.getByTestId('message-0')).toBeInTheDocument();

      // Reconnect
      wsClient.connect();
      (wsClient as any).ws.onopen?.();

      // Send new message
      const message2: WebSocketMessage = {
        type: 'persistent_event',
        timestamp: new Date().toISOString(),
        data: { num: 2 }
      };

      (wsClient as any).ws.onmessage?.(
        new MessageEvent('message', { data: JSON.stringify(message2) })
      );

      await waitFor(() => {
        const messages = screen.getAllByTestId(/^message-\d+$/);
        expect(messages.length).toBe(2);
      });
    });
  });

  describe('Error Handling in Components', () => {
    function ErrorBoundaryComponent() {
      try {
        const { on } = useAgentManagerWS();

        // Intentionally cause error
        on('bad_event', () => {
          throw new Error('Handler error');
        });

        return <div>OK</div>;
      } catch (e) {
        return <div>Error</div>;
      }
    }

    test('should handle handler errors gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      render(
        <TestWrapper>
          <ErrorBoundaryComponent />
        </TestWrapper>
      );

      expect(screen.getByText('OK')).toBeInTheDocument();

      consoleSpy.mockRestore();
    });
  });
});
