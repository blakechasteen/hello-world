# @hololoom/components

React component library for building streaming AI interfaces with HoloLoom.

## Features

- 🚀 **Real-time Streaming** - WebSocket-based streaming with automatic reconnection
- 📊 **Confidence Visualization** - Arc gauge with sparkline history
- ✨ **Token Animation** - Smooth token-by-token rendering with markdown support
- 🔄 **Zustand State Management** - Centralized store with immer and devtools
- 🎨 **Tailwind CSS** - Utility-first styling with dark mode support
- ♿ **Accessible** - ARIA attributes and keyboard navigation
- 📦 **TypeScript** - Full type safety with exported types

## Installation

```bash
npm install @hololoom/components

# Peer dependencies
npm install react react-dom
```

## Quick Start

```tsx
import {
  ConnectedConfidenceGauge,
  ConnectedStreamingResponse,
  useWebSocket,
  useHoloLoomStore,
} from '@hololoom/components';

function App() {
  const { connect, sendQuery, isConnected } = useWebSocket({
    url: 'ws://localhost:8001/ws/stream',
  });

  const handleSubmit = async (query: string) => {
    if (!isConnected()) {
      await connect();
    }
    sendQuery(query, 'direct');
  };

  return (
    <div className="flex gap-4">
      <ConnectedConfidenceGauge size="lg" showTrend />
      <ConnectedStreamingResponse markdown showCursor />
    </div>
  );
}
```

## Components

### ConfidenceGauge

Displays confidence as an arc gauge with optional sparkline trend.

```tsx
import { ConfidenceGauge } from '@hololoom/components';

<ConfidenceGauge
  confidence={0.85}
  epistemicConfidence={0.72}
  history={[0.6, 0.7, 0.75, 0.8, 0.85]}
  size="md"
  showValue
  showTrend
  animated
/>
```

**Props:**
- `confidence: number` - Current confidence value (0-1)
- `epistemicConfidence?: number` - Meta-uncertainty (0-1)
- `history?: number[]` - Historical values for sparkline
- `size?: 'sm' | 'md' | 'lg'` - Component size
- `showValue?: boolean` - Show percentage text
- `showTrend?: boolean` - Show sparkline
- `label?: string` - Custom label
- `animated?: boolean` - Animate changes

### StreamingResponse

Displays streaming text with markdown rendering and cursor animation.

```tsx
import { StreamingResponse } from '@hololoom/components';

<StreamingResponse
  text={accumulatedText}
  isStreaming={true}
  markdown
  showCursor
  cursorChar="▋"
/>
```

**Props:**
- `text: string` - Accumulated response text
- `isStreaming: boolean` - Whether stream is active
- `markdown?: boolean` - Render as markdown
- `showCursor?: boolean` - Show blinking cursor
- `cursorChar?: string` - Cursor character
- `tokenAnimation?: boolean` - Animate tokens individually
- `onTokenRender?: (token, index) => void` - Token callback

### Connected Components

Use `ConnectedConfidenceGauge` and `ConnectedStreamingResponse` to automatically connect to the Zustand store:

```tsx
import {
  ConnectedConfidenceGauge,
  ConnectedStreamingResponse,
} from '@hololoom/components';

// No props needed - pulls from store
<ConnectedConfidenceGauge size="lg" />
<ConnectedStreamingResponse markdown />
```

## WebSocket Client

```tsx
import { useWebSocket, getWebSocketClient } from '@hololoom/components';

// Hook usage
const { connect, disconnect, sendQuery, cancelStream, isConnected } = useWebSocket({
  url: 'ws://localhost:8001/ws/stream',
  autoReconnect: true,
  maxReconnectAttempts: 5,
});

// Direct client usage
const client = getWebSocketClient({ url: '...' });
await client.connect();
client.sendQuery('What is Thompson Sampling?', 'research');
```

## Store

The Zustand store manages all streaming state:

```tsx
import {
  useHoloLoomStore,
  selectConfidence,
  selectIsStreaming,
  selectAccumulatedText,
} from '@hololoom/components';

function MyComponent() {
  const confidence = useHoloLoomStore(selectConfidence);
  const isStreaming = useHoloLoomStore(selectIsStreaming);
  const text = useHoloLoomStore(selectAccumulatedText);

  return <div>Confidence: {Math.round(confidence * 100)}%</div>;
}
```

**Available Selectors:**
- `selectConnectionState` / `selectIsConnected`
- `selectStreamState` / `selectIsStreaming` / `selectIsComplete`
- `selectAccumulatedText` / `selectTokens` / `selectTokenCount`
- `selectContextChunks` / `selectContextTokenCount`
- `selectConfidence` / `selectEpistemicConfidence` / `selectConfidenceHistory`
- `selectStages` / `selectActiveStage` / `selectCompletedStages`
- `selectReasoningSteps` / `selectActiveReasoningStep`
- `selectMetrics` / `selectError` / `selectHasError`
- `selectProgress` / `selectCurrentStageDisplay`

## Server Setup

Start the streaming server:

```bash
# From mythRL root
PYTHONPATH=. uvicorn HoloLoom.server.streaming_api:app --port 8001
```

The server provides:
- WebSocket endpoint: `ws://localhost:8001/ws/stream`
- Health check: `http://localhost:8001/health`

## Message Protocol

### Client → Server

```typescript
// Start query
{ "type": "query", "query": "...", "mode": "direct" }

// Cancel stream
{ "type": "cancel" }

// Heartbeat
{ "type": "heartbeat", "timestamp": "..." }
```

### Server → Client

```typescript
// Stream start
{ "type": "stream_start", "session_id": "...", "query": "...", "mode": "direct" }

// Context chunk
{ "type": "context_chunk", "chunk_index": 0, "nodes": [...], "hop_distance": 0 }

// Token
{ "type": "token", "token": "Hello", "cumulative_text": "Hello", "token_index": 0 }

// Confidence update
{ "type": "confidence_update", "confidence": 0.85, "epistemic_confidence": 0.72 }

// Stage complete
{ "type": "stage_complete", "stage": "yarn_graph", "duration_ms": 45.2 }

// Stream end
{ "type": "stream_end", "total_duration_ms": 1250, "final_confidence": 0.92 }

// Error
{ "type": "error", "error_code": "TIMEOUT", "message": "...", "recoverable": true }
```

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build library
npm run build

# Run tests
npm test

# Run Storybook
npm run storybook
```

## File Structure

```
src/
├── index.ts              # Library entry point
├── types.ts              # TypeScript definitions
├── store.ts              # Zustand store
├── WebSocketClient.ts    # WebSocket client + hook
└── components/
    ├── ConfidenceGauge.tsx
    └── StreamingResponse.tsx
```

## License

MIT
