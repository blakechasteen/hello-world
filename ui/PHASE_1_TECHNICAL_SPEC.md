# Phase 1: Foundation & Real-Time Infrastructure - Technical Specification

**Version**: 1.0.0
**Created**: 2025-11-30
**Status**: Implementation Ready

---

## Executive Summary

Phase 1 establishes the foundational infrastructure for HoloLoom's revolutionary UX. This specification details the WebSocket streaming protocol, React component library architecture, and confidence visualization prototypes that will power all subsequent phases.

### Key Deliverables

| Component | Priority | Effort | Dependencies |
|-----------|----------|--------|--------------|
| WebSocket Streaming Protocol | P0 | 3 days | agentic_api.py (exists) |
| React Component Library | P0 | 4 days | Vite scaffold (exists) |
| Confidence Visualization | P1 | 2 days | Component library |
| Streaming Response UI | P1 | 2 days | WebSocket protocol |
| State Management (Zustand) | P0 | 1 day | Component library |

**Total Estimated Effort**: 12 days (can parallelize to ~8 days with 2 developers)

---

## 1. WebSocket Streaming Protocol

### 1.1 Architecture Overview

```
┌──────────────────┐     WebSocket      ┌──────────────────┐
│   React Client   │◄──────────────────►│  FastAPI Server  │
│                  │                     │  (agentic_api)   │
│ ┌──────────────┐ │                     │ ┌──────────────┐ │
│ │ HoloLoomWS   │ │  ws://host:8000/    │ │ WeavingOrch  │ │
│ │   Client     │ │  ws/stream          │ │   estrator   │ │
│ └──────────────┘ │                     │ └──────────────┘ │
│                  │                     │                  │
│ ┌──────────────┐ │                     │ ┌──────────────┐ │
│ │ Zustand      │ │  Message Types:     │ │ Interleaved  │ │
│ │ Store        │ │  - stream_start     │ │ Generation   │ │
│ └──────────────┘ │  - context_chunk    │ └──────────────┘ │
│                  │  - token            │                  │
│ ┌──────────────┐ │  - confidence       │ ┌──────────────┐ │
│ │ React        │ │  - stage_complete   │ │ Memory       │ │
│ │ Components   │ │  - stream_end       │ │ Symphony     │ │
│ └──────────────┘ │  - error            │ └──────────────┘ │
└──────────────────┘                     └──────────────────┘
```

### 1.2 WebSocket Message Schema

#### Base Message Format

```typescript
// All messages follow this base structure
interface WSMessage {
  type: MessageType;
  timestamp: string;        // ISO 8601
  session_id: string;       // Unique per query session
  sequence: number;         // Monotonically increasing per session
}

type MessageType =
  | 'stream_start'
  | 'context_chunk'
  | 'token'
  | 'confidence_update'
  | 'stage_complete'
  | 'reasoning_step'
  | 'stream_end'
  | 'error'
  | 'heartbeat';
```

#### Message Type Definitions

```typescript
// 1. Stream Start - First message when query begins
interface StreamStartMessage extends WSMessage {
  type: 'stream_start';
  query: {
    text: string;
    mode: 'direct' | 'verify' | 'research' | 'plan_execute';
    max_steps: number;
  };
  estimated_duration_ms: number;
  stages: string[];  // ['retrieval', 'reasoning', 'generation', 'verification']
}

// 2. Context Chunk - Knowledge retrieved from memory
interface ContextChunkMessage extends WSMessage {
  type: 'context_chunk';
  chunk: {
    nodes: string[];          // Node IDs
    relevance_scores: Record<string, number>;
    hop_distance: number;
    token_count: number;
    cumulative_tokens: number;
  };
  is_final: boolean;
}

// 3. Token - Streaming generation token
interface TokenMessage extends WSMessage {
  type: 'token';
  token: string;
  cumulative_text: string;
  token_index: number;
  is_final: boolean;
  generation_metadata?: {
    model: string;
    temperature: number;
  };
}

// 4. Confidence Update - Real-time confidence tracking
interface ConfidenceUpdateMessage extends WSMessage {
  type: 'confidence_update';
  confidence: number;           // 0.0-1.0
  epistemic_confidence: number; // Meta-confidence
  source: 'retrieval' | 'generation' | 'verification' | 'aggregate';
  factors: {
    source_count: number;
    relevance_avg: number;
    coherence: number;
    activation_level: number;
  };
}

// 5. Stage Complete - Pipeline stage finished
interface StageCompleteMessage extends WSMessage {
  type: 'stage_complete';
  stage: string;
  duration_ms: number;
  metadata: {
    items_processed?: number;
    cache_hit?: boolean;
    strategy_used?: string;
  };
}

// 6. Reasoning Step - Multi-step reasoning progress
interface ReasoningStepMessage extends WSMessage {
  type: 'reasoning_step';
  step: {
    index: number;
    total: number;
    type: 'query' | 'verify' | 'synthesize' | 'plan' | 'execute';
    description: string;
    confidence: number;
    duration_ms: number;
  };
}

// 7. Stream End - Final message
interface StreamEndMessage extends WSMessage {
  type: 'stream_end';
  summary: {
    total_duration_ms: number;
    tokens_generated: number;
    context_chunks: number;
    final_confidence: number;
    reasoning_steps: number;
    cache_effectiveness: number;
  };
  spacetime_id: string;  // For audit trail lookup
}

// 8. Error - Error during processing
interface ErrorMessage extends WSMessage {
  type: 'error';
  error: {
    code: string;
    message: string;
    recoverable: boolean;
    retry_after_ms?: number;
  };
}

// 9. Heartbeat - Keep connection alive
interface HeartbeatMessage extends WSMessage {
  type: 'heartbeat';
  server_time: string;
  active_sessions: number;
}
```

### 1.3 Client Request Schema

```typescript
// Client-to-server messages
interface WSClientMessage {
  type: ClientMessageType;
  request_id: string;
}

type ClientMessageType =
  | 'query'
  | 'cancel'
  | 'subscribe'
  | 'unsubscribe'
  | 'ping';

interface QueryRequest extends WSClientMessage {
  type: 'query';
  query: {
    text: string;
    mode?: 'direct' | 'verify' | 'research' | 'plan_execute';
    max_steps?: number;
    context?: {
      languageId?: string;
      fileName?: string;
      selection?: string;
      workspace?: string;
    };
  };
}

interface CancelRequest extends WSClientMessage {
  type: 'cancel';
  session_id: string;
}

interface SubscribeRequest extends WSClientMessage {
  type: 'subscribe';
  channels: ('all' | 'confidence' | 'tokens' | 'stages')[];
}
```

### 1.4 Server Implementation

**File**: `hololoom/server/streaming_api.py`

```python
"""
HoloLoom Streaming WebSocket API
================================
Real-time streaming endpoint with interleaved generation.
"""

from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
from typing import AsyncIterator, Optional
from datetime import datetime
import asyncio
import json
import uuid

from hololoom.memory.interleaved_generation import (
    stream_interleaved_expansion_generation,
    StreamMode,
    ContextChunk,
    GenerationToken,
    StreamMetadata
)


@dataclass
class StreamSession:
    """Active streaming session."""
    session_id: str
    websocket: WebSocket
    query_text: str
    mode: str
    start_time: datetime
    sequence: int = 0
    cancelled: bool = False


class StreamingManager:
    """Manages WebSocket streaming sessions."""

    def __init__(self):
        self.sessions: dict[str, StreamSession] = {}
        self.heartbeat_interval = 30.0  # seconds

    async def handle_connection(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
        await websocket.accept()

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(websocket)
        )

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await self._handle_message(websocket, message)
        except WebSocketDisconnect:
            pass
        finally:
            heartbeat_task.cancel()
            # Cleanup any active sessions for this socket
            for sid, session in list(self.sessions.items()):
                if session.websocket == websocket:
                    session.cancelled = True
                    del self.sessions[sid]

    async def _handle_message(self, websocket: WebSocket, message: dict):
        """Route incoming message to handler."""
        msg_type = message.get('type')

        if msg_type == 'query':
            await self._handle_query(websocket, message)
        elif msg_type == 'cancel':
            await self._handle_cancel(message)
        elif msg_type == 'ping':
            await websocket.send_json({'type': 'pong'})

    async def _handle_query(self, websocket: WebSocket, message: dict):
        """Process streaming query."""
        session_id = str(uuid.uuid4())
        query = message.get('query', {})

        session = StreamSession(
            session_id=session_id,
            websocket=websocket,
            query_text=query.get('text', ''),
            mode=query.get('mode', 'direct'),
            start_time=datetime.now()
        )
        self.sessions[session_id] = session

        try:
            # Send stream start
            await self._send(session, {
                'type': 'stream_start',
                'query': query,
                'estimated_duration_ms': 500,
                'stages': ['retrieval', 'reasoning', 'generation']
            })

            # Stream interleaved expansion + generation
            async for item in self._stream_query(session, query):
                if session.cancelled:
                    break
                await self._send(session, item)

            # Send stream end
            await self._send(session, {
                'type': 'stream_end',
                'summary': {
                    'total_duration_ms': (datetime.now() - session.start_time).total_seconds() * 1000,
                    'final_confidence': 0.85
                }
            })

        except Exception as e:
            await self._send(session, {
                'type': 'error',
                'error': {
                    'code': 'PROCESSING_ERROR',
                    'message': str(e),
                    'recoverable': True
                }
            })
        finally:
            if session_id in self.sessions:
                del self.sessions[session_id]

    async def _stream_query(
        self,
        session: StreamSession,
        query: dict
    ) -> AsyncIterator[dict]:
        """Stream query results."""
        from hololoom.config import Config
        from hololoom.memory.backend_factory import create_memory_backend

        config = Config.fast()
        memory = await create_memory_backend(config)

        # Get graph and seed nodes
        graph = memory.graph if hasattr(memory, 'graph') else None
        seed_nodes = ['default_seed']  # Would come from initial retrieval

        async for item in stream_interleaved_expansion_generation(
            query=query.get('text', ''),
            seed_nodes=seed_nodes,
            graph=graph,
            token_budget=2000,
            max_generation_tokens=500,
            stream_mode=StreamMode.CONCURRENT
        ):
            if isinstance(item, ContextChunk):
                yield {
                    'type': 'context_chunk',
                    'chunk': {
                        'nodes': item.nodes,
                        'relevance_scores': item.relevance_scores,
                        'hop_distance': item.hop_distance,
                        'token_count': item.token_count,
                        'cumulative_tokens': item.cumulative_tokens
                    },
                    'is_final': item.is_final
                }
            elif isinstance(item, GenerationToken):
                yield {
                    'type': 'token',
                    'token': item.token,
                    'cumulative_text': item.cumulative_text,
                    'token_index': item.token_index,
                    'is_final': item.is_final
                }
            elif isinstance(item, StreamMetadata):
                yield {
                    'type': 'stage_complete',
                    'stage': item.event_type,
                    'duration_ms': 0
                }

    async def _send(self, session: StreamSession, message: dict):
        """Send message to session."""
        session.sequence += 1
        message['timestamp'] = datetime.now().isoformat()
        message['session_id'] = session.session_id
        message['sequence'] = session.sequence
        await session.websocket.send_json(message)

    async def _handle_cancel(self, message: dict):
        """Cancel active session."""
        session_id = message.get('session_id')
        if session_id in self.sessions:
            self.sessions[session_id].cancelled = True

    async def _heartbeat_loop(self, websocket: WebSocket):
        """Send periodic heartbeats."""
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await websocket.send_json({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat(),
                    'server_time': datetime.now().isoformat(),
                    'active_sessions': len(self.sessions)
                })
            except:
                break


# Global manager instance
streaming_manager = StreamingManager()
```

---

## 2. React Component Library

### 2.1 Architecture

```
ui/hololoom-components/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
│
├── src/
│   ├── index.ts                    # Public exports
│   │
│   ├── core/                       # Foundation
│   │   ├── WebSocketClient.ts      # WS connection manager
│   │   ├── store.ts                # Zustand store
│   │   ├── types.ts                # TypeScript types
│   │   └── hooks.ts                # Custom hooks
│   │
│   ├── components/                 # UI Components
│   │   ├── ConfidenceGauge/
│   │   │   ├── ConfidenceGauge.tsx
│   │   │   ├── ConfidenceGauge.css
│   │   │   └── index.ts
│   │   │
│   │   ├── StreamingResponse/
│   │   │   ├── StreamingResponse.tsx
│   │   │   ├── TokenRenderer.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ReasoningTree/
│   │   │   ├── ReasoningTree.tsx
│   │   │   ├── TreeNode.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── PipelineWaterfall/
│   │   │   ├── PipelineWaterfall.tsx
│   │   │   └── index.ts
│   │   │
│   │   ├── ContextExplorer/
│   │   │   ├── ContextExplorer.tsx
│   │   │   ├── ChunkCard.tsx
│   │   │   └── index.ts
│   │   │
│   │   └── KnowledgeGraph/
│   │       ├── KnowledgeGraph.tsx
│   │       ├── ForceLayout.ts
│   │       └── index.ts
│   │
│   └── utils/
│       ├── formatters.ts
│       ├── animations.ts
│       └── accessibility.ts
│
└── stories/                        # Storybook stories
    ├── ConfidenceGauge.stories.tsx
    ├── StreamingResponse.stories.tsx
    └── ...
```

### 2.2 Core Types

```typescript
// src/core/types.ts

// ============================================================================
// WebSocket Types
// ============================================================================

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketState {
  status: ConnectionStatus;
  lastHeartbeat: Date | null;
  reconnectAttempts: number;
}

// ============================================================================
// Query Types
// ============================================================================

export type ReasoningMode = 'direct' | 'verify' | 'research' | 'plan_execute';

export interface Query {
  text: string;
  mode: ReasoningMode;
  maxSteps: number;
  context?: CodeContext;
}

export interface CodeContext {
  languageId?: string;
  fileName?: string;
  selection?: string;
  workspace?: string;
}

// ============================================================================
// Stream Types
// ============================================================================

export interface StreamSession {
  sessionId: string;
  query: Query;
  startTime: Date;
  status: 'active' | 'completed' | 'cancelled' | 'error';

  // Accumulated state
  contextChunks: ContextChunk[];
  tokens: string[];
  confidenceHistory: ConfidencePoint[];
  stages: StageInfo[];
  reasoningSteps: ReasoningStep[];

  // Final results
  response?: string;
  finalConfidence?: number;
  totalDurationMs?: number;
}

export interface ContextChunk {
  nodes: string[];
  relevanceScores: Record<string, number>;
  hopDistance: number;
  tokenCount: number;
  cumulativeTokens: number;
  isFinal: boolean;
}

export interface ConfidencePoint {
  timestamp: Date;
  confidence: number;
  epistemicConfidence: number;
  source: 'retrieval' | 'generation' | 'verification' | 'aggregate';
}

export interface StageInfo {
  name: string;
  startTime: Date;
  endTime?: Date;
  durationMs?: number;
  status: 'pending' | 'active' | 'completed' | 'error';
  metadata?: Record<string, unknown>;
}

export interface ReasoningStep {
  index: number;
  total: number;
  type: 'query' | 'verify' | 'synthesize' | 'plan' | 'execute';
  description: string;
  confidence: number;
  durationMs: number;
}

// ============================================================================
// Component Props
// ============================================================================

export interface ConfidenceGaugeProps {
  confidence: number;
  epistemicConfidence?: number;
  size?: 'sm' | 'md' | 'lg';
  showHistory?: boolean;
  history?: ConfidencePoint[];
  animated?: boolean;
  className?: string;
}

export interface StreamingResponseProps {
  session: StreamSession | null;
  onComplete?: (response: string) => void;
  showMetadata?: boolean;
  className?: string;
}

export interface ReasoningTreeProps {
  steps: ReasoningStep[];
  currentStep?: number;
  expandable?: boolean;
  className?: string;
}

export interface PipelineWaterfallProps {
  stages: StageInfo[];
  totalDurationMs?: number;
  showBottlenecks?: boolean;
  className?: string;
}

export interface ContextExplorerProps {
  chunks: ContextChunk[];
  onNodeClick?: (nodeId: string) => void;
  expandedByDefault?: boolean;
  className?: string;
}
```

### 2.3 Zustand Store

```typescript
// src/core/store.ts

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type {
  StreamSession,
  WebSocketState,
  Query,
  ContextChunk,
  ConfidencePoint,
  StageInfo,
  ReasoningStep
} from './types';

// ============================================================================
// Store Types
// ============================================================================

interface HoloLoomState {
  // Connection
  websocket: WebSocketState;

  // Sessions
  activeSession: StreamSession | null;
  sessionHistory: StreamSession[];

  // UI State
  isExpanded: boolean;
  selectedTab: 'response' | 'reasoning' | 'context' | 'graph';

  // Actions
  setWebSocketStatus: (status: WebSocketState['status']) => void;
  startSession: (query: Query) => void;
  updateSession: (updates: Partial<StreamSession>) => void;
  addContextChunk: (chunk: ContextChunk) => void;
  addToken: (token: string) => void;
  updateConfidence: (point: ConfidencePoint) => void;
  updateStage: (stage: StageInfo) => void;
  addReasoningStep: (step: ReasoningStep) => void;
  endSession: (summary: SessionSummary) => void;
  cancelSession: () => void;
  clearHistory: () => void;
  setExpanded: (expanded: boolean) => void;
  setTab: (tab: HoloLoomState['selectedTab']) => void;
}

interface SessionSummary {
  totalDurationMs: number;
  tokensGenerated: number;
  finalConfidence: number;
  spacetimeId: string;
}

// ============================================================================
// Store Implementation
// ============================================================================

export const useHoloLoomStore = create<HoloLoomState>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        // Initial state
        websocket: {
          status: 'disconnected',
          lastHeartbeat: null,
          reconnectAttempts: 0,
        },
        activeSession: null,
        sessionHistory: [],
        isExpanded: false,
        selectedTab: 'response',

        // Actions
        setWebSocketStatus: (status) => set((state) => {
          state.websocket.status = status;
          if (status === 'connected') {
            state.websocket.reconnectAttempts = 0;
          }
        }),

        startSession: (query) => set((state) => {
          const sessionId = crypto.randomUUID();
          state.activeSession = {
            sessionId,
            query,
            startTime: new Date(),
            status: 'active',
            contextChunks: [],
            tokens: [],
            confidenceHistory: [],
            stages: [],
            reasoningSteps: [],
          };
        }),

        updateSession: (updates) => set((state) => {
          if (state.activeSession) {
            Object.assign(state.activeSession, updates);
          }
        }),

        addContextChunk: (chunk) => set((state) => {
          if (state.activeSession) {
            state.activeSession.contextChunks.push(chunk);
          }
        }),

        addToken: (token) => set((state) => {
          if (state.activeSession) {
            state.activeSession.tokens.push(token);
            state.activeSession.response = state.activeSession.tokens.join('');
          }
        }),

        updateConfidence: (point) => set((state) => {
          if (state.activeSession) {
            state.activeSession.confidenceHistory.push(point);
          }
        }),

        updateStage: (stage) => set((state) => {
          if (state.activeSession) {
            const existing = state.activeSession.stages.find(
              s => s.name === stage.name
            );
            if (existing) {
              Object.assign(existing, stage);
            } else {
              state.activeSession.stages.push(stage);
            }
          }
        }),

        addReasoningStep: (step) => set((state) => {
          if (state.activeSession) {
            state.activeSession.reasoningSteps.push(step);
          }
        }),

        endSession: (summary) => set((state) => {
          if (state.activeSession) {
            state.activeSession.status = 'completed';
            state.activeSession.totalDurationMs = summary.totalDurationMs;
            state.activeSession.finalConfidence = summary.finalConfidence;

            // Move to history
            state.sessionHistory.unshift(state.activeSession);
            if (state.sessionHistory.length > 50) {
              state.sessionHistory.pop();
            }
            state.activeSession = null;
          }
        }),

        cancelSession: () => set((state) => {
          if (state.activeSession) {
            state.activeSession.status = 'cancelled';
            state.sessionHistory.unshift(state.activeSession);
            state.activeSession = null;
          }
        }),

        clearHistory: () => set((state) => {
          state.sessionHistory = [];
        }),

        setExpanded: (expanded) => set((state) => {
          state.isExpanded = expanded;
        }),

        setTab: (tab) => set((state) => {
          state.selectedTab = tab;
        }),
      }))
    ),
    { name: 'hololoom-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectCurrentResponse = (state: HoloLoomState) =>
  state.activeSession?.response ?? '';

export const selectLatestConfidence = (state: HoloLoomState) => {
  const history = state.activeSession?.confidenceHistory ?? [];
  return history[history.length - 1]?.confidence ?? 0;
};

export const selectIsStreaming = (state: HoloLoomState) =>
  state.activeSession?.status === 'active';

export const selectTotalTokens = (state: HoloLoomState) =>
  state.activeSession?.tokens.length ?? 0;
```

### 2.4 WebSocket Client

```typescript
// src/core/WebSocketClient.ts

import { useHoloLoomStore } from './store';
import type { Query } from './types';

interface WebSocketOptions {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

export class HoloLoomWebSocketClient {
  private ws: WebSocket | null = null;
  private options: Required<WebSocketOptions>;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  constructor(options: WebSocketOptions) {
    this.options = {
      reconnectInterval: 3000,
      maxReconnectAttempts: 5,
      heartbeatInterval: 30000,
      ...options,
    };
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const store = useHoloLoomStore.getState();
      store.setWebSocketStatus('connecting');

      this.ws = new WebSocket(this.options.url);

      this.ws.onopen = () => {
        store.setWebSocketStatus('connected');
        this.startHeartbeat();
        resolve();
      };

      this.ws.onclose = () => {
        store.setWebSocketStatus('disconnected');
        this.stopHeartbeat();
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        store.setWebSocketStatus('error');
        reject(error);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(JSON.parse(event.data));
      };
    });
  }

  disconnect(): void {
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  async query(query: Query): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    const store = useHoloLoomStore.getState();
    store.startSession(query);

    this.ws.send(JSON.stringify({
      type: 'query',
      request_id: crypto.randomUUID(),
      query: {
        text: query.text,
        mode: query.mode,
        max_steps: query.maxSteps,
        context: query.context,
      },
    }));
  }

  cancel(): void {
    const store = useHoloLoomStore.getState();
    const sessionId = store.activeSession?.sessionId;

    if (sessionId && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'cancel',
        session_id: sessionId,
      }));
      store.cancelSession();
    }
  }

  private handleMessage(message: any): void {
    const store = useHoloLoomStore.getState();

    switch (message.type) {
      case 'stream_start':
        // Session already started in query()
        break;

      case 'context_chunk':
        store.addContextChunk({
          nodes: message.chunk.nodes,
          relevanceScores: message.chunk.relevance_scores,
          hopDistance: message.chunk.hop_distance,
          tokenCount: message.chunk.token_count,
          cumulativeTokens: message.chunk.cumulative_tokens,
          isFinal: message.is_final,
        });
        break;

      case 'token':
        store.addToken(message.token);
        break;

      case 'confidence_update':
        store.updateConfidence({
          timestamp: new Date(message.timestamp),
          confidence: message.confidence,
          epistemicConfidence: message.epistemic_confidence,
          source: message.source,
        });
        break;

      case 'stage_complete':
        store.updateStage({
          name: message.stage,
          startTime: new Date(),
          endTime: new Date(),
          durationMs: message.duration_ms,
          status: 'completed',
          metadata: message.metadata,
        });
        break;

      case 'reasoning_step':
        store.addReasoningStep({
          index: message.step.index,
          total: message.step.total,
          type: message.step.type,
          description: message.step.description,
          confidence: message.step.confidence,
          durationMs: message.step.duration_ms,
        });
        break;

      case 'stream_end':
        store.endSession({
          totalDurationMs: message.summary.total_duration_ms,
          tokensGenerated: message.summary.tokens_generated,
          finalConfidence: message.summary.final_confidence,
          spacetimeId: message.spacetime_id,
        });
        break;

      case 'error':
        store.updateSession({ status: 'error' });
        console.error('Stream error:', message.error);
        break;

      case 'heartbeat':
        useHoloLoomStore.setState((state) => ({
          websocket: {
            ...state.websocket,
            lastHeartbeat: new Date(message.server_time),
          },
        }));
        break;
    }
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, this.options.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect(): void {
    const store = useHoloLoomStore.getState();

    if (store.websocket.reconnectAttempts >= this.options.maxReconnectAttempts) {
      return;
    }

    useHoloLoomStore.setState((state) => ({
      websocket: {
        ...state.websocket,
        reconnectAttempts: state.websocket.reconnectAttempts + 1,
      },
    }));

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(console.error);
    }, this.options.reconnectInterval);
  }
}

// Singleton instance
let clientInstance: HoloLoomWebSocketClient | null = null;

export function getWebSocketClient(url?: string): HoloLoomWebSocketClient {
  if (!clientInstance && url) {
    clientInstance = new HoloLoomWebSocketClient({ url });
  }
  if (!clientInstance) {
    throw new Error('WebSocket client not initialized. Call with URL first.');
  }
  return clientInstance;
}
```

---

## 3. Confidence Visualization Component

### 3.1 ConfidenceGauge Component

```typescript
// src/components/ConfidenceGauge/ConfidenceGauge.tsx

import React, { useMemo, useEffect, useRef } from 'react';
import type { ConfidenceGaugeProps, ConfidencePoint } from '../../core/types';
import './ConfidenceGauge.css';

const SIZE_MAP = {
  sm: { width: 80, height: 80, strokeWidth: 6 },
  md: { width: 120, height: 120, strokeWidth: 8 },
  lg: { width: 160, height: 160, strokeWidth: 10 },
};

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  confidence,
  epistemicConfidence,
  size = 'md',
  showHistory = false,
  history = [],
  animated = true,
  className = '',
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const { width, height, strokeWidth } = SIZE_MAP[size];
  const radius = (Math.min(width, height) - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  // Calculate colors based on confidence level
  const getColor = (value: number): string => {
    if (value >= 0.8) return 'var(--confidence-high)';
    if (value >= 0.6) return 'var(--confidence-medium)';
    if (value >= 0.4) return 'var(--confidence-low)';
    return 'var(--confidence-very-low)';
  };

  // Calculate stroke dashoffset for arc
  const dashOffset = useMemo(() => {
    const progress = Math.min(Math.max(confidence, 0), 1);
    return circumference * (1 - progress * 0.75); // 270 degree arc
  }, [confidence, circumference]);

  // Rating text
  const rating = useMemo(() => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Good';
    if (confidence >= 0.4) return 'Moderate';
    return 'Low';
  }, [confidence]);

  // Sparkline for history
  const sparklinePath = useMemo(() => {
    if (!showHistory || history.length < 2) return '';

    const maxPoints = 20;
    const points = history.slice(-maxPoints);
    const sparkWidth = width * 0.6;
    const sparkHeight = 20;
    const xStep = sparkWidth / (points.length - 1);

    return points
      .map((p, i) => {
        const x = i * xStep;
        const y = sparkHeight - p.confidence * sparkHeight;
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  }, [history, showHistory, width]);

  return (
    <div
      className={`confidence-gauge ${className}`}
      role="meter"
      aria-valuenow={Math.round(confidence * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Confidence: ${Math.round(confidence * 100)}%`}
    >
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="confidence-gauge__svg"
      >
        {/* Background arc */}
        <circle
          cx={width / 2}
          cy={height / 2}
          r={radius}
          fill="none"
          stroke="var(--gauge-bg)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${circumference * 0.75} ${circumference * 0.25}`}
          transform={`rotate(135 ${width / 2} ${height / 2})`}
        />

        {/* Confidence arc */}
        <circle
          cx={width / 2}
          cy={height / 2}
          r={radius}
          fill="none"
          stroke={getColor(confidence)}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          transform={`rotate(135 ${width / 2} ${height / 2})`}
          className={animated ? 'confidence-gauge__arc--animated' : ''}
        />

        {/* Epistemic confidence ring (if provided) */}
        {epistemicConfidence !== undefined && (
          <circle
            cx={width / 2}
            cy={height / 2}
            r={radius - strokeWidth - 2}
            fill="none"
            stroke={getColor(epistemicConfidence)}
            strokeWidth={2}
            strokeLinecap="round"
            strokeDasharray={circumference * 0.75}
            strokeDashoffset={circumference * 0.75 * (1 - epistemicConfidence)}
            transform={`rotate(135 ${width / 2} ${height / 2})`}
            opacity={0.6}
          />
        )}

        {/* Center text */}
        <text
          x={width / 2}
          y={height / 2 - 5}
          textAnchor="middle"
          dominantBaseline="middle"
          className="confidence-gauge__value"
          fill="var(--text-primary)"
        >
          {Math.round(confidence * 100)}%
        </text>

        <text
          x={width / 2}
          y={height / 2 + 15}
          textAnchor="middle"
          dominantBaseline="middle"
          className="confidence-gauge__rating"
          fill={getColor(confidence)}
        >
          {rating}
        </text>
      </svg>

      {/* Sparkline history */}
      {showHistory && history.length >= 2 && (
        <svg
          width={width * 0.6}
          height={24}
          className="confidence-gauge__sparkline"
          viewBox={`0 0 ${width * 0.6} 20`}
        >
          <path
            d={sparklinePath}
            fill="none"
            stroke="var(--text-secondary)"
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {/* Endpoint dot */}
          <circle
            cx={width * 0.6}
            cy={20 - history[history.length - 1].confidence * 20}
            r={3}
            fill={getColor(history[history.length - 1].confidence)}
          />
        </svg>
      )}
    </div>
  );
};

export default ConfidenceGauge;
```

### 3.2 ConfidenceGauge CSS

```css
/* src/components/ConfidenceGauge/ConfidenceGauge.css */

:root {
  --confidence-high: #10B981;      /* Emerald 500 */
  --confidence-medium: #F59E0B;    /* Amber 500 */
  --confidence-low: #EF4444;       /* Red 500 */
  --confidence-very-low: #6B7280;  /* Gray 500 */
  --gauge-bg: #E5E7EB;             /* Gray 200 */
  --text-primary: #111827;         /* Gray 900 */
  --text-secondary: #6B7280;       /* Gray 500 */
}

.dark {
  --gauge-bg: #374151;             /* Gray 700 */
  --text-primary: #F9FAFB;         /* Gray 50 */
  --text-secondary: #9CA3AF;       /* Gray 400 */
}

.confidence-gauge {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.confidence-gauge__svg {
  display: block;
}

.confidence-gauge__arc--animated {
  transition: stroke-dashoffset 0.6s cubic-bezier(0.4, 0, 0.2, 1),
              stroke 0.3s ease;
}

.confidence-gauge__value {
  font-size: 1.5rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  font-family: var(--font-mono, monospace);
}

.confidence-gauge__rating {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.confidence-gauge__sparkline {
  opacity: 0.8;
  transition: opacity 0.2s ease;
}

.confidence-gauge:hover .confidence-gauge__sparkline {
  opacity: 1;
}

/* Size variants */
.confidence-gauge--sm .confidence-gauge__value {
  font-size: 1rem;
}

.confidence-gauge--lg .confidence-gauge__value {
  font-size: 2rem;
}

/* Pulse animation for active streaming */
@keyframes confidence-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.confidence-gauge--streaming .confidence-gauge__arc--animated {
  animation: confidence-pulse 1.5s ease-in-out infinite;
}
```

---

## 4. Streaming Response Component

### 4.1 StreamingResponse Component

```typescript
// src/components/StreamingResponse/StreamingResponse.tsx

import React, { useEffect, useRef, useMemo } from 'react';
import { useHoloLoomStore, selectIsStreaming } from '../../core/store';
import type { StreamingResponseProps } from '../../core/types';
import { TokenRenderer } from './TokenRenderer';
import './StreamingResponse.css';

export const StreamingResponse: React.FC<StreamingResponseProps> = ({
  session,
  onComplete,
  showMetadata = true,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const isStreaming = useHoloLoomStore(selectIsStreaming);
  const prevSessionRef = useRef(session);

  // Auto-scroll during streaming
  useEffect(() => {
    if (isStreaming && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [session?.tokens.length, isStreaming]);

  // Call onComplete when session completes
  useEffect(() => {
    if (
      prevSessionRef.current?.status === 'active' &&
      session?.status === 'completed' &&
      session?.response &&
      onComplete
    ) {
      onComplete(session.response);
    }
    prevSessionRef.current = session;
  }, [session?.status, session?.response, onComplete]);

  // Format duration
  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  // Tokens per second
  const tokensPerSecond = useMemo(() => {
    if (!session?.totalDurationMs || !session?.tokens.length) return null;
    return (session.tokens.length / (session.totalDurationMs / 1000)).toFixed(1);
  }, [session?.totalDurationMs, session?.tokens.length]);

  if (!session) {
    return (
      <div className={`streaming-response streaming-response--empty ${className}`}>
        <p className="streaming-response__placeholder">
          Enter a query to begin...
        </p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`streaming-response ${className}`}
      role="log"
      aria-live="polite"
      aria-label="AI response"
    >
      {/* Response content */}
      <div className="streaming-response__content">
        <TokenRenderer
          tokens={session.tokens}
          isStreaming={isStreaming}
        />

        {/* Streaming cursor */}
        {isStreaming && (
          <span className="streaming-response__cursor" aria-hidden="true">
            |
          </span>
        )}
      </div>

      {/* Metadata footer */}
      {showMetadata && session.status === 'completed' && (
        <div className="streaming-response__metadata">
          <span className="streaming-response__stat">
            <span className="streaming-response__stat-label">Duration:</span>
            <span className="streaming-response__stat-value">
              {formatDuration(session.totalDurationMs ?? 0)}
            </span>
          </span>

          <span className="streaming-response__stat">
            <span className="streaming-response__stat-label">Tokens:</span>
            <span className="streaming-response__stat-value">
              {session.tokens.length}
            </span>
          </span>

          {tokensPerSecond && (
            <span className="streaming-response__stat">
              <span className="streaming-response__stat-label">Speed:</span>
              <span className="streaming-response__stat-value">
                {tokensPerSecond} tok/s
              </span>
            </span>
          )}

          <span className="streaming-response__stat">
            <span className="streaming-response__stat-label">Confidence:</span>
            <span
              className="streaming-response__stat-value"
              style={{
                color: session.finalConfidence && session.finalConfidence >= 0.7
                  ? 'var(--confidence-high)'
                  : 'var(--confidence-medium)'
              }}
            >
              {session.finalConfidence
                ? `${Math.round(session.finalConfidence * 100)}%`
                : 'N/A'}
            </span>
          </span>
        </div>
      )}

      {/* Error state */}
      {session.status === 'error' && (
        <div className="streaming-response__error" role="alert">
          An error occurred while processing your query.
        </div>
      )}
    </div>
  );
};

export default StreamingResponse;
```

### 4.2 TokenRenderer Component

```typescript
// src/components/StreamingResponse/TokenRenderer.tsx

import React, { memo, useMemo } from 'react';

interface TokenRendererProps {
  tokens: string[];
  isStreaming: boolean;
}

export const TokenRenderer: React.FC<TokenRendererProps> = memo(({
  tokens,
  isStreaming,
}) => {
  // Join tokens into full text
  const text = useMemo(() => tokens.join(''), [tokens]);

  // Parse markdown-like formatting
  const formattedContent = useMemo(() => {
    // Simple markdown parsing
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let inCodeBlock = false;
    let codeBlockContent = '';
    let codeBlockLang = '';

    lines.forEach((line, lineIndex) => {
      // Code block detection
      if (line.startsWith('```')) {
        if (!inCodeBlock) {
          inCodeBlock = true;
          codeBlockLang = line.slice(3).trim();
          codeBlockContent = '';
        } else {
          elements.push(
            <pre key={`code-${lineIndex}`} className="token-code-block">
              <code className={`language-${codeBlockLang}`}>
                {codeBlockContent}
              </code>
            </pre>
          );
          inCodeBlock = false;
        }
        return;
      }

      if (inCodeBlock) {
        codeBlockContent += (codeBlockContent ? '\n' : '') + line;
        return;
      }

      // Headers
      if (line.startsWith('### ')) {
        elements.push(
          <h3 key={lineIndex} className="token-heading-3">
            {line.slice(4)}
          </h3>
        );
        return;
      }

      if (line.startsWith('## ')) {
        elements.push(
          <h2 key={lineIndex} className="token-heading-2">
            {line.slice(3)}
          </h2>
        );
        return;
      }

      if (line.startsWith('# ')) {
        elements.push(
          <h1 key={lineIndex} className="token-heading-1">
            {line.slice(2)}
          </h1>
        );
        return;
      }

      // Lists
      if (line.match(/^[-*] /)) {
        elements.push(
          <li key={lineIndex} className="token-list-item">
            {formatInlineMarkdown(line.slice(2))}
          </li>
        );
        return;
      }

      // Numbered lists
      if (line.match(/^\d+\. /)) {
        elements.push(
          <li key={lineIndex} className="token-list-item token-list-item--numbered">
            {formatInlineMarkdown(line.replace(/^\d+\. /, ''))}
          </li>
        );
        return;
      }

      // Empty line = paragraph break
      if (line.trim() === '') {
        elements.push(<br key={lineIndex} />);
        return;
      }

      // Regular paragraph
      elements.push(
        <p key={lineIndex} className="token-paragraph">
          {formatInlineMarkdown(line)}
        </p>
      );
    });

    // Handle unclosed code block
    if (inCodeBlock && codeBlockContent) {
      elements.push(
        <pre key="code-unclosed" className="token-code-block token-code-block--streaming">
          <code className={`language-${codeBlockLang}`}>
            {codeBlockContent}
          </code>
        </pre>
      );
    }

    return elements;
  }, [text]);

  return <>{formattedContent}</>;
});

// Format inline markdown (bold, italic, code)
function formatInlineMarkdown(text: string): React.ReactNode {
  // Split on inline code first
  const parts = text.split(/(`[^`]+`)/g);

  return parts.map((part, i) => {
    // Inline code
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={i} className="token-inline-code">
          {part.slice(1, -1)}
        </code>
      );
    }

    // Bold
    let result: React.ReactNode = part;
    if (part.includes('**')) {
      const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
      result = boldParts.map((bp, j) => {
        if (bp.startsWith('**') && bp.endsWith('**')) {
          return <strong key={j}>{bp.slice(2, -2)}</strong>;
        }
        return bp;
      });
    }

    return <React.Fragment key={i}>{result}</React.Fragment>;
  });
}

TokenRenderer.displayName = 'TokenRenderer';
```

---

## 5. Implementation Timeline

### Week 1: Core Infrastructure

| Day | Tasks | Owner | Status |
|-----|-------|-------|--------|
| 1 | WebSocket streaming endpoint (server) | Backend | Pending |
| 1 | WebSocket client setup (React) | Frontend | Pending |
| 2 | Zustand store implementation | Frontend | Pending |
| 2 | Message handlers & state sync | Frontend | Pending |
| 3 | ConfidenceGauge component | Frontend | Pending |
| 3 | ConfidenceGauge Storybook stories | Frontend | Pending |
| 4 | StreamingResponse component | Frontend | Pending |
| 4 | TokenRenderer with markdown | Frontend | Pending |
| 5 | Integration testing | Both | Pending |

### Week 2: Refinement & Additional Components

| Day | Tasks | Owner | Status |
|-----|-------|-------|--------|
| 6 | PipelineWaterfall component | Frontend | Pending |
| 6 | ReasoningTree component | Frontend | Pending |
| 7 | ContextExplorer component | Frontend | Pending |
| 7 | Accessibility audit | Frontend | Pending |
| 8 | Performance optimization | Both | Pending |
| 8 | Documentation | Both | Pending |

---

## 6. Testing Strategy

### 6.1 Unit Tests

```typescript
// Example test for ConfidenceGauge
describe('ConfidenceGauge', () => {
  it('renders confidence value correctly', () => {
    render(<ConfidenceGauge confidence={0.85} />);
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('High')).toBeInTheDocument();
  });

  it('updates color based on confidence level', () => {
    const { rerender } = render(<ConfidenceGauge confidence={0.9} />);
    expect(screen.getByText('High')).toHaveStyle({ color: 'var(--confidence-high)' });

    rerender(<ConfidenceGauge confidence={0.4} />);
    expect(screen.getByText('Moderate')).toHaveStyle({ color: 'var(--confidence-low)' });
  });

  it('renders sparkline when history provided', () => {
    const history = [
      { confidence: 0.8, timestamp: new Date() },
      { confidence: 0.85, timestamp: new Date() },
    ];
    render(<ConfidenceGauge confidence={0.85} showHistory history={history} />);
    expect(document.querySelector('.confidence-gauge__sparkline')).toBeInTheDocument();
  });

  it('has correct ARIA attributes', () => {
    render(<ConfidenceGauge confidence={0.75} />);
    const gauge = screen.getByRole('meter');
    expect(gauge).toHaveAttribute('aria-valuenow', '75');
    expect(gauge).toHaveAttribute('aria-valuemin', '0');
    expect(gauge).toHaveAttribute('aria-valuemax', '100');
  });
});
```

### 6.2 Integration Tests

```typescript
// WebSocket integration test
describe('WebSocket Integration', () => {
  let server: WS;
  let client: HoloLoomWebSocketClient;

  beforeEach(async () => {
    server = new WS('ws://localhost:8000/ws/stream');
    client = new HoloLoomWebSocketClient({ url: 'ws://localhost:8000/ws/stream' });
    await client.connect();
  });

  afterEach(() => {
    client.disconnect();
    server.close();
  });

  it('receives stream_start message on query', async () => {
    const store = useHoloLoomStore.getState();

    await client.query({
      text: 'Test query',
      mode: 'direct',
      maxSteps: 3,
    });

    server.send(JSON.stringify({
      type: 'stream_start',
      session_id: 'test-session',
      query: { text: 'Test query' },
    }));

    await waitFor(() => {
      expect(store.activeSession).not.toBeNull();
      expect(store.activeSession?.query.text).toBe('Test query');
    });
  });

  it('accumulates tokens during streaming', async () => {
    await client.query({ text: 'Test', mode: 'direct', maxSteps: 1 });

    const tokens = ['Hello', ' ', 'world', '!'];
    for (const token of tokens) {
      server.send(JSON.stringify({
        type: 'token',
        token,
        cumulative_text: tokens.slice(0, tokens.indexOf(token) + 1).join(''),
      }));
    }

    await waitFor(() => {
      const store = useHoloLoomStore.getState();
      expect(store.activeSession?.response).toBe('Hello world!');
    });
  });
});
```

---

## 7. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to First Token | <100ms | WebSocket message latency |
| Render Performance | 60fps | React DevTools Profiler |
| Bundle Size (core) | <50KB gzip | Webpack Bundle Analyzer |
| Memory Usage | <50MB | Chrome DevTools Memory |
| Accessibility Score | 100 | Lighthouse |
| WebSocket Reconnect | <3s | Network monitor |

---

## 8. Dependencies

### Production Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "zustand": "^4.4.0",
    "immer": "^10.0.0"
  }
}
```

### Development Dependencies

```json
{
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "tailwindcss": "^3.4.0",
    "@testing-library/react": "^14.1.0",
    "@testing-library/jest-dom": "^6.1.0",
    "vitest": "^1.0.0",
    "@storybook/react-vite": "^7.6.0"
  }
}
```

---

## 9. Acceptance Criteria

### Phase 1 Complete When:

- [ ] WebSocket streaming endpoint deployed and tested
- [ ] WebSocket client connects, handles all message types
- [ ] Zustand store manages all streaming state
- [ ] ConfidenceGauge renders with animation, history, accessibility
- [ ] StreamingResponse displays tokens with markdown formatting
- [ ] All unit tests pass (>80% coverage)
- [ ] Integration tests pass
- [ ] Performance targets met
- [ ] Accessibility audit passes (WCAG 2.1 AA)
- [ ] Documentation complete
- [ ] Storybook stories for all components

---

## Appendix A: File Locations

```
mythRL/
├── hololoom/
│   └── server/
│       ├── agentic_api.py          # Existing (add streaming endpoint)
│       └── streaming_api.py        # New WebSocket streaming
│
└── ui/
    ├── PHASE_1_TECHNICAL_SPEC.md   # This document
    │
    └── hololoom-components/        # New React component library
        ├── package.json
        ├── src/
        │   ├── core/
        │   │   ├── WebSocketClient.ts
        │   │   ├── store.ts
        │   │   └── types.ts
        │   └── components/
        │       ├── ConfidenceGauge/
        │       └── StreamingResponse/
        └── stories/
```

---

*Document Version: 1.0.0 | Created: 2025-11-30 | Status: Implementation Ready*
