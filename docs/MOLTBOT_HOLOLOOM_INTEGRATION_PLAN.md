# Moltbot + HoloLoom Integration Plan

**Date**: 2026-01-28
**Updated**: 2026-01-28 (expanded with implementation details)
**Status**: Proposal
**Authors**: blakechasteen

---

## Executive Summary

**Moltbot** is a multi-channel personal AI assistant (TypeScript, 61.5k GitHub stars) that routes messages from WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Matrix, and Teams through a Gateway daemon to LLM providers. It has broad reach but no persistent memory, no learning, and no semantic safety layer.

**HoloLoom** is a neural decision-making system (Python, ~165K LOC) with 11 memory systems, 7 learning loops, alignment guardrails, Level 4 agentic RAG, and SAE-based interpretability. It has deep intelligence but limited distribution channels (FastAPI, VS Code, CLI).

**The integration** creates a symbiotic system: HoloLoom becomes the "brain stem" providing memory, reasoning, safety, and learning. Moltbot becomes the "nervous system" providing multi-channel I/O, real-world tool access, and user session management.

---

## Value Exchange

### What Moltbot Gains

| Capability | Current State | With HoloLoom |
|------------|--------------|---------------|
| **Memory** | None between sessions | 11 memory systems, cross-session, cross-channel |
| **Reasoning** | Raw LLM calls | 4 agentic modes (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE) |
| **Safety** | Docker sandbox only | Alignment framework: risk gating, deception detection, audit trail |
| **Learning** | None | 7 parallel loops: Thompson Sampling, PPO, hot patterns, adaptive routing |
| **Interpretability** | Opaque LLM calls | Dark Trace SAE decomposition, 228D semantic space |

### What HoloLoom Gains

| Capability | Current State | With Moltbot |
|------------|--------------|-------------|
| **Distribution** | FastAPI / VS Code / CLI | 8+ messaging platforms simultaneously |
| **Tools** | None (reasoning only) | CDP browser, camera, screen, cron, webhooks, Gmail |
| **User identity** | Per-query sessions | Per-user DM pairing across channels |
| **Feedback volume** | Manual testing | Continuous multi-channel user interactions |

---

## Technical Bridge

```
Moltbot (Node 22, TypeScript)                  HoloLoom (Python, FastAPI)
+---------------------------------+            +--------------------------------+
|  Gateway Daemon                 |            |  Unified Server :8000          |
|  ws://127.0.0.1:18789          |            |                                |
|                                 |  HTTP/REST |  /lite/experience              |
|  +---------------------------+  | ---------> |  /lite/recall                  |
|  | hololoom-bridge extension |  |  JSON      |  /lite/reflect                 |
|  | (TypeScript SDK)          |  | <--------- |  /lite/reason                  |
|  +---------------------------+  |            |  /lite/query                   |
|                                 |            |  /safety/gate                  |
|  Channels:                      |  WebSocket |  /safety/audit-trail           |
|  WhatsApp, Telegram, Slack,     | ---------> |                                |
|  Discord, Signal, iMessage,     |  streaming |  Streaming API :8001           |
|  Matrix, Teams                  |            |  ws://127.0.0.1:8001/ws        |
|                                 |            |                                |
|  Tools:                         |            |  Backends:                     |
|  CDP Browser, Device Nodes,     |            |  Neo4j (graph)                 |
|  Cron, Webhooks, Gmail          |            |  Qdrant (vector)               |
+---------------------------------+            +--------------------------------+
```

**Primary path**: Moltbot extension makes HTTP calls to HoloLoom's FastAPI server using the existing TypeScript SDK (`@hololoom/sdk`).

**Streaming path**: For long-running reasoning (RESEARCH/PLAN_EXECUTE), use HoloLoom's WebSocket streaming API with Moltbot's "typing" indicators.

### Existing Infrastructure

The following components already exist and are ready to use:

| Component | Location | Status |
|-----------|----------|--------|
| TypeScript SDK | `sdk/typescript/src/client.ts` | Complete — `experience()`, `recall()`, `reflect()`, `reason()`, `query()` with retry + backoff |
| SDK Types | `sdk/typescript/src/types/index.ts` | Complete — `Memory`, `Spacetime`, `RecallStrategy`, `ReasoningMode`, etc. |
| Lite API Server | `HoloLoom/server/unified_server.py` | Complete — `/lite/*`, `/safety/*`, `/learning/*`, `/viz/*` |
| Agentic API | `HoloLoom/server/agentic_api.py` | Complete — `QueryRequest`/`AgenticResponse` models |
| Streaming Server | `HoloLoom/server/streaming_api.py` | Complete — WebSocket on :8001 with typed message protocol |
| Safety Gate | `HoloLoom/alignment/safety_guardrails.py` | Complete — `ActionRequest` -> `SafetyDecision` (0.103ms) |
| Audit Trail | `HoloLoom/alignment/audit_trail.py` | Complete — Searchable, temporal queries |
| ChatOps Handlers | `HoloLoom/chatops/handlers/` | Complete — 50+ handlers, handler registry |
| SDK Package | `sdk/typescript/package.json` | Complete — `@hololoom/sdk` v1.0.0, zero dependencies |

---

## Phase 1: Memory Bridge

**Goal**: Give Moltbot persistent, cross-session, cross-channel memory.

### 1.1 Moltbot HoloLoom Skill

**File**: `~/clawd/skills/hololoom-memory/SKILL.md`

Moltbot's skill system loads `SKILL.md` files that describe tool capabilities to the LLM agent. This skill registers three tools that the Moltbot LLM can invoke:

```markdown
# HoloLoom Memory

Connect to HoloLoom for persistent memory across sessions and channels.

## Tools

### hololoom_remember

Store information in long-term memory.

Parameters:
- content (string, required): The information to remember
- tags (string[], optional): Tags for categorization
- importance (number, optional): 0.0-1.0 importance score

### hololoom_recall

Retrieve relevant memories for context.

Parameters:
- query (string, required): What to search for
- limit (number, optional): Max results (default: 5)
- strategy (string, optional): RECENT | SIMILAR | CONNECTED | BALANCED

### hololoom_reflect

Provide feedback on memory quality.

Parameters:
- memoryIds (string[], required): Which memories were used
- helpful (boolean, required): Was the recall helpful?
- quality (number, optional): 0.0-1.0 quality score
```

### 1.2 Bridge Extension

**File**: `extensions/hololoom-bridge/src/index.ts`

The bridge extension is the runtime code that implements the skill's tools. It imports the existing `@hololoom/sdk` and handles the Moltbot <-> HoloLoom translation:

```typescript
import { HoloLoomClient } from '@hololoom/sdk';
import type { Memory, RecallStrategy } from '@hololoom/sdk/types';

// --- Configuration ---

interface HoloLoomBridgeConfig {
  baseUrl: string;           // HoloLoom API (default: http://127.0.0.1:8000)
  apiKey?: string;           // Optional auth token
  autoRemember: boolean;     // Store all messages automatically (default: true)
  autoRecall: boolean;       // Inject context before LLM (default: true)
  recallLimit: number;       // Max memories per recall (default: 5)
  recallStrategy: string;    // Default: 'BALANCED'
}

const DEFAULT_CONFIG: HoloLoomBridgeConfig = {
  baseUrl: process.env.HOLOLOOM_API_URL || 'http://127.0.0.1:8000',
  autoRemember: true,
  autoRecall: true,
  recallLimit: 5,
  recallStrategy: 'BALANCED',
};

// --- User Identity ---

interface MoltbotIdentity {
  userId: string;         // Stable cross-channel user ID
  channel: string;        // whatsapp | telegram | slack | discord | ...
  sessionId: string;      // Current session ID
  displayName?: string;   // Human-readable name
}

function buildMetadata(identity: MoltbotIdentity) {
  return {
    moltbot_user_id: identity.userId,
    moltbot_channel: identity.channel,
    moltbot_session_id: identity.sessionId,
    moltbot_display_name: identity.displayName,
    timestamp: new Date().toISOString(),
  };
}

// --- Tool Implementations ---

// hololoom_remember: Store a memory
async function remember(
  client: HoloLoomClient,
  identity: MoltbotIdentity,
  content: string,
  tags?: string[],
  importance?: number,
) {
  return client.experience({
    content,
    metadata: {
      ...buildMetadata(identity),
      tags,
      importance: importance ?? 0.5,
    },
  });
}

// hololoom_recall: Retrieve relevant memories
async function recall(
  client: HoloLoomClient,
  identity: MoltbotIdentity,
  query: string,
  limit: number = 5,
  strategy: RecallStrategy = 'BALANCED',
) {
  const result = await client.recall(query, {
    strategy,
    limit,
    metadata: { moltbot_user_id: identity.userId },
  });
  return result.data.memories;
}

// hololoom_reflect: Provide feedback
async function reflect(
  client: HoloLoomClient,
  memoryIds: string[],
  helpful: boolean,
  quality?: number,
) {
  return client.reflect({
    memoryIds,
    feedback: {
      helpful,
      quality: quality ?? (helpful ? 0.8 : 0.2),
    },
  });
}
```

### 1.3 Message Flow (Auto-Mode)

When `autoRemember` and `autoRecall` are enabled, the extension intercepts every message:

```
1. User sends "What did we discuss about the API redesign?"
   on WhatsApp

2. Extension: auto-remember
   POST /lite/experience
   {
     content: "User asked: What did we discuss about the API redesign?",
     metadata: {
       moltbot_user_id: "user_abc",
       moltbot_channel: "whatsapp",
       moltbot_session_id: "sess_xyz",
       timestamp: "2026-01-28T14:30:00Z"
     }
   }

3. Extension: auto-recall
   POST /lite/recall
   {
     query: "API redesign discussion",
     strategy: "CONNECTED",
     limit: 5,
     metadata: { moltbot_user_id: "user_abc" }
   }

   Response:
   {
     memories: [
       { id: "mem_1", content: "Decided to use REST over GraphQL for v2...",
         relevance: 0.92, metadata: { moltbot_channel: "slack" } },
       { id: "mem_2", content: "API rate limiting set to 100 req/min...",
         relevance: 0.85, metadata: { moltbot_channel: "slack" } },
       { id: "mem_3", content: "Blake proposed pagination with cursor-based...",
         relevance: 0.78, metadata: { moltbot_channel: "telegram" } }
     ]
   }

4. Extension: inject context into Moltbot's LLM prompt
   System prompt addition:
   """
   [Memory Context — retrieved from past conversations]
   - (Slack, 3 days ago) Decided to use REST over GraphQL for v2...
   - (Slack, 3 days ago) API rate limiting set to 100 req/min...
   - (Telegram, 5 days ago) Blake proposed pagination with cursor-based...
   [End Memory Context]
   """

5. Moltbot LLM generates response WITH memory context
   "Based on our previous discussions: We decided to go with REST
    over GraphQL for v2. The rate limiting was set at 100 req/min,
    and Blake proposed cursor-based pagination..."

6. Extension: auto-remember the response
   POST /lite/experience
   {
     content: "Bot responded with API redesign summary: REST over GraphQL...",
     metadata: { moltbot_user_id: "user_abc", moltbot_channel: "whatsapp", role: "assistant" }
   }

7. Later: user reacts with 👍 emoji
   Extension: auto-reflect
   POST /lite/reflect
   {
     memoryIds: ["mem_1", "mem_2", "mem_3"],
     feedback: { helpful: true, quality: 0.9 }
   }
```

### 1.4 User Identity Mapping

Moltbot identifies users differently per channel. The bridge maintains a stable cross-channel identity:

```typescript
// Identity resolution strategy:
// 1. Check if Moltbot provides a unified user ID (from DM pairing)
// 2. Fall back to channel-specific ID with channel prefix
// 3. Store mapping in HoloLoom metadata for future cross-channel resolution

interface IdentityMapping {
  // Moltbot's DM pairing gives us a stable ID
  pairedUserId?: string;

  // Channel-specific IDs (fallback)
  channelIds: {
    whatsapp?: string;   // phone number hash
    telegram?: string;   // telegram user ID
    slack?: string;      // slack member ID
    discord?: string;    // discord user ID
    signal?: string;     // signal phone hash
    matrix?: string;     // @user:homeserver
  };
}

function resolveUserId(session: MoltbotSession): string {
  // Prefer Moltbot's paired identity (stable across channels)
  if (session.pairedUserId) {
    return `moltbot:${session.pairedUserId}`;
  }
  // Fall back to channel-specific
  return `moltbot:${session.channel}:${session.channelUserId}`;
}
```

### 1.5 HoloLoom Server: Lite API Endpoints

The existing `unified_server.py` already exposes the Lite API. The following endpoints are used by Phase 1:

| Endpoint | Method | SDK Method | Purpose |
|----------|--------|------------|---------|
| `/lite/experience` | POST | `client.experience()` | Store memory |
| `/lite/experience/batch` | POST | `client.experienceBatch()` | Batch store |
| `/lite/recall` | POST | `client.recall()` | Retrieve memories |
| `/lite/reflect` | POST | `client.reflect()` | Learning feedback |
| `/health` | GET | `client.health()` | Health check |

**No changes needed on the Python side for Phase 1.** The Lite API already supports metadata filtering, recall strategies, and feedback.

### 1.6 Graceful Degradation

If HoloLoom is unreachable, Moltbot continues working in stateless mode:

```typescript
async function withFallback<T>(
  fn: () => Promise<T>,
  fallback: T,
  label: string,
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    // SDK already retries 3x with exponential backoff
    // If still failing, degrade gracefully
    console.warn(`[hololoom-bridge] ${label} failed, degrading to stateless mode`);
    return fallback;
  }
}

// Usage:
const memories = await withFallback(
  () => recall(client, identity, query),
  [],  // No memories = stateless mode
  'recall',
);
```

### 1.7 File Structure

```
extensions/hololoom-bridge/
├── package.json              # Depends on @hololoom/sdk
├── tsconfig.json
├── src/
│   ├── index.ts              # Extension entry point + lifecycle
│   ├── config.ts             # HoloLoomBridgeConfig
│   ├── identity.ts           # User identity resolution
│   ├── tools/
│   │   ├── remember.ts       # hololoom_remember tool
│   │   ├── recall.ts         # hololoom_recall tool
│   │   └── reflect.ts        # hololoom_reflect tool
│   ├── hooks/
│   │   ├── pre-message.ts    # Auto-recall before LLM
│   │   └── post-message.ts   # Auto-remember after response
│   └── context-formatter.ts  # Format memories for LLM injection
├── skills/
│   └── hololoom-memory/
│       └── SKILL.md          # Skill definition for Moltbot
└── test/
    ├── remember.test.ts
    ├── recall.test.ts
    └── identity.test.ts
```

### 1.8 Configuration

Users configure the bridge in `~/.clawdbot/moltbot.json`:

```json
{
  "extensions": {
    "hololoom-bridge": {
      "enabled": true,
      "baseUrl": "http://127.0.0.1:8000",
      "autoRemember": true,
      "autoRecall": true,
      "recallLimit": 5,
      "recallStrategy": "BALANCED"
    }
  }
}
```

### 1.9 Testing Plan

| Test | What It Validates |
|------|------------------|
| `remember.test.ts` | Stores message with correct metadata, handles errors |
| `recall.test.ts` | Retrieves memories, formats context, respects limit |
| `reflect.test.ts` | Sends feedback, maps emoji reactions correctly |
| `identity.test.ts` | Cross-channel identity resolution, paired vs unpaired |
| `fallback.test.ts` | Graceful degradation when HoloLoom unreachable |
| `e2e: cross-channel` | Store on Slack, recall on WhatsApp |
| `e2e: persistence` | Memories survive Moltbot restart |

### 1.10 Success Criteria

- A WhatsApp conversation recalls context from a Slack message sent days ago
- Memory persists across Moltbot restarts
- Zero-config for end users (default config works)
- <50ms added latency per message (recall is cached after first hit)

---

## Phase 2: Agentic Reasoning Bridge

**Goal**: Give Moltbot access to HoloLoom's 4 reasoning modes for complex queries.

### 2.1 Reasoning Skill

**File**: `extensions/hololoom-bridge/skills/hololoom-reasoning/SKILL.md`

```markdown
# HoloLoom Reasoning

Deep multi-step reasoning powered by HoloLoom's agentic orchestrator.

## Tools

### hololoom_reason

Perform multi-step reasoning with verification.

Parameters:
- query (string, required): The question or task
- mode (string, optional): DIRECT | VERIFY | RESEARCH | PLAN_EXECUTE
- maxSteps (number, optional): Maximum reasoning steps (1-20, default: 5)

Use VERIFY for factual claims. Use RESEARCH for open-ended exploration.
Use PLAN_EXECUTE for multi-step tasks. Omit mode for auto-detection.

### hololoom_query

Full weaving cycle query with complete provenance.

Parameters:
- query (string, required): The question
- includeProvenance (boolean, optional): Include reasoning trace (default: false)
```

### 2.2 Auto Mode Selection

The extension detects query complexity and selects the appropriate mode:

```typescript
import type { ReasoningMode } from '@hololoom/sdk/types';

interface ModeSelection {
  mode: ReasoningMode;
  confidence: number;
  reason: string;
}

function selectMode(query: string): ModeSelection {
  const q = query.toLowerCase().trim();

  // TRIVIAL — don't even call HoloLoom
  if (/^(hi|hello|hey|thanks|ok|bye)[\s!?.]*$/i.test(q)) {
    return { mode: 'DIRECT', confidence: 0.99, reason: 'greeting/ack' };
  }

  // PLAN_EXECUTE — task decomposition
  if (/\b(step[- ]by[- ]step|how (do|can|should) i|plan|build|create|implement)\b/i.test(q)) {
    return { mode: 'PLAN_EXECUTE', confidence: 0.85, reason: 'planning keywords' };
  }

  // VERIFY — fact checking
  if (/\b(is it true|verify|fact[- ]check|confirm|really|actually)\b/i.test(q)) {
    return { mode: 'VERIFY', confidence: 0.85, reason: 'verification keywords' };
  }

  // RESEARCH — open-ended exploration
  if (/\b(compare|analyze|tradeoffs?|pros and cons|comprehensive|everything about)\b/i.test(q)) {
    return { mode: 'RESEARCH', confidence: 0.80, reason: 'research keywords' };
  }

  // Word count heuristic
  const wordCount = q.split(/\s+/).length;
  if (wordCount > 20) {
    return { mode: 'RESEARCH', confidence: 0.60, reason: 'long query' };
  }

  // Default
  return { mode: 'DIRECT', confidence: 0.70, reason: 'default' };
}
```

Note: HoloLoom's server-side query classifier (`HoloLoom/routing/`) also performs classification. The client-side selection above is a fast pre-filter; the server may override it.

### 2.3 Streaming Protocol

For modes that take >500ms (VERIFY, RESEARCH, PLAN_EXECUTE), the extension connects to HoloLoom's WebSocket streaming API to deliver progressive responses:

```typescript
// WebSocket message types from HoloLoom/server/streaming_api.py
type StreamMessageType =
  | 'stream_start'
  | 'context_chunk'
  | 'token'
  | 'confidence_update'
  | 'stage_complete'
  | 'reasoning_step'
  | 'stream_end'
  | 'error'
  | 'heartbeat';

interface StreamMessage {
  type: StreamMessageType;
  data: any;
  timestamp: string;
}

// Streaming delivery to messaging channels
async function streamReasoning(
  ws: WebSocket,
  channel: MoltbotChannel,
  query: string,
) {
  // 1. Send typing indicator immediately
  await channel.sendTyping();

  let accumulatedText = '';
  let messageId: string | null = null;

  ws.onmessage = async (event) => {
    const msg: StreamMessage = JSON.parse(event.data);

    switch (msg.type) {
      case 'stream_start':
        // Keep typing indicator active
        break;

      case 'token':
        accumulatedText += msg.data.token;
        // For channels that support editing (Slack, Discord):
        if (channel.supportsEditing && messageId) {
          await channel.editMessage(messageId, accumulatedText + ' ⏳');
        }
        break;

      case 'reasoning_step':
        // Optional: show reasoning progress
        // "🔍 Step 2/5: Verifying claim against sources..."
        break;

      case 'confidence_update':
        // Internal signal, don't expose to user
        break;

      case 'stream_end':
        if (messageId) {
          await channel.editMessage(messageId, accumulatedText);
        } else {
          await channel.sendMessage(accumulatedText);
        }
        break;

      case 'error':
        await channel.sendMessage(
          `⚠️ Reasoning failed: ${msg.data.message}. Falling back to standard mode.`
        );
        break;
    }
  };

  // Send query to streaming endpoint
  ws.send(JSON.stringify({
    type: 'query',
    data: { query, mode: selectMode(query).mode },
  }));
}
```

### 2.4 Channel Capabilities Matrix

Different messaging platforms have different capabilities for streaming delivery:

| Channel | Edit Messages | Typing Indicator | Reactions | Max Length |
|---------|---------------|-----------------|-----------|------------|
| **Slack** | Yes | Yes | Yes | 40,000 chars |
| **Discord** | Yes | Yes | Yes | 2,000 chars |
| **Telegram** | Yes | Yes | Yes | 4,096 chars |
| **WhatsApp** | No | Yes | Yes (emoji) | 65,536 chars |
| **iMessage** | No | Yes (bubbles) | Yes (tapback) | No limit |
| **Signal** | No | No | Yes (emoji) | ~2,000 chars |
| **Matrix** | Yes | Yes | Yes | No limit |
| **Teams** | Yes | Yes | Yes | 28,000 chars |

**Delivery strategy by capability**:

- **Supports editing** (Slack, Discord, Telegram, Matrix, Teams): Send initial message, progressively edit with new tokens
- **Typing only** (WhatsApp, iMessage): Show typing indicator during reasoning, send final message
- **No streaming** (Signal): Send final message after reasoning completes

### 2.5 Dual-Mode Operation

Users can control how HoloLoom integrates:

**Context mode** (default): HoloLoom's reasoning result is injected as context for Moltbot's LLM. The LLM synthesizes the final response in its own voice.

```
LLM System Prompt:
  [HoloLoom Reasoning Context]
  Mode: VERIFY (2 steps)
  Confidence: 0.92
  Finding: "Thompson Sampling converges to optimal in O(log T)"
  Sources: 3 verified memories
  Verification: Confirmed with 2 independent sources
  [End Context]

  Use this verified information to answer the user's question.
```

**Direct mode**: HoloLoom's response is sent directly to the user, bypassing Moltbot's LLM entirely. Faster and cheaper (no LLM call), but loses Moltbot's personality/voice.

Toggle via config:
```json
{
  "extensions": {
    "hololoom-bridge": {
      "reasoningMode": "context",  // "context" | "direct"
    }
  }
}
```

Or per-message via command:
```
!hololoom direct What are the tradeoffs of Thompson Sampling?
```

### 2.6 Latency Budget

| Mode | HoloLoom | Streaming | Moltbot LLM | Total (Context) | Total (Direct) |
|------|----------|-----------|-------------|-----------------|----------------|
| DIRECT | ~150ms | — | ~500ms | ~650ms | ~150ms |
| VERIFY | ~600ms | typing | ~500ms | ~1,100ms | ~600ms |
| RESEARCH | ~900ms | progressive | ~500ms | ~1,400ms | ~900ms |
| PLAN_EXECUTE | ~750ms | progressive | ~500ms | ~1,250ms | ~750ms |

With query cache (100x speedup for repeated queries):
- Cached DIRECT: <5ms
- Cached VERIFY: <5ms

### 2.7 Additional Files

```
extensions/hololoom-bridge/
├── src/
│   ├── reasoning/
│   │   ├── mode-selector.ts    # Auto mode selection
│   │   ├── stream-handler.ts   # WebSocket streaming logic
│   │   ├── context-injector.ts # Format reasoning for LLM injection
│   │   └── channel-adapter.ts  # Per-channel delivery strategy
│   └── tools/
│       ├── reason.ts           # hololoom_reason tool
│       └── query.ts            # hololoom_query tool
├── skills/
│   └── hololoom-reasoning/
│       └── SKILL.md
└── test/
    ├── mode-selector.test.ts
    ├── stream-handler.test.ts
    └── channel-adapter.test.ts
```

### 2.8 Testing Plan

| Test | What It Validates |
|------|------------------|
| `mode-selector.test.ts` | Correct mode for 20+ query patterns |
| `stream-handler.test.ts` | WebSocket message handling, reconnection |
| `channel-adapter.test.ts` | Per-channel delivery (edit vs typing vs final) |
| `context-injector.test.ts` | LLM context formatting, token budget |
| `e2e: verify-mode` | Fact-check claim, return verification status |
| `e2e: research-mode` | Multi-step research with streaming |
| `e2e: fallback` | Degrade to DIRECT when streaming fails |

### 2.9 Success Criteria

- Complex questions produce multi-step verified answers
- Streaming delivery shows typing indicators on all channels
- Channels supporting editing show progressive text updates
- Response quality measurably improves over raw LLM calls
- <100ms overhead for mode selection

---

## Phase 3: Safety and Alignment Layer

**Goal**: Gate Moltbot's tool executions through HoloLoom's alignment framework.

### 3.1 Safety Gate Integration

Before Moltbot executes any tool, the extension calls HoloLoom's `/safety/gate` endpoint:

```typescript
import type { ActionCategory } from './types';

interface SafetyGateRequest {
  action_id: string;           // Unique action ID (UUID)
  category: ActionCategory;    // EXECUTION | DATA_ACCESS | COMMUNICATION | ...
  description: string;         // Human-readable description
  context: Record<string, any>; // Tool-specific context
  user_id?: string;            // Moltbot user ID
  session_id?: string;         // Moltbot session ID
}

interface SafetyGateResponse {
  allowed: boolean;
  risk_level: 'SAFE' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  reason: string;
  requires_approval: boolean;
  alternative_action?: string;
  metadata: Record<string, any>;
}

async function gateAction(
  client: HoloLoomClient,
  request: SafetyGateRequest,
): Promise<SafetyGateResponse> {
  const response = await fetch(`${client.baseUrl}/safety/gate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return response.json();
}
```

### 3.2 Risk Response Matrix

| Risk Level | Action | User Experience |
|------------|--------|----------------|
| **SAFE** | Execute immediately | No indication |
| **LOW** | Execute + log | No indication (audit only) |
| **MEDIUM** | Execute + log + notify | "ℹ️ Executed: browsed example.com" |
| **HIGH** | Ask user first | "⚠️ This will access your file system. Proceed? (yes/no)" |
| **CRITICAL** | Block + explain | "🚫 Blocked: This action was classified as dangerous. Reason: {reason}" |

### 3.3 Tool-to-Category Mapping

Every Moltbot tool maps to a HoloLoom `ActionCategory`:

```typescript
const TOOL_CATEGORY_MAP: Record<string, {
  category: ActionCategory;
  defaultRisk: string;
  contextExtractor: (args: any) => Record<string, any>;
}> = {

  // --- Browser Tools ---
  'browser.navigate': {
    category: 'EXECUTION',
    defaultRisk: 'MEDIUM',
    contextExtractor: (args) => ({
      url: args.url,
      purpose: 'web_navigation',
    }),
  },
  'browser.execute_script': {
    category: 'EXECUTION',
    defaultRisk: 'HIGH',
    contextExtractor: (args) => ({
      script_preview: args.script?.substring(0, 200),
      purpose: 'script_execution',
    }),
  },
  'browser.screenshot': {
    category: 'DATA_ACCESS',
    defaultRisk: 'LOW',
    contextExtractor: (args) => ({
      url: args.url,
      purpose: 'visual_capture',
    }),
  },

  // --- Device Node Tools ---
  'node.camera.snap': {
    category: 'DATA_ACCESS',
    defaultRisk: 'HIGH',
    contextExtractor: () => ({
      device: 'camera',
      purpose: 'photo_capture',
    }),
  },
  'node.screen.record': {
    category: 'DATA_ACCESS',
    defaultRisk: 'HIGH',
    contextExtractor: (args) => ({
      device: 'screen',
      duration_s: args.duration,
      purpose: 'screen_recording',
    }),
  },
  'node.location': {
    category: 'DATA_ACCESS',
    defaultRisk: 'MEDIUM',
    contextExtractor: () => ({
      device: 'gps',
      purpose: 'location_access',
    }),
  },

  // --- Execution Tools ---
  'bash.execute': {
    category: 'EXECUTION',
    defaultRisk: 'CRITICAL',
    contextExtractor: (args) => ({
      command_preview: args.command?.substring(0, 200),
      purpose: 'shell_execution',
    }),
  },
  'code.run': {
    category: 'EXECUTION',
    defaultRisk: 'HIGH',
    contextExtractor: (args) => ({
      language: args.language,
      code_preview: args.code?.substring(0, 200),
      purpose: 'code_execution',
    }),
  },

  // --- Automation Tools ---
  'cron.schedule': {
    category: 'EXECUTION',
    defaultRisk: 'MEDIUM',
    contextExtractor: (args) => ({
      schedule: args.cron,
      action: args.action,
      purpose: 'scheduled_task',
    }),
  },
  'webhook.create': {
    category: 'COMMUNICATION',
    defaultRisk: 'MEDIUM',
    contextExtractor: (args) => ({
      url: args.url,
      purpose: 'webhook_creation',
    }),
  },

  // --- Communication Tools ---
  'gmail.send': {
    category: 'COMMUNICATION',
    defaultRisk: 'MEDIUM',
    contextExtractor: (args) => ({
      to: args.to,
      subject: args.subject,
      purpose: 'email_send',
    }),
  },
  'sessions.send': {
    category: 'COMMUNICATION',
    defaultRisk: 'LOW',
    contextExtractor: (args) => ({
      target_session: args.sessionId,
      purpose: 'cross_session_message',
    }),
  },

  // --- File Tools ---
  'fs.read': {
    category: 'DATA_ACCESS',
    defaultRisk: 'MEDIUM',
    contextExtractor: (args) => ({
      path: args.path,
      purpose: 'file_read',
    }),
  },
  'fs.write': {
    category: 'DATA_ACCESS',
    defaultRisk: 'HIGH',
    contextExtractor: (args) => ({
      path: args.path,
      size_bytes: args.content?.length,
      purpose: 'file_write',
    }),
  },
  'fs.delete': {
    category: 'DATA_ACCESS',
    defaultRisk: 'CRITICAL',
    contextExtractor: (args) => ({
      path: args.path,
      recursive: args.recursive,
      purpose: 'file_delete',
    }),
  },
};
```

### 3.4 Pre-Execution Hook

The hook wraps every Moltbot tool execution:

```typescript
async function preExecutionHook(
  toolName: string,
  toolArgs: any,
  session: MoltbotSession,
  client: HoloLoomClient,
): Promise<{ proceed: boolean; message?: string }> {

  const mapping = TOOL_CATEGORY_MAP[toolName];
  if (!mapping) {
    // Unknown tool — default to HIGH risk
    return gateUnknownTool(toolName, toolArgs, session, client);
  }

  const request: SafetyGateRequest = {
    action_id: crypto.randomUUID(),
    category: mapping.category,
    description: `Moltbot tool: ${toolName}`,
    context: {
      ...mapping.contextExtractor(toolArgs),
      tool_name: toolName,
      channel: session.channel,
    },
    user_id: resolveUserId(session),
    session_id: session.id,
  };

  try {
    const gate = await gateAction(client, request);

    // Log to audit trail (all risk levels)
    await logToAudit(client, request, gate);

    switch (gate.risk_level) {
      case 'SAFE':
      case 'LOW':
        return { proceed: true };

      case 'MEDIUM':
        // Execute but notify
        await session.sendMessage(
          `ℹ️ Executed: ${toolName} — ${gate.reason}`
        );
        return { proceed: true };

      case 'HIGH':
        // Ask user for confirmation
        if (gate.requires_approval) {
          const confirmed = await session.askConfirmation(
            `⚠️ ${gate.reason}\n\nProceed with ${toolName}? (yes/no)`
          );
          return { proceed: confirmed };
        }
        return { proceed: true };

      case 'CRITICAL':
        // Block
        await session.sendMessage(
          `🚫 Blocked: ${toolName}\nReason: ${gate.reason}`
          + (gate.alternative_action
            ? `\nSuggestion: ${gate.alternative_action}`
            : '')
        );
        return { proceed: false, message: gate.reason };
    }
  } catch (error) {
    // FAIL-CLOSED: if HoloLoom is unreachable, block high-risk actions
    if (mapping.defaultRisk === 'HIGH' || mapping.defaultRisk === 'CRITICAL') {
      await session.sendMessage(
        `⚠️ Safety system unavailable. Blocking ${toolName} as a precaution.`
      );
      return { proceed: false, message: 'Safety system unreachable' };
    }
    // Low/medium risk: allow with warning
    return { proceed: true };
  }
}
```

### 3.5 Audit Trail Integration

Every action (allowed or blocked) is logged:

```typescript
async function logToAudit(
  client: HoloLoomClient,
  request: SafetyGateRequest,
  gate: SafetyGateResponse,
) {
  await fetch(`${client.baseUrl}/safety/audit-trail`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action_id: request.action_id,
      action: request.description,
      category: request.category,
      context: request.context,
      user_id: request.user_id,
      session_id: request.session_id,
      risk_level: gate.risk_level,
      allowed: gate.allowed,
      reason: gate.reason,
      timestamp: new Date().toISOString(),
      source: 'moltbot-bridge',
    }),
  });
}
```

Audit entries are queryable via HoloLoom's existing endpoints:
- `GET /safety/audit-trail?user_id=X` — by user
- `GET /safety/audit-trail?category=EXECUTION` — by category
- `GET /safety/audit-trail?risk_level=CRITICAL` — by risk level
- `GET /safety/audit-trail?start=2026-01-01&end=2026-01-31` — by date range

### 3.6 Defense in Depth

```
Layer 1: Moltbot DM Pairing       WHO can talk to the bot
         (authentication)          Unknown senders get pairing codes

Layer 2: Moltbot Docker Sandbox    WHERE code runs
         (execution isolation)     Non-main sessions in Docker containers

Layer 3: HoloLoom Safety Gate      WHAT is allowed to happen
         (semantic safety)         Risk-based action gating with context

Layer 4: HoloLoom Audit Trail      WHAT happened (forensics)
         (provenance)              Complete log of all decisions
```

### 3.7 Additional Files

```
extensions/hololoom-bridge/
├── src/
│   ├── safety/
│   │   ├── gate.ts             # Safety gate client
│   │   ├── tool-mapping.ts     # Tool-to-category mapping
│   │   ├── pre-exec-hook.ts    # Pre-execution hook
│   │   ├── audit.ts            # Audit trail logging
│   │   └── confirmation.ts     # User confirmation UI
│   └── ...
└── test/
    ├── gate.test.ts
    ├── tool-mapping.test.ts
    ├── pre-exec-hook.test.ts
    └── fail-closed.test.ts
```

### 3.8 Testing Plan

| Test | What It Validates |
|------|------------------|
| `gate.test.ts` | All 5 risk levels produce correct responses |
| `tool-mapping.test.ts` | All 15+ Moltbot tools map to correct categories |
| `pre-exec-hook.test.ts` | Hook blocks CRITICAL, confirms HIGH, allows LOW |
| `fail-closed.test.ts` | HIGH/CRITICAL blocked when HoloLoom unreachable |
| `audit.test.ts` | All actions logged with correct fields |
| `e2e: rm-rf` | `bash.execute("rm -rf /")` -> CRITICAL -> blocked |
| `e2e: safe-browse` | `browser.navigate("example.com")` -> LOW -> allowed |
| `e2e: confirmation` | HIGH risk tool prompts user, respects yes/no |

### 3.9 Success Criteria

- All 15+ Moltbot tools have safety gate mappings
- CRITICAL actions (shell exec, file delete) are always blocked without confirmation
- HIGH actions (camera, file write, code exec) require user confirmation
- All actions have audit trail entries
- Safety gate adds <1ms overhead (HoloLoom benchmark: 0.103ms)
- Fail-closed: unreachable HoloLoom blocks HIGH+ actions

---

## Phase 4: Learning Feedback Loop

**Goal**: Close the loop so HoloLoom continuously improves from Moltbot interactions.

### 4.1 Feedback Signal Mapping

```typescript
// Map Moltbot user behaviors to HoloLoom learning signals

interface FeedbackSignal {
  type: 'explicit' | 'implicit';
  quality: number;     // 0.0-1.0
  helpful: boolean;
  confidence: number;  // How confident we are in this signal
  source: string;      // What triggered it
}

const SIGNAL_MAP: Record<string, FeedbackSignal> = {
  // --- Explicit Signals ---
  'reaction:thumbsup':    { type: 'explicit', quality: 0.9, helpful: true,  confidence: 0.95, source: 'emoji_positive' },
  'reaction:thumbsdown':  { type: 'explicit', quality: 0.1, helpful: false, confidence: 0.95, source: 'emoji_negative' },
  'reaction:heart':       { type: 'explicit', quality: 0.95, helpful: true, confidence: 0.90, source: 'emoji_love' },
  'reaction:confused':    { type: 'explicit', quality: 0.3, helpful: false, confidence: 0.80, source: 'emoji_confused' },
  'text:thanks':          { type: 'explicit', quality: 0.85, helpful: true, confidence: 0.85, source: 'text_gratitude' },
  'text:wrong':           { type: 'explicit', quality: 0.15, helpful: false, confidence: 0.90, source: 'text_correction' },

  // --- Implicit Signals ---
  'behavior:follow_up':   { type: 'implicit', quality: 0.6, helpful: true,  confidence: 0.50, source: 'follow_up_question' },
  'behavior:no_reply':    { type: 'implicit', quality: 0.4, helpful: false, confidence: 0.30, source: 'no_reply_5min' },
  'behavior:share':       { type: 'implicit', quality: 0.95, helpful: true, confidence: 0.85, source: 'shared_response' },
  'behavior:copy':        { type: 'implicit', quality: 0.8, helpful: true,  confidence: 0.70, source: 'copied_text' },
  'behavior:long_read':   { type: 'implicit', quality: 0.7, helpful: true,  confidence: 0.40, source: 'read_time_long' },
  'behavior:quick_dismiss': { type: 'implicit', quality: 0.3, helpful: false, confidence: 0.40, source: 'quick_dismiss' },
};
```

### 4.2 Feedback Collection Pipeline

```typescript
// Collect signals and batch them to HoloLoom

class FeedbackCollector {
  private buffer: Array<{
    memoryIds: string[];
    signal: FeedbackSignal;
    channel: string;
    userId: string;
    timestamp: string;
  }> = [];

  private flushInterval: NodeJS.Timeout;

  constructor(
    private client: HoloLoomClient,
    flushIntervalMs: number = 60_000,  // Flush every 60s
  ) {
    this.flushInterval = setInterval(() => this.flush(), flushIntervalMs);
  }

  // Record a feedback signal
  record(
    memoryIds: string[],
    signal: FeedbackSignal,
    channel: string,
    userId: string,
  ) {
    this.buffer.push({
      memoryIds,
      signal,
      channel,
      userId,
      timestamp: new Date().toISOString(),
    });

    // Flush immediately for high-confidence explicit signals
    if (signal.type === 'explicit' && signal.confidence >= 0.9) {
      this.flush();
    }
  }

  // Send accumulated feedback to HoloLoom
  async flush() {
    if (this.buffer.length === 0) return;

    const batch = [...this.buffer];
    this.buffer = [];

    for (const entry of batch) {
      try {
        await this.client.reflect({
          memoryIds: entry.memoryIds,
          feedback: {
            helpful: entry.signal.helpful,
            quality: entry.signal.quality,
            source: entry.signal.source,
            channel: entry.channel,
            confidence: entry.signal.confidence,
          },
          tags: [
            `channel:${entry.channel}`,
            `signal:${entry.signal.source}`,
          ],
        });
      } catch (error) {
        // Re-buffer on failure (will retry next flush)
        this.buffer.push(entry);
      }
    }
  }

  stop() {
    clearInterval(this.flushInterval);
    this.flush();
  }
}
```

### 4.3 Implicit Signal Detection

```typescript
// Detect implicit signals from user behavior

class ImplicitSignalDetector {
  private lastResponseTime = new Map<string, number>();  // userId -> timestamp
  private lastMemoryIds = new Map<string, string[]>();   // userId -> memoryIds

  // Called after bot sends a response
  onBotResponse(userId: string, memoryIds: string[]) {
    this.lastResponseTime.set(userId, Date.now());
    this.lastMemoryIds.set(userId, memoryIds);
  }

  // Called when user sends a follow-up message
  onUserMessage(userId: string, message: string): FeedbackSignal | null {
    const lastTime = this.lastResponseTime.get(userId);
    if (!lastTime) return null;

    const elapsed = Date.now() - lastTime;

    // Quick follow-up (< 2 min) = engaged
    if (elapsed < 120_000) {
      // Check if it's a correction
      if (/\b(no|wrong|incorrect|actually|not right)\b/i.test(message)) {
        return SIGNAL_MAP['text:wrong'];
      }
      // Check if it's gratitude
      if (/\b(thanks|thank you|perfect|great|awesome)\b/i.test(message)) {
        return SIGNAL_MAP['text:thanks'];
      }
      // Otherwise it's a follow-up question
      return SIGNAL_MAP['behavior:follow_up'];
    }

    // No reply for 5+ minutes = weak negative
    if (elapsed > 300_000) {
      return SIGNAL_MAP['behavior:no_reply'];
    }

    return null;
  }

  // Called when user reacts to a message
  onReaction(userId: string, reaction: string): FeedbackSignal | null {
    const key = `reaction:${reaction}`;
    return SIGNAL_MAP[key] || null;
  }
}
```

### 4.4 Per-Channel Thompson Sampling

HoloLoom learns which reasoning mode works best for each channel:

```typescript
// This runs on the HoloLoom side (Python), but the Moltbot bridge
// feeds it by including channel metadata in every reflect() call.
//
// HoloLoom's Thompson Sampling infrastructure maintains:
//   - Beta(α, β) priors per (channel, reasoning_mode) pair
//   - α += quality on success
//   - β += (1 - quality) on failure
//
// After enough data (~50 queries per channel), the system converges:
//
// Example learned preferences:
//   slack:    VERIFY (α=45, β=8)    -> 85% success -> prefer accuracy
//   whatsapp: DIRECT (α=38, β=5)   -> 88% success -> prefer speed
//   discord:  RESEARCH (α=30, β=12) -> 71% success -> prefer depth
//   telegram: DIRECT (α=42, β=7)   -> 86% success -> prefer speed
//
// The extension can query this:
//   GET /learning/channel-preferences?channel=slack
//   -> { recommended_mode: "VERIFY", confidence: 0.85, sample_size: 53 }
```

### 4.5 New Server Endpoint (Python side)

Phase 4 requires one new endpoint on the HoloLoom server:

```python
# Add to HoloLoom/server/unified_server.py

@app.get("/learning/channel-preferences")
async def get_channel_preferences(channel: str):
    """
    Get Thompson Sampling recommendations for a specific channel.
    Returns optimal reasoning mode based on learned user preferences.
    """
    # Query the policy engine's per-channel bandits
    bandit = policy_engine.get_channel_bandit(channel)
    if not bandit:
        return {"recommended_mode": "DIRECT", "confidence": 0.5, "sample_size": 0}

    recommendation = bandit.recommend()
    return {
        "recommended_mode": recommendation.mode,
        "confidence": recommendation.confidence,
        "sample_size": recommendation.total_queries,
        "mode_scores": {
            mode: bandit.expected_reward(mode)
            for mode in ["DIRECT", "VERIFY", "RESEARCH", "PLAN_EXECUTE"]
        },
    }
```

### 4.6 Learning Feedback Architecture

```
User Interaction                    Feedback Collection             HoloLoom Learning
+-----------------+                 +-------------------+           +------------------+
| 👍 reaction     | ---> record --> | FeedbackCollector  |           |                  |
| "thanks" text   |                 | (60s buffer)       | --flush-> | /lite/reflect    |
| follow-up Q     |                 |                    |           |   |              |
| no reply 5min   |                 | Batches by:        |           |   v              |
| shares response |                 | - userId           |           | Thompson Update  |
+-----------------+                 | - channel          |           | α += quality     |
                                    | - memoryIds        |           | β += (1-quality) |
                                    +-------------------+           |   |              |
                                                                     |   v              |
                                                                     | Hot Patterns     |
                                                                     | heat += access   |
                                                                     |   |              |
                                                                     |   v              |
                                                                     | Adaptive Routing |
                                                                     | per-channel mode |
                                                                     +------------------+
```

### 4.7 Privacy-Preserving Feedback

Feedback signals are **aggregated, not raw**:

```typescript
// What IS sent to HoloLoom:
{
  memoryIds: ["mem_123", "mem_456"],
  feedback: {
    helpful: true,
    quality: 0.9,
    source: "emoji_positive",      // Signal type, not content
    channel: "whatsapp",           // Channel, not user message
    confidence: 0.95,
  }
}

// What is NOT sent:
// - Raw user message text (already stored in Phase 1)
// - User's personal information
// - Other users' data
// - Cross-user correlations
```

### 4.8 Additional Files

```
extensions/hololoom-bridge/
├── src/
│   ├── learning/
│   │   ├── feedback-collector.ts     # Batched feedback dispatch
│   │   ├── signal-detector.ts        # Implicit signal detection
│   │   ├── signal-map.ts             # Signal type definitions
│   │   ├── channel-preferences.ts    # Per-channel mode selection
│   │   └── privacy-filter.ts         # Strip PII from feedback
│   └── ...
└── test/
    ├── feedback-collector.test.ts
    ├── signal-detector.test.ts
    ├── channel-preferences.test.ts
    └── privacy-filter.test.ts
```

### 4.9 Testing Plan

| Test | What It Validates |
|------|------------------|
| `feedback-collector.test.ts` | Batching, flushing, retry on failure |
| `signal-detector.test.ts` | All explicit + implicit signals detected correctly |
| `channel-preferences.test.ts` | Thompson Sampling recommendations converge |
| `privacy-filter.test.ts` | No PII in feedback payloads |
| `e2e: thumbs-up` | Emoji reaction -> reflect(helpful=true) -> quality improves |
| `e2e: cross-channel` | WhatsApp feedback improves Telegram recall |
| `e2e: convergence` | After 50 queries, channel mode preferences stabilize |

### 4.10 Success Criteria

- Response quality measurably improves after 100+ interactions
- Per-channel reasoning mode selection converges to user preferences
- Hot patterns reflect actual user interests
- Feedback latency < 100ms (batched, async)
- No PII in feedback payloads (privacy filter verified by test)

---

## Deployment

### Docker Compose (Development)

```yaml
# docker-compose.moltbot-hololoom.yml
version: '3.8'

services:
  # HoloLoom API + backends
  hololoom-api:
    build: .
    ports:
      - "8000:8000"   # REST API
      - "8001:8001"   # WebSocket streaming
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=hololoom123
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      neo4j:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s

  neo4j:
    image: neo4j:5.13-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/hololoom123
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "hololoom123", "RETURN 1"]
      interval: 10s
      timeout: 5s

  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s

volumes:
  neo4j_data:
  qdrant_data:
```

Start HoloLoom:
```bash
docker compose -f docker-compose.moltbot-hololoom.yml up -d
```

Then configure Moltbot to use the bridge:
```bash
# In Moltbot's workspace
cd ~/clawd
npm install @hololoom/sdk
# Add extension config to ~/.clawdbot/moltbot.json
```

### Production Deployment

For production, both services should run as system daemons:

```
                         ┌─ Moltbot Gateway (systemd/launchd)
                         │    ws://127.0.0.1:18789
Internet (messaging)  ───┤
                         │    hololoom-bridge extension
                         │    ↓ HTTP to localhost
                         │
                         └─ HoloLoom API (Docker Compose)
                              :8000 REST + :8001 WebSocket
                              ├─ Neo4j (graph persistence)
                              └─ Qdrant (vector persistence)
```

Both services bind to localhost. No external network exposure needed for the HoloLoom<->Moltbot bridge.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| HoloLoom server down | Moltbot loses memory/reasoning | Graceful degradation to stateless LLM mode. SDK has exponential backoff retry (3x). |
| Cross-language type drift | Integration breaks silently | TypeScript SDK types (`sdk/typescript/src/types/index.ts`) serve as contract. CI validates both sides. |
| Latency on messaging | Slow responses lose users | DIRECT mode ~150ms. For slow modes, streaming + typing indicators. Query cache provides 100x speedup for repeated queries. |
| Memory bloat from chat volume | Storage grows unbounded | HoloLoom consolidation runs every 60min: deduplication (>95% similar), archival (>30 days), pruning (>90 days). Context packing achieves 40-90% token savings. |
| Adversarial multi-channel attacks | Prompt injection across platforms | Phase 3 deception detection monitors cross-channel patterns. Audit trail provides forensics. Fail-closed design blocks suspicious actions. |
| Privacy concerns | User data stored in graph DB | Per-user memory namespaces. No cross-user data leakage. Archive-not-delete policy. Privacy filter strips PII from feedback. |
| Moltbot skill format changes | Extension breaks on Moltbot update | Skill definitions are Markdown (stable format). SDK client uses HTTP (stable protocol). Pin Moltbot version in CI. |

---

## Implementation Priority

```
Phase 1: Memory Bridge          [Week 1-2]   <- Start here
  Deliverables:
  - SKILL.md for hololoom-memory
  - Bridge extension (remember/recall/reflect)
  - User identity mapping
  - Auto-remember + auto-recall hooks
  - Graceful degradation
  |
  v
Phase 2: Agentic Reasoning      [Week 3-5]   <- Biggest user-visible impact
  Deliverables:
  - SKILL.md for hololoom-reasoning
  - Mode auto-selection
  - WebSocket streaming handler
  - Channel capability matrix
  - Dual-mode (context/direct)
  |
  v
Phase 3: Safety & Alignment     [Week 6-8]   <- Required before public deployment
  Deliverables:
  - Pre-execution hook
  - 15+ tool-to-category mappings
  - Risk response matrix
  - Audit trail logging
  - Fail-closed fallback
  - User confirmation UI
  |
  v
Phase 4: Learning Feedback       [Week 9-11]  <- Long-term differentiation
  Deliverables:
  - Feedback signal map (12+ signals)
  - FeedbackCollector (60s batching)
  - Implicit signal detector
  - Per-channel Thompson Sampling endpoint
  - Privacy filter
```

Each phase is independently useful. Phase 1 alone transforms Moltbot from stateless to stateful. Phase 2 alone transforms it from shallow to deep reasoning. Phase 3 is mandatory before any production deployment with tool access. Phase 4 is what makes the system get smarter over time.

---

## Future Possibilities

Beyond the 4 phases:

- **HoloLoom Dark Trace for Moltbot**: Explain why the assistant made specific decisions, visible to users ("I recalled this memory because it has 0.92 relevance to your question")
- **SpinningWheel adapters for Moltbot channels**: Ingest Slack threads, Discord channels, email threads directly into HoloLoom's knowledge graph via the 47 existing adapters
- **Moltbot's CDP browser as HoloLoom tool**: Give HoloLoom web browsing capability for RESEARCH mode — the weaving orchestrator can invoke Moltbot's browser to gather live information
- **Moltbot Canvas as Jenny renderer**: Render HoloLoom's Jenny visualization panels (confidence trajectories, knowledge graphs, pipeline waterfalls) in Moltbot's Canvas A2UI
- **Federated multi-user HoloLoom**: Multiple Moltbot instances sharing a HoloLoom cluster via the Federation module (SWIM gossip + Kademlia DHT)
- **Voice integration**: Moltbot's TTS/STT + HoloLoom's voice system for hands-free agentic reasoning
- **SOUL.md integration**: Auto-generate Moltbot's `SOUL.md` personality file from HoloLoom's semantic profile (228D axes: warmth, formality, urgency, etc.)
