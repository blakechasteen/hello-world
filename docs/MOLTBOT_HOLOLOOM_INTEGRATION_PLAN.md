# Moltbot + HoloLoom Integration Plan

**Date**: 2026-01-28
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

---

## Phase 1: Memory Bridge

**Goal**: Give Moltbot persistent, cross-session, cross-channel memory.

**Duration**: 1-2 weeks

### Deliverables

1. **Moltbot HoloLoom Skill** (`~/clawd/skills/hololoom/SKILL.md`)

   A Moltbot skill that wraps HoloLoom's Lite API with three operations:
   - `remember(content, metadata)` -> calls `experience()` to persist information
   - `recall(query, options)` -> calls `recall()` to retrieve relevant memories
   - `reflect(feedback)` -> calls `reflect()` to provide learning signals

2. **User Identity Mapping**

   Translate Moltbot's per-channel user IDs into a stable HoloLoom namespace:
   ```typescript
   await client.experience({
     content: userMessage,
     metadata: {
       moltbot_user_id: session.userId,
       moltbot_channel: 'whatsapp',
       moltbot_session_id: session.id,
       timestamp: new Date().toISOString()
     }
   });
   ```

3. **Message Flow**
   ```
   User sends message on WhatsApp
     -> Moltbot receives via Gateway
     -> Extension calls experience() to store message
     -> Extension calls recall() to get relevant past context
     -> Recalled context injected into LLM prompt
     -> LLM generates response with memory context
     -> User gets response that "remembers" past conversations
   ```

### Security

- Both services run on localhost (no TLS needed for Phase 1)
- HoloLoom rate limiter (60 req/min) is sufficient for single-user
- Moltbot's sandbox mode isolates execution

### Success Criteria

- A user's WhatsApp conversation recalls context from a Slack message sent days ago
- Memory persists across Moltbot restarts
- Zero-config for end users (the skill handles everything)

---

## Phase 2: Agentic Reasoning Bridge

**Goal**: Give Moltbot access to HoloLoom's 4 reasoning modes for complex queries.

**Duration**: 2-3 weeks

### Deliverables

1. **Reasoning Skill Extension**

   Extends Phase 1 skill with reasoning capabilities:
   - `reason(query, mode, maxSteps)` -> calls HoloLoom's agentic orchestrator
   - `query(text, options)` -> calls full weaving cycle

2. **Auto Mode Selection**

   HoloLoom's query classifier automatically routes:
   | Query Pattern | Reasoning Mode | Latency |
   |---|---|---|
   | Simple chat ("hi", "thanks") | DIRECT | ~150ms |
   | Factual ("what is X?") | DIRECT | ~150ms |
   | Verification ("is X true?") | VERIFY | ~600ms |
   | Research ("tell me everything about X") | RESEARCH | ~900ms |
   | Planning ("how do I do X step by step?") | PLAN_EXECUTE | ~750ms |

3. **Streaming Response Delivery**

   For modes >500ms, use WebSocket streaming:
   - `STREAM_START` -> "typing" indicator on messaging channel
   - `TOKEN` -> progressive text update (Slack/Discord support message editing)
   - `STREAM_COMPLETE` -> final message delivery

4. **Dual-Mode Operation**

   Users choose how to use HoloLoom:
   - **Context mode** (default): HoloLoom reasoning injected as context for Moltbot's LLM
   - **Direct mode**: HoloLoom's response delivered directly, bypassing Moltbot's LLM

### Success Criteria

- Complex questions ("compare X and Y with tradeoffs") produce multi-step verified answers
- Streaming delivery shows typing indicators on messaging channels
- Response quality measurably improves over raw LLM calls

---

## Phase 3: Safety and Alignment Layer

**Goal**: Gate Moltbot's tool executions through HoloLoom's alignment framework.

**Duration**: 2-3 weeks

### Deliverables

1. **Safety Gate Pre-Execution Hook**

   Before Moltbot executes any tool, the extension calls HoloLoom's safety gate:
   ```typescript
   const gate = await client.safetyGate({
     action: 'browser_navigate',
     context: { url: targetUrl, user_id: session.userId },
     category: 'TOOL_EXECUTION'
   });

   switch (gate.risk_level) {
     case 'LOW':    execute(); break;
     case 'MEDIUM': execute(); log(gate); break;
     case 'HIGH':   await confirmWithUser(gate.reason); break;
     case 'CRITICAL': block(gate.reason); break;
   }
   ```

2. **Gated Tool Categories**

   | Moltbot Tool | Risk Category | Default Gate |
   |---|---|---|
   | CDP browser navigation | TOOL_EXECUTION | MEDIUM |
   | File system access | DATA_ACCESS | HIGH |
   | Camera/screen capture | DATA_ACCESS | HIGH |
   | Cron scheduling | TOOL_EXECUTION | MEDIUM |
   | Gmail pub/sub | COMMUNICATION | MEDIUM |
   | Code execution | TOOL_EXECUTION | HIGH |
   | System commands | TOOL_EXECUTION | CRITICAL |

3. **Audit Trail Integration**

   Every gated action logged to HoloLoom's `AuditTrail`:
   - Complete provenance of all tool executions
   - Searchable by user, channel, action type, time range
   - Forensic capability for incident response

4. **Epistemic Confidence Gating**

   HoloLoom's consciousness integration adjusts risk levels:
   - Epistemic confidence < 0.3 -> escalate to HIGH risk
   - Epistemic confidence < 0.6 -> escalate to MEDIUM risk
   - This prevents tool execution when the system is uncertain about what it's doing

5. **Fail-Closed Design**

   If HoloLoom is unreachable, Moltbot blocks HIGH+ risk actions rather than executing ungated.

### Defense in Depth

```
Layer 1: Moltbot DM Pairing     (authentication - who can talk)
Layer 2: Moltbot Docker Sandbox  (execution isolation - blast radius)
Layer 3: HoloLoom Safety Gate    (semantic safety - should this happen?)
Layer 4: HoloLoom Audit Trail    (forensics - what happened?)
```

### Success Criteria

- Dangerous commands (`rm -rf`, navigation to known-malicious URLs) are blocked with explanation
- All tool executions have audit trail entries
- Safety gate adds <1ms overhead per action (alignment framework benchmark: 0.103ms)

---

## Phase 4: Learning Feedback Loop

**Goal**: Close the loop so HoloLoom continuously improves from Moltbot interactions.

**Duration**: 2-3 weeks

### Deliverables

1. **Implicit Feedback Signals**

   Map Moltbot user behaviors to HoloLoom learning signals:
   | User Behavior | Signal | Quality Score |
   |---|---|---|
   | Positive emoji / "thanks" | Success | 0.9 |
   | Negative emoji / "wrong" | Failure | 0.2 |
   | Follow-up question | Moderate success | 0.6 |
   | No reply (>N minutes) | Weak negative | 0.4 |
   | Shares response to another channel | Strong success | 0.95 |

2. **Cross-Channel Learning**

   Feedback from WhatsApp improves recall quality on Telegram. HoloLoom's Memory Conductor coordinates across all 7 memory systems, and Thompson Sampling exploration means successful patterns get boosted everywhere.

3. **Per-Channel Optimization**

   Thompson Sampling bandits learn optimal reasoning modes per platform:
   - Slack users may prefer VERIFY mode (technical accuracy)
   - WhatsApp users may prefer DIRECT mode (fast responses)
   - The system discovers this automatically through interaction data

4. **Hot Pattern Feedback**

   HoloLoom's hot pattern engine tracks what knowledge gets accessed:
   ```
   heat = access_count * success_rate * avg_confidence * decay
   ```
   Moltbot interactions feed `access_count` and `success_rate`. Hot patterns get 2x retrieval boost.

5. **Adaptive Routing Feedback**

   Moltbot provides massive classification data for HoloLoom's adaptive query routing. Channel-specific patterns (Slack = technical, WhatsApp = casual) improve routing accuracy over time.

### Success Criteria

- Response quality measurably improves after 100+ interactions
- Per-channel reasoning mode selection converges to user preferences
- Hot patterns reflect actual user interests (not just initial seed data)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| HoloLoom server down | Moltbot loses memory/reasoning | Graceful degradation to stateless LLM mode. SDK has exponential backoff retry. |
| Cross-language type drift | Integration breaks silently | TypeScript SDK types serve as contract. CI validates both sides. |
| Latency on messaging | Slow responses lose users | DIRECT mode ~150ms. For slow modes, streaming + typing indicators. Query cache provides 100x speedup for repeated queries. |
| Memory bloat from chat volume | Storage grows unbounded | HoloLoom consolidation runs every 60min: deduplication (>95% similar), archival (>30 days), pruning (>90 days). Context packing achieves 40-90% token savings. |
| Adversarial multi-channel attacks | Prompt injection across platforms | Phase 3 deception detection monitors cross-channel patterns. Audit trail provides forensics. |
| Privacy concerns | User data stored in graph DB | Per-user memory namespaces. No cross-user data leakage. Archive-not-delete policy. |

---

## Implementation Priority

```
Phase 1: Memory Bridge          [Week 1-2]   <- Start here
  |
  v
Phase 2: Agentic Reasoning      [Week 3-5]   <- Biggest user-visible impact
  |
  v
Phase 3: Safety & Alignment     [Week 6-8]   <- Required before public deployment
  |
  v
Phase 4: Learning Feedback       [Week 9-11]  <- Long-term differentiation
```

Each phase is independently useful. Phase 1 alone transforms Moltbot from stateless to stateful. Phase 2 alone transforms it from shallow to deep reasoning. Phase 3 is mandatory before any production deployment with tool access. Phase 4 is what makes the system get smarter over time.

---

## Future Possibilities

Beyond the 4 phases:

- **HoloLoom Dark Trace for Moltbot**: Explain why the assistant made specific decisions, visible to users
- **SpinningWheel adapters for Moltbot channels**: Ingest Slack threads, Discord channels, email threads directly into HoloLoom's knowledge graph
- **Moltbot's CDP browser as HoloLoom tool**: Give HoloLoom web browsing capability for research mode
- **Moltbot Canvas as Jenny renderer**: Render HoloLoom's Jenny visualization panels in Moltbot's Canvas UI
- **Federated multi-user HoloLoom**: Multiple Moltbot instances sharing a HoloLoom cluster via the Federation module (SWIM gossip + Kademlia DHT)
