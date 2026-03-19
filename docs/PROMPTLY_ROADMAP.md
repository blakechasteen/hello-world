# Promptly Agentic Roadmap

Three tracks. Each independently useful. No dependencies between them unless noted.

---

## Track 1: Smart Routing

Learns which model to use for which query. Never done — the fleet grows, the signal improves.

### Built

| Feature | Location | Status |
|---------|----------|--------|
| Model registry | `model_router.py:ModelSpec` | 6 models, config-driven |
| Query classifier | `model_router.py:classify_intent` | Keyword heuristic, 6 intents |
| Thompson Sampling bandit | `model_router.py:ModelBandit` | Beta(α,β) per (intent, model) |
| Health caching | `model_router.py:HealthCache` | 30s TTL, async probes |
| Speed/quality blending | `model_router.py:ModelBandit.select` | `speed_weight` 0.0–1.0 |
| Fallback chain | `model_router.py:RoutingDecision.fallback` | Auto-fallback to always_available |
| Multi-pass escalation | `promptly_chat.py:_run_refinement_pass` | Later passes favor quality |
| Bandit feedback | `promptly_chat.py` | Wired on pass 1 + refinement |
| Status endpoint | `promptly_chat.py:/status` | Models + bandit stats |

### Next

- **Better classifier** — The keyword heuristic is a placeholder. Replace with a lightweight embedding classifier (sentence-transformers, ~50ms). Train on (query, intent) pairs from bandit feedback history. The bandit already collects the signal — use it.

- **Feedback quality** — Currently binary (got content = success). Add signal richness: user reactions from Matrix (thumbs up/down), refinement pass count (more passes = harder query = penalize fast models), response length ratio (truncated = model too small).

- **Model capability tags** — `strengths` is coarse. Add fine-grained capability tags: `{"context_128k", "tool_use", "json_mode", "code_execution"}`. Route based on what the query *needs*, not just what the model is *good at*.

- **Cost accounting** — Track tokens per model per query. When two models are close on quality, prefer the cheaper one. AirLLM rig has real electricity cost; local Ollama is "free." This becomes a third axis alongside speed and quality.

- **Dynamic model discovery** — Poll Ollama `/api/tags` on startup and periodically. Auto-register new models with conservative priors (quality=0.5, let the bandit learn). Remove models that disappear. No manual registry updates when you `ollama pull` something new.

- **Bandit persistence** — Save/load bandit state to disk. Currently resets on restart. A few hundred interactions of learning shouldn't be lost. Simple JSON dump of `{(intent, model_id): {α, β}}`.

### Speculative

- **Cascade routing** — Instead of picking one model, try the fast model first. If confidence is low (detected via output length, hedging language, "I'm not sure"), automatically escalate to a bigger model. The multi-pass system already does this between passes — cascade does it *within* a single pass.

- **Context-aware routing** — Factor in conversation history. If the user has been asking code questions for 10 turns, bias toward code models even if the current message looks like chat. Conversation intent has momentum.

- **Capacity-aware scheduling** — When the rig is busy (health check returns "busy" not "ready"), queue deep queries and serve shallow ones locally. Don't block chat on a 108B model that's mid-generation.

- **A/B routing** — Occasionally route the same query to two models, show the fast one, compare quality offline. Accelerates bandit learning for new models without user-visible latency.

---

## Track 2: Observability

Internal visibility into what the system is doing. Debuggability without user-facing noise.

### Built

| Feature | Location | Status |
|---------|----------|--------|
| Model + intent on pass 1 | `hololoom-runner.ts` | Structured pino log |
| Model per refinement pass | `hololoom-runner.ts` | Structured pino log |
| Routing trail | `model_router.py` | `logger.info("Routed: ...")` |
| Bandit stats endpoint | `promptly_chat.py:/status` | JSON via API |
| Relay logging | `mention-router.ts` | Create/consume/expire logged |

### Next

- **Routing decision log** — Append each routing decision to a structured log file (JSONL). Fields: `timestamp, query_hash, intent, confidence, model_id, speed_weight, bandit_scores, fallback_used, duration_ms, tokens, success`. This is the dataset for training a better classifier and debugging routing failures.

- **Conversation flow tracing** — Assign a `trace_id` to each conversation turn that follows through pass 1 → refinement passes → relay messages. Currently you have to correlate by `refinement_id` manually. A single trace ID makes `grep` work.

- **Bandit dashboard** — Simple HTML page served from `/status/dashboard` that visualizes bandit arm distributions. Bar chart of Beta(α,β) means per intent. No React — just inline SVG from the API response. See which models are winning for which intents at a glance.

- **Relay observability** — Add `pendingRelayCount()` to the `/status` equivalent on the NanoClaw side. Log relay round-trip time (create → consume). Alert if relays are expiring (target agent not responding).

### Speculative

- **Token budget tracking** — Per-room running total of tokens consumed. Alert when a conversation is getting expensive. Useful when the rig is doing real work and you want to know where the compute is going.

- **Anomaly detection** — Flag when a model's bandit score drops suddenly (something changed — model update? prompt regression?). Flag when a room's token usage spikes. Simple z-score on rolling windows, not ML.

- **Replay** — From the routing decision log, replay historical queries through the current router to see "what would have changed." Useful after tuning classifier weights or adding models.

---

## Track 3: Agent Coordination

Agents talking to agents. Matrix as the bus. @mentions as the protocol.

### Built

| Feature | Location | Status |
|---------|----------|--------|
| @mention detection | `mention-router.ts:detectMentions` | Regex scan against registered groups |
| One-shot relay | `mention-router.ts:createRelay/consumeRelay` | FIFO, 5min TTL |
| DB + Matrix dual delivery | `index.ts` | Synthetic msg in DB, echo to Matrix |
| Trigger bypass for relays | `index.ts` | `[from ` prefix check |
| External agent flag | `types.ts:RegisteredGroup.externalAgent` | OpenClaw compat |
| External relay-back | `index.ts` | Cursor advance + relay consume |
| IPC registration | `ipc.ts:register_group` | `externalAgent` passthrough |

### Next

- **Multi-turn relay** — Current relay is one-shot (request → response). For complex coordination, agents need back-and-forth. Extend relay to support a `conversation_id` that keeps the relay channel open until either agent says "done" or a longer TTL (15min) expires. The `[from agent]` prefix already identifies relay messages — add `[conv:abc123]` to track threads.

- **Structured envelopes** — Currently agents coordinate via free text. Add an optional structured envelope for programmatic coordination:
  ```
  [from promptly] {"type": "request", "task": "check_hive_temp", "params": {"hive_id": 3}, "reply_to": "!room:server"}
  ```
  Agents that understand envelopes parse them. Agents that don't just see the text. Graceful degradation.

- **Agent discovery** — `detectMentions` only knows about NanoClaw registered groups. Add a `/agents` endpoint that returns all known agents (NanoClaw + external). Agents can query it to know who they can talk to. Also useful for SOUL.md — agents can describe themselves and their capabilities.

- **Rate limiting** — An agent stuck in a loop could spam another agent with @mentions. Add a per-source rate limit: max N relays per minute from a single agent. Log and drop excess. The main group is exempt (it's the orchestrator).

- **Authorization matrix** — Currently: main group can mention anyone, others can mention anyone registered. Add explicit rules: `farm` can talk to `promptly` and `sous`, but not `main`. Define in a config file, not code. Default: open (current behavior).

### Speculative

- **Subagent spawning** — Promptly says `@spawn code-reviewer "review this PR"`. NanoClaw creates a temporary agent with a short-lived Matrix room, runs the task, collects the result, destroys the room. The `register_group` IPC already supports dynamic registration — spawning is register + run + deregister.

- **Shared context protocol** — When agent A mentions agent B, optionally attach context: the last N turns of conversation, relevant HoloLoom memories, the current intent. Agent B gets a richer starting point than just the message text. Transported as a JSON blob in the synthetic DB message, invisible in Matrix.

- **Consensus protocol** — Ask multiple agents the same question, collect responses, synthesize. Promptly says `@farm @sous what should we cook tonight?` — both respond, Promptly synthesizes. The mention router already handles multiple mentions per message. Need: collect multiple relay responses before synthesizing.

- **Agent reputation** — Track relay success rate per agent pair. If `farm` consistently times out when `promptly` asks it things, lower its relay priority. Natural extension of the bandit — but over agents instead of models.

- **Matrix threading** — Use Matrix reply threads for relay conversations instead of flat messages. The relay response would be a reply to the injected message. Keeps the room tidy when multiple cross-agent conversations happen simultaneously.

---

## Ordering

No strict dependencies, but natural sequence:

1. **Bandit persistence** (routing) — prevents learning loss, 30 minutes of work
2. **Trace IDs** (observability) — makes everything else debuggable
3. **Multi-turn relay** (coordination) — unlocks real agent collaboration
4. **Routing decision log** (observability) — dataset for better classifier
5. **Better classifier** (routing) — trained on real data from step 4
6. **Structured envelopes** (coordination) — programmatic agent communication
7. Everything else in any order

---

## What This Is Not

This is not a framework. There is no `AgentCoordinator` class, no `RoutingPipeline`, no plugin system. It's three sets of focused primitives:

- A bandit that learns model preferences
- A log that records decisions
- A relay that forwards messages between rooms

They compose because they're small. Adding a model is adding a dict. Adding an agent is registering a room. Adding observability is logging a field. The complexity budget is: can you explain it in one sentence?
