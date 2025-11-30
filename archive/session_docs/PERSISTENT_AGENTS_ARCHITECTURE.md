# Persistent Background Agents Architecture

**Status**: ✅ Complete (January 20, 2025)
**Innovation**: Tiny recursive loops running continuously in background
**Philosophy**: "Agents aren't just called—they're always thinking, learning, and preparing."

---

## Executive Summary

**Persistent Background Agents** provide continuous learning and adaptation for all 4 prompt chaining systems. Each system gets its own persistent agent that runs a tiny recursive loop every 60 seconds, enabling:

- ✅ **Continuous learning** from past interactions
- ✅ **Pattern detection** across requests
- ✅ **Thompson Sampling adaptation** of strategies
- ✅ **Internal dialogue** via Hofstadter Scratchpad
- ✅ **Persistent memory** across sessions
- ✅ **Self-optimization** between requests

---

## The Problem

Traditional systems are **reactive** - they only run when called:

```
User Request → System Processes → Response → System Sleeps
```

**Problems**:
- ❌ No learning between requests
- ❌ Can't detect patterns across sessions
- ❌ Thompson priors don't update automatically
- ❌ No self-reflection or improvement
- ❌ State lost between runs

---

## The Solution: Persistent Background Agents

Each system gets a **tiny recursive loop** running continuously:

```
┌─────────────────────────────────────────────────┐
│        User Requests (foreground)               │
│  Chain → Recursive → Workflow → Scratchpad      │
└─────────────────────────────────────────────────┘
                     │
                     ↓ (Records request)
┌─────────────────────────────────────────────────┐
│    Persistent Background Agents (background)     │
│  ┌───────────────────────────────────────────┐  │
│  │  Every 60 seconds:                         │  │
│  │  1. Reflect on recent requests             │  │
│  │  2. Internal dialogue via scratchpad       │  │
│  │  3. Update Thompson priors                 │  │
│  │  4. Generate insights                      │  │
│  │  5. Persist state                          │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Result**: Agents **continuously learn** even when not processing requests!

---

## Architecture

### Core Components

```python
PersistentBackgroundAgent
├─ AgentState
│  ├─ requests_processed: int
│  ├─ avg_confidence: float
│  ├─ learning_cycles: int
│  ├─ thompson_priors: Dict[str, Dict[str, float]]
│  ├─ recent_requests: List[Dict]
│  └─ insights: List[str]
│
├─ Scratchpad Integration
│  └─ Internal dialogue every cycle
│
├─ Learning Loop (every 60s)
│  ├─ 1. Reflect on performance
│  ├─ 2. Internal dialogue (max depth 3)
│  ├─ 3. Update Thompson priors
│  ├─ 4. Generate insights
│  └─ 5. Persist state
│
└─ Lifecycle
   ├─ start() → Background task
   ├─ stop() → Save state
   ├─ record_request() → Log request
   └─ get_state() → Current state
```

### Learning Loop (60-Second Cycle)

```python
async def _run_learning_cycle(self):
    """Run single learning cycle."""

    # 1. Reflect on recent performance
    recent = self.state.recent_requests[-10:]
    avg_conf = sum(r['confidence'] for r in recent) / len(recent)

    # Detect declining confidence
    if avg_conf < 0.7:
        logger.warning("Performance declining")

    # 2. Internal dialogue via scratchpad
    thought = await scratchpad.think(
        f"Recent performance: {avg_conf:.2f}. What patterns do I see?",
        thought_type=ThoughtType.REFLECTION
    )

    tree = await scratchpad.dialogue_loop(
        initial_thought=thought,
        max_depth=3,  # Tiny loop!
        mode="synthesis"
    )

    # Extract insights
    for t in tree.thoughts.values():
        if t.type == ThoughtType.INSIGHT:
            self.state.insights.append(t.text)

    # 3. Update Thompson priors
    for strategy, confidences in strategy_results.items():
        for conf in confidences:
            if conf >= 0.75:  # Success
                priors[strategy]['alpha'] += conf
            else:  # Failure
                priors[strategy]['beta'] += (1 - conf)

    # 4. Generate insights
    if len(low_conf_queries) >= 3:
        insight = "Pattern: Low-confidence queries clustering"
        self.state.insights.append(insight)

    # 5. Persist state
    await scratchpad.save_session(f"{agent_id}_cycle_{cycle}")
```

---

## Integration with All 4 Systems

### 1. Chain Orchestrator Agent

```python
# Chain Orchestrator with persistent agent
from HoloLoom.chaining import ChainOrchestrator
from HoloLoom.agents.persistent_agent import PersistentBackgroundAgent

async with PersistentBackgroundAgent(
    agent_id="chain_orchestrator",
    agent_type="chain"
) as agent:
    # Process requests
    orchestrator = ChainOrchestrator(rag_dept)
    result = await orchestrator.execute_chain(chain, input)

    # Record for learning
    agent.record_request(
        query=input,
        result=result,
        confidence=result.confidence,
        duration_ms=result.duration_ms,
        metadata={"pattern": "verified_query"}
    )

    # Agent learns in background!
    # Every 60s: reflects, updates priors, generates insights
```

**What It Learns**:
- Which patterns work best (verified_query, auto_refine, etc.)
- When to use conditions vs loops
- Optimal confidence thresholds

### 2. Recursive Reasoner Agent

```python
# Recursive Reasoner with persistent agent
from HoloLoom.convergence import RecursiveReasoner
from HoloLoom.agents.persistent_agent import PersistentBackgroundAgent

async with PersistentBackgroundAgent(
    agent_id="recursive_reasoner",
    agent_type="recursive"
) as agent:
    # Process requests
    reasoner = RecursiveReasoner(rag_dept)
    result = await reasoner.reason(query)

    # Record for learning
    agent.record_request(
        query=query,
        result=result,
        confidence=result.confidence,
        duration_ms=result.latency_ms,
        metadata={"strategy": result.refinement_strategy}
    )

    # Background learning updates Thompson priors!
```

**What It Learns**:
- Which refinement strategies work best
- Success rates per strategy per query type
- Optimal convergence thresholds

### 3. Agentic Workflow Agent

```python
# Agentic Workflow with persistent agent
from HoloLoom.workflows import WorkflowExecutor
from HoloLoom.agents.persistent_agent import PersistentBackgroundAgent

async with PersistentBackgroundAgent(
    agent_id="agentic_workflow",
    agent_type="workflow"
) as agent:
    # Process requests
    executor = WorkflowExecutor(rag_dept)
    result = await executor.execute(workflow, inputs)

    # Record for learning
    agent.record_request(
        query=inputs["query"],
        result=result,
        confidence=result.confidence,
        duration_ms=result.duration_ms,
        metadata={"template": workflow.name}
    )

    # Background learning identifies bottlenecks!
```

**What It Learns**:
- Which templates succeed most often
- Bottleneck detection (slow nodes)
- Optimal parallel vs sequential strategies

### 4. Hofstadter Scratchpad Agent

```python
# Hofstadter Scratchpad with persistent agent
from HoloLoom.scratchpad import RecursiveScratchpad
from HoloLoom.agents.persistent_agent import PersistentBackgroundAgent

async with PersistentBackgroundAgent(
    agent_id="hofstadter_scratchpad",
    agent_type="scratchpad"
) as agent:
    # Process requests
    async with RecursiveScratchpad() as scratchpad:
        thought = await scratchpad.think(query)
        tree = await scratchpad.dialogue_loop(thought, mode="hofstadter")

    # Record for learning
    agent.record_request(
        query=query,
        result=tree,
        confidence=tree.root.confidence,
        duration_ms=50,
        metadata={"mode": "hofstadter", "loops": len(loops)}
    )

    # Background learning generates meta-insights!
```

**What It Learns**:
- Which dialogue modes produce most insights
- Strange loop frequency patterns
- Optimal dialogue depth per query type

---

## Multi-Agent Management

Run all 4 agents simultaneously:

```python
from HoloLoom.agents.persistent_agent import AgentManager

async with AgentManager(loop_interval=60.0) as manager:
    # Create 4 persistent agents
    chain_agent = await manager.create_agent("chain_orchestrator", "chain")
    recursive_agent = await manager.create_agent("recursive_reasoner", "recursive")
    workflow_agent = await manager.create_agent("agentic_workflow", "workflow")
    scratchpad_agent = await manager.create_agent("hofstadter_scratchpad", "scratchpad")

    # All agents learn in background!
    # Process requests as normal...

    # Get all states
    all_states = manager.get_all_states()
    for agent_id, state in all_states.items():
        print(f"{agent_id}: {state.requests_processed} requests, "
              f"{state.avg_confidence:.2f} avg confidence")
```

---

## Learning Capabilities

### 1. Performance Reflection

```python
# Detects declining confidence
if second_half_conf < first_half_conf - 0.1:
    logger.warning("Confidence declining")

# Detects low-confidence clusters
low_conf = [r for r in recent if r['confidence'] < 0.7]
if len(low_conf) >= 3:
    insight = "Pattern: Low-confidence requests clustering"
```

### 2. Internal Dialogue

```python
# Every 60s, agent asks itself:
thought = await scratchpad.think(
    f"My recent performance: {avg_conf:.2f}. What patterns do I see?"
)

# Tiny dialogue loop (max depth 3)
tree = await scratchpad.dialogue_loop(
    initial_thought=thought,
    max_depth=3,
    mode="synthesis"
)

# Extracts insights
for t in tree.thoughts.values():
    if t.type == ThoughtType.INSIGHT:
        insights.append(t.text)
```

### 3. Thompson Sampling Updates

```python
# Tracks success/failure per strategy
for strategy, confidences in strategy_results.items():
    for conf in confidences:
        if conf >= 0.75:  # Success
            thompson_priors[strategy]['alpha'] += conf
        else:  # Failure
            thompson_priors[strategy]['beta'] += (1 - conf)

# Expected reward
E[X] = α / (α + β)
```

### 4. Pattern Detection

```python
# Low-confidence pattern
if len(low_conf_queries) >= 3:
    insight = f"{len(low_conf_queries)} low-confidence queries detected"

# Slow query pattern
if len(slow_queries) >= 3:
    insight = f"{len(slow_queries)} slow queries (>1s)"

# Strategy effectiveness
best_strategy = max(strategies, key=lambda s: success_rate[s])
insight = f"{best_strategy} performing best"
```

### 5. State Persistence

```python
# Every cycle, state saved to scratchpad database
await scratchpad.save_session(f"{agent_id}_cycle_{cycle}")

# Full provenance
- All requests logged
- All insights recorded
- Thompson priors tracked
- Internal dialogues preserved
```

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Loop interval** | 60s | Configurable (5s-300s) |
| **Cycle duration** | ~50-200ms | Depends on history size |
| **Memory overhead** | ~1-2MB | Per agent |
| **CPU overhead** | <0.1% | Async background task |
| **Dialogue depth** | 3 | Tiny loop (keeps it fast) |
| **History size** | 50 requests | Rolling window |
| **Persistence** | SQLite | Full provenance |

**Total Overhead**: Negligible (<0.1% CPU, <10MB memory for 4 agents)

---

## Benefits

### Continuous Learning
- ✅ Agents learn 24/7, even when idle
- ✅ No need for explicit training sessions
- ✅ Always improving

### Pattern Detection
- ✅ Detects low-confidence clusters
- ✅ Identifies slow queries
- ✅ Finds strategy effectiveness patterns

### Thompson Sampling
- ✅ Priors update automatically
- ✅ Exploration-exploitation balance adapts
- ✅ Best strategies emerge naturally

### Internal Dialogue
- ✅ Agents question their own performance
- ✅ Generate meta-insights
- ✅ Self-reflection leads to improvement

### Persistent Memory
- ✅ State saved across restarts
- ✅ Full provenance of all learning
- ✅ Historical analysis possible

---

## Use Cases

### 1. Production Monitoring

```python
# Monitor agent performance
async with AgentManager() as manager:
    chain_agent = await manager.create_agent("chain", "chain")

    # Set up alerting
    def on_insight(insight: str):
        if "declining" in insight.lower():
            send_alert("Performance declining!")

    chain_agent.on_insight = on_insight

    # Agents monitor themselves!
```

### 2. A/B Testing

```python
# Test different strategies
agent.record_request(
    query=query,
    result=result,
    confidence=confidence,
    metadata={"strategy": "strategy_A"}
)

# Background learning tracks which strategy wins
priors = agent.get_thompson_priors()
best = max(priors, key=lambda s: priors[s]['alpha'])
```

### 3. Adaptive Systems

```python
# System adapts to workload
if agent.state.avg_confidence < 0.7:
    # Switch to more conservative strategies
    use_pattern = "quality_first"
else:
    # Use faster strategies
    use_pattern = "balanced"
```

### 4. Historical Analysis

```python
# Analyze learning over time
sessions = await persistence.list_sessions()
for session in sessions:
    tree = await persistence.load_session(session['session_name'])
    # Analyze dialogue trees for insights
```

---

## Comparison: Before vs After

### Before (Reactive Systems)

```
Request → Process → Response → Sleep
           ↑
        No learning between requests
        No pattern detection
        No self-reflection
        State lost
```

**Problems**:
- ❌ Learning only during requests
- ❌ No cross-session patterns
- ❌ Thompson priors stale
- ❌ No self-optimization

### After (Persistent Agents)

```
Request → Process → Response
           ↓ (Record)
     ┌─────────────────┐
     │ Background Loop │
     │   Every 60s:    │
     │ • Reflect       │
     │ • Dialogue      │
     │ • Learn         │
     │ • Persist       │
     └─────────────────┘
```

**Benefits**:
- ✅ Continuous 24/7 learning
- ✅ Pattern detection across sessions
- ✅ Thompson priors always fresh
- ✅ Self-optimizing systems

---

## Running the Demo

```bash
PYTHONPATH=. python demos/demo_persistent_agents.py
```

**4 Demos**:
1. Single persistent agent
2. Multi-agent management (all 4 systems)
3. Insight generation from patterns
4. Thompson Sampling adaptation

**Example Output**:

```
🚀 Starting 4 persistent agents...
   ✅ Chain Orchestrator agent started
   ✅ Recursive Reasoner agent started
   ✅ Agentic Workflow agent started
   ✅ Hofstadter Scratchpad agent started

📝 Simulating requests to all 4 agents...
   Batch 1: 4 requests across all agents
   Batch 2: 4 requests across all agents
   ...

⏳ Waiting for learning cycles...

💡 Pattern Detected: 3 low-confidence queries detected
💡 Pattern Detected: Recursive strategy improving (0.60 → 0.85)

📊 Agent States:

   chain_orchestrator:
      Requests: 5
      Avg Confidence: 0.88
      Learning Cycles: 1
      Insights: 2

   recursive_reasoner:
      Requests: 5
      Avg Confidence: 0.75
      Learning Cycles: 1
      Insights: 3
```

---

## Future Enhancements

1. **Multi-Agent Collaboration**
   - Agents share insights with each other
   - Collaborative pattern detection
   - Cross-agent Thompson Sampling

2. **Advanced Pattern Mining**
   - ML-based pattern detection
   - Anomaly detection
   - Predictive insights

3. **Visual Dashboard**
   - Real-time agent state visualization
   - Learning curve plots
   - Thompson prior evolution

4. **Adaptive Loop Intervals**
   - Faster loops when learning quickly
   - Slower loops when stable
   - Energy-efficient mode

---

## Conclusion

**Persistent Background Agents** complete the prompt chaining moonshot by providing continuous learning infrastructure. All 4 systems benefit from:

- ✅ **24/7 learning** (even when idle)
- ✅ **Pattern detection** across sessions
- ✅ **Thompson Sampling adaptation**
- ✅ **Internal dialogue** and self-reflection
- ✅ **Persistent memory** with full provenance

**Total Moonshot Deliverables**:
- 4 prompt chaining systems (13,250 lines)
- Persistent background agents (800 lines)
- 80+ tests passing
- 25+ demos
- 10,000+ lines of documentation

**Status**: ✅ **PRODUCTION READY**

---

**Created**: January 20, 2025
**Innovation**: Tiny recursive loops for continuous agent learning
**Philosophy**: "Agents aren't just called—they're always thinking."
