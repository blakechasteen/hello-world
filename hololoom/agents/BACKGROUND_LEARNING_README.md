# Background MCTS Learning System

**Agents that learn continuously in the background while serving user queries.**

## Overview

The Background Learning System enables specialized agents to build expertise over time by:
- Processing user queries in the foreground (low latency)
- Learning from feedback in the background (MCTS exploration)
- Persisting learned patterns across sessions
- Serving multiple specialized agents simultaneously

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Agentic Server (FastAPI) - Port 8002                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User Query (HTTP/WebSocket)                            │
│           │                                              │
│           v                                              │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │ Agent Pool       │────────>│ MCTS Agent       │    │
│  │ (orchestrates)   │         │ - budget         │    │
│  └──────────────────┘         │ - architecture   │    │
│           │                    │ - research       │    │
│           │                    │ - code_review    │    │
│           v                    └──────────────────┘    │
│  ┌──────────────────┐                  │              │
│  │ Learning Queue   │<─────────────────┘              │
│  │ (experiences)    │                                  │
│  └──────────────────┘                                  │
│           │                                             │
│           v                                             │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │ Background       │────────>│ Pattern Store    │    │
│  │ Learner          │         │ (persistent)     │    │
│  │ (MCTS explorer)  │         │ JSON files       │    │
│  └──────────────────┘         └──────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. AgentPool

Manages multiple specialized agents with shared knowledge graph and embeddings.

**Features**:
- Spawns agents on demand
- Collects experiences for background learning
- Persists learned patterns to disk
- Reloads patterns on startup

**Usage**:
```python
from hololoom.agents.background_learner import create_agent_pool

async with await create_agent_pool(
    kg=kg,
    emb=emb,
    persist_path=Path("./agent_patterns"),
    enable_background_learning=True,
    mcts_simulations=100,
    exploration_simulations=50
) as pool:
    # Process query
    result = await pool.query('budget', Query(text="What is Q4 revenue?"))

    # Provide feedback (goes to background learning)
    await pool.feedback('budget', query, result, {'helpful': True})

    # Agents learn in background automatically
```

### 2. LearningQueue

Thread-safe async queue for collecting experiences.

**Features**:
- Non-blocking queue operations
- Statistics tracking (added, processed, pending)
- Configurable max size

### 3. BackgroundLearner

Async task that continuously explores and learns.

**Process**:
1. Wait for experience from queue
2. Run MCTS exploration on similar queries
3. Update agent's learned patterns
4. Persist patterns to disk
5. Repeat

**Statistics**:
- Learning cycles completed
- Patterns learned
- Total exploration time
- Average time per cycle

### 4. PatternStore

Persistent storage for learned patterns (JSON files).

**Features**:
- Per-agent pattern files
- In-memory cache for fast access
- Statistics persistence
- Automatic reload on startup

**Storage Format**:
```
./agent_patterns/
├── budget_patterns.json
├── budget_stats.json
├── architecture_patterns.json
├── architecture_stats.json
└── ...
```

## Agent Profiles

The system includes 6 pre-configured agent profiles:

| Agent | Domain | Focus | Threshold |
|-------|--------|-------|-----------|
| **budget** | Financial | Revenue, expenses, profit | 0.80 |
| **architecture** | Software | Scalability, modularity | 0.75 |
| **code_review** | Code quality | Bugs, style, performance | 0.85 |
| **research** | Analysis | Exploration, synthesis | 0.65 |
| **planning** | Strategy | Goals, roadmaps | 0.75 |
| **general** | Generic | All domains | 0.70 |

Each profile has:
- Custom confidence threshold
- Attention radius (semantic focus)
- Domain-specific dimensions
- Tailored activation strategy

## FastAPI Integration

The system is integrated into the agentic server at `http://localhost:8002`.

### Endpoints

**1. Query an Agent**
```bash
POST /api/agent/query
Content-Type: application/json

{
  "agent": "budget",
  "query": "What is Q4 revenue?",
  "use_mcts": true,
  "feedback": {
    "helpful": true,
    "confidence": 0.85
  }
}

# Response:
{
  "response": "Q4 revenue was $500,000...",
  "confidence": 0.85,
  "agent": "budget",
  "mcts_used": true,
  "pattern_count": 5
}
```

**2. Get Statistics**
```bash
GET /api/agent/stats

# Response:
{
  "total_queries": 42,
  "active_agents": ["budget", "research"],
  "query_count_by_agent": {"budget": 30, "research": 12},
  "patterns_by_agent": {"budget": 5, "research": 3},
  "learning_queue": {
    "queue_size": 2,
    "total_added": 42,
    "total_processed": 40,
    "pending": 2
  },
  "background_learner": {
    "running": true,
    "learning_cycles": 40,
    "patterns_learned": 8,
    "total_exploration_time_s": 120.5,
    "avg_time_per_cycle_ms": 3012.5
  }
}
```

**3. Provide Feedback**
```bash
POST /api/agent/feedback
Content-Type: application/json

{
  "agent": "budget",
  "query": "What is Q4 revenue?",
  "feedback": {
    "helpful": true,
    "confidence": 0.85,
    "latency_ms": 150
  }
}

# Response:
{
  "status": "queued",
  "queue_size": 3
}
```

## Usage Examples

### Example 1: Simple Query

```python
from hololoom.agents.background_learner import create_agent_pool
from hololoom.documentation.types import Query

async with await create_agent_pool(kg, emb) as pool:
    result = await pool.query('budget', Query(text="What is Q4 profit?"))
    print(result.response)
```

### Example 2: With Feedback

```python
async with await create_agent_pool(kg, emb) as pool:
    query = Query(text="What is Q4 profit?")
    result = await pool.query('budget', query)

    # Provide feedback (queued for background learning)
    await pool.feedback('budget', query, result, {
        'helpful': True,
        'confidence': result.confidence,
        'latency_ms': 150
    })

    # Background learner processes feedback asynchronously
```

### Example 3: Persistent Patterns

```python
# Session 1: Build patterns
async with await create_agent_pool(
    kg, emb, persist_path=Path("./patterns")
) as pool:
    for query in training_queries:
        result = await pool.query('budget', query)
        await pool.feedback('budget', query, result, {'helpful': True})

    # Patterns saved to ./patterns/budget_patterns.json

# Session 2: Patterns automatically reloaded
async with await create_agent_pool(
    kg, emb, persist_path=Path("./patterns")
) as pool:
    # Patterns from session 1 are available
    stats = pool.stats()
    print(f"Loaded {stats['patterns_by_agent']['budget']} patterns")
```

### Example 4: Multiple Agents

```python
async with await create_agent_pool(kg, emb) as pool:
    # Budget agent
    budget_result = await pool.query('budget', Query(text="Q4 revenue?"))

    # Architecture agent
    arch_result = await pool.query('architecture', Query(text="System scalability?"))

    # Research agent
    research_result = await pool.query('research', Query(text="Market trends?"))

    # All agents learn in parallel in background
```

## Running the Demo

```bash
# Run comprehensive demo (all 3 scenarios)
python demos/demo_background_learning.py

# Demo 1: Interactive learning with feedback
# Demo 2: Persistent learning across sessions
# Demo 3: Multi-agent learning in parallel
```

## Starting the Server

```bash
# Start agentic server with background learning
python hololoom/web_dashboard/agentic_server.py

# Server starts at http://localhost:8002
# Background learning automatically enabled
```

## Performance Characteristics

| Operation | Overhead | When |
|-----------|----------|------|
| Foreground query (standard) | ~150ms | Every query |
| Foreground query (MCTS) | ~3-4s | Exploration enabled |
| Background learning | ~50ms | Every feedback |
| Pattern persistence | <5ms | On shutdown |
| Pattern reload | <10ms | On startup |

**Key Insights**:
- Foreground queries remain fast (standard mode)
- MCTS overhead only when explicitly enabled
- Background learning has zero impact on query latency
- Patterns persist instantly on shutdown

## Configuration

### AgentPool Parameters

```python
pool = await create_agent_pool(
    kg=kg,                                    # Knowledge graph
    emb=emb,                                  # Embeddings
    persist_path=Path("./agent_patterns"),   # Pattern storage
    enable_background_learning=True,          # Enable/disable
    mcts_simulations=100,                     # MCTS depth (queries)
    exploration_simulations=50                # MCTS depth (learning)
)
```

### BackgroundLearner Parameters

```python
learner = BackgroundLearner(
    agent_pool=pool,
    learning_queue=queue,
    pattern_store=store,
    exploration_simulations=50,   # MCTS simulations per learning cycle
    learn_interval=1.0            # Seconds between learning cycles
)
```

## Trinity Working Memory

Each agent has a "trinity substrate" working memory:

1. **Semantic Substrate**: Focus vector in 244D/768D space with momentum
2. **Graph Substrate**: Activation propagation through knowledge graph edges
3. **Computational Substrate**: Tensioned threads ready for warp space

See [AGENT_SYSTEM_README.md](AGENT_SYSTEM_README.md) for details.

## MCTS Integration

Full MCTS integration at every decision point:

1. **Working Memory**: MCTS explores semantic space for optimal focus
2. **Pattern Validation**: MCTS simulates pattern outcomes before applying
3. **Orchestrator**: Full MCTS-powered query processing
4. **Hierarchical Planning**: Multi-scale MCTS (macro/meso/micro)

See [MCTS_README.md](MCTS_README.md) for details.

## Monitoring

### Server Logs

```bash
[AgentPool] Spawned agent: budget
[BackgroundLearner] Started (exploration: 25 sims)
[BackgroundLearner] budget: +2 patterns from 'What is Q4 revenue?'
[BackgroundLearner] budget: +1 patterns from 'What is Q4 profit?'
[AgentPool] Closed (42 queries)
  - Total queries: 42
  - Patterns learned: 8
```

### Statistics API

```bash
curl http://localhost:8002/api/agent/stats

{
  "total_queries": 42,
  "background_learner": {
    "running": true,
    "patterns_learned": 8,
    "avg_time_per_cycle_ms": 3012.5
  }
}
```

## Troubleshooting

### Agent Pool Not Available

**Symptom**: `503 Service Unavailable - Agent pool not available`

**Causes**:
1. Agent pool initialization failed
2. Knowledge graph or embeddings not available

**Solution**:
```bash
# Check server logs
python hololoom/web_dashboard/agentic_server.py

# Look for:
# "Agent Pool initialized with background learning"
# or
# "Agent Pool initialization failed: <error>"
```

### Patterns Not Persisting

**Symptom**: Patterns reset between sessions

**Causes**:
1. Incorrect persist_path
2. Permission issues
3. Patterns not saved on shutdown

**Solution**:
```python
# Verify persist_path exists
Path("./agent_patterns").mkdir(parents=True, exist_ok=True)

# Check files exist
ls -la ./agent_patterns/budget_patterns.json
```

### Slow Queries

**Symptom**: Queries take >5 seconds

**Causes**:
1. MCTS enabled with high simulation count
2. Background learner consuming CPU

**Solution**:
```python
# Reduce MCTS simulations
pool = await create_agent_pool(
    kg, emb,
    mcts_simulations=25,         # Lower for speed
    exploration_simulations=10    # Lower for speed
)

# Or disable MCTS for queries
result = await pool.query('budget', query, use_mcts=False)
```

## Future Enhancements

1. **Pattern Injection**: Method to inject learned patterns into agent.learner
2. **Cross-Agent Learning**: Share patterns between related agents
3. **Adaptive Exploration**: Adjust MCTS simulations based on query complexity
4. **Pattern Pruning**: Remove stale or low-quality patterns
5. **Distributed Learning**: Multiple background learners in parallel

## Related Documentation

- [AGENT_SYSTEM_README.md](AGENT_SYSTEM_README.md) - Trinity working memory
- [MCTS_README.md](MCTS_README.md) - MCTS integration details
- [CLAUDE.md](../../CLAUDE.md) - Complete HoloLoom documentation

## License

Part of the HoloLoom project.
