# HoloLoom Agent System

**Specialized agent bots with working memory and learning**

## Overview

The Agent System provides domain-specific AI agents that:
- **Learn from success**: Identify patterns in successful queries and apply them automatically
- **Maintain context**: "Pins" persist across queries, creating agent-specific working memory
- **Share knowledge**: All agents access a shared knowledge graph but prioritize different semantic regions
- **Persist state**: Learning survives across sessions

## Architecture: Trinity Working Memory

Working memory operates on **three complementary substrates**:

### 1. Semantic (Geometric)
- **WHERE** in 244D semantic space
- Focus vector with momentum
- Attention radius for semantic neighborhood

### 2. Graph (Network)
- **WHICH** nodes are activated
- Activation propagation through edges
- Decay dynamics

### 3. Computational (Warp Space)
- **WHAT'S** ready for computation
- Tensioned threads in warp manifold
- Persistent tension profiles (pins)

**Key Insight**: The three levels interact naturally:
```
Semantic focus → activates nearby graph nodes → tension into warp space
Results → update semantic focus (feedback loop)
```

## Quick Start

```python
from HoloLoom.agents import create_agent
from HoloLoom.memory.graph import KG
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Query

# Create shared knowledge
kg = KG()
emb = MatryoshkaEmbeddings()

# Create budget advisor
async with create_agent('budget', kg, emb) as agent:
    # Pin persistent context
    agent.pin("company budget policy 2024", weight=0.7)

    # Query
    result = await agent.query(Query(text="What's the Q4 marketing budget?"))

    # View working memory
    print(agent.get_working_memory_summary())
    # {
    #   'semantic_focus': ['monetary_value', 'temporal_scope', 'resource_allocation'],
    #   'activated_nodes': 12,
    #   'tensioned_threads': 8,
    #   'pinned_concepts': 1
    # }

    # View learning stats
    print(agent.get_learning_stats())
    # {
    #   'total_snapshots': 5,
    #   'patterns_learned': 2,
    #   'avg_confidence': 0.87
    # }
```

## Available Agent Profiles

### Budget Advisor
- **Domain**: Financial planning and cost analysis
- **Priorities**: Accuracy, sustainability, risk awareness
- **Semantic dimensions**: `monetary_value`, `cost_benefit`, `resource_allocation`, `risk_tolerance`
- **High confidence bar**: 0.80 success threshold

### Architecture Reviewer
- **Domain**: Software design evaluation
- **Priorities**: Scalability, maintainability, modularity
- **Semantic dimensions**: `hierarchical_depth`, `modularity`, `coupling_strength`, `abstraction_level`
- **Broad attention**: 0.35 attention radius

### Code Reviewer
- **Domain**: Code quality and best practices
- **Priorities**: Correctness, readability, performance, security
- **Semantic dimensions**: `code_clarity`, `maintainability`, `performance`, `security`

### Research Assistant
- **Domain**: Open-ended exploration
- **Priorities**: Thoroughness, synthesis, critical thinking
- **Semantic dimensions**: `conceptual_breadth`, `analytical_depth`, `novelty`, `synthesis`
- **Very broad attention**: 0.40 attention radius

### Planning Agent
- **Domain**: Task decomposition and sequencing
- **Priorities**: Completeness, logical ordering, feasibility
- **Semantic dimensions**: `temporal_ordering`, `dependency_structure`, `goal_decomposition`

### General Agent
- **Domain**: Jack of all trades
- **Priorities**: Accuracy, clarity, practicality
- **Balanced configuration**

## Pattern Learning

Agents automatically learn from successful queries:

```python
# Query 1: High confidence
result = await agent.query(Query(text="Q4 budget breakdown"))
# confidence: 0.92 → LEARNS pattern

# Later: Similar query benefits from learned pattern
result = await agent.query(Query(text="Q3 budget breakdown"))
# confidence: 0.95 (improved via pattern application)
```

**What Gets Learned**:
1. **Semantic regions**: "When focus is near [cost_analysis, quarterly_planning], activate these threads"
2. **Critical threads**: "Budget queries need these threads 80%+ of time: ['fiscal_policy', 'Q4_targets']"
3. **Activation patterns**: "For marketing questions, 'marketing_spend' should be at 0.85 activation"
4. **Warp operations**: Successful computational patterns

## Persistence

Learning state persists across sessions:

```python
# Session 1: Train agent
async with create_agent('budget', kg, emb, persist_dir=Path('./my_agents')) as agent:
    await agent.query(Query(text="budget query"))
    # Learning happens automatically

# Session 2: Load agent (weeks later)
async with create_agent('budget', kg, emb, persist_dir=Path('./my_agents')) as agent:
    # Learned patterns automatically loaded
    stats = agent.get_learning_stats()
    print(f"Loaded {stats['patterns_learned']} patterns")
```

**What Persists**:
- Learned patterns (semantic regions, critical threads, activation patterns)
- Snapshot history (rolling window of 1000)
- Pattern statistics (success rates, confidence)

## Working Memory Operations

### Pinning (Persistent Context)
```python
# Pin concepts across all three substrates
agent.pin("company budget policy 2024", weight=0.7)
agent.pin("Q4 fiscal targets", weight=0.6)

# Pins affect:
# - Semantic: Pull focus toward concept
# - Graph: Keep node highly activated
# - Computational: Keep thread tensioned
```

### Relaxation
```python
# Decay activations, detension threads (but keep pins)
await agent.relax()
```

### State Summary
```python
summary = agent.get_working_memory_summary()
# {
#   'semantic_focus': ['top', 'semantic', 'dimensions'],
#   'focus_magnitude': 1.0,
#   'activated_nodes': 12,
#   'highly_activated_nodes': 5,  # activation > 0.7
#   'tensioned_threads': 8,
#   'pinned_concepts': 2,
#   'attention_radius': 0.30
# }
```

## Statistics

### Agent Stats
```python
stats = agent.get_agent_stats()
# AgentStats(
#   total_queries=25,
#   successful_queries=22,
#   avg_confidence=0.87,
#   cache_hits=18,
#   cache_misses=7
# )

print(f"Success rate: {stats.success_rate():.1%}")  # 88.0%
print(f"Cache hit rate: {stats.cache_hit_rate():.1%}")  # 72.0%
```

### Learning Stats
```python
learning = agent.get_learning_stats()
# {
#   'total_snapshots': 25,
#   'successful_snapshots': 22,
#   'success_rate': 0.88,
#   'patterns_learned': 5,
#   'avg_confidence': 0.87,
#   'avg_confidence_successful': 0.91
# }
```

## Advanced: Custom Profiles

```python
from HoloLoom.agents.types import AgentProfile, AgentDomain

custom_profile = AgentProfile(
    agent_id="custom_bot",
    name="Custom Bot",
    domain=AgentDomain.GENERAL,

    system_prompt="You are a custom agent...",

    priorities=["accuracy", "speed"],

    semantic_dimensions=[
        "dimension_1",
        "dimension_2",
        # ... from EXTENDED_244_DIMENSIONS
    ],

    preferred_tools=["tool1", "tool2"],

    # Memory configuration
    context_window_size=15,
    attention_radius=0.35,
    momentum=0.6,

    # Learning configuration
    enable_learning=True,
    success_threshold=0.75,
    refinement_threshold=0.70
)

agent = AgentOrchestrator(
    profile=custom_profile,
    shared_knowledge=kg,
    embedding_model=emb
)
```

## Cross-Agent Knowledge Sharing

```python
# Shared knowledge graph
kg = KG()
emb = MatryoshkaEmbeddings()

# Create multiple agents sharing same KG
budget = create_agent('budget', kg, emb)
arch = create_agent('architecture', kg, emb)

async with budget, arch:
    # Budget agent can access architecture knowledge
    result1 = await budget.query(Query(text="Infrastructure cost of microservices?"))

    # Architecture agent can access budget knowledge
    result2 = await arch.query(Query(text="Budget for scaling infrastructure?"))

    # Same knowledge graph, different semantic priorities
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentOrchestrator                        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              AgentWorkingMemory                       │ │
│  │                                                       │ │
│  │  LEVEL 1: Semantic (244D focus vector)               │ │
│  │    - Focus with momentum                              │ │
│  │    - Attention radius                                 │ │
│  │                                                       │ │
│  │  LEVEL 2: Graph (activation map)                     │ │
│  │    - Node activation levels                           │ │
│  │    - Propagation through edges                        │ │
│  │                                                       │ │
│  │  LEVEL 3: Computational (tensioned threads)          │ │
│  │    - Warp space readiness                             │ │
│  │    - Persistent tension profile (pins)                │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           WorkingMemoryLearner                        │ │
│  │                                                       │ │
│  │  - Records snapshots (semantic + graph + comp)       │ │
│  │  - Learns patterns from success                       │ │
│  │  - Suggests improvements                              │ │
│  │  - Persists to disk                                   │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              AgentProfile                             │ │
│  │                                                       │ │
│  │  - Domain specialization                              │ │
│  │  - Semantic priorities                                │ │
│  │  - Tool preferences                                   │ │
│  │  - Learning configuration                             │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              Shared Knowledge Graph (KG)
```

## Implementation Details

### File Structure
```
HoloLoom/agents/
├── __init__.py              # Public API
├── types.py                 # Data structures
├── profiles.py              # Profile templates
├── working_memory.py        # Trinity substrate
├── learner.py               # Pattern learning
├── orchestrator.py          # Main agent class
├── test_agent_system.py     # Tests
└── README.md                # This file
```

### Dependencies
- `HoloLoom.memory.graph` - Knowledge graph (KG)
- `HoloLoom.embedding.spectral` - Matryoshka embeddings
- `HoloLoom.semantic_calculus` - 244D semantic space
- `HoloLoom.documentation.types` - Query, Spacetime, MemoryShard

### Performance
- **Working memory overhead**: <3ms per query
- **Learning overhead**: <1ms per query (snapshot recording)
- **Persistence overhead**: ~50ms (save on close)
- **Pattern application**: <2ms (if patterns exist)

### Storage
- **Snapshots**: JSONL format (rolling window of 1000)
- **Patterns**: JSON format (all patterns)
- **Location**: `{persist_dir}/{agent_id}/`

## Testing

```bash
# Run tests
pytest HoloLoom/agents/test_agent_system.py -v

# Run specific test
pytest HoloLoom/agents/test_agent_system.py::test_agent_query -v
```

## Demo

```bash
# Run comprehensive demo
python demos/demo_agent_system.py
```

## Future Enhancements

1. **Agent Collaboration**: Agents consulting each other
2. **Cross-Agent Learning**: Agents share successful patterns
3. **Routine Detection**: Auto-detect repeated tasks
4. **Context Compression**: Summarize long interaction history
5. **Agent Dashboard**: Real-time monitoring UI
6. **Template Library**: More domain-specific profiles

## Philosophy

> **"Working memory is not a separate data structure - it's heightened activation in the awareness graph."**

The trinity substrate (semantic + graph + computational) provides three complementary views of the same phenomenon: what's currently "top of mind" for the agent.

Pins create persistent context. Learning saves what works. The result is an agent that gets smarter over time, maintains domain-specific focus, and shares knowledge with other agents while maintaining its own identity.
