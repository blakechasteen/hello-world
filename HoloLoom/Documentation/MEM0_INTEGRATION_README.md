# Mem0 Integration for HoloLoom

This directory contains documentation and implementation for integrating [mem0](https://github.com/mem0ai/mem0) with HoloLoom's memory system.

## Overview

**Mem0** is an intelligent memory layer for AI agents that provides:
- Multi-level memory (User, Session, Agent)
- LLM-based intelligent extraction
- Memory filtering and decay
- Cross-session learning

**HoloLoom** has a sophisticated multi-scale retrieval system with:
- Matryoshka embeddings (96d, 192d, 384d)
- Knowledge graph reasoning
- Domain-specific motif detection
- Spectral features

**The Integration** combines both systems collaboratively:
- Both systems contribute to **WHAT to remember** (from different perspectives)
- Both systems contribute to **HOW to recall** (with different strategies)
- Results are fused using weighted combination

**Mem0's Perspective**:
- What to remember: User preferences, important facts (LLM-filtered)
- How to recall: User-specific filtering, session awareness

**HoloLoom's Perspective**:
- What to remember: Domain entities, graph relationships, temporal context
- How to recall: Multi-scale semantic search, motif matching, graph traversal

## Quick Links

- **[Integration Analysis](./MEM0_INTEGRATION_ANALYSIS.md)** - Comprehensive architectural analysis, benefits, risks, and implementation phases
- **[Quickstart Guide](./MEM0_QUICKSTART.md)** - Get started in under 10 minutes
- **[Implementation Code](../memory/mem0_adapter.py)** - The actual integration adapter
- **[Example Script](../examples/hybrid_memory_example.py)** - Working demo

## Architecture Diagram

```
User Query: "Winter prep for hives?"
    ↓
┌───────────────────────────────────────────────────────┐
│        HybridMemoryManager (Parallel Processing)      │
├────────────────────┬──────────────────────────────────┤
│      Mem0          │         HoloLoom                 │
│                    │                                  │
│ What to Remember:  │  What to Remember:               │
│  • User prefs      │   • Domain entities              │
│  • Important facts │   • Graph relations              │
│  • Decay old info  │   • Temporal threads             │
│                    │                                  │
│ How to Recall:     │  How to Recall:                  │
│  • User filtering  │   • Multi-scale search           │
│  • Session context │   • Motif matching               │
│  • Relevance score │   • Graph traversal              │
└────────┬───────────┴──────────────┬───────────────────┘
         │                          │
         └──────── Weighted Fusion ─┘
                       ↓
            ┌──────────────────┐
            │  Fused Context   │
            │ (Personal +      │
            │  Domain-rich)    │
            └──────────────────┘
```

## Quick Start (60 seconds)

1. **Install mem0**:
   ```bash
   pip install mem0ai
   ```

2. **Basic usage**:
   ```python
   from HoloLoom.memory.mem0_adapter import create_hybrid_memory, Mem0Config
   
   # Create config
   config = Mem0Config(enabled=True)
   
   # Create hybrid manager
   hybrid = create_hybrid_memory(hololoom_memory, config)
   
   # Store with intelligent extraction
   await hybrid.store(query, results, features, user_id="blake")
   
   # Retrieve with fusion
   context = await hybrid.retrieve(query, user_id="blake")
   ```

3. **Run the demo**:
   ```bash
   python HoloLoom/examples/hybrid_memory_example.py
   ```

## Key Benefits

| Feature | Without Mem0 | With Mem0 |
|---------|-------------|-----------|
| Entity Extraction | Manual/heuristic | LLM-based |
| User Personalization | Limited | Multi-level tracking |
| Memory Filtering | None | Intelligent decay |
| Cross-session Learning | Manual | Automatic |
| Token Usage | Baseline | -90% (mem0 benchmark) |

## When to Use Mem0 Integration

**Use mem0 when**:
- ✅ You need user-specific personalization
- ✅ You want intelligent memory extraction
- ✅ You need cross-session continuity
- ✅ You want production-ready memory management
- ✅ You need to reduce token costs

**Stick with HoloLoom-only when**:
- ✅ You want full control over extraction
- ✅ You don't need user tracking
- ✅ You want minimal dependencies
- ✅ Multi-scale retrieval is sufficient

## Configuration Options

### HoloLoom-Only (Default)
```python
config = Mem0Config(enabled=False)
```

### Hybrid (Recommended)
```python
config = Mem0Config(
    enabled=True,
    extraction_enabled=True,
    graph_sync_enabled=True,
    mem0_weight=0.3,        # 30% mem0
    hololoom_weight=0.7,    # 70% HoloLoom
)
```

### Mem0-Heavy (More Personalization)
```python
config = Mem0Config(
    enabled=True,
    mem0_weight=0.7,        # 70% mem0
    hololoom_weight=0.3,    # 30% HoloLoom
)
```

## Implementation Status

- ✅ **Phase 0**: Analysis and design complete
- 🚧 **Phase 1**: POC implementation (mem0_adapter.py)
- ⏳ **Phase 2**: Deep integration with orchestrator
- ⏳ **Phase 3**: Optimization and benchmarking

## Files in This Integration

```
HoloLoom/
├── Documentation/
│   ├── MEM0_INTEGRATION_README.md           ← You are here
│   ├── MEM0_INTEGRATION_ANALYSIS.md         ← Full analysis
│   ├── MEM0_QUICKSTART.md                   ← Quick start guide
│   └── MATHEMATICAL_MODULES_DESIGN.md       ← Math modules design
├── memory/
│   ├── mem0_adapter.py                      ← Integration code
│   ├── neo4j_adapter.py                     ← TODO: Neo4j threads
│   └── qdrant_store.py                      ← TODO: Qdrant vectors
├── math/
│   ├── hofstadter.py                        ← Hofstadter sequences
│   ├── strange_loops.py                     ← TODO: Loop detection
│   └── spectral_advanced.py                 ← TODO: Enhanced spectral
└── examples/
    ├── hybrid_memory_example.py             ← Working demo
    └── mathematical_memory_example.py       ← TODO: Math demo
```

## Performance Expectations

Based on mem0's benchmarks:

- **Accuracy**: +26% improvement over baseline memory
- **Speed**: 91% faster than full-context approaches
- **Cost**: 90% fewer tokens than full-context
- **Latency**: Sub-50ms memory lookups

HoloLoom's multi-scale retrieval adds:
- Coarse-to-fine search (96d → 384d)
- Domain-specific reasoning
- Graph-based entity relationships

## Example Use Cases

### 1. Beekeeping Assistant
```python
# User asks about hive status
query = Query(text="How is Hive Jodi doing?")

# Mem0 remembers:
# - User has 3 hives: Jodi, Aurora, Luna
# - Last inspection: Oct 13, 2025
# - User prefers organic treatments

# HoloLoom retrieves:
# - Multi-scale semantic search on "hive status"
# - Knowledge graph: Jodi → inspection notes
# - Domain motifs: HIVE_INSPECTION, SEASONAL
```

### 2. Personalized Recommendations
```python
# Mem0 tracks user preferences across sessions
# HoloLoom provides domain-specific context

# Combined result:
# "Based on your preference for organic treatments (mem0)
#  and current varroa mite season (HoloLoom),
#  I recommend formic acid treatment."
```

### 3. Cross-Session Learning
```python
# Session 1: "I prefer morning inspections"
# (Mem0 stores user preference)

# Session 2: "When should I inspect?"
# (Mem0 recalls preference, HoloLoom provides seasonal timing)

# Response: "Morning inspections are best. Based on the season..."
```

## Testing

Run the test suite:
```bash
# Test HoloLoom memory only
python HoloLoom/tests/test_orchestrator.py

# Test hybrid memory (requires mem0ai)
python HoloLoom/examples/hybrid_memory_example.py

# Test specific components
python -m pytest HoloLoom/tests/test_mem0_adapter.py  # TODO: Create tests
```

## Troubleshooting

### Mem0 not installed
```bash
pip install mem0ai
```

### Import errors
```python
# Make sure you're running from repository root
import sys
sys.path.insert(0, '/path/to/mythRL')
```

### API key issues (managed platform)
```bash
export MEM0_API_KEY=your_key_here
```

Or in code:
```python
config = Mem0Config(api_key="your_key_here")
```

## Roadmap

### Phase 1: POC (Current)
- [x] Architecture analysis
- [x] Integration adapter (`mem0_adapter.py`)
- [x] Example script
- [x] Mathematical modules design
- [x] Hofstadter sequences implementation
- [ ] Basic tests
- [ ] Benchmark comparison

### Phase 2: Deep Integration
- [ ] Orchestrator integration
- [ ] Config file updates
- [ ] Neo4j adapter implementation
- [ ] Qdrant vector store integration
- [ ] Entity extraction improvements
- [ ] Graph sync optimization

### Phase 3: Mathematical Enhancement
- [ ] Strange loop detection
- [ ] Spectral graph features
- [ ] Hofstadter resonance indexing
- [ ] Thread-based retrieval (Neo4j)
- [ ] Multi-scale Qdrant search

### Phase 4: Production Ready
- [ ] Performance tuning
- [ ] Memory decay implementation
- [ ] User dashboard
- [ ] Documentation polish
- [ ] Production deployment guide

## Contributing

To contribute to this integration:

1. Read the [Integration Analysis](./MEM0_INTEGRATION_ANALYSIS.md)
2. Review the [code](../memory/mem0_adapter.py)
3. Run the [example](../examples/hybrid_memory_example.py)
4. Open issues or PRs for improvements

## Resources

### Mem0
- [GitHub](https://github.com/mem0ai/mem0)
- [Documentation](https://docs.mem0.ai/)
- [Research Paper](https://mem0.ai/research)
- [Discord Community](https://mem0.dev/DiG)

### HoloLoom
- [Orchestrator](../Orchestrator.py)
- [Memory Cache](../memory/cache.py)
- [Knowledge Graph](../memory/graph.py)
- [Config](../config.py)

## FAQ

**Q: Does this replace HoloLoom's memory system?**  
A: No. It augments it. You can use both together or disable mem0 entirely.

**Q: Do I need the mem0 managed platform?**  
A: No. The open-source version works locally without an API key.

**Q: Will this slow down my system?**  
A: Minimal overhead. Mem0 adds sub-50ms lookups. You can use HoloLoom's fast mode for speed.

**Q: Can I use only mem0 without HoloLoom?**  
A: Not recommended. You'd lose multi-scale retrieval and domain reasoning.

**Q: How do fusion weights work?**  
A: Results from both systems are weighted and combined. 30% mem0 + 70% HoloLoom is the default.

**Q: What if mem0 is down or unavailable?**  
A: The system automatically falls back to HoloLoom-only mode.

## License

This integration follows the same license as HoloLoom. Mem0 itself is Apache 2.0 licensed.

## Support

- **HoloLoom Integration**: Open an issue in this repository
- **Mem0 Issues**: [mem0ai/mem0 Issues](https://github.com/mem0ai/mem0/issues)
- **General Questions**: Discord or GitHub Discussions

---

**Status**: 🚧 POC Complete, Phase 1 Implementation  
**Last Updated**: October 22, 2025  
**Maintainer**: HoloLoom Team
