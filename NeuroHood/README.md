# NeuroHood

**Consciousness-Level Neighborhood Simulator**

> "The Sims on steroids, ultra marathon running, and DMTxLSD."

---

## Quick Start

```python
from NeuroHood import NeuroHood, NeuroHoodConfig

config = NeuroHoodConfig.prototype()

async with NeuroHood(config) as hood:
    # Bootstrap neighborhood
    state = await hood.bootstrap_neighborhood(
        seed="A quiet suburban street with diverse neighbors"
    )

    # Get resident thoughts
    thoughts = await hood.get_resident_thoughts("alice", state)

    # Evolve simulation
    new_state = await hood.step(
        player_action="Alice knocks on Bob's door",
        state=state
    )
```

## Run the Demo

```bash
PYTHONPATH=. python demos/demo_neurohood.py
```

## What Makes This Different

- **🧠 Simulated Consciousness** - NPCs have internal dialogue, strange loops, meta-awareness
- **♾️ Recursive Self-Awareness** - Characters question their own reasoning
- **🌊 Physics-Based Relationships** - Social dynamics modeled as springs
- **⏱️ Multi-Timescale Learning** - Characters evolve across 7 parallel loops
- **🎭 Emergent Drama** - Conversations from message bus, not scripts

## Documentation

- **[NEUROHOOD_DESIGN.md](../NEUROHOOD_DESIGN.md)** - Complete design document
- **[demos/demo_neurohood.py](../demos/demo_neurohood.py)** - Working demo

## Status

**Phase 1 Complete** ✅ (Week 1-2)
- Neighborhood engine
- Conscious resident agents
- Visualization layer
- Working demo

**Next**: Phase 2 - Social Physics Integration

## Architecture

```
NeuroHood/
├── __init__.py          # Public API
├── config.py            # Configuration
├── engine.py            # Main orchestrator
├── agents/              # Resident agents
│   └── resident_agent.py
├── consciousness/       # Visualization
│   └── view.py
└── social/              # Physics engine
    └── physics.py
```

## Built On

**HoloLoom Infrastructure** (~150,000 lines):
- DreamWeaver (world building)
- Collaborative Agents (consciousness)
- Strange Loops (self-awareness)
- Internal Dialogue (reflection)
- Spring Dynamics (physics)
- 7 Learning Loops (evolution)

---

**Version**: 0.1.0
**Last Updated**: 2025-11-22
