# DreamWeaver: Open-Source World Building Component

**Status**: Phase 0 - Architecture Complete ✅
**Version**: 0.1.0
**License**: Apache 2.0 (planned)

---

## What is DreamWeaver?

DreamWeaver transforms HoloLoom from a decision-making engine into a **collaborative storytelling platform**. It enables authors, game designers, and creators to build persistent, coherent fictional worlds that evolve through interaction while maintaining narrative consistency.

### Core Philosophy

> **"Dreams are not random - they are recursive narratives seeking coherence."**

DreamWeaver treats world building as a **constraint satisfaction problem in narrative space**, where consistency, causality, and creativity must balance. It extends HoloLoom's weaving metaphor to world building:

| HoloLoom Concept | DreamWeaver Equivalent |
|-----------------|----------------------|
| **Yarn Graph** | World Memory (persistent entities) |
| **Warp Space** | Narrative Manifold (generation space) |
| **Convergence Engine** | Story Collapse (probabilities → events) |
| **Reflection Buffer** | Player Feedback (learn from interactions) |
| **Thompson Sampling** | Narrative Exploration (story branches) |

---

## Quick Start

### Installation

```bash
# DreamWeaver is part of HoloLoom
cd mythRL
pip install -e .

# Optional dependencies for LLM integration (Phase 2+)
pip install openai anthropic  # Or your preferred provider
```

### Basic Usage

```python
from HoloLoom.dreamweaving import DreamWeaver, DreamWeaverConfig
from HoloLoom.config import Config

# Configure DreamWeaver
config = DreamWeaverConfig(
    world_id="my_fantasy_world",
    generation_mode=GenerationMode.HYBRID,
    consistency_level=ConsistencyLevel.BALANCED,
)

# Configure HoloLoom (use FUSED for best quality)
holo_config = Config.fused()

# Initialize DreamWeaver
async with DreamWeaver(config, holo_config) as weaver:
    # Generate initial world
    world_state = await weaver.bootstrap_world(
        seed_prompt="A medieval fantasy world with magic and dragons"
    )

    # Evolve world with user input
    world_state = await weaver.step(
        user_input="I explore the ancient forest",
        world_state=world_state
    )

    # Query world state
    result = await weaver.query_world(
        "What factions exist in this world?",
        world_state
    )

    print(result)
```

### Running the Demo

```bash
PYTHONPATH=. python demos/demo_dreamweaver.py
```

This demonstrates:
- World bootstrapping
- Entity generation (characters, locations)
- Event generation (discoveries, conflicts)
- World evolution
- Consistency checking
- Narrative thread management

---

## Features

### Phase 0 (Current) ✅

**Architecture Complete**:
- ✅ Core data structures (WorldEntity, NarrativeEvent, WorldState)
- ✅ Protocol-based design (WorldBuilderProtocol, ConsistencyCheckerProtocol)
- ✅ DreamWeaver orchestrator skeleton
- ✅ HoloLoom integration points
- ✅ Async lifecycle management
- ✅ World state persistence (JSON)
- ✅ Metrics tracking

**What Works**:
- Basic world initialization
- Entity/event data structures
- State management
- Configuration system

**What's Stubbed** (Phase 1+):
- Rule-based entity generation
- Consistency rule checking
- LLM integration
- Story-aware knowledge graph

### Phase 1 (Weeks 1-8) 🚧

**Core World Building**:
- Narrative memory (story-aware KG)
- World fabric (versioning + persistence)
- Consistency engine (rule-based checking)
- Generative loom (template-based generation)

### Phase 2 (Weeks 9-12) 📅

**LLM Integration**:
- Natural language generation
- Dynamic dialogue
- Context-aware descriptions

### Phase 3 (Weeks 13-18) 📅

**Collaborative Authoring**:
- Human-AI co-creation
- Authorship tracking
- Web dashboard

See [DREAMWEAVER_ROADMAP.md](../../../DREAMWEAVER_ROADMAP.md) for complete 6-phase plan.

---

## Architecture

### 5 Core Components

```
DreamWeaver (Orchestrator)
    │
    ├── World Fabric (State Management)
    │   └── Versioning, persistence, rollback
    │
    ├── Narrative Memory (Story-Aware KG)
    │   └── Entities, events, causal chains
    │
    ├── Consistency Engine (Rule Checking)
    │   └── Physical, logical, narrative, stylistic
    │
    └── Generative Loom (Content Generation)
        └── Entities, events, descriptions

All components integrate with HoloLoom's 9-layer weaving cycle.
```

### Data Flow

```
User Input (Query/Action/Choice)
    ↓
Context Retrieval (Narrative Memory)
    ↓
Consistency Check (Validate against rules)
    ↓
Generation (via HoloLoom + LLM)
    ↓
Event Extraction (Parse entities/events)
    ↓
World Update (Apply to World Fabric)
    ↓
Reflection (Learn from outcome)
    ↓
Response to User
```

---

## API Reference

### Core API (10 Methods)

#### Initialization

```python
# Create DreamWeaver instance
weaver = DreamWeaver(config: DreamWeaverConfig, hololoom_config: Config)

# Bootstrap new world
world_state = await weaver.bootstrap_world(seed_prompt: str)

# Load existing world
world_state = await weaver.load_world(world_id: str)
```

#### Evolution

```python
# Evolve world with user input
world_state = await weaver.step(
    user_input: str,
    world_state: Optional[WorldState] = None
)

# Query world (read-only)
response = await weaver.query_world(
    query: str,
    world_state: Optional[WorldState] = None
)
```

#### Generation

```python
# Generate entity
entity = await weaver.generate_entity(
    entity_type: EntityType,
    context: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None
)

# Generate event
event = await weaver.generate_event(
    event_type: EventType,
    participants: List[str],
    context: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None
)
```

#### Consistency

```python
# Check consistency
violations = await weaver.check_consistency(
    world_state: WorldState,
    rules: Optional[List[ConsistencyRule]] = None
)

# Resolve contradiction
fixed_state = await weaver.resolve_contradiction(
    world_state: WorldState,
    violation: Dict[str, Any]
)
```

#### Persistence

```python
# Save world
await weaver.save_world(
    world_state: WorldState,
    save_path: Optional[str] = None
)
```

### Extended API (Future)

- Thread management: `create_thread()`, `resolve_thread()`, etc.
- Relationship operations: `add_relationship()`, `remove_relationship()`
- Temporal operations: `rewind()`, `fast_forward()`, `branch_timeline()`
- Authorship tracking: `get_authorship()`, `approve_suggestion()`
- Analytics: `get_metrics()`, `get_statistics()`, `export_graph()`

---

## Data Structures

### WorldEntity

Represents an entity in the world (character, location, faction, etc).

```python
@dataclass
class WorldEntity:
    entity_id: str
    entity_type: EntityType  # CHARACTER, LOCATION, FACTION, ARTIFACT, EVENT, CONCEPT
    name: str
    description: str
    attributes: Dict[str, Any]
    relationships: List[Dict[str, Any]]  # {target_id, relation_type, strength}
    history: List[NarrativeEvent]
    metadata: Dict[str, Any]
```

### NarrativeEvent

Represents an event in the narrative timeline.

```python
@dataclass
class NarrativeEvent:
    event_id: str
    event_type: EventType  # DIALOGUE, ACTION, DISCOVERY, CONFLICT, RESOLUTION, etc.
    timestamp: float  # In-world time
    participants: List[str]  # Entity IDs
    location_id: Optional[str]
    description: str
    consequences: Dict[str, Any]
    causal_links: Dict[str, List[str]]  # {causes: [...], effects: [...]}
    metadata: Dict[str, Any]
```

### WorldState

Complete snapshot of world at a point in time.

```python
@dataclass
class WorldState:
    timestamp: float
    entities: Dict[str, WorldEntity]
    active_threads: List[NarrativeThread]
    recent_events: List[NarrativeEvent]
    global_state: Dict[str, Any]
    consistency_violations: List[Dict[str, Any]]
    generation_context: Dict[str, Any]
```

---

## Configuration

### DreamWeaverConfig

```python
@dataclass
class DreamWeaverConfig:
    # World persistence
    world_id: str = "default_world"
    save_path: Optional[str] = None
    auto_save: bool = True
    auto_save_interval: int = 300  # 5 minutes

    # Generation settings
    generation_mode: GenerationMode = GenerationMode.HYBRID
    consistency_level: ConsistencyLevel = ConsistencyLevel.BALANCED
    max_entities: int = 10000
    max_events: int = 100000

    # Narrative settings
    max_active_threads: int = 10
    thread_tension_decay: float = 0.1  # Per time unit
    event_causality_window: int = 20  # Look back N events

    # Generation parameters
    creativity_temperature: float = 0.8
    use_thompson_sampling: bool = True
    exploration_epsilon: float = 0.15

    # LLM settings (Phase 2+)
    llm_provider: Optional[str] = None  # "openai", "anthropic", "local"
    llm_model: Optional[str] = None
    llm_temperature: float = 0.8

    # Safety
    enable_content_filtering: bool = True
    track_authorship: bool = True
    enable_version_control: bool = True
```

### Enums

```python
class GenerationMode(Enum):
    PROCEDURAL = "procedural"      # Rule-based
    NARRATIVE = "narrative"        # Story-driven
    COLLABORATIVE = "collaborative" # Human-AI co-creation
    EMERGENT = "emergent"          # Player/reader-driven
    HYBRID = "hybrid"              # Mix of above

class ConsistencyLevel(Enum):
    STRICT = "strict"        # No contradictions
    BALANCED = "balanced"    # Minor inconsistencies if narratively valuable
    LOOSE = "loose"          # Prioritize creativity
    DREAMLIKE = "dreamlike"  # Embrace surrealism
```

---

## Integration with HoloLoom

DreamWeaver sits **above** HoloLoom's 9-layer weaving cycle as a **domain-specific orchestrator**:

```
Layer 10 (New): DreamWeaver - Domain orchestration
    ↓
Layer 9: Reflection Buffer - Learn from player feedback
Layer 8: Spacetime Fabric - Complete story provenance
Layer 7: Convergence Engine - Collapse story possibilities
Layer 6: Tool Execution - Generate content (LLM calls)
Layer 5: Warp Space - Narrative manifold mathematics
Layer 4: DotPlasma - Flowing story features
Layer 3: Resonance Shed - Multi-modal story extraction
Layer 2: Chrono Trigger - In-world temporal control
Layer 1: Loom Command - Story pattern selection
Layer 0: Yarn Graph - Persistent world memory
```

### Integration Points

| HoloLoom Component | DreamWeaver Usage |
|-------------------|------------------|
| **WeavingOrchestrator** | Main reasoning engine for queries |
| **Yarn Graph (KG)** | Persistent world memory storage |
| **Warp Space** | Narrative manifold tensioning |
| **Convergence Engine** | Story possibility collapse |
| **Thompson Sampling** | Narrative branch exploration |
| **Reflection Buffer** | Player feedback learning |
| **Matryoshka Gate** | Multi-scale consistency checking |
| **Chrono Trigger** | In-world temporal control |
| **Alignment Framework** | Content filtering, safety |

---

## Performance Targets

| Operation | Target Latency | Acceptable Latency |
|-----------|---------------|-------------------|
| Entity retrieval | <5ms | <20ms |
| Event generation (rule-based) | <50ms | <200ms |
| Event generation (LLM) | <1s | <3s |
| Consistency check | <100ms | <500ms |
| World state save | <200ms | <1s |
| World state load | <300ms | <1.5s |

---

## Examples

### Example 1: Fantasy World

```python
# Create fantasy world
config = DreamWeaverConfig(
    world_id="middle_kingdoms",
    generation_mode=GenerationMode.NARRATIVE,
    consistency_level=ConsistencyLevel.BALANCED,
)

async with DreamWeaver(config, Config.fused()) as weaver:
    world = await weaver.bootstrap_world(
        "A medieval fantasy world with five kingdoms, ancient magic, and dragon riders"
    )

    # Generate protagonist
    hero = await weaver.generate_entity(
        EntityType.CHARACTER,
        context={"role": "protagonist", "world_theme": "high-fantasy"},
        constraints={"personality": "brave", "magical_affinity": "fire"}
    )

    # Player explores
    world = await weaver.step(
        "I enter the throne room and challenge the corrupt king",
        world_state=world
    )
```

### Example 2: Sci-Fi World

```python
# Create sci-fi world
config = DreamWeaverConfig(
    world_id="distant_colonies",
    generation_mode=GenerationMode.PROCEDURAL,
    consistency_level=ConsistencyLevel.STRICT,
)

async with DreamWeaver(config, Config.fused()) as weaver:
    world = await weaver.bootstrap_world(
        "A distant future where humanity has colonized the galaxy, "
        "but an ancient alien threat awakens"
    )

    # Generate location
    station = await weaver.generate_entity(
        EntityType.LOCATION,
        context={"setting": "sci-fi", "tech_level": "advanced"},
        constraints={"has_ai": True, "danger_level": "high"}
    )
```

---

## Testing

### Running Tests

```bash
# Unit tests (Phase 1+)
pytest HoloLoom/dreamweaving/tests/unit/ -v

# Integration tests (Phase 1+)
pytest HoloLoom/dreamweaving/tests/integration/ -v

# Run demo
PYTHONPATH=. python demos/demo_dreamweaver.py
```

### Test Coverage Targets

- Phase 1: 80%+ coverage
- Phase 2: 85%+ coverage
- Phase 3: 90%+ coverage

---

## Roadmap

### Phase 0: Foundation ✅ (Current)

- ✅ Architecture complete
- ✅ Core data structures
- ✅ Protocol-based design
- ✅ DreamWeaver orchestrator skeleton

### Phase 1: Core World Building (Weeks 1-8) 🚧

- Narrative memory (story-aware KG)
- World fabric (versioning + persistence)
- Consistency engine (rule-based)
- Generative loom (templates)

### Phase 2: LLM Integration (Weeks 9-12) 📅

- Natural language generation
- Dynamic dialogue
- Context-aware descriptions

### Phase 3: Collaborative Authoring (Weeks 13-18) 📅

- Human-AI co-creation
- Authorship tracking
- Web dashboard

### Phase 4: Advanced Features (Weeks 19-26) 📅

- Multi-agent collaboration
- Advanced consistency (semantic, statistical)
- Learning system
- Performance optimization

### Phase 5: Polish & Documentation (Weeks 27-30) 📅

- Complete test suite
- Full documentation
- Example worlds
- Community prep

### Phase 6: Advanced Applications (Months 7-18) 📅

- Interactive fiction engine
- Game world builder
- Writing assistant
- Collaborative storytelling

See [DREAMWEAVER_ROADMAP.md](../../../DREAMWEAVER_ROADMAP.md) for complete details.

---

## Contributing

**Phase 1 is ready for implementation!** We need:

1. **Backend Developers** (Python):
   - Implement narrative memory
   - Build consistency engine
   - Create generative loom

2. **Test Engineers**:
   - Write unit/integration tests
   - Create example worlds
   - Performance benchmarks

3. **Documentation Writers**:
   - API documentation
   - User guides
   - Tutorial creation

4. **Domain Experts**:
   - Fantasy/sci-fi world building
   - Interactive fiction design
   - Game design

---

## License

Apache 2.0 (planned for open-source release)

---

## Contact

See [CLAUDE.md](../../../CLAUDE.md) for contribution guidelines.

---

## Acknowledgments

DreamWeaver builds on:
- HoloLoom's weaving metaphor and 9-layer architecture
- Thompson Sampling for exploration
- Matryoshka embeddings for multi-scale consistency
- Alignment framework for content safety

---

**Status**: Phase 0 complete, Phase 1 ready to begin
**Version**: 0.1.0
**Last Updated**: 2025-11-05

*"Dreams are not random - they are recursive narratives seeking coherence."*
