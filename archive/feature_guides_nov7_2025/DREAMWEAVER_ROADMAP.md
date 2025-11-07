# DreamWeaver: Open-Source World Building Component

**Status**: 🚧 Phase 0 - Foundation (Architecture Complete)
**Timeline**: 6-phase roadmap (18 months)
**Location**: `HoloLoom/dreamweaving/`

---

## Vision Statement

> **"Dreams are not random - they are recursive narratives seeking coherence."**

DreamWeaver transforms HoloLoom from a decision-making engine into a collaborative storytelling platform. It enables authors, game designers, and creators to build persistent, coherent fictional worlds that evolve through interaction while maintaining narrative consistency.

## Core Philosophy

### The Weaving Metaphor Extended

HoloLoom's weaving metaphor naturally extends to world building:

| HoloLoom Concept | DreamWeaver Equivalent | Purpose |
|-----------------|----------------------|---------|
| **Yarn Graph** | World Memory | Persistent entities/relationships |
| **Warp Space** | Narrative Manifold | Continuous generation space |
| **Convergence Engine** | Story Collapse | Probabilities → Concrete events |
| **Reflection Buffer** | Player Feedback | Learn from interactions |
| **Thompson Sampling** | Narrative Exploration | Explore story branches |
| **Matryoshka Gate** | Multi-Scale Coherence | Scene → Chapter → Book consistency |
| **Chrono Trigger** | In-World Time | Temporal event ordering |

### Key Principles

1. **Consistency as Soft Constraint**: Allow minor inconsistencies if narratively valuable
2. **Human-AI Collaboration**: Blend authorial intent with generative creativity
3. **Graceful Degradation**: Works without LLM (rule-based), better with LLM
4. **Complete Provenance**: Track human vs AI contributions
5. **Multi-Scale Coherence**: Maintain consistency from scene to saga level
6. **Narrative Physics**: World has rules (physical, logical, narrative, stylistic)

---

## Architecture Overview

### 5 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        DreamWeaver                          │
│                     (Core Orchestrator)                     │
└──────────────┬──────────────────────────────────┬───────────┘
               │                                  │
      ┌────────▼────────┐              ┌─────────▼──────────┐
      │ World Fabric    │              │ Consistency Engine │
      │ (Persistent     │              │ (Rule Checking)    │
      │  State Manager) │              └─────────┬──────────┘
      └────────┬────────┘                        │
               │                                  │
      ┌────────▼────────┐              ┌─────────▼──────────┐
      │ Narrative Memory│              │ Generative Loom    │
      │ (Story-Aware KG)│              │ (Content Gen)      │
      └────────┬────────┘              └─────────┬──────────┘
               │                                  │
               └──────────────┬───────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ HoloLoom Integration│
                    │ (Weaving           │
                    │  Orchestrator)     │
                    └────────────────────┘
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

### Integration with HoloLoom's 9-Layer System

DreamWeaver sits **above** the 9-layer weaving cycle as a **domain-specific orchestrator**:

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

---

## 6-Phase Roadmap

### Phase 0: Foundation (Complete) ✅

**Duration**: 2 weeks
**Status**: Architecture complete, ready for Phase 1

**Deliverables**:
- ✅ `HoloLoom/dreamweaving/__init__.py` - Public API (600 lines)
- ✅ `HoloLoom/dreamweaving/core.py` - Main orchestrator (800 lines)
- ✅ `DREAMWEAVER_ROADMAP.md` - This document

**Features**:
- Core data structures (WorldEntity, NarrativeEvent, WorldState, etc.)
- Protocol-based design (WorldBuilderProtocol, ConsistencyCheckerProtocol, etc.)
- DreamWeaver orchestrator skeleton
- HoloLoom integration points defined
- Configuration system

**What Works**:
- Async lifecycle management (`async with DreamWeaver(...)`)
- World state persistence (JSON serialization)
- Metrics tracking
- Background auto-save

**What's Stubbed**:
- Entity/event extraction (placeholder parsers)
- Consistency checking (no actual rules implemented)
- LLM integration (prompts defined but not connected)
- Narrative memory (uses basic KG, not story-aware)

---

### Phase 1: Core World Building (8 weeks)

**Goal**: Implement basic world generation and evolution without LLM.

**Priority**: Must-Have
**Timeline**: Weeks 1-8
**Dependencies**: Phase 0

#### Week 1-2: Narrative Memory Implementation

**File**: `HoloLoom/dreamweaving/narrative_memory.py` (~500 lines)

**Features**:
- Story-aware knowledge graph (extends Yarn Graph)
- Entity relationship tracking (IS_A, CONTAINS, ALLY_OF, ENEMY_OF, etc.)
- Temporal event ordering (Chrono Trigger integration)
- Causal link detection (Event A → Event B causality)
- Character personality tracking (traits, goals, relationships)
- Location connectivity (geography, travel time, borders)

**Implementation**:
```python
class NarrativeMemory:
    """Story-aware knowledge graph."""

    def __init__(self, kg: KG):
        self.kg = kg
        self.entity_index: Dict[str, WorldEntity] = {}
        self.event_timeline: List[NarrativeEvent] = []
        self.causal_graph: nx.DiGraph = nx.DiGraph()

    async def add_entity(self, entity: WorldEntity):
        """Add entity to memory with relationships."""
        # Add to KG
        for rel in entity.relationships:
            self.kg.add_edge(KGEdge(
                head=entity.entity_id,
                tail=rel["target_id"],
                rel_type=rel["relation_type"],
                weight=rel.get("strength", 1.0)
            ))

        # Index for fast lookup
        self.entity_index[entity.entity_id] = entity

    async def add_event(self, event: NarrativeEvent):
        """Add event to timeline with causal links."""
        # Binary search insert (maintain temporal order)
        self.event_timeline.append(event)
        self.event_timeline.sort(key=lambda e: e.timestamp)

        # Add causal links
        for cause_id in event.causal_links["causes"]:
            self.causal_graph.add_edge(cause_id, event.event_id)

    async def retrieve_context(
        self,
        query: str,
        world_state: WorldState,
        k: int = 10
    ) -> Dict[str, Any]:
        """Retrieve relevant world context for query."""
        # 1. Extract entities from query (NER)
        # 2. Expand via KG traversal (BFS, depth 2)
        # 3. Find recent events involving entities
        # 4. Return context dict
        pass

    async def find_causal_chain(
        self,
        event_id: str,
        max_depth: int = 5
    ) -> List[NarrativeEvent]:
        """Find causal chain leading to event."""
        # DFS on causal graph
        pass
```

**Tests**:
- `test_narrative_memory.py` (15 tests)
- Test entity storage/retrieval
- Test event timeline ordering
- Test causal chain detection
- Test context expansion

---

#### Week 3-4: World Fabric (State Management)

**File**: `HoloLoom/dreamweaving/world_fabric.py` (~400 lines)

**Features**:
- World state versioning (branching timelines)
- Atomic state updates (all-or-nothing event application)
- Rollback/undo support
- Diff generation (what changed between states?)
- Snapshot persistence (fast save/load)

**Implementation**:
```python
class WorldFabric:
    """Persistent world state manager with versioning."""

    def __init__(self, config: DreamWeaverConfig):
        self.config = config
        self.versions: Dict[str, WorldState] = {}
        self.current_version: str = "v0"
        self.version_graph: nx.DiGraph = nx.DiGraph()  # Branching

    async def commit_state(
        self,
        world_state: WorldState,
        message: str = ""
    ) -> str:
        """
        Commit new world state (like git commit).

        Returns:
            Version ID
        """
        version_id = f"v{len(self.versions)}"

        # Store version
        self.versions[version_id] = world_state

        # Add to version graph
        self.version_graph.add_edge(self.current_version, version_id)

        # Update current
        self.current_version = version_id

        return version_id

    async def rollback(self, version_id: str) -> WorldState:
        """Rollback to previous version."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        self.current_version = version_id
        return self.versions[version_id]

    async def branch(self, branch_name: str) -> str:
        """Create alternative timeline branch."""
        # Copy current state
        branch_id = f"branch_{branch_name}_v0"
        self.versions[branch_id] = copy.deepcopy(
            self.versions[self.current_version]
        )

        return branch_id

    async def diff(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """Generate diff between versions."""
        state_a = self.versions[version_a]
        state_b = self.versions[version_b]

        diff = {
            "entities_added": [],
            "entities_removed": [],
            "entities_modified": [],
            "events_added": [],
        }

        # Compare entities
        ids_a = set(state_a.entities.keys())
        ids_b = set(state_b.entities.keys())

        diff["entities_added"] = list(ids_b - ids_a)
        diff["entities_removed"] = list(ids_a - ids_b)

        # Compare events
        events_a = {e.event_id for e in state_a.recent_events}
        events_b = {e.event_id for e in state_b.recent_events}

        diff["events_added"] = list(events_b - events_a)

        return diff
```

**Tests**:
- `test_world_fabric.py` (12 tests)
- Test state versioning
- Test rollback/branch
- Test diff generation
- Test persistence

---

#### Week 5-6: Consistency Engine

**File**: `HoloLoom/dreamweaving/consistency.py` (~600 lines)

**Features**:
- Rule-based consistency checking (no LLM required)
- 4 rule types: Physical, Logical, Narrative, Stylistic
- Automatic violation detection
- Severity scoring (0.0-1.0)
- Auto-fix for simple violations
- Violation explanation generation

**Default Rules**:

1. **Physical Rules** (Severity: 0.9):
   - Temporal causality (no backwards time travel unless explicit)
   - Entity location consistency (can't be two places at once)
   - Object permanence (can't destroy then use)

2. **Logical Rules** (Severity: 0.8):
   - Entity identity (same entity has consistent properties)
   - Relationship symmetry (if A→B, check B→A makes sense)
   - Numerical consistency (populations, distances, etc.)

3. **Narrative Rules** (Severity: 0.6):
   - Character personality (actions align with traits)
   - Dialogue style (character voice consistency)
   - Foreshadowing payoff (setups have resolutions)

4. **Stylistic Rules** (Severity: 0.3):
   - Genre conventions (fantasy vs sci-fi tropes)
   - Tone consistency (dark vs lighthearted)
   - Description density (match author's style)

**Implementation**:
```python
class ConsistencyEngine:
    """Maintains logical/causal coherence in worlds."""

    def __init__(
        self,
        rules: List[ConsistencyRule],
        level: ConsistencyLevel = ConsistencyLevel.BALANCED
    ):
        self.rules = rules
        self.level = level
        self.violation_cache: Dict[str, List[Dict]] = {}

    async def check(
        self,
        world_state: WorldState
    ) -> List[Dict[str, Any]]:
        """
        Check world state against all rules.

        Returns:
            List of violations with metadata
        """
        violations = []

        for rule in self.rules:
            # Skip low-severity rules if level is LOOSE
            if self.level == ConsistencyLevel.LOOSE and rule.severity < 0.5:
                continue

            # Execute check function
            if callable(rule.check_function):
                result = await rule.check_function(world_state)
            else:
                # String function name - lookup in registry
                result = await self._execute_check(
                    rule.check_function,
                    world_state
                )

            if not result["passed"]:
                violations.append({
                    "rule_id": rule.rule_id,
                    "rule_type": rule.rule_type,
                    "severity": rule.severity,
                    "description": rule.description,
                    "details": result.get("details", ""),
                    "can_auto_fix": rule.auto_fix,
                })

        return violations

    async def auto_fix(
        self,
        world_state: WorldState,
        violation: Dict[str, Any]
    ) -> Optional[WorldState]:
        """Attempt automatic fix for violation."""
        if not violation["can_auto_fix"]:
            return None

        rule_id = violation["rule_id"]

        # Dispatch to specific fix function
        if rule_id == "entity_location_consistency":
            return await self._fix_location_violation(world_state, violation)
        elif rule_id == "temporal_causality":
            return await self._fix_temporal_violation(world_state, violation)

        return None

    async def _execute_check(
        self,
        check_name: str,
        world_state: WorldState
    ) -> Dict[str, Any]:
        """Execute named check function."""
        # Registry of check functions
        checks = {
            "check_temporal_ordering": self._check_temporal_ordering,
            "check_entity_location_consistency": self._check_entity_location,
            "check_character_consistency": self._check_character_personality,
        }

        if check_name not in checks:
            return {"passed": True, "details": "Check not implemented"}

        return await checks[check_name](world_state)

    async def _check_temporal_ordering(
        self,
        world_state: WorldState
    ) -> Dict[str, Any]:
        """Verify events are temporally ordered."""
        events = world_state.recent_events

        for i in range(len(events) - 1):
            if events[i].timestamp > events[i+1].timestamp:
                return {
                    "passed": False,
                    "details": f"Event {events[i].event_id} occurs after {events[i+1].event_id} but has earlier timestamp"
                }

        return {"passed": True}

    async def _check_entity_location(
        self,
        world_state: WorldState
    ) -> Dict[str, Any]:
        """Verify entities aren't in multiple locations."""
        # Group events by timestamp
        from collections import defaultdict
        locations_by_time: Dict[float, Dict[str, str]] = defaultdict(dict)

        for event in world_state.recent_events:
            if event.location_id:
                for entity_id in event.participants:
                    if entity_id in locations_by_time[event.timestamp]:
                        prev_loc = locations_by_time[event.timestamp][entity_id]
                        if prev_loc != event.location_id:
                            return {
                                "passed": False,
                                "details": f"Entity {entity_id} in two locations at time {event.timestamp}"
                            }
                    locations_by_time[event.timestamp][entity_id] = event.location_id

        return {"passed": True}

    async def _check_character_personality(
        self,
        world_state: WorldState
    ) -> Dict[str, Any]:
        """Verify character actions align with personality."""
        # This would require personality embeddings + action similarity
        # Placeholder for Phase 2 (requires LLM)
        return {"passed": True, "details": "Requires LLM for personality analysis"}
```

**Tests**:
- `test_consistency_engine.py` (20 tests)
- Test each rule type
- Test auto-fix mechanisms
- Test severity filtering
- Test violation reporting

---

#### Week 7-8: Generative Loom (Rule-Based)

**File**: `HoloLoom/dreamweaving/generative.py` (~500 lines)

**Features**:
- Template-based generation (no LLM required)
- Entity generation from templates
- Event generation from patterns
- Name generation (Markov chains)
- Description composition (template + attributes)
- Relationship inference (heuristic rules)

**Implementation**:
```python
class GenerativeLoom:
    """Procedural content generation for worlds."""

    def __init__(self, config: DreamWeaverConfig):
        self.config = config
        self.templates: Dict[EntityType, List[str]] = {}
        self.name_generator: NameGenerator = NameGenerator()

        self._load_templates()

    async def generate_entity(
        self,
        entity_type: EntityType,
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> WorldEntity:
        """Generate entity from template."""
        # Select template
        template = self._select_template(entity_type, context)

        # Generate attributes
        attributes = self._generate_attributes(entity_type, context, constraints)

        # Generate name
        name = self.name_generator.generate(entity_type, context)

        # Compose description
        description = template.format(**attributes)

        # Infer relationships
        relationships = await self._infer_relationships(
            entity_type,
            attributes,
            context
        )

        return WorldEntity(
            entity_id=f"{entity_type.value}_{uuid.uuid4().hex[:8]}",
            entity_type=entity_type,
            name=name,
            description=description,
            attributes=attributes,
            relationships=relationships,
            history=[],
            metadata={
                "generated": datetime.utcnow().isoformat(),
                "generator": "rule_based",
            }
        )

    def _select_template(
        self,
        entity_type: EntityType,
        context: Dict[str, Any]
    ) -> str:
        """Select appropriate template."""
        templates = self.templates.get(entity_type, [])

        if not templates:
            return "{description}"

        # Simple random selection
        # (In production, use context to pick best match)
        import random
        return random.choice(templates)

    def _generate_attributes(
        self,
        entity_type: EntityType,
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate entity attributes."""
        attributes = {}

        if entity_type == EntityType.CHARACTER:
            attributes = {
                "age": random.randint(18, 80),
                "personality": random.choice(["brave", "cunning", "wise", "reckless"]),
                "occupation": random.choice(["merchant", "warrior", "scholar", "thief"]),
            }
        elif entity_type == EntityType.LOCATION:
            attributes = {
                "terrain": random.choice(["forest", "mountain", "desert", "city"]),
                "population": random.randint(100, 10000),
                "notable_features": [],
            }

        # Apply constraints
        if constraints:
            attributes.update(constraints)

        return attributes

    async def _infer_relationships(
        self,
        entity_type: EntityType,
        attributes: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Infer relationships with existing entities."""
        relationships = []

        # Heuristic rules
        if entity_type == EntityType.CHARACTER:
            # Characters in same location know each other
            if "location" in context:
                relationships.append({
                    "target_id": context["location"],
                    "relation_type": "LOCATED_IN",
                    "strength": 1.0
                })

        return relationships

    def _load_templates(self):
        """Load entity templates."""
        self.templates[EntityType.CHARACTER] = [
            "{name} is a {age}-year-old {occupation} known for being {personality}.",
            "A {personality} {occupation} named {name}, aged {age}.",
        ]

        self.templates[EntityType.LOCATION] = [
            "{name} is a {terrain} with a population of {population}.",
            "The {terrain} of {name}, home to {population} souls.",
        ]


class NameGenerator:
    """Generates names using Markov chains."""

    def __init__(self):
        self.chains: Dict[EntityType, MarkovChain] = {}
        self._initialize_chains()

    def generate(
        self,
        entity_type: EntityType,
        context: Dict[str, Any]
    ) -> str:
        """Generate name for entity type."""
        chain = self.chains.get(entity_type)

        if not chain:
            return f"Generated{entity_type.value.capitalize()}"

        return chain.generate()

    def _initialize_chains(self):
        """Initialize Markov chains with seed data."""
        # Character names
        character_names = [
            "Aric", "Brenna", "Caelan", "Dara", "Eira",
            "Finn", "Gwen", "Haldor", "Isla", "Jarek"
        ]

        self.chains[EntityType.CHARACTER] = MarkovChain(
            samples=character_names,
            order=2
        )

        # Location names
        location_names = [
            "Avalon", "Brightwood", "Cinderfell", "Drakeshore",
            "Eldergrove", "Frostpeak", "Goldenhaven"
        ]

        self.chains[EntityType.LOCATION] = MarkovChain(
            samples=location_names,
            order=2
        )


class MarkovChain:
    """Simple Markov chain for name generation."""

    def __init__(self, samples: List[str], order: int = 2):
        self.order = order
        self.chains: Dict[str, List[str]] = {}

        for sample in samples:
            self._train(sample)

    def _train(self, sample: str):
        """Train on sample string."""
        padded = "#" * self.order + sample + "#"

        for i in range(len(padded) - self.order):
            prefix = padded[i:i+self.order]
            suffix = padded[i+self.order]

            if prefix not in self.chains:
                self.chains[prefix] = []

            self.chains[prefix].append(suffix)

    def generate(self, max_length: int = 12) -> str:
        """Generate new string."""
        import random

        result = []
        prefix = "#" * self.order

        for _ in range(max_length):
            if prefix not in self.chains:
                break

            suffixes = self.chains[prefix]
            suffix = random.choice(suffixes)

            if suffix == "#":
                break

            result.append(suffix)
            prefix = prefix[1:] + suffix

        return "".join(result).capitalize()
```

**Tests**:
- `test_generative_loom.py` (18 tests)
- Test entity generation
- Test name generation (Markov chains)
- Test template selection
- Test relationship inference

---

### Phase 1 Summary

**Total Lines**: ~2,000 lines of production code
**Total Tests**: ~65 tests
**Key Deliverables**:
- ✅ Narrative memory (story-aware KG)
- ✅ World fabric (versioning + persistence)
- ✅ Consistency engine (rule-based checking)
- ✅ Generative loom (template-based generation)

**What Works After Phase 1**:
- Create worlds without LLM (rule-based)
- Add entities and events
- Check consistency automatically
- Version control with rollback
- Persist worlds to disk

**What's Missing**:
- Natural language generation (requires Phase 2 LLM)
- Complex reasoning (requires Phase 2)
- Multi-agent collaboration (requires Phase 3)

---

### Phase 2: LLM Integration (4 weeks)

**Goal**: Add natural language generation and understanding.

**Priority**: High
**Timeline**: Weeks 9-12
**Dependencies**: Phase 1 complete

#### Week 9-10: LLM Abstraction Layer

**File**: `HoloLoom/dreamweaving/llm_bridge.py` (~400 lines)

**Features**:
- Provider abstraction (OpenAI, Anthropic, Local, HuggingFace)
- Prompt templates for world building
- Response parsing (structured output extraction)
- Token budget management
- Caching for repeated prompts
- Graceful fallback to rule-based generation

**Prompts**:
- Entity generation (character, location, faction, etc.)
- Event generation (dialogue, action, discovery, etc.)
- Description enrichment
- Consistency checking (narrative/stylistic rules)
- Relationship inference

#### Week 11-12: Enhanced Generation

**File**: `HoloLoom/dreamweaving/llm_generator.py` (~600 lines)

**Features**:
- Natural language entity generation
- Dynamic dialogue generation
- Context-aware description
- Personality-driven behavior
- Multi-turn narrative generation
- Foreshadowing suggestion

---

### Phase 3: Collaborative Authoring (6 weeks)

**Goal**: Enable human-AI co-creation with authorship tracking.

**Priority**: High
**Timeline**: Weeks 13-18
**Dependencies**: Phase 2 complete

#### Week 13-14: Authorship System

**File**: `HoloLoom/dreamweaving/authorship.py` (~300 lines)

**Features**:
- Track human vs AI contributions
- Edit attribution (who created/modified what)
- Approval workflows (AI suggests, human approves)
- Style learning (match author's voice)
- Suggestion system (AI offers alternatives)

#### Week 15-16: Interactive Editor

**File**: `HoloLoom/dreamweaving/editor.py` (~500 lines)

**Features**:
- Real-time collaboration
- Inline suggestions
- Alternative generation (multiple options)
- "What if?" branching
- Undo/redo with attribution

#### Week 17-18: Web Dashboard

**Location**: `HoloLoom/web_dashboard/dreamweaver.html`

**Features**:
- Visual world graph
- Entity/event timeline
- Consistency violation dashboard
- Authorship analytics
- Export to formats (JSON, Markdown, EPUB)

---

### Phase 4: Advanced Features (8 weeks)

**Goal**: Multi-agent collaboration, advanced consistency, learning.

**Priority**: Medium
**Timeline**: Weeks 19-26
**Dependencies**: Phase 3 complete

#### Week 19-20: Multi-Agent System

**File**: `HoloLoom/dreamweaving/agents.py` (~700 lines)

**Features**:
- Specialized agents (PlotAgent, CharacterAgent, WorldAgent, ConsistencyAgent)
- Agent coordination (debate, vote, consensus)
- Adversarial generation (one agent generates, another critiques)
- Ensemble generation (multiple agents, pick best)

#### Week 21-22: Advanced Consistency

**File**: `HoloLoom/dreamweaving/advanced_consistency.py` (~500 lines)

**Features**:
- Semantic consistency (embeddings for personality/tone)
- Statistical consistency (numerical ranges, distributions)
- Graph consistency (relationship transitivity, cycles)
- Temporal consistency (timeline validation, paradox detection)

#### Week 23-24: Learning System

**File**: `HoloLoom/dreamweaving/learning.py` (~400 lines)

**Features**:
- Player feedback loop (track ratings, reactions)
- Preference learning (what does player enjoy?)
- Thompson Sampling for narrative branches
- A/B testing (try different approaches)
- Reflection buffer integration

#### Week 25-26: Performance Optimization

**Focus**:
- Caching strategies (entity cache, prompt cache, embedding cache)
- Parallel generation (batch entity creation)
- Incremental consistency (only check changed parts)
- Fast rollback (copy-on-write state)

---

### Phase 5: Polish & Documentation (4 weeks)

**Goal**: Production-ready release.

**Priority**: High
**Timeline**: Weeks 27-30
**Dependencies**: Phase 4 complete

#### Week 27: Testing & Validation

- Comprehensive test suite (80%+ coverage)
- Integration tests with HoloLoom
- Performance benchmarks
- Edge case testing

#### Week 28: Documentation

- User guide (getting started, tutorials)
- API reference (complete)
- Architecture deep dive
- Best practices guide

#### Week 29: Examples & Demos

- Example worlds (fantasy, sci-fi, modern)
- Demo notebooks (Jupyter)
- Video tutorials
- Sample applications (interactive fiction, game, writing tool)

#### Week 30: Community Prep

- GitHub repo setup
- Issue templates
- Contributing guide
- License (Apache 2.0)
- Roadmap for future features

---

### Phase 6: Advanced Applications (Ongoing)

**Goal**: Domain-specific applications and extensions.

**Timeline**: Post-release (months 7-18)
**Priority**: Low

#### Potential Applications

1. **Interactive Fiction Engine**
   - Choice-based narratives (Twine-style)
   - Parser-based adventures (Inform-style)
   - Visual novel backend

2. **Game World Builder**
   - RPG world generation (D&D, Pathfinder)
   - NPC personality simulation
   - Quest generation
   - Procedural dungeon design

3. **Writing Assistant**
   - Novel outline generation
   - Character arc tracking
   - Plot hole detection
   - Research assistant (historical accuracy)

4. **Collaborative Storytelling**
   - Multi-player world building
   - Shared universe management
   - Canon enforcement
   - Fan fiction integration

5. **Educational Applications**
   - Historical simulation
   - Science fiction thought experiments
   - Ethical dilemma exploration
   - Creative writing pedagogy

---

## Technical Specifications

### Performance Targets

| Operation | Target Latency | Acceptable Latency |
|-----------|---------------|-------------------|
| Entity retrieval | <5ms | <20ms |
| Event generation (rule-based) | <50ms | <200ms |
| Event generation (LLM) | <1s | <3s |
| Consistency check | <100ms | <500ms |
| World state save | <200ms | <1s |
| World state load | <300ms | <1.5s |

### Scalability Targets

| Metric | Phase 1 (MVP) | Phase 3 (Production) | Phase 6 (Enterprise) |
|--------|--------------|---------------------|---------------------|
| Max entities | 1,000 | 10,000 | 100,000+ |
| Max events | 10,000 | 100,000 | 1,000,000+ |
| Max active threads | 5 | 20 | 100+ |
| World save size | <1MB | <10MB | <100MB |
| Concurrent users | 1 | 10 | 100+ |

### Memory Architecture

```
In-Memory (Hot):
  - Current world state (~10MB)
  - Active entities (<1,000)
  - Recent events (<5,000)
  - Entity cache (LRU, 10k entities)

On-Disk (Cold):
  - Full world history (unlimited)
  - All entities (compressed)
  - Complete event log
  - Version snapshots
```

### API Surface

**Core API** (10 methods):
```python
# Initialization
DreamWeaver(config, hololoom_config)
bootstrap_world(seed_prompt)
load_world(world_id)

# Evolution
step(user_input, world_state)
query_world(query, world_state)

# Generation
generate_entity(entity_type, context)
generate_event(event_type, participants, context)

# Consistency
check_consistency(world_state, rules)
resolve_contradiction(world_state, violation)

# Persistence
save_world(world_state, path)
```

**Extended API** (20+ methods):
- Thread management (create_thread, resolve_thread, etc.)
- Relationship operations (add_relationship, remove_relationship)
- Temporal operations (rewind, fast_forward, branch_timeline)
- Authorship tracking (get_authorship, approve_suggestion)
- Analytics (get_metrics, get_statistics, export_graph)

---

## Integration Points

### With HoloLoom Core

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

### With External Systems

| System | Integration | Use Case |
|--------|------------|----------|
| **LLM APIs** | OpenAI, Anthropic, Local | Natural language generation |
| **Vector DBs** | Qdrant, Pinecone | Semantic entity search |
| **Game Engines** | Unity, Unreal | Real-time world interaction |
| **Writing Tools** | Scrivener, Obsidian | Author collaboration |
| **Web Apps** | FastAPI, React | Dashboard, editor UI |

---

## Safety & Alignment

### Content Filtering

- **Sensitive topics**: Track and warn on violence, adult content, etc.
- **Bias detection**: Monitor for stereotypes, unfair representation
- **Toxicity**: Block hate speech, harassment
- **Copyright**: Avoid reproducing copyrighted characters/worlds

### Authorship Transparency

- **Attribution**: Every entity/event tracks creator (human/AI)
- **Approval workflow**: AI suggests, human approves
- **Edit history**: Complete provenance (git-style)
- **License tracking**: Track source material licenses

### Privacy

- **User data**: No PII in world states
- **Anonymization**: Remove identifying info before sharing
- **Export control**: User controls what data is shared

---

## Success Metrics

### Phase 1 (MVP)

- ✅ Generate 1000-entity world in <5 seconds (rule-based)
- ✅ Consistency checking <100ms for typical world
- ✅ Zero data loss (robust persistence)
- ✅ 80%+ test coverage

### Phase 2 (LLM)

- ✅ Natural language entity generation <3s
- ✅ Coherent dialogue generation
- ✅ 90%+ user approval rate for suggestions
- ✅ Graceful fallback when LLM unavailable

### Phase 3 (Collaborative)

- ✅ Real-time collaboration (2+ users)
- ✅ Complete authorship tracking
- ✅ Web dashboard functional
- ✅ Export to 3+ formats (JSON, Markdown, EPUB)

### Phase 4 (Advanced)

- ✅ Multi-agent generation working
- ✅ Advanced consistency checks (semantic, statistical)
- ✅ Learning system improves over time
- ✅ 10k+ entities with <1s query latency

### Phase 5 (Production)

- ✅ Complete documentation
- ✅ 10+ example worlds
- ✅ Video tutorials
- ✅ Community feedback positive

### Phase 6 (Applications)

- ✅ 1+ domain-specific application live
- ✅ 100+ active users
- ✅ 10+ community contributions
- ✅ Published research paper (optional)

---

## Open Questions

### Technical

1. **Entity ID strategy**: UUID vs sequential vs hash-based?
2. **Event granularity**: How fine-grained should events be?
3. **Consistency strictness**: Default to BALANCED or STRICT?
4. **LLM provider**: OpenAI, Anthropic, or local models?
5. **Vector DB**: Qdrant, Pinecone, or FAISS?

### Design

1. **User interface**: Web, CLI, or both?
2. **Import/Export formats**: JSON, YAML, custom binary?
3. **Versioning strategy**: Git-like, timestamped, or hybrid?
4. **Multi-user**: Real-time or async collaboration?

### Business

1. **Licensing**: Apache 2.0, MIT, or custom?
2. **Monetization**: Open-source + paid hosting? Freemium?
3. **Support model**: Community-driven or commercial support?

---

## Related Work

### Existing World Building Tools

| Tool | Strengths | Weaknesses |
|------|-----------|-----------|
| **World Anvil** | Feature-rich, templates | No AI generation |
| **Campfire** | Good UI, collaboration | Limited AI |
| **LegendKeeper** | Markdown-based | No consistency checking |
| **AI Dungeon** | Strong AI generation | No world persistence |
| **NovelAI** | Good LLM integration | Limited world building |

**DreamWeaver Differentiators**:
- ✅ Full world consistency checking
- ✅ Version control with branching
- ✅ Complete authorship tracking
- ✅ Multi-agent collaboration
- ✅ HoloLoom reasoning integration
- ✅ Open-source + extensible

---

## Call to Action

### For Contributors

**Phase 1** is architected and ready for implementation. We need:

1. **Backend Developers** (Python):
   - Implement narrative memory system
   - Build consistency checking engine
   - Create generative loom templates

2. **Test Engineers**:
   - Write unit/integration tests
   - Create example worlds for testing
   - Performance benchmarking

3. **Documentation Writers**:
   - API documentation
   - User guides
   - Tutorial creation

4. **Domain Experts**:
   - Fantasy/sci-fi world building
   - Interactive fiction design
   - Game design

### For Users

**Join the beta** (Phase 3+):
- Test early versions
- Provide feedback
- Share use cases
- Create example worlds

---

## Conclusion

DreamWeaver extends HoloLoom's weaving metaphor to world building, creating a collaborative storytelling platform that balances human creativity with AI generation. By integrating consistency checking, authorship tracking, and multi-agent collaboration, it enables authors and creators to build rich, coherent fictional worlds at scale.

**The future of storytelling is collaborative - human imagination meets AI intelligence.**

---

**Status**: Phase 0 complete, Phase 1 ready to begin
**Next Steps**: Implement narrative memory (Week 1-2)
**Contact**: See CLAUDE.md for contribution guidelines

---

*Generated with HoloLoom DreamWeaver v0.1.0*
*Last Updated: 2025-11-05*