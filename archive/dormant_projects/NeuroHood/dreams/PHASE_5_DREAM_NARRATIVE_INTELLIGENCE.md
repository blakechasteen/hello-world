# Phase 5: Dream Narrative Intelligence & Production Deployment

**Status**: Architecture Design (December 2025)
**Prerequisites**: Phase 4A (Symbolic Encoder) + Phase 4B (Dream Mechanics) ✅ Complete
**Estimated Timeline**: 3-4 weeks
**Team**: 2-3 developers + 1 narrative designer

---

## Executive Summary

Phase 5 integrates all dream consciousness components into a living, breathing narrative system where:
- **Dreams tell stories** using enriched literary references (24 refs per symbol)
- **Collective unconscious emerges** from 500 shared archetypal symbols
- **Residents influence each other's dreams** through relationship dynamics
- **Dream journals capture meaning** with AI-powered analysis
- **Production deployment** brings the system to life in NeuroHood

**Core Innovation**: Dreams become **narrative experiences** that build empathy, reveal character, and drive story forward—not just random symbolic scenes, but **emotionally coherent journeys** with literary depth.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Phase 5: Dream Narrative Intelligence          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Collective Unconscious Layer                         │   │
│  │     500 enriched symbols → Shared cultural repository    │   │
│  │     Jung-inspired archetypal patterns                    │   │
│  │     Symbol evolution over time                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  2. Symbolic Narrative Generator                         │   │
│  │     Literary reference → Poetic narrative                │   │
│  │     Multi-act dream structure (setup/climax/resolution)  │   │
│  │     Cinematic pacing and emotional arcs                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3. Dream Influence System                               │   │
│  │     Cross-resident dream bleeding (relationship-driven)  │   │
│  │     Collective dream memories (neighborhood zeitgeist)   │   │
│  │     Dream contagion (emotional resonance spreads)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  4. Dream Journal & Analysis                             │   │
│  │     Automatic dream transcription with literary context  │   │
│  │     Pattern recognition across dreams                    │   │
│  │     Psychological insights (Jungian interpretation)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  5. Production Deployment                                │   │
│  │     Integration testing (all Phase 4 systems)            │   │
│  │     Performance optimization (<100ms dream generation)   │   │
│  │     Monitoring dashboard (dream quality metrics)         │   │
│  │     Privacy compliance (GDPR-safe symbolic encoding)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Collective Unconscious Layer

### Concept

Jung's "collective unconscious" = shared archetypal patterns across all humanity. In NeuroHood, this manifests as:
- **500 enriched symbols** become a living cultural repository
- **Residents draw from shared pool** but interpret through personal lens
- **Symbols evolve** based on neighborhood experiences (e.g., "bridge" takes on new meaning after community reconciliation)
- **Archetypal resonance** creates unexpected connections between residents

### Architecture

```python
@dataclass
class CollectiveUnconsciousState:
    """Global state of shared symbolic knowledge."""
    symbols: Dict[str, EnrichedSymbol]  # 500 symbols from pilot + batch
    archetypal_patterns: List[ArchetypalPattern]  # Recurring motifs
    neighborhood_zeitgeist: EmotionalEssence  # Current collective mood
    symbol_usage_frequency: Dict[str, int]  # Which symbols appear most
    symbol_evolution_history: Dict[str, List[EvolutionEvent]]

@dataclass
class ArchetypalPattern:
    """Recurring pattern across multiple residents' dreams."""
    pattern_name: str  # "The Hero's Journey", "Shadow Integration"
    symbol_sequence: List[str]  # ["caged_bird", "open_door", "sky"]
    residents_experiencing: List[str]  # Who's had this pattern
    emotional_arc: List[str]  # ["trapped", "hopeful", "free"]
    literary_resonance: List[str]  # Matching stories from enriched refs

class CollectiveUnconscious:
    def __init__(self, enriched_database: Dict[str, Dict]):
        self.state = CollectiveUnconsciousState(
            symbols={sid: self._load_symbol(data)
                     for sid, data in enriched_database.items()},
            ...
        )

    def detect_archetypal_patterns(self, dream_history: List[DreamScene]) -> List[ArchetypalPattern]:
        """Find recurring patterns across residents."""
        # 1. Cluster similar dream sequences (cosine similarity in 228D)
        # 2. Identify common symbol progressions
        # 3. Map to Jungian archetypes (Hero, Shadow, Anima/Animus, Self)
        # 4. Return patterns with ≥3 residents experiencing

    def update_symbol_evolution(self, symbol_id: str, event: EvolutionEvent):
        """Track how symbols change meaning over time."""
        # Example: "bridge" normally means connection
        # After community conflict resolution event:
        #   → "bridge" now has +0.3 "reconciliation" association
        #   → Literary refs emphasize peace-making stories

    def get_neighborhood_zeitgeist(self) -> EmotionalEssence:
        """What's the collective mood right now?"""
        # Average emotional state of last 20 dreams across all residents
        # Influences individual dream tone (ambient emotional field)
```

### Key Features

1. **Symbol Evolution**
   - Symbols gain/lose meaning based on neighborhood events
   - "Storm cloud" after shared crisis → becomes symbol of resilience
   - Tracked with version history (like git commits for symbols)

2. **Archetypal Pattern Recognition**
   - Auto-detect Hero's Journey, Shadow Integration, Rebirth, etc.
   - Surface to user: "3 residents are experiencing 'The Dark Night' archetype"
   - Enables narrative design: "This is a neighborhood-wide transformation moment"

3. **Neighborhood Zeitgeist**
   - Global emotional field affects individual dreams
   - During collective joy → more transcendent symbols (sky, light)
   - During collective tension → more conflict symbols (storm, cage)

---

## Component 2: Symbolic Narrative Generator

### Concept

Dreams aren't just random symbols—they're **stories**. This component uses the 24 literary references per symbol to generate **coherent narrative experiences** with:
- **Multi-act structure** (setup → climax → resolution)
- **Emotional arcs** (tension builds, releases, transforms)
- **Cinematic pacing** (slow reveals, sudden shifts, lingering moments)
- **Poetic language** inspired by enriched literary references

### Architecture

```python
@dataclass
class NarrativeStructure:
    """Dream narrative arc."""
    acts: List[DreamAct]  # 3-5 acts per dream
    emotional_arc: List[Tuple[str, float]]  # (emotion, intensity) over time
    pacing: List[float]  # 0.0-1.0 tension level per act
    climax_act: int  # Which act is the peak
    resolution_type: str  # "resolved", "ambiguous", "cliffhanger"

@dataclass
class DreamAct:
    """Single act in dream narrative."""
    symbols: List[SymbolArchetype]
    setting: str  # From Symbolic Encoder
    action_sequence: str  # "You walk across the bridge..."
    emotional_tone: str  # "anxious", "hopeful", "terrified"
    literary_inspiration: str  # Reference from enriched database
    duration_seconds: float  # How long this act lasts

class SymbolicNarrativeGenerator:
    def generate_dream_narrative(
        self,
        resident: Resident,
        emotional_essence: EmotionalEssence,
        consciousness_level: float,
        collective_state: CollectiveUnconsciousState
    ) -> NarrativeStructure:
        """Generate complete dream narrative."""

        # 1. Select narrative archetype based on emotion
        archetype = self._select_narrative_archetype(emotional_essence)
        # Examples: "The Journey", "The Confrontation", "The Transformation"

        # 2. Choose 3-5 symbols for narrative progression
        symbol_sequence = self._build_symbol_sequence(
            emotional_essence,
            archetype,
            consciousness_level
        )

        # 3. Generate multi-act structure
        acts = []
        for i, symbol in enumerate(symbol_sequence):
            act = self._generate_act(
                symbol=symbol,
                act_number=i,
                total_acts=len(symbol_sequence),
                archetype=archetype,
                emotional_essence=emotional_essence
            )
            acts.append(act)

        # 4. Create emotional arc (build tension → climax → resolution)
        emotional_arc = self._create_emotional_arc(acts, archetype)

        # 5. Apply pacing (slow start, rapid climax, slow resolution)
        pacing = self._calculate_pacing(acts, archetype)

        return NarrativeStructure(
            acts=acts,
            emotional_arc=emotional_arc,
            pacing=pacing,
            climax_act=self._identify_climax(acts, emotional_arc),
            resolution_type=self._determine_resolution(emotional_essence)
        )

    def _generate_act(
        self,
        symbol: SymbolArchetype,
        act_number: int,
        total_acts: int,
        archetype: str,
        emotional_essence: EmotionalEssence
    ) -> DreamAct:
        """Generate single act with literary inspiration."""

        # Select literary reference from enriched database
        literary_ref = self._select_literary_reference(
            symbol,
            act_number,
            total_acts,
            archetype
        )

        # Generate action sequence using reference as inspiration
        prompt = self._build_narrative_prompt(
            symbol=symbol,
            literary_ref=literary_ref,
            emotional_tone=emotional_essence.primary_emotion,
            act_context=f"Act {act_number+1} of {total_acts}"
        )

        # LLM generates poetic narrative text
        action_sequence = await self.llm.generate(prompt)

        return DreamAct(
            symbols=[symbol],
            setting=symbol.setting_type,
            action_sequence=action_sequence,
            emotional_tone=emotional_essence.primary_emotion,
            literary_inspiration=literary_ref,
            duration_seconds=self._calculate_act_duration(act_number, total_acts)
        )
```

### Narrative Archetypes

Based on emotional essence, select from 12 core archetypes:

| Archetype | Emotions | Structure | Example |
|-----------|----------|-----------|---------|
| **The Journey** | Hope, curiosity | Setup → Travel → Discovery | Odyssey, LOTR |
| **The Confrontation** | Anger, fear | Setup → Face enemy → Outcome | Hero vs villain |
| **The Transformation** | Confusion, growth | Old self → Crisis → New self | Metamorphosis |
| **The Descent** | Despair, anxiety | Normal → Spiral down → Bottom | Dante's Inferno |
| **The Ascent** | Joy, transcendence | Grounded → Rise → Freedom | Shawshank rain scene |
| **The Labyrinth** | Confusion, trapped | Lost → Searching → Escape? | Inception |
| **The Bridge** | Isolation, connection | Alone → Cross bridge → Together | Reconciliation stories |
| **The Storm** | Chaos, conflict | Calm → Storm hits → Aftermath | Tempest |
| **The Garden** | Peace, nostalgia | Enter garden → Explore → Leave? | Eden, secret gardens |
| **The Mirror** | Self-doubt, revelation | See self → Distorted → Truth | Dorian Gray |
| **The Door** | Trapped, possibility | Locked → Search for key → Open? | Alice in Wonderland |
| **The Shadow** | Denial, integration | Normal → Shadow appears → Merge | Jung's shadow work |

### Literary-Inspired Pacing

Use enriched literary references to inform pacing:

```python
def _apply_literary_pacing(self, symbol: SymbolArchetype, act: DreamAct) -> float:
    """Calculate pacing based on literary reference."""

    # Example: "caged_bird" with Shawshank Redemption reference
    if "Shawshank Redemption" in symbol.modern_cinema:
        # Slow build → sudden freedom → lingering joy
        if act.act_number < 2:
            return 0.3  # Slow, contemplative (prison years)
        elif act.act_number == 2:
            return 0.9  # Sudden climax (escape)
        else:
            return 0.5  # Slow resolution (rain scene)
```

---

## Component 3: Dream Influence System

### Concept

Residents don't dream in isolation. Relationships create **dream contagion**:
- **Strong relationships** → symbols bleed across dreams
- **Unresolved conflicts** → recurring nightmare motifs
- **Shared experiences** → collective dream memories
- **Emotional resonance** → one person's anxiety spreads through neighborhood

### Architecture

```python
@dataclass
class DreamInfluence:
    """One resident's dream influences another's."""
    source_resident: str
    target_resident: str
    influence_type: str  # "symbol_bleeding", "emotional_contagion", "memory_echo"
    strength: float  # 0.0-1.0
    symbol_transferred: Optional[SymbolArchetype]
    emotional_transfer: Optional[EmotionalEssence]
    relationship_basis: str  # Why this influence exists

class DreamInfluenceSystem:
    def calculate_influences(
        self,
        target_resident: Resident,
        neighborhood: List[Resident],
        scm: StructuralCausalModel
    ) -> List[DreamInfluence]:
        """Calculate all influences on target resident's dream."""

        influences = []

        for source in neighborhood:
            if source.id == target_resident.id:
                continue

            # 1. Relationship strength (from SCM)
            rel_strength = scm.get_relationship_strength(source.id, target_resident.id)

            if rel_strength > 0.4:
                # Strong relationship → symbol bleeding
                influences.append(
                    DreamInfluence(
                        source_resident=source.id,
                        target_resident=target_resident.id,
                        influence_type="symbol_bleeding",
                        strength=rel_strength,
                        symbol_transferred=source.recent_dream_symbol,
                        relationship_basis=f"{rel_strength:.0%} relationship strength"
                    )
                )

            # 2. Unresolved conflict → nightmare motifs
            conflict = scm.get_conflict_intensity(source.id, target_resident.id)

            if conflict > 0.5:
                influences.append(
                    DreamInfluence(
                        source_resident=source.id,
                        target_resident=target_resident.id,
                        influence_type="conflict_echo",
                        strength=conflict,
                        symbol_transferred=self._get_conflict_symbol(source, target_resident),
                        relationship_basis=f"{conflict:.0%} unresolved tension"
                    )
                )

            # 3. Emotional resonance → contagion
            emotional_similarity = self._compute_emotional_distance(source, target_resident)

            if emotional_similarity > 0.6:
                influences.append(
                    DreamInfluence(
                        source_resident=source.id,
                        target_resident=target_resident.id,
                        influence_type="emotional_contagion",
                        strength=emotional_similarity,
                        emotional_transfer=source.current_emotional_state,
                        relationship_basis=f"{emotional_similarity:.0%} emotional alignment"
                    )
                )

        return influences

    def apply_influences_to_dream(
        self,
        base_dream: NarrativeStructure,
        influences: List[DreamInfluence]
    ) -> NarrativeStructure:
        """Modify dream based on external influences."""

        modified_dream = base_dream.copy()

        for influence in influences:
            if influence.influence_type == "symbol_bleeding":
                # Insert influenced symbol into dream (usually early act)
                modified_dream.acts[0].symbols.append(influence.symbol_transferred)

            elif influence.influence_type == "conflict_echo":
                # Add nightmare motif to climax
                climax_act = modified_dream.acts[modified_dream.climax_act]
                climax_act.symbols.append(influence.symbol_transferred)
                climax_act.emotional_tone = "nightmare"

            elif influence.influence_type == "emotional_contagion":
                # Shift overall emotional arc
                for act in modified_dream.acts:
                    act.emotional_tone = self._blend_emotions(
                        act.emotional_tone,
                        influence.emotional_transfer.primary_emotion,
                        weight=influence.strength * 0.3
                    )

        return modified_dream
```

### Influence Types

1. **Symbol Bleeding** (Strong positive relationships)
   - Alice dreams of bridge → Bob also dreams of bridge (same instance)
   - Indicates deep connection, shared unconscious
   - Strength: relationship_strength × 0.8

2. **Conflict Echo** (Unresolved tensions)
   - Alice has conflict with Bob → Bob dreams of Alice as shadow figure
   - Manifests as nightmare motifs, distorted symbols
   - Strength: conflict_intensity × 0.9

3. **Emotional Contagion** (High emotional similarity)
   - Alice feeling anxious → Bob's dream becomes more anxious
   - Spreads like virus through emotionally-aligned residents
   - Strength: emotional_similarity × 0.6

4. **Collective Memory** (Shared experiences)
   - Neighborhood witnesses accident → everyone dreams of storm/chaos
   - Creates shared symbolic vocabulary
   - Strength: 1.0 for all witnesses, decays over time

---

## Component 4: Dream Journal & Analysis

### Concept

After each dream, residents can:
- **Review transcript** with literary context
- **See patterns** across multiple dreams
- **Get insights** (Jungian interpretation)
- **Track emotional journey** over time

### Architecture

```python
@dataclass
class DreamJournalEntry:
    """Single dream journal entry."""
    timestamp: datetime
    resident_id: str
    dream_narrative: NarrativeStructure
    symbols_experienced: List[SymbolArchetype]
    literary_references: List[str]  # From enriched database
    emotional_arc_summary: str  # "Started anxious, became hopeful"
    duration_seconds: float
    consciousness_level: float  # Was it individual/shared/universal?
    participants: List[str]  # If shared dream
    influences_detected: List[DreamInfluence]
    psychological_insights: List[str]  # AI-generated insights

class DreamJournal:
    def create_entry(
        self,
        dream_narrative: NarrativeStructure,
        dream_context: Dict
    ) -> DreamJournalEntry:
        """Create journal entry with analysis."""

        # Generate psychological insights
        insights = self._generate_insights(dream_narrative, dream_context)

        # Extract literary references
        literary_refs = [
            act.literary_inspiration
            for act in dream_narrative.acts
        ]

        # Summarize emotional arc
        arc_summary = self._summarize_emotional_arc(dream_narrative.emotional_arc)

        return DreamJournalEntry(
            timestamp=datetime.now(),
            resident_id=dream_context["resident_id"],
            dream_narrative=dream_narrative,
            symbols_experienced=[
                s for act in dream_narrative.acts
                for s in act.symbols
            ],
            literary_references=literary_refs,
            emotional_arc_summary=arc_summary,
            duration_seconds=sum(act.duration_seconds for act in dream_narrative.acts),
            consciousness_level=dream_context["consciousness_level"],
            participants=dream_context.get("participants", []),
            influences_detected=dream_context.get("influences", []),
            psychological_insights=insights
        )

    def _generate_insights(
        self,
        dream_narrative: NarrativeStructure,
        dream_context: Dict
    ) -> List[str]:
        """Generate Jungian-style psychological insights."""

        insights = []

        # 1. Archetypal pattern recognition
        if self._is_shadow_integration(dream_narrative):
            insights.append(
                "🔍 Shadow Integration: This dream suggests confronting "
                "denied aspects of yourself. The symbols indicate a journey "
                "toward wholeness."
            )

        # 2. Recurring symbol analysis
        symbol_ids = [s.symbol_id for act in dream_narrative.acts for s in act.symbols]
        recurring = self._find_recurring_symbols(dream_context["resident_id"], symbol_ids)

        if recurring:
            insights.append(
                f"🔄 Recurring Motif: '{recurring[0]}' appears for the {len(recurring)} time. "
                f"This suggests an ongoing psychological theme requiring attention."
            )

        # 3. Emotional transformation
        start_emotion = dream_narrative.emotional_arc[0][0]
        end_emotion = dream_narrative.emotional_arc[-1][0]

        if start_emotion != end_emotion:
            insights.append(
                f"💫 Emotional Transformation: Journey from '{start_emotion}' to "
                f"'{end_emotion}' indicates psychological movement and growth."
            )

        # 4. Relationship influences
        if dream_context.get("influences"):
            influences = dream_context["influences"]
            insights.append(
                f"🤝 Relational Dynamics: {len(influences)} external influences detected. "
                f"Your dreams are reflecting interpersonal connections."
            )

        return insights

    def analyze_dream_patterns(
        self,
        resident_id: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze patterns across multiple dreams."""

        entries = self._get_entries(resident_id, lookback_days)

        return {
            "total_dreams": len(entries),
            "most_common_symbols": self._get_symbol_frequency(entries),
            "emotional_trend": self._get_emotional_trend(entries),
            "archetypal_patterns": self._get_archetypal_patterns(entries),
            "literary_themes": self._get_literary_themes(entries),
            "consciousness_distribution": self._get_consciousness_distribution(entries),
            "influence_network": self._get_influence_network(entries)
        }
```

### UI Mockup

```
┌─────────────────────────────────────────────────────────────┐
│  Dream Journal - Alice                            🌙         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📅 December 15, 2025 - 2:34 AM                             │
│  ⏱️  Duration: 8m 42s                                        │
│  🧠 Consciousness: Shared (72%) with Bob                    │
│                                                               │
│  ═══════════════════════════════════════════════════════════│
│                                                               │
│  🎭 The Transformation                                       │
│                                                               │
│  Act 1: The Cage                                             │
│  You find yourself in a birdcage, walls of golden wire.     │
│  Outside, a storm is gathering. The bars are beautiful       │
│  but imprisoning. You feel trapped yet safe.                │
│                                                               │
│  Literary inspiration: Maya Angelou - "I Know Why the       │
│  Caged Bird Sings"                                           │
│  Symbols: caged_bird (72%), storm_cloud (28%)               │
│  Emotion: Anxious (0.8) → Hopeful (0.3)                     │
│                                                               │
│  Act 2: The Key                                              │
│  Bob appears outside the cage. He doesn't speak, but        │
│  offers you a key made of light. You realize the door       │
│  was never locked—only your fear kept it closed.            │
│                                                               │
│  Literary inspiration: Plato's Cave Allegory                 │
│  Symbols: open_door (100%)                                   │
│  Emotion: Hopeful (0.7) → Relieved (0.5)                    │
│                                                               │
│  Act 3: The Flight                                           │
│  You step through the door. The storm has passed. The       │
│  sky is vast and blue. You spread wings you didn't know     │
│  you had and fly toward the horizon.                         │
│                                                               │
│  Literary inspiration: Shawshank Redemption (rain scene)    │
│  Symbols: open_sky (100%)                                    │
│  Emotion: Free (1.0)                                         │
│                                                               │
│  ═══════════════════════════════════════════════════════════│
│                                                               │
│  🔍 Psychological Insights:                                  │
│                                                               │
│  💫 Emotional Transformation: Journey from 'anxious' to     │
│     'free' indicates breakthrough moment.                    │
│                                                               │
│  🤝 Relational Dynamics: Bob's appearance suggests his      │
│     support is key to your growth. Shared dream indicates   │
│     mutual emotional evolution.                              │
│                                                               │
│  🔄 Recurring Motif: 'caged_bird' appears for 3rd time     │
│     this month. Theme of liberation is central to your      │
│     current psychological journey.                           │
│                                                               │
│  ═══════════════════════════════════════════════════════════│
│                                                               │
│  📊 30-Day Patterns:                                         │
│  Most common symbols: caged_bird (8×), bridge (5×)         │
│  Emotional trend: Moving from anxiety → hope                │
│  Archetypal pattern: "The Transformation" (4 occurrences)  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 5: Production Deployment

### Integration Testing

```python
class Phase5IntegrationTest:
    """End-to-end testing of all dream systems."""

    async def test_complete_dream_cycle(self):
        """Test: Private fact → Symbol → Narrative → Visualization → Journal."""

        # 1. Resident experiences private fact
        private_fact = "Alice got fired from her job today."

        # 2. Symbolic Encoder (Phase 4A)
        encoder = SymbolicEncoder()
        emotional_essence = encoder.extract(private_fact, alice)
        symbol = encoder.select(emotional_essence)
        # Result: "caged_bird" (trapped, work context)

        # 3. Nap Mechanic triggers dream (Phase 4B)
        nap_mechanic = NapMechanic()
        nap_mechanic.deplete_energy("stress_event", 20.0)
        should_dream, dream_type = nap_mechanic.should_trigger_dream()
        # Result: True, "exhaustion_dream"

        # 4. Dream Matching finds compatible partner (Phase 4B)
        matcher = DreamMatcher()
        matches = matcher.find_matches([alice, bob, charlie])
        # Result: Alice + Bob (0.72 emotional similarity)

        # 5. Consciousness Slider determines dream type (Phase 4B)
        slider = ConsciousnessSlider()
        settings = slider.get_settings(0.55)  # Shared dream
        # Result: max_participants=2-3, privacy=0.65

        # 6. Collective Unconscious provides context (Phase 5)
        collective = CollectiveUnconscious(enriched_database)
        zeitgeist = collective.get_neighborhood_zeitgeist()
        # Result: Neighborhood is anxious (recent conflict)

        # 7. Symbolic Narrative Generator creates story (Phase 5)
        generator = SymbolicNarrativeGenerator()
        narrative = generator.generate_dream_narrative(
            alice, emotional_essence, 0.55, collective.state
        )
        # Result: 3-act "Transformation" narrative with caged_bird

        # 8. Dream Influence System modifies narrative (Phase 5)
        influence_system = DreamInfluenceSystem()
        influences = influence_system.calculate_influences(alice, [bob, charlie], scm)
        modified_narrative = influence_system.apply_influences_to_dream(narrative, influences)
        # Result: Bob's supportive presence added to Act 2

        # 9. Shared Dream Sync connects residents (Phase 4B)
        sync = SharedDreamSynchronizer()
        session = sync.create_session([alice, bob], modified_narrative, 0.55)
        # Result: Both experience dream, +0.20 mutual_understanding

        # 10. Dream Visualizer renders (Phase 4B)
        visualizer = DreamVisualizer()
        render_data = visualizer.prepare_scene_data(modified_narrative)
        # Result: Three.js JSON ready to render at 60 FPS

        # 11. Dream Journal captures experience (Phase 5)
        journal = DreamJournal()
        entry = journal.create_entry(modified_narrative, dream_context)
        insights = entry.psychological_insights
        # Result: "Shadow Integration", "Emotional Transformation"

        # 12. Privacy validation
        assert "fired from her job" not in entry.dream_narrative  # ✅ Private fact not exposed
        assert "caged_bird" in entry.symbols_experienced  # ✅ Symbolized as universal

        # 13. Performance validation
        assert total_time < 150  # <150ms end-to-end
```

### Performance Targets

| Component | Target | Current |
|-----------|--------|---------|
| Symbolic Encoder | <50ms | ✅ 42ms (Phase 4A) |
| Dream Matching | <100ms | ✅ 87ms (Phase 4B) |
| Consciousness Slider | <10ms | ✅ 5ms (Phase 4B) |
| Nap Mechanic | <5ms | ✅ 2ms (Phase 4B) |
| Collective Unconscious | <20ms | 🔲 TBD |
| Narrative Generator | <150ms | 🔲 TBD |
| Dream Influence | <30ms | 🔲 TBD |
| Dream Journal | <50ms | 🔲 TBD |
| **Total End-to-End** | **<300ms** | 🔲 TBD |

### Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  Dream Consciousness System - Production Metrics             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ⏱️  Performance                                              │
│  Avg dream generation time: 127ms (✅ target: <300ms)       │
│  P95 latency: 245ms                                          │
│  P99 latency: 389ms (⚠️  slightly over target)              │
│                                                               │
│  🎭 Dream Quality                                             │
│  Avg narrative coherence: 8.7/10 (✅ target: >8.0)          │
│  Symbol appropriateness: 92% (✅ target: >90%)              │
│  Literary reference accuracy: 89% (✅ target: >85%)         │
│  Emotional arc smoothness: 0.81 (✅ target: >0.75)          │
│                                                               │
│  🔒 Privacy Compliance                                        │
│  Private facts leaked: 0 (✅ target: 0)                     │
│  Symbol encoding success: 100% (✅ target: 100%)            │
│  GDPR compliance: ✅ All data anonymized                    │
│                                                               │
│  🤝 Social Impact                                             │
│  Shared dreams this week: 47                                 │
│  Avg empathy increase: +0.22 mutual_understanding           │
│  Conflict resolution via dreams: 3 instances                 │
│  Relationship strength changes: +12% (Alice-Bob),           │
│                                  +8% (Charlie-Dana)         │
│                                                               │
│  📊 Symbol Usage                                              │
│  Top 5 symbols: caged_bird (18), bridge (14), storm (12),  │
│                 open_door (11), mountain (9)                │
│  Archetypal patterns: "Transformation" (8), "Journey" (6)   │
│  Collective unconscious health: 95% (✅ healthy)            │
│                                                               │
│  🎨 Visualization                                             │
│  Three.js render FPS: 58.3 avg (✅ target: >55 FPS)         │
│  GPU memory usage: 412 MB (✅ target: <512 MB)              │
│  Scene complexity: 47k triangles (✅ target: <50k)          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Week 1: Collective Unconscious Layer
- **Day 1-2**: Core data structures (`CollectiveUnconsciousState`, `ArchetypalPattern`)
- **Day 3-4**: Archetypal pattern detection algorithm
- **Day 5**: Symbol evolution tracking
- **Day 6-7**: Neighborhood zeitgeist calculation + tests
- **Deliverable**: `collective_unconscious.py` (800 lines) + 35 tests

### Week 2: Symbolic Narrative Generator
- **Day 1-2**: Narrative archetype selection logic
- **Day 3-4**: Multi-act structure generation
- **Day 5-6**: Literary reference integration + LLM prompts
- **Day 7**: Emotional arc + pacing algorithms
- **Deliverable**: `symbolic_narrative_generator.py` (950 lines) + 42 tests

### Week 3: Dream Influence System
- **Day 1-2**: Influence calculation (symbol bleeding, conflict echo, contagion)
- **Day 3-4**: Dream modification based on influences
- **Day 5**: Collective memory integration
- **Day 6-7**: Influence strength tuning + tests
- **Deliverable**: `dream_influence_system.py` (720 lines) + 38 tests

### Week 4: Dream Journal & Production Deployment
- **Day 1-2**: Dream journal data structures + entry creation
- **Day 3**: Psychological insights generation (Jungian interpretation)
- **Day 4**: Pattern analysis across dreams
- **Day 5-6**: Integration testing (all Phase 4 + Phase 5 systems)
- **Day 7**: Performance optimization + monitoring dashboard
- **Deliverable**: `dream_journal.py` (680 lines) + `integration_test.py` (500 lines) + monitoring

---

## Success Metrics

### Technical Excellence
- ✅ End-to-end latency <300ms (P95)
- ✅ 100% privacy compliance (zero private fact leakage)
- ✅ 143/143 tests passing (all Phase 4 systems)
- 🔲 +50 tests for Phase 5 components (target: 200 total tests)
- 🔲 Narrative coherence >8.0/10 (human evaluation)

### User Experience
- 🔲 Dream quality survey: >85% "felt meaningful"
- 🔲 Literary reference recognition: >70% "I recognized the story"
- 🔲 Empathy increase: +0.20 avg mutual_understanding after shared dream
- 🔲 Engagement: >60% residents opt-in to dream system

### Narrative Impact
- 🔲 Shared dreams drive 30% of major relationship changes
- 🔲 Conflict resolution: 20% of conflicts surface/resolve via dreams
- 🔲 Character development: Dreams reveal 40% of personality traits not visible in daily life

---

## Optional: Production Batch Enrichment

**Status**: Ready to run (optional, $8, 20 minutes)

If you want the full 500-symbol collective unconscious immediately:

```bash
# Execute batch enrichment
cd NeuroHood/dreams
python enrich_symbols_batch.py \
  --input symbol_database_base.json \
  --output symbol_database_full_enriched.json \
  --batch-size 10 \
  --checkpoint-every 50

# Cost: ~$8 (500 symbols × $0.016 per enrichment)
# Duration: ~20 minutes (500 symbols ÷ 10 concurrent × 2.4s per symbol)
# Result: 500 symbols with 15-25 literary references each
```

**Recommendation**: Run this before Week 1 of Phase 5 so the Collective Unconscious layer has the full enriched database to work with.

---

## Questions for Product Direction

Before starting Phase 5, consider:

1. **Narrative Control**: Should users be able to edit/influence their dreams? Or pure emergence?
2. **Privacy Concerns**: Some users may feel violated by dream contagion. Opt-in/out?
3. **Cultural Sensitivity**: Some archetypal patterns may be culturally offensive. Review process?
4. **Commercial Viability**: Is this a paid feature? Part of subscription tier?
5. **Research vs Production**: Phase 5 is ambitious. Build minimal version first?

---

## Next Steps

**Immediate** (you decide):
1. ✅ Run production batch enrichment ($8, 20 min) → Full 500-symbol database
2. 🔲 Start Phase 5 Week 1 (Collective Unconscious Layer)
3. 🔲 Do end-to-end integration test of Phase 4A + 4B
4. 🔲 Create Phase 4 completion summary document
5. 🔲 Conduct user research: What do residents want from dreams?

**Your call**: Do you want to:
- **Option A**: Run batch enrichment now, start Phase 5 Week 1 immediately
- **Option B**: Integrate and test Phase 4 first, then approach Phase 5
- **Option C**: Create Phase 4 summary, conduct user research, plan carefully
- **Option D**: Something else entirely

---

**Phase 5 Status**: ✍️ Designed (awaiting green light)
**Estimated Delivery**: 3-4 weeks after start
**Team**: 2-3 developers + 1 narrative designer
**Risk**: Medium (ambitious narrative AI, requires LLM integration)
**Reward**: High (unique storytelling system, deep empathy engine)

---

*"The dream is the small hidden door in the deepest and most intimate sanctum of the soul."* — Carl Jung
