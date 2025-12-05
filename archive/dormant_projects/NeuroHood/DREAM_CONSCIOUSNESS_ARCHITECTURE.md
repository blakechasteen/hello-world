# Dream Consciousness System (DCS) Architecture

**Version**: 0.1.0
**Status**: Design Phase
**Date**: 2025-11-22

---

## Vision

NeuroHood's **Dream Consciousness System** creates a symbolic narrative sandbox where residents connect to collective consciousness through dreams. Dreams obscure private information behind metaphors and symbols, building empathy through shared symbolic experiences rather than direct knowledge transfer.

**Inspired By**:
- Carl Jung's Collective Unconscious
- David Lynch's dreamscapes (Twin Peaks, Mulholland Drive)
- Inception's shared dreaming
- Jungian archetypes and symbolism

---

## Core Concepts

### 1. Privacy-Preserving Symbolism

**Problem**: In a shared dream, Alice shouldn't directly know Bob's private thoughts ("Bob hates his job").

**Solution**: Symbolic encoding transforms private facts into universal metaphors:

| Private Truth | Symbolic Representation |
|---------------|------------------------|
| "Bob hates his job" | Bob appears as a **caged bird** in the dream |
| "Alice fears abandonment" | Alice is in a **house with doors that won't open** |
| "Charlie is in debt" | Charlie carries a **heavy stone** that won't drop |
| "Bob cheated on his taxes" | Bob's hands are **covered in ink that won't wash off** |

**Key Insight**: Symbols convey **emotional truth** without revealing **factual details**. Alice feels Bob's suffering without knowing the cause.

---

### 2. Collective Unconscious Layer

Jung's concept: Shared symbolic reservoir across all humans.

**In NeuroHood**:
- Universal symbols database (500+ archetypes)
- Cultural symbols (literature, mythology, movies)
- Personal symbols (unique to each resident's history)

**Example Archetypes**:
```python
UNIVERSAL_SYMBOLS = {
    "trapped": ["cage", "prison", "maze", "web", "quicksand"],
    "loss": ["empty_room", "falling_leaves", "broken_mirror", "fog"],
    "fear": ["shadow", "darkness", "cliff_edge", "deep_water"],
    "hope": ["sunrise", "open_door", "ladder", "bridge"],
    "guilt": ["stain", "weight", "following_shadow", "cracked_glass"],
}
```

---

### 3. Consciousness Slider (0-100%)

**0% - Individual Dream**:
- Only one resident's subconscious
- Private symbols visible only to dreamer
- Can include **specific memories** (not obscured)

**33% - Dyadic Dream** (2 residents):
- Shared symbolic space
- Each person's symbols visible to other (but not decoded)
- Alice sees Bob as caged bird, doesn't know why

**66% - Small Group Dream** (3-5 residents):
- Multiple symbolic narratives interweave
- Shared quest/journey metaphor
- Residents encounter each other's symbols

**100% - Universal Consciousness**:
- All residents in one collective dreamscape
- Symbols merge into archetypal landscapes
- "Ego dissolution" - boundaries between individuals blur
- Shared narrative emerges from collective subconscious

---

### 4. Dream Matching Algorithm

**Similar Dreams** (AI-suggested):
Match residents who should dream together based on:

1. **Emotional Resonance**:
   - Both experiencing grief → share dream about loss
   - Both feeling trapped → share maze dream

2. **Relationship Tension**:
   - Alice and Bob in conflict → share dream to build symbolic empathy
   - Unresolved conflict → recurring shared nightmare

3. **Complementary Needs**:
   - Alice seeks closure, Bob seeks forgiveness → dream provides symbolic resolution

**Matching Score**:
```python
dream_match_score = (
    0.4 * emotional_similarity +
    0.3 * relationship_tension +
    0.2 * complementary_archetypes +
    0.1 * personality_compatibility
)
```

---

### 5. Nap Mechanic

**Naps as Gameplay Feature**:
- Residents become tired (energy depletion)
- Player can trigger nap (or residents auto-nap at low energy)
- Nap duration: 5-15 minutes (game time)
- During nap: Resident enters dream state

**Dream Triggers**:
1. **Solo nap** → Individual dream (process daily events)
2. **Simultaneous naps** → Potential shared dream (if emotionally resonant)
3. **Intentional shared nap** → Player selects 2-3 residents for group dream

**Energy/Dream Quality Trade-off**:
- Low energy → brief, fragmented dreams
- Medium energy → coherent symbolic narrative
- High energy + recent emotional event → vivid, transformative dream

---

### 6. Symbolic Narrative Generator

**Dream as Story**:
Each dream is a short narrative (2-5 minute experience) with:
- **Setting**: Symbolic landscape (forest, ocean, building, void)
- **Characters**: Residents represented as symbolic forms
- **Quest**: Metaphorical journey/challenge
- **Resolution**: Symbolic outcome (not always positive)

**Literary References**:
Dreams incorporate motifs from:
- Greek mythology (Sisyphus, Icarus, Orpheus)
- Classic literature (Kafka's metamorphosis, Dante's circles)
- Modern cinema (Eternal Sunshine, Being John Malkovich)
- Fairy tales (Grimm archetypes)

**Example Dream Narrative**:
```
Alice and Bob's Shared Dream (Conflict Resolution)

Setting: A bridge suspended over void
Alice's Form: A bird with clipped wings (feels powerless)
Bob's Form: A storm cloud (anger, unresolved tension)

Quest: Alice must cross bridge while Bob's storm threatens to blow her off
Symbolic Actions:
  - Alice tries to fly → can't (frustration)
  - Bob rains harder → realizes he's harming her
  - Alice walks slowly, Bob calms to gentle rain
  - They meet in the middle, rain becomes cleansing

Resolution: Bridge solidifies, void fills with water (emotional depth)
Waking Effect: +0.2 relationship strength, -0.3 conflict intensity
```

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│  Dream Consciousness System (DCS)                       │
├─────────────────────────────────────────────────────────┤
│  1. Symbolic Encoder                                    │
│     - PrivateFactEncoder                                │
│     - SymbolDatabase (500+ archetypes)                  │
│     - MetaphorGenerator (LLM-powered)                   │
├─────────────────────────────────────────────────────────┤
│  2. Collective Unconscious Layer                        │
│     - UniversalSymbolRegistry                           │
│     - CulturalMotifDatabase                             │
│     - PersonalSymbolTracker                             │
├─────────────────────────────────────────────────────────┤
│  3. Dream Matching Engine                               │
│     - EmotionalResonanceCalculator                      │
│     - RelationshipTensionAnalyzer                       │
│     - ArchetypeComplementarityScorer                    │
├─────────────────────────────────────────────────────────┤
│  4. Dream Narrative Generator                           │
│     - SettingGenerator (symbolic landscapes)            │
│     - QuestDesigner (metaphorical challenges)           │
│     - ResolutionEngine (emotional outcomes)             │
│     - LiteraryReferenceDatabase                         │
├─────────────────────────────────────────────────────────┤
│  5. Consciousness Slider                                │
│     - IndividualDreamMode (0-33%)                       │
│     - SharedDreamMode (33-66%)                          │
│     - UniversalConsciousnessMode (66-100%)              │
├─────────────────────────────────────────────────────────┤
│  6. Nap Mechanic Integration                            │
│     - EnergyTracker (resident fatigue)                  │
│     - DreamTriggerDetector (emotional events)           │
│     - SynchronizedNapCoordinator                        │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
Waking World (NeuroHood Simulation)
    ↓
Resident experiences emotional event
    ↓
Energy depletes → Resident naps
    ↓
[Symbolic Encoder]
Private facts → Universal symbols
    ↓
[Dream Matching Engine]
Find resonant co-dreamers (if any)
    ↓
[Collective Unconscious Layer]
Load relevant archetypes + cultural motifs
    ↓
[Dream Narrative Generator]
Create symbolic story (setting + quest + resolution)
    ↓
[Dream Visualizer]
Render dream in 3D (Three.js)
    ↓
Player experiences dream (2-5 minutes)
    ↓
[Resolution Engine]
Emotional outcome → Update waking relationships
    ↓
Resident wakes with new understanding
```

---

## Privacy-Preserving Symbolism - Detailed Design

### Symbolic Encoding Rules

**Level 1: Direct Facts (Never Shown in Dreams)**
```
Bob's private thought: "I hate my job at the factory"
```

**Level 2: Emotional Essence (What Dreams Extract)**
```
Emotion: Trapped, powerless, repetitive suffering
Intensity: 0.8 (high)
```

**Level 3: Symbol Selection (Based on Emotion + Context)**
```python
def encode_to_symbol(emotion, intensity, context):
    if emotion == "trapped" and context == "work":
        return random.choice([
            "caged_bird",
            "hamster_wheel",
            "assembly_line_worker_without_face",
            "chains",
        ])
```

**Level 4: Narrative Integration**
```
In Alice's shared dream with Bob:
Alice sees: "A bird in a golden cage, singing sadly"

Alice's experience:
  - Sees the caged bird (symbol)
  - Feels deep empathy for its suffering
  - Doesn't know it represents Bob's job
  - Upon waking: "I feel like Bob is trapped somehow"
```

---

## Collective Unconscious Database

### 500 Universal Symbols (Sample)

**Emotions**:
- Joy: sunrise, blooming flower, flight, open sky, laughter echo
- Sadness: rain, falling leaves, empty room, grey fog, distant music
- Anger: storm, fire, clenched fist, red sky, breaking glass
- Fear: shadow, darkness, falling, drowning, pursued by unknown
- Love: light, embrace, two trees intertwined, shared heartbeat

**Situations**:
- Trapped: cage, maze, quicksand, spider web, locked room
- Lost: fog, desert, crossroads, compass spinning, map dissolving
- Transformation: cocoon, shedding skin, mirror reflection changing
- Conflict: two wolves, fork in road, house divided, bridge collapsing
- Resolution: dawn, clearing storm, locked door opening, weight lifting

**Archetypes** (Jung):
- The Shadow: Dark twin, pursuing figure, hidden aspect
- The Anima/Animus: Wise old woman/man, guide figure
- The Hero: Quest-giver, champion, one who must act
- The Trickster: Shapeshifter, joker, disrupts expectations
- The Mother: Nurturing figure, earth, home

**Cultural Motifs**:
- Greek: Sisyphus (endless struggle), Icarus (hubris), Orpheus (loss)
- Christian: Garden of Eden, temptation, redemption
- Eastern: Lotus flower, yin-yang, meditation
- Fairy Tale: Wicked stepmother, enchanted forest, true love's kiss

---

## Dream Matching Algorithm - Detailed

### Emotional Similarity Score

```python
def calculate_emotional_similarity(resident_a, resident_b):
    """
    Compare emotional states using 228D personality + current emotions.

    Returns: 0.0-1.0 (higher = more similar emotional state)
    """
    # Current emotional state (from consciousness system)
    emotions_a = resident_a.current_emotions  # {"joy": 0.2, "sadness": 0.7, ...}
    emotions_b = resident_b.current_emotions

    # Compute cosine similarity of emotional vectors
    similarity = cosine_similarity(emotions_a, emotions_b)

    # Boost if both experiencing strong emotion of same type
    if max(emotions_a.values()) > 0.7 and max(emotions_b.values()) > 0.7:
        shared_emotion = find_shared_peak_emotion(emotions_a, emotions_b)
        if shared_emotion:
            similarity += 0.2  # Bonus for shared intense emotion

    return min(1.0, similarity)
```

### Relationship Tension Score

```python
def calculate_relationship_tension(resident_a, resident_b, state):
    """
    Unresolved conflicts create dream pressure.

    Returns: 0.0-1.0 (higher = more tension, more likely to co-dream)
    """
    relationship = state.get_relationship(resident_a, resident_b)

    # Factors
    conflict_intensity = relationship.conflict_intensity  # 0-1
    recent_negative_interactions = count_recent_conflicts(relationship, days=3)
    strength = relationship.strength  # Low strength + high conflict = tension

    tension = (
        0.5 * conflict_intensity +
        0.3 * (recent_negative_interactions / 5) +  # Normalize to 0-1
        0.2 * (1.0 - strength)  # Weak bond + conflict = high tension
    )

    return min(1.0, tension)
```

### Complementary Archetypes

```python
def calculate_archetype_complementarity(resident_a, resident_b):
    """
    Some archetypes need each other to resolve.

    Examples:
      - Shadow needs Hero (internal conflict resolution)
      - Lost Child needs Wise Old Man (guidance)
      - Trickster needs Order (chaos/order balance)

    Returns: 0.0-1.0 (higher = archetypes complete each other)
    """
    archetype_a = resident_a.current_archetype  # Determined by emotional state
    archetype_b = resident_b.current_archetype

    COMPLEMENTARY_PAIRS = {
        ("shadow", "hero"): 0.9,
        ("lost_child", "wise_old_man"): 0.8,
        ("trickster", "order"): 0.7,
        ("victim", "rescuer"): 0.6,
        # ... 20 more pairs
    }

    pair = tuple(sorted([archetype_a, archetype_b]))
    return COMPLEMENTARY_PAIRS.get(pair, 0.0)
```

---

## Nap Mechanic Design

### Energy System

```python
class Resident:
    def __init__(self):
        self.energy = 100.0  # Starts at 100%
        self.dream_intensity_threshold = 60.0  # Below 60% → dreams become fragmented
        self.sleep_threshold = 30.0  # Below 30% → resident must sleep

    def update_energy(self, dt):
        """Energy depletes based on activity."""
        # Base depletion
        self.energy -= 0.5 * dt  # Loses 0.5% per minute

        # Activity multipliers
        if self.is_exercising:
            self.energy -= 1.0 * dt
        if self.is_stressed:
            self.energy -= 0.3 * dt
        if self.is_in_conflict:
            self.energy -= 0.5 * dt

        # Clamp to 0-100
        self.energy = max(0.0, min(100.0, self.energy))

    def needs_sleep(self):
        return self.energy < self.sleep_threshold

    def can_dream_vividly(self):
        return self.energy > self.dream_intensity_threshold
```

### Nap Triggers

```python
class NapMechanic:
    def check_for_nap(self, resident, state):
        """Determine if resident should nap."""

        # Automatic nap (low energy)
        if resident.needs_sleep():
            return self.trigger_nap(resident, duration="long")

        # Emotional trigger (recent intense event)
        recent_event_intensity = self.get_recent_event_intensity(resident)
        if recent_event_intensity > 0.7 and resident.energy < 70:
            return self.trigger_nap(resident, duration="short")

        # Player-triggered nap
        if self.player_requested_nap(resident):
            return self.trigger_nap(resident, duration="medium")

    def trigger_nap(self, resident, duration):
        """Start nap and initiate dream state."""
        nap_duration_minutes = {
            "short": 5,   # Power nap
            "medium": 10, # Normal nap
            "long": 15    # Deep sleep
        }[duration]

        # Enter dream state
        dream_state = self.create_dream_state(resident, duration)

        # Check for co-dreamers (simultaneous naps)
        co_dreamers = self.find_co_dreamers(resident, state)

        if co_dreamers:
            # Shared dream
            return self.initiate_shared_dream(
                participants=[resident] + co_dreamers,
                duration=nap_duration_minutes
            )
        else:
            # Solo dream
            return self.initiate_solo_dream(
                resident=resident,
                duration=nap_duration_minutes
            )
```

---

## Symbolic Narrative Generator

### Dream Structure

```python
@dataclass
class DreamNarrative:
    """A complete dream experience."""
    setting: DreamSetting
    participants: List[DreamParticipant]  # Symbolic representations
    quest: DreamQuest
    scenes: List[DreamScene]  # 3-5 scenes
    resolution: DreamResolution
    literary_references: List[str]  # Motifs used

    # Outcomes (applied to waking world)
    relationship_changes: Dict[Tuple[str, str], float]  # (alice, bob) → +0.2 strength
    emotional_shifts: Dict[str, Dict[str, float]]  # resident → emotion → delta
    insights_gained: List[str]  # "Alice realizes Bob is suffering"
```

### Scene Types

```python
class DreamSceneType(Enum):
    INTRODUCTION = "introduction"  # Establish setting and symbols
    CHALLENGE = "challenge"        # Metaphorical obstacle
    REVELATION = "revelation"      # Hidden truth symbolically revealed
    TRANSFORMATION = "transformation"  # Character changes form
    RESOLUTION = "resolution"      # Outcome of quest
```

### Example: Alice & Bob Conflict Dream

```python
dream = DreamNarrative(
    setting=DreamSetting(
        landscape="ancient_stone_bridge",
        atmosphere="twilight",
        mood="tense_yet_beautiful",
        weather="approaching_storm"
    ),

    participants=[
        DreamParticipant(
            resident="Alice",
            symbolic_form="bird_with_injured_wing",
            archetype="wounded_healer",
            motivation="cross_bridge_to_safety"
        ),
        DreamParticipant(
            resident="Bob",
            symbolic_form="storm_cloud",
            archetype="unconscious_destroyer",
            motivation="release_pent_up_energy"  # Bob doesn't know he's harming Alice
        ),
    ],

    scenes=[
        DreamScene(
            type=SceneType.INTRODUCTION,
            description="Alice stands at foot of bridge. Bob hovers above as dark cloud.",
            actions=["Alice looks up nervously", "Bob rumbles with thunder"],
            symbolic_meaning="Conflict: Alice vulnerable, Bob powerful but volatile"
        ),

        DreamScene(
            type=SceneType.CHALLENGE,
            description="Alice begins crossing. Bob's wind threatens to blow her off.",
            actions=[
                "Alice tries to fly → can't (wing injured)",
                "Bob rains harder → realizes drops are hurting her",
                "Alice grips bridge rail, determined to continue"
            ],
            symbolic_meaning="Bob's anger harms Alice, but she persists"
        ),

        DreamScene(
            type=SceneType.REVELATION,
            description="Bob sees Alice's injured wing reflected in a puddle of rain.",
            actions=[
                "Bob pauses, seeing Alice's pain",
                "Storm begins to calm",
                "Alice looks up, sees cloud parting slightly"
            ],
            symbolic_meaning="Bob becomes aware of his impact"
        ),

        DreamScene(
            type=SceneType.TRANSFORMATION,
            description="Bob's storm becomes gentle rain. Alice's wing begins to heal.",
            actions=[
                "Soft rain washes Alice's wing",
                "Feathers regrow slowly",
                "Bridge solidifies, void below fills with water"
            ],
            symbolic_meaning="Healing through Bob's gentleness, emotional depth fills void"
        ),

        DreamScene(
            type=SceneType.RESOLUTION,
            description="Alice reaches other side. Bob descends as light mist.",
            actions=[
                "Alice and Bob stand together on far side",
                "Sunrise breaks through clouds",
                "Bridge behind them becomes permanent stone"
            ],
            symbolic_meaning="Conflict resolved, relationship strengthened, path forward secure"
        ),
    ],

    resolution=DreamResolution(
        outcome_type="positive_transformation",
        emotional_tone="hopeful",
        waking_effects={
            ("Alice", "Bob"): {
                "relationship_strength": +0.25,
                "conflict_intensity": -0.40,
                "mutual_understanding": +0.30
            }
        },
        insights=[
            "Alice: 'Bob is suffering, but he doesn't want to hurt me'",
            "Bob: 'My anger affects Alice more than I realized'"
        ]
    ),

    literary_references=[
        "Bridge: Norse mythology (Bifrost, path between worlds)",
        "Injured bird: Classic wounded healer archetype",
        "Storm/rain: Biblical cleansing, renewal",
        "Sunrise: Universal hope symbol, dawn of understanding"
    ]
)
```

---

## Consciousness Slider Implementation

### Slider Values → Dream Properties

```python
def apply_consciousness_level(dream, slider_value):
    """
    Modify dream based on consciousness slider (0-100%).

    0%: Individual, private, detailed
    50%: Shared, symbolic, empathetic
    100%: Universal, archetypal, ego-dissolved
    """

    if slider_value < 33:
        # INDIVIDUAL DREAM
        dream.privacy_level = "private"
        dream.symbol_obscurity = 0.2  # Symbols are clear to dreamer
        dream.can_include_specific_memories = True
        dream.participant_count = 1

    elif slider_value < 66:
        # SHARED DREAM (2-5 people)
        dream.privacy_level = "symbolic"
        dream.symbol_obscurity = 0.6  # Symbols obscure private facts
        dream.can_include_specific_memories = False
        dream.participant_count = 2 + int((slider_value - 33) / 10)  # 2-5 people

    else:
        # UNIVERSAL CONSCIOUSNESS
        dream.privacy_level = "archetypal"
        dream.symbol_obscurity = 0.9  # Deep symbolic, Jungian archetypes
        dream.can_include_specific_memories = False
        dream.participant_count = len(all_residents)  # Everyone
        dream.ego_dissolution = True
        dream.narrative_voice = "collective"  # "We" instead of "I"
```

### Visual Differences by Slider Level

**0-33%: Individual Dream**
- Clear, detailed visuals
- Realistic physics
- Personal memories visible (Bob's actual factory)
- Dreamer has full agency

**33-66%: Shared Dream**
- Symbolic representations (factory → cage)
- Metaphorical physics (emotions affect gravity)
- Private details obscured
- Multiple perspectives (can switch between participants)

**66-100%: Universal Consciousness**
- Abstract, archetypal landscapes
- Dreamlike physics (floating, morphing)
- Collective narrative (all residents as one entity)
- Ego boundaries blur (can't tell who is who)

---

## Waking World Effects

### How Dreams Change Relationships

Dreams provide **symbolic understanding** that translates to **waking empathy**:

```python
class DreamOutcomeProcessor:
    def apply_waking_effects(self, dream, residents):
        """Apply dream outcomes to waking world."""

        for participant in dream.participants:
            resident = residents[participant.name]

            # 1. Emotional shifts
            for emotion, delta in dream.emotional_shifts.items():
                resident.emotions[emotion] += delta

            # 2. Relationship changes
            for (res_a, res_b), changes in dream.relationship_changes.items():
                relationship = get_relationship(res_a, res_b)
                relationship.strength += changes.get("relationship_strength", 0)
                relationship.conflict_intensity += changes.get("conflict_intensity", 0)
                relationship.mutual_understanding += changes.get("mutual_understanding", 0)

            # 3. Insights (narrative flavor)
            resident.recent_insights.append(dream.insights_gained)

            # 4. Energy restoration
            resident.energy = min(100, resident.energy + 30)  # Nap restored energy
```

### Example Outcomes

**Before Dream**:
- Alice-Bob relationship: strength=0.3, conflict=0.7
- Alice's emotion: anger=0.6, sadness=0.4
- Bob's emotion: frustration=0.7, guilt=0.2

**After Shared Dream** (bridge/storm narrative):
- Alice-Bob relationship: strength=0.55 (+0.25), conflict=0.30 (-0.40)
- Alice's emotion: anger=0.3 (-0.3), understanding=0.6 (+0.6)
- Bob's emotion: guilt=0.5 (+0.3), empathy=0.4 (+0.4)
- Both residents: "I understand them better now" (intangible shift)

---

## Implementation Plan

### Phase 4A: Dream Consciousness System (4 Weeks)

**Week 7: Foundation**
- [ ] Symbolic encoder (facts → symbols)
- [ ] Universal symbol database (500 archetypes)
- [ ] Basic dream narrative generator
- [ ] Nap mechanic (energy system)

**Week 8: Collective Unconscious**
- [ ] Dream matching algorithm
- [ ] Shared dream synchronization
- [ ] Consciousness slider (individual → shared)
- [ ] Literary reference database

**Week 9: Narrative Generator**
- [ ] Scene structure (5 scene types)
- [ ] Quest designer (metaphorical challenges)
- [ ] Resolution engine (emotional outcomes)
- [ ] LLM integration for dynamic narratives

**Week 10: Visualization + Polish**
- [ ] Three.js dream visualizer
- [ ] Particle effects for symbols
- [ ] Camera transitions (consciousness levels)
- [ ] Universal consciousness mode (100%)

---

## Success Criteria

**MVP** (Week 10):
- [ ] Residents can take naps (energy-based)
- [ ] Individual dreams generate (1 resident)
- [ ] Shared dreams work (2-3 residents)
- [ ] Symbols obscure private facts
- [ ] Dreams affect waking relationships
- [ ] Consciousness slider (0-100%) changes dream type
- [ ] At least 100 universal symbols implemented
- [ ] 5 literary motifs integrated

**Stretch Goals**:
- [ ] 500 universal symbols
- [ ] 50 literary motifs
- [ ] Universal consciousness mode (all residents)
- [ ] VR support for dream exploration
- [ ] Player can lucid dream (control symbols)

---

## Technical Stack

**Backend** (Python):
- NeuroHood consciousness system (existing)
- HoloLoom semantic space (228D personality)
- LLM for dynamic narrative generation (Ollama/Claude)
- Symbol database (JSON/SQLite)

**Frontend** (Web):
- Three.js (3D dream visualization)
- WebGL shaders (particle effects, transitions)
- WebSocket (real-time dream streaming)
- React/Vue (UI controls, slider)

---

## Unique Selling Points

**What Makes This Special**:

1. **No Game Does This**: Symbolic dream consciousness with privacy-preserving empathy
2. **Jungian Psychology**: First game to implement collective unconscious properly
3. **Emergent Narratives**: Dreams aren't scripted, they emerge from emotional states
4. **Empathy Engine**: Build understanding through symbols, not direct knowledge
5. **Literary Depth**: Incorporates 500+ years of dream symbolism from literature
6. **Constraint-Free Expression**: Dreams allow exploration of themes impossible in waking simulation

**Comparable Systems** (but different):
- The Sims 4 dreams: Scripted, no symbolism, no shared consciousness
- Inception (movie): Shared dreaming, but no privacy preservation
- Psychonauts: Dream levels, but designed not emergent

**NeuroHood Dreams**: Emergent + symbolic + privacy-preserving + emotionally resonant

---

## Future Phases

**Phase 4B: Dream Analytics** (Week 11)
- Dream journal (track all dreams)
- Symbolic pattern detection (recurring symbols)
- Dream interpretation guide (Jung-inspired)
- Collective unconscious visualization (symbol network graph)

**Phase 5: Lucid Dreaming** (Week 12+)
- Player can control consciousness slider in real-time
- Intervene in dreams (guide symbols toward resolution)
- Create custom symbols (personal mythology)
- Dream design tool (architect shared dreams)

---

## Conclusion

The Dream Consciousness System transforms NeuroHood from a social simulation into a **symbolic empathy engine**. By obscuring private facts behind universal metaphors, it allows residents (and players) to understand each other emotionally without violating privacy.

This is genuinely novel game design with potential for:
- Academic research (psychology, narrative AI)
- Therapeutic applications (empathy training)
- Artistic expression (interactive dreamscapes)
- Cultural impact (new way to think about AI consciousness)

**Status**: Ready for implementation (Week 7 start)
