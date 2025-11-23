# Symbolic Encoder: Architecture Strategy

**Version**: 1.0.0
**Date**: 2025-11-22
**Status**: Design Complete - Ready for Implementation

---

## Vision

The **Symbolic Encoder** is the privacy-preserving heart of dream consciousness. It transforms private facts into universal symbols while preserving emotional truth and enabling empathy without knowledge transfer.

**Core Principle**: *"Obscure the details, preserve the essence, amplify the resonance."*

---

## Design Philosophy

### The Three Constraints

1. **Privacy**: Never reveal private facts in shared dreams
2. **Truth**: Symbols must be emotionally authentic (no random mapping)
3. **Beauty**: Symbols should be poetic, universal, timeless

### The Moonshot Goals

1. **Semantic Distance Preservation**: Similar emotions → similar symbols
2. **Temporal Consistency**: Same issue → same recurring symbol
3. **Relationship-Aware**: Symbols adapt to who's dreaming together
4. **Cultural Universality**: Symbols transcend cultural boundaries
5. **Infinite Extensibility**: New symbol domains plug in seamlessly

---

## Four-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: EXTRACTION                                        │
│  Private Fact → Emotional Essence                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Input: "Bob secretly hates his factory job"         │   │
│  │ Output: {                                            │   │
│  │   emotion: "trapped",                                │   │
│  │   intensity: 0.82,                                   │   │
│  │   context: "work",                                   │   │
│  │   temporal: "chronic",                               │   │
│  │   valence: "negative"                                │   │
│  │ }                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: SYMBOL SELECTION                                  │
│  Emotional Essence → Candidate Symbols                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Input: {emotion: "trapped", context: "work", ...}   │   │
│  │ Output: [                                            │   │
│  │   ("caged_bird", 0.93),        # Best match         │   │
│  │   ("hamster_wheel", 0.87),                           │   │
│  │   ("chains", 0.79),                                  │   │
│  │   ("quicksand", 0.65)                                │   │
│  │ ]                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: DISAMBIGUATION                                    │
│  Candidate Symbols → Selected Symbol                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Inputs:                                              │   │
│  │   - Candidates: [caged_bird, hamster_wheel, ...]    │   │
│  │   - Dream context: (Alice's dream, work theme)      │   │
│  │   - Prior symbols: Bob was "caged_bird" yesterday   │   │
│  │   - Relationship: Alice-Bob (0.4 strength)          │   │
│  │                                                       │   │
│  │ Decision: "caged_bird" (temporal consistency)        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: NARRATIVE INTEGRATION                             │
│  Selected Symbol → Dream Scene                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Input: "caged_bird" + dream_setting + participants  │   │
│  │ Output: DreamScene(                                  │   │
│  │   description="A golden cage hangs from a tree.     │   │
│  │                A songbird sits inside, silent.",     │   │
│  │   symbolic_meaning="Bob's entrapment in his job",   │   │
│  │   visual_elements=["ornate_cage", "sad_bird"],      │   │
│  │   metaphorical_physics="Cage sways but won't open"  │   │
│  │ )                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Emotional Extraction

### The Challenge

Transform rich private facts into **emotion signatures** that preserve truth but discard details.

**Example Transformations**:

| Private Fact | Emotional Signature |
|--------------|---------------------|
| "Bob hates his factory job, feels trapped by bills" | `{emotion: "trapped", intensity: 0.8, context: "obligation", temporal: "chronic"}` |
| "Alice fears her husband will leave her" | `{emotion: "abandonment_fear", intensity: 0.9, context: "relationship", temporal: "acute"}` |
| "Charlie cheated on taxes, fears getting caught" | `{emotion: "guilt", intensity: 0.7, context: "transgression", temporal: "chronic"}` |

### Extraction Algorithm

```python
class EmotionalExtractor:
    """Extract emotional essence from private facts."""

    def extract(self, private_fact: str, resident: Resident) -> EmotionalEssence:
        """
        Transform private fact → emotional signature.

        Uses:
        1. NLP emotion detection (from text)
        2. Resident's current emotional state (from consciousness system)
        3. LLM analysis (for nuanced emotions)
        """

        # 1. NLP emotion detection
        base_emotions = self.nlp_emotion_analyzer.analyze(private_fact)
        # Output: {"anger": 0.3, "sadness": 0.6, "fear": 0.5}

        # 2. Resident's live emotional state
        current_state = resident.consciousness.current_emotions
        # Output: {"trapped": 0.8, "powerless": 0.7}

        # 3. LLM nuanced analysis
        llm_analysis = self.llm_analyze_emotion(
            fact=private_fact,
            context=resident.recent_history
        )
        # Output: {
        #   "primary_emotion": "trapped",
        #   "intensity": 0.82,
        #   "context_type": "work",
        #   "temporal_pattern": "chronic",
        #   "complexity": "mixed_with_resignation"
        # }

        # 4. Blend all sources (weighted)
        essence = self.blend_emotion_sources(
            nlp=base_emotions,
            live=current_state,
            llm=llm_analysis,
            weights=(0.2, 0.3, 0.5)  # Trust LLM most for nuance
        )

        return EmotionalEssence(
            primary_emotion=essence["primary_emotion"],
            intensity=essence["intensity"],
            context=essence["context_type"],
            temporal=essence["temporal_pattern"],
            valence=self.compute_valence(essence),
            secondary_emotions=self.extract_secondary(essence),
            complexity_score=self.compute_complexity(essence)
        )
```

### Key Innovation: Semantic Embedding Alignment

**Problem**: How do we ensure "trapped at work" and "trapped in relationship" feel similar but distinct?

**Solution**: Embed emotional essences in **228D semantic space** (same as personality):

```python
def embed_emotional_essence(essence: EmotionalEssence) -> np.ndarray:
    """
    Project emotion into HoloLoom's 228D semantic space.

    This allows:
    - Distance metrics (how similar are two emotions?)
    - Clustering (what emotions co-occur?)
    - Interpolation (blend emotions smoothly)
    """
    # Use HoloLoom's MatryoshkaSemanticCalculus
    from HoloLoom.semantic_calculus import SemanticSpectrum

    # Create semantic description
    semantic_text = f"{essence.primary_emotion} in {essence.context} context, " \
                    f"intensity {essence.intensity}, {essence.temporal} pattern"

    # Project to 228D
    embedding = SemanticSpectrum.project(semantic_text)

    return embedding  # 228D vector
```

**Why This Matters**:
- Symbol selection can use **cosine similarity** in semantic space
- Emotions cluster naturally (trapped/suffocated/confined are nearby)
- Enables **symbol interpolation** (blend symbols for complex emotions)

---

## Layer 2: Symbol Selection

### Symbol Database Structure

```python
@dataclass
class SymbolArchetype:
    """A universal symbol with semantic properties."""

    # Identity
    symbol_id: str  # "caged_bird"
    category: str   # "trapped", "loss", "transformation", etc.

    # Semantic
    embedding: np.ndarray  # 228D semantic vector
    emotion_tags: List[str]  # ["trapped", "powerless", "yearning"]
    context_tags: List[str]  # ["work", "relationship", "self"]

    # Cultural
    universality_score: float  # 0-1 (how universal across cultures?)
    cultural_variants: Dict[str, str]  # {"western": "caged_bird", "eastern": "lotus_in_ice"}

    # Literary
    literary_references: List[str]  # ["Kafka Metamorphosis", "Maya Angelou Caged Bird"]
    archetypal_roots: List[str]  # ["Jungian Shadow", "Christian Captivity"]

    # Visual
    visual_complexity: str  # "simple", "moderate", "complex"
    color_palette: List[str]  # ["grey", "gold", "shadow"]

    # Narrative
    typical_transformations: List[str]  # ["cage_opens", "bird_escapes", "wings_heal"]
    complementary_symbols: List[str]  # ["open_sky", "broken_chains", "sunrise"]

    # Extensibility
    custom_properties: Dict[str, Any]  # Plugin-defined properties
```

### Symbol Database (500 Archetypes)

**Organization** (10 categories × 50 symbols):

1. **Trapped** (50 symbols)
   - Physical: cage, prison, maze, quicksand, web, chains
   - Psychological: hamster_wheel, mirror_maze, fog, drowning
   - Social: invisible_walls, crowd_pressure, spotlight_prison

2. **Loss** (50 symbols)
   - Absence: empty_room, fading_photo, extinguished_flame
   - Separation: broken_bridge, receding_shore, disappearing_path
   - Grief: falling_leaves, winter_tree, grey_rain

3. **Fear** (50 symbols)
   - Unknown: shadow, darkness, fog, void, shapeshifter
   - Falling: cliff_edge, crumbling_ground, vertigo
   - Pursuit: unseen_follower, closing_walls, ticking_clock

4. **Transformation** (50 symbols)
   - Growth: cocoon, shedding_skin, sprouting_seed
   - Decay: rust, erosion, withering_flower
   - Rebirth: phoenix, sunrise, breaking_ice

5. **Connection** (50 symbols)
   - Bond: intertwined_trees, shared_heartbeat, bridge
   - Distance: parallel_roads, opposite_shores, glass_wall
   - Reunion: crossing_paths, merging_rivers, eclipse

6. **Power** (50 symbols)
   - Authority: throne, crown, mountain_peak
   - Powerlessness: puppet_strings, sinking, small_figure
   - Empowerment: breaking_chains, rising_sun, blooming

7. **Guilt** (50 symbols)
   - Stain: unwashable_mark, spreading_ink, bloodstain
   - Weight: boulder, anchor, heavy_chains
   - Judgment: mirror_reflection, pointing_finger, scale

8. **Hope** (50 symbols)
   - Light: sunrise, candle_in_darkness, star
   - Opening: door_ajar, parting_clouds, dawn
   - Growth: green_shoot, rainbow, horizon

9. **Conflict** (50 symbols)
   - Internal: two_wolves, split_path, cracked_mirror
   - External: clashing_waves, storm_meeting, wall_between
   - Resolution: merging_colors, balanced_scale, handshake

10. **Mystery** (50 symbols)
    - Hidden: veil, fog, locked_box, encrypted_text
    - Revelation: unveiling, sunrise, opening_eye
    - Unknown: blank_book, unopened_door, distant_light

### Symbol Selection Algorithm

```python
class SymbolSelector:
    """Select best symbol for emotional essence."""

    def __init__(self, symbol_database: List[SymbolArchetype]):
        self.symbols = symbol_database
        # Pre-compute embeddings for fast similarity search
        self.symbol_embeddings = np.array([s.embedding for s in self.symbols])

    def select_candidates(
        self,
        essence: EmotionalEssence,
        context: DreamContext,
        k: int = 10
    ) -> List[Tuple[SymbolArchetype, float]]:
        """
        Select top-k candidate symbols for emotional essence.

        Uses multi-criteria scoring:
        1. Semantic similarity (cosine distance in 228D space)
        2. Emotional tag match (primary + secondary emotions)
        3. Context appropriateness (work, relationship, self)
        4. Cultural universality (prefer universal symbols)
        5. Temporal consistency (prefer previously used symbols)
        """

        # 1. Semantic similarity (40% weight)
        essence_embedding = embed_emotional_essence(essence)
        semantic_scores = cosine_similarity(
            essence_embedding,
            self.symbol_embeddings
        )

        # 2. Emotional tag match (30% weight)
        emotion_scores = np.array([
            self._emotion_tag_match(essence, symbol)
            for symbol in self.symbols
        ])

        # 3. Context appropriateness (15% weight)
        context_scores = np.array([
            self._context_match(essence.context, symbol.context_tags)
            for symbol in self.symbols
        ])

        # 4. Cultural universality (10% weight)
        universal_scores = np.array([
            symbol.universality_score
            for symbol in self.symbols
        ])

        # 5. Temporal consistency (5% weight)
        temporal_scores = np.array([
            self._temporal_consistency_bonus(symbol, context)
            for symbol in self.symbols
        ])

        # Weighted blend
        final_scores = (
            0.40 * semantic_scores +
            0.30 * emotion_scores +
            0.15 * context_scores +
            0.10 * universal_scores +
            0.05 * temporal_scores
        )

        # Get top-k
        top_k_indices = np.argsort(final_scores)[-k:][::-1]
        candidates = [
            (self.symbols[i], final_scores[i])
            for i in top_k_indices
        ]

        return candidates

    def _emotion_tag_match(self, essence: EmotionalEssence, symbol: SymbolArchetype) -> float:
        """How well do emotion tags match?"""
        primary_match = 1.0 if essence.primary_emotion in symbol.emotion_tags else 0.0

        secondary_matches = sum(
            1.0 for e in essence.secondary_emotions
            if e in symbol.emotion_tags
        ) / max(1, len(essence.secondary_emotions))

        return 0.7 * primary_match + 0.3 * secondary_matches

    def _context_match(self, essence_context: str, symbol_contexts: List[str]) -> float:
        """How appropriate is symbol for this context?"""
        if essence_context in symbol_contexts:
            return 1.0
        # Partial match (e.g., "work" matches "obligation")
        return jaccard_similarity(essence_context, symbol_contexts)

    def _temporal_consistency_bonus(self, symbol: SymbolArchetype, context: DreamContext) -> float:
        """Bonus for previously used symbols (narrative coherence)."""
        dreamer = context.primary_dreamer

        # Has this symbol appeared in dreamer's recent dreams?
        recent_symbols = dreamer.dream_history.get_recent_symbols(days=7)

        if symbol.symbol_id in recent_symbols:
            # Recurring symbol = stronger narrative thread
            frequency = recent_symbols.count(symbol.symbol_id)
            return min(1.0, frequency * 0.3)  # Cap at 1.0

        return 0.0
```

### Key Innovation: Symbol Interpolation

**Problem**: What if emotion is **between** two archetypes? (e.g., 60% trapped, 40% exhausted)

**Solution**: **Interpolate symbols** in semantic space:

```python
def interpolate_symbols(
    symbol_a: SymbolArchetype,
    symbol_b: SymbolArchetype,
    alpha: float  # 0.0 = all A, 1.0 = all B
) -> HybridSymbol:
    """
    Create hybrid symbol by blending two archetypes.

    Example:
      "caged_bird" (trapped) + "withering_flower" (exhausted)
      → "caged_bird_with_drooping_feathers"
    """
    # Interpolate embeddings
    hybrid_embedding = (1 - alpha) * symbol_a.embedding + alpha * symbol_b.embedding

    # Blend visual elements
    hybrid_visuals = blend_visual_elements(symbol_a, symbol_b, alpha)

    # Merge emotion tags
    hybrid_emotions = list(set(symbol_a.emotion_tags + symbol_b.emotion_tags))

    # Generate hybrid description (LLM)
    hybrid_description = generate_hybrid_description(symbol_a, symbol_b, alpha)

    return HybridSymbol(
        base_symbols=(symbol_a, symbol_b),
        interpolation_alpha=alpha,
        embedding=hybrid_embedding,
        description=hybrid_description,
        visual_elements=hybrid_visuals,
        emotion_tags=hybrid_emotions
    )
```

**Example Hybrids**:
- `caged_bird` (0.6) + `drowning` (0.4) → "Bird in cage slowly filling with water"
- `empty_room` (0.7) + `fog` (0.3) → "Empty room with creeping fog"
- `cracked_mirror` (0.5) + `stain` (0.5) → "Mirror with spreading dark stain in cracks"

**Why This Works**:
- Nuanced emotions deserve nuanced symbols
- No need to pre-define every hybrid (infinite combinations)
- LLM generates poetic descriptions on-the-fly

---

## Layer 3: Disambiguation

### The Challenge

Given 10 candidate symbols, which one should appear in **this specific dream**?

**Disambiguation Factors**:

1. **Dream Context** (who's dreaming, what's the setting?)
2. **Temporal Consistency** (has this symbol appeared before?)
3. **Relationship Dynamics** (Alice-Bob dream needs bridge symbols)
4. **Narrative Coherence** (symbols should tell a story together)
5. **Aesthetic Balance** (don't overwhelm with too many complex symbols)

### Disambiguation Algorithm

```python
class SymbolDisambiguator:
    """Select final symbol from candidates."""

    def disambiguate(
        self,
        candidates: List[Tuple[SymbolArchetype, float]],
        dream_context: DreamContext,
        existing_symbols: List[SymbolArchetype]
    ) -> SymbolArchetype:
        """
        Select best symbol for this specific dream.

        Args:
            candidates: Top-k symbols from selection phase
            dream_context: Who's dreaming, setting, participants
            existing_symbols: Symbols already chosen for this dream

        Returns:
            Single best symbol
        """

        scores = {}

        for symbol, base_score in candidates:
            # Start with selection score
            total_score = base_score

            # 1. Temporal consistency (+20% if recurring)
            if self._is_recurring_symbol(symbol, dream_context):
                total_score *= 1.2

            # 2. Relationship appropriateness (+15% if matches relationship)
            if self._matches_relationship_archetype(symbol, dream_context):
                total_score *= 1.15

            # 3. Narrative coherence (+10% if complements existing symbols)
            coherence_score = self._narrative_coherence(symbol, existing_symbols)
            total_score *= (1.0 + 0.1 * coherence_score)

            # 4. Aesthetic balance (-20% if too complex given other symbols)
            if self._creates_visual_clutter(symbol, existing_symbols):
                total_score *= 0.8

            # 5. Novelty bonus (+5% if introduces new archetype category)
            if self._introduces_new_category(symbol, existing_symbols):
                total_score *= 1.05

            scores[symbol] = total_score

        # Select best
        return max(scores.items(), key=lambda x: x[1])[0]

    def _is_recurring_symbol(self, symbol: SymbolArchetype, context: DreamContext) -> bool:
        """Has this appeared in dreamer's recent dreams?"""
        recent = context.primary_dreamer.dream_history.get_recent_symbols(days=7)
        return symbol.symbol_id in recent

    def _matches_relationship_archetype(self, symbol: SymbolArchetype, context: DreamContext) -> bool:
        """Does symbol fit the relationship between participants?"""
        if not context.is_shared_dream:
            return True  # N/A for solo dreams

        relationship = context.get_relationship()

        # High conflict → prefer "wall", "storm", "cliff" symbols
        if relationship.conflict_intensity > 0.7:
            return "conflict" in symbol.category

        # Low strength → prefer "distance", "separation" symbols
        if relationship.strength < 0.3:
            return "connection" in symbol.category and "distance" in symbol.emotion_tags

        # Improving relationship → prefer "bridge", "dawn" symbols
        if relationship.is_improving:
            return "hope" in symbol.category

        return True  # Default: no penalty

    def _narrative_coherence(self, symbol: SymbolArchetype, existing: List[SymbolArchetype]) -> float:
        """
        How well does symbol fit with existing symbols in dream?

        Returns: 0.0-1.0 (higher = better fit)
        """
        if not existing:
            return 1.0  # First symbol, always fits

        # Check complementary symbols
        complementarity = sum(
            1.0 for e in existing
            if e.symbol_id in symbol.complementary_symbols
        ) / len(existing)

        # Check thematic consistency (same category family)
        same_family = sum(
            1.0 for e in existing
            if self._same_category_family(symbol.category, e.category)
        ) / len(existing)

        # Blend
        return 0.6 * complementarity + 0.4 * same_family

    def _creates_visual_clutter(self, symbol: SymbolArchetype, existing: List[SymbolArchetype]) -> bool:
        """Too many complex symbols = visual noise."""
        complex_count = sum(1 for e in existing if e.visual_complexity == "complex")

        if symbol.visual_complexity == "complex" and complex_count >= 2:
            return True  # Already 2+ complex symbols, adding more is clutter

        return False
```

---

## Layer 4: Narrative Integration

### The Challenge

Transform selected symbol into **living dream scene** that:
- Has visual beauty
- Tells a story
- Feels emotionally true
- Integrates with other symbols

### Scene Generation

```python
class NarrativeIntegrator:
    """Transform symbols into dream scenes."""

    def integrate_symbol(
        self,
        symbol: SymbolArchetype,
        dream_setting: DreamSetting,
        participants: List[DreamParticipant],
        llm_client: LLMClient
    ) -> DreamScene:
        """
        Create dream scene from symbol.

        Uses LLM to generate:
        1. Visual description (poetic, evocative)
        2. Symbolic actions (what happens with this symbol?)
        3. Metaphorical physics (dream logic)
        4. Emotional tone (how does it feel?)
        """

        # Build prompt for LLM
        prompt = self._build_scene_prompt(
            symbol=symbol,
            setting=dream_setting,
            participants=participants
        )

        # Generate scene description
        scene_description = llm_client.generate(prompt)

        # Parse LLM output into structured scene
        scene = self._parse_scene_description(scene_description)

        # Add metadata
        scene.symbol_id = symbol.symbol_id
        scene.symbolic_meaning = symbol.emotion_tags
        scene.literary_references = symbol.literary_references

        return scene

    def _build_scene_prompt(
        self,
        symbol: SymbolArchetype,
        setting: DreamSetting,
        participants: List[DreamParticipant]
    ) -> str:
        """Build LLM prompt for scene generation."""
        return f"""
You are a dream architect creating a symbolic dream scene.

SYMBOL: {symbol.symbol_id}
  Emotions: {", ".join(symbol.emotion_tags)}
  Literary refs: {", ".join(symbol.literary_references[:3])}

SETTING: {setting.landscape} at {setting.time_of_day}
  Atmosphere: {setting.atmosphere}

PARTICIPANTS:
{self._format_participants(participants)}

Generate a dream scene that:
1. Integrates the symbol naturally into the setting
2. Creates visual poetry (evocative, memorable imagery)
3. Implies symbolic meaning without being heavy-handed
4. Includes metaphorical physics (dream logic, not realistic)
5. Sets up potential symbolic actions

Format:
DESCRIPTION: [2-3 sentences of vivid visual description]
SYMBOLIC_ACTIONS: [3-5 potential actions participants could take]
METAPHORICAL_PHYSICS: [How does the symbol behave in dream logic?]
EMOTIONAL_TONE: [What does it feel like to be in this scene?]

Example for "caged_bird" symbol:
DESCRIPTION: A golden cage hangs from an ancient oak tree, swaying gently in a wind that touches nothing else. Inside, a songbird with iridescent feathers sits perfectly still, its eyes reflecting distant stars. The cage's door is ornate but has no visible lock.

SYMBOLIC_ACTIONS:
- Approach the cage and realize it's larger than it first appeared
- Hear the bird's silent song (felt, not heard)
- Try to open the cage, find the door opens easily but the bird doesn't fly
- Look into the bird's eyes and see your own reflection

METAPHORICAL_PHYSICS: The cage exists in superposition—simultaneously enormous and small, present and distant. When observed directly, it becomes solid; when seen peripherally, it phases in and out of existence. The bird's weight bends the tree branch impossibly far, suggesting the cage contains something heavier than it appears.

EMOTIONAL_TONE: Melancholic beauty. There's a profound sadness mixed with strange comfort—the cage is prison and sanctuary simultaneously. The scene feels both deeply personal and universally familiar, like a memory from a life you never lived.

Now generate a scene for the symbol "{symbol.symbol_id}" in the {setting.landscape} setting:
"""

    def _parse_scene_description(self, llm_output: str) -> DreamScene:
        """Parse LLM output into structured DreamScene object."""
        # Extract sections (DESCRIPTION, SYMBOLIC_ACTIONS, etc.)
        sections = self._extract_sections(llm_output)

        return DreamScene(
            description=sections["DESCRIPTION"],
            symbolic_actions=self._parse_actions(sections["SYMBOLIC_ACTIONS"]),
            metaphorical_physics=sections["METAPHORICAL_PHYSICS"],
            emotional_tone=sections["EMOTIONAL_TONE"],
            visual_elements=self._extract_visual_elements(sections["DESCRIPTION"])
        )
```

### Key Innovation: Metaphorical Physics

**Normal Physics**: Ball falls down, fire is hot, water is wet

**Dream Physics**: Emotions affect gravity, time flows backwards, impossible geometry

**Examples**:

| Symbol | Metaphorical Physics |
|--------|---------------------|
| **Caged Bird** | Cage is larger inside than outside; door has no lock but won't open when bird tries |
| **Quicksand** | Sinks faster when you panic, solidifies when you accept |
| **Mirror** | Reflection acts independently, shows not appearance but emotional truth |
| **Bridge** | Solidifies as you trust it, dissolves if you doubt |
| **Fog** | Clears when you speak truth, thickens with secrets |
| **Weight** | Gets heavier with guilt, lighter with forgiveness |

**Implementation**:

```python
def generate_metaphorical_physics(
    symbol: SymbolArchetype,
    emotional_context: EmotionalEssence
) -> str:
    """
    Define how symbol behaves in dream logic.

    Rules:
    1. Physics responds to emotions (fear → gravity, hope → levitation)
    2. Paradoxes are allowed (infinite room, non-Euclidean space)
    3. Symbolic meaning is amplified (guilt → literal weight)
    """

    physics_templates = {
        "caged_bird": [
            "Cage's size changes based on observer's distance from freedom",
            "Bird's song creates visible ripples in air, distorting space",
            "Cage door opens only when no one is watching"
        ],
        "quicksand": [
            "Sinking speed proportional to panic level",
            "Spreads to consume nearby ground when feared",
            "Becomes solid platform when accepted"
        ],
        "mirror": [
            "Reflection shows emotional truth, not physical appearance",
            "Multiple reflections show possible selves",
            "Cracks spread from points of self-deception"
        ]
    }

    # Get template or generate via LLM
    if symbol.symbol_id in physics_templates:
        return random.choice(physics_templates[symbol.symbol_id])
    else:
        return llm_generate_physics(symbol, emotional_context)
```

---

## Extensibility: Plugin Architecture

### Design Goal

New symbol domains (animals, elements, architecture) should plug in without modifying core code.

### Plugin Interface

```python
class SymbolDomainPlugin(Protocol):
    """
    Plugin interface for new symbol domains.

    Implement this to add new symbol categories:
    - Animal symbols (wolf, butterfly, snake, ...)
    - Elemental symbols (fire, water, earth, air, ...)
    - Architectural symbols (tower, ruin, labyrinth, ...)
    - Natural symbols (mountain, ocean, forest, ...)
    """

    @property
    def domain_name(self) -> str:
        """Domain name (e.g., "animals", "elements")."""
        ...

    def get_symbols(self) -> List[SymbolArchetype]:
        """Return all symbols in this domain."""
        ...

    def customize_selection(
        self,
        essence: EmotionalEssence,
        base_candidates: List[SymbolArchetype]
    ) -> List[SymbolArchetype]:
        """
        Optional: Modify candidate selection for this domain.

        Example: Animal domain might prefer "pack animals" for
                 emotions related to community/belonging.
        """
        return base_candidates  # Default: no modification

    def customize_physics(
        self,
        symbol: SymbolArchetype,
        scene: DreamScene
    ) -> str:
        """
        Optional: Domain-specific metaphorical physics.

        Example: Animal symbols might have pack behavior,
                 elemental symbols might have transformation rules.
        """
        return None  # Default: use standard physics
```

### Example Plugin: Animal Symbols

```python
class AnimalSymbolDomain(SymbolDomainPlugin):
    """Animal-based symbols (wolf, butterfly, bird, snake, ...)."""

    @property
    def domain_name(self) -> str:
        return "animals"

    def get_symbols(self) -> List[SymbolArchetype]:
        return [
            SymbolArchetype(
                symbol_id="wolf",
                category="power",
                embedding=embed_text("wild, primal, instinctual power"),
                emotion_tags=["strength", "ferocity", "independence"],
                context_tags=["self", "instinct", "survival"],
                universality_score=0.95,
                literary_references=["Jack London White Fang", "Cherokee Two Wolves"],
                visual_complexity="moderate"
            ),
            SymbolArchetype(
                symbol_id="butterfly",
                category="transformation",
                embedding=embed_text("delicate transformation, fleeting beauty"),
                emotion_tags=["metamorphosis", "fragility", "freedom"],
                context_tags=["change", "beauty", "transience"],
                universality_score=0.98,
                literary_references=["Zhuangzi Butterfly Dream", "Butterfly Effect"],
                visual_complexity="simple"
            ),
            # ... 48 more animal symbols
        ]

    def customize_physics(self, symbol: SymbolArchetype, scene: DreamScene) -> str:
        """Animals in dreams follow pack/flock/herd dynamics."""

        if symbol.symbol_id == "wolf":
            return "Appears alone initially, but howl summons pack from shadows. " \
                   "Pack size reflects dreamer's sense of support/isolation."

        if symbol.symbol_id == "butterfly":
            return "Wings generate visible emotion-colored dust. Lands on objects, " \
                   "transforming them temporarily into more beautiful versions."

        return None  # Use default physics
```

### Plugin Registration

```python
class SymbolicEncoder:
    """Main encoder with plugin support."""

    def __init__(self):
        self.domains: Dict[str, SymbolDomainPlugin] = {}
        self.symbols: List[SymbolArchetype] = []

        # Load core symbols
        self._load_core_symbols()

    def register_domain(self, domain: SymbolDomainPlugin):
        """Register a new symbol domain plugin."""
        self.domains[domain.domain_name] = domain

        # Add domain's symbols to database
        new_symbols = domain.get_symbols()
        self.symbols.extend(new_symbols)

        logger.info(f"Registered {len(new_symbols)} symbols from {domain.domain_name} domain")

    def encode(self, private_fact: str, context: DreamContext) -> SymbolArchetype:
        """Encode private fact → symbol (with plugin support)."""

        # Layer 1: Extract emotional essence
        essence = self.extractor.extract(private_fact, context.dreamer)

        # Layer 2: Select candidates (from all registered domains)
        candidates = self.selector.select_candidates(essence, context, k=10)

        # Allow plugins to customize selection
        for domain in self.domains.values():
            candidates = domain.customize_selection(essence, candidates)

        # Layer 3: Disambiguate
        symbol = self.disambiguator.disambiguate(
            candidates,
            dream_context=context,
            existing_symbols=context.scene_symbols
        )

        # Layer 4: Narrative integration (with plugin physics)
        scene = self.integrator.integrate_symbol(symbol, context.setting, context.participants)

        # Check if plugin has custom physics
        for domain in self.domains.values():
            custom_physics = domain.customize_physics(symbol, scene)
            if custom_physics:
                scene.metaphorical_physics = custom_physics

        return symbol

# Usage
encoder = SymbolicEncoder()

# Register plugins
encoder.register_domain(AnimalSymbolDomain())
encoder.register_domain(ElementalSymbolDomain())
encoder.register_domain(ArchitecturalSymbolDomain())

# Now encoder has 50 (core) + 50 (animals) + 50 (elements) + 50 (architecture) = 200 symbols
```

---

## Privacy Gradient: Configurable Obscurity

### The Problem

Different dream contexts need different privacy levels:
- **Solo dream**: Can show specific details (Bob's factory job)
- **Shared dream with stranger**: Maximum obscurity (generic "trapped" symbol)
- **Shared dream with spouse**: Moderate obscurity (more context)

### Solution: Obscurity Slider

```python
def apply_privacy_gradient(
    symbol: SymbolArchetype,
    obscurity_level: float,  # 0.0 = no privacy, 1.0 = maximum privacy
    private_fact: str
) -> SymbolArchetype:
    """
    Adjust symbol specificity based on privacy needs.

    obscurity_level:
      0.0-0.3: Specific (can include contextual details)
      0.3-0.7: Moderate (generic symbol, no specific context)
      0.7-1.0: Maximum (archetypal, extremely abstract)
    """

    if obscurity_level < 0.3:
        # Low obscurity: Include contextual hints
        # "caged_bird" → "bird in factory-shaped cage"
        return add_contextual_hints(symbol, private_fact)

    elif obscurity_level < 0.7:
        # Moderate obscurity: Pure symbol, no context
        # "caged_bird" (no factory hints)
        return symbol

    else:
        # Maximum obscurity: Abstract to archetype
        # "caged_bird" → "sense of confinement" (no visual form)
        return abstract_to_archetype(symbol)
```

**Example Transformations**:

| Private Fact | Obscurity 0.2 | Obscurity 0.5 | Obscurity 0.9 |
|--------------|---------------|---------------|---------------|
| "Bob hates factory job" | Bird in industrial cage | Caged bird | Vague sense of confinement |
| "Alice fears husband leaving" | Person at empty dinner table | Empty room | Feeling of absence |
| "Charlie owes $50k" | Heavy boulder labeled with numbers | Heavy stone | Oppressive weight |

---

## Implementation Roadmap

### Week 7: Foundation (4 days)

**Day 1: Emotional Extraction**
- [ ] `EmotionalExtractor` class
- [ ] NLP emotion analyzer (spaCy)
- [ ] LLM integration (Claude/Ollama)
- [ ] 228D semantic embedding

**Day 2: Symbol Database**
- [ ] `SymbolArchetype` dataclass
- [ ] Core 100 symbols (10 categories × 10 each)
- [ ] Load symbols from JSON
- [ ] Pre-compute embeddings

**Day 3: Symbol Selection**
- [ ] `SymbolSelector` class
- [ ] Multi-criteria scoring
- [ ] Top-k candidate selection
- [ ] Symbol interpolation (hybrid symbols)

**Day 4: Disambiguation + Integration**
- [ ] `SymbolDisambiguator` class
- [ ] `NarrativeIntegrator` class
- [ ] LLM scene generation
- [ ] Metaphorical physics templates

### Week 8: Extension (3 days)

**Day 5-6: Expand Symbol Database**
- [ ] Add 400 more symbols (total 500)
- [ ] Categorize by 10 emotions
- [ ] Add literary references
- [ ] Add cultural variants

**Day 7: Plugin Architecture**
- [ ] `SymbolDomainPlugin` protocol
- [ ] `AnimalSymbolDomain` example plugin
- [ ] Plugin registration system
- [ ] Custom physics support

### Week 9: Polish (2 days)

**Day 8: Privacy Gradient**
- [ ] Obscurity slider implementation
- [ ] Contextual hint system
- [ ] Archetype abstraction

**Day 9: Testing + Demo**
- [ ] End-to-end encoding test
- [ ] Demo: "Bob hates job" → dream scene
- [ ] Performance benchmarks
- [ ] Documentation

---

## Success Metrics

### Quantitative

- [ ] **Symbol Database**: 500 archetypes across 10 categories
- [ ] **Encoding Speed**: <100ms per private fact
- [ ] **Semantic Coherence**: >0.8 cosine similarity for related emotions
- [ ] **Plugin Support**: 3+ domain plugins working
- [ ] **Privacy**: 100% success rate (no private facts leak in shared dreams)

### Qualitative

- [ ] **Emotional Resonance**: Playtesters report symbols "feel right"
- [ ] **Poetic Beauty**: Scenes are evocative and memorable
- [ ] **Narrative Coherence**: Symbols tell a story together
- [ ] **Cultural Universality**: Symbols work across cultures
- [ ] **Extensibility**: New domains add easily

---

## Conclusion

The Symbolic Encoder is the **architectural heart** of NeuroHood's dream consciousness. By transforming private facts into universal symbols while preserving emotional truth, it enables:

1. **Privacy-Preserving Empathy**: Understand without knowing
2. **Poetic Expression**: Beauty in abstraction
3. **Narrative Coherence**: Symbols tell stories
4. **Infinite Extensibility**: Plugins add new domains forever

**This is elegant, extensible, meaningful, and moonshot-level ambitious.**

Ready to build the future of symbolic consciousness? 🌙✨

**Next**: Begin Week 7, Day 1 - Emotional Extraction implementation.
