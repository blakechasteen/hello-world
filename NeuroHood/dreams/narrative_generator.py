"""
Symbolic Narrative Generator - Phase 5 Week 2

Generates coherent multi-act dream narratives using:
- Collective Unconscious symbols as vocabulary
- Jungian archetypal patterns for story structure
- Neighborhood zeitgeist for emotional tone
- 12 narrative archetypes for variety
- Literary-inspired pacing from enriched references

The generator creates dreams that feel like meaningful stories rather
than random symbol sequences, following the three-act structure:
  Act 1 (Setup): Establish situation and introduce tension
  Act 2 (Climax): Conflict intensifies, transformation begins
  Act 3 (Resolution): Integration or new equilibrium

Author: Phase 5 Implementation
Date: November 2025
"""

import random
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from collective_unconscious import (
    CollectiveUnconscious,
    ArchetypalPattern,
    EmotionalEssence,
)


class NarrativeArchetype(Enum):
    """12 narrative archetypes for dream story structure."""
    JOURNEY = "journey"  # Movement through spaces, seeking
    CONFRONTATION = "confrontation"  # Facing opposition or self
    TRANSFORMATION = "transformation"  # Metamorphosis, change
    PURSUIT = "pursuit"  # Being chased or chasing
    DISCOVERY = "discovery"  # Finding hidden truth or object
    LOSS = "loss"  # Losing something precious
    REUNION = "reunion"  # Coming together again
    TRIAL = "trial"  # Testing, proving worth
    RESCUE = "rescue"  # Saving or being saved
    RETURN = "return"  # Coming back to origin
    THRESHOLD = "threshold"  # Crossing between worlds
    INTEGRATION = "integration"  # Becoming whole


class DreamMood(Enum):
    """Overall emotional tone of a dream."""
    NIGHTMARE = "nightmare"  # Fear, terror, anxiety
    MELANCHOLY = "melancholy"  # Sadness, loss, grief
    WONDER = "wonder"  # Awe, discovery, magic
    PEACEFUL = "peaceful"  # Calm, contentment
    TENSE = "tense"  # Anxiety, anticipation
    ECSTATIC = "ecstatic"  # Joy, transcendence
    MYSTERIOUS = "mysterious"  # Enigma, uncertainty
    TRANSFORMATIVE = "transformative"  # Change, growth


@dataclass
class NarrativeAct:
    """A single act in a dream narrative."""
    act_number: int  # 1, 2, or 3
    act_type: str  # "setup", "climax", "resolution"
    description: str
    symbols: List[Dict]
    emotional_intensity: float  # 0-1
    literary_echo: Optional[str] = None  # Reference to literature


@dataclass
class DreamNarrative:
    """A complete dream narrative with three acts."""
    title: str
    archetype: NarrativeArchetype
    mood: DreamMood
    acts: List[NarrativeAct]
    protagonist_state: Dict[str, float]  # Emotional state
    symbols_used: List[str]
    literary_inspirations: List[str]
    jungian_interpretation: str
    generation_timestamp: datetime = field(default_factory=datetime.now)

    def full_narrative(self) -> str:
        """Generate the complete narrative text."""
        parts = [f"# {self.title}\n"]
        parts.append(f"*A {self.mood.value} dream of {self.archetype.value}*\n")

        for act in self.acts:
            parts.append(f"\n## Act {act.act_number}: {act.act_type.title()}")
            parts.append(act.description)
            if act.literary_echo:
                parts.append(f"\n*Echoing: {act.literary_echo}*")

        parts.append(f"\n---\n*{self.jungian_interpretation}*")
        return "\n".join(parts)


class NarrativeGenerator:
    """
    Generates coherent dream narratives using symbols and archetypes.

    Usage:
        generator = NarrativeGenerator(collective_unconscious)

        narrative = generator.generate(
            resident_id="resident_001",
            emotional_state={"fear": 0.6, "hope": 0.4},
            consciousness_level=0.5,
            archetype=NarrativeArchetype.JOURNEY
        )

        print(narrative.full_narrative())
    """

    def __init__(self, collective: CollectiveUnconscious):
        self.collective = collective
        self._narrative_templates = self._init_narrative_templates()
        self._mood_mappings = self._init_mood_mappings()
        self._literary_patterns = self._init_literary_patterns()

    def generate(
        self,
        resident_id: str,
        emotional_state: Dict[str, float],
        consciousness_level: float = 0.5,
        archetype: Optional[NarrativeArchetype] = None,
        seed_symbols: Optional[List[str]] = None,
        length: str = "medium"  # "short", "medium", "long"
    ) -> DreamNarrative:
        """
        Generate a complete dream narrative.

        Args:
            resident_id: The resident having the dream
            emotional_state: Dict of emotion -> intensity
            consciousness_level: 0=individual, 1=collective
            archetype: Narrative archetype (auto-selected if None)
            seed_symbols: Starting symbols to build from
            length: Dream length (short=1 act, medium=3 acts, long=5 acts)

        Returns:
            Complete DreamNarrative with acts and interpretation
        """
        # Select or use provided archetype
        if archetype is None:
            archetype = self._select_archetype(emotional_state)

        # Determine mood from emotional state
        mood = self._determine_mood(emotional_state)

        # Get symbols for the dream
        num_symbols = {"short": 3, "medium": 5, "long": 8}.get(length, 5)
        symbols = self.collective.get_dream_symbols(
            resident_id=resident_id,
            emotional_state=emotional_state,
            count=num_symbols,
            consciousness_level=consciousness_level,
            existing_symbols=seed_symbols
        )

        # Generate narrative acts
        num_acts = {"short": 1, "medium": 3, "long": 5}.get(length, 3)
        acts = self._generate_acts(
            archetype=archetype,
            mood=mood,
            symbols=symbols,
            emotional_state=emotional_state,
            num_acts=num_acts
        )

        # Generate title
        title = self._generate_title(archetype, symbols, mood)

        # Get literary inspirations
        inspirations = self._get_literary_inspirations(symbols)

        # Generate Jungian interpretation
        interpretation = self._generate_interpretation(
            archetype, mood, symbols, emotional_state
        )

        return DreamNarrative(
            title=title,
            archetype=archetype,
            mood=mood,
            acts=acts,
            protagonist_state=emotional_state,
            symbols_used=[s["symbol_id"] for s in symbols],
            literary_inspirations=inspirations,
            jungian_interpretation=interpretation
        )

    def _select_archetype(self, emotional_state: Dict[str, float]) -> NarrativeArchetype:
        """Select narrative archetype based on emotional state."""
        # Map dominant emotions to likely archetypes
        mappings = {
            "fear": [NarrativeArchetype.PURSUIT, NarrativeArchetype.CONFRONTATION],
            "hope": [NarrativeArchetype.JOURNEY, NarrativeArchetype.DISCOVERY],
            "grief": [NarrativeArchetype.LOSS, NarrativeArchetype.RETURN],
            "transformation": [NarrativeArchetype.TRANSFORMATION, NarrativeArchetype.THRESHOLD],
            "connection": [NarrativeArchetype.REUNION, NarrativeArchetype.INTEGRATION],
            "conflict": [NarrativeArchetype.CONFRONTATION, NarrativeArchetype.TRIAL],
            "anxiety": [NarrativeArchetype.PURSUIT, NarrativeArchetype.TRIAL],
        }

        # Find dominant emotion
        if not emotional_state:
            return random.choice(list(NarrativeArchetype))

        dominant = max(emotional_state, key=emotional_state.get)
        possible = mappings.get(dominant, list(NarrativeArchetype))

        return random.choice(possible)

    def _determine_mood(self, emotional_state: Dict[str, float]) -> DreamMood:
        """Determine overall dream mood from emotional state."""
        if not emotional_state:
            return DreamMood.MYSTERIOUS

        fear = emotional_state.get("fear", 0) + emotional_state.get("anxiety", 0)
        hope = emotional_state.get("hope", 0) + emotional_state.get("joy", 0)
        grief = emotional_state.get("grief", 0) + emotional_state.get("loss", 0)
        transformation = emotional_state.get("transformation", 0)

        if fear > 0.7:
            return DreamMood.NIGHTMARE
        elif grief > 0.6:
            return DreamMood.MELANCHOLY
        elif hope > 0.6:
            return DreamMood.WONDER
        elif transformation > 0.5:
            return DreamMood.TRANSFORMATIVE
        elif fear > 0.4:
            return DreamMood.TENSE
        elif hope > 0.3:
            return DreamMood.PEACEFUL
        else:
            return DreamMood.MYSTERIOUS

    def _generate_acts(
        self,
        archetype: NarrativeArchetype,
        mood: DreamMood,
        symbols: List[Dict],
        emotional_state: Dict[str, float],
        num_acts: int
    ) -> List[NarrativeAct]:
        """Generate narrative acts with increasing intensity."""
        template = self._narrative_templates.get(archetype.value, {})
        acts = []

        # Distribute symbols across acts
        symbols_per_act = max(1, len(symbols) // num_acts)

        # Generate each act
        for i in range(num_acts):
            # Determine act type
            if num_acts == 1:
                act_type = "vision"
            elif i == 0:
                act_type = "setup"
            elif i == num_acts - 1:
                act_type = "resolution"
            else:
                act_type = "climax"

            # Get symbols for this act
            start_idx = i * symbols_per_act
            end_idx = start_idx + symbols_per_act if i < num_acts - 1 else len(symbols)
            act_symbols = symbols[start_idx:end_idx] if start_idx < len(symbols) else []

            # Calculate emotional intensity (builds to climax, then resolves)
            if num_acts == 1:
                intensity = 0.7
            else:
                peak = (num_acts - 1) // 2 + 1
                if i < peak:
                    intensity = 0.3 + (i / peak) * 0.5
                else:
                    intensity = 0.8 - ((i - peak) / (num_acts - peak)) * 0.3

            # Generate description
            description = self._generate_act_description(
                archetype, mood, act_type, act_symbols, intensity
            )

            # Get literary echo
            literary_echo = self._get_literary_echo(act_symbols, mood)

            acts.append(NarrativeAct(
                act_number=i + 1,
                act_type=act_type,
                description=description,
                symbols=act_symbols,
                emotional_intensity=intensity,
                literary_echo=literary_echo
            ))

        return acts

    def _generate_act_description(
        self,
        archetype: NarrativeArchetype,
        mood: DreamMood,
        act_type: str,
        symbols: List[Dict],
        intensity: float
    ) -> str:
        """Generate narrative description for an act."""
        # Get template elements
        template_key = f"{archetype.value}_{act_type}"
        templates = self._get_act_templates(archetype, act_type)

        # Build symbol descriptions
        symbol_phrases = []
        for sym in symbols:
            desc = sym.get("description", sym.get("symbol_id", "unknown symbol"))
            symbol_phrases.append(desc)

        # Combine into narrative
        if not templates:
            # Fallback narrative construction
            if symbols:
                return self._construct_fallback_description(
                    symbols, mood, act_type, intensity
                )
            return "The dream unfolds in mysterious ways..."

        template = random.choice(templates)

        # Fill template with symbols
        if symbol_phrases:
            filled = template.format(
                symbol1=symbol_phrases[0] if len(symbol_phrases) > 0 else "a mysterious presence",
                symbol2=symbol_phrases[1] if len(symbol_phrases) > 1 else "shifting shadows",
                symbol3=symbol_phrases[2] if len(symbol_phrases) > 2 else "distant echoes",
                mood=mood.value,
                intensity="intensely" if intensity > 0.7 else "gently" if intensity < 0.4 else ""
            )
            return filled

        return template

    def _get_act_templates(self, archetype: NarrativeArchetype, act_type: str) -> List[str]:
        """Get narrative templates for an archetype and act type."""
        templates = {
            "journey_setup": [
                "You find yourself at the beginning of a path. Ahead lies {symbol1}. "
                "The way forward calls to you, though {symbol2} lurks at the edges.",
                "The journey begins where {symbol1} meets the horizon. "
                "Something within urges you forward, past {symbol2}.",
            ],
            "journey_climax": [
                "Deep into the journey now, you encounter {symbol1} in its most vivid form. "
                "The way becomes difficult as {symbol2} demands your attention. "
                "There is no turning back.",
                "The path twists through {symbol1}. Here, at the heart of the journey, "
                "{symbol2} reveals what was hidden. The air thrums with possibility.",
            ],
            "journey_resolution": [
                "The journey nears its end. {symbol1} transforms before your eyes, "
                "becoming something you never expected. Understanding dawns.",
                "You emerge from the journey changed. What was {symbol1} is now "
                "a part of you. The path continues, but you are different.",
            ],
            "confrontation_setup": [
                "Before you stands {symbol1}, demanding to be faced. "
                "Behind you, {symbol2} blocks any retreat. The moment of truth approaches.",
                "The confrontation you've avoided materializes as {symbol1}. "
                "There is no escape, no alternative. Only facing what must be faced.",
            ],
            "confrontation_climax": [
                "Face to face with {symbol1}, you feel {intensity} the weight of this moment. "
                "{symbol2} watches from the shadows, waiting to see what you'll become.",
                "The confrontation reaches its peak. {symbol1} reveals its true nature, "
                "and you must choose: stand or fall, integrate or shatter.",
            ],
            "confrontation_resolution": [
                "The confrontation ends not with victory or defeat, but with understanding. "
                "{symbol1} becomes part of your story, no longer an enemy.",
                "What you confronted was yourself all along. {symbol1} dissolves into light, "
                "leaving only the truth of who you are.",
            ],
            "transformation_setup": [
                "Something is changing. You feel it in the air around {symbol1}. "
                "The old form cannot hold; {symbol2} signals the beginning.",
                "The chrysalis of the self begins to crack. {symbol1} represents "
                "what you're leaving behind. {symbol2} is what awaits.",
            ],
            "transformation_climax": [
                "The transformation accelerates. {symbol1} tears apart and reforms, "
                "neither one thing nor another. You are caught in the becoming.",
                "In the crucible of change, {symbol1} and {symbol2} merge and separate. "
                "You cannot tell where you end and the transformation begins.",
            ],
            "transformation_resolution": [
                "The metamorphosis completes. Where {symbol1} once stood, "
                "something new emerges. You are not who you were.",
                "Transformation complete, you look upon {symbol1} with new eyes. "
                "The world is the same, but you are reborn.",
            ],
            "pursuit_setup": [
                "They are coming. {symbol1} appears in flashes behind you. "
                "Your heart pounds as {symbol2} closes the distance.",
                "The pursuit has begun. No matter which way you turn, "
                "{symbol1} is there, always closer than before.",
            ],
            "pursuit_climax": [
                "You cannot outrun {symbol1}. It catches you, surrounds you, "
                "becomes everything. {symbol2} offers no escape.",
                "The chase reaches its desperate peak. Cornered by {symbol1}, "
                "you must face what you've been fleeing.",
            ],
            "pursuit_resolution": [
                "The pursuit ends when you stop running. {symbol1} was never the enemy, "
                "only a messenger. Understanding replaces fear.",
                "What chased you dissolves into mist. {symbol1} was your own shadow, "
                "desperate to be acknowledged. Now you are whole.",
            ],
        }

        key = f"{archetype.value}_{act_type}"
        return templates.get(key, [])

    def _construct_fallback_description(
        self,
        symbols: List[Dict],
        mood: DreamMood,
        act_type: str,
        intensity: float
    ) -> str:
        """Construct a description when no template matches."""
        mood_phrases = {
            DreamMood.NIGHTMARE: "In the grip of terror",
            DreamMood.MELANCHOLY: "Weighted by sorrow",
            DreamMood.WONDER: "Filled with awe",
            DreamMood.PEACEFUL: "In gentle stillness",
            DreamMood.TENSE: "With bated breath",
            DreamMood.ECSTATIC: "Overwhelmed with joy",
            DreamMood.MYSTERIOUS: "In enigmatic depths",
            DreamMood.TRANSFORMATIVE: "Through the crucible of change",
        }

        opener = mood_phrases.get(mood, "In the dream")

        symbol_descs = [s.get("description", s.get("symbol_id")) for s in symbols[:2]]
        if symbol_descs:
            return f"{opener}, you encounter {symbol_descs[0]}. " + (
                f"Nearby, {symbol_descs[1]} shifts in the periphery." if len(symbol_descs) > 1 else ""
            )

        return f"{opener}, the dream unfolds in ways beyond words."

    def _get_literary_echo(self, symbols: List[Dict], mood: DreamMood) -> Optional[str]:
        """Get a literary reference that echoes this moment."""
        for sym in symbols:
            refs = sym.get("literary_references", {})
            category_map = {
                DreamMood.NIGHTMARE: "modern_cinema",
                DreamMood.MELANCHOLY: "poetry_visual_arts",
                DreamMood.WONDER: "classical_mythology",
                DreamMood.MYSTERIOUS: "philosophy_religion",
            }
            preferred = category_map.get(mood, "world_literature")
            category_refs = refs.get(preferred, [])

            if category_refs:
                ref = random.choice(category_refs)
                title = ref.get("title", "")
                author = ref.get("author", ref.get("culture", ""))
                if title:
                    return f"{title} ({author})" if author else title

        return None

    def _generate_title(
        self,
        archetype: NarrativeArchetype,
        symbols: List[Dict],
        mood: DreamMood
    ) -> str:
        """Generate an evocative title for the dream."""
        title_patterns = {
            NarrativeArchetype.JOURNEY: ["The Path of {symbol}", "Journey Through {symbol}", "Where {symbol} Leads"],
            NarrativeArchetype.CONFRONTATION: ["Facing {symbol}", "The {symbol} Within", "Stand Before {symbol}"],
            NarrativeArchetype.TRANSFORMATION: ["{symbol} Transformed", "Becoming {symbol}", "The Metamorphosis of {symbol}"],
            NarrativeArchetype.PURSUIT: ["Flight from {symbol}", "The Chase of {symbol}", "{symbol} Pursues"],
            NarrativeArchetype.DISCOVERY: ["Finding {symbol}", "The {symbol} Revealed", "Discovery of {symbol}"],
            NarrativeArchetype.LOSS: ["Losing {symbol}", "The Absence of {symbol}", "When {symbol} Fades"],
            NarrativeArchetype.REUNION: ["Return to {symbol}", "{symbol} Found Again", "Reunion with {symbol}"],
            NarrativeArchetype.TRIAL: ["The Trial of {symbol}", "Proving {symbol}", "Test of {symbol}"],
            NarrativeArchetype.RESCUE: ["Saving {symbol}", "{symbol} Delivered", "The Rescue from {symbol}"],
            NarrativeArchetype.RETURN: ["Coming Home to {symbol}", "The Return of {symbol}", "Back to {symbol}"],
            NarrativeArchetype.THRESHOLD: ["Crossing {symbol}", "The Gate of {symbol}", "{symbol}'s Threshold"],
            NarrativeArchetype.INTEGRATION: ["Becoming Whole Through {symbol}", "{symbol} United", "Integration of {symbol}"],
        }

        patterns = title_patterns.get(archetype, ["The Dream of {symbol}"])
        pattern = random.choice(patterns)

        if symbols:
            # Use first symbol's ID, formatted nicely
            symbol_name = symbols[0].get("symbol_id", "shadow").replace("_", " ").title()
            return pattern.format(symbol=symbol_name)

        return pattern.format(symbol="Shadow")

    def _get_literary_inspirations(self, symbols: List[Dict]) -> List[str]:
        """Get literary inspirations from symbols."""
        inspirations = []
        seen = set()

        for sym in symbols:
            refs = sym.get("literary_references", {})
            for category_refs in refs.values():
                for ref in category_refs[:2]:
                    title = ref.get("title", "")
                    if title and title not in seen:
                        seen.add(title)
                        author = ref.get("author", "")
                        if author:
                            inspirations.append(f"{title} by {author}")
                        else:
                            inspirations.append(title)

                        if len(inspirations) >= 3:
                            return inspirations

        return inspirations

    def _generate_interpretation(
        self,
        archetype: NarrativeArchetype,
        mood: DreamMood,
        symbols: List[Dict],
        emotional_state: Dict[str, float]
    ) -> str:
        """Generate a Jungian psychological interpretation."""
        interpretations = {
            NarrativeArchetype.JOURNEY: (
                "This journey dream represents the individuation process - "
                "the psyche's movement toward wholeness. The path is the Self "
                "calling you toward integration."
            ),
            NarrativeArchetype.CONFRONTATION: (
                "Confrontation dreams often involve Shadow integration. "
                "What you face is not external but a disowned part of yourself "
                "seeking acknowledgment and reintegration."
            ),
            NarrativeArchetype.TRANSFORMATION: (
                "Transformation dreams signal deep psychic change. "
                "The death of the old self makes way for rebirth. "
                "Trust the process of becoming."
            ),
            NarrativeArchetype.PURSUIT: (
                "Being pursued represents running from aspects of yourself "
                "that demand attention. The pursuer is often the Shadow, "
                "seeking integration rather than harm."
            ),
            NarrativeArchetype.DISCOVERY: (
                "Discovery dreams reveal hidden aspects of the Self. "
                "What you find has always been within you, "
                "waiting for the right moment to emerge."
            ),
            NarrativeArchetype.LOSS: (
                "Loss dreams process grief and change. "
                "What is lost in the dream often represents what must be released "
                "for growth to occur."
            ),
            NarrativeArchetype.REUNION: (
                "Reunion dreams often involve the Anima/Animus - "
                "the contrasexual element of the psyche. "
                "Coming together represents inner marriage and wholeness."
            ),
            NarrativeArchetype.TRIAL: (
                "Trial dreams test the developing ego. "
                "Success or failure matters less than the willingness "
                "to face the challenge and grow."
            ),
            NarrativeArchetype.THRESHOLD: (
                "Threshold dreams mark transitions between life phases. "
                "Crossing represents leaving the known for the unknown, "
                "a necessary step in individuation."
            ),
            NarrativeArchetype.INTEGRATION: (
                "Integration dreams show the psyche moving toward wholeness. "
                "Disparate elements coming together represents "
                "the union of conscious and unconscious."
            ),
        }

        base = interpretations.get(archetype, "This dream invites reflection on the journey of the Self.")

        # Add symbol-specific insight
        if symbols:
            sym = symbols[0]
            archetypal_roots = sym.get("archetypal_roots", {})
            primary = archetypal_roots.get("primary", "")
            if primary:
                base += f"\n\nThe central symbol connects to {primary}, "
                "suggesting work at a deep archetypal level."

        return base

    def _init_narrative_templates(self) -> Dict[str, Dict]:
        """Initialize narrative templates for each archetype."""
        return {
            archetype.value: {"name": archetype.value}
            for archetype in NarrativeArchetype
        }

    def _init_mood_mappings(self) -> Dict[str, DreamMood]:
        """Initialize emotion to mood mappings."""
        return {
            "fear": DreamMood.NIGHTMARE,
            "anxiety": DreamMood.TENSE,
            "grief": DreamMood.MELANCHOLY,
            "hope": DreamMood.WONDER,
            "joy": DreamMood.ECSTATIC,
            "transformation": DreamMood.TRANSFORMATIVE,
            "mystery": DreamMood.MYSTERIOUS,
            "peace": DreamMood.PEACEFUL,
        }

    def _init_literary_patterns(self) -> Dict[str, List[str]]:
        """Initialize literary pattern references."""
        return {
            "journey": ["Odyssey", "Divine Comedy", "Pilgrim's Progress"],
            "confrontation": ["Beowulf", "Paradise Lost", "Crime and Punishment"],
            "transformation": ["Metamorphoses", "Kafka's Metamorphosis"],
        }


async def create_narrative_generator(
    collective: Optional[CollectiveUnconscious] = None
) -> NarrativeGenerator:
    """Create a narrative generator with collective unconscious."""
    if collective is None:
        from collective_unconscious import create_collective_unconscious
        collective = await create_collective_unconscious()
    return NarrativeGenerator(collective)


if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 70)
        print("SYMBOLIC NARRATIVE GENERATOR DEMO")
        print("=" * 70)

        # Create generator with collective unconscious
        from collective_unconscious import create_collective_unconscious
        collective = await create_collective_unconscious()
        generator = NarrativeGenerator(collective)

        # Generate a journey dream
        print("\n--- Journey Dream (Fear + Hope) ---")
        narrative1 = generator.generate(
            resident_id="demo_resident",
            emotional_state={"fear": 0.4, "hope": 0.6, "transformation": 0.3},
            consciousness_level=0.5,
            archetype=NarrativeArchetype.JOURNEY,
            length="medium"
        )
        print(narrative1.full_narrative())

        # Generate a confrontation nightmare
        print("\n" + "=" * 70)
        print("\n--- Confrontation Nightmare ---")
        narrative2 = generator.generate(
            resident_id="demo_resident",
            emotional_state={"fear": 0.8, "conflict": 0.7},
            consciousness_level=0.3,
            archetype=NarrativeArchetype.CONFRONTATION,
            length="medium"
        )
        print(narrative2.full_narrative())

        # Auto-select archetype based on grief
        print("\n" + "=" * 70)
        print("\n--- Auto-Selected Archetype (Grief) ---")
        narrative3 = generator.generate(
            resident_id="demo_resident",
            emotional_state={"grief": 0.7, "hope": 0.2},
            consciousness_level=0.7,
            length="short"
        )
        print(f"Auto-selected archetype: {narrative3.archetype.value}")
        print(f"Mood: {narrative3.mood.value}")
        print(narrative3.full_narrative())

        # Summary stats
        print("\n" + "=" * 70)
        print("GENERATION SUMMARY")
        print("=" * 70)
        print(f"\nDream 1: '{narrative1.title}'")
        print(f"  Archetype: {narrative1.archetype.value}")
        print(f"  Mood: {narrative1.mood.value}")
        print(f"  Symbols: {', '.join(narrative1.symbols_used)}")
        print(f"  Literary echoes: {narrative1.literary_inspirations[:2]}")

    asyncio.run(demo())