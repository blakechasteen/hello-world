"""
Dream Influence System - Phase 5 Week 3

Implements the bidirectional influence between dreams and waking life:
- Dreams affect resident mood, relationships, and actions
- Waking experiences shape future dreams
- Collective dreams can shift neighborhood zeitgeist

The system models Jung's concept that dreams are not random but purposeful
communications from the unconscious, offering guidance and integration.

Author: Claude (Phase 5 Implementation)
Date: November 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import random
import math
import json


class InfluenceType(Enum):
    """Types of influence dreams can have on waking life."""
    MOOD_SHIFT = "mood_shift"          # Changes emotional state
    INSIGHT = "insight"                 # Provides clarity or understanding
    WARNING = "warning"                 # Alerts to danger or issue
    HEALING = "healing"                 # Processes trauma or grief
    CONNECTION = "connection"           # Strengthens relationships
    CREATIVITY = "creativity"           # Inspires new ideas
    PREMONITION = "premonition"         # Hints at future events
    SHADOW_WORK = "shadow_work"         # Confronts repressed aspects


class InfluenceStrength(Enum):
    """Strength of dream influence on waking life."""
    SUBTLE = 0.1      # Barely noticeable
    MILD = 0.25       # Vague feeling
    MODERATE = 0.5    # Clear impact
    STRONG = 0.75     # Significant change
    PROFOUND = 1.0    # Life-altering


@dataclass
class DreamInfluence:
    """Represents a single dream's influence on waking life."""
    influence_id: str
    dream_id: str
    resident_id: str
    influence_type: InfluenceType
    strength: InfluenceStrength
    description: str
    affected_traits: Dict[str, float]  # trait -> delta
    affected_relationships: Dict[str, float]  # other_resident_id -> delta
    duration_hours: float
    created_at: datetime
    expires_at: datetime
    symbols_involved: List[str]
    archetype: Optional[str] = None
    is_collective: bool = False  # From shared dream

    def is_active(self) -> bool:
        """Check if influence is still active."""
        return datetime.now() < self.expires_at

    def remaining_strength(self) -> float:
        """Calculate remaining strength based on decay."""
        if not self.is_active():
            return 0.0

        total_duration = (self.expires_at - self.created_at).total_seconds()
        elapsed = (datetime.now() - self.created_at).total_seconds()
        decay_factor = 1 - (elapsed / total_duration) ** 0.5  # Square root decay

        return self.strength.value * decay_factor


@dataclass
class WakingExperience:
    """An experience from waking life that shapes dreams."""
    experience_id: str
    resident_id: str
    experience_type: str  # "conversation", "event", "emotion", "discovery"
    description: str
    emotional_impact: Dict[str, float]  # emotion -> intensity
    people_involved: List[str]
    symbols_evoked: List[str]  # Symbols this might trigger in dreams
    timestamp: datetime
    processed_in_dream: bool = False

    def dream_probability(self) -> float:
        """Calculate probability this experience appears in dreams."""
        # Higher emotional intensity = more likely to dream about
        max_emotion = max(self.emotional_impact.values()) if self.emotional_impact else 0.1
        recency = 1.0 / (1 + (datetime.now() - self.timestamp).days)
        return min(1.0, max_emotion * recency)


@dataclass
class InfluenceState:
    """Complete influence state for a resident."""
    resident_id: str
    active_influences: List[DreamInfluence] = field(default_factory=list)
    pending_experiences: List[WakingExperience] = field(default_factory=list)
    processed_experiences: List[str] = field(default_factory=list)  # experience_ids
    cumulative_mood_shift: Dict[str, float] = field(default_factory=dict)
    relationship_deltas: Dict[str, float] = field(default_factory=dict)
    insight_count: int = 0
    healing_progress: float = 0.0  # 0-1 scale
    shadow_integration: float = 0.0  # 0-1 scale


class DreamInfluenceSystem:
    """
    Main system for managing dream-waking interactions.

    Models the Jungian concept of dreams as meaningful communications
    that serve the psyche's development and integration.
    """

    def __init__(self):
        self.resident_states: Dict[str, InfluenceState] = {}
        self.collective_influences: List[DreamInfluence] = []
        self.neighborhood_mood: Dict[str, float] = {
            "anxiety": 0.0,
            "hope": 0.0,
            "connection": 0.0,
            "tension": 0.0,
            "creativity": 0.0,
            "healing": 0.0,
        }

        # Influence type -> trait mappings
        self.influence_trait_map = {
            InfluenceType.MOOD_SHIFT: ["happiness", "anxiety", "peace"],
            InfluenceType.INSIGHT: ["wisdom", "clarity", "confidence"],
            InfluenceType.WARNING: ["caution", "awareness", "anxiety"],
            InfluenceType.HEALING: ["peace", "acceptance", "hope"],
            InfluenceType.CONNECTION: ["empathy", "trust", "openness"],
            InfluenceType.CREATIVITY: ["inspiration", "energy", "curiosity"],
            InfluenceType.PREMONITION: ["intuition", "unease", "awareness"],
            InfluenceType.SHADOW_WORK: ["self_awareness", "tension", "growth"],
        }

        # Symbol -> influence type associations
        self.symbol_influence_map = {
            # Trapped symbols often warn or heal
            "cage": [InfluenceType.WARNING, InfluenceType.SHADOW_WORK],
            "chains": [InfluenceType.SHADOW_WORK, InfluenceType.HEALING],
            "locked_door": [InfluenceType.WARNING, InfluenceType.INSIGHT],

            # Loss symbols trigger healing
            "empty_chair": [InfluenceType.HEALING, InfluenceType.CONNECTION],
            "fading_photo": [InfluenceType.HEALING, InfluenceType.INSIGHT],
            "torn_letter": [InfluenceType.HEALING, InfluenceType.CONNECTION],

            # Hope symbols uplift mood
            "sunrise": [InfluenceType.MOOD_SHIFT, InfluenceType.HEALING],
            "sprout": [InfluenceType.CREATIVITY, InfluenceType.MOOD_SHIFT],
            "rainbow": [InfluenceType.MOOD_SHIFT, InfluenceType.CONNECTION],

            # Transformation symbols drive growth
            "butterfly": [InfluenceType.INSIGHT, InfluenceType.HEALING],
            "phoenix": [InfluenceType.HEALING, InfluenceType.SHADOW_WORK],
            "molting_snake": [InfluenceType.SHADOW_WORK, InfluenceType.INSIGHT],

            # Fear symbols warn
            "shadow_figure": [InfluenceType.WARNING, InfluenceType.SHADOW_WORK],
            "teeth_falling": [InfluenceType.WARNING, InfluenceType.INSIGHT],
            "drowning": [InfluenceType.WARNING, InfluenceType.HEALING],

            # Connection symbols strengthen bonds
            "bridge": [InfluenceType.CONNECTION, InfluenceType.HEALING],
            "handshake": [InfluenceType.CONNECTION, InfluenceType.INSIGHT],
            "shared_meal": [InfluenceType.CONNECTION, InfluenceType.MOOD_SHIFT],
        }

        # Archetype -> influence mapping
        self.archetype_influence_map = {
            "heros_journey": InfluenceType.INSIGHT,
            "shadow_integration": InfluenceType.SHADOW_WORK,
            "death_rebirth": InfluenceType.HEALING,
            "great_mother": InfluenceType.HEALING,
            "wise_old_one": InfluenceType.INSIGHT,
            "divine_child": InfluenceType.CREATIVITY,
            "trickster": InfluenceType.INSIGHT,
            "anima_animus": InfluenceType.CONNECTION,
        }

    def get_or_create_state(self, resident_id: str) -> InfluenceState:
        """Get or create influence state for a resident."""
        if resident_id not in self.resident_states:
            self.resident_states[resident_id] = InfluenceState(resident_id=resident_id)
        return self.resident_states[resident_id]

    def process_dream(
        self,
        dream: Dict[str, Any],
        resident_id: str
    ) -> List[DreamInfluence]:
        """
        Process a completed dream and generate influences.

        Args:
            dream: Dream data with symbols, archetype, mood, narrative
            resident_id: Dreamer's ID

        Returns:
            List of generated influences
        """
        influences = []
        state = self.get_or_create_state(resident_id)

        symbols = dream.get("symbols_used", [])
        archetype = dream.get("archetype")
        mood = dream.get("mood", "mysterious")

        # Determine influence types from symbols
        influence_types = self._determine_influence_types(symbols, archetype)

        # Calculate base strength from dream intensity
        base_strength = self._calculate_dream_intensity(dream)

        for inf_type in influence_types:
            # Generate influence
            influence = self._create_influence(
                dream_id=dream.get("id", f"dream_{datetime.now().timestamp()}"),
                resident_id=resident_id,
                influence_type=inf_type,
                base_strength=base_strength,
                symbols=symbols,
                archetype=archetype,
                mood=mood,
            )

            influences.append(influence)
            state.active_influences.append(influence)

        # Mark pending experiences as processed
        self._process_pending_experiences(state, symbols)

        # Update cumulative effects
        self._update_cumulative_effects(state, influences)

        return influences

    def _determine_influence_types(
        self,
        symbols: List[str],
        archetype: Optional[str]
    ) -> List[InfluenceType]:
        """Determine influence types from dream content."""
        types = set()

        # From symbols
        for symbol in symbols:
            if symbol in self.symbol_influence_map:
                types.update(self.symbol_influence_map[symbol])

        # From archetype
        if archetype and archetype in self.archetype_influence_map:
            types.add(self.archetype_influence_map[archetype])

        # Default if none found
        if not types:
            types.add(InfluenceType.INSIGHT)

        # Limit to 3 influence types per dream
        return list(types)[:3]

    def _calculate_dream_intensity(self, dream: Dict[str, Any]) -> float:
        """Calculate dream intensity from content."""
        intensity = 0.5  # Base

        # More symbols = more intense
        symbol_count = len(dream.get("symbols_used", []))
        intensity += min(0.2, symbol_count * 0.04)

        # Certain moods are more intense
        intense_moods = {"nightmare", "ecstatic", "transformative"}
        if dream.get("mood") in intense_moods:
            intensity += 0.2

        # Archetypes add intensity
        if dream.get("archetype"):
            intensity += 0.1

        return min(1.0, intensity)

    def _create_influence(
        self,
        dream_id: str,
        resident_id: str,
        influence_type: InfluenceType,
        base_strength: float,
        symbols: List[str],
        archetype: Optional[str],
        mood: str,
    ) -> DreamInfluence:
        """Create a single influence from dream."""
        # Determine strength level
        if base_strength >= 0.9:
            strength = InfluenceStrength.PROFOUND
        elif base_strength >= 0.7:
            strength = InfluenceStrength.STRONG
        elif base_strength >= 0.5:
            strength = InfluenceStrength.MODERATE
        elif base_strength >= 0.25:
            strength = InfluenceStrength.MILD
        else:
            strength = InfluenceStrength.SUBTLE

        # Generate affected traits
        affected_traits = {}
        for trait in self.influence_trait_map.get(influence_type, ["awareness"]):
            # Random delta based on strength
            delta = random.uniform(0.05, 0.2) * strength.value
            # Some influences decrease traits (anxiety from warnings)
            if trait in ["anxiety", "tension", "unease"] and influence_type != InfluenceType.WARNING:
                delta = -delta
            affected_traits[trait] = round(delta, 3)

        # Generate description
        description = self._generate_influence_description(
            influence_type, symbols, mood
        )

        # Duration based on strength (8-72 hours)
        duration = 8 + (strength.value * 64)

        now = datetime.now()

        return DreamInfluence(
            influence_id=f"inf_{now.timestamp()}_{random.randint(1000, 9999)}",
            dream_id=dream_id,
            resident_id=resident_id,
            influence_type=influence_type,
            strength=strength,
            description=description,
            affected_traits=affected_traits,
            affected_relationships={},
            duration_hours=duration,
            created_at=now,
            expires_at=now + timedelta(hours=duration),
            symbols_involved=symbols[:5],
            archetype=archetype,
        )

    def _generate_influence_description(
        self,
        influence_type: InfluenceType,
        symbols: List[str],
        mood: str
    ) -> str:
        """Generate human-readable influence description."""
        descriptions = {
            InfluenceType.MOOD_SHIFT: [
                "A lingering feeling from the dream colors the day",
                "The dream's atmosphere echoes through waking hours",
                f"Something about {symbols[0] if symbols else 'the dream'} stays with you",
            ],
            InfluenceType.INSIGHT: [
                "A sudden clarity emerges from the dream's imagery",
                "The dream reveals something previously hidden",
                "Understanding dawns, though you're not sure why",
            ],
            InfluenceType.WARNING: [
                "An unsettling feeling persists, urging caution",
                "The dream's warning echoes in the back of your mind",
                "Something feels off; the dream suggests care is needed",
            ],
            InfluenceType.HEALING: [
                "A sense of release, as if something finally let go",
                "The dream brought old pain to the surface to process",
                "Healing happens in the night, leaving peace by morning",
            ],
            InfluenceType.CONNECTION: [
                "Warmth toward someone lingers from the dream",
                "The dream reminds you of what matters in relationships",
                "A desire to reach out to someone grows stronger",
            ],
            InfluenceType.CREATIVITY: [
                "Ideas flow more freely after the dream",
                "Inspiration struck in sleep; creativity awakens",
                "The dream planted seeds of new possibilities",
            ],
            InfluenceType.PREMONITION: [
                "A sense of déjà vu haunts the day",
                "The dream felt more like prophecy than fantasy",
                "Something about today feels familiar, foretold",
            ],
            InfluenceType.SHADOW_WORK: [
                "Confronting something difficult, but necessary",
                "The dream dragged hidden aspects into light",
                "Integration of shadow aspects continues by day",
            ],
        }

        options = descriptions.get(influence_type, ["The dream lingers"])
        return random.choice(options)

    def _process_pending_experiences(
        self,
        state: InfluenceState,
        dream_symbols: List[str]
    ):
        """Mark experiences that appeared in dream as processed."""
        for exp in state.pending_experiences:
            # If dream symbols overlap with experience symbols
            if set(exp.symbols_evoked) & set(dream_symbols):
                exp.processed_in_dream = True
                state.processed_experiences.append(exp.experience_id)

    def _update_cumulative_effects(
        self,
        state: InfluenceState,
        new_influences: List[DreamInfluence]
    ):
        """Update cumulative mood and relationship effects."""
        for inf in new_influences:
            # Update mood shifts
            for trait, delta in inf.affected_traits.items():
                current = state.cumulative_mood_shift.get(trait, 0.0)
                state.cumulative_mood_shift[trait] = max(-1.0, min(1.0, current + delta))

            # Update relationship deltas
            for rel_id, delta in inf.affected_relationships.items():
                current = state.relationship_deltas.get(rel_id, 0.0)
                state.relationship_deltas[rel_id] = max(-1.0, min(1.0, current + delta))

            # Track special influences
            if inf.influence_type == InfluenceType.INSIGHT:
                state.insight_count += 1
            elif inf.influence_type == InfluenceType.HEALING:
                state.healing_progress = min(1.0, state.healing_progress + 0.1)
            elif inf.influence_type == InfluenceType.SHADOW_WORK:
                state.shadow_integration = min(1.0, state.shadow_integration + 0.05)

    def record_waking_experience(
        self,
        resident_id: str,
        experience_type: str,
        description: str,
        emotional_impact: Dict[str, float],
        people_involved: Optional[List[str]] = None,
    ) -> WakingExperience:
        """
        Record a waking experience that may appear in dreams.

        Args:
            resident_id: Who had the experience
            experience_type: Type of experience
            description: What happened
            emotional_impact: Emotional intensity by emotion
            people_involved: Other people in the experience

        Returns:
            Created WakingExperience
        """
        state = self.get_or_create_state(resident_id)

        # Determine which symbols this might evoke
        symbols_evoked = self._experience_to_symbols(
            experience_type, emotional_impact
        )

        experience = WakingExperience(
            experience_id=f"exp_{datetime.now().timestamp()}_{random.randint(1000, 9999)}",
            resident_id=resident_id,
            experience_type=experience_type,
            description=description,
            emotional_impact=emotional_impact,
            people_involved=people_involved or [],
            symbols_evoked=symbols_evoked,
            timestamp=datetime.now(),
        )

        state.pending_experiences.append(experience)
        return experience

    def _experience_to_symbols(
        self,
        experience_type: str,
        emotional_impact: Dict[str, float]
    ) -> List[str]:
        """Map waking experience to potential dream symbols."""
        symbols = []

        # High fear -> fear symbols
        if emotional_impact.get("fear", 0) > 0.5:
            symbols.extend(["shadow_figure", "falling", "teeth_falling"])

        # High grief -> loss symbols
        if emotional_impact.get("grief", 0) > 0.5:
            symbols.extend(["empty_chair", "fading_photo", "withered_flower"])

        # High joy -> hope symbols
        if emotional_impact.get("joy", 0) > 0.5:
            symbols.extend(["sunrise", "rainbow", "blooming_flower"])

        # High anxiety -> trapped symbols
        if emotional_impact.get("anxiety", 0) > 0.5:
            symbols.extend(["cage", "locked_door", "maze"])

        # High connection -> connection symbols
        if emotional_impact.get("connection", 0) > 0.5:
            symbols.extend(["bridge", "handshake", "shared_meal"])

        return symbols[:5]

    def get_active_influences(
        self,
        resident_id: str
    ) -> List[DreamInfluence]:
        """Get all active (non-expired) influences for a resident."""
        state = self.get_or_create_state(resident_id)

        # Filter to active and update list
        active = [inf for inf in state.active_influences if inf.is_active()]
        state.active_influences = active

        return active

    def get_current_trait_modifiers(
        self,
        resident_id: str
    ) -> Dict[str, float]:
        """
        Get current trait modifiers from all active influences.

        Returns dict of trait -> modifier value.
        """
        influences = self.get_active_influences(resident_id)
        modifiers: Dict[str, float] = {}

        for inf in influences:
            current_strength = inf.remaining_strength()
            for trait, base_delta in inf.affected_traits.items():
                adjusted_delta = base_delta * (current_strength / inf.strength.value)
                modifiers[trait] = modifiers.get(trait, 0.0) + adjusted_delta

        # Clamp to reasonable bounds
        return {k: max(-0.5, min(0.5, v)) for k, v in modifiers.items()}

    def get_pending_dream_content(
        self,
        resident_id: str
    ) -> Dict[str, Any]:
        """
        Get content that should appear in next dream.

        Returns symbols and themes from unprocessed experiences.
        """
        state = self.get_or_create_state(resident_id)

        # Filter unprocessed experiences
        unprocessed = [
            exp for exp in state.pending_experiences
            if not exp.processed_in_dream
        ]

        if not unprocessed:
            return {"symbols": [], "emotions": {}, "people": []}

        # Aggregate
        all_symbols = []
        all_emotions: Dict[str, float] = {}
        all_people = set()

        for exp in unprocessed:
            # Weight by probability of appearing
            prob = exp.dream_probability()
            if random.random() < prob:
                all_symbols.extend(exp.symbols_evoked)
                for emotion, intensity in exp.emotional_impact.items():
                    all_emotions[emotion] = max(
                        all_emotions.get(emotion, 0),
                        intensity * prob
                    )
                all_people.update(exp.people_involved)

        return {
            "symbols": list(set(all_symbols))[:10],
            "emotions": all_emotions,
            "people": list(all_people)[:5],
        }

    def process_collective_dream(
        self,
        dream: Dict[str, Any],
        dreamer_ids: List[str]
    ) -> List[DreamInfluence]:
        """
        Process a shared/collective dream that affects multiple residents.

        Collective dreams have wider but weaker individual impact,
        and also shift the neighborhood zeitgeist.
        """
        all_influences = []

        # Generate base influences
        for resident_id in dreamer_ids:
            influences = self.process_dream(dream, resident_id)

            # Mark as collective and reduce individual strength
            for inf in influences:
                inf.is_collective = True
                # Collective dreams spread influence thinner
                for trait in inf.affected_traits:
                    inf.affected_traits[trait] *= 0.5

            all_influences.extend(influences)

        # Update neighborhood mood
        self._update_neighborhood_mood(dream, len(dreamer_ids))

        return all_influences

    def _update_neighborhood_mood(
        self,
        dream: Dict[str, Any],
        participant_count: int
    ):
        """Update neighborhood-wide mood from collective dream."""
        # Scale by participation
        scale = min(1.0, participant_count / 10)  # Max effect at 10 dreamers

        mood = dream.get("mood", "mysterious")

        # Map dream moods to neighborhood moods
        mood_effects = {
            "nightmare": {"anxiety": 0.1, "tension": 0.1},
            "melancholy": {"healing": 0.05, "connection": 0.05},
            "wonder": {"creativity": 0.1, "hope": 0.1},
            "peaceful": {"healing": 0.1, "anxiety": -0.1},
            "tense": {"tension": 0.1, "anxiety": 0.05},
            "ecstatic": {"hope": 0.15, "creativity": 0.1},
            "mysterious": {"creativity": 0.05},
            "transformative": {"healing": 0.1, "hope": 0.1},
        }

        effects = mood_effects.get(mood, {})

        for mood_type, delta in effects.items():
            current = self.neighborhood_mood.get(mood_type, 0.0)
            self.neighborhood_mood[mood_type] = max(
                -1.0, min(1.0, current + delta * scale)
            )

    def get_neighborhood_mood(self) -> Dict[str, float]:
        """Get current neighborhood emotional state."""
        return self.neighborhood_mood.copy()

    def decay_neighborhood_mood(self, hours: float = 1.0):
        """Apply time decay to neighborhood mood (call periodically)."""
        decay_rate = 0.95 ** hours  # 5% decay per hour

        for mood_type in self.neighborhood_mood:
            self.neighborhood_mood[mood_type] *= decay_rate

    def get_influence_summary(self, resident_id: str) -> Dict[str, Any]:
        """Get a summary of a resident's dream influence state."""
        state = self.get_or_create_state(resident_id)
        active = self.get_active_influences(resident_id)

        return {
            "resident_id": resident_id,
            "active_influence_count": len(active),
            "pending_experiences": len([
                e for e in state.pending_experiences
                if not e.processed_in_dream
            ]),
            "cumulative_mood": state.cumulative_mood_shift,
            "relationship_changes": state.relationship_deltas,
            "insight_count": state.insight_count,
            "healing_progress": round(state.healing_progress, 2),
            "shadow_integration": round(state.shadow_integration, 2),
            "active_influences": [
                {
                    "type": inf.influence_type.value,
                    "strength": inf.strength.value,
                    "remaining": round(inf.remaining_strength(), 2),
                    "description": inf.description,
                    "expires_in_hours": round(
                        (inf.expires_at - datetime.now()).total_seconds() / 3600, 1
                    ),
                }
                for inf in active
            ],
        }

    def save_state(self, path: str):
        """Save complete system state to JSON."""
        state = {
            "neighborhood_mood": self.neighborhood_mood,
            "resident_states": {
                rid: {
                    "cumulative_mood_shift": s.cumulative_mood_shift,
                    "relationship_deltas": s.relationship_deltas,
                    "insight_count": s.insight_count,
                    "healing_progress": s.healing_progress,
                    "shadow_integration": s.shadow_integration,
                }
                for rid, s in self.resident_states.items()
            },
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """Load system state from JSON."""
        with open(path, "r") as f:
            state = json.load(f)

        self.neighborhood_mood = state.get("neighborhood_mood", self.neighborhood_mood)

        for rid, data in state.get("resident_states", {}).items():
            s = self.get_or_create_state(rid)
            s.cumulative_mood_shift = data.get("cumulative_mood_shift", {})
            s.relationship_deltas = data.get("relationship_deltas", {})
            s.insight_count = data.get("insight_count", 0)
            s.healing_progress = data.get("healing_progress", 0.0)
            s.shadow_integration = data.get("shadow_integration", 0.0)


# Convenience function
def create_dream_influence_system() -> DreamInfluenceSystem:
    """Create a new Dream Influence System instance."""
    return DreamInfluenceSystem()


if __name__ == "__main__":
    print("=" * 70)
    print("DREAM INFLUENCE SYSTEM DEMO")
    print("=" * 70)

    system = create_dream_influence_system()
    resident = "maya_001"

    # Record some waking experiences
    print("\n--- Recording Waking Experiences ---")

    exp1 = system.record_waking_experience(
        resident_id=resident,
        experience_type="conversation",
        description="Difficult conversation with neighbor about boundaries",
        emotional_impact={"anxiety": 0.7, "tension": 0.6, "fear": 0.3},
        people_involved=["neighbor_jorge"],
    )
    print(f"Recorded: {exp1.description}")
    print(f"  Dream probability: {exp1.dream_probability():.1%}")
    print(f"  Symbols evoked: {exp1.symbols_evoked}")

    exp2 = system.record_waking_experience(
        resident_id=resident,
        experience_type="event",
        description="Found old family photo while cleaning",
        emotional_impact={"grief": 0.6, "connection": 0.5, "joy": 0.3},
        people_involved=["grandmother_deceased"],
    )
    print(f"\nRecorded: {exp2.description}")
    print(f"  Dream probability: {exp2.dream_probability():.1%}")
    print(f"  Symbols evoked: {exp2.symbols_evoked}")

    # Get pending dream content
    print("\n--- Pending Dream Content ---")
    pending = system.get_pending_dream_content(resident)
    print(f"Symbols to appear: {pending['symbols']}")
    print(f"Emotions to process: {pending['emotions']}")

    # Simulate a dream
    print("\n--- Processing Dream ---")
    dream = {
        "id": "dream_001",
        "symbols_used": ["shadow_figure", "fading_photo", "bridge", "sunrise"],
        "archetype": "shadow_integration",
        "mood": "transformative",
    }

    influences = system.process_dream(dream, resident)
    print(f"Generated {len(influences)} influences:")

    for inf in influences:
        print(f"\n  {inf.influence_type.value.upper()}")
        print(f"    Strength: {inf.strength.name}")
        print(f"    Description: {inf.description}")
        print(f"    Traits affected: {inf.affected_traits}")
        print(f"    Duration: {inf.duration_hours:.0f} hours")

    # Get trait modifiers
    print("\n--- Current Trait Modifiers ---")
    modifiers = system.get_current_trait_modifiers(resident)
    for trait, mod in modifiers.items():
        direction = "+" if mod > 0 else "-"
        print(f"  {trait}: {direction}{abs(mod):.3f}")

    # Simulate collective dream
    print("\n--- Processing Collective Dream ---")
    collective_dream = {
        "id": "collective_001",
        "symbols_used": ["great_flood", "ark", "rainbow"],
        "archetype": "death_rebirth",
        "mood": "transformative",
    }

    collective_influences = system.process_collective_dream(
        collective_dream,
        ["maya_001", "jorge_002", "elena_003", "carlos_004"]
    )
    print(f"Generated {len(collective_influences)} collective influences")

    # Check neighborhood mood
    print("\n--- Neighborhood Mood ---")
    mood = system.get_neighborhood_mood()
    for m, v in mood.items():
        if abs(v) > 0.01:
            print(f"  {m}: {v:+.2f}")

    # Get full summary
    print("\n--- Resident Influence Summary ---")
    summary = system.get_influence_summary(resident)
    print(f"Active influences: {summary['active_influence_count']}")
    print(f"Pending experiences: {summary['pending_experiences']}")
    print(f"Insights gained: {summary['insight_count']}")
    print(f"Healing progress: {summary['healing_progress']:.0%}")
    print(f"Shadow integration: {summary['shadow_integration']:.0%}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
