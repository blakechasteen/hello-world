"""
Resident - Neighborhood Resident Model

Each resident has:
- Personality traits (Big Five)
- Emotional state (dynamic)
- Relationships with other residents
- Memory of interactions
- Daily routines and preferences

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import random


class PersonalityTrait(Enum):
    """Big Five personality traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class RelationshipType(Enum):
    """Types of relationships between residents."""
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    NEIGHBOR = "neighbor"
    FRIEND = "friend"
    CLOSE_FRIEND = "close_friend"
    FAMILY = "family"
    ROMANTIC = "romantic"
    RIVAL = "rival"


@dataclass
class Personality:
    """Big Five personality model."""
    openness: float = 0.5          # Curiosity, creativity
    conscientiousness: float = 0.5  # Organization, dependability
    extraversion: float = 0.5       # Sociability, energy
    agreeableness: float = 0.5      # Cooperation, trust
    neuroticism: float = 0.5        # Emotional instability

    def get_trait(self, trait: PersonalityTrait) -> float:
        return getattr(self, trait.value)

    def describe(self) -> str:
        """Get personality description."""
        traits = []
        if self.openness > 0.7:
            traits.append("creative and curious")
        elif self.openness < 0.3:
            traits.append("practical and traditional")

        if self.extraversion > 0.7:
            traits.append("outgoing and energetic")
        elif self.extraversion < 0.3:
            traits.append("reserved and introspective")

        if self.agreeableness > 0.7:
            traits.append("warm and compassionate")
        elif self.agreeableness < 0.3:
            traits.append("direct and competitive")

        if self.neuroticism > 0.7:
            traits.append("sensitive and emotional")
        elif self.neuroticism < 0.3:
            traits.append("calm and resilient")

        if self.conscientiousness > 0.7:
            traits.append("organized and reliable")
        elif self.conscientiousness < 0.3:
            traits.append("spontaneous and flexible")

        return ", ".join(traits) if traits else "balanced"


@dataclass
class EmotionalState:
    """Current emotional state of a resident."""
    happiness: float = 0.5
    anxiety: float = 0.2
    energy: float = 0.6
    social_need: float = 0.5
    loneliness: float = 0.2

    def mood_description(self) -> str:
        """Get mood description."""
        if self.happiness > 0.7 and self.energy > 0.6:
            return "cheerful and energetic"
        elif self.happiness > 0.7:
            return "content"
        elif self.anxiety > 0.6:
            return "anxious"
        elif self.loneliness > 0.6:
            return "lonely"
        elif self.energy < 0.3:
            return "tired"
        elif self.happiness < 0.3:
            return "melancholy"
        else:
            return "neutral"

    def needs_social(self) -> bool:
        """Check if resident needs social interaction."""
        return self.social_need > 0.6 or self.loneliness > 0.5


@dataclass
class Relationship:
    """Relationship between two residents."""
    target_id: str
    relationship_type: RelationshipType = RelationshipType.STRANGER
    affection: float = 0.5      # How much they like each other
    trust: float = 0.5          # How much they trust each other
    familiarity: float = 0.0    # How well they know each other
    last_interaction: Optional[datetime] = None
    shared_memories: List[str] = field(default_factory=list)
    topics_discussed: List[str] = field(default_factory=list)

    def quality(self) -> float:
        """Overall relationship quality."""
        return (self.affection + self.trust + self.familiarity) / 3

    def can_discuss_deep_topics(self) -> bool:
        """Check if relationship allows deep conversations."""
        return self.trust > 0.6 and self.familiarity > 0.5


@dataclass
class Memory:
    """A memory of an interaction or event."""
    timestamp: datetime
    event_type: str
    description: str
    participants: List[str]
    emotional_impact: Dict[str, float]
    location: str
    importance: float = 0.5


@dataclass
class DailySchedule:
    """Typical daily schedule preferences."""
    wake_time: int = 7           # Hour (24h format)
    sleep_time: int = 22         # Hour (24h format)
    work_start: int = 9
    work_end: int = 17
    preferred_social_times: List[int] = field(default_factory=lambda: [12, 18, 20])
    hobbies: List[str] = field(default_factory=list)


class Resident:
    """A neighborhood resident with full life simulation."""

    def __init__(
        self,
        resident_id: str,
        name: str,
        age: int,
        occupation: str,
        personality: Optional[Personality] = None,
        backstory: str = "",
    ):
        self.resident_id = resident_id
        self.name = name
        self.age = age
        self.occupation = occupation
        self.backstory = backstory

        # Generate personality if not provided
        self.personality = personality or self._generate_personality()

        # Dynamic state
        self.emotional_state = EmotionalState()
        self.current_location = "home"
        self.current_activity = "resting"

        # Relationships (resident_id -> Relationship)
        self.relationships: Dict[str, Relationship] = {}

        # Memories
        self.memories: List[Memory] = []
        self.recent_conversations: List[Dict] = []

        # Schedule
        self.schedule = DailySchedule()
        self._assign_hobbies()

        # Conversation style
        self.speaking_style = self._generate_speaking_style()
        self.favorite_topics = self._generate_favorite_topics()

    def _generate_personality(self) -> Personality:
        """Generate random but coherent personality."""
        return Personality(
            openness=random.uniform(0.3, 0.9),
            conscientiousness=random.uniform(0.3, 0.9),
            extraversion=random.uniform(0.2, 0.9),
            agreeableness=random.uniform(0.3, 0.9),
            neuroticism=random.uniform(0.2, 0.7),
        )

    def _assign_hobbies(self):
        """Assign hobbies based on personality."""
        all_hobbies = {
            "high_openness": ["painting", "writing", "music", "philosophy", "travel"],
            "high_conscientiousness": ["gardening", "cooking", "organizing", "crafts"],
            "high_extraversion": ["dancing", "sports", "parties", "volunteering"],
            "high_agreeableness": ["helping neighbors", "community events", "charity"],
            "low_extraversion": ["reading", "meditation", "puzzles", "nature walks"],
        }

        hobbies = []
        if self.personality.openness > 0.6:
            hobbies.extend(random.sample(all_hobbies["high_openness"], 2))
        if self.personality.conscientiousness > 0.6:
            hobbies.extend(random.sample(all_hobbies["high_conscientiousness"], 1))
        if self.personality.extraversion > 0.6:
            hobbies.extend(random.sample(all_hobbies["high_extraversion"], 1))
        elif self.personality.extraversion < 0.4:
            hobbies.extend(random.sample(all_hobbies["low_extraversion"], 2))
        if self.personality.agreeableness > 0.6:
            hobbies.extend(random.sample(all_hobbies["high_agreeableness"], 1))

        self.schedule.hobbies = list(set(hobbies))[:4]

    def _generate_speaking_style(self) -> Dict[str, any]:
        """Generate speaking style based on personality."""
        style = {
            "verbosity": "average",
            "formality": "casual",
            "emotionality": "moderate",
            "humor": "occasional",
        }

        if self.personality.extraversion > 0.7:
            style["verbosity"] = "talkative"
            style["humor"] = "frequent"
        elif self.personality.extraversion < 0.3:
            style["verbosity"] = "brief"

        if self.personality.openness > 0.7:
            style["formality"] = "informal"
        elif self.personality.conscientiousness > 0.7:
            style["formality"] = "polite"

        if self.personality.neuroticism > 0.6:
            style["emotionality"] = "expressive"
        elif self.personality.neuroticism < 0.3:
            style["emotionality"] = "reserved"

        return style

    def _generate_favorite_topics(self) -> List[str]:
        """Generate favorite conversation topics."""
        topics = []

        # Based on occupation
        occupation_topics = {
            "teacher": ["education", "students", "learning"],
            "artist": ["art", "creativity", "inspiration"],
            "gardener": ["plants", "seasons", "nature"],
            "retired": ["memories", "family", "neighborhood changes"],
            "chef": ["food", "recipes", "restaurants"],
            "writer": ["stories", "books", "ideas"],
            "nurse": ["health", "community", "helping others"],
            "musician": ["music", "concerts", "instruments"],
        }
        topics.extend(occupation_topics.get(self.occupation.lower(), ["work"]))

        # Based on hobbies
        topics.extend(self.schedule.hobbies)

        # Universal topics
        topics.extend(["weather", "neighborhood", "dreams"])

        return list(set(topics))

    def get_relationship(self, other_id: str) -> Relationship:
        """Get or create relationship with another resident."""
        if other_id not in self.relationships:
            self.relationships[other_id] = Relationship(target_id=other_id)
        return self.relationships[other_id]

    def update_relationship(
        self,
        other_id: str,
        affection_delta: float = 0,
        trust_delta: float = 0,
        familiarity_delta: float = 0,
    ):
        """Update relationship with another resident."""
        rel = self.get_relationship(other_id)
        rel.affection = max(0, min(1, rel.affection + affection_delta))
        rel.trust = max(0, min(1, rel.trust + trust_delta))
        rel.familiarity = max(0, min(1, rel.familiarity + familiarity_delta))
        rel.last_interaction = datetime.now()

        # Update relationship type based on metrics
        quality = rel.quality()
        if quality > 0.8:
            rel.relationship_type = RelationshipType.CLOSE_FRIEND
        elif quality > 0.6:
            rel.relationship_type = RelationshipType.FRIEND
        elif quality > 0.4:
            rel.relationship_type = RelationshipType.NEIGHBOR
        elif quality > 0.2:
            rel.relationship_type = RelationshipType.ACQUAINTANCE

    def add_memory(
        self,
        event_type: str,
        description: str,
        participants: List[str],
        emotional_impact: Dict[str, float],
        location: str,
        importance: float = 0.5,
    ):
        """Add a memory of an event."""
        memory = Memory(
            timestamp=datetime.now(),
            event_type=event_type,
            description=description,
            participants=participants,
            emotional_impact=emotional_impact,
            location=location,
            importance=importance,
        )
        self.memories.append(memory)

        # Update emotional state based on impact
        for emotion, delta in emotional_impact.items():
            if hasattr(self.emotional_state, emotion):
                current = getattr(self.emotional_state, emotion)
                setattr(self.emotional_state, emotion, max(0, min(1, current + delta * 0.3)))

    def update_emotional_state(self, hours_passed: float = 1.0):
        """Update emotional state over time (decay toward baseline)."""
        decay_rate = 0.05 * hours_passed

        # Emotions decay toward baseline
        self.emotional_state.happiness = self._decay_toward(self.emotional_state.happiness, 0.5, decay_rate)
        self.emotional_state.anxiety = self._decay_toward(self.emotional_state.anxiety, 0.2, decay_rate)
        self.emotional_state.energy = self._decay_toward(self.emotional_state.energy, 0.6, decay_rate)

        # Social need increases over time without interaction
        self.emotional_state.social_need = min(1.0, self.emotional_state.social_need + 0.02 * hours_passed)
        self.emotional_state.loneliness = min(1.0, self.emotional_state.loneliness + 0.01 * hours_passed)

    def _decay_toward(self, current: float, target: float, rate: float) -> float:
        """Decay a value toward a target."""
        if current > target:
            return max(target, current - rate)
        else:
            return min(target, current + rate)

    def satisfy_social_need(self, amount: float = 0.3):
        """Satisfy social need after interaction."""
        self.emotional_state.social_need = max(0, self.emotional_state.social_need - amount)
        self.emotional_state.loneliness = max(0, self.emotional_state.loneliness - amount * 0.5)

    def wants_to_talk(self) -> bool:
        """Check if resident wants to engage in conversation."""
        # Extraverts more likely to want to talk
        base_chance = 0.3 + self.personality.extraversion * 0.4

        # Social need increases desire
        base_chance += self.emotional_state.social_need * 0.3

        # Low energy decreases desire
        if self.emotional_state.energy < 0.3:
            base_chance *= 0.5

        # High anxiety might decrease desire (unless high agreeableness)
        if self.emotional_state.anxiety > 0.6 and self.personality.agreeableness < 0.5:
            base_chance *= 0.7

        return random.random() < base_chance

    def get_greeting(self, other_name: str, relationship: Relationship) -> str:
        """Generate a greeting based on personality and relationship."""
        greetings_by_familiarity = {
            "stranger": [
                f"Oh, hello there.",
                f"Hi, I don't think we've met.",
                f"Good day.",
            ],
            "acquaintance": [
                f"Hello, {other_name}.",
                f"Oh, hi {other_name}!",
                f"Hey there, {other_name}.",
            ],
            "friend": [
                f"Hey {other_name}! Great to see you!",
                f"{other_name}! How are you?",
                f"There you are, {other_name}!",
            ],
            "close": [
                f"{other_name}! I was just thinking about you!",
                f"My favorite person! Hi {other_name}!",
                f"Oh {other_name}, I'm so glad to see you!",
            ],
        }

        if relationship.familiarity < 0.2:
            level = "stranger"
        elif relationship.familiarity < 0.5:
            level = "acquaintance"
        elif relationship.familiarity < 0.8:
            level = "friend"
        else:
            level = "close"

        return random.choice(greetings_by_familiarity[level])

    def describe(self) -> str:
        """Get full description of resident."""
        return f"""{self.name} ({self.age}, {self.occupation})
Personality: {self.personality.describe()}
Current mood: {self.emotional_state.mood_description()}
Hobbies: {', '.join(self.schedule.hobbies)}
Favorite topics: {', '.join(self.favorite_topics[:5])}"""

    def __repr__(self):
        return f"Resident({self.name}, {self.occupation})"


# Pre-defined residents for the neighborhood
def create_neighborhood_residents() -> Dict[str, Resident]:
    """Create the default neighborhood residents."""
    residents = {}

    # Maya - The Creative Soul
    maya = Resident(
        resident_id="maya_001",
        name="Maya",
        age=34,
        occupation="Artist",
        personality=Personality(
            openness=0.9,
            conscientiousness=0.4,
            extraversion=0.6,
            agreeableness=0.8,
            neuroticism=0.5,
        ),
        backstory="Maya moved to the neighborhood to find inspiration. She paints murals and teaches art to children.",
    )
    maya.schedule.hobbies = ["painting", "meditation", "nature walks", "music"]
    maya.favorite_topics = ["art", "dreams", "nature", "creativity", "emotions"]
    residents["maya_001"] = maya

    # Jorge - The Community Gardener
    jorge = Resident(
        resident_id="jorge_002",
        name="Jorge",
        age=58,
        occupation="Retired Teacher",
        personality=Personality(
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.5,
            agreeableness=0.9,
            neuroticism=0.3,
        ),
        backstory="Jorge retired from teaching and now tends the community garden. He loves sharing stories and vegetables.",
    )
    jorge.schedule.hobbies = ["gardening", "cooking", "reading", "helping neighbors"]
    jorge.favorite_topics = ["plants", "seasons", "neighborhood", "family", "memories"]
    residents["jorge_002"] = jorge

    # Elena - The Young Professional
    elena = Resident(
        resident_id="elena_003",
        name="Elena",
        age=28,
        occupation="Software Developer",
        personality=Personality(
            openness=0.6,
            conscientiousness=0.7,
            extraversion=0.4,
            agreeableness=0.6,
            neuroticism=0.5,
        ),
        backstory="Elena works from home and is still finding her place in the community. She's curious but a bit reserved.",
    )
    elena.schedule.hobbies = ["reading", "puzzles", "music", "cooking"]
    elena.favorite_topics = ["technology", "books", "ideas", "work-life balance"]
    residents["elena_003"] = elena

    # Carlos - The Storyteller
    carlos = Resident(
        resident_id="carlos_004",
        name="Carlos",
        age=67,
        occupation="Retired Journalist",
        personality=Personality(
            openness=0.8,
            conscientiousness=0.5,
            extraversion=0.8,
            agreeableness=0.7,
            neuroticism=0.4,
        ),
        backstory="Carlos has lived in the neighborhood for 40 years. He knows everyone and loves to share stories.",
    )
    carlos.schedule.hobbies = ["writing", "storytelling", "coffee", "neighborhood walks"]
    carlos.favorite_topics = ["neighborhood history", "stories", "news", "people"]
    residents["carlos_004"] = carlos

    # Sofia - The Healer
    sofia = Resident(
        resident_id="sofia_005",
        name="Sofia",
        age=45,
        occupation="Nurse",
        personality=Personality(
            openness=0.6,
            conscientiousness=0.9,
            extraversion=0.5,
            agreeableness=0.9,
            neuroticism=0.4,
        ),
        backstory="Sofia works at the local clinic. She's the one neighbors call when they need help or advice.",
    )
    sofia.schedule.hobbies = ["yoga", "cooking", "volunteering", "gardening"]
    sofia.favorite_topics = ["health", "community", "family", "helping others"]
    residents["sofia_005"] = sofia

    # Diego - The Young Dreamer
    diego = Resident(
        resident_id="diego_006",
        name="Diego",
        age=22,
        occupation="Music Student",
        personality=Personality(
            openness=0.9,
            conscientiousness=0.3,
            extraversion=0.7,
            agreeableness=0.6,
            neuroticism=0.6,
        ),
        backstory="Diego is studying music at the conservatory. He's full of dreams and uncertainties about the future.",
    )
    diego.schedule.hobbies = ["music", "songwriting", "skateboarding", "parties"]
    diego.favorite_topics = ["music", "dreams", "future", "art", "love"]
    residents["diego_006"] = diego

    # Initialize relationships (everyone knows each other somewhat)
    for r1_id, r1 in residents.items():
        for r2_id, r2 in residents.items():
            if r1_id != r2_id:
                # Random initial familiarity
                familiarity = random.uniform(0.2, 0.6)
                affection = random.uniform(0.4, 0.7)
                r1.relationships[r2_id] = Relationship(
                    target_id=r2_id,
                    relationship_type=RelationshipType.NEIGHBOR,
                    affection=affection,
                    trust=random.uniform(0.4, 0.6),
                    familiarity=familiarity,
                )

    # Special relationships
    # Maya and Jorge are good friends (she paints in his garden)
    residents["maya_001"].update_relationship("jorge_002", 0.3, 0.3, 0.3)
    residents["jorge_002"].update_relationship("maya_001", 0.3, 0.3, 0.3)

    # Carlos knows everyone well
    for r_id in residents:
        if r_id != "carlos_004":
            residents["carlos_004"].update_relationship(r_id, 0.2, 0.2, 0.4)

    # Diego and Elena have a budding friendship
    residents["diego_006"].update_relationship("elena_003", 0.2, 0.1, 0.2)
    residents["elena_003"].update_relationship("diego_006", 0.2, 0.1, 0.2)

    return residents


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("RESIDENT SYSTEM DEMO")
    print("=" * 60)

    residents = create_neighborhood_residents()

    print("\n--- NEIGHBORHOOD RESIDENTS ---\n")
    for r_id, resident in residents.items():
        print(resident.describe())
        print()

    print("\n--- RELATIONSHIPS ---\n")
    maya = residents["maya_001"]
    for r_id, rel in maya.relationships.items():
        other = residents[r_id]
        print(f"Maya -> {other.name}: {rel.relationship_type.value} (quality: {rel.quality():.2f})")

    print("\n--- SAMPLE INTERACTION ---\n")
    # Simulate an interaction
    jorge = residents["jorge_002"]
    rel = maya.get_relationship("jorge_002")

    print(f"Maya sees Jorge in the garden...")
    print(f"Maya: \"{maya.get_greeting('Jorge', rel)}\"")

    # Update after interaction
    maya.update_relationship("jorge_002", affection_delta=0.05, familiarity_delta=0.02)
    maya.satisfy_social_need(0.3)
    maya.add_memory(
        event_type="conversation",
        description="Chatted with Jorge about the garden",
        participants=["jorge_002"],
        emotional_impact={"happiness": 0.1},
        location="community_garden",
    )

    print(f"\nAfter conversation:")
    print(f"  Maya's mood: {maya.emotional_state.mood_description()}")
    print(f"  Relationship quality: {rel.quality():.2f}")
