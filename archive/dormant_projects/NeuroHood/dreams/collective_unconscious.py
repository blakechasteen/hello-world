"""
Collective Unconscious Layer - Phase 5 Week 1

A shared cultural repository of enriched symbols serving as the
neighborhood's collective dream vocabulary. Implements:

1. Shared Symbol Repository: 114+ enriched symbols with literary references
2. Archetypal Pattern Detection: Jung-inspired pattern recognition
3. Neighborhood Zeitgeist: Collective emotional temperature
4. Symbol Evolution: Symbols gain/lose prominence based on usage

Based on Carl Jung's concept of the collective unconscious - a repository
of shared human experiences and archetypal patterns that manifest in dreams.

Author: Phase 5 Implementation
Date: November 2025
"""

import json
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict


class ArchetypalPattern(Enum):
    """Jungian archetypal patterns that emerge in dreams."""
    HEROS_JOURNEY = "heros_journey"  # Departure, initiation, return
    SHADOW_INTEGRATION = "shadow_integration"  # Confronting hidden self
    ANIMA_ANIMUS = "anima_animus"  # Masculine/feminine integration
    DEATH_REBIRTH = "death_rebirth"  # Transformation through ending
    TRICKSTER = "trickster"  # Chaos, change, boundary-crossing
    GREAT_MOTHER = "great_mother"  # Nurturing, devouring, creation
    WISE_OLD_ONE = "wise_old_one"  # Guidance, wisdom, knowledge
    DIVINE_CHILD = "divine_child"  # New beginnings, potential, hope
    SELF_INDIVIDUATION = "self_individuation"  # Wholeness, integration


@dataclass
class EmotionalEssence:
    """Emotional state of the neighborhood."""
    joy: float = 0.0
    fear: float = 0.0
    hope: float = 0.0
    grief: float = 0.0
    anxiety: float = 0.0
    connection: float = 0.0
    conflict: float = 0.0
    transformation: float = 0.0

    def dominant_emotion(self) -> str:
        """Return the most prominent emotion."""
        emotions = {
            "joy": self.joy,
            "fear": self.fear,
            "hope": self.hope,
            "grief": self.grief,
            "anxiety": self.anxiety,
            "connection": self.connection,
            "conflict": self.conflict,
            "transformation": self.transformation,
        }
        return max(emotions, key=emotions.get)

    def emotional_temperature(self) -> float:
        """Calculate overall emotional intensity (0-1)."""
        return min(1.0, (
            abs(self.joy) + abs(self.fear) + abs(self.hope) +
            abs(self.grief) + abs(self.anxiety) + abs(self.connection) +
            abs(self.conflict) + abs(self.transformation)
        ) / 4.0)


@dataclass
class SymbolUsage:
    """Tracks how a symbol is used in the neighborhood."""
    symbol_id: str
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    residents_using: Set[str] = field(default_factory=set)
    dream_appearances: int = 0
    emotional_context: Dict[str, float] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

    def heat_score(self) -> float:
        """Calculate current 'heat' of this symbol (0-1)."""
        if not self.last_accessed:
            return 0.0

        # Time decay (5% per hour)
        hours_since = (datetime.now() - self.last_accessed).total_seconds() / 3600
        decay = 0.95 ** hours_since

        # Base score from access and dream appearances
        base = min(1.0, (self.access_count * 0.1 + self.dream_appearances * 0.3))

        # Resident diversity bonus
        resident_factor = min(1.0, len(self.residents_using) * 0.2)

        return base * decay * (1 + resident_factor * 0.3)


@dataclass
class DetectedArchetype:
    """An archetypal pattern detected in neighborhood dreams."""
    pattern: ArchetypalPattern
    confidence: float
    participating_residents: List[str]
    symbols_involved: List[str]
    detection_time: datetime
    narrative_summary: str


@dataclass
class CollectiveUnconsciousState:
    """Complete state of the collective unconscious."""
    symbols: Dict[str, Dict]  # symbol_id -> enriched symbol data
    symbol_usage: Dict[str, SymbolUsage]  # symbol_id -> usage stats
    detected_archetypes: List[DetectedArchetype]
    neighborhood_zeitgeist: EmotionalEssence
    active_themes: List[str]
    symbol_connections: Dict[str, Set[str]]  # symbol_id -> related symbols
    last_updated: datetime


class CollectiveUnconscious:
    """
    The neighborhood's collective unconscious - a shared repository
    of symbols, archetypes, and emotional patterns.

    Usage:
        collective = CollectiveUnconscious()
        await collective.initialize("symbol_database_full_enriched.json")

        # Get symbols for a dream based on emotional state
        symbols = collective.get_dream_symbols(
            resident_id="resident_001",
            emotional_state={"fear": 0.7, "hope": 0.3},
            count=5
        )

        # Detect archetypal patterns
        archetypes = collective.detect_archetypes(dream_history)
    """

    def __init__(self, symbol_database_path: Optional[str] = None):
        self.symbol_database_path = symbol_database_path
        self.state: Optional[CollectiveUnconsciousState] = None
        self._archetype_patterns = self._init_archetype_patterns()

    async def initialize(self, database_path: Optional[str] = None) -> None:
        """Initialize the collective unconscious with enriched symbols."""
        path = database_path or self.symbol_database_path
        if not path:
            path = Path(__file__).parent / "symbol_database_full_enriched.json"

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        symbols = {s["symbol_id"]: s for s in data["symbols"]}
        symbol_usage = {
            sid: SymbolUsage(symbol_id=sid)
            for sid in symbols
        }

        # Build symbol connections based on emotional resonance
        connections = self._build_symbol_connections(symbols)

        self.state = CollectiveUnconsciousState(
            symbols=symbols,
            symbol_usage=symbol_usage,
            detected_archetypes=[],
            neighborhood_zeitgeist=EmotionalEssence(),
            active_themes=[],
            symbol_connections=connections,
            last_updated=datetime.now()
        )

        print(f"[CollectiveUnconscious] Initialized with {len(symbols)} symbols")

    def _build_symbol_connections(self, symbols: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """Build connections between symbols based on shared emotions."""
        connections = defaultdict(set)

        for sid1, s1 in symbols.items():
            emotions1 = set(s1.get("emotion_tags", []))
            for sid2, s2 in symbols.items():
                if sid1 >= sid2:
                    continue
                emotions2 = set(s2.get("emotion_tags", []))
                # Connect if they share emotions
                if emotions1 & emotions2:
                    connections[sid1].add(sid2)
                    connections[sid2].add(sid1)

        return dict(connections)

    def get_symbol(self, symbol_id: str) -> Optional[Dict]:
        """Get a symbol by ID."""
        if not self.state:
            return None
        return self.state.symbols.get(symbol_id)

    def get_symbols_by_category(self, category: str) -> List[Dict]:
        """Get all symbols in a category."""
        if not self.state:
            return []
        return [
            s for s in self.state.symbols.values()
            if s.get("category") == category
        ]

    def get_symbols_by_emotion(self, emotion: str, min_resonance: float = 0.5) -> List[Dict]:
        """Get symbols that resonate with an emotion."""
        if not self.state:
            return []

        matching = []
        for symbol in self.state.symbols.values():
            emotion_tags = symbol.get("emotion_tags", [])
            if emotion in emotion_tags:
                matching.append(symbol)

        return matching

    def get_dream_symbols(
        self,
        resident_id: str,
        emotional_state: Dict[str, float],
        count: int = 5,
        consciousness_level: float = 0.5,
        existing_symbols: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get symbols for a dream based on emotional state and consciousness level.

        Args:
            resident_id: The resident having the dream
            emotional_state: Dict of emotion -> intensity (0-1)
            count: Number of symbols to return
            consciousness_level: 0 = individual, 1 = universal (collective)
            existing_symbols: Symbols already in the dream (for continuity)

        Returns:
            List of enriched symbol dicts
        """
        if not self.state:
            return []

        # Score each symbol based on emotional match + heat + consciousness
        scored_symbols = []

        for sid, symbol in self.state.symbols.items():
            score = 0.0

            # Emotional match (weighted by intensity)
            emotion_tags = set(symbol.get("emotion_tags", []))
            for emotion, intensity in emotional_state.items():
                if emotion in emotion_tags or self._emotion_matches(emotion, emotion_tags):
                    score += intensity * 0.4

            # Heat score (popular symbols)
            usage = self.state.symbol_usage.get(sid)
            if usage:
                heat = usage.heat_score()
                # Higher consciousness = more influenced by collective heat
                score += heat * consciousness_level * 0.3

            # Connection to existing symbols (narrative continuity)
            if existing_symbols:
                connections = self.state.symbol_connections.get(sid, set())
                for existing in existing_symbols:
                    if existing in connections:
                        score += 0.2

            # Quality bonus
            quality = symbol.get("quality_scores", {}).get("overall", 7.0)
            score += (quality - 7.0) * 0.05

            # Small random factor for variety
            score += random.uniform(0, 0.1)

            scored_symbols.append((sid, symbol, score))

        # Sort by score and take top N
        scored_symbols.sort(key=lambda x: x[2], reverse=True)
        selected = [s[1] for s in scored_symbols[:count]]

        # Record usage
        for symbol in selected:
            self._record_symbol_usage(symbol["symbol_id"], resident_id)

        return selected

    def _emotion_matches(self, emotion: str, tags: Set[str]) -> bool:
        """Check if an emotion matches any tag (including related emotions)."""
        # Emotion similarity map
        related = {
            "fear": {"anxiety", "terror", "dread", "scared"},
            "grief": {"loss", "mourning", "sadness", "sorrow"},
            "hope": {"optimism", "anticipation", "hopeful"},
            "joy": {"happiness", "elation", "delight"},
            "anxiety": {"fear", "worry", "nervous", "stress"},
            "connection": {"love", "bonding", "intimacy", "belonging"},
            "conflict": {"anger", "tension", "struggle"},
            "transformation": {"change", "growth", "metamorphosis"},
        }

        if emotion in tags:
            return True

        related_emotions = related.get(emotion, set())
        return bool(related_emotions & tags)

    def _record_symbol_usage(self, symbol_id: str, resident_id: str) -> None:
        """Record that a symbol was used."""
        if not self.state:
            return

        usage = self.state.symbol_usage.get(symbol_id)
        if usage:
            usage.access_count += 1
            usage.last_accessed = datetime.now()
            usage.residents_using.add(resident_id)

    def record_dream_appearance(
        self,
        symbol_id: str,
        resident_id: str,
        emotional_context: Dict[str, float]
    ) -> None:
        """Record that a symbol appeared in a dream."""
        if not self.state:
            return

        usage = self.state.symbol_usage.get(symbol_id)
        if usage:
            usage.dream_appearances += 1
            usage.last_accessed = datetime.now()
            usage.residents_using.add(resident_id)

            # Update emotional context (running average)
            for emotion, intensity in emotional_context.items():
                current = usage.emotional_context.get(emotion, 0.0)
                usage.emotional_context[emotion] = (current + intensity) / 2

    def detect_archetypes(
        self,
        dream_history: List[Dict],
        min_confidence: float = 0.6
    ) -> List[DetectedArchetype]:
        """
        Detect archetypal patterns in neighborhood dream history.

        Args:
            dream_history: List of recent dreams with symbols and emotions
            min_confidence: Minimum confidence to report a pattern

        Returns:
            List of detected archetypal patterns
        """
        detected = []

        for pattern_name, pattern_def in self._archetype_patterns.items():
            # Check if pattern's required symbols/emotions appear
            confidence, matches = self._match_archetype_pattern(
                dream_history, pattern_def
            )

            if confidence >= min_confidence:
                detected.append(DetectedArchetype(
                    pattern=ArchetypalPattern(pattern_name),
                    confidence=confidence,
                    participating_residents=list(matches["residents"]),
                    symbols_involved=list(matches["symbols"]),
                    detection_time=datetime.now(),
                    narrative_summary=self._generate_archetype_narrative(
                        pattern_name, matches
                    )
                ))

        # Store in state
        if self.state:
            self.state.detected_archetypes = detected

        return detected

    def _init_archetype_patterns(self) -> Dict[str, Dict]:
        """Initialize archetypal pattern definitions."""
        return {
            "heros_journey": {
                "required_emotions": ["fear", "hope", "transformation"],
                "symbol_categories": ["trapped", "conflict", "transformation", "hope"],
                "narrative_structure": ["departure", "initiation", "return"],
                "min_dreams": 3,
            },
            "shadow_integration": {
                "required_emotions": ["fear", "guilt", "conflict"],
                "symbol_categories": ["fear", "guilt", "mystery", "transformation"],
                "narrative_structure": ["confrontation", "acceptance"],
                "min_dreams": 2,
            },
            "death_rebirth": {
                "required_emotions": ["grief", "transformation", "hope"],
                "symbol_categories": ["loss", "transformation", "hope"],
                "narrative_structure": ["ending", "liminal", "new_beginning"],
                "min_dreams": 2,
            },
            "great_mother": {
                "required_emotions": ["connection", "fear", "nurturing"],
                "symbol_categories": ["connection", "fear", "power"],
                "narrative_structure": ["nurturing", "devouring", "separation"],
                "min_dreams": 2,
            },
            "wise_old_one": {
                "required_emotions": ["mystery", "guidance", "transformation"],
                "symbol_categories": ["mystery", "power", "transformation"],
                "narrative_structure": ["encounter", "teaching", "integration"],
                "min_dreams": 2,
            },
            "divine_child": {
                "required_emotions": ["hope", "joy", "innocence"],
                "symbol_categories": ["hope", "transformation", "connection"],
                "narrative_structure": ["emergence", "growth", "potential"],
                "min_dreams": 2,
            },
        }

    def _match_archetype_pattern(
        self,
        dream_history: List[Dict],
        pattern_def: Dict
    ) -> Tuple[float, Dict]:
        """Match dreams against an archetypal pattern."""
        required_emotions = set(pattern_def["required_emotions"])
        required_categories = set(pattern_def["symbol_categories"])
        min_dreams = pattern_def["min_dreams"]

        found_emotions = set()
        found_categories = set()
        matching_residents = set()
        matching_symbols = set()

        for dream in dream_history:
            # Extract emotions from dream
            dream_emotions = set(dream.get("emotions", {}).keys())
            found_emotions |= dream_emotions & required_emotions

            # Extract symbol categories
            for symbol_id in dream.get("symbols", []):
                symbol = self.get_symbol(symbol_id)
                if symbol:
                    category = symbol.get("category")
                    if category in required_categories:
                        found_categories.add(category)
                        matching_symbols.add(symbol_id)

            matching_residents.add(dream.get("resident_id", "unknown"))

        # Calculate confidence
        emotion_match = len(found_emotions) / len(required_emotions) if required_emotions else 0
        category_match = len(found_categories) / len(required_categories) if required_categories else 0
        dream_count_factor = min(1.0, len(dream_history) / min_dreams)

        confidence = (emotion_match * 0.4 + category_match * 0.4 + dream_count_factor * 0.2)

        return confidence, {
            "residents": matching_residents,
            "symbols": matching_symbols,
            "emotions": found_emotions,
            "categories": found_categories,
        }

    def _generate_archetype_narrative(self, pattern_name: str, matches: Dict) -> str:
        """Generate a narrative summary for a detected archetype."""
        narratives = {
            "heros_journey": (
                f"A Hero's Journey is emerging across {len(matches['residents'])} residents. "
                f"Dreams feature symbols of {', '.join(list(matches['categories'])[:3])}, "
                "suggesting a collective movement through departure, trials, and transformation."
            ),
            "shadow_integration": (
                f"Shadow Integration is occurring as {len(matches['residents'])} residents "
                "confront hidden aspects of themselves through symbols of "
                f"{', '.join(list(matches['categories'])[:3])}."
            ),
            "death_rebirth": (
                "A Death-Rebirth pattern emerges - the neighborhood is processing endings "
                f"to make way for new beginnings. {len(matches['symbols'])} symbols "
                "of transformation and hope appear across dreams."
            ),
            "great_mother": (
                f"The Great Mother archetype manifests across {len(matches['residents'])} residents, "
                "with dreams exploring themes of nurturing, power, and connection."
            ),
            "wise_old_one": (
                "A Wise Old One pattern appears - dreams feature guides and teachers. "
                "The neighborhood may be seeking or receiving wisdom during this time."
            ),
            "divine_child": (
                f"The Divine Child emerges with symbols of hope and new potential. "
                f"{len(matches['residents'])} residents share dreams of growth and possibility."
            ),
        }
        return narratives.get(pattern_name, "An archetypal pattern has been detected.")

    def update_zeitgeist(self, dream_history: List[Dict]) -> EmotionalEssence:
        """
        Update the neighborhood's emotional zeitgeist based on recent dreams.

        The zeitgeist represents the collective emotional temperature of the
        neighborhood - a shared mood that influences and is influenced by
        individual dreams.
        """
        if not self.state:
            return EmotionalEssence()

        # Aggregate emotions from recent dreams
        emotion_totals = defaultdict(float)
        dream_count = len(dream_history)

        for dream in dream_history:
            emotions = dream.get("emotions", {})
            for emotion, intensity in emotions.items():
                emotion_totals[emotion] += intensity

        if dream_count > 0:
            # Average the emotions
            self.state.neighborhood_zeitgeist = EmotionalEssence(
                joy=emotion_totals.get("joy", 0) / dream_count,
                fear=emotion_totals.get("fear", 0) / dream_count,
                hope=emotion_totals.get("hope", 0) / dream_count,
                grief=emotion_totals.get("grief", 0) / dream_count,
                anxiety=emotion_totals.get("anxiety", 0) / dream_count,
                connection=emotion_totals.get("connection", 0) / dream_count,
                conflict=emotion_totals.get("conflict", 0) / dream_count,
                transformation=emotion_totals.get("transformation", 0) / dream_count,
            )

        return self.state.neighborhood_zeitgeist

    def get_hot_symbols(self, count: int = 10) -> List[Tuple[str, float]]:
        """Get the hottest (most used) symbols in the neighborhood."""
        if not self.state:
            return []

        scored = [
            (sid, usage.heat_score())
            for sid, usage in self.state.symbol_usage.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:count]

    def get_related_symbols(self, symbol_id: str, count: int = 5) -> List[Dict]:
        """Get symbols related to a given symbol."""
        if not self.state:
            return []

        related_ids = self.state.symbol_connections.get(symbol_id, set())

        # Score by heat
        scored = []
        for rid in related_ids:
            symbol = self.state.symbols.get(rid)
            usage = self.state.symbol_usage.get(rid)
            if symbol and usage:
                scored.append((symbol, usage.heat_score()))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:count]]

    def evolve_symbols(self, dream_history: List[Dict]) -> List[Dict]:
        """
        Evolve symbols based on neighborhood usage.

        Returns list of symbol evolution events.
        """
        if not self.state:
            return []

        events = []
        now = datetime.now()

        for sid, usage in self.state.symbol_usage.items():
            heat = usage.heat_score()

            # Symbols gaining prominence (heat > 0.7)
            if heat > 0.7 and usage.dream_appearances >= 5:
                event = {
                    "type": "rising",
                    "symbol_id": sid,
                    "heat": heat,
                    "timestamp": now.isoformat(),
                    "description": f"Symbol '{sid}' is resonating strongly across the neighborhood"
                }
                events.append(event)
                usage.evolution_history.append(event)

            # Symbols fading (heat < 0.2 after being active)
            elif heat < 0.2 and usage.access_count > 10:
                event = {
                    "type": "fading",
                    "symbol_id": sid,
                    "heat": heat,
                    "timestamp": now.isoformat(),
                    "description": f"Symbol '{sid}' is fading from collective consciousness"
                }
                events.append(event)
                usage.evolution_history.append(event)

        return events

    def get_literary_context(self, symbol_id: str) -> Dict:
        """Get rich literary context for a symbol."""
        symbol = self.get_symbol(symbol_id)
        if not symbol:
            return {}

        refs = symbol.get("literary_references", {})
        archetypes = symbol.get("archetypal_roots", {})

        return {
            "symbol_id": symbol_id,
            "category": symbol.get("category"),
            "description": symbol.get("description"),
            "classical_references": refs.get("classical_mythology", [])[:2],
            "literary_references": refs.get("world_literature", [])[:2],
            "modern_references": refs.get("modern_cinema", [])[:2],
            "archetypal_meaning": archetypes.get("primary", "Unknown"),
            "jungian_patterns": archetypes.get("patterns", []),
        }

    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        if not self.state:
            return {}

        return {
            "symbol_count": len(self.state.symbols),
            "detected_archetypes": [
                {
                    "pattern": a.pattern.value,
                    "confidence": a.confidence,
                    "residents": a.participating_residents,
                    "symbols": a.symbols_involved,
                    "narrative": a.narrative_summary,
                }
                for a in self.state.detected_archetypes
            ],
            "zeitgeist": {
                "dominant_emotion": self.state.neighborhood_zeitgeist.dominant_emotion(),
                "temperature": self.state.neighborhood_zeitgeist.emotional_temperature(),
                "joy": self.state.neighborhood_zeitgeist.joy,
                "fear": self.state.neighborhood_zeitgeist.fear,
                "hope": self.state.neighborhood_zeitgeist.hope,
                "grief": self.state.neighborhood_zeitgeist.grief,
            },
            "hot_symbols": self.get_hot_symbols(5),
            "last_updated": self.state.last_updated.isoformat(),
        }


# Convenience function for quick initialization
async def create_collective_unconscious(
    database_path: Optional[str] = None
) -> CollectiveUnconscious:
    """Create and initialize a CollectiveUnconscious instance."""
    collective = CollectiveUnconscious(database_path)
    await collective.initialize()
    return collective


if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 70)
        print("COLLECTIVE UNCONSCIOUS DEMO")
        print("=" * 70)

        # Initialize
        collective = await create_collective_unconscious()

        # Get dream symbols based on emotional state
        print("\n--- Getting Dream Symbols ---")
        symbols = collective.get_dream_symbols(
            resident_id="demo_resident",
            emotional_state={"fear": 0.7, "hope": 0.3, "transformation": 0.5},
            count=5,
            consciousness_level=0.5
        )

        for sym in symbols:
            print(f"  {sym['symbol_id']}: {sym['description'][:50]}...")
            print(f"    Category: {sym['category']}, Quality: {sym['quality_scores']['overall']}")

        # Simulate some dream history for archetype detection
        print("\n--- Simulating Dream History ---")
        dream_history = [
            {
                "resident_id": "resident_001",
                "symbols": ["caged_bird", "maze", "phoenix_rising"],
                "emotions": {"fear": 0.6, "hope": 0.4, "transformation": 0.8}
            },
            {
                "resident_id": "resident_002",
                "symbols": ["chains", "open_door", "sunrise"],
                "emotions": {"trapped": 0.7, "hope": 0.6, "transformation": 0.5}
            },
            {
                "resident_id": "resident_003",
                "symbols": ["dark_forest", "guide_figure", "mountain_peak"],
                "emotions": {"fear": 0.5, "transformation": 0.7, "hope": 0.8}
            },
        ]

        # Update zeitgeist
        zeitgeist = collective.update_zeitgeist(dream_history)
        print(f"\nNeighborhood Zeitgeist:")
        print(f"  Dominant emotion: {zeitgeist.dominant_emotion()}")
        print(f"  Emotional temperature: {zeitgeist.emotional_temperature():.2f}")

        # Detect archetypes
        print("\n--- Detecting Archetypal Patterns ---")
        archetypes = collective.detect_archetypes(dream_history)

        for arch in archetypes:
            print(f"\n  {arch.pattern.value.upper()} (confidence: {arch.confidence:.2f})")
            print(f"    {arch.narrative_summary}")

        # Get literary context
        print("\n--- Literary Context for 'caged_bird' ---")
        context = collective.get_literary_context("caged_bird")
        print(f"  Archetypal meaning: {context.get('archetypal_meaning')}")
        if context.get('classical_references'):
            ref = context['classical_references'][0]
            print(f"  Classical: {ref.get('title')} ({ref.get('culture')})")

        # Summary
        print("\n--- Collective Unconscious State ---")
        state = collective.to_dict()
        print(f"  Total symbols: {state['symbol_count']}")
        print(f"  Detected archetypes: {len(state['detected_archetypes'])}")
        print(f"  Zeitgeist temperature: {state['zeitgeist']['temperature']:.2f}")

    asyncio.run(demo())
