"""
Dream Journal & Analysis System - Phase 5 Week 4

A comprehensive dream journaling and analysis system that:
- Records and stores complete dream narratives
- Analyzes patterns across dream history
- Generates insights about psychological development
- Tracks symbol frequency, recurring themes, and emotional arcs
- Provides personalized Jungian interpretations

The journal serves as both a practical record and a tool for
understanding the psyche's journey toward individuation.

Author: Claude (Phase 5 Implementation)
Date: November 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import math
import random


class DreamSignificance(Enum):
    """Significance level of a dream."""
    MUNDANE = 1         # Processing daily events
    NOTABLE = 2         # Worth remembering
    SIGNIFICANT = 3     # Important message
    PIVOTAL = 4         # Life-changing potential
    NUMINOUS = 5        # Transcendent experience


class PatternType(Enum):
    """Types of patterns detected in dream analysis."""
    RECURRING_SYMBOL = "recurring_symbol"
    EMOTIONAL_TREND = "emotional_trend"
    ARCHETYPE_SEQUENCE = "archetype_sequence"
    RELATIONSHIP_THEME = "relationship_theme"
    SHADOW_EMERGENCE = "shadow_emergence"
    TRANSFORMATION_ARC = "transformation_arc"
    COMPENSATION = "compensation"  # Dreams compensating for waking imbalance
    PROSPECTIVE = "prospective"    # Dreams pointing to future development


@dataclass
class DreamEntry:
    """A single dream journal entry."""
    entry_id: str
    resident_id: str
    timestamp: datetime
    title: str
    narrative: str
    symbols: List[str]
    emotions: Dict[str, float]
    archetype: Optional[str]
    mood: str
    significance: DreamSignificance
    people_present: List[str]
    locations: List[str]
    actions: List[str]
    interpretation: Optional[str] = None
    personal_notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_lucid: bool = False
    is_recurring: bool = False
    shared_with: List[str] = field(default_factory=list)  # Other dreamers
    influences_generated: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entry_id": self.entry_id,
            "resident_id": self.resident_id,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "narrative": self.narrative,
            "symbols": self.symbols,
            "emotions": self.emotions,
            "archetype": self.archetype,
            "mood": self.mood,
            "significance": self.significance.value,
            "people_present": self.people_present,
            "locations": self.locations,
            "actions": self.actions,
            "interpretation": self.interpretation,
            "personal_notes": self.personal_notes,
            "tags": self.tags,
            "is_lucid": self.is_lucid,
            "is_recurring": self.is_recurring,
            "shared_with": self.shared_with,
            "influences_generated": self.influences_generated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DreamEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            resident_id=data["resident_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            title=data["title"],
            narrative=data["narrative"],
            symbols=data["symbols"],
            emotions=data["emotions"],
            archetype=data.get("archetype"),
            mood=data["mood"],
            significance=DreamSignificance(data["significance"]),
            people_present=data.get("people_present", []),
            locations=data.get("locations", []),
            actions=data.get("actions", []),
            interpretation=data.get("interpretation"),
            personal_notes=data.get("personal_notes"),
            tags=data.get("tags", []),
            is_lucid=data.get("is_lucid", False),
            is_recurring=data.get("is_recurring", False),
            shared_with=data.get("shared_with", []),
            influences_generated=data.get("influences_generated", []),
        )


@dataclass
class DetectedPattern:
    """A pattern detected in dream analysis."""
    pattern_id: str
    pattern_type: PatternType
    description: str
    confidence: float
    supporting_dreams: List[str]  # entry_ids
    symbols_involved: List[str]
    first_occurrence: datetime
    last_occurrence: datetime
    occurrence_count: int
    psychological_meaning: str
    suggested_action: Optional[str] = None


@dataclass
class DreamAnalysis:
    """Complete analysis of a resident's dream journal."""
    resident_id: str
    analysis_date: datetime
    dream_count: int
    date_range: Tuple[datetime, datetime]

    # Symbol analysis
    most_common_symbols: List[Tuple[str, int]]
    symbol_categories: Dict[str, int]  # category -> count
    emerging_symbols: List[str]  # New symbols in recent dreams
    fading_symbols: List[str]  # Symbols decreasing in frequency

    # Emotional analysis
    emotional_baseline: Dict[str, float]
    emotional_trends: Dict[str, str]  # emotion -> "increasing"/"stable"/"decreasing"
    dominant_mood: str
    mood_variety: float  # 0-1, how varied moods are

    # Archetype analysis
    archetype_distribution: Dict[str, int]
    dominant_archetype: Optional[str]
    archetype_sequence: List[str]  # Recent progression

    # Pattern detection
    patterns: List[DetectedPattern]

    # Development metrics
    individuation_score: float  # 0-1, progress toward wholeness
    shadow_integration: float   # 0-1, shadow work progress
    anima_animus_development: float  # 0-1
    self_awareness: float       # 0-1

    # Recommendations
    insights: List[str]
    suggestions: List[str]
    areas_of_focus: List[str]


class DreamJournal:
    """
    Dream Journal and Analysis System.

    Stores dream entries and provides comprehensive analysis
    of dream patterns, psychological development, and insights.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.entries: Dict[str, List[DreamEntry]] = {}  # resident_id -> entries
        self.patterns: Dict[str, List[DetectedPattern]] = {}  # resident_id -> patterns

        # Symbol category mappings (from collective unconscious)
        self.symbol_categories = {
            "trapped": ["cage", "chains", "locked_door", "maze", "pit", "prison"],
            "loss": ["empty_chair", "fading_photo", "torn_letter", "withered_flower"],
            "fear": ["shadow_figure", "falling", "teeth_falling", "drowning", "chased"],
            "hope": ["sunrise", "sprout", "rainbow", "star", "candle_light"],
            "transformation": ["butterfly", "phoenix", "molting_snake", "cocoon"],
            "connection": ["bridge", "handshake", "shared_meal", "reunion"],
            "power": ["crown", "throne", "sword", "mountain_peak"],
            "mystery": ["fog", "hidden_door", "oracle", "ancient_book"],
            "conflict": ["battle", "storm", "fire", "flood"],
            "guilt": ["blood", "broken_mirror", "accusation", "judgment"],
        }

        # Load existing data if storage path provided
        if storage_path:
            self._load_data()

    def _load_data(self):
        """Load journal data from storage."""
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                for resident_id, entries in data.get("entries", {}).items():
                    self.entries[resident_id] = [
                        DreamEntry.from_dict(e) for e in entries
                    ]
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_data(self):
        """Save journal data to storage."""
        if not self.storage_path:
            return

        data = {
            "entries": {
                rid: [e.to_dict() for e in entries]
                for rid, entries in self.entries.items()
            }
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def record_dream(
        self,
        resident_id: str,
        title: str,
        narrative: str,
        symbols: List[str],
        emotions: Dict[str, float],
        archetype: Optional[str] = None,
        mood: str = "mysterious",
        people_present: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        personal_notes: Optional[str] = None,
        is_lucid: bool = False,
        shared_with: Optional[List[str]] = None,
    ) -> DreamEntry:
        """
        Record a new dream entry.

        Args:
            resident_id: Dreamer's ID
            title: Dream title
            narrative: Full narrative text
            symbols: Symbols appearing in dream
            emotions: Emotional state dict
            archetype: Jungian archetype if applicable
            mood: Overall dream mood
            people_present: People who appeared
            locations: Dream locations
            actions: Key actions taken
            personal_notes: Dreamer's own interpretation
            is_lucid: Whether dream was lucid
            shared_with: Other dreamers (collective dream)

        Returns:
            Created DreamEntry
        """
        # Determine significance
        significance = self._calculate_significance(
            symbols, emotions, archetype, is_lucid
        )

        # Check for recurring patterns
        is_recurring = self._check_recurring(
            resident_id, symbols, locations, people_present or []
        )

        # Generate automatic interpretation
        interpretation = self._generate_interpretation(
            symbols, emotions, archetype, mood
        )

        # Generate entry ID
        entry_id = f"dream_{resident_id}_{datetime.now().timestamp()}"

        entry = DreamEntry(
            entry_id=entry_id,
            resident_id=resident_id,
            timestamp=datetime.now(),
            title=title,
            narrative=narrative,
            symbols=symbols,
            emotions=emotions,
            archetype=archetype,
            mood=mood,
            significance=significance,
            people_present=people_present or [],
            locations=locations or [],
            actions=actions or [],
            interpretation=interpretation,
            personal_notes=personal_notes,
            is_lucid=is_lucid,
            is_recurring=is_recurring,
            shared_with=shared_with or [],
        )

        # Store entry
        if resident_id not in self.entries:
            self.entries[resident_id] = []
        self.entries[resident_id].append(entry)

        # Update patterns
        self._update_patterns(resident_id, entry)

        # Save if storage configured
        self._save_data()

        return entry

    def _calculate_significance(
        self,
        symbols: List[str],
        emotions: Dict[str, float],
        archetype: Optional[str],
        is_lucid: bool
    ) -> DreamSignificance:
        """Calculate dream significance based on content."""
        score = 1  # Start at mundane

        # More symbols = more significant
        if len(symbols) >= 5:
            score += 1
        if len(symbols) >= 10:
            score += 1

        # Strong emotions increase significance
        max_emotion = max(emotions.values()) if emotions else 0
        if max_emotion > 0.7:
            score += 1
        if max_emotion > 0.9:
            score += 1

        # Archetypes are significant
        if archetype:
            score += 1

        # Lucid dreams are notable
        if is_lucid:
            score += 1

        return DreamSignificance(min(5, score))

    def _check_recurring(
        self,
        resident_id: str,
        symbols: List[str],
        locations: List[str],
        people: List[str]
    ) -> bool:
        """Check if this dream matches recurring patterns."""
        if resident_id not in self.entries:
            return False

        recent_entries = self.entries[resident_id][-10:]  # Last 10 dreams

        for entry in recent_entries:
            # Check symbol overlap
            symbol_overlap = len(set(symbols) & set(entry.symbols))
            if symbol_overlap >= 3:
                return True

            # Check location match
            if locations and entry.locations:
                if set(locations) & set(entry.locations):
                    return True

        return False

    def _generate_interpretation(
        self,
        symbols: List[str],
        emotions: Dict[str, float],
        archetype: Optional[str],
        mood: str
    ) -> str:
        """Generate automatic Jungian interpretation."""
        interpretations = []

        # Archetype interpretation
        archetype_meanings = {
            "heros_journey": "The psyche calls for growth through challenge and transformation.",
            "shadow_integration": "The unconscious invites you to acknowledge hidden aspects of self.",
            "death_rebirth": "A cycle ends; transformation and renewal await.",
            "great_mother": "Themes of nurturing, protection, or devouring mother emerge.",
            "wise_old_one": "Inner wisdom seeks expression; guidance is available.",
            "divine_child": "New beginnings, innocence, or unrealized potential surface.",
            "anima_animus": "The inner feminine/masculine seeks integration.",
            "trickster": "The unexpected disrupts; look for hidden wisdom in chaos.",
        }

        if archetype and archetype in archetype_meanings:
            interpretations.append(archetype_meanings[archetype])

        # Dominant emotion interpretation
        if emotions:
            dominant = max(emotions.items(), key=lambda x: x[1])
            emotion_meanings = {
                "fear": "Fear in dreams often points to what we must face to grow.",
                "joy": "Joy signals alignment between conscious and unconscious.",
                "grief": "The psyche processes loss, allowing healing to begin.",
                "anxiety": "Unresolved tensions seek acknowledgment and resolution.",
                "hope": "The Self signals movement toward positive development.",
                "connection": "Relationship needs or desires emerge from the unconscious.",
            }
            if dominant[0] in emotion_meanings and dominant[1] > 0.5:
                interpretations.append(emotion_meanings[dominant[0]])

        # Symbol category interpretation
        categories_found = set()
        for symbol in symbols:
            for category, cat_symbols in self.symbol_categories.items():
                if symbol in cat_symbols:
                    categories_found.add(category)

        category_meanings = {
            "trapped": "Feelings of limitation or constraint seek expression.",
            "transformation": "The psyche is ready for significant change.",
            "connection": "Relationship or belonging needs emerge.",
            "power": "Questions of agency and effectiveness are active.",
            "mystery": "The unknown beckons; trust your intuition.",
        }

        for cat in categories_found:
            if cat in category_meanings:
                interpretations.append(category_meanings[cat])
                break  # One category meaning is enough

        if not interpretations:
            interpretations.append(
                "This dream invites reflection on its symbols and emotional tone."
            )

        return " ".join(interpretations[:3])

    def _update_patterns(self, resident_id: str, new_entry: DreamEntry):
        """Update pattern detection with new entry."""
        if resident_id not in self.patterns:
            self.patterns[resident_id] = []

        entries = self.entries.get(resident_id, [])
        if len(entries) < 3:
            return  # Need more dreams for pattern detection

        # Check for recurring symbol patterns
        symbol_counts = Counter()
        for entry in entries[-20:]:  # Last 20 dreams
            symbol_counts.update(entry.symbols)

        for symbol, count in symbol_counts.most_common(5):
            if count >= 3:
                # Check if pattern already exists
                existing = next(
                    (p for p in self.patterns[resident_id]
                     if p.pattern_type == PatternType.RECURRING_SYMBOL
                     and symbol in p.symbols_involved),
                    None
                )

                if existing:
                    existing.occurrence_count = count
                    existing.last_occurrence = new_entry.timestamp
                    if new_entry.entry_id not in existing.supporting_dreams:
                        existing.supporting_dreams.append(new_entry.entry_id)
                else:
                    pattern = DetectedPattern(
                        pattern_id=f"pattern_{datetime.now().timestamp()}",
                        pattern_type=PatternType.RECURRING_SYMBOL,
                        description=f"'{symbol}' appears frequently in your dreams",
                        confidence=min(0.95, 0.5 + count * 0.1),
                        supporting_dreams=[e.entry_id for e in entries if symbol in e.symbols][-5:],
                        symbols_involved=[symbol],
                        first_occurrence=min(e.timestamp for e in entries if symbol in e.symbols),
                        last_occurrence=new_entry.timestamp,
                        occurrence_count=count,
                        psychological_meaning=self._get_symbol_meaning(symbol),
                        suggested_action=self._get_symbol_action(symbol),
                    )
                    self.patterns[resident_id].append(pattern)

    def _get_symbol_meaning(self, symbol: str) -> str:
        """Get psychological meaning for a symbol."""
        meanings = {
            "cage": "Feeling restricted or trapped in some life area",
            "sunrise": "Hope for new beginnings emerging from unconscious",
            "shadow_figure": "Unintegrated shadow aspects seeking acknowledgment",
            "bridge": "Desire to connect or transition between life phases",
            "water": "The unconscious itself; emotional depths",
            "fire": "Transformation, passion, or destructive forces",
            "flying": "Desire for freedom or transcendence",
            "falling": "Loss of control or fear of failure",
        }
        return meanings.get(symbol, f"The symbol '{symbol}' has personal significance")

    def _get_symbol_action(self, symbol: str) -> str:
        """Get suggested action for recurring symbol."""
        actions = {
            "cage": "Explore what feels limiting in your life",
            "shadow_figure": "Consider what aspects of yourself you may be avoiding",
            "water": "Pay attention to your emotional state",
            "fire": "Channel passion or address anger constructively",
            "bridge": "Consider what transitions you're navigating",
        }
        return actions.get(symbol, "Reflect on what this symbol means to you personally")

    def get_entries(
        self,
        resident_id: str,
        limit: int = 50,
        since: Optional[datetime] = None,
        archetype: Optional[str] = None,
        mood: Optional[str] = None,
        min_significance: Optional[DreamSignificance] = None,
    ) -> List[DreamEntry]:
        """
        Get dream entries with optional filtering.

        Args:
            resident_id: Dreamer's ID
            limit: Maximum entries to return
            since: Only entries after this date
            archetype: Filter by archetype
            mood: Filter by mood
            min_significance: Minimum significance level

        Returns:
            List of matching DreamEntry objects
        """
        entries = self.entries.get(resident_id, [])

        # Apply filters
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        if archetype:
            entries = [e for e in entries if e.archetype == archetype]

        if mood:
            entries = [e for e in entries if e.mood == mood]

        if min_significance:
            entries = [e for e in entries if e.significance.value >= min_significance.value]

        # Sort by timestamp descending (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries[:limit]

    def analyze(
        self,
        resident_id: str,
        lookback_days: int = 30
    ) -> DreamAnalysis:
        """
        Perform comprehensive analysis of dream journal.

        Args:
            resident_id: Dreamer's ID
            lookback_days: How many days to analyze

        Returns:
            Complete DreamAnalysis
        """
        since = datetime.now() - timedelta(days=lookback_days)
        entries = self.get_entries(resident_id, limit=1000, since=since)

        if not entries:
            # Return empty analysis
            now = datetime.now()
            return DreamAnalysis(
                resident_id=resident_id,
                analysis_date=now,
                dream_count=0,
                date_range=(now, now),
                most_common_symbols=[],
                symbol_categories={},
                emerging_symbols=[],
                fading_symbols=[],
                emotional_baseline={},
                emotional_trends={},
                dominant_mood="none",
                mood_variety=0.0,
                archetype_distribution={},
                dominant_archetype=None,
                archetype_sequence=[],
                patterns=[],
                individuation_score=0.0,
                shadow_integration=0.0,
                anima_animus_development=0.0,
                self_awareness=0.0,
                insights=["Record more dreams to enable analysis."],
                suggestions=["Try to record dreams immediately upon waking."],
                areas_of_focus=["Building dream recall"],
            )

        # Symbol analysis
        all_symbols = []
        for entry in entries:
            all_symbols.extend(entry.symbols)

        symbol_counts = Counter(all_symbols)
        most_common = symbol_counts.most_common(10)

        # Categorize symbols
        cat_counts: Dict[str, int] = {}
        for symbol in all_symbols:
            for cat, cat_symbols in self.symbol_categories.items():
                if symbol in cat_symbols:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # Find emerging/fading symbols (compare recent half to older half)
        midpoint = len(entries) // 2
        recent_symbols = Counter()
        older_symbols = Counter()

        for i, entry in enumerate(entries):
            if i < midpoint:
                recent_symbols.update(entry.symbols)
            else:
                older_symbols.update(entry.symbols)

        emerging = [
            s for s in recent_symbols
            if recent_symbols[s] > older_symbols.get(s, 0) + 2
        ]
        fading = [
            s for s in older_symbols
            if older_symbols[s] > recent_symbols.get(s, 0) + 2
        ]

        # Emotional analysis
        all_emotions: Dict[str, List[float]] = defaultdict(list)
        for entry in entries:
            for emotion, value in entry.emotions.items():
                all_emotions[emotion].append(value)

        emotional_baseline = {
            emotion: sum(values) / len(values)
            for emotion, values in all_emotions.items()
        }

        # Emotional trends
        emotional_trends = {}
        for emotion, values in all_emotions.items():
            if len(values) >= 3:
                recent = sum(values[:len(values)//2]) / (len(values)//2)
                older = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
                if recent > older + 0.1:
                    emotional_trends[emotion] = "increasing"
                elif recent < older - 0.1:
                    emotional_trends[emotion] = "decreasing"
                else:
                    emotional_trends[emotion] = "stable"

        # Mood analysis
        mood_counts = Counter(e.mood for e in entries)
        dominant_mood = mood_counts.most_common(1)[0][0] if mood_counts else "unknown"
        mood_variety = len(mood_counts) / max(8, len(entries))  # 8 possible moods

        # Archetype analysis
        archetype_counts = Counter(
            e.archetype for e in entries if e.archetype
        )
        dominant_archetype = (
            archetype_counts.most_common(1)[0][0]
            if archetype_counts else None
        )
        archetype_sequence = [
            e.archetype for e in entries[:10]
            if e.archetype
        ]

        # Development metrics
        individuation_score = self._calculate_individuation(entries)
        shadow_integration = self._calculate_shadow_work(entries)
        anima_animus_dev = self._calculate_anima_animus(entries)
        self_awareness = self._calculate_self_awareness(entries)

        # Get patterns
        patterns = self.patterns.get(resident_id, [])

        # Generate insights and suggestions
        insights = self._generate_insights(
            entries, patterns, emotional_baseline, symbol_counts
        )
        suggestions = self._generate_suggestions(
            entries, patterns, emotional_trends
        )
        areas_of_focus = self._identify_focus_areas(
            shadow_integration, anima_animus_dev, individuation_score
        )

        return DreamAnalysis(
            resident_id=resident_id,
            analysis_date=datetime.now(),
            dream_count=len(entries),
            date_range=(entries[-1].timestamp, entries[0].timestamp),
            most_common_symbols=most_common,
            symbol_categories=cat_counts,
            emerging_symbols=emerging[:5],
            fading_symbols=fading[:5],
            emotional_baseline=emotional_baseline,
            emotional_trends=emotional_trends,
            dominant_mood=dominant_mood,
            mood_variety=mood_variety,
            archetype_distribution=dict(archetype_counts),
            dominant_archetype=dominant_archetype,
            archetype_sequence=archetype_sequence,
            patterns=patterns,
            individuation_score=individuation_score,
            shadow_integration=shadow_integration,
            anima_animus_development=anima_animus_dev,
            self_awareness=self_awareness,
            insights=insights,
            suggestions=suggestions,
            areas_of_focus=areas_of_focus,
        )

    def _calculate_individuation(self, entries: List[DreamEntry]) -> float:
        """Calculate progress toward individuation (wholeness)."""
        if not entries:
            return 0.0

        score = 0.0

        # Variety of archetypes (integration)
        archetypes = set(e.archetype for e in entries if e.archetype)
        score += min(0.3, len(archetypes) * 0.05)

        # Balance of emotions (neither suppressed nor overwhelming)
        emotions = defaultdict(list)
        for entry in entries:
            for emotion, value in entry.emotions.items():
                emotions[emotion].append(value)

        if emotions:
            variance = sum(
                (sum(v)/len(v)) ** 2 for v in emotions.values()
            ) / len(emotions)
            # Low variance with moderate values = good
            score += max(0, 0.3 - variance * 0.3)

        # Transformation symbols indicate growth
        transformation_count = sum(
            1 for e in entries
            for s in e.symbols
            if s in self.symbol_categories.get("transformation", [])
        )
        score += min(0.2, transformation_count * 0.02)

        # Lucid dreams indicate awareness
        lucid_count = sum(1 for e in entries if e.is_lucid)
        score += min(0.2, lucid_count * 0.05)

        return min(1.0, score)

    def _calculate_shadow_work(self, entries: List[DreamEntry]) -> float:
        """Calculate shadow integration progress."""
        if not entries:
            return 0.0

        score = 0.0

        # Shadow-related archetypes
        shadow_archetypes = ["shadow_integration", "trickster"]
        shadow_dreams = sum(
            1 for e in entries if e.archetype in shadow_archetypes
        )
        score += min(0.4, shadow_dreams * 0.05)

        # Fear symbols appearing and being processed
        fear_symbols = self.symbol_categories.get("fear", [])
        fear_count = sum(
            1 for e in entries
            for s in e.symbols
            if s in fear_symbols
        )
        # Some fear is healthy (facing shadow)
        score += min(0.3, fear_count * 0.03)

        # Conflict resolution (conflict followed by hope)
        for i in range(len(entries) - 1):
            current_has_conflict = any(
                s in self.symbol_categories.get("conflict", [])
                for s in entries[i].symbols
            )
            next_has_hope = any(
                s in self.symbol_categories.get("hope", [])
                for s in entries[i+1].symbols
            )
            if current_has_conflict and next_has_hope:
                score += 0.1

        return min(1.0, score)

    def _calculate_anima_animus(self, entries: List[DreamEntry]) -> float:
        """Calculate anima/animus development."""
        if not entries:
            return 0.0

        score = 0.0

        # Anima/animus archetype appearances
        aa_dreams = sum(
            1 for e in entries if e.archetype == "anima_animus"
        )
        score += min(0.3, aa_dreams * 0.1)

        # Connection symbols (relationship with inner other)
        connection_symbols = self.symbol_categories.get("connection", [])
        conn_count = sum(
            1 for e in entries
            for s in e.symbols
            if s in connection_symbols
        )
        score += min(0.3, conn_count * 0.02)

        # Emotional balance (neither too masculine nor feminine)
        # Approximated by balance of power vs connection themes
        power_count = sum(
            1 for e in entries
            for s in e.symbols
            if s in self.symbol_categories.get("power", [])
        )
        balance = 1 - abs(power_count - conn_count) / max(power_count + conn_count, 1)
        score += balance * 0.4

        return min(1.0, score)

    def _calculate_self_awareness(self, entries: List[DreamEntry]) -> float:
        """Calculate self-awareness from dreams."""
        if not entries:
            return 0.0

        score = 0.0

        # Lucid dreams = highest awareness
        lucid_count = sum(1 for e in entries if e.is_lucid)
        score += min(0.4, lucid_count * 0.1)

        # Higher significance dreams
        high_sig = sum(
            1 for e in entries if e.significance.value >= 3
        )
        score += min(0.3, high_sig * 0.03)

        # Personal notes indicate reflection
        notes_count = sum(1 for e in entries if e.personal_notes)
        score += min(0.3, notes_count * 0.03)

        return min(1.0, score)

    def _generate_insights(
        self,
        entries: List[DreamEntry],
        patterns: List[DetectedPattern],
        emotional_baseline: Dict[str, float],
        symbol_counts: Counter
    ) -> List[str]:
        """Generate personalized insights."""
        insights = []

        # Top symbol insight
        if symbol_counts:
            top_symbol, count = symbol_counts.most_common(1)[0]
            insights.append(
                f"'{top_symbol}' appears in {count} dreams, "
                f"suggesting its significance in your psyche."
            )

        # Emotional insight
        if emotional_baseline:
            top_emotion = max(emotional_baseline.items(), key=lambda x: x[1])
            insights.append(
                f"Your dreams are predominantly colored by {top_emotion[0]} "
                f"(avg intensity: {top_emotion[1]:.1%})."
            )

        # Pattern insight
        for pattern in patterns[:2]:
            insights.append(pattern.psychological_meaning)

        # Recurring dream insight
        recurring = sum(1 for e in entries if e.is_recurring)
        if recurring >= 3:
            insights.append(
                f"{recurring} dreams show recurring elements, "
                "indicating unresolved material seeking attention."
            )

        return insights[:5]

    def _generate_suggestions(
        self,
        entries: List[DreamEntry],
        patterns: List[DetectedPattern],
        emotional_trends: Dict[str, str]
    ) -> List[str]:
        """Generate actionable suggestions."""
        suggestions = []

        # Pattern-based suggestions
        for pattern in patterns:
            if pattern.suggested_action:
                suggestions.append(pattern.suggested_action)

        # Emotional trend suggestions
        if emotional_trends.get("anxiety") == "increasing":
            suggestions.append(
                "Anxiety is increasing in your dreams. "
                "Consider what waking stressors may be contributing."
            )

        if emotional_trends.get("fear") == "increasing":
            suggestions.append(
                "Fear is becoming more present. "
                "Your dreams may be inviting you to face something."
            )

        if emotional_trends.get("hope") == "decreasing":
            suggestions.append(
                "Hope symbols are fading. "
                "Nurture sources of optimism in waking life."
            )

        # General suggestions
        lucid_count = sum(1 for e in entries if e.is_lucid)
        if lucid_count == 0:
            suggestions.append(
                "Try lucid dreaming techniques to increase "
                "active engagement with your unconscious."
            )

        return suggestions[:5]

    def _identify_focus_areas(
        self,
        shadow: float,
        anima_animus: float,
        individuation: float
    ) -> List[str]:
        """Identify areas needing attention."""
        areas = []

        if shadow < 0.3:
            areas.append("Shadow work: Facing and integrating hidden aspects")

        if anima_animus < 0.3:
            areas.append("Anima/animus: Developing relationship with inner other")

        if individuation < 0.4:
            areas.append("Individuation: Moving toward psychological wholeness")

        if not areas:
            areas.append("Continue current dream work: Good progress on all fronts")

        return areas

    def get_dream_timeline(
        self,
        resident_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get dream timeline for visualization."""
        since = datetime.now() - timedelta(days=days)
        entries = self.get_entries(resident_id, limit=100, since=since)

        timeline = []
        for entry in entries:
            timeline.append({
                "date": entry.timestamp.isoformat(),
                "title": entry.title,
                "mood": entry.mood,
                "significance": entry.significance.name,
                "archetype": entry.archetype,
                "symbol_count": len(entry.symbols),
                "is_lucid": entry.is_lucid,
                "is_recurring": entry.is_recurring,
            })

        return timeline

    def search_dreams(
        self,
        resident_id: str,
        query: str,
        limit: int = 10
    ) -> List[DreamEntry]:
        """Search dreams by text content."""
        entries = self.entries.get(resident_id, [])

        query_lower = query.lower()
        matches = []

        for entry in entries:
            # Search in title, narrative, symbols
            if (query_lower in entry.title.lower() or
                query_lower in entry.narrative.lower() or
                any(query_lower in s.lower() for s in entry.symbols)):
                matches.append(entry)

        # Sort by relevance (exact matches first)
        matches.sort(
            key=lambda e: (
                query_lower in e.title.lower(),
                any(query_lower == s.lower() for s in e.symbols),
            ),
            reverse=True
        )

        return matches[:limit]


def create_dream_journal(storage_path: Optional[str] = None) -> DreamJournal:
    """Create a new Dream Journal instance."""
    return DreamJournal(storage_path)


if __name__ == "__main__":
    print("=" * 70)
    print("DREAM JOURNAL & ANALYSIS DEMO")
    print("=" * 70)

    journal = create_dream_journal()
    resident = "maya_001"

    # Record several dreams
    print("\n--- Recording Dreams ---")

    dream1 = journal.record_dream(
        resident_id=resident,
        title="The Bridge in Mist",
        narrative="I stood before an ancient stone bridge shrouded in fog. "
                  "On the other side, a figure waited. I began to cross, "
                  "but the bridge started to crumble beneath my feet.",
        symbols=["bridge", "fog", "shadow_figure", "falling"],
        emotions={"fear": 0.6, "hope": 0.4, "curiosity": 0.5},
        archetype="heros_journey",
        mood="mysterious",
        locations=["bridge", "chasm"],
        actions=["crossing", "falling"],
    )
    print(f"Recorded: '{dream1.title}'")
    print(f"  Significance: {dream1.significance.name}")
    print(f"  Interpretation: {dream1.interpretation[:100]}...")

    dream2 = journal.record_dream(
        resident_id=resident,
        title="Garden of Butterflies",
        narrative="A garden filled with thousands of butterflies. Each one "
                  "carried a memory. When I touched one, I saw my grandmother's face.",
        symbols=["butterfly", "garden", "fading_photo", "blooming_flower"],
        emotions={"grief": 0.5, "joy": 0.6, "connection": 0.7},
        archetype="death_rebirth",
        mood="wonder",
        people_present=["grandmother_deceased"],
        is_lucid=True,
    )
    print(f"\nRecorded: '{dream2.title}'")
    print(f"  Significance: {dream2.significance.name}")
    print(f"  Lucid: {dream2.is_lucid}")

    dream3 = journal.record_dream(
        resident_id=resident,
        title="The Locked Room",
        narrative="Trapped in a room with no doors. The walls were closing in. "
                  "Then I noticed a small key in my pocket.",
        symbols=["cage", "locked_door", "key", "darkness"],
        emotions={"anxiety": 0.8, "fear": 0.6, "hope": 0.3},
        archetype="shadow_integration",
        mood="nightmare",
        personal_notes="Feels related to work stress",
    )
    print(f"\nRecorded: '{dream3.title}'")
    print(f"  Recurring: {dream3.is_recurring}")

    # More dreams to build patterns
    for i in range(5):
        journal.record_dream(
            resident_id=resident,
            title=f"Dream {i+4}",
            narrative="A wandering dream...",
            symbols=random.sample(
                ["water", "fire", "bridge", "shadow_figure", "sunrise", "cage"],
                3
            ),
            emotions={"fear": random.random(), "hope": random.random()},
            mood=random.choice(["mysterious", "peaceful", "tense"]),
        )

    # Run analysis
    print("\n" + "=" * 70)
    print("DREAM ANALYSIS")
    print("=" * 70)

    analysis = journal.analyze(resident, lookback_days=30)

    print(f"\nDreams Analyzed: {analysis.dream_count}")
    print(f"Date Range: {analysis.date_range[0].date()} to {analysis.date_range[1].date()}")

    print("\n--- Symbol Analysis ---")
    print("Most Common Symbols:")
    for symbol, count in analysis.most_common_symbols[:5]:
        print(f"  {symbol}: {count} appearances")

    if analysis.symbol_categories:
        print("\nSymbol Categories:")
        for cat, count in sorted(analysis.symbol_categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    print("\n--- Emotional Profile ---")
    print(f"Dominant Mood: {analysis.dominant_mood}")
    print(f"Mood Variety: {analysis.mood_variety:.1%}")
    print("Emotional Baseline:")
    for emotion, value in sorted(analysis.emotional_baseline.items(), key=lambda x: -x[1])[:5]:
        print(f"  {emotion}: {value:.1%}")

    print("\n--- Development Metrics ---")
    print(f"Individuation Score: {analysis.individuation_score:.1%}")
    print(f"Shadow Integration: {analysis.shadow_integration:.1%}")
    print(f"Anima/Animus Development: {analysis.anima_animus_development:.1%}")
    print(f"Self-Awareness: {analysis.self_awareness:.1%}")

    print("\n--- Insights ---")
    for i, insight in enumerate(analysis.insights, 1):
        print(f"{i}. {insight}")

    print("\n--- Suggestions ---")
    for i, suggestion in enumerate(analysis.suggestions, 1):
        print(f"{i}. {suggestion}")

    print("\n--- Areas of Focus ---")
    for area in analysis.areas_of_focus:
        print(f"  * {area}")

    # Search dreams
    print("\n--- Search: 'bridge' ---")
    results = journal.search_dreams(resident, "bridge")
    for entry in results:
        print(f"  {entry.title}: {entry.symbols}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)