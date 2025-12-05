"""
Dream System - Phase 5 Complete Integration

Unified interface to the complete Dream Consciousness System:
- Collective Unconscious (shared symbols, archetypes, zeitgeist)
- Narrative Generator (three-act dream stories)
- Dream Influence (bidirectional waking-dream effects)
- Dream Journal (recording, analysis, patterns)

This module provides the main entry point for integrating
dreams into NeuroHood's neighborhood simulation.

Author: Claude (Phase 5 Implementation)
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

# Import all Phase 5 components
from collective_unconscious import (
    CollectiveUnconscious,
    create_collective_unconscious,
    EmotionalEssence,
    DetectedArchetype,
)
from narrative_generator import (
    NarrativeGenerator,
    NarrativeArchetype,
    DreamMood,
    DreamNarrative,
)
from dream_influence import (
    DreamInfluenceSystem,
    create_dream_influence_system,
    DreamInfluence,
    WakingExperience,
    InfluenceType,
)
from dream_journal import (
    DreamJournal,
    create_dream_journal,
    DreamEntry,
    DreamAnalysis,
    DreamSignificance,
)


@dataclass
class DreamSession:
    """Complete dream session data."""
    resident_id: str
    narrative: DreamNarrative
    journal_entry: DreamEntry
    influences: List[DreamInfluence]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DreamSystemState:
    """System-wide state snapshot."""
    neighborhood_zeitgeist: EmotionalEssence
    neighborhood_mood: Dict[str, float]
    active_archetypes: Dict[str, int]  # archetype -> count
    total_dreams_today: int
    collective_dream_count: int


class DreamSystem:
    """
    Unified Dream Consciousness System.

    Coordinates all dream components into a cohesive system
    for neighborhood-wide dream generation and influence.
    """

    def __init__(
        self,
        symbol_database_path: Optional[str] = None,
        journal_storage_path: Optional[str] = None,
    ):
        """
        Initialize the Dream System.

        Args:
            symbol_database_path: Path to enriched symbol database JSON
            journal_storage_path: Path for journal persistence
        """
        self.symbol_database_path = symbol_database_path
        self.journal_storage_path = journal_storage_path

        # Components (initialized in setup())
        self.collective: Optional[CollectiveUnconscious] = None
        self.generator: Optional[NarrativeGenerator] = None
        self.influence: Optional[DreamInfluenceSystem] = None
        self.journal: Optional[DreamJournal] = None

        # State tracking
        self.dreams_today: List[DreamSession] = []
        self.is_initialized = False

    async def initialize(self):
        """Initialize all dream system components."""
        if self.is_initialized:
            return

        # Initialize collective unconscious with symbols
        self.collective = await create_collective_unconscious(
            self.symbol_database_path
        )

        # Initialize narrative generator (uses collective)
        self.generator = NarrativeGenerator(self.collective)

        # Initialize influence system
        self.influence = create_dream_influence_system()

        # Initialize journal
        self.journal = create_dream_journal(self.journal_storage_path)

        self.is_initialized = True
        symbol_count = len(self.collective.state.symbols) if self.collective.state else 0
        print(f"[DreamSystem] Initialized with {symbol_count} symbols")

    async def generate_dream(
        self,
        resident_id: str,
        emotional_state: Dict[str, float],
        consciousness_level: float = 0.5,
        archetype: Optional[NarrativeArchetype] = None,
        seed_symbols: Optional[List[str]] = None,
        record_to_journal: bool = True,
        apply_influences: bool = True,
    ) -> DreamSession:
        """
        Generate a complete dream for a resident.

        This is the main entry point for dream generation, coordinating
        all system components.

        Args:
            resident_id: Dreamer's ID
            emotional_state: Current emotional state dict
            consciousness_level: 0=individual, 1=universal
            archetype: Optional forced archetype
            seed_symbols: Optional starting symbols
            record_to_journal: Whether to record in journal
            apply_influences: Whether to generate influences

        Returns:
            Complete DreamSession with all data
        """
        if not self.is_initialized:
            await self.initialize()

        # Get pending waking experiences to incorporate
        pending_content = self.influence.get_pending_dream_content(resident_id)

        # Merge seed symbols with pending content
        all_seeds = list(seed_symbols or [])
        all_seeds.extend(pending_content.get("symbols", []))

        # Merge emotions
        combined_emotions = emotional_state.copy()
        for emotion, value in pending_content.get("emotions", {}).items():
            combined_emotions[emotion] = max(
                combined_emotions.get(emotion, 0),
                value
            )

        # Generate narrative
        narrative = self.generator.generate(
            resident_id=resident_id,
            emotional_state=combined_emotions,
            consciousness_level=consciousness_level,
            archetype=archetype,
            seed_symbols=all_seeds[:5] if all_seeds else None,
        )

        # Create journal entry
        journal_entry = None
        if record_to_journal:
            journal_entry = self.journal.record_dream(
                resident_id=resident_id,
                title=narrative.title,
                narrative=self._narrative_to_text(narrative),
                symbols=narrative.symbols_used,
                emotions=combined_emotions,
                archetype=narrative.archetype.value if narrative.archetype else None,
                mood=narrative.mood.value,
                personal_notes=narrative.jungian_interpretation,
            )

        # Generate influences
        influences = []
        if apply_influences:
            dream_data = {
                "id": journal_entry.entry_id if journal_entry else f"dream_{datetime.now().timestamp()}",
                "symbols_used": narrative.symbols_used,
                "archetype": narrative.archetype.value if narrative.archetype else None,
                "mood": narrative.mood.value,
            }
            influences = self.influence.process_dream(dream_data, resident_id)

        # Create session
        session = DreamSession(
            resident_id=resident_id,
            narrative=narrative,
            journal_entry=journal_entry,
            influences=influences,
        )

        self.dreams_today.append(session)
        return session

    def _narrative_to_text(self, narrative: DreamNarrative) -> str:
        """Convert narrative to plain text."""
        parts = []
        for act in narrative.acts:
            parts.append(f"{act.act_type.title()}: {act.description}")
        return " ".join(parts)

    async def generate_collective_dream(
        self,
        dreamer_ids: List[str],
        shared_emotional_state: Dict[str, float],
        consciousness_level: float = 0.7,  # Higher for collective
    ) -> List[DreamSession]:
        """
        Generate a shared dream for multiple residents.

        Args:
            dreamer_ids: List of participating dreamer IDs
            shared_emotional_state: Collective emotional state
            consciousness_level: Typically higher (more universal)

        Returns:
            List of DreamSessions (one per dreamer)
        """
        if not self.is_initialized:
            await self.initialize()

        # Generate one narrative for all
        narrative = self.generator.generate(
            resident_id=dreamer_ids[0],
            emotional_state=shared_emotional_state,
            consciousness_level=consciousness_level,
        )

        sessions = []

        for resident_id in dreamer_ids:
            # Record to each dreamer's journal
            entry = self.journal.record_dream(
                resident_id=resident_id,
                title=f"Shared: {narrative.title}",
                narrative=self._narrative_to_text(narrative),
                symbols=narrative.symbols_used,
                emotions=shared_emotional_state,
                archetype=narrative.archetype.value if narrative.archetype else None,
                mood=narrative.mood.value,
                shared_with=[d for d in dreamer_ids if d != resident_id],
            )

            sessions.append(DreamSession(
                resident_id=resident_id,
                narrative=narrative,
                journal_entry=entry,
                influences=[],  # Processed separately below
            ))

        # Process collective influence
        dream_data = {
            "id": f"collective_{datetime.now().timestamp()}",
            "symbols_used": narrative.symbols_used,
            "archetype": narrative.archetype.value if narrative.archetype else None,
            "mood": narrative.mood.value,
        }

        all_influences = self.influence.process_collective_dream(
            dream_data, dreamer_ids
        )

        # Distribute influences to sessions
        for session in sessions:
            session.influences = [
                inf for inf in all_influences
                if inf.resident_id == session.resident_id
            ]

        return sessions

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
            experience_type: Type (conversation, event, emotion, discovery)
            description: What happened
            emotional_impact: Emotional intensity by emotion
            people_involved: Other people involved

        Returns:
            Created WakingExperience
        """
        return self.influence.record_waking_experience(
            resident_id=resident_id,
            experience_type=experience_type,
            description=description,
            emotional_impact=emotional_impact,
            people_involved=people_involved,
        )

    def get_active_influences(self, resident_id: str) -> List[DreamInfluence]:
        """Get all active dream influences for a resident."""
        return self.influence.get_active_influences(resident_id)

    def get_trait_modifiers(self, resident_id: str) -> Dict[str, float]:
        """Get current trait modifiers from dream influences."""
        return self.influence.get_current_trait_modifiers(resident_id)

    def analyze_dreams(
        self,
        resident_id: str,
        lookback_days: int = 30
    ) -> DreamAnalysis:
        """Run full analysis on a resident's dream journal."""
        return self.journal.analyze(resident_id, lookback_days)

    def get_zeitgeist(self) -> EmotionalEssence:
        """Get current neighborhood emotional zeitgeist."""
        if self.collective and self.collective.state:
            return self.collective.state.neighborhood_zeitgeist
        return EmotionalEssence()

    def update_zeitgeist(self, recent_dreams: List[Dict[str, Any]]) -> EmotionalEssence:
        """Update neighborhood zeitgeist from recent dreams."""
        return self.collective.update_zeitgeist(recent_dreams)

    def get_neighborhood_mood(self) -> Dict[str, float]:
        """Get neighborhood mood from influence system."""
        return self.influence.get_neighborhood_mood()

    def detect_archetypes(
        self,
        resident_id: str,
        min_confidence: float = 0.6
    ) -> List[DetectedArchetype]:
        """Detect archetypal patterns in a resident's dream history."""
        entries = self.journal.get_entries(resident_id, limit=20)

        dream_history = [
            {
                "resident_id": e.resident_id,
                "symbols": e.symbols,
                "emotions": e.emotions,
                "mood": e.mood,
            }
            for e in entries
        ]

        return self.collective.detect_archetypes(dream_history, min_confidence)

    def get_system_state(self) -> DreamSystemState:
        """Get complete system state snapshot."""
        # Count archetypes from today's dreams
        archetype_counts: Dict[str, int] = {}
        collective_count = 0

        for session in self.dreams_today:
            if session.narrative.archetype:
                arch = session.narrative.archetype.value
                archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

            if session.journal_entry and session.journal_entry.shared_with:
                collective_count += 1

        return DreamSystemState(
            neighborhood_zeitgeist=self.get_zeitgeist(),
            neighborhood_mood=self.influence.get_neighborhood_mood(),
            active_archetypes=archetype_counts,
            total_dreams_today=len(self.dreams_today),
            collective_dream_count=collective_count,
        )

    def reset_daily_tracking(self):
        """Reset daily tracking (call at day boundary)."""
        self.dreams_today = []
        self.influence.decay_neighborhood_mood(hours=8)  # Night cycle decay

    async def save_state(self, path: str):
        """Save complete system state."""
        self.influence.save_state(f"{path}_influence.json")
        # Journal auto-saves if storage path configured

    async def load_state(self, path: str):
        """Load system state."""
        self.influence.load_state(f"{path}_influence.json")


async def create_dream_system(
    symbol_database_path: Optional[str] = None,
    journal_storage_path: Optional[str] = None,
) -> DreamSystem:
    """
    Create and initialize a complete Dream System.

    Args:
        symbol_database_path: Path to enriched symbol database
        journal_storage_path: Path for journal persistence

    Returns:
        Initialized DreamSystem ready for use
    """
    system = DreamSystem(
        symbol_database_path=symbol_database_path,
        journal_storage_path=journal_storage_path,
    )
    await system.initialize()
    return system


# Demo
if __name__ == "__main__":
    async def main():
        print("=" * 70)
        print("DREAM SYSTEM INTEGRATION DEMO")
        print("=" * 70)

        # Create system
        system = await create_dream_system()
        print(f"\n[System] Initialized successfully")

        # Simulate a day in the neighborhood
        residents = ["maya_001", "jorge_002", "elena_003"]

        # Morning: Record waking experiences
        print("\n--- Morning: Recording Experiences ---")

        system.record_waking_experience(
            resident_id="maya_001",
            experience_type="conversation",
            description="Deep talk with neighbor about community garden",
            emotional_impact={"connection": 0.7, "hope": 0.5},
            people_involved=["jorge_002"],
        )

        system.record_waking_experience(
            resident_id="jorge_002",
            experience_type="event",
            description="Received news about family illness",
            emotional_impact={"grief": 0.6, "fear": 0.4, "connection": 0.3},
        )

        print("Experiences recorded for maya and jorge")

        # Night: Generate dreams
        print("\n--- Night: Generating Dreams ---")

        # Maya's dream
        maya_dream = await system.generate_dream(
            resident_id="maya_001",
            emotional_state={"hope": 0.6, "connection": 0.5},
            consciousness_level=0.4,
        )
        print(f"\nMaya dreamed: '{maya_dream.narrative.title}'")
        print(f"  Archetype: {maya_dream.narrative.archetype.value}")
        print(f"  Mood: {maya_dream.narrative.mood.value}")
        print(f"  Influences generated: {len(maya_dream.influences)}")

        # Jorge's dream
        jorge_dream = await system.generate_dream(
            resident_id="jorge_002",
            emotional_state={"grief": 0.5, "fear": 0.4, "hope": 0.3},
            consciousness_level=0.5,
        )
        print(f"\nJorge dreamed: '{jorge_dream.narrative.title}'")
        print(f"  Archetype: {jorge_dream.narrative.archetype.value}")
        print(f"  Mood: {jorge_dream.narrative.mood.value}")

        # Collective dream
        print("\n--- Collective Dream ---")
        collective_sessions = await system.generate_collective_dream(
            dreamer_ids=["maya_001", "jorge_002", "elena_003"],
            shared_emotional_state={"transformation": 0.7, "connection": 0.6},
            consciousness_level=0.8,
        )
        print(f"Shared dream: '{collective_sessions[0].narrative.title}'")
        print(f"  Dreamers: {[s.resident_id for s in collective_sessions]}")

        # Morning after: Check influences
        print("\n--- Morning After: Active Influences ---")

        for resident_id in residents:
            influences = system.get_active_influences(resident_id)
            modifiers = system.get_trait_modifiers(resident_id)

            print(f"\n{resident_id}:")
            print(f"  Active influences: {len(influences)}")
            if modifiers:
                print(f"  Trait modifiers: ", end="")
                mods = [f"{k}: {v:+.2f}" for k, v in list(modifiers.items())[:3]]
                print(", ".join(mods))

        # Analysis
        print("\n--- Dream Analysis: Maya ---")
        analysis = system.analyze_dreams("maya_001", lookback_days=30)
        print(f"Dreams analyzed: {analysis.dream_count}")
        print(f"Individuation score: {analysis.individuation_score:.0%}")
        print(f"Shadow integration: {analysis.shadow_integration:.0%}")
        print("Insights:")
        for insight in analysis.insights[:2]:
            print(f"  - {insight}")

        # System state
        print("\n--- Neighborhood State ---")
        state = system.get_system_state()
        print(f"Dreams today: {state.total_dreams_today}")
        print(f"Collective dreams: {state.collective_dream_count}")
        print(f"Active archetypes: {state.active_archetypes}")

        mood = state.neighborhood_mood
        active_moods = {k: v for k, v in mood.items() if abs(v) > 0.01}
        if active_moods:
            print(f"Neighborhood mood: {active_moods}")

        print("\n" + "=" * 70)
        print("INTEGRATION DEMO COMPLETE")
        print("=" * 70)

    asyncio.run(main())