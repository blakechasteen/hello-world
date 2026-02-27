"""
DreamSpinner - NeuroHood Dream Narrative to MemoryShard Converter
=================================================================

Converts NeuroHood DreamNarrative objects into HoloLoom MemoryShards for
memory system integration. Part of the NeuroHood-HoloLoom Phase 2 integration.

Key Features:
- Converts DreamNarrative to structured MemoryShards
- Extracts entities from dream content (symbols, literary references)
- Extracts motifs (archetypes, moods, themes)
- Importance scoring based on emotional intensity and symbol density
- Streaming support for batch narrative processing
- Graceful degradation when NeuroHood not available

Philosophy:
"Dreams are memories of the unconscious. The DreamSpinner weaves these
ephemeral experiences into the fabric of HoloLoom's memory system."

Author: Phase 2 Integration
Date: November 2025
"""

from typing import List, Optional, Dict, Any, AsyncIterator, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import time
import logging

from hololoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinResult,
    ImportanceSignals,
    ImportanceScore,
)
from hololoom.protocols.types import MemoryShard

# Graceful import of NeuroHood types
try:
    from NeuroHood.dreams.narrative_generator import (
        DreamNarrative,
        NarrativeAct,
        NarrativeArchetype,
        DreamMood
    )
    NEUROHOOD_AVAILABLE = True
except ImportError:
    NEUROHOOD_AVAILABLE = False
    # Define minimal stubs for type checking
    DreamNarrative = None
    NarrativeAct = None
    NarrativeArchetype = None
    DreamMood = None


logger = logging.getLogger(__name__)


@dataclass
class DreamSpinnerConfig:
    """Configuration for DreamSpinner."""
    # Shard generation
    shard_per_act: bool = True  # Create separate shard for each act
    include_interpretation: bool = True  # Include Jungian interpretation
    include_literary_echoes: bool = True  # Include literary references

    # Importance scoring weights
    emotional_intensity_weight: float = 0.3
    symbol_density_weight: float = 0.25
    archetype_significance_weight: float = 0.25
    literary_depth_weight: float = 0.2

    # Entity extraction
    max_entities_per_shard: int = 20
    max_motifs_per_shard: int = 10

    # Processing
    importance_threshold: float = 0.3


class DreamSpinner(BaseSpinner):
    """
    Spinner that converts NeuroHood DreamNarrative objects to MemoryShards.

    Converts dream narratives with their full structure (acts, symbols,
    interpretations) into HoloLoom-compatible memory shards.

    Example:
        >>> spinner = DreamSpinner()
        >>> narrative = create_dream_narrative(...)
        >>> result = await spinner.spin(narrative)
        >>> print(f"Created {result.shard_count} shards")
        Created 4 shards

        >>> # Process multiple narratives
        >>> narratives = [narrative1, narrative2, narrative3]
        >>> result = await spinner.spin(narratives)
        >>> print(f"Total shards: {result.shard_count}")
    """

    def __init__(
        self,
        config: Optional[DreamSpinnerConfig] = None,
        importance_threshold: float = 0.3,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize DreamSpinner.

        Args:
            config: Optional spinner configuration
            importance_threshold: Min importance to include (0.0-1.0)
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(
            name="dream",
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir
        )

        self.config = config or DreamSpinnerConfig()

        # Archetype significance scores (from Jungian psychology)
        self._archetype_significance = {
            "journey": 0.8,       # Hero's journey - universal
            "confrontation": 0.9, # Shadow work - transformative
            "transformation": 0.95,  # Death-rebirth - most significant
            "discovery": 0.75,
            "protection": 0.7,
            "creation": 0.85,
        }

        # Mood intensity scores
        self._mood_intensity = {
            "nightmare": 0.9,      # High emotional impact
            "wonder": 0.7,
            "mysterious": 0.6,
            "transformative": 0.85,
            "peaceful": 0.5,
            "anxious": 0.75,
        }

        logger.info(f"DreamSpinner initialized (NeuroHood available: {NEUROHOOD_AVAILABLE})")

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get spinner capabilities."""
        return SpinnerCapabilities(
            basic_processing=NEUROHOOD_AVAILABLE,
            streaming=True,  # Supports streaming multiple narratives
            incremental=False,  # Dreams are atomic, no incremental updates
            batch_processing=True,  # Can process narrative batches
            importance_scoring=True,  # Dream-specific importance scoring
            entity_extraction=True,  # Symbol/character extraction
            motif_extraction=True,  # Archetype/theme extraction
            embeddings=False,  # Embeddings generated by HoloLoom
            multimodal=False,  # Text-only dreams (for now)
            supported_formats=["dream_narrative", "narrative_act"]
        )

    def is_available(self) -> bool:
        """Check if NeuroHood is available."""
        return NEUROHOOD_AVAILABLE

    async def _spin_impl(
        self,
        source: Any,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Convert dream narrative(s) to MemoryShards.

        Args:
            source: DreamNarrative or List[DreamNarrative]
            **kwargs: Optional resident_id, session_id

        Returns:
            List of MemoryShards
        """
        if not NEUROHOOD_AVAILABLE:
            logger.warning("NeuroHood not available, returning empty result")
            return []

        resident_id = kwargs.get("resident_id", "unknown")
        session_id = kwargs.get("session_id", "unknown")

        # Handle single narrative or list
        if isinstance(source, list):
            narratives = source
        else:
            narratives = [source]

        shards = []

        for narrative in narratives:
            narrative_shards = self._narrative_to_shards(
                narrative,
                resident_id=resident_id,
                session_id=session_id
            )
            shards.extend(narrative_shards)

        return shards

    def _narrative_to_shards(
        self,
        narrative: 'DreamNarrative',
        resident_id: str,
        session_id: str
    ) -> List[MemoryShard]:
        """Convert a single DreamNarrative to MemoryShards."""
        shards = []

        # Get timestamp
        timestamp = narrative.generation_timestamp.timestamp() if hasattr(narrative, 'generation_timestamp') else time.time()

        # Create main narrative shard
        main_shard = self._create_narrative_shard(
            narrative,
            resident_id=resident_id,
            session_id=session_id,
            timestamp=timestamp
        )
        shards.append(main_shard)

        # Create per-act shards if configured
        if self.config.shard_per_act and hasattr(narrative, 'acts'):
            for act in narrative.acts:
                act_shard = self._create_act_shard(
                    act,
                    narrative=narrative,
                    resident_id=resident_id,
                    session_id=session_id,
                    timestamp=timestamp
                )
                shards.append(act_shard)

        return shards

    def _create_narrative_shard(
        self,
        narrative: 'DreamNarrative',
        resident_id: str,
        session_id: str,
        timestamp: float
    ) -> MemoryShard:
        """Create MemoryShard from full DreamNarrative."""
        # Build content text
        parts = [f"Dream: {narrative.title}"]

        # Add archetype and mood
        archetype = narrative.archetype.value if hasattr(narrative.archetype, 'value') else str(narrative.archetype)
        mood = narrative.mood.value if hasattr(narrative.mood, 'value') else str(narrative.mood)
        parts.append(f"A {mood} dream of {archetype}.")
        parts.append("")

        # Add acts
        if hasattr(narrative, 'acts'):
            for act in narrative.acts:
                parts.append(f"Act {act.act_number} ({act.act_type}): {act.description}")
                if hasattr(act, 'literary_echo') and act.literary_echo:
                    parts.append(f"[Echoing: {act.literary_echo}]")
                parts.append("")

        # Add interpretation
        if self.config.include_interpretation and hasattr(narrative, 'jungian_interpretation'):
            parts.append(f"Interpretation: {narrative.jungian_interpretation}")

        # Add symbols
        if hasattr(narrative, 'symbols_used') and narrative.symbols_used:
            parts.append(f"Symbols: {', '.join(narrative.symbols_used)}")

        text = "\n".join(parts)

        # Extract entities
        entities = self._extract_entities(narrative)

        # Extract motifs
        motifs = self._extract_motifs(narrative)

        # Score importance
        importance = self.score_importance(narrative)

        # Build metadata
        metadata = {
            "type": "dream_narrative",
            "spinner": "dream",
            "resident_id": resident_id,
            "session_id": session_id,
            "archetype": archetype,
            "mood": mood,
            "act_count": len(narrative.acts) if hasattr(narrative, 'acts') else 0,
            "importance_score": importance.score,
            "importance_reason": importance.reason,
            "symbols_used": narrative.symbols_used if hasattr(narrative, 'symbols_used') else [],
            "literary_inspirations": narrative.literary_inspirations if hasattr(narrative, 'literary_inspirations') else [],
            "protagonist_state": narrative.protagonist_state if hasattr(narrative, 'protagonist_state') else {},
        }

        return MemoryShard(
            id=f"dream_{resident_id}_{int(timestamp)}",
            text=text,
            entities=entities[:self.config.max_entities_per_shard],
            motifs=motifs[:self.config.max_motifs_per_shard],
            episode=f"dream_{session_id}",
            timestamp=timestamp,
            metadata=metadata
        )

    def _create_act_shard(
        self,
        act: 'NarrativeAct',
        narrative: 'DreamNarrative',
        resident_id: str,
        session_id: str,
        timestamp: float
    ) -> MemoryShard:
        """Create MemoryShard from a single NarrativeAct."""
        # Build content
        parts = [
            f"Dream Scene (Act {act.act_number}): {act.description}",
        ]

        if hasattr(act, 'literary_echo') and act.literary_echo:
            parts.append(f"Literary echo: {act.literary_echo}")

        text = "\n".join(parts)

        # Extract entities from act symbols
        entities = []
        if hasattr(act, 'symbols') and act.symbols:
            for sym in act.symbols:
                if isinstance(sym, dict):
                    entities.append(sym.get('symbol_id', str(sym)))
                else:
                    entities.append(str(sym))

        # Add literary echo as entity if present
        if hasattr(act, 'literary_echo') and act.literary_echo:
            entities.append(act.literary_echo)

        # Motifs based on act type and emotional intensity
        motifs = [act.act_type]
        if hasattr(act, 'emotional_intensity') and act.emotional_intensity > 0.7:
            motifs.append("high_intensity")

        # Score importance for this act
        act_importance = self._score_act_importance(act)

        # Build metadata
        metadata = {
            "type": "dream_scene",
            "spinner": "dream",
            "resident_id": resident_id,
            "session_id": session_id,
            "dream_title": narrative.title if hasattr(narrative, 'title') else "Unknown",
            "act_number": act.act_number,
            "act_type": act.act_type,
            "emotional_intensity": act.emotional_intensity if hasattr(act, 'emotional_intensity') else 0.5,
            "importance_score": act_importance,
            "symbols": [s.get('symbol_id', str(s)) if isinstance(s, dict) else str(s)
                       for s in (act.symbols if hasattr(act, 'symbols') else [])],
        }

        if hasattr(act, 'literary_echo') and act.literary_echo:
            metadata["literary_echo"] = act.literary_echo

        return MemoryShard(
            id=f"dream_act_{resident_id}_{int(timestamp)}_{act.act_number}",
            text=text,
            entities=entities[:self.config.max_entities_per_shard],
            motifs=motifs[:self.config.max_motifs_per_shard],
            episode=f"dream_{session_id}",
            timestamp=timestamp,
            metadata=metadata
        )

    def _extract_entities(self, narrative: 'DreamNarrative') -> List[str]:
        """Extract entities from dream narrative."""
        entities = []

        # Symbols as entities
        if hasattr(narrative, 'symbols_used'):
            entities.extend(narrative.symbols_used)

        # Literary inspirations as entities
        if hasattr(narrative, 'literary_inspirations'):
            entities.extend(narrative.literary_inspirations)

        # Extract symbols from acts
        if hasattr(narrative, 'acts'):
            for act in narrative.acts:
                if hasattr(act, 'symbols') and act.symbols:
                    for sym in act.symbols:
                        if isinstance(sym, dict):
                            sym_id = sym.get('symbol_id')
                            if sym_id:
                                entities.append(sym_id)
                        else:
                            entities.append(str(sym))

                # Literary echoes from acts
                if hasattr(act, 'literary_echo') and act.literary_echo:
                    entities.append(act.literary_echo)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)

        return unique

    def _extract_motifs(self, narrative: 'DreamNarrative') -> List[str]:
        """Extract motifs from dream narrative."""
        motifs = []

        # Archetype as primary motif
        if hasattr(narrative, 'archetype'):
            archetype = narrative.archetype.value if hasattr(narrative.archetype, 'value') else str(narrative.archetype)
            motifs.append(f"archetype:{archetype}")

        # Mood as motif
        if hasattr(narrative, 'mood'):
            mood = narrative.mood.value if hasattr(narrative.mood, 'value') else str(narrative.mood)
            motifs.append(f"mood:{mood}")

        # Protagonist emotional states as motifs
        if hasattr(narrative, 'protagonist_state'):
            for state, intensity in narrative.protagonist_state.items():
                if intensity > 0.5:  # Only significant states
                    motifs.append(f"state:{state}")

        # Extract themes from symbols
        if hasattr(narrative, 'symbols_used'):
            for symbol in narrative.symbols_used[:5]:  # Top 5 symbols
                motifs.append(f"symbol:{symbol}")

        return motifs

    def score_importance(self, data: Any) -> ImportanceScore:
        """
        Score importance of a dream narrative.

        Uses dream-specific signals:
        - Emotional intensity of acts
        - Symbol density (more symbols = richer dream)
        - Archetype significance (transformation > journey > ...)
        - Literary depth (literary echoes)
        """
        if not NEUROHOOD_AVAILABLE or data is None:
            return ImportanceScore(
                score=0.5,
                signals=ImportanceSignals(),
                reason="default importance (NeuroHood unavailable)"
            )

        narrative = data

        # Calculate signal scores

        # 1. Emotional intensity (average across acts)
        emotional_scores = []
        if hasattr(narrative, 'acts'):
            for act in narrative.acts:
                if hasattr(act, 'emotional_intensity'):
                    emotional_scores.append(act.emotional_intensity)

        emotional_intensity = sum(emotional_scores) / len(emotional_scores) if emotional_scores else 0.5

        # 2. Symbol density
        symbol_count = len(narrative.symbols_used) if hasattr(narrative, 'symbols_used') else 0
        symbol_density = min(1.0, symbol_count / 10)  # Max score at 10+ symbols

        # 3. Archetype significance
        archetype = narrative.archetype.value if hasattr(narrative, 'archetype') and hasattr(narrative.archetype, 'value') else "unknown"
        archetype_sig = self._archetype_significance.get(archetype, 0.5)

        # 4. Literary depth
        literary_count = 0
        if hasattr(narrative, 'literary_inspirations'):
            literary_count += len(narrative.literary_inspirations)
        if hasattr(narrative, 'acts'):
            for act in narrative.acts:
                if hasattr(act, 'literary_echo') and act.literary_echo:
                    literary_count += 1
        literary_depth = min(1.0, literary_count / 5)  # Max score at 5+ references

        # Create signals
        signals = ImportanceSignals(
            technical_score=archetype_sig,  # Archetype as "technical content"
            structural_score=symbol_density,  # Symbol richness as structure
            engagement_score=emotional_intensity,  # Emotional intensity as engagement
            reference_score=literary_depth,  # Literary echoes as references
            custom_signals={
                'emotional_intensity': emotional_intensity,
                'symbol_density': symbol_density,
                'archetype_significance': archetype_sig,
                'literary_depth': literary_depth
            }
        )

        # Weighted combination
        total = (
            emotional_intensity * self.config.emotional_intensity_weight +
            symbol_density * self.config.symbol_density_weight +
            archetype_sig * self.config.archetype_significance_weight +
            literary_depth * self.config.literary_depth_weight
        )

        # Generate explanation
        factors = []
        if emotional_intensity > 0.7:
            factors.append("high emotional intensity")
        if symbol_density > 0.6:
            factors.append("rich symbolism")
        if archetype_sig > 0.8:
            factors.append(f"significant archetype ({archetype})")
        if literary_depth > 0.5:
            factors.append("literary depth")

        reason = " + ".join(factors) if factors else "moderate dream significance"

        return ImportanceScore(
            score=max(0.0, min(1.0, total)),
            signals=signals,
            reason=reason
        )

    def _score_act_importance(self, act: 'NarrativeAct') -> float:
        """Score importance of a single act."""
        score = 0.5  # Base score

        # Emotional intensity
        if hasattr(act, 'emotional_intensity'):
            score = 0.4 * act.emotional_intensity + 0.3 * score

        # Climax acts are more important
        if hasattr(act, 'act_type') and act.act_type == 'climax':
            score += 0.2

        # Literary echoes add importance
        if hasattr(act, 'literary_echo') and act.literary_echo:
            score += 0.1

        # Symbol count
        if hasattr(act, 'symbols') and act.symbols:
            score += min(0.2, len(act.symbols) * 0.05)

        return min(1.0, max(0.0, score))

    async def spin_stream(
        self,
        source: Any,
        **kwargs
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream MemoryShards from narratives as they're processed.

        Args:
            source: List of DreamNarrative objects
            **kwargs: Optional resident_id, session_id

        Yields:
            MemoryShards one at a time
        """
        if not NEUROHOOD_AVAILABLE:
            return

        narratives = source if isinstance(source, list) else [source]
        resident_id = kwargs.get("resident_id", "unknown")
        session_id = kwargs.get("session_id", "unknown")

        for narrative in narratives:
            shards = self._narrative_to_shards(
                narrative,
                resident_id=resident_id,
                session_id=session_id
            )
            for shard in shards:
                # Apply importance filter
                importance = shard.metadata.get('importance_score', 0.5)
                if importance >= self.importance_threshold:
                    yield shard


# ============================================================================
# Factory Functions
# ============================================================================

def create_dream_spinner(
    config: Optional[DreamSpinnerConfig] = None,
    importance_threshold: float = 0.3
) -> DreamSpinner:
    """
    Create a DreamSpinner instance.

    Args:
        config: Optional configuration
        importance_threshold: Min importance to include

    Returns:
        Configured DreamSpinner

    Example:
        >>> spinner = create_dream_spinner()
        >>> result = await spinner.spin(narrative)
    """
    return DreamSpinner(
        config=config,
        importance_threshold=importance_threshold
    )


def is_dream_spinner_available() -> bool:
    """Check if DreamSpinner is available (NeuroHood installed)."""
    return NEUROHOOD_AVAILABLE


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'DreamSpinner',
    'DreamSpinnerConfig',
    'create_dream_spinner',
    'is_dream_spinner_available',
]
