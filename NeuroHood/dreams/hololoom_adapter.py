"""
HoloLoom Adapter for NeuroHood Dreams - Phase 1 MVP

Bridges NeuroHood's dream system with HoloLoom's unified memory API.
Enables dreams to be stored, queried, and reflected upon using HoloLoom's
memory infrastructure.

Key Insight: Both systems use compatible 228D semantic embeddings and
identical decay rates (0.95^hours), making integration straightforward.

Usage:
    from HoloLoom import HoloLoom
    from NeuroHood.dreams.hololoom_adapter import store_dream_narrative, recall_dreams

    async with HoloLoom() as loom:
        # Store a dream
        memory = await store_dream_narrative(loom, narrative, "resident_001")

        # Query dreams
        water_dreams = await recall_dreams(loom, "dreams about water")

        # Reflect on dream quality
        await reflect_on_dream(loom, [memory], {"meaningful": True})

Author: Phase 1 MVP Integration
Date: November 2025
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

# Graceful degradation - NeuroHood works without HoloLoom
try:
    from HoloLoom import HoloLoom
    from HoloLoom.memory.protocol import Memory
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HoloLoom = None
    Memory = None
    HOLOLOOM_AVAILABLE = False

if TYPE_CHECKING:
    from narrative_generator import DreamNarrative, NarrativeAct
    from nap_mechanic import NapEvent


async def store_dream_narrative(
    loom: 'HoloLoom',
    narrative: 'DreamNarrative',
    resident_id: str,
    nap_event: Optional['NapEvent'] = None
) -> Optional['Memory']:
    """
    Store a complete dream narrative in HoloLoom memory.

    Converts a DreamNarrative into HoloLoom-compatible format and stores it
    with full metadata for later retrieval.

    Args:
        loom: HoloLoom instance (must be initialized)
        narrative: The dream narrative to store
        resident_id: ID of the resident who had the dream
        nap_event: Optional nap event for temporal context

    Returns:
        Memory object if stored successfully, None if HoloLoom unavailable

    Example:
        async with HoloLoom() as loom:
            narrative = generator.generate(resident_id="r001", ...)
            memory = await store_dream_narrative(loom, narrative, "r001")
    """
    if not HOLOLOOM_AVAILABLE or loom is None:
        return None

    # Build rich content string from narrative
    content = _narrative_to_content(narrative)

    # Build context with dream metadata
    context = {
        "type": "dream_narrative",
        "resident_id": resident_id,
        "archetype": narrative.archetype.value,
        "mood": narrative.mood.value,
        "symbols_used": narrative.symbols_used,
        "literary_inspirations": narrative.literary_inspirations,
        "jungian_interpretation": narrative.jungian_interpretation,
        "generation_timestamp": narrative.generation_timestamp.isoformat(),
        "act_count": len(narrative.acts),
    }

    # Add nap event context if available
    if nap_event:
        context["nap_event"] = {
            "start_time": nap_event.start_time.isoformat() if hasattr(nap_event, 'start_time') else None,
            "duration_minutes": getattr(nap_event, 'duration_minutes', None),
            "quality": getattr(nap_event, 'quality', None),
        }

    # Store in HoloLoom
    memory = await loom.experience(content, context=context)

    return memory


async def store_dream_scene(
    loom: 'HoloLoom',
    scene: 'NarrativeAct',
    resident_id: str,
    dream_title: Optional[str] = None
) -> Optional['Memory']:
    """
    Store a single dream scene/act in HoloLoom memory.

    Useful for storing individual dream segments when full narrative
    isn't available or when you want finer-grained retrieval.

    Args:
        loom: HoloLoom instance
        scene: A single NarrativeAct (dream scene)
        resident_id: ID of the resident
        dream_title: Optional title of parent dream

    Returns:
        Memory object if stored successfully

    Example:
        for act in narrative.acts:
            await store_dream_scene(loom, act, "r001", narrative.title)
    """
    if not HOLOLOOM_AVAILABLE or loom is None:
        return None

    # Build content from scene
    content = f"Dream Scene (Act {scene.act_number}): {scene.description}"

    # Build context
    context = {
        "type": "dream_scene",
        "resident_id": resident_id,
        "act_number": scene.act_number,
        "act_type": scene.act_type,
        "emotional_intensity": scene.emotional_intensity,
        "symbols": [s.get("symbol_id", str(s)) for s in scene.symbols] if scene.symbols else [],
    }

    if scene.literary_echo:
        context["literary_echo"] = scene.literary_echo

    if dream_title:
        context["dream_title"] = dream_title

    memory = await loom.experience(content, context=context)

    return memory


async def recall_dreams(
    loom: 'HoloLoom',
    query: str,
    resident_id: Optional[str] = None,
    limit: int = 10
) -> List['Memory']:
    """
    Recall dreams from HoloLoom memory.

    Query can be semantic (e.g., "dreams about water") or emotional
    (e.g., "fearful nightmares"). Results are ranked by relevance.

    Args:
        loom: HoloLoom instance
        query: Search query (semantic or emotional)
        resident_id: Optional filter by resident
        limit: Maximum number of dreams to return

    Returns:
        List of Memory objects containing dream content

    Example:
        # Semantic search
        water_dreams = await recall_dreams(loom, "dreams about water")

        # Resident-specific search
        my_dreams = await recall_dreams(loom, "transformation", resident_id="r001")
    """
    if not HOLOLOOM_AVAILABLE or loom is None:
        return []

    # Add dream context to query for better retrieval
    enhanced_query = f"dream: {query}"

    # Recall from HoloLoom
    memories = await loom.recall(enhanced_query, limit=limit)

    # Filter by resident if specified
    if resident_id:
        memories = [
            m for m in memories
            if m.context.get("resident_id") == resident_id
        ]

    # Filter to only dream-related memories
    dream_types = {"dream_narrative", "dream_scene"}
    memories = [
        m for m in memories
        if m.context.get("type") in dream_types
    ]

    return memories[:limit]


async def recall_resident_dreams(
    loom: 'HoloLoom',
    resident_id: str,
    limit: int = 10
) -> List['Memory']:
    """
    Recall all dreams for a specific resident.

    Returns dreams ordered by relevance/recency.

    Args:
        loom: HoloLoom instance
        resident_id: ID of the resident
        limit: Maximum dreams to return

    Returns:
        List of Memory objects for that resident's dreams

    Example:
        my_dreams = await recall_resident_dreams(loom, "resident_001")
        for dream in my_dreams:
            print(f"Dream: {dream.text[:100]}...")
    """
    return await recall_dreams(loom, "dream", resident_id=resident_id, limit=limit)


async def reflect_on_dream(
    loom: 'HoloLoom',
    dream_memories: List['Memory'],
    feedback: Dict[str, Any]
) -> None:
    """
    Reflect on dream quality to improve future retrieval.

    Provides feedback to HoloLoom's learning system about dream
    relevance, meaningfulness, and emotional impact.

    Args:
        loom: HoloLoom instance
        dream_memories: Dreams to reflect on
        feedback: Feedback dictionary with signals like:
            - {"meaningful": True/False}
            - {"emotional_impact": 0.8}
            - {"helped_insight": True}
            - {"relevant_to_waking": True}

    Example:
        dreams = await recall_dreams(loom, "water symbolism")
        # User found these dreams insightful
        await reflect_on_dream(loom, dreams, {
            "meaningful": True,
            "emotional_impact": 0.8,
            "helped_insight": True
        })
    """
    if not HOLOLOOM_AVAILABLE or loom is None:
        return

    # Map dream-specific feedback to HoloLoom's format
    hololoom_feedback = {
        "helpful": feedback.get("meaningful", feedback.get("helpful", True)),
        "relevance": feedback.get("emotional_impact", 0.8),
    }

    # Add any custom feedback signals
    if "selected" in feedback:
        hololoom_feedback["selected"] = feedback["selected"]

    await loom.reflect(dream_memories, feedback=hololoom_feedback)


# =============================================================================
# Helper Functions
# =============================================================================

def _narrative_to_content(narrative: 'DreamNarrative') -> str:
    """
    Convert a DreamNarrative to a rich text content string.

    Preserves narrative structure while creating searchable text.
    """
    parts = [
        f"Dream: {narrative.title}",
        f"A {narrative.mood.value} dream of {narrative.archetype.value}.",
        "",
    ]

    # Add each act
    for act in narrative.acts:
        parts.append(f"Act {act.act_number} ({act.act_type}): {act.description}")
        if act.literary_echo:
            parts.append(f"[Echoing: {act.literary_echo}]")
        parts.append("")

    # Add interpretation
    if narrative.jungian_interpretation:
        parts.append(f"Interpretation: {narrative.jungian_interpretation}")

    # Add symbols for searchability
    if narrative.symbols_used:
        parts.append(f"Symbols: {', '.join(narrative.symbols_used)}")

    return "\n".join(parts)


def is_hololoom_available() -> bool:
    """
    Check if HoloLoom integration is available.

    Returns:
        True if HoloLoom can be imported, False otherwise
    """
    return HOLOLOOM_AVAILABLE


# =============================================================================
# Convenience: Context Manager Pattern
# =============================================================================

class DreamMemoryBridge:
    """
    Context manager for HoloLoom dream integration.

    Provides convenient async context manager pattern for managing
    HoloLoom lifecycle during dream operations.

    Usage:
        async with DreamMemoryBridge() as bridge:
            await bridge.store(narrative, "resident_001")
            dreams = await bridge.recall("water")
    """

    def __init__(self):
        self._loom: Optional['HoloLoom'] = None

    async def __aenter__(self) -> 'DreamMemoryBridge':
        """Enter context and initialize HoloLoom."""
        if not HOLOLOOM_AVAILABLE:
            return self

        self._loom = HoloLoom()
        await self._loom.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup HoloLoom."""
        if self._loom:
            await self._loom.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def available(self) -> bool:
        """Check if HoloLoom is available."""
        return HOLOLOOM_AVAILABLE and self._loom is not None

    async def store(
        self,
        narrative: 'DreamNarrative',
        resident_id: str,
        nap_event: Optional['NapEvent'] = None
    ) -> Optional['Memory']:
        """Store a dream narrative."""
        return await store_dream_narrative(self._loom, narrative, resident_id, nap_event)

    async def recall(
        self,
        query: str,
        resident_id: Optional[str] = None,
        limit: int = 10
    ) -> List['Memory']:
        """Recall dreams matching query."""
        return await recall_dreams(self._loom, query, resident_id, limit)

    async def recall_for_resident(
        self,
        resident_id: str,
        limit: int = 10
    ) -> List['Memory']:
        """Recall all dreams for a resident."""
        return await recall_resident_dreams(self._loom, resident_id, limit)

    async def reflect(
        self,
        dreams: List['Memory'],
        feedback: Dict[str, Any]
    ) -> None:
        """Reflect on dream quality."""
        return await reflect_on_dream(self._loom, dreams, feedback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get HoloLoom memory metrics."""
        if not self._loom:
            return {"available": False}

        metrics = self._loom.get_metrics()
        metrics["available"] = True
        return metrics
