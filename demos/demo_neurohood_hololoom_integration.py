"""
Demo: NeuroHood Dreams ↔ HoloLoom Integration

Demonstrates the Phase 1 MVP integration between NeuroHood's dream system
and HoloLoom's unified memory API.

Key Features:
- Store dream narratives in HoloLoom memory
- Query dreams semantically ("dreams about water")
- Reflect on dream quality for improved retrieval
- Graceful degradation when HoloLoom unavailable

Run with:
    cd mythRL
    PYTHONPATH=. python demos/demo_neurohood_hololoom_integration.py

Author: Phase 1 MVP Demo
Date: November 2025
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "NeuroHood" / "dreams"))


# =============================================================================
# Mock Classes (to run demo without full NeuroHood setup)
# =============================================================================

class NarrativeArchetype(Enum):
    JOURNEY = "journey"
    CONFRONTATION = "confrontation"
    TRANSFORMATION = "transformation"


class DreamMood(Enum):
    NIGHTMARE = "nightmare"
    WONDER = "wonder"
    MYSTERIOUS = "mysterious"
    TRANSFORMATIVE = "transformative"


@dataclass
class NarrativeAct:
    act_number: int
    act_type: str
    description: str
    symbols: List[Dict]
    emotional_intensity: float
    literary_echo: Optional[str] = None


@dataclass
class DreamNarrative:
    title: str
    archetype: NarrativeArchetype
    mood: DreamMood
    acts: List[NarrativeAct]
    protagonist_state: Dict[str, float]
    symbols_used: List[str]
    literary_inspirations: List[str]
    jungian_interpretation: str
    generation_timestamp: datetime = field(default_factory=datetime.now)


def create_sample_narratives() -> List[DreamNarrative]:
    """Create sample dream narratives for demo."""
    return [
        DreamNarrative(
            title="Journey Through the Deep Ocean",
            archetype=NarrativeArchetype.JOURNEY,
            mood=DreamMood.WONDER,
            acts=[
                NarrativeAct(
                    act_number=1,
                    act_type="setup",
                    description="You find yourself in a vast ocean, waves crashing around you. "
                               "The water is warm and welcoming, despite its depth.",
                    symbols=[{"symbol_id": "ocean", "description": "vast sea"}],
                    emotional_intensity=0.6,
                    literary_echo="The Odyssey"
                ),
                NarrativeAct(
                    act_number=2,
                    act_type="climax",
                    description="A great whale emerges from the depths, its eyes ancient and wise. "
                               "It speaks without words, communicating through pure emotion.",
                    symbols=[{"symbol_id": "whale", "description": "ancient creature"}],
                    emotional_intensity=0.9
                ),
                NarrativeAct(
                    act_number=3,
                    act_type="resolution",
                    description="You ride upon its back to a shore of golden light. "
                               "The journey has changed you.",
                    symbols=[{"symbol_id": "golden_light", "description": "radiance of dawn"}],
                    emotional_intensity=0.7
                )
            ],
            protagonist_state={"hope": 0.7, "wonder": 0.8, "transformation": 0.5},
            symbols_used=["ocean", "whale", "golden_light"],
            literary_inspirations=["The Odyssey", "Moby Dick", "Life of Pi"],
            jungian_interpretation="This journey represents the individuation process - "
                                   "descending into the unconscious (ocean) and emerging transformed."
        ),
        DreamNarrative(
            title="The Shadow in the Mirror",
            archetype=NarrativeArchetype.CONFRONTATION,
            mood=DreamMood.NIGHTMARE,
            acts=[
                NarrativeAct(
                    act_number=1,
                    act_type="setup",
                    description="You stand before a mirror in a dark room. Your reflection "
                               "moves independently, smiling when you frown.",
                    symbols=[{"symbol_id": "mirror", "description": "gateway to self"}],
                    emotional_intensity=0.7
                ),
                NarrativeAct(
                    act_number=2,
                    act_type="climax",
                    description="The reflection steps out of the mirror, embodying everything "
                               "you've hidden from yourself. It demands acknowledgment.",
                    symbols=[{"symbol_id": "shadow_self", "description": "hidden aspect"}],
                    emotional_intensity=0.95,
                    literary_echo="Dr. Jekyll and Mr. Hyde"
                ),
                NarrativeAct(
                    act_number=3,
                    act_type="resolution",
                    description="Instead of fighting, you embrace your shadow. It merges back "
                               "into you, but now you feel whole rather than fractured.",
                    symbols=[{"symbol_id": "integration", "description": "becoming whole"}],
                    emotional_intensity=0.6
                )
            ],
            protagonist_state={"fear": 0.6, "conflict": 0.7, "transformation": 0.8},
            symbols_used=["mirror", "shadow_self", "integration"],
            literary_inspirations=["Dr. Jekyll and Mr. Hyde", "Fight Club"],
            jungian_interpretation="Classic shadow integration - confronting and accepting "
                                   "the disowned aspects of the psyche leads to wholeness."
        ),
        DreamNarrative(
            title="The Burning Phoenix",
            archetype=NarrativeArchetype.TRANSFORMATION,
            mood=DreamMood.TRANSFORMATIVE,
            acts=[
                NarrativeAct(
                    act_number=1,
                    act_type="setup",
                    description="You feel yourself catching fire, but it doesn't hurt. "
                               "The flames consume everything you thought you were.",
                    symbols=[{"symbol_id": "fire", "description": "purifying flames"}],
                    emotional_intensity=0.8
                ),
                NarrativeAct(
                    act_number=2,
                    act_type="climax",
                    description="From the ashes, a phoenix rises - and you realize you are "
                               "the phoenix. Wings of golden fire spread from your back.",
                    symbols=[{"symbol_id": "phoenix", "description": "rebirth symbol"}],
                    emotional_intensity=0.95,
                    literary_echo="Harry Potter and the Chamber of Secrets"
                ),
                NarrativeAct(
                    act_number=3,
                    act_type="resolution",
                    description="You soar above the clouds, leaving behind what was to embrace "
                               "what can be. The transformation is complete.",
                    symbols=[{"symbol_id": "flight", "description": "transcendence"}],
                    emotional_intensity=0.85
                )
            ],
            protagonist_state={"transformation": 0.9, "hope": 0.8, "fear": 0.2},
            symbols_used=["fire", "phoenix", "flight"],
            literary_inspirations=["Harry Potter", "Fahrenheit 451", "Phoenix mythology"],
            jungian_interpretation="The death-rebirth archetype - transformation requires "
                                   "the destruction of the old self to birth the new."
        )
    ]


async def demo():
    """Run the integration demo."""
    print("=" * 70)
    print("NEUROHOOD DREAMS <-> HOLOLOOM INTEGRATION DEMO")
    print("Phase 1 MVP - November 2025")
    print("=" * 70)

    # Import adapter
    from NeuroHood.dreams.hololoom_adapter import (
        DreamMemoryBridge,
        is_hololoom_available,
        _narrative_to_content
    )

    print(f"\n✓ HoloLoom available: {is_hololoom_available()}")

    # Create sample narratives
    narratives = create_sample_narratives()
    print(f"✓ Created {len(narratives)} sample dream narratives")

    # Show content conversion
    print("\n" + "-" * 70)
    print("CONTENT CONVERSION EXAMPLE")
    print("-" * 70)
    content = _narrative_to_content(narratives[0])
    print(content[:500] + "...")

    # Use the bridge
    print("\n" + "-" * 70)
    print("HOLOLOOM MEMORY INTEGRATION")
    print("-" * 70)

    async with DreamMemoryBridge() as bridge:
        if not bridge.available:
            print("\n⚠️  HoloLoom not available - showing graceful degradation")
            print("   (Dreams would still work, just without HoloLoom memory)")

            # Show what would happen
            for narrative in narratives:
                memory = await bridge.store(narrative, "demo_resident")
                print(f"   Store '{narrative.title}': {memory}")  # Will be None

            dreams = await bridge.recall("ocean journey")
            print(f"   Recall 'ocean journey': {dreams}")  # Will be []

            print("\n✓ Graceful degradation working - no crashes!")
        else:
            print("\n✓ HoloLoom connected!")

            # Store all narratives
            print("\n[1] Storing dreams in HoloLoom memory...")
            stored_memories = []
            for narrative in narratives:
                memory = await bridge.store(narrative, "demo_resident")
                stored_memories.append(memory)
                print(f"   ✓ Stored: '{narrative.title}'")
                print(f"     Memory ID: {memory.id[:50]}...")

            # Query semantically
            print("\n[2] Semantic dream queries...")

            queries = [
                "dreams about water and ocean",
                "shadow and mirror dreams",
                "transformation and rebirth",
                "confronting fears",
            ]

            for query in queries:
                results = await bridge.recall(query, limit=2)
                print(f"\n   Query: '{query}'")
                print(f"   Found: {len(results)} dream(s)")
                for r in results[:2]:
                    title = r.context.get("archetype", "unknown")
                    print(f"     - {title}: {r.text[:60]}...")

            # Reflect on quality
            print("\n[3] Reflecting on dream quality...")
            await bridge.reflect(stored_memories, {
                "meaningful": True,
                "emotional_impact": 0.9,
                "helped_insight": True
            })
            print("   ✓ Feedback provided to HoloLoom learning system")

            # Get metrics
            print("\n[4] Memory system metrics...")
            metrics = bridge.get_metrics()
            print(f"   Memories stored: {metrics.get('n_memories', 'N/A')}")
            print(f"   Connections: {metrics.get('n_connections', 'N/A')}")
            print(f"   Active: {metrics.get('n_active', 'N/A')}")

    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("""
Phase 1 MVP Features Demonstrated:
  ✓ store_dream_narrative() - Store dreams with full metadata
  ✓ recall_dreams() - Semantic search for dreams
  ✓ reflect_on_dream() - Provide feedback for learning
  ✓ DreamMemoryBridge - Convenient context manager
  ✓ Graceful degradation - Works without HoloLoom

Key Insight: Both systems use compatible 228D embeddings and
identical decay rates (0.95^hours), enabling seamless integration.

Next Steps (Phase 2):
  - DreamSpinner for SpinningWheel protocol
  - DreamDepartment for Department protocol
  - DS-STAR verification for dream quality
""")


if __name__ == "__main__":
    asyncio.run(demo())
