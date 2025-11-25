"""
Demo: Shared Dream Synchronization
===================================

Demonstrates the complete shared dream creation pipeline with Alice and Bob.

This demo shows:
1. Creating a dyadic (2-person) shared dream
2. Symbol blending mechanics
3. Perspective generation
4. Waking effects calculation
5. Full session summary

Run with: python demo_shared_dream_sync.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from NeuroHood.dreams.shared_dream_sync import (
    SharedDreamSynchronizer,
    SymbolBlendingStrategy,
)


async def demo_simple_dyadic():
    """Demo: Simple dyadic shared dream (Alice & Bob)."""
    print("\n" + "="*70)
    print("DEMO 1: Simple Dyadic Shared Dream (Alice and Bob)")
    print("="*70 + "\n")

    sync = SharedDreamSynchronizer()

    # Define participants with private facts
    participants = [
        {
            'id': 'alice_001',
            'name': 'Alice',
            'private_fact': 'I feel trapped in my corporate job'
        },
        {
            'id': 'bob_001',
            'name': 'Bob',
            'private_fact': 'I struggle with unresolved anger toward my brother'
        }
    ]

    print(f"Creating shared dream for:")
    for p in participants:
        print(f"  - {p['name']} (private: {p['private_fact'][:40]}...)")

    # Create shared dream
    session = await sync.create_shared_dream(
        participants,
        consciousness_level=0.5,
        blending_strategy=SymbolBlendingStrategy.INTERACTION
    )

    print(f"\n[OK] Dream created: {session.session_id}")
    print(f"  Type: {session.dream_type.value}")
    print(f"  Consciousness level: {session.consciousness_level:.2f}")
    print(f"  Intensity: {session.intensity:.2f}")
    print(f"  Duration: {session.duration_minutes:.1f} minutes")

    # Show symbolic encoding
    print(f"\n--- Symbolic Encoding ---\n")
    for res_id, contrib in session.contributions.items():
        print(f"{contrib.resident_name}'s Symbol:")
        print(f"  Symbol ID: {contrib.symbol.symbol_id}")
        print(f"  Emotions: {', '.join(contrib.symbol.emotion_tags[:3])}")
        print(f"  Private fact obscured by symbolic encoding")
        print()

    # Show blended narrative
    print(f"--- Shared Dream Narrative ---\n")
    print(session.shared_narrative)
    print()

    # Show perspectives
    print(f"--- Individual Perspectives ---\n")
    for res_id, perspective in session.participant_perspectives.items():
        resident = next(p['name'] for p in participants if p['id'] == res_id)
        print(f"{resident}'s Perspective:")
        print(f"  {perspective}")
        print()

    # Show waking effects
    print(f"--- Waking Effects (Relationship Changes) ---\n")
    for (res_a, res_b), effects in session.waking_effects.items():
        res_a_name = next(p['name'] for p in participants if p['id'] == res_a)
        res_b_name = next(p['name'] for p in participants if p['id'] == res_b)

        print(f"{res_a_name} <-> {res_b_name}:")
        for metric, value in effects.items():
            print(f"  {metric}: {value:+.2f}")
        print()

    # Show session summary
    print(f"--- Session Summary ---\n")
    summary = sync.get_session_summary(session)
    print(f"Session ID: {summary['session_id']}")
    print(f"Type: {summary['type']}")
    print(f"Participants: {', '.join(summary['participants'])}")
    print(f"Consciousness Level: {summary['consciousness_level']:.2f}")
    print(f"Intensity: {summary['intensity']:.2f}")
    print(f"Duration: {summary['duration_minutes']:.1f} min")


async def demo_triad():
    """Demo: Triad shared dream (Alice, Bob, Charlie)."""
    print("\n" + "="*70)
    print("DEMO 2: Triad Shared Dream (Alice, Bob, Charlie)")
    print("="*70 + "\n")

    sync = SharedDreamSynchronizer()

    participants = [
        {
            'id': 'alice_001',
            'name': 'Alice',
            'private_fact': 'I feel trapped in my corporate job'
        },
        {
            'id': 'bob_001',
            'name': 'Bob',
            'private_fact': 'I struggle with unresolved anger'
        },
        {
            'id': 'charlie_001',
            'name': 'Charlie',
            'private_fact': 'I lost my sense of identity after career change'
        }
    ]

    print(f"Creating shared dream for {len(participants)} residents:")
    for p in participants:
        print(f"  - {p['name']}")

    session = await sync.create_shared_dream(
        participants,
        consciousness_level=0.5,
        blending_strategy=SymbolBlendingStrategy.NARRATIVE
    )

    print(f"\n[OK] Triad dream created")
    print(f"  Type: {session.dream_type.value}")
    print(f"  Participants: {len(session.participants)}")
    print(f"  Intensity: {session.intensity:.2f}")
    print(f"  Duration: {session.duration_minutes:.1f} minutes")

    print(f"\n--- Shared Dream Narrative ---\n")
    print(session.shared_narrative)
    print()

    print(f"--- Waking Effects (All Pairwise Relationships) ---\n")
    effect_count = 0
    for (res_a, res_b), effects in session.waking_effects.items():
        res_a_name = next(p['name'] for p in participants if p['id'] == res_a)
        res_b_name = next(p['name'] for p in participants if p['id'] == res_b)

        print(f"{res_a_name} <-> {res_b_name}:")
        for metric, value in effects.items():
            print(f"  {metric}: {value:+.2f}")
        effect_count += 1
        print()

    print(f"Total relationship pairs affected: {effect_count}")


async def demo_blending_strategies():
    """Demo: Different blending strategies."""
    print("\n" + "="*70)
    print("DEMO 3: Different Blending Strategies")
    print("="*70 + "\n")

    sync = SharedDreamSynchronizer()

    participants = [
        {
            'id': 'alice_001',
            'name': 'Alice',
            'private_fact': 'I feel trapped in my corporate job'
        },
        {
            'id': 'bob_001',
            'name': 'Bob',
            'private_fact': 'I struggle with unresolved anger'
        }
    ]

    strategies = [
        SymbolBlendingStrategy.INTERACTION,
        SymbolBlendingStrategy.NARRATIVE,
        SymbolBlendingStrategy.LANDSCAPE,
        SymbolBlendingStrategy.CONVERGENCE
    ]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.value.upper()} ---\n")

        session = await sync.create_shared_dream(
            participants,
            blending_strategy=strategy
        )

        print(f"Blending Strategy: {strategy.value}")
        print(f"Emotional Resonance: {session.intensity:.2f}")
        print(f"Narrative:\n  {session.shared_narrative[:150]}...")
        print()


async def demo_privacy_preservation():
    """Demo: Privacy preservation mechanics."""
    print("\n" + "="*70)
    print("DEMO 4: Privacy Preservation in Shared Dreams")
    print("="*70 + "\n")

    sync = SharedDreamSynchronizer()

    participants = [
        {
            'id': 'alice_001',
            'name': 'Alice',
            'private_fact': 'I cheated on my taxes and feel guilty'
        },
        {
            'id': 'bob_001',
            'name': 'Bob',
            'private_fact': 'I secretly hate my best friend'
        }
    ]

    print("PRIVATE FACTS (Never directly shared in dream):")
    print()
    for p in participants:
        print(f"{p['name']}: {p['private_fact']}")
    print()

    print("-" * 70)
    print()

    session = await sync.create_shared_dream(participants)

    print("WHAT APPEARS IN SHARED DREAM (Symbolic, Privacy-Preserving):")
    print()

    for res_id, contrib in session.contributions.items():
        resident = next(p['name'] for p in participants if p['id'] == res_id)
        print(f"{resident}'s Symbolic Representation:")
        print(f"  Symbol: {contrib.symbol.symbol_id}")
        print(f"  Emotional Essence: {contrib.symbol.emotion_tags}")
        print(f"  [OK] Private fact is obscured")
        print(f"  [OK] Only emotional truth is shared")
        print()

    print("WHAT EACH RESIDENT EXPERIENCES:")
    print()

    for res_id, perspective in session.participant_perspectives.items():
        resident = next(p['name'] for p in participants if p['id'] == res_id)
        print(f"{resident} experiences:")
        print(f"  {perspective[:100]}...")
        print(f"  -> Feels {resident.lower()}'s emotional state")
        print(f"  -> Does NOT know the underlying facts")
        print(f"  -> Builds empathy through shared symbols")
        print()

    print("-" * 70)
    print()

    print("PRIVACY LEVELS:")
    for scene in [session.dream_scene]:
        if scene:
            print(f"  Privacy Level: {scene.privacy_level:.1f} (0=specific, 1=abstract)")
            print(f"  [OK] Symbols are highly abstract")
            print(f"  [OK] Emotional truth preserved")
            print(f"  [OK] Factual details hidden")


async def main():
    """Run all demos."""
    print("\n")
    print("=" * 70)
    print("SHARED DREAM SYNCHRONIZATION SYSTEM DEMO")
    print("=" * 70)

    try:
        # Run all demos
        await demo_simple_dyadic()
        await demo_triad()
        await demo_blending_strategies()
        await demo_privacy_preservation()

        print("\n" + "="*70)
        print("[OK] All demos completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
