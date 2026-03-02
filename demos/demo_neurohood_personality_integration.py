"""
NeuroHood Personality Integration Demo

Demonstrates Phase 2 complete integration:
1. Spring Dynamics for relationship physics
2. Personality mapping to 228D semantic space
3. Personality drift from social interactions
4. Combined social + psychological simulation

Run:
    PYTHONPATH=. python demos/demo_neurohood_personality_integration.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NeuroHood import NeuroHood, NeuroHoodConfig
from NeuroHood.personality import PersonalitySystem
from hololoom.embedding.spectral import create_embedder


async def main():
    """Run integrated personality + spring dynamics demo."""

    print("\n" + "="*70)
    print("  NeuroHood Phase 2 Complete Integration")
    print("  Spring Dynamics + 228D Personality Space")
    print("="*70 + "\n")

    # ========== 1. Initialize System ==========

    print("[ Step 1: Initializing NeuroHood + Personality System ]")
    print("-" * 70)

    config = NeuroHoodConfig.prototype()
    config.enable_spring_dynamics = True  # Enable social physics

    # Create embedder for personality system
    embedder = create_embedder(sizes=[384])
    embed_fn = lambda text: embedder.encode([text])[0]

    # Create personality system
    personality = PersonalitySystem(embed_fn)

    print(f"  Spring Physics: ENABLED")
    print(f"  Personality System: ENABLED (16 dimensions)")
    print()

    async with NeuroHood(config) as hood:

        # ========== 2. Bootstrap Neighborhood ==========

        print("[ Step 2: Creating Neighborhood with Personality Profiles ]")
        print("-" * 70)

        state = await hood.bootstrap_neighborhood(
            seed="A quiet suburban street where Alice values peace, "
                 "Bob loves parties, and Charlie tries to keep the peace"
        )

        residents = list(state.residents.values())
        alice, bob, charlie = residents[0], residents[1], residents[2]

        print(f"  Created {len(residents)} residents")
        print()

        # Initialize personalities
        print("  Initializing Personality Profiles:")
        print()

        alice_personality = personality.initialize_personality(
            resident_id=alice.resident_id,
            description=(
                "Alice is a warm, caring, and loving person who deeply values peace and tranquility. "
                "She is introverted, calm, and reflective, preferring gentle interactions. "
                "She is creative, patient, and values harmony."
            )
        )
        print(f"    {alice.name}:")
        print(f"      {personality.get_personality_description(alice.resident_id, use_current=False)}")
        print()

        bob_personality = personality.initialize_personality(
            resident_id=bob.resident_id,
            description=(
                "Bob is excited, energetic, and intense, loving action and activity. "
                "He is extroverted and thrives on arousal and excitement. "
                "He is direct and straightforward, preferring casual interactions. "
                "He values urgency and seeks novel experiences."
            )
        )
        print(f"    {bob.name}:")
        print(f"      {personality.get_personality_description(bob.resident_id, use_current=False)}")
        print()

        charlie_personality = personality.initialize_personality(
            resident_id=charlie.resident_id,
            description=(
                "Charlie is balanced and diplomatic, seeking harmony and peace. "
                "He is generous and caring, with moderate energy and formality. "
                "He is certain and stable in his approach, preferring concrete solutions. "
                "He values completion and finishing things properly."
            )
        )
        print(f"    {charlie.name}:")
        print(f"      {personality.get_personality_description(charlie.resident_id, use_current=False)}")
        print()

        # ========== 3. Initial Personality Compatibility ==========

        print("[ Step 3: Initial Personality Compatibility ]")
        print("-" * 70)

        alice_bob_dist = personality.compute_personality_distance(
            alice.resident_id, bob.resident_id
        )
        alice_charlie_dist = personality.compute_personality_distance(
            alice.resident_id, charlie.resident_id
        )
        bob_charlie_dist = personality.compute_personality_distance(
            bob.resident_id, charlie.resident_id
        )

        print(f"  Personality Distances (0-8, lower = more compatible):")
        print(f"    {alice.name} <-> {bob.name}: {alice_bob_dist:.3f}")
        print(f"    {alice.name} <-> {charlie.name}: {alice_charlie_dist:.3f}")
        print(f"    {bob.name} <-> {charlie.name}: {bob_charlie_dist:.3f}")
        print()

        # ========== 4. Trigger Social Event (Noise Complaint) ==========

        print("[ Step 4: Social Event - Noise Complaint ]")
        print("-" * 70)

        print(f"  {alice.name} complains to {bob.name} about loud music...")
        print()

        # Apply negative interaction to relationship
        alice_bob_key = hood._relationship_key(alice.resident_id, bob.resident_id)
        alice_bob_rel = state.relationships[alice_bob_key]

        print(f"  Before Confrontation:")
        print(f"    Relationship strength: {alice_bob_rel.strength:.3f}")
        print(f"    Relationship tension: {alice_bob_rel.tension:.3f}")
        print()

        hood.relationship_physics.apply_social_interaction(
            relationship=alice_bob_rel,
            interaction_type="negative",
            residents=state.residents,
            intensity=0.4
        )

        print(f"  After Confrontation:")
        print(f"    Relationship strength: {alice_bob_rel.strength:.3f} (weakened)")
        print(f"    Relationship tension: {alice_bob_rel.tension:.3f} (increased)")
        print()

        # ========== 5. Update Personalities from Experience ==========

        print("[ Step 5: Personality Evolution from Experience ]")
        print("-" * 70)

        # Alice becomes stressed and negative
        print(f"  {alice.name} experiences stress and irritation...")
        alice_updated, alice_drifts = personality.update_personality(
            resident_id=alice.resident_id,
            description=(
                "Alice has become cold and harsh toward Bob, feeling negative and miserable. "
                "She is intensely stressed and irritated, more powerful in her demands. "
                "Less patient, more urgent, volatile and unstable."
            ),
            event_context=f"Noise complaint against {bob.name}"
        )

        print(f"    Updated personality: {personality.get_personality_description(alice.resident_id)}")
        print(f"    Significant drifts: {len(alice_drifts)}")
        for dim_name, drift in list(alice_drifts.items())[:5]:  # Top 5
            print(f"      {dim_name}: D{drift.delta:+.3f}")
        print()

        # Bob becomes defensive
        print(f"  {bob.name} becomes defensive and annoyed...")
        bob_updated, bob_drifts = personality.update_personality(
            resident_id=bob.resident_id,
            description=(
                "Bob becomes defensive and annoyed, feeling attacked and powerless. "
                "He is tense and volatile, more direct and confrontational. "
                "Less relaxed, more urgent in defending his lifestyle."
            ),
            event_context=f"Confrontation with {alice.name}"
        )

        print(f"    Updated personality: {personality.get_personality_description(bob.resident_id)}")
        print(f"    Significant drifts: {len(bob_drifts)}")
        for dim_name, drift in list(bob_drifts.items())[:5]:  # Top 5
            print(f"      {dim_name}: D{drift.delta:+.3f}")
        print()

        # ========== 6. Charlie Intervenes (Mediator) ==========

        print("[ Step 6: Charlie Intervenes as Mediator ]")
        print("-" * 70)

        print(f"  {charlie.name} tries to mediate the conflict...")
        print()

        charlie_updated, charlie_drifts = personality.update_personality(
            resident_id=charlie.resident_id,
            description=(
                "Charlie feels stressed trying to maintain peace, becoming more urgent. "
                "He is active and doing much to mediate, more direct and explicit. "
                "Still generous and caring, but feeling the pressure."
            ),
            event_context=f"Mediating between {alice.name} and {bob.name}"
        )

        print(f"    Updated personality: {personality.get_personality_description(charlie.resident_id)}")
        print(f"    Significant drifts: {len(charlie_drifts)}")
        for dim_name, drift in charlie_drifts.items():
            print(f"      {dim_name}: D{drift.delta:+.3f}")
        print()

        # ========== 7. Visualize Alice's Personality Drift ==========

        print("[ Step 7: Detailed Personality Drift (Alice) ]")
        print("-" * 70)
        print()
        print(personality.visualize_drift(alice.resident_id))
        print()

        # ========== 8. Drift Summary ==========

        print("[ Step 8: Drift Statistics ]")
        print("-" * 70)

        alice_summary = personality.get_drift_summary(alice.resident_id)
        bob_summary = personality.get_drift_summary(bob.resident_id)
        charlie_summary = personality.get_drift_summary(charlie.resident_id)

        print(f"  {alice.name}:")
        print(f"    Total drift: {alice_summary['total_drift']:.3f}")
        print(f"    Max drift: {alice_summary['max_drift_dimension']} (D{alice_summary['max_drift_value']:+.3f})")
        print()

        print(f"  {bob.name}:")
        print(f"    Total drift: {bob_summary['total_drift']:.3f}")
        print(f"    Max drift: {bob_summary['max_drift_dimension']} (D{bob_summary['max_drift_value']:+.3f})")
        print()

        print(f"  {charlie.name}:")
        print(f"    Total drift: {charlie_summary['total_drift']:.3f}")
        print(f"    Max drift: {charlie_summary['max_drift_dimension']} (D{charlie_summary['max_drift_value']:+.3f})")
        print()

        # ========== 9. Updated Compatibility ==========

        print("[ Step 9: Updated Personality Compatibility ]")
        print("-" * 70)

        alice_bob_dist_after = personality.compute_personality_distance(
            alice.resident_id, bob.resident_id
        )
        alice_charlie_dist_after = personality.compute_personality_distance(
            alice.resident_id, charlie.resident_id
        )
        bob_charlie_dist_after = personality.compute_personality_distance(
            bob.resident_id, charlie.resident_id
        )

        print(f"  Updated Distances:")
        print(f"    {alice.name} <-> {bob.name}: {alice_bob_dist_after:.3f} (was {alice_bob_dist:.3f}, D{alice_bob_dist_after - alice_bob_dist:+.3f})")
        print(f"    {alice.name} <-> {charlie.name}: {alice_charlie_dist_after:.3f} (was {alice_charlie_dist:.3f}, D{alice_charlie_dist_after - alice_charlie_dist:+.3f})")
        print(f"    {bob.name} <-> {charlie.name}: {bob_charlie_dist_after:.3f} (was {bob_charlie_dist:.3f}, D{bob_charlie_dist_after - bob_charlie_dist:+.3f})")
        print()

        # ========== 10. Combined Metrics ==========

        print("[ Step 10: Combined Social Physics + Personality Metrics ]")
        print("-" * 70)

        # Relationship energy (spring physics)
        relationship_energy = hood.get_relationship_energy(state)

        # Personality drift (psychological)
        total_personality_drift = (
            alice_summary['total_drift'] +
            bob_summary['total_drift'] +
            charlie_summary['total_drift']
        )

        print(f"  Social Physics:")
        print(f"    Relationship Energy: {relationship_energy:.4f} (higher = more conflict)")
        print()

        print(f"  Psychological:")
        print(f"    Total Personality Drift: {total_personality_drift:.3f}")
        print(f"    Residents with Significant Changes: {sum(1 for s in [alice_summary, bob_summary, charlie_summary] if s['significant_count'] > 0)}/3")
        print()

        print(f"  Interpretation:")
        if relationship_energy > 0.15 and total_personality_drift > 2.0:
            print(f"    High social tension + significant personality changes")
            print(f"    The neighborhood is under stress and residents are changing")
        elif relationship_energy < 0.05 and total_personality_drift < 1.0:
            print(f"    Low tension + minimal personality drift")
            print(f"    The neighborhood is stable and harmonious")
        else:
            print(f"    Moderate dynamics - some stress, some adaptation")
        print()

        # ========== 11. Conclusion ==========

        print("="*70)
        print("  Phase 2 Integration Demo Complete!")
        print("="*70)
        print()
        print("  What We Demonstrated:")
        print("    ✓ 228D Personality space (16 interpretable dimensions)")
        print("    ✓ Personality initialization from descriptions")
        print("    ✓ Automatic drift tracking from experiences")
        print("    ✓ Integration with Spring Dynamics physics")
        print("    ✓ Combined social (relationship) + psychological (personality) metrics")
        print("    ✓ Realistic personality evolution from social interactions")
        print()
        print("  The Magic:")
        print("    - Residents don't just have relationships - they EVOLVE psychologically")
        print("    - Social stress causes measurable personality drift")
        print("    - Physics (springs) + Psychology (228D space) = emergent realism")
        print("    - Every interaction changes residents in interpretable ways")
        print()
        print("  Phase 2 Status: ✅ COMPLETE")
        print("    - Spring Dynamics: ✅")
        print("    - Personality System: ✅")
        print("    - Drift Tracking: ✅")
        print()


if __name__ == "__main__":
    asyncio.run(main())
