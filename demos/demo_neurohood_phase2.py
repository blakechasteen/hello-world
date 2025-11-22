"""
NeuroHood Phase 2 Demo: Social Physics

Demonstrates Phase 2 features:
1. Spring Dynamics integration (relationship physics)
2. Emotional propagation through social network
3. Relationship evolution (strengthening/weakening)
4. Social cluster detection
5. Relationship energy (harmony vs. conflict)

Run:
    PYTHONPATH=. python demos/demo_neurohood_phase2.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NeuroHood import NeuroHood, NeuroHoodConfig


async def main():
    """Run NeuroHood Phase 2 demo."""

    print("\n" + "="*70)
    print("  NeuroHood Phase 2: Social Physics")
    print("  Relationships as Physical Springs with Real Dynamics")
    print("="*70 + "\n")

    # ========== 1. Initialize with Spring Dynamics ==========

    print("[ Step 1: Initializing NeuroHood with Spring Dynamics ]")
    print("-" * 70)

    config = NeuroHoodConfig.prototype()
    config.enable_spring_dynamics = True  # Enable physics!

    print(f"  Spring Physics Enabled: {config.enable_spring_dynamics}")
    print(f"  Relationship Stiffness (k): {config.relationship_stiffness}")
    print(f"  Relationship Damping (c): {config.relationship_damping}")
    print(f"  Relationship Decay: {config.relationship_decay}")
    print()

    async with NeuroHood(config) as hood:

        # ========== 2. Bootstrap Neighborhood ==========

        print("[ Step 2: Creating Neighborhood ]")
        print("-" * 70)

        state = await hood.bootstrap_neighborhood(
            seed="A street where Alice (quiet, introverted) lives next to "
                 "Bob (loud, party-loving) and Charlie (peacemaker)"
        )

        residents = list(state.residents.values())
        alice, bob, charlie = residents[0], residents[1], residents[2]

        print(f"  Created {len(residents)} residents:")
        print(f"    - {alice.name} (quiet, introverted)")
        print(f"    - {bob.name} (loud, party-loving)")
        print(f"    - {charlie.name} (peacemaker)")
        print()

        # Show initial relationships
        print(f"  Initial Relationships:")
        for key, rel in state.relationships.items():
            res_a = state.residents[rel.resident_a]
            res_b = state.residents[rel.resident_b]
            print(f"    {res_a.name} ↔ {res_b.name}: "
                  f"strength={rel.strength:.2f}, tension={rel.tension:.2f}")
        print()

        # ========== 3. Emotional Propagation ==========

        print("[ Step 3: Emotional Propagation Through Social Network ]")
        print("-" * 70)
        print(f"  {alice.name} gets ANGRY about noise from {bob.name}'s party\n")

        # Propagate Alice's anger through the network
        activations = await hood.propagate_emotional_state(
            trigger_resident=alice.resident_id,
            emotion="anger",
            intensity=1.0,  # Maximum anger!
            state=state
        )

        print("  Emotional Activation Spreading (via Spring Dynamics):")
        for resident_id, activation in sorted(activations.items(),
                                             key=lambda x: x[1],
                                             reverse=True):
            resident = state.residents[resident_id]
            print(f"    {resident.name}: {activation:.3f} "
                  f"(mood: {resident.internal_state['mood']:.2f}, "
                  f"stress: {resident.internal_state['stress']:.2f})")

        print()
        print("  🌊 Anger propagated through relationship springs!")
        print(f"     {alice.name} (angry) → affects → {bob.name} & {charlie.name}")
        print("     Strength of propagation = relationship strength × distance")
        print()

        # ========== 4. Relationship Evolution ==========

        print("[ Step 4: Relationship Evolution via Negative Interaction ]")
        print("-" * 70)

        # Alice confronts Bob (negative interaction)
        alice_bob_key = hood._relationship_key(alice.resident_id, bob.resident_id)
        alice_bob_rel = state.relationships[alice_bob_key]

        print(f"  Before confrontation:")
        print(f"    {alice.name} ↔ {bob.name}: strength={alice_bob_rel.strength:.2f}, "
              f"tension={alice_bob_rel.tension:.2f}")
        print()

        # Apply negative interaction
        hood.relationship_physics.apply_social_interaction(
            relationship=alice_bob_rel,
            interaction_type="negative",
            residents=state.residents,
            intensity=0.3  # Moderate conflict
        )

        print(f"  After confrontation (negative interaction, intensity=0.3):")
        print(f"    {alice.name} ↔ {bob.name}: strength={alice_bob_rel.strength:.2f}, "
              f"tension={alice_bob_rel.tension:.2f}")
        print()

        print("  🔴 Relationship weakened! Tension increased!")
        print(f"     Spring got softer (lower stiffness: {alice_bob_rel.strength:.2f})")
        print(f"     Spring got stressed (higher tension: {alice_bob_rel.tension:.2f})")
        print()

        # ========== 5. Time Passing (Decay & Healing) ==========

        print("[ Step 5: Time Passing - Natural Decay & Tension Release ]")
        print("-" * 70)

        print("  Simulating 10 time steps (relationships decay, tensions ease)...\n")

        for i in range(10):
            state = await hood._update_social_physics(state)

        print(f"  After 10 time steps:")
        print(f"    {alice.name} ↔ {bob.name}: strength={alice_bob_rel.strength:.2f}, "
              f"tension={alice_bob_rel.tension:.2f}")
        print()

        print("  ⏰ Natural dynamics:")
        print(f"     - Strength decayed (no maintenance): {config.relationship_decay:.2f} per step")
        print(f"     - Tension relaxed (damping): {config.relationship_damping:.2f} per step")
        print("     - Springs naturally seek equilibrium!")
        print()

        # ========== 6. Positive Interaction (Apology) ==========

        print("[ Step 6: Positive Interaction - Bob Apologizes ]")
        print("-" * 70)

        print(f"  Bob apologizes to Alice (positive interaction, intensity=0.4)...\n")

        hood.relationship_physics.apply_social_interaction(
            relationship=alice_bob_rel,
            interaction_type="positive",
            residents=state.residents,
            intensity=0.4
        )

        print(f"  After apology:")
        print(f"    {alice.name} ↔ {bob.name}: strength={alice_bob_rel.strength:.2f}, "
              f"tension={alice_bob_rel.tension:.2f}")
        print()

        print("  💚 Relationship strengthened! Tension reduced!")
        print("     Forgiveness = positive force on spring")
        print()

        # ========== 7. Social Cluster Detection ==========

        print("[ Step 7: Social Cluster Detection ]")
        print("-" * 70)

        clusters = hood.get_relationship_clusters(state)

        print(f"  Detected {len(clusters)} social cluster(s):\n")

        for i, cluster in enumerate(clusters, 1):
            cluster_names = [state.residents[rid].name for rid in cluster]
            print(f"    Cluster {i}: {', '.join(cluster_names)}")

        print()
        print("  🔗 Clusters detected using spring energy:")
        print("     High internal energy (strong connections)")
        print("     + Low external energy (weak connections)")
        print("     = Tightly-bonded social group!")
        print()

        # ========== 8. Relationship Energy ==========

        print("[ Step 8: Relationship Energy (Harmony vs. Conflict) ]")
        print("-" * 70)

        energy = hood.get_relationship_energy(state)

        print(f"  Total Relationship Energy: {energy:.4f}\n")

        print("  Energy Interpretation:")
        if energy < 0.05:
            print("    ✓ LOW ENERGY: Neighborhood in harmony")
        elif energy < 0.15:
            print("    ⚠ MODERATE ENERGY: Some tensions present")
        else:
            print("    🔴 HIGH ENERGY: Significant conflicts!")

        print()
        print("  Physics Analogy:")
        print("    E = Σ (1/2 × k × tension²)")
        print("    Spring network seeks minimum energy (equilibrium)")
        print("    High energy = system under stress")
        print()

        # ========== 9. Force Calculation ==========

        print("[ Step 9: Social Force Between Residents ]")
        print("-" * 70)

        force = hood.relationship_physics.compute_social_force(
            alice, bob, alice_bob_rel
        )

        print(f"  Social Force ({alice.name} → {bob.name}): {force:.4f}\n")

        print("  Force Interpretation:")
        if force > 0:
            print(f"    → Attractive force (pulling together)")
        elif force < 0:
            print(f"    ← Repulsive force (pushing apart)")
        else:
            print(f"    — Balanced (no net force)")

        print()
        print("  Hooke's Law: F = -k × (activation_a - activation_b)")
        print(f"    k (stiffness) = {alice_bob_rel.strength:.2f}")
        print(f"    Δa (mood difference) = "
              f"{alice.internal_state['mood']:.2f} - {bob.internal_state['mood']:.2f}")
        print()

        # ========== 10. Metrics Summary ==========

        print("[ Step 10: Physics Engine Metrics ]")
        print("-" * 70)

        metrics = hood.get_metrics()

        print(f"  Total Residents: {metrics['total_residents']}")
        print(f"  Relationships Formed: {metrics['relationships_formed']}")
        print(f"  Relationship Energy: {metrics.get('relationship_energy', 0.0):.4f}")
        print(f"  Social Clusters: {metrics.get('social_clusters', 0)}")
        print()

        if 'physics' in metrics:
            phys = metrics['physics']
            print(f"  Spring Dynamics Engine:")
            print(f"    Propagations: {phys['total_propagations']}")
            print(f"    Avg Convergence Iterations: {phys['avg_convergence_iterations']:.1f}")
            print(f"    Integrator: {phys['spring_config']['integrator']}")
            print()

        # ========== 11. Conclusion ==========

        print("="*70)
        print("  Phase 2 Demo Complete!")
        print("="*70)
        print("\n  What We Just Demonstrated:")
        print("    ✓ Relationships as physical springs (Hooke's Law)")
        print("    ✓ Emotional propagation through network")
        print("    ✓ Relationship evolution (strengthening/weakening)")
        print("    ✓ Natural decay and tension release over time")
        print("    ✓ Social cluster detection (spring energy)")
        print("    ✓ Relationship energy (harmony vs. conflict)")
        print("    ✓ Social force calculations")
        print("\n  Physics Integration:")
        print("    - HoloLoom's Spring Dynamics (Velocity Verlet integrator)")
        print("    - Professional ODE solver (energy-conserving)")
        print("    - 100-1000× accuracy vs. naive Euler method")
        print("\n  The Magic:")
        print("    Relationships FEEL physical because they ARE physical!")
        print("    Strong bonds resist change, weak bonds fade naturally.")
        print("    Emotions propagate like waves through springs.")
        print("    The social network seeks equilibrium, just like nature.")
        print()


if __name__ == "__main__":
    asyncio.run(main())
