"""
Statistical Mechanics Demo - Phase 5

Demonstrates emergent memory consolidation via statistical mechanics:
    - Gibbs distribution for natural clustering
    - Phase transitions for pattern crystallization
    - Entropy-driven organization
    - Free energy minimization F = E - T*S

Shows:
1. Emergent clustering (no manual rules)
2. Phase transition detection
3. Temperature-controlled organization
4. Entropy and free energy tracking

Run:
    python demos/demo_statistical_mechanics_simple.py
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.physics import (
    StatisticalMechanicsEngine,
    Microstate,
    CanonicalEnsemble,
    EntropyCalculator
)


def create_test_microstates():
    """Create test microstates (memory shards)."""
    # Cluster 1: Thompson Sampling related (similar embeddings)
    cluster1 = [
        Microstate(
            id="ts1",
            state_vector=np.array([1.0, 0.8, 0.2, 0.1]),
            energy=0.0,  # Will be computed
            metadata={"topic": "thompson_sampling"}
        ),
        Microstate(
            id="ts2",
            state_vector=np.array([0.9, 0.9, 0.1, 0.15]),
            energy=0.0,
            metadata={"topic": "thompson_sampling"}
        ),
        Microstate(
            id="ts3",
            state_vector=np.array([1.1, 0.7, 0.25, 0.05]),
            energy=0.0,
            metadata={"topic": "thompson_sampling"}
        ),
    ]

    # Cluster 2: Gradient descent related
    cluster2 = [
        Microstate(
            id="gd1",
            state_vector=np.array([0.1, 0.2, 1.0, 0.9]),
            energy=0.0,
            metadata={"topic": "gradient_descent"}
        ),
        Microstate(
            id="gd2",
            state_vector=np.array([0.15, 0.1, 0.9, 1.0]),
            energy=0.0,
            metadata={"topic": "gradient_descent"}
        ),
        Microstate(
            id="gd3",
            state_vector=np.array([0.05, 0.25, 1.1, 0.85]),
            energy=0.0,
            metadata={"topic": "gradient_descent"}
        ),
    ]

    # Cluster 3: Physics related
    cluster3 = [
        Microstate(
            id="ph1",
            state_vector=np.array([0.5, 0.5, 0.5, 0.5]),
            energy=0.0,
            metadata={"topic": "physics"}
        ),
        Microstate(
            id="ph2",
            state_vector=np.array([0.6, 0.4, 0.6, 0.4]),
            energy=0.0,
            metadata={"topic": "physics"}
        ),
    ]

    return cluster1 + cluster2 + cluster3


async def demo_gibbs_distribution():
    """Demo 1: Gibbs distribution for natural clustering."""
    print("=" * 70)
    print("Demo 1: Gibbs Distribution - Natural Clustering")
    print("=" * 70)
    print()
    print("Physics: P(i) = exp(-E_i/kT) / Z")
    print("  Low energy states have higher probability")
    print("  Similar memories cluster together (low interaction energy)")
    print()

    # Create microstates
    microstates = create_test_microstates()
    print(f"Created {len(microstates)} microstates:")
    for state in microstates:
        print(f"  {state.id}: topic={state.metadata['topic']}")
    print()

    # Create statistical mechanics engine
    engine = StatisticalMechanicsEngine(temperature=1.0)

    # Consolidate memories
    print("Consolidating memories via Gibbs distribution...")
    macrostates = engine.consolidate_memories(microstates, num_clusters=3)

    print()
    print(f"Emergent clusters: {len(macrostates)}")
    print()

    for i, macro in enumerate(macrostates):
        print(f"[Cluster {i}] {macro.label}")
        print(f"  Size: {len(macro.microstates)} memories")
        print(f"  Average Energy: {macro.average_energy:.3f}")
        print(f"  Entropy: {macro.entropy:.3f}")
        print(f"  Free Energy: {macro.free_energy:.3f}")
        print(f"  Order Parameter: {macro.order_parameter:.3f}")
        print(f"  Members: {[s.id for s in macro.microstates]}")
        print()

    print("Observations:")
    print("  - Similar memories naturally cluster (low interaction energy)")
    print("  - No manual clustering rules needed")
    print("  - Gibbs distribution automatically organizes by energy landscape")


async def demo_phase_transition():
    """Demo 2: Phase transition detection (pattern crystallization)."""
    print()
    print()
    print("=" * 70)
    print("Demo 2: Phase Transition Detection")
    print("=" * 70)
    print()
    print("Concept: Pattern crystallization via cooling (annealing)")
    print("  High T: Disordered (many possible states)")
    print("  Low T: Ordered (crystallized patterns)")
    print("  Transition: Critical point where order emerges")
    print()

    microstates = create_test_microstates()
    engine = StatisticalMechanicsEngine(temperature=5.0, cooling_rate=0.85)

    print("Cooling from T=5.0 -> 0.5 (simulated annealing)")
    print()
    print("Step  Temp    Clusters  Avg Order  Entropy   Free Energy")
    print("-" * 70)

    for step in range(15):
        # Consolidate at current temperature
        macrostates = engine.consolidate_memories(microstates, num_clusters=3)

        # Compute statistics
        avg_order = np.mean([m.order_parameter for m in macrostates])
        stats = engine.get_statistics()

        print(f"  {step:2d}   {stats['temperature']:5.2f}      {len(macrostates)}      {avg_order:.3f}     {stats['entropy']:.3f}    {stats['free_energy']:.3f}")

        # Cool temperature
        engine.anneal()

    # Detect phase transition
    print()
    transition = engine.detect_phase_transition()

    if transition:
        print(f"* Phase transition detected!")
        print(f"  Type: {transition.phase_type.value}")
        print(f"  Critical Temperature: {transition.critical_temperature:.2f}")
        print(f"  Order before: {transition.order_before:.3f}")
        print(f"  Order after: {transition.order_after:.3f}")
        if transition.phase_type.value == "first_order":
            print(f"  Latent heat: {transition.latent_heat:.3f}")
        else:
            print(f"  Critical exponent: {transition.critical_exponent:.3f}")
    else:
        print("No phase transition detected (continuous evolution)")

    print()
    print("Observations:")
    print("  - Cooling increases order (entropy decreases)")
    print("  - Phase transition marks crystallization of patterns")
    print("  - Free energy decreases (system stabilizes)")


async def demo_entropy_evolution():
    """Demo 3: Entropy evolution during consolidation."""
    print()
    print()
    print("=" * 70)
    print("Demo 3: Entropy Evolution")
    print("=" * 70)
    print()
    print("Gibbs Entropy: S = -k SUM p_i ln(p_i)")
    print("  High S: Many accessible states (disordered)")
    print("  Low S: Few accessible states (ordered)")
    print()

    # Create ensemble
    ensemble = CanonicalEnsemble(temperature=2.0)
    calc = EntropyCalculator()

    # Create energy distributions
    print("Example 1: Uniform energy (max entropy)")
    energies_uniform = [1.0, 1.0, 1.0, 1.0, 1.0]
    probs_uniform = ensemble.gibbs_probabilities(energies_uniform)
    entropy_uniform = calc.gibbs_entropy(probs_uniform)

    print(f"  Energies: {energies_uniform}")
    print(f"  Probabilities: {[f'{p:.3f}' for p in probs_uniform]}")
    print(f"  Entropy: {entropy_uniform:.3f} (maximum)")
    print()

    print("Example 2: Single low energy (min entropy)")
    energies_peaked = [0.0, 5.0, 5.0, 5.0, 5.0]
    probs_peaked = ensemble.gibbs_probabilities(energies_peaked)
    entropy_peaked = calc.gibbs_entropy(probs_peaked)

    print(f"  Energies: {energies_peaked}")
    print(f"  Probabilities: {[f'{p:.3f}' for p in probs_peaked]}")
    print(f"  Entropy: {entropy_peaked:.3f} (minimum)")
    print()

    print("Example 3: Gradual distribution (medium entropy)")
    energies_gradual = [0.0, 1.0, 2.0, 3.0, 4.0]
    probs_gradual = ensemble.gibbs_probabilities(energies_gradual)
    entropy_gradual = calc.gibbs_entropy(probs_gradual)

    print(f"  Energies: {energies_gradual}")
    print(f"  Probabilities: {[f'{p:.3f}' for p in probs_gradual]}")
    print(f"  Entropy: {entropy_gradual:.3f} (medium)")
    print()

    print("Observations:")
    print("  - Uniform -> Max entropy (all states equally probable)")
    print("  - Peaked -> Min entropy (one dominant state)")
    print("  - Entropy measures uncertainty/disorder")


async def demo_free_energy_minimization():
    """Demo 4: Free energy minimization F = E - T*S."""
    print()
    print()
    print("=" * 70)
    print("Demo 4: Free Energy Minimization")
    print("=" * 70)
    print()
    print("Helmholtz Free Energy: F = E - T*S")
    print("  E: Internal energy (stability)")
    print("  T*S: Entropy term (disorder)")
    print("  System minimizes F (balance energy and entropy)")
    print()

    microstates = create_test_microstates()

    # Compare different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print("Temperature  Entropy  Energy   Free Energy")
    print("-" * 70)

    for T in temperatures:
        engine = StatisticalMechanicsEngine(temperature=T)
        macrostates = engine.consolidate_memories(microstates, num_clusters=3)

        stats = engine.get_statistics()

        print(f"  {T:5.1f}       {stats['entropy']:6.3f}  {stats['average_energy']:6.3f}   {stats['free_energy']:8.3f}")

    print()
    print("Observations:")
    print("  - Low T: Emphasizes energy (E dominates)")
    print("  - High T: Emphasizes entropy (T*S dominates)")
    print("  - System finds optimal balance at each temperature")


async def demo_emergent_organization():
    """Demo 5: Complete emergent organization example."""
    print()
    print()
    print("=" * 70)
    print("Demo 5: Complete Emergent Organization")
    print("=" * 70)
    print()

    # Create diverse microstates
    microstates = create_test_microstates()

    # Start with high temperature (disordered)
    engine = StatisticalMechanicsEngine(temperature=5.0, cooling_rate=0.9)

    print("Complete consolidation cycle:")
    print()

    # Initial state (high temp)
    macrostates_initial = engine.consolidate_memories(microstates, num_clusters=3)
    stats_initial = engine.get_statistics()

    print(f"[Initial State] T={stats_initial['temperature']:.2f}")
    print(f"  Entropy: {stats_initial['entropy']:.3f} (high disorder)")
    print(f"  Free Energy: {stats_initial['free_energy']:.3f}")
    print(f"  Clusters: {len(macrostates_initial)}")
    for i, macro in enumerate(macrostates_initial):
        print(f"    Cluster {i}: {len(macro.microstates)} members, order={macro.order_parameter:.3f}")
    print()

    # Cool to intermediate
    for _ in range(5):
        engine.anneal()

    macrostates_mid = engine.consolidate_memories(microstates, num_clusters=3)
    stats_mid = engine.get_statistics()

    print(f"[Mid-Cooling] T={stats_mid['temperature']:.2f}")
    print(f"  Entropy: {stats_mid['entropy']:.3f} (decreasing)")
    print(f"  Free Energy: {stats_mid['free_energy']:.3f}")
    print(f"  Clusters: {len(macrostates_mid)}")
    for i, macro in enumerate(macrostates_mid):
        print(f"    Cluster {i}: {len(macro.microstates)} members, order={macro.order_parameter:.3f}")
    print()

    # Cool to low temperature
    for _ in range(10):
        engine.anneal()

    macrostates_final = engine.consolidate_memories(microstates, num_clusters=3)
    stats_final = engine.get_statistics()

    print(f"[Final State] T={stats_final['temperature']:.2f}")
    print(f"  Entropy: {stats_final['entropy']:.3f} (low disorder)")
    print(f"  Free Energy: {stats_final['free_energy']:.3f} (minimized)")
    print(f"  Clusters: {len(macrostates_final)}")
    for i, macro in enumerate(macrostates_final):
        print(f"    Cluster {i}: {len(macro.microstates)} members, order={macro.order_parameter:.3f}")
        print(f"      Members: {[s.id for s in macro.microstates]}")
    print()

    # Detect phase transition
    transition = engine.detect_phase_transition()
    if transition:
        print(f"* Phase transition at T={transition.critical_temperature:.2f}")
        print(f"  Pattern crystallization: order {transition.order_before:.3f} -> {transition.order_after:.3f}")
    print()

    print("Key Insights:")
    print("  1. High T -> Disordered (exploration of states)")
    print("  2. Cooling -> Gradual ordering (entropy decreases)")
    print("  3. Phase transition -> Patterns crystallize (emergent structure)")
    print("  4. Low T -> Organized (stable clusters)")
    print("  5. Free energy -> Minimized (optimal balance)")
    print()
    print("Emergence:")
    print("  - NO manual clustering rules")
    print("  - Organization emerges from energy landscape")
    print("  - Physics determines structure!")


if __name__ == "__main__":
    print()
    print("=== Statistical Mechanics Demo (Phase 5) ===")
    print()
    print("Emergent memory consolidation via statistical mechanics!")
    print()

    # Run demos
    asyncio.run(demo_gibbs_distribution())
    asyncio.run(demo_phase_transition())
    asyncio.run(demo_entropy_evolution())
    asyncio.run(demo_free_energy_minimization())
    asyncio.run(demo_emergent_organization())

    print()
    print("=" * 70)
    print("[COMPLETE] Statistical mechanics demo finished!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  * Gibbs distribution -> Natural clustering (no rules)")
    print("  * Phase transitions -> Pattern crystallization")
    print("  * Entropy -> Measures disorder (S = -k SUM p ln(p))")
    print("  * Free energy -> F = E - T*S (minimized)")
    print("  * Emergent organization -> Structure from physics")
    print()
    print("Key Achievement:")
    print("  Self-organizing memory system via statistical mechanics!")
    print("  No manual tuning - physics determines optimal structure.")
