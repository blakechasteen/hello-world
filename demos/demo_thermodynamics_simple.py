"""
Thermodynamics Demo - Phase 3

Demonstrates free energy minimization for exploration/exploitation balance.

Physics Model:
    Helmholtz Free Energy: F = E - T*S

    High Temperature (T → ∞):
        F ≈ -T*S (entropy dominates)
        → Explore (maximize diversity)

    Low Temperature (T → 0):
        F ≈ E (energy dominates)
        → Exploit (minimize cost)

Shows:
1. Temperature-controlled exploration/exploitation
2. Automatic annealing (hot start → cold finish)
3. Free energy tracking system health
4. Action selection via Boltzmann distribution

Run:
    python demos/demo_thermodynamics_simple.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.physics import (
    ThermodynamicOptimizer,
    EnergyCalculator,
    EntropyCalculator,
    CoolingSchedule
)


def demo_temperature_exploration():
    """Demo 1: Temperature controls exploration vs exploitation."""
    print("=" * 60)
    print("Demo 1: Temperature-Controlled Exploration")
    print("=" * 60)
    print()
    print("Physics: Boltzmann distribution P(a) ∝ exp(-E(a) / T)")
    print("  High temp (T=10): Uniform (exploration)")
    print("  Low temp (T=0.1): Greedy (exploitation)")
    print()

    # Define actions with different energies
    actions = ["cheap", "balanced", "expensive"]
    energies = {
        "cheap": 0.1,       # Low energy (good!)
        "balanced": 0.5,    # Medium energy
        "expensive": 0.9    # High energy (bad!)
    }

    print("Actions and energies:")
    for action, energy in energies.items():
        print(f"  {action:12s}: energy={energy:.1f}")

    print()

    # Test different temperatures
    temperatures = [10.0, 1.0, 0.1]

    for temp in temperatures:
        print(f"Temperature T={temp:.1f}:")

        # Create optimizer
        thermo = ThermodynamicOptimizer(
            initial_temperature=temp,
            auto_anneal=False  # Manual temperature for this demo
        )

        # Sample actions 100 times
        action_counts = {action: 0 for action in actions}
        for _ in range(100):
            action = thermo.select_action(actions, energies, temperature=temp)
            action_counts[action] += 1

        # Show distribution
        print("  Action distribution:")
        for action in actions:
            count = action_counts[action]
            bar = "#" * (count // 2)
            print(f"    {action:12s}: [{bar:<40}] {count}%")

        print()

    print("Observation:")
    print("  - High temp (T=10): Explores all actions (uniform)")
    print("  - Low temp (T=0.1): Exploits best action (greedy)")
    print("  - Temperature is the exploration parameter!")


def demo_annealing():
    """Demo 2: Simulated annealing (hot start → cold finish)."""
    print()
    print()
    print("=" * 60)
    print("Demo 2: Simulated Annealing")
    print("=" * 60)
    print()
    print("Concept: Start hot (explore), cool down (exploit)")
    print("  Early: High temp → Explore widely")
    print("  Late: Low temp → Exploit best option")
    print()

    # Create optimizer with exponential cooling
    thermo = ThermodynamicOptimizer(
        initial_temperature=10.0,
        cooling_schedule="exponential",
        cooling_rate=0.9,
        auto_anneal=True  # Automatic temperature cooling
    )

    actions = ["cheap", "balanced", "expensive"]
    energies = {"cheap": 0.1, "balanced": 0.5, "expensive": 0.9}

    print("Cooling schedule: T(t) = T0 * rate^t")
    print("  T0 = 10.0, rate = 0.9")
    print()

    # Run 20 timesteps
    print("Timestep   Temperature   Selected Action")
    print("-" * 60)

    for t in range(20):
        # Select action at current temperature
        action = thermo.select_action(actions, energies)

        # Update (auto-cools temperature)
        thermo.update(action, energies[action], actions)

        # Print state
        stats = thermo.get_statistics()
        print(f"  {t:2d}       {stats['temperature']:8.3f}      {action}")

    print()
    print("Observation:")
    print("  - Early (high temp): Explores all actions")
    print("  - Late (low temp): Converges to 'cheap' (optimal)")
    print("  - Automatic exploration → exploitation!")


def demo_free_energy():
    """Demo 3: Free energy tracks system health."""
    print()
    print()
    print("=" * 60)
    print("Demo 3: Free Energy Minimization")
    print("=" * 60)
    print()
    print("Free Energy: F = E - T*S")
    print("  E: Energy (cost)")
    print("  T: Temperature (exploration parameter)")
    print("  S: Entropy (diversity)")
    print()
    print("System minimizes F automatically:")
    print("  - High T: Maximize entropy S (explore)")
    print("  - Low T: Minimize energy E (exploit)")
    print()

    thermo = ThermodynamicOptimizer(
        initial_temperature=5.0,
        cooling_schedule="exponential",
        cooling_rate=0.92,
        auto_anneal=True
    )

    actions = ["cheap", "balanced", "expensive"]
    energies = {"cheap": 0.1, "balanced": 0.5, "expensive": 0.9}

    print("Timestep    E       S      T      F = E - T*S")
    print("-" * 60)

    for t in range(15):
        action = thermo.select_action(actions, energies)
        thermo.update(action, energies[action], actions)

        stats = thermo.get_statistics()
        print(f"  {t:2d}      {stats['energy']:.3f}  {stats['entropy']:.3f}  {stats['temperature']:.2f}   {stats['free_energy']:.3f}")

    print()
    print("Observation:")
    print("  - Free energy F decreases over time")
    print("  - Early: High entropy S (diverse actions)")
    print("  - Late: Low energy E (optimal actions)")
    print("  - F tracks system health!")


def demo_action_diversity():
    """Demo 4: Entropy measures action diversity."""
    print()
    print()
    print("=" * 60)
    print("Demo 4: Entropy Measures Diversity")
    print("=" * 60)
    print()
    print("Shannon Entropy: H = -sum(p * log(p))")
    print("  High entropy: Diverse (all actions used equally)")
    print("  Low entropy: Focused (one action dominates)")
    print()

    # Test different action distributions
    test_cases = [
        {"name": "Uniform (max diversity)", "counts": {"A": 33, "B": 33, "C": 34}},
        {"name": "Balanced", "counts": {"A": 50, "B": 30, "C": 20}},
        {"name": "Focused", "counts": {"A": 80, "B": 15, "C": 5}},
        {"name": "Greedy (min diversity)", "counts": {"A": 100, "B": 0, "C": 0}}
    ]

    print("Action distribution → Entropy:")
    for case in test_cases:
        actions = list(case["counts"].keys())
        counts = case["counts"]

        diversity = EntropyCalculator.diversity_score(actions, counts)

        # Show distribution
        total = sum(counts.values())
        dist_str = ", ".join(f"{action}={counts[action]*100//total}%" for action in actions)

        print(f"  {case['name']:25s}: [{dist_str:20s}] entropy={diversity:.3f}")

    print()
    print("Observation:")
    print("  - Uniform distribution → High entropy (1.0)")
    print("  - Greedy distribution → Low entropy (0.0)")
    print("  - Entropy measures exploration!")


def demo_comparison():
    """Demo 5: Compare different cooling schedules."""
    print()
    print()
    print("=" * 60)
    print("Demo 5: Cooling Schedule Comparison")
    print("=" * 60)
    print()

    schedules = [
        ("exponential", 0.9, "Fast cooling (geometric)"),
        ("inverse", 0.5, "Slow cooling (asymptotic)"),
        ("linear", 0.5, "Linear cooling")
    ]

    print("Temperature over 10 timesteps:")
    print()

    for schedule, rate, desc in schedules:
        thermo = ThermodynamicOptimizer(
            initial_temperature=10.0,
            cooling_schedule=schedule,
            cooling_rate=rate,
            auto_anneal=True
        )

        print(f"{schedule:12s} ({desc}):")
        temps = []
        for t in range(10):
            temps.append(thermo.state.temperature)
            thermo.scheduler.step()

        # Show temperature trajectory
        temp_str = " -> ".join(f"{t:.2f}" for t in temps[::2])  # Every 2nd
        print(f"  {temp_str}")
        print()

    print("Observation:")
    print("  - Exponential: Fast convergence (best for short tasks)")
    print("  - Inverse: Slow convergence (best for long tasks)")
    print("  - Linear: Medium convergence (predictable)")


if __name__ == "__main__":
    print()
    print("=== Thermodynamics Demo (Phase 3) ===")
    print()
    print("Free Energy Minimization: F = E - T*S")
    print()

    # Run demos
    demo_temperature_exploration()
    demo_annealing()
    demo_free_energy()
    demo_action_diversity()
    demo_comparison()

    print()
    print("=" * 60)
    print("[DONE] Thermodynamics demo complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  1. Temperature controls exploration vs exploitation")
    print("  2. Simulated annealing: hot start → cold finish")
    print("  3. Free energy F = E - T*S tracks system health")
    print("  4. Entropy measures action diversity")
    print("  5. Zero manual tuning - physics handles balance!")
    print()
    print("Next: Integrate thermodynamics into WeavingOrchestrator!")
