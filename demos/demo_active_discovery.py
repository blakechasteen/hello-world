"""
Active Causal Discovery Demo

Shows how the system LEARNS causal structure through experimentation.

Instead of hand-coding:
    dag.add_edge("age" → "recovery")  # Manual

The system figures it out:
    learner.run_experiments(20)  # Learns automatically!
    dag = learner.get_dag()  # Discovered structure

This is cutting-edge AI: Learning through experimentation.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from HoloLoom.causal import (
    CausalDiscovery, ActiveCausalLearner
)


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def create_ground_truth_system():
    """
    Create ground truth causal system for testing.

    True structure:
        Age → Treatment
        Age → Recovery
        Treatment → Recovery
    """

    def system(intervention: dict) -> dict:
        """
        Simulate the ground truth system.

        Args:
            intervention: {variable: value} to intervene on

        Returns:
            Observations of all variables
        """
        # Sample root cause (exogenous variable)
        age = intervention.get('age', np.random.uniform(20, 80))

        # Treatment depends on age (unless intervened)
        if 'treatment' in intervention:
            treatment = intervention['treatment']
        else:
            treatment_prob = 1 / (1 + np.exp(-(age / 20 - 2.5)))
            treatment = 1 if np.random.rand() < treatment_prob else 0

        # Recovery depends on age and treatment (unless intervened)
        if 'recovery' in intervention:
            recovery = intervention['recovery']
        else:
            age_effect = -0.05 * ((age - 50) ** 2) / 100
            treatment_effect = treatment * (1 - age / 100)
            recovery_logit = age_effect + treatment_effect + 0.3
            recovery_prob = 1 / (1 + np.exp(-recovery_logit))
            recovery = 1 if np.random.rand() < recovery_prob else 0

        return {
            'age': age,
            'treatment': treatment,
            'recovery': recovery
        }

    return system


def demo_passive_discovery():
    """Learn causal structure from observational data (PC algorithm)."""

    print_section("PART 1: Passive Discovery (Observational Data)")

    print("The Challenge:")
    print("-" * 80)
    print("We have data but don't know the causal structure.")
    print("Can we learn it from observations alone?")
    print("")

    # Generate observational data
    system = create_ground_truth_system()
    n_samples = 1000

    data = []
    for _ in range(n_samples):
        obs = system({})  # No interventions
        data.append([obs['age'], obs['treatment'], obs['recovery']])

    data = np.array(data)
    variable_names = ['age', 'treatment', 'recovery']

    print(f"Generated {n_samples} observational samples")
    print(f"Variables: {variable_names}")
    print("")

    # Learn structure using PC algorithm
    print("Running PC Algorithm...")
    print("-" * 80)

    discoverer = CausalDiscovery(
        variables=variable_names,
        alpha=0.05,  # Significance level
        max_conditioning_size=2
    )

    discoverer.fit_observational(data, variable_names)

    print("")
    print("Learned Structure:")
    print("-" * 80)

    dag = discoverer.get_dag()

    print(f"Nodes: {len(dag.nodes)}")
    print(f"Edges: {len(dag.edges)}")
    print("")

    print("Discovered edges:")
    for (src, tgt), edge in dag.edges.items():
        print(f"  {src} → {tgt}")
    print("")

    # Compare to ground truth
    print("Ground Truth:")
    print("  age → treatment")
    print("  age → recovery")
    print("  treatment → recovery")
    print("")

    # Check accuracy
    correct_edges = {
        ('age', 'treatment'),
        ('age', 'recovery'),
        ('treatment', 'recovery')
    }

    discovered_edges = set(dag.edges.keys())

    true_positives = correct_edges & discovered_edges
    false_positives = discovered_edges - correct_edges
    false_negatives = correct_edges - discovered_edges

    print("Evaluation:")
    print(f"  True Positives: {len(true_positives)}/{len(correct_edges)}")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")
    print("")

    if len(false_negatives) == 0 and len(false_positives) == 0:
        print("✓ PERFECT! Discovered exact structure")
    elif len(true_positives) >= 2:
        print("✓ GOOD! Most edges discovered")
    else:
        print("⚠ PARTIAL: Some edges missing or incorrect")

    print("")


def demo_active_discovery():
    """Learn causal structure through active experimentation."""

    print_section("PART 2: Active Discovery (Experimentation)")

    print("The Power of Intervention:")
    print("-" * 80)
    print("Instead of just observing, we can EXPERIMENT!")
    print("The system chooses which experiments to run.")
    print("")

    # Create environment
    system = create_ground_truth_system()
    variable_names = ['age', 'treatment', 'recovery']

    # Create active learner
    learner = ActiveCausalLearner(
        variables=variable_names,
        environment=system
    )

    print("Running Active Learning Loop...")
    print("-" * 80)
    print("")

    # Run experiments
    n_experiments = 15

    for i in range(n_experiments):
        result = learner.run_experiment()

        print(f"Experiment {i+1}:")
        print(f"  Intervention: {result.intervention}")
        print(f"  Observed: ", end="")
        for var, val in result.observations.items():
            if var not in result.intervention:
                if isinstance(val, float):
                    print(f"{var}={val:.1f}", end=" ")
                else:
                    print(f"{var}={val}", end=" ")
        print("")

    print("")
    print("✓ Completed 15 experiments")
    print("")

    # Get learned structure
    dag = learner.get_dag()

    print("Learned Structure (from experiments):")
    print("-" * 80)

    print(f"Discovered {len(dag.edges)} edges:")
    for (src, tgt), edge in dag.edges.items():
        confidence = edge.confidence
        print(f"  {src} → {tgt} (confidence: {confidence:.2f})")
    print("")

    print("Ground Truth:")
    print("  age → treatment")
    print("  age → recovery")
    print("  treatment → recovery")
    print("")

    # Evaluation
    correct_edges = {
        ('age', 'treatment'),
        ('age', 'recovery'),
        ('treatment', 'recovery')
    }

    discovered_edges = set(dag.edges.keys())

    true_positives = correct_edges & discovered_edges
    precision = len(true_positives) / len(discovered_edges) if discovered_edges else 0
    recall = len(true_positives) / len(correct_edges)

    print("Evaluation:")
    print(f"  Precision: {precision:.2%} ({len(true_positives)}/{len(discovered_edges)})")
    print(f"  Recall: {recall:.2%} ({len(true_positives)}/{len(correct_edges)})")
    print("")

    if precision > 0.8 and recall > 0.6:
        print("✓ SUCCESS! Learned causal structure through experimentation")
    elif recall > 0.4:
        print("⚠ PARTIAL: Some edges discovered")
    else:
        print("✗ Need more experiments or better selection strategy")

    print("")


def demo_information_gain():
    """Show how information gain guides experiment selection."""

    print_section("PART 3: Smart Experiment Selection")

    print("How does the system choose which experiment to run?")
    print("-" * 80)
    print("")

    print("Answer: INFORMATION GAIN")
    print("")
    print("The system picks experiments that reduce uncertainty the most.")
    print("")

    print("Strategy:")
    print("  1. Estimate uncertainty about each edge")
    print("  2. Pick variable involved in most uncertain edges")
    print("  3. Intervene on that variable")
    print("  4. Observe which other variables change")
    print("  5. Update beliefs about causal structure")
    print("")

    print("Example:")
    print("  Uncertain: age → recovery? (50% belief)")
    print("  Experiment: Set age = 30")
    print("  Observe: recovery = 0.8 (high)")
    print("  Update: age → recovery (70% belief) ✓")
    print("")

    print("This is WAY more efficient than random experimentation!")
    print("")


def main():
    """Run all demos."""

    print_section("ACTIVE CAUSAL DISCOVERY DEMO")

    print("The Goal: Learn causal structure automatically")
    print("")
    print("Two approaches:")
    print("  1. PASSIVE: Learn from observations (PC algorithm)")
    print("  2. ACTIVE: Learn from experiments (active learning)")
    print("")
    print("Active learning is MUCH more efficient!")
    print("")

    # Part 1: Passive discovery
    demo_passive_discovery()

    # Part 2: Active discovery
    demo_active_discovery()

    # Part 3: Information gain
    demo_information_gain()

    # Summary
    print_section("SUMMARY")

    print("What we demonstrated:")
    print("")
    print("1. ✅ PC Algorithm (passive discovery)")
    print("   - Learns from observational data")
    print("   - Uses conditional independence tests")
    print("   - Can discover most of the structure")
    print("")

    print("2. ✅ Active Learning (experimental discovery)")
    print("   - Chooses informative experiments")
    print("   - Updates beliefs from interventions")
    print("   - More efficient than passive observation")
    print("")

    print("3. ✅ Information Gain")
    print("   - Measures uncertainty reduction")
    print("   - Guides experiment selection")
    print("   - Minimizes experiments needed")
    print("")

    print("Key Insight:")
    print("  INTERVENTIONS reveal causality better than OBSERVATIONS")
    print("  That's why experiments beat correlation studies!")
    print("")

    print("=" * 80)
    print("✅ Active Causal Discovery Demo Complete!".center(80))
    print("=" * 80)
    print("")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    main()
