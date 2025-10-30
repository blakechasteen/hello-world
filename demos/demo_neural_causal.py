"""
Neural-Causal Integration Demo

Shows the power of combining:
- Symbolic causal structure (interpretable)
- Neural mechanisms (learned from data)

Example: Medical treatment with complex, non-linear relationships
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from HoloLoom.causal import (
    CausalNode, CausalEdge, CausalDAG, NodeType,
    NeuralStructuralCausalModel
)


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def generate_synthetic_data(n_samples: int = 1000):
    """
    Generate synthetic medical data with KNOWN causal structure.

    Ground truth:
        Age → Treatment (elderly more likely to get treatment)
        Age → Recovery (age affects recovery, NON-LINEAR)
        Treatment → Recovery (treatment helps, NON-LINEAR interaction with age)

    Non-linear relationships:
        P(Treatment | Age) = sigmoid(age/20 - 2.5)
        P(Recovery | Age, Treatment) = sigmoid(
            -0.05 * (age - 50)^2 / 100 +  # Quadratic age effect
            treatment * (1 - age/100)      # Treatment better for younger patients
        )
    """
    print("Generating synthetic medical data...")
    print("  Ground truth causal structure:")
    print("    Age → Treatment")
    print("    Age → Recovery")
    print("    Treatment → Recovery")
    print("")

    # Age: 20-80 years old
    age = np.random.uniform(20, 80, n_samples)

    # Treatment: Elderly more likely to receive treatment
    treatment_prob = 1 / (1 + np.exp(-(age / 20 - 2.5)))
    treatment = (np.random.rand(n_samples) < treatment_prob).astype(float)

    # Recovery: Complex non-linear relationship
    # - Age has quadratic effect (50 is optimal)
    # - Treatment helps more for younger patients
    age_effect = -0.05 * ((age - 50) ** 2) / 100
    treatment_effect = treatment * (1 - age / 100)
    recovery_logit = age_effect + treatment_effect + 0.3
    recovery_prob = 1 / (1 + np.exp(-recovery_logit))
    recovery = (np.random.rand(n_samples) < recovery_prob).astype(float)

    # Combine into dataset
    data = np.column_stack([age, treatment, recovery])
    variable_names = ['age', 'treatment', 'recovery']

    print(f"✓ Generated {n_samples} samples")
    print(f"  Age range: {age.min():.1f} - {age.max():.1f}")
    print(f"  Treatment rate: {treatment.mean():.2%}")
    print(f"  Recovery rate: {recovery.mean():.2%}")
    print("")

    return data, variable_names


def demo_neural_causal():
    """Main demo."""

    print_section("NEURAL-CAUSAL INTEGRATION DEMO")

    print("The Problem:")
    print("-" * 80)
    print("Traditional approaches either:")
    print("  1. Use symbolic models (interpretable but can't learn complex patterns)")
    print("  2. Use pure neural networks (powerful but can't do causal inference)")
    print("")
    print("Our Solution:")
    print("  Combine BOTH:")
    print("  - Symbolic causal structure (human knowledge)")
    print("  - Neural mechanisms (learned from data)")
    print("")

    # ========================================================================
    # 1. Generate Data
    # ========================================================================

    print_section("STEP 1: Generate Synthetic Data")

    data, variable_names = generate_synthetic_data(n_samples=1000)

    # ========================================================================
    # 2. Define Causal Structure
    # ========================================================================

    print_section("STEP 2: Define Causal Structure (Domain Knowledge)")

    print("We KNOW the causal structure (from domain knowledge):")
    print("  Age → Treatment")
    print("  Age → Recovery")
    print("  Treatment → Recovery")
    print("")

    # Create DAG
    dag = CausalDAG()

    dag.add_node(CausalNode("age", NodeType.OBSERVABLE, description="Patient age"))
    dag.add_node(CausalNode("treatment", NodeType.OBSERVABLE, description="Received treatment"))
    dag.add_node(CausalNode("recovery", NodeType.OBSERVABLE, description="Patient recovered"))

    dag.add_edge(CausalEdge("age", "treatment"))
    dag.add_edge(CausalEdge("age", "recovery"))
    dag.add_edge(CausalEdge("treatment", "recovery"))

    print("✓ Causal DAG created")
    print(f"  Nodes: {len(dag.nodes)}")
    print(f"  Edges: {len(dag.edges)}")
    print("")

    # ========================================================================
    # 3. Learn Neural Mechanisms
    # ========================================================================

    print_section("STEP 3: Learn Neural Mechanisms from Data")

    print("Now we learn the MECHANISMS (how variables relate):")
    print("  Each variable = neural_network(parents)")
    print("")

    # Create Neural SCM
    nscm = NeuralStructuralCausalModel(dag)

    # Learn mechanisms
    print("Training neural networks...")
    nscm.fit(data, variable_names, hidden_dim=32, epochs=200)

    print("")
    print("✓ Neural mechanisms learned!")
    print(f"  {nscm}")
    print("")

    # ========================================================================
    # 4. Test Learned Model
    # ========================================================================

    print_section("STEP 4: Test Learned Model")

    print("Generate samples from learned model:")

    # Sample without intervention
    samples = nscm.sample(n_samples=100)

    print(f"  Generated {len(samples)} samples")
    print(f"  Age: {samples[:, 0].mean():.1f} ± {samples[:, 0].std():.1f}")
    print(f"  Treatment rate: {samples[:, 1].mean():.2%}")
    print(f"  Recovery rate: {samples[:, 2].mean():.2%}")
    print("")

    # ========================================================================
    # 5. Causal Inference (Interventions)
    # ========================================================================

    print_section("STEP 5: Causal Inference via Interventions")

    print("Question: What's the causal effect of treatment on recovery?")
    print("-" * 80)
    print("")

    # Estimate ATE
    ate = nscm.estimate_ate(
        treatment='treatment',
        outcome='recovery',
        treatment_value=1.0,
        control_value=0.0,
        n_samples=1000
    )

    print(f"Average Treatment Effect (ATE): {ate:.3f}")
    print("")

    if ate > 0.2:
        print("✓ STRONG positive effect - treatment causes recovery")
    elif ate > 0.05:
        print("⚠ MODERATE positive effect - treatment helps somewhat")
    elif ate > -0.05:
        print("~ WEAK effect - treatment doesn't matter much")
    else:
        print("✗ NEGATIVE effect - treatment is harmful!")

    print("")

    # ========================================================================
    # 6. Compare: Observational vs Causal
    # ========================================================================

    print_section("STEP 6: Observational vs Causal")

    print("Key Insight: Correlation ≠ Causation")
    print("-" * 80)
    print("")

    # Observational correlation
    treated_idx = data[:, 1] == 1
    control_idx = data[:, 1] == 0

    obs_treated_recovery = data[treated_idx, 2].mean()
    obs_control_recovery = data[control_idx, 2].mean()
    obs_diff = obs_treated_recovery - obs_control_recovery

    print("Observational (just correlation):")
    print(f"  P(Recovery | Treatment=1) = {obs_treated_recovery:.3f}")
    print(f"  P(Recovery | Treatment=0) = {obs_control_recovery:.3f}")
    print(f"  Difference: {obs_diff:.3f}")
    print("")

    print("Causal (intervention):")
    print(f"  P(Recovery | do(Treatment=1)) - P(Recovery | do(Treatment=0))")
    print(f"  = {ate:.3f}")
    print("")

    print("Confounding bias:")
    print(f"  Observational - Causal = {obs_diff - ate:.3f}")
    print("")

    if abs(obs_diff - ate) > 0.05:
        print("⚠️  WARNING: Observational estimate is biased!")
        print("   Elderly patients get more treatment AND have different recovery rates")
        print("   Age is a confounder")
    else:
        print("✓ Observational and causal estimates align")

    print("")

    # ========================================================================
    # 7. Counterfactual Reasoning
    # ========================================================================

    print_section("STEP 7: Counterfactual Reasoning")

    print("Question: Would a patient have recovered WITHOUT treatment?")
    print("-" * 80)
    print("")

    # Find a patient who got treatment and recovered
    recovered_treated = np.where((data[:, 1] == 1) & (data[:, 2] == 1))[0]
    if len(recovered_treated) > 0:
        patient_idx = recovered_treated[0]
        patient_age = data[patient_idx, 0]

        print(f"Patient #{patient_idx}:")
        print(f"  Age: {patient_age:.1f}")
        print(f"  Treatment: 1 (received)")
        print(f"  Recovery: 1 (recovered)")
        print("")

        # Counterfactual: What if no treatment?
        cf_recovery = nscm.counterfactual(
            intervention={'treatment': 0},
            evidence={'age': patient_age, 'treatment': 1, 'recovery': 1},
            query='recovery',
            n_samples=1000
        )

        print("Counterfactual: What if no treatment?")
        print(f"  P(Recovery | do(Treatment=0)) ≈ {cf_recovery:.3f}")
        print("")

        if cf_recovery < 0.3:
            print("✓ Treatment was likely NECESSARY")
            print("  Patient probably wouldn't have recovered without it")
        elif cf_recovery < 0.6:
            print("⚠ Treatment MAY have been necessary")
            print("  Unclear if patient would have recovered")
        else:
            print("✗ Treatment probably NOT necessary")
            print("  Patient likely would have recovered anyway")

    print("")

    # ========================================================================
    # Summary
    # ========================================================================

    print_section("SUMMARY: Neural-Causal Integration")

    print("What we did:")
    print("  1. ✅ Defined causal structure (domain knowledge)")
    print("  2. ✅ Learned neural mechanisms (from data)")
    print("  3. ✅ Performed causal inference (interventions)")
    print("  4. ✅ Computed counterfactuals")
    print("")

    print("Key advantages:")
    print("  ✓ Interpretable: Causal structure is explicit")
    print("  ✓ Powerful: Neural networks learn complex patterns")
    print("  ✓ Causal: Can do interventions and counterfactuals")
    print("  ✓ Data-driven: Learns from observations")
    print("")

    print("vs Pure symbolic models:")
    print("  ✓ No need to hand-code mechanisms")
    print("  ✓ Captures non-linear relationships")
    print("  ✓ Learns from data automatically")
    print("")

    print("vs Pure neural networks:")
    print("  ✓ Can answer causal questions")
    print("  ✓ Interpretable structure")
    print("  ✓ Requires less data (structure is prior knowledge)")
    print("")

    print("=" * 80)
    print("✅ Neural-Causal Integration Demo Complete!".center(80))
    print("=" * 80)
    print("")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    demo_neural_causal()
