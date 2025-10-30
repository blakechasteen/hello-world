"""
Abductive Reasoning Demo - Medical Diagnosis

Demonstrates inference to best explanation through medical diagnosis scenarios:
1. Simple Diagnosis: Single disease from symptoms
2. Differential Diagnosis: Multiple possible diseases
3. Multi-Cause: Multiple diseases simultaneously
4. Uncertain Evidence: Noisy/unreliable observations
5. Hypothesis Testing: Comparing specific diagnoses

Shows the power of abductive reasoning for explanation generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.reasoning.abductive import (
    AbductiveReasoner, CausalRule, Observation,
    create_causal_rule, create_observation
)


# ============================================================================
# Example 1: Simple Diagnosis
# ============================================================================

def demo_simple_diagnosis():
    """Single disease explaining symptoms."""

    print("=" * 80)
    print("EXAMPLE 1: SIMPLE MEDICAL DIAGNOSIS".center(80))
    print("=" * 80)
    print()

    print("Scenario: Patient presents with fever, cough, and fatigue.")
    print("Question: What disease best explains these symptoms?")
    print()

    # Causal knowledge: diseases → symptoms
    causal_rules = [
        # Flu causes various symptoms
        create_causal_rule("disease", "flu", "symptom", "fever", 0.90),
        create_causal_rule("disease", "flu", "symptom", "cough", 0.80),
        create_causal_rule("disease", "flu", "symptom", "fatigue", 0.85),

        # Cold causes symptoms
        create_causal_rule("disease", "cold", "symptom", "cough", 0.75),
        create_causal_rule("disease", "cold", "symptom", "fatigue", 0.60),
        create_causal_rule("disease", "cold", "symptom", "fever", 0.30),

        # COVID causes symptoms
        create_causal_rule("disease", "covid", "symptom", "fever", 0.88),
        create_causal_rule("disease", "covid", "symptom", "cough", 0.82),
        create_causal_rule("disease", "covid", "symptom", "fatigue", 0.90),
    ]

    # Prior probabilities (base rates)
    priors = {
        "disease": {
            "flu": 0.10,      # 10% of population has flu
            "cold": 0.20,     # 20% has cold
            "covid": 0.05,    # 5% has COVID
        }
    }

    # Create reasoner
    reasoner = AbductiveReasoner(causal_rules, priors)

    # Observations
    observations = [
        create_observation("symptom", "fever", confidence=1.0),
        create_observation("symptom", "cough", confidence=1.0),
        create_observation("symptom", "fatigue", confidence=1.0),
    ]

    print("Causal Knowledge:")
    print("-" * 80)
    print("Diseases → Symptoms:")
    for rule in causal_rules[:9]:
        print(f"  {rule}")
    print()

    print("Observations:")
    print("-" * 80)
    for obs in observations:
        print(f"  {obs}")
    print()

    # Find explanations
    print("Abductive Reasoning:")
    print("-" * 80)
    explanations = reasoner.explain(observations, max_hypotheses=5)

    print(f"Generated {len(explanations)} candidate explanations:\n")

    for i, hyp in enumerate(explanations, 1):
        print(f"{i}. {hyp.explanation}")
        print(f"   Score: {hyp.score():.4f}")
        print(f"   Likelihood: {hyp.likelihood:.4f} (how well it explains symptoms)")
        print(f"   Prior: {hyp.prior:.4f} (how common the disease is)")
        print(f"   Complexity: {hyp.complexity:.1f} (simpler = better)")
        print(f"   Explains: {', '.join(hyp.observations_explained)}")
        print()

    # Best explanation
    best = explanations[0]
    print("=" * 80)
    print(f"✓ Best Explanation: {best.explanation}")
    print(f"  Confidence: {best.score():.1%}")
    print("=" * 80)
    print()


# ============================================================================
# Example 2: Differential Diagnosis
# ============================================================================

def demo_differential_diagnosis():
    """Multiple diseases with overlapping symptoms."""

    print("=" * 80)
    print("EXAMPLE 2: DIFFERENTIAL DIAGNOSIS".center(80))
    print("=" * 80)
    print()

    print("Scenario: Patient has chest pain and shortness of breath.")
    print("Question: Heart attack, pneumonia, or anxiety?")
    print()

    # Causal rules
    causal_rules = [
        # Heart attack
        create_causal_rule("disease", "heart_attack", "symptom", "chest_pain", 0.95),
        create_causal_rule("disease", "heart_attack", "symptom", "shortness_of_breath", 0.85),
        create_causal_rule("disease", "heart_attack", "symptom", "sweating", 0.80),

        # Pneumonia
        create_causal_rule("disease", "pneumonia", "symptom", "chest_pain", 0.70),
        create_causal_rule("disease", "pneumonia", "symptom", "shortness_of_breath", 0.90),
        create_causal_rule("disease", "pneumonia", "symptom", "fever", 0.85),

        # Anxiety
        create_causal_rule("disease", "anxiety", "symptom", "chest_pain", 0.60),
        create_causal_rule("disease", "anxiety", "symptom", "shortness_of_breath", 0.75),
        create_causal_rule("disease", "anxiety", "symptom", "sweating", 0.70),
    ]

    priors = {
        "disease": {
            "heart_attack": 0.01,   # 1% (rare but serious)
            "pneumonia": 0.05,      # 5%
            "anxiety": 0.15,        # 15% (common)
        }
    }

    reasoner = AbductiveReasoner(causal_rules, priors)

    # Observations: chest pain + shortness of breath
    observations = [
        create_observation("symptom", "chest_pain", confidence=1.0),
        create_observation("symptom", "shortness_of_breath", confidence=1.0),
    ]

    print("Observations:")
    print("-" * 80)
    for obs in observations:
        print(f"  {obs}")
    print()

    print("Differential Diagnosis:")
    print("-" * 80)
    explanations = reasoner.explain(observations, max_hypotheses=3)

    for i, hyp in enumerate(explanations, 1):
        disease = hyp.explanation.get("disease", "unknown")
        score = hyp.score()
        likelihood = hyp.likelihood
        prior = hyp.prior

        print(f"{i}. {disease.upper()}")
        print(f"   Score: {score:.4f} (Likelihood: {likelihood:.3f}, Prior: {prior:.3f})")
        print(f"   Evidence: {', '.join(hyp.supporting_evidence)}")
        print()

    # Show how additional evidence changes diagnosis
    print("=" * 80)
    print("Adding additional symptom: FEVER")
    print("=" * 80)
    print()

    observations.append(create_observation("symptom", "fever", confidence=1.0))
    explanations_with_fever = reasoner.explain(observations, max_hypotheses=3)

    for i, hyp in enumerate(explanations_with_fever, 1):
        disease = hyp.explanation.get("disease", "unknown")
        score = hyp.score()
        print(f"{i}. {disease.upper()}: {score:.4f}")

    print()
    print("✓ Fever shifts diagnosis toward PNEUMONIA")
    print()


# ============================================================================
# Example 3: Multi-Cause Diagnosis
# ============================================================================

def demo_multi_cause():
    """Patient with multiple simultaneous conditions."""

    print("=" * 80)
    print("EXAMPLE 3: MULTI-CAUSE DIAGNOSIS".center(80))
    print("=" * 80)
    print()

    print("Scenario: Patient has many symptoms from different systems.")
    print("Question: Could multiple diseases be present simultaneously?")
    print()

    # Causal rules
    causal_rules = [
        # Diabetes
        create_causal_rule("disease", "diabetes", "symptom", "thirst", 0.85),
        create_causal_rule("disease", "diabetes", "symptom", "frequent_urination", 0.90),
        create_causal_rule("disease", "diabetes", "symptom", "fatigue", 0.70),

        # Hypertension
        create_causal_rule("disease", "hypertension", "symptom", "headache", 0.60),
        create_causal_rule("disease", "hypertension", "symptom", "dizziness", 0.55),

        # Anemia
        create_causal_rule("disease", "anemia", "symptom", "fatigue", 0.85),
        create_causal_rule("disease", "anemia", "symptom", "weakness", 0.80),
        create_causal_rule("disease", "anemia", "symptom", "pale_skin", 0.75),
    ]

    priors = {
        "disease": {
            "diabetes": 0.08,
            "hypertension": 0.25,
            "anemia": 0.06,
        }
    }

    reasoner = AbductiveReasoner(causal_rules, priors)

    # Complex observation set
    observations = [
        create_observation("symptom", "thirst"),
        create_observation("symptom", "frequent_urination"),
        create_observation("symptom", "fatigue"),
        create_observation("symptom", "headache"),
        create_observation("symptom", "dizziness"),
        create_observation("symptom", "weakness"),
    ]

    print("Observations:")
    print("-" * 80)
    for obs in observations:
        print(f"  {obs}")
    print()

    print("Abductive Reasoning (allowing multi-cause):")
    print("-" * 80)

    # Allow multiple diseases
    explanations = reasoner.explain(
        observations,
        max_hypotheses=10,
        allow_multi_cause=True
    )

    print("Top explanations:\n")
    for i, hyp in enumerate(explanations[:5], 1):
        diseases = list(hyp.explanation.values())
        if len(diseases) == 1:
            label = f"Single: {diseases[0]}"
        else:
            label = f"Multi: {' + '.join(diseases)}"

        print(f"{i}. {label}")
        print(f"   Score: {hyp.score():.4f}")
        print(f"   Explains {len(hyp.observations_explained)} symptoms: "
              f"{', '.join(sorted(hyp.observations_explained))}")
        print()

    # Find best multi-cause explanation
    multi_cause = [h for h in explanations if len(h.explanation) > 1]
    if multi_cause:
        best_multi = multi_cause[0]
        print("=" * 80)
        print(f"✓ Best Multi-Cause Explanation: {list(best_multi.explanation.values())}")
        print(f"  Score: {best_multi.score():.4f}")
        print(f"  Explains: {', '.join(sorted(best_multi.observations_explained))}")
        print("=" * 80)

    print()


# ============================================================================
# Example 4: Uncertain Evidence
# ============================================================================

def demo_uncertain_evidence():
    """Diagnosis with unreliable observations."""

    print("=" * 80)
    print("EXAMPLE 4: UNCERTAIN EVIDENCE".center(80))
    print("=" * 80)
    print()

    print("Scenario: Patient reports symptoms, but some are uncertain.")
    print("Question: How does observation confidence affect diagnosis?")
    print()

    # Causal rules (simple flu vs cold)
    causal_rules = [
        create_causal_rule("disease", "flu", "symptom", "fever", 0.90),
        create_causal_rule("disease", "flu", "symptom", "body_aches", 0.85),
        create_causal_rule("disease", "flu", "symptom", "fatigue", 0.80),

        create_causal_rule("disease", "cold", "symptom", "fever", 0.30),
        create_causal_rule("disease", "cold", "symptom", "runny_nose", 0.90),
        create_causal_rule("disease", "cold", "symptom", "fatigue", 0.60),
    ]

    priors = {"disease": {"flu": 0.10, "cold": 0.20}}
    reasoner = AbductiveReasoner(causal_rules, priors)

    # Observations with varying confidence
    observations = [
        create_observation("symptom", "fever", confidence=1.0),         # Certain
        create_observation("symptom", "body_aches", confidence=0.7),    # Somewhat sure
        create_observation("symptom", "fatigue", confidence=0.5),       # Uncertain
    ]

    print("Observations (with confidence):")
    print("-" * 80)
    for obs in observations:
        certainty = "certain" if obs.confidence > 0.9 else \
                   "likely" if obs.confidence > 0.7 else \
                   "uncertain"
        print(f"  {obs.variable}={obs.value} [{certainty}, conf={obs.confidence:.1f}]")
    print()

    print("Diagnosis:")
    print("-" * 80)
    explanations = reasoner.explain(observations, max_hypotheses=2)

    for i, hyp in enumerate(explanations, 1):
        disease = hyp.explanation.get("disease")
        print(f"{i}. {disease.upper()}")
        print(f"   Score: {hyp.score():.4f}")
        print(f"   Likelihood: {hyp.likelihood:.4f} (accounts for observation confidence)")
        print()

    print("✓ Uncertain observations reduce likelihood, affecting final score")
    print()


# ============================================================================
# Example 5: Hypothesis Testing
# ============================================================================

def demo_hypothesis_testing():
    """Compare two specific diagnostic hypotheses."""

    print("=" * 80)
    print("EXAMPLE 5: HYPOTHESIS TESTING".center(80))
    print("=" * 80)
    print()

    print("Scenario: Clinician suspects either strep throat or viral pharyngitis.")
    print("Question: Which hypothesis better explains the observations?")
    print()

    # Causal rules
    causal_rules = [
        # Strep throat (bacterial)
        create_causal_rule("disease", "strep", "symptom", "sore_throat", 0.95),
        create_causal_rule("disease", "strep", "symptom", "fever", 0.80),
        create_causal_rule("disease", "strep", "symptom", "swollen_lymph_nodes", 0.75),
        create_causal_rule("disease", "strep", "symptom", "white_patches", 0.70),

        # Viral pharyngitis
        create_causal_rule("disease", "viral", "symptom", "sore_throat", 0.90),
        create_causal_rule("disease", "viral", "symptom", "runny_nose", 0.70),
        create_causal_rule("disease", "viral", "symptom", "cough", 0.65),
        create_causal_rule("disease", "viral", "symptom", "fever", 0.50),
    ]

    priors = {"disease": {"strep": 0.05, "viral": 0.15}}
    reasoner = AbductiveReasoner(causal_rules, priors)

    # Observations
    observations = [
        create_observation("symptom", "sore_throat"),
        create_observation("symptom", "fever"),
        create_observation("symptom", "swollen_lymph_nodes"),
        create_observation("symptom", "white_patches"),
    ]

    print("Observations:")
    print("-" * 80)
    for obs in observations:
        print(f"  {obs}")
    print()

    # Compare specific hypotheses
    print("Hypothesis Testing:")
    print("-" * 80)

    h1 = {"disease": "strep"}
    h2 = {"disease": "viral"}

    scored_h1, scored_h2 = reasoner.compare_hypotheses(h1, h2, observations)

    print(f"H1: Strep Throat")
    print(f"    Score: {scored_h1.score():.4f}")
    print(f"    Likelihood: {scored_h1.likelihood:.4f}")
    print(f"    Prior: {scored_h1.prior:.4f}")
    print()

    print(f"H2: Viral Pharyngitis")
    print(f"    Score: {scored_h2.score():.4f}")
    print(f"    Likelihood: {scored_h2.likelihood:.4f}")
    print(f"    Prior: {scored_h2.prior:.4f}")
    print()

    winner = "Strep Throat" if scored_h1.score() > scored_h2.score() else "Viral Pharyngitis"
    ratio = max(scored_h1.score(), scored_h2.score()) / min(scored_h1.score(), scored_h2.score())

    print("=" * 80)
    print(f"✓ Winner: {winner} ({ratio:.1f}× more likely)")
    print("=" * 80)
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all abductive reasoning demos."""

    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + "ABDUCTIVE REASONING DEMONSTRATION".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run examples
    demo_simple_diagnosis()
    print("\n\n")

    demo_differential_diagnosis()
    print("\n\n")

    demo_multi_cause()
    print("\n\n")

    demo_uncertain_evidence()
    print("\n\n")

    demo_hypothesis_testing()
    print("\n\n")

    # Summary
    print("=" * 80)
    print("SUMMARY".center(80))
    print("=" * 80)
    print()

    print("What We Demonstrated:")
    print()

    print("1. ✅ Hypothesis Generation")
    print("   - Backward reasoning: observations → possible causes")
    print("   - Single-cause and multi-cause hypotheses")
    print("   - Causal rule-based generation")
    print()

    print("2. ✅ Bayesian Scoring")
    print("   - Likelihood: P(observations | hypothesis)")
    print("   - Prior: P(hypothesis) base rate")
    print("   - Complexity: Occam's razor penalty")
    print("   - Combined: (likelihood × prior) / complexity")
    print()

    print("3. ✅ Best Explanation Selection")
    print("   - Rank hypotheses by score")
    print("   - Return top-k explanations")
    print("   - Confidence thresholding")
    print()

    print("4. ✅ Differential Diagnosis")
    print("   - Multiple competing explanations")
    print("   - Evidence accumulation shifts probabilities")
    print("   - Discriminating between similar hypotheses")
    print()

    print("5. ✅ Multi-Cause Reasoning")
    print("   - Multiple diseases simultaneously")
    print("   - Composite explanations for complex symptoms")
    print("   - Coverage vs. parsimony tradeoff")
    print()

    print("6. ✅ Uncertain Evidence")
    print("   - Observation confidence weighting")
    print("   - Robustness to noisy data")
    print("   - Likelihood adjustments for reliability")
    print()

    print("Applications:")
    print("   - Medical diagnosis (symptoms → diseases)")
    print("   - Fault diagnosis (errors → root causes)")
    print("   - Scientific discovery (data → theories)")
    print("   - Natural language understanding (text → meaning)")
    print("   - Debugging (bug → code defect)")
    print()

    print("Research Alignment:")
    print("   - Peirce (1878): Origin of abductive reasoning")
    print("   - Josephson & Josephson (1996): Abductive Inference")
    print("   - Pearl (2000): Causality and explanation")
    print("   - Hobbs et al. (1993): Interpretation as Abduction")
    print()

    print("Key Insights:")
    print("   - Abduction ≠ Deduction: Finds likely explanations, not proofs")
    print("   - Combines likelihood (fit) + prior (plausibility) + parsimony (simplicity)")
    print("   - Multiple explanations possible → maintain uncertainty")
    print("   - Evidence accumulation refines hypotheses")
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()
