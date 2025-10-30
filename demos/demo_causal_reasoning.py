"""
Causal Reasoning Demo

Demonstrates Pearl's three-level causal hierarchy:
1. Association (observational): P(Y|X)
2. Intervention (do-calculus): P(Y|do(X=x))
3. Counterfactual (twin networks): P(Y_x|X',Y')

Example: Medical treatment scenario
"""

import sys
sys.path.insert(0, '.')

from HoloLoom.causal import (
    CausalNode, CausalEdge, CausalDAG, NodeType,
    CausalQuery, QueryType,
    InterventionEngine, CounterfactualEngine
)


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def demo_medical_treatment():
    """
    Medical treatment scenario.

    Causal structure:
        Age → Treatment (older patients get treatment more often)
        Age → Recovery (age affects recovery)
        Treatment → Recovery (treatment helps recovery)

    Questions we'll answer:
    1. Level 1: What's the correlation between treatment and recovery?
    2. Level 2: What's the CAUSAL effect of treatment on recovery?
    3. Level 3: Would patient have recovered without treatment?
    """

    print_section("CAUSAL REASONING ENGINE DEMO")

    print("Scenario: Medical Treatment Study")
    print("-" * 80)
    print("Variables:")
    print("  - Age: Patient age (young/elderly)")
    print("  - Treatment: Received experimental drug (0/1)")
    print("  - Recovery: Patient recovered (0/1)")
    print("")
    print("Causal Structure:")
    print("  Age → Treatment (confounds)")
    print("  Age → Recovery")
    print("  Treatment → Recovery")
    print("")

    # ========================================================================
    # 1. Build Causal DAG
    # ========================================================================

    print_section("STEP 1: Building Causal DAG")

    dag = CausalDAG()

    # Add nodes
    dag.add_node(CausalNode(
        "age",
        NodeType.OBSERVABLE,
        domain=["young", "elderly"],
        description="Patient age group"
    ))

    dag.add_node(CausalNode(
        "treatment",
        NodeType.OBSERVABLE,
        domain=[0, 1],
        description="Received experimental drug"
    ))

    dag.add_node(CausalNode(
        "recovery",
        NodeType.OBSERVABLE,
        domain=[0, 1],
        description="Patient recovered"
    ))

    # Add edges with strengths
    dag.add_edge(CausalEdge(
        "age", "treatment",
        strength=0.3,
        mechanism="Elderly patients more likely to receive treatment",
        confidence=0.9
    ))

    dag.add_edge(CausalEdge(
        "age", "recovery",
        strength=0.2,
        mechanism="Age affects baseline recovery rate",
        confidence=0.95
    ))

    dag.add_edge(CausalEdge(
        "treatment", "recovery",
        strength=0.6,
        mechanism="Treatment improves recovery",
        confidence=0.85
    ))

    print(f"✓ Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    print("")
    print("Edges:")
    for (src, tgt), edge in dag.edges.items():
        print(f"  {src} → {tgt} (strength={edge.strength:.2f})")
    print("")

    # Graph properties
    print("Graph Properties:")
    print(f"  Confounders (Treatment, Recovery): {dag.find_confounders('treatment', 'recovery')}")
    print(f"  Mediators (Treatment, Recovery): {dag.find_mediators('treatment', 'recovery')}")
    print(f"  Topological order: {dag.topological_order()}")
    print("")

    # ========================================================================
    # 2. Level 1: Association (Observational)
    # ========================================================================

    print_section("LEVEL 1: ASSOCIATION (Observational Queries)")

    print("Question: What's the correlation between treatment and recovery?")
    print("-" * 80)
    print("")
    print("This is a Level 1 (observational) question.")
    print("We observe: P(Recovery=1 | Treatment=1) = 0.75")
    print("But this includes confounding from age!")
    print("")
    print("⚠️  Observational correlation ≠ Causal effect")
    print("   Confounders create spurious associations.")
    print("")

    # ========================================================================
    # 3. Level 2: Intervention (Causal Effects)
    # ========================================================================

    print_section("LEVEL 2: INTERVENTION (Causal Effects)")

    engine = InterventionEngine(dag)

    print("Question: What's the CAUSAL effect of treatment on recovery?")
    print("-" * 80)
    print("")

    # Identify causal effect
    identification = engine.identify_causal_effect("treatment", "recovery")

    print(f"Identifiable: {identification.identifiable}")
    print(f"Method: {identification.identification_method}")
    if identification.adjustment_set:
        print(f"Adjustment set: {identification.adjustment_set}")
    print("")
    print("Explanation:")
    print(identification.explanation)
    print("")

    # Show what do() operator does
    print("do() Operator - Graph Surgery:")
    print("-" * 80)
    print("Original graph: Age → Treatment → Recovery")
    print("                Age → Recovery")
    print("")
    print("After do(Treatment=1):")
    print("  Remove: Age → Treatment (break confounding)")
    print("  Keep: Treatment → Recovery (causal effect)")
    print("  Keep: Age → Recovery (direct effect)")
    print("")

    result = engine.do({"treatment": 1})
    print(f"✓ Applied do(Treatment=1)")
    print(f"  Edges removed: Age → Treatment")
    print(f"  Edges kept: {len(result.mutilated_graph.edges)}")
    print("")

    # Answer causal query
    query = CausalQuery(
        query_type=QueryType.INTERVENTION,
        outcome="recovery",
        treatment="treatment",
        treatment_value=1,
        description="What is the causal effect of treatment on recovery?"
    )

    answer = engine.query(query)

    print("Causal Effect Estimate:")
    print("-" * 80)
    print(answer.to_natural_language())
    print("")

    # ========================================================================
    # 4. Level 3: Counterfactual (Twin Networks)
    # ========================================================================

    print_section("LEVEL 3: COUNTERFACTUAL (Twin Networks)")

    cf_engine = CounterfactualEngine(dag)

    print("Question: Would this patient have recovered WITHOUT treatment?")
    print("-" * 80)
    print("")
    print("Patient facts:")
    print("  Age: elderly")
    print("  Treatment: 1 (received treatment)")
    print("  Recovery: 1 (recovered)")
    print("")
    print("Counterfactual question:")
    print("  What if Treatment = 0? (no treatment)")
    print("")

    # Counterfactual query
    result = cf_engine.counterfactual(
        intervention={"treatment": 0},
        evidence={"age": "elderly", "treatment": 1, "recovery": 1},
        query="recovery"
    )

    print("Counterfactual Outcome:")
    print(f"  Factual: Recovery = {result.factual_outcome} (what happened)")
    print(f"  Counterfactual: Recovery = {result.counterfactual_outcome} (what would have happened)")
    print(f"  Probability: {result.probability:.3f}")
    print("")

    print("Interpretation:")
    if result.counterfactual_outcome == 0:
        print("  ✓ Treatment was likely NECESSARY for recovery")
        print("    Patient would NOT have recovered without treatment")
    else:
        print("  ⚠ Patient might have recovered anyway")
        print("    Treatment may not have been necessary")
    print("")

    # Probability of necessity
    print("Probability of Necessity (PN):")
    print("-" * 80)
    print("PN = P(Recovery_0=0 | Treatment=1, Recovery=1)")
    print("'Was treatment necessary for recovery?'")
    print("")

    necessity = cf_engine.probability_of_necessity(
        treatment="treatment",
        outcome="recovery",
        evidence={"treatment": 1, "recovery": 1}
    )

    print(f"PN = {necessity:.3f}")
    if necessity > 0.7:
        print("✓ HIGH: Treatment was likely necessary")
    elif necessity > 0.4:
        print("⚠ MEDIUM: Treatment may have been necessary")
    else:
        print("✗ LOW: Treatment probably wasn't necessary")
    print("")

    # Probability of sufficiency
    print("Probability of Sufficiency (PS):")
    print("-" * 80)
    print("PS = P(Recovery_1=1 | Treatment=0, Recovery=0)")
    print("'Would treatment be sufficient for recovery?'")
    print("")

    sufficiency = cf_engine.probability_of_sufficiency(
        treatment="treatment",
        outcome="recovery",
        evidence={"treatment": 0, "recovery": 0}
    )

    print(f"PS = {sufficiency:.3f}")
    if sufficiency > 0.7:
        print("✓ HIGH: Treatment would likely be sufficient")
    elif sufficiency > 0.4:
        print("⚠ MEDIUM: Treatment might be sufficient")
    else:
        print("✗ LOW: Treatment probably wouldn't be sufficient")
    print("")

    # ========================================================================
    # Summary
    # ========================================================================

    print_section("SUMMARY: Pearl's Causal Hierarchy")

    print("Level 1: ASSOCIATION (Observational)")
    print("  Question: P(Recovery | Treatment)")
    print("  Answer: 0.75 (includes confounding)")
    print("  ⚠️  Cannot distinguish causation from correlation")
    print("")

    print("Level 2: INTERVENTION (Causal)")
    print("  Question: P(Recovery | do(Treatment=1))")
    print("  Answer: Use backdoor adjustment, control for Age")
    print("  ✓ Identifies true causal effect")
    print("")

    print("Level 3: COUNTERFACTUAL (Retrospective)")
    print("  Question: P(Recovery_0 | Treatment=1, Recovery=1)")
    print(f"  Answer: PN={necessity:.3f}, PS={sufficiency:.3f}")
    print("  ✓ Answers 'what if?' questions about specific cases")
    print("")

    print("=" * 80)
    print("Key Insight: Each level requires stronger assumptions!")
    print("=" * 80)
    print("  Level 1: Just needs data")
    print("  Level 2: Needs causal graph + identification")
    print("  Level 3: Needs structural equations + exogenous variables")
    print("")

    print("✅ Causal Reasoning Engine Demo Complete!")
    print("")


if __name__ == "__main__":
    demo_medical_treatment()
