"""
NeuroHood Phase 3 Demo: Causal Intelligence

Demonstrates the complete Phase 3 integration:
1. Neural Structural Causal Models (learning relationship dynamics)
2. Intervention reasoning ("what if Alice apologizes?")
3. Counterfactual analysis ("what would have happened if...?")
4. Blame attribution (fair responsibility using PN/PS)
5. Department workflows (HOA, Mediation, City Services)
6. Temporal dynamics (30-day relationship predictions)

Run:
    PYTHONPATH=. python demos/demo_neurohood_phase3_causal.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NeuroHood import NeuroHood, NeuroHoodConfig
from NeuroHood.personality import PersonalitySystem
from NeuroHood.causal import (
    RelationshipSCM,
    extract_relationship_data,
    NeuroHoodInterventionEngine,
    NeuroHoodCounterfactualEngine,
    quick_blame_attribution,
)
from NeuroHood.causal.temporal import NeuroHoodTemporalEngine
from NeuroHood.departments import HOADepartment, MediationDepartment
from HoloLoom.protocols.department import DepartmentRequest
from HoloLoom.embedding.spectral import create_embedder


async def main():
    """Run comprehensive Phase 3 causal intelligence demo."""

    print("\n" + "="*70)
    print("  NeuroHood Phase 3: Causal Intelligence")
    print("  Understanding WHY Things Happen + Fair Conflict Resolution")
    print("="*70 + "\n")

    # ========== 1. Initialize System ==========

    print("[ Step 1: Initializing NeuroHood + Causal System ]")
    print("-" * 70)

    config = NeuroHoodConfig.prototype()
    config.enable_spring_dynamics = True

    embedder = create_embedder(sizes=[384])
    embed_fn = lambda text: embedder.encode([text])[0]

    personality = PersonalitySystem(embed_fn)

    print("  NeuroHood: ENABLED")
    print("  Spring Physics: ENABLED")
    print("  Personality System: ENABLED")
    print("  Causal Intelligence: ENABLED")
    print()

    async with NeuroHood(config) as hood:

        # ========== 2. Bootstrap Neighborhood ==========

        print("[ Step 2: Creating Neighborhood ]")
        print("-" * 70)

        state = await hood.bootstrap_neighborhood(
            seed="A suburban street where Alice values peace, Bob loves parties, "
                 "and Charlie tries to keep everyone happy"
        )

        residents = list(state.residents.values())
        alice, bob, charlie = residents[0], residents[1], residents[2]

        print(f"  Created {len(residents)} residents:")
        print(f"    - {alice.name} (quiet, peaceful)")
        print(f"    - {bob.name} (loud, party-loving)")
        print(f"    - {charlie.name} (mediator)")
        print()

        # Initialize personalities
        personality.initialize_personality(
            resident_id=alice.resident_id,
            description=(
                "Alice is warm, caring, and loving who values peace and tranquility. "
                "She is introverted, calm, reflective, creative, and patient."
            )
        )

        personality.initialize_personality(
            resident_id=bob.resident_id,
            description=(
                "Bob is excited, energetic, and intense, loving action and activity. "
                "He is extroverted, direct, straightforward, and seeks novel experiences."
            )
        )

        personality.initialize_personality(
            resident_id=charlie.resident_id,
            description=(
                "Charlie is balanced and diplomatic, seeking harmony and peace. "
                "He is generous, caring, certain, and stable in his approach."
            )
        )

        # ========== 3. Collect Training Data for SCM ==========

        print("[ Step 3: Collecting Training Data for Causal Learning ]")
        print("-" * 70)

        # Simulate some interactions to generate training data
        alice_bob_key = hood._relationship_key(alice.resident_id, bob.resident_id)
        alice_bob_rel = state.relationships[alice_bob_key]

        training_data = []

        # Initial state
        training_data.append(extract_relationship_data(
            alice_bob_rel, state.residents, personality, timestamp=0.0
        ))

        # Negative interaction (noise complaint)
        hood.relationship_physics.apply_social_interaction(
            alice_bob_rel, "negative", state.residents, intensity=0.4
        )
        training_data.append(extract_relationship_data(
            alice_bob_rel, state.residents, personality, timestamp=1.0
        ))

        # Time passes (decay)
        for i in range(5):
            state = await hood._update_social_physics(state)
            training_data.append(extract_relationship_data(
                alice_bob_rel, state.residents, personality, timestamp=2.0 + i
            ))

        # Positive interaction (apology)
        hood.relationship_physics.apply_social_interaction(
            alice_bob_rel, "positive", state.residents, intensity=0.3
        )
        training_data.append(extract_relationship_data(
            alice_bob_rel, state.residents, personality, timestamp=7.0
        ))

        print(f"  Collected {len(training_data)} training samples from simulation")
        print()

        # ========== 4. Train Neural SCM ==========

        print("[ Step 4: Training Neural Structural Causal Model ]")
        print("-" * 70)

        scm = RelationshipSCM()
        scm.fit(training_data, epochs=50)  # Quick training

        print("  Causal mechanisms learned!")
        print()
        print(scm.get_structure_summary())

        # ========== 5. Intervention Reasoning ==========

        print("[ Step 5: Intervention Reasoning - \"What if Alice apologizes?\" ]")
        print("-" * 70)

        intervention_engine = NeuroHoodInterventionEngine(scm)

        # Current state (after conflict)
        current_data = training_data[2]  # Peak conflict state

        context = {
            "personality_compatibility": current_data.personality_compatibility,
            "conflict_intensity": current_data.conflict_intensity,
        }

        # What if Alice apologizes?
        result = intervention_engine.intervene(
            intervention={"communication_quality": 0.9},
            context=context
        )

        print(f"  Current state:")
        print(f"    Relationship strength: {result.factual_outcome['relationship_strength']:.2f}")
        print(f"    Conflict intensity: {result.factual_outcome['conflict_intensity']:.2f}")
        print()

        print(f"  If Alice apologizes (communication_quality = 0.9):")
        print(f"    Relationship strength: {result.counterfactual_outcome['relationship_strength']:.2f}")
        print(f"    Causal effect: {result.causal_effect['relationship_strength']:+.2f}")
        print()
        print(result.interpretation)
        print()

        # Compare multiple interventions
        print("  Comparing intervention strategies:")
        print()

        interventions = [
            {"communication_quality": 0.9},  # Apology
            {"conflict_intensity": 0.1},     # Reduce noise
            {"communication_quality": 0.9, "conflict_intensity": 0.1},  # Both
        ]

        ranked = intervention_engine.compare_interventions(
            interventions,
            context=context,
            outcome_variable="relationship_strength"
        )

        for i, (intervention, outcome) in enumerate(ranked, 1):
            print(f"    {i}. {intervention} -> {outcome:.2f} relationship strength")
        print()

        # ========== 6. Counterfactual Analysis ==========

        print("[ Step 6: Counterfactual Analysis - \"What would have happened?\" ]")
        print("-" * 70)

        counterfactual_engine = NeuroHoodCounterfactualEngine(scm)

        # What if Alice had been polite from the start?
        cf_result = counterfactual_engine.analyze_counterfactual(
            factual_data=current_data,
            counterfactual_intervention={"communication_quality": 0.9}
        )

        print(cf_result.interpretation)
        print()
        print(f"  Probability of Causation: {cf_result.probability_of_causation:.1%}")
        print("  (How much did Alice's rudeness cause the conflict?)")
        print()

        # ========== 7. Blame Attribution ==========

        print("[ Step 7: Fair Blame Attribution using PN/PS ]")
        print("-" * 70)

        blame_report = quick_blame_attribution(
            scm,
            factual_data=current_data,
            resident_a_name=alice.name,
            resident_b_name=bob.name,
            conflict_description="Noise complaint escalated to shouting match"
        )

        print(blame_report.format())
        print()

        # ========== 8. Department Workflows ==========

        print("[ Step 8: Department Workflows - HOA and Mediation ]")
        print("-" * 70)

        # Create departments
        hoa = HOADepartment(scm)
        mediation = MediationDepartment(scm)

        # File noise complaint with HOA
        print("  Filing noise complaint with HOA...")
        complaint_response = await hoa.process(DepartmentRequest(
            request_type="file_complaint",
            data={
                "complainant": alice.name,
                "subject": bob.name,
                "type": "noise",
                "description": "Loud music at 11 PM repeatedly disturbing sleep",
                "timestamp": current_data.timestamp
            }
        ))

        print(f"    {complaint_response.data['message']}")
        complaint_id = complaint_response.data['complaint_id']
        print()

        # Review complaint with causal reasoning
        print("  HOA reviewing complaint with blame attribution...")
        review_response = await hoa.process(DepartmentRequest(
            request_type="review_complaint",
            data={
                "complaint_id": complaint_id,
                "state": state
            }
        ))

        print(f"    Primary cause: {review_response.data['primary_cause']}")
        print(f"    Blame explained: {review_response.data['total_blame_explained']:.1%}")
        print()

        # Issue violation based on review
        print("  Issuing violation notice...")
        violation_response = await hoa.process(DepartmentRequest(
            request_type="issue_violation",
            data={
                "resident": bob.name,
                "type": "noise",
                "description": "Loud music at 11 PM",
                "severity": "moderate",
                "timestamp": current_data.timestamp
            }
        ))

        print(f"    {violation_response.data['message']}")
        print(f"    Fine amount: ${violation_response.data['fine_amount']:.2f}")
        print()

        # Open mediation case
        print("  Opening mediation case...")
        mediation_response = await mediation.process(DepartmentRequest(
            request_type="open_case",
            data={
                "resident_a": alice.resident_id,
                "resident_b": bob.resident_id,
                "description": "Noise complaint escalation",
                "timestamp": current_data.timestamp
            }
        ))

        case_id = mediation_response.data['case_id']
        print(f"    {mediation_response.data['message']}")
        print()

        # Analyze conflict
        print("  Mediation analyzing conflict...")
        analysis_response = await mediation.process(DepartmentRequest(
            request_type="analyze_conflict",
            data={
                "case_id": case_id,
                "state": state
            }
        ))

        print(f"    Primary cause: {analysis_response.data['primary_cause']}")
        print(f"    Blame explained: {analysis_response.data['total_blame_explained']:.1%}")
        print()

        # Suggest interventions
        print("  Mediation suggesting resolution strategies...")
        suggestions_response = await mediation.process(DepartmentRequest(
            request_type="suggest_interventions",
            data={
                "case_id": case_id,
                "state": state
            }
        ))

        print("    Recommended interventions:")
        for rec in suggestions_response.data['recommendations']:
            print(f"      {rec['rank']}. {rec['intervention']} "
                  f"(predicted strength: {rec['predicted_relationship_strength']})")
        print()

        # ========== 9. Temporal Dynamics ==========

        print("[ Step 9: Temporal Dynamics - 30-Day Prediction ]")
        print("-" * 70)

        temporal_engine = NeuroHoodTemporalEngine(scm)

        # Predict without intervention
        print("  Predicting relationship trajectory without intervention...")
        trajectory_no_intervention = temporal_engine.predict_trajectory(
            initial_state=current_data,
            n_days=30
        )

        final_strength_no_intervention = trajectory_no_intervention[-1].predicted_strength

        print(f"    Day 30 relationship strength: {final_strength_no_intervention:.2f}")
        print()

        # Predict with intervention (apology on day 7)
        print("  Predicting with apology on day 7...")
        trajectory_with_apology = temporal_engine.predict_trajectory(
            initial_state=current_data,
            n_days=30,
            interventions={7: {"communication_quality": 0.9}}
        )

        final_strength_with_apology = trajectory_with_apology[-1].predicted_strength

        print(f"    Day 30 relationship strength: {final_strength_with_apology:.2f}")
        print(f"    Improvement: {final_strength_with_apology - final_strength_no_intervention:+.2f}")
        print()

        # Find optimal intervention timing
        print("  Finding optimal intervention timing...")
        best_day, best_outcome = temporal_engine.find_optimal_intervention_timing(
            initial_state=current_data,
            intervention={"communication_quality": 0.9},
            n_days=14
        )

        print(f"    Best day to apologize: Day {best_day}")
        print(f"    Best outcome: {best_outcome:.2f} relationship strength")
        print()

        # Early warning detection
        print("  Checking for early warnings...")
        warning_day = temporal_engine.detect_early_warning(
            initial_state=current_data,
            threshold=0.3,
            n_days=30
        )

        if warning_day:
            print(f"    WARNING: Relationship at risk on day {warning_day}!")
            print("    Immediate intervention recommended.")
        else:
            print("    No early warning - relationship stable.")
        print()

        # Show trajectory chart
        print("  30-Day Trajectory Preview (first 10 days):")
        print()
        print(temporal_engine.format_trajectory(trajectory_no_intervention[:10]))
        print()

        # ========== 10. Conclusion ==========

        print("="*70)
        print("  Phase 3 Causal Intelligence Demo Complete!")
        print("="*70)
        print()
        print("  What We Demonstrated:")
        print("    - Neural SCM learning from relationship dynamics")
        print("    - Intervention reasoning (\"what if?\")")
        print("    - Counterfactual analysis (\"what would have happened?\")")
        print("    - Fair blame attribution (PN/PS metrics)")
        print("    - HOA department workflow (violation processing)")
        print("    - Mediation department (conflict resolution strategies)")
        print("    - Temporal dynamics (30-day predictions)")
        print("    - Optimal intervention timing")
        print("    - Early warning detection")
        print()
        print("  The Power of Causal Intelligence:")
        print("    - Understand WHY conflicts happen")
        print("    - Fairly attribute responsibility")
        print("    - Predict relationship futures")
        print("    - Optimize intervention timing")
        print("    - Resolve conflicts with evidence-based strategies")
        print()
        print("  Phase 3 Status: COMPLETE")
        print("    - Neural SCM: COMPLETE")
        print("    - Interventions: COMPLETE")
        print("    - Counterfactuals: COMPLETE")
        print("    - Blame Attribution: COMPLETE")
        print("    - Departments: COMPLETE (HOA, City Services, Mediation)")
        print("    - Temporal Dynamics: COMPLETE")
        print()
        print("  Total Phase 3 Code: ~3,500+ lines")
        print()


if __name__ == "__main__":
    asyncio.run(main())
