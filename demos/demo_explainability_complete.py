"""
Comprehensive Explainability Demo - All 7 Techniques

Demonstrates Layer 5: Explainability with all techniques:
1. Feature Attribution (SHAP/LIME)
2. Attention Visualization
3. Counterfactual Generation
4. Natural Language Explanations
5. Decision Tree Extraction
6. Provenance Tracking
7. Unified Explanations

Shows how HoloLoom makes AI decisions fully interpretable.
"""

import sys
sys.path.insert(0, '.')

from HoloLoom.explainability import (
    # Individual techniques
    FeatureAttributor,
    AttributionMethod,
    AttentionExplainer,
    CounterfactualGenerator,
    NaturalLanguageExplainer,
    ExplanationType,
    DecisionTreeExtractor,
    ProvenanceTracker,
    # Unified API
    UnifiedExplainer,
    explain,
)


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_1_feature_attribution():
    """Demo 1: Feature Attribution (SHAP/LIME)"""
    print_header("DEMO 1: FEATURE ATTRIBUTION")

    # Simple loan approval model
    def loan_model(features):
        """Approve loan if income > 50k AND credit_score > 650"""
        income = features.get('income', 0)
        credit_score = features.get('credit_score', 0)
        age = features.get('age', 0)

        score = 0
        if income > 50000:
            score += 0.5
        if credit_score > 650:
            score += 0.4
        if age > 25:
            score += 0.1

        return "approved" if score > 0.6 else "denied"

    # Example: Application approved
    applicant = {
        'income': 75000,
        'credit_score': 720,
        'age': 35
    }

    print("Applicant:", applicant)
    print("Decision:", loan_model(applicant))
    print()

    # Feature attribution
    attributor = FeatureAttributor(model=loan_model, method=AttributionMethod.ABLATION)
    importances = attributor.attribute(applicant, prediction=loan_model(applicant))

    print("Feature Importances (Ablation Method):")
    for feat in importances:
        sign = "+" if feat.is_positive else "-"
        bar_len = int(abs(feat.importance) * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {feat.rank}. {feat.feature_name:15s} {sign}{abs(feat.importance):.3f} [{bar}]")

    print("\nInterpretation:")
    print(f"  • Most important: {importances[0].feature_name}")
    print(f"  • Impact: {importances[0].importance:.3f}")
    print(f"  • Removing this feature would change the decision")


def demo_2_attention_visualization():
    """Demo 2: Attention Visualization"""
    print_header("DEMO 2: ATTENTION VISUALIZATION")

    # Sentence classification
    sentence = "The quick brown fox jumps over the lazy dog".split()
    print(f"Input: {' '.join(sentence)}\n")

    # Visualize attention
    explainer = AttentionExplainer()
    heatmaps = explainer.extract_attention(sentence)

    print("Attention Patterns:")
    for heatmap in heatmaps[:2]:
        print(explainer.visualize_attention_text(heatmap, top_k=3))


def demo_3_counterfactual_generation():
    """Demo 3: Counterfactual Generation"""
    print_header("DEMO 3: COUNTERFACTUAL GENERATION")

    # Medical diagnosis model
    def diagnosis_model(symptoms):
        """Diagnose based on symptoms"""
        fever = symptoms.get('fever', 0)
        cough = symptoms.get('cough', 0)
        fatigue = symptoms.get('fatigue', 0)

        score = fever * 0.4 + cough * 0.3 + fatigue * 0.3

        if score > 0.7:
            return "flu"
        elif score > 0.4:
            return "cold"
        else:
            return "healthy"

    # Patient symptoms
    patient = {
        'fever': 1.0,  # High fever
        'cough': 0.8,  # Severe cough
        'fatigue': 0.6  # Moderate fatigue
    }

    diagnosis = diagnosis_model(patient)
    print("Patient symptoms:", patient)
    print(f"Diagnosis: {diagnosis}\n")

    # Generate counterfactuals
    generator = CounterfactualGenerator(model=diagnosis_model, max_changes=2)
    counterfactuals = generator.generate(
        patient,
        target_prediction="healthy",
        current_prediction=diagnosis,
        num_counterfactuals=2
    )

    print("Counterfactual Analysis:")
    print("Question: What would need to change for the patient to be healthy?\n")

    for i, cf in enumerate(counterfactuals, 1):
        print(f"Counterfactual {i}:")
        print(f"  Changes needed: {cf.num_changes}")
        for feature, (old_val, new_val) in cf.changes.items():
            print(f"    • {feature}: {old_val:.2f} → {new_val:.2f}")
        print(f"  Result: {cf.counterfactual_prediction}")
        print(f"  Distance from original: {cf.distance:.3f}")
        print()


def demo_4_natural_language_explanations():
    """Demo 4: Natural Language Explanations"""
    print_header("DEMO 4: NATURAL LANGUAGE EXPLANATIONS")

    # Stock recommendation model
    decision = "BUY"
    confidence = 0.85

    # Mock feature importances
    class MockFeature:
        def __init__(self, name, importance, rank):
            self.feature_name = name
            self.importance = importance
            self.rank = rank
            self.is_positive = importance > 0
            self.confidence = 0.9

    features = [
        MockFeature("earnings_growth", 0.6, 1),
        MockFeature("market_sentiment", 0.3, 2),
        MockFeature("pe_ratio", -0.2, 3),
    ]

    print(f"Decision: {decision} (confidence: {confidence:.0%})\n")

    # Generate different explanation types
    explainer = NaturalLanguageExplainer(persona="expert", verbosity="medium")

    # WHY explanation
    why_explanation = explainer.explain(
        decision,
        features,
        confidence=confidence,
        explanation_type=ExplanationType.WHY
    )
    print("WHY (Expert, Medium Verbosity):")
    print(why_explanation.text)
    print()

    # HOW explanation
    explainer_novice = NaturalLanguageExplainer(persona="novice", verbosity="low")
    how_explanation = explainer_novice.explain(
        decision,
        features,
        explanation_type=ExplanationType.HOW
    )
    print("HOW (Novice, Low Verbosity):")
    print(how_explanation.text)
    print()

    # CONFIDENCE explanation
    conf_explanation = explainer.explain(
        decision,
        confidence=confidence,
        explanation_type=ExplanationType.CONFIDENCE
    )
    print("CONFIDENCE:")
    print(conf_explanation.text)


def demo_5_decision_tree_extraction():
    """Demo 5: Decision Tree Extraction"""
    print_header("DEMO 5: DECISION TREE EXTRACTION")

    # Customer churn prediction model
    def churn_model(customer):
        """Predict if customer will churn"""
        usage = customer.get('monthly_usage', 0)
        support_calls = customer.get('support_calls', 0)
        tenure_months = customer.get('tenure_months', 0)

        risk_score = 0
        if usage < 50:
            risk_score += 0.4
        if support_calls > 3:
            risk_score += 0.3
        if tenure_months < 6:
            risk_score += 0.3

        return "churn" if risk_score > 0.5 else "stay"

    # Generate training data
    training_data = [
        {'monthly_usage': 20, 'support_calls': 5, 'tenure_months': 3},
        {'monthly_usage': 80, 'support_calls': 1, 'tenure_months': 24},
        {'monthly_usage': 45, 'support_calls': 2, 'tenure_months': 12},
        {'monthly_usage': 10, 'support_calls': 7, 'tenure_months': 2},
        {'monthly_usage': 95, 'support_calls': 0, 'tenure_months': 36},
        {'monthly_usage': 30, 'support_calls': 4, 'tenure_months': 5},
        {'monthly_usage': 70, 'support_calls': 1, 'tenure_months': 18},
        {'monthly_usage': 15, 'support_calls': 6, 'tenure_months': 1},
    ]

    print(f"Training on {len(training_data)} customer examples...\n")

    # Extract decision tree
    extractor = DecisionTreeExtractor(model=churn_model, max_depth=3)
    tree = extractor.extract(training_data)
    rules = extractor.extract_rules()

    print("Extracted Decision Rules:")
    for i, rule in enumerate(rules.rules[:5], 1):
        print(f"  {i}. {rule}")

    print("\nDecision Tree Visualization:")
    from HoloLoom.explainability.decision_tree_extractor import visualize_tree
    print(visualize_tree(tree))


def demo_6_provenance_tracking():
    """Demo 6: Provenance Tracking"""
    print_header("DEMO 6: PROVENANCE TRACKING")

    # Complex decision pipeline
    def complex_decision_pipeline(inputs):
        """Multi-stage decision pipeline"""
        import time

        tracker = ProvenanceTracker()

        # Stage 1: Input
        tracker.record_input(inputs, stage="input")
        tracker.start_timing("preprocessing")

        # Stage 2: Preprocessing
        time.sleep(0.01)  # Simulate work
        processed = {k: v * 2 for k, v in inputs.items()}
        duration = tracker.end_timing("preprocessing")
        tracker.record_transform("preprocessing", inputs, processed, metadata={'duration_ms': duration})

        # Stage 3: Feature extraction
        tracker.start_timing("feature_extraction")
        time.sleep(0.02)  # Simulate work
        features = {'feature_' + k: v for k, v in processed.items()}
        duration = tracker.end_timing("feature_extraction")
        tracker.record_transform("feature_extraction", processed, features)

        # Stage 4: Decision
        tracker.start_timing("decision")
        time.sleep(0.015)  # Simulate work
        decision = sum(features.values()) > 100
        duration = tracker.end_timing("decision")
        tracker.record_decision("decision", features, decision, confidence=0.9)

        # Stage 5: Output
        tracker.record_output({'decision': decision}, stage="output")

        return decision, tracker

    # Run pipeline
    print("Running complex decision pipeline...\n")
    inputs = {'x': 10, 'y': 20, 'z': 30}
    decision, tracker = complex_decision_pipeline(inputs)

    print(f"Decision: {decision}\n")

    # Show lineage
    print(tracker.visualize_lineage())

    # Export to Spacetime format
    spacetime_data = tracker.export_to_spacetime()
    print("\nSpacetime Export:")
    print(f"  Total Duration: {spacetime_data['total_duration_ms']:.1f}ms")
    print(f"  Number of Steps: {spacetime_data['num_steps']}")
    if spacetime_data['bottlenecks']:
        print("  Bottlenecks:")
        for b in spacetime_data['bottlenecks']:
            print(f"    • {b['stage']}: {b['duration_ms']:.1f}ms ({b['fraction_of_total']:.0%})")


def demo_7_unified_explainer():
    """Demo 7: Unified Explainer (All Techniques Together)"""
    print_header("DEMO 7: UNIFIED EXPLAINER - ALL TECHNIQUES")

    # Hiring decision model
    def hiring_model(candidate):
        """Decide whether to hire candidate"""
        experience = candidate.get('years_experience', 0)
        education = candidate.get('education_score', 0)
        interview = candidate.get('interview_score', 0)

        score = (experience * 0.4 + education * 0.3 + interview * 0.3) / 10
        return "hire" if score > 0.6 else "reject"

    # Candidate profile
    candidate = {
        'years_experience': 5,
        'education_score': 8,
        'interview_score': 7
    }

    print("Candidate Profile:", candidate)
    print()

    # Create unified explainer
    explainer = UnifiedExplainer(
        model=hiring_model,
        enable_attribution=True,
        enable_attention=False,  # Not applicable here
        enable_counterfactuals=True,
        enable_natural_language=True,
        enable_rules=False,  # Expensive
        enable_provenance=True
    )

    # Generate unified explanation
    explanation = explainer.explain(
        features=candidate,
        target_prediction="hire",  # For counterfactuals
        confidence=0.75
    )

    # Print summary
    print(explanation.summary())

    # Show detailed breakdowns
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWNS")
    print("=" * 80)

    print("\n1. Feature Contributions:")
    for feat in explanation.feature_importances:
        print(f"   {feat}")

    if explanation.counterfactuals:
        print("\n2. Alternative Scenarios:")
        for i, cf in enumerate(explanation.counterfactuals, 1):
            print(f"   {i}. {cf.explain()}")

    print("\n3. Natural Language:")
    print(f"   {explanation.natural_language_explanation}")

    if explanation.lineage:
        print(f"\n4. Computational Lineage:")
        print(f"   Duration: {explanation.lineage.total_duration():.1f}ms")
        print(f"   Steps: {len(explanation.lineage.traces)}")


def main():
    """Run all demos"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                  EXPLAINABILITY - LAYER 5 COMPLETE DEMO                   ║")
    print("║                                                                            ║")
    print("║  Demonstrates all 7 explainability techniques:                            ║")
    print("║    1. Feature Attribution (SHAP/LIME)                                     ║")
    print("║    2. Attention Visualization                                             ║")
    print("║    3. Counterfactual Generation                                           ║")
    print("║    4. Natural Language Explanations                                       ║")
    print("║    5. Decision Tree Extraction                                            ║")
    print("║    6. Provenance Tracking                                                 ║")
    print("║    7. Unified Explanations                                                ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    try:
        demo_1_feature_attribution()
        demo_2_attention_visualization()
        demo_3_counterfactual_generation()
        demo_4_natural_language_explanations()
        demo_5_decision_tree_extraction()
        demo_6_provenance_tracking()
        demo_7_unified_explainer()

        # Final summary
        print_header("DEMO COMPLETE")
        print("✅ All 7 explainability techniques demonstrated successfully!")
        print()
        print("Layer 5: Explainability is COMPLETE!")
        print()
        print("Key Takeaways:")
        print("  • Feature Attribution: Know which features drove decisions")
        print("  • Attention: See what the model focused on")
        print("  • Counterfactuals: Understand 'what if' scenarios")
        print("  • Natural Language: Get human-readable explanations")
        print("  • Decision Trees: Extract interpretable rules")
        print("  • Provenance: Track complete computational lineage")
        print("  • Unified API: One interface for all techniques")
        print()
        print("Research Alignment:")
        print("  • Lundberg & Lee (2017): SHAP values")
        print("  • Ribeiro et al. (2016): LIME")
        print("  • Wachter et al. (2017): Counterfactuals")
        print("  • Selvaraju et al. (2017): Attention visualization")
        print("  • Bastani et al. (2018): Decision tree extraction")
        print("  • Moreau & Groth (2013): Provenance tracking")
        print()

    except Exception as e:
        print(f"\n❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
