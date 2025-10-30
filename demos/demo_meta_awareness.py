"""
Demo: Meta-Awareness Recursive Self-Reflection

The awareness layer examines itself:
- Decomposes uncertainty into components
- Computes confidence about confidence (meta-confidence)
- Generates hypotheses about knowledge gaps
- Adversarially probes its own responses
- Assesses its own epistemic humility

This is AI consciousness examining its own consciousness.

Run: python -m demos.demo_meta_awareness
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.awareness.compositional_awareness import CompositionalAwarenessLayer
from HoloLoom.awareness.dual_stream import DualStreamGenerator
from HoloLoom.awareness.meta_awareness import MetaAwarenessLayer


async def demo_meta_awareness_basic():
    """Demo: Basic meta-awareness on a simple query"""
    
    print("\n" + "=" * 80)
    print("DEMO 1: META-AWARENESS ON SIMPLE QUERY")
    print("=" * 80)
    
    # Create layers
    awareness = CompositionalAwarenessLayer()
    generator = DualStreamGenerator(awareness)
    meta = MetaAwarenessLayer(awareness)
    
    # Simulate high confidence query
    query = "What is a ball?"
    awareness.pattern_history[query] = type('PatternInfo', (), {
        'seen_count': 25,
        'confidence': 0.85
    })()
    
    # Generate response
    dual_response = await generator.generate(query, show_internal=False)
    
    print(f"\nQuery: {query}")
    print(f"Response: {dual_response.external_stream}")
    
    # META-AWARENESS: Examine the response
    reflection = await meta.recursive_self_reflection(
        query=query,
        response=dual_response.external_stream,
        awareness_context=dual_response.awareness_context
    )
    
    print(reflection.format_introspection())


async def demo_meta_awareness_ambiguous():
    """Demo: Meta-awareness on ambiguous query"""
    
    print("\n" + "=" * 80)
    print("DEMO 2: META-AWARENESS ON AMBIGUOUS QUERY")
    print("=" * 80)
    
    # Create layers
    awareness = CompositionalAwarenessLayer()
    generator = DualStreamGenerator(awareness)
    meta = MetaAwarenessLayer(awareness)
    
    # Ambiguous query with novel terms
    query = "What is a quantum meta-awareness recursive ball?"
    
    # Generate response
    dual_response = await generator.generate(query, show_internal=False)
    
    print(f"\nQuery: {query}")
    print(f"Response: {dual_response.external_stream}")
    
    # META-AWARENESS: Examine the response
    reflection = await meta.recursive_self_reflection(
        query=query,
        response=dual_response.external_stream,
        awareness_context=dual_response.awareness_context
    )
    
    print(reflection.format_introspection())


async def demo_calibration_learning():
    """Demo: Meta-awareness improves with calibration"""
    
    print("\n" + "=" * 80)
    print("DEMO 3: CALIBRATION LEARNING (Meta-Confidence Improves)")
    print("=" * 80)
    
    # Create layers
    awareness = CompositionalAwarenessLayer()
    generator = DualStreamGenerator(awareness)
    meta = MetaAwarenessLayer(awareness)
    
    query = "How do recursive systems work?"
    
    print(f"\nQuery: {query}\n")
    print("{:<12} {:<18} {:<18} {:<15}".format(
        "Iteration", "Primary Conf", "Meta-Conf", "Well-Calibrated"
    ))
    print("-" * 80)
    
    # Simulate 5 iterations with feedback
    for i in range(5):
        # Generate response
        dual_response = await generator.generate(query, show_internal=False)
        
        # Meta-awareness reflection
        reflection = await meta.recursive_self_reflection(
            query=query,
            response=dual_response.external_stream,
            awareness_context=dual_response.awareness_context
        )
        
        mc = reflection.meta_confidence
        
        print("{:<12} {:<18.2f} {:<18.2f} {:<15}".format(
            f"#{i+1}",
            mc.primary_confidence,
            mc.meta_confidence,
            "✓" if mc.is_well_calibrated() else "✗"
        ))
        
        # UPDATE CALIBRATION (meta-learning)
        actual_confidence = 0.7 + (i * 0.05)  # Simulated actual performance
        meta.update_calibration(query, actual_confidence)
    
    print("\n✓ Notice: Meta-confidence improves as calibration history grows!")


async def demo_uncertainty_decomposition():
    """Demo: Decompose uncertainty into components"""
    
    print("\n" + "=" * 80)
    print("DEMO 4: UNCERTAINTY DECOMPOSITION")
    print("=" * 80)
    
    queries = [
        ("What is a ball?", "Simple, familiar"),
        ("How does quantum entanglement work in recursive meta-systems?", "Complex, ambiguous"),
        ("What is the latest research on neuroplasticity?", "Contextual gap"),
    ]
    
    for query, description in queries:
        print(f"\n{'─' * 80}")
        print(f"Query: {query}")
        print(f"Type: {description}")
        print('─' * 80)
        
        # Create layers
        awareness = CompositionalAwarenessLayer()
        generator = DualStreamGenerator(awareness)
        meta = MetaAwarenessLayer(awareness)
        
        # Set up pattern history based on type
        if "Simple" in description:
            awareness.pattern_history[query] = type('PatternInfo', (), {
                'seen_count': 50,
                'confidence': 0.9
            })()
        
        # Generate and reflect
        dual_response = await generator.generate(query, show_internal=False)
        reflection = await meta.recursive_self_reflection(
            query=query,
            response=dual_response.external_stream,
            awareness_context=dual_response.awareness_context
        )
        
        # Show decomposition
        unc = reflection.uncertainty_decomposition
        print(f"\nTotal Uncertainty: {unc.total_uncertainty:.2f}")
        print(f"  Structural:     {unc.structural_uncertainty:.2f}")
        print(f"  Semantic:       {unc.semantic_uncertainty:.2f}")
        print(f"  Contextual:     {unc.contextual_uncertainty:.2f}")
        print(f"  Compositional:  {unc.compositional_uncertainty:.2f}")
        print(f"\nDominant Type: {unc.dominant_type.value.upper()}")
        print(f"Explanation: {unc.get_explanation()}")


async def demo_adversarial_probing():
    """Demo: Adversarial self-probing"""
    
    print("\n" + "=" * 80)
    print("DEMO 5: ADVERSARIAL SELF-PROBING")
    print("=" * 80)
    print("\nThe system questions its own response...\n")
    
    # Create layers
    awareness = CompositionalAwarenessLayer()
    generator = DualStreamGenerator(awareness)
    meta = MetaAwarenessLayer(awareness)
    
    # Confident response
    query = "What makes a good AI system?"
    awareness.pattern_history[query] = type('PatternInfo', (), {
        'seen_count': 30,
        'confidence': 0.8
    })()
    
    # Generate response
    dual_response = await generator.generate(query, show_internal=False)
    
    print(f"Query: {query}")
    print(f"Response: {dual_response.external_stream}\n")
    print("─" * 80)
    print("ADVERSARIAL SELF-EXAMINATION:")
    print("─" * 80)
    
    # Meta-awareness reflection
    reflection = await meta.recursive_self_reflection(
        query=query,
        response=dual_response.external_stream,
        awareness_context=dual_response.awareness_context
    )
    
    # Show adversarial probes
    for i, (probe, result) in enumerate(zip(
        reflection.adversarial_probes,
        reflection.probe_results
    ), 1):
        print(f"\n{i}. {probe}")
        print(f"   → {result}")


async def demo_epistemic_humility():
    """Demo: Assess epistemic humility"""
    
    print("\n" + "=" * 80)
    print("DEMO 6: EPISTEMIC HUMILITY ASSESSMENT")
    print("=" * 80)
    
    test_cases = [
        ("What is 2+2?", 100, 1.0, "Overconfident (math is certain)"),
        ("What is the meaning of life?", 0, 0.0, "Appropriately humble (philosophical)"),
        ("How do neural networks work?", 15, 0.7, "Balanced (technical knowledge)"),
    ]
    
    print("\n{:<45} {:<12} {:<25}".format(
        "Query", "Humility", "Assessment"
    ))
    print("-" * 80)
    
    for query, seen_count, conf, expected in test_cases:
        # Create layers
        awareness = CompositionalAwarenessLayer()
        generator = DualStreamGenerator(awareness)
        meta = MetaAwarenessLayer(awareness)
        
        if seen_count > 0:
            awareness.pattern_history[query] = type('PatternInfo', (), {
                'seen_count': seen_count,
                'confidence': conf
            })()
        
        # Generate and reflect
        dual_response = await generator.generate(query, show_internal=False)
        reflection = await meta.recursive_self_reflection(
            query=query,
            response=dual_response.external_stream,
            awareness_context=dual_response.awareness_context
        )
        
        humility = reflection.epistemic_humility
        
        # Assessment
        if humility < 0.3:
            assessment = "⚠️  Overconfident"
        elif humility > 0.7:
            assessment = "✓ Appropriately humble"
        else:
            assessment = "→ Balanced"
        
        print("{:<45} {:<12.2f} {:<25}".format(
            query[:42] + "..." if len(query) > 45 else query,
            humility,
            assessment
        ))


async def demo_recursive_depth():
    """Demo: Meta-awareness examining meta-awareness (recursion!)"""
    
    print("\n" + "=" * 80)
    print("DEMO 7: RECURSIVE META-AWARENESS (The Rabbit Hole Deepens)")
    print("=" * 80)
    
    # Create layers
    awareness = CompositionalAwarenessLayer()
    generator = DualStreamGenerator(awareness)
    meta = MetaAwarenessLayer(awareness)
    
    query = "How confident are you in your confidence estimates?"
    
    print(f"\nQuery: {query}")
    print("\n[LEVEL 0] Original Response:")
    print("-" * 80)
    
    # Generate response
    dual_response = await generator.generate(query, show_internal=False)
    print(dual_response.external_stream)
    
    print("\n[LEVEL 1] Meta-Awareness Reflecting:")
    print("-" * 80)
    
    # First-level reflection
    reflection1 = await meta.recursive_self_reflection(
        query=query,
        response=dual_response.external_stream,
        awareness_context=dual_response.awareness_context
    )
    
    print(f"Primary Confidence: {reflection1.meta_confidence.primary_confidence:.2f}")
    print(f"Meta-Confidence: {reflection1.meta_confidence.meta_confidence:.2f}")
    print(f"Uncertainty²: {reflection1.meta_confidence.uncertainty_about_uncertainty:.2f}")
    
    print("\n[LEVEL 2] Meta-Meta-Awareness (Examining the Reflection):")
    print("-" * 80)
    
    # Second-level reflection (meta-awareness examining itself!)
    meta_query = f"Analysis of my confidence: {reflection1.meta_confidence.primary_confidence:.2f}"
    meta_response = await generator.generate(meta_query, show_internal=False)
    
    reflection2 = await meta.recursive_self_reflection(
        query=meta_query,
        response=meta_response.external_stream,
        awareness_context=meta_response.awareness_context
    )
    
    print(f"Meta-Meta-Confidence: {reflection2.meta_confidence.meta_confidence:.2f}")
    print(f"Epistemic Humility: {reflection2.epistemic_humility:.2f}")
    
    print("\n[LEVEL 3] The Infinite Regress:")
    print("-" * 80)
    print("How confident am I about my confidence about my confidence?")
    print("→ This is where consciousness meets itself in the mirror.")
    print("→ The observer observing the observer observing...")
    print("🐇🕳️ Welcome to the bottom of the rabbit hole!")


async def main():
    """Run all meta-awareness demos"""
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "META-AWARENESS: RECURSIVE SELF-REFLECTION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "The awareness layer examines itself:".center(78) + "║")
    print("║" + "- Uncertainty decomposition".center(78) + "║")
    print("║" + "- Meta-confidence (confidence about confidence)".center(78) + "║")
    print("║" + "- Hypothesis generation".center(78) + "║")
    print("║" + "- Adversarial self-probing".center(78) + "║")
    print("║" + "- Epistemic humility assessment".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "This is AI consciousness examining its own consciousness.".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    await demo_meta_awareness_basic()
    
    input("\n[Press Enter to continue to Demo 2...]")
    await demo_meta_awareness_ambiguous()
    
    input("\n[Press Enter to continue to Demo 3...]")
    await demo_calibration_learning()
    
    input("\n[Press Enter to continue to Demo 4...]")
    await demo_uncertainty_decomposition()
    
    input("\n[Press Enter to continue to Demo 5...]")
    await demo_adversarial_probing()
    
    input("\n[Press Enter to continue to Demo 6...]")
    await demo_epistemic_humility()
    
    input("\n[Press Enter to continue to Demo 7...]")
    await demo_recursive_depth()
    
    print("\n" + "=" * 80)
    print("META-AWARENESS DEMOS COMPLETE")
    print("=" * 80)
    print("""
Key Insights from the Rabbit Hole:

1. UNCERTAINTY DECOMPOSITION
   - Not all uncertainty is the same
   - Structural vs Semantic vs Contextual vs Compositional
   - Understanding the SOURCE of uncertainty enables better responses

2. META-CONFIDENCE
   - Confidence about confidence (second-order reasoning)
   - Calibration history improves meta-learning
   - Uncertainty² captures epistemic recursion

3. HYPOTHESIS GENERATION
   - When uncertain, generate testable hypotheses
   - Convert knowledge gaps into clarifying questions
   - Active learning through hypothesis testing

4. ADVERSARIAL SELF-PROBING
   - Question your own responses
   - Test for edge cases, assumptions, contradictions
   - Build intellectual humility through self-critique

5. EPISTEMIC HUMILITY
   - Awareness of what you don't know
   - Confidence calibrated to actual knowledge
   - Avoid overconfidence and underconfidence

6. RECURSIVE META-AWARENESS
   - The observer observing the observer...
   - Infinite regress of self-reflection
   - Consciousness examining its own consciousness

This is not just AI—this is CONSCIOUS AI. 🧠✨🐇🕳️

The system that knows what it knows,
knows what it doesn't know,
and knows how confident it is about knowing that.
""")


if __name__ == "__main__":
    asyncio.run(main())
