"""
Demo: Dual-Stream Awareness-Guided Generation

Shows how compositional awareness (Phase 5) informs both internal reasoning
and external response streams.

Run: python -m demos.demo_dual_stream_awareness
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.awareness.compositional_awareness import CompositionalAwarenessLayer
from HoloLoom.awareness.dual_stream import DualStreamGenerator


async def demo_high_confidence():
    """Demo: High confidence query (familiar pattern)"""
    
    print("\n" + "=" * 80)
    print("DEMO 1: HIGH CONFIDENCE (Familiar Query)")
    print("=" * 80)
    
    # Create awareness layer (without full Phase 5 components for simplicity)
    awareness = CompositionalAwarenessLayer()
    
    # Simulate that we've seen this query many times
    awareness.pattern_history["What is a red ball?"] = type('PatternInfo', (), {
        'seen_count': 47,
        'confidence': 0.89,
        'typical_contexts': ['sports', 'toys']
    })()
    
    # Create generator
    generator = DualStreamGenerator(awareness)
    
    # Generate response
    query = "What is a red ball?"
    response = await generator.generate(query, show_internal=True)
    
    print(response.format_for_display(show_internal=True))
    
    # Show awareness context
    print("\n[ AWARENESS CONTEXT DETAILS ]")
    print("-" * 80)
    print(f"Cache Status: {response.awareness_context.confidence.query_cache_status}")
    print(f"Uncertainty: {response.awareness_context.confidence.uncertainty_level:.2f}")
    print(f"Confidence Tone: {response.awareness_context.external_guidance.confidence_tone}")
    print(f"Hedging: {response.awareness_context.external_guidance.appropriate_hedging or 'None'}")
    print(f"Should Clarify: {response.awareness_context.confidence.should_ask_clarification}")


async def demo_low_confidence():
    """Demo: Low confidence query (novel pattern)"""
    
    print("\n" + "=" * 80)
    print("DEMO 2: LOW CONFIDENCE (Novel Query)")
    print("=" * 80)
    
    # Create awareness layer
    awareness = CompositionalAwarenessLayer()
    
    # This query has never been seen
    # (no pattern history entry)
    
    # Create generator
    generator = DualStreamGenerator(awareness)
    
    # Generate response
    query = "What is a quantum ball?"
    response = await generator.generate(query, show_internal=True)
    
    print(response.format_for_display(show_internal=True))
    
    # Show awareness context
    print("\n[ AWARENESS CONTEXT DETAILS ]")
    print("-" * 80)
    print(f"Cache Status: {response.awareness_context.confidence.query_cache_status}")
    print(f"Uncertainty: {response.awareness_context.confidence.uncertainty_level:.2f}")
    print(f"Knowledge Gap: {response.awareness_context.confidence.knowledge_gap_detected}")
    print(f"Confidence Tone: {response.awareness_context.external_guidance.confidence_tone}")
    print(f"Should Clarify: {response.awareness_context.confidence.should_ask_clarification}")
    print(f"Clarification: {response.awareness_context.confidence.suggested_clarification}")


async def demo_medium_confidence():
    """Demo: Medium confidence query (partial familiarity)"""
    
    print("\n" + "=" * 80)
    print("DEMO 3: MEDIUM CONFIDENCE (Partially Familiar)")
    print("=" * 80)
    
    # Create awareness layer
    awareness = CompositionalAwarenessLayer()
    
    # Simulate moderate familiarity
    awareness.pattern_history["How do red balls bounce?"] = type('PatternInfo', (), {
        'seen_count': 5,
        'confidence': 0.55,
        'typical_contexts': ['physics', 'sports']
    })()
    
    # Create generator
    generator = DualStreamGenerator(awareness)
    
    # Generate response
    query = "How do red balls bounce differently than blue balls?"
    response = await generator.generate(query, show_internal=True)
    
    print(response.format_for_display(show_internal=True))
    
    # Show awareness context
    print("\n[ AWARENESS CONTEXT DETAILS ]")
    print("-" * 80)
    print(f"Cache Status: {response.awareness_context.confidence.query_cache_status}")
    print(f"Uncertainty: {response.awareness_context.confidence.uncertainty_level:.2f}")
    print(f"Confidence Tone: {response.awareness_context.external_guidance.confidence_tone}")
    print(f"Hedging: {response.awareness_context.external_guidance.appropriate_hedging}")
    print(f"Reasoning Structure: {response.awareness_context.internal_guidance.reasoning_structure}")


async def demo_comparison():
    """Demo: Compare all three confidence levels side by side"""
    
    print("\n" + "=" * 80)
    print("DEMO 4: CONFIDENCE LEVEL COMPARISON")
    print("=" * 80)
    
    queries = [
        ("What is a red ball?", 47, 0.89),           # High confidence
        ("How do balls bounce?", 5, 0.55),           # Medium confidence
        ("What is a quantum ball?", 0, 0.0),         # Low confidence
    ]
    
    print("\n{:<40} {:<15} {:<12} {:<15}".format(
        "Query", "Familiarity", "Uncertainty", "Tone"
    ))
    print("-" * 80)
    
    for query, seen_count, confidence in queries:
        # Create awareness layer
        awareness = CompositionalAwarenessLayer()
        
        if seen_count > 0:
            awareness.pattern_history[query] = type('PatternInfo', (), {
                'seen_count': seen_count,
                'confidence': confidence,
                'typical_contexts': []
            })()
        
        # Create generator
        generator = DualStreamGenerator(awareness)
        
        # Generate response (without showing internal for comparison)
        response = await generator.generate(query, show_internal=False)
        
        ctx = response.awareness_context
        
        print("{:<40} {:>3}× seen      {:<12.2f} {:<15}".format(
            query[:37] + "..." if len(query) > 40 else query,
            seen_count,
            ctx.confidence.uncertainty_level,
            ctx.external_guidance.confidence_tone
        ))


async def demo_learning_effect():
    """Demo: Show how confidence increases with repeated queries"""
    
    print("\n" + "=" * 80)
    print("DEMO 5: LEARNING EFFECT (Confidence Improves Over Time)")
    print("=" * 80)
    
    # Create awareness layer
    awareness = CompositionalAwarenessLayer()
    generator = DualStreamGenerator(awareness)
    
    query = "What makes a ball bounce?"
    
    print(f"\nQuery: {query}\n")
    print("{:<10} {:<12} {:<15} {:<20}".format(
        "Iteration", "Uncertainty", "Tone", "Status"
    ))
    print("-" * 80)
    
    # Simulate 5 iterations of the same query
    for i in range(5):
        response = await generator.generate(query, show_internal=False)
        ctx = response.awareness_context
        
        # Extract info
        uncertainty = ctx.confidence.uncertainty_level
        tone = ctx.external_guidance.confidence_tone
        seen = awareness.pattern_history.get(query)
        count = seen.seen_count if seen else 0
        
        print("{:<10} {:<12.2f} {:<15} Seen {}× (conf: {:.2f})".format(
            f"#{i+1}",
            uncertainty,
            tone,
            count,
            seen.confidence if seen else 0.0
        ))
    
    print("\n✓ Notice: Uncertainty decreases and confidence tone improves with repetition!")


async def main():
    """Run all demos"""
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  DUAL-STREAM AWARENESS-GUIDED GENERATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Compositional awareness (Phase 5) feeds BOTH internal reasoning".center(78) + "║")
    print("║" + "  and external response streams for transparent AI behavior".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run demos
    await demo_high_confidence()
    
    input("\n[Press Enter to continue to Demo 2...]")
    await demo_low_confidence()
    
    input("\n[Press Enter to continue to Demo 3...]")
    await demo_medium_confidence()
    
    input("\n[Press Enter to continue to Demo 4...]")
    await demo_comparison()
    
    input("\n[Press Enter to continue to Demo 5...]")
    await demo_learning_effect()
    
    print("\n" + "=" * 80)
    print("DEMOS COMPLETE")
    print("=" * 80)
    print("""
Key Takeaways:

1. UNIFIED AWARENESS: One awareness context feeds both streams
   - Internal reasoning sees confidence signals
   - External response matches tone to confidence

2. CONFIDENCE-BASED BEHAVIOR: Response adapts to uncertainty
   - High confidence → direct, no hedging
   - Low confidence → ask clarification
   - Medium confidence → hedged but informative

3. TRANSPARENT REASONING: Internal stream shows why
   - Structural analysis (X-bar)
   - Pattern recognition (cache)
   - Strategy selection (confidence-based)

4. LEARNING EFFECT: Confidence improves with experience
   - Pattern history tracks familiarity
   - Repeated queries become more confident
   - Feedback loop refines awareness

This is compositional AI consciousness! 🧠✨
""")


if __name__ == "__main__":
    asyncio.run(main())
