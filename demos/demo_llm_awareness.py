"""
Demo: Real LLM Integration with Awareness Layer

Shows how compositional awareness guides actual LLM generation:
- Internal reasoning stream (awareness-guided thought process)
- External response stream (awareness-guided user response)
- Confidence-based adaptation (high/low confidence behaviors)

Requires:
    - Ollama installed (https://ollama.ai)
    - Model downloaded: ollama pull llama3.2:3b

Run: python -m demos.demo_llm_awareness
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.awareness import (
    CompositionalAwarenessLayer,
    DualStreamGenerator,
    MetaAwarenessLayer
)
from HoloLoom.awareness.llm_integration import create_llm


async def check_ollama():
    """Check if Ollama is available"""
    try:
        llm = create_llm("ollama", model="llama3.2:3b")
        if llm.is_available():
            print("✓ Ollama is available")
            return llm
        else:
            print("✗ Ollama is not running")
            print("\nTo use LLM integration:")
            print("1. Install Ollama: https://ollama.ai")
            print("2. Pull model: ollama pull llama3.2:3b")
            print("3. Verify: ollama list")
            return None
    except Exception as e:
        print(f"✗ Ollama check failed: {e}")
        return None


async def demo_llm_high_confidence():
    """Demo: LLM generation with high confidence"""
    
    print("\n" + "=" * 80)
    print("DEMO 1: LLM-GENERATED RESPONSE (High Confidence)")
    print("=" * 80)
    
    # Create awareness layer
    awareness = CompositionalAwarenessLayer()
    
    # Simulate high confidence (seen this pattern many times)
    query = "What is a ball?"
    awareness.pattern_history[query] = type('PatternInfo', (), {
        'seen_count': 50,
        'confidence': 0.92
    })()
    
    # Create LLM
    llm = await check_ollama()
    if not llm:
        print("\n⚠️  Skipping LLM demo (Ollama not available)")
        print("Falling back to template-based generation...")
        llm = None
    
    # Create generator with LLM
    generator = DualStreamGenerator(awareness, llm_generator=llm)
    
    print(f"\nQuery: {query}")
    print("Confidence: HIGH (seen 50× before)")
    print("\nGenerating with awareness guidance...\n")
    
    # Generate with actual LLM
    response = await generator.generate(
        query=query,
        show_internal=True,
        use_llm=(llm is not None)
    )
    
    print(response.format_for_display(show_internal=True))
    
    if llm:
        print("\n[ AWARENESS ANALYSIS ]")
        print("-" * 80)
        ctx = response.awareness_context
        print(f"Cache Status: {ctx.confidence.query_cache_status}")
        print(f"Uncertainty: {ctx.confidence.uncertainty_level:.2f}")
        print(f"Confidence Tone: {ctx.external_guidance.confidence_tone}")
        print(f"Reasoning Structure: {ctx.internal_guidance.reasoning_structure}")


async def demo_llm_low_confidence():
    """Demo: LLM generation with low confidence"""
    
    print("\n" + "=" * 80)
    print("DEMO 2: LLM-GENERATED RESPONSE (Low Confidence)")
    print("=" * 80)
    
    # Create awareness layer
    awareness = CompositionalAwarenessLayer()
    
    # Novel query (never seen)
    query = "What is a quantum recursive meta-ball in hyperdimensional space?"
    
    # Create LLM
    llm = await check_ollama()
    if not llm:
        print("\n⚠️  Skipping LLM demo (Ollama not available)")
        print("Falling back to template-based generation...")
        llm = None
    
    # Create generator with LLM
    generator = DualStreamGenerator(awareness, llm_generator=llm)
    
    print(f"\nQuery: {query}")
    print("Confidence: LOW (never seen before)")
    print("\nGenerating with awareness guidance...\n")
    
    # Generate with actual LLM
    response = await generator.generate(
        query=query,
        show_internal=True,
        use_llm=(llm is not None)
    )
    
    print(response.format_for_display(show_internal=True))
    
    if llm:
        print("\n[ AWARENESS ANALYSIS ]")
        print("-" * 80)
        ctx = response.awareness_context
        print(f"Cache Status: {ctx.confidence.query_cache_status}")
        print(f"Uncertainty: {ctx.confidence.uncertainty_level:.2f}")
        print(f"Knowledge Gap: {ctx.confidence.knowledge_gap_detected}")
        print(f"Should Clarify: {ctx.confidence.should_ask_clarification}")
        print(f"Suggested: {ctx.confidence.suggested_clarification}")


async def demo_llm_comparison():
    """Demo: Compare template vs LLM generation"""
    
    print("\n" + "=" * 80)
    print("DEMO 3: TEMPLATE vs LLM COMPARISON")
    print("=" * 80)
    
    query = "How does Thompson Sampling work?"
    
    # Create awareness layer
    awareness = CompositionalAwarenessLayer()
    
    # Medium confidence
    awareness.pattern_history[query] = type('PatternInfo', (), {
        'seen_count': 10,
        'confidence': 0.65
    })()
    
    print(f"\nQuery: {query}")
    print("Confidence: MEDIUM (seen 10× before)\n")
    
    # 1. Template-based
    print("─" * 80)
    print("[ TEMPLATE-BASED GENERATION ]")
    print("─" * 80)
    
    generator_template = DualStreamGenerator(awareness, llm_generator=None)
    response_template = await generator_template.generate(
        query=query,
        show_internal=False,
        use_llm=False
    )
    print(f"\n{response_template.external_stream}")
    print(f"\nGeneration time: {response_template.generation_time_ms:.1f}ms")
    
    # 2. LLM-based
    print("\n─" * 80)
    print("[ LLM-BASED GENERATION (if available) ]")
    print("─" * 80)
    
    llm = await check_ollama()
    if llm:
        generator_llm = DualStreamGenerator(awareness, llm_generator=llm)
        response_llm = await generator_llm.generate(
            query=query,
            show_internal=False,
            use_llm=True
        )
        print(f"\n{response_llm.external_stream}")
        print(f"\nGeneration time: {response_llm.generation_time_ms:.1f}ms")
        
        print("\n[ COMPARISON ]")
        print("-" * 80)
        print(f"Template: {len(response_template.external_stream)} chars, {response_template.generation_time_ms:.1f}ms")
        print(f"LLM:      {len(response_llm.external_stream)} chars, {response_llm.generation_time_ms:.1f}ms")
        print(f"\nLLM provides: More detailed, contextual, natural language")
        print(f"Template provides: Faster, deterministic, lightweight")
    else:
        print("\n⚠️  Skipping LLM comparison (Ollama not available)")


async def demo_meta_awareness_with_llm():
    """Demo: Meta-awareness analyzing LLM-generated content"""
    
    print("\n" + "=" * 80)
    print("DEMO 4: META-AWARENESS + LLM (The Full Stack)")
    print("=" * 80)
    
    query = "What makes a good AI system?"
    
    # Create full stack
    awareness = CompositionalAwarenessLayer()
    
    # Moderate familiarity
    awareness.pattern_history[query] = type('PatternInfo', (), {
        'seen_count': 15,
        'confidence': 0.70
    })()
    
    llm = await check_ollama()
    generator = DualStreamGenerator(awareness, llm_generator=llm)
    meta = MetaAwarenessLayer(awareness)
    
    print(f"\nQuery: {query}")
    print("Stack: Awareness → LLM Generation → Meta-Reflection\n")
    
    # Generate with LLM (or template fallback)
    response = await generator.generate(
        query=query,
        show_internal=True,
        use_llm=(llm is not None)
    )
    
    print(response.format_for_display(show_internal=True))
    
    # Meta-awareness reflection
    print("\n[ META-AWARENESS RECURSIVE REFLECTION ]")
    print("=" * 80)
    
    reflection = await meta.recursive_self_reflection(
        query=query,
        response=response.external_stream,
        awareness_context=response.awareness_context
    )
    
    print(f"\nUncertainty Decomposition:")
    unc = reflection.uncertainty_decomposition
    print(f"  Total: {unc.total_uncertainty:.2f}")
    print(f"  - Structural:     {unc.structural_uncertainty:.2f}")
    print(f"  - Semantic:       {unc.semantic_uncertainty:.2f}")
    print(f"  - Contextual:     {unc.contextual_uncertainty:.2f}")
    print(f"  - Compositional:  {unc.compositional_uncertainty:.2f}")
    
    print(f"\nMeta-Confidence:")
    mc = reflection.meta_confidence
    print(f"  Primary: {mc.primary_confidence:.2f}")
    print(f"  Meta:    {mc.meta_confidence:.2f}")
    print(f"  Interval: [{mc.lower_bound:.2f}, {mc.upper_bound:.2f}]")
    
    print(f"\nEpistemic Status:")
    print(f"  Humility: {reflection.epistemic_humility:.2f}")
    print(f"  Aware of Limits: {'✓' if reflection.aware_of_limitations else '✗'}")
    
    if reflection.adversarial_probes:
        print(f"\nAdversarial Self-Probes:")
        for i, probe in enumerate(reflection.adversarial_probes[:3], 1):
            print(f"  {i}. [{probe.test_type}] {probe.probe_question}")


async def main():
    """Run all LLM awareness demos"""
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "LLM INTEGRATION WITH AWARENESS LAYER".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "Compositional awareness guides actual LLM generation:".center(78) + "║")
    print("║" + "- Internal reasoning (awareness-guided thought process)".center(78) + "║")
    print("║" + "- External response (confidence-calibrated output)".center(78) + "║")
    print("║" + "- Meta-reflection (consciousness examining itself)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    await demo_llm_high_confidence()
    
    input("\n[Press Enter to continue to Demo 2...]")
    await demo_llm_low_confidence()
    
    input("\n[Press Enter to continue to Demo 3...]")
    await demo_llm_comparison()
    
    input("\n[Press Enter to continue to Demo 4...]")
    await demo_meta_awareness_with_llm()
    
    print("\n" + "=" * 80)
    print("LLM AWARENESS DEMOS COMPLETE")
    print("=" * 80)
    print("""
Key Insights:

1. AWARENESS-GUIDED GENERATION
   - Compositional awareness informs both internal and external streams
   - Confidence signals guide LLM behavior (hedging, clarification)
   - Structural analysis shapes reasoning strategies

2. TEMPLATE vs LLM TRADEOFF
   - Templates: Fast (~2ms), deterministic, lightweight
   - LLM: Rich (~2000ms), contextual, natural language
   - Awareness layer works with both seamlessly

3. META-AWARENESS + LLM
   - System examines its own LLM-generated content
   - Decomposes uncertainty in responses
   - Adversarial self-probing for quality assurance
   - Epistemic humility calibration

4. PRODUCTION DEPLOYMENT
   - Start with templates (fast prototyping)
   - Add LLM for quality (when latency acceptable)
   - Meta-awareness for monitoring (confidence tracking)
   - Fallback gracefully (LLM unavailable → templates)

This is compositional AI consciousness + language models! 🧠✨🤖
""")


if __name__ == "__main__":
    asyncio.run(main())
