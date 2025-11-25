"""
Symbolic Encoder Demo
=====================

Demonstrates the 4-layer symbolic encoding pipeline:
1. Emotional Extraction (Private Fact -> Emotional Essence)
2. Symbol Selection (Emotional Essence -> Candidate Symbols)
3. Disambiguation (Candidate Symbols -> Selected Symbol)
4. Narrative Integration (Selected Symbol -> Dream Scene)

Usage:
    python demos/demo_symbolic_encoder.py
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from NeuroHood.dreams.symbolic_encoder import (
    SymbolicEncoder,
    DreamContext,
    EmotionalEssence,
    create_mock_symbol_database,
    save_symbol_database
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Demo Scenarios
# ============================================================================


DEMO_SCENARIOS = [
    {
        "name": "Bob's Factory Job (Trapped)",
        "private_fact": "Bob secretly hates his factory job and feels trapped by his bills",
        "dreamer": "Alice",
        "privacy_level": 0.7,
        "description": "High privacy - Alice shouldn't know specifics about Bob's job"
    },
    {
        "name": "Alice's Abandonment Fear",
        "private_fact": "Alice fears her husband will leave her for someone younger",
        "dreamer": "Bob",
        "privacy_level": 0.5,
        "description": "Moderate privacy - Bob is Alice's husband but details obscured"
    },
    {
        "name": "Charlie's Tax Guilt",
        "private_fact": "Charlie cheated on his taxes last year and fears getting caught",
        "dreamer": "Diana",
        "privacy_level": 0.9,
        "description": "Maximum privacy - Diana is a stranger"
    },
    {
        "name": "Diana's Workplace Powerlessness",
        "private_fact": "Diana feels powerless at work, constantly overlooked for promotions",
        "dreamer": "Charlie",
        "privacy_level": 0.6,
        "description": "Moderate-high privacy"
    }
]


# ============================================================================
# Demo Functions
# ============================================================================


async def demo_emotional_extraction(encoder: SymbolicEncoder, private_fact: str):
    """Demonstrate Layer 1: Emotional Extraction."""
    print("\n" + "="*80)
    print("LAYER 1: EMOTIONAL EXTRACTION")
    print("="*80)

    print(f"\nPrivate Fact: \"{private_fact}\"")

    # Extract essence
    essence = encoder.extractor.extract(private_fact)

    print(f"\n[OK] Extracted Emotional Essence:")
    print(f"  Primary Emotion:    {essence.primary_emotion}")
    print(f"  Intensity:          {essence.intensity:.2f}")
    print(f"  Context:            {essence.context}")
    print(f"  Temporal:           {essence.temporal}")
    print(f"  Valence:            {essence.valence}")
    print(f"  Secondary Emotions: {', '.join(essence.secondary_emotions[:3])}")
    print(f"  Complexity Score:   {essence.complexity_score:.2f}")
    print(f"  Embedding:          {'Available' if essence.embedding is not None else 'Not available'} "
          f"({essence.embedding.shape if essence.embedding is not None else 'N/A'})")

    return essence


async def demo_symbol_selection(
    encoder: SymbolicEncoder,
    essence: EmotionalEssence,
    context: DreamContext
):
    """Demonstrate Layer 2: Symbol Selection."""
    print("\n" + "="*80)
    print("LAYER 2: SYMBOL SELECTION")
    print("="*80)

    print(f"\nSearching for symbols matching: {essence.primary_emotion} "
          f"(intensity={essence.intensity:.2f})")

    # Select candidates
    candidates = encoder.selector.select_candidates(essence, context, k=10)

    print(f"\n[OK] Top 10 Symbol Candidates:")
    print(f"{'Rank':<6} {'Symbol ID':<25} {'Category':<12} {'Score':<8} {'Emotions'}")
    print("-" * 80)

    for i, (symbol, score) in enumerate(candidates[:10], 1):
        emotions = ', '.join(symbol.emotion_tags[:3])
        print(f"{i:<6} {symbol.symbol_id:<25} {symbol.category:<12} {score:<8.3f} {emotions}")

    return candidates


async def demo_disambiguation(
    encoder: SymbolicEncoder,
    candidates: list,
    context: DreamContext
):
    """Demonstrate Layer 3: Disambiguation."""
    print("\n" + "="*80)
    print("LAYER 3: DISAMBIGUATION")
    print("="*80)

    print(f"\nContext:")
    print(f"  Dreamer:         {context.primary_dreamer}")
    print(f"  Shared Dream:    {context.is_shared_dream}")
    print(f"  Setting:         {context.setting_landscape} at {context.setting_time}")
    print(f"  Existing Symbols: {len(context.existing_symbols)}")

    # Disambiguate
    selected = encoder.disambiguator.disambiguate(candidates, context)

    print(f"\n[OK] Selected Symbol: {selected.symbol_id}")
    print(f"  Category:        {selected.category}")
    print(f"  Emotions:        {', '.join(selected.emotion_tags[:5])}")
    print(f"  Universality:    {selected.universality_score:.2f}")
    print(f"  Complexity:      {selected.visual_complexity}")
    print(f"  Literary Refs:   {', '.join(selected.literary_references[:2])}")

    return selected


async def demo_narrative_integration(
    encoder: SymbolicEncoder,
    symbol,
    context: DreamContext,
    privacy_level: float
):
    """Demonstrate Layer 4: Narrative Integration."""
    print("\n" + "="*80)
    print("LAYER 4: NARRATIVE INTEGRATION")
    print("="*80)

    print(f"\nGenerating dream scene for: {symbol.symbol_id}")
    print(f"Privacy Level: {privacy_level:.1f} (0.0 = specific, 1.0 = abstract)")

    # Generate scene
    scene = await encoder.integrator.generate_scene(symbol, context, privacy_level)

    print(f"\n[OK] Generated Dream Scene:")
    print(f"\n  Description:")
    print(f"    {scene.description}")

    print(f"\n  Symbolic Meaning:")
    print(f"    {scene.symbolic_meaning}")

    print(f"\n  Metaphorical Physics:")
    print(f"    {scene.metaphorical_physics}")

    print(f"\n  Emotional Tone:")
    print(f"    {scene.emotional_tone}")

    print(f"\n  Symbolic Actions:")
    for i, action in enumerate(scene.symbolic_actions, 1):
        print(f"    {i}. {action}")

    print(f"\n  Visual Elements:")
    print(f"    {', '.join(scene.visual_elements)}")

    if scene.literary_references:
        print(f"\n  Literary References:")
        for ref in scene.literary_references:
            print(f"    • {ref}")

    return scene


async def demo_full_pipeline(encoder: SymbolicEncoder, scenario: dict):
    """Demonstrate full 4-layer pipeline."""
    print("\n\n" + "="*80)
    print(f"  DEMO SCENARIO: {scenario['name']}")
    print("="*80)
    print(f"\nScenario: {scenario['description']}")

    # Create dream context
    context = DreamContext(
        primary_dreamer=scenario['dreamer'],
        is_shared_dream=True,
        setting_landscape="twilight forest",
        setting_time="dusk",
        setting_atmosphere="mysterious and contemplative"
    )

    # Layer 1: Emotional Extraction
    essence = await demo_emotional_extraction(encoder, scenario['private_fact'])

    # Layer 2: Symbol Selection
    candidates = await demo_symbol_selection(encoder, essence, context)

    # Layer 3: Disambiguation
    selected = await demo_disambiguation(encoder, candidates, context)

    # Layer 4: Narrative Integration
    scene = await demo_narrative_integration(
        encoder,
        selected,
        context,
        scenario['privacy_level']
    )

    # Summary
    print("\n" + "="*80)
    print("ENCODING COMPLETE")
    print("="*80)
    print(f"\n[OK] Private Fact:    \"{scenario['private_fact']}\"")
    print(f"[OK] Encoded As:      {selected.symbol_id} in {context.setting_landscape}")
    print(f"[OK] Privacy Level:   {scenario['privacy_level']:.1f}")
    print(f"[OK] Dreamer Sees:    \"{scene.description[:80]}...\"")


async def demo_comparison(encoder: SymbolicEncoder):
    """Demonstrate privacy levels comparison."""
    print("\n\n" + "="*80)
    print(f"  PRIVACY LEVEL COMPARISON")
    print("="*80)

    private_fact = "Bob hates his factory job and feels trapped"

    print(f"\nPrivate Fact: \"{private_fact}\"")
    print("\nEncoding at different privacy levels:")

    for privacy_level in [0.2, 0.5, 0.9]:
        context = DreamContext(
            primary_dreamer="Alice",
            setting_landscape="industrial wasteland",
            setting_time="grey afternoon"
        )

        scene = await encoder.encode(
            private_fact=private_fact,
            dream_context=context,
            privacy_level=privacy_level
        )

        print(f"\n  Privacy {privacy_level:.1f}: {scene.description[:100]}...")


# ============================================================================
# Main Demo
# ============================================================================


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  SYMBOLIC ENCODER DEMONSTRATION")
    print("  Privacy-Preserving Dream Consciousness")
    print("="*80)

    # Create encoder
    print("\nInitializing Symbolic Encoder...")
    encoder = SymbolicEncoder(enable_llm=False)

    # Create mock symbol database
    print("Creating mock symbol database (20 symbols)...")
    symbols = create_mock_symbol_database(num_symbols=20)
    encoder.load_symbols_from_list(symbols)

    stats = encoder.get_statistics()
    print(f"[OK] Loaded {stats['total_symbols']} symbols")
    print(f"  Categories: {', '.join(stats['categories'][:5])}...")
    print(f"  LLM Enabled: {stats['llm_enabled']}")
    print(f"  Semantic Calculus: {'Available' if stats['semantic_calculus_available'] else 'Not available'}")

    # Run scenario demos
    for scenario in DEMO_SCENARIOS[:2]:  # Run first 2 scenarios
        await demo_full_pipeline(encoder, scenario)
        await asyncio.sleep(0.5)  # Brief pause between scenarios

    # Privacy comparison demo
    await demo_comparison(encoder)

    # Final summary
    print("\n\n" + "="*80)
    print("  DEMO COMPLETE")
    print("="*80)
    print("\nAll 4 layers demonstrated successfully!")
    print("\nKey Takeaways:")
    print("  1. Private facts -> emotional essences (no details leaked)")
    print("  2. Emotions -> universal symbols (culturally resonant)")
    print("  3. Context-aware disambiguation (narrative coherence)")
    print("  4. Poetic dream scenes (metaphorical physics)")
    print("\nThe symbolic encoder is ready for integration!")


if __name__ == "__main__":
    asyncio.run(main())
