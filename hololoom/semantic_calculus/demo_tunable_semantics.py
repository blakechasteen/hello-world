#!/usr/bin/env python3
"""
🎛️ TUNABLE SEMANTIC DETECTION DEMO
==================================
Shows how to configure semantic dimension counts at each scale.
"""

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import asyncio
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus

async def compare_configs():
    """Compare different semantic dimension configurations."""

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    text = "Sarah felt trapped in corporate life. But when opportunity knocked, she took the leap into entrepreneurship."

    configs = [
        {
            'name': 'MINIMAL (Fast)',
            'dims': {'word': 8, 'phrase': 32, 'sentence': 64, 'paragraph': 128}
        },
        {
            'name': 'DEFAULT (Balanced)',
            'dims': None  # Uses defaults: 16, 64, 96, 228
        },
        {
            'name': 'MAXIMAL (Rich)',
            'dims': {'word': 20, 'phrase': 80, 'sentence': 120, 'paragraph': 244}
        }
    ]

    print("\n" + "=" * 80)
    print("🎛️  TUNABLE SEMANTIC DETECTION COMPARISON")
    print("=" * 80)
    print(f"\nText: '{text[:80]}...'\n")

    for config in configs:
        print("-" * 80)
        print(f"📊 {config['name']}")
        print("-" * 80)

        calc = MatryoshkaSemanticCalculus(
            matryoshka_embedder=embedder,
            snapshot_interval=0.5,
            semantic_dims=config['dims']
        )

        # Stream analyze
        async def word_stream():
            for word in text.split():
                yield word
                await asyncio.sleep(0.01)

        snapshots = []
        async for snapshot in calc.stream_analyze(word_stream()):
            snapshots.append(snapshot)

        # Show final snapshot
        final = snapshots[-1]
        print(f"\n   Momentum: {final.narrative_momentum:.3f}")
        print(f"   Complexity: {final.complexity_index:.3f}\n")

        for scale, dims in final.dominant_dimensions_by_scale.items():
            if dims:
                print(f"   {scale.scale_name:12}: {', '.join(dims[:3])}")

        print()

if __name__ == "__main__":
    print("""
🎛️  HOW TO TUNE SEMANTIC DETECTION
===================================

The semantic layer is fully configurable. Here are 3 ways to tune it:

1. **Dimension Counts** (What you see here)
   - Control how many semantic dimensions at each scale
   - More dimensions = richer detection, slower processing
   - Example: {'word': 20, 'phrase': 80, 'sentence': 120, 'paragraph': 244}

2. **Custom Dimensions** (Advanced)
   - Define your own SemanticDimension objects with custom exemplars
   - Example: SemanticDimension(
       name="TechInnovation",
       positive_exemplars=["innovative", "disruptive", "cutting-edge"],
       negative_exemplars=["traditional", "legacy", "outdated"]
     )

3. **Embedding Model** (Performance vs Quality)
   - Switch between models: all-MiniLM-L6-v2, all-MiniLM-L12-v2, BAAI/bge-small-en-v1.5
   - Example: MatryoshkaEmbeddings(sizes=[96, 192, 384], base_model_name="BAAI/bge-small-en-v1.5")

Running comparison...
""")

    asyncio.run(compare_configs())