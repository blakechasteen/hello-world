"""
Demo: Verification Mode Comparison
===================================
Demonstrates different reasoning modes and shows when verification adds value.

Shows:
    - Direct mode: Fast single-pass answers (~150ms)
    - Verify mode: Answer with verification checks (~600ms)
    - Research mode: Multi-query exploration (~900ms)
    - Confidence scoring differences
    - When to use each mode

Usage:
    PYTHONPATH=. python demos/demo_rag_with_verification.py

Output:
    - Comparison of response times across modes
    - Confidence scores by mode
    - Source attribution and count
    - Recommendations for mode selection
"""

import asyncio
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.rag import SimpleRAG


async def main():
    """Run verification mode demo."""
    print("=" * 70)
    print("Demo: Verification Mode Comparison")
    print("=" * 70)
    print()

    try:
        async with SimpleRAG() as rag:
            # 1. Ingest knowledge base
            print("1. Ingesting knowledge base...")
            documents = [
                "Thompson Sampling is theoretically optimal for multi-armed bandits.",
                "It maintains posterior distributions over reward probabilities.",
                "Epsilon-greedy is a simpler but less optimal alternative.",
                "Thompson Sampling naturally adapts exploration to uncertainty.",
                "Applications include A/B testing, recommendations, and trials.",
                "Thompson Sampling has logarithmic regret bounds (optimal).",
                "The main advantage is automatic uncertainty-driven exploration.",
            ]

            for doc in documents:
                await rag.ingest(doc)

            print(f"   ✓ Ingested {len(documents)} documents")

            # 2. Define test question
            question = "Is Thompson Sampling better than epsilon-greedy?"
            print(f"\n2. Test Question: {question}")

            # 3. Query with different modes
            print("\n3. Querying with different modes...")
            print("-" * 70)

            modes = ["direct", "verify", "research"]

            results_by_mode = {}

            for mode in modes:
                print(f"\n   Mode: {mode}")

                result = await rag.query(question, mode=mode)
                results_by_mode[mode] = result

                # Display result
                print(f"   Response: {result.response[:75]}...")
                print(f"   Confidence: {result.confidence:.2f}")
                print(f"   Sources: {len(result.sources)}")
                latency = result.metadata.get('latency_ms', 'unknown')
                print(f"   Latency: {latency}ms" if latency != 'unknown' else "   Latency: unknown")

            # 4. Comparison analysis
            print("\n4. Mode Comparison Analysis:")
            print("-" * 70)

            print("\n   DIRECT Mode:")
            print("   ├─ Speed: Fastest (~150ms)")
            print("   ├─ Confidence: Lower (less verification)")
            print("   ├─ Sources: Minimal")
            print("   └─ Best for: Simple factual queries")

            print("\n   VERIFY Mode (Recommended):")
            print("   ├─ Speed: Moderate (~600ms)")
            print("   ├─ Confidence: Higher (with verification)")
            print("   ├─ Sources: Good coverage")
            print("   └─ Best for: Claims needing verification")

            print("\n   RESEARCH Mode:")
            print("   ├─ Speed: Slowest (~900ms)")
            print("   ├─ Confidence: Highest (multi-query)")
            print("   ├─ Sources: Comprehensive")
            print("   └─ Best for: Open-ended exploration")

            # 5. Confidence comparison
            print("\n5. Confidence Scores by Mode:")
            print("-" * 70)

            for mode in modes:
                conf = results_by_mode[mode].confidence
                bar = "#" * int(conf * 20)
                print(f"   {mode:12} [{bar:<20}] {conf:.2f}")

            # 6. Source count comparison
            print("\n6. Sources Retrieved by Mode:")
            print("-" * 70)

            for mode in modes:
                sources = len(results_by_mode[mode].sources)
                print(f"   {mode:12} {sources} sources")

            # 7. Recommendations
            print("\n7. Mode Selection Guide:")
            print("-" * 70)
            print("\n   Choose DIRECT mode if:")
            print("   • You need fast responses (<200ms)")
            print("   • Questions are simple and factual")
            print("   • You don't need verification")

            print("\n   Choose VERIFY mode if:")
            print("   • You want a good balance (recommended default)")
            print("   • You need moderate confidence (60-85%)")
            print("   • Questions might have multiple perspectives")

            print("\n   Choose RESEARCH mode if:")
            print("   • You have time for comprehensive answers")
            print("   • You need high confidence (85%+)")
            print("   • Questions are open-ended or complex")

            # 8. Summary
            print("\n8. Summary:")
            print("   ✓ Demonstrated 3 reasoning modes")
            print("   ✓ Compared confidence, latency, and sources")
            print("   ✓ Showed when to use each mode")
            print()

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
