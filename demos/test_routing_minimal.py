#!/usr/bin/env python3
"""
Minimal test of query routing - isolate the issue.
"""

import asyncio
from hololoom.routing.query_classifier import QueryClassifier
from hololoom.routing.fast_paths import handle_trivial_query, handle_simple_query
from hololoom.documentation.types import Query


async def test_trivial():
    """Test trivial query handler directly."""
    print("\n" + "=" * 70)
    print("  TRIVIAL QUERY TEST (Direct Handler)")
    print("=" * 70)

    query = Query(text="hi")
    print(f"\nQuery: '{query.text}'")

    try:
        spacetime = await handle_trivial_query(query)
        print(f"✅ Success!")
        print(f"   Response: {spacetime.response}")
        print(f"   Confidence: {spacetime.confidence}")
        print(f"   Latency: {spacetime.metadata.get('latency_ms', 0):.3f}ms")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


async def test_classifier():
    """Test query classifier."""
    print("\n" + "=" * 70)
    print("  QUERY CLASSIFIER TEST")
    print("=" * 70)

    classifier = QueryClassifier()

    test_cases = [
        ("hi", "TRIVIAL"),
        ("thanks", "TRIVIAL"),
        ("what is X?", "SIMPLE"),
        ("Explain Thompson Sampling in detail", "COMPLEX"),
    ]

    for query_text, expected in test_cases:
        classification = classifier.classify(query_text)
        match = "✅" if classification.complexity.value.upper() == expected else "❌"
        print(f"\n{match} '{query_text}'")
        print(f"   Expected: {expected}")
        print(f"   Got: {classification.complexity.value.upper()}")
        print(f"   Confidence: {classification.confidence:.2f}")
        print(f"   Reasoning: {classification.reasoning}")


async def main():
    """Run all tests."""
    await test_classifier()
    await test_trivial()
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    asyncio.run(main())
