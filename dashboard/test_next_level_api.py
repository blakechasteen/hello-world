#!/usr/bin/env python3
"""
🚀 Test Script for NEXT-LEVEL Query API
=========================================
Demonstrates all the legendary features!
"""

import requests
import json
import time
from typing import Dict, Any


BASE_URL = "http://localhost:8001"


def print_section(title: str):
    """Print a fancy section header."""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80 + "\n")


def pretty_print(data: Dict[Any, Any]):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))


def test_basic_query():
    """Test basic enhanced query."""
    print_section("1. Basic Enhanced Query")

    response = requests.post(f"{BASE_URL}/api/query", json={
        "text": "What is Thompson Sampling?",
        "pattern": "fast",
        "enable_narrative_depth": True,
        "include_trace": True
    })

    data = response.json()
    print(f"Query: {data['query_text']}")
    print(f"Response: {data['response'][:200]}...")
    print(f"Confidence: {data['confidence']:.2%}")
    print(f"Tool: {data['tool_used']}")
    print(f"Duration: {data['duration_ms']:.1f}ms")
    print(f"Cache Hit: {data['cache_hit']}")

    if data.get('insights'):
        print(f"\nInsights:")
        print(f"  Entities: {data['insights'].get('entities_detected', [])[:5]}")
        print(f"  Narrative Depth: {data['insights'].get('narrative_depth')}")


def test_query_enhancement():
    """Test AI-powered query enhancement."""
    print_section("2. AI-Powered Query Enhancement")

    # Test with abbreviated query
    response = requests.post(f"{BASE_URL}/api/query/enhance", params={
        "text": "What is TS"
    })

    data = response.json()
    print(f"Original: {data['original']}")
    print(f"Enhanced: {data['enhanced']}")
    print(f"Enhancements: {data['enhancements']}")
    print(f"Alternatives:")
    for alt in data['alternatives']:
        print(f"  • {alt}")


def test_semantic_cache():
    """Test semantic caching with similar queries."""
    print_section("3. Semantic Caching")

    queries = [
        "What is Thompson Sampling?",
        "Explain Thompson Sampling to me",  # Similar!
        "Tell me about Thompson Sampling",  # Also similar!
    ]

    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")
        start = time.time()

        response = requests.post(f"{BASE_URL}/api/query", json={
            "text": query,
            "pattern": "fast"
        })

        duration = (time.time() - start) * 1000
        data = response.json()

        print(f"  Cache Hit: {data['cache_hit']}")
        print(f"  Duration: {duration:.1f}ms")

        if 'metadata' in data and 'cache_similarity' in data.get('metadata', {}):
            print(f"  Similarity: {data['metadata']['cache_similarity']:.2%}")


def test_query_chain():
    """Test query chain orchestration."""
    print_section("4. Query Chain Orchestration")

    response = requests.post(f"{BASE_URL}/api/query/chain", params={
        "chain_id": "exploration"
    }, json={
        "topic": "Matryoshka embeddings"
    })

    data = response.json()
    print(f"Chain: {data['chain_id']}")
    print(f"Queries Executed: {data['queries_executed']}")
    print(f"Total Duration: {data['total_duration_ms']:.1f}ms\n")

    for i, result in enumerate(data['results']):
        print(f"Query {i+1}: {result['query_text']}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Confidence: {result['confidence']:.2%}\n")


def test_ab_testing():
    """Test A/B testing framework."""
    print_section("5. A/B Testing Framework")

    response = requests.post(f"{BASE_URL}/api/query/ab-test", params={
        "text": "What is Thompson Sampling?"
    }, json={
        "patterns": ["bare", "fast", "fused"]
    })

    data = response.json()
    print(f"Test Query: {data['test_query']}")
    print(f"Winner: {data['winner']}")
    print(f"Win Margin: {data['win_margin']:.2%}\n")

    print("Results:")
    for pattern, result in data['results'].items():
        print(f"  {pattern.upper()}:")
        print(f"    Confidence: {result['confidence']:.2%}")
        print(f"    Duration: {result['duration_ms']:.1f}ms")
        print(f"    Tool: {result['tool_used']}")


def test_predictive_engine():
    """Test predictive query suggestions."""
    print_section("6. Predictive Engine")

    # First, make a query to establish context
    requests.post(f"{BASE_URL}/api/query", json={
        "text": "What is Thompson Sampling?"
    })

    # Then make a follow-up
    requests.post(f"{BASE_URL}/api/query", json={
        "text": "How does it compare to epsilon-greedy?"
    })

    # Now get predictions
    response = requests.get(f"{BASE_URL}/api/query/predict", params={
        "current_query": "What is Thompson Sampling?",
        "k": 5
    })

    data = response.json()
    print(f"Current Query: {data['current_query']}\n")
    print("Predicted Next Queries:")

    for pred in data['predictions']:
        print(f"  • {pred['query']} ({pred['probability']:.0%})")


def test_templates():
    """Test query templates."""
    print_section("7. Query Templates")

    # Create template
    requests.post(f"{BASE_URL}/api/templates/create", params={
        "template_id": "comparison",
        "template": "Compare {thing1} and {thing2} in the context of {domain}"
    })

    print("Created template: 'comparison'")

    # Execute template
    response = requests.post(
        f"{BASE_URL}/api/templates/comparison/execute",
        json={
            "thing1": "Thompson Sampling",
            "thing2": "UCB",
            "domain": "multi-armed bandits"
        }
    )

    data = response.json()
    print(f"\nGenerated Query: {data['query_text']}")
    print(f"Response: {data['response'][:150]}...")
    print(f"Confidence: {data['confidence']:.2%}")


def test_analytics():
    """Test analytics dashboard."""
    print_section("8. Analytics Dashboard")

    response = requests.get(f"{BASE_URL}/api/analytics")
    data = response.json()

    print(f"Total Queries: {data['total_queries']}")
    print(f"Average Duration: {data['average_duration_ms']:.1f}ms")
    print(f"Average Confidence: {data['average_confidence']:.2%}")

    print("\nPattern Distribution:")
    for pattern, count in data['pattern_distribution'].items():
        print(f"  {pattern}: {count} queries")

    print(f"\nQueries Last Hour: {data['queries_last_hour']}")


def test_cache_stats():
    """Test cache statistics."""
    print_section("9. Cache Statistics")

    response = requests.get(f"{BASE_URL}/api/cache/stats")
    data = response.json()

    print(f"Cache Size: {data['cache_size']}/{data['max_size']}")
    print(f"Similarity Threshold: {data['threshold']:.0%}")

    if data.get('oldest_entry'):
        print(f"Oldest Entry: {data['oldest_entry']}")
        print(f"Newest Entry: {data['newest_entry']}")


def test_performance_flamegraph():
    """Test performance flamegraph."""
    print_section("10. Performance Flamegraph")

    response = requests.get(f"{BASE_URL}/api/performance/flamegraph")
    data = response.json()

    print(f"Total Query Time: {data['value']:.1f}ms\n")
    print("Stage Breakdown:")

    for child in data['children']:
        percentage = (child['value'] / data['value']) * 100
        print(f"  {child['name']}: {child['value']:.1f}ms ({percentage:.0f}%)")


def test_cool_factor():
    """Test the cool factor endpoint."""
    print_section("11. Cool Factor Check")

    response = requests.get(f"{BASE_URL}/api/cool-factor")
    data = response.json()

    print(f"Cool Factor: {data['cool_factor']}/{data['max_cool']}")
    print(f"Percentage: {data['percentage']}")
    print(f"Status: {data['status']}\n")

    print("Coolness Breakdown:")
    for feature, points in data['factors'].items():
        print(f"  {feature.replace('_', ' ').title()}: {points} points")


def main():
    """Run all tests."""
    print("=" * 80)
    print("🚀 NEXT-LEVEL QUERY API TEST SUITE")
    print("=" * 80)
    print("\nMake sure the API is running at http://localhost:8001")
    print("Run: python enhanced_query_api.py\n")

    input("Press Enter to start tests...")

    try:
        # Run all tests
        test_basic_query()
        test_query_enhancement()
        test_semantic_cache()
        test_query_chain()
        test_ab_testing()
        test_predictive_engine()
        test_templates()
        test_analytics()
        test_cache_stats()
        test_performance_flamegraph()
        test_cool_factor()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETE!")
        print("=" * 80)
        print("\n🌟 Query is officially NEXT-LEVEL! 🌟\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running: python enhanced_query_api.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
