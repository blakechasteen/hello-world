"""
HoloLoom API Example Usage
===========================

Example Python client demonstrating all API endpoints.

Requirements:
    pip install requests

Usage:
    python HoloLoom/api/example_usage.py

Author: HoloLoom Team
Date: 2025-11-18 (Week 8B)
"""

import requests
import json
from datetime import datetime, timedelta
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"  # Set your API key

# Headers
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_response(response):
    """Pretty print response."""
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
    print()


# ============================================================================
# 1. Health Check
# ============================================================================

print_section("1. Health Check (No Auth Required)")

response = requests.get(f"{API_BASE_URL}/health")
print_response(response)


# ============================================================================
# 2. Create Memories
# ============================================================================

print_section("2. Create Episodic Memories")

memories_to_create = [
    "Thompson Sampling balances exploration and exploitation using Bayesian priors",
    "Multi-armed bandits are a class of sequential decision problems",
    "Upper Confidence Bound (UCB) is an alternative to Thompson Sampling",
    "Reinforcement learning involves learning from trial and error",
    "Exploration vs exploitation is the fundamental tradeoff in RL"
]

memory_ids = []
for content in memories_to_create:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/experience",
        headers=HEADERS,
        json={
            "content": content,
            "scope": "SESSION",
            "metadata": {"source": "example_script"}
        }
    )
    if response.status_code == 200:
        memory_id = response.json()["memory_id"]
        memory_ids.append(memory_id)
        print(f"✓ Created memory: {memory_id}")
    else:
        print(f"✗ Failed to create memory: {response.text}")

print(f"\nTotal memories created: {len(memory_ids)}")


# ============================================================================
# 3. Recall Memories
# ============================================================================

print_section("3. Recall Memories")

queries = [
    "What is Thompson Sampling?",
    "Tell me about exploration vs exploitation",
    "What are multi-armed bandits?"
]

for query in queries:
    print(f"\nQuery: {query}")
    response = requests.post(
        f"{API_BASE_URL}/api/v1/recall",
        headers=HEADERS,
        json={
            "query": query,
            "max_results": 3,
            "include_metadata": False
        }
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['count']} memories (latency: {data['latency_ms']:.1f}ms)")
        for mem in data['memories'][:2]:  # Show first 2
            print(f"  - [{mem['relevance_score']:.2f}] {mem['content'][:60]}...")
    else:
        print(f"Error: {response.text}")


# ============================================================================
# 4. Trigger Consolidation
# ============================================================================

print_section("4. Trigger Consolidation")

response = requests.post(
    f"{API_BASE_URL}/api/v1/consolidation/trigger",
    headers=HEADERS,
    json={
        "strategy": "fact_extraction",
        "max_episodes": 10,
        "prune_episodes": False
    }
)
print_response(response)


# ============================================================================
# 5. Detect Patterns
# ============================================================================

print_section("5. Detect Episodic Patterns")

# Wait a bit for memories to be processed
time.sleep(2)

response = requests.post(
    f"{API_BASE_URL}/api/v1/semantic/detect-patterns",
    headers=HEADERS,
    json={
        "threshold": 2,
        "similarity_threshold": 0.7,
        "window_days": 7,
        "max_patterns": 10
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Detected {data['count']} patterns (time: {data['detection_time_ms']:.1f}ms)")
    for pattern in data['patterns'][:3]:  # Show first 3
        print(f"\nPattern: {pattern['pattern_id']}")
        print(f"  Type: {pattern['pattern_type']}")
        print(f"  Frequency: {pattern['frequency']}")
        print(f"  Similarity: {pattern['similarity_score']:.2f}")
        print(f"  Query: {pattern['representative_query'][:60]}...")
else:
    print(f"Error: {response.text}")


# ============================================================================
# 6. Temporal Query
# ============================================================================

print_section("6. Temporal Evolution Query")

# Query understanding at current time
timestamp = datetime.now().isoformat() + "Z"

response = requests.post(
    f"{API_BASE_URL}/api/v1/temporal/query",
    headers=HEADERS,
    json={
        "concept": "thompson_sampling",
        "timestamp": timestamp
    }
)

if response.status_code == 200:
    data = response.json()
    if data['found']:
        snapshot = data['snapshot']
        print(f"Concept: {snapshot['concept']}")
        print(f"State: {snapshot['state']}")
        print(f"Memories: {snapshot['memory_count']}")
        print(f"Confidence: {snapshot['confidence']:.2f}")
    else:
        print("No temporal data found for this concept")
else:
    print(f"Error: {response.text}")


# ============================================================================
# 7. Get Evolution Summary
# ============================================================================

print_section("7. Evolution Summary")

response = requests.post(
    f"{API_BASE_URL}/api/v1/temporal/summary",
    headers=HEADERS,
    json={"days": 30}
)

if response.status_code == 200:
    data = response.json()
    print(f"Period: {data['period_days']} days")
    print(f"Total concepts: {data['total_concepts']}")
    print(f"New concepts: {data['new_concepts']}")
    print(f"Mastered: {data['mastered_concepts']}")
    print(f"\nConcepts by state:")
    for state, count in data['concepts_by_state'].items():
        print(f"  {state}: {count}")
    print(f"\nTop learning areas:")
    for area in data['top_learning_areas'][:5]:
        print(f"  - {area}")
else:
    print(f"Error: {response.text}")


# ============================================================================
# 8. Curiosity Suggestions
# ============================================================================

print_section("8. Curiosity Suggestions")

response = requests.post(
    f"{API_BASE_URL}/api/v1/curiosity/suggest",
    headers=HEADERS,
    json={
        "limit": 5,
        "importance_threshold": 0.5,
        "include_serendipity": True
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Generated {data['count']} suggestions (time: {data['generation_time_ms']:.1f}ms)")
    for suggestion in data['suggestions']:
        print(f"\n💡 {suggestion['concept']} ({suggestion['importance']:.0%} important)")
        print(f"   Type: {suggestion['type']}")
        print(f"   Reason: {suggestion['reason'][:80]}...")
        print(f"   Suggested: \"{suggestion['suggested_query']}\"")
else:
    print(f"Error: {response.text}")


# ============================================================================
# 9. Multi-Hop Graph Query
# ============================================================================

print_section("9. Multi-Hop Graph Reasoning")

response = requests.post(
    f"{API_BASE_URL}/api/v1/graph/multi-hop",
    headers=HEADERS,
    json={
        "query": "What concepts are related to Thompson Sampling?",
        "max_hops": 2,
        "max_results": 5
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Query: {data['query']}")
    print(f"Explored {data['total_paths_explored']} paths (latency: {data['latency_ms']:.1f}ms)")
    print(f"Found entities: {', '.join(data['query_entities'])}")
    print(f"\nResults ({len(data['results'])}):")
    for result in data['results'][:3]:
        print(f"  [{result['relevance_score']:.2f}] {result['content'][:60]}...")
else:
    print(f"Error: {response.text}")


# ============================================================================
# 10. System Statistics
# ============================================================================

print_section("10. System Statistics")

response = requests.get(f"{API_BASE_URL}/api/v1/stats", headers=HEADERS)

if response.status_code == 200:
    data = response.json()
    print("\nConsolidation:")
    print(f"  Total: {data['consolidation']['total_consolidations']}")
    print(f"  Episodes processed: {data['consolidation']['total_episodes_processed']}")
    print(f"  Facts created: {data['consolidation']['total_facts_created']}")

    print("\nSemantic:")
    print(f"  Patterns detected: {data['semantic']['total_patterns_detected']}")
    print(f"  Concepts created: {data['semantic']['total_concepts_created']}")

    print("\nCuriosity:")
    print(f"  Suggestions generated: {data['curiosity']['total_suggestions_generated']}")
    print(f"  Follow rate: {data['curiosity']['follow_rate']:.1%}")
else:
    print(f"Error: {response.text}")


# ============================================================================
# Complete!
# ============================================================================

print_section("Example Complete!")
print("All API endpoints demonstrated successfully.")
print("\nNext steps:")
print("  - Explore API docs: http://localhost:8000/docs")
print("  - View metrics: http://localhost:8000/metrics")
print("  - Monitor with Prometheus: http://localhost:9090")
print("  - Visualize with Grafana: http://localhost:3000")
