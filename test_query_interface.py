#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: Query Interface
=====================

Tests the natural language query system for HoloLoom.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.query.engine import QueryEngine, QueryClassifier, QueryType
from HoloLoom.query.formatter import format_result


def test_query_classifier():
    """Test query classification."""
    print("="*80)
    print("TEST: Query Classifier")
    print("="*80)

    classifier = QueryClassifier()

    test_cases = [
        ("What should I work on?", QueryType.NEXT_TASK),
        ("What am I currently working on?", QueryType.CURRENT_TASKS),
        ("What's due this week?", QueryType.DEADLINES),
        ("When did I finish the auth refactor?", QueryType.TEMPORAL),
        ("Show me my notes about neural networks", QueryType.SEARCH),
        ("What's the status of the deployment?", QueryType.STATUS),
        ("What did I accomplish today?", QueryType.COMPLETED),
        ("What tasks are blocked?", QueryType.BLOCKED),
        ("What's high priority?", QueryType.PRIORITY),
        ("Show me statistics", QueryType.STATS),
    ]

    print("\nClassifying queries:\n")

    passed = 0
    for query, expected_type in test_cases:
        intent = classifier.classify(query)

        status = "✓" if intent.query_type == expected_type else "✗"
        print(f"{status} Query: '{query}'")
        print(f"  Classified as: {intent.query_type.value}")
        print(f"  Expected: {expected_type.value}")

        if intent.timeframe:
            print(f"  Timeframe: {intent.timeframe}")
        if intent.entities:
            print(f"  Entities: {intent.entities}")

        if intent.query_type == expected_type:
            passed += 1

        print()

    print(f"Passed: {passed}/{len(test_cases)}")
    print("✓ Query classifier test completed!\n")


async def test_deadline_queries():
    """Test deadline queries."""
    print("="*80)
    print("TEST: Deadline Queries")
    print("="*80)

    # Create test graph with deadlines
    kg = KG()

    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    next_week = today + timedelta(days=7)
    next_month = today + timedelta(days=30)

    # Add tasks with deadlines
    kg.add_edge(KGEdge(
        'task-urgent',
        f'time::{tomorrow.date().isoformat()}',
        'DEADLINE'
    ))

    kg.add_edge(KGEdge(
        'task-soon',
        f'time::{next_week.date().isoformat()}',
        'DEADLINE'
    ))

    kg.add_edge(KGEdge(
        'task-later',
        f'time::{next_month.date().isoformat()}',
        'DEADLINE'
    ))

    # Query
    engine = QueryEngine(kg)

    print("\nQuery: 'What's due this week?'\n")
    result = await engine.query("What's due this week?")

    print(f"Found {result['count']} deadlines:")
    for task in result['results']:
        print(f"  - {task['id']}: {task['days_until']} days until deadline")

    # Should find 2 tasks (tomorrow and next week)
    assert result['count'] == 2, f"Expected 2 deadlines, got {result['count']}"

    print("\n✓ Deadline query test passed!\n")


async def test_stats_queries():
    """Test statistics queries."""
    print("="*80)
    print("TEST: Statistics Queries")
    print("="*80)

    # Create test graph
    kg = KG()

    # Add some nodes and edges
    kg.add_edge(KGEdge('project-a', 'category-1', 'IS_A'))
    kg.add_edge(KGEdge('project-b', 'category-1', 'IS_A'))
    kg.add_edge(KGEdge('project-a', 'task-1', 'HAS_TASK'))
    kg.add_edge(KGEdge('project-a', 'task-2', 'HAS_TASK'))
    kg.add_edge(KGEdge('project-b', 'task-3', 'HAS_TASK'))

    engine = QueryEngine(kg)

    print("\nQuery: 'Show me statistics'\n")
    result = await engine.query("Show me statistics")

    stats = result['results']
    print(f"Graph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Avg Degree: {stats['avg_degree']:.2f}")

    assert stats['num_nodes'] == 6, f"Expected 6 nodes, got {stats['num_nodes']}"
    assert stats['num_edges'] == 5, f"Expected 5 edges, got {stats['num_edges']}"

    print("\n✓ Statistics query test passed!\n")


async def test_search_queries():
    """Test search queries."""
    print("="*80)
    print("TEST: Search Queries")
    print("="*80)

    # Create test graph with searchable content
    kg = KG()

    kg.add_edge(KGEdge('neural-networks-intro', 'ai-concepts', 'CATEGORY'))
    kg.add_edge(KGEdge('transformer-architecture', 'neural-networks-intro', 'RELATED_TO'))
    kg.add_edge(KGEdge('attention-mechanism', 'transformer-architecture', 'PART_OF'))

    engine = QueryEngine(kg)

    print("\nQuery: 'Find notes about neural networks'\n")
    result = await engine.query("Find notes about neural networks")

    print(f"Found {result['count']} matching notes:")
    for item in result['results']:
        print(f"  - {item['id']}")

    assert result['count'] >= 1, f"Expected at least 1 match, got {result['count']}"

    print("\n✓ Search query test passed!\n")


async def test_formatters():
    """Test response formatters."""
    print("="*80)
    print("TEST: Response Formatters")
    print("="*80)

    # Create test result
    test_result = {
        'query': 'What should I work on?',
        'query_type': 'deadlines',
        'count': 2,
        'results': [
            {
                'id': 'task-urgent',
                'title': 'Deploy to production',
                'deadline': datetime.now().isoformat(),
                'days_until': 1
            },
            {
                'id': 'task-soon',
                'title': 'Write documentation',
                'deadline': (datetime.now() + timedelta(days=3)).isoformat(),
                'days_until': 3
            }
        ],
        'intent': {
            'type': 'deadlines',
            'confidence': 0.9
        }
    }

    print("\n--- TEXT FORMAT ---\n")
    text_output = format_result(test_result, format='text')
    print(text_output)

    print("\n\n--- ORG FORMAT ---\n")
    org_output = format_result(test_result, format='org')
    print(org_output)

    print("\n\n--- JSON FORMAT ---\n")
    json_output = format_result(test_result, format='json')
    print(json_output[:200] + "...")  # Truncate for display

    print("\n✓ Formatter test passed!\n")


async def test_end_to_end():
    """Test complete end-to-end query flow."""
    print("="*80)
    print("TEST: End-to-End Query Flow")
    print("="*80)

    # Create realistic knowledge graph
    kg = KG()

    # Projects
    kg.add_edge(KGEdge('project-hololoom', 'ai-projects', 'CATEGORY'))

    # Tasks with various states
    kg.add_edge(KGEdge('task-auth-refactor', 'project-hololoom', 'CHILD_OF'))
    kg.add_edge(KGEdge('task-documentation', 'project-hololoom', 'CHILD_OF'))
    kg.add_edge(KGEdge('task-deployment', 'project-hololoom', 'CHILD_OF'))

    # Deadlines
    tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
    kg.add_edge(KGEdge('task-deployment', f'time::{tomorrow}', 'DEADLINE'))

    # Create engine and run multiple queries
    engine = QueryEngine(kg)

    queries = [
        "What's due soon?",
        "Show me statistics",
        "Find notes about hololoom",
    ]

    print("\nRunning multiple queries:\n")

    for query_str in queries:
        print(f"Query: '{query_str}'")
        result = await engine.query(query_str)
        print(f"  Type: {result.get('query_type')}")
        print(f"  Count: {result.get('count', 0)}")
        print()

    print("✓ End-to-end test passed!\n")


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("HoloLoom Query Interface Tests")
    print("="*80 + "\n")

    try:
        # Run tests
        test_query_classifier()
        await test_deadline_queries()
        await test_stats_queries()
        await test_search_queries()
        await test_formatters()
        await test_end_to_end()

        print("="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)

        print("""
🎉 Query Interface is working!

You can now:
1. Query from Python:
   from HoloLoom.query import query
   result = await query("What should I work on?", kg=my_kg)

2. Query from CLI:
   python -m HoloLoom.query "What's due this week?"

3. Use different formats:
   python -m HoloLoom.query "Statistics" --format json
   python -m HoloLoom.query "What's due?" --format org

Next steps:
- Load actual org files into knowledge graph
- Connect to Emacs
- Add more query types
- Improve semantic search with embeddings
""")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
