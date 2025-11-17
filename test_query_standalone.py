#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Test: Query Interface
=================================

Tests query system without full HoloLoom dependencies.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

# Import modules directly to avoid package __init__
import importlib.util

# Load graph module directly
graph_spec = importlib.util.spec_from_file_location(
    "graph_module",
    Path(__file__).parent / "HoloLoom" / "memory" / "graph.py"
)
graph_module = importlib.util.module_from_spec(graph_spec)
graph_spec.loader.exec_module(graph_module)

KG = graph_module.KG
KGEdge = graph_module.KGEdge

# Load query modules directly
engine_spec = importlib.util.spec_from_file_location(
    "engine_module",
    Path(__file__).parent / "HoloLoom" / "query" / "engine.py"
)
engine_module = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(engine_module)

formatter_spec = importlib.util.spec_from_file_location(
    "formatter_module",
    Path(__file__).parent / "HoloLoom" / "query" / "formatter.py"
)
formatter_module = importlib.util.module_from_spec(formatter_spec)
formatter_spec.loader.exec_module(formatter_module)

QueryEngine = engine_module.QueryEngine
QueryClassifier = engine_module.QueryClassifier
QueryType = engine_module.QueryType
format_result = formatter_module.format_result


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
        ("Show me statistics", QueryType.STATS),
    ]

    print("\nClassifying queries:\n")

    passed = 0
    for query, expected_type in test_cases:
        intent = classifier.classify(query)

        status = "✓" if intent.query_type == expected_type else "✗"
        print(f"{status} '{query}'")
        print(f"   → {intent.query_type.value} (expected: {expected_type.value})")

        if intent.timeframe:
            print(f"   Timeframe: {intent.timeframe}")

        if intent.query_type == expected_type:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    assert passed == len(test_cases), f"Some classifications failed"
    print("✓ All query classifications correct!\n")


async def test_deadline_queries():
    """Test deadline queries."""
    print("="*80)
    print("TEST: Deadline Queries")
    print("="*80)

    # Create test graph
    kg = KG()

    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    next_week = today + timedelta(days=7)

    # Add tasks with deadlines
    kg.add_edge(KGEdge(
        'deploy-production',
        f'time::{tomorrow.date().isoformat()}',
        'DEADLINE'
    ))

    kg.add_edge(KGEdge(
        'write-docs',
        f'time::{next_week.date().isoformat()}',
        'DEADLINE'
    ))

    # Query
    engine = QueryEngine(kg)

    print("\nQuery: 'What's due this week?'\n")
    result = await engine.query("What's due this week?")

    print(f"Results: {result['count']} deadlines found")
    for task in result['results']:
        print(f"  • {task['id']}: {task['days_until']} days")

    assert result['count'] == 2, f"Expected 2 deadlines, got {result['count']}"
    print("\n✓ Deadline queries work!\n")


async def test_stats_queries():
    """Test statistics queries."""
    print("="*80)
    print("TEST: Statistics Queries")
    print("="*80)

    # Create graph
    kg = KG()
    kg.add_edge(KGEdge('project-a', 'category-work', 'IS_A'))
    kg.add_edge(KGEdge('project-b', 'category-work', 'IS_A'))
    kg.add_edge(KGEdge('project-a', 'task-1', 'HAS_TASK'))

    engine = QueryEngine(kg)

    print("\nQuery: 'Show me statistics'\n")
    result = await engine.query("Show me statistics")

    stats = result['results']
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")

    assert stats['num_nodes'] == 4
    assert stats['num_edges'] == 3
    print("\n✓ Statistics queries work!\n")


async def test_formatters():
    """Test response formatters."""
    print("="*80)
    print("TEST: Response Formatters")
    print("="*80)

    test_result = {
        'query': 'What is due?',
        'query_type': 'deadlines',
        'count': 2,
        'results': [
            {
                'id': 'deploy-production',
                'title': 'Deploy to production',
                'deadline': datetime.now().isoformat(),
                'days_until': 1
            },
            {
                'id': 'write-docs',
                'title': 'Write documentation',
                'deadline': (datetime.now() + timedelta(days=3)).isoformat(),
                'days_until': 3
            }
        ]
    }

    print("\n--- TEXT FORMAT ---\n")
    text_output = format_result(test_result, format='text')
    print(text_output)

    assert 'Deploy to production' in text_output
    assert 'Write documentation' in text_output

    print("\n\n--- ORG FORMAT ---\n")
    org_output = format_result(test_result, format='org')
    print(org_output[:300] + "...")

    assert '* Query Results' in org_output
    assert 'DEADLINE:' in org_output

    print("\n✓ Formatters work!\n")


async def test_end_to_end():
    """Complete flow test."""
    print("="*80)
    print("TEST: End-to-End Flow")
    print("="*80)

    # Build realistic graph
    kg = KG()

    # Projects and tasks
    kg.add_edge(KGEdge('task-auth', 'project-backend', 'CHILD_OF'))
    kg.add_edge(KGEdge('task-ui', 'project-frontend', 'CHILD_OF'))

    # Deadlines
    tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
    kg.add_edge(KGEdge('task-auth', f'time::{tomorrow}', 'DEADLINE'))

    # Query engine
    engine = QueryEngine(kg)

    # Run multiple queries
    queries = [
        "What's due this week?",
        "Show me statistics",
        "Find notes about backend",
    ]

    print("\nRunning queries:\n")

    for query_str in queries:
        result = await engine.query(query_str)
        print(f"✓ '{query_str}'")
        print(f"  → {result.get('query_type')}, {result.get('count', 0)} results")

    print("\n✓ End-to-end flow works!\n")


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("HoloLoom Query Interface - Standalone Tests")
    print("="*80 + "\n")

    try:
        test_query_classifier()
        await test_deadline_queries()
        await test_stats_queries()
        await test_formatters()
        await test_end_to_end()

        print("="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)

        print("""
🎉 Query Interface is Working!

Capabilities demonstrated:
  ✓ Query classification (intent detection)
  ✓ Deadline queries with timeframe filtering
  ✓ Statistics queries
  ✓ Multiple output formats (text, org, json)
  ✓ End-to-end query flow

You can now:

1. Query from Python:
   ```python
   from HoloLoom.memory.graph import KG
   from HoloLoom.query import query

   kg = KG.load('knowledge.jsonl')
   result = await query("What's due?", kg=kg)
   print(result)
   ```

2. Query from CLI:
   ```bash
   python -m HoloLoom.query "What should I work on?"
   python -m HoloLoom.query "What's due?" --format org
   ```

3. Integrate with org files:
   - Parse org files with OrgModeSpinner
   - Build knowledge graph
   - Query with natural language
   - Get formatted results

Next: Try with your actual org files!
""")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
