#!/usr/bin/env python3
"""
Examples of using Promptly plugins

Demonstrates:
- Custom evaluators (keyword, semantic, exact_match)
- Different storage backends (sqlite, json)
- Evaluator metrics and scoring
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plugins import get_evaluator, get_storage_backend, list_plugins


def example_1_keyword_evaluator():
    """Example 1: Keyword-based evaluation"""
    print("\n" + "=" * 70)
    print("Example 1: Keyword Evaluator")
    print("=" * 70 + "\n")

    evaluator = get_evaluator("keyword", case_sensitive=False)

    # Simple word overlap
    actual = "The quick brown fox jumps over the lazy dog"
    expected = "fox dog brown"

    score = evaluator.evaluate(actual, expected)
    print(f"Text: {actual}")
    print(f"Expected words: {expected}")
    print(f"Score: {score:.2f}\n")

    # Advanced with context
    actual = "Python is a powerful programming language for data science"
    context = {
        'required_keywords': ['python', 'programming'],
        'optional_keywords': ['data', 'science'],
        'forbidden_keywords': ['java', 'javascript'],
        'min_length': 10,
        'max_length': 200
    }

    score = evaluator.evaluate(actual, "", context=context)
    metrics = evaluator.get_metrics(actual, "", context=context)

    print(f"Text: {actual}")
    print(f"Context: {context}")
    print(f"Score: {score:.2f}")
    print(f"Metrics: {metrics}\n")


def example_2_semantic_evaluator():
    """Example 2: Semantic similarity evaluation"""
    print("\n" + "=" * 70)
    print("Example 2: Semantic Similarity Evaluator")
    print("=" * 70 + "\n")

    evaluator = get_evaluator("semantic", backend="tfidf")

    # Similar meanings
    actual = "Machine learning allows computers to learn from data"
    expected = "AI systems can be trained using datasets"

    score = evaluator.evaluate(actual, expected)
    metrics = evaluator.get_metrics(actual, expected)

    print(f"Actual: {actual}")
    print(f"Expected: {expected}")
    print(f"Similarity score: {score:.2f}")
    print(f"Metrics: {metrics}\n")

    # Different meanings
    actual2 = "The weather is sunny today"
    expected2 = "Machine learning is fascinating"

    score2 = evaluator.evaluate(actual2, expected2)

    print(f"Actual: {actual2}")
    print(f"Expected: {expected2}")
    print(f"Similarity score: {score2:.2f}\n")


def example_3_exact_match_evaluator():
    """Example 3: Exact match evaluation"""
    print("\n" + "=" * 70)
    print("Example 3: Exact Match Evaluator")
    print("=" * 70 + "\n")

    evaluator = get_evaluator("exact_match", case_sensitive=False, ignore_whitespace=True)

    tests = [
        ("hello world", "hello world", True),
        ("Hello World", "hello world", True),  # Case insensitive
        ("hello  world", "hello world", True),  # Whitespace ignored
        ("hello world", "goodbye world", False)
    ]

    for actual, expected, should_match in tests:
        score = evaluator.evaluate(actual, expected)
        match = "✓" if (score == 1.0) == should_match else "✗"
        print(f"{match} '{actual}' vs '{expected}': {score:.0f}")

    print()


def example_4_comparing_evaluators():
    """Example 4: Compare different evaluators on same data"""
    print("\n" + "=" * 70)
    print("Example 4: Comparing Evaluators")
    print("=" * 70 + "\n")

    test_cases = [
        {
            "actual": "Python is great for data science",
            "expected": "Python is excellent for data analysis"
        },
        {
            "actual": "The cat sat on the mat",
            "expected": "A feline rested on the rug"
        },
        {
            "actual": "Hello world",
            "expected": "Hello world"
        }
    ]

    evaluators = {
        "keyword": get_evaluator("keyword"),
        "semantic": get_evaluator("semantic", backend="tfidf"),
        "exact_match": get_evaluator("exact_match", case_sensitive=False)
    }

    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"  Actual: {test['actual']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Scores:")

        for name, evaluator in evaluators.items():
            score = evaluator.evaluate(test['actual'], test['expected'])
            print(f"    {name:15s}: {score:.2f}")

        print()


def example_5_storage_backends():
    """Example 5: Using different storage backends"""
    print("\n" + "=" * 70)
    print("Example 5: Storage Backends")
    print("=" * 70 + "\n")

    # List available backends
    plugins = list_plugins()

    print("Available Storage Backends:")
    for backend in plugins['storage_backends']:
        print(f"  - {backend['name']}: {backend['description']}")

    print("\n" + "-" * 70 + "\n")

    # SQLite example
    print("SQLite Storage:")
    sqlite_storage = get_storage_backend("sqlite")
    print(f"  Name: {sqlite_storage.name}")
    print(f"  Description: {sqlite_storage.description}")

    # JSON example
    print("\nJSON Storage:")
    json_storage = get_storage_backend("json")
    print(f"  Name: {json_storage.name}")
    print(f"  Description: {json_storage.description}")

    print()


def example_6_custom_evaluator_context():
    """Example 6: Using context for advanced evaluation"""
    print("\n" + "=" * 70)
    print("Example 6: Advanced Context-Based Evaluation")
    print("=" * 70 + "\n")

    evaluator = get_evaluator("keyword")

    # Code review context
    code_output = "The function implements binary search efficiently using O(log n) time complexity"

    contexts = [
        {
            "name": "Requires efficiency mention",
            "context": {
                'required_keywords': ['efficient', 'complexity'],
                'optional_keywords': ['fast', 'optimized'],
                'min_length': 20
            }
        },
        {
            "name": "Requires algorithm details",
            "context": {
                'required_keywords': ['binary search', 'time complexity'],
                'optional_keywords': ['logarithmic', 'O(log n)'],
            }
        },
        {
            "name": "Forbids wrong info",
            "context": {
                'required_keywords': ['binary search'],
                'forbidden_keywords': ['O(n)', 'linear', 'slow']
            }
        }
    ]

    print(f"Code Output: {code_output}\n")

    for test in contexts:
        score = evaluator.evaluate(code_output, "", context=test['context'])
        metrics = evaluator.get_metrics(code_output, "", context=test['context'])

        print(f"{test['name']}:")
        print(f"  Score: {score:.2f}")
        if 'required_keywords_found' in metrics:
            print(f"  Required found: {metrics['required_keywords_found']}")
        if 'required_keywords_missing' in metrics:
            print(f"  Required missing: {metrics['required_keywords_missing']}")
        if 'optional_keywords_found' in metrics:
            print(f"  Optional found: {metrics['optional_keywords_found']}")
        print()


def example_7_list_all_plugins():
    """Example 7: List all available plugins"""
    print("\n" + "=" * 70)
    print("Example 7: All Available Plugins")
    print("=" * 70 + "\n")

    plugins = list_plugins()

    print("Evaluators:")
    for evaluator in plugins['evaluators']:
        print(f"  • {evaluator['name']:20s} - {evaluator['description']}")

    print("\nStorage Backends:")
    for backend in plugins['storage_backends']:
        print(f"  • {backend['name']:20s} - {backend['description']}")

    print("\nChain Processors:")
    for processor in plugins['chain_processors']:
        print(f"  • {processor['name']:20s} - {processor['description']}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Promptly Plugin System - Usage Examples")
    print("=" * 70)

    try:
        example_1_keyword_evaluator()
        example_2_semantic_evaluator()
        example_3_exact_match_evaluator()
        example_4_comparing_evaluators()
        example_5_storage_backends()
        example_6_custom_evaluator_context()
        example_7_list_all_plugins()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
