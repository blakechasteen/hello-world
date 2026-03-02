"""
Codebase Spinner Usage Examples
================================

Demonstrates how to use CodebaseSpinner to ingest source code
into HoloLoom memory.

Requirements:
    # No external dependencies - uses stdlib ast module

Optional (for better entity extraction):
    pip install spacy
    python -m spacy download en_core_web_sm

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
from typing import List

from hololoom.spinningWheel.codebase_spinner import (
    CodebaseSpinner,
    spin_codebase,
    create_codebase_scorer
)
from hololoom.spinningWheel.protocol import SpinResult
from hololoom.documentation.types import MemoryShard


# =============================================================================
# Example 1: Basic Single File Ingestion
# =============================================================================

async def example_1_basic_file():
    """
    Basic example: Spin a single Python file into MemoryShards

    Use case: Understand code structure and documentation
    """
    print("=" * 60)
    print("Example 1: Basic Single File Ingestion")
    print("=" * 60)

    # Initialize spinner
    spinner = CodebaseSpinner(
        importance_threshold=0.3,  # Filter out trivial files
        languages=['python'],
        include_tests=False
    )

    try:
        # Spin a single file
        file_path = Path("./HoloLoom/policy/unified.py")
        result: SpinResult = await spinner.spin(file_path)

        print(f"\nProcessed: {result.items_processed} files")
        print(f"Shards created: {len(result.shards)}")

        # Inspect first shard
        if result.shards:
            shard = result.shards[0]
            print(f"\nCode file analysis:")
            print(f"  File: {shard.metadata.get('file_name', 'N/A')}")
            print(f"  Language: {shard.metadata.get('language', 'N/A')}")
            print(f"  Lines: {shard.metadata.get('total_lines', 'N/A')}")
            print(f"  Classes: {shard.metadata.get('class_count', 0)}")
            print(f"  Functions: {shard.metadata.get('function_count', 0)}")
            print(f"  Complexity: {shard.metadata.get('complexity_score', 0):.2f}")
            print(f"  Entities: {shard.entities[:5]}")  # First 5
            print(f"  Motifs: {shard.motifs}")

    except FileNotFoundError:
        print(f"\nNote: File not found - adjust path to your codebase")


# =============================================================================
# Example 2: Directory Ingestion with Filtering
# =============================================================================

async def example_2_directory():
    """
    Ingest entire directory with filtering

    Use case: Index project for code search
    """
    print("=" * 60)
    print("Example 2: Directory Ingestion")
    print("=" * 60)

    spinner = CodebaseSpinner(
        importance_threshold=0.3,
        languages=['python'],
        include_tests=False,  # Exclude test files
        max_files=50  # Limit for demo
    )

    try:
        # Spin entire directory
        code_dir = Path("./HoloLoom/policy/")
        result = await spinner.spin_directory(code_dir, recursive=True)

        print(f"\nProcessed: {result.items_processed} files")
        print(f"Filtered: {result.items_filtered} files")
        print(f"Shards created: {len(result.shards)}")

        # Show statistics
        total_classes = sum(s.metadata.get('class_count', 0) for s in result.shards)
        total_functions = sum(s.metadata.get('function_count', 0) for s in result.shards)
        total_lines = sum(s.metadata.get('total_lines', 0) for s in result.shards)

        print(f"\nCodebase statistics:")
        print(f"  Total classes: {total_classes}")
        print(f"  Total functions: {total_functions}")
        print(f"  Total lines: {total_lines}")

        # Show most complex files
        complex_files = sorted(
            result.shards,
            key=lambda s: s.metadata.get('complexity_score', 0),
            reverse=True
        )[:3]

        print("\nMost complex files:")
        for shard in complex_files:
            print(f"  - {shard.metadata.get('file_name', 'Unknown')}")
            print(f"    Complexity: {shard.metadata.get('complexity_score', 0):.2f}")
            print(f"    Classes: {shard.metadata.get('class_count', 0)}, "
                  f"Functions: {shard.metadata.get('function_count', 0)}")

    except FileNotFoundError:
        print(f"\nNote: Adjust directory path to your codebase")


# =============================================================================
# Example 3: Importance Filtering
# =============================================================================

async def example_3_importance_filtering():
    """
    Demonstrate importance filtering at different thresholds

    Use case: Focus on core implementation vs include everything
    """
    print("=" * 60)
    print("Example 3: Importance Filtering")
    print("=" * 60)

    code_dir = Path("./HoloLoom/")
    thresholds = [0.2, 0.5, 0.8]

    for threshold in thresholds:
        spinner = CodebaseSpinner(
            importance_threshold=threshold,
            languages=['python'],
            include_tests=False,
            max_files=100
        )

        try:
            result = await spinner.spin_directory(code_dir, recursive=True)

            print(f"\nThreshold {threshold}:")
            print(f"  Files processed: {result.items_processed}")
            print(f"  Shards created: {len(result.shards)}")
            print(f"  Filter rate: {result.items_filtered / result.items_processed * 100:.1f}%")

            # Show importance distribution
            if result.shards:
                scores = [s.metadata.get('importance_score', 0) for s in result.shards]
                avg_score = sum(scores) / len(scores)
                print(f"  Average importance: {avg_score:.3f}")

                # Average complexity
                avg_complexity = sum(
                    s.metadata.get('complexity_score', 0) for s in result.shards
                ) / len(result.shards)
                print(f"  Average complexity: {avg_complexity:.2f}")

        except FileNotFoundError:
            print(f"\nNote: Adjust directory path to your codebase")
            break


# =============================================================================
# Example 4: Multi-Language Support
# =============================================================================

async def example_4_multi_language():
    """
    Process codebases with multiple programming languages

    Use case: Full-stack project indexing
    """
    print("=" * 60)
    print("Example 4: Multi-Language Support")
    print("=" * 60)

    spinner = CodebaseSpinner(
        importance_threshold=0.3,
        languages=['python', 'javascript', 'typescript'],
        include_tests=False,
        max_files=100
    )

    code_dir = Path("./")

    try:
        result = await spinner.spin_directory(code_dir, recursive=True)

        print(f"\nProcessed: {result.items_processed} files")
        print(f"Shards created: {len(result.shards)}")

        # Group by language
        by_language = {}
        for shard in result.shards:
            lang = shard.metadata.get('language', 'unknown')
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(shard)

        print("\nFiles by language:")
        for lang, shards in sorted(by_language.items(), key=lambda x: len(x[1]), reverse=True):
            total_lines = sum(s.metadata.get('total_lines', 0) for s in shards)
            print(f"  {lang}: {len(shards)} files, {total_lines} lines")

    except FileNotFoundError:
        print(f"\nNote: Adjust directory path to your codebase")


# =============================================================================
# Example 5: Streaming Large Codebases
# =============================================================================

async def example_5_streaming():
    """
    Stream MemoryShards for memory-efficient processing

    Use case: Very large codebases (1000+ files)
    """
    print("=" * 60)
    print("Example 5: Streaming Large Codebases")
    print("=" * 60)

    spinner = CodebaseSpinner(
        importance_threshold=0.3,
        languages=['python'],
        include_tests=False,
        max_files=1000  # Large codebase
    )

    code_dir = Path("./HoloLoom/")

    try:
        shard_count = 0
        total_classes = 0
        total_functions = 0

        # Process shards one at a time
        async for shard in spinner.spin_stream(code_dir, recursive=True, batch_size=10):
            shard_count += 1
            total_classes += shard.metadata.get('class_count', 0)
            total_functions += shard.metadata.get('function_count', 0)

            # Process shard (e.g., add to memory, index, etc.)
            # await memory.add_shard(shard)

            # Progress indicator
            if shard_count % 10 == 0:
                print(f"  Processed {shard_count} files...")

        print(f"\nTotal files streamed: {shard_count}")
        print(f"Total classes: {total_classes}")
        print(f"Total functions: {total_functions}")

    except FileNotFoundError:
        print(f"\nNote: Adjust directory path to your codebase")


# =============================================================================
# Example 6: Custom Importance Scoring
# =============================================================================

async def example_6_custom_scoring():
    """
    Customize importance scoring for domain-specific needs

    Use case: Prioritize specific patterns or architectural components
    """
    print("=" * 60)
    print("Example 6: Custom Importance Scoring")
    print("=" * 60)

    # Create custom scorer for ML/AI codebases
    scorer = create_codebase_scorer()
    scorer.add_technical_terms({
        'neural', 'network', 'layer', 'tensor', 'embedding',
        'transformer', 'attention', 'activation', 'optimizer',
        'loss', 'gradient', 'backprop', 'forward', 'training'
    })

    spinner = CodebaseSpinner(
        importance_threshold=0.3,
        languages=['python'],
        include_tests=False
    )

    # Override scorer
    spinner.importance_scorer = scorer

    code_dir = Path("./HoloLoom/policy/")

    try:
        result = await spinner.spin_directory(code_dir, recursive=True)

        print(f"\nProcessed: {result.items_processed} files")
        print(f"High-priority files: {len(result.shards)}")

        # Show ML/AI related files
        ml_files = [
            s for s in result.shards
            if any(term in s.text.lower()
                   for term in ['neural', 'transformer', 'attention', 'embedding'])
        ]

        print(f"\nML/AI related files: {len(ml_files)}")
        for shard in ml_files[:5]:  # Show first 5
            print(f"  - {shard.metadata.get('file_name', 'Unknown')}")
            classes = [e for e in shard.entities if e.isupper() or e[0].isupper()][:3]
            print(f"    Classes: {', '.join(classes) if classes else 'None'}")

    except FileNotFoundError:
        print(f"\nNote: Adjust directory path to your codebase")


# =============================================================================
# Example 7: Include Tests for Full Coverage
# =============================================================================

async def example_7_with_tests():
    """
    Include test files for comprehensive codebase analysis

    Use case: Understanding test coverage and patterns
    """
    print("=" * 60)
    print("Example 7: Include Tests for Full Coverage")
    print("=" * 60)

    # Include tests
    spinner = CodebaseSpinner(
        importance_threshold=0.2,  # Lower threshold for tests
        languages=['python'],
        include_tests=True,  # Include test files
        max_files=100
    )

    code_dir = Path("./HoloLoom/tests/")

    try:
        result = await spinner.spin_directory(code_dir, recursive=True)

        print(f"\nProcessed: {result.items_processed} files")
        print(f"Shards created: {len(result.shards)}")

        # Separate production vs test files
        test_files = [
            s for s in result.shards
            if 'test_' in s.metadata.get('file_name', '').lower()
        ]
        prod_files = [
            s for s in result.shards
            if 'test_' not in s.metadata.get('file_name', '').lower()
        ]

        print(f"\nProduction files: {len(prod_files)}")
        print(f"Test files: {len(test_files)}")

        # Test coverage analysis
        test_functions = sum(s.metadata.get('function_count', 0) for s in test_files)
        prod_functions = sum(s.metadata.get('function_count', 0) for s in prod_files)

        print(f"\nTest coverage estimate:")
        print(f"  Production functions: {prod_functions}")
        print(f"  Test functions: {test_functions}")
        if prod_functions > 0:
            ratio = test_functions / prod_functions
            print(f"  Test/Prod ratio: {ratio:.2f}x")

        # Show test file structure
        print("\nTest files found:")
        for shard in test_files[:5]:
            print(f"  - {shard.metadata.get('file_name', 'Unknown')}")
            print(f"    Tests: {shard.metadata.get('function_count', 0)}")

    except FileNotFoundError:
        print(f"\nNote: Adjust directory path to your test suite")


# =============================================================================
# Main Runner
# =============================================================================

async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Codebase Spinner Examples")
    print("=" * 60 + "\n")

    # NOTE: Update these with your actual codebase paths
    print("SETUP REQUIRED:")
    print("1. Python AST parsing uses stdlib (no dependencies)")
    print("2. Optional: pip install spacy (better entity extraction)")
    print("3. Adjust paths in examples to your codebase")
    print("4. Currently supports: Python (more languages coming soon)\n")

    examples = [
        ("Basic Single File Ingestion", example_1_basic_file),
        ("Directory Ingestion", example_2_directory),
        ("Importance Filtering", example_3_importance_filtering),
        ("Multi-Language Support", example_4_multi_language),
        ("Streaming Large Codebases", example_5_streaming),
        ("Custom Importance Scoring", example_6_custom_scoring),
        ("Include Tests for Full Coverage", example_7_with_tests),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nUncomment examples in main() to run them individually.")

    # Uncomment to run examples:
    # await example_1_basic_file()
    # await example_2_directory()
    # await example_3_importance_filtering()
    # await example_4_multi_language()
    # await example_5_streaming()
    # await example_6_custom_scoring()
    # await example_7_with_tests()


if __name__ == "__main__":
    asyncio.run(main())
