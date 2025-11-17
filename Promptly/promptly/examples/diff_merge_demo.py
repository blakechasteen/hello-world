#!/usr/bin/env python3
"""
Diff & Merge Demo - Showcase advanced diff and merge capabilities
"""

import tempfile
import shutil
from pathlib import Path

# Import Promptly and diff/merge modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptly import Promptly
from diff import (
    DiffEngine,
    DiffLevel,
    ComparisonEngine,
    TerminalDiff,
    HTMLDiff,
    quick_diff
)
from merge import (
    MergeTool,
    ThreeWayMerge,
    MergeStrategy,
    InteractiveMerge
)


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_1_basic_diff():
    """Demo 1: Basic diffing capabilities"""
    print_section("Demo 1: Basic Diffing")

    old_text = """Summarize the following text:

{text}

Provide a brief summary."""

    new_text = """Provide a concise summary of the following text, highlighting the key points:

{text}

Your summary should be 2-3 sentences maximum."""

    # Character-level diff
    print("Character-level diff:")
    engine = DiffEngine()
    result = engine.diff(old_text, new_text, DiffLevel.CHARACTER)
    print(f"Stats: {result.stats}\n")

    # Word-level diff (better for text)
    print("Word-level diff:")
    result = engine.diff(old_text, new_text, DiffLevel.WORD)
    print(TerminalDiff.render(result))

    # Quick diff
    print("\nQuick diff stats:")
    result = quick_diff(old_text, new_text)
    print(f"Additions: {result.stats.additions}")
    print(f"Deletions: {result.stats.deletions}")
    print(f"Similarity: {result.stats.similarity:.1%}")


def demo_2_version_comparison():
    """Demo 2: Compare prompt versions"""
    print_section("Demo 2: Version Comparison")

    # Create temporary Promptly repository
    test_dir = tempfile.mkdtemp()
    try:
        promptly = Promptly(test_dir)
        promptly.init()

        # Create evolving prompt
        promptly.add(
            "translator",
            "Translate to {language}: {text}",
            {"version_note": "Basic translation"}
        )

        promptly.add(
            "translator",
            "Translate the following text to {language}, preserving tone and style:\n\n{text}",
            {"version_note": "Added tone preservation"}
        )

        promptly.add(
            "translator",
            """Translate the following text to {language}.

Requirements:
- Preserve tone and style
- Maintain formatting
- Use natural, idiomatic expressions

Text: {text}""",
            {"version_note": "Added detailed requirements"}
        )

        # Compare versions
        engine = ComparisonEngine(promptly)

        print("Comparing v1 to v2:")
        comparison = engine.compare_versions("translator", 1, 2, DiffLevel.LINE)
        print(TerminalDiff.render_comparison(comparison))

        print("\n" + "-" * 70 + "\n")

        print("Comparing v2 to v3:")
        comparison = engine.compare_versions("translator", 2, 3, DiffLevel.LINE)
        print(TerminalDiff.render_comparison(comparison))

    finally:
        shutil.rmtree(test_dir)


def demo_3_branch_comparison():
    """Demo 3: Branch comparison"""
    print_section("Demo 3: Branch Comparison")

    test_dir = tempfile.mkdtemp()
    try:
        promptly = Promptly(test_dir)
        promptly.init()

        # Setup main branch
        promptly.add("greeter", "Hello, {name}!")
        promptly.add("farewell", "Goodbye, {name}!")

        # Create feature branch
        promptly.branch("feature", "main")
        promptly.checkout("feature")

        # Modify on feature
        promptly.add("greeter", "Hey {name}, welcome to our app!")
        promptly.add("new_prompt", "This is a new feature prompt")

        # Add to main
        promptly.checkout("main")
        promptly.add("greeter", "Hello {name}, nice to see you!")
        promptly.add("main_only", "Main branch exclusive")

        # Compare branches
        engine = ComparisonEngine(promptly)
        comparison = engine.compare_branches("main", "feature")

        print(TerminalDiff.render_branch_comparison(comparison))

        # Show detailed diff for modified prompts
        if comparison.prompts_modified:
            print("\n" + "-" * 70)
            print("Detailed diff for 'greeter':")
            print("-" * 70 + "\n")
            diff = comparison.diff_results["greeter"]
            print(TerminalDiff.render(diff))

    finally:
        shutil.rmtree(test_dir)


def demo_4_merge():
    """Demo 4: Merging branches"""
    print_section("Demo 4: Branch Merging")

    test_dir = tempfile.mkdtemp()
    try:
        promptly = Promptly(test_dir)
        promptly.init()

        # Setup base
        promptly.add("prompt1", "Base content for prompt 1")
        promptly.add("shared", "Shared base content")

        # Create feature branch
        promptly.branch("feature", "main")
        promptly.checkout("feature")
        promptly.add("prompt2", "Feature-specific prompt")
        promptly.add("shared", "Feature modified shared content")

        # Modify main
        promptly.checkout("main")
        promptly.add("prompt3", "Main-specific prompt")
        # Don't modify 'shared' on main to avoid conflict

        # Merge
        merge_tool = MergeTool(promptly)
        print("Merging 'feature' into 'main'...\n")

        results = merge_tool.merge_branches("feature", "main", MergeStrategy.AUTO)

        # Display results
        for name, result in results.items():
            if result.success:
                print(f"✓ {name}: Merged successfully")
            else:
                print(f"✗ {name}: Has conflicts")
                for conflict in result.conflicts:
                    print(f"  Conflict: {conflict}")

        print(f"\nTotal prompts merged: {len(results)}")

    finally:
        shutil.rmtree(test_dir)


def demo_5_three_way_merge():
    """Demo 5: Three-way merge with conflicts"""
    print_section("Demo 5: Three-Way Merge & Conflict Resolution")

    base = """Analyze the following data:

{data}

Provide insights."""

    ours = """Analyze the following data carefully:

{data}

Provide detailed insights and recommendations."""

    theirs = """Perform a comprehensive analysis of the data:

{data}

Provide insights with statistical evidence."""

    print("Base version:")
    print(base)
    print("\n" + "-" * 70)

    print("\nOur version:")
    print(ours)
    print("\n" + "-" * 70)

    print("\nTheir version:")
    print(theirs)
    print("\n" + "-" * 70 + "\n")

    # Try different merge strategies
    strategies = [
        MergeStrategy.AUTO,
        MergeStrategy.OURS,
        MergeStrategy.THEIRS,
        MergeStrategy.UNION
    ]

    for strategy in strategies:
        result = ThreeWayMerge.merge(base, ours, theirs, strategy)

        print(f"\nStrategy: {strategy.value}")
        print(f"Success: {result.success}")
        print(f"Conflicts: {len(result.conflicts)}")
        print(f"Stats: {result.stats}")

        if result.conflicts:
            print("\nConflict details:")
            for i, conflict in enumerate(result.conflicts, 1):
                print(f"\nConflict {i}:")
                print(conflict.to_marker_format())


def demo_6_html_export():
    """Demo 6: HTML export"""
    print_section("Demo 6: HTML Export")

    old_text = "Simple prompt: {input}"
    new_text = """Enhanced prompt with detailed instructions:

Input: {input}

Please provide a comprehensive response."""

    # Generate diff
    engine = DiffEngine()
    result = engine.diff(old_text, new_text, DiffLevel.LINE)

    # Export to HTML
    html = HTMLDiff.render(result, title="Prompt Enhancement")

    # Save to file
    output_file = Path(tempfile.gettempdir()) / "promptly_diff_demo.html"
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"HTML diff report generated: {output_file}")
    print(f"File size: {output_file.stat().st_size} bytes")
    print("\nHTML preview (first 500 chars):")
    print(html[:500] + "...")

    # Generate side-by-side HTML
    html_sbs = HTMLDiff.render_side_by_side(
        old_text,
        new_text,
        "Original",
        "Enhanced"
    )

    output_file_sbs = Path(tempfile.gettempdir()) / "promptly_diff_sbs_demo.html"
    with open(output_file_sbs, 'w') as f:
        f.write(html_sbs)

    print(f"\nSide-by-side HTML generated: {output_file_sbs}")


def demo_7_semantic_diff():
    """Demo 7: Semantic diff"""
    print_section("Demo 7: Semantic Diff")

    old_text = "You must complete all tasks. For example, finish the report."
    new_text = "You should complete the tasks. This includes finishing reports and documentation."

    engine = DiffEngine()
    result = engine.diff(old_text, new_text, DiffLevel.SEMANTIC)

    print("Semantic diff with context annotations:\n")
    print(TerminalDiff.render(result))

    print("\nSemantic contexts found:")
    for chunk in result.chunks:
        if chunk.context:
            print(f"  - {chunk.type.value}: {chunk.context}")


def demo_8_stats_and_similarity():
    """Demo 8: Statistics and similarity"""
    print_section("Demo 8: Statistics & Similarity Analysis")

    test_pairs = [
        ("Hello world", "Hello world"),  # Identical
        ("Hello world", "Hello there"),  # Similar
        ("Hello world", "Goodbye moon"),  # Different
        ("Short", "This is a much longer text with more content"),  # Length change
    ]

    engine = DiffEngine()

    print(f"{'Old':<30} {'New':<30} {'Similarity':>12} {'Changes':>10}")
    print("-" * 85)

    for old, new in test_pairs:
        result = engine.diff(old, new, DiffLevel.WORD)
        stats = result.stats

        similarity = f"{stats.similarity:.1%}"
        changes = f"+{stats.additions}/-{stats.deletions}"

        print(f"{old[:28]:<30} {new[:28]:<30} {similarity:>12} {changes:>10}")


def run_all_demos():
    """Run all demos"""
    demos = [
        demo_1_basic_diff,
        demo_2_version_comparison,
        demo_3_branch_comparison,
        demo_4_merge,
        demo_5_three_way_merge,
        demo_6_html_export,
        demo_7_semantic_diff,
        demo_8_stats_and_similarity,
    ]

    print("\n" + "=" * 70)
    print("  PROMPTLY DIFF & MERGE DEMO")
    print("  Showcasing advanced version control for prompts")
    print("=" * 70)

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n⚠ Error in {demo.__name__}: {e}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_demos()
