#!/usr/bin/env python3
"""
Promptly Rich CLI Demo - Windows Compatible
============================================
Demonstrates rich terminal output without progress bar Unicode issues.
"""

import sys
from pathlib import Path

# Add promptly to path
sys.path.insert(0, str(Path(__file__).parent / "promptly"))

from cli_output import (
    success, error, warning, info,
    heading, separator,
    print_table, print_dict_table, print_panel
)


def main():
    """Run rich CLI demo."""

    print("\n")
    heading("Promptly Rich CLI Demo", "bold magenta")
    separator()

    # Basic messages
    print("\n[1] Basic Messages:")
    success("Prompt created successfully!")
    error("Failed to load prompt")
    warning("This prompt has no tags")
    info("Tip: Use auto-tagging to suggest tags")

    separator()

    # Tables
    print("\n[2] Tables:")
    print_table(
        "Top Prompts by Quality",
        ["Name", "Quality", "Usage", "Tags"],
        [
            ["SQL Optimizer", "0.92", "42", "sql, optimization"],
            ["Code Reviewer", "0.89", "67", "code, review"],
            ["Bug Detective", "0.85", "89", "debugging"],
            ["Docs Generator", "0.78", "34", "documentation"]
        ]
    )

    separator()

    # Dict table
    print("\n[3] Key-Value Display:")
    print_dict_table({
        "Total Prompts": "15",
        "Total Executions": "340",
        "Success Rate": "94.2%",
        "Avg Quality": "0.88",
        "Most Used": "Bug Detective (89 uses)"
    }, title="System Statistics")

    separator()

    # Panel
    print("\n[4] Panels:")
    print_panel(
        """
This is a styled panel! You can use it for:
- Important notifications
- Configuration display
- Results summary
- Status messages
        """.strip(),
        title="Example Panel",
        style="cyan"
    )

    separator()

    # Search results simulation
    print("\n[5] Search Results:")
    from cli_output import print_search_results

    mock_results = [
        {
            'context': {'name': 'SQL Performance Analyzer', 'tags': ['sql', 'performance']},
            'relevance': 0.92
        },
        {
            'context': {'name': 'Database Query Optimizer', 'tags': ['sql', 'optimization']},
            'relevance': 0.87
        },
        {
            'context': {'name': 'SQL Best Practices', 'tags': ['sql', 'guide']},
            'relevance': 0.75
        }
    ]

    print_search_results(mock_results, "sql optimization")

    separator()

    # Analytics summary
    print("\n[6] Analytics Display:")
    from cli_output import print_analytics_summary

    mock_analytics = {
        'total_executions': 340,
        'unique_prompts': 15,
        'success_rate': 0.942,
        'avg_execution_time': 2.3,
        'avg_quality': 0.88,
        'total_cost': 12.45,
        'total_tokens': 125000
    }

    print_analytics_summary(mock_analytics)

    separator()

    # Loop execution
    print("\n[7] Loop Execution Display:")
    from cli_output import print_loop_execution

    print_loop_execution("REFINE", 3, 0.92)
    print_loop_execution("CRITIQUE", 2, 0.85)
    print_loop_execution("VERIFY", 1, 0.78)

    separator()

    # Summary
    heading("\nDemo Complete!", "bold green")
    print("\nAll rich CLI features demonstrated:")
    print("  [+] Basic messages (success, error, warning, info)")
    print("  [+] Tables (single and multi-column)")
    print("  [+] Key-value displays")
    print("  [+] Styled panels")
    print("  [+] Search results formatting")
    print("  [+] Analytics summaries")
    print("  [+] Loop execution display")
    print("\nWindows-compatible: No Unicode errors!")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
