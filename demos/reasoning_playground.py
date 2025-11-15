#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Engine Interactive Playground
========================================

Interactive demo for testing and comparing reasoning modes.

Author: Claude Code
Date: 2025-11-15
Phase: 4 (Visualization)

Usage:
    python demos/reasoning_playground.py --query "What is Thompson Sampling?"
    python demos/reasoning_playground.py --compare  # Compare all modes
    python demos/reasoning_playground.py --interactive  # Interactive mode
"""

import asyncio
import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.reasoning.engine import ReasoningEngine, reason_with_mode, auto_reason
from HoloLoom.reasoning.types import ReasoningMode, ReasoningResult
from HoloLoom.visualization.reasoning_chain import (
    render_reasoning_chain,
    render_from_reasoning_result
)
from HoloLoom.performance.reasoning_metrics import (
    ReasoningMetrics,
    track_reasoning
)
from HoloLoom.documentation.types import Query, Features, Context, MemoryShard


# ============================================================================
# Test Data Creation
# ============================================================================

def create_test_features(query_text: str) -> Features:
    """Create mock features for testing."""
    # Simple keyword extraction
    keywords = query_text.lower().split()

    return Features(
        motifs=keywords[:5],
        embeddings=[0.1] * 384,  # Mock embedding
        spectral_features={
            'graph_laplacian': [0.5, 0.3, 0.2],
            'topic_components': [0.6, 0.4]
        }
    )


def create_test_context(query_text: str) -> Context:
    """Create mock context for testing."""
    # Mock shards based on query type
    shards = []

    if "thompson" in query_text.lower():
        shards.append(MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits. "
                 "It samples from posterior distributions to balance exploration and exploitation.",
            entities=["thompson_sampling", "bayesian", "bandit"],
            created_at=time.time()
        ))
        shards.append(MemoryShard(
            id="shard_2",
            text="The algorithm maintains beta distributions (alpha, beta) for each arm. "
                 "It samples from these distributions and selects the arm with highest sample.",
            entities=["beta_distribution", "alpha", "beta"],
            created_at=time.time()
        ))

    elif "transformer" in query_text.lower():
        shards.append(MemoryShard(
            id="shard_1",
            text="Transformers are neural networks that use self-attention mechanisms. "
                 "They process sequences in parallel unlike RNNs.",
            entities=["transformer", "attention", "neural_network"],
            created_at=time.time()
        ))

    else:
        # Generic context
        shards.append(MemoryShard(
            id="shard_1",
            text=f"Relevant information about: {query_text}",
            entities=query_text.split()[:3],
            created_at=time.time()
        ))

    return Context(shards=shards)


# ============================================================================
# Playground Functions
# ============================================================================

async def run_single_query(
    query_text: str,
    mode: ReasoningMode = ReasoningMode.STANDARD,
    output_file: Optional[str] = None
) -> ReasoningResult:
    """
    Run reasoning on a single query.

    Args:
        query_text: Query text
        mode: Reasoning mode to use
        output_file: Optional HTML output file path

    Returns:
        ReasoningResult
    """
    print(f"\n{'='*60}")
    print(f"Query: {query_text}")
    print(f"Mode: {mode.value.upper()}")
    print(f"{'='*60}\n")

    # Create test data
    query = Query(text=query_text)
    features = create_test_features(query_text)
    context = create_test_context(query_text)

    # Run reasoning with metrics tracking
    metrics = ReasoningMetrics()

    with track_reasoning(mode=mode.value, metrics=metrics) as tracker:
        result = await reason_with_mode(query, features, context, mode=mode)
        tracker.set_result(result)

    # Print results
    print(result.summary())
    print()

    # Print chain
    for i, step in enumerate(result.chain):
        print(f"Step {i+1}: [{step.confidence:.2f}] {step.thought}")
        if step.evidence:
            print(f"  Evidence: {step.evidence[:100]}...")
        print()

    # Generate visualization if requested
    if output_file:
        html = render_from_reasoning_result(
            result,
            title=f"Query: {query_text}"
        )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

        print(f"✓ Visualization saved to: {output_file}\n")

    return result


async def compare_modes(
    query_text: str,
    output_file: Optional[str] = None
) -> Dict[str, ReasoningResult]:
    """
    Compare all reasoning modes on the same query.

    Args:
        query_text: Query text
        output_file: Optional HTML output file for comparison

    Returns:
        Dict mapping mode name to result
    """
    print(f"\n{'='*60}")
    print(f"COMPARING REASONING MODES")
    print(f"Query: {query_text}")
    print(f"{'='*60}\n")

    # Create test data (shared)
    query = Query(text=query_text)
    features = create_test_features(query_text)
    context = create_test_context(query_text)

    # Run all modes
    results = {}
    metrics = ReasoningMetrics()

    for mode in [ReasoningMode.FAST, ReasoningMode.STANDARD, ReasoningMode.DEEP]:
        print(f"\nRunning {mode.value.upper()} mode...")

        with track_reasoning(mode=mode.value, metrics=metrics) as tracker:
            result = await reason_with_mode(query, features, context, mode=mode)
            tracker.set_result(result)

        results[mode.value] = result

        print(f"  Steps: {len(result.chain)}")
        print(f"  Confidence: {result.total_confidence:.2f}")
        print(f"  Duration: {result.duration_ms:.1f}ms")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<15} {'Steps':<8} {'Confidence':<12} {'Duration (ms)':<15}")
    print("-" * 60)

    for mode_name, result in results.items():
        print(f"{mode_name.upper():<15} {len(result.chain):<8} "
              f"{result.total_confidence:<12.2f} {result.duration_ms:<15.1f}")

    print()

    # Get metrics summary
    summary = metrics.get_summary()
    print("\nPerformance Metrics:")
    print(f"  Avg Duration: {summary['duration_stats']['avg']:.1f}ms")
    print(f"  Avg Confidence: {summary['confidence_stats']['avg']:.2f}")

    # Generate comparison visualization
    if output_file:
        html_parts = [
            "<html><head><title>Reasoning Mode Comparison</title></head><body>",
            "<h1>Reasoning Mode Comparison</h1>",
            f"<p><strong>Query:</strong> {query_text}</p>",
            "<hr>"
        ]

        for mode_name, result in results.items():
            html_parts.append(
                render_from_reasoning_result(
                    result,
                    title=f"{mode_name.upper()} Mode"
                )
            )
            html_parts.append("<hr>")

        html_parts.append("</body></html>")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(html_parts))

        print(f"✓ Comparison saved to: {output_file}\n")

    return results


async def interactive_mode():
    """Run interactive playground."""
    print("\n" + "="*60)
    print("REASONING ENGINE INTERACTIVE PLAYGROUND")
    print("="*60)
    print("\nCommands:")
    print("  Type a query to analyze")
    print("  'compare <query>' - Compare all modes")
    print("  'mode <fast|standard|deep>' - Set default mode")
    print("  'metrics' - Show performance metrics")
    print("  'export <filename>' - Export last result to HTML")
    print("  'quit' - Exit")
    print()

    current_mode = ReasoningMode.STANDARD
    last_result = None
    metrics = ReasoningMetrics()

    while True:
        try:
            user_input = input(f"\n[{current_mode.value}]> ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\nGoodbye!\n")
                break

            elif user_input.lower() == 'metrics':
                summary = metrics.get_summary()
                print("\nPerformance Metrics:")
                print(f"  Total Operations: {summary['total_operations']}")
                print(f"  Mode Distribution: {summary['mode_distribution']}")
                print(f"  Avg Duration: {summary['duration_stats']['avg']:.1f}ms")
                print(f"  Avg Confidence: {summary['confidence_stats']['avg']:.2f}")
                print(f"  Verification Failure Rate: {summary['verification_failure_rate']:.2%}")

            elif user_input.lower().startswith('mode '):
                mode_str = user_input[5:].strip().lower()
                try:
                    current_mode = ReasoningMode(mode_str)
                    print(f"\n✓ Mode set to: {current_mode.value.upper()}")
                except ValueError:
                    print(f"\n✗ Invalid mode. Use: fast, standard, or deep")

            elif user_input.lower().startswith('compare '):
                query_text = user_input[8:].strip()
                if query_text:
                    results = await compare_modes(query_text)
                else:
                    print("\n✗ Please provide a query")

            elif user_input.lower().startswith('export '):
                filename = user_input[7:].strip()
                if last_result and filename:
                    html = render_from_reasoning_result(last_result)
                    Path(filename).write_text(html)
                    print(f"\n✓ Exported to: {filename}")
                else:
                    print("\n✗ No result to export or no filename provided")

            else:
                # Regular query
                query = Query(text=user_input)
                features = create_test_features(user_input)
                context = create_test_context(user_input)

                with track_reasoning(mode=current_mode.value, metrics=metrics) as tracker:
                    result = await reason_with_mode(
                        query, features, context, mode=current_mode
                    )
                    tracker.set_result(result)

                last_result = result

                print(f"\n{result.summary()}")
                print("\nReasoning Chain:")

                for i, step in enumerate(result.chain):
                    print(f"\n  {i+1}. [{step.confidence:.2f}] {step.thought}")
                    if step.evidence:
                        print(f"     Evidence: {step.evidence[:80]}...")

        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


# ============================================================================
# Demo Queries
# ============================================================================

DEMO_QUERIES = [
    "What is Thompson Sampling?",
    "How do transformers work?",
    "Compare Bayesian and frequentist approaches",
    "Explain the relationship between exploration and exploitation",
    "What are the tradeoffs between FAST and DEEP reasoning modes?",
]


async def run_demo_queries(output_dir: str = "demos/output"):
    """Run demonstration queries and generate visualizations."""
    print("\n" + "="*60)
    print("RUNNING DEMO QUERIES")
    print("="*60 + "\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, query_text in enumerate(DEMO_QUERIES):
        print(f"\n[{i+1}/{len(DEMO_QUERIES)}] {query_text}")

        # Run with auto mode selection
        query = Query(text=query_text)
        features = create_test_features(query_text)
        context = create_test_context(query_text)

        result = await auto_reason(query, features, context)

        # Save visualization
        filename = f"reasoning_demo_{i+1}.html"
        filepath = output_path / filename

        html = render_from_reasoning_result(
            result,
            title=f"Demo Query {i+1}: {query_text}"
        )

        filepath.write_text(html)
        print(f"  ✓ Saved to: {filepath}")
        print(f"  Mode: {result.mode.value}, Confidence: {result.total_confidence:.2f}")

    print(f"\n✓ All demos complete. Output in: {output_path}\n")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reasoning Engine Interactive Playground"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Single query to analyze"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "standard", "deep"],
        default="standard",
        help="Reasoning mode (default: standard)"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all modes on the query"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration queries"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output HTML file path"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="demos/output",
        help="Output directory for demo mode (default: demos/output)"
    )

    args = parser.parse_args()

    # Run appropriate mode
    if args.interactive:
        await interactive_mode()

    elif args.demo:
        await run_demo_queries(output_dir=args.output_dir)

    elif args.query:
        if args.compare:
            await compare_modes(args.query, output_file=args.output)
        else:
            mode = ReasoningMode(args.mode)
            await run_single_query(args.query, mode=mode, output_file=args.output)

    else:
        # No arguments - show help and run demo
        parser.print_help()
        print("\n" + "="*60)
        print("Running demonstration queries...")
        print("="*60)
        await run_demo_queries(output_dir=args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
