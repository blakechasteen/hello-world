#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 FUSED 36D SMART SELECTION DEMO
==================================
Demonstrates intelligent dimension selection for FUSED mode.

Shows:
1. Different selection strategies (BALANCED, NARRATIVE, HYBRID)
2. Category distribution for each strategy
3. Visualization of which dimensions are chosen
4. Comparison with simple first-N selection

The smart selector chooses 36 dimensions from 244 based on:
- Category balance (ensure representation)
- Discriminative power (high variance)
- Domain relevance (narrative/dialogue/technical)
"""

# Force UTF-8 encoding for Windows console
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

from HoloLoom.semantic_calculus.dimension_selector import (
    SmartDimensionSelector,
    SelectionStrategy,
)
from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS
from HoloLoom.embedding.spectral import create_embedder


def visualize_selection_comparison(output_dir: Path):
    """
    Compare different selection strategies visually.
    """
    print("\n" + "="*80)
    print("🎯 COMPARING SELECTION STRATEGIES")
    print("="*80)

    # Create selector
    selector = SmartDimensionSelector()

    # Test all strategies
    strategies = [
        SelectionStrategy.BALANCED,
        SelectionStrategy.NARRATIVE,
        SelectionStrategy.HYBRID,
    ]

    results = {}
    for strategy in strategies:
        print(f"\n--- {strategy.value.upper()} Strategy ---")
        selected = selector.select(
            n_dimensions=36,
            strategy=strategy,
            embed_fn=None  # Will use balance-only for BALANCED/NARRATIVE
        )
        results[strategy.value] = selected

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (strategy_name, dims) in enumerate(results.items()):
        ax = axes[idx]

        # Count by category
        category_counts = {}
        for dim in dims:
            cat = selector.dim_to_category[dim.name]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Sort by count
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        categories = [cat for cat, _ in sorted_cats]
        counts = [count for _, count in sorted_cats]

        # Colors
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                  '#9b59b6', '#1abc9c', '#95a5a6', '#34495e']

        # Bar chart
        bars = ax.barh(categories, counts, color=colors[:len(categories)], alpha=0.8)

        # Labels
        ax.set_xlabel('Number of Dimensions', fontsize=11)
        ax.set_title(f'{strategy_name.upper()}\n(36 dimensions)',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(0, max(counts) + 2)

        # Add counts on bars
        for i, (cat, count) in enumerate(zip(categories, counts)):
            ax.text(count + 0.3, i, str(count),
                   va='center', fontweight='bold', fontsize=10)

        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "fused_36d_strategy_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison: {output_path}")

    return fig


def show_top_dimensions_per_strategy(output_dir: Path):
    """
    Show which specific dimensions are chosen by each strategy.
    """
    print("\n" + "="*80)
    print("📊 TOP DIMENSIONS BY STRATEGY")
    print("="*80)

    selector = SmartDimensionSelector()

    strategies_to_test = {
        'BALANCED': SelectionStrategy.BALANCED,
        'NARRATIVE': SelectionStrategy.NARRATIVE,
        'HYBRID': SelectionStrategy.HYBRID,
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for idx, (name, strategy) in enumerate(strategies_to_test.items()):
        ax = axes[idx]

        # Select dimensions
        selected = selector.select(
            n_dimensions=36,
            strategy=strategy,
            embed_fn=None
        )

        # Get dimension names and categories
        dim_names = [dim.name for dim in selected[:20]]  # Top 20
        categories = [selector.dim_to_category[dim.name] for dim in selected[:20]]

        # Color by category
        category_colors = {
            'Core': '#3498db',
            'Narrative': '#e74c3c',
            'Emotional': '#9b59b6',
            'Archetypal': '#f39c12',
            'Philosophical': '#2ecc71',
            'Theme': '#1abc9c',
            'Plot': '#e67e22',
            'Other': '#95a5a6',
        }
        colors = [category_colors.get(cat, '#95a5a6') for cat in categories]

        # Horizontal bar chart
        y_pos = np.arange(len(dim_names))
        ax.barh(y_pos, [1] * len(dim_names), color=colors, alpha=0.7)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(dim_names, fontsize=9)
        ax.set_xlabel('Selected', fontsize=10)
        ax.set_title(f'{name} Strategy - Top 20 Dimensions',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.5)
        ax.set_xticks([])

        # Add category labels
        for i, (dim_name, cat) in enumerate(zip(dim_names, categories)):
            ax.text(1.05, i, f'[{cat}]', va='center', fontsize=8, style='italic')

        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "fused_36d_top_dimensions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved top dimensions: {output_path}")

    return fig


def create_mode_comparison_chart(output_dir: Path):
    """
    Compare BARE (8D) vs FAST (16D) vs FUSED (36D) vs RESEARCH (244D).
    """
    print("\n" + "="*80)
    print("📈 MODE COMPARISON: BARE → FAST → FUSED → RESEARCH")
    print("="*80)

    modes = {
        'BARE': {'dims': 8, 'latency': 50, 'quality': 3, 'detail': 2},
        'FAST': {'dims': 16, 'latency': 200, 'quality': 6, 'detail': 5},
        'FUSED': {'dims': 36, 'latency': 1000, 'quality': 8, 'detail': 8},
        'RESEARCH': {'dims': 244, 'latency': 5000, 'quality': 10, 'detail': 10},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Dimensions
    ax = axes[0, 0]
    mode_names = list(modes.keys())
    dims = [modes[m]['dims'] for m in mode_names]
    colors = ['#95a5a6', '#3498db', '#f39c12', '#e74c3c']
    ax.bar(mode_names, dims, color=colors, alpha=0.8)
    ax.set_ylabel('Number of Dimensions', fontsize=11, fontweight='bold')
    ax.set_title('Semantic Dimensions', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (mode, dim) in enumerate(zip(mode_names, dims)):
        ax.text(i, dim + 5, str(dim), ha='center', fontweight='bold')

    # 2. Latency
    ax = axes[0, 1]
    latencies = [modes[m]['latency'] for m in mode_names]
    ax.bar(mode_names, latencies, color=colors, alpha=0.8)
    ax.set_ylabel('Target Latency (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Response Speed', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    for i, (mode, lat) in enumerate(zip(mode_names, latencies)):
        ax.text(i, lat * 1.3, f'{lat}ms', ha='center', fontsize=9)

    # 3. Quality
    ax = axes[1, 0]
    qualities = [modes[m]['quality'] for m in mode_names]
    ax.bar(mode_names, qualities, color=colors, alpha=0.8)
    ax.set_ylabel('Analysis Quality (1-10)', fontsize=11, fontweight='bold')
    ax.set_title('Insight Quality', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 11)
    ax.grid(axis='y', alpha=0.3)
    for i, (mode, qual) in enumerate(zip(mode_names, qualities)):
        ax.text(i, qual + 0.3, str(qual), ha='center', fontweight='bold')

    # 4. Detail
    ax = axes[1, 1]
    details = [modes[m]['detail'] for m in mode_names]
    ax.bar(mode_names, details, color=colors, alpha=0.8)
    ax.set_ylabel('Narrative Detail (1-10)', fontsize=11, fontweight='bold')
    ax.set_title('Literary Analysis Depth', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 11)
    ax.grid(axis='y', alpha=0.3)
    for i, (mode, det) in enumerate(zip(mode_names, details)):
        ax.text(i, det + 0.3, str(det), ha='center', fontweight='bold')

    plt.suptitle('Pattern Card Mode Comparison\n(8D → 16D → 36D → 244D)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = output_dir / "mode_comparison_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved mode comparison: {output_path}")

    return fig


def main():
    """Run full FUSED 36D selection demo."""
    print("🎯 FUSED 36D SMART SELECTION DEMO")
    print("="*80)
    print("Demonstrating intelligent dimension selection for middle-range mode")
    print("="*80)

    # Create output directory
    output_dir = Path("demos/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run visualizations
    visualize_selection_comparison(output_dir)
    show_top_dimensions_per_strategy(output_dir)
    create_mode_comparison_chart(output_dir)

    # Summary
    print("\n" + "="*80)
    print("✅ FUSED 36D SELECTION DEMO COMPLETE!")
    print("="*80)
    print("\n📊 Key Insights:")
    print("   • BALANCED: Equal representation from all categories")
    print("   • NARRATIVE: Focus on story/archetype/plot dimensions")
    print("   • HYBRID: Optimal blend (default for FUSED mode)")
    print("\n📁 Visualizations saved to:", output_dir.absolute())
    print("\n🎯 FUSED Mode (36D) provides:")
    print("   ✓ 2.25x more dimensions than FAST (16D)")
    print("   ✓ Smart selection from 244D space")
    print("   ✓ Balanced category representation")
    print("   ✓ ~1s latency (vs 50ms FAST, 5s RESEARCH)")
    print("   ✓ Rich narrative analysis without full 244D overhead")
    print("="*80)


if __name__ == "__main__":
    main()
