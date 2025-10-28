#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 DOMAIN-SPECIFIC FUSED MODE COMPARISON
=========================================
Demonstrates how different FUSED modes select different dimensions
based on domain optimization (narrative, dialogue, technical, general).

Shows:
1. Which dimensions each domain selects
2. Category distribution differences
3. Overlap and unique dimensions across domains
4. Example analysis with domain-appropriate text
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
from typing import Dict, List, Set

from HoloLoom.semantic_calculus.dimension_selector import (
    SmartDimensionSelector,
    SelectionStrategy,
)
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig


def compare_domain_selections(output_dir: Path):
    """
    Compare dimension selections across different domains.
    """
    print("\n" + "="*80)
    print("🎯 DOMAIN-SPECIFIC FUSED MODE COMPARISON")
    print("="*80)

    selector = SmartDimensionSelector()

    # Define domain configurations
    domains = {
        'NARRATIVE': {
            'strategy': SelectionStrategy.NARRATIVE,
            'domain': 'narrative',
            'color': '#e74c3c',
            'icon': '📖',
        },
        'DIALOGUE': {
            'strategy': SelectionStrategy.DIALOGUE,
            'domain': 'dialogue',
            'color': '#3498db',
            'icon': '💬',
        },
        'TECHNICAL': {
            'strategy': SelectionStrategy.BALANCED,
            'domain': 'technical',
            'color': '#2ecc71',
            'icon': '🔧',
        },
        'GENERAL': {
            'strategy': SelectionStrategy.HYBRID,
            'domain': 'general',
            'color': '#f39c12',
            'icon': '🌐',
        },
    }

    # Select dimensions for each domain
    selections = {}
    for name, config in domains.items():
        print(f"\n{config['icon']} {name} Mode")
        print("-" * 80)
        selected = selector.select(
            n_dimensions=36,
            strategy=config['strategy'],
            embed_fn=None,  # Will use balance-based selection
            domain=config['domain']
        )
        selections[name] = {
            'dimensions': selected,
            'config': config,
        }

    # Create visualization
    create_domain_comparison_visualization(selections, selector, output_dir)
    create_overlap_analysis(selections, selector, output_dir)
    create_category_comparison(selections, selector, output_dir)

    return selections


def create_domain_comparison_visualization(
    selections: Dict,
    selector: SmartDimensionSelector,
    output_dir: Path
):
    """
    Create visualization showing which dimensions each domain selects.
    """
    print("\n" + "="*80)
    print("📊 CREATING DOMAIN COMPARISON VISUALIZATION")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(selections.items()):
        ax = axes[idx]
        dims = data['dimensions']
        config = data['config']

        # Get top 15 dimensions
        dim_names = [dim.name for dim in dims[:15]]
        categories = [selector.dim_to_category[dim.name] for dim in dims[:15]]

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
        ax.set_title(f'{config["icon"]} {name} Mode - Top 15 Dimensions',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.5)
        ax.set_xticks([])

        # Add category labels
        for i, (dim_name, cat) in enumerate(zip(dim_names, categories)):
            ax.text(1.05, i, f'[{cat}]', va='center', fontsize=8, style='italic')

        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "domain_fused_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison: {output_path}")

    return fig


def create_overlap_analysis(
    selections: Dict,
    selector: SmartDimensionSelector,
    output_dir: Path
):
    """
    Analyze overlap between different domain selections.
    """
    print("\n" + "="*80)
    print("🔄 ANALYZING DIMENSION OVERLAP")
    print("="*80)

    # Convert to sets of dimension names
    dim_sets = {}
    for name, data in selections.items():
        dim_sets[name] = set([dim.name for dim in data['dimensions']])

    # Calculate pairwise overlaps
    print("\nPairwise Overlap (dimensions in common):")
    overlap_matrix = np.zeros((len(dim_sets), len(dim_sets)))
    names = list(dim_sets.keys())

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            overlap = len(dim_sets[name1] & dim_sets[name2])
            overlap_matrix[i, j] = overlap
            if i < j:  # Only print upper triangle
                print(f"  {name1} ∩ {name2}: {overlap}/36 dimensions ({overlap/36*100:.1f}%)")

    # Find unique dimensions per domain
    print("\nUnique Dimensions (not shared with any other domain):")
    for name, dims in dim_sets.items():
        # Find dimensions unique to this domain
        other_dims = set()
        for other_name, other_set in dim_sets.items():
            if other_name != name:
                other_dims |= other_set
        unique = dims - other_dims
        print(f"\n  {selections[name]['config']['icon']} {name}: {len(unique)} unique")
        if unique:
            for dim_name in sorted(list(unique))[:5]:  # Show first 5
                cat = selector.dim_to_category[dim_name]
                print(f"      - {dim_name} [{cat}]")
            if len(unique) > 5:
                print(f"      ... and {len(unique) - 5} more")

    # Create overlap heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(overlap_matrix, cmap='YlOrRd', vmin=0, vmax=36)

    # Labels
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add overlap counts
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, int(overlap_matrix[i, j]),
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title("Dimension Overlap Between Domain Modes\n(number of shared dimensions)",
                 fontsize=13, fontweight='bold', pad=20)
    fig.colorbar(im, ax=ax, label='Shared Dimensions')

    plt.tight_layout()
    output_path = output_dir / "domain_overlap_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved overlap heatmap: {output_path}")

    return fig


def create_category_comparison(
    selections: Dict,
    selector: SmartDimensionSelector,
    output_dir: Path
):
    """
    Compare category distributions across domains.
    """
    print("\n" + "="*80)
    print("📈 CATEGORY DISTRIBUTION COMPARISON")
    print("="*80)

    # Collect category distributions
    category_data = {}
    for name, data in selections.items():
        category_counts = {}
        for dim in data['dimensions']:
            cat = selector.dim_to_category[dim.name]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        category_data[name] = category_counts

    # Get all unique categories
    all_categories = set()
    for counts in category_data.values():
        all_categories.update(counts.keys())
    all_categories = sorted(list(all_categories))

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(all_categories))
    width = 0.2
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for idx, (name, counts) in enumerate(category_data.items()):
        values = [counts.get(cat, 0) for cat in all_categories]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=name,
                      color=colors[idx],
                      alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Semantic Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Dimensions Selected', fontsize=12, fontweight='bold')
    ax.set_title('Category Distribution Across Domain-Specific FUSED Modes',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "domain_category_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved category distribution: {output_path}")

    # Print summary table
    print("\nCategory Distribution Summary:")
    print("-" * 80)
    header = "Category".ljust(20)
    for name in category_data.keys():
        header += name.ljust(12)
    print(header)
    print("-" * 80)

    for cat in all_categories:
        row = cat.ljust(20)
        for name in category_data.keys():
            count = category_data[name].get(cat, 0)
            row += str(count).ljust(12)
        print(row)

    return fig


def main():
    """Run domain-specific FUSED mode comparison."""
    print("🎯 DOMAIN-SPECIFIC FUSED MODE COMPARISON DEMO")
    print("="*80)
    print("Comparing dimension selection across different domain optimizations")
    print("="*80)

    # Create output directory
    output_dir = Path("demos/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    selections = compare_domain_selections(output_dir)

    # Summary
    print("\n" + "="*80)
    print("✅ DOMAIN-SPECIFIC FUSED MODE COMPARISON COMPLETE!")
    print("="*80)
    print("\n📊 Key Insights:")
    print("   📖 NARRATIVE: Emphasizes story, archetype, and plot dimensions")
    print("   💬 DIALOGUE: Focuses on core emotional and relational dimensions")
    print("   🔧 TECHNICAL: Balanced selection across all categories")
    print("   🌐 GENERAL: Hybrid approach for versatile analysis")
    print("\n📁 Visualizations saved to:", output_dir.absolute())
    print("\n🎯 Use Cases:")
    print("   • Analyzing novels/myths → fused_narrative.yaml")
    print("   • Chat/conversation analysis → fused_dialogue.yaml")
    print("   • Documentation/code docs → fused_technical.yaml")
    print("   • General text analysis → fused.yaml (general)")
    print("="*80)


if __name__ == "__main__":
    main()
