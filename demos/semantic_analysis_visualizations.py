#!/usr/bin/env python3
"""
Semantic Analysis Visualizations Demo
======================================

Creates rich visualizations of semantic calculus analysis:
1. Semantic trajectory in 3D space
2. Velocity, acceleration, curvature plots
3. Semantic dimensions analysis
4. Ethical evaluation heatmap
5. Pattern Card configuration comparison

Usage:
    python demos/semantic_analysis_visualizations.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import asyncio

# HoloLoom imports
from HoloLoom.semantic_calculus.flow_calculus import SemanticFlowCalculus, analyze_text_flow
from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum, visualize_semantic_spectrum
from HoloLoom.semantic_calculus.ethics import EthicalSemanticPolicy, COMPASSIONATE_COMMUNICATION
from HoloLoom.embedding.spectral import create_embedder
from HoloLoom.loom.card_loader import PatternCard

# Create output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


def create_semantic_trajectory_3d(trajectory, title="Semantic Trajectory"):
    """Create 3D visualization of semantic trajectory."""
    print(f"\n[1/5] Creating 3D trajectory visualization...")

    # Use PCA to reduce to 3D
    from sklearn.decomposition import PCA

    positions = trajectory.positions
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(positions)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 3D trajectory
    ax1 = fig.add_subplot(gs[0, :], projection='3d')

    # Color by speed
    speeds = np.array([s.speed for s in trajectory.states])
    scatter = ax1.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=speeds,
        cmap='plasma',
        s=100,
        alpha=0.6
    )

    # Draw path
    ax1.plot(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        'k-',
        alpha=0.3,
        linewidth=1
    )

    # Label points
    for i, (state, coord) in enumerate(zip(trajectory.states, coords_3d)):
        if i % 3 == 0:  # Label every 3rd word to avoid clutter
            ax1.text(coord[0], coord[1], coord[2], state.word, fontsize=8)

    ax1.set_title(f'{title}\n(colored by semantic velocity)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    plt.colorbar(scatter, ax=ax1, label='Speed')

    # Velocity over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(speeds, 'o-', color='#FF6B6B', linewidth=2, markersize=6)
    ax2.set_title('Semantic Velocity', fontweight='bold')
    ax2.set_xlabel('Word Position')
    ax2.set_ylabel('Speed')
    ax2.grid(True, alpha=0.3)

    # Acceleration over time
    ax3 = fig.add_subplot(gs[1, 1])
    accels = np.array([s.acceleration_magnitude for s in trajectory.states])
    ax3.plot(accels, 'o-', color='#4ECDC4', linewidth=2, markersize=6)
    ax3.set_title('Semantic Acceleration', fontweight='bold')
    ax3.set_xlabel('Word Position')
    ax3.set_ylabel('Acceleration')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = output_dir / "semantic_trajectory_3d.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {filename}")
    plt.close()

    return filename


def create_flow_metrics_plot(trajectory, title="Semantic Flow Metrics"):
    """Create detailed flow metrics visualization."""
    print(f"\n[2/5] Creating flow metrics visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Velocity
    ax = axes[0, 0]
    speeds = [s.speed for s in trajectory.states]
    words = [s.word for s in trajectory.states]

    ax.bar(range(len(speeds)), speeds, color='#FF6B6B', alpha=0.7)
    ax.set_title('Velocity (Rate of Meaning Change)', fontweight='bold')
    ax.set_xlabel('Word Position')
    ax.set_ylabel('Speed')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
    ax.axhline(np.mean(speeds), color='red', linestyle='--', label=f'Mean: {np.mean(speeds):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Acceleration
    ax = axes[0, 1]
    accels = [s.acceleration_magnitude for s in trajectory.states]

    ax.bar(range(len(accels)), accels, color='#4ECDC4', alpha=0.7)
    ax.set_title('Acceleration (Change in Direction)', fontweight='bold')
    ax.set_xlabel('Word Position')
    ax.set_ylabel('Acceleration')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
    ax.axhline(np.mean(accels), color='blue', linestyle='--', label=f'Mean: {np.mean(accels):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Curvature
    ax = axes[1, 0]
    curvatures = [trajectory.curvature(i) for i in range(len(trajectory.states))]

    ax.bar(range(len(curvatures)), curvatures, color='#95E1D3', alpha=0.7)
    ax.set_title('Curvature (Semantic Bending)', fontweight='bold')
    ax.set_xlabel('Word Position')
    ax.set_ylabel('Curvature')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
    ax.axhline(np.mean(curvatures), color='green', linestyle='--', label=f'Mean: {np.mean(curvatures):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy (if available)
    ax = axes[1, 1]
    energies = [s.kinetic if s.kinetic is not None else 0 for s in trajectory.states]

    ax.bar(range(len(energies)), energies, color='#F38181', alpha=0.7)
    ax.set_title('Kinetic Energy (Semantic Momentum)', fontweight='bold')
    ax.set_xlabel('Word Position')
    ax.set_ylabel('Energy')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
    if any(energies):
        ax.axhline(np.mean(energies), color='darkred', linestyle='--', label=f'Mean: {np.mean(energies):.3f}')
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = output_dir / "flow_metrics.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {filename}")
    plt.close()

    return filename


def create_semantic_dimensions_heatmap(trajectory, spectrum, title="Semantic Dimensions Analysis"):
    """Create heatmap of semantic dimensions over time."""
    print(f"\n[3/5] Creating semantic dimensions heatmap...")

    # Project trajectory onto dimensions
    projections = spectrum.project_trajectory(trajectory.positions)

    # Create matrix: dimensions x time
    dim_names = [dim.name for dim in spectrum.dimensions]
    matrix = np.array([projections[name] for name in dim_names])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Heatmap
    im = ax1.imshow(matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax1.set_yticks(range(len(dim_names)))
    ax1.set_yticklabels(dim_names, fontsize=10)
    ax1.set_xticks(range(len(trajectory.words)))
    ax1.set_xticklabels(trajectory.words, rotation=45, ha='right', fontsize=8)
    ax1.set_title('Semantic Dimension Activation Over Time', fontweight='bold')
    ax1.set_xlabel('Word')
    ax1.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax1, label='Activation')

    # Dominant dimensions
    semantic_forces = spectrum.analyze_semantic_forces(trajectory.positions, dt=1.0)
    dominant = semantic_forces['dominant_velocity'][:8]  # Top 8

    dim_names_dom = [name for name, _ in dominant]
    velocities = [vel for _, vel in dominant]

    colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in velocities]
    ax2.barh(range(len(dim_names_dom)), velocities, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(dim_names_dom)))
    ax2.set_yticklabels(dim_names_dom, fontsize=10)
    ax2.set_xlabel('Velocity (rate of change)')
    ax2.set_title('Dominant Semantic Dimensions', fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')

    # Legend
    red_patch = mpatches.Patch(color='#FF6B6B', label='Increasing')
    blue_patch = mpatches.Patch(color='#4ECDC4', label='Decreasing')
    ax2.legend(handles=[red_patch, blue_patch])

    plt.tight_layout()
    filename = output_dir / "semantic_dimensions.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {filename}")
    plt.close()

    return filename


def create_ethical_analysis_chart(ethical_analysis, title="Ethical Analysis"):
    """Create ethical analysis visualization."""
    print(f"\n[4/5] Creating ethical analysis chart...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Virtue scores over trajectory
    ax1.plot(ethical_analysis['virtue_scores'], 'o-', color='#95E1D3', linewidth=2, markersize=8, label='Virtue')
    ax1.plot(ethical_analysis['manipulation_scores'], 'o-', color='#F38181', linewidth=2, markersize=8, label='Manipulation')
    ax1.set_title('Ethical Metrics Over Trajectory', fontweight='bold')
    ax1.set_xlabel('Position in Text')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add thresholds
    ax1.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold')

    # Summary metrics
    metrics = {
        'Mean Virtue': ethical_analysis['mean_virtue'],
        'Max Manipulation': ethical_analysis['max_manipulation'],
        'Ethical': 1.0 if ethical_analysis['is_ethical'] else 0.0
    }

    colors = ['#95E1D3', '#F38181', '#4ECDC4']
    ax2.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7)
    ax2.set_title('Overall Ethical Assessment', fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add annotations
    for i, (key, value) in enumerate(metrics.items()):
        ax2.text(i, value + 0.05, f'{value:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    filename = output_dir / "ethical_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {filename}")
    plt.close()

    return filename


def create_pattern_card_comparison():
    """Create pattern card configuration comparison chart."""
    print(f"\n[5/5] Creating pattern card comparison...")

    # Load cards
    cards = {
        'bare': PatternCard.load('bare'),
        'fast': PatternCard.load('fast'),
        'fused': PatternCard.load('fused')
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Pattern Card Configuration Comparison', fontsize=16, fontweight='bold')

    # Semantic dimensions
    ax = axes[0, 0]
    dims = []
    for name, card in cards.items():
        sem = card.math_capabilities.semantic_calculus
        if sem.get('enabled'):
            dims.append(sem['config']['dimensions'])
        else:
            dims.append(0)

    ax.bar(cards.keys(), dims, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    ax.set_title('Semantic Dimensions', fontweight='bold')
    ax.set_ylabel('Dimensions')
    ax.grid(True, alpha=0.3, axis='y')

    # Embedding scales
    ax = axes[0, 1]
    scales_count = [
        len(card.math_capabilities.spectral_embedding.get('scales', []))
        for card in cards.values()
    ]

    ax.bar(cards.keys(), scales_count, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    ax.set_title('Embedding Scales', fontweight='bold')
    ax.set_ylabel('Number of Scales')
    ax.grid(True, alpha=0.3, axis='y')

    # Tools enabled
    ax = axes[1, 0]
    tools_count = [len(card.tools_config.enabled) for card in cards.values()]

    ax.bar(cards.keys(), tools_count, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    ax.set_title('Tools Enabled', fontweight='bold')
    ax.set_ylabel('Number of Tools')
    ax.grid(True, alpha=0.3, axis='y')

    # Target latency
    ax = axes[1, 1]
    latencies = [card.performance_profile.target_latency_ms for card in cards.values()]

    ax.bar(cards.keys(), latencies, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7)
    ax.set_title('Target Latency', fontweight='bold')
    ax.set_ylabel('Milliseconds')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = output_dir / "pattern_cards_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {filename}")
    plt.close()

    return filename


async def main():
    """Run all visualizations."""
    print("=" * 70)
    print("SEMANTIC ANALYSIS VISUALIZATIONS DEMO")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}")

    # Example text for analysis
    text = ("Thompson Sampling balances exploration and exploitation using Bayesian inference "
            "The algorithm maintains posterior distributions over reward parameters "
            "By sampling from these distributions it naturally explores uncertain options")

    print(f"\nAnalyzing text:")
    print(f"  '{text[:80]}...'")
    print(f"\n  Words: {len(text.split())}")

    # Create embedder
    print("\n[Setup] Creating embedding model...")
    embed_model = create_embedder(sizes=[384])
    embed_fn = lambda words: embed_model.encode(words)

    # Compute trajectory
    print("[Setup] Computing semantic trajectory...")
    calculus = SemanticFlowCalculus(embed_fn, dt=1.0, enable_cache=True)
    trajectory = calculus.compute_trajectory(text.split())

    # Create spectrum
    print("[Setup] Creating semantic spectrum...")
    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    # Ethical analysis
    print("[Setup] Running ethical analysis...")
    dim_names = [dim.name for dim in spectrum.dimensions]
    policy = EthicalSemanticPolicy(COMPASSIONATE_COMMUNICATION, dim_names)
    projections = spectrum.project_trajectory(trajectory.positions)
    q_semantic = np.column_stack([projections[name] for name in dim_names])
    ethical_analysis = policy.analyze_conversation_ethics(q_semantic)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Create visualizations
    viz1 = create_semantic_trajectory_3d(trajectory)
    viz2 = create_flow_metrics_plot(trajectory)
    viz3 = create_semantic_dimensions_heatmap(trajectory, spectrum)
    viz4 = create_ethical_analysis_chart(ethical_analysis)
    viz5 = create_pattern_card_comparison()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✓ Generated 5 visualizations in: {output_dir.absolute()}")
    print(f"\n  1. {viz1.name} - 3D semantic trajectory")
    print(f"  2. {viz2.name} - Flow metrics (velocity/accel/curvature)")
    print(f"  3. {viz3.name} - Semantic dimensions heatmap")
    print(f"  4. {viz4.name} - Ethical analysis")
    print(f"  5. {viz5.name} - Pattern cards comparison")

    print(f"\nKey metrics:")
    print(f"  Average velocity: {np.mean([s.speed for s in trajectory.states]):.4f}")
    print(f"  Average acceleration: {np.mean([s.acceleration_magnitude for s in trajectory.states]):.4f}")
    print(f"  Total distance: {trajectory.total_distance():.4f}")
    print(f"  Mean virtue score: {ethical_analysis['mean_virtue']:.3f}")
    print(f"  Max manipulation: {ethical_analysis['max_manipulation']:.3f}")

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
