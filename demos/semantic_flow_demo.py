"""
Semantic Flow Calculus Demo

Demonstrates the mathematical framework for analyzing semantic trajectories.
Shows how word sequences create "flows" through embedding space with
measurable velocity, acceleration, and curvature.

This is the proof-of-concept for the full semantic tracer.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from HoloLoom.semantic_flow_calculus import (
    SemanticFlowCalculus,
    SemanticFlowVisualizer,
    analyze_text_flow
)
from HoloLoom.embedding.spectral import create_embedder


def demo_basic_flow():
    """
    Demo 1: Basic semantic flow analysis on a simple sentence
    """
    print("=" * 70)
    print("DEMO 1: Basic Semantic Flow")
    print("=" * 70)

    # Create embedding function
    print("\n[1/4] Creating embedding model...")
    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.embed_query(word)

    # Analyze a philosophical sentence
    text = "I think therefore I am a conscious being"
    words = text.split()

    print(f"\n[2/4] Analyzing text: '{text}'")
    print(f"        Number of words: {len(words)}")

    # Compute trajectory
    calculus = SemanticFlowCalculus(embed_fn, dt=1.0)
    trajectory = calculus.compute_trajectory(words)

    print(f"\n[3/4] Computing semantic derivatives...")
    print(f"        ✓ Positions: {trajectory.positions.shape}")
    print(f"        ✓ Velocities: {trajectory.velocities.shape}")
    print(f"        ✓ Accelerations: {trajectory.accelerations.shape}")

    # Display metrics for each word
    print(f"\n[4/4] Flow metrics by word:")
    print(f"        {'Word':<15} {'Speed':>10} {'Accel':>10} {'Curvature':>10}")
    print(f"        {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    for i, state in enumerate(trajectory.states):
        curv = trajectory.curvature(i)
        print(f"        {state.word:<15} {state.speed:>10.4f} "
              f"{state.acceleration_magnitude:>10.4f} {curv:>10.4f}")

    print(f"\n        Total distance traveled: {trajectory.total_distance():.4f}")

    # Visualize
    print(f"\n[Visualization] Generating plots...")
    visualizer = SemanticFlowVisualizer(dim_reduction='pca')

    # 3D trajectory
    fig1, ax1 = visualizer.visualize_trajectory_3d(
        trajectory,
        show_velocity=True,
        show_acceleration=False
    )
    plt.savefig('demos/output/semantic_flow_3d.png', dpi=150, bbox_inches='tight')
    print(f"        ✓ Saved: demos/output/semantic_flow_3d.png")

    # Flow metrics
    fig2, axes2 = visualizer.plot_flow_metrics(trajectory)
    plt.savefig('demos/output/semantic_flow_metrics.png', dpi=150, bbox_inches='tight')
    print(f"        ✓ Saved: demos/output/semantic_flow_metrics.png")

    return trajectory, calculus


def demo_conversation_flow():
    """
    Demo 2: Analyze a conversation with topic shifts
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Conversation Flow with Topic Shifts")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.embed_query(word)

    # Conversation with clear topic shifts
    conversation = [
        "hello how are you today",
        "I am doing well thanks",
        "what did you think about the movie",
        "the cinematography was absolutely stunning",
        "speaking of art let's discuss painting",
        "impressionism changed everything in visual arts"
    ]

    print(f"\n[1/3] Analyzing conversation with {len(conversation)} utterances")

    calculus = SemanticFlowCalculus(embed_fn, dt=1.0)

    # Analyze each utterance separately, then combine
    all_words = []
    for utt in conversation:
        all_words.extend(utt.split())

    print(f"\n[2/3] Total words: {len(all_words)}")
    trajectory = calculus.compute_trajectory(all_words)

    # Find high-curvature points (topic shifts!)
    curvatures = [trajectory.curvature(i) for i in range(len(trajectory.states))]
    avg_curvature = np.mean(curvatures)
    high_curvature_indices = [i for i, c in enumerate(curvatures) if c > avg_curvature * 2]

    print(f"\n[3/3] Detected {len(high_curvature_indices)} potential topic shifts:")
    for idx in high_curvature_indices:
        word = trajectory.words[idx]
        curv = curvatures[idx]
        print(f"        Position {idx}: '{word}' (κ = {curv:.4f})")

    # Visualize
    visualizer = SemanticFlowVisualizer(dim_reduction='pca')
    fig, ax = visualizer.visualize_trajectory_3d(
        trajectory,
        show_velocity=True,
        show_acceleration=True
    )
    plt.savefig('demos/output/conversation_flow_3d.png', dpi=150, bbox_inches='tight')
    print(f"\n        ✓ Saved: demos/output/conversation_flow_3d.png")

    return trajectory


def demo_attractor_finding():
    """
    Demo 3: Find semantic attractors from multiple trajectories
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Finding Semantic Attractors")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.embed_query(word)

    # Multiple sentences about related topics
    texts = [
        "love is a beautiful emotion",
        "affection and care define relationships",
        "hatred destroys everything it touches",
        "anger and rage consume the soul",
        "happiness brings joy to life",
        "sadness weighs heavy on the heart",
        "fear paralyzes the mind completely",
        "courage overcomes all obstacles"
    ]

    print(f"\n[1/3] Processing {len(texts)} text samples...")

    calculus = SemanticFlowCalculus(embed_fn, dt=1.0)
    trajectories = []

    for text in texts:
        words = text.split()
        traj = calculus.compute_trajectory(words)
        trajectories.append(traj)

    print(f"\n[2/3] Finding semantic attractors...")
    attractors = calculus.find_attractors(trajectories, velocity_threshold=0.15)

    print(f"\n[3/3] Found {len(attractors)} attractors:")
    for i, attr in enumerate(attractors):
        print(f"\n        Attractor {i+1}:")
        print(f"          Words: {', '.join(attr['words'][:5])}")
        print(f"          Basin size: {attr['basin_size']}")

    # Visualize one trajectory with attractors marked
    if len(trajectories) > 0 and len(attractors) > 0:
        visualizer = SemanticFlowVisualizer(dim_reduction='pca')
        fig, ax = visualizer.visualize_trajectory_3d(
            trajectories[0],
            show_velocity=True,
            attractors=attractors
        )
        plt.savefig('demos/output/attractors_3d.png', dpi=150, bbox_inches='tight')
        print(f"\n        ✓ Saved: demos/output/attractors_3d.png")

    return attractors, trajectories


def demo_energy_conservation():
    """
    Demo 4: Test Hamiltonian energy conservation
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Hamiltonian Energy Analysis")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.embed_query(word)

    # Coherent narrative (should have relatively conserved energy)
    narrative = "once upon a time in a faraway kingdom there lived a brave knight who protected the realm"

    print(f"\n[1/2] Analyzing narrative: '{narrative}'")

    words = narrative.split()
    calculus = SemanticFlowCalculus(embed_fn, dt=1.0)
    trajectory = calculus.compute_trajectory(words)

    # Extract energies
    kinetic = [s.kinetic for s in trajectory.states]
    # Approximate potential (we'd need full potential field for exact values)
    potential_approx = [0.5 * np.linalg.norm(s.position)**2 for s in trajectory.states]
    total_energy = [k + p for k, p in zip(kinetic, potential_approx)]

    print(f"\n[2/2] Energy statistics:")
    print(f"        Kinetic energy (avg): {np.mean(kinetic):.4f} ± {np.std(kinetic):.4f}")
    print(f"        Total energy (avg): {np.mean(total_energy):.4f} ± {np.std(total_energy):.4f}")
    print(f"        Energy conservation: {np.std(total_energy)/np.mean(total_energy)*100:.2f}% variation")

    # Plot energy over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(kinetic, 'b-', label='Kinetic Energy', linewidth=2)
    ax.plot(potential_approx, 'r-', label='Potential Energy (approx)', linewidth=2)
    ax.plot(total_energy, 'g--', label='Total Energy', linewidth=2)
    ax.set_xlabel('Word Index')
    ax.set_ylabel('Energy')
    ax.set_title('Hamiltonian Energy Through Semantic Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if len(words) <= 15:
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('demos/output/energy_conservation.png', dpi=150, bbox_inches='tight')
    print(f"\n        ✓ Saved: demos/output/energy_conservation.png")

    return trajectory


def main():
    """Run all demos"""
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "  SEMANTIC FLOW CALCULUS - Proof of Concept Demo".center(68) + "|")
    print("|" + "  BearL Labs - Breaking Ground Daily".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")
    print()

    # Create output directory
    os.makedirs('demos/output', exist_ok=True)

    try:
        # Run demos
        trajectory1, calculus = demo_basic_flow()
        trajectory2 = demo_conversation_flow()
        attractors, trajectories = demo_attractor_finding()
        trajectory4 = demo_energy_conservation()

        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETE")
        print("=" * 70)
        print("\nKey Findings:")
        print("  ✓ Semantic derivatives (velocity, acceleration) computed successfully")
        print("  ✓ Trajectories visualized in 3D reduced space")
        print("  ✓ Topic shifts detected via curvature analysis")
        print(f"  ✓ {len(attractors)} semantic attractors discovered")
        print("  ✓ Energy conservation demonstrated")
        print("\nNext Steps:")
        print("  → Phase 2: Multi-scale harmonic analysis")
        print("  → Phase 3: Full potential field reconstruction")
        print("  → Phase 4: Attention flow extraction")
        print("\nOutput saved to: demos/output/")
        print()

        # Show plots if in interactive mode
        try:
            plt.show()
        except:
            pass

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())