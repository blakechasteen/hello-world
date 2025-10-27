"""
Semantic Spectrum Demo

Shows how conversations move along interpretable semantic dimensions.
This is the "special sauce" - instead of opaque embeddings, we see:
  "Warmth increased, Formality decreased, Certainty fluctuated"

Demonstrates conjugate pairs and semantic forces acting on language.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from HoloLoom.semantic_calculus import (
    SemanticFlowCalculus,
    SemanticSpectrum,
    visualize_semantic_spectrum,
    print_spectrum_summary,
    STANDARD_DIMENSIONS
)
from HoloLoom.embedding.spectral import create_embedder


def demo_dimension_learning():
    """
    Demo 1: Learn semantic dimension axes from exemplars
    """
    print("=" * 70)
    print("DEMO 1: Learning Semantic Dimensions")
    print("=" * 70)

    print("\n[1/2] Creating embedding model...")
    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.encode([word])[0]

    print("\n[2/2] Learning dimension axes from exemplars...")
    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    # Test projection on sample words
    test_words = ["happy", "sad", "professional", "casual", "intense", "calm"]

    print(f"\n{'Word':<15} {'Warmth':>10} {'Formality':>10} {'Intensity':>10} {'Certainty':>10}")
    print("-" * 65)

    for word in test_words:
        vec = embed_fn(word)
        proj = spectrum.project_vector(vec)
        print(f"{word:<15} {proj['Warmth']:>10.3f} {proj['Formality']:>10.3f} "
              f"{proj['Intensity']:>10.3f} {proj['Certainty']:>10.3f}")

    return spectrum, embed_fn


def demo_conversation_spectrum():
    """
    Demo 2: Analyze conversation along semantic dimensions
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Conversation Semantic Spectrum")
    print("=" * 70)

    # Setup
    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.encode([word])[0]

    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    # Analyze a conversation that shifts tone
    conversation = "Hey there friend how are you doing today I hope everything is wonderful"
    words = conversation.split()

    print(f"\n[1/3] Analyzing: '{conversation}'")
    print(f"        Words: {len(words)}")

    # Compute trajectory
    calculus = SemanticFlowCalculus(embed_fn, dt=1.0)
    trajectory = calculus.compute_trajectory(words)

    print(f"\n[2/3] Projecting onto semantic dimensions...")
    analysis = spectrum.analyze_semantic_forces(trajectory.positions, dt=1.0)

    print_spectrum_summary(analysis, words)

    print(f"\n[3/3] Generating spectrum visualization...")
    fig, axes = visualize_semantic_spectrum(
        analysis, words, top_k=8,
        save_path='demos/output/conversation_spectrum.png'
    )

    return analysis


def demo_emotional_arc():
    """
    Demo 3: Track emotional arc through a narrative
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Emotional Arc Analysis")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.encode([word])[0]

    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    # A narrative with clear emotional progression
    narrative = (
        "once upon a time there lived a happy peaceful prince "
        "but suddenly disaster struck and chaos erupted everywhere "
        "the brave courageous hero fought fiercely against evil "
        "finally peace and joy returned to the grateful kingdom"
    )
    words = narrative.split()

    print(f"\n[1/2] Analyzing emotional arc: {len(words)} words")

    calculus = SemanticFlowCalculus(embed_fn, dt=1.0)
    trajectory = calculus.compute_trajectory(words)
    analysis = spectrum.analyze_semantic_forces(trajectory.positions, dt=1.0)

    print(f"\n[2/2] Emotional dimensions:")

    # Focus on emotional dimensions
    emotional_dims = ['Valence', 'Arousal', 'Intensity', 'Warmth']

    for dim in emotional_dims:
        proj = analysis['projections'][dim]
        start, middle, end = proj[0], proj[len(proj)//2], proj[-1]
        print(f"  {dim:<12}: {start:>6.2f} -> {middle:>6.2f} -> {end:>6.2f}")

    # Visualize just emotional dimensions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, dim in enumerate(emotional_dims):
        ax = axes[i]
        proj = analysis['projections'][dim]
        vel = analysis['velocities'][dim]

        # Plot projection and velocity
        ax2 = ax.twinx()
        l1 = ax.plot(proj, 'b-', linewidth=2, label='Position')
        l2 = ax2.plot(vel, 'r--', linewidth=2, label='Velocity', alpha=0.7)

        ax.set_ylabel(f'{dim} Position', color='b')
        ax2.set_ylabel(f'{dim} Velocity', color='r')
        ax.set_title(dim, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Combine legends
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')

    plt.tight_layout()
    plt.savefig('demos/output/emotional_arc.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: demos/output/emotional_arc.png")

    return analysis


def demo_rhetorical_shift():
    """
    Demo 4: Detect rhetorical strategy shifts
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Rhetorical Strategy Detection")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.encode([word])[0]

    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    # Two speaking styles: formal argument -> casual persuasion
    speech = (
        "we must consider the evidence carefully and analyze objectively "
        "hey look this is really simple just think about it for a second"
    )
    words = speech.split()

    print(f"\n[1/2] Analyzing rhetorical shift...")

    calculus = SemanticFlowCalculus(embed_fn, dt=1.0)
    trajectory = calculus.compute_trajectory(words)
    analysis = spectrum.analyze_semantic_forces(trajectory.positions, dt=1.0)

    # Focus on rhetorical dimensions
    rhetorical_dims = ['Formality', 'Directness', 'Certainty', 'Complexity']

    print(f"\n[2/2] Rhetorical dimensions (first half vs second half):")
    print(f"        {'Dimension':<15} {'First Half':>12} {'Second Half':>12} {'Change':>12}")
    print("        " + "-" * 60)

    for dim in rhetorical_dims:
        proj = analysis['projections'][dim]
        midpoint = len(proj) // 2

        first_half = np.mean(proj[:midpoint])
        second_half = np.mean(proj[midpoint:])
        change = second_half - first_half

        arrow = ">>>" if abs(change) > 0.1 else "->"
        print(f"        {dim:<15} {first_half:>12.3f} {arrow:>3} {second_half:>10.3f}  ({change:+.3f})")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))

    for dim in rhetorical_dims:
        ax.plot(analysis['projections'][dim], label=dim, linewidth=2.5, alpha=0.8)

    ax.axvline(x=len(words)//2, color='red', linestyle='--',
               linewidth=2, label='Style Shift', alpha=0.7)
    ax.set_xlabel('Word Index', fontsize=12)
    ax.set_ylabel('Dimension Value', fontsize=12)
    ax.set_title('Rhetorical Strategy Shift', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if len(words) <= 20:
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('demos/output/rhetorical_shift.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: demos/output/rhetorical_shift.png")

    return analysis


def main():
    """Run all demos"""
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "    SEMANTIC SPECTRUM ANALYSIS - Conjugate Dimensions".center(68) + "|")
    print("|" + "        BearL Labs - Unlocking Meaning".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")
    print()

    # Create output directory
    os.makedirs('demos/output', exist_ok=True)

    try:
        spectrum, embed_fn = demo_dimension_learning()
        analysis1 = demo_conversation_spectrum()
        analysis2 = demo_emotional_arc()
        analysis3 = demo_rhetorical_shift()

        print("\n" + "=" * 70)
        print("ALL SPECTRUM DEMOS COMPLETE")
        print("=" * 70)
        print("\nKey Findings:")
        print("  OK Learned 16 semantic dimensions from exemplars")
        print("  OK Tracked conversations along interpretable axes")
        print("  OK Detected emotional arcs in narratives")
        print("  OK Identified rhetorical strategy shifts")
        print("\nSemantic Forces Discovered:")
        print("  - Conversations have momentum along dimensions")
        print("  - Emotional words create acceleration in Valence/Arousal space")
        print("  - Rhetorical shifts visible as dimension velocity changes")
        print("\nThis is what unlocks meaning:")
        print("  Instead of: 'embedding changed by [0.02, -0.01, ...]'")
        print("  We see: 'Warmth +0.3, Formality -0.5, Certainty +0.2'")
        print("\nOutput saved to: demos/output/")
        print()

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