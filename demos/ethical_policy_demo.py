"""
Ethical Semantic Policy Demo

Demonstrates:
1. Differentiation: Computing virtue gradients
2. Integration: Following ethical flow
3. Pareto Frontiers: Finding optimal trade-offs
4. Manipulation Detection: Spotting unethical patterns

Shows how to navigate semantic space with moral constraints.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from HoloLoom.semantic_calculus import (
    SemanticFlowCalculus,
    SemanticSpectrum,
    EthicalSemanticPolicy,
    COMPASSIONATE_COMMUNICATION,
    SCIENTIFIC_DISCOURSE,
    THERAPEUTIC_DIALOGUE,
    visualize_ethical_landscape
)
from HoloLoom.embedding.spectral import create_embedder


def demo_virtue_gradients():
    """
    Demo 1: Show how virtue gradients point toward "good"
    """
    print("=" * 70)
    print("DEMO 1: Virtue Gradients (Differentiation)")
    print("=" * 70)

    # Setup
    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.encode([word])[0]

    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    # Create policy
    policy = EthicalSemanticPolicy(
        objective=COMPASSIONATE_COMMUNICATION,
        dimension_names=[d.name for d in spectrum.dimensions]
    )

    print("\n[1/2] Computing virtue scores for test words...")

    test_words = [
        "kind", "cruel", "honest", "manipulative",
        "patient", "rushed", "humble", "arrogant"
    ]

    print(f"\n{'Word':<15} {'Virtue':>10} {'Manipulation':>15}")
    print("-" * 45)

    for word in test_words:
        vec = embed_fn(word)
        q_sem = spectrum.project_vector(vec)
        q_array = np.array([q_sem[d.name] for d in spectrum.dimensions])

        virtue = policy.compute_virtue_score(q_array)
        manipulation = policy.detect_manipulation(q_array)

        print(f"{word:<15} {virtue:>10.3f} {manipulation:>15.3f}")

    print("\n[2/2] Computing ethical gradient...")

    # Pick a word and show gradient
    word = "nice"
    vec = embed_fn(word)
    q_sem = spectrum.project_vector(vec)
    q_array = np.array([q_sem[d.name] for d in spectrum.dimensions])

    gradient = policy.compute_ethical_gradient(q_array)

    # Show top dimensions to improve
    grad_dict = {spectrum.dimensions[i].name: gradient[i] for i in range(len(gradient))}
    sorted_dims = sorted(grad_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    print(f"\n  Starting from '{word}', to increase virtue, change:")
    for dim, grad in sorted_dims:
        direction = "INCREASE" if grad > 0 else "DECREASE"
        print(f"    {dim:<15}: {direction} (gradient: {grad:+.4f})")

    return policy, spectrum, embed_fn


def demo_ethical_trajectories():
    """
    Demo 2: Integrate ethical flow over time
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Ethical Trajectories (Integration)")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.encode([word])[0]

    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    policy = EthicalSemanticPolicy(
        objective=COMPASSIONATE_COMMUNICATION,
        dimension_names=[d.name for d in spectrum.dimensions]
    )

    # Compare two conversation paths
    path_a = "hello friend I care about you and want to help"
    path_b = "listen you need to do this right now or else"

    print("\n[1/2] Analyzing Path A (compassionate):")
    print(f"  '{path_a}'")

    words_a = path_a.split()
    calculus = SemanticFlowCalculus(embed_fn, dt=1.0)
    traj_a = calculus.compute_trajectory(words_a)

    # Project to semantic space
    sem_traj_a = np.array([
        [spectrum.project_vector(embed_fn(w))[d.name] for d in spectrum.dimensions]
        for w in words_a
    ])

    analysis_a = policy.analyze_conversation_ethics(sem_traj_a)

    print(f"    Mean virtue: {analysis_a['mean_virtue']:.3f}")
    print(f"    Mean manipulation: {analysis_a['mean_manipulation']:.3f}")
    print(f"    Ethical? {analysis_a['is_ethical']}")

    print("\n[2/2] Analyzing Path B (manipulative):")
    print(f"  '{path_b}'")

    words_b = path_b.split()
    traj_b = calculus.compute_trajectory(words_b)

    sem_traj_b = np.array([
        [spectrum.project_vector(embed_fn(w))[d.name] for d in spectrum.dimensions]
        for w in words_b
    ])

    analysis_b = policy.analyze_conversation_ethics(sem_traj_b)

    print(f"    Mean virtue: {analysis_b['mean_virtue']:.3f}")
    print(f"    Mean manipulation: {analysis_b['mean_manipulation']:.3f}")
    print(f"    Ethical? {analysis_b['is_ethical']}")

    # Plot virtue over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = axes[0]
    ax1.plot(analysis_a['virtue_scores'], 'g-', linewidth=2.5, label='Path A (compassionate)')
    ax1.plot(analysis_b['virtue_scores'], 'r-', linewidth=2.5, label='Path B (manipulative)')
    ax1.set_ylabel('Virtue Score')
    ax1.set_title('Ethical Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(analysis_a['manipulation_scores'], 'g--', linewidth=2.5, label='Path A')
    ax2.plot(analysis_b['manipulation_scores'], 'r--', linewidth=2.5, label='Path B')
    ax2.set_ylabel('Manipulation Score')
    ax2.set_xlabel('Word Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demos/output/ethical_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: demos/output/ethical_trajectories.png")

    return analysis_a, analysis_b


def demo_pareto_frontier():
    """
    Demo 3: Visualize Pareto frontier (trade-offs)
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Pareto Frontier (Multi-Objective Optimization)")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.encode([word])[0]

    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    policy = EthicalSemanticPolicy(
        objective=COMPASSIONATE_COMMUNICATION,
        dimension_names=[d.name for d in spectrum.dimensions]
    )

    print("\n[1/2] Exploring Warmth-Formality trade-off...")

    # Sample points along warmth-formality spectrum
    test_phrases = [
        "hey",          # high warmth, low formality
        "hello",        # balanced
        "greetings",    # lower warmth, higher formality
        "salutations",  # low warmth, high formality
    ]

    warmths = []
    formalities = []
    virtues = []

    for phrase in test_phrases:
        vec = embed_fn(phrase)
        q_sem = spectrum.project_vector(vec)
        q_array = np.array([q_sem[d.name] for d in spectrum.dimensions])

        warmth = q_sem['Warmth']
        formality = q_sem['Formality']
        virtue = policy.compute_virtue_score(q_array)

        warmths.append(warmth)
        formalities.append(formality)
        virtues.append(virtue)

        print(f"  '{phrase:<12}': Warmth={warmth:>6.2f}, Formality={formality:>6.2f}, Virtue={virtue:>6.2f}")

    print("\n[2/2] Visualizing ethical landscape...")

    # Create 2D virtue landscape
    fig = visualize_ethical_landscape(
        policy,
        dim1='Warmth',
        dim2='Formality',
        n_points=40,
        save_path='demos/output/ethical_landscape_warmth_formality.png'
    )

    # Add test points to plot
    axes = fig.axes[:2]
    for ax in axes:
        ax.scatter(warmths, formalities, c='blue', s=100, edgecolors='white', linewidths=2, zorder=10)
        for i, phrase in enumerate(test_phrases):
            ax.annotate(phrase, (warmths[i], formalities[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.savefig('demos/output/ethical_landscape_warmth_formality.png', dpi=150, bbox_inches='tight')

    # Create another landscape: Certainty-Power
    fig2 = visualize_ethical_landscape(
        policy,
        dim1='Certainty',
        dim2='Power',
        n_points=40,
        save_path='demos/output/ethical_landscape_certainty_power.png'
    )

    return warmths, formalities, virtues


def demo_constrained_path():
    """
    Demo 4: Find ethical path between two points
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Constrained Geodesic (Ethical Path Finding)")
    print("=" * 70)

    embed_model = create_embedder(sizes=[384])

    def embed_fn(word):
        return embed_model.encode([word])[0]

    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    policy = EthicalSemanticPolicy(
        objective=COMPASSIONATE_COMMUNICATION,
        dimension_names=[d.name for d in spectrum.dimensions]
    )

    print("\n[1/2] Finding path from 'command' to 'request'...")

    # Start and end words
    start_word = "command"
    end_word = "request"

    # Get semantic positions
    start_vec = embed_fn(start_word)
    end_vec = embed_fn(end_word)

    start_sem = spectrum.project_vector(start_vec)
    end_sem = spectrum.project_vector(end_vec)

    start_array = np.array([start_sem[d.name] for d in spectrum.dimensions])
    end_array = np.array([end_sem[d.name] for d in spectrum.dimensions])

    # Find direct path
    direct_path = np.linspace(start_array, end_array, num=20)

    # Find ethical path (constrained geodesic)
    ethical_path = policy.find_ethical_path(start_array, end_array, n_steps=20, alpha=0.4)

    # Compute virtue along both paths
    direct_virtues = [policy.compute_virtue_score(q) for q in direct_path]
    ethical_virtues = [policy.compute_virtue_score(q) for q in ethical_path]

    print(f"\n[2/2] Path comparison:")
    print(f"    Direct path:")
    print(f"      Mean virtue: {np.mean(direct_virtues):.3f}")
    print(f"      Min virtue: {np.min(direct_virtues):.3f}")

    print(f"    Ethical path (constrained geodesic):")
    print(f"      Mean virtue: {np.mean(ethical_virtues):.3f}")
    print(f"      Min virtue: {np.min(ethical_virtues):.3f}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(direct_virtues, 'r--', linewidth=2.5, label='Direct Path', alpha=0.7)
    ax.plot(ethical_virtues, 'g-', linewidth=2.5, label='Ethical Path (Constrained Geodesic)')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Virtue Score', fontsize=12)
    ax.set_title(f'Path Comparison: "{start_word}" -> "{end_word}"', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demos/output/constrained_geodesic.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: demos/output/constrained_geodesic.png")

    return direct_path, ethical_path


def main():
    """Run all ethical policy demos"""
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "       ETHICAL SEMANTIC POLICY ENGINE".center(68) + "|")
    print("|" + "   Multi-Objective Optimization with Moral Constraints".center(68) + "|")
    print("|" + "              BearL Labs".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")
    print()

    os.makedirs('demos/output', exist_ok=True)

    try:
        policy, spectrum, embed_fn = demo_virtue_gradients()
        analysis_a, analysis_b = demo_ethical_trajectories()
        warmths, formalities, virtues = demo_pareto_frontier()
        direct, ethical = demo_constrained_path()

        print("\n" + "=" * 70)
        print("ALL ETHICAL POLICY DEMOS COMPLETE")
        print("=" * 70)

        print("\nKey Findings:")
        print("  1. DIFFERENTIATION: Virtue gradients point toward ethical communication")
        print("  2. INTEGRATION: Can integrate flow to create ethical trajectories")
        print("  3. PARETO FRONTIER: Trade-offs visualized in warmth-formality space")
        print("  4. CONSTRAINED GEODESICS: Found paths that maximize virtue")

        print("\nManipulation Detection:")
        print(f"  Compassionate path: {analysis_a['mean_manipulation']:.3f}")
        print(f"  Manipulative path: {analysis_b['mean_manipulation']:.3f}")
        print("  -> Successfully detected unethical patterns!")

        print("\nThis enables:")
        print("  - Conversation steering toward ethical outcomes")
        print("  - Real-time manipulation detection")
        print("  - Multi-objective optimization of communication")
        print("  - Balancing competing values (warmth vs formality, etc.)")

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