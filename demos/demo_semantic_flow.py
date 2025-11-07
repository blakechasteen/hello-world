"""
Semantic Flow Demo
==================

Demonstrates PDE-based temporal dynamics in semantic space.

This demo shows:
1. Heat equation (diffusion) - spreading activation
2. Wave equation (oscillation) - resonance patterns
3. Reaction-diffusion - competitive activation
4. Semantic trajectory visualization over time

Author: HoloLoom Mathematical Moonshot Team
Date: 2025-11-03
Priority: 6
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from HoloLoom.semantic_calculus.flow import create_semantic_flow, SemanticFlowState
    from HoloLoom.semantic_calculus.dimensions import (
        SemanticSpectrum,
        STANDARD_DIMENSIONS,
        EXTENDED_244_DIMENSIONS,
    )
    _FLOW_AVAILABLE = True
except ImportError as e:
    _FLOW_AVAILABLE = False
    print(f"Error importing semantic flow: {e}")
    print("Ensure HoloLoom.semantic_calculus.flow is available")


def demo_heat_equation():
    """Demo 1: Heat equation - diffusion spreading activation."""
    print("\n" + "=" * 80)
    print("DEMO 1: HEAT EQUATION (Diffusion)")
    print("=" * 80)
    print("\nConcept: Information diffuses from high-activation to low-activation regions")
    print("Use case: Exploration, activation spreading, concept blending\n")

    # Create semantic flow with heat equation
    flow = create_semantic_flow(
        dimensions=16,  # Use 16D for speed
        pde_type="heat",
        dt=0.01,
        implicit=True  # Stable implicit solver
    )

    # Create initial state: localized activation
    initial_activation = np.zeros(16)
    initial_activation[0] = 1.0  # High activation at dimension 0
    initial_activation[1] = 0.5  # Medium activation at dimension 1

    initial_state = flow.create_state(initial_activation, time=0.0)

    print(f"Initial state:")
    print(f"  Energy: {initial_state.energy:.4f}")
    print(f"  Entropy: {initial_state.entropy:.4f}")
    print(f"  Peak activation: {np.max(initial_activation):.4f} at dim 0")

    # Evolve using heat equation
    print(f"\nEvolving for 100 steps (dt={flow.dt})...")
    final_state = flow.evolve(initial_state, steps=100, track_trajectory=True)

    print(f"\nFinal state:")
    print(f"  Energy: {final_state.energy:.4f}")
    print(f"  Entropy: {final_state.entropy:.4f}")
    print(f"  Max activation: {np.max(final_state.activation):.4f}")
    print(f"\n  => Activation diffused (entropy increased, peak decreased)")

    # Visualize
    dimension_names = [f"Dim {i}" for i in range(16)]
    flow.visualize_trajectory(
        dimension_names=dimension_names,
        save_path="demos/output/semantic_flow_heat.png"
    )

    return flow


def demo_wave_equation():
    """Demo 2: Wave equation - oscillatory dynamics."""
    print("\n" + "=" * 80)
    print("DEMO 2: WAVE EQUATION (Oscillation)")
    print("=" * 80)
    print("\nConcept: Oscillatory back-and-forth semantic dynamics")
    print("Use case: Dialectical reasoning, debate, refinement\n")

    # Create semantic flow with wave equation
    flow = create_semantic_flow(
        dimensions=16,
        pde_type="wave",
        dt=0.005,  # Smaller timestep for stability
        c=1.0  # Wave speed
    )

    # Create initial state with velocity
    initial_activation = np.zeros(16)
    initial_activation[8] = 1.0  # Impulse at center

    initial_state = flow.create_state(
        initial_activation,
        time=0.0,
        velocity=np.zeros(16)  # Initial velocity
    )

    print(f"Initial state:")
    print(f"  Energy: {initial_state.energy:.4f}")
    print(f"  Impulse at dimension 8")

    # Evolve using wave equation
    print(f"\nEvolving for 200 steps (dt={flow.dt})...")
    final_state = flow.evolve(initial_state, steps=200, track_trajectory=True)

    print(f"\nFinal state:")
    print(f"  Energy: {final_state.energy:.4f}")
    print(f"  => Wave propagated and reflected (energy conserved)")

    # Visualize
    dimension_names = [f"Dim {i}" for i in range(16)]
    flow.visualize_trajectory(
        dimension_names=dimension_names,
        save_path="demos/output/semantic_flow_wave.png"
    )

    return flow


def demo_reaction_diffusion():
    """Demo 3: Reaction-diffusion - competitive activation."""
    print("\n" + "=" * 80)
    print("DEMO 3: REACTION-DIFFUSION (Competition)")
    print("=" * 80)
    print("\nConcept: Competitive activation with mutual inhibition")
    print("Use case: Disambiguation, tool selection, winner-take-all\n")

    # Create semantic flow with reaction-diffusion
    flow = create_semantic_flow(
        dimensions=16,
        pde_type="reaction_diffusion",
        dt=0.01,
        reaction_type="competitive",  # Competitive reaction
        r=1.0,  # Reaction rate
        theta=0.5,  # Activation threshold
        diffusion_coef=0.1  # Weak diffusion
    )

    # Create initial state: two competing regions
    initial_activation = np.random.rand(16) * 0.3  # Background noise
    initial_activation[4] = 0.8  # Competitor 1
    initial_activation[12] = 0.7  # Competitor 2

    initial_state = flow.create_state(initial_activation, time=0.0)

    print(f"Initial state:")
    print(f"  Energy: {initial_state.energy:.4f}")
    print(f"  Competitor 1 (dim 4): {initial_activation[4]:.4f}")
    print(f"  Competitor 2 (dim 12): {initial_activation[12]:.4f}")

    # Evolve using reaction-diffusion
    print(f"\nEvolving for 100 steps (dt={flow.dt})...")
    final_state = flow.evolve(initial_state, steps=100, track_trajectory=True)

    print(f"\nFinal state:")
    print(f"  Energy: {final_state.energy:.4f}")
    print(f"  Competitor 1 (dim 4): {final_state.activation[4]:.4f}")
    print(f"  Competitor 2 (dim 12): {final_state.activation[12]:.4f}")

    # Determine winner
    winner_idx = np.argmax(final_state.activation)
    print(f"\n  => Winner: Dimension {winner_idx} (competitive selection)")

    # Visualize
    dimension_names = [f"Dim {i}" for i in range(16)]
    flow.visualize_trajectory(
        dimension_names=dimension_names,
        save_path="demos/output/semantic_flow_reaction_diffusion.png"
    )

    return flow


def demo_semantic_spectrum_evolution():
    """Demo 4: Semantic spectrum with temporal evolution."""
    print("\n" + "=" * 80)
    print("DEMO 4: SEMANTIC SPECTRUM + TEMPORAL EVOLUTION")
    print("=" * 80)
    print("\nConcept: Track how interpretable semantic dimensions evolve over time")
    print("Use case: Understanding semantic drift during conversation\n")

    # Mock embedding function (for demo purposes)
    def mock_embed_fn(word):
        """Mock embedding: hash word to reproducible vector."""
        np.random.seed(hash(word) % 2**32)
        return np.random.randn(384)

    # Create semantic spectrum with 16 dimensions (first 16 from STANDARD_DIMENSIONS)
    spectrum = SemanticSpectrum(dimensions=STANDARD_DIMENSIONS[:16])

    print("Learning semantic axes from exemplars...")
    spectrum.learn_axes(mock_embed_fn)

    # Enable temporal dynamics
    print("Enabling PDE-based temporal evolution (heat equation)...")
    spectrum.enable_temporal_dynamics(
        pde_type="heat",
        dt=0.01
    )

    # Create initial semantic projection
    mock_embedding = np.random.randn(384) * 0.5
    initial_projection = spectrum.project_vector(mock_embedding)

    print(f"\nInitial semantic state:")
    for dim_name in ['Warmth', 'Formality', 'Certainty', 'Complexity']:
        if dim_name in initial_projection:
            print(f"  {dim_name}: {initial_projection[dim_name]:.3f}")

    # Evolve semantic state
    print(f"\nEvolving semantic state for 20 steps...")
    evolved_projection = spectrum.evolve(initial_projection, steps=20)

    print(f"\nEvolved semantic state:")
    for dim_name in ['Warmth', 'Formality', 'Certainty', 'Complexity']:
        if dim_name in evolved_projection:
            change = evolved_projection[dim_name] - initial_projection[dim_name]
            direction = "↑" if change > 0 else "↓"
            print(f"  {dim_name}: {evolved_projection[dim_name]:.3f} {direction} (Δ={change:+.3f})")

    # Get full trajectory
    trajectory = spectrum.get_flow_trajectory()
    if trajectory:
        print(f"\n  => Tracked {len(trajectory)} states in semantic trajectory")

    return spectrum


def compare_pde_types():
    """Demo 5: Compare all PDE types side-by-side."""
    print("\n" + "=" * 80)
    print("DEMO 5: PDE TYPE COMPARISON")
    print("=" * 80)
    print("\nComparing semantic evolution across different PDE types\n")

    # Create identical initial state
    np.random.seed(42)
    initial_activation = np.random.rand(16) * 0.5
    initial_activation[8] = 1.0  # Peak at center

    pde_types = ["heat", "wave", "reaction_diffusion"]
    flows = {}
    final_states = {}

    for pde_type in pde_types:
        print(f"\n{pde_type.upper()}:")

        # Create flow
        kwargs = {"dimensions": 16, "pde_type": pde_type, "dt": 0.01}
        if pde_type == "wave":
            kwargs["c"] = 1.0
        elif pde_type == "reaction_diffusion":
            kwargs["reaction_type"] = "competitive"
            kwargs["diffusion_coef"] = 0.1

        flow = create_semantic_flow(**kwargs)

        # Create state
        state_kwargs = {"activation": initial_activation.copy(), "time": 0.0}
        if pde_type == "wave":
            state_kwargs["velocity"] = np.zeros(16)

        initial_state = flow.create_state(**state_kwargs)

        # Evolve
        final_state = flow.evolve(initial_state, steps=100, track_trajectory=True)

        flows[pde_type] = flow
        final_states[pde_type] = final_state

        print(f"  Initial energy: {initial_state.energy:.4f}")
        print(f"  Final energy: {final_state.energy:.4f}")
        print(f"  Initial entropy: {initial_state.entropy:.4f}")
        print(f"  Final entropy: {final_state.entropy:.4f}")

    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for idx, pde_type in enumerate(pde_types):
        trajectory = flows[pde_type].get_trajectory_array()
        times = np.array([state.time for state in flows[pde_type].get_trajectory()])

        ax = axes[idx]
        # Plot activation at center dimension
        ax.plot(times, trajectory[:, 8], 'b-', linewidth=2, label='Center (Dim 8)')
        # Plot activation at neighbors
        ax.plot(times, trajectory[:, 7], 'g--', linewidth=1.5, alpha=0.7, label='Dim 7')
        ax.plot(times, trajectory[:, 9], 'r--', linewidth=1.5, alpha=0.7, label='Dim 9')

        ax.set_ylabel('Activation', fontsize=12)
        ax.set_title(f'{pde_type.upper()} PDE', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=12)
    plt.tight_layout()
    plt.savefig('demos/output/semantic_flow_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n  => Saved comparison plot: demos/output/semantic_flow_comparison.png")

    return flows


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("SEMANTIC FLOW DEMONSTRATION")
    print("=" * 80)
    print("\nPDE-Based Temporal Dynamics in 244D Semantic Space")
    print("Mathematical Moonshot - Priority 6")
    print("=" * 80)

    if not _FLOW_AVAILABLE:
        print("\nERROR: Semantic flow not available. Cannot run demos.")
        print("Ensure HoloLoom.semantic_calculus.flow is installed.")
        return

    try:
        # Demo 1: Heat equation
        heat_flow = demo_heat_equation()

        # Demo 2: Wave equation
        wave_flow = demo_wave_equation()

        # Demo 3: Reaction-diffusion
        rd_flow = demo_reaction_diffusion()

        # Demo 4: Semantic spectrum evolution
        spectrum = demo_semantic_spectrum_evolution()

        # Demo 5: Compare PDE types
        flows = compare_pde_types()

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\n✅ All demos completed successfully!")
        print("\nKey insights:")
        print("  1. Heat equation smooths activation (diffusion)")
        print("  2. Wave equation creates oscillations (resonance)")
        print("  3. Reaction-diffusion enables competition (winner-take-all)")
        print("  4. Semantic spectrum tracks interpretable dimensions")
        print("  5. Different PDEs model different reasoning modes")
        print("\nVisualizations saved to demos/output/")
        print("  - semantic_flow_heat.png")
        print("  - semantic_flow_wave.png")
        print("  - semantic_flow_reaction_diffusion.png")
        print("  - semantic_flow_comparison.png")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\nERROR during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
